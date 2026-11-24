//! 10-slot Sliding-Window Queue + Non-Blocking Delayed Retry Min-Heap + S5 `429`/`432` AIMD Router.
//!
//! Eliminates in-slot retry starvation (`buffer_unordered(10)` + blocking `retry_http` sleep)
//! and decouples transient HTTP `429` (AIMD halve + `Retry-After` cooldown + additive recovery)
//! from permanent HTTP `432` quota exhaustion (`ProviderState::PermanentOpen` + instant failover).

use futures_util::stream::{FuturesUnordered, StreamExt};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::future::Future;
use std::time::{Duration, Instant};

pub const DEFAULT_SLIDING_SLOTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderState {
    Closed,
    Cooldown(Instant),
    PermanentOpen,
}

#[derive(Debug, Clone)]
pub struct ProviderSlot {
    pub name: String,
    pub state: ProviderState,
    pub aimd_limit: usize,
    pub max_limit: usize,
    pub active: usize,
    pub consec_ok: usize,
    pub dispatched: usize,
    pub ok_calls: usize,
    pub err_429: usize,
    pub err_432: usize,
    pub aimd_drops: usize,
}

impl ProviderSlot {
    pub fn new(name: impl Into<String>, initial_limit: usize, max_limit: usize) -> Self {
        let max_lim = max_limit.max(1);
        Self {
            name: name.into(),
            state: ProviderState::Closed,
            aimd_limit: initial_limit.clamp(1, max_lim),
            max_limit: max_lim,
            active: 0,
            consec_ok: 0,
            dispatched: 0,
            ok_calls: 0,
            err_429: 0,
            err_432: 0,
            aimd_drops: 0,
        }
    }

    pub fn is_ready(&self, now: Instant) -> bool {
        match self.state {
            ProviderState::Closed => true,
            ProviderState::Cooldown(until) => now >= until,
            ProviderState::PermanentOpen => false,
        }
    }
}

pub enum DispatchOutcome<T> {
    Ok(T),
    /// Transient HTTP `429`: AIMD halve (`max(1)`) + timed `Retry-After` cooldown (NEVER permanent open).
    RateLimited429 { retry_after: Duration },
    /// Permanent HTTP `432` billing/quota exhaustion: `PermanentOpen` + instant failover.
    QuotaExhausted432,
    /// Transient `5xx`/timeout/network error: non-blocking backoff via delayed retry min-heap.
    TransientErr { backoff: Duration, fallback: T },
    /// Non-retryable terminal response (e.g. 400/401/403/404).
    Terminal(T),
}

#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    pub dispatched: usize,
    pub retries: usize,
    pub failovers: usize,
    pub max_in_flight: usize,
}

#[derive(Debug, Clone)]
pub struct SlidingRouter {
    providers: Vec<ProviderSlot>,
    max_slots: usize,
    max_attempts: u32,
    rr_cursor: usize,
}

impl SlidingRouter {
    pub fn new(providers: Vec<ProviderSlot>, max_slots: usize, max_attempts: u32) -> Self {
        Self {
            providers,
            max_slots: max_slots.max(1),
            max_attempts: max_attempts.max(1),
            rr_cursor: 0,
        }
    }

    #[cfg(test)]
    pub fn providers(&self) -> &[ProviderSlot] {
        &self.providers
    }

    pub fn has_non_permanent_provider(&self) -> bool {
        self.providers.iter().any(|p| p.state != ProviderState::PermanentOpen)
    }

    fn wake_expired_cooldowns(&mut self, now: Instant) {
        for p in &mut self.providers {
            if let ProviderState::Cooldown(until) = p.state
                && now >= until
            {
                p.state = ProviderState::Closed;
            }
        }
    }

    pub fn pick_healthy_provider(&mut self, now: Instant) -> Option<usize> {
        self.wake_expired_cooldowns(now);
        let n = self.providers.len();
        if n == 0 {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for offset in 0..n {
            let idx = (self.rr_cursor + offset) % n;
            let p = &self.providers[idx];
            if p.is_ready(now) && p.active < p.aimd_limit {
                match best {
                    None => best = Some((p.active, idx)),
                    Some((best_active, _)) if p.active < best_active => {
                        best = Some((p.active, idx));
                    }
                    _ => {}
                }
            }
        }
        if let Some((_, idx)) = best {
            self.rr_cursor = (idx + 1) % n;
            Some(idx)
        } else {
            None
        }
    }

    fn has_other_ready_provider(&self, exclude_idx: usize, now: Instant) -> bool {
        self.providers
            .iter()
            .enumerate()
            .any(|(i, p)| i != exclude_idx && p.is_ready(now) && p.active < p.aimd_limit)
    }

    fn earliest_cooldown(&self) -> Option<Instant> {
        self.providers
            .iter()
            .filter_map(|p| match p.state {
                ProviderState::Cooldown(until) => Some(until),
                _ => None,
            })
            .min()
    }

    fn record_ok(&mut self, pidx: usize) {
        let p = &mut self.providers[pidx];
        p.active = p.active.saturating_sub(1);
        p.ok_calls += 1;
        p.consec_ok += 1;
        if p.consec_ok >= 4 && p.aimd_limit < p.max_limit {
            p.aimd_limit += 1;
            p.consec_ok = 0;
            tracing::debug!(
                provider = %p.name,
                aimd_limit = p.aimd_limit,
                ok_calls = p.ok_calls,
                dispatched = p.dispatched,
                "provider AIMD additive recovery"
            );
        }
    }

    fn record_432(&mut self, pidx: usize) {
        let p = &mut self.providers[pidx];
        p.active = p.active.saturating_sub(1);
        p.err_432 += 1;
        p.consec_ok = 0;
        p.state = ProviderState::PermanentOpen;
        tracing::warn!(
            provider = %p.name,
            err_432 = p.err_432,
            "HTTP 432 permanent quota exhausted; provider marked PermanentOpen"
        );
    }

    fn record_429(&mut self, pidx: usize, now: Instant, retry_after: Duration) {
        let p = &mut self.providers[pidx];
        p.active = p.active.saturating_sub(1);
        p.err_429 += 1;
        p.consec_ok = 0;
        let next_lim = (p.aimd_limit / 2).max(1);
        if next_lim < p.aimd_limit {
            p.aimd_drops += 1;
            p.aimd_limit = next_lim;
        }
        if p.state != ProviderState::PermanentOpen {
            p.state = ProviderState::Cooldown(now + retry_after);
        }
        tracing::debug!(
            provider = %p.name,
            err_429 = p.err_429,
            aimd_limit = p.aimd_limit,
            aimd_drops = p.aimd_drops,
            "HTTP 429 rate limited; AIMD halved with timed cooldown"
        );
    }

    fn record_other(&mut self, pidx: usize) {
        let p = &mut self.providers[pidx];
        p.active = p.active.saturating_sub(1);
        p.consec_ok = 0;
    }

    fn claim_next_dispatch<T>(
        &mut self,
        now: Instant,
        state: &mut BatchState<T>,
    ) -> Option<DispatchMeta> {
        if state.ready_queue.is_empty() {
            return None;
        }
        let pidx = self.pick_healthy_provider(now)?;
        let (tid, att) = state.ready_queue.pop_front()?;
        self.providers[pidx].active += 1;
        self.providers[pidx].dispatched += 1;
        state.stats.dispatched += 1;
        Some(DispatchMeta { tid, pidx, att })
    }

    fn next_wakeup(
        &self,
        retry_heap: &BinaryHeap<Reverse<(Instant, usize, u32)>>,
    ) -> Option<Instant> {
        let next_retry = retry_heap.peek().map(|entry| entry.0.0);
        let next_cooldown = self.earliest_cooldown();
        match (next_retry, next_cooldown) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    fn handle_quota_exhausted<T, E>(
        &mut self,
        meta: DispatchMeta,
        state: &mut BatchState<T>,
        on_exhausted: &mut E,
    ) where
        E: FnMut(usize, &'static str) -> T,
    {
        self.record_432(meta.pidx);
        if self.has_non_permanent_provider() {
            state.stats.failovers += 1;
            state.ready_queue.push_front((meta.tid, meta.att));
        } else {
            state.finish_task(meta.tid, on_exhausted(meta.tid, "quota_exhausted_432"));
        }
    }

    fn handle_rate_limited<T, E>(
        &mut self,
        meta: DispatchMeta,
        finish_now: Instant,
        retry_after: Duration,
        state: &mut BatchState<T>,
        on_exhausted: &mut E,
    ) where
        E: FnMut(usize, &'static str) -> T,
    {
        self.record_429(meta.pidx, finish_now, retry_after);
        if self.has_other_ready_provider(meta.pidx, finish_now) {
            state.stats.failovers += 1;
            state.ready_queue.push_front((meta.tid, meta.att));
        } else if meta.att + 1 < self.max_attempts {
            state.stats.retries += 1;
            state.retry_heap.push(Reverse((finish_now + retry_after, meta.tid, meta.att + 1)));
        } else {
            state.finish_task(meta.tid, on_exhausted(meta.tid, "rate_limited_429"));
        }
    }

    fn handle_transient_err<T>(
        &mut self,
        meta: DispatchMeta,
        finish_now: Instant,
        backoff: Duration,
        fallback: T,
        state: &mut BatchState<T>,
    ) {
        self.record_other(meta.pidx);
        if meta.att + 1 < self.max_attempts {
            state.stats.retries += 1;
            state.retry_heap.push(Reverse((finish_now + backoff, meta.tid, meta.att + 1)));
        } else {
            state.finish_task(meta.tid, fallback);
        }
    }

    fn handle_completion<T, E>(
        &mut self,
        meta: DispatchMeta,
        outcome: DispatchOutcome<T>,
        state: &mut BatchState<T>,
        on_exhausted: &mut E,
    ) where
        E: FnMut(usize, &'static str) -> T,
    {
        let finish_now = Instant::now();
        match outcome {
            DispatchOutcome::Ok(val) => {
                self.record_ok(meta.pidx);
                state.finish_task(meta.tid, val);
            }
            DispatchOutcome::Terminal(val) => {
                self.record_other(meta.pidx);
                state.finish_task(meta.tid, val);
            }
            DispatchOutcome::QuotaExhausted432 => {
                self.handle_quota_exhausted(meta, state, on_exhausted);
            }
            DispatchOutcome::RateLimited429 { retry_after } => {
                self.handle_rate_limited(meta, finish_now, retry_after, state, on_exhausted);
            }
            DispatchOutcome::TransientErr { backoff, fallback } => {
                self.handle_transient_err(meta, finish_now, backoff, fallback, state);
            }
        }
    }

    /// Execute a heterogeneous batch over at most `max_slots` concurrent slots with a
    /// non-blocking delayed retry min-heap (`BinaryHeap<Reverse<...>>`) and S5 `429`/`432` routing.
    pub async fn execute_batch<I, T, F, Fut, E>(
        &mut self,
        items: &[I],
        mut dispatch: F,
        mut on_exhausted: E,
    ) -> (Vec<T>, RouterStats)
    where
        F: FnMut(usize, usize, u32) -> Fut,
        Fut: Future<Output = DispatchOutcome<T>>,
        E: FnMut(usize, &'static str) -> T,
    {
        let total = items.len();
        if total == 0 {
            return (Vec::new(), RouterStats::default());
        }

        let mut state = BatchState::new(total);
        let mut in_flight = FuturesUnordered::new();

        while state.completed < total {
            let now = Instant::now();
            state.drain_due_retries(now);
            self.wake_expired_cooldowns(now);

            while in_flight.len() < self.max_slots
                && let Some(meta) = self.claim_next_dispatch(now, &mut state)
            {
                let fut = dispatch(meta.pidx, meta.tid, meta.att);
                in_flight.push(async move {
                    let out = fut.await;
                    (meta, out)
                });
                state.stats.max_in_flight = state.stats.max_in_flight.max(in_flight.len());
            }

            let next_due = self.next_wakeup(&state.retry_heap);
            if !in_flight.is_empty() {
                if let Some((meta, outcome)) = await_next_in_flight(&mut in_flight, next_due).await
                {
                    self.handle_completion(meta, outcome, &mut state, &mut on_exhausted);
                }
            } else if let Some(due) = next_due {
                let deadline = tokio::time::Instant::from_std(due);
                tokio::time::sleep_until(deadline).await;
            } else {
                // Every provider is `PermanentOpen`; resolve remaining queued tasks immediately.
                state.fail_remaining_queued(&mut on_exhausted);
                break;
            }
        }

        let ordered = state
            .results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| on_exhausted(i, "router_incomplete")))
            .collect();
        (ordered, state.stats)
    }
}

#[derive(Clone, Copy)]
struct DispatchMeta {
    tid: usize,
    pidx: usize,
    att: u32,
}

struct BatchState<T> {
    results: Vec<Option<T>>,
    ready_queue: VecDeque<(usize, u32)>,
    retry_heap: BinaryHeap<Reverse<(Instant, usize, u32)>>,
    stats: RouterStats,
    completed: usize,
}

impl<T> BatchState<T> {
    fn new(total: usize) -> Self {
        Self {
            results: (0..total).map(|_| None).collect(),
            ready_queue: (0..total).map(|i| (i, 0u32)).collect(),
            retry_heap: BinaryHeap::new(),
            stats: RouterStats::default(),
            completed: 0,
        }
    }

    fn finish_task(&mut self, tid: usize, val: T) {
        self.results[tid] = Some(val);
        self.completed += 1;
    }

    fn drain_due_retries(&mut self, now: Instant) {
        while let Some(&Reverse((due, _, _))) = self.retry_heap.peek() {
            if due > now {
                break;
            }
            if let Some(Reverse((_, tid, att))) = self.retry_heap.pop() {
                self.ready_queue.push_back((tid, att));
            }
        }
    }

    fn fail_remaining_queued<E>(&mut self, on_exhausted: &mut E)
    where
        E: FnMut(usize, &'static str) -> T,
    {
        while let Some((tid, _)) = self.ready_queue.pop_front() {
            if self.results[tid].is_none() {
                self.results[tid] = Some(on_exhausted(tid, "quota_exhausted_432"));
            }
        }
    }
}

async fn await_next_in_flight<Fut, T>(
    in_flight: &mut FuturesUnordered<Fut>,
    next_due: Option<Instant>,
) -> Option<T>
where
    Fut: Future<Output = T>,
{
    if let Some(due) = next_due {
        let deadline = tokio::time::Instant::from_std(due);
        tokio::select! {
            ev = in_flight.next() => ev,
            () = tokio::time::sleep_until(deadline) => None,
        }
    } else {
        in_flight.next().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + (self.next_u64() % (hi - lo + 1))
        }
    }

    #[derive(Clone)]
    struct SimItem {
        key: u32,
        latency_ms: u64,
        fail_attempts: u32,
    }

    fn make_500_tasks() -> Vec<SimItem> {
        let mut rng = Rng::new(0x59E5_B810_C0DE_2026);
        let unique = 400usize;
        let mut base = Vec::with_capacity(unique);
        for id in 0..unique {
            let latency_ms = if rng.range(1, 100) <= 85 { 1 } else { 6 };
            let froll = rng.range(1, 100);
            let fail_attempts = if froll <= 12 { 1 } else if froll <= 15 { 2 } else { 0 };
            base.push(SimItem { key: id as u32, latency_ms, fail_attempts });
        }
        let mut all = Vec::with_capacity(500);
        all.extend_from_slice(&base);
        for _ in unique..500 {
            let pick = (rng.next_u64() as usize) % unique;
            all.push(base[pick].clone());
        }
        all
    }

    #[tokio::test]
    async fn s1_sliding_10_queue_non_blocking_retry_heap_and_dedup() {
        let tasks = make_500_tasks();
        let mut seen = HashMap::new();
        let mut unique = Vec::new();
        let mut deduped = 0usize;
        for t in &tasks {
            if seen.contains_key(&t.key) {
                deduped += 1;
            } else {
                seen.insert(t.key, unique.len());
                unique.push(t.clone());
            }
        }
        assert_eq!((unique.len(), deduped), (400, 100));

        let mut router = SlidingRouter::new(
            vec![ProviderSlot::new("primary", DEFAULT_SLIDING_SLOTS, DEFAULT_SLIDING_SLOTS)],
            DEFAULT_SLIDING_SLOTS,
            4,
        );
        let items_arc = Arc::new(unique);
        let (out, stats) = router
            .execute_batch(
                &items_arc,
                |_pidx, tid, att| {
                    let item = items_arc[tid].clone();
                    async move {
                        if att < item.fail_attempts {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            DispatchOutcome::TransientErr {
                                backoff: Duration::from_millis(2),
                                fallback: false,
                            }
                        } else {
                            tokio::time::sleep(Duration::from_millis(item.latency_ms)).await;
                            DispatchOutcome::Ok(true)
                        }
                    }
                },
                |_tid, _| false,
            )
            .await;

        assert_eq!(out.iter().filter(|&&ok| ok).count(), 400);
        assert_eq!(out.len() + deduped, 500);
        assert!(stats.max_in_flight <= 10 && stats.max_in_flight >= 2);
        assert!(stats.retries > 0, "must exercise non-blocking retry heap");
    }

    #[tokio::test]
    async fn s5_five_provider_429_432_storm_aimd_and_permanent_open() {
        let tasks = make_500_tasks();
        let mut seen = HashMap::new();
        let mut unique = Vec::new();
        let mut deduped = 0usize;
        for t in &tasks {
            if seen.contains_key(&t.key) {
                deduped += 1;
            } else {
                seen.insert(t.key, unique.len());
                unique.push(t.clone());
            }
        }

        let providers = vec![
            ProviderSlot::new("p0_tavily_432", 3, 4),
            ProviderSlot::new("p1_exa_429", 3, 4),
            ProviderSlot::new("p2_firecrawl", 3, 4),
            ProviderSlot::new("p3_keenable", 3, 4),
            ProviderSlot::new("p4_parallel", 3, 4),
        ];
        let mut router = SlidingRouter::new(providers, DEFAULT_SLIDING_SLOTS, 4);
        let p1_seq = Arc::new(AtomicUsize::new(0));

        let (out, stats) = router
            .execute_batch(
                &unique,
                |pidx, _tid, _att| {
                    let seq_counter = p1_seq.clone();
                    async move {
                        if pidx == 0 {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            DispatchOutcome::QuotaExhausted432
                        } else if pidx == 1 {
                            let seq = seq_counter.fetch_add(1, Ordering::Relaxed) + 1;
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            if (3..=5).contains(&seq) {
                                DispatchOutcome::RateLimited429 {
                                    retry_after: Duration::from_millis(3),
                                }
                            } else {
                                DispatchOutcome::Ok(true)
                            }
                        } else {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            DispatchOutcome::Ok(true)
                        }
                    }
                },
                |_tid, _| false,
            )
            .await;

        assert_eq!(out.iter().filter(|&&ok| ok).count() + deduped, 500);
        assert!(stats.max_in_flight <= 10, "max_in_flight={}", stats.max_in_flight);
        assert!(stats.failovers >= 1);

        let ps = router.providers();
        // P0 tripped PermanentOpen on HTTP 432 and stopped receiving traffic after initial wave.
        assert_eq!(ps[0].state, ProviderState::PermanentOpen);
        assert!(ps[0].err_432 >= 1 && ps[0].err_432 <= 10);

        // P1 executed AIMD decrease on 429 burst, stayed NOT PermanentOpen, and recovered healthy throughput.
        assert!(ps[1].err_429 >= 3);
        assert!(ps[1].aimd_drops >= 1);
        assert_ne!(ps[1].state, ProviderState::PermanentOpen);
        assert!(ps[1].ok_calls > 0);
    }
}
