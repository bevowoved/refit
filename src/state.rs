use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use tokio::sync::Notify;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitState {
    pub phase: String,
    pub attempt: u32,
    pub loc_before: usize,
    pub loc_after: usize,
    pub hygiene_after: f64,
    pub last_blake3: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateDigest {
    pub tracked_units: usize,
    pub committed_units: u64,
    pub in_flight_units: u64,
    pub blocked_units: usize,
    pub loc_before: u64,
    pub loc_after: u64,
    pub stage_hardlinked_files: u64,
    pub telemetry_dedup_skipped: u64,
    pub research_queries: u64,
    pub research_cache_hits: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteerState {
    pub paused: bool,
    pub abort_requested: bool,
    pub k_override: Option<usize>,
    pub priority_queue: Vec<String>,
    pub skipped_units: Vec<String>,
}

pub struct RunState {
    units: DashMap<String, UnitState>,
    committed_units: AtomicU64,
    in_flight_units: AtomicU64,
    loc_before: AtomicU64,
    loc_after: AtomicU64,
    stage_hardlinked_files: AtomicU64,
    telemetry_dedup_skipped: AtomicU64,
    research_queries: AtomicU64,
    research_cache_hits: AtomicU64,
    // Event-driven live steering controls (zero polling via Notify).
    paused: AtomicBool,
    abort_requested: AtomicBool,
    k_override: AtomicUsize,
    priority_units: Mutex<VecDeque<String>>,
    skipped_units: DashMap<String, ()>,
    blocked_units: DashMap<String, String>,
    steer_notify: Notify,
}

impl RunState {
    pub fn new(initial_loc: usize) -> Arc<Self> {
        Arc::new(Self {
            units: DashMap::new(),
            committed_units: AtomicU64::new(0),
            in_flight_units: AtomicU64::new(0),
            loc_before: AtomicU64::new(initial_loc as u64),
            loc_after: AtomicU64::new(initial_loc as u64),
            stage_hardlinked_files: AtomicU64::new(0),
            telemetry_dedup_skipped: AtomicU64::new(0),
            research_queries: AtomicU64::new(0),
            research_cache_hits: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            abort_requested: AtomicBool::new(false),
            k_override: AtomicUsize::new(0),
            priority_units: Mutex::new(VecDeque::new()),
            skipped_units: DashMap::new(),
            blocked_units: DashMap::new(),
            steer_notify: Notify::new(),
        })
    }

    pub fn steer_pause(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
        if !paused {
            self.steer_notify.notify_waiters();
        }
    }

    pub fn steer_abort(&self) {
        self.abort_requested.store(true, Ordering::Release);
        self.paused.store(false, Ordering::Release);
        self.steer_notify.notify_waiters();
    }

    pub fn steer_k(&self, k: Option<usize>) {
        self.k_override
            .store(k.unwrap_or(0).min(64), Ordering::Release);
        self.steer_notify.notify_waiters();
    }

    pub fn steer_prioritize(&self, unit_id: String) {
        if unit_id.is_empty() {
            return;
        }
        let mut q = self.priority_units.lock();
        if !q.contains(&unit_id) {
            q.push_back(unit_id);
        }
        drop(q);
        self.steer_notify.notify_waiters();
    }

    pub fn steer_skip(&self, unit_id: String) {
        if !unit_id.is_empty() {
            self.skipped_units.insert(unit_id, ());
            self.steer_notify.notify_waiters();
        }
    }

    pub async fn await_resume(&self) {
        while self.paused.load(Ordering::Acquire) && !self.abort_requested.load(Ordering::Acquire) {
            let notified = self.steer_notify.notified();
            if !self.paused.load(Ordering::Acquire) || self.abort_requested.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
    }

    pub fn abort_asked(&self) -> bool {
        self.abort_requested.load(Ordering::Acquire)
    }

    pub fn k_override(&self) -> Option<usize> {
        match self.k_override.load(Ordering::Acquire) {
            0 => None,
            k => Some(k),
        }
    }

    #[cfg(test)]
    pub fn pop_prio(&self) -> Option<String> {
        self.priority_units.lock().pop_front()
    }

    /// Pop the first queued priority unit that is currently ready, preserving unready requests.
    pub fn pop_ready<F: Fn(&str) -> bool>(&self, is_ready: F) -> Option<String> {
        let mut q = self.priority_units.lock();
        if let Some(pos) = q.iter().position(|uid| is_ready(uid)) {
            q.remove(pos)
        } else {
            None
        }
    }

    pub fn mark_blocked(&self, unit_id: &str, reason: &str) {
        self.blocked_units
            .insert(unit_id.to_string(), reason.to_string());
    }

    pub fn blocked_count(&self) -> usize {
        self.blocked_units.len()
    }

    pub fn is_skipped(&self, unit_id: &str) -> bool {
        self.skipped_units.contains_key(unit_id)
    }

    pub fn steering_snapshot(&self) -> SteerState {
        let mut skipped: Vec<String> = self.skipped_units.iter().map(|e| e.key().clone()).collect();
        skipped.sort();
        SteerState {
            paused: self.paused.load(Ordering::Acquire),
            abort_requested: self.abort_requested.load(Ordering::Acquire),
            k_override: self.k_override(),
            priority_queue: self.priority_units.lock().iter().cloned().collect(),
            skipped_units: skipped,
        }
    }

    pub fn set_phase(&self, unit_id: &str, phase: &str, attempt: u32, loc_before: usize) {
        self.units
            .entry(unit_id.to_string())
            .and_modify(|u| {
                u.phase = phase.to_string();
                u.attempt = attempt;
            })
            .or_insert_with(|| UnitState {
                phase: phase.to_string(),
                attempt,
                loc_before,
                loc_after: loc_before,
                hygiene_after: 0.0,
                last_blake3: String::new(),
            });
    }

    pub fn step_inflight(&self, delta: i64) {
        if delta >= 0 {
            self.in_flight_units
                .fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            let sub = (-delta) as u64;
            let _ = self
                .in_flight_units
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    Some(cur.saturating_sub(sub))
                });
        }
    }

    pub fn record_commit(
        &self,
        unit_id: &str,
        attempt: u32,
        loc_before: usize,
        loc_after: usize,
        hygiene_after: f64,
        blake3_hash: String,
    ) {
        self.committed_units.fetch_add(1, Ordering::Relaxed);
        if loc_after < loc_before {
            let saved = (loc_before - loc_after) as u64;
            let _ = self
                .loc_after
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    Some(cur.saturating_sub(saved))
                });
        } else if loc_after > loc_before {
            self.loc_after
                .fetch_add((loc_after - loc_before) as u64, Ordering::Relaxed);
        }
        self.units.insert(
            unit_id.to_string(),
            UnitState {
                phase: "committed".into(),
                attempt,
                loc_before,
                loc_after,
                hygiene_after,
                last_blake3: blake3_hash,
            },
        );
    }

    pub fn add_hardlinks(&self, count: u64) {
        if count > 0 {
            self.stage_hardlinked_files
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    pub fn count_dedup(&self, count: u64) {
        if count > 0 {
            self.telemetry_dedup_skipped
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    pub fn count_search(&self, cache_hit: bool) {
        self.research_queries.fetch_add(1, Ordering::Relaxed);
        if cache_hit {
            self.research_cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn digest(&self) -> StateDigest {
        StateDigest {
            tracked_units: self.units.len(),
            committed_units: self.committed_units.load(Ordering::Relaxed),
            in_flight_units: self.in_flight_units.load(Ordering::Relaxed),
            blocked_units: self.blocked_units.len(),
            loc_before: self.loc_before.load(Ordering::Relaxed),
            loc_after: self.loc_after.load(Ordering::Relaxed),
            stage_hardlinked_files: self.stage_hardlinked_files.load(Ordering::Relaxed),
            telemetry_dedup_skipped: self.telemetry_dedup_skipped.load(Ordering::Relaxed),
            research_queries: self.research_queries.load(Ordering::Relaxed),
            research_cache_hits: self.research_cache_hits.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn probe_43_state_counters() {
        let st = RunState::new(1000);
        std::thread::scope(|s| {
            for i in 0..8usize {
                let c = st.clone();
                s.spawn(move || {
                    c.step_inflight(1);
                    c.set_phase(&format!("mod_{i}"), "streaming", 1, 100);
                    c.record_commit(&format!("mod_{i}"), 1, 100, 80, 0.92, format!("h{i}"));
                    c.step_inflight(-1);
                    c.add_hardlinks(10);
                    c.count_search(i % 2 == 0);
                });
            }
        });

        let d = st.digest();
        assert_eq!(d.tracked_units, 8);
        assert_eq!(d.committed_units, 8);
        assert_eq!(d.in_flight_units, 0);
        assert_eq!(d.loc_before, 1000);
        assert_eq!(d.loc_after, 840);
        assert_eq!(d.stage_hardlinked_files, 80);
        assert_eq!(d.research_queries, 8);
        assert_eq!(d.research_cache_hits, 4);
    }

    #[tokio::test]
    async fn probe_65_steer_bus() {
        let st = RunState::new(500);
        st.steer_pause(true);
        st.steer_k(Some(6));
        st.steer_prioritize("hot_mod".into());
        st.steer_skip("legacy_mod".into());

        let snap = st.steering_snapshot();
        assert!(snap.paused);
        assert_eq!(snap.k_override, Some(6));
        assert!(st.is_skipped("legacy_mod"));
        assert_eq!(st.pop_prio().as_deref(), Some("hot_mod"));

        let st_c = st.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            st_c.steer_pause(false);
        });
        let t0 = Instant::now();
        st.await_resume().await;
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(!st.steering_snapshot().paused);
    }

    #[test]
    fn probe_78_prio_preserve() {
        let st = RunState::new(100);
        st.steer_prioritize("blocked_child".into());
        st.steer_prioritize("ready_leaf".into());
        assert_eq!(
            st.pop_ready(|uid| uid == "ready_leaf").as_deref(),
            Some("ready_leaf")
        );
        // Unready unit must remain queued until its dependencies become ready.
        assert_eq!(st.pop_ready(|_| false), None);
        assert_eq!(
            st.pop_ready(|uid| uid == "blocked_child").as_deref(),
            Some("blocked_child")
        );
        st.mark_blocked("orphan_u", "dependency skipped");
        assert_eq!(st.blocked_count(), 1);
    }
}
