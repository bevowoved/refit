use crate::types::{LlmCallMetrics, ProxySnapshot};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const MAX_TIMING_SAMPLES: usize = 4096;
const KEEPWARM_MAX_PINGS: u32 = 11;

struct RollingEntry {
    req_id: u64,
    at: Instant,
    uncached: u64,
    total: u64,
}

#[derive(Default)]
struct KeepWarmState {
    last_activity: Option<Instant>,
    last_ping: Option<Instant>,
    pings_sent: u32,
    disabled_reason: Option<&'static str>,
}

#[derive(Default)]
pub(super) struct PacerState {
    window: VecDeque<RollingEntry>,
    uncached_tpm_cap: u64,
    total_tpm_cap: u64,
    rpm_cap: u64,
    http_429_count: u64,
    http_5xx_count: u64,
    consecutive_outage: u32,
    circuit_state: String,
    circuit_open_until: Option<Instant>,
    circuit_pauses: u64,
    prefix_hashes: HashMap<String, String>,
    prefix_drift_events: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_completion_tokens: u64,
    total_uncached_tokens: u64,
    queue_ms_samples: VecDeque<f64>,
    decode_tps_samples: VecDeque<f64>,
    pub(super) requests_sent: u64,
    retries_sent: u64,
    keep_warm: KeepWarmState,
}

impl PacerState {
    pub(super) fn new() -> Self {
        Self {
            uncached_tpm_cap: 44_000,
            total_tpm_cap: 440_000,
            rpm_cap: 500,
            circuit_state: "CLOSED".into(),
            queue_ms_samples: VecDeque::with_capacity(256),
            decode_tps_samples: VecDeque::with_capacity(256),
            ..Self::default()
        }
    }

    fn prune(&mut self) {
        let window = Duration::from_secs(60);
        while self.window.front().is_some_and(|e| e.at.elapsed() >= window) {
            self.window.pop_front();
        }
    }

    fn rolling_totals(&mut self) -> (u64, u64, u64) {
        self.prune();
        let (mut unc, mut tot) = (0u64, 0u64);
        for e in &self.window {
            unc = unc.saturating_add(e.uncached);
            tot = tot.saturating_add(e.total);
        }
        (unc, tot, self.window.len() as u64)
    }

    pub(super) fn snapshot(&mut self) -> ProxySnapshot {
        let (unc, tot, rpm) = self.rolling_totals();
        ProxySnapshot {
            circuit_state: self.circuit_state.clone(),
            uncached_tpm_rolling: unc,
            uncached_tpm_cap: self.uncached_tpm_cap,
            total_tpm_rolling: tot,
            total_tpm_cap: self.total_tpm_cap,
            rpm_rolling: rpm,
            rpm_cap: self.rpm_cap,
            http_429_count: self.http_429_count,
            http_5xx_count: self.http_5xx_count,
            circuit_pauses: self.circuit_pauses,
            prefix_drift_events: self.prefix_drift_events,
            total_prompt_tokens: self.total_prompt_tokens,
            total_cached_tokens: self.total_cached_tokens,
            total_completion_tokens: self.total_completion_tokens,
            total_uncached_tokens: self.total_uncached_tokens,
        }
    }

    pub(super) fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) {
        (
            self.queue_ms_samples.iter().copied().collect(),
            self.decode_tps_samples.iter().copied().collect(),
        )
    }

    pub(super) fn allow_retry(&mut self) -> bool {
        let budget = (self.requests_sent / 5).max(4);
        if self.retries_sent >= budget {
            return false;
        }
        self.retries_sent += 1;
        true
    }

    pub(super) fn try_reserve(&mut self, req_id: u64, est_unc: u64, est_tot: u64) -> Option<Duration> {
        let mut half_open = false;
        if let Some(until) = self.circuit_open_until {
            let now = Instant::now();
            if now < until {
                return Some(until.saturating_duration_since(now));
            }
            self.circuit_open_until = None;
            self.circuit_state = "HALF_OPEN".into();
            half_open = true;
        }

        let (unc, tot, rpm) = self.rolling_totals();
        let req_unc = est_unc.min(self.uncached_tpm_cap);
        let req_tot = est_tot.min(self.total_tpm_cap);
        let empty = self.window.is_empty();

        let fits_unc = unc.saturating_add(req_unc) <= self.uncached_tpm_cap || empty;
        let fits_tot = tot.saturating_add(req_tot) <= self.total_tpm_cap || empty;
        if fits_unc && fits_tot && rpm < self.rpm_cap {
            let entry = RollingEntry { req_id, at: Instant::now(), uncached: req_unc, total: req_tot };
            self.window.push_back(entry);
            return None;
        }
        if half_open {
            return Some(Duration::from_millis(500));
        }

        let wait = self.window.front().map_or(Duration::from_millis(500), |e| {
            let rem = Duration::from_secs(60).saturating_sub(e.at.elapsed());
            rem.clamp(Duration::from_millis(250), Duration::from_secs(5))
        });
        Some(wait)
    }

    pub(super) fn free_budget(&mut self, req_id: u64) {
        if let Some(pos) = self.window.iter().position(|e| e.req_id == req_id) {
            self.window.remove(pos);
        }
    }

    pub(super) fn apply_status(
        &mut self,
        status: u16,
        limit_tpm: Option<u64>,
        limit_rpm: Option<u64>,
        retry_after: Option<u64>,
    ) -> Option<Value> {
        if let Some(v) = limit_tpm {
            self.uncached_tpm_cap = ((v as f64) * 0.88).round().max(10_000.0) as u64;
        }
        if let Some(v) = limit_rpm {
            self.rpm_cap = ((v as f64) * 0.80).round().max(30.0) as u64;
        }

        if crate::net::is_quota_exhausted(status) {
            return Some(self.on_432());
        }
        if status == 429 {
            return self.on_429(retry_after.unwrap_or(5));
        }
        if status >= 500 {
            return Some(self.on_5xx(status));
        }
        if (200..300).contains(&status) {
            return self.on_2xx();
        }
        None
    }

    fn on_432(&mut self) -> Value {
        self.circuit_state = "PERMANENT_OPEN".into();
        self.circuit_open_until = None;
        self.circuit_pauses += 1;
        json!({"state": "PERMANENT_OPEN", "status": 432, "reason": "quota_exhausted"})
    }

    fn on_429(&mut self, retry_after: u64) -> Option<Value> {
        self.http_429_count += 1;
        if retry_after > 10 {
            self.consecutive_outage += 1;
        }
        if self.consecutive_outage < 3 {
            return None;
        }
        let pause_s = retry_after.clamp(15, 60);
        self.open_circuit(pause_s);
        Some(json!({"state": "OPEN", "pause_s": pause_s, "reason": "429_storm"}))
    }

    fn on_5xx(&mut self, status: u16) -> Value {
        self.http_5xx_count += 1;
        self.consecutive_outage += 1;
        let pause_s = (5u64 * (1u64 << self.consecutive_outage.min(3))).min(60);
        self.open_circuit(pause_s);
        json!({"state": "OPEN", "pause_s": pause_s, "status": status})
    }

    fn on_2xx(&mut self) -> Option<Value> {
        let was_open = self.circuit_state != "CLOSED";
        self.consecutive_outage = 0;
        self.circuit_state = "CLOSED".into();
        self.circuit_open_until = None;
        self.keep_warm.last_activity = Some(Instant::now());
        self.keep_warm.pings_sent = 0;
        was_open.then(|| json!({"state": "CLOSED"}))
    }

    fn open_circuit(&mut self, pause_s: u64) {
        self.circuit_state = "OPEN".into();
        self.circuit_open_until = Some(Instant::now() + Duration::from_secs(pause_s));
        self.circuit_pauses += 1;
    }

    pub(super) fn record_usage(&mut self, req_id: u64, m: &LlmCallMetrics) -> u64 {
        self.total_prompt_tokens = self.total_prompt_tokens.saturating_add(m.prompt_tokens);
        self.total_cached_tokens = self.total_cached_tokens.saturating_add(m.cached_tokens);
        self.total_completion_tokens =
            self.total_completion_tokens.saturating_add(m.completion_tokens);
        self.total_uncached_tokens = self.total_uncached_tokens.saturating_add(m.uncached_tokens);
        push_sample(&mut self.queue_ms_samples, m.queue_ms);
        push_sample(&mut self.decode_tps_samples, m.decode_tps);

        let actual_tot = m.prompt_tokens.saturating_add(m.completion_tokens);
        if let Some(slot) = self.window.iter_mut().find(|e| e.req_id == req_id) {
            slot.uncached = m.uncached_tokens;
            slot.total = actual_tot;
        } else if m.uncached_tokens > 0 || actual_tot > 0 {
            let entry = RollingEntry {
                req_id,
                at: Instant::now(),
                uncached: m.uncached_tokens,
                total: actual_tot,
            };
            self.window.push_back(entry);
        }
        self.rolling_totals().0
    }

    pub(super) fn check_drift(&mut self, lang_key: &str, hash: String) -> bool {
        match self.prefix_hashes.get(lang_key) {
            None => {
                self.prefix_hashes.insert(lang_key.to_string(), hash);
                false
            }
            Some(prev) if prev == &hash => false,
            Some(_) => {
                self.prefix_drift_events += 1;
                true
            }
        }
    }

    pub(super) fn keepwarm_check(
        &self,
        ttl_ms: u64,
        in_flight: u64,
        due_fn: fn(u64, u64, u64, u64, u32, u32, bool) -> bool,
    ) -> bool {
        if self.circuit_state == "OPEN" || self.circuit_state == "PERMANENT_OPEN" {
            return false;
        }
        let Some(last) = self.keep_warm.last_activity else { return false };

        let idle = last.elapsed().as_millis() as u64;
        let since = self.keep_warm.last_ping.map_or(idle, |t| t.elapsed().as_millis() as u64);
        let dis = self.keep_warm.disabled_reason.is_some();
        due_fn(idle, since, ttl_ms, in_flight, self.keep_warm.pings_sent, KEEPWARM_MAX_PINGS, dis)
    }

    pub(super) fn keepwarm_done(&mut self, cached: Option<u64>) {
        let Some(tokens) = cached else {
            self.keep_warm.disabled_reason = Some("ping_failed");
            return;
        };
        self.keep_warm.last_ping = Some(Instant::now());
        self.keep_warm.pings_sent += 1;
        if tokens == 0 {
            self.keep_warm.disabled_reason = Some("zero_cached_tokens");
        }
    }
}

fn push_sample(q: &mut VecDeque<f64>, val: f64) {
    if val <= 0.0 {
        return;
    }
    if q.len() >= MAX_TIMING_SAMPLES {
        q.pop_front();
    }
    q.push_back(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_classifies_432_permanent_distinct_from_429() {
        let mut p = PacerState::new();
        let ev429 = p.apply_status(429, None, None, Some(2));
        assert!(ev429.is_none());
        assert_eq!(p.snapshot().circuit_state, "CLOSED");
        assert_eq!(p.snapshot().http_429_count, 1);

        let ev432 = p.apply_status(432, None, None, None).unwrap();
        assert_eq!(ev432["state"], "PERMANENT_OPEN");
        assert_eq!(ev432["reason"], "quota_exhausted");
        assert_eq!(p.snapshot().circuit_state, "PERMANENT_OPEN");
        assert_eq!(p.snapshot().circuit_pauses, 1);
    }
}
