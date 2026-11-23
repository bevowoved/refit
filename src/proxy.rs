use crate::mvcc::MvccStore;
use crate::platform::{
    stream_http, is_retryable, retry_ms, backoff_ms,
};
use crate::state::RunState;
use crate::telemetry::TelemetryHub;
use crate::types::{LlmCallMetrics, ProxySnapshot};
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use dashmap::DashMap;
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

const MAX_PROMPT_CACHE_KEY_LEN: usize = 1024;
const KEEPWARM_MAX_PINGS: u32 = 11;
const DEFAULT_SSE_IDLE_TIMEOUT_MS: u64 = 45_000;
const MAX_TIMING_SAMPLES: usize = 4096;

fn stall_ms() -> u64 {
    std::env::var("REFIT_SSE_STALL_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|v| v.clamp(100, 300_000))
        .unwrap_or(DEFAULT_SSE_IDLE_TIMEOUT_MS)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Short,
    Long,
    None,
}

impl CacheMode {
    pub fn parse_mode(explicit: Option<&str>) -> Self {
        let raw = explicit
            .map(String::from)
            .or_else(|| std::env::var("REFIT_CACHE_RETENTION").ok())
            .unwrap_or_else(|| "short".into());
        match raw.trim().to_ascii_lowercase().as_str() {
            "long" | "1h" | "24h" => Self::Long,
            "none" | "off" => Self::None,
            _ => Self::Short,
        }
    }

    pub fn ttl_ms(self, upstream_base_url: &str) -> u64 {
        let u = upstream_base_url.to_ascii_lowercase();
        match self {
            Self::None => 0,
            Self::Long
                if u.contains("moonshot.ai")
                    || u.contains("kimi.ai")
                    || u.contains("anthropic.com") =>
            {
                3_600_000
            }
            Self::Long if u.contains("openai.com") => 1_800_000,
            _ => 300_000,
        }
    }
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_base_url: String,
    pub api_key: String,
    pub model_override: String,
    pub ui_index_path: PathBuf,
    pub static_prefix_override: Option<String>,
    pub cache_retention: CacheMode,
    pub keep_warm: bool,
    pub auth_token: Option<String>,
    pub lang_prefixes: BTreeMap<String, String>,
    pub preserve_tools: bool,
}

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

pub fn keepwarm_due(
    idle_ms: u64,
    since_last_ping_ms: u64,
    ttl_ms: u64,
    in_flight: u64,
    pings_sent: u32,
    max_pings: u32,
    disabled: bool,
) -> bool {
    if disabled || ttl_ms < 60_000 || in_flight > 0 || pings_sent >= max_pings {
        return false;
    }
    let trigger_ms = (ttl_ms * 80) / 100;
    let expiry_guard_ms = ttl_ms.saturating_sub(15_000);
    idle_ms >= trigger_ms && idle_ms < expiry_guard_ms && since_last_ping_ms >= trigger_ms
}

struct PacerState {
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
    requests_sent: u64,
    retries_sent: u64,
    keep_warm: KeepWarmState,
}

impl Default for PacerState {
    fn default() -> Self {
        Self {
            window: VecDeque::new(),
            uncached_tpm_cap: 44_000,
            total_tpm_cap: 440_000,
            rpm_cap: 500,
            http_429_count: 0,
            http_5xx_count: 0,
            consecutive_outage: 0,
            circuit_state: "CLOSED".to_string(),
            circuit_open_until: None,
            circuit_pauses: 0,
            prefix_hashes: HashMap::new(),
            prefix_drift_events: 0,
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
            total_completion_tokens: 0,
            total_uncached_tokens: 0,
            queue_ms_samples: VecDeque::with_capacity(256),
            decode_tps_samples: VecDeque::with_capacity(256),
            requests_sent: 0,
            retries_sent: 0,
            keep_warm: KeepWarmState::default(),
        }
    }
}

impl PacerState {
    fn prune(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(60);
        while let Some(front) = self.window.front() {
            if front.at < cutoff {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }

    fn rolling_totals(&mut self) -> (u64, u64, u64) {
        self.prune();
        let mut unc = 0u64;
        let mut tot = 0u64;
        for e in &self.window {
            unc += e.uncached;
            tot += e.total;
        }
        (unc, tot, self.window.len() as u64)
    }

    fn push_sample(q: &mut VecDeque<f64>, val: f64) {
        if q.len() >= MAX_TIMING_SAMPLES {
            q.pop_front();
        }
        q.push_back(val);
    }

    fn allow_retry(&mut self) -> bool {
        let allowed_budget = (self.requests_sent / 5).max(4);
        if self.retries_sent < allowed_budget {
            self.retries_sent += 1;
            true
        } else {
            false
        }
    }
}

pub struct ProxyState {
    cfg: ProxyConfig,
    client: reqwest::Client,
    telemetry: TelemetryHub,
    pacer: Mutex<PacerState>,
    module_metrics: DashMap<String, LlmCallMetrics>,
    in_flight: AtomicU64,
    req_seq: AtomicU64,
    repo_root: Mutex<Option<PathBuf>>,
    shared_state: Mutex<Option<Arc<RunState>>>,
    ui_cache: Mutex<Option<(u64, String)>>,
}

impl ProxyState {
    pub fn bind_repo(&self, root: PathBuf) {
        *self.repo_root.lock() = Some(root);
    }

    pub fn attach_state(&self, st: Arc<RunState>) {
        *self.shared_state.lock() = Some(st);
    }

    pub fn snapshot(&self) -> ProxySnapshot {
        let mut p = self.pacer.lock();
        let (unc, tot, rpm) = p.rolling_totals();
        ProxySnapshot {
            circuit_state: p.circuit_state.clone(),
            uncached_tpm_rolling: unc,
            uncached_tpm_cap: p.uncached_tpm_cap,
            total_tpm_rolling: tot,
            total_tpm_cap: p.total_tpm_cap,
            rpm_rolling: rpm,
            rpm_cap: p.rpm_cap,
            http_429_count: p.http_429_count,
            http_5xx_count: p.http_5xx_count,
            circuit_pauses: p.circuit_pauses,
            prefix_drift_events: p.prefix_drift_events,
            total_prompt_tokens: p.total_prompt_tokens,
            total_cached_tokens: p.total_cached_tokens,
            total_completion_tokens: p.total_completion_tokens,
            total_uncached_tokens: p.total_uncached_tokens,
        }
    }

    pub fn take_metrics(&self, module_id: &str) -> Option<LlmCallMetrics> {
        self.module_metrics.remove(module_id).map(|(_, v)| v)
    }

    pub fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) {
        let p = self.pacer.lock();
        (
            p.queue_ms_samples.iter().copied().collect(),
            p.decode_tps_samples.iter().copied().collect(),
        )
    }

    fn is_authorized(&self, headers: &HeaderMap) -> bool {
        let Some(ref expected) = self.cfg.auth_token else {
            return true;
        };
        if expected.is_empty() {
            return true;
        }
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(|tok| tok.trim() == expected)
            .unwrap_or(false)
    }

    async fn await_budget(&self, req_id: u64, est_uncached: u64, est_total: u64) {
        loop {
            let sleep_dur = {
                let mut p = self.pacer.lock();
                if let Some(until) = p.circuit_open_until {
                    let now = Instant::now();
                    if now < until {
                        Some(until - now)
                    } else {
                        p.circuit_open_until = None;
                        p.circuit_state = "HALF_OPEN".into();
                        // Fall through to enforce TPM/RPM budget & record reservation in window.
                        let (unc, tot, rpm) = p.rolling_totals();
                        let req_unc = est_uncached.min(p.uncached_tpm_cap);
                        let req_tot = est_total.min(p.total_tpm_cap);
                        if (unc + req_unc <= p.uncached_tpm_cap || p.window.is_empty())
                            && (tot + req_tot <= p.total_tpm_cap || p.window.is_empty())
                            && rpm + 1 <= p.rpm_cap
                        {
                            p.window.push_back(RollingEntry {
                                req_id,
                                at: Instant::now(),
                                uncached: req_unc,
                                total: req_tot,
                            });
                            None
                        } else {
                            Some(Duration::from_millis(500))
                        }
                    }
                } else {
                    let (unc, tot, rpm) = p.rolling_totals();
                    // Clamp single-request admission to cap so huge SCCs never deadlock forever.
                    let req_unc = est_uncached.min(p.uncached_tpm_cap);
                    let req_tot = est_total.min(p.total_tpm_cap);
                    if (unc + req_unc <= p.uncached_tpm_cap || p.window.is_empty())
                        && (tot + req_tot <= p.total_tpm_cap || p.window.is_empty())
                        && rpm + 1 <= p.rpm_cap
                    {
                        p.window.push_back(RollingEntry {
                            req_id,
                            at: Instant::now(),
                            uncached: req_unc,
                            total: req_tot,
                        });
                        None
                    } else {
                        let wait = p
                            .window
                            .front()
                            .map(|e| {
                                let age = e.at.elapsed();
                                Duration::from_secs(60)
                                    .saturating_sub(age)
                                    .min(Duration::from_secs(5))
                                    .max(Duration::from_millis(250))
                            })
                            .unwrap_or(Duration::from_millis(500));
                        Some(wait)
                    }
                }
            };

            match sleep_dur {
                None => return,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
    }

    fn free_budget(&self, req_id: u64) {
        let mut p = self.pacer.lock();
        if let Some(pos) = p.window.iter().position(|e| e.req_id == req_id) {
            p.window.remove(pos);
        }
    }

    fn apply_headers(&self, status: u16, headers: &reqwest::header::HeaderMap) {
        let mut p = self.pacer.lock();
        if let Some(v) = header_u64(headers, "x-ratelimit-limit-tokens-minute") {
            p.uncached_tpm_cap = ((v as f64) * 0.88).round().max(10_000.0) as u64;
        }
        if let Some(v) = header_u64(headers, "x-ratelimit-limit-requests-minute") {
            p.rpm_cap = ((v as f64) * 0.80).round().max(30.0) as u64;
        }

        if status == 429 {
            p.http_429_count += 1;
            let retry_after = header_u64(headers, "retry-after").unwrap_or(5);
            if retry_after > 10 {
                p.consecutive_outage += 1;
            }
            if p.consecutive_outage >= 3 {
                let pause_s = retry_after.clamp(15, 60);
                p.circuit_state = "OPEN".into();
                p.circuit_open_until = Some(Instant::now() + Duration::from_secs(pause_s));
                p.circuit_pauses += 1;
                drop(p);
                self.telemetry.emit(
                    "circuit_state",
                    None,
                    None,
                    None,
                    None,
                    Some(serde_json::json!({"state": "OPEN", "pause_s": pause_s, "reason": "429_storm"})),
                    None,
                );
            }
        } else if status >= 500 {
            p.http_5xx_count += 1;
            p.consecutive_outage += 1;
            let pause_s = (5u64 * (1u64 << p.consecutive_outage.min(3))).min(60);
            p.circuit_state = "OPEN".into();
            p.circuit_open_until = Some(Instant::now() + Duration::from_secs(pause_s));
            p.circuit_pauses += 1;
            drop(p);
            self.telemetry.emit(
                "circuit_state",
                None,
                None,
                None,
                None,
                Some(serde_json::json!({"state": "OPEN", "pause_s": pause_s, "status": status})),
                None,
            );
        } else if (200..300).contains(&status) {
            let was_open = p.circuit_state != "CLOSED";
            p.consecutive_outage = 0;
            p.circuit_state = "CLOSED".into();
            p.circuit_open_until = None;
            p.keep_warm.last_activity = Some(Instant::now());
            p.keep_warm.pings_sent = 0;
            if was_open {
                drop(p);
                self.telemetry.emit(
                    "circuit_state",
                    None,
                    None,
                    None,
                    None,
                    Some(serde_json::json!({"state": "CLOSED"})),
                    None,
                );
            }
        }
    }

    /// Reconcile actual token usage against the pre-request budget reservation.
    fn record_usage(&self, req_id: u64, m: LlmCallMetrics) {
        let rolling_uncached = {
            let mut p = self.pacer.lock();
            p.total_prompt_tokens += m.prompt_tokens;
            p.total_cached_tokens += m.cached_tokens;
            p.total_completion_tokens += m.completion_tokens;
            p.total_uncached_tokens += m.uncached_tokens;
            if m.queue_ms > 0.0 {
                PacerState::push_sample(&mut p.queue_ms_samples, m.queue_ms);
            }
            if m.decode_tps > 0.0 {
                PacerState::push_sample(&mut p.decode_tps_samples, m.decode_tps);
            }
            let actual_tot = m.prompt_tokens + m.completion_tokens;
            if let Some(slot) = p.window.iter_mut().find(|e| e.req_id == req_id) {
                slot.uncached = m.uncached_tokens;
                slot.total = actual_tot;
            } else if m.uncached_tokens > 0 || actual_tot > 0 {
                p.window.push_back(RollingEntry {
                    req_id,
                    at: Instant::now(),
                    uncached: m.uncached_tokens,
                    total: actual_tot,
                });
            }
            p.rolling_totals().0
        };

        if !m.module_id.is_empty() {
            self.module_metrics.insert(m.module_id.clone(), m.clone());
        }
        self.telemetry.emit(
            "llm_done",
            if m.module_id.is_empty() {
                None
            } else {
                Some(m.module_id.clone())
            },
            None,
            None,
            Some(rolling_uncached),
            Some(serde_json::to_value(&m).unwrap_or(Value::Null)),
            None,
        );
    }
}

/// Infer the retry attempt number from `[attempt=N]` tags or turn count.
pub fn infer_attempt(payload: &Value) -> u32 {
    if let Some(msgs) = payload.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs.iter().rev() {
            if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                if let Some(pos) = text.find("[attempt=") {
                    let tail = &text[pos + 9..];
                    if let Some(end) = tail.find(']') {
                        if let Ok(n) = tail[..end].trim().parse::<u32>() {
                            return n.max(1);
                        }
                    }
                }
            }
        }
        if msgs.len() >= 4 {
            return ((msgs.len() / 2) as u32).max(2);
        }
    }
    1
}

pub fn guard_payload(
    payload: &mut Value,
    cfg: &ProxyConfig,
    module_id: &str,
    attempt: u32,
) {
    guard_lang(payload, cfg, module_id, attempt, None);
}

pub fn guard_lang(
    payload: &mut Value,
    cfg: &ProxyConfig,
    module_id: &str,
    attempt: u32,
    lang_hint: Option<&str>,
) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };
    if !cfg.model_override.is_empty() {
        obj.insert("model".into(), Value::String(cfg.model_override.clone()));
    }
    // Preserve adaptive caller reasoning_effort (`low` | `medium` | `high`) while blocking unsupported values.
    let req_effort = obj
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .unwrap_or("low");
    let safe_effort = match req_effort {
        "low" | "medium" | "high" => req_effort,
        _ => "low",
    };
    obj.insert("reasoning_effort".into(), Value::String(safe_effort.into()));
    obj.insert("reasoning_format".into(), Value::String("parsed".into()));
    obj.insert("temperature".into(), Value::from(1.0));
    obj.remove("disable_reasoning");
    obj.remove("top_p");
    obj.remove("clear_thinking");
    if let Some(mt) = obj.remove("max_tokens") {
        obj.entry("max_completion_tokens").or_insert(mt);
    }
    let max_comp = obj
        .get("max_completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(3200)
        .clamp(1, 6144);
    obj.insert("max_completion_tokens".into(), Value::from(max_comp));

    // Always deterministically sort tools when preserved so even non-ASRF OpenClaw tool turns avoid KV-prefix drift.
    let has_tools = obj
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if cfg.preserve_tools {
        if let Some(tools) = obj.get_mut("tools").and_then(|t| t.as_array_mut()) {
            tools.sort_by(|a, b| {
                let na = a["function"]["name"].as_str().unwrap_or("");
                let nb = b["function"]["name"].as_str().unwrap_or("");
                na.cmp(nb)
            });
        }
    }

    if !module_id.is_empty() && module_id != "probe" {
        if !cfg.preserve_tools {
            obj.remove("tools");
            obj.remove("tool_choice");
        }
        // Only override system prompt on ASRF turns (never clobber caller's tool-use instructions when tools are active).
        if !(cfg.preserve_tools && has_tools) {
            let chosen_prefix = lang_hint
                .and_then(|l| cfg.lang_prefixes.get(l))
                .or(cfg.static_prefix_override.as_ref());
            if let Some(static_sys) = chosen_prefix {
                if let Some(msgs) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
                    if let Some(first) = msgs.first_mut() {
                        if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                            first["content"] = Value::String(static_sys.clone());
                        }
                    }
                }
            }
        }
    }
    let is_stream = obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    if is_stream {
        obj.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
    }
    if attempt <= 1 || cfg.cache_retention == CacheMode::None {
        obj.remove("prompt_cache_key");
    } else if let Some(k) = obj
        .get("prompt_cache_key")
        .and_then(|v| v.as_str())
        .map(String::from)
    {
        if k.len() > MAX_PROMPT_CACHE_KEY_LEN {
            let clamped: String = k.chars().take(MAX_PROMPT_CACHE_KEY_LEN).collect();
            obj.insert("prompt_cache_key".into(), Value::String(clamped));
        }
    } else if !module_id.is_empty() && module_id != "probe" {
        // Synthesize fallback prompt_cache_key on retries if OpenClaw stripped the caller's key.
        obj.insert(
            "prompt_cache_key".into(),
            Value::String(format!("refit:auto:m:{module_id}")),
        );
    }

    let upstream_l = cfg.upstream_base_url.to_ascii_lowercase();
    if cfg.cache_retention == CacheMode::Long {
        if upstream_l.contains("moonshot.ai") || upstream_l.contains("kimi.ai") {
            obj.insert(
                "prompt_cache_options".into(),
                serde_json::json!({"mode": "implicit", "ttl": "1h"}),
            );
        } else if upstream_l.contains("openai.com") {
            obj.insert("prompt_cache_retention".into(), Value::String("24h".into()));
        }
    }
}

fn localip(origin: &HeaderValue) -> bool {
    let Ok(s) = origin.to_str() else {
        return false;
    };
    let l = s.to_ascii_lowercase();
    let rest = if let Some(r) = l.strip_prefix("http://") {
        r
    } else if let Some(r) = l.strip_prefix("https://") {
        r
    } else {
        return false;
    };
    if rest.contains('/') || rest.contains('@') || rest.contains('?') || rest.contains('#') {
        return false;
    }
    for host in ["127.0.0.1", "localhost", "[::1]"] {
        if let Some(after) = rest.strip_prefix(host) {
            if after.is_empty() {
                return true;
            }
            if let Some(port) = after.strip_prefix(':') {
                return !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit());
            }
        }
    }
    false
}

pub async fn start_proxy(
    cfg: ProxyConfig,
    telemetry: TelemetryHub,
) -> Result<(Arc<ProxyState>, tokio::task::JoinHandle<()>)> {
    let client = stream_http("cerebras-cloud-sdk/1.91.0 (refit/0.5.0)", 15, 600, 32)?;

    let state = Arc::new(ProxyState {
        cfg: cfg.clone(),
        client,
        telemetry,
        pacer: Mutex::new(PacerState::default()),
        module_metrics: DashMap::new(),
        in_flight: AtomicU64::new(0),
        req_seq: AtomicU64::new(1),
        repo_root: Mutex::new(std::env::current_dir().ok()),
        shared_state: Mutex::new(None),
        ui_cache: Mutex::new(None),
    });

    if cfg.keep_warm && cfg.static_prefix_override.is_some() {
        let kw_state = state.clone();
        tokio::spawn(async move {
            keepwarm_loop(kw_state).await;
        });
    }

    // Restrict CORS strictly to loopback origins.
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| localip(origin)))
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/health", get(handle_health))
        .route("/events", get(serve_events))
        .route("/api/timeline", get(timeline_get))
        .route("/api/diff/{tx_id}", get(serve_diff))
        .route("/api/steer", get(steer_get).post(steer_post))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(chat_handler))
        .layer(cors)
        .with_state(state.clone());

    let listener = TcpListener::bind(&cfg.listen_addr)
        .await
        .with_context(|| format!("bind proxy on {}", cfg.listen_addr))?;

    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            tracing::error!("proxy server stopped: {err}");
        }
    });

    Ok((state, handle))
}

async fn keepwarm_loop(state: Arc<ProxyState>) {
    let ttl_ms = state.cfg.cache_retention.ttl_ms(&state.cfg.upstream_base_url);
    if ttl_ms == 0 {
        return;
    }
    let Some(prefix) = state.cfg.static_prefix_override.clone() else {
        return;
    };
    let mut ticker = tokio::time::interval(Duration::from_secs(15));
    loop {
        ticker.tick().await;
        let should_ping = {
            let p = state.pacer.lock();
            if p.circuit_state == "OPEN" {
                continue;
            }
            let Some(last) = p.keep_warm.last_activity else {
                continue;
            };
            let idle_ms = last.elapsed().as_millis() as u64;
            let since_ping_ms = p
                .keep_warm
                .last_ping
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(idle_ms);
            keepwarm_due(
                idle_ms,
                since_ping_ms,
                ttl_ms,
                state.in_flight.load(Ordering::Relaxed),
                p.keep_warm.pings_sent,
                KEEPWARM_MAX_PINGS,
                p.keep_warm.disabled_reason.is_some(),
            )
        };
        if !should_ping {
            continue;
        }
        let url = format!(
            "{}/chat/completions",
            state.cfg.upstream_base_url.trim_end_matches('/')
        );
        let mut body = serde_json::json!({
            "model": state.cfg.model_override,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": "."}
            ],
            "stream": false,
            "max_completion_tokens": 1
        });
        guard_payload(&mut body, &state.cfg, "probe", 2);
        match state
            .client
            .post(&url)
            .bearer_auth(&state.cfg.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let v: Value = resp.json().await.unwrap_or(Value::Null);
                let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0);
                let mut p = state.pacer.lock();
                p.keep_warm.last_ping = Some(Instant::now());
                p.keep_warm.pings_sent += 1;
                if cached == 0 {
                    p.keep_warm.disabled_reason = Some("zero_cached_tokens");
                }
            }
            _ => {
                let mut p = state.pacer.lock();
                p.keep_warm.disabled_reason = Some("ping_failed");
            }
        }
    }
}

async fn serve_ui(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let sig = std::fs::metadata(&state.cfg.ui_index_path)
        .ok()
        .map(|m| {
            let mt = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            m.len() ^ mt.rotate_left(17)
        });
    if let Some(cur_sig) = sig {
        if let Some((cached_sig, html)) = state.ui_cache.lock().as_ref() {
            if *cached_sig == cur_sig {
                return (StatusCode::OK, Html(html.clone())).into_response();
            }
        }
    }
    match std::fs::read_to_string(&state.cfg.ui_index_path) {
        Ok(html) => {
            if let Some(cur_sig) = sig {
                *state.ui_cache.lock() = Some((cur_sig, html.clone()));
            }
            (StatusCode::OK, Html(html)).into_response()
        }
        Err(_) => (
            StatusCode::OK,
            Html("<html><body><h1>refit telemetry proxy active</h1></body></html>".to_string()),
        )
            .into_response(),
    }
}

fn err_401() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": "missing or invalid proxy bearer token"})),
    )
        .into_response()
}

async fn steer_get(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let snap = state
        .shared_state
        .lock()
        .as_ref()
        .map(|s| s.steering_snapshot());
    Json(serde_json::json!({
        "attached": snap.is_some(),
        "steering": snap
    }))
    .into_response()
}

async fn steer_post(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Json(cmd): Json<Value>,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let shared = state.shared_state.lock().clone();
    let Some(st) = shared else {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "no active run attached for live steering"})),
        )
            .into_response();
    };
    let action = cmd["action"].as_str().unwrap_or("");
    match action {
        "pause" => st.steer_pause(true),
        "resume" => st.steer_pause(false),
        "abort" => st.steer_abort(),
        "set_k" => st.steer_k(cmd["k"].as_u64().map(|v| v as usize)),
        "prioritize" => {
            if let Some(uid) = cmd["unit_id"].as_str() {
                st.steer_prioritize(uid.to_string());
            }
        }
        "skip" => {
            if let Some(uid) = cmd["unit_id"].as_str() {
                st.steer_skip(uid.to_string());
            }
        }
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("unknown steering action `{other}`")})),
            )
                .into_response();
        }
    }
    let snap = st.steering_snapshot();
    state.telemetry.emit(
        "live_steer",
        cmd["unit_id"].as_str().map(String::from),
        None,
        None,
        None,
        Some(serde_json::json!({"action": action, "steering": snap})),
        None,
    );
    (StatusCode::OK, Json(serde_json::json!({"ok": true, "steering": snap}))).into_response()
}

async fn handle_health(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(serde_json::json!({
        "ok": true,
        "proxy": state.snapshot()
    }))
}

async fn timeline_get(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let root = state.repo_root.lock().clone();
    let entries = root
        .as_deref()
        .and_then(|r| MvccStore::wal_timeline(r).ok())
        .unwrap_or_default();
    Json(serde_json::json!({ "entries": entries })).into_response()
}

async fn serve_diff(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Path(tx_id): Path<String>,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let root = state.repo_root.lock().clone();
    if let Some(ref r) = root {
        if let Ok(Some(report)) = MvccStore::diff_tx(r, &tx_id) {
            return (StatusCode::OK, Json(report)).into_response();
        }
    }
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": format!("tx `{tx_id}` not found") })),
    )
        .into_response()
}

async fn serve_events(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let query_ok = state
        .cfg
        .auth_token
        .as_deref()
        .map(|expected| {
            expected.is_empty()
                || params.get("token").map(|t| t.trim() == expected).unwrap_or(false)
        })
        .unwrap_or(true);
    if !state.is_authorized(&headers) && !query_ok {
        return err_401();
    }
    let history = state.telemetry.snapshot_events();
    let rx = state.telemetry.subscribe();

    let hist_stream = futures_util::stream::iter(history.into_iter().map(|ev| {
        let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
        Ok::<Event, Infallible>(Event::default().data(data))
    }));

    let live_stream = BroadcastStream::new(rx).filter_map(|item| async move {
        match item {
            Ok(ev) => {
                let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
                Some(Ok::<Event, Infallible>(Event::default().data(data)))
            }
            Err(_) => None,
        }
    });

    Sse::new(hist_stream.chain(live_stream))
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn handle_models(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let url = format!("{}/models", state.cfg.upstream_base_url.trim_end_matches('/'));
    let seq = state.req_seq.fetch_add(1, Ordering::Relaxed);
    let mut last_err = String::new();
    for attempt in 0..3u32 {
        match state
            .client
            .get(&url)
            .bearer_auth(&state.cfg.api_key)
            .timeout(Duration::from_secs(15))
            .send()
            .await
        {
            Ok(resp) => {
                let code = resp.status().as_u16();
                if is_retryable(code) && attempt < 2 {
                    let ra = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| retry_ms(s, 5_000));
                    let wait_ms = backoff_ms(attempt, 150, 5_000, ra, seq);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
                let status = StatusCode::from_u16(code).unwrap_or(StatusCode::OK);
                let bytes = resp.bytes().await.unwrap_or_default();
                return (status, [("content-type", "application/json")], bytes).into_response();
            }
            Err(err) => {
                last_err = err.to_string();
                if attempt < 2 {
                    let wait_ms = backoff_ms(attempt, 150, 5_000, None, seq);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
            }
        }
    }
    (
        StatusCode::BAD_GATEWAY,
        Json(serde_json::json!({"error": last_err})),
    )
        .into_response()
}

struct InFlightGuard(Arc<ProxyState>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn chat_handler(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    state.in_flight.fetch_add(1, Ordering::Relaxed);
    let _in_flight = InFlightGuard(state.clone());

    let mut module_id = headers
        .get("x-refit-module")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if module_id.is_empty() {
        module_id = find_module(&payload);
    }
    let attempt: u32 = headers
        .get("x-refit-attempt")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| infer_attempt(&payload));
    let lang_hint = headers.get("x-refit-lang").and_then(|v| v.to_str().ok());

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    guard_lang(&mut payload, &state.cfg, &module_id, attempt, lang_hint);
    let prefix_drift = check_drift(&state, &payload, lang_hint.unwrap_or("default"));

    let sys_bytes = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|a| a.first())
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.len())
        .unwrap_or(0);
    let body_bytes = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());
    let max_comp = payload
        .get("max_completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1600);
    // Dynamic byte-exact uncached token estimate.
    let dynamic_input_toks = (body_bytes.len().saturating_sub(sys_bytes) as u64 / 4).max(150);
    let est_out = (max_comp / 3).clamp(200, 1800);
    let est_total = (body_bytes.len() as u64 / 4).max(300) + est_out;
    let est_uncached = if attempt <= 1 {
        dynamic_input_toks + est_out
    } else {
        (dynamic_input_toks / 3).max(200) + est_out
    };

    let seq = state.req_seq.fetch_add(1, Ordering::Relaxed);
    state.await_budget(seq, est_uncached, est_total).await;

    let url = format!(
        "{}/chat/completions",
        state.cfg.upstream_base_url.trim_end_matches('/')
    );
    let started = Instant::now();
    {
        state.pacer.lock().requests_sent += 1;
    }

    let mut upstream_opt = None;
    let mut last_err = String::new();
    for retry_idx in 0..3u32 {
        let req_builder = state
            .client
            .post(&url)
            .bearer_auth(&state.cfg.api_key)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(body_bytes.clone());

        match req_builder.send().await {
            Ok(resp) => {
                let code = resp.status().as_u16();
                if is_retryable(code) && retry_idx < 2 && state.pacer.lock().allow_retry() {
                    let ra = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| retry_ms(s, 30_000));
                    state.apply_headers(code, resp.headers());
                    let wait_ms = backoff_ms(retry_idx, 250, 30_000, ra, seq);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
                upstream_opt = Some(resp);
                break;
            }
            Err(err) => {
                last_err = err.to_string();
                state.apply_headers(503, &reqwest::header::HeaderMap::new());
                if retry_idx < 2 && state.pacer.lock().allow_retry() {
                    let wait_ms = backoff_ms(retry_idx, 250, 10_000, None, seq);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
                break;
            }
        }
    }

    let Some(upstream) = upstream_opt else {
        state.free_budget(seq);
        return (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": last_err})),
        )
            .into_response();
    };

    let status_u16 = upstream.status().as_u16();
    let status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let up_headers = upstream.headers().clone();
    state.apply_headers(status_u16, &up_headers);

    let limit_tpm = header_u64(&up_headers, "x-ratelimit-limit-tokens-minute");
    let remaining_tpm = header_u64(&up_headers, "x-ratelimit-remaining-tokens-minute");
    let limit_rpm = header_u64(&up_headers, "x-ratelimit-limit-requests-minute");

    if !is_stream || status_u16 != 200 {
        let bytes = upstream.bytes().await.unwrap_or_default();
        let wall_ms = started.elapsed().as_millis() as u64;
        if let Ok(json) = serde_json::from_slice::<Value>(&bytes) {
            let mut m = parse_metrics(&json, &module_id, attempt, status_u16, wall_ms, prefix_drift);
            m.limit_tpm_minute = limit_tpm;
            m.remaining_tpm_minute = remaining_tpm;
            m.limit_rpm_minute = limit_rpm;
            state.record_usage(seq, m);
        } else {
            state.free_budget(seq);
        }
        return (status, [("content-type", "application/json")], bytes).into_response();
    }

    let state_clone = state.clone();
    let idle_ms = headers
        .get("x-refit-sse-stall-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|v| v.clamp(50, 300_000))
        .unwrap_or_else(stall_ms);
    let stream = tap_stream(
        upstream,
        state_clone,
        seq,
        module_id,
        attempt,
        status_u16,
        started,
        prefix_drift,
        limit_tpm,
        remaining_tpm,
        limit_rpm,
        idle_ms,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream error").into_response())
}

#[allow(clippy::too_many_arguments)]
fn tap_stream(
    upstream: reqwest::Response,
    state: Arc<ProxyState>,
    req_id: u64,
    module_id: String,
    attempt: u32,
    status_u16: u16,
    started: Instant,
    prefix_drift: bool,
    limit_tpm: Option<u64>,
    remaining_tpm: Option<u64>,
    limit_rpm: Option<u64>,
    idle_timeout_ms: u64,
) -> impl futures_util::Stream<Item = std::result::Result<Bytes, std::io::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut byte_stream = upstream.bytes_stream();
        // Raw byte buffer split on `b'\n'` so multi-byte UTF-8 chunks never drop SSE usage metrics.
        let mut byte_buf: Vec<u8> = Vec::with_capacity(4096);
        let mut last_usage: Option<Value> = None;
        let mut last_time: Option<Value> = None;
        let watchdog = Duration::from_millis(idle_timeout_ms.max(50));

        loop {
            let next_item = match tokio::time::timeout(watchdog, byte_stream.next()).await {
                Ok(item) => item,
                Err(_) => {
                    state.telemetry.emit(
                        "sse_stall",
                        if module_id.is_empty() { None } else { Some(module_id.clone()) },
                        None,
                        None,
                        None,
                        Some(serde_json::json!({
                            "idle_timeout_ms": idle_timeout_ms,
                            "attempt": attempt
                        })),
                        None,
                    );
                    let _ = tx
                        .send(Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!("upstream SSE stream stalled after {idle_timeout_ms}ms"),
                        )))
                        .await;
                    break;
                }
            };
            let Some(chunk_res) = next_item else {
                break;
            };
            match chunk_res {
                Ok(chunk) => {
                    byte_buf.extend_from_slice(&chunk);
                    while let Some(nl) = byte_buf.iter().position(|&b| b == b'\n') {
                        let raw_line: Vec<u8> = byte_buf.drain(..=nl).collect();
                        let line = String::from_utf8_lossy(&raw_line).trim().to_string();
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data != "[DONE]" {
                                if let Ok(v) = serde_json::from_str::<Value>(data) {
                                    if v.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
                                        last_usage = v.get("usage").cloned();
                                    }
                                    if v.get("time_info").map(|t| !t.is_null()).unwrap_or(false) {
                                        last_time = v.get("time_info").cloned();
                                    }
                                }
                            }
                        }
                    }
                    if tx.send(Ok(chunk)).await.is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx
                        .send(Err(std::io::Error::other(err.to_string())))
                        .await;
                    break;
                }
            }
        }

        if !byte_buf.is_empty() {
            let line = String::from_utf8_lossy(&byte_buf).trim().to_string();
            if let Some(data) = line.strip_prefix("data: ") {
                if data != "[DONE]" {
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        if v.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
                            last_usage = v.get("usage").cloned();
                        }
                        if v.get("time_info").map(|t| !t.is_null()).unwrap_or(false) {
                            last_time = v.get("time_info").cloned();
                        }
                    }
                }
            }
        }

        let wall_ms = started.elapsed().as_millis() as u64;
        let synthetic = serde_json::json!({
            "usage": last_usage,
            "time_info": last_time,
        });
        let mut m = parse_metrics(
            &synthetic,
            &module_id,
            attempt,
            status_u16,
            wall_ms,
            prefix_drift,
        );
        m.limit_tpm_minute = limit_tpm;
        m.remaining_tpm_minute = remaining_tpm;
        m.limit_rpm_minute = limit_rpm;
        state.record_usage(req_id, m);
    });

    tokio_stream::wrappers::ReceiverStream::new(rx)
}

fn check_drift(state: &ProxyState, payload: &Value, lang_key: &str) -> bool {
    let sys = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|arr| arr.first())
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if sys.len() < 120 {
        return false;
    }
    let tools = payload
        .get("tools")
        .map(|t| t.to_string())
        .unwrap_or_default();
    let hash = MvccStore::hash_bytes(format!("{sys}\n---\n{tools}").as_bytes());
    let mut p = state.pacer.lock();
    match p.prefix_hashes.get(lang_key) {
        None => {
            p.prefix_hashes.insert(lang_key.to_string(), hash);
            false
        }
        Some(prev) if prev == &hash => false,
        Some(_) => {
            p.prefix_drift_events += 1;
            true
        }
    }
}

fn parse_metrics(
    json: &Value,
    module_id: &str,
    attempt: u32,
    status: u16,
    wall_ms: u64,
    prefix_drift: bool,
) -> LlmCallMetrics {
    let usage = &json["usage"];
    let time_info = &json["time_info"];

    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    let reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0);
    let uncached_tokens = prompt_tokens.saturating_sub(cached_tokens) + completion_tokens;

    let queue_ms = time_info["queue_time"].as_f64().unwrap_or(0.0) * 1000.0;
    let prompt_ms = time_info["prompt_time"].as_f64().unwrap_or(0.0) * 1000.0;
    let completion_s = time_info["completion_time"].as_f64().unwrap_or(0.0);
    let completion_ms = completion_s * 1000.0;
    let decode_tps = if completion_s > 0.0005 && completion_tokens > 0 {
        (completion_tokens as f64) / completion_s
    } else {
        0.0
    };

    LlmCallMetrics {
        module_id: module_id.to_string(),
        attempt,
        http_status: status,
        wall_ms,
        queue_ms,
        prompt_ms,
        completion_ms,
        decode_tps,
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        reasoning_tokens,
        uncached_tokens,
        prefix_drift,
        limit_tpm_minute: None,
        remaining_tpm_minute: None,
        limit_rpm_minute: None,
    }
}

fn find_module(payload: &Value) -> String {
    let Some(msgs) = payload.get("messages").and_then(|m| m.as_array()) else {
        return String::new();
    };
    for msg in msgs {
        if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
            if let Some(pos) = text.find("Refactor unit `") {
                let rest = &text[pos + "Refactor unit `".len()..];
                if let Some(end) = rest.find('`') {
                    return rest[..end].to_string();
                }
            }
        }
    }
    String::new()
}

fn header_u64(headers: &reqwest::header::HeaderMap, name: &str) -> Option<u64> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn test_cfg(upstream: &str, retention: CacheMode) -> ProxyConfig {
        ProxyConfig {
            listen_addr: "127.0.0.1:0".into(),
            upstream_base_url: upstream.into(),
            api_key: "test-key".into(),
            model_override: "moonshotai-kimi-k3.1-pro-trial".into(),
            ui_index_path: PathBuf::from("ui/index.html"),
            static_prefix_override: Some("S".repeat(160)),
            cache_retention: retention,
            keep_warm: false,
            auth_token: None,
            lang_prefixes: BTreeMap::new(),
            preserve_tools: true,
        }
    }

    #[test]
    fn probe_01_wire_guard() {
        let mut cfg = test_cfg("https://api.cerebras.ai/v1", CacheMode::Short);
        cfg.preserve_tools = false;
        let mut p = serde_json::json!({
            "model": "wrong",
            "disable_reasoning": true,
            "top_p": 0.9,
            "clear_thinking": true,
            "max_tokens": 1000,
            "tools": [{"type": "function"}],
            "messages": [{"role": "system", "content": "bloated"}]
        });
        guard_payload(&mut p, &cfg, "mod_01", 1);
        assert_eq!(p["model"], "moonshotai-kimi-k3.1-pro-trial");
        assert_eq!(p["reasoning_effort"], "low");
        assert_eq!(p["reasoning_format"], "parsed");
        assert_eq!(p["temperature"], 1.0);
        assert!(p.get("disable_reasoning").is_none());
        assert!(p.get("top_p").is_none());
        assert!(p.get("tools").is_none());
        assert_eq!(p["max_completion_tokens"], 1000);
        assert_eq!(p["messages"][0]["content"].as_str().unwrap().len(), 160);
    }

    #[test]
    fn probe_02_cache_key_clamp() {
        let cfg = test_cfg("https://api.cerebras.ai/v1", CacheMode::Short);
        let mut p1 = serde_json::json!({"prompt_cache_key": "k".repeat(2000)});
        guard_payload(&mut p1, &cfg, "m1", 1);
        assert!(p1.get("prompt_cache_key").is_none());

        let mut p2 = serde_json::json!({"prompt_cache_key": "k".repeat(2000)});
        guard_payload(&mut p2, &cfg, "m1", 2);
        assert_eq!(p2["prompt_cache_key"].as_str().unwrap().len(), MAX_PROMPT_CACHE_KEY_LEN);
    }

    #[test]
    fn probe_03_cache_ttl_route() {
        let mut p_cb = serde_json::json!({});
        guard_payload(&mut p_cb, &test_cfg("https://api.cerebras.ai/v1", CacheMode::Long), "m", 2);
        assert!(p_cb.get("prompt_cache_options").is_none());
        assert!(p_cb.get("prompt_cache_retention").is_none());

        let mut p_kimi = serde_json::json!({});
        guard_payload(&mut p_kimi, &test_cfg("https://api.moonshot.ai/v1", CacheMode::Long), "m", 2);
        assert_eq!(p_kimi["prompt_cache_options"]["ttl"], "1h");

        let mut p_oa = serde_json::json!({});
        guard_payload(&mut p_oa, &test_cfg("https://api.openai.com/v1", CacheMode::Long), "m", 2);
        assert_eq!(p_oa["prompt_cache_retention"], "24h");
    }

    #[test]
    fn probe_04_keepwarm_window() {
        assert!(!keepwarm_due(100_000, 100_000, 300_000, 0, 0, 11, false));
        assert!(keepwarm_due(250_000, 250_000, 300_000, 0, 0, 11, false));
        assert!(!keepwarm_due(295_000, 295_000, 300_000, 0, 0, 11, false));
        assert!(!keepwarm_due(400_000, 400_000, 300_000, 0, 0, 11, false));
    }

    #[test]
    fn probe_05_keepwarm_limits() {
        assert!(!keepwarm_due(250_000, 250_000, 300_000, 1, 0, 11, false));
        assert!(!keepwarm_due(250_000, 250_000, 300_000, 0, 11, 11, false));
        assert!(!keepwarm_due(250_000, 250_000, 300_000, 0, 2, 11, true));
    }

    #[test]
    fn probe_06_prefix_drift() {
        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_pd_{}.jsonl", now_ms()))).unwrap();
        let st = ProxyState {
            cfg: test_cfg("https://api.cerebras.ai/v1", CacheMode::Short),
            client: reqwest::Client::new(),
            telemetry: tel,
            pacer: Mutex::new(PacerState::default()),
            module_metrics: DashMap::new(),
            in_flight: AtomicU64::new(0),
            req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(None),
            shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        };
        let tiny = serde_json::json!({"messages": [{"role": "system", "content": "OK"}]});
        assert!(!check_drift(&st, &tiny, "default"));

        let base = "A".repeat(150);
        let p1 = serde_json::json!({"messages": [{"role": "system", "content": base}]});
        let p2 = serde_json::json!({"messages": [{"role": "system", "content": "A".repeat(149) + "B"}]});
        assert!(!check_drift(&st, &p1, "default"));
        assert!(!check_drift(&st, &p1, "default"));
        assert!(check_drift(&st, &p2, "default"));
        assert_eq!(st.snapshot().prefix_drift_events, 1);
    }

    #[tokio::test]
    async fn probe_12_proxy_retry_503() {
        let up = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let up_addr = up.local_addr().unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let calls_c = calls.clone();

        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut sock, _) = up.accept().await.unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let n = calls_c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    let r = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0.02\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndown";
                    let _ = sock.write_all(r.as_bytes()).await;
                } else {
                    let body = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#;
                    let r = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
                    let _ = sock.write_all(r.as_bytes()).await;
                }
            }
        });

        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx12_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg(&format!("http://{up_addr}/v1"), CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        let (_st, handle) = start_proxy(cfg, tel).await.unwrap();

        let resp = reqwest::Client::new()
            .post(format!("http://{p_addr}/v1/chat/completions"))
            .json(&serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "stream": false}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        handle.abort();
    }

    #[tokio::test]
    async fn probe_13_proxy_skip_4xx() {
        let up = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let up_addr = up.local_addr().unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let calls_c = calls.clone();

        tokio::spawn(async move {
            let (mut sock, _) = up.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            calls_c.fetch_add(1, Ordering::SeqCst);
            let body = r#"{"error":"model_not_found"}"#;
            let r = format!("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            let _ = sock.write_all(r.as_bytes()).await;
        });

        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx13_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg(&format!("http://{up_addr}/v1"), CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        let (_st, handle) = start_proxy(cfg, tel).await.unwrap();

        let resp = reqwest::Client::new()
            .post(format!("http://{p_addr}/v1/chat/completions"))
            .json(&serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "stream": false}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        handle.abort();
    }

    #[test]
    fn probe_14_circuit_breaker() {
        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_cb_{}.jsonl", now_ms()))).unwrap();
        let st = ProxyState {
            cfg: test_cfg("https://api.cerebras.ai/v1", CacheMode::Short),
            client: reqwest::Client::new(),
            telemetry: tel,
            pacer: Mutex::new(PacerState::default()),
            module_metrics: DashMap::new(),
            in_flight: AtomicU64::new(0),
            req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(None),
            shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        };
        let empty = reqwest::header::HeaderMap::new();
        st.apply_headers(503, &empty);
        assert_eq!(st.snapshot().circuit_state, "OPEN");
        assert_eq!(st.snapshot().circuit_pauses, 1);
        st.apply_headers(200, &empty);
        assert_eq!(st.snapshot().circuit_state, "CLOSED");
    }

    #[test]
    fn probe_15_retry_budget() {
        let mut p = PacerState::default();
        p.requests_sent = 50;
        let mut allowed = 0;
        for _ in 0..30 {
            if p.allow_retry() {
                allowed += 1;
            }
        }
        assert_eq!(allowed, 10);
    }

    #[tokio::test]
    async fn probe_51_sse_watchdog() {
        let up = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let up_addr = up.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut sock, _) = up.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
            let frame = format!("{:x}\r\n{}\r\n", chunk.len(), chunk);
            let _ = sock.write_all(hdr.as_bytes()).await;
            let _ = sock.write_all(frame.as_bytes()).await;
            let _ = sock.flush().await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        });

        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx51_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg(&format!("http://{up_addr}/v1"), CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        let (_st, handle) = start_proxy(cfg, tel).await.unwrap();

        let started = Instant::now();
        let resp = reqwest::Client::new()
            .post(format!("http://{p_addr}/v1/chat/completions"))
            .header("x-refit-sse-stall-ms", "120")
            .json(&serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "stream": true}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let mut s = resp.bytes_stream();
        let mut saw_err = false;
        while let Some(item) = s.next().await {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "SSE watchdog must surface error when upstream stalls");
        assert!(started.elapsed() < Duration::from_secs(2), "watchdog must abort promptly, not hang");
        handle.abort();
    }

    #[tokio::test]
    async fn probe_63_proxy_timeline() {
        let dir = std::env::temp_dir().join(format!("refit_prx63_{}_{}", std::process::id(), now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let fpath = dir.join("app.py");
        std::fs::write(&fpath, "def f():\n    return 1\n").unwrap();
        let h0 = MvccStore::hash_file(&fpath).unwrap();
        let store = MvccStore::open(&dir, "prx63").unwrap();
        let mut exp = std::collections::BTreeMap::new();
        exp.insert("app.py".into(), h0);
        let mut next = std::collections::BTreeMap::new();
        next.insert("app.py".into(), b"def f():\n    return 42\n".to_vec());
        let tx_id = match store.commit_edits("app", &exp, &next).unwrap() {
            crate::mvcc::CommitOutcome::Committed { tx_id, .. } => tx_id,
            other => panic!("expected commit, got {other:?}"),
        };

        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx63_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg("http://127.0.0.1:1/v1", CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        let (st, handle) = start_proxy(cfg, tel).await.unwrap();
        st.bind_repo(dir.clone());

        let client = reqwest::Client::new();
        let tl: Value = client
            .get(format!("http://{p_addr}/api/timeline"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(tl["entries"][0]["tx_id"], serde_json::json!(tx_id));

        let diff_resp = client
            .get(format!("http://{p_addr}/api/diff/{tx_id}"))
            .send()
            .await
            .unwrap();
        assert_eq!(diff_resp.status().as_u16(), 200);
        let diff_json: Value = diff_resp.json().await.unwrap();
        assert!(diff_json["files"][0]["unified_diff"].as_str().unwrap().contains("+    return 42"));

        let missing = client
            .get(format!("http://{p_addr}/api/diff/tx-does-not-exist"))
            .send()
            .await
            .unwrap();
        assert_eq!(missing.status().as_u16(), 404);
        handle.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn probe_73_proxy_steer_api() {
        let up = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let up_addr = up.local_addr().unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let calls_c = calls.clone();
        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut sock, _) = up.accept().await.unwrap();
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf).await;
                let n = calls_c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    let r = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0.01\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbusy";
                    let _ = sock.write_all(r.as_bytes()).await;
                } else {
                    let body = r#"{"data":[{"id":"kimi"}]}"#;
                    let r = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
                    let _ = sock.write_all(r.as_bytes()).await;
                }
            }
        });

        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);
        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx73_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg(&format!("http://{up_addr}/v1"), CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        let (st, handle) = start_proxy(cfg, tel).await.unwrap();
        let shared = RunState::new(100);
        st.attach_state(shared.clone());

        let client = reqwest::Client::new();
        let m_resp = client.get(format!("http://{p_addr}/v1/models")).send().await.unwrap();
        assert_eq!(m_resp.status().as_u16(), 200);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let s1: Value = client
            .post(format!("http://{p_addr}/api/steer"))
            .json(&serde_json::json!({"action": "set_k", "k": 8}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(s1["steering"]["k_override"], serde_json::json!(8));
        assert_eq!(shared.k_override(), Some(8));

        let _ = client
            .post(format!("http://{p_addr}/api/steer"))
            .json(&serde_json::json!({"action": "prioritize", "unit_id": "core_u"}))
            .send()
            .await
            .unwrap();
        assert_eq!(shared.pop_prio().as_deref(), Some("core_u"));
        handle.abort();
    }

    #[tokio::test]
    async fn probe_84_proxy_tools_tpm() {
        let cfg = test_cfg("https://api.cerebras.ai/v1", CacheMode::Short);
        let mut p = serde_json::json!({
            "reasoning_effort": "medium",
            "tools": [
                {"type": "function", "function": {"name": "z_tool"}},
                {"type": "function", "function": {"name": "a_tool"}}
            ],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Refactor unit `m1`"},
                {"role": "assistant", "content": "bad"},
                {"role": "user", "content": "[attempt=3] retry"}
            ],
            "prompt_cache_key": "refit:r1:m:m1"
        });
        let inferred = infer_attempt(&p);
        assert_eq!(inferred, 3);
        guard_payload(&mut p, &cfg, "m1", inferred);
        assert_eq!(p["reasoning_effort"], "medium");
        assert_eq!(p["tools"][0]["function"]["name"], "a_tool");
        assert_eq!(p["prompt_cache_key"], "refit:r1:m:m1");

        // Verify TPM refund reconciles overestimated reservation down to actual usage.
        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx84_{}.jsonl", now_ms()))).unwrap();
        let st = ProxyState {
            cfg,
            client: reqwest::Client::new(),
            telemetry: tel,
            pacer: Mutex::new(PacerState::default()),
            module_metrics: DashMap::new(),
            in_flight: AtomicU64::new(0),
            req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(None),
            shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        };
        st.await_budget(42, 4000, 8000).await;
        assert_eq!(st.snapshot().uncached_tpm_rolling, 4000);
        st.record_usage(
            42,
            LlmCallMetrics {
                uncached_tokens: 350,
                prompt_tokens: 300,
                completion_tokens: 50,
                ..Default::default()
            },
        );
        assert_eq!(st.snapshot().uncached_tpm_rolling, 350);
    }

    #[tokio::test]
    async fn probe_85_proxy_auth_cors() {
        let p_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_addr = p_listener.local_addr().unwrap().to_string();
        drop(p_listener);

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_prx85_{}.jsonl", now_ms()))).unwrap();
        let mut cfg = test_cfg("http://127.0.0.1:1/v1", CacheMode::Short);
        cfg.listen_addr = p_addr.clone();
        cfg.auth_token = Some("secret-run-token".into());
        let (_st, handle) = start_proxy(cfg, tel).await.unwrap();

        let client = reqwest::Client::new();
        let unauth = client
            .get(format!("http://{p_addr}/api/timeline"))
            .header("origin", "https://evil.example")
            .send()
            .await
            .unwrap();
        assert_eq!(unauth.status().as_u16(), 401);
        assert!(unauth.headers().get("access-control-allow-origin").is_none());

        let authed = client
            .get(format!("http://{p_addr}/api/timeline"))
            .bearer_auth("secret-run-token")
            .header("origin", "http://127.0.0.1:18790")
            .send()
            .await
            .unwrap();
        assert_eq!(authed.status().as_u16(), 200);
        assert_eq!(
            authed
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://127.0.0.1:18790")
        );
        handle.abort();
    }

    #[test]
    fn probe_97_cors_and_drift() {
        assert!(localip(&HeaderValue::from_static("http://127.0.0.1:18790")));
        assert!(localip(&HeaderValue::from_static("http://localhost:3000")));
        assert!(localip(&HeaderValue::from_static("https://[::1]:8443")));
        assert!(!localip(&HeaderValue::from_static("http://localhost.evil.com")));
        assert!(!localip(&HeaderValue::from_static("https://127.0.0.1.attacker.io")));
        assert!(!localip(&HeaderValue::from_static("http://127.0.0.1@evil.com")));

        let tel = TelemetryHub::new(std::env::temp_dir().join(format!("refit_pd97_{}.jsonl", now_ms()))).unwrap();
        let st = ProxyState {
            cfg: test_cfg("https://api.cerebras.ai/v1", CacheMode::Short),
            client: reqwest::Client::new(),
            telemetry: tel,
            pacer: Mutex::new(PacerState::default()),
            module_metrics: DashMap::new(),
            in_flight: AtomicU64::new(0),
            req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(None),
            shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        };
        let py_p = serde_json::json!({"messages": [{"role": "system", "content": "P".repeat(150)}]});
        let rs_p = serde_json::json!({"messages": [{"role": "system", "content": "R".repeat(150)}]});
        assert!(!check_drift(&st, &py_p, "python"));
        assert!(!check_drift(&st, &rs_p, "rust"));
        assert!(!check_drift(&st, &py_p, "python"));
        assert_eq!(st.snapshot().prefix_drift_events, 0);
    }
}
