use crate::mvcc::MvccStore;
use crate::net::{drain_sse_typed, header_str, header_u64, is_quota_exhausted, retry_http};
use crate::platform::{SieveCache, env_parse, env_trimmed, file_mtime_size, stream_http};
use crate::state::RunState;
use crate::telemetry::{EventSpec, TelemetryHub};
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
use futures_util::StreamExt;
use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{Error as IoError, ErrorKind};
use std::mem::DropGuard;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::broadcast::error::RecvError;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

const DEFAULT_SSE_IDLE_MS: u64 = 45_000;
const MAX_KEY_LEN: usize = 1024;
const MAX_TIMING_SAMPLES: usize = 4096;
const KEEPWARM_MAX_PINGS: u32 = 11;

static ATTEMPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[attempt=[ \t]*([0-9]+)[ \t]*\]").expect("valid attempt regex"));
static MODULE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Refactor unit `([^`]*)`").expect("valid module regex"));

#[derive(Debug, Clone, Default, Deserialize)]
struct PromptTokDetails { #[serde(default)] cached_tokens: u64 }
#[derive(Debug, Clone, Default, Deserialize)]
struct CompTokDetails { #[serde(default)] reasoning_tokens: u64 }
#[derive(Debug, Clone, Default, Deserialize)]
struct Usage {
    #[serde(default)] prompt_tokens: u64, #[serde(default)] completion_tokens: u64,
    #[serde(default)] prompt_tokens_details: PromptTokDetails,
    #[serde(default)] completion_tokens_details: CompTokDetails,
}
#[derive(Debug, Clone, Default, Deserialize)]
struct TimeInfo {
    #[serde(default)] queue_time: f64, #[serde(default)] prompt_time: f64,
    #[serde(default)] completion_time: f64,
}
#[derive(Debug, Clone, Default, Deserialize)]
struct ChatRespMeta { #[serde(default)] usage: Option<Usage>, #[serde(default)] time_info: Option<TimeInfo> }
#[derive(Deserialize)]
struct SteerCmd {
    #[serde(default)] action: String, #[serde(default)] unit_id: Option<String>,
    #[serde(default)] k: Option<usize>,
}
#[derive(Serialize)]
struct CircuitEvent {
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")] pause_s: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")] status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")] reason: Option<&'static str>,
}
#[derive(Serialize)] struct ErrBody { error: String }
#[derive(Serialize)] struct HealthBody { ok: bool, proxy: ProxySnapshot }
#[derive(Serialize)] struct SteerGetBody<T: Serialize> { attached: bool, steering: Option<T> }
#[derive(Serialize)] struct SteerOkBody<T: Serialize> { ok: bool, steering: T }
#[derive(Serialize)] struct SteerEventBody<'a, T: Serialize> { action: &'a str, steering: &'a T }
#[derive(Serialize)] struct TimelineBody<T: Serialize> { entries: Vec<T> }
#[derive(Serialize)] struct SseStallBody { idle_timeout_ms: u64, attempt: u32 }
#[derive(Serialize)] struct StreamOpts { include_usage: bool }
#[derive(Serialize)] struct PromptCacheOpts { mode: &'static str, ttl: &'static str }
#[derive(Serialize)] struct ChatMsg<'a> { role: &'a str, content: &'a str }
#[derive(Serialize)]
struct KeepWarmReq<'a> {
    model: &'a str, messages: [ChatMsg<'a>; 2], stream: bool, max_completion_tokens: u64,
}

fn circuit_ev(
    state: &'static str, pause_s: Option<u64>, status: Option<u16>, reason: Option<&'static str>,
) -> Option<Value> {
    to_val(CircuitEvent { state, pause_s, status, reason })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CacheMode { Short, Long, None }

impl CacheMode {
    pub fn parse_mode(explicit: Option<&str>) -> Self {
        let raw = explicit.map(str::to_string)
            .or_else(|| env_trimmed("REFIT_CACHE_RETENTION")).unwrap_or_default();
        match raw.trim().to_ascii_lowercase().as_str() {
            "long" | "1h" | "24h" => Self::Long,
            "none" | "off" => Self::None,
            _ => Self::Short,
        }
    }

    pub fn ttl_ms(self, upstream_base_url: &str) -> u64 {
        let u = upstream_base_url.to_ascii_lowercase();
        let long_host = ["moonshot.ai", "kimi.ai", "anthropic.com"].iter().any(|h| u.contains(h));
        match self {
            Self::None => 0,
            Self::Long if long_host => 3_600_000,
            Self::Long if u.contains("openai.com") => 1_800_000,
            _ => 300_000,
        }
    }
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub listen_addr: String, pub upstream_base_url: String, pub api_key: String,
    pub model_override: String, pub ui_index_path: PathBuf,
    pub static_prefix_override: Option<String>, pub cache_retention: CacheMode,
    pub keep_warm: bool, pub auth_token: Option<String>,
    pub lang_prefixes: BTreeMap<String, String>, pub preserve_tools: bool,
}

impl ProxyConfig {
    pub fn new(listen: String, upstream: String, api_key: String, model: String) -> Self {
        Self {
            listen_addr: listen, upstream_base_url: upstream, api_key, model_override: model,
            ui_index_path: PathBuf::from("ui/index.html"), static_prefix_override: None,
            cache_retention: CacheMode::Short, keep_warm: false, auth_token: None,
            lang_prefixes: BTreeMap::new(), preserve_tools: true,
        }
    }
}

struct RollingEntry { req_id: u64, at: Instant, uncached: u64, total: u64 }

#[derive(Default)]
struct KeepWarmState {
    last_activity: Option<Instant>, last_ping: Option<Instant>,
    pings_sent: u32, disabled_reason: Option<&'static str>,
}

#[derive(Default)]
struct PacerState {
    window: VecDeque<RollingEntry>, snap: ProxySnapshot, consecutive_outage: u32,
    circuit_open_until: Option<Instant>, prefix_hashes: HashMap<String, String>,
    queue_ms_samples: VecDeque<f64>, decode_tps_samples: VecDeque<f64>,
    requests_sent: u64, retries_sent: u64, keep_warm: KeepWarmState,
}

impl PacerState {
    fn new() -> Self {
        let snap = ProxySnapshot {
            circuit_state: "CLOSED".into(), uncached_tpm_cap: 44_000,
            total_tpm_cap: 440_000, rpm_cap: 500, ..ProxySnapshot::default()
        };
        Self {
            snap, queue_ms_samples: VecDeque::with_capacity(256),
            decode_tps_samples: VecDeque::with_capacity(256), ..Self::default()
        }
    }

    fn rolling_totals(&mut self) -> (u64, u64, u64) {
        let win = Duration::from_secs(60);
        while self.window.front().is_some_and(|e| e.at.elapsed() >= win) { self.window.pop_front(); }
        let (unc, tot) = self.window.iter().fold((0u64, 0u64), |(u, t), e| {
            (u.saturating_add(e.uncached), t.saturating_add(e.total))
        });
        (unc, tot, self.window.len() as u64)
    }

    fn snapshot(&mut self) -> ProxySnapshot {
        let (unc, tot, rpm) = self.rolling_totals();
        ProxySnapshot {
            uncached_tpm_rolling: unc, total_tpm_rolling: tot, rpm_rolling: rpm, ..self.snap.clone()
        }
    }

    fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) {
        let queue = self.queue_ms_samples.iter().copied().collect();
        let decode = self.decode_tps_samples.iter().copied().collect();
        (queue, decode)
    }

    fn allow_retry(&mut self) -> bool {
        if self.retries_sent >= (self.requests_sent / 5).max(4) { return false; }
        self.retries_sent += 1;
        true
    }

    fn try_reserve(&mut self, req_id: u64, est_unc: u64, est_tot: u64) -> Option<Duration> {
        pacer_try_reserve(self, req_id, est_unc, est_tot)
    }

    fn free_budget(&mut self, req_id: u64) {
        if let Some(pos) = self.window.iter().position(|e| e.req_id == req_id) { self.window.remove(pos); }
    }

    fn open_circuit(&mut self, pause_s: u64) {
        self.snap.circuit_state = "OPEN".into();
        self.circuit_open_until = Some(Instant::now() + Duration::from_secs(pause_s));
        self.snap.circuit_pauses += 1;
    }

    fn apply_status(
        &mut self, status: u16, limit_tpm: Option<u64>, limit_rpm: Option<u64>, retry_after: Option<u64>,
    ) -> Option<Value> {
        pacer_apply_status(self, status, limit_tpm, limit_rpm, retry_after)
    }

    fn record_usage(&mut self, req_id: u64, met: &LlmCallMetrics) -> u64 {
        pacer_record_usage(self, req_id, met)
    }

    fn check_drift(&mut self, lang_key: &str, hash: String) -> bool {
        match self.prefix_hashes.get(lang_key) {
            None => { self.prefix_hashes.insert(lang_key.to_string(), hash); false }
            Some(prev) if prev == &hash => false,
            Some(_) => { self.snap.prefix_drift_events += 1; true }
        }
    }

    fn keepwarm_check(&self, ttl_ms: u64, in_flight: u64) -> bool {
        if matches!(self.snap.circuit_state.as_str(), "OPEN" | "PERMANENT_OPEN") { return false; }
        let Some(last) = self.keep_warm.last_activity else { return false };
        let idle = last.elapsed().as_millis() as u64;
        let since = self.keep_warm.last_ping.map_or(idle, |t| t.elapsed().as_millis() as u64);
        keepwarm_due(idle, since, ttl_ms, in_flight, self.keep_warm.pings_sent,
            KEEPWARM_MAX_PINGS, self.keep_warm.disabled_reason.is_some())
    }
}

fn pacer_try_reserve(p: &mut PacerState, req_id: u64, est_unc: u64, est_tot: u64) -> Option<Duration> {
    let mut half_open = false;
    if let Some(until) = p.circuit_open_until {
        let now = Instant::now();
        if now < until { return Some(until.saturating_duration_since(now)); }
        p.circuit_open_until = None;
        p.snap.circuit_state = "HALF_OPEN".into();
        half_open = true;
    }

    let (unc, tot, rpm) = p.rolling_totals();
    let req_unc = est_unc.min(p.snap.uncached_tpm_cap);
    let req_tot = est_tot.min(p.snap.total_tpm_cap);
    let empty = p.window.is_empty();
    if (unc.saturating_add(req_unc) <= p.snap.uncached_tpm_cap || empty)
        && (tot.saturating_add(req_tot) <= p.snap.total_tpm_cap || empty)
        && rpm < p.snap.rpm_cap
    {
        p.window.push_back(RollingEntry { req_id, at: Instant::now(), uncached: req_unc, total: req_tot });
        return None;
    }
    if half_open { return Some(Duration::from_millis(500)); }
    Some(p.window.front().map_or(Duration::from_millis(500), |e| {
        Duration::from_secs(60).saturating_sub(e.at.elapsed())
            .clamp(Duration::from_millis(250), Duration::from_secs(5))
    }))
}

fn pacer_apply_ok(p: &mut PacerState) -> Option<Value> {
    let was_open = p.snap.circuit_state != "CLOSED";
    p.consecutive_outage = 0;
    p.snap.circuit_state = "CLOSED".into();
    p.circuit_open_until = None;
    p.keep_warm.last_activity = Some(Instant::now());
    p.keep_warm.pings_sent = 0;
    was_open.then(|| circuit_ev("CLOSED", None, None, None)).flatten()
}

fn pacer_apply_429_5xx(p: &mut PacerState, status: u16, retry_after: Option<u64>) -> Option<Value> {
    if status == 429 {
        let ra = retry_after.unwrap_or(5);
        p.snap.http_429_count += 1;
        if ra > 10 { p.consecutive_outage += 1; }
        if p.consecutive_outage < 3 { return None; }
        let pause_s = ra.clamp(15, 60);
        p.open_circuit(pause_s);
        return circuit_ev("OPEN", Some(pause_s), None, Some("429_storm"));
    }
    p.snap.http_5xx_count += 1;
    p.consecutive_outage += 1;
    let pause_s = (5u64 * (1u64 << p.consecutive_outage.min(3))).min(60);
    p.open_circuit(pause_s);
    circuit_ev("OPEN", Some(pause_s), Some(status), None)
}

fn pacer_apply_status(
    p: &mut PacerState, status: u16, limit_tpm: Option<u64>, limit_rpm: Option<u64>, retry_after: Option<u64>,
) -> Option<Value> {
    if let Some(v) = limit_tpm {
        p.snap.uncached_tpm_cap = ((v as f64) * 0.88).round().max(10_000.0) as u64;
    }
    if let Some(v) = limit_rpm {
        p.snap.rpm_cap = ((v as f64) * 0.80).round().max(30.0) as u64;
    }
    if is_quota_exhausted(status) {
        p.snap.circuit_state = "PERMANENT_OPEN".into();
        p.circuit_open_until = None;
        p.snap.circuit_pauses += 1;
        return circuit_ev("PERMANENT_OPEN", None, Some(432), Some("quota_exhausted"));
    }
    if status == 429 || status >= 500 { return pacer_apply_429_5xx(p, status, retry_after); }
    if (200..300).contains(&status) { return pacer_apply_ok(p); }
    None
}

fn pacer_record_usage(p: &mut PacerState, req_id: u64, met: &LlmCallMetrics) -> u64 {
    let snap = &mut p.snap;
    snap.total_prompt_tokens = snap.total_prompt_tokens.saturating_add(met.prompt_tokens);
    snap.total_cached_tokens = snap.total_cached_tokens.saturating_add(met.cached_tokens);
    snap.total_completion_tokens = snap.total_completion_tokens.saturating_add(met.completion_tokens);
    snap.total_uncached_tokens = snap.total_uncached_tokens.saturating_add(met.uncached_tokens);
    push_sample(&mut p.queue_ms_samples, met.queue_ms);
    push_sample(&mut p.decode_tps_samples, met.decode_tps);

    let actual_tot = met.prompt_tokens.saturating_add(met.completion_tokens);
    if let Some(slot) = p.window.iter_mut().find(|e| e.req_id == req_id) {
        (slot.uncached, slot.total) = (met.uncached_tokens, actual_tot);
    } else if met.uncached_tokens > 0 || actual_tot > 0 {
        p.window.push_back(RollingEntry {
            req_id, at: Instant::now(), uncached: met.uncached_tokens, total: actual_tot,
        });
    }
    p.rolling_totals().0
}

impl PacerState {

    fn keepwarm_done(&mut self, cached: Option<u64>) {
        let Some(tokens) = cached else { self.keep_warm.disabled_reason = Some("ping_failed"); return };
        self.keep_warm.last_ping = Some(Instant::now());
        self.keep_warm.pings_sent += 1;
        if tokens == 0 { self.keep_warm.disabled_reason = Some("zero_cached_tokens"); }
    }
}

fn to_val<T: Serialize>(v: T) -> Option<Value> { serde_json::to_value(v).ok() }

fn push_sample(q: &mut VecDeque<f64>, val: f64) {
    if val <= 0.0 { return; }
    if q.len() >= MAX_TIMING_SAMPLES { q.pop_front(); }
    q.push_back(val);
}

fn stall_ms() -> u64 {
    env_parse::<u64>("REFIT_SSE_STALL_MS").map_or(DEFAULT_SSE_IDLE_MS, |v| v.clamp(100, 300_000))
}

fn eq_token(got: &str, expected: &str) -> bool {
    let key = blake3::derive_key("refit proxy token compare", b"token");
    blake3::keyed_hash(&key, got.trim().as_bytes()) == blake3::keyed_hash(&key, expected.as_bytes())
}

pub struct ProxyState {
    cfg: ProxyConfig, client: reqwest::Client, telemetry: TelemetryHub,
    pacer: Mutex<PacerState>, module_metrics: SieveCache<String, LlmCallMetrics>,
    in_flight: AtomicU64, req_seq: AtomicU64, repo_root: Mutex<Option<PathBuf>>,
    shared_state: Mutex<Option<Arc<RunState>>>, ui_cache: Mutex<Option<(u64, String)>>,
}

impl ProxyState {
    fn new(
        cfg: ProxyConfig, client: reqwest::Client, telemetry: TelemetryHub, repo_root: Option<PathBuf>,
    ) -> Self {
        Self {
            cfg, client, telemetry, pacer: Mutex::new(PacerState::new()),
            module_metrics: SieveCache::new(1_024, 2 * 1024 * 1024),
            in_flight: AtomicU64::new(0), req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(repo_root), shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        }
    }

    pub fn bind_repo(&self, root: PathBuf) { *self.repo_root.lock() = Some(root); }
    pub fn attach_state(&self, st: Arc<RunState>) { *self.shared_state.lock() = Some(st); }
    pub fn snapshot(&self) -> ProxySnapshot { self.pacer.lock().snapshot() }
    pub fn take_metrics(&self, module_id: &str) -> Option<LlmCallMetrics> {
        self.module_metrics.remove(&module_id.to_string())
    }
    pub fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) { self.pacer.lock().timing_samples() }

    fn upstream_url(&self, path: &str) -> String {
        format!("{}/{path}", self.cfg.upstream_base_url.trim_end_matches('/'))
    }

    fn is_authorized(&self, headers: &HeaderMap) -> bool {
        let Some(ref expected) = self.cfg.auth_token else { return true };
        if expected.is_empty() { return true; }
        header_str(headers, "authorization")
            .and_then(|s| s.strip_prefix("Bearer "))
            .is_some_and(|tok| eq_token(tok, expected))
    }

    async fn await_budget(&self, req_id: u64, est_unc: u64, est_tot: u64) {
        loop {
            let delay = self.pacer.lock().try_reserve(req_id, est_unc, est_tot);
            let Some(wait) = delay else { break };
            tokio::time::sleep(wait).await;
        }
    }

    fn apply_headers(&self, status: u16, headers: &reqwest::header::HeaderMap) {
        let tpm = header_u64(headers, "x-ratelimit-limit-tokens-minute");
        let rpm = header_u64(headers, "x-ratelimit-limit-requests-minute");
        let ra = header_u64(headers, "retry-after");
        if let Some(payload) = self.pacer.lock().apply_status(status, tpm, rpm, ra) {
            self.telemetry.emit_data("circuit_state", payload);
        }
    }

    fn record_usage(&self, req_id: u64, met: LlmCallMetrics) {
        let rolling = self.pacer.lock().record_usage(req_id, &met);
        if !met.module_id.is_empty() {
            self.module_metrics.insert(met.module_id.clone(), met.clone(), 256);
        }
        let mid = (!met.module_id.is_empty()).then(|| met.module_id.clone());
        let data = serde_json::to_value(&met).unwrap_or(Value::Null);
        let spec = EventSpec {
            module_id: mid, uncached_tpm: Some(rolling), data: Some(data), ..EventSpec::default()
        };
        self.telemetry.emit("llm_done", spec);
    }
}

pub async fn start_proxy(
    cfg: ProxyConfig, telemetry: TelemetryHub,
) -> Result<(Arc<ProxyState>, tokio::task::JoinHandle<()>)> {
    let ua = concat!("refit/", env!("CARGO_PKG_VERSION"));
    let client = stream_http(ua, 15, 600, crate::platform::budget().http_pool_max)?;
    let state = Arc::new(ProxyState::new(cfg.clone(), client, telemetry, std::env::current_dir().ok()));
    if cfg.keep_warm && cfg.static_prefix_override.is_some() {
        let kw = state.clone();
        tokio::spawn(async move { keepwarm_loop(kw).await });
    }

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| localip(origin)))
        .allow_methods(Any).allow_headers(Any);
    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/health", get(handle_health))
        .route("/events", get(serve_events))
        .route("/api/timeline", get(timeline_get))
        .route("/api/diff/{tx_id}", get(serve_diff))
        .route("/api/steer", get(steer_get).post(steer_post))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(chat_handler))
        .layer(cors).with_state(state.clone());

    let listener = TcpListener::bind(&cfg.listen_addr).await
        .with_context(|| format!("bind proxy on {}", cfg.listen_addr))?;
    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await { tracing::error!("proxy server stopped: {err}"); }
    });
    Ok((state, handle))
}

async fn keepwarm_loop(state: Arc<ProxyState>) {
    let ttl_ms = state.cfg.cache_retention.ttl_ms(&state.cfg.upstream_base_url);
    let Some(prefix) = state.cfg.static_prefix_override.clone().filter(|_| ttl_ms > 0) else { return };
    let mut ticker = tokio::time::interval(Duration::from_secs(15));
    let url = state.upstream_url("chat/completions");
    loop {
        ticker.tick().await;
        if !state.pacer.lock().keepwarm_check(ttl_ms, state.in_flight.load(Ordering::Relaxed)) { continue; }
        let cached = send_keepwarm(&state, &url, &prefix).await;
        state.pacer.lock().keepwarm_done(cached);
    }
}

async fn send_keepwarm(state: &ProxyState, url: &str, prefix: &str) -> Option<u64> {
    let req_body = KeepWarmReq {
        model: &state.cfg.model_override,
        messages: [ChatMsg { role: "system", content: prefix }, ChatMsg { role: "user", content: "." }],
        stream: false, max_completion_tokens: 1,
    };
    let mut body = serde_json::to_value(&req_body).ok()?;
    guard_lang(&mut body, &state.cfg, "probe", 2, None);
    let req = state.client.post(url).bearer_auth(&state.cfg.api_key).json(&body);
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() { return None; }
    let meta: ChatRespMeta = resp.json().await.unwrap_or_default();
    Some(meta.usage.map_or(0, |u| u.prompt_tokens_details.cached_tokens))
}

fn json_err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(ErrBody { error: msg.into() })).into_response()
}

fn err_401() -> Response { json_err(StatusCode::UNAUTHORIZED, "missing or invalid proxy bearer token") }

async fn serve_ui(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let sig = file_mtime_size(&state.cfg.ui_index_path)
        .map(|(mtime_ns, size)| size ^ mtime_ns.rotate_left(17));
    if let Some(cur) = sig
        && let Some((_, html)) = state.ui_cache.lock().as_ref().filter(|(s, _)| *s == cur)
    {
        return (StatusCode::OK, Html(html.clone())).into_response();
    }
    let html = match std::fs::read_to_string(&state.cfg.ui_index_path) {
        Ok(text) => {
            if let Some(cur) = sig { *state.ui_cache.lock() = Some((cur, text.clone())); }
            text
        }
        Err(_) => "<html><body><h1>refit telemetry proxy active</h1></body></html>".into(),
    };
    (StatusCode::OK, Html(html)).into_response()
}

async fn steer_get(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    let steering = state.shared_state.lock().as_ref().map(|s| s.steering_snapshot());
    Json(SteerGetBody { attached: steering.is_some(), steering }).into_response()
}

async fn steer_post(
    State(state): State<Arc<ProxyState>>, headers: HeaderMap, Json(cmd): Json<SteerCmd>,
) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    let Some(st) = state.shared_state.lock().clone() else {
        return json_err(StatusCode::CONFLICT, "no active run attached for live steering");
    };
    if !apply_steer(&st, &cmd) {
        return json_err(StatusCode::BAD_REQUEST, format!("unknown steering action `{}`", cmd.action));
    }
    let snap = st.steering_snapshot();
    let data = to_val(SteerEventBody { action: &cmd.action, steering: &snap });
    state.telemetry.emit("live_steer", EventSpec { module_id: cmd.unit_id, data, ..EventSpec::default() });
    (StatusCode::OK, Json(SteerOkBody { ok: true, steering: snap })).into_response()
}

fn apply_steer(st: &RunState, cmd: &SteerCmd) -> bool {
    match cmd.action.as_str() {
        "pause" => st.steer_pause(true),
        "resume" => st.steer_pause(false),
        "abort" => st.steer_abort(),
        "set_k" => st.steer_k(cmd.k),
        "prioritize" => if let Some(u) = cmd.unit_id.clone() { st.steer_prioritize(u); },
        "skip" => if let Some(u) = cmd.unit_id.clone() { st.steer_skip(u); },
        _ => return false,
    }
    true
}

async fn handle_health(State(state): State<Arc<ProxyState>>) -> Json<HealthBody> {
    Json(HealthBody { ok: true, proxy: state.snapshot() })
}

async fn timeline_get(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    let root = state.repo_root.lock().clone();
    let entries = root
        .as_deref()
        .and_then(|r| MvccStore::wal_timeline(r).ok())
        .unwrap_or_default();
    Json(TimelineBody { entries }).into_response()
}

async fn serve_diff(
    State(state): State<Arc<ProxyState>>, headers: HeaderMap, Path(tx_id): Path<String>,
) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    let root = state.repo_root.lock().clone();
    let diff = root
        .as_deref()
        .and_then(|r| MvccStore::diff_tx(r, &tx_id).ok())
        .flatten();
    match diff {
        Some(report) => (StatusCode::OK, Json(report)).into_response(),
        None => json_err(StatusCode::NOT_FOUND, format!("tx `{tx_id}` not found")),
    }
}

async fn serve_events(
    State(state): State<Arc<ProxyState>>, headers: HeaderMap, Query(params): Query<HashMap<String, String>>,
) -> Response {
    let query_ok = state.cfg.auth_token.as_deref().is_none_or(|exp| {
        exp.is_empty() || params.get("token").is_some_and(|t| eq_token(t, exp))
    });
    if !state.is_authorized(&headers) && !query_ok { return err_401(); }
    let hist = futures_util::stream::iter(state.telemetry.snapshot_events().into_iter().map(sse_event));
    let live = futures_util::stream::unfold(state.telemetry.subscribe(), |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(ev) => return Some((sse_event(ev), rx)),
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(hist.chain(live)).keep_alive(KeepAlive::default()).into_response()
}

fn sse_event(ev: crate::types::TelemetryEvent) -> std::result::Result<Event, !> {
    Ok(Event::default().data(serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into())))
}

async fn handle_models(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    let (url, seq) = (state.upstream_url("models"), state.req_seq.fetch_add(1, Ordering::Relaxed));
    let res = retry_http(3, 150, 5_000, seq, |_| true, || {
        state.client
            .get(&url)
            .bearer_auth(&state.cfg.api_key)
            .timeout(Duration::from_secs(15))
            .send()
    }).await;
    match res {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = resp.bytes().await.unwrap_or_default();
            (status, [("content-type", "application/json")], bytes).into_response()
        }
        Err(err) => json_err(StatusCode::BAD_GATEWAY, err.to_string()),
    }
}

struct ChatCtx { seq: u64, module_id: String, attempt: u32, is_stream: bool, drift: bool, idle_ms: u64 }
#[derive(Clone, Copy)]
struct RateLimits { tpm: Option<u64>, rem_tpm: Option<u64>, rpm: Option<u64> }

async fn chat_handler(
    State(state): State<Arc<ProxyState>>, headers: HeaderMap, Json(mut payload): Json<Value>,
) -> Response {
    if !state.is_authorized(&headers) { return err_401(); }
    state.in_flight.fetch_add(1, Ordering::Relaxed);
    let _guard = DropGuard::new(state.clone(), |s| { s.in_flight.fetch_sub(1, Ordering::Relaxed); });

    let module_id = header_str(&headers, "x-refit-module").filter(|s| !s.is_empty())
        .map_or_else(|| find_module(&payload), String::from);
    let attempt = header_str(&headers, "x-refit-attempt").and_then(|s| s.parse().ok())
        .unwrap_or_else(|| infer_attempt(&payload));
    let lang = header_str(&headers, "x-refit-lang");

    guard_lang(&mut payload, &state.cfg, &module_id, attempt, lang);
    let drift = check_drift(&state, &payload, lang.unwrap_or("default"));
    let (body_bytes, est_unc, est_tot) = estimate_body(&payload, attempt);

    let seq = state.req_seq.fetch_add(1, Ordering::Relaxed);
    state.await_budget(seq, est_unc, est_tot).await;
    state.pacer.lock().requests_sent += 1;

    let upstream = match send_chat(&state, &body_bytes, seq).await {
        Ok(resp) => resp,
        Err(err) => {
            state.apply_headers(503, &reqwest::header::HeaderMap::new());
            state.pacer.lock().free_budget(seq);
            return json_err(StatusCode::BAD_GATEWAY, err.to_string());
        }
    };
    let idle_ms = header_u64(&headers, "x-refit-sse-stall-ms")
        .map_or_else(stall_ms, |v| v.clamp(50, 300_000));
    let is_stream = payload.get("stream").and_then(Value::as_bool).unwrap_or(false);
    finish_chat(state, upstream, ChatCtx { seq, module_id, attempt, is_stream, drift, idle_ms }).await
}

async fn send_chat(
    state: &ProxyState, body_bytes: &Bytes, seq: u64,
) -> std::result::Result<reqwest::Response, reqwest::Error> {
    let url = state.upstream_url("chat/completions");
    retry_http(3, 250, 30_000, seq, |resp| {
        let can = state.pacer.lock().allow_retry();
        match resp {
            Some(r) if can => state.apply_headers(r.status().as_u16(), r.headers()),
            None => state.apply_headers(503, &reqwest::header::HeaderMap::new()),
            Some(_) => {}
        }
        can
    }, || {
        state.client.post(&url).bearer_auth(&state.cfg.api_key)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(body_bytes.clone()).send()
    }).await
}

fn finish_metrics(
    state: &ProxyState, meta: &ChatRespMeta, ctx: &ChatCtx, code: u16, wall: u64, lim: RateLimits,
) {
    let mut met = parse_metrics(meta, &ctx.module_id, ctx.attempt, code, wall, ctx.drift);
    (met.limit_tpm_minute, met.remaining_tpm_minute, met.limit_rpm_minute) = (lim.tpm, lim.rem_tpm, lim.rpm);
    state.record_usage(ctx.seq, met);
}

async fn finish_chat(state: Arc<ProxyState>, upstream: reqwest::Response, ctx: ChatCtx) -> Response {
    let (started, code) = (Instant::now(), upstream.status().as_u16());
    let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let up_hdrs = upstream.headers().clone();
    state.apply_headers(code, &up_hdrs);
    let lim = RateLimits {
        tpm: header_u64(&up_hdrs, "x-ratelimit-limit-tokens-minute"),
        rem_tpm: header_u64(&up_hdrs, "x-ratelimit-remaining-tokens-minute"),
        rpm: header_u64(&up_hdrs, "x-ratelimit-limit-requests-minute"),
    };

    if !ctx.is_stream || code != 200 {
        let bytes = upstream.bytes().await.unwrap_or_default();
        if let Ok(meta) = serde_json::from_slice::<ChatRespMeta>(&bytes) {
            finish_metrics(&state, &meta, &ctx, code, started.elapsed().as_millis() as u64, lim);
        } else {
            state.pacer.lock().free_budget(ctx.seq);
        }
        return (status, [("content-type", "application/json")], bytes).into_response();
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Bytes, IoError>>(64);
    let st_c = state.clone();
    tokio::spawn(async move {
        let meta = forward_sse(
            upstream, &tx, &st_c.telemetry, &ctx.module_id, ctx.attempt, ctx.idle_ms,
        ).await;
        finish_metrics(&st_c, &meta, &ctx, code, started.elapsed().as_millis() as u64, lim);
    });
    let body = futures_util::stream::unfold(rx, |mut r| async move { r.recv().await.map(|i| (i, r)) });
    Response::builder().status(StatusCode::OK)
        .header("content-type", "text/event-stream").header("cache-control", "no-cache")
        .body(Body::from_stream(body))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream error").into_response())
}

async fn forward_sse(
    upstream: reqwest::Response, tx: &tokio::sync::mpsc::Sender<std::result::Result<Bytes, IoError>>,
    telemetry: &TelemetryHub, module_id: &str, attempt: u32, idle_ms: u64,
) -> ChatRespMeta {
    let (mut byte_stream, mut line_buf) = (upstream.bytes_stream(), Vec::<u8>::with_capacity(4096));
    let max_line = crate::platform::budget().sse_line_cap_bytes;
    let watchdog = Duration::from_millis(idle_ms.max(50));
    let mut meta = ChatRespMeta::default();
    let mut record_meta = |chunk: ChatRespMeta| {
        if chunk.usage.is_some() { meta.usage = chunk.usage; }
        if chunk.time_info.is_some() { meta.time_info = chunk.time_info; }
    };

    loop {
        let Ok(next_item) = tokio::time::timeout(watchdog, byte_stream.next()).await else {
            let mid = (!module_id.is_empty()).then(|| module_id.to_string());
            let data = to_val(SseStallBody { idle_timeout_ms: idle_ms, attempt });
            telemetry.emit("sse_stall", EventSpec { module_id: mid, data, ..EventSpec::default() });
            let msg = format!("upstream SSE stream stalled after {idle_ms}ms");
            let _ = tx.send(Err(IoError::new(ErrorKind::TimedOut, msg))).await;
            break;
        };
        let Some(chunk_res) = next_item else { break };
        let chunk = match chunk_res {
            Ok(c) => c,
            Err(err) => {
                let _ = tx.send(Err(IoError::other(err.to_string()))).await;
                break;
            }
        };
        line_buf.extend_from_slice(&chunk);
        drain_sse_typed(&mut line_buf, false, &mut record_meta);
        if line_buf.len() > max_line {
            line_buf.clear();
            line_buf.shrink_to(4096);
        }
        if tx.send(Ok(chunk)).await.is_err() { break; }
    }
    drain_sse_typed(&mut line_buf, true, &mut record_meta);
    meta
}

fn keepwarm_due(
    idle_ms: u64, since_last_ping_ms: u64, ttl_ms: u64, in_flight: u64,
    pings_sent: u32, max_pings: u32, disabled: bool,
) -> bool {
    if disabled || ttl_ms < 60_000 || in_flight > 0 || pings_sent >= max_pings { return false; }
    let (trigger, expiry) = ((ttl_ms * 80) / 100, ttl_ms.saturating_sub(15_000));
    idle_ms >= trigger && idle_ms < expiry && since_last_ping_ms >= trigger
}

fn msg_slice(payload: &Value) -> &[Value] {
    payload.get("messages").and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn infer_attempt(payload: &Value) -> u32 {
    let msgs = msg_slice(payload);
    for msg in msgs.iter().rev() {
        if let Some(text) = msg.get("content").and_then(Value::as_str)
            && let Some(cap) = ATTEMPT_RE.captures(text)
            && let Ok(num) = cap[1].parse::<u32>()
        {
            return num.max(1);
        }
    }
    if msgs.len() >= 4 { ((msgs.len() / 2) as u32).max(2) } else { 1 }
}

fn guard_lang(
    payload: &mut Value, cfg: &ProxyConfig, module_id: &str, attempt: u32, lang_hint: Option<&str>,
) {
    let Some(obj) = payload.as_object_mut() else { return };
    guard_core(obj, cfg);
    let has_tools = sort_tools(obj, cfg.preserve_tools);
    guard_module(obj, cfg, module_id, lang_hint, has_tools);
    guard_cache(obj, cfg, module_id, attempt);
}

fn guard_core(obj: &mut Map<String, Value>, cfg: &ProxyConfig) {
    if !cfg.model_override.is_empty() {
        obj.insert("model".into(), Value::String(cfg.model_override.clone()));
    }
    let req = obj.get("reasoning_effort").and_then(Value::as_str).unwrap_or("low");
    let effort = if matches!(req, "low" | "medium" | "high") { req } else { "low" };

    obj.insert("reasoning_effort".into(), Value::String(effort.into()));
    obj.insert("reasoning_format".into(), Value::String("parsed".into()));
    obj.insert("temperature".into(), Value::from(1.0));
    for drop_key in ["disable_reasoning", "top_p", "clear_thinking"] { obj.remove(drop_key); }

    if let Some(mt) = obj.remove("max_tokens") { obj.entry("max_completion_tokens").or_insert(mt); }
    let raw_max = obj.get("max_completion_tokens").and_then(Value::as_u64).unwrap_or(3200);
    let max_comp = raw_max.clamp(1, 6144);
    obj.insert("max_completion_tokens".into(), Value::from(max_comp));
    if obj.get("stream").and_then(Value::as_bool).unwrap_or(false)
        && let Some(v) = to_val(StreamOpts { include_usage: true })
    {
        obj.insert("stream_options".into(), v);
    }
}

fn sort_tools(obj: &mut Map<String, Value>, preserve: bool) -> bool {
    let Some(tools) = obj.get_mut("tools").and_then(Value::as_array_mut) else { return false };
    if preserve {
        tools.sort_by(|a, b| {
            let left = a["function"]["name"].as_str().unwrap_or("");
            let right = b["function"]["name"].as_str().unwrap_or("");
            left.cmp(right)
        });
    }
    !tools.is_empty()
}

fn guard_module(
    obj: &mut Map<String, Value>, cfg: &ProxyConfig, module_id: &str,
    lang_hint: Option<&str>, has_tools: bool,
) {
    if module_id.is_empty() || module_id == "probe" { return; }
    if !cfg.preserve_tools {
        obj.remove("tools");
        obj.remove("tool_choice");
    }
    if cfg.preserve_tools && has_tools { return; }
    let prefix = lang_hint.and_then(|l| cfg.lang_prefixes.get(l)).or(cfg.static_prefix_override.as_ref());
    let Some(static_sys) = prefix else { return };
    if let Some(msgs) = obj.get_mut("messages").and_then(Value::as_array_mut)
        && let Some(first) = msgs.first_mut()
        && first.get("role").and_then(Value::as_str) == Some("system")
    {
        first["content"] = Value::String(static_sys.clone());
    }
}

fn guard_cache(obj: &mut Map<String, Value>, cfg: &ProxyConfig, module_id: &str, attempt: u32) {
    if attempt <= 1 || cfg.cache_retention == CacheMode::None {
        obj.remove("prompt_cache_key");
    } else if let Some(k) = obj.get("prompt_cache_key").and_then(Value::as_str) {
        if k.len() > MAX_KEY_LEN {
            obj.insert("prompt_cache_key".into(), Value::String(k.chars().take(MAX_KEY_LEN).collect()));
        }
    } else if !module_id.is_empty() && module_id != "probe" {
        obj.insert("prompt_cache_key".into(), Value::String(format!("refit:auto:m:{module_id}")));
    }

    if cfg.cache_retention != CacheMode::Long { return; }
    let up = cfg.upstream_base_url.to_ascii_lowercase();
    if (up.contains("moonshot.ai") || up.contains("kimi.ai"))
        && let Some(v) = to_val(PromptCacheOpts { mode: "implicit", ttl: "1h" })
    {
        obj.insert("prompt_cache_options".into(), v);
    } else if up.contains("openai.com") {
        obj.insert("prompt_cache_retention".into(), Value::String("24h".into()));
    }
}

fn sys_text(payload: &Value) -> &str {
    msg_slice(payload).first()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|m| m.get("content").and_then(Value::as_str)).unwrap_or("")
}

fn find_module(payload: &Value) -> String {
    msg_slice(payload).iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .find_map(|text| MODULE_RE.captures(text).map(|cap| cap[1].to_string()))
        .unwrap_or_default()
}

fn localip(origin: &HeaderValue) -> bool {
    let Ok(raw) = origin.to_str() else { return false };
    if raw.is_empty() || raw.ends_with(':') { return false; }
    let Ok(parsed) = url::Url::parse(raw) else { return false };
    let after = raw.split_once("://").map_or("", |(_, rest)| rest);
    let bad_scheme = !matches!(parsed.scheme(), "http" | "https");
    let has_extra = !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some();
    let bad_addr = parsed.port() == Some(0) || after.contains('/') || after.contains('@');
    if bad_scheme || has_extra || bad_addr { return false; }
    match parsed.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => std::net::IpAddr::V4(ip).is_loopback(),
        Some(url::Host::Ipv6(ip)) => std::net::IpAddr::V6(ip).is_loopback(),
        None => false,
    }
}

fn parse_metrics(
    meta: &ChatRespMeta, module_id: &str, attempt: u32, status: u16, wall_ms: u64, prefix_drift: bool,
) -> LlmCallMetrics {
    let (u, ti) = (meta.usage.as_ref(), meta.time_info.as_ref());
    let prompt_tokens = u.map_or(0, |x| x.prompt_tokens);
    let cached_tokens = u.map_or(0, |x| x.prompt_tokens_details.cached_tokens);
    let completion_tokens = u.map_or(0, |x| x.completion_tokens);
    let reasoning_tokens = u.map_or(0, |x| x.completion_tokens_details.reasoning_tokens);
    let comp_s = ti.map_or(0.0, |x| x.completion_time);
    let decode_tps = if comp_s > 0.0005 && completion_tokens > 0 {
        (completion_tokens as f64) / comp_s
    } else { 0.0 };
    LlmCallMetrics {
        module_id: module_id.to_string(), attempt, http_status: status, wall_ms,
        queue_ms: ti.map_or(0.0, |x| x.queue_time) * 1000.0,
        prompt_ms: ti.map_or(0.0, |x| x.prompt_time) * 1000.0,
        completion_ms: comp_s * 1000.0, decode_tps, prompt_tokens, cached_tokens, completion_tokens,
        reasoning_tokens, uncached_tokens: prompt_tokens.saturating_sub(cached_tokens) + completion_tokens,
        prefix_drift, ..LlmCallMetrics::default()
    }
}

fn estimate_body(payload: &Value, attempt: u32) -> (Bytes, u64, u64) {
    let sys_bytes = sys_text(payload).len();
    let body_bytes = Bytes::from(serde_json::to_vec(payload).unwrap_or_default());
    let max_comp = payload["max_completion_tokens"].as_u64().unwrap_or(1600);
    let dyn_in = (body_bytes.len().saturating_sub(sys_bytes) as u64 / 4).max(150);
    let est_out = (max_comp / 3).clamp(200, 1800);
    let est_tot = (body_bytes.len() as u64 / 4).max(300) + est_out;
    let est_unc = if attempt <= 1 { dyn_in + est_out } else { (dyn_in / 3).max(200) + est_out };
    (body_bytes, est_unc, est_tot)
}

fn check_drift(state: &ProxyState, payload: &Value, lang_key: &str) -> bool {
    let sys = sys_text(payload);
    if sys.len() < 120 { return false; }
    let mut hasher = blake3::Hasher::new();
    hasher.update(sys.as_bytes());
    hasher.update(b"\n---\n");
    if let Some(tools) = payload.get("tools") { let _ = serde_json::to_writer(&mut hasher, tools); }
    let digest = hasher.finalize().to_hex().to_string();
    state.pacer.lock().check_drift(lang_key, digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        MockHttp, TempRepo, TestProxy, commit_file, http_res, mock_sse, proxy_cfg, sse_delta,
    };
    use serde_json::json;

    const CB_URL: &str = "https://api.cerebras.ai/v1";

    fn make_cfg(upstream: &str, retention: CacheMode) -> ProxyConfig {
        let mut cfg = proxy_cfg(upstream, retention);
        cfg.static_prefix_override = Some("S".repeat(160));
        cfg.preserve_tools = true;
        cfg
    }

    fn sys_msg(content: impl Into<String>) -> Value {
        json!({"messages": [{"role": "system", "content": content.into()}]})
    }

    fn chat_body(stream: bool) -> Value {
        json!({"messages": [{"role": "user", "content": "hi"}], "stream": stream})
    }

    async fn spawn_mock(resps: Vec<Vec<u8>>, tok: Option<&str>) -> (MockHttp, TestProxy) {
        let up = MockHttp::spawn(resps).await;
        let prx = TestProxy::spawn(&up.url("/v1"), CacheMode::Short, tok).await;
        (up, prx)
    }

    async fn get_origin(prx: &TestProxy, tok: Option<&str>, origin: &str) -> reqwest::Response {
        let mut req = prx.client.get(prx.url("/api/timeline")).header("origin", origin);
        if let Some(t) = tok { req = req.bearer_auth(t); }
        req.send().await.unwrap()
    }

    #[test]
    fn proxy_guard_and_routing() {
        let mut cfg = make_cfg(CB_URL, CacheMode::Short);
        cfg.preserve_tools = false;
        let mut p = json!({
            "model": "wrong", "disable_reasoning": true, "top_p": 0.9, "clear_thinking": true,
            "max_tokens": 1000, "tools": [{"type": "function"}],
            "messages": [{"role": "system", "content": "bloated"}]
        });
        guard_lang(&mut p, &cfg, "mod_01", 1, None);
        assert_eq!(p["model"], "moonshotai-kimi-k3.1-pro-trial");
        assert_eq!((&p["reasoning_effort"], &p["reasoning_format"]), (&json!("low"), &json!("parsed")));
        assert_eq!((&p["temperature"], &p["max_completion_tokens"]), (&json!(1.0), &json!(1000)));
        assert!(p.get("disable_reasoning").is_none() && p.get("top_p").is_none());
        assert!(p.get("tools").is_none());
        let sys_content = p["messages"][0]["content"].as_str().unwrap();
        assert_eq!(sys_content.len(), 160);
        verify_cache_and_tools(&cfg);
    }

    fn verify_cache_and_tools(cfg: &ProxyConfig) {
        for (att, want) in [(1, None), (2, Some(MAX_KEY_LEN))] {
            let mut b = json!({"prompt_cache_key": "k".repeat(2000)});
            guard_lang(&mut b, cfg, "m1", att, None);
            assert_eq!(b.get("prompt_cache_key").and_then(Value::as_str).map(str::len), want);
        }
        for (url, ttl, ret) in [
            (CB_URL, None, None),
            ("https://api.moonshot.ai/v1", Some("1h"), None),
            ("https://api.openai.com/v1", None, Some("24h")),
        ] {
            let mut b = json!({});
            guard_lang(&mut b, &make_cfg(url, CacheMode::Long), "m", 2, None);
            assert_eq!((b["prompt_cache_options"]["ttl"].as_str(), b["prompt_cache_retention"].as_str()), (ttl, ret));
        }

        let mut tp = json!({
            "reasoning_effort": "medium", "prompt_cache_key": "refit:r1:m:m1",
            "tools": [
                {"type": "function", "function": {"name": "z_tool"}},
                {"type": "function", "function": {"name": "a_tool"}}
            ],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Refactor unit `m1` now"},
                {"role": "assistant", "content": "bad"},
                {"role": "user", "content": "[attempt=3] retry"}
            ]
        });
        let inf = infer_attempt(&tp);
        assert_eq!(inf, 3);
        assert_eq!(find_module(&tp), "m1");
        guard_lang(&mut tp, &make_cfg(CB_URL, CacheMode::Short), "m1", inf, None);
        assert_eq!(tp["reasoning_effort"], "medium");
        assert_eq!(tp["tools"][0]["function"]["name"], "a_tool");
        assert_eq!(tp["prompt_cache_key"], "refit:r1:m:m1");
    }

    #[test]
    fn proxy_keepwarm_policy() {
        for (ms, inflight, sent, open, want) in [
            (100_000, 0, 0, false, false), (250_000, 0, 0, false, true),
            (295_000, 0, 0, false, false), (400_000, 0, 0, false, false),
            (250_000, 1, 0, false, false), (250_000, 0, 11, false, false),
            (250_000, 0, 2, true, false),
        ] {
            assert_eq!(keepwarm_due(ms, ms, 300_000, inflight, sent, 11, open), want);
        }
    }

    #[tokio::test]
    async fn proxy_pacer_and_cors() {
        let tmp = TempRepo::new("pac");
        let tel = TelemetryHub::new(tmp.join("run.jsonl")).unwrap();
        let st = ProxyState::new(make_cfg(CB_URL, CacheMode::Short), reqwest::Client::new(), tel, None);

        let (p1, p2) = (sys_msg("A".repeat(150)), sys_msg("A".repeat(149) + "B"));
        for (msg, lang, drifted) in [
            (sys_msg("OK"), "default", false), (p1.clone(), "default", false),
            (p1, "default", false), (p2, "default", true),
            (sys_msg("P".repeat(150)), "python", false), (sys_msg("R".repeat(150)), "rust", false),
            (sys_msg("P".repeat(150)), "python", false),
        ] {
            assert_eq!(check_drift(&st, &msg, lang), drifted);
        }
        assert_eq!(st.snapshot().prefix_drift_events, 1);

        let empty = reqwest::header::HeaderMap::new();
        st.apply_headers(503, &empty);
        assert_eq!((st.snapshot().circuit_state.as_str(), st.snapshot().circuit_pauses), ("OPEN", 1));
        st.apply_headers(200, &empty);
        assert_eq!(st.snapshot().circuit_state, "CLOSED");

        st.await_budget(42, 4000, 8000).await;
        assert_eq!(st.snapshot().uncached_tpm_rolling, 4000);
        st.record_usage(42, LlmCallMetrics {
            uncached_tokens: 350, prompt_tokens: 300, completion_tokens: 50, ..Default::default()
        });
        assert_eq!(st.snapshot().uncached_tpm_rolling, 350);

        let mut p = PacerState::new();
        assert!(p.apply_status(429, None, None, Some(2)).is_none());
        assert_eq!((p.snapshot().circuit_state.as_str(), p.snapshot().http_429_count), ("CLOSED", 1));
        let ev = p.apply_status(432, None, None, None).unwrap();
        assert_eq!((&ev["state"], &ev["reason"]), (&json!("PERMANENT_OPEN"), &json!("quota_exhausted")));
        p.requests_sent = 50;
        assert_eq!((0..30).filter(|_| p.allow_retry()).count(), 10);

        for (orig, want) in [
            ("http://127.0.0.1:18790", true), ("http://127.0.0.2:65535", true),
            ("http://localhost", true), ("http://localhost:3000", true),
            ("https://[::1]:8443", true), ("http://127.0.0.1:65536", false),
            ("http://127.0.0.1:0", false), ("http://127.0.0.1:", false),
            ("http://user@127.0.0.1:3000", false), ("http://127.0.0.1:3000/api", false),
            ("http://127.0.0.1:3000?x=1", false), ("http://127.0.0.1:3000#h", false),
            ("ftp://127.0.0.1:21", false), ("http://0.0.0.0:3000", false),
            ("http://192.168.1.1:3000", false), ("http://localhost.evil.com", false),
            ("https://127.0.0.1.attacker.io", false), ("http://127.0.0.1@evil.com", false),
        ] {
            assert_eq!(localip(&HeaderValue::from_static(orig)), want, "origin={orig}");
        }
    }

    #[tokio::test]
    async fn proxy_retry_policy() {
        let ok = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#;
        let body = chat_body(false);
        for (resps, want_code, want_calls) in [
            (vec![http_res("503 Service Unavailable", "Retry-After: 0.02\r\n", "down"), http_res("200 OK", "", ok)], 200, 2),
            (vec![http_res("404 Not Found", "", r#"{"error":"not_found"}"#)], 404, 1),
        ] {
            let (up, prx) = spawn_mock(resps, None).await;
            let (code, _) = prx.post_json("/v1/chat/completions", &body).await;
            assert_eq!((code, up.calls()), (want_code, want_calls));
        }
    }

    #[tokio::test]
    async fn proxy_sse_stall_watchdog() {
        let up_addr = mock_sse(vec![sse_delta("hi")], 0, 10).await;
        let prx = TestProxy::spawn(&format!("http://{up_addr}/v1"), CacheMode::Short, None).await;
        let started = Instant::now();
        let req = prx.client
            .post(prx.url("/v1/chat/completions"))
            .header("x-refit-sse-stall-ms", "120")
            .json(&chat_body(true));
        let resp = req.send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let saw_err = resp.bytes_stream().any(|item| std::future::ready(item.is_err())).await;
        assert!(saw_err && started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn proxy_auth_and_steer() {
        let tmp = TempRepo::with_files("prx_api", &[("app.py", "def f():\n    return 1\n")]);
        let store = MvccStore::open(tmp.path(), "prx_api").unwrap();
        let (_, tx_id, _) = commit_file(&store, tmp.path(), "app", "app.py", "def f():\n    return 42\n");
        let (up, prx) = spawn_mock(vec![
            http_res("503 Service Unavailable", "Retry-After: 0.01\r\n", "busy"),
            http_res("200 OK", "", r#"{"data":[{"id":"kimi"}]}"#),
        ], Some("secret-run-token")).await;
        let shared = RunState::new(100);
        prx.state.bind_repo(tmp.path().to_path_buf());
        prx.state.attach_state(shared.clone());

        let acao = |r: &reqwest::Response| header_str(r.headers(), "access-control-allow-origin").map(str::to_string);
        let unauth = get_origin(&prx, None, "https://evil.example").await;
        assert_eq!((unauth.status().as_u16(), acao(&unauth)), (401, None));
        let authed = get_origin(&prx, Some("secret-run-token"), "http://127.0.0.1:18790").await;
        assert_eq!((authed.status().as_u16(), acao(&authed).as_deref()), (200, Some("http://127.0.0.1:18790")));

        let (_, tl) = prx.get_json("/api/timeline").await;
        assert_eq!(tl["entries"][0]["tx_id"], json!(tx_id));
        let (code, dj) = prx.get_json(&format!("/api/diff/{tx_id}")).await;
        assert_eq!(code, 200);
        let diff_text = dj["files"][0]["unified_diff"].as_str().unwrap();
        assert!(diff_text.contains("+    return 42"));
        assert_eq!(
            (prx.get_json("/api/diff/tx-none").await.0, prx.get_json("/v1/models").await.0, up.calls()),
            (404, 200, 2)
        );

        let (_, s1) = prx.post_json("/api/steer", &json!({"action": "set_k", "k": 8})).await;
        assert_eq!((&s1["steering"]["k_override"], shared.k_override()), (&json!(8), Some(8)));
        let _ = prx.post_json("/api/steer", &json!({"action": "prioritize", "unit_id": "core_u"})).await;
        assert_eq!(shared.pop_prio().as_deref(), Some("core_u"));
    }
}
