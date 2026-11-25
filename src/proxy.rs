mod guard;
mod pacer;
mod stream;

pub use guard::{
    CacheMode, ProxyConfig, guard_lang, guard_payload, infer_attempt, keepwarm_due,
};

use crate::mvcc::MvccStore;
use crate::net::retry_http;
use crate::platform::{SieveCache, stream_http};
use crate::state::RunState;
use crate::telemetry::{EventSpec, TelemetryHub};
use crate::types::{LlmCallMetrics, ProxySnapshot};
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::StreamExt;
use guard::{find_module, header_u64, localip, parse_metrics, sys_text};
use pacer::PacerState;
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::mem::DropGuard;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::broadcast::error::RecvError;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

const DEFAULT_SSE_IDLE_MS: u64 = 45_000;

fn stall_ms() -> u64 {
    std::env::var("REFIT_SSE_STALL_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(DEFAULT_SSE_IDLE_MS, |v| v.clamp(100, 300_000))
}

pub struct ProxyState {
    cfg: ProxyConfig,
    client: reqwest::Client,
    telemetry: TelemetryHub,
    pacer: Mutex<PacerState>,
    module_metrics: SieveCache<String, LlmCallMetrics>,
    in_flight: AtomicU64,
    req_seq: AtomicU64,
    repo_root: Mutex<Option<PathBuf>>,
    shared_state: Mutex<Option<Arc<RunState>>>,
    ui_cache: Mutex<Option<(u64, String)>>,
}

fn eq_token(got: &str, expected: &str) -> bool {
    constant_time_eq::constant_time_eq(got.trim().as_bytes(), expected.as_bytes())
}

impl ProxyState {
    fn new(
        cfg: ProxyConfig,
        client: reqwest::Client,
        telemetry: TelemetryHub,
        repo_root: Option<PathBuf>,
    ) -> Self {
        Self {
            cfg,
            client,
            telemetry,
            pacer: Mutex::new(PacerState::new()),
            module_metrics: SieveCache::new(1_024, 2 * 1024 * 1024),
            in_flight: AtomicU64::new(0),
            req_seq: AtomicU64::new(1),
            repo_root: Mutex::new(repo_root),
            shared_state: Mutex::new(None),
            ui_cache: Mutex::new(None),
        }
    }

    pub fn bind_repo(&self, root: PathBuf) {
        *self.repo_root.lock() = Some(root);
    }

    pub fn attach_state(&self, st: Arc<RunState>) {
        *self.shared_state.lock() = Some(st);
    }

    pub fn snapshot(&self) -> ProxySnapshot {
        self.pacer.lock().snapshot()
    }

    pub fn take_metrics(&self, module_id: &str) -> Option<LlmCallMetrics> {
        self.module_metrics.remove(&module_id.to_string())
    }

    pub fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) {
        self.pacer.lock().timing_samples()
    }

    fn is_authorized(&self, headers: &HeaderMap) -> bool {
        let Some(ref expected) = self.cfg.auth_token else { return true };
        if expected.is_empty() {
            return true;
        }
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .is_some_and(|tok| eq_token(tok, expected))
    }

    async fn await_budget(&self, req_id: u64, est_unc: u64, est_tot: u64) {
        loop {
            let sleep_dur = self.pacer.lock().try_reserve(req_id, est_unc, est_tot);
            match sleep_dur {
                None => return,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
    }

    fn free_budget(&self, req_id: u64) {
        self.pacer.lock().free_budget(req_id);
    }

    fn apply_headers(&self, status: u16, headers: &reqwest::header::HeaderMap) {
        let tpm = header_u64(headers, "x-ratelimit-limit-tokens-minute");
        let rpm = header_u64(headers, "x-ratelimit-limit-requests-minute");
        let ra = header_u64(headers, "retry-after");
        if let Some(payload) = self.pacer.lock().apply_status(status, tpm, rpm, ra) {
            self.telemetry.emit_data("circuit_state", payload);
        }
    }

    /// Reconcile actual token usage against the pre-request budget reservation.
    fn record_usage(&self, req_id: u64, m: LlmCallMetrics) {
        let rolling = self.pacer.lock().record_usage(req_id, &m);
        if !m.module_id.is_empty() {
            self.module_metrics.insert(m.module_id.clone(), m.clone(), 256);
        }
        let mid = (!m.module_id.is_empty()).then(|| m.module_id.clone());
        let data = serde_json::to_value(&m).unwrap_or(Value::Null);
        let spec = EventSpec {
            module_id: mid,
            uncached_tpm: Some(rolling),
            data: Some(data),
            ..EventSpec::default()
        };
        self.telemetry.emit("llm_done", spec);
    }
}

pub async fn start_proxy(
    cfg: ProxyConfig,
    telemetry: TelemetryHub,
) -> Result<(Arc<ProxyState>, tokio::task::JoinHandle<()>)> {
    let pool_max = crate::platform::budget().http_pool_max;
    let client = stream_http("cerebras-cloud-sdk/1.91.0 (refit/0.5.0)", 15, 600, pool_max)?;
    let state = Arc::new(ProxyState::new(
        cfg.clone(),
        client,
        telemetry,
        std::env::current_dir().ok(),
    ));

    if cfg.keep_warm && cfg.static_prefix_override.is_some() {
        let kw = state.clone();
        tokio::spawn(async move { keepwarm_loop(kw).await });
    }

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
    let Some(prefix) = state.cfg.static_prefix_override.clone() else { return };
    if ttl_ms == 0 {
        return;
    }

    let mut ticker = tokio::time::interval(Duration::from_secs(15));
    let url = format!("{}/chat/completions", state.cfg.upstream_base_url.trim_end_matches('/'));

    loop {
        ticker.tick().await;
        let inflight = state.in_flight.load(Ordering::Relaxed);
        if !state.pacer.lock().keepwarm_check(ttl_ms, inflight, keepwarm_due) {
            continue;
        }
        let cached = send_keepwarm(&state, &url, &prefix).await;
        state.pacer.lock().keepwarm_done(cached);
    }
}

async fn send_keepwarm(state: &ProxyState, url: &str, prefix: &str) -> Option<u64> {
    let mut body = json!({
        "model": state.cfg.model_override,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "."}
        ],
        "stream": false,
        "max_completion_tokens": 1
    });
    guard_payload(&mut body, &state.cfg, "probe", 2);

    let resp = state
        .client
        .post(url)
        .bearer_auth(&state.cfg.api_key)
        .json(&body)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }

    let v: Value = resp.json().await.unwrap_or(Value::Null);
    Some(v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0))
}

async fn serve_ui(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let sig = ui_sig(&state.cfg.ui_index_path);
    if let Some(cur) = sig {
        let guard = state.ui_cache.lock();
        if let Some((_, html)) = guard.as_ref().filter(|(s, _)| *s == cur) {
            return (StatusCode::OK, Html(html.clone())).into_response();
        }
    }

    match std::fs::read_to_string(&state.cfg.ui_index_path) {
        Ok(html) => {
            if let Some(cur) = sig {
                *state.ui_cache.lock() = Some((cur, html.clone()));
            }
            (StatusCode::OK, Html(html)).into_response()
        }
        Err(_) => {
            let body = "<html><body><h1>refit telemetry proxy active</h1></body></html>";
            (StatusCode::OK, Html(body.to_string())).into_response()
        }
    }
}

fn ui_sig(path: &std::path::Path) -> Option<u64> {
    let m = std::fs::metadata(path).ok()?;
    let mt = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    Some(m.len() ^ mt.rotate_left(17))
}

fn err_401() -> Response {
    let body = json!({"error": "missing or invalid proxy bearer token"});
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

async fn steer_get(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let snap = state.shared_state.lock().as_ref().map(|s| s.steering_snapshot());
    Json(json!({"attached": snap.is_some(), "steering": snap})).into_response()
}

async fn steer_post(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Json(cmd): Json<Value>,
) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let Some(st) = state.shared_state.lock().clone() else {
        let body = json!({"error": "no active run attached for live steering"});
        return (StatusCode::CONFLICT, Json(body)).into_response();
    };

    let action = cmd["action"].as_str().unwrap_or("");
    if !apply_steer(&st, action, &cmd) {
        let msg = format!("unknown steering action `{action}`");
        return (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response();
    }

    let snap = st.steering_snapshot();
    let uid = cmd["unit_id"].as_str().map(String::from);
    let data = json!({"action": action, "steering": snap});
    let spec = EventSpec { module_id: uid, data: Some(data), ..EventSpec::default() };
    state.telemetry.emit("live_steer", spec);
    (StatusCode::OK, Json(json!({"ok": true, "steering": snap}))).into_response()
}

fn apply_steer(st: &RunState, action: &str, cmd: &Value) -> bool {
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
        _ => return false,
    }
    true
}

async fn handle_health(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(json!({"ok": true, "proxy": state.snapshot()}))
}

async fn timeline_get(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }
    let root = state.repo_root.lock().clone();
    let entries = root
        .as_deref()
        .and_then(|r| MvccStore::wal_timeline(r).ok())
        .unwrap_or_default();
    Json(json!({"entries": entries})).into_response()
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
    let found = root
        .as_deref()
        .and_then(|r| MvccStore::diff_tx(r, &tx_id).ok())
        .flatten();

    match found {
        Some(report) => (StatusCode::OK, Json(report)).into_response(),
        None => {
            let body = json!({"error": format!("tx `{tx_id}` not found")});
            (StatusCode::NOT_FOUND, Json(body)).into_response()
        }
    }
}

async fn serve_events(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let query_ok = state.cfg.auth_token.as_deref().is_none_or(|exp| {
        exp.is_empty() || params.get("token").is_some_and(|t| eq_token(t, exp))
    });
    if !state.is_authorized(&headers) && !query_ok {
        return err_401();
    }

    let history = state.telemetry.snapshot_events();
    let rx = state.telemetry.subscribe();
    let hist = futures_util::stream::iter(history.into_iter().map(sse_event));
    let live = futures_util::stream::unfold(rx, |mut r| async move {
        loop {
            match r.recv().await {
                Ok(ev) => return Some((sse_event(ev), r)),
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(hist.chain(live)).keep_alive(KeepAlive::default()).into_response()
}

fn sse_event(ev: crate::types::TelemetryEvent) -> std::result::Result<Event, !> {
    let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
    Ok(Event::default().data(data))
}

async fn handle_models(State(state): State<Arc<ProxyState>>, headers: HeaderMap) -> Response {
    if !state.is_authorized(&headers) {
        return err_401();
    }

    let url = format!("{}/models", state.cfg.upstream_base_url.trim_end_matches('/'));
    let seq = state.req_seq.fetch_add(1, Ordering::Relaxed);
    let res = retry_http(3, 150, 5_000, seq, |_| true, || {
        state
            .client
            .get(&url)
            .bearer_auth(&state.cfg.api_key)
            .timeout(Duration::from_secs(15))
            .send()
    })
    .await;

    match res {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = resp.bytes().await.unwrap_or_default();
            (status, [("content-type", "application/json")], bytes).into_response()
        }
        Err(err) => {
            let body = json!({"error": err.to_string()});
            (StatusCode::BAD_GATEWAY, Json(body)).into_response()
        }
    }
}

struct ChatCtx {
    seq: u64,
    module_id: String,
    attempt: u32,
    is_stream: bool,
    drift: bool,
    idle_ms: u64,
}

#[derive(Clone, Copy)]
struct RateLimits {
    tpm: Option<u64>,
    rem_tpm: Option<u64>,
    rpm: Option<u64>,
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
    let _guard = DropGuard::new(state.clone(), |s| {
        s.in_flight.fetch_sub(1, Ordering::Relaxed);
    });

    let module_id = hdr_str(&headers, "x-refit-module")
        .filter(|s| !s.is_empty())
        .map_or_else(|| find_module(&payload), String::from);
    let attempt = hdr_str(&headers, "x-refit-attempt")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| infer_attempt(&payload));
    let lang = hdr_str(&headers, "x-refit-lang");

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
            state.free_budget(seq);
            let body = json!({"error": err.to_string()});
            return (StatusCode::BAD_GATEWAY, Json(body)).into_response();
        }
    };

    let idle_ms = hdr_str(&headers, "x-refit-sse-stall-ms")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or_else(stall_ms, |v| v.clamp(50, 300_000));
    let cctx = ChatCtx {
        seq,
        module_id,
        attempt,
        is_stream: payload["stream"].as_bool().unwrap_or(false),
        drift,
        idle_ms,
    };
    finish_chat(state, upstream, cctx).await
}

async fn send_chat(
    state: &ProxyState,
    body_bytes: &Bytes,
    seq: u64,
) -> std::result::Result<reqwest::Response, reqwest::Error> {
    let url = format!("{}/chat/completions", state.cfg.upstream_base_url.trim_end_matches('/'));
    retry_http(
        3,
        250,
        30_000,
        seq,
        |resp| {
            let can = state.pacer.lock().allow_retry();
            if let Some(r) = resp {
                if can {
                    state.apply_headers(r.status().as_u16(), r.headers());
                }
            } else {
                state.apply_headers(503, &reqwest::header::HeaderMap::new());
            }
            can
        },
        || {
            state
                .client
                .post(&url)
                .bearer_auth(&state.cfg.api_key)
                .header("content-type", "application/json")
                .header("accept", "application/json, text/event-stream")
                .body(body_bytes.clone())
                .send()
        },
    )
    .await
}

fn finish_metrics(
    state: &ProxyState,
    v: &Value,
    module_id: &str,
    ctx: &ChatCtx,
    code: u16,
    wall: u64,
    lim: RateLimits,
) {
    let mut m = parse_metrics(v, module_id, ctx.attempt, code, wall, ctx.drift);
    m.limit_tpm_minute = lim.tpm;
    m.remaining_tpm_minute = lim.rem_tpm;
    m.limit_rpm_minute = lim.rpm;
    state.record_usage(ctx.seq, m);
}

async fn finish_chat(state: Arc<ProxyState>, upstream: reqwest::Response, ctx: ChatCtx) -> Response {
    let started = Instant::now();
    let code = upstream.status().as_u16();
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
        let wall = started.elapsed().as_millis() as u64;
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            finish_metrics(&state, &v, &ctx.module_id, &ctx, code, wall, lim);
        } else {
            state.free_budget(ctx.seq);
        }
        return (status, [("content-type", "application/json")], bytes).into_response();
    }

    let st_c = state.clone();
    let mid_c = ctx.module_id.clone();
    let body = stream::tap_stream(
        upstream,
        state.telemetry.clone(),
        ctx.module_id.clone(),
        ctx.attempt,
        ctx.idle_ms,
        started,
        move |usage, time, wall| {
            let synth = json!({"usage": usage, "time_info": time});
            finish_metrics(&st_c, &synth, &mid_c, &ctx, code, wall, lim);
        },
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(body))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream error").into_response())
}

fn hdr_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
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
    if sys.len() < 120 {
        return false;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(sys.as_bytes());
    hasher.update(b"\n---\n");
    if let Some(tools) = payload.get("tools") {
        let _ = serde_json::to_writer(&mut hasher, tools);
    }
    let hex = hasher.finalize().to_hex();
    state.pacer.lock().check_drift(lang_key, hex.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        MockHttp, TempRepo, TestProxy, commit_file, http_res, mock_sse, proxy_cfg, sse_delta,
    };
    use axum::http::HeaderValue;

    const CB_URL: &str = "https://api.cerebras.ai/v1";

    fn test_cfg(upstream: &str, retention: CacheMode) -> ProxyConfig {
        let mut cfg = proxy_cfg(upstream, retention);
        cfg.static_prefix_override = Some("S".repeat(160));
        cfg.preserve_tools = true;
        cfg
    }

    fn mock_state(tag: &str, cfg: ProxyConfig) -> (ProxyState, TempRepo) {
        let tmp = TempRepo::new(tag);
        let tel = TelemetryHub::new(tmp.join("run.jsonl")).unwrap();
        (ProxyState::new(cfg, reqwest::Client::new(), tel, None), tmp)
    }

    fn sys_msg(content: impl Into<String>) -> Value {
        json!({"messages": [{"role": "system", "content": content.into()}]})
    }

    #[test]
    fn test_payload_guard_and_routing() {
        let mut cfg = test_cfg(CB_URL, CacheMode::Short);
        cfg.preserve_tools = false;
        let mut p = json!({
            "model": "wrong", "disable_reasoning": true, "top_p": 0.9, "clear_thinking": true,
            "max_tokens": 1000, "tools": [{"type": "function"}],
            "messages": [{"role": "system", "content": "bloated"}]
        });
        guard_payload(&mut p, &cfg, "mod_01", 1);
        assert_eq!(p["model"], "moonshotai-kimi-k3.1-pro-trial");
        assert_eq!((&p["reasoning_effort"], &p["reasoning_format"]), (&json!("low"), &json!("parsed")));
        assert_eq!((&p["temperature"], &p["max_completion_tokens"]), (&json!(1.0), &json!(1000)));
        let sys_str = p["messages"][0]["content"].as_str().unwrap();
        assert!(p.get("disable_reasoning").is_none() && p.get("top_p").is_none());
        assert!(p.get("tools").is_none() && sys_str.len() == 160);

        for (attempt, want) in [(1, None), (2, Some(guard::MAX_KEY_LEN))] {
            let mut body = json!({"prompt_cache_key": "k".repeat(2000)});
            guard_payload(&mut body, &cfg, "m1", attempt);
            let len = body.get("prompt_cache_key").and_then(Value::as_str).map(str::len);
            assert_eq!(len, want, "attempt={attempt}");
        }

        for (url, ttl, ret) in [
            (CB_URL, None, None),
            ("https://api.moonshot.ai/v1", Some("1h"), None),
            ("https://api.openai.com/v1", None, Some("24h")),
        ] {
            let mut body = json!({});
            guard_payload(&mut body, &test_cfg(url, CacheMode::Long), "m", 2);
            assert_eq!(body["prompt_cache_options"]["ttl"].as_str(), ttl, "{url}");
            assert_eq!(body["prompt_cache_retention"].as_str(), ret, "{url}");
        }

        let mut tools_p = json!({
            "reasoning_effort": "medium", "prompt_cache_key": "refit:r1:m:m1",
            "tools": [
                {"type": "function", "function": {"name": "z_tool"}},
                {"type": "function", "function": {"name": "a_tool"}}
            ],
            "messages": [
                {"role": "system", "content": "sys"}, {"role": "user", "content": "Refactor `m1`"},
                {"role": "assistant", "content": "bad"}, {"role": "user", "content": "[attempt=3] retry"}
            ]
        });
        let inf = infer_attempt(&tools_p);
        assert_eq!(inf, 3);
        guard_payload(&mut tools_p, &test_cfg(CB_URL, CacheMode::Short), "m1", inf);
        assert_eq!(tools_p["reasoning_effort"], "medium");
        assert_eq!(tools_p["tools"][0]["function"]["name"], "a_tool");
        assert_eq!(tools_p["prompt_cache_key"], "refit:r1:m:m1");
    }

    #[test]
    fn test_keepwarm_policy() {
        let cases = [
            (100_000, 0, 0, false, false),
            (250_000, 0, 0, false, true),
            (295_000, 0, 0, false, false),
            (400_000, 0, 0, false, false),
            (250_000, 1, 0, false, false),
            (250_000, 0, 11, false, false),
            (250_000, 0, 2, true, false),
        ];
        for (ms, inflight, sent, open, want) in cases {
            assert_eq!(keepwarm_due(ms, ms, 300_000, inflight, sent, 11, open), want);
        }
    }

    #[tokio::test]
    async fn test_pacer_drift_circuit_and_cors() {
        let (st, _tmp) = mock_state("pac", test_cfg(CB_URL, CacheMode::Short));
        let p1 = sys_msg("A".repeat(150));
        let p2 = sys_msg("A".repeat(149) + "B");
        for (msg, lang, drifted) in [
            (sys_msg("OK"), "default", false),
            (p1.clone(), "default", false),
            (p1, "default", false),
            (p2, "default", true),
            (sys_msg("P".repeat(150)), "python", false),
            (sys_msg("R".repeat(150)), "rust", false),
            (sys_msg("P".repeat(150)), "python", false),
        ] {
            assert_eq!(check_drift(&st, &msg, lang), drifted, "lang={lang}");
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
        p.requests_sent = 50;
        assert_eq!((0..30).filter(|_| p.allow_retry()).count(), 10);

        for (orig, want) in [
            ("http://127.0.0.1:18790", true), ("http://localhost:3000", true),
            ("https://[::1]:8443", true), ("http://localhost.evil.com", false),
            ("https://127.0.0.1.attacker.io", false), ("http://127.0.0.1@evil.com", false),
        ] {
            assert_eq!(localip(&HeaderValue::from_static(orig)), want, "{orig}");
        }
    }

    #[tokio::test]
    async fn test_proxy_upstream_retry_policy() {
        let ok = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#;
        let cases = [
            (vec![http_res("503 Service Unavailable", "Retry-After: 0.02\r\n", "down"), http_res("200 OK", "", ok)], 200, 2),
            (vec![http_res("404 Not Found", "", r#"{"error":"not_found"}"#)], 404, 1),
        ];
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": false});
        for (resps, want_code, want_calls) in cases {
            let up = MockHttp::spawn(resps).await;
            let prx = TestProxy::spawn(&up.url("/v1"), CacheMode::Short, None).await;
            let (code, _) = prx.post_json("/v1/chat/completions", &body).await;
            assert_eq!((code, up.calls()), (want_code, want_calls));
        }
    }

    #[tokio::test]
    async fn test_sse_stall_watchdog() {
        let up_addr = mock_sse(vec![sse_delta("hi")], 0, 10).await;
        let prx = TestProxy::spawn(&format!("http://{up_addr}/v1"), CacheMode::Short, None).await;

        let started = Instant::now();
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true});
        let resp = prx
            .client
            .post(prx.url("/v1/chat/completions"))
            .header("x-refit-sse-stall-ms", "120")
            .json(&body)
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
        assert!(saw_err && started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn test_proxy_api_auth_timeline_and_steer() {
        let tmp = TempRepo::with_files("prx_api", &[("app.py", "def f():\n    return 1\n")]);
        let store = MvccStore::open(tmp.path(), "prx_api").unwrap();
        let (_, tx_id, _) = commit_file(&store, tmp.path(), "app", "app.py", "def f():\n    return 42\n");

        let up = MockHttp::spawn(vec![
            http_res("503 Service Unavailable", "Retry-After: 0.01\r\n", "busy"),
            http_res("200 OK", "", r#"{"data":[{"id":"kimi"}]}"#),
        ])
        .await;
        let tok = "secret-run-token";
        let prx = TestProxy::spawn(&up.url("/v1"), CacheMode::Short, Some(tok)).await;
        let shared = RunState::new(100);
        prx.state.bind_repo(tmp.path().to_path_buf());
        prx.state.attach_state(shared.clone());

        let unauth = prx
            .client
            .get(prx.url("/api/timeline"))
            .header("origin", "https://evil.example")
            .send()
            .await
            .unwrap();
        assert_eq!(unauth.status().as_u16(), 401);
        assert!(unauth.headers().get("access-control-allow-origin").is_none());

        let authed = prx
            .client
            .get(prx.url("/api/timeline"))
            .bearer_auth(tok)
            .header("origin", "http://127.0.0.1:18790")
            .send()
            .await
            .unwrap();
        assert_eq!(authed.status().as_u16(), 200);
        let cors = authed.headers().get("access-control-allow-origin");
        assert_eq!(cors.and_then(|v| v.to_str().ok()), Some("http://127.0.0.1:18790"));

        let (_, tl) = prx.get_json("/api/timeline").await;
        assert_eq!(tl["entries"][0]["tx_id"], json!(tx_id));
        let (code, dj) = prx.get_json(&format!("/api/diff/{tx_id}")).await;
        let udiff = dj["files"][0]["unified_diff"].as_str().unwrap();
        assert_eq!(code, 200);
        assert!(udiff.contains("+    return 42"));
        assert_eq!(prx.get_json("/api/diff/tx-none").await.0, 404);

        assert_eq!((prx.get_json("/v1/models").await.0, up.calls()), (200, 2));
        let (_, s1) = prx.post_json("/api/steer", &json!({"action": "set_k", "k": 8})).await;
        assert_eq!((&s1["steering"]["k_override"], shared.k_override()), (&json!(8), Some(8)));
        let _ = prx.post_json("/api/steer", &json!({"action": "prioritize", "unit_id": "core_u"})).await;
        assert_eq!(shared.pop_prio().as_deref(), Some("core_u"));
    }
}
