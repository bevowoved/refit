use crate::mvcc::MvccStore;
use crate::telemetry::TelemetryHub;
use crate::types::{LlmCallMetrics, ProxySnapshot};
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dashmap::DashMap;
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_base_url: String,
    pub api_key: String,
    pub model_override: String,
    pub ui_index_path: PathBuf,
    pub static_prefix_override: Option<String>,
}

struct RollingEntry {
    at: Instant,
    uncached: u64,
    total: u64,
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
    prefix_hash: Option<String>,
    prefix_drift_events: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_completion_tokens: u64,
    total_uncached_tokens: u64,
    queue_ms_samples: Vec<f64>,
    decode_tps_samples: Vec<f64>,
}

impl Default for PacerState {
    fn default() -> Self {
        Self {
            window: VecDeque::new(),
            // Live-verified hackathon defaults: 50K uncached / 500K total / 1,000 RPM -> 88% safe ceiling
            uncached_tpm_cap: 44_000,
            total_tpm_cap: 440_000,
            rpm_cap: 500,
            http_429_count: 0,
            http_5xx_count: 0,
            consecutive_outage: 0,
            circuit_state: "CLOSED".to_string(),
            circuit_open_until: None,
            circuit_pauses: 0,
            prefix_hash: None,
            prefix_drift_events: 0,
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
            total_completion_tokens: 0,
            total_uncached_tokens: 0,
            queue_ms_samples: Vec::new(),
            decode_tps_samples: Vec::new(),
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
}

pub struct ProxyState {
    cfg: ProxyConfig,
    client: reqwest::Client,
    telemetry: TelemetryHub,
    pacer: Mutex<PacerState>,
    module_metrics: DashMap<String, LlmCallMetrics>,
}

impl ProxyState {
    pub fn snapshot(&self) -> ProxySnapshot {
        let mut p = self.pacer.lock().unwrap();
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

    pub fn take_module_metrics(&self, module_id: &str) -> Option<LlmCallMetrics> {
        self.module_metrics.remove(module_id).map(|(_, v)| v)
    }

    pub fn timing_samples(&self) -> (Vec<f64>, Vec<f64>) {
        let p = self.pacer.lock().unwrap();
        (p.queue_ms_samples.clone(), p.decode_tps_samples.clone())
    }

    async fn wait_for_budget(&self, est_uncached: u64, est_total: u64) {
        loop {
            let sleep_dur = {
                let mut p = self.pacer.lock().unwrap();
                // 1. Outage circuit breaker check
                if let Some(until) = p.circuit_open_until {
                    let now = Instant::now();
                    if now < until {
                        Some(until - now)
                    } else {
                        p.circuit_open_until = None;
                        p.circuit_state = "HALF_OPEN".into();
                        None
                    }
                } else {
                    let (unc, tot, rpm) = p.rolling_totals();
                    if unc + est_uncached <= p.uncached_tpm_cap
                        && tot + est_total <= p.total_tpm_cap
                        && rpm + 1 <= p.rpm_cap
                    {
                        // Reserve estimated budget immediately so parallel callers don't overshoot
                        p.window.push_back(RollingEntry {
                            at: Instant::now(),
                            uncached: est_uncached,
                            total: est_total,
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

    fn update_from_headers(&self, status: u16, headers: &reqwest::header::HeaderMap) {
        let mut p = self.pacer.lock().unwrap();
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

    fn record_usage_metrics(&self, m: LlmCallMetrics, est_uncached: u64, est_total: u64) {
        let rolling_uncached = {
            let mut p = self.pacer.lock().unwrap();
            p.total_prompt_tokens += m.prompt_tokens;
            p.total_cached_tokens += m.cached_tokens;
            p.total_completion_tokens += m.completion_tokens;
            p.total_uncached_tokens += m.uncached_tokens;
            if m.queue_ms > 0.0 {
                p.queue_ms_samples.push(m.queue_ms);
            }
            if m.decode_tps > 0.0 {
                p.decode_tps_samples.push(m.decode_tps);
            }
            // Adjust most recent reservation delta
            let extra_unc = m.uncached_tokens.saturating_sub(est_uncached);
            let actual_tot = m.prompt_tokens + m.completion_tokens;
            let extra_tot = actual_tot.saturating_sub(est_total);
            if extra_unc > 0 || extra_tot > 0 {
                p.window.push_back(RollingEntry {
                    at: Instant::now(),
                    uncached: extra_unc,
                    total: extra_tot,
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

pub async fn start_proxy_server(
    cfg: ProxyConfig,
    telemetry: TelemetryHub,
) -> Result<(Arc<ProxyState>, tokio::task::JoinHandle<()>)> {
    let client = reqwest::Client::builder()
        .user_agent("cerebras-cloud-sdk/1.91.0 (refit/0.3.0)")
        .pool_max_idle_per_host(32)
        .timeout(Duration::from_secs(180))
        .build()
        .context("build reqwest HTTP client")?;

    let state = Arc::new(ProxyState {
        cfg: cfg.clone(),
        client,
        telemetry,
        pacer: Mutex::new(PacerState::default()),
        module_metrics: DashMap::new(),
    });

    let app = Router::new()
        .route("/", get(handle_ui_index))
        .route("/health", get(handle_health))
        .route("/events", get(handle_sse_events))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .layer(CorsLayer::permissive())
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

async fn handle_ui_index(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    match std::fs::read_to_string(&state.cfg.ui_index_path) {
        Ok(html) => (StatusCode::OK, Html(html)).into_response(),
        Err(_) => (
            StatusCode::OK,
            Html("<html><body><h1>refit telemetry proxy active</h1></body></html>".to_string()),
        )
            .into_response(),
    }
}

async fn handle_health(State(state): State<Arc<ProxyState>>) -> Json<Value> {
    Json(serde_json::json!({
        "ok": true,
        "proxy": state.snapshot()
    }))
}

async fn handle_sse_events(
    State(state): State<Arc<ProxyState>>,
) -> Sse<impl futures_util::Stream<Item = std::result::Result<Event, Infallible>>> {
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

    Sse::new(hist_stream.chain(live_stream)).keep_alive(KeepAlive::default())
}

async fn handle_models(State(state): State<Arc<ProxyState>>) -> Response {
    let url = format!("{}/models", state.cfg.upstream_base_url.trim_end_matches('/'));
    match state
        .client
        .get(&url)
        .bearer_auth(&state.cfg.api_key)
        .send()
        .await
    {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = resp.bytes().await.unwrap_or_default();
            (status, [("content-type", "application/json")], bytes).into_response()
        }
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

async fn handle_chat_completions(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Response {
    let mut module_id = headers
        .get("x-refit-module")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if module_id.is_empty() {
        module_id = infer_module_id_from_payload(&payload);
    }
    let attempt: u32 = headers
        .get("x-refit-attempt")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // 1. Exact Cerebras Kimi K3.1 Pro wire guard
    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(obj) = payload.as_object_mut() {
        if !state.cfg.model_override.is_empty() {
            obj.insert("model".into(), Value::String(state.cfg.model_override.clone()));
        }
        // Strict Cerebras Kimi K3.1 Pro hackathon wire guard
        obj.insert("reasoning_effort".into(), Value::String("low".into()));
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
            .clamp(256, 6144);
        obj.insert("max_completion_tokens".into(), Value::from(max_comp));
        // Compact OpenClaw's ~21K-token default workspace/tool schema overhead on ASRF refit turns
        // into the byte-identical static SKILL.md prefix so every worker stays inside the 50K uncached TPM cap.
        if !module_id.is_empty() && module_id != "probe" {
            obj.remove("tools");
            obj.remove("tool_choice");
            if let Some(ref static_sys) = state.cfg.static_prefix_override {
                if let Some(msgs) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
                    if let Some(first) = msgs.first_mut() {
                        if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                            first["content"] = Value::String(static_sys.clone());
                        }
                    }
                }
            }
        }
        if is_stream {
            obj.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }
        // Never funnel parallel Turn 1 workers onto one cache key
        if attempt <= 1 {
            obj.remove("prompt_cache_key");
        }
    }

    // 2. BLAKE3 static prefix drift check (messages[0] + tools)
    let prefix_drift = check_prefix_drift(&state, &payload);

    // 3. Estimate token spend and wait cleanly for 3-bucket pacer permit
    let body_bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let est_total = ((body_bytes.len() as u64) / 4).max(300) + 450;
    let est_uncached = if attempt <= 1 { 950 } else { 700 };
    state.wait_for_budget(est_uncached, est_total).await;

    let url = format!(
        "{}/chat/completions",
        state.cfg.upstream_base_url.trim_end_matches('/')
    );
    let started = Instant::now();

    let req_builder = state
        .client
        .post(&url)
        .bearer_auth(&state.cfg.api_key)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .body(body_bytes);

    let upstream = match req_builder.send().await {
        Ok(r) => r,
        Err(err) => {
            state.update_from_headers(503, &reqwest::header::HeaderMap::new());
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };

    let status_u16 = upstream.status().as_u16();
    let status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let up_headers = upstream.headers().clone();
    state.update_from_headers(status_u16, &up_headers);

    let limit_tpm = header_u64(&up_headers, "x-ratelimit-limit-tokens-minute");
    let remaining_tpm = header_u64(&up_headers, "x-ratelimit-remaining-tokens-minute");
    let limit_rpm = header_u64(&up_headers, "x-ratelimit-limit-requests-minute");

    if !is_stream || status_u16 != 200 {
        let bytes = upstream.bytes().await.unwrap_or_default();
        let wall_ms = started.elapsed().as_millis() as u64;
        if let Ok(json) = serde_json::from_slice::<Value>(&bytes) {
            let mut m = extract_metrics_from_json(&json, &module_id, attempt, status_u16, wall_ms, prefix_drift);
            m.limit_tpm_minute = limit_tpm;
            m.remaining_tpm_minute = remaining_tpm;
            m.limit_rpm_minute = limit_rpm;
            state.record_usage_metrics(m, est_uncached, est_total);
        }
        return (status, [("content-type", "application/json")], bytes).into_response();
    }

    // Stream SSE transparently while tapping usage & time_info from final chunk
    let state_clone = state.clone();
    let stream = async_stream_sse_tap(
        upstream,
        state_clone,
        module_id,
        attempt,
        status_u16,
        started,
        prefix_drift,
        limit_tpm,
        remaining_tpm,
        limit_rpm,
        est_uncached,
        est_total,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn async_stream_sse_tap(
    upstream: reqwest::Response,
    state: Arc<ProxyState>,
    module_id: String,
    attempt: u32,
    status_u16: u16,
    started: Instant,
    prefix_drift: bool,
    limit_tpm: Option<u64>,
    remaining_tpm: Option<u64>,
    limit_rpm: Option<u64>,
    est_uncached: u64,
    est_total: u64,
) -> impl futures_util::Stream<Item = std::result::Result<bytes::Bytes, std::io::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<bytes::Bytes, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut byte_stream = upstream.bytes_stream();
        let mut line_buf = String::new();
        let mut last_usage: Option<Value> = None;
        let mut last_time: Option<Value> = None;

        while let Some(chunk_res) = byte_stream.next().await {
            match chunk_res {
                Ok(chunk) => {
                    if let Ok(s) = std::str::from_utf8(&chunk) {
                        line_buf.push_str(s);
                        while let Some(nl) = line_buf.find('\n') {
                            let line = line_buf[..nl].trim().to_string();
                            line_buf.drain(..=nl);
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
                    }
                    if tx.send(Ok(chunk)).await.is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx
                        .send(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            err.to_string(),
                        )))
                        .await;
                    break;
                }
            }
        }

        let wall_ms = started.elapsed().as_millis() as u64;
        let synthetic = serde_json::json!({
            "usage": last_usage,
            "time_info": last_time,
        });
        let mut m = extract_metrics_from_json(
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
        state.record_usage_metrics(m, est_uncached, est_total);
    });

    tokio_stream::wrappers::ReceiverStream::new(rx)
}

fn check_prefix_drift(state: &ProxyState, payload: &Value) -> bool {
    let sys = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|arr| arr.first())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if sys.len() < 120 {
        return false; // Ignore tiny 1-token probes
    }
    let tools = payload
        .get("tools")
        .map(|t| t.to_string())
        .unwrap_or_default();
    let hash = MvccStore::hash_bytes(format!("{sys}\n---\n{tools}").as_bytes());
    let mut p = state.pacer.lock().unwrap();
    match &p.prefix_hash {
        None => {
            p.prefix_hash = Some(hash);
            false
        }
        Some(prev) if prev == &hash => false,
        Some(_) => {
            p.prefix_drift_events += 1;
            true
        }
    }
}

fn extract_metrics_from_json(
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

fn infer_module_id_from_payload(payload: &Value) -> String {
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
