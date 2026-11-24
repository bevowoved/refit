use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::future::Future;
use std::time::{Duration, SystemTime};

#[inline]
pub fn is_retryable(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
}

pub fn retry_ms(header_val: &str, max_cap_ms: u64) -> Option<u64> {
    let trimmed = header_val.trim();
    if let Ok(secs) = trimmed.parse::<f64>()
        && secs.is_finite()
        && secs >= 0.0
    {
        return Some(((secs * 1000.0).round() as u64).min(max_cap_ms));
    }

    let target = httpdate::parse_http_date(trimmed).ok()?;
    let wait_ms = target
        .duration_since(SystemTime::now())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    Some(wait_ms.min(max_cap_ms))
}

pub fn backoff_ms(
    attempt: u32,
    base_ms: u64,
    cap_ms: u64,
    retry_after_ms: Option<u64>,
    jitter_seed: u64,
) -> u64 {
    if let Some(ra) = retry_after_ms {
        return ra.min(cap_ms);
    }

    let shift = 1u64.checked_shl(attempt.min(10)).unwrap_or(1024);
    let exp = base_ms.saturating_mul(shift).min(cap_ms);
    let half = exp / 2;
    let span = (exp - half).max(1);
    let mix = jitter_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(attempt as u64);
    half + (mix % span)
}

pub fn build_http(
    user_agent: &str,
    connect_secs: u64,
    idle_secs: u64,
    pool_max: usize,
    total_timeout_secs: Option<u64>,
) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(user_agent)
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .connect_timeout(Duration::from_secs(connect_secs))
        .pool_max_idle_per_host(pool_max);

    builder = builder
        .pool_idle_timeout(Some(Duration::from_secs(idle_secs)))
        .http2_keep_alive_interval(Some(Duration::from_secs(15)))
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .http2_adaptive_window(true);

    if let Some(secs) = total_timeout_secs {
        builder = builder.timeout(Duration::from_secs(secs));
    }
    builder.build().context("build HTTP/2 keep-alive client")
}

/// Drain complete SSE `data: {...}` frames using one cursor pass.
pub fn drain_sse(buf: &mut Vec<u8>, flush: bool, mut on_val: impl FnMut(Value)) {
    let mut cursor = 0usize;
    while let Some(rel_nl) = buf[cursor..].iter().position(|&b| b == b'\n') {
        let end = cursor + rel_nl;
        parse_sse(&buf[cursor..end], &mut on_val);
        cursor = end + 1;
    }

    if cursor > 0 {
        buf.drain(..cursor);
    }
    if flush && !buf.is_empty() {
        parse_sse(buf, &mut on_val);
        buf.clear();
    }
}

fn parse_sse(raw: &[u8], on_val: &mut impl FnMut(Value)) {
    let line = String::from_utf8_lossy(raw);
    let Some(data) = line.trim().strip_prefix("data: ") else { return };
    if data == "[DONE]" {
        return;
    }
    if let Ok(v) = serde_json::from_str::<Value>(data) {
        on_val(v);
    }
}

pub fn retry_hdr(headers: &reqwest::header::HeaderMap, cap_ms: u64) -> Option<u64> {
    let raw = headers.get("retry-after")?.to_str().ok()?;
    retry_ms(raw, cap_ms)
}

pub async fn retry_http<F, Fut>(
    attempts: u32,
    base_ms: u64,
    cap_ms: u64,
    seed: u64,
    mut allow_retry: impl FnMut(Option<&reqwest::Response>) -> bool,
    mut send: F,
) -> std::result::Result<reqwest::Response, reqwest::Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = std::result::Result<reqwest::Response, reqwest::Error>>,
{
    for attempt in 0..attempts {
        let more = attempt + 1 < attempts;
        let res = send().await;
        let ra = match &res {
            Ok(r) if more && is_retryable(r.status().as_u16()) && allow_retry(Some(r)) => {
                Some(retry_hdr(r.headers(), cap_ms))
            }
            Err(_) if more && allow_retry(None) => Some(None),
            _ => return res,
        };
        let wait = backoff_ms(attempt, base_ms, cap_ms, ra.flatten(), seed);
        tokio::time::sleep(Duration::from_millis(wait)).await;
    }
    unreachable!("attempts must be >= 1")
}

pub fn json_str(v: &Value, key: &str) -> String {
    let raw = v.get(key).and_then(Value::as_str);
    raw.unwrap_or("").to_string()
}

pub fn json_path(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .replace('\\', "/")
}

pub fn rpc_ok(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn rpc_err(id: Value, code: i64, msg: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg.into() } })
}

pub fn rpc_text(id: Option<Value>, payload: &Value) -> Value {
    let text = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".into());
    rpc_ok(id, json!({ "content": [{ "type": "text", "text": text }] }))
}
