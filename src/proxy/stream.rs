use crate::net::drain_sse;
use crate::telemetry::{EventSpec, TelemetryHub};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::io::{Error as IoError, ErrorKind};
use std::time::{Duration, Instant};

pub(super) fn tap_stream(
    upstream: reqwest::Response,
    telemetry: TelemetryHub,
    module_id: String,
    attempt: u32,
    idle_ms: u64,
    started: Instant,
    on_done: impl FnOnce(Option<Value>, Option<Value>, u64) + Send + 'static,
) -> impl futures_util::Stream<Item = std::result::Result<Bytes, IoError>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Bytes, IoError>>(64);

    tokio::spawn(async move {
        let (usage, time) =
            forward_sse(upstream, &tx, &telemetry, &module_id, attempt, idle_ms).await;
        on_done(usage, time, started.elapsed().as_millis() as u64);
    });
    futures_util::stream::unfold(rx, |mut r| async move {
        r.recv().await.map(|item| (item, r))
    })
}

async fn forward_sse(
    upstream: reqwest::Response,
    tx: &tokio::sync::mpsc::Sender<std::result::Result<Bytes, IoError>>,
    telemetry: &TelemetryHub,
    module_id: &str,
    attempt: u32,
    idle_ms: u64,
) -> (Option<Value>, Option<Value>) {
    let mut byte_stream = upstream.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let (mut last_usage, mut last_time) = (None, None);
    let watchdog = Duration::from_millis(idle_ms.max(50));

    loop {
        let Ok(next_item) = tokio::time::timeout(watchdog, byte_stream.next()).await else {
            emit_stall(telemetry, module_id, attempt, idle_ms);
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

        buf.extend_from_slice(&chunk);
        drain_sse(&mut buf, false, |v| capture_meta(&v, &mut last_usage, &mut last_time));
        if tx.send(Ok(chunk)).await.is_err() {
            break;
        }
    }

    drain_sse(&mut buf, true, |v| capture_meta(&v, &mut last_usage, &mut last_time));
    (last_usage, last_time)
}

fn capture_meta(v: &Value, usage: &mut Option<Value>, time: &mut Option<Value>) {
    if v.get("usage").is_some_and(|u| !u.is_null()) {
        *usage = v.get("usage").cloned();
    }
    if v.get("time_info").is_some_and(|t| !t.is_null()) {
        *time = v.get("time_info").cloned();
    }
}

fn emit_stall(telemetry: &TelemetryHub, module_id: &str, attempt: u32, idle_ms: u64) {
    let mid = (!module_id.is_empty()).then(|| module_id.to_string());
    let data = json!({"idle_timeout_ms": idle_ms, "attempt": attempt});
    telemetry.emit(
        "sse_stall",
        EventSpec { module_id: mid, data: Some(data), ..EventSpec::default() },
    );
}
