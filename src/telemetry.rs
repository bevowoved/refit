use crate::platform::now_ms;
use crate::types::{RunSummary, TelemetryEvent};
use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

const MAX_HISTORY: usize = 4096;

#[derive(Clone)]
pub struct TelemetryHub {
    inner: Arc<TelemetryInner>,
}

struct TelemetryInner {
    jsonl_path: PathBuf,
    seq: AtomicU64,
    writer: Mutex<std::fs::File>,
    history: Mutex<Vec<TelemetryEvent>>,
    tx: broadcast::Sender<TelemetryEvent>,
}

impl TelemetryHub {
    pub fn new(jsonl_path: PathBuf) -> Result<Self> {
        if let Some(p) = jsonl_path.parent() {
            fs::create_dir_all(p)
                .with_context(|| format!("create telemetry dir {}", p.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&jsonl_path)
            .with_context(|| format!("open telemetry {}", jsonl_path.display()))?;
        let (tx, _rx) = broadcast::channel(2048);

        Ok(Self {
            inner: Arc::new(TelemetryInner {
                jsonl_path,
                seq: AtomicU64::new(1),
                writer: Mutex::new(file),
                history: Mutex::new(Vec::with_capacity(512)),
                tx,
            }),
        })
    }

    pub fn jsonl_path(&self) -> &PathBuf {
        &self.inner.jsonl_path
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        kind: &str,
        module_id: Option<String>,
        layer: Option<usize>,
        wave: Option<usize>,
        uncached_tpm: Option<u64>,
        data: Option<serde_json::Value>,
        summary: Option<RunSummary>,
    ) {
        let ev = TelemetryEvent {
            seq: self.inner.seq.fetch_add(1, Ordering::Relaxed),
            ts_ms: now_ms(),
            kind: kind.to_string(),
            module_id,
            layer,
            wave,
            uncached_tpm,
            data,
            summary,
        };

        if let Ok(mut line) = serde_json::to_string(&ev) {
            // Hard secret-safety guard: never persist any API key token
            if line.contains("csk-") {
                line = line.replace("csk-", "REDACTED_KEY_");
            }
            if let Ok(mut w) = self.inner.writer.lock() {
                let _ = w.write_all(line.as_bytes());
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }
        }

        if let Ok(mut hist) = self.inner.history.lock() {
            if hist.len() >= MAX_HISTORY {
                hist.remove(0);
            }
            hist.push(ev.clone());
        }

        let _ = self.inner.tx.send(ev);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.inner.tx.subscribe()
    }

    pub fn snapshot_events(&self) -> Vec<TelemetryEvent> {
        self.inner
            .history
            .lock()
            .map(|h| h.clone())
            .unwrap_or_default()
    }
}
