use crate::platform::{write_atomic, now_ms};
use crate::state::RunState;
use crate::types::{RunSummary, TelemetryEvent};
use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;

const MAX_HISTORY: usize = 2048;

#[derive(Clone)]
pub struct TelemetryHub {
    inner: Arc<TelemetryInner>,
}

struct TelemetryInner {
    jsonl_path: PathBuf,
    seq: AtomicU64,
    writer: Mutex<BufWriter<File>>,
    history: Mutex<VecDeque<TelemetryEvent>>,
    last_unit_sig: DashMap<String, u64>,
    dedup_skipped: AtomicU64,
    bytes_written: AtomicU64,
    shared_state: Mutex<Option<Arc<RunState>>>,
    tx: broadcast::Sender<TelemetryEvent>,
}

impl TelemetryHub {
    pub fn new(jsonl_path: PathBuf) -> Result<Self> {
        Self::open_hub(jsonl_path, false)
    }

    pub fn open_hub(jsonl_path: PathBuf, append: bool) -> Result<Self> {
        if let Some(p) = jsonl_path.parent() {
            fs::create_dir_all(p)
                .with_context(|| format!("create telemetry dir {}", p.display()))?;
        }
        let mut opts = OpenOptions::new();
        opts.create(true).write(true);
        if append {
            opts.append(true);
        } else {
            opts.truncate(true);
        }
        let file = opts
            .open(&jsonl_path)
            .with_context(|| format!("open telemetry {}", jsonl_path.display()))?;
        let (tx, _rx) = broadcast::channel(2048);

        Ok(Self {
            inner: Arc::new(TelemetryInner {
                jsonl_path,
                seq: AtomicU64::new(1),
                writer: Mutex::new(BufWriter::with_capacity(32 * 1024, file)),
                history: Mutex::new(VecDeque::with_capacity(512)),
                last_unit_sig: DashMap::new(),
                dedup_skipped: AtomicU64::new(0),
                bytes_written: AtomicU64::new(0),
                shared_state: Mutex::new(None),
                tx,
            }),
        })
    }

    pub fn bind_state(&self, st: Arc<RunState>) {
        *self.inner.shared_state.lock() = Some(st);
    }

    pub fn dedup_skipped(&self) -> u64 {
        self.inner.dedup_skipped.load(Ordering::Relaxed)
    }

    pub fn bytes_written(&self) -> u64 {
        self.inner.bytes_written.load(Ordering::Relaxed)
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
        if matches!(kind, "llm_stream" | "wave_dispatch" | "heartbeat") {
            if let Some(ref mid) = module_id {
                let extra = data
                    .as_ref()
                    .and_then(|d| d.get("attempt"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let sig = fast_sig(kind, uncached_tpm.unwrap_or(0) ^ extra.rotate_left(19));
                let prev = self.inner.last_unit_sig.insert(mid.clone(), sig);
                if prev == Some(sig) {
                    self.inner.dedup_skipped.fetch_add(1, Ordering::Relaxed);
                    if let Some(st) = self.inner.shared_state.lock().as_ref() {
                        st.count_dedup(1);
                    }
                    return;
                }
            }
        }

        let redacted_data = data.map(redact_json);
        let ev = TelemetryEvent {
            seq: self.inner.seq.fetch_add(1, Ordering::Relaxed),
            ts_ms: now_ms(),
            kind: kind.to_string(),
            module_id: module_id.map(redact_secrets),
            layer,
            wave,
            uncached_tpm,
            data: redacted_data,
            summary,
        };

        if let Ok(mut line) = serde_json::to_string(&ev) {
            line = redact_secrets(line);
            let mut w = self.inner.writer.lock();
            if w.write_all(line.as_bytes()).is_ok() && w.write_all(b"\n").is_ok() {
                self.inner
                    .bytes_written
                    .fetch_add((line.len() + 1) as u64, Ordering::Relaxed);
                if matches!(
                    kind,
                    "run_start"
                        | "dag_ready"
                        | "mvcc_commit"
                        | "run_end"
                        | "circuit_state"
                        | "unit_blocked"
                ) {
                    let _ = w.flush();
                }
            }
        }

        {
            let mut hist = self.inner.history.lock();
            if hist.len() >= MAX_HISTORY {
                hist.pop_front();
            }
            hist.push_back(ev.clone());
        }

        let _ = self.inner.tx.send(ev);
    }

    pub fn flush_zstd(&self) -> Result<(PathBuf, u64, u64)> {
        {
            let mut w = self.inner.writer.lock();
            w.flush()?;
        }
        let raw = fs::read(&self.inner.jsonl_path)?;
        let raw_len = raw.len() as u64;
        let compressed = zstd::stream::encode_all(raw.as_slice(), 6)
            .context("zstd compress telemetry")?;
        let zst_len = compressed.len() as u64;
        let zst_path = self.inner.jsonl_path.with_extension("jsonl.zst");
        write_atomic(&zst_path, &compressed)
            .with_context(|| format!("write compact zstd {}", zst_path.display()))?;
        Ok((zst_path, raw_len, zst_len))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.inner.tx.subscribe()
    }

    pub fn snapshot_events(&self) -> Vec<TelemetryEvent> {
        self.inner.history.lock().iter().cloned().collect()
    }
}

pub use crate::platform::redact_secrets;

fn redact_json(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => serde_json::Value::String(redact_secrets(s)),
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(redact_json).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .map(|(k, val)| (k, redact_json(val)))
                .collect(),
        ),
        other => other,
    }
}

fn fast_sig(kind: &str, val: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ val;
    for b in kind.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_41_telemetry_zstd() {
        let tmp = std::env::temp_dir().join(format!(
            "refit_tel_test_{}_{}.jsonl",
            std::process::id(),
            now_ms()
        ));
        let hub = TelemetryHub::new(tmp.clone()).unwrap();

        for _ in 0..8 {
            hub.emit("llm_stream", Some("mod_01".into()), Some(0), Some(0), Some(24000), None, None);
        }
        assert_eq!(hub.dedup_skipped(), 7);

        let fake_csk = ["csk", "-", "dummy000000000000"].concat();
        let fake_keen = ["keen", "_", "dummy000000000000"].concat();
        hub.emit(
            "run_start",
            None,
            None,
            None,
            Some(0),
            Some(serde_json::json!({
                "a": fake_csk,
                "b": fake_keen
            })),
            None,
        );
        for i in 0..30 {
            hub.emit(
                "mvcc_commit",
                Some(format!("mod_{i:02}")),
                Some(i / 10),
                Some(i / 10),
                Some(32000),
                Some(serde_json::json!({"head_rev": i + 1, "ruff_errors": 0, "ty_errors": 0})),
                None,
            );
        }

        let (zst_path, raw_len, zst_len) = hub.flush_zstd().unwrap();
        let disk_text = fs::read_to_string(&tmp).unwrap();
        assert!(!disk_text.contains("csk-"));
        assert!(!disk_text.contains("keen_"));
        assert!(zst_len < raw_len / 2);

        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(&zst_path);
    }

    #[test]
    fn probe_42_redact_multi() {
        let line = format!(
            "first={} mid={} second={} tail={}",
            ["csk", "-", "aaa111222333"].concat(),
            ["keen", "_", "bbb444555666"].concat(),
            ["csk", "-", "ccc777888999"].concat(),
            ["keen", "_", "ddd000111222"].concat()
        );
        let clean = redact_secrets(line);
        assert!(!clean.contains("csk-"));
        assert!(!clean.contains("keen_"));
        assert_eq!(clean.matches("REDACTED_CSK_").count(), 2);
        assert_eq!(clean.matches("REDACTED_KEEN_").count(), 2);
    }

    #[test]
    fn probe_68_telemetry_resume() {
        let tmp = std::env::temp_dir().join(format!("refit_tel68_{}_{}.jsonl", std::process::id(), now_ms()));
        {
            let h1 = TelemetryHub::new(tmp.clone()).unwrap();
            h1.emit("run_start", None, None, None, Some(0), None, None);
            let _ = h1.flush_zstd().unwrap();
        }
        {
            let h2 = TelemetryHub::open_hub(tmp.clone(), true).unwrap();
            h2.emit("run_end", None, None, None, Some(0), None, None);
            let (zst, raw_len, zst_len) = h2.flush_zstd().unwrap();
            assert!(raw_len > 0 && zst_len > 0);
            let text = fs::read_to_string(&tmp).unwrap();
            assert!(text.contains("run_start") && text.contains("run_end"));
            let _ = fs::remove_file(&zst);
        }
        let _ = fs::remove_file(&tmp);
    }
}
