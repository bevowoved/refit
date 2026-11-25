use crate::platform::{SieveCache, now_ms};
use crate::state::RunState;
use crate::types::{ModuleUnit, RunSummary, TelemetryEvent};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;

pub use crate::platform::redact_secrets;

const MAX_HISTORY: usize = 2048;

#[derive(Default)]
pub struct EventSpec {
    pub data: Option<Value>,
    pub summary: Option<RunSummary>,
    pub uncached_tpm: Option<u64>,
    pub module_id: Option<String>,
    pub layer: Option<usize>,
    pub wave: Option<usize>,
}

impl EventSpec {
    pub fn unit(u: &ModuleUnit) -> Self {
        Self {
            module_id: Some(u.id.clone()),
            layer: Some(u.layer),
            wave: Some(u.wave),
            ..Self::default()
        }
    }
}

#[derive(Clone)]
pub struct TelemetryHub {
    inner: Arc<TelemetryInner>,
}

struct TelemetryInner {
    jsonl_path: PathBuf,
    seq: AtomicU64,
    writer: Mutex<BufWriter<File>>,
    history: Mutex<VecDeque<TelemetryEvent>>,
    last_unit_sig: SieveCache<String, u64>,
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
        let (tx, _rx) = broadcast::channel(MAX_HISTORY);

        Ok(Self {
            inner: Arc::new(TelemetryInner {
                jsonl_path,
                seq: AtomicU64::new(1),
                writer: Mutex::new(BufWriter::with_capacity(32 * 1024, file)),
                history: Mutex::new(VecDeque::with_capacity(512)),
                last_unit_sig: SieveCache::new(4_096, 512 * 1024),
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

    pub fn emit_data(&self, kind: &str, data: Value) {
        self.emit(kind, EventSpec { data: Some(data), ..EventSpec::default() });
    }

    pub fn emit_unit(&self, kind: &str, unit: &ModuleUnit, data: Value) {
        self.emit(kind, EventSpec { data: Some(data), ..EventSpec::unit(unit) });
    }

    fn skip_dedup(&self, kind: &str, spec: &EventSpec) -> bool {
        if !matches!(kind, "llm_stream" | "wave_dispatch" | "heartbeat") {
            return false;
        }
        let Some(id) = spec.module_id.as_deref() else { return false };

        let extra = spec
            .data
            .as_ref()
            .and_then(|d| d.get("attempt"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let tpm = spec.uncached_tpm.unwrap_or(0);
        let sig = fast_sig(kind, tpm ^ extra.rotate_left(19));
        let weight = id.len().saturating_add(32);
        if self.inner.last_unit_sig.insert(id.to_string(), sig, weight) != Some(sig) {
            return false;
        }

        self.inner.dedup_skipped.fetch_add(1, Ordering::Relaxed);
        if let Some(st) = self.inner.shared_state.lock().as_ref() {
            st.count_dedup(1);
        }
        true
    }

    fn write_line(&self, kind: &str, ev: &TelemetryEvent) {
        let Ok(raw) = serde_json::to_string(ev) else { return };
        let line = redact_secrets(raw);
        let mut w = self.inner.writer.lock();

        if w.write_all(line.as_bytes()).is_err() || w.write_all(b"\n").is_err() {
            return;
        }
        self.inner
            .bytes_written
            .fetch_add((line.len() + 1) as u64, Ordering::Relaxed);

        let sync = matches!(
            kind,
            "run_start"
                | "dag_ready"
                | "mvcc_commit"
                | "run_end"
                | "circuit_state"
                | "unit_blocked"
        );
        if sync {
            let _ = w.flush();
        }
    }

    pub fn emit(&self, kind: &str, spec: EventSpec) {
        if self.skip_dedup(kind, &spec) {
            return;
        }

        let ev = TelemetryEvent {
            seq: self.inner.seq.fetch_add(1, Ordering::Relaxed),
            ts_ms: now_ms(),
            kind: kind.to_string(),
            module_id: spec.module_id.map(redact_secrets),
            layer: spec.layer,
            wave: spec.wave,
            uncached_tpm: spec.uncached_tpm,
            data: spec.data.map(redact_json),
            summary: spec.summary,
        };

        self.write_line(kind, &ev);

        let mut hist = self.inner.history.lock();
        if hist.len() >= MAX_HISTORY {
            hist.pop_front();
        }
        hist.push_back(ev.clone());
        drop(hist);

        let _ = self.inner.tx.send(ev);
    }

    pub fn flush_zstd(&self) -> Result<(PathBuf, u64, u64)> {
        self.inner.writer.lock().flush()?;
        let path = self.inner.jsonl_path.with_extension("jsonl.zst");
        let (raw_len, zst_len) =
            crate::platform::io::compress_file_zstd(&self.inner.jsonl_path, &path, 3)
                .with_context(|| format!("write compact zstd {}", path.display()))?;
        Ok((path, raw_len, zst_len))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.inner.tx.subscribe()
    }

    pub fn snapshot_events(&self) -> Vec<TelemetryEvent> {
        let hist = self.inner.history.lock();
        hist.iter().cloned().collect()
    }
}

fn redact_json(v: Value) -> Value {
    match v {
        Value::String(s) => Value::String(redact_secrets(s)),
        Value::Array(arr) => Value::Array(arr.into_iter().map(redact_json).collect()),
        Value::Object(map) => Value::Object(
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
    use crate::testutil::TempRepo;
    use serde_json::json;

    #[test]
    fn test_telemetry_dedup_zstd_and_resume() {
        let dir = TempRepo::new("tel_all");
        let log = dir.join("run.jsonl");
        let hub = TelemetryHub::new(log.clone()).unwrap();

        for _ in 0..8 {
            hub.emit("llm_stream", EventSpec {
                module_id: Some("mod_01".into()), layer: Some(0), wave: Some(0),
                uncached_tpm: Some(24000), ..EventSpec::default()
            });
        }
        assert_eq!(hub.dedup_skipped(), 7);

        let fake_csk = ["csk", "-", "dummy000000000000"].concat();
        let fake_keen = ["keen", "_", "dummy000000000000"].concat();
        hub.emit_data("run_start", json!({ "a": fake_csk, "b": fake_keen }));

        for i in 0..30 {
            let payload = json!({"head_rev": i + 1, "ruff_errors": 0, "ty_errors": 0});
            hub.emit("mvcc_commit", EventSpec {
                module_id: Some(format!("mod_{i:02}")), layer: Some(i / 10), wave: Some(i / 10),
                uncached_tpm: Some(32000), data: Some(payload), ..EventSpec::default()
            });
        }

        let (_zst, raw_len, zst_len) = hub.flush_zstd().unwrap();
        assert!(zst_len < raw_len / 2);
        drop(hub);

        let h2 = TelemetryHub::open_hub(log, true).unwrap();
        h2.emit("run_end", EventSpec { uncached_tpm: Some(0), ..EventSpec::default() });
        let (zst2, raw2, zst_len2) = h2.flush_zstd().unwrap();
        assert!(raw2 > raw_len && zst_len2 > 0 && zst2.exists());

        let disk = dir.read("run.jsonl");
        assert!(!disk.contains("csk-") && !disk.contains("keen_"));
        assert!(disk.contains("run_start") && disk.contains("run_end"));
    }

    #[test]
    fn test_redact_multi_secrets() {
        let line = format!(
            "first={} mid={} second={} tail={}",
            ["csk", "-", "aaa111222333"].concat(),
            ["keen", "_", "bbb444555666"].concat(),
            ["csk", "-", "ccc777888999"].concat(),
            ["keen", "_", "ddd000111222"].concat()
        );
        let clean = redact_secrets(line);
        assert!(!clean.contains("csk-") && !clean.contains("keen_"));
        assert_eq!((clean.matches("REDACTED_CSK_").count(), clean.matches("REDACTED_KEEN_").count()), (2, 2));
    }
}
