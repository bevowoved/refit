use crate::platform::{SieveCache, compress_file_zstd, now_ms};
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
    pub fn unit(unit: &ModuleUnit) -> Self {
        Self {
            module_id: Some(unit.id.clone()), layer: Some(unit.layer), wave: Some(unit.wave),
            ..Self::default()
        }
    }
}

#[derive(Clone)]
pub struct TelemetryHub { inner: Arc<TelemetryInner> }

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
    pub fn new(jsonl_path: PathBuf) -> Result<Self> { Self::open_hub(jsonl_path, false) }

    pub fn open_hub(jsonl_path: PathBuf, append: bool) -> Result<Self> {
        if let Some(parent) = jsonl_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create telemetry dir {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
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

    pub fn bind_state(&self, state: Arc<RunState>) { *self.inner.shared_state.lock() = Some(state); }
    pub fn dedup_skipped(&self) -> u64 { self.inner.dedup_skipped.load(Ordering::Relaxed) }
    pub fn bytes_written(&self) -> u64 { self.inner.bytes_written.load(Ordering::Relaxed) }
    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> { self.inner.tx.subscribe() }
    pub fn snapshot_events(&self) -> Vec<TelemetryEvent> {
        let hist = self.inner.history.lock();
        hist.iter().cloned().collect()
    }

    pub fn emit_data(&self, kind: &str, data: Value) {
        self.emit(kind, EventSpec { data: Some(data), ..EventSpec::default() });
    }

    pub fn emit_unit(&self, kind: &str, unit: &ModuleUnit, data: Value) {
        self.emit(kind, EventSpec { data: Some(data), ..EventSpec::unit(unit) });
    }

    fn skip_dedup(&self, kind: &str, spec: &EventSpec) -> bool {
        if !matches!(kind, "llm_stream" | "wave_dispatch" | "heartbeat") { return false; }
        let Some(id) = spec.module_id.as_deref() else { return false };
        let extra = spec.data.as_ref().and_then(|val| val.get("attempt")).and_then(Value::as_u64);
        let sig = fast_sig(kind, spec.uncached_tpm.unwrap_or(0) ^ extra.unwrap_or(0).rotate_left(19));
        if self.inner.last_unit_sig.insert(id.to_string(), sig, id.len().saturating_add(32)) != Some(sig) {
            return false;
        }
        self.inner.dedup_skipped.fetch_add(1, Ordering::Relaxed);
        if let Some(state) = self.inner.shared_state.lock().as_ref() { state.count_dedup(1); }
        true
    }

    fn write_line(&self, kind: &str, ev: &TelemetryEvent) {
        let Ok(raw) = serde_json::to_string(ev) else { return };
        let line = redact_secrets(raw);
        let mut writer = self.inner.writer.lock();
        if writer.write_all(line.as_bytes()).is_err() || writer.write_all(b"\n").is_err() { return; }
        self.inner.bytes_written.fetch_add((line.len() + 1) as u64, Ordering::Relaxed);
        if matches!(
            kind,
            "run_start" | "dag_ready" | "mvcc_commit" | "run_end" | "circuit_state" | "unit_blocked"
        ) {
            let _ = writer.flush();
        }
    }

    pub fn emit(&self, kind: &str, spec: EventSpec) {
        if self.skip_dedup(kind, &spec) { return; }
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
        if hist.len() >= MAX_HISTORY { hist.pop_front(); }
        hist.push_back(ev.clone());
        drop(hist);
        let _ = self.inner.tx.send(ev);
    }

    pub fn flush_zstd(&self) -> Result<(PathBuf, u64, u64)> {
        self.inner.writer.lock().flush()?;
        let path = self.inner.jsonl_path.with_extension("jsonl.zst");
        let (raw_len, zst_len) = compress_file_zstd(&self.inner.jsonl_path, &path, 3)
            .with_context(|| format!("write compact zstd {}", path.display()))?;
        Ok((path, raw_len, zst_len))
    }
}

fn redact_json(val: Value) -> Value {
    match val {
        Value::String(text) => Value::String(redact_secrets(text)),
        Value::Array(arr) => Value::Array(arr.into_iter().map(redact_json).collect()),
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(key, item)| (key, redact_json(item))).collect())
        }
        other => other,
    }
}

fn fast_sig(kind: &str, val: u64) -> u64 {
    kind.bytes().fold(0xcbf29ce484222325u64 ^ val, |acc, byte| {
        (acc ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use serde_json::json;

    #[test]
    fn telemetry_dedup_zstd_resume() {
        let dir = TempRepo::new("tel_all");
        let log = dir.join("run.jsonl");
        let hub = TelemetryHub::new(log.clone()).unwrap();
        let mk = |mid: &str, layer: usize, tpm: u64, data: Option<Value>| EventSpec {
            module_id: Some(mid.into()), layer: Some(layer), wave: Some(layer),
            uncached_tpm: Some(tpm), data, ..EventSpec::default()
        };

        for _ in 0..8 { hub.emit("llm_stream", mk("mod_01", 0, 24000, None)); }
        assert_eq!(hub.dedup_skipped(), 7);

        let fake_csk = ["csk", "-", "dummy000000000000"].concat();
        let fake_keen = ["keen", "_", "dummy000000000000"].concat();
        hub.emit_data("run_start", json!({ "a": fake_csk, "b": fake_keen }));

        for idx in 0..30 {
            let payload = json!({"head_rev": idx + 1, "ruff_errors": 0, "ty_errors": 0});
            hub.emit("mvcc_commit", mk(&format!("mod_{idx:02}"), idx / 10, 32000, Some(payload)));
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
    fn redact_multi_secrets() {
        let parts = [
            ("first", ["csk", "-", "aaa111222333"].concat()),
            ("mid", ["keen", "_", "bbb444555666"].concat()),
            ("second", ["csk", "-", "ccc777888999"].concat()),
            ("tail", ["keen", "_", "ddd000111222"].concat()),
        ];
        let clean = redact_secrets(parts.map(|(key, val)| format!("{key}={val}")).join(" "));
        assert!(!clean.contains("csk-") && !clean.contains("keen_"));
        assert_eq!(clean.matches("REDACTED_CSK_").count(), 2);
        assert_eq!(clean.matches("REDACTED_KEEN_").count(), 2);
    }
}
