mod model;
mod ops;
mod simd;
pub mod strict;
mod weights;

pub use ops::{EvalOptions, SparseAttnTelemetry};

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{
    MemGate, SieveCache, budget, collect_os_memory, ensure_cachedir_tag, under_memory_pressure,
    write_atomic,
};
use crate::source::{quantile, read_text};
use crate::types::{FileEgoContext, HygieneResult, Squiggle};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const CACHE_MAGIC: &[u8; 8] = b"RFHG0002";
const CACHE_ZSTD_WINDOW_LOG: u32 = 18;

fn estimate_neural_bytes(content: &str) -> usize {
    let lines = content.as_bytes().iter().filter(|&&b| b == b'\n').count() + 1;
    (lines.saturating_mul(2048)).clamp(64 * 1024, 16 * 1024 * 1024)
}

#[derive(Debug, Clone, Copy)]
pub struct StyleConfig {
    pub max_line_width: usize,
    pub indent_width: usize,
}

/// Read repository formatter configuration.
pub fn repo_style(repo_root: &Path) -> StyleConfig {
    let mut cfg = StyleConfig { max_line_width: 88, indent_width: 4 };
    if toml_style(&repo_root.join("pyproject.toml"), "line-length", "indent-width", &mut cfg)
        || toml_style(&repo_root.join("rustfmt.toml"), "max_width", "tab_spaces", &mut cfg)
        || json_style(repo_root, &mut cfg)
    {
        return cfg;
    }
    if repo_root.join("Cargo.toml").exists() {
        cfg = StyleConfig { max_line_width: 100, indent_width: 4 };
    } else if repo_root.join("package.json").exists() {
        cfg = StyleConfig { max_line_width: 100, indent_width: 2 };
    }
    cfg
}

fn find_toml_usize(val: &toml::Value, key: &str) -> Option<usize> {
    let tbl = val.as_table()?;
    tbl.get(key)
        .and_then(toml::Value::as_integer)
        .and_then(|v| usize::try_from(v).ok())
        .or_else(|| tbl.values().find_map(|c| find_toml_usize(c, key)))
}

fn toml_style(path: &Path, w_key: &str, i_key: &str, cfg: &mut StyleConfig) -> bool {
    let Ok(raw) = fs::read_to_string(path) else { return false };
    let Ok(parsed) = toml::from_str::<toml::Value>(&raw) else { return false };
    let (w, ind) = (find_toml_usize(&parsed, w_key), find_toml_usize(&parsed, i_key));
    if let Some(n) = w { cfg.max_line_width = n.clamp(60, 160); }
    if let Some(n) = ind { cfg.indent_width = n.clamp(2, 8); }
    w.is_some() || ind.is_some()
}

fn json_num(v: &serde_json::Value, top: &str, nested: &str) -> Option<usize> {
    v[top].as_u64().or_else(|| v["formatter"][nested].as_u64()).map(|n| n as usize)
}

fn json_style(repo_root: &Path, cfg: &mut StyleConfig) -> bool {
    for f in [".prettierrc", ".prettierrc.json", "biome.json", "biome.jsonc"] {
        let Ok(raw) = fs::read_to_string(repo_root.join(f)) else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&strip_jsonc(&raw)) else { continue };
        cfg.max_line_width = json_num(&v, "printWidth", "lineWidth").unwrap_or(100).clamp(60, 160);
        cfg.indent_width = json_num(&v, "tabWidth", "indentWidth").unwrap_or(2).clamp(2, 8);
        return true;
    }
    false
}

fn write_str_u16(out: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    let _ = out.write_u16::<LittleEndian>(len as u16);
    out.extend_from_slice(&bytes[..len]);
}

fn read_str_u16(rd: &mut Cursor<&[u8]>) -> Option<String> {
    let len = rd.read_u16::<LittleEndian>().ok()? as usize;
    let mut buf = vec![0u8; len];
    rd.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn write_opt_f64(out: &mut Vec<u8>, v: Option<f64>) {
    if let Some(x) = v { let _ = out.write_f64::<LittleEndian>(x); }
}

fn read_opt_f64(rd: &mut Cursor<&[u8]>, flags: u8, bit: u8) -> Option<Option<f64>> {
    if (flags & bit) != 0 {
        Some(Some(rd.read_f64::<LittleEndian>().ok()?))
    } else {
        Some(None)
    }
}

/// Safe explicit little-endian binary codec for `HygieneResult` caches (`RFHG0002`, zero UB).
fn encode_compact_cache(map: &HashMap<String, HygieneResult>) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 96);
    out.extend_from_slice(CACHE_MAGIC);
    let _ = out.write_u32::<LittleEndian>(map.len() as u32);
    for (key, r) in map {
        write_str_u16(&mut out, key);
        let _ = out.write_f64::<LittleEndian>(r.score);
        let _ = out.write_u64::<LittleEndian>(r.latency_us);
        let flags = u8::from(r.strict_score.is_some())
            | (u8::from(r.veto_cap.is_some()) << 1)
            | (u8::from(r.total_squiggles.is_some()) << 2)
            | (u8::from(r.event_load.is_some()) << 3)
            | (u8::from(r.multiscale_score.is_some()) << 4)
            | (u8::from(r.calibration_log.is_some()) << 5);
        out.push(flags);
        write_opt_f64(&mut out, r.strict_score);
        write_opt_f64(&mut out, r.veto_cap);
        if let Some(v) = r.total_squiggles {
            let _ = out.write_u32::<LittleEndian>(v.min(u32::MAX as usize) as u32);
        }
        write_opt_f64(&mut out, r.event_load);
        write_opt_f64(&mut out, r.multiscale_score);
        if let Some(ref s) = r.calibration_log { write_str_u16(&mut out, s); }

        let sq_cnt = r.squiggles.len().min(u16::MAX as usize);
        let _ = out.write_u16::<LittleEndian>(sq_cnt as u16);
        for sq in r.squiggles.iter().take(sq_cnt) {
            let _ = out.write_u32::<LittleEndian>(sq.line.min(u32::MAX as usize) as u32);
            let spec_idx = model::SQUIGGLE_SPEC
                .iter()
                .position(|(k, m, _)| *k == sq.kind && *m == sq.message);
            if let Some(idx) = spec_idx {
                out.push(idx as u8);
            } else {
                out.push(255);
                write_str_u16(&mut out, &sq.kind);
                write_str_u16(&mut out, &sq.message);
            }
            let _ = out.write_f64::<LittleEndian>(sq.probability);
        }
        let veto_cnt = r.triggered_vetoes.len().min(u16::MAX as usize);
        let _ = out.write_u16::<LittleEndian>(veto_cnt as u16);
        for veto in r.triggered_vetoes.iter().take(veto_cnt) { write_str_u16(&mut out, veto); }
    }
    out
}

fn decode_compact_cache(bytes: &[u8]) -> Option<HashMap<String, HygieneResult>> {
    if !bytes.starts_with(CACHE_MAGIC) { return None; }
    let mut rd = Cursor::new(&bytes[8..]);
    let count = rd.read_u32::<LittleEndian>().ok()? as usize;
    if count > 1_000_000 { return None; }
    let mut map = HashMap::with_capacity(count.min(4096));
    for _ in 0..count {
        let key = read_str_u16(&mut rd)?;
        let score = rd.read_f64::<LittleEndian>().ok()?;
        let latency_us = rd.read_u64::<LittleEndian>().ok()?;
        let flags = rd.read_u8().ok()?;
        let strict_score = read_opt_f64(&mut rd, flags, 1)?;
        let veto_cap = read_opt_f64(&mut rd, flags, 2)?;
        let total_squiggles = if (flags & 4) != 0 {
            Some(rd.read_u32::<LittleEndian>().ok()? as usize)
        } else { None };
        let event_load = read_opt_f64(&mut rd, flags, 8)?;
        let multiscale_score = read_opt_f64(&mut rd, flags, 16)?;
        let calibration_log = if (flags & 32) != 0 { Some(read_str_u16(&mut rd)?) } else { None };

        let sq_cnt = rd.read_u16::<LittleEndian>().ok()? as usize;
        let mut squiggles = Vec::with_capacity(sq_cnt.min(64));
        for _ in 0..sq_cnt {
            let line = rd.read_u32::<LittleEndian>().ok()? as usize;
            let kind_id = rd.read_u8().ok()?;
            let (kind, message) = match model::SQUIGGLE_SPEC.get(kind_id as usize) {
                Some(&(sk, sm, _)) => (sk.to_string(), sm.to_string()),
                None if kind_id == 255 => (read_str_u16(&mut rd)?, read_str_u16(&mut rd)?),
                None => return None,
            };
            let probability = rd.read_f64::<LittleEndian>().ok()?;
            squiggles.push(Squiggle { line, kind, probability, message });
        }
        let veto_cnt = rd.read_u16::<LittleEndian>().ok()? as usize;
        let mut triggered_vetoes = Vec::with_capacity(veto_cnt.min(32));
        for _ in 0..veto_cnt { triggered_vetoes.push(read_str_u16(&mut rd)?); }
        map.insert(key, HygieneResult {
            score, squiggles, cached: false, latency_us, strict_score, veto_cap,
            total_squiggles, event_load, triggered_vetoes, multiscale_score, calibration_log,
        });
    }
    Some(map)
}

fn write_compact_cache_zstd(path: &Path, map: &HashMap<String, HygieneResult>) -> anyhow::Result<()> {
    let raw = encode_compact_cache(map);
    let mut enc = zstd::stream::Encoder::new(Vec::new(), 3)?;
    let _ = enc.window_log(CACHE_ZSTD_WINDOW_LOG);
    enc.write_all(&raw)?;
    let compressed = enc.finish()?;
    write_atomic(path, &compressed)
}

fn read_hygiene_cache(path: &Path, active_prefix: &str) -> Option<HashMap<String, HygieneResult>> {
    let decoded = zstd::stream::decode_all(fs::read(path).ok()?.as_slice()).ok()?;
    if decoded.starts_with(CACHE_MAGIC) { return decode_compact_cache(&decoded); }
    let legacy: HashMap<String, HygieneResult> = serde_json::from_slice(&decoded).ok()?;
    let def_pfx = weights::default_weights_hash_prefix();
    Some(legacy.into_iter().map(|(k, v)| {
        let nk = if active_prefix == def_pfx
            && let Some(rest) = k.strip_prefix("v07:")
            && !rest.contains(':')
        {
            format!("v07:{def_pfx}:{rest}")
        } else { k };
        (nk, v)
    }).collect())
}

fn hygiene_entry_weight(key: &str, res: &HygieneResult) -> usize {
    let veto_b: usize = res.triggered_vetoes.iter().map(String::len).sum();
    let cal_b = res.calibration_log.as_ref().map_or(0, String::len);
    key.len()
        .saturating_add(160)
        .saturating_add(res.squiggles.len().saturating_mul(96))
        .saturating_add(veto_b)
        .saturating_add(cal_b)
}

fn env_approx_long() -> bool {
    std::env::var("REFIT_APPROX_LONG")
        .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

pub struct HygieneScorer {
    cache: SieveCache<String, HygieneResult>,
    in_flight: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    disk_path: Option<PathBuf>,
    override_weights: Option<Arc<weights::ResolvedWeights>>,
    default_opts: EvalOptions,
    dirty: AtomicBool,
    last_warm_us: AtomicU64,
}

impl HygieneScorer {
    #[cfg(test)]
    pub fn new() -> Self { Self::with_path(None, None, EvalOptions::default()) }

    fn with_path(
        disk_path: Option<PathBuf>,
        override_weights: Option<Arc<weights::ResolvedWeights>>,
        default_opts: EvalOptions,
    ) -> Self {
        Self {
            cache: SieveCache::new(16_384, 32 * 1024 * 1024),
            in_flight: Mutex::new(HashMap::new()),
            disk_path, override_weights, default_opts,
            dirty: AtomicBool::new(false),
            last_warm_us: AtomicU64::new(0),
        }
    }

    pub fn set_approx_long(&mut self, enabled: bool) {
        if enabled { self.default_opts.approx_long = true; }
    }

    fn active_weights(&self) -> &weights::ResolvedWeights {
        match self.override_weights.as_deref() {
            Some(w) => w,
            None => weights::v07_weights(),
        }
    }

    /// Expose the active weights' 16-hex BLAKE3 prefix without inflating embedded `WEIGHTS_NPZ`.
    pub fn weights_blake3_prefix(&self) -> &str {
        match self.override_weights.as_deref() {
            Some(w) => w.hash_prefix.as_str(),
            None => weights::default_weights_hash_prefix(),
        }
    }

    pub fn open_persistent(repo_root: &Path) -> Self {
        let ov = weights::load_verified_override(repo_root).unwrap_or_else(|err| {
            tracing::warn!("rejecting unverified local hygiene weights override: {err}");
            None
        });
        Self::open_with_weights(repo_root, ov)
    }

    /// Open a persistent scorer while propagating errors from any invalid `.refit/weights/` override.
    #[cfg(test)]
    pub fn open_persistent_checked(repo_root: &Path) -> anyhow::Result<Self> {
        Ok(Self::open_with_weights(repo_root, weights::load_verified_override(repo_root)?))
    }

    fn open_with_weights(
        repo_root: &Path,
        override_weights: Option<Arc<weights::ResolvedWeights>>,
    ) -> Self {
        let disk_path = repo_root.join(".refit/cache/hygiene_v07_cache.zst");
        let opts = EvalOptions { approx_long: env_approx_long(), ..EvalOptions::default() };
        let scorer = Self::with_path(Some(disk_path.clone()), override_weights, opts);
        let prefix = scorer.weights_blake3_prefix().to_string();
        if let Some(saved) = read_hygiene_cache(&disk_path, &prefix) {
            for (k, v) in saved {
                let w = hygiene_entry_weight(&k, &v);
                scorer.cache.insert(k, v, w);
            }
        }
        scorer
    }

    pub fn warm_us(&self) -> u64 { self.last_warm_us.load(Ordering::Relaxed) }

    pub fn flush_cache(&self) {
        if !self.dirty.swap(false, Ordering::AcqRel) { return; }
        let Some(ref path) = self.disk_path else { return };
        if let Some(parent) = path.parent() { ensure_cachedir_tag(parent); }
        let map: HashMap<String, HygieneResult> = self.cache.snapshot().into_iter().collect();
        let _ = write_compact_cache_zstd(path, &map);
    }

    fn cache_hit(&self, key: &String, t0: Instant) -> Option<HygieneResult> {
        let mut res = self.cache.get(key)?;
        let us = t0.elapsed().as_micros().max(1) as u64;
        res.cached = true;
        res.latency_us = us;
        self.last_warm_us.store(us, Ordering::Relaxed);
        Some(res)
    }

    fn release_in_flight(&self, key: &str, lock: &Arc<Mutex<()>>) {
        let mut guard = self.in_flight.lock();
        if matches!(guard.get(key), Some(existing) if Arc::ptr_eq(existing, lock)) {
            guard.remove(key);
        }
    }

    pub fn score_content(&self, path: &str, content: &str) -> HygieneResult {
        self.score_content_with_opts(path, content, self.default_opts).0
    }

    pub fn score_content_with_opts(
        &self,
        _path: &str,
        content: &str,
        opts: EvalOptions,
    ) -> (HygieneResult, Option<SparseAttnTelemetry>) {
        let line_count = content.as_bytes().iter().filter(|&&b| b == b'\n').count() + 1;
        let approx_active = opts.approx_long && line_count > opts.approx_threshold;
        let namespace = if approx_active { "v07-approx" } else { "v07" };
        let key = format!(
            "{namespace}:{}:{}",
            self.weights_blake3_prefix(),
            MvccStore::hash_bytes(content.as_bytes())
        );
        let t0 = Instant::now();
        if let Some(res) = self.cache_hit(&key, t0) {
            return (res, None);
        }

        let lock = {
            let mut inflight = self.in_flight.lock();
            inflight
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let guard = lock.lock();
        if let Some(res) = self.cache_hit(&key, t0) {
            drop(guard);
            self.release_in_flight(&key, &lock);
            return (res, None);
        }

        let est = estimate_neural_bytes(content);
        let gate = MemGate::global();
        let req = est.min(gate.capacity());
        let permit = gate.reserve_or_wait(req, Duration::from_secs(5)).ok();
        let started = Instant::now();
        let (mut res, telemetry) = model::score_v07_with_opts(content, self.active_weights(), opts);
        res.latency_us = started.elapsed().as_micros().max(1) as u64;
        let weight = hygiene_entry_weight(&key, &res);
        self.cache.insert(key.clone(), res.clone(), weight);
        self.dirty.store(true, Ordering::Release);

        drop(permit);
        drop(guard);
        self.release_in_flight(&key, &lock);
        (res, telemetry)
    }

    /// Score content with exact `v0.7.0` `score` plus additive `CH-V4..V7` strict/calibration track.
    #[cfg(test)]
    pub fn score_content_strict(&self, path: &str, content: &str) -> HygieneResult {
        self.score_content_strict_with_ego(path, content, None)
    }

    /// Score content with exact `v0.7.0` `score` plus live `FileEgoContext` & truthful `CalibrationAudit`.
    pub fn score_content_strict_with_ego(
        &self,
        path: &str,
        content: &str,
        ego: Option<&FileEgoContext>,
    ) -> HygieneResult {
        let (base, telemetry) = self.score_content_with_opts(path, content, self.default_opts);
        let zero_ratio = model::weight_zero_ratio(self.active_weights());
        strict::score_strict_calibrated(path, content, &base, ego, telemetry.as_ref(), zero_ratio)
    }

    pub fn score_files(
        &self,
        repo_root: &Path,
        rel_paths: &[String],
        preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    ) -> (f64, f64, HashMap<String, HygieneResult>) {
        let was_dirty = self.dirty.load(Ordering::Acquire);
        let b = budget();
        let results: Vec<(String, HygieneResult)> = if b.is_constrained() {
            rel_paths
                .iter()
                .map(|rel| {
                    let text = read_text(repo_root, rel, preloaded).unwrap_or_default();
                    (rel.clone(), self.score_content(rel, &text))
                })
                .collect()
        } else if b.rayon_threads <= 2 && rel_paths.len() > 8 {
            let next = AtomicUsize::new(0);
            let out = Mutex::new(Vec::with_capacity(rel_paths.len()));
            std::thread::scope(|s| {
                let run = || {
                    while let Some(rel) = rel_paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let text = read_text(repo_root, rel, preloaded).unwrap_or_default();
                        out.lock().push((rel.clone(), self.score_content(rel, &text)));
                    }
                };
                s.spawn(run);
                run();
            });
            out.into_inner()
        } else {
            rel_paths
                .par_iter()
                .map(|rel| {
                    let text = read_text(repo_root, rel, preloaded).unwrap_or_default();
                    (rel.clone(), self.score_content(rel, &text))
                })
                .collect()
        };

        let mut per_file = HashMap::with_capacity(results.len());
        let mut scores = Vec::with_capacity(results.len());
        for (rel, res) in results {
            scores.push(res.score);
            per_file.insert(rel, res);
        }

        if scores.is_empty() {
            return (1.0, 1.0, per_file);
        }
        let median = quantile(&mut scores, 0.5);
        let p10 = quantile(&mut scores, 0.10);
        let scored_cold = self.dirty.load(Ordering::Acquire) && !was_dirty;
        self.flush_cache();
        if scored_cold || under_memory_pressure() {
            collect_os_memory();
        }
        (median, p10, per_file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::write_zstd;
    use crate::testutil::{TempRepo, build_single_entry_zip, map_conds, nest_src, repeat_fn, walk_rs_files};
    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    use std::collections::{BTreeSet, HashSet};
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    fn sq_tuples(list: &[Squiggle]) -> Vec<(usize, String, i64)> {
        list.iter().map(|s| (s.line, s.kind.clone(), (s.probability * 10000.0).round() as i64)).collect()
    }

    fn js_sq_tuples(list: &[serde_json::Value]) -> Vec<(usize, String, i64)> {
        list.iter().map(|s| (
            s["line"].as_u64().unwrap() as usize,
            s["kind"].as_str().unwrap().to_string(),
            (s["probability"].as_f64().unwrap() * 10000.0).round() as i64,
        )).collect()
    }

    #[test]
    fn test_hygiene_scoring_and_cache() {
        let tmp = TempRepo::new("hyg_disk");
        let s1 = HygieneScorer::open_persistent(tmp.path());

        for (rel, src) in [("e.py", ""), ("o.rs", "pub const X: usize = 1;\n")] {
            let sc = s1.score_content(rel, src).score;
            assert!((0.80..=0.99).contains(&sc), "{rel}={sc}");
        }

        let clean = "def add(a: int, b: int) -> int:\n    return a + b\n";
        let conds = map_conds(&["0", "1", "2", "3", "4"], |i| format!("if x > {i}:"));
        let tail = "return 'a' * 120 + 'extra_padding_to_exceed_one_hundred_cols_now'";
        let messy = nest_src("def bad(x):", "    ", &conds, tail, None);

        let cold = s1.score_content("clean.py", clean);
        assert!(!cold.cached);
        let warm = s1.score_content("clean.py", clean);
        assert!(warm.cached && warm.latency_us <= 1000);
        let bad = s1.score_content("messy.py", &messy);
        assert!(cold.score > bad.score + 0.05 && !bad.squiggles.is_empty());

        let rs_clean = concat!(
            "pub struct S;\nimpl S {\n    pub fn run(&self, x: Option<i32>) -> i32 {\n",
            "        match x { Some(v) => v + 1, None => 0 }\n    }\n}\n",
        );
        let r1 = s1.score_content("s.rs", rs_clean);
        assert!(!r1.cached && r1.score > 0.70);
        s1.flush_cache();

        let s2 = HygieneScorer::open_persistent(tmp.path());
        let r2 = s2.score_content("s.rs", rs_clean);
        assert!(r2.cached && (r1.score - r2.score).abs() < 1e-9 && s2.warm_us() >= 1);

        // Verify legacy JSON-zstd cache backward compatibility
        let legacy_tmp = TempRepo::new("hyg_legacy_json");
        let legacy_path = legacy_tmp.path().join(".refit/cache/hygiene_v07_cache.zst");
        let legacy_key = format!("v07:{}", MvccStore::hash_bytes(rs_clean.as_bytes()));
        let mut legacy_map = HashMap::new();
        legacy_map.insert(legacy_key, r1.clone());
        write_zstd(&legacy_path, &legacy_map).unwrap();
        let s_leg = HygieneScorer::open_persistent(legacy_tmp.path());
        let r_leg = s_leg.score_content("s.rs", rs_clean);
        assert!(r_leg.cached && (r_leg.score - r1.score).abs() < 1e-9);

        let ts_head = "export function messy(a: number, b: number, c: number, d: number, e: number, f: number) {";
        let ts_c = map_conds(&["a", "b", "c", "d", "e", "f"], |v| format!("if ({v} > 0) {{"));
        let ts_body = "return a ? (b ? (c ? d + e + f : a + b) : c) : (d ? e : f) + 1234567;";
        let r_ts = s2.score_content("messy.ts", &nest_src(ts_head, "  ", &ts_c, ts_body, Some("}")));
        assert!(r_ts.score < 0.85 && !r_ts.squiggles.is_empty(), "got {}", r_ts.score);
    }

    #[test]
    fn test_hygiene_all_six_smells() {
        let scorer = HygieneScorer::new();
        let conds = map_conds(&["a", "b", "c", "d", "e"], |v| format!("if {v}:"));
        let nested = nest_src("def f(a, b, c, d, e):", "    ", &conds, "return a+b+c+d+e", None);
        let dense = concat!(
            "def d(x):\n    return (x + 1 if x > 0 else (x - 1 if x < -5 else ",
            "(x * 2 if x == 0 else x * 3))) + len(str(x).strip().lower().replace('a', 'b'))\n",
        );

        let steps = [
            "one = normalize_payload(raw_input)", "two = validate_schema(step_one)",
            "three = enrich_metadata(step_two)", "four = compute_digest(step_three)",
            "five = serialize_record(step_four)", "six = write_storage(step_five)",
            "seven = emit_audit_log(step_six)",
        ];
        let mut dup = String::new();
        for n in 1..=2 {
            dup.push_str(&format!("def run_pipeline_{n}(raw_input):\n"));
            for s in steps {
                dup.push_str(&format!("    step_{s}\n"));
            }
            dup.push_str("    return step_seven\n\n");
        }

        let rep = repeat_fn("def pick(k):", 1..=8, |i| format!("    if k == {i}:\n        return {i} * 10\n"), "");
        let cramped = repeat_fn("def cramped_fn(x):", 1..=45, |i| format!("    v_{i} = x + {i}\n"), "    return v_45\n");
        let echo = repeat_fn("def calc(x, y, z):", 0..=14, |i| format!("    # step_{i} = compute_legacy_{i}(x, y, z)\n"), "    return x\n");

        let mut kinds: HashSet<String> = HashSet::new();
        for src in [nested.as_str(), dense, &dup, &rep, &cramped, &echo] {
            for q in scorer.score_content("t.py", src).squiggles {
                kinds.insert(q.kind);
            }
        }
        for k in ["deep_nesting", "dense_line", "duplicate_block", "repetitive_structure", "cramped_layout", "comment_echo"] {
            assert!(kinds.contains(k), "missing {k} in {kinds:?}");
        }
    }

    #[test]
    fn v07_canonical_zero_delta_parity() {
        let scorer = HygieneScorer::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));

        // 1. Pinned canonical v0.7.0 baselines
        assert_eq!(scorer.score_content("empty.rs", "").score, 0.8808);
        assert_eq!(scorer.score_content("one.rs", "pub const X: usize = 1;\n").score, 0.8808);

        // 2. Verify pure-AVX2 non-FMA Estrin exp accuracy & underflow mask on x86_64 AVX2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let mut max_ulp = 0u64;
            let mut out = [0.0f64; 4];
            let underflow = [-708.5, -709.0, -1000.0, -1e9];
            // SAFETY: AVX2 support verified above.
            unsafe {
                let v = std::arch::x86_64::_mm256_loadu_pd(underflow.as_ptr());
                let r = simd::exp_f64x4_avx2_non_fma(v);
                std::arch::x86_64::_mm256_storeu_pd(out.as_mut_ptr(), r);
            }
            assert_eq!(out, [0.0, 0.0, 0.0, 0.0], "underflow mask failed: {out:?}");

            for step in 0..2500 {
                let mut xs = [0.0f64; 4];
                for (j, slot) in xs.iter_mut().enumerate() {
                    let t = ((step * 4 + j) as f64) / 9999.0;
                    *slot = -708.0 + t * 1417.0;
                }
                // SAFETY: AVX2 support verified above.
                unsafe {
                    let v = std::arch::x86_64::_mm256_loadu_pd(xs.as_ptr());
                    let r = simd::exp_f64x4_avx2_non_fma(v);
                    std::arch::x86_64::_mm256_storeu_pd(out.as_mut_ptr(), r);
                }
                for j in 0..4 {
                    let ref_v = xs[j].exp();
                    let ulp = out[j].to_bits().abs_diff(ref_v.to_bits());
                    max_ulp = max_ulp.max(ulp);
                }
            }
            assert!(max_ulp <= 2, "Estrin exp_f64x4 ULP {max_ulp} > 2");
        }

        // 3. Collect all src/**/*.rs files + edge-case fixtures (CRLF, tabs, Unicode, single-line)
        let mut cases: Vec<(String, String)> = Vec::new();
        let mut rs_files = Vec::new();
        walk_rs_files(&root.join("src"), &root.join("src"), &mut rs_files);
        rs_files.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(rs_files.len() >= 47, "expected >= 47 src/**/*.rs files, got {}", rs_files.len());
        for (rel, abs) in rs_files {
            cases.push((format!("src/{rel}"), fs::read_to_string(abs).unwrap()));
        }
        cases.push(("edge_crlf.rs".into(), "pub fn a() -> i32 {\r\n    1\r\n}\r\n".into()));
        cases.push(("edge_tabs.go".into(), "package p\nfunc F(x int) int {\n\tif x > 0 {\n\t\treturn x\n\t}\n\treturn 0\n}\n".into()));
        cases.push(("edge_utf8.rs".into(), "pub fn greet() -> &'static str {\n    \"héllo 🚀 日本語\"\n}\n".into()));
        cases.push(("edge_nul_vtab.txt".into(), "a\x00b\x0bc\u{2028}d\r\ne".into()));

        // 4. Spawn live JS v0.7.0 worker and assert 0.000000000 max_abs_delta across every single file
        let worker = root.join("vendor/codehygiene/worker.mjs");
        let npz = root.join("vendor/codehygiene/weights.npz");
        let mut child = Command::new("node")
            .arg(&worker)
            .arg(&npz)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn node vendor/codehygiene/worker.mjs");

        let mut stdin = child.stdin.take().unwrap();
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let mut max_abs_delta = 0.0f64;

        for (idx, (label, content)) in cases.iter().enumerate() {
            let req = serde_json::json!({ "id": idx + 1, "content": content })
                .to_string()
                .replace('\u{2028}', "\\u2028")
                .replace('\u{2029}', "\\u2029");
            writeln!(stdin, "{req}").unwrap();
            stdin.flush().unwrap();

            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let js: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert!(js.get("error").is_none(), "JS error on {label}: {js}");

            let js_score = js["score"].as_f64().unwrap();
            let js_sq = js["squiggles"].as_array().unwrap();
            let rust_res = scorer.score_content(label, content);
            let rust_sq_vec = sq_tuples(&rust_res.squiggles);
            if idx < 8 || label.starts_with("edge_") || label.contains("hygiene") {
                let scalar_res = model::score_v07_scalar_only(content, weights::v07_weights());
                assert_eq!(rust_res.score, scalar_res.score, "AVX2 vs scalar score on {label}");
                assert_eq!(
                    rust_sq_vec,
                    sq_tuples(&scalar_res.squiggles),
                    "AVX2 vs scalar squiggles on {label}"
                );
            }

            let delta = (rust_res.score - js_score).abs();
            max_abs_delta = max_abs_delta.max(delta);
            assert_eq!(
                rust_res.score, js_score,
                "score mismatch on {label}: rust={} js={} delta={delta}",
                rust_res.score, js_score
            );
            assert_eq!(
                rust_res.total_squiggles,
                Some(js_sq.len()),
                "total_squiggles mismatch on {label}"
            );
            if js_sq.len() <= 48 {
                assert_eq!(rust_res.squiggles.len(), js_sq.len(), "sq len mismatch on {label}");
                let mut r_sorted = rust_sq_vec;
                let mut j_sorted = js_sq_tuples(js_sq);
                r_sorted.sort();
                j_sorted.sort();
                for (r_item, j_item) in r_sorted.iter().zip(j_sorted.iter()) {
                    assert_eq!((r_item.0, &r_item.1), (j_item.0, &j_item.1), "sq kind on {label}");
                    assert!(
                        (r_item.2 - j_item.2).abs() <= 1,
                        "sq prob mismatch on {label}: rust={r_item:?} js={j_item:?}"
                    );
                }
            }
        }
        drop(stdin);
        let status = child.wait().unwrap();
        assert!(status.success());
        assert_eq!(max_abs_delta, 0.0, "expected 0.000000000 max_abs_delta, got {max_abs_delta}");
    }

    #[test]
    fn anti_gaming_25_vectors_and_zero_false_vetoes() {
        let scorer = HygieneScorer::new();

        // 1. All 25 adversarial gaming vectors MUST be vetoed (<= 0.50; BiDi/homoglyph <= 0.15)
        let adv_cases: [(&str, &str, &str); 25] = [
            ("adv_01.rs", "bidi_trojan_source", "pub fn check(admin: bool) -> bool {\n    /* \u{202E} } \u{2066} if admin \u{2069} \u{2066} */\n    false\n}\n"),
            ("adv_02.rs", "bidi_trojan_source", "pub fn hidden() -> &'static str {\n    \"ok\u{200B}token\"\n}\n"),
            ("adv_03.rs", "uts39_homoglyph", "pub fn auth() -> bool {\n    let \u{0430}dmin = false;\n    let us\u{0435}r = \u{0430}dmin;\n    us\u{0435}r\n}\n"),
            ("adv_04.rs", "uts39_homoglyph", "pub fn check() -> i32 {\n    let t\u{03BF}ken = 1\u{037E}\n    t\u{03BF}ken\n}\n"),
            ("adv_05.rs", "lint_suppression", concat!("#", "[allow(dead_code, clippy::too_many_arguments)]\npub fn suppressed(val: i32) -> i32 {\n    val + 1\n}\n")),
            ("adv_06.rs", "lint_suppression", concat!("#!", "[allow(warnings)]\npub fn noisy(val: i32) -> i32 {\n    val + 1\n}\n")),
            ("adv_07.rs", "lint_suppression", concat!("#", "[cfg_attr(all(), allow(clippy::all))]\npub fn sneaky(val: i32) -> i32 {\n    val + 1\n}\n")),
            ("adv_08.rs", "serde_flatten", "pub struct Payload {\n    pub name: String,\n    #[serde(default, flatten)]\n    pub extra: std::collections::BTreeMap<String, String>,\n}\n"),
            ("adv_09.rs", "identifier_golf", "pub fn run(p: i32) -> i32 {\n    let a = p + 1;\n    let b = a * 2;\n    let c = b - 3;\n    let d = c + 4;\n    let f = d * 5;\n    let g = f - 6;\n    g\n}\n"),
            ("adv_10.rs", "identifier_golf", "pub fn f(value: i32) -> i32 {\n    value + 1\n}\n"),
            ("adv_11.rs", "blank_line_stuffing", "pub fn padded(value: i32) -> i32 {\n    let step_one = value + 1;\n\n\n\n\n    let step_two = step_one * 2;\n    let step_three = step_two + 3;\n    let step_four = step_three - 1;\n    let step_five = step_four + 5;\n    step_five\n}\n"),
            ("adv_12.rs", "blank_line_stuffing", "pub fn sparse(input: i32) -> i32 {\n\n    let alpha = input + 1;\n\n    let beta = alpha + 2;\n\n    let gamma = beta + 3;\n\n    let delta = gamma + 4;\n\n    let eps = delta + 5;\n\n    let zeta = eps + 6;\n\n    let eta = zeta + 7;\n\n    eta\n}\n"),
            ("adv_13.rs", "comment_echo_stuffing", "pub fn echoed(base_score: i32) -> i32 {\n    // compute total_score from base_score\n    let total_score = base_score + 10;\n    // apply bonus_weight to total_score\n    let bonus_weight = total_score * 2;\n    // clamp final_metric using bonus_weight\n    let final_metric = bonus_weight.min(100);\n    final_metric\n}\n"),
            ("adv_14.rs", "comment_echo_stuffing", "// Step 1: initialize variable\n// Step 2: compute addition\n// Step 3: validate input\n// Step 4: check result\n// Step 5: prepare output\n// Step 6: finalize return\n// Step 7: complete function\n// Step 8: end marker\npub fn add(left_val: i32, right_val: i32) -> i32 {\n    left_val + right_val\n}\n"),
            ("adv_15.rs", "prod_placeholder_macro", "pub fn incomplete(value: i32) -> i32 {\n    if value > 0 {\n        todo!(\"implement later\")\n    } else {\n        0\n    }\n}\n"),
            ("adv_16.rs", "prod_placeholder_macro", "pub fn stub(value: i32) -> i32 {\n    let _ = value;\n    unimplemented!()\n}\n"),
            ("adv_17.rs", "hidden_dense_tail", "pub fn sneak(alpha: i32, beta: i32) -> i32 {\n    let base = alpha + 1;                                                                                                                 let hidden = if beta > 0 { base * 2 + alpha } else { base - 1 };\n    hidden\n}\n"),
            ("adv_18.rs", "hidden_dense_tail", "pub fn packed(val: i32) -> i32 {\n    let out = val + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20 + 21 + 22 + 23 + 24 + 25 + 26 + 27 + 28 + 29 + 30 + 31;\n    out\n}\n"),
            ("adv_19.rs", "deep_nesting_spaghetti", "pub fn pyramid(v1: i32, v2: i32, v3: i32, v4: i32, v5: i32, v6: i32) -> i32 {\n    if v1 > 0 {\n        if v2 > 0 {\n            if v3 > 0 {\n                if v4 > 0 {\n                    if v5 > 0 {\n                        if v6 > 0 {\n                            return v1 + v6;\n                        }\n                    }\n                }\n            }\n        }\n    }\n    0\n}\n"),
            ("adv_20.rs", "deep_nesting_spaghetti", "pub fn spoofed(v1: i32, v2: i32, v3: i32, v4: i32, v5: i32, v6: i32) -> i32 {\nif v1 > 0 {\n if v2 > 0 {\n  if v3 > 0 {\n   if v4 > 0 {\n    if v5 > 0 {\n     if v6 > 0 {\n      return v1 + v6;\n     }\n    }\n   }\n  }\n }\n}\n0\n}\n"),
            ("adv_21.go", "deep_nesting_spaghetti", "package sloppy\nfunc Deep(a, b, c, d, e, f bool) int {\n\tif a {\n\t\tif b {\n\t\t\tif c {\n\t\t\t\tif d {\n\t\t\t\t\tif e {\n\t\t\t\t\t\tif f {\n\t\t\t\t\t\t\treturn 1\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n\treturn 0\n}\n"),
            ("adv_22.rs", "unsafe_undocumented", "pub fn cast_bits(bits: u64) -> f64 {\n    unsafe { std::mem::transmute::<u64, f64>(bits) }\n}\n"),
            ("adv_23.rs", "std_lock_unwrap", "pub fn bump(lock: &std::sync::Mutex<u64>) -> u64 {\n    let mut guard = lock.lock().unwrap();\n    *guard += 1;\n    *guard\n}\n"),
            ("adv_24.rs", "deep_nesting_spaghetti", "pub fn crammed(a: bool, b: bool, c: bool) -> i32 {\n    if a { if b { if c { return 1; } } }\n    0\n}\n"),
            ("adv_25.rs", "deep_nesting_spaghetti", "pub fn multi(a: bool, b: bool, c: bool) -> i32 {\n    let mut acc = 0;\n    if a { acc += 1; } if b { acc += 2; } if c { acc += 4; }\n    acc\n}\n"),
        ];

        for (path, expected_rule, code) in adv_cases {
            let res = scorer.score_content_strict(path, code);
            let strict_sc = res.strict_score.unwrap();
            assert!(
                res.triggered_vetoes.iter().any(|v| v.starts_with(expected_rule)),
                "{path}: expected `{expected_rule}`, got {:?}",
                res.triggered_vetoes
            );
            assert!(strict_sc <= 0.50, "{path}: expected strict_score <= 0.50, got {strict_sc}");
            if expected_rule == "bidi_trojan_source" || expected_rule == "uts39_homoglyph" {
                assert!(strict_sc <= 0.15, "{path}: expected <= 0.15, got {strict_sc}");
            }
        }

        // Verify expect-with-reason is allowed while bare expect is vetoed
        let expect_ok = scorer.score_content_strict(
            "ok_expect.rs",
            concat!("#", "[expect(dead_code, reason = \"kept for ABI compatibility\")]\npub fn helper(val: i32) -> i32 {\n    val + 1\n}\n"),
        );
        assert!(expect_ok.triggered_vetoes.is_empty(), "{:?}", expect_ok.triggered_vetoes);
        let expect_bad = scorer.score_content_strict(
            "bad_expect.rs",
            concat!("#", "[expect(dead_code)]\npub fn helper(val: i32) -> i32 {\n    val + 1\n}\n"),
        );
        assert!(expect_bad.triggered_vetoes.iter().any(|v| v.starts_with("lint_suppression")));

        // 2. Clean idiom controls (incl. Go hard tabs) -> 0 vetoes & exact clean score
        let clean_go = "package ledger\nfunc ApplyCredit(total int, amount int) int {\n\tif amount <= 0 {\n\t\treturn total\n\t}\n\treturn total + amount\n}\n";
        let go_res = scorer.score_content_strict("ledger.go", clean_go);
        assert!(go_res.triggered_vetoes.is_empty());
        assert_eq!(go_res.strict_score, Some(go_res.score));
        let audit_cmp = strict::audit_anti_gaming(
            "clean.rs",
            "pub fn add(left: i32, right: i32) -> i32 {\n    left + right\n}\n",
            "pub fn add(left: i32, right: i32) -> i32 {\n    todo!()\n}\n",
        );
        assert!(!audit_cmp.passed && audit_cmp.introduced_vetoes.len() == 1);

        // 3. Audit ALL real src/**/*.rs files -> ZERO false vetoes!
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk_rs_files(&src_dir, &src_dir, &mut files);
        files.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(files.len() >= 47, "expected >= 47 real src files, got {}", files.len());

        let mut cfg_test_mods = BTreeSet::new();
        for (rel, abs) in &files {
            let src = fs::read_to_string(abs).unwrap();
            for target in strict::find_cfg_test_mod_decls(rel, &src) {
                cfg_test_mods.insert(target);
            }
        }
        let mut false_vetoes = Vec::new();
        for (rel, abs) in &files {
            let src = fs::read_to_string(abs).unwrap();
            let is_test = strict::is_test_only_path(rel) || cfg_test_mods.contains(rel);
            let rep = strict::audit_source_with_test_flag(rel, &src, 0.8808, is_test);
            for v in rep.formatted_vetoes() {
                false_vetoes.push(format!("{rel}: {v}"));
            }
        }
        assert!(
            false_vetoes.is_empty(),
            "false vetoes across {} real src/**/*.rs files:\n{}",
            files.len(),
            false_vetoes.join("\n")
        );
    }

    #[test]
    fn weights_override_zip_bomb_and_digest_guard() {
        let real_npz = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/codehygiene/weights.npz")).unwrap();
        let real_digest = blake3::hash(&real_npz).to_hex().to_string();
        let ok_repo = TempRepo::new("weights_ok");
        let wdir = ok_repo.path().join(".refit/weights");
        fs::create_dir_all(&wdir).unwrap();
        fs::write(wdir.join("hygiene.npz"), &real_npz).unwrap();
        let write_manifest = |v: &str, a: &str, f: &str, d: &str| {
            let body = format!(r#"{{"version":"{v}","arch":"{a}","file":"{f}","blake3":"{d}"}}"#);
            fs::write(wdir.join("hygiene.json"), body).unwrap();
        };
        write_manifest("0.7.0", "v07", "hygiene.npz", &real_digest);
        let scorer = HygieneScorer::open_persistent_checked(ok_repo.path()).unwrap();
        assert_eq!(scorer.weights_blake3_prefix(), &real_digest[..16]);
        assert_eq!(scorer.score_content("a.rs", "pub const X: usize = 1;\n").score, 0.8808);

        let bad_digest = "0".repeat(64);
        for (v, a, f, d, want) in [
            ("0.7.0", "v07", "hygiene.npz", bad_digest.as_str(), "digest mismatch"),
            ("0.8.0", "v08", "hygiene.npz", real_digest.as_str(), "unsupported weights arch"),
            ("0.7.0", "v07", "../hygiene.npz", real_digest.as_str(), "unsafe weights filename"),
        ] {
            write_manifest(v, a, f, d);
            let err = HygieneScorer::open_persistent_checked(ok_repo.path()).err().unwrap();
            assert!(err.to_string().contains(want), "got {err}");
        }
        fs::remove_file(wdir.join("hygiene.json")).unwrap();
        assert!(HygieneScorer::open_persistent_checked(ok_repo.path()).err().unwrap().to_string().contains("unpinned"));
        let oversized = vec![0u8; weights::MAX_NPZ_ARCHIVE_BYTES + 1];
        assert!(weights::load_weights_from_npz(&oversized).err().unwrap().to_string().contains("exceeds"));

        let bomb_raw = vec![0u8; weights::MAX_NPY_ENTRY_BYTES + 4096];
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&bomb_raw).unwrap();
        let comp = enc.finish().unwrap();
        assert!(comp.len() < 16 * 1024);
        let bomb_zip = build_single_entry_zip("trits.npy", 8, &comp, bomb_raw.len() as u32);
        assert!(weights::load_weights_from_npz(&bomb_zip).err().unwrap().to_string().contains("zip-bomb"));
        assert!(weights::load_weights_from_npz(&real_npz[..40]).is_err());
    }

    #[test]
    fn test_ego_graph_wired_into_strict_with_v07_parity() {
        let scorer = HygieneScorer::new();
        let src = "pub struct Order {\n    pub total_cents: u64,\n}\n\
            pub fn compute_tax(amount: u64, rate_bps: u64) -> u64 {\n    (amount * rate_bps) / 10_000\n}\n";
        let base = scorer.score_content("src/order.rs", src);

        let clean_ego = FileEgoContext { fan_in: 2, fan_out: 1, ..FileEgoContext::default() };
        let clean_res = scorer.score_content_strict_with_ego("src/order.rs", src, Some(&clean_ego));
        assert_eq!((clean_res.score, clean_res.strict_score, clean_res.multiscale_score), (base.score, Some(base.score), Some(base.score)));
        assert!(clean_res.triggered_vetoes.is_empty());
        let ca: strict::CalibrationAudit = serde_json::from_str(clean_res.calibration_log.as_deref().unwrap()).unwrap();
        assert_eq!((ca.raw_v07, ca.delta_cross, ca.use_5trit_lut), (base.score, 0.0, false));

        let bad_ego = FileEgoContext {
            fan_in: 9, fan_out: 8, neighbor_dirs: 3, has_module_cycle: true, tangle: true,
            external_field_readers: 4, clone_cluster_hits: 2, churn_commits: 25,
        };
        let bad_res = scorer.score_content_strict_with_ego("src/order.rs", src, Some(&bad_ego));
        assert_eq!(bad_res.score, base.score);
        assert!(bad_res.strict_score.unwrap() <= 0.48 && bad_res.multiscale_score.unwrap() <= 0.48 && bad_res.veto_cap.unwrap() <= 0.48);
        for rule in ["module-cycle", "tangle", "field-leak"] {
            assert!(bad_res.triggered_vetoes.iter().any(|r| r == rule), "missing `{rule}`");
        }
        let ba: strict::CalibrationAudit = serde_json::from_str(bad_res.calibration_log.as_deref().unwrap()).unwrap();
        assert!(ba.delta_cross > 0.0 && ba.delta_block > 0.0);
    }

    #[test]
    fn default_v07_zero_delta_parity() {
        let mut long_src = String::with_capacity(48 * 1024);
        for i in 0..1400 {
            long_src.push_str(&format!("pub const ITEM_{i}: usize = {};\n", i * 3 + 1));
        }
        let w = weights::v07_weights();
        let (res, sparse_alloc) = model::reset_and_check_sparse_alloc_on_default(&long_src, w);
        let scalar = model::score_v07_scalar_only(&long_src, w);
        assert!(!sparse_alloc, "default path must keep SparseScratch unallocated (None)");
        assert_eq!(res.score, scalar.score, "default path must maintain 0.000000000 scalar parity");
    }

    const CLONE_FN_BLOCK: &str = concat!(
        "pub fn process_order_alpha(raw_input: &[u64], limit: u64, fee_bps: u64) -> u64 {\n",
        "    let mut total = raw_input.iter().copied().sum::<u64>();\n",
        "    let normalized = total.saturating_mul(17).saturating_add(limit);\n",
        "    let clamped = normalized.clamp(100, 100_000);\n",
        "    let weighted = (clamped * 3) / 2 + raw_input.len() as u64;\n",
        "    let fee_deduct = (weighted * fee_bps) / 10_000;\n",
        "    let adjusted = weighted.saturating_sub(fee_deduct + limit / 4);\n",
        "    let checksum = adjusted.rotate_left(5) ^ 0x5A5A_1234;\n",
        "    let bucket = (checksum % 997).saturating_add(adjusted);\n",
        "    let reserve = (bucket / 3).clamp(16, 4096);\n",
        "    let net_delta = bucket.saturating_sub(reserve).wrapping_add(clamped);\n",
        "    let audited = (net_delta ^ checksum).rotate_right(3);\n",
        "    total = total.wrapping_add(audited).saturating_sub(clamped);\n",
        "    total ^ checksum\n}\n\n",
    );

    fn build_far_clone_fixture(renamed: bool) -> String {
        let block_b = if renamed {
            CLONE_FN_BLOCK
                .replace("process_order_alpha", "process_order_omega")
                .replace("raw_input", "src_slice")
                .replace("limit", "bound")
                .replace("fee_bps", "tax_bps")
                .replace("total", "accum")
        } else {
            CLONE_FN_BLOCK.to_string()
        };
        let mut out = String::with_capacity(48 * 1024);
        for i in 0..30 {
            out.push_str(&format!("pub fn preamble_fn_{i}(v: u64) -> u64 {{\n    v.wrapping_add({})\n}}\n", i * 7 + 3));
        }
        out.push_str(CLONE_FN_BLOCK);
        for i in 0..260 {
            out.push_str(&format!(
                "pub fn mid_worker_{i}(arg_{i}: u64) -> u64 {{\n    let res_{i} = arg_{i}.wrapping_mul({}) ^ {};\n    res_{i}\n}}\n",
                i * 11 + 5, i * 13 + 9
            ));
        }
        out.push_str(&block_b);
        for i in 0..80 {
            out.push_str(&format!("pub fn epilogue_fn_{i}(z: u64) -> u64 {{\n    z.saturating_sub({})\n}}\n", i * 5 + 2));
        }
        out
    }

    #[test]
    fn wired_cobs2_engram_lsa_mli_far_clone_recall() {
        let scorer = HygieneScorer::new();
        let approx_opts = EvalOptions { approx_long: true, approx_threshold: 1024 };
        for (label, renamed, min_hits) in [("exact", false, 12usize), ("renamed", true, 7usize)] {
            let src = build_far_clone_fixture(renamed);
            let default_res = scorer.score_content(&format!("far_{label}.rs"), &src);
            let (approx_res, tel_opt) = scorer.score_content_with_opts(&format!("far_{label}.rs"), &src, approx_opts);
            let tel = tel_opt.expect("cold approx_long run must return SparseAttnTelemetry");
            let dup_hits = approx_res.squiggles.iter().filter(|s| s.kind == "duplicate_block" && s.probability >= 0.80).count();
            assert!(dup_hits >= min_hits, "{label}: expected >= {min_hits} duplicate_block hits >= 0.80, got {dup_hits} (def={}, app={})", default_res.score, approx_res.score);
            assert!(default_res.score - approx_res.score >= 0.08, "{label}: score drop < 0.08 (def={}, app={})", default_res.score, approx_res.score);
            assert!(tel.mli_reused && tel.engram_bytes > 0 && tel.engram_bytes <= 128 * 1024 && tel.cobs_blocks > 0);
            assert_eq!(tel.attn_mode, "v07_cobs2_engram_lsa_mli");
            assert_eq!(scorer.score_content(&format!("far_{label}.rs"), &src).score, default_res.score);
        }
    }

    #[test]
    fn wired_cobs2_real_strict_preserves_local_squiggles() {
        let scorer = HygieneScorer::new();
        let strict_src = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hygiene/strict.rs")).unwrap();
        let def_strict = scorer.score_content("src/hygiene/strict.rs", &strict_src);
        let (app_strict, tel_opt) = scorer.score_content_with_opts("src/hygiene/strict.rs", &strict_src, EvalOptions { approx_long: true, approx_threshold: 256 });
        assert!(tel_opt.is_some() && app_strict.score <= def_strict.score + 0.005);
        for sq in def_strict.squiggles.iter().filter(|s| s.kind == "duplicate_block") {
            assert!(app_strict.squiggles.iter().any(|a| a.kind == "duplicate_block" && a.line == sq.line), "lost strict.rs:L{}", sq.line);
        }

        let mut long_local = String::with_capacity(48 * 1024);
        for i in 0..380 {
            long_local.push_str(&format!("pub fn pad_top_{i}(x: u64) -> u64 {{\n    x.wrapping_add({})\n}}\n", i * 3 + 1));
        }
        long_local.push_str(CLONE_FN_BLOCK);
        long_local.push_str(&CLONE_FN_BLOCK.replace("process_order_alpha", "process_order_beta"));
        for i in 0..120 {
            long_local.push_str(&format!("pub fn pad_bot_{i}(y: u64) -> u64 {{\n    y.wrapping_mul({})\n}}\n", i * 5 + 2));
        }
        let def_loc = scorer.score_content("long_local.rs", &long_local);
        let (app_loc, _) = scorer.score_content_with_opts("long_local.rs", &long_local, EvalOptions { approx_long: true, approx_threshold: 1024 });
        let def_dups: Vec<usize> = def_loc.squiggles.iter().filter(|s| s.kind == "duplicate_block").map(|s| s.line).collect();
        assert!(!def_dups.is_empty(), "fixture must have local duplicate_block squiggles");
        for ln in def_dups {
            assert!(app_loc.squiggles.iter().any(|s| s.kind == "duplicate_block" && s.line == ln), "lost local dup at L{ln}");
        }
        assert!(app_loc.score <= def_loc.score + 0.005);
    }

    #[test]
    fn wired_dense_qk_partition_unified_and_cross_layer_mli() {
        let (mae_2048, tel_2048) = ops::bench_dense_qk_two_layer(2048);
        let (_mae_4096, tel_4096) = ops::bench_dense_qk_two_layer(4096);
        let ratio = (tel_4096.pairs_evaluated as f64) / (tel_2048.pairs_evaluated as f64);
        assert!(mae_2048 <= 0.01, "DenseQk MAE {mae_2048:.5} > 0.01");
        assert!(ratio <= 2.15, "DenseQk scaling ratio {ratio:.3} > 2.15");
        assert!(tel_2048.mli_reused && tel_2048.mli_reuse_hits >= 1);
        assert!(tel_4096.mli_reused && tel_4096.mli_reuse_hits >= 1);
    }
}
