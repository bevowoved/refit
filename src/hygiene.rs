mod model;
mod ops;
mod simd;
pub mod strict;
mod weights;

pub use ops::{EvalOptions, SparseStats};

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{
    MemGate, SieveCache, budget, collect_os_memory, ensure_cachedir_tag, env_truthy,
    under_memory_pressure, write_atomic,
};
use crate::source::{quantile, read_text};
use crate::types::{FileEgoContext, HygieneResult, Squiggle};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const CACHE_MAGIC: &[u8; 8] = b"RFHG0002";
const CACHE_ZSTD_WINDOW_LOG: u32 = 18;

fn estimate_neural_bytes(content: &str) -> usize {
    let lines = crate::source::count_lines(content.as_bytes());
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
    let raw = v[top]
        .as_u64()
        .or_else(|| v["formatter"][nested].as_u64())?;
    Some(raw as usize)
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

use crate::platform::{read_u16_str as read_str_u16, write_u16_str as write_str_u16};

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

/// Encode cached hygiene results into compact little-endian bytes.
fn encode_compact_cache(map: &HashMap<String, HygieneResult>) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 96);
    out.extend_from_slice(CACHE_MAGIC);
    let _ = out.write_u32::<LittleEndian>(map.len() as u32);
    for (key, r) in map {
        write_str_u16(&mut out, key);
        let _ = out.write_f64::<LittleEndian>(r.score);
        let _ = out.write_u64::<LittleEndian>(r.latency_us);
        let bits = [
            r.strict_score.is_some(),
            r.veto_cap.is_some(),
            r.total_squiggles.is_some(),
            r.event_load.is_some(),
            r.multiscale_score.is_some(),
            r.calibration_log.is_some(),
        ];
        let mut flags = 0u8;
        for (idx, present) in bits.into_iter().enumerate() {
            if present {
                flags |= 1u8 << idx;
            }
        }
        out.push(flags);
        write_opt_f64(&mut out, r.strict_score);
        write_opt_f64(&mut out, r.veto_cap);
        if let Some(v) = r.total_squiggles {
            let _ = out.write_u32::<LittleEndian>(v.min(u32::MAX as usize) as u32);
        }
        write_opt_f64(&mut out, r.event_load);
        write_opt_f64(&mut out, r.multiscale_score);
        if let Some(ref s) = r.calibration_log {
            write_str_u16(&mut out, s);
        }

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
        for veto in r.triggered_vetoes.iter().take(veto_cnt) {
            write_str_u16(&mut out, veto);
        }
    }
    out
}

fn decode_cache_entry(rd: &mut Cursor<&[u8]>) -> Option<(String, HygieneResult)> {
    let key = read_str_u16(rd)?;
    let (score, latency_us, flags) = (
        rd.read_f64::<LittleEndian>().ok()?,
        rd.read_u64::<LittleEndian>().ok()?,
        rd.read_u8().ok()?,
    );
    let strict_score = read_opt_f64(rd, flags, 1)?;
    let veto_cap = read_opt_f64(rd, flags, 2)?;
    let total_squiggles = if (flags & 4) != 0 {
        Some(rd.read_u32::<LittleEndian>().ok()? as usize)
    } else {
        None
    };
    let event_load = read_opt_f64(rd, flags, 8)?;
    let multiscale_score = read_opt_f64(rd, flags, 16)?;
    let calibration_log = if (flags & 32) != 0 {
        Some(read_str_u16(rd)?)
    } else {
        None
    };

    let sq_cnt = rd.read_u16::<LittleEndian>().ok()? as usize;
    let mut squiggles = Vec::with_capacity(sq_cnt.min(64));
    for _ in 0..sq_cnt {
        let line = rd.read_u32::<LittleEndian>().ok()? as usize;
        let kind_id = rd.read_u8().ok()?;
        let (kind, message) = match model::SQUIGGLE_SPEC.get(kind_id as usize) {
            Some(&(sk, sm, _)) => (sk.to_string(), sm.to_string()),
            None if kind_id == 255 => (read_str_u16(rd)?, read_str_u16(rd)?),
            None => return None,
        };
        let probability = rd.read_f64::<LittleEndian>().ok()?;
        squiggles.push(Squiggle { line, kind, probability, message });
    }
    let veto_cnt = rd.read_u16::<LittleEndian>().ok()? as usize;
    let mut triggered_vetoes = Vec::with_capacity(veto_cnt.min(32));
    for _ in 0..veto_cnt {
        triggered_vetoes.push(read_str_u16(rd)?);
    }
    Some((key, HygieneResult {
        score, squiggles, cached: false, latency_us, strict_score, veto_cap,
        total_squiggles, event_load, triggered_vetoes, multiscale_score, calibration_log,
    }))
}

fn decode_compact_cache(bytes: &[u8]) -> Option<HashMap<String, HygieneResult>> {
    if !bytes.starts_with(CACHE_MAGIC) {
        return None;
    }
    let mut rd = Cursor::new(&bytes[8..]);
    let count = rd.read_u32::<LittleEndian>().ok()? as usize;
    if count > 1_000_000 {
        return None;
    }
    let mut map = HashMap::with_capacity(count.min(4096));
    for _ in 0..count {
        let (key, val) = decode_cache_entry(&mut rd)?;
        map.insert(key, val);
    }
    Some(map)
}

fn write_compact_cache_zstd(
    path: &Path,
    map: &HashMap<String, HygieneResult>,
) -> anyhow::Result<()> {
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
            None => weights::base_weights(),
        }
    }

    /// Return the 16-hex BLAKE3 prefix of the active weights.
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

    /// Open a persistent scorer and validate any weights override.
    #[cfg(test)]
    pub fn open_persistent_checked(repo_root: &Path) -> anyhow::Result<Self> {
        Ok(Self::open_with_weights(repo_root, weights::load_verified_override(repo_root)?))
    }

    fn open_with_weights(
        repo_root: &Path,
        override_weights: Option<Arc<weights::ResolvedWeights>>,
    ) -> Self {
        let disk_path = repo_root.join(".refit/cache/hygiene_cache.zst");
        let opts = EvalOptions {
            approx_long: env_truthy("REFIT_APPROX_LONG"),
            ..EvalOptions::default()
        };
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
    ) -> (HygieneResult, Option<SparseStats>) {
        let line_count = crate::source::count_lines(content.as_bytes()).max(1);
        let approx_active = opts.approx_long && line_count > opts.approx_threshold;
        let eval_opts = EvalOptions { approx_long: approx_active, ..opts };
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
        let (mut res, stats) = model::score_opts(content, self.active_weights(), eval_opts);
        res.latency_us = started.elapsed().as_micros().max(1) as u64;
        let weight = hygiene_entry_weight(&key, &res);
        self.cache.insert(key.clone(), res.clone(), weight);
        self.dirty.store(true, Ordering::Release);

        drop(permit);
        drop(guard);
        self.release_in_flight(&key, &lock);
        (res, stats)
    }

    /// Score content with strict checks using default context.
    #[cfg(test)]
    pub fn score_content_strict(&self, path: &str, content: &str) -> HygieneResult {
        self.score_strict_ego(path, content, None)
    }

    /// Score content with strict checks and optional ego context.
    pub fn score_strict_ego(
        &self,
        path: &str,
        content: &str,
        ego: Option<&FileEgoContext>,
    ) -> HygieneResult {
        let (base, stats) = self.score_content_with_opts(path, content, self.default_opts);
        let zero_ratio = self.active_weights().weight_zero_ratio;
        strict::score_strict(path, content, &base, ego, stats.as_ref(), zero_ratio)
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
    use crate::testutil::{
        TempRepo, build_single_entry_zip, map_conds, nest_src, repeat_fn, walk_rs_files,
    };
    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    use std::collections::{BTreeSet, HashSet};
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    const CLONE_BLOCK: &str = concat!(
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

    fn sq_tuples(list: &[Squiggle]) -> Vec<(usize, String, i64)> {
        let mut out = Vec::with_capacity(list.len());
        for s in list {
            let prob = (s.probability * 10_000.0).round() as i64;
            out.push((s.line, s.kind.clone(), prob));
        }
        out
    }

    fn js_sq_tuples(list: &[serde_json::Value]) -> Vec<(usize, String, i64)> {
        let mut out = Vec::with_capacity(list.len());
        for s in list {
            let ln = s["line"].as_u64().unwrap() as usize;
            let kind = s["kind"].as_str().unwrap();
            let raw_p = s["probability"].as_f64().unwrap();
            out.push((ln, String::from(kind), (raw_p * 10_000.0).round() as i64));
        }
        out
    }

    fn collect_src_rs() -> Vec<(String, PathBuf)> {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk_rs_files(&src_dir, &src_dir, &mut files);
        files.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(files.len() >= 47, "expected >= 47 src/**/*.rs files, got {}", files.len());
        files
    }

    fn dup_lines(res: &HygieneResult) -> Vec<usize> {
        let mut lines = Vec::new();
        for sq in &res.squiggles {
            if sq.kind == "duplicate_block" {
                lines.push(sq.line);
            }
        }
        lines
    }

    fn parse_audit(res: &HygieneResult) -> strict::CalibrationAudit {
        let raw = res.calibration_log.as_deref().unwrap();
        serde_json::from_str(raw).unwrap()
    }

    fn build_clone_fixture(pre: usize, mid: usize, post: usize, renamed: bool) -> String {
        let mut second = CLONE_BLOCK.to_string();
        if renamed {
            for (from, to) in [
                ("process_order_alpha", "process_order_omega"),
                ("raw_input", "src_slice"),
                ("limit", "bound"),
                ("fee_bps", "tax_bps"),
                ("total", "accum"),
            ] {
                second = second.replace(from, to);
            }
        }
        let mut out = String::with_capacity(48 * 1024);
        for i in 0..pre {
            let v = i * 7 + 3;
            out.push_str(&format!("pub fn pre_{i}(x: u64) -> u64 {{\n    x.wrapping_add({v})\n}}\n"));
        }
        out.push_str(CLONE_BLOCK);
        for i in 0..mid {
            let (m, k) = (i * 11 + 5, i * 13 + 9);
            out.push_str(&format!(
                "pub fn mid_{i}(a_{i}: u64) -> u64 {{\n    \
                 let r_{i} = a_{i}.wrapping_mul({m}) ^ {k};\n    r_{i}\n}}\n"
            ));
        }
        out.push_str(&second);
        for i in 0..post {
            let v = i * 5 + 2;
            out.push_str(&format!("pub fn post_{i}(z: u64) -> u64 {{\n    z.saturating_sub({v})\n}}\n"));
        }
        out
    }

    fn strict_vectors() -> Vec<(&'static str, &'static str, String)> {
        let pad_tail = format!(
            "pub fn sneak(a: i32, b: i32) -> i32 {{\n    let x = a + 1;{}\
             let y = if b > 0 {{ x * 2 }} else {{ x - 1 }};\n    y\n}}\n",
            " ".repeat(115),
        );
        let terms: String = (1..=35).map(|i| format!(" + {i}")).collect();
        let wide_ops = format!("pub fn pk(val: i32) -> i32 {{\n    val{terms}\n}}\n");
        let rs_c = map_conds(&["v1", "v2", "v3", "v4", "v5", "v6"], |v| format!("if {v} > 0 {{"));
        let rs_h = "pub fn deep(v1: i32, v2: i32, v3: i32, v4: i32, v5: i32, v6: i32) -> i32 {";
        let deep_if = nest_src(rs_h, "    ", &rs_c, "return v1 + v6;", Some("}"));
        let flat_indent = nest_src(rs_h, " ", &rs_c, "return v1 + v6;", Some("}"));
        let go_c = map_conds(&["a", "b", "c", "d", "e", "f"], |v| format!("if {v} {{"));
        let go_h = "package sample\nfunc Deep(a, b, c, d, e, f bool) int {";
        let deep_go = nest_src(go_h, "\t", &go_c, "return 1", Some("}"));

        let blank_run = repeat_fn(
            "pub fn pad(v: i32) -> i32 {", 1..=6,
            |i| format!("    let v_{i} = v + {i};\n"), "\n\n\n\n    v_6\n}\n",
        );
        let blank_alt = repeat_fn(
            "pub fn sp(v: i32) -> i32 {", 1..=7,
            |i| format!("\n    let s_{i} = v + {i};\n"), "\n    s_7\n}\n",
        );
        let echo_cmt = repeat_fn(
            "pub fn echo(base_val: i32) -> i32 {",
            1..=3,
            |i| {
                format!(
                    "    // compute step_val_{i} from base_val\n    \
                     let step_val_{i} = base_val + {i};\n"
                )
            },
            "    step_val_3\n}\n",
        );
        let banner_cmt = repeat_fn(
            "// Step 0: init stage",
            1..=7,
            |i| format!("// Step {i}: process stage {i}\n"),
            "pub fn add(left: i32, right: i32) -> i32 {\n    left + right\n}\n",
        );

        let statics: [(&str, &str, &str); 16] = [
            ("bidi.rs", "bidi_trojan_source", "pub fn ok() { /* \u{202E} \u{2066} */ }\n"),
            ("zwsp.rs", "bidi_trojan_source", "pub fn hid() -> &'static str { \"a\u{200B}b\" }\n"),
            ("cyrl.rs", "uts39_homoglyph", "pub fn auth() -> bool { let \u{0430}d = true; \u{0430}d }\n"),
            ("grk.rs", "uts39_homoglyph", "pub fn chk() -> i32 { let t\u{03BF}k = 1\u{037E} t\u{03BF}k }\n"),
            ("alw.rs", "lint_suppression", concat!("#", "[allow(dead_code)]\npub fn sup() {}\n")),
            ("inw.rs", "lint_suppression", concat!("#!", "[allow(warnings)]\npub fn sup() {}\n")),
            ("cfg.rs", "lint_suppression", concat!("#", "[cfg_attr(all(), allow(clippy::all))]\n")),
            ("serde_f.rs", "serde_flatten", "pub struct P2 { #[serde(flatten)] pub m: String }\n"),
            ("short_v.rs", "short_ident", concat!(
                "pub fn run(p: i32) -> i32 {\n",
                "    let (a, b, c, d, f) = (p, 1, 2, 3, 4);\n    a + b + c + d + f\n}\n",
            )),
            ("short_f.rs", "short_ident", "pub fn f(val: i32) -> i32 { val + 1 }\n"),
            ("todo_m.rs", "prod_placeholder_macro", "pub fn inc() -> i32 { todo!(\"x\") }\n"),
            ("unimpl.rs", "prod_placeholder_macro", "pub fn stub() -> i32 { unimplemented!() }\n"),
            (
                "raw_u.rs",
                "unsafe_undocumented",
                "pub fn bits(v: u64) -> f64 {\n    unsafe { std::mem::transmute(v) }\n}\n",
            ),
            (
                "lock_u.rs",
                "std_lock_unwrap",
                "pub fn bump(m: &std::sync::Mutex<u64>) {\n    *m.lock().unwrap() += 1;\n}\n",
            ),
            ("pack_c.rs", "deep_nesting_spaghetti", concat!(
                "pub fn pk(a: bool, b: bool, c: bool) -> i32 {\n",
                "    if a { if b { if c { return 1; } } }\n    0\n}\n",
            )),
            ("multi_c.rs", "deep_nesting_spaghetti", concat!(
                "pub fn mc(a: bool, b: bool, c: bool) -> i32 {\n    let mut n = 0;\n",
                "    if a { n += 1; } if b { n += 2; } if c { n += 4; }\n    n\n}\n",
            )),
        ];
        let mut out: Vec<(&'static str, &'static str, String)> =
            statics.into_iter().map(|(p, r, s)| (p, r, s.to_string())).collect();
        out.extend([
            ("blank_r.rs", "blank_line_stuffing", blank_run),
            ("blank_a.rs", "blank_line_stuffing", blank_alt),
            ("echo_c.rs", "comment_echo_stuffing", echo_cmt),
            ("ban_c.rs", "comment_echo_stuffing", banner_cmt),
            ("pad_t.rs", "hidden_dense_tail", pad_tail),
            ("wide_o.rs", "hidden_dense_tail", wide_ops),
            ("deep_i.rs", "deep_nesting_spaghetti", deep_if),
            ("flat_i.rs", "deep_nesting_spaghetti", flat_indent),
            ("deep_g.go", "deep_nesting_spaghetti", deep_go),
        ]);
        out
    }

    #[cfg(target_arch = "x86_64")]
    fn check_avx2_exp() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let eval_exp = |xs: &[f64; 4]| -> [f64; 4] {
            let mut out = [0.0f64; 4];
            // SAFETY: AVX2 support verified above.
            unsafe {
                let v = std::arch::x86_64::_mm256_loadu_pd(xs.as_ptr());
                let r = simd::exp_f64x4_avx2_non_fma(v);
                std::arch::x86_64::_mm256_storeu_pd(out.as_mut_ptr(), r);
            }
            out
        };
        assert_eq!(eval_exp(&[-708.5, -709.0, -1000.0, -1e9]), [0.0; 4]);
        let mut max_ulp = 0u64;
        for step in 0..2500 {
            let xs: [f64; 4] = std::array::from_fn(|j| {
                -708.0 + (((step * 4 + j) as f64) / 9999.0) * 1417.0
            });
            let got = eval_exp(&xs);
            for j in 0..4 {
                let want = xs[j].exp().to_bits();
                max_ulp = max_ulp.max(got[j].to_bits().abs_diff(want));
            }
        }
        assert!(max_ulp <= 2, "exp_f64x4 ULP {max_ulp} > 2");
    }

    fn verify_js_case(
        scorer: &HygieneScorer,
        idx: usize,
        label: &str,
        content: &str,
        js: &serde_json::Value,
    ) -> f64 {
        assert!(js.get("error").is_none(), "JS error on {label}: {js}");
        let js_score = js["score"].as_f64().unwrap();
        let js_sq = js["squiggles"].as_array().unwrap();
        let rust = scorer.score_content(label, content);
        let rust_sq = sq_tuples(&rust.squiggles);

        if idx < 8 || label.starts_with("edge_") || label.contains("hygiene") {
            let sc = model::score_scalar(content, weights::base_weights());
            assert_eq!(rust.score, sc.score, "AVX2 vs scalar on {label}");
            assert_eq!(rust_sq, sq_tuples(&sc.squiggles), "scalar squiggles on {label}");
        }
        assert_eq!(rust.score, js_score, "score mismatch on {label}");
        assert_eq!(rust.total_squiggles, Some(js_sq.len()), "count on {label}");

        if js_sq.len() <= 48 {
            let (mut rs, mut jv) = (rust_sq, js_sq_tuples(js_sq));
            assert_eq!(rs.len(), jv.len(), "sq len on {label}");
            rs.sort();
            jv.sort();
            for (r_item, j_item) in rs.iter().zip(jv.iter()) {
                assert_eq!((r_item.0, &r_item.1), (j_item.0, &j_item.1), "kind on {label}");
                assert!((r_item.2 - j_item.2).abs() <= 1, "prob on {label}");
            }
        }
        (rust.score - js_score).abs()
    }

    #[test]
    fn cache_and_six_smells() {
        let tmp = TempRepo::new("hyg_disk");
        let s1 = HygieneScorer::open_persistent(tmp.path());
        let rs_clean = concat!(
            "pub struct S;\nimpl S {\n    pub fn run(&self, x: Option<i32>) -> i32 {\n",
            "        x.map_or(0, |v| v + 1)\n    }\n}\n",
        );
        let cold = s1.score_content("s.rs", rs_clean);
        let warm = s1.score_content("s.rs", rs_clean);
        assert!(!cold.cached && warm.cached && warm.latency_us <= 1000 && cold.score > 0.70);
        s1.flush_cache();

        let reopened = HygieneScorer::open_persistent(tmp.path());
        let disk = reopened.score_content("s.rs", rs_clean);
        assert!(disk.cached && reopened.warm_us() >= 1 && (cold.score - disk.score).abs() < 1e-9);

        let leg = TempRepo::new("hyg_legacy_json");
        let leg_key = format!("v07:{}", MvccStore::hash_bytes(rs_clean.as_bytes()));
        let leg_path = leg.path().join(".refit/cache/hygiene_cache.zst");
        write_zstd(&leg_path, &HashMap::from([(leg_key, cold.clone())])).unwrap();
        let migrated = HygieneScorer::open_persistent(leg.path()).score_content("s.rs", rs_clean);
        assert!(migrated.cached && (migrated.score - cold.score).abs() < 1e-9);

        let conds = map_conds(&["a", "b", "c", "d", "e"], |v| format!("if {v}:"));
        let nested = nest_src("def f(a, b, c, d, e):", "    ", &conds, "return a+b+c+d+e", None);
        let dense = concat!(
            "def d(x):\n    return (x + 1 if x > 0 else (x - 1 if x < -5 else ",
            "(x * 2 if x == 0 else x * 3))) + len(str(x).strip().lower().replace('a', 'b'))\n",
        );
        let dup = format!("{CLONE_BLOCK}{CLONE_BLOCK}");
        let rep = repeat_fn(
            "def pick(k):", 1..=8, |i| format!("    if k == {i}:\n        return {i} * 10\n"), "",
        );
        let cr = repeat_fn(
            "def cr(x):", 1..=45, |i| format!("    v_{i} = x + {i}\n"), "    return v_45\n",
        );
        let echo = repeat_fn(
            "def c(x, y, z):", 0..=14, |i| format!("    # step_{i} = old_{i}(x, y, z)\n"),
            "    return x\n",
        );

        let mut kinds = HashSet::new();
        for src in [nested.as_str(), dense, &dup, &rep, &cr, &echo] {
            for sq in reopened.score_content("t.py", src).squiggles {
                kinds.insert(sq.kind);
            }
        }
        let expected = [
            "deep_nesting", "dense_line", "duplicate_block",
            "repetitive_structure", "cramped_layout", "comment_echo",
        ];
        for want in expected {
            assert!(kinds.contains(want), "missing {want} in {kinds:?}");
        }
    }

    #[test]
    fn parity_vs_js_and_scalar() {
        let scorer = HygieneScorer::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(scorer.score_content("empty.rs", "").score, 0.8808);
        assert_eq!(scorer.score_content("one.rs", "pub const X: usize = 1;\n").score, 0.8808);

        let long_src: String = (0..1400)
            .map(|i| format!("pub const N_{i}: usize = {};\n", i * 3 + 1))
            .collect();
        let w = weights::base_weights();
        let (def_res, sparse_alloc) = model::check_default_no_sparse(&long_src, w);
        assert!(!sparse_alloc && def_res.score == model::score_scalar(&long_src, w).score);

        let lf = "pub fn add(left: i32, right: i32) -> i32 {\n    left + right\n}\n";
        let lf_r = scorer.score_content("lf.rs", lf);
        let crlf_r = scorer.score_content("crlf.rs", &lf.replace('\n', "\r\n"));
        assert_eq!(crlf_r.score, lf_r.score);
        assert_eq!(sq_tuples(&crlf_r.squiggles), sq_tuples(&lf_r.squiggles));

        #[cfg(target_arch = "x86_64")]
        check_avx2_exp();

        let mut cases: Vec<(String, String)> = collect_src_rs()
            .into_iter()
            .map(|(rel, abs)| (format!("src/{rel}"), fs::read_to_string(abs).unwrap()))
            .collect();
        let go_tab = concat!(
            "package p\nfunc F(x int) int {\n",
            "\tif x > 0 {\n\t\treturn x\n\t}\n\treturn 0\n}\n",
        );
        let utf8_src = "pub fn hi() -> &'static str {\n    \"héllo 🚀 日本語\"\n}\n";
        cases.extend([
            ("edge_crlf.rs".into(), "pub fn a() -> i32 {\r\n    1\r\n}\r\n".into()),
            ("edge_tabs.go".into(), go_tab.into()),
            ("edge_utf8.rs".into(), utf8_src.into()),
            ("edge_nul.txt".into(), "a\x00b\x0bc\u{2028}d\r\ne".into()),
        ]);

        let mut child = Command::new("node")
            .arg(root.join("vendor/codehygiene/worker.mjs"))
            .arg(root.join("vendor/codehygiene/weights.npz"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn node worker");
        let mut stdin = child.stdin.take().unwrap();
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let mut max_delta = 0.0f64;

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
            max_delta = max_delta.max(verify_js_case(&scorer, idx, label, content, &js));
        }
        drop(stdin);
        assert!(child.wait().unwrap().success());
        assert_eq!(max_delta, 0.0, "expected 0.0 delta, got {max_delta}");
    }

    #[test]
    fn strict_rules_and_repo_scan() {
        let scorer = HygieneScorer::new();
        let mut cases = strict_vectors();
        assert_eq!(cases.len(), 25);
        let bare_ex = concat!("#", "[expect(dead_code)]\npub fn h1() {}\n");
        cases.push(("bad_ex.rs", "lint_suppression", bare_ex.to_string()));

        for (path, want, code) in &cases {
            let res = scorer.score_content_strict(path, code);
            let cap = match *want {
                "bidi_trojan_source" | "uts39_homoglyph" => 0.15,
                _ => 0.50,
            };
            let hit = res.triggered_vetoes.iter().any(|v| v.starts_with(want));
            assert!(hit, "{path}: expected `{want}`, got {:?}", res.triggered_vetoes);
            assert!(res.strict_score.unwrap() <= cap, "{path}: expected <= {cap}");
        }

        let ok_ex = concat!("#", "[expect(dead_code, reason = \"ABI\")]\npub fn h1() {}\n");
        let clean_go = concat!(
            "package ledger\nfunc Add(a int, b int) int {\n",
            "\tif b <= 0 {\n\t\treturn a\n\t}\n\treturn a + b\n}\n",
        );
        for (path, src) in [("ok_ex.rs", ok_ex), ("ledger.go", clean_go)] {
            let res = scorer.score_content_strict(path, src);
            assert!(res.triggered_vetoes.is_empty());
            assert_eq!(res.strict_score, Some(res.score));
        }

        let cmp = strict::audit_anti_gaming(
            "clean.rs",
            "pub fn add(left: i32, right: i32) -> i32 { left + right }\n",
            "pub fn add(left: i32, right: i32) -> i32 { todo!() }\n",
        );
        assert!(!cmp.passed && cmp.introduced_vetoes.len() == 1);
        assert!(cmp.introduced_vetoes[0].starts_with("prod_placeholder_macro"));

        let files = collect_src_rs();
        let mut cfg_test_mods = BTreeSet::new();
        let mut loaded = Vec::with_capacity(files.len());
        for (rel, abs) in files {
            let src = fs::read_to_string(abs).unwrap();
            cfg_test_mods.extend(strict::find_cfg_test_mod_decls(&rel, &src));
            loaded.push((rel, src));
        }
        let mut false_vetoes = Vec::new();
        for (rel, src) in &loaded {
            let is_test = strict::is_test_only_path(rel) || cfg_test_mods.contains(rel);
            let rep = strict::audit_source(rel, src, 0.8808, is_test);
            for veto in rep.formatted_vetoes() {
                false_vetoes.push(format!("{rel}: {veto}"));
            }
        }
        assert!(false_vetoes.is_empty(), "false vetoes:\n{}", false_vetoes.join("\n"));
    }

    fn score_exact_and_sparse(
        scorer: &HygieneScorer,
        path: &str,
        src: &str,
        approx_threshold: usize,
    ) -> (HygieneResult, HygieneResult, SparseStats) {
        let exact = scorer.score_content(path, src);
        let opts = EvalOptions { approx_long: true, approx_threshold };
        let (sparse, stats) = scorer.score_content_with_opts(path, src, opts);
        (exact, sparse, stats.expect("cold approx_long run must return SparseStats"))
    }

    #[test]
    fn ego_strict_parity() {
        let scorer = HygieneScorer::new();
        let src = concat!(
            "pub struct Order {\n    pub total_cents: u64,\n}\n",
            "pub fn tax(a: u64, bps: u64) -> u64 {\n    (a * bps) / 10_000\n}\n",
        );
        let base = scorer.score_content("src/order.rs", src);

        let eval = |ego: FileEgoContext, tier: &str| {
            let res = scorer.score_strict_ego("src/order.rs", src, Some(&ego));
            let audit = parse_audit(&res);
            assert_eq!(res.score, base.score);
            assert_eq!(audit.raw_score, base.score);
            assert_eq!(audit.proof_tier, tier);
            (res, audit)
        };

        let clean_ctx = FileEgoContext { fan_in: 2, fan_out: 1, ..FileEgoContext::default() };
        let (clean, ca) = eval(clean_ctx, "exact");
        assert!(clean.triggered_vetoes.is_empty() && !ca.use_5trit_lut);
        assert_eq!(clean.strict_score, Some(base.score));
        assert_eq!(clean.multiscale_score, Some(base.score));
        assert_eq!((ca.delta_cross, ca.delta_idiom, ca.attn_mode.as_str()), (0.0, 0.0, "exact"));

        let bad_ctx = FileEgoContext {
            fan_in: 9, fan_out: 8, neighbor_dirs: 3, has_module_cycle: true, tangle: true,
            external_field_readers: 4, clone_cluster_hits: 2, churn_commits: 25,
        };
        let (bad, ba) = eval(bad_ctx, "vetoed");
        let caps = [bad.strict_score, bad.multiscale_score, bad.veto_cap];
        assert!(caps.into_iter().all(|v| v.unwrap() <= 0.48));
        assert!(ba.delta_cross > 0.0 && ba.delta_block > 0.0);
        for rule in ["module-cycle", "tangle", "field-leak"] {
            assert!(bad.triggered_vetoes.iter().any(|r| r == rule), "missing `{rule}`");
        }
    }

    #[test]
    fn sparse_far_clone_recall() {
        let scorer = HygieneScorer::new();
        for (label, renamed, min_hits) in [("exact", false, 12usize), ("renamed", true, 7usize)] {
            let src = build_clone_fixture(30, 260, 80, renamed);
            let path = format!("far_{label}.rs");
            let (def, app, st) = score_exact_and_sparse(&scorer, &path, &src, 1024);
            let hits = app
                .squiggles
                .iter()
                .filter(|s| s.kind == "duplicate_block" && s.probability >= 0.80)
                .count();
            assert!(hits >= min_hits, "{label}: hits={hits} < {min_hits}");
            assert!(def.score - app.score >= 0.08, "{label}: {} -> {}", def.score, app.score);
            assert!(st.reused && st.blocks > 0 && st.attn_mode == "sparse");
            assert!(st.index_bytes > 0 && st.index_bytes <= 128 * 1024);
            assert_eq!(scorer.score_content(&path, &src).score, def.score);
        }
    }

    #[test]
    fn dense_qk_sparse_and_reuse() {
        let (mae_2048, st_2048) = ops::bench_dense_qk_two_layer(2048);
        let (_mae_4096, st_4096) = ops::bench_dense_qk_two_layer(4096);
        let ratio = (st_4096.pairs_evaluated as f64) / (st_2048.pairs_evaluated as f64);
        assert!(mae_2048 <= 0.01 && ratio <= 2.15, "mae={mae_2048:.5} ratio={ratio:.3}");
        assert!(st_2048.reused && st_2048.reuse_hits >= 1);
        assert!(st_4096.reused && st_4096.reuse_hits >= 1);
    }

    #[test]
    fn sparse_keeps_local_squiggles() {
        let scorer = HygieneScorer::new();
        let strict_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hygiene/strict.rs");
        let strict_src = fs::read_to_string(strict_path).unwrap();
        let long_local = build_clone_fixture(380, 0, 120, false);

        for (path, src, threshold, min_dups) in [
            ("src/hygiene/strict.rs", strict_src.as_str(), 256usize, 0usize),
            ("long_local.rs", long_local.as_str(), 1024usize, 4usize),
        ] {
            let (def, app, _st) = score_exact_and_sparse(&scorer, path, src, threshold);
            assert!(app.score <= def.score + 0.005);
            let local_lines = dup_lines(&def);
            assert!(local_lines.len() >= min_dups, "{path}: expected >= {min_dups} local dups");
            let app_lines = dup_lines(&app);
            for ln in local_lines {
                assert!(app_lines.contains(&ln), "{path}: lost local dup at L{ln}");
            }
        }
    }

    #[test]
    fn weights_bounds_and_tamper_limits() {
        let npz_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/codehygiene/weights.npz");
        let real_npz = fs::read(npz_path).unwrap();
        let real_digest = blake3::hash(&real_npz).to_hex().to_string();
        let repo = TempRepo::new("weights_limits");
        let wdir = repo.path().join(".refit/weights");
        fs::create_dir_all(&wdir).unwrap();
        fs::write(wdir.join("hygiene.npz"), &real_npz).unwrap();

        let write_manifest = |v: &str, a: &str, f: &str, d: &str| {
            let body = format!(r#"{{"version":"{v}","arch":"{a}","file":"{f}","blake3":"{d}"}}"#);
            fs::write(wdir.join("hygiene.json"), body).unwrap();
        };
        write_manifest("0.7.0", "v07", "hygiene.npz", &real_digest);
        let scorer = HygieneScorer::open_persistent_checked(repo.path()).unwrap();
        assert_eq!(scorer.weights_blake3_prefix(), &real_digest[..16]);
        assert_eq!(scorer.score_content("a.rs", "pub const X: usize = 1;\n").score, 0.8808);

        let bad_digest = "0".repeat(64);
        for (v, a, f, d, want) in [
            ("0.7.0", "v07", "hygiene.npz", bad_digest.as_str(), "digest mismatch"),
            ("0.8.0", "v08", "hygiene.npz", real_digest.as_str(), "unsupported weights arch"),
            ("0.7.0", "v07", "../hygiene.npz", real_digest.as_str(), "unsafe weights filename"),
        ] {
            write_manifest(v, a, f, d);
            let err = HygieneScorer::open_persistent_checked(repo.path()).err().unwrap();
            assert!(err.to_string().contains(want), "got {err}");
        }
        fs::remove_file(wdir.join("hygiene.json")).unwrap();
        let unpinned = HygieneScorer::open_persistent_checked(repo.path()).err().unwrap();
        assert!(unpinned.to_string().contains("unpinned"));

        let assert_npz_err = |bytes: &[u8], want: &str| {
            let err = weights::load_weights_from_npz(bytes).err().unwrap();
            assert!(err.to_string().contains(want), "expected `{want}`, got {err}");
        };
        assert_npz_err(&vec![0u8; weights::MAX_NPZ_ARCHIVE_BYTES + 1], "exceeds");

        let bomb_raw = vec![0u8; weights::MAX_NPY_ENTRY_BYTES + 1];
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&bomb_raw).unwrap();
        let comp = enc.finish().unwrap();
        assert!(comp.len() < 16 * 1024);
        let bomb_zip = build_single_entry_zip("trits.npy", 8, &comp, bomb_raw.len() as u32);
        assert_npz_err(&bomb_zip, "zip-bomb");

        let hdr = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n";
        let mut bad_npy = Vec::with_capacity(10 + hdr.len() + 3);
        bad_npy.extend_from_slice(b"\x93NUMPY\x01\x00");
        bad_npy.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
        bad_npy.extend_from_slice(hdr);
        bad_npy.extend_from_slice(&[1u8, 2, 3]);
        let bad_f4_zip = build_single_entry_zip("vecs.npy", 0, &bad_npy, bad_npy.len() as u32);
        assert_npz_err(&bad_f4_zip, "misaligned <f4");
        assert_npz_err(&bad_f4_zip[..bad_f4_zip.len() - 4], "EOCD");
        assert_npz_err(&real_npz[..40], "EOCD");
    }
}
