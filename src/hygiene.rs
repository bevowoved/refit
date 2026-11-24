mod model;
mod ops;
mod weights;

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{
    MemGate, TangoCreditRing, budget, collect_os_memory, read_zstd, under_memory_pressure,
    write_zstd,
};
use crate::source::{quantile, read_text};
use crate::types::HygieneResult;
use dashmap::DashMap;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
    if toml_style(&repo_root.join("pyproject.toml"), "line-length", "indent-width", &mut cfg) {
        return cfg;
    }
    if toml_style(&repo_root.join("rustfmt.toml"), "max_width", "tab_spaces", &mut cfg) {
        return cfg;
    }
    if json_style(repo_root, &mut cfg) {
        return cfg;
    }

    if repo_root.join("Cargo.toml").exists() {
        cfg.max_line_width = 100;
        cfg.indent_width = 4;
    } else if repo_root.join("package.json").exists() {
        cfg.max_line_width = 100;
        cfg.indent_width = 2;
    }
    cfg
}

fn toml_style(path: &Path, w_key: &str, i_key: &str, cfg: &mut StyleConfig) -> bool {
    let Ok(raw) = std::fs::read_to_string(path) else { return false };
    let mut found = false;

    for line in raw.lines() {
        let head = line.split('#').next().unwrap_or("");
        let Some((k, val)) = head.trim().split_once('=') else { continue };
        let Ok(n) = val.trim().parse::<usize>() else { continue };
        let key = k.trim();

        if key == w_key {
            cfg.max_line_width = n.clamp(60, 160);
            found = true;
        } else if key == i_key {
            cfg.indent_width = n.clamp(2, 8);
            found = true;
        }
    }
    found
}

fn json_num(v: &serde_json::Value, top: &str, nested: &str) -> Option<usize> {
    let direct = v[top].as_u64();
    let fallback = v["formatter"][nested].as_u64();
    direct.or(fallback).map(|n| n as usize)
}

fn json_style(repo_root: &Path, cfg: &mut StyleConfig) -> bool {
    const FILES: [&str; 4] = [".prettierrc", ".prettierrc.json", "biome.json", "biome.jsonc"];
    for f in FILES {
        let Ok(raw) = std::fs::read_to_string(repo_root.join(f)) else { continue };
        let cleaned = strip_jsonc(&raw);
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned) else { continue };

        let w = json_num(&v, "printWidth", "lineWidth").unwrap_or(100);
        let ind = json_num(&v, "tabWidth", "indentWidth").unwrap_or(2);
        cfg.max_line_width = w.clamp(60, 160);
        cfg.indent_width = ind.clamp(2, 8);
        return true;
    }
    false
}

pub struct HygieneScorer {
    cache: DashMap<String, HygieneResult>,
    in_flight: DashMap<String, Arc<Mutex<()>>>,
    disk_path: Option<PathBuf>,
    dirty: AtomicBool,
    last_warm_us: AtomicU64,
}

impl HygieneScorer {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::with_path(None)
    }

    fn with_path(disk_path: Option<PathBuf>) -> Self {
        Self {
            cache: DashMap::new(),
            in_flight: DashMap::new(),
            disk_path,
            dirty: AtomicBool::new(false),
            last_warm_us: AtomicU64::new(0),
        }
    }

    pub fn open_persistent(repo_root: &Path) -> Self {
        let disk_path = repo_root.join(".refit/cache/hygiene_v07_cache.zst");
        let scorer = Self::with_path(Some(disk_path.clone()));
        if let Some(saved) = read_zstd::<HashMap<String, HygieneResult>>(&disk_path) {
            for (k, v) in saved {
                scorer.cache.insert(k, v);
            }
        }
        scorer
    }

    pub fn warm_us(&self) -> u64 {
        self.last_warm_us.load(Ordering::Relaxed)
    }

    pub fn flush_cache(&self) {
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        let Some(ref p) = self.disk_path else { return };
        let mut map = HashMap::with_capacity(self.cache.len());
        for entry in self.cache.iter() {
            map.insert(entry.key().clone(), entry.value().clone());
        }
        let _ = write_zstd(p, &map);
    }

    fn cache_hit(&self, key: &str, t0: Instant) -> Option<HygieneResult> {
        let mut res = self.cache.get(key)?.clone();
        let us = t0.elapsed().as_micros().max(1) as u64;
        res.cached = true;
        res.latency_us = us;
        self.last_warm_us.store(us, Ordering::Relaxed);
        Some(res)
    }

    pub fn score_content(&self, _path: &str, content: &str) -> HygieneResult {
        let key = format!("v07:{}", MvccStore::hash_bytes(content.as_bytes()));
        let t0 = Instant::now();
        if let Some(res) = self.cache_hit(&key, t0) {
            return res;
        }

        let lock = self
            .in_flight
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let guard = lock.lock();
        if let Some(res) = self.cache_hit(&key, t0) {
            drop(guard);
            self.in_flight.remove_if(&key, |_, v| Arc::ptr_eq(v, &lock));
            return res;
        }

        let est = estimate_neural_bytes(content);
        let gate = MemGate::global();
        let req = est.min(gate.capacity());
        let permit = gate.reserve_or_wait(req, Duration::from_secs(5)).ok();
        let started = Instant::now();
        let mut res = model::score_v07(content);
        res.latency_us = started.elapsed().as_micros().max(1) as u64;
        self.cache.insert(key.clone(), res.clone());
        self.dirty.store(true, Ordering::Release);

        drop(permit);
        drop(guard);
        self.in_flight.remove_if(&key, |_, v| Arc::ptr_eq(v, &lock));
        if content.len() > 128 * 1024 {
            collect_os_memory();
        }
        res
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
            // Bounded 2-lane credit ring scheduling (zero Rayon oversubscription on 2-vCPU hosts).
            let ring = TangoCreditRing::<usize, 64>::new();
            let out = Mutex::new(Vec::with_capacity(rel_paths.len()));
            let done = AtomicBool::new(false);
            std::thread::scope(|s| {
                let r_ref = &ring;
                let o_ref = &out;
                let d_ref = &done;
                let worker = s.spawn(move || {
                    loop {
                        if let Some(idx) = r_ref.try_pop() {
                            let rel = &rel_paths[idx];
                            let text = read_text(repo_root, rel, preloaded).unwrap_or_default();
                            let res = self.score_content(rel, &text);
                            o_ref.lock().push((rel.clone(), res));
                        } else if d_ref.load(Ordering::Acquire) {
                            break;
                        } else {
                            std::thread::yield_now();
                        }
                    }
                });
                for (idx, rel) in rel_paths.iter().enumerate() {
                    if r_ref.try_push(idx).is_err() {
                        let text = read_text(repo_root, rel, preloaded).unwrap_or_default();
                        let res = self.score_content(rel, &text);
                        o_ref.lock().push((rel.clone(), res));
                    }
                }
                d_ref.store(true, Ordering::Release);
                let _ = worker.join();
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
    use crate::testutil::TempRepo;
    use std::collections::HashSet;

    fn nest_src(head: &str, step: &str, open: &[String], body: &str, close: Option<&str>) -> String {
        let mut out = format!("{head}\n");
        for (i, line) in open.iter().enumerate() {
            out.push_str(&format!("{}{line}\n", step.repeat(i + 1)));
        }
        out.push_str(&format!("{}{body}\n", step.repeat(open.len() + 1)));
        if let Some(ch) = close {
            for i in (0..open.len()).rev() {
                out.push_str(&format!("{}{ch}\n", step.repeat(i + 1)));
            }
            out.push_str("  return 0;\n}\n");
        }
        out
    }

    fn repeat_fn(
        head: &str,
        range: std::ops::RangeInclusive<usize>,
        mut line: impl FnMut(usize) -> String,
        tail: &str,
    ) -> String {
        let mut out = format!("{head}\n");
        for i in range {
            out.push_str(&line(i));
        }
        out.push_str(tail);
        out
    }

    fn map_conds(vars: &[&str], fmt: impl Fn(&str) -> String) -> Vec<String> {
        vars.iter().map(|v| fmt(v)).collect()
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
}
