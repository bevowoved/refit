use crate::mvcc::MvccStore;
use crate::platform::{read_zstd, write_zstd};
use crate::types::{HygieneResult, Squiggle};
use dashmap::DashMap;
use flate2::read::DeflateDecoder;
use parking_lot::{Condvar, Mutex};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Embedded 8 KB ternary weights from `brianbell-x/codehygiene v0.7.0`.
static WEIGHTS_NPZ: &[u8] = include_bytes!("../vendor/codehygiene/weights.npz");
static WEIGHTS_CACHE: OnceLock<HygieneWeights> = OnceLock::new();
static NEURAL_PERMITS: OnceLock<(Mutex<usize>, Condvar)> = OnceLock::new();

struct ScorePermit;
impl ScorePermit {
    fn acquire() -> Self {
        let (lock, cvar) = NEURAL_PERMITS.get_or_init(|| (Mutex::new(4), Condvar::new()));
        let mut avail = lock.lock();
        while *avail == 0 {
            cvar.wait(&mut avail);
        }
        *avail -= 1;
        Self
    }
}
impl Drop for ScorePermit {
    fn drop(&mut self) {
        if let Some((lock, cvar)) = NEURAL_PERMITS.get() {
            let mut avail = lock.lock();
            *avail += 1;
            cvar.notify_one();
        }
    }
}

const MAX_COLS: usize = 128;
const MAX_REL_DIST: isize = 32;
const SKETCH_DIM: usize = 40;
const PRIOR_LINES: f64 = 16.0;
const PEAK_WINDOW: usize = 32;
const PEAK_MIN_GATE: f32 = 10.0;
const PEAK_PRIOR: f32 = 6.0;
const SPAN_CONT_WEIGHT: f32 = 0.25;
const RIBBON_GUARD: usize = 6;
const POW3: [u8; 5] = [1, 3, 9, 27, 81];
const SQUIGGLE_THRESHOLD: f32 = 0.55;

const SQUIGGLE_SPEC: [(&str, &str, f64); 6] = [
    (
        "deep_nesting",
        "Deep nesting pyramid; flatten with guard clauses or helpers",
        0.42,
    ),
    (
        "dense_line",
        "Wide line, packed statements, call train, or nested ternary",
        0.10,
    ),
    (
        "cramped_layout",
        "Unbroken statement wall or ragged off-grid indentation",
        0.08,
    ),
    (
        "duplicate_block",
        "Duplicated or structurally cloned multi-line block",
        0.58,
    ),
    (
        "repetitive_structure",
        "Repetitive parallel equality or branch ladder",
        0.34,
    ),
    (
        "comment_echo",
        "Comment echoes adjacent code or leaves commented-out code",
        0.18,
    ),
];

#[derive(Debug, Clone, Copy)]
pub struct StyleConfig {
    pub max_line_width: usize,
    pub indent_width: usize,
}

/// Read repository formatter configuration.
pub fn repo_style(repo_root: &Path) -> StyleConfig {
    let mut cfg = StyleConfig {
        max_line_width: 88,
        indent_width: 4,
    };
    if let Ok(py) = std::fs::read_to_string(repo_root.join("pyproject.toml")) {
        let mut found = false;
        for line in py.lines() {
            let t = line.split('#').next().unwrap_or("").trim();
            if let Some(rest) = t.strip_prefix("line-length") {
                if let Some((_, val)) = rest.split_once('=') {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        cfg.max_line_width = n.clamp(60, 160);
                        found = true;
                    }
                }
            } else if let Some(rest) = t.strip_prefix("indent-width") {
                if let Some((_, val)) = rest.split_once('=') {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        cfg.indent_width = n.clamp(2, 8);
                        found = true;
                    }
                }
            }
        }
        if found {
            return cfg;
        }
    }
    if let Ok(rf) = std::fs::read_to_string(repo_root.join("rustfmt.toml")) {
        let mut found = false;
        for line in rf.lines() {
            let t = line.split('#').next().unwrap_or("").trim();
            if let Some(rest) = t.strip_prefix("max_width") {
                if let Some((_, val)) = rest.split_once('=') {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        cfg.max_line_width = n.clamp(60, 160);
                        found = true;
                    }
                }
            } else if let Some(rest) = t.strip_prefix("tab_spaces") {
                if let Some((_, val)) = rest.split_once('=') {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        cfg.indent_width = n.clamp(2, 8);
                        found = true;
                    }
                }
            }
        }
        if found {
            return cfg;
        }
    }
    for f in [".prettierrc", ".prettierrc.json", "biome.json", "biome.jsonc"] {
        if let Ok(raw) = std::fs::read_to_string(repo_root.join(f)) {
            let cleaned = strip_jsonc(&raw);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned) {
                cfg.max_line_width = 100;
                cfg.indent_width = 2;
                if let Some(w) = v["printWidth"]
                    .as_u64()
                    .or_else(|| v["formatter"]["lineWidth"].as_u64())
                {
                    cfg.max_line_width = (w as usize).clamp(60, 160);
                }
                if let Some(ind) = v["tabWidth"]
                    .as_u64()
                    .or_else(|| v["formatter"]["indentWidth"].as_u64())
                {
                    cfg.indent_width = (ind as usize).clamp(2, 8);
                }
                return cfg;
            }
        }
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

fn strip_jsonc(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    let mut in_str = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            out.push(b as char);
            if b == b'\\' && i + 1 < bytes.len() {
                i += 1;
                out.push(bytes[i] as char);
            } else if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_str = true;
            out.push('"');
            i += 1;
        } else if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    out
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
        Self {
            cache: DashMap::new(),
            in_flight: DashMap::new(),
            disk_path: None,
            dirty: AtomicBool::new(false),
            last_warm_us: AtomicU64::new(0),
        }
    }

    pub fn open_persistent(repo_root: &Path) -> Self {
        let disk_path = repo_root.join(".refit/cache/hygiene_v07_cache.zst");
        let cache = DashMap::new();
        if let Some(saved) = read_zstd::<HashMap<String, HygieneResult>>(&disk_path) {
            for (k, v) in saved {
                cache.insert(k, v);
            }
        }
        Self {
            cache,
            in_flight: DashMap::new(),
            disk_path: Some(disk_path),
            dirty: AtomicBool::new(false),
            last_warm_us: AtomicU64::new(0),
        }
    }

    pub fn warm_us(&self) -> u64 {
        self.last_warm_us.load(Ordering::Relaxed)
    }

    pub fn flush_cache(&self) {
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        let Some(ref p) = self.disk_path else {
            return;
        };
        let mut map = HashMap::with_capacity(self.cache.len());
        for entry in self.cache.iter() {
            map.insert(entry.key().clone(), entry.value().clone());
        }
        let _ = write_zstd(p, &map);
    }

    pub fn score_content(&self, _path: &str, content: &str) -> HygieneResult {
        let key = format!("v07:{}", MvccStore::hash_bytes(content.as_bytes()));
        let t0 = Instant::now();
        if let Some(hit) = self.cache.get(&key) {
            let mut res = hit.clone();
            let us = t0.elapsed().as_micros().max(1) as u64;
            res.cached = true;
            res.latency_us = us;
            self.last_warm_us.store(us, Ordering::Relaxed);
            return res;
        }

        // Per-hash single-flight coalescing prevents duplicate parallel neural passes on the same file.
        let lock = self
            .in_flight
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = lock.lock();
        if let Some(hit) = self.cache.get(&key) {
            let mut res = hit.clone();
            let us = t0.elapsed().as_micros().max(1) as u64;
            res.cached = true;
            res.latency_us = us;
            self.last_warm_us.store(us, Ordering::Relaxed);
            drop(_guard);
            self.in_flight.remove_if(&key, |_, v| Arc::ptr_eq(v, &lock));
            return res;
        }

        let _permit = ScorePermit::acquire();
        let started = Instant::now();
        let mut res = score_v07(content);
        res.latency_us = started.elapsed().as_micros().max(1) as u64;
        self.cache.insert(key.clone(), res.clone());
        self.dirty.store(true, Ordering::Release);
        drop(_permit);
        drop(_guard);
        self.in_flight.remove_if(&key, |_, v| Arc::ptr_eq(v, &lock));
        res
    }

    pub fn score_files(
        &self,
        repo_root: &Path,
        rel_paths: &[String],
        preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    ) -> (f64, f64, HashMap<String, HygieneResult>) {
        let results: Vec<(String, HygieneResult)> = rel_paths
            .par_iter()
            .map(|rel| {
                let res = if let Some(bytes) = preloaded.and_then(|m| m.get(rel)) {
                    let content = String::from_utf8_lossy(bytes);
                    self.score_content(rel, &content)
                } else {
                    let abs = repo_root.join(rel);
                    let content = std::fs::read_to_string(&abs).unwrap_or_default();
                    self.score_content(rel, &content)
                };
                (rel.clone(), res)
            })
            .collect();

        let mut per_file = HashMap::with_capacity(results.len());
        let mut scores = Vec::with_capacity(results.len());
        for (rel, res) in results {
            scores.push(res.score);
            per_file.insert(rel, res);
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if scores.is_empty() {
            return (1.0, 1.0, per_file);
        }
        let median = scores[scores.len() / 2];
        let p10_idx = ((scores.len() as f64) * 0.10).floor() as usize;
        let p10 = scores[p10_idx.min(scores.len() - 1)];
        self.flush_cache();
        (median, p10, per_file)
    }
}

// ============================================================================
// Exact Pure-Rust Port of codehygiene v0.7.0 Sparse Ternary Geometric Scorer
// ============================================================================

struct Sparse2D {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f32>,
    active_rows: Vec<usize>,
    non_zero_count: usize,
}

struct SparseConv {
    k_size: usize,
    rows: usize,
    cols: usize,
    taps: Vec<Sparse2D>,
    non_zero_count: usize,
}

struct HygieneWeights {
    mats: HashMap<String, Sparse2D>,
    convs: HashMap<String, SparseConv>,
    vecs: HashMap<String, Vec<f32>>,
}

impl HygieneWeights {
    fn mat(&self, k: &str) -> &Sparse2D {
        self.mats.get(k).unwrap_or_else(|| panic!("missing mat `{k}`"))
    }
    fn conv(&self, k: &str) -> &SparseConv {
        self.convs.get(k).unwrap_or_else(|| panic!("missing conv `{k}`"))
    }
    fn vec(&self, k: &str) -> &[f32] {
        self.vecs.get(k).unwrap_or_else(|| panic!("missing vec `{k}`"))
    }
}

fn v07_weights() -> &'static HygieneWeights {
    WEIGHTS_CACHE.get_or_init(|| unpack_npz(WEIGHTS_NPZ))
}

fn le_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

enum NpyData {
    U8(Vec<u8>),
    F32(Vec<f32>),
    Str(String),
}

fn parse_npy(bytes: &[u8]) -> NpyData {
    let major = bytes[6];
    let header_len = if major == 1 {
        le_u16(bytes, 8) as usize
    } else {
        le_u32(bytes, 8) as usize
    };
    let data_offset = (if major == 1 { 10 } else { 12 }) + header_len;
    let header_text = String::from_utf8_lossy(&bytes[8..data_offset]);
    let descr = header_text
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .unwrap_or("");
    let body = &bytes[data_offset..];

    if descr.ends_with("u1") {
        NpyData::U8(body.to_vec())
    } else if descr == "<f4" {
        let vals = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        NpyData::F32(vals)
    } else if descr.starts_with("<U") {
        let s: String = body
            .chunks_exact(4)
            .filter_map(|c| char::from_u32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .filter(|&c| c != '\0')
            .collect();
        NpyData::Str(s)
    } else {
        panic!("unsupported npy descr `{descr}`");
    }
}

fn unpack_npz(bytes: &[u8]) -> HygieneWeights {
    let min_off = bytes.len().saturating_sub(65557);
    let mut eocd = None;
    for pos in (min_off..=bytes.len().saturating_sub(22)).rev() {
        if le_u32(bytes, pos) == 0x06054b50 {
            eocd = Some(pos);
            break;
        }
    }
    let eocd_pos = eocd.expect("valid weights.npz EOCD");
    let entry_count = le_u16(bytes, eocd_pos + 10) as usize;
    let mut dir_pos = le_u32(bytes, eocd_pos + 16) as usize;

    let mut trits = Vec::new();
    let mut vec_buf = Vec::new();
    let mut meta_str = String::new();

    for _ in 0..entry_count {
        let method = le_u16(bytes, dir_pos + 10);
        let comp_size = le_u32(bytes, dir_pos + 20) as usize;
        let name_len = le_u16(bytes, dir_pos + 28) as usize;
        let extra_len = le_u16(bytes, dir_pos + 30) as usize;
        let comment_len = le_u16(bytes, dir_pos + 32) as usize;
        let local_pos = le_u32(bytes, dir_pos + 42) as usize;
        let entry_name =
            String::from_utf8_lossy(&bytes[dir_pos + 46..dir_pos + 46 + name_len]).to_string();

        let local_name_len = le_u16(bytes, local_pos + 26) as usize;
        let local_extra_len = le_u16(bytes, local_pos + 28) as usize;
        let data_start = local_pos + 30 + local_name_len + local_extra_len;
        let raw_slice = &bytes[data_start..data_start + comp_size];

        let unzipped = if method == 8 {
            let mut dec = DeflateDecoder::new(raw_slice);
            let mut out = Vec::new();
            dec.read_to_end(&mut out).expect("inflate weights.npz entry");
            out
        } else {
            raw_slice.to_vec()
        };

        let key = entry_name.strip_suffix(".npy").unwrap_or(&entry_name);
        match (key, parse_npy(&unzipped)) {
            ("trits", NpyData::U8(v)) => trits = v,
            ("vecs", NpyData::F32(v)) => vec_buf = v,
            ("meta", NpyData::Str(s)) => meta_str = s,
            _ => {}
        }
        dir_pos += 46 + name_len + extra_len + comment_len;
    }

    unpack_weights(&trits, &vec_buf, &meta_str)
}

fn make_sparse2d(dense: Vec<f32>, rows: usize, cols: usize) -> Sparse2D {
    let mut row_offsets = vec![0usize; rows + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut active_rows = Vec::new();

    for row in 0..rows {
        row_offsets[row] = col_indices.len();
        for col in 0..cols {
            let val = dense[row * cols + col];
            if val != 0.0 {
                col_indices.push(col);
                values.push(val);
            }
        }
        if col_indices.len() > row_offsets[row] {
            active_rows.push(row);
        }
    }
    row_offsets[rows] = col_indices.len();
    let non_zero_count = col_indices.len();
    Sparse2D {
        data: dense,
        rows,
        cols,
        row_offsets,
        col_indices,
        values,
        active_rows,
        non_zero_count,
    }
}

fn make_conv(dense: &[f32], k_size: usize, in_ch: usize, out_ch: usize) -> SparseConv {
    let span = in_ch * out_ch;
    let mut taps = Vec::with_capacity(k_size);
    let mut nz = 0usize;
    for tap in 0..k_size {
        let s = make_sparse2d(dense[tap * span..(tap + 1) * span].to_vec(), in_ch, out_ch);
        nz += s.non_zero_count;
        taps.push(s);
    }
    SparseConv {
        k_size,
        rows: in_ch,
        cols: out_ch,
        taps,
        non_zero_count: nz,
    }
}

fn parse_meta(entry: &str) -> (String, Vec<usize>, usize) {
    let mut parts = entry.split(':');
    let name = parts.next().unwrap_or("").to_string();
    let dims: Vec<usize> = parts
        .next()
        .unwrap_or("")
        .split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect();
    let count = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
    (name, dims, count)
}

fn decode_trits(trit_buf: &[u8]) -> Vec<i8> {
    let mut flat = vec![0i8; trit_buf.len() * 5];
    for (byte_idx, &bv) in trit_buf.iter().enumerate() {
        for slot in 0..5 {
            let digit = (bv / POW3[slot]) % 3;
            flat[byte_idx * 5 + slot] = ((digit as i8 + 1) % 3) - 1;
        }
    }
    flat
}

fn find_scale(name: &str) -> String {
    if name == "col_sum_proj" {
        return "col_sum_s".to_string();
    }
    if let Some(stem) = name.strip_suffix("_w") {
        format!("{stem}_s")
    } else {
        format!("{name}_s")
    }
}

fn unpack_weights(trit_buf: &[u8], vec_buf: &[f32], meta_str: &str) -> HygieneWeights {
    let (trit_part, vec_part) = meta_str.split_once('|').expect("meta '|' separator");
    let mut raw_vecs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut vec_off = 0usize;
    for entry in vec_part.split(" ; ") {
        let (name, _, count) = parse_meta(entry);
        raw_vecs.insert(name, vec_buf[vec_off..vec_off + count].to_vec());
        vec_off += count;
    }

    let flat_trits = decode_trits(trit_buf);
    let mut mats = HashMap::new();
    let mut convs = HashMap::new();
    let mut byte_off = 0usize;

    for entry in trit_part.split(" ; ") {
        let (name, shape, packed_len) = parse_meta(entry);
        let num_elems: usize = shape.iter().product();
        let scale_key = find_scale(&name);
        let scale_vec = raw_vecs.get(&scale_key);
        let trit_start = byte_off * 5;
        let mut dense = vec![0.0f32; num_elems];
        for idx in 0..num_elems {
            let t = flat_trits[trit_start + idx];
            if t != 0 {
                dense[idx] = match scale_vec {
                    Some(sv) => (t as f32) * sv[idx % sv.len()],
                    None => t as f32,
                };
            }
        }
        byte_off += packed_len;
        if shape.len() == 3 {
            convs.insert(name, make_conv(&dense, shape[0], shape[1], shape[2]));
        } else {
            mats.insert(name, make_sparse2d(dense, shape[0], shape[1]));
        }
    }

    let mut vecs = HashMap::new();
    for (name, v) in raw_vecs {
        if !name.ends_with("_s") {
            vecs.insert(name, v);
        }
    }
    HygieneWeights { mats, convs, vecs }
}

fn expand_tabs(text: &str) -> String {
    if !text.contains('\t') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 32);
    let mut col_pos = 0usize;
    for ch in text.chars() {
        if ch == '\t' {
            let pad = 2 - (col_pos % 2);
            for _ in 0..pad {
                out.push(' ');
            }
            col_pos += pad;
        } else {
            out.push(ch);
            col_pos = if ch == '\n' || ch == '\r' { 0 } else { col_pos + 1 };
        }
    }
    out
}

fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let is_break = matches!(
            ch,
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        );
        if is_break {
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    parts.push(cur);
    if parts.len() > 1 && parts.last().map(|s| s.is_empty()).unwrap_or(false) {
        parts.pop();
    }
    parts
}

fn encode_grid(code: &str) -> (Vec<u8>, Vec<String>, usize) {
    let expanded = expand_tabs(code);
    let lines = split_lines(&expanded);
    let num_lines = lines.len();
    let mut grid = vec![0u8; num_lines * MAX_COLS];
    for (row, line) in lines.iter().enumerate() {
        for (col, ch) in line.chars().take(MAX_COLS).enumerate() {
            let cp = ch as u32;
            grid[row * MAX_COLS + col] = if cp > 127 {
                127
            } else if cp == 0 {
                32
            } else {
                cp as u8
            };
        }
    }
    (grid, lines, num_lines)
}

struct ActiveSpans {
    span_lens: Vec<usize>,
    line_starts: Vec<usize>,
    line_ids: Vec<usize>,
    within_col: Vec<usize>,
    cell_rows: Vec<usize>,
    total: usize,
}

fn active_spans(grid: &[u8], num_lines: usize) -> ActiveSpans {
    let mut nonblank = Vec::new();
    let mut span_lens = Vec::new();
    for row in 0..num_lines {
        let mut len = MAX_COLS;
        while len > 0 && grid[row * MAX_COLS + len - 1] == 0 {
            len -= 1;
        }
        if len > 0 {
            nonblank.push(row);
            span_lens.push((len + RIBBON_GUARD).min(MAX_COLS));
        }
    }
    let mut line_starts = vec![0usize; nonblank.len()];
    let mut total = 0usize;
    for (idx, &len) in span_lens.iter().enumerate() {
        line_starts[idx] = total;
        total += len;
    }
    let mut line_ids = vec![0usize; total];
    let mut within_col = vec![0usize; total];
    let mut cell_rows = vec![0usize; total];
    let mut line = 0usize;
    for cell in 0..total {
        while line + 1 < nonblank.len() && cell >= line_starts[line + 1] {
            line += 1;
        }
        line_ids[cell] = line;
        within_col[cell] = cell - line_starts[line];
        cell_rows[cell] = nonblank[line];
    }
    ActiveSpans {
        span_lens,
        line_starts,
        line_ids,
        within_col,
        cell_rows,
        total,
    }
}

fn ribbon_positions(spans: &ActiveSpans, guard: usize) -> (Vec<usize>, usize) {
    let mut starts = vec![0usize; spans.span_lens.len() + 1];
    starts[0] = guard;
    let mut running = 0usize;
    for (idx, &len) in spans.span_lens.iter().enumerate() {
        running += len + guard;
        starts[idx + 1] = running;
    }
    let mut pos = vec![0usize; spans.total];
    for cell in 0..spans.total {
        pos[cell] = starts[spans.line_ids[cell]] + spans.within_col[cell];
    }
    let length = starts[starts.len() - 1] + guard;
    (pos, length)
}

#[inline]
fn accumulate_row(src: &[f32], src_base: usize, weight: &Sparse2D, acc: &mut [f32]) {
    for &in_idx in &weight.active_rows {
        let in_val = src[src_base + in_idx];
        if in_val == 0.0 {
            continue;
        }
        let start = weight.row_offsets[in_idx];
        let end = weight.row_offsets[in_idx + 1];
        for ptr in start..end {
            acc[weight.col_indices[ptr]] += in_val * weight.values[ptr];
        }
    }
}

fn sparse_parts(
    parts: &[(&[f32], usize)],
    num_rows: usize,
    weight: &Sparse2D,
    bias: Option<&[f32]>,
    apply_relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_rows * weight.cols];
    let mut acc = vec![0.0f32; weight.cols];

    struct Slice<'a> {
        src: &'a [f32],
        dim: usize,
        in_off: usize,
        start: usize,
        end: usize,
    }
    let mut slices = Vec::with_capacity(parts.len());
    let mut in_off = 0usize;
    let mut cursor = 0usize;
    for &(src, dim) in parts {
        let next_off = in_off + dim;
        let start = cursor;
        while cursor < weight.active_rows.len() && weight.active_rows[cursor] < next_off {
            cursor += 1;
        }
        if cursor > start {
            slices.push(Slice {
                src,
                dim,
                in_off,
                start,
                end: cursor,
            });
        }
        in_off = next_off;
    }

    for row in 0..num_rows {
        if let Some(b) = bias {
            acc.copy_from_slice(b);
        } else {
            acc.fill(0.0);
        }
        for sl in &slices {
            let row_base = row * sl.dim;
            for &in_idx in &weight.active_rows[sl.start..sl.end] {
                let in_val = sl.src[row_base + (in_idx - sl.in_off)];
                if in_val == 0.0 {
                    continue;
                }
                let ro_s = weight.row_offsets[in_idx];
                let ro_e = weight.row_offsets[in_idx + 1];
                for ptr in ro_s..ro_e {
                    acc[weight.col_indices[ptr]] += in_val * weight.values[ptr];
                }
            }
        }
        if apply_relu {
            for v in &mut acc {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        out[row * weight.cols..(row + 1) * weight.cols].copy_from_slice(&acc);
    }
    out
}

fn linear_sparse(
    input: &[f32],
    num_rows: usize,
    weight: &Sparse2D,
    bias: Option<&[f32]>,
    apply_relu: bool,
) -> Vec<f32> {
    sparse_parts(&[(input, weight.rows)], num_rows, weight, bias, apply_relu)
}

fn line_cumsum(vals: &[f32], dim: usize, spans: &ActiveSpans) -> Vec<f32> {
    let mut csum = vec![0.0f32; vals.len()];
    if vals.len() >= dim {
        csum[..dim].copy_from_slice(&vals[..dim]);
        for idx in dim..vals.len() {
            csum[idx] = csum[idx - dim] + vals[idx];
        }
    }
    let mut out = vec![0.0f32; vals.len()];
    for cell in 0..spans.total {
        let start = spans.line_starts[spans.line_ids[cell]];
        for chan in 0..dim {
            let base = if start > 0 {
                csum[(start - 1) * dim + chan]
            } else {
                0.0
            };
            out[cell * dim + chan] = csum[cell * dim + chan] - base;
        }
    }
    out
}

fn line_prefix(
    parts: &[(&[f32], usize)],
    n_cells: usize,
    proj: &Sparse2D,
    spans: &ActiveSpans,
    exclusive: bool,
) -> Vec<f32> {
    let projected = sparse_parts(parts, n_cells, proj, None, false);
    let mut prefix = line_cumsum(&projected, proj.cols, spans);
    if exclusive {
        for (p, &v) in prefix.iter_mut().zip(projected.iter()) {
            *p -= v;
        }
    }
    prefix
}

fn fuse_cells(params: &HygieneWeights, grid: &[u8], spans: &ActiveSpans) -> Vec<f32> {
    let n_cells = spans.total;
    let tok_emb = params.mat("tok_emb");
    let col_emb = params.mat("col_emb");
    let emb_dim = tok_emb.cols;
    let mut emb = vec![0.0f32; n_cells * emb_dim];
    for cell in 0..n_cells {
        let col = spans.within_col[cell];
        let code = grid[spans.cell_rows[cell] * MAX_COLS + col] as usize;
        for chan in 0..emb_dim {
            emb[cell * emb_dim + chan] =
                tok_emb.data[code * emb_dim + chan] + col_emb.data[col * emb_dim + chan];
        }
    }

    let pre_w = params.conv("pre_conv3_w");
    let pre_b = params.vec("pre_conv3_b");
    let (rib_pos, rib_len) = ribbon_positions(spans, 1);
    let mut rib = vec![0.0f32; rib_len * emb_dim];
    for cell in 0..n_cells {
        let dst = rib_pos[cell] * emb_dim;
        rib[dst..dst + emb_dim].copy_from_slice(&emb[cell * emb_dim..(cell + 1) * emb_dim]);
    }
    let mut lex = vec![0.0f32; n_cells * pre_w.cols];
    let mut acc = vec![0.0f32; pre_w.cols];
    for cell in 0..n_cells {
        acc.copy_from_slice(pre_b);
        for tap in 0..pre_w.k_size {
            accumulate_row(&rib, (rib_pos[cell] - 1 + tap) * emb_dim, &pre_w.taps[tap], &mut acc);
        }
        for v in &mut acc {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        lex[cell * pre_w.cols..(cell + 1) * pre_w.cols].copy_from_slice(&acc);
    }

    let pref1 = line_prefix(
        &[(&emb, emb_dim), (&lex, pre_w.cols)],
        n_cells,
        params.mat("pref1_proj"),
        spans,
        true,
    );
    let gate1_w = params.mat("gate1_w");
    let gates = sparse_parts(
        &[
            (&emb, emb_dim),
            (&lex, pre_w.cols),
            (&pref1, params.mat("pref1_proj").cols),
        ],
        n_cells,
        gate1_w,
        Some(params.vec("gate1_b")),
        true,
    );
    let pref2 = line_prefix(
        &[(&gates, gate1_w.cols)],
        n_cells,
        params.mat("pref2_proj"),
        spans,
        false,
    );
    sparse_parts(
        &[
            (&emb, emb_dim),
            (&lex, pre_w.cols),
            (&gates, gate1_w.cols),
            (&pref2, params.mat("pref2_proj").cols),
        ],
        n_cells,
        params.mat("col_fuse_w"),
        Some(params.vec("col_fuse_b")),
        true,
    )
}

struct CompactCells {
    offsets: Vec<usize>,
    chans: Vec<usize>,
    vals: Vec<f32>,
}

fn ribbon_conv(
    rib_pos: &[usize],
    cell_at: &[i32],
    pad_fill: &[f32],
    sparse_cells: &CompactCells,
    spans: &ActiveSpans,
    conv_w: &SparseConv,
    bias: &[f32],
    num_lines: usize,
) -> Vec<f32> {
    let out_dim = conv_w.cols;
    let mut out = vec![0.0f32; num_lines * out_dim];
    if conv_w.non_zero_count == 0 {
        return out;
    }
    let mut pad_taps: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(conv_w.k_size);
    for tap in &conv_w.taps {
        let mut prod = vec![0.0f32; out_dim];
        accumulate_row(pad_fill, 0, tap, &mut prod);
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (c, &v) in prod.iter().enumerate() {
            if v != 0.0 {
                cols.push(c);
                vals.push(v);
            }
        }
        pad_taps.push((cols, vals));
    }

    let pad = (conv_w.k_size - 1) >> 1;
    let mut acc = vec![0.0f32; out_dim];
    for cell in 0..spans.total {
        acc.copy_from_slice(bias);
        let rpos = rib_pos[cell];
        for tap in 0..conv_w.k_size {
            let tap_mat = &conv_w.taps[tap];
            if tap_mat.non_zero_count == 0 {
                continue;
            }
            let src_cell = cell_at[rpos - pad + tap];
            if src_cell >= 0 {
                let sc = src_cell as usize;
                for idx in sparse_cells.offsets[sc]..sparse_cells.offsets[sc + 1] {
                    let ch = sparse_cells.chans[idx];
                    let iv = sparse_cells.vals[idx];
                    for ptr in tap_mat.row_offsets[ch]..tap_mat.row_offsets[ch + 1] {
                        acc[tap_mat.col_indices[ptr]] += iv * tap_mat.values[ptr];
                    }
                }
            } else {
                let (ref cols, ref vals) = pad_taps[tap];
                for (&c, &v) in cols.iter().zip(vals.iter()) {
                    acc[c] += v;
                }
            }
        }
        let dst_base = spans.cell_rows[cell] * out_dim;
        for chan in 0..out_dim {
            if acc[chan] > 0.0 {
                out[dst_base + chan] += acc[chan];
            }
        }
    }
    out
}

fn excl_prefix(deltas: &[f32], dim: usize) -> Vec<f32> {
    let mut running = vec![0.0f32; deltas.len()];
    let mut out = vec![0.0f32; deltas.len()];
    for idx in 0..deltas.len() {
        running[idx] = if idx >= dim { running[idx - dim] } else { 0.0 } + deltas[idx];
        out[idx] = running[idx] - deltas[idx];
    }
    out
}

fn axis_conv(deltas: &[f32], num_lines: usize, conv_w: &SparseConv, bias: &[f32]) -> Vec<f32> {
    let in_dim = conv_w.rows;
    let pad = ((conv_w.k_size - 1) >> 1) as isize;
    let mut out = vec![0.0f32; num_lines * conv_w.cols];
    let mut acc = vec![0.0f32; conv_w.cols];
    for row in 0..num_lines {
        acc.copy_from_slice(bias);
        let lo = (pad - row as isize).max(0) as usize;
        let hi = (num_lines as isize - row as isize + pad).min(conv_w.k_size as isize) as usize;
        for tap in lo..hi {
            let src_row = (row as isize + tap as isize - pad) as usize;
            accumulate_row(deltas, src_row * in_dim, &conv_w.taps[tap], &mut acc);
        }
        for v in &mut acc {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        out[row * conv_w.cols..(row + 1) * conv_w.cols].copy_from_slice(&acc);
    }
    out
}

fn norm_sketch(sketch: &mut [f32], num_lines: usize, total_sketch: usize) {
    for v in sketch.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    let total = num_lines * total_sketch;
    let mut base = 0usize;
    while base < total {
        let end = (base + SKETCH_DIM).min(total);
        let mut norm_sq = 0.0f32;
        for &v in &sketch[base..end] {
            norm_sq += v * v;
        }
        let norm = (norm_sq + 1e-6f32).sqrt();
        for v in &mut sketch[base..end] {
            *v /= norm;
        }
        base += SKETCH_DIM;
    }
}

fn sink_attention(
    state: &[f32],
    num_lines: usize,
    wq: Option<&Sparse2D>,
    wk: Option<&Sparse2D>,
    wv: &Sparse2D,
    wo: &Sparse2D,
    b_rel: &[f32],
    b_sink: &[f32],
) -> Vec<f32> {
    let num_heads = b_sink.len();
    let layer_has_query = wq.map(|m| m.non_zero_count > 0).unwrap_or(false)
        && wk.map(|m| m.non_zero_count > 0).unwrap_or(false);
    let qk_total = wq.map(|m| m.cols).unwrap_or(0);
    let head_dim = if qk_total > 0 { qk_total / num_heads } else { 1 };
    let val_total = wv.cols;
    let val_dim = val_total / num_heads;
    let inv_scale = 1.0f32 / (head_dim as f32).sqrt();

    let q_proj = if layer_has_query {
        Some(linear_sparse(state, num_lines, wq.unwrap(), None, false))
    } else {
        None
    };
    let k_proj = if layer_has_query {
        Some(linear_sparse(state, num_lines, wk.unwrap(), None, false))
    } else {
        None
    };
    let v_proj = linear_sparse(state, num_lines, wv, None, false);

    let mut merged = vec![0.0f32; num_lines * val_total];
    let mut logits = vec![0.0f64; num_lines];
    let mut keys = vec![0usize; num_lines];
    let rel_span = (2 * MAX_REL_DIST + 1) as usize;

    for head_idx in 0..num_heads {
        let rel_base = head_idx * rel_span;
        let qk_off = head_idx * head_dim;
        let val_off = head_idx * val_dim;
        let has_query = layer_has_query
            && wq
                .map(|m| m.col_indices.iter().any(|&c| c >= qk_off && c < qk_off + head_dim))
                .unwrap_or(false);
        let banded = b_rel[rel_base] <= -1e3 && b_rel[rel_base + rel_span - 1] <= -1e3;
        let uniform = !banded
            && !has_query
            && b_rel[rel_base..rel_base + rel_span].iter().all(|&v| v == 0.0);
        let sink = b_sink[head_idx] as f64;

        if uniform {
            let weight = (1.0f64 / (num_lines as f64 + sink.exp())) as f32;
            let mut sum_vec = vec![0.0f64; val_dim];
            for row in 0..num_lines {
                let base = row * val_total + val_off;
                for d in 0..val_dim {
                    sum_vec[d] += v_proj[base + d] as f64;
                }
            }
            for row in 0..num_lines {
                let base = row * val_total + val_off;
                for d in 0..val_dim {
                    merged[base + d] = (sum_vec[d] * weight as f64) as f32;
                }
            }
            continue;
        }

        for q_line in 0..num_lines {
            let k_start = if banded {
                (q_line as isize - MAX_REL_DIST + 1).max(0) as usize
            } else {
                0
            };
            let k_end = if banded {
                (q_line as isize + MAX_REL_DIST).min(num_lines as isize) as usize
            } else {
                num_lines
            };
            let mut count = 0usize;
            let mut max_logit = sink;
            for k_line in k_start..k_end {
                let rel = (k_line as isize - q_line as isize).clamp(-MAX_REL_DIST, MAX_REL_DIST);
                let rel_bias = b_rel[rel_base + (rel + MAX_REL_DIST) as usize] as f64;
                if banded && rel_bias <= -1e3 {
                    continue;
                }
                let qk = if has_query {
                    let qp = q_proj.as_ref().unwrap();
                    let kp = k_proj.as_ref().unwrap();
                    let qb = q_line * qk_total + qk_off;
                    let kb = k_line * qk_total + qk_off;
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += (qp[qb + d] as f64) * (kp[kb + d] as f64);
                    }
                    dot * (inv_scale as f64)
                } else {
                    0.0
                };
                let logit = rel_bias + qk;
                keys[count] = k_line;
                logits[count] = logit;
                if logit > max_logit {
                    max_logit = logit;
                }
                count += 1;
            }
            let mut exp_sum = (sink - max_logit).exp();
            for idx in 0..count {
                logits[idx] = (logits[idx] - max_logit).exp();
                exp_sum += logits[idx];
            }
            let dst_base = q_line * val_total + val_off;
            for idx in 0..count {
                let aw = logits[idx] / exp_sum;
                let v_base = keys[idx] * val_total + val_off;
                for d in 0..val_dim {
                    merged[dst_base + d] += (aw * (v_proj[v_base + d] as f64)) as f32;
                }
            }
        }
    }
    linear_sparse(&merged, num_lines, wo, None, false)
}

#[inline]
fn sigmoid_f64(val: f64) -> f64 {
    1.0 / (1.0 + (-val.clamp(-30.0, 30.0)).exp())
}

fn pool_doc(
    has_ml: bool,
    state: &[f32],
    active_gate: &[f32],
    num_lines: usize,
    state_dim: usize,
) -> Vec<f32> {
    let mut doc_vec = vec![0.0f32; state_dim * 2];
    let mut mean_acc = vec![0.0f64; state_dim];
    let mut gate_sum = if has_ml { PRIOR_LINES } else { 1e-5 };
    for row in 0..num_lines {
        let g = active_gate[row] as f64;
        gate_sum += g;
        let base = row * state_dim;
        for ch in 0..state_dim {
            let s = state[base + ch];
            mean_acc[ch] += (s as f64) * g;
            if !has_ml {
                let pv = s.max(0.0) * active_gate[row];
                if pv > doc_vec[state_dim + ch] {
                    doc_vec[state_dim + ch] = pv;
                }
            }
        }
    }
    for ch in 0..state_dim {
        doc_vec[ch] = (mean_acc[ch] / gate_sum) as f32;
    }

    if has_ml {
        let mut pos = vec![0.0f32; num_lines * state_dim];
        for row in 0..num_lines {
            for ch in 0..state_dim {
                let raw = state[row * state_dim + ch].max(0.0) * active_gate[row];
                let prev1 = if row >= 1 {
                    state[(row - 1) * state_dim + ch].max(0.0) * active_gate[row - 1]
                } else {
                    0.0
                };
                let prev2 = if row >= 2 {
                    state[(row - 2) * state_dim + ch].max(0.0) * active_gate[row - 2]
                } else {
                    0.0
                };
                let overlap = if num_lines > 1 {
                    raw.min(prev1.max(prev2))
                } else {
                    0.0
                };
                pos[row * state_dim + ch] = raw - (1.0 - SPAN_CONT_WEIGHT) * overlap;
            }
        }
        let mut prefix_w = vec![0.0f32; (num_lines + 1) * state_dim];
        let mut prefix_g = vec![0.0f32; num_lines + 1];
        for row in 0..num_lines {
            prefix_g[row + 1] = prefix_g[row] + active_gate[row];
            for ch in 0..state_dim {
                prefix_w[(row + 1) * state_dim + ch] =
                    prefix_w[row * state_dim + ch] + pos[row * state_dim + ch];
            }
        }
        let mut peak = vec![f32::NEG_INFINITY; state_dim];
        for hi in 1..=num_lines {
            let lo = hi.saturating_sub(PEAK_WINDOW);
            let denom = (prefix_g[hi] - prefix_g[lo]).max(PEAK_MIN_GATE) + PEAK_PRIOR;
            for ch in 0..state_dim {
                let span = prefix_w[hi * state_dim + ch] - prefix_w[lo * state_dim + ch];
                let val = span / denom;
                if val > peak[ch] {
                    peak[ch] = val;
                }
            }
        }
        doc_vec[state_dim..].copy_from_slice(&peak);
    }
    doc_vec
}

fn score_v07(content: &str) -> HygieneResult {
    let params = v07_weights();
    let (grid, lines, num_lines) = encode_grid(content);
    let spans = active_spans(&grid, num_lines);

    let col_sum_cols = params.mat("col_sum_proj").cols;
    let conv3_cols = params.conv("conv3_w").cols;
    let conv7_cols = params.conv("conv7_w").cols;
    let sketch_cols = params.conv("sketch_w").cols;

    let (col_sums, conv3, conv7, mut sketch) = if spans.total == 0 {
        (
            vec![0.0f32; num_lines * col_sum_cols],
            vec![0.0f32; num_lines * conv3_cols],
            vec![0.0f32; num_lines * conv7_cols],
            vec![0.0f32; num_lines * sketch_cols],
        )
    } else {
        let cells = fuse_cells(params, &grid, &spans);
        let hc_dim = params.mat("col_fuse_w").cols;
        let mut cell_sums = vec![0.0f32; num_lines * hc_dim];
        for cell in 0..spans.total {
            let dst = spans.cell_rows[cell] * hc_dim;
            let src = cell * hc_dim;
            for ch in 0..hc_dim {
                cell_sums[dst + ch] += cells[src + ch];
            }
        }
        let col_sums = linear_sparse(&cell_sums, num_lines, params.mat("col_sum_proj"), None, false);

        let (rib_pos, rib_len) = ribbon_positions(&spans, RIBBON_GUARD);
        let mut cell_at = vec![-1i32; rib_len];
        for (cell, &rp) in rib_pos.iter().enumerate() {
            cell_at[rp] = cell as i32;
        }
        let pad_fill: Vec<f32> = params
            .vec("col_fuse_b")
            .iter()
            .map(|&v| if v > 0.0 { v } else { 0.0 })
            .collect();
        let mut offsets = vec![0usize; spans.total + 1];
        let mut chans = Vec::new();
        let mut vals = Vec::new();
        for cell in 0..spans.total {
            offsets[cell] = chans.len();
            let base = cell * hc_dim;
            for ch in 0..hc_dim {
                let v = cells[base + ch];
                if v != 0.0 {
                    chans.push(ch);
                    vals.push(v);
                }
            }
        }
        offsets[spans.total] = chans.len();
        let sc = CompactCells {
            offsets,
            chans,
            vals,
        };
        let c3 = ribbon_conv(
            &rib_pos,
            &cell_at,
            &pad_fill,
            &sc,
            &spans,
            params.conv("conv3_w"),
            params.vec("conv3_b"),
            num_lines,
        );
        let c7 = ribbon_conv(
            &rib_pos,
            &cell_at,
            &pad_fill,
            &sc,
            &spans,
            params.conv("conv7_w"),
            params.vec("conv7_b"),
            num_lines,
        );
        let sk = ribbon_conv(
            &rib_pos,
            &cell_at,
            &pad_fill,
            &sc,
            &spans,
            params.conv("sketch_w"),
            params.vec("sketch_b"),
            num_lines,
        );
        (col_sums, c3, c7, sk)
    };

    let has_ml = params.mats.contains_key("ml_proj");
    let line_hidden = if has_ml {
        let ml_proj = params.mat("ml_proj");
        let deltas = sparse_parts(
            &[
                (&col_sums, col_sum_cols),
                (&conv3, conv3_cols),
                (&conv7, conv7_cols),
            ],
            num_lines,
            ml_proj,
            None,
            false,
        );
        let ml_saw = linear_sparse(
            &excl_prefix(&deltas, ml_proj.cols),
            num_lines,
            params.mat("ml_gate_w"),
            Some(params.vec("ml_gate_b")),
            true,
        );
        let paren_suf = axis_conv(
            &deltas,
            num_lines,
            params.conv("ml_paren_w"),
            params.vec("ml_paren_b"),
        );
        sparse_parts(
            &[
                (&col_sums, col_sum_cols),
                (&conv3, conv3_cols),
                (&conv7, conv7_cols),
                (&ml_saw, params.mat("ml_gate_w").cols),
                (&paren_suf, params.conv("ml_paren_w").cols),
            ],
            num_lines,
            params.mat("lmlp1_w"),
            Some(params.vec("lmlp1_b")),
            true,
        )
    } else {
        sparse_parts(
            &[
                (&col_sums, col_sum_cols),
                (&conv3, conv3_cols),
                (&conv7, conv7_cols),
            ],
            num_lines,
            params.mat("lmlp1_w"),
            Some(params.vec("lmlp1_b")),
            true,
        )
    };

    let lmlp2_w = params.mat("lmlp2_w");
    let line_stats = linear_sparse(
        &line_hidden,
        num_lines,
        lmlp2_w,
        Some(params.vec("lmlp2_b")),
        false,
    );
    norm_sketch(&mut sketch, num_lines, sketch_cols);

    let state_dim = lmlp2_w.cols + sketch_cols;
    let mut state = vec![0.0f32; num_lines * state_dim];
    for row in 0..num_lines {
        let dst = row * state_dim;
        state[dst..dst + lmlp2_w.cols]
            .copy_from_slice(&line_stats[row * lmlp2_w.cols..(row + 1) * lmlp2_w.cols]);
        state[dst + lmlp2_w.cols..dst + state_dim]
            .copy_from_slice(&sketch[row * sketch_cols..(row + 1) * sketch_cols]);
    }

    for layer in ["1", "2"] {
        let attn = sink_attention(
            &state,
            num_lines,
            params.mats.get(&format!("attn{layer}_wq")),
            params.mats.get(&format!("attn{layer}_wk")),
            params.mat(&format!("attn{layer}_wv")),
            params.mat(&format!("attn{layer}_wo")),
            params.vec(&format!("attn{layer}_brel")),
            params.vec(&format!("attn{layer}_bsink")),
        );
        for (s, a) in state.iter_mut().zip(attn.iter()) {
            *s += *a;
        }
        let ctx_h = linear_sparse(
            &state,
            num_lines,
            params.mat(&format!("cmlp{layer}_w")),
            Some(params.vec(&format!("cmlp{layer}_b"))),
            true,
        );
        let ctx_o = linear_sparse(
            &ctx_h,
            num_lines,
            params.mat(&format!("cmlp{layer}_2_w")),
            Some(params.vec(&format!("cmlp{layer}_2_b"))),
            false,
        );
        for (s, c) in state.iter_mut().zip(ctx_o.iter()) {
            *s += *c;
        }
    }

    let sq_logits = linear_sparse(
        &state,
        num_lines,
        params.mat("sq_w"),
        Some(params.vec("sq_b")),
        false,
    );
    let gate_logits = linear_sparse(
        &state,
        num_lines,
        params.mat("pool_gate_w"),
        Some(params.vec("pool_gate_b")),
        false,
    );
    let active_gate: Vec<f32> = gate_logits
        .iter()
        .map(|&v| sigmoid_f64(v as f64) as f32)
        .collect();

    let doc_vec = pool_doc(has_ml, &state, &active_gate, num_lines, state_dim);
    let doc_hidden = linear_sparse(
        &doc_vec,
        1,
        params.mat("doc1_w"),
        Some(params.vec("doc1_b")),
        true,
    );
    let doc_logits = linear_sparse(
        &doc_hidden,
        1,
        params.mat("doc2_w"),
        Some(params.vec("doc2_b")),
        false,
    );
    let score = (sigmoid_f64(doc_logits[0] as f64) * 10000.0).round() / 10000.0;

    // Collect all squiggles ranked by severity * confidence so deep_nesting never starves other smells.
    let mut ranked: Vec<(f64, Squiggle)> = Vec::new();
    let n_kinds = SQUIGGLE_SPEC.len();
    for row in 0..lines.len() {
        let base = row * n_kinds;
        let mut best_idx = 0usize;
        let mut best_p = sigmoid_f64(sq_logits[base] as f64) as f32;
        for k in 1..n_kinds {
            let p = sigmoid_f64(sq_logits[base + k] as f64) as f32;
            if p > best_p {
                best_p = p;
                best_idx = k;
            }
        }
        if best_p < SQUIGGLE_THRESHOLD {
            continue;
        }
        let (kind, msg, sev) = SQUIGGLE_SPEC[best_idx];
        let prob = ((best_p as f64) * 10000.0).round() / 10000.0;
        ranked.push((
            sev * prob,
            Squiggle {
                line: row + 1,
                kind: kind.to_string(),
                probability: prob,
                message: msg.to_string(),
            },
        ));
    }
    // Reserve up to 4 top squiggles per present smell family so no single smell starves others,
    // then fill up to 48 ordered by descending severity * confidence.
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.line.cmp(&b.1.line))
    });
    let mut per_kind_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut selected: Vec<(f64, Squiggle)> = Vec::with_capacity(48);
    let mut overflow: Vec<(f64, Squiggle)> = Vec::new();
    for item in ranked {
        let k_static = SQUIGGLE_SPEC
            .iter()
            .find(|(k, _, _)| *k == item.1.kind)
            .map(|(k, _, _)| *k)
            .unwrap_or("dense_line");
        let cnt = per_kind_counts.entry(k_static).or_insert(0);
        if *cnt < 4 {
            *cnt += 1;
            selected.push(item);
        } else {
            overflow.push(item);
        }
    }
    for item in overflow {
        if selected.len() >= 48 {
            break;
        }
        selected.push(item);
    }
    selected.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.line.cmp(&b.1.line))
    });
    selected.truncate(48);
    let squiggles: Vec<Squiggle> = selected.into_iter().map(|(_, s)| s).collect();

    HygieneResult {
        score,
        squiggles,
        cached: false,
        latency_us: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn probe_22_hygiene_cache() {
        let scorer = HygieneScorer::new();
        let clean = "def add(a: int, b: int) -> int:\n    return a + b\n";
        let messy = concat!(
            "def bad(x):\n",
            "    if x > 0:\n",
            "        if x > 1:\n",
            "            if x > 2:\n",
            "                if x > 3:\n",
            "                    if x > 4:\n",
            "                        return 'a' * 120 + 'extra_padding_to_exceed_one_hundred_columns_easily_here_yes'\n"
        );

        let r_clean_cold = scorer.score_content("clean.py", clean);
        let r_clean_warm = scorer.score_content("clean.py", clean);
        let r_messy = scorer.score_content("messy.py", messy);

        assert!(!r_clean_cold.cached);
        assert!(r_clean_warm.cached);
        assert!(r_clean_warm.latency_us <= 1000);
        assert!(
            r_clean_cold.score > r_messy.score + 0.05,
            "clean={} messy={}",
            r_clean_cold.score,
            r_messy.score
        );
        assert!(!r_messy.squiggles.is_empty());
    }

    #[test]
    fn probe_23_hygiene_smells() {
        let scorer = HygieneScorer::new();
        let nested = concat!(
            "def f(a, b, c, d, e):\n",
            "    if a:\n",
            "        if b:\n",
            "            if c:\n",
            "                if d:\n",
            "                    if e:\n",
            "                        return a + b + c + d + e\n"
        );
        let dense = "def d(x):\n    return (x + 1 if x > 0 else (x - 1 if x < -5 else (x * 2 if x == 0 else x * 3))) + len(str(x).strip().lower().replace('a', 'b'))\n";
        let mut dup = String::new();
        for n in 1..=2 {
            dup.push_str(&format!(
                "def run_pipeline_{n}(raw_input):\n\
                 \x20   step_one = normalize_payload(raw_input)\n\
                 \x20   step_two = validate_schema(step_one)\n\
                 \x20   step_three = enrich_metadata(step_two)\n\
                 \x20   step_four = compute_digest(step_three)\n\
                 \x20   step_five = serialize_record(step_four)\n\
                 \x20   step_six = write_storage(step_five)\n\
                 \x20   step_seven = emit_audit_log(step_six)\n\
                 \x20   return step_seven\n\n"
            ));
        }
        let mut rep = String::from("def pick(k):\n");
        for i in 1..=8 {
            rep.push_str(&format!("    if k == {i}:\n        return {i} * 10\n"));
        }
        let mut cramped = String::from("def cramped_fn(x):\n");
        for i in 1..=45 {
            cramped.push_str(&format!("    v_{i} = x + {i}\n"));
        }
        cramped.push_str("    return v_45\n");
        let mut echo = String::from("def calc(x, y, z):\n");
        for i in 0..15 {
            echo.push_str(&format!("    # step_{i} = compute_legacy_{i}(x, y, z)\n"));
        }
        echo.push_str("    return x\n");

        let mut kinds: HashSet<String> = HashSet::new();
        for src in [nested, dense, &dup, &rep, &cramped, &echo] {
            for q in scorer.score_content("t.py", src).squiggles {
                kinds.insert(q.kind);
            }
        }
        assert!(kinds.contains("deep_nesting"), "kinds={kinds:?}");
        assert!(kinds.contains("dense_line"), "kinds={kinds:?}");
        assert!(kinds.contains("duplicate_block"), "kinds={kinds:?}");
        assert!(kinds.contains("repetitive_structure"), "kinds={kinds:?}");
        assert!(kinds.contains("cramped_layout"), "kinds={kinds:?}");
        assert!(kinds.contains("comment_echo"), "kinds={kinds:?}");
    }

    #[test]
    fn probe_24_hygiene_bounds() {
        let scorer = HygieneScorer::new();
        let empty = scorer.score_content("e.py", "");
        let one = scorer.score_content("o.rs", "pub const X: usize = 1;\n");
        assert!((0.80..=0.99).contains(&empty.score), "empty={}", empty.score);
        assert!((0.80..=0.99).contains(&one.score), "one={}", one.score);
    }

    #[test]
    fn probe_69_hygiene_disk() {
        let dir = std::env::temp_dir().join(format!(
            "refit_hyg69_{}_{}",
            std::process::id(),
            crate::platform::now_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let src = "pub fn stable(x: i32) -> i32 {\n    x + 1\n}\n";
        let s1 = HygieneScorer::open_persistent(&dir);
        let r1 = s1.score_content("a.rs", src);
        assert!(!r1.cached);
        s1.flush_cache();

        let s2 = HygieneScorer::open_persistent(&dir);
        let r2 = s2.score_content("a.rs", src);
        assert!(r2.cached);
        assert!((r1.score - r2.score).abs() < 1e-9);
        assert!(s2.warm_us() >= 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_79_hygiene_v07() {
        let scorer = HygieneScorer::new();
        let ts_messy = concat!(
            "export function messy(a: number, b: number, c: number, d: number, e: number, f: number) {\n",
            "  if (a > 0) {\n",
            "    if (b > 0) {\n",
            "      if (c > 0) {\n",
            "        if (d > 0) {\n",
            "          if (e > 0) {\n",
            "            if (f > 0) {\n",
            "              return a ? (b ? (c ? d + e + f : a + b) : c) : (d ? e : f) + 1234567;\n",
            "            }\n",
            "          }\n",
            "        }\n",
            "      }\n",
            "    }\n",
            "  }\n",
            "  return 0;\n",
            "}\n"
        );
        let r_ts = scorer.score_content("messy.ts", ts_messy);
        // Real v0.7.0 weights catch 2-space nesting (~0.75) instead of old heuristic 0.950 false pass!
        assert!(r_ts.score < 0.85, "expected <0.85 on 6-level 2-space TS, got {}", r_ts.score);
        assert!(!r_ts.squiggles.is_empty());

        let clean_rust = "pub struct S;\nimpl S {\n    pub fn run(&self, x: Option<i32>) -> i32 {\n        match x {\n            Some(v) => v + 1,\n            None => 0,\n        }\n    }\n}\n";
        let r_rs = scorer.score_content("s.rs", clean_rust);
        assert!(r_rs.score > 0.70, "normal Rust impl+match must not collapse to 0.270 floor, got {}", r_rs.score);
    }
}
