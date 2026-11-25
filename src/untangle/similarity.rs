//! Unified code-similarity engine:
//! 1. Adaptive Type-1 & Type-2 clone detector:
//!    - `< 500` files: single-pass exact `FxHashMap` (zero `3.06 MiB` Bloom alloc)
//!    - `>= 500` files: bounded-RAM (`<= 3.06 MiB`) 2-pass Bloom+CM using `FixedBitSet`
//! 2. Zero-copy SimiFlow fuzzy helper & capability clustering (`&[u8]` slices, stack lowercase
//!    hashing, `n <= 128` direct pairwise SimHash, `n > 128` sparse LSH bands).

use crate::lex::{ident_byte, iter_idents, quoted_lit};
use crate::source::stem_of;
use crate::stem::analysis::{rapid_hash_u64, rapid_hash64};
use crate::types::{CloneCluster, RoleCluster};
use fixedbitset::FixedBitSet;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use strsim::jaro_winkler;

// ============================================================================
// 1. Adaptive Type-1 & Type-2 Clone Detector
// ============================================================================

const EXACT_CLONE_FILE_CAP: usize = 500;
const BLOOM_BITS: usize = 1 << 23; // 1.00 MiB L1 seen Bloom
const BLOOM_MASK: usize = BLOOM_BITS - 1;
const L2_BITS: usize = 1 << 22; // 0.50 MiB L2 duplicate Bloom
const L2_MASK: usize = L2_BITS - 1;
const CM_SLOTS: usize = 1 << 20; // 0.50 MiB 4-bit CM DF sketch
const CM_MASK: usize = CM_SLOTS - 1;
const FB_BUCKETS: usize = 8192; // 8192 x 16 u64 = 1.00 MiB File-Bloom
const FB_WORDS: usize = 16;
const SB_BITS: usize = 1 << 19; // 64 KiB Surviving-Dup Bloom
const SB_MASK: usize = SB_BITS - 1;
const DF_BOILER_CAP: u8 = 8;

const SEED_CM: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_FB: u64 = 0xbf58_476d_1ce4_e5b9;
const SEED_L2: u64 = 0x94d0_49bb_1331_11eb;

#[derive(Clone, Copy)]
struct LineFeat { norm_hash: u64, valid: bool }

fn build_log2_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (i, slot) in lut.iter_mut().enumerate().skip(1) {
        let f = i as f32;
        *slot = f * f.log2();
    }
    lut
}

#[inline]
fn is_clone_keyword(tok: &[u8]) -> bool {
    matches!(
        tok,
        b"const" | b"let" | b"var" | b"return" | b"if" | b"else" | b"elif" | b"for" | b"while"
            | b"fn" | b"pub" | b"async" | b"await" | b"match" | b"struct" | b"enum" | b"impl"
            | b"function" | b"export" | b"def" | b"class" | b"func" | b"this" | b"self"
    )
}

#[inline]
fn is_ignored_line_prefix(t: &[u8]) -> bool {
    if t.len() < 12 {
        return true;
    }
    let prefixes = [
        b"//".as_slice(), b"#", b"/*", b"* ", b"import ", b"from ", b"use ", b"package ",
        b"pub const ", b"pub(crate) const ", b"pub(super) const ",
    ];
    if prefixes.iter().any(|p| t.starts_with(p)) {
        return true;
    }
    let field_rest = t
        .strip_prefix(b"pub(crate) ")
        .or_else(|| t.strip_prefix(b"pub(super) "))
        .or_else(|| t.strip_prefix(b"pub "));
    field_rest.is_some_and(|r| {
        r.ends_with(b",") && r.contains(&b':') && !r.contains(&b'=') && !r.contains(&b'(')
    })
}

fn shannon_entropy(bytes: &[u8], lut: &[f32; 256]) -> f32 {
    let mut counts = [0u16; 128];
    for &b in bytes {
        counts[(b & 0x7f) as usize] += 1;
    }
    let sum_c: f32 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| lut[(c as usize).min(255)])
        .sum();
    let (n, nf) = (bytes.len(), bytes.len() as f32);
    let n_log_n = if n < 256 { lut[n] } else { nf * nf.log2() };
    (n_log_n - sum_c) / nf
}

struct AlphaRenamer {
    hash: u64, var_slot: u32, local_map: [(u64, u32); 16], local_len: usize,
    distinct_toks: u16, op_count: u16,
}

impl AlphaRenamer {
    fn new() -> Self {
        Self { hash: 0x99, var_slot: 0, local_map: [(0, 0); 16], local_len: 0, distinct_toks: 0, op_count: 0 }
    }

    fn intern_var(&mut self, tok_hash: u64) -> u32 {
        if let Some(&(_, v)) = self.local_map[..self.local_len].iter().find(|&&(k, _)| k == tok_hash) {
            return v;
        }
        let id = self.var_slot;
        self.var_slot += 1;
        self.distinct_toks += 1;
        if self.local_len < self.local_map.len() {
            self.local_map[self.local_len] = (tok_hash, id);
            self.local_len += 1;
        }
        id
    }

    fn feed_ident(&mut self, tok: &[u8]) {
        if is_clone_keyword(tok) {
            self.hash = rapid_hash_u64(rapid_hash64(tok, 0x11), self.hash);
            self.distinct_toks += 1;
            return;
        }
        let id = self.intern_var(rapid_hash64(tok, 0x77));
        self.hash = rapid_hash_u64(0x1000_0000 | (id as u64), self.hash);
    }

    fn feed_symbol(&mut self, byte: u8) {
        self.hash = rapid_hash_u64(0x3000_0000 | (byte as u64), self.hash);
        if matches!(byte, b'+' | b'-' | b'*' | b'/' | b'^' | b'|' | b'&' | b'=' | b'<' | b'>' | b'?' | b'(' | b'.') {
            self.op_count += 1;
        }
    }
}

fn scan_line_type2(raw: &[u8], lut: &[f32; 256]) -> LineFeat {
    let bytes = raw.trim_ascii();
    if is_ignored_line_prefix(bytes) || shannon_entropy(bytes, lut) < 2.75 {
        return LineFeat { norm_hash: 0, valid: false };
    }
    let (mut scanner, mut idx) = (AlphaRenamer::new(), 0usize);
    while idx < bytes.len() {
        let byte = bytes[idx];
        if byte.is_ascii_whitespace() {
            idx += 1;
        } else if matches!(byte, b'"' | b'\'' | b'`') {
            idx = quoted_lit(bytes, idx, byte != b'`', false).0;
            scanner.hash = rapid_hash_u64(0x2000_0001, scanner.hash);
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = idx;
            while idx < bytes.len() && ident_byte(bytes[idx]) { idx += 1; }
            scanner.feed_ident(&bytes[start..idx]);
        } else if byte.is_ascii_digit() {
            while idx < bytes.len() && ident_byte(bytes[idx]) { idx += 1; }
            scanner.hash = rapid_hash_u64(0x2000_0000, scanner.hash);
        } else {
            scanner.feed_symbol(byte);
            idx += 1;
        }
    }
    LineFeat { norm_hash: scanner.hash, valid: scanner.distinct_toks >= 4 && scanner.op_count >= 2 }
}

#[inline]
fn kgram_hash(feats: &[LineFeat], i: usize) -> u64 {
    feats[i..i + 4].iter().fold(0x1337u64, |h, f| rapid_hash_u64(f.norm_hash, h))
}

#[inline]
fn is_valid_kgram(f: &[LineFeat], i: usize) -> bool { f[i..i + 4].iter().all(|x| x.valid) }

#[inline]
fn mark_boiler_window(mask: &mut [bool], pos: usize) { mask[pos..pos + 4].fill(true); }

fn filter_boiler_kgrams(raw: &[(u64, u32)], mask: &[bool], out: &mut Vec<(u64, u32)>) {
    out.clear();
    for &(hash, pos) in raw {
        let p = pos as usize;
        if !mask[p..p + 4].iter().any(|&b| b) {
            out.push((hash, pos));
        }
    }
}

fn winnow_run(run: &[(u64, u32)], on_pick: &mut impl FnMut(u64, u32)) {
    if run.len() < 3 {
        let mut best = run[0];
        for &item in &run[1..] {
            if item.0 <= best.0 {
                best = item;
            }
        }
        on_pick(best.0, best.1);
        return;
    }
    let mut last_picked = u32::MAX;
    for w in run.windows(3) {
        let mut best = w[0];
        if w[1].0 <= best.0 {
            best = w[1];
        }
        if w[2].0 <= best.0 {
            best = w[2];
        }
        if best.1 != last_picked {
            last_picked = best.1;
            on_pick(best.0, best.1);
        }
    }
}

fn winnow_contiguous(win_slice: &[(u64, u32)], mut on_pick: impl FnMut(u64, u32)) {
    let (n, mut run_start) = (win_slice.len(), 0usize);
    while run_start < n {
        let mut run_end = run_start + 1;
        while run_end < n && win_slice[run_end].1 == win_slice[run_end - 1].1 + 1 {
            run_end += 1;
        }
        winnow_run(&win_slice[run_start..run_end], &mut on_pick);
        run_start = run_end;
    }
}

fn scan_bytes_feats(bytes: &[u8], lut: &[f32; 256], out: &mut Vec<LineFeat>) {
    out.clear();
    for line in bytes.split(|&b| b == b'\n') {
        let clean = line.strip_suffix(b"\r").unwrap_or(line);
        out.push(scan_line_type2(clean, lut));
    }
}

fn extract_line(bytes: &[u8], line_idx: usize) -> String {
    let Some(raw) = bytes.split(|&b| b == b'\n').nth(line_idx) else {
        return String::new();
    };
    let clean = raw.strip_suffix(b"\r").unwrap_or(raw);
    String::from_utf8_lossy(clean).trim().to_string()
}

type ExactKgrams = Vec<(usize, Vec<(u64, u32)>)>;

fn collect_exact_kgrams(
    files: &[(&str, &[u8])],
    log2_lut: &[f32; 256],
) -> (FxHashMap<u64, u32>, ExactKgrams) {
    let mut df_counts: FxHashMap<u64, u32> = FxHashMap::default();
    let mut per_file = Vec::with_capacity(files.len());
    let mut feats = Vec::with_capacity(256);
    let mut seen_in_file: FxHashSet<u64> = FxHashSet::default();

    for &(_, bytes) in files {
        scan_bytes_feats(bytes, log2_lut, &mut feats);
        let mut kgrams = Vec::new();
        seen_in_file.clear();
        for idx in 0..feats.len().saturating_sub(3) {
            if !is_valid_kgram(&feats, idx) {
                continue;
            }
            let hash = kgram_hash(&feats, idx);
            kgrams.push((hash, idx as u32));
            if seen_in_file.insert(hash) {
                *df_counts.entry(hash).or_insert(0) += 1;
            }
        }
        per_file.push((feats.len(), kgrams));
    }
    (df_counts, per_file)
}

/// Fast single-pass exact clone detector for `< 500` files (zero `3.06 MiB` Bloom alloc).
fn find_clones_exact(files: &[(&str, &[u8])], log2_lut: &[f32; 256]) -> Vec<CloneCluster> {
    let (df_counts, per_file) = collect_exact_kgrams(files, log2_lut);
    let mut buckets: FxHashMap<u64, SmallVec<[(u32, u32); 2]>> = FxHashMap::default();
    let (mut boiler_mask, mut filtered) = (Vec::new(), Vec::new());
    let cap = DF_BOILER_CAP as u32;

    for (fid, (n_lines, raw_kgrams)) in per_file.into_iter().enumerate() {
        let has_boiler = raw_kgrams
            .iter()
            .any(|&(h, _)| df_counts.get(&h).copied().unwrap_or(0) >= cap);
        let win_slice: &[(u64, u32)] = if has_boiler {
            boiler_mask.clear();
            boiler_mask.resize(n_lines, false);
            for &(hash, pos) in &raw_kgrams {
                if df_counts.get(&hash).copied().unwrap_or(0) >= cap {
                    mark_boiler_window(&mut boiler_mask, pos as usize);
                }
            }
            filter_boiler_kgrams(&raw_kgrams, &boiler_mask, &mut filtered);
            &filtered
        } else {
            &raw_kgrams
        };
        winnow_contiguous(win_slice, |hash, pos| {
            let df = df_counts.get(&hash).copied().unwrap_or(0);
            if (2..cap).contains(&df) {
                buckets.entry(hash).or_default().push((fid as u32, pos));
            }
        });
    }

    let mut postings = Vec::new();
    for (hash, occs) in buckets {
        if occs.len() >= 2 {
            postings.extend(occs.into_iter().map(|(fid, pos)| (hash, fid, pos)));
        }
    }
    assemble_clusters(files, postings)
}

#[inline]
fn cm4_get(sketch: &[u8], h: u64) -> u8 {
    let hm = rapid_hash_u64(h, SEED_CM);
    let p1 = (hm as usize) & CM_MASK;
    let p2 = ((hm >> 22) as usize) & CM_MASK;
    let c1 = (sketch[p1 >> 1] >> ((p1 & 1) * 4)) & 0x0f;
    let c2 = (sketch[p2 >> 1] >> ((p2 & 1) * 4)) & 0x0f;
    c1.min(c2)
}

#[inline]
fn cm4_inc_conservative(sketch: &mut [u8], h: u64) -> u8 {
    let hm = rapid_hash_u64(h, SEED_CM);
    let p1 = (hm as usize) & CM_MASK;
    let p2 = ((hm >> 22) as usize) & CM_MASK;
    let (s1, s2) = ((p1 & 1) * 4, (p2 & 1) * 4);
    let c1 = (sketch[p1 >> 1] >> s1) & 0x0f;
    let c2 = (sketch[p2 >> 1] >> s2) & 0x0f;
    let min_c = c1.min(c2);
    let next_c = min_c.saturating_add(1).min(15);
    if c1 == min_c && c1 < 15 {
        sketch[p1 >> 1] = (sketch[p1 >> 1] & !(0x0f << s1)) | (next_c << s1);
    }
    if c2 == min_c && c2 < 15 {
        sketch[p2 >> 1] = (sketch[p2 >> 1] & !(0x0f << s2)) | (next_c << s2);
    }
    next_c
}

#[inline]
fn stamp_table_insert(table: &mut [(u64, u32); 256], hash: u64, stamp: u32) -> bool {
    let mut slot = (hash as usize) & 255;
    for _ in 0..8 {
        if table[slot].1 != stamp {
            table[slot] = (hash, stamp);
            return true;
        }
        if table[slot].0 == hash {
            return false;
        }
        slot = (slot + 1) & 255;
    }
    true
}

#[inline]
fn bitset_test_and_set_3(bits: &mut FixedBitSet, p1: usize, p2: usize, p3: usize) -> bool {
    if bits.contains(p1) && bits.contains(p2) && bits.contains(p3) {
        return true;
    }
    bits.insert(p1);
    bits.insert(p2);
    bits.insert(p3);
    false
}

struct BloomScratch {
    kg_tab: [(u64, u32); 256],
    wn_tab: [(u64, u32); 256],
    feats: Vec<LineFeat>,
    raw_kgrams: Vec<(u64, u32)>,
    kgrams: Vec<(u64, u32)>,
    boiler_mask: Vec<bool>,
    file_winnow_uniq: Vec<u64>,
}

impl BloomScratch {
    fn new() -> Self {
        Self {
            kg_tab: [(0, 0); 256],
            wn_tab: [(0, 0); 256],
            feats: Vec::with_capacity(256),
            raw_kgrams: Vec::with_capacity(256),
            kgrams: Vec::with_capacity(256),
            boiler_mask: Vec::with_capacity(256),
            file_winnow_uniq: Vec::with_capacity(256),
        }
    }
}

struct BloomPass1 {
    df_sketch: Vec<u8>,
    file_bloom: Vec<u64>,
    file_has_boiler: FixedBitSet,
    dup_hashes: Vec<u64>,
}

fn bloom_build_pass1_kgrams(
    scratch: &mut BloomScratch,
    df_sketch: &mut [u8],
    file_has_boiler: &mut FixedBitSet,
    fid: usize,
) -> bool {
    let stamp = (fid as u32) + 1;
    scratch.raw_kgrams.clear();
    let mut has_boiler = false;
    for idx in 0..scratch.feats.len().saturating_sub(3) {
        if !is_valid_kgram(&scratch.feats, idx) {
            continue;
        }
        let hash = kgram_hash(&scratch.feats, idx);
        scratch.raw_kgrams.push((hash, idx as u32));
        if stamp_table_insert(&mut scratch.kg_tab, hash, stamp)
            && cm4_inc_conservative(df_sketch, hash) >= DF_BOILER_CAP
        {
            has_boiler = true;
        }
    }
    if !has_boiler {
        return false;
    }
    file_has_boiler.insert(fid);
    scratch.boiler_mask.clear();
    scratch.boiler_mask.resize(scratch.feats.len(), false);
    for &(hash, pos) in &scratch.raw_kgrams {
        if cm4_get(df_sketch, hash) >= DF_BOILER_CAP {
            mark_boiler_window(&mut scratch.boiler_mask, pos as usize);
        }
    }
    filter_boiler_kgrams(&scratch.raw_kgrams, &scratch.boiler_mask, &mut scratch.kgrams);
    true
}

fn run_bloom_pass1(
    files: &[(&str, &[u8])],
    log2_lut: &[f32; 256],
    scratch: &mut BloomScratch,
) -> BloomPass1 {
    let mut l1_seen = FixedBitSet::with_capacity(BLOOM_BITS);
    let mut l2_dup = FixedBitSet::with_capacity(L2_BITS);
    let mut file_bloom = vec![0u64; FB_BUCKETS * FB_WORDS];
    let mut df_sketch = vec![0u8; CM_SLOTS / 2];
    let mut file_has_boiler = FixedBitSet::with_capacity(files.len());
    let mut dup_hashes = Vec::with_capacity(1024);

    for (fid, &(_, bytes)) in files.iter().enumerate() {
        let stamp = (fid as u32) + 1;
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let use_filtered = bloom_build_pass1_kgrams(scratch, &mut df_sketch, &mut file_has_boiler, fid);
        let win_slice: &[(u64, u32)] = if use_filtered { &scratch.kgrams } else { &scratch.raw_kgrams };

        scratch.file_winnow_uniq.clear();
        let (wn_tab, uniq) = (&mut scratch.wn_tab, &mut scratch.file_winnow_uniq);
        winnow_contiguous(win_slice, |hash, _| {
            if stamp_table_insert(wn_tab, hash, stamp) {
                uniq.push(hash);
            }
        });

        let fb_base = (fid & (FB_BUCKETS - 1)) * FB_WORDS;
        for &hash in &scratch.file_winnow_uniq {
            let h_fb = rapid_hash_u64(hash, SEED_FB);
            for s in 0..6 {
                let bit = ((h_fb >> (s * 10)) as usize) & 1023;
                file_bloom[fb_base + (bit >> 6)] |= 1u64 << (bit & 63);
            }
            let p1 = (hash as usize) & BLOOM_MASK;
            let p2 = ((hash >> 23) as usize) & BLOOM_MASK;
            let p3 = ((hash.wrapping_mul(0x517c_c1b7_2722_0a95) >> 41) as usize) & BLOOM_MASK;
            if bitset_test_and_set_3(&mut l1_seen, p1, p2, p3)
                && (2..DF_BOILER_CAP).contains(&cm4_get(&df_sketch, hash))
            {
                let h_l2 = rapid_hash_u64(hash, SEED_L2);
                let q1 = (h_l2 as usize) & L2_MASK;
                let q2 = ((h_l2 >> 22) as usize) & L2_MASK;
                let q3 = ((h_l2.wrapping_mul(SEED_CM) >> 42) as usize) & L2_MASK;
                if !bitset_test_and_set_3(&mut l2_dup, q1, q2, q3) {
                    dup_hashes.push(hash);
                }
            }
        }
    }
    BloomPass1 { df_sketch, file_bloom, file_has_boiler, dup_hashes }
}

fn bloom_build_pass2_kgrams(scratch: &mut BloomScratch, df_sketch: &[u8], may_boiler: bool) -> bool {
    scratch.raw_kgrams.clear();
    if !may_boiler {
        for idx in 0..scratch.feats.len().saturating_sub(3) {
            if is_valid_kgram(&scratch.feats, idx) {
                scratch.raw_kgrams.push((kgram_hash(&scratch.feats, idx), idx as u32));
            }
        }
        return false;
    }
    scratch.boiler_mask.clear();
    scratch.boiler_mask.resize(scratch.feats.len(), false);
    let mut has_b = false;
    for idx in 0..scratch.feats.len().saturating_sub(3) {
        if !is_valid_kgram(&scratch.feats, idx) {
            continue;
        }
        let hash = kgram_hash(&scratch.feats, idx);
        if cm4_get(df_sketch, hash) >= DF_BOILER_CAP {
            has_b = true;
            mark_boiler_window(&mut scratch.boiler_mask, idx);
        } else {
            scratch.raw_kgrams.push((hash, idx as u32));
        }
    }
    if has_b {
        filter_boiler_kgrams(&scratch.raw_kgrams, &scratch.boiler_mask, &mut scratch.kgrams);
    }
    has_b
}

/// Bounded-RAM (`<= 3.06 MiB`) 2-pass Bloom+CM clone detector for `>= 500`-file repos.
fn find_clones_bloom(files: &[(&str, &[u8])], log2_lut: &[f32; 256]) -> Vec<CloneCluster> {
    let mut scratch = BloomScratch::new();
    let pass1 = run_bloom_pass1(files, log2_lut, &mut scratch);

    let mut surviving_dups: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(pass1.dup_hashes.len(), Default::default());
    let mut surv_bloom = FixedBitSet::with_capacity(SB_BITS);
    for hash in pass1.dup_hashes {
        if (2..DF_BOILER_CAP).contains(&cm4_get(&pass1.df_sketch, hash)) {
            surviving_dups.insert(hash);
            surv_bloom.insert((hash as usize) & SB_MASK);
            surv_bloom.insert(((hash >> 19) as usize) & SB_MASK);
        }
    }
    if surviving_dups.is_empty() {
        return Vec::new();
    }

    let cand_bucket = select_candidate_buckets(files.len(), &surviving_dups, &pass1.file_bloom);
    drop(pass1.file_bloom);

    let mut postings = Vec::new();
    for (fid, &(_, bytes)) in files.iter().enumerate() {
        if !cand_bucket[fid & (FB_BUCKETS - 1)] {
            continue;
        }
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let may_boiler = pass1.file_has_boiler.contains(fid) || fid < 1024;
        let use_filtered = bloom_build_pass2_kgrams(&mut scratch, &pass1.df_sketch, may_boiler);
        let win_slice: &[(u64, u32)] = if use_filtered { &scratch.kgrams } else { &scratch.raw_kgrams };
        winnow_contiguous(win_slice, |hash, pos| {
            if surv_bloom.contains((hash as usize) & SB_MASK)
                && surv_bloom.contains(((hash >> 19) as usize) & SB_MASK)
                && surviving_dups.contains(&hash)
            {
                postings.push((hash, fid as u32, pos));
            }
        });
    }
    assemble_clusters(files, postings)
}

fn select_candidate_buckets(
    n_files: usize,
    surviving_dups: &FxHashSet<u64>,
    file_bloom: &[u64],
) -> Vec<bool> {
    let mut cand = vec![false; FB_BUCKETS];
    let active = n_files.min(FB_BUCKETS);
    if surviving_dups.len() > 512 {
        cand[..active].fill(true);
        return cand;
    }
    let probes: Vec<([u8; 6], [u64; 6])> = surviving_dups
        .iter()
        .map(|&h| {
            let h_fb = rapid_hash_u64(h, SEED_FB);
            let (mut words, mut masks) = ([0u8; 6], [0u64; 6]);
            for s in 0..6 {
                let bit = ((h_fb >> (s * 10)) as usize) & 1023;
                words[s] = (bit >> 6) as u8;
                masks[s] = 1u64 << (bit & 63);
            }
            (words, masks)
        })
        .collect();
    for (bkt, flag) in cand.iter_mut().enumerate().take(active) {
        let slice = &file_bloom[bkt * FB_WORDS..(bkt + 1) * FB_WORDS];
        *flag = probes
            .iter()
            .any(|(w, m)| (0..6).all(|k| (slice[(w[k] & 15) as usize] & m[k]) != 0));
    }
    cand
}

fn assemble_clusters(
    files: &[(&str, &[u8])],
    mut postings: Vec<(u64, u32, u32)>,
) -> Vec<CloneCluster> {
    postings.sort_unstable_by_key(|&(h, fid, line)| (h, fid, line));
    let mut clusters = Vec::new();
    let mut idx = 0usize;
    while idx < postings.len() {
        let hash = postings[idx].0;
        let start = idx;
        while idx < postings.len() && postings[idx].0 == hash {
            idx += 1;
        }
        let mut occ: Vec<(String, usize)> = Vec::new();
        let (mut last_fid, mut sample_fid, mut sample_line) = (u32::MAX, 0usize, 0usize);
        for &(_, fid, line_idx) in &postings[start..idx] {
            if fid != last_fid {
                if occ.is_empty() {
                    sample_fid = fid as usize;
                    sample_line = line_idx as usize;
                }
                occ.push((files[fid as usize].0.to_string(), (line_idx as usize) + 1));
                last_fid = fid;
            }
        }
        if occ.len() >= 2 && (occ.len() as u8) < DF_BOILER_CAP {
            occ.sort();
            let sample = extract_line(files[sample_fid].1, sample_line);
            clusters.push(CloneCluster {
                fingerprint: format!("{:012x}", hash & 0x0000_ffff_ffff_ffff),
                line_count: 4,
                sample_line: sample.chars().take(100).collect(),
                occurrences: occ,
            });
        }
    }

    clusters.sort_by(|a, b| {
        let imp_a = a.occurrences.len().saturating_sub(1) * a.line_count;
        let imp_b = b.occurrences.len().saturating_sub(1) * b.line_count;
        imp_b
            .cmp(&imp_a)
            .then_with(|| b.occurrences.len().cmp(&a.occurrences.len()))
            .then_with(|| a.occurrences.cmp(&b.occurrences))
            .then_with(|| a.fingerprint.cmp(&b.fingerprint))
    });

    let mut out: Vec<CloneCluster> = Vec::new();
    for c in clusters {
        let overlaps = out.iter().any(|prev| {
            prev.occurrences.len() == c.occurrences.len()
                && prev
                    .occurrences
                    .iter()
                    .zip(&c.occurrences)
                    .all(|((pf, pl), (cf, cl))| pf == cf && pl.abs_diff(*cl) <= 4)
        });
        if !overlaps {
            out.push(c);
            if out.len() >= 25 {
                break;
            }
        }
    }
    out
}

pub fn find_clones(
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
) -> Vec<CloneCluster> {
    let pairs: Vec<(&str, &[u8])> = source_rels
        .iter()
        .filter_map(|rel| preloaded.get(rel).map(|b| (rel.as_str(), b.as_slice())))
        .collect();
    find_clones_from_slices(&pairs)
}

pub fn find_clones_from_slices(files: &[(&str, &[u8])]) -> Vec<CloneCluster> {
    if files.len() < 2 {
        return Vec::new();
    }
    let log2_lut = build_log2_lut();
    if files.len() < EXACT_CLONE_FILE_CAP {
        find_clones_exact(files, &log2_lut)
    } else {
        find_clones_bloom(files, &log2_lut)
    }
}

// ============================================================================
// 2. Zero-Copy SimiFlow Fuzzy Helper & Capability Clustering
// ============================================================================

const N_BANDS: usize = 16;
const B_MAX: usize = 32;
const HAMMING_MAX: u32 = 16;
const DIRECT_PAIRWISE_MAX: usize = 128;
const WEIGHTED_JACCARD_MIN: f64 = 0.55;
const JARO_WINKLER_MIN: f64 = 0.80;
const STRONG_BODY_JACCARD: f64 = 0.85;

const SYNTAX_STOP: &[&[u8]] = &[
    b"const", b"let", b"var", b"return", b"export", b"import", b"from", b"function", b"async",
    b"await", b"pub", b"fn", b"struct", b"enum", b"impl", b"trait", b"mod", b"use", b"crate",
    b"self", b"super", b"def", b"class", b"pass", b"none", b"true", b"false", b"func", b"package",
    b"type", b"interface", b"string", b"number", b"boolean", b"void", b"usize", b"u32", b"u64",
    b"i32", b"i64", b"f64", b"bool", b"if", b"else", b"elif", b"for", b"while", b"match",
    b"break", b"continue", b"in", b"of", b"new",
];

#[derive(Debug, Clone)]
pub struct SimiItem<'a> {
    pub name: Cow<'a, str>,
    pub body: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct SimiMatch {
    pub a: usize,
    pub b: usize,
}

#[derive(Debug, Clone)]
pub struct SimiStats {
    pub matches: Vec<SimiMatch>,
    pub lsh_candidates: usize,
    pub hot_buckets_capped: usize,
}

fn simhash64_idf(tok_weights: &[(u64, f32)]) -> u64 {
    let mut v = [0.0f32; 64];
    for &(h, w) in tok_weights {
        for (bit, acc) in v.iter_mut().enumerate() {
            *acc += if (h & (1u64 << bit)) != 0 { w } else { -w };
        }
    }
    let mut fp = 0u64;
    for (bit, &acc) in v.iter().enumerate() {
        if acc > 0.0 { fp |= 1u64 << bit; }
    }
    fp
}

fn weighted_jaccard(a: &[(u64, f32)], b: &[(u64, f32)]) -> f64 {
    let (mut i, mut j, mut inter, mut uni) = (0usize, 0usize, 0.0f64, 0.0f64);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                inter += a[i].1.min(b[j].1) as f64;
                uni += a[i].1.max(b[j].1) as f64;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                uni += a[i].1 as f64;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                uni += b[j].1 as f64;
                j += 1;
            }
        }
    }
    let tail_a: f64 = a[i..].iter().map(|x| x.1 as f64).sum();
    let tail_b: f64 = b[j..].iter().map(|x| x.1 as f64).sum();
    uni += tail_a + tail_b;
    if uni > 0.0 { inter / uni } else { 0.0 }
}

fn extract_domain_token_hashes(body: &[u8], filter_syntax: bool) -> Vec<u64> {
    let mut toks = Vec::with_capacity(24);
    let mut stack_buf = [0u8; 64];
    for raw in iter_idents(body).filter(|r| r.len() >= 2) {
        let hash = if raw.len() <= stack_buf.len() {
            for (dst, &src) in stack_buf[..raw.len()].iter_mut().zip(raw) {
                *dst = src.to_ascii_lowercase();
            }
            let lower = &stack_buf[..raw.len()];
            if filter_syntax && SYNTAX_STOP.contains(&lower) {
                continue;
            }
            rapid_hash64(lower, 0x333)
        } else {
            let owned: Vec<u8> = raw.iter().map(u8::to_ascii_lowercase).collect();
            rapid_hash64(&owned, 0x333)
        };
        toks.push(hash);
    }
    toks.sort_unstable();
    toks.dedup();
    toks
}

type WeightedDocs = Vec<Vec<(u64, f32)>>;
type SparseLshBands = FxHashMap<u16, SmallVec<[u32; 4]>>;

fn build_weighted_docs_and_fps(items: &[SimiItem<'_>], filter_syntax: bool) -> (WeightedDocs, Vec<u64>) {
    let n = items.len();
    let mut tok_df: FxHashMap<u64, u32> = FxHashMap::with_capacity_and_hasher(n * 8 + 64, Default::default());
    let mut raw_doc_toks = Vec::with_capacity(n);
    for item in items {
        let toks = extract_domain_token_hashes(item.body, filter_syntax);
        for &h in &toks { *tok_df.entry(h).or_insert(0) += 1; }
        raw_doc_toks.push(toks);
    }
    let (df_stop_cap, n_f) = ((n / 10).max(8) as u32, (n as f64) + 1.0);
    let (mut doc_vecs, mut fps) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for toks in raw_doc_toks {
        let weighted: Vec<(u64, f32)> = toks.into_iter().filter_map(|h| {
            let df = tok_df.get(&h).copied().unwrap_or(1);
            (df <= df_stop_cap).then(|| (h, ((n_f / ((df as f64) + 1.0)).ln() + 1.0) as f32))
        }).collect();
        fps.push(if weighted.len() >= 3 { simhash64_idf(&weighted) } else { 0 });
        doc_vecs.push(weighted);
    }
    (doc_vecs, fps)
}

fn collect_candidates(fps: &[u64]) -> (FxHashSet<(usize, usize)>, usize) {
    let mut cands: FxHashSet<(usize, usize)> = FxHashSet::default();
    if fps.len() <= DIRECT_PAIRWISE_MAX {
        for (lo, &a) in fps.iter().enumerate().filter(|&(_, &f)| f != 0) {
            for (hi, &b) in fps.iter().enumerate().skip(lo + 1) {
                if b != 0 && (a ^ b).count_ones() <= HAMMING_MAX { cands.insert((lo, hi)); }
            }
        }
        return (cands, 0);
    }
    let mut bands: SparseLshBands = FxHashMap::with_capacity_and_hasher(fps.len() * 4, Default::default());
    for (idx, &fp) in fps.iter().enumerate().filter(|&(_, &f)| f != 0) {
        for band_idx in 0..N_BANDS {
            let key = ((fp.rotate_left((band_idx * 4) as u32)) & 0xff) as u16;
            bands.entry(((band_idx as u16) << 8) | key).or_default().push(idx as u32);
        }
    }
    let mut hot_capped = 0usize;
    for bucket in bands.values().filter(|b| b.len() >= 2) {
        if bucket.len() > B_MAX { hot_capped += 1; continue; }
        for (i, &u) in bucket.iter().enumerate() {
            for &v in &bucket[i + 1..] {
                let (lo, hi) = if u < v { (u as usize, v as usize) } else { (v as usize, u as usize) };
                if (fps[lo] ^ fps[hi]).count_ones() <= HAMMING_MAX { cands.insert((lo, hi)); }
            }
        }
    }
    (cands, hot_capped)
}

pub fn find_similar_items(items: &[SimiItem<'_>], filter_syntax: bool) -> SimiStats {
    if items.len() < 2 {
        return SimiStats { matches: Vec::new(), lsh_candidates: 0, hot_buckets_capped: 0 };
    }
    let (doc_vecs, fps) = build_weighted_docs_and_fps(items, filter_syntax);
    let (lsh_cands, hot_buckets_capped) = collect_candidates(&fps);
    let mut matches: Vec<SimiMatch> = lsh_cands.iter().filter_map(|&(lo, hi)| {
        let wj = weighted_jaccard(&doc_vecs[lo], &doc_vecs[hi]);
        let ok = wj >= WEIGHTED_JACCARD_MIN
            && (jaro_winkler(&items[lo].name, &items[hi].name) >= JARO_WINKLER_MIN || wj >= STRONG_BODY_JACCARD);
        ok.then_some(SimiMatch { a: lo, b: hi })
    }).collect();
    matches.sort_by_key(|m| (m.a, m.b));
    SimiStats { matches, lsh_candidates: lsh_cands.len(), hot_buckets_capped }
}

fn split_ident_lower(s: &str) -> Vec<String> {
    let (mut tokens, mut cur) = (Vec::new(), String::new());
    for ch in s.chars() {
        let camel = ch.is_ascii_uppercase() && cur.chars().last().is_some_and(|c| c.is_ascii_lowercase());
        if !ch.is_ascii_alphanumeric() {
            if !cur.is_empty() { tokens.push(cur.to_ascii_lowercase()); cur.clear(); }
        } else if camel {
            tokens.push(cur.to_ascii_lowercase());
            cur.clear();
            cur.push(ch);
        } else { cur.push(ch); }
    }
    if !cur.is_empty() { tokens.push(cur.to_ascii_lowercase()); }
    tokens
}

fn canonical_stem(rel: &str) -> String {
    let (p, stem) = (Path::new(rel), stem_of(Path::new(rel)));
    if matches!(stem, "index" | "mod" | "lib" | "main")
        && let Some(dir) = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str())
    { return dir.to_ascii_lowercase(); }
    stem.to_ascii_lowercase()
}

fn infer_capability_label(files: &[String]) -> String {
    let mut counts: FxHashMap<String, usize> = FxHashMap::default();
    for rel in files {
        for tok in split_ident_lower(&canonical_stem(rel)) {
            if tok.len() >= 2 && !tok.bytes().all(|b| b.is_ascii_digit())
                && !matches!(tok.as_str(), "v1" | "v2" | "v3")
            { *counts.entry(tok).or_insert(0) += 1; }
        }
    }
    counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map_or_else(|| "shared_helper".into(), |(k, _)| k)
}

/// Discover duplicated capability/helper modules borrowing directly from preloaded `&[u8]`.
pub fn find_capability_clusters(source_rels: &[String], preloaded: &BTreeMap<String, Vec<u8>>) -> Vec<RoleCluster> {
    if source_rels.len() < 2 { return Vec::new(); }
    let items: Vec<SimiItem<'_>> = source_rels.iter().map(|rel| SimiItem {
        name: Cow::Owned(canonical_stem(rel)),
        body: preloaded.get(rel).map(Vec::as_slice).unwrap_or(&[]),
    }).collect();
    let stats = find_similar_items(&items, true);
    tracing::debug!(
        lsh_candidates = stats.lsh_candidates, hot_buckets_capped = stats.hot_buckets_capped,
        matches = stats.matches.len(), "simiflow capability clustering complete"
    );
    if stats.matches.is_empty() { return Vec::new(); }

    let mut parent: Vec<usize> = (0..source_rels.len()).collect();
    fn find_root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x { p[x] = p[p[x]]; x = p[x]; }
        x
    }
    for m in &stats.matches {
        let (ra, rb) = (find_root(&mut parent, m.a), find_root(&mut parent, m.b));
        if ra != rb { parent[rb] = ra; }
    }
    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (i, rel) in source_rels.iter().enumerate() {
        groups.entry(find_root(&mut parent, i)).or_default().push(rel.clone());
    }
    let mut clusters: Vec<RoleCluster> = groups.into_values().filter(|g| g.len() >= 2).map(|mut files| {
        files.sort();
        let cap = infer_capability_label(&files);
        let rec = format!("Detected {} structurally similar `{cap}` modules; consolidate into one canonical module", files.len());
        RoleCluster { recommendation: rec, capability: cap, files }
    }).collect();
    clusters.sort_by(|a, b| {
        b.files.len().cmp(&a.files.len())
            .then_with(|| a.capability.cmp(&b.capability))
            .then_with(|| a.files.cmp(&b.files))
    });
    clusters.truncate(20);
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clones_type1_type2_and_zero_boilerplate_fp() {
        const OPS: [&str; 6] = ["+", "-", "*", "^", "|", "&"];
        let mut owned: Vec<(String, String)> = Vec::new();

        for fid in 0..48usize {
            let mut lines = Vec::new();
            if fid < 16 {
                for _ in 0..6 {
                    lines.push("    a = 0; a = 0; a = 0; a = 0; a = 0;".to_string());
                }
            } else if fid < 32 {
                for ln in 0..6 {
                    lines.push(format!(
                        "    this.fieldSlot_{fid}_{ln} = inputParam_{fid}_{ln}; return this.fieldSlot_{fid}_{ln};"
                    ));
                }
            }
            for ln in lines.len()..20 {
                let h = rapid_hash_u64(rapid_hash_u64(((fid as u64) << 16) | (ln as u64), 0xabc1), 0xdef2);
                let op1 = OPS[(h as usize) % 6];
                let op2 = OPS[((h >> 4) as usize) % 6];
                let c = ((h >> 8) % 997) as usize;
                lines.push(format!(
                    "const uniq_{fid}_{ln} = calc_{fid}(in_{ln} {op1} {c}, ctx_{fid}) {op2} salt_{ln};"
                ));
            }
            owned.push((format!("src/boiler_{fid:02}.ts"), lines.join("\n")));
        }

        let mut t1 = String::new();
        for ln in 0..8 {
            t1.push_str(&format!(
                "const acc_{ln} = validatePayload(req.body, ctx.tenant) + computeDigest({ln}, 31) ^ 7;\n"
            ));
        }
        owned.push(("src/t1_a.ts".into(), t1.clone()));
        owned.push(("src/t1_b.ts".into(), t1));

        let (mut t2_a, mut t2_b) = (String::new(), String::new());
        for ln in 0..8 {
            t2_a.push_str(&format!(
                "const userRec_{ln} = verifySignature(rawToken, 11) * normalizeEpoch(userRec, {ln}) + 19;\n"
            ));
            t2_b.push_str(&format!(
                "const acctObj_{ln} = verifySignature(jwtSlice, 11) * normalizeEpoch(acctObj, {ln}) + 19;\n"
            ));
        }
        owned.push(("src/t2_a.ts".into(), t2_a));
        owned.push(("src/t2_b.ts".into(), t2_b));

        let slices: Vec<(&str, &[u8])> = owned.iter().map(|(p, b)| (p.as_str(), b.as_bytes())).collect();
        let lut = build_log2_lut();
        for (label, clusters) in [
            ("exact", find_clones_from_slices(&slices)),
            ("bloom", find_clones_bloom(&slices, &lut)),
        ] {
            let boiler_fp = clusters
                .iter()
                .filter(|c| c.occurrences.iter().any(|(f, ln)| f.contains("boiler_") && *ln <= 6))
                .count();
            assert_eq!(boiler_fp, 0, "[{label}] boilerplate leaked: {clusters:?}");
            assert!(
                clusters.iter().any(|c| c.occurrences.iter().any(|(f, _)| f == "src/t1_a.ts")
                    && c.occurrences.iter().any(|(f, _)| f == "src/t1_b.ts")),
                "[{label}] missed Type-1 clone"
            );
            assert!(
                clusters.iter().any(|c| c.occurrences.iter().any(|(f, _)| f == "src/t2_a.ts")
                    && c.occurrences.iter().any(|(f, _)| f == "src/t2_b.ts")),
                "[{label}] missed Type-2 clone"
            );
        }
    }

    #[test]
    fn test_simiflow_idf_bmax32_and_zero_false_positives() {
        const DOMAINS: [(&str, &str); 25] = [
            ("AuthToken", "header bearer jwt signature expiry issuer audience claims decode parse slice trim"),
            ("DateEpoch", "timestamp millis timezone offset isoformat calendar leapyear weekday utc epoch parse clamp"),
            ("CsvEscape", "delimiter quote comma newline backslash buffer append row column field escape parser"),
            ("PathCanon", "segment dotdot slash root normalize symlink prefix mount drive relative join clean"),
            ("CacheKey", "namespace version shard bucket digest prefix suffix ttl evict lookup entry key"),
            ("SlugTitle", "unicode ascii hyphen lowercase collapse whitespace diacritic url permalink title slug trim"),
            ("CidrMatch", "ipv4 ipv6 subnet mask prefixlen octets bitwise network broadcast range cidr route"),
            ("MimeSniff", "magic bytes header charset subtype boundary multipart contenttype media sniff detect table"),
            ("SemverCmp", "major minor patch prerelease buildmeta caret tilde comparator range version order parse"),
            ("RetryWait", "backoff jitter exponential cap attempt deadline timeout sleep schedule delay factor limit"),
            ("RateLimit", "tokenbucket refill capacity burst window quota throttle counter sliding clock permit allow"),
            ("CsrfGuard", "nonce cookie origin referer double submit secret hmac constanttime compare token guard"),
            ("HtmlClean", "tag attribute script iframe sanitize entity strip whitespace dom fragment node filter"),
            ("YamlQuote", "scalar indent block fold anchor alias colon boolean null string quote emitter"),
            ("UrlEncode", "percent hex query fragment reserved unreserved plus space decode encode uri param"),
            ("HashChunk", "block digest state update finalize merkle leaf branch padding buffer stream chunk"),
            ("ByteOrder", "endian swap little big network host u16 u32 u64 slice read write"),
            ("LineWrap", "column width break indent hyphen word margin paragraph wrap measure glyph line"),
            ("AnsiStrip", "escape bracket csi sgr color reset cursor sequence terminal strip control byte"),
            ("GlobMatch", "wildcard star question bracket negate recursive segment directory pattern glob match path"),
            ("SqlBind", "placeholder parameter dialect escape quote prepared query tuple value bind statement exec"),
            ("XmlEntity", "ampersand lt gt apos quot numeric codepoint cdata namespace element decode entity"),
            ("HexDigest", "nibble lookup lower upper encode decode byte pair buffer validate length checksum"),
            ("UuidParse", "variant version hyphen canonical bytes clockseq node timestamp random parse format uuid"),
            ("DiffHunk", "context insert delete line range header patch offset apply merge hunk diff"),
        ];

        let mut owned = Vec::with_capacity(1_000);
        for (i, &(dom, vocab)) in DOMAINS.iter().enumerate() {
            owned.push((format!("normalize{dom}V1"), format!("const return async await ctx logger {vocab} verify_{i}")));
            owned.push((format!("normalize{dom}V2"), format!("const return async await ctx logger {vocab} check_{i}")));
        }
        for i in 50..1_000 {
            owned.push((
                format!("domainSpecificOp{i:04}"),
                format!(
                    "const return async await unique pipeline transform stage alpha_{} beta_{} gamma_{}",
                    i * 3 + 1, i * 7 + 2, i * 13 + 3
                ),
            ));
        }
        let items: Vec<SimiItem<'_>> = owned
            .iter()
            .map(|(n, b)| SimiItem { name: Cow::Borrowed(n.as_str()), body: b.as_bytes() })
            .collect();

        let stats = find_similar_items(&items, false);
        assert!(stats.lsh_candidates <= 150, "too many LSH candidates: {}", stats.lsh_candidates);
        let true_hits = stats.matches.iter().filter(|m| m.a < 50 && m.a % 2 == 0 && m.b == m.a + 1).count();
        assert!(true_hits >= 24, "expected >= 24/25 domain twins, got {true_hits}");
        assert_eq!(stats.matches.len(), true_hits);

        let mut skew_owned = Vec::with_capacity(500);
        for i in 0..45 {
            skew_owned.push((format!("hotBucketTwin_{i}"), "merkle digest verify payload signature nonce".to_string()));
        }
        for i in 45..500 {
            skew_owned.push((format!("distinctOp_{i}"), format!("tok_a_{i} tok_b_{i} tok_c_{i} tok_d_{i}")));
        }
        let skew_items: Vec<SimiItem<'_>> = skew_owned
            .iter()
            .map(|(n, b)| SimiItem { name: Cow::Borrowed(n.as_str()), body: b.as_bytes() })
            .collect();
        let skew_stats = find_similar_items(&skew_items, false);
        assert!(skew_stats.hot_buckets_capped >= N_BANDS);
        assert_eq!(skew_stats.matches.len(), 0);
    }

    #[test]
    fn test_simiflow_matches_identical_4line_auth_bodies() {
        let block = concat!(
            "    const normalized = raw.trim().toLowerCase();\n",
            "    const parts = normalized.split(':');\n",
            "    const checksum = parts.length * 31;\n",
            "    const output = checksum + normalized.length;\n",
        );
        let mut preloaded = BTreeMap::new();
        preloaded.insert("src/auth_v1.ts".to_string(), format!("export function authOne(raw: string) {{\n{block}    return output;\n}}\n").into_bytes());
        preloaded.insert("src/auth_v2.ts".to_string(), format!("export function authTwo(raw: string) {{\n{block}    return output;\n}}\n").into_bytes());
        preloaded.insert("src/app.py".to_string(), b"import os\n".to_vec());
        preloaded.insert("src/main.go".to_string(), b"package main\nfunc main() {}\n".to_vec());
        let rels: Vec<String> = preloaded.keys().cloned().collect();

        let clusters = find_capability_clusters(&rels, &preloaded);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].files, vec!["src/auth_v1.ts", "src/auth_v2.ts"]);
    }
}
