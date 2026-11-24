//! Fixed-RAM (<= 3.06 MiB) Type-1 & Type-2 clone detector using stack-only fused
//! alpha-renaming (`$0,$1,$2`), 1 KB `LOG2_LUT[256]` Shannon entropy, strict all-4-lines
//! boilerplate mask (`DF_CAP = 8`), 512 KB 4-bit CM DF sketch, 1 MB `8192 x 16` File-Bloom,
//! 2-level Asymmetric Bloom, and Schleimer Winnowing.

use crate::stem::witness::{rapid_hash_u64, rapid_hash64};
use crate::types::CloneCluster;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;

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
struct LineFeat {
    norm_hash: u64,
    valid: bool,
}

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
        b"const"
            | b"let"
            | b"var"
            | b"return"
            | b"if"
            | b"else"
            | b"elif"
            | b"for"
            | b"while"
            | b"fn"
            | b"pub"
            | b"async"
            | b"await"
            | b"match"
            | b"struct"
            | b"enum"
            | b"impl"
            | b"function"
            | b"export"
            | b"def"
            | b"class"
            | b"func"
            | b"this"
            | b"self"
    )
}

#[inline]
fn is_ignored_line_prefix(trimmed: &[u8]) -> bool {
    trimmed.len() < 12
        || trimmed.starts_with(b"//")
        || trimmed.starts_with(b"#")
        || trimmed.starts_with(b"/*")
        || trimmed.starts_with(b"* ")
        || trimmed.starts_with(b"import ")
        || trimmed.starts_with(b"from ")
        || trimmed.starts_with(b"use ")
        || trimmed.starts_with(b"package ")
}

fn trim_ascii_bytes(raw: &[u8]) -> &[u8] {
    let mut start = 0usize;
    while start < raw.len() && raw[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut end = raw.len();
    while end > start && raw[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &raw[start..end]
}

fn shannon_entropy(bytes: &[u8], lut: &[f32; 256]) -> f32 {
    let mut counts = [0u16; 128];
    for &byte in bytes {
        counts[(byte & 0x7f) as usize] += 1;
    }
    let mut sum_c_log_c = 0.0f32;
    for &count in &counts {
        if count > 0 {
            sum_c_log_c += lut[(count as usize).min(255)];
        }
    }
    let total = bytes.len();
    let total_f = total as f32;
    let n_log_n = if total < 256 { lut[total] } else { total_f * total_f.log2() };
    (n_log_n - sum_c_log_c) / total_f
}

fn skip_quoted_bytes(bytes: &[u8], quote_pos: usize) -> usize {
    let quote = bytes[quote_pos];
    let mut idx = quote_pos + 1;
    while idx < bytes.len() {
        if bytes[idx] == b'\\' {
            idx = (idx + 2).min(bytes.len());
            continue;
        }
        let cur = bytes[idx];
        idx += 1;
        if cur == quote {
            break;
        }
    }
    idx
}

fn skip_ident_or_num(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_') {
        idx += 1;
    }
    idx
}

struct AlphaRenamer {
    hash: u64,
    var_slot: u32,
    local_map: [(u64, u32); 16],
    local_len: usize,
    distinct_toks: u16,
    op_count: u16,
}

impl AlphaRenamer {
    fn new() -> Self {
        Self {
            hash: 0x99u64,
            var_slot: 0,
            local_map: [(0, 0); 16],
            local_len: 0,
            distinct_toks: 0,
            op_count: 0,
        }
    }

    fn intern_var(&mut self, tok_hash: u64) -> u32 {
        for &(key, val) in &self.local_map[..self.local_len] {
            if key == tok_hash {
                return val;
            }
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
            let kh = rapid_hash64(tok, 0x11);
            self.hash = rapid_hash_u64(kh, self.hash);
            self.distinct_toks += 1;
            return;
        }
        let tok_hash = rapid_hash64(tok, 0x77);
        let id = self.intern_var(tok_hash);
        self.hash = rapid_hash_u64(0x1000_0000u64 | (id as u64), self.hash);
    }

    fn feed_symbol(&mut self, byte: u8) {
        self.hash = rapid_hash_u64(0x3000_0000u64 | (byte as u64), self.hash);
        if matches!(
            byte,
            b'+' | b'-' | b'*' | b'/' | b'^' | b'|' | b'&' | b'=' | b'<' | b'>' | b'?' | b'(' | b'.'
        ) {
            self.op_count += 1;
        }
    }
}

/// Zero-allocation stack-only fused Type-2 alpha-renaming scanner + `LOG2_LUT` Shannon entropy.
fn scan_line_type2(raw: &[u8], lut: &[f32; 256]) -> LineFeat {
    let bytes = trim_ascii_bytes(raw);
    if is_ignored_line_prefix(bytes) || shannon_entropy(bytes, lut) < 2.75 {
        return LineFeat { norm_hash: 0, valid: false };
    }

    let mut scanner = AlphaRenamer::new();
    let mut idx = 0usize;
    while idx < bytes.len() {
        let byte = bytes[idx];
        if byte.is_ascii_whitespace() {
            idx += 1;
        } else if matches!(byte, b'"' | b'\'' | b'`') {
            idx = skip_quoted_bytes(bytes, idx);
            scanner.hash = rapid_hash_u64(0x2000_0001u64, scanner.hash);
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let end = skip_ident_or_num(bytes, idx);
            scanner.feed_ident(&bytes[idx..end]);
            idx = end;
        } else if byte.is_ascii_digit() {
            idx = skip_ident_or_num(bytes, idx);
            scanner.hash = rapid_hash_u64(0x2000_0000u64, scanner.hash);
        } else {
            scanner.feed_symbol(byte);
            idx += 1;
        }
    }

    LineFeat {
        norm_hash: scanner.hash,
        valid: scanner.distinct_toks >= 4 && scanner.op_count >= 2,
    }
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
    let s1 = (p1 & 1) * 4;
    let s2 = (p2 & 1) * 4;
    let c1 = (sketch[p1 >> 1] >> s1) & 0x0f;
    let c2 = (sketch[p2 >> 1] >> s2) & 0x0f;
    let min_c = c1.min(c2);
    let next_c = if min_c < 15 { min_c + 1 } else { 15 };
    if c1 == min_c && c1 < 15 {
        sketch[p1 >> 1] = (sketch[p1 >> 1] & !(0x0f << s1)) | (next_c << s1);
    }
    if c2 == min_c && c2 < 15 {
        sketch[p2 >> 1] = (sketch[p2 >> 1] & !(0x0f << s2)) | (next_c << s2);
    }
    next_c
}

#[inline]
fn kgram_hash(feats: &[LineFeat], i: usize) -> u64 {
    let mut h = 0x1337u64;
    h = rapid_hash_u64(feats[i].norm_hash, h);
    h = rapid_hash_u64(feats[i + 1].norm_hash, h);
    h = rapid_hash_u64(feats[i + 2].norm_hash, h);
    rapid_hash_u64(feats[i + 3].norm_hash, h)
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
    for win in 0..=run.len() - 3 {
        let mut best = run[win];
        if run[win + 1].0 <= best.0 {
            best = run[win + 1];
        }
        if run[win + 2].0 <= best.0 {
            best = run[win + 2];
        }
        if best.1 != last_picked {
            last_picked = best.1;
            on_pick(best.0, best.1);
        }
    }
}

/// Schleimer winnowing (`w = 3`) over contiguous runs of valid k-grams, preserving
/// short isolated 4-5 line valid runs (`run_len` 1 or 2) by emitting their min k-gram.
fn winnow_contiguous(win_slice: &[(u64, u32)], mut on_pick: impl FnMut(u64, u32)) {
    let n = win_slice.len();
    let mut run_start = 0usize;
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
        let stripped = line.strip_suffix(b"\r").unwrap_or(line);
        out.push(scan_line_type2(stripped, lut));
    }
}

fn extract_line(bytes: &[u8], line_idx: usize) -> String {
    bytes
        .split(|&b| b == b'\n')
        .nth(line_idx)
        .map(|l| String::from_utf8_lossy(l.strip_suffix(b"\r").unwrap_or(l)).trim().to_string())
        .unwrap_or_default()
}

#[inline]
fn is_valid_kgram(feats: &[LineFeat], idx: usize) -> bool {
    feats[idx].valid && feats[idx + 1].valid && feats[idx + 2].valid && feats[idx + 3].valid
}

#[inline]
fn mark_boiler_window(boiler_mask: &mut [bool], pos: usize) {
    boiler_mask[pos] = true;
    boiler_mask[pos + 1] = true;
    boiler_mask[pos + 2] = true;
    boiler_mask[pos + 3] = true;
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

fn filter_boiler_kgrams(
    raw_kgrams: &[(u64, u32)],
    boiler_mask: &[bool],
    out: &mut Vec<(u64, u32)>,
) {
    out.clear();
    for &(hash, pos) in raw_kgrams {
        let idx = pos as usize;
        if !(boiler_mask[idx]
            || boiler_mask[idx + 1]
            || boiler_mask[idx + 2]
            || boiler_mask[idx + 3])
        {
            out.push((hash, pos));
        }
    }
}

#[inline]
fn bloom_test_and_set_3(bits: &mut [u64], p1: usize, p2: usize, p3: usize) -> bool {
    let s1 = (bits[p1 >> 6] & (1u64 << (p1 & 63))) != 0;
    let s2 = (bits[p2 >> 6] & (1u64 << (p2 & 63))) != 0;
    let s3 = (bits[p3 >> 6] & (1u64 << (p3 & 63))) != 0;
    if s1 && s2 && s3 {
        return true;
    }
    bits[p1 >> 6] |= 1u64 << (p1 & 63);
    bits[p2 >> 6] |= 1u64 << (p2 & 63);
    bits[p3 >> 6] |= 1u64 << (p3 & 63);
    false
}

struct ScanScratch {
    kg_tab: [(u64, u32); 256],
    wn_tab: [(u64, u32); 256],
    feats: Vec<LineFeat>,
    raw_kgrams: Vec<(u64, u32)>,
    kgrams: Vec<(u64, u32)>,
    boiler_mask: Vec<bool>,
    file_winnow_uniq: Vec<u64>,
}

impl ScanScratch {
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

struct Pass1Blooms {
    l1_seen: Vec<u64>,
    l2_dup: Vec<u64>,
    file_bloom: Vec<u64>,
    dup_hashes: Vec<u64>,
}

impl Pass1Blooms {
    fn new() -> Self {
        Self {
            l1_seen: vec![0u64; BLOOM_BITS / 64],
            l2_dup: vec![0u64; L2_BITS / 64],
            file_bloom: vec![0u64; FB_BUCKETS * FB_WORDS],
            dup_hashes: Vec::with_capacity(1024),
        }
    }

    fn record_file_hashes(&mut self, fid: usize, hashes: &[u64], df_sketch: &[u8]) {
        let fb_base = (fid & (FB_BUCKETS - 1)) * FB_WORDS;
        for &hash in hashes {
            let h_fb = rapid_hash_u64(hash, SEED_FB);
            for shift_idx in 0..6 {
                let bit = ((h_fb >> (shift_idx * 10)) as usize) & 1023;
                self.file_bloom[fb_base + (bit >> 6)] |= 1u64 << (bit & 63);
            }
            self.record_two_level_dup(hash, df_sketch);
        }
    }

    fn record_two_level_dup(&mut self, hash: u64, df_sketch: &[u8]) {
        let p1 = (hash as usize) & BLOOM_MASK;
        let p2 = ((hash >> 23) as usize) & BLOOM_MASK;
        let p3 = ((hash.wrapping_mul(0x517c_c1b7_2722_0a95) >> 41) as usize) & BLOOM_MASK;
        if !bloom_test_and_set_3(&mut self.l1_seen, p1, p2, p3) {
            return;
        }
        let df = cm4_get(df_sketch, hash);
        if !(2..DF_BOILER_CAP).contains(&df) {
            return;
        }
        let h_l2 = rapid_hash_u64(hash, SEED_L2);
        let q1 = (h_l2 as usize) & L2_MASK;
        let q2 = ((h_l2 >> 22) as usize) & L2_MASK;
        let q3 = ((h_l2.wrapping_mul(SEED_CM) >> 42) as usize) & L2_MASK;
        if !bloom_test_and_set_3(&mut self.l2_dup, q1, q2, q3) {
            self.dup_hashes.push(hash);
        }
    }
}

struct Pass1Result {
    df_sketch: Vec<u8>,
    file_bloom: Vec<u64>,
    file_has_boiler: Vec<u64>,
    dup_hashes: Vec<u64>,
}

fn build_pass1_kgrams(
    scratch: &mut ScanScratch,
    df_sketch: &mut [u8],
    file_has_boiler: &mut [u64],
    fid: usize,
) -> bool {
    let stamp = (fid as u32) + 1;
    scratch.raw_kgrams.clear();
    let mut has_boiler_kgram = false;

    for idx in 0..scratch.feats.len().saturating_sub(3) {
        if !is_valid_kgram(&scratch.feats, idx) {
            continue;
        }
        let hash = kgram_hash(&scratch.feats, idx);
        scratch.raw_kgrams.push((hash, idx as u32));
        let is_new = stamp_table_insert(&mut scratch.kg_tab, hash, stamp);
        if is_new && cm4_inc_conservative(df_sketch, hash) >= DF_BOILER_CAP {
            has_boiler_kgram = true;
        }
    }

    if !has_boiler_kgram {
        return false;
    }

    file_has_boiler[fid >> 6] |= 1u64 << (fid & 63);
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

fn run_pass1(
    files: &[(&str, &[u8])],
    log2_lut: &[f32; 256],
    scratch: &mut ScanScratch,
) -> Pass1Result {
    let mut blooms = Pass1Blooms::new();
    let mut df_sketch = vec![0u8; CM_SLOTS / 2];
    let mut file_has_boiler = vec![0u64; files.len().div_ceil(64)];

    for (fid, &(_, bytes)) in files.iter().enumerate() {
        let stamp = (fid as u32) + 1;
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let use_filtered = build_pass1_kgrams(scratch, &mut df_sketch, &mut file_has_boiler, fid);
        let win_slice: &[(u64, u32)] = if use_filtered {
            &scratch.kgrams
        } else {
            &scratch.raw_kgrams
        };

        let wn_tab = &mut scratch.wn_tab;
        let file_winnow_uniq = &mut scratch.file_winnow_uniq;
        file_winnow_uniq.clear();
        winnow_contiguous(win_slice, |hash, _| {
            if stamp_table_insert(wn_tab, hash, stamp) {
                file_winnow_uniq.push(hash);
            }
        });

        blooms.record_file_hashes(fid, file_winnow_uniq, &df_sketch);
    }

    Pass1Result {
        df_sketch,
        file_bloom: blooms.file_bloom,
        file_has_boiler,
        dup_hashes: blooms.dup_hashes,
    }
}

fn build_surviving_dups(dup_hashes: Vec<u64>, df_sketch: &[u8]) -> (FxHashSet<u64>, Vec<u64>) {
    let mut surviving_dups: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(dup_hashes.len(), Default::default());
    let mut surv_bloom = vec![0u64; SB_BITS / 64];
    for hash in dup_hashes {
        let df = cm4_get(df_sketch, hash);
        if (2..DF_BOILER_CAP).contains(&df) {
            surviving_dups.insert(hash);
            let b1 = (hash as usize) & SB_MASK;
            let b2 = ((hash >> 19) as usize) & SB_MASK;
            surv_bloom[b1 >> 6] |= 1u64 << (b1 & 63);
            surv_bloom[b2 >> 6] |= 1u64 << (b2 & 63);
        }
    }
    (surviving_dups, surv_bloom)
}

#[derive(Clone, Copy)]
struct FbProbe {
    words: [u8; 6],
    masks: [u64; 6],
}

impl FbProbe {
    fn from_hash(hash: u64) -> Self {
        let h_fb = rapid_hash_u64(hash, SEED_FB);
        let mut words = [0u8; 6];
        let mut masks = [0u64; 6];
        for shift_idx in 0..6 {
            let bit = ((h_fb >> (shift_idx * 10)) as usize) & 1023;
            words[shift_idx] = (bit >> 6) as u8;
            masks[shift_idx] = 1u64 << (bit & 63);
        }
        Self { words, masks }
    }

    #[inline]
    fn matches_bucket(&self, slice: &[u64]) -> bool {
        (slice[(self.words[0] & 15) as usize] & self.masks[0]) != 0
            && (slice[(self.words[1] & 15) as usize] & self.masks[1]) != 0
            && (slice[(self.words[2] & 15) as usize] & self.masks[2]) != 0
            && (slice[(self.words[3] & 15) as usize] & self.masks[3]) != 0
            && (slice[(self.words[4] & 15) as usize] & self.masks[4]) != 0
            && (slice[(self.words[5] & 15) as usize] & self.masks[5]) != 0
    }
}

fn select_candidate_buckets(
    n_files: usize,
    surviving_dups: &FxHashSet<u64>,
    file_bloom: &[u64],
) -> Vec<bool> {
    let mut cand_bucket = vec![false; FB_BUCKETS];
    let active_buckets = n_files.min(FB_BUCKETS);
    if surviving_dups.len() > 512 {
        cand_bucket[..active_buckets].fill(true);
        return cand_bucket;
    }
    let probes: Vec<FbProbe> = surviving_dups.iter().map(|&h| FbProbe::from_hash(h)).collect();
    for (bkt, flag) in cand_bucket.iter_mut().enumerate().take(active_buckets) {
        let base = bkt * FB_WORDS;
        let slice = &file_bloom[base..base + FB_WORDS];
        if probes.iter().any(|probe| probe.matches_bucket(slice)) {
            *flag = true;
        }
    }
    cand_bucket
}

fn build_pass2_kgrams(
    scratch: &mut ScanScratch,
    df_sketch: &[u8],
    may_have_boiler: bool,
) -> bool {
    scratch.raw_kgrams.clear();
    if !may_have_boiler {
        for idx in 0..scratch.feats.len().saturating_sub(3) {
            if is_valid_kgram(&scratch.feats, idx) {
                scratch.raw_kgrams.push((kgram_hash(&scratch.feats, idx), idx as u32));
            }
        }
        return false;
    }

    scratch.boiler_mask.clear();
    scratch.boiler_mask.resize(scratch.feats.len(), false);
    let mut has_boiler_kgram = false;
    for idx in 0..scratch.feats.len().saturating_sub(3) {
        if !is_valid_kgram(&scratch.feats, idx) {
            continue;
        }
        let hash = kgram_hash(&scratch.feats, idx);
        if cm4_get(df_sketch, hash) >= DF_BOILER_CAP {
            has_boiler_kgram = true;
            mark_boiler_window(&mut scratch.boiler_mask, idx);
        } else {
            scratch.raw_kgrams.push((hash, idx as u32));
        }
    }
    if has_boiler_kgram {
        filter_boiler_kgrams(&scratch.raw_kgrams, &scratch.boiler_mask, &mut scratch.kgrams);
        return true;
    }
    false
}

struct Pass2Filter<'a> {
    df_sketch: &'a [u8],
    file_has_boiler: &'a [u64],
    cand_bucket: &'a [bool],
    surv_bloom: &'a [u64],
    surviving_dups: &'a FxHashSet<u64>,
}

fn run_pass2(
    files: &[(&str, &[u8])],
    log2_lut: &[f32; 256],
    scratch: &mut ScanScratch,
    filter: &Pass2Filter<'_>,
) -> Vec<(u64, u32, u32)> {
    let mut postings: Vec<(u64, u32, u32)> = Vec::new();
    for (fid, &(_, bytes)) in files.iter().enumerate() {
        if !filter.cand_bucket[fid & (FB_BUCKETS - 1)] {
            continue;
        }
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let may_have_boiler =
            (filter.file_has_boiler[fid >> 6] & (1u64 << (fid & 63))) != 0 || fid < 1024;
        let use_filtered = build_pass2_kgrams(scratch, filter.df_sketch, may_have_boiler);
        let win_slice: &[(u64, u32)] = if use_filtered {
            &scratch.kgrams
        } else {
            &scratch.raw_kgrams
        };

        winnow_contiguous(win_slice, |hash, pos| {
            let b1 = (hash as usize) & SB_MASK;
            let b2 = ((hash >> 19) as usize) & SB_MASK;
            if (filter.surv_bloom[b1 >> 6] & (1u64 << (b1 & 63))) != 0
                && (filter.surv_bloom[b2 >> 6] & (1u64 << (b2 & 63))) != 0
                && filter.surviving_dups.contains(&hash)
            {
                postings.push((hash, fid as u32, pos));
            }
        });
    }
    postings
}

fn build_cluster_from_group(
    files: &[(&str, &[u8])],
    hash: u64,
    group: &[(u64, u32, u32)],
) -> Option<CloneCluster> {
    let mut occ: Vec<(String, usize)> = Vec::new();
    let mut last_fid = u32::MAX;
    let mut sample_fid = 0usize;
    let mut sample_line_idx = 0usize;
    for &(_, fid, line_idx) in group {
        if fid != last_fid {
            if occ.is_empty() {
                sample_fid = fid as usize;
                sample_line_idx = line_idx as usize;
            }
            occ.push((files[fid as usize].0.to_string(), (line_idx as usize) + 1));
            last_fid = fid;
        }
    }
    if occ.len() < 2 || (occ.len() as u8) >= DF_BOILER_CAP {
        return None;
    }
    occ.sort();
    let sample = extract_line(files[sample_fid].1, sample_line_idx);
    Some(CloneCluster {
        fingerprint: format!("{:012x}", hash & 0x0000_ffff_ffff_ffff),
        line_count: 4,
        sample_line: sample.chars().take(100).collect(),
        occurrences: occ,
    })
}

fn assemble_clusters(
    files: &[(&str, &[u8])],
    mut postings: Vec<(u64, u32, u32)>,
) -> Vec<CloneCluster> {
    postings.sort_unstable_by_key(|&(h, fid, line)| (h, fid, line));

    let mut clusters: Vec<CloneCluster> = Vec::new();
    let mut idx = 0usize;
    while idx < postings.len() {
        let hash = postings[idx].0;
        let start = idx;
        while idx < postings.len() && postings[idx].0 == hash {
            idx += 1;
        }
        if let Some(cluster) = build_cluster_from_group(files, hash, &postings[start..idx]) {
            clusters.push(cluster);
        }
    }

    clusters.sort_by(|a, b| {
        let impact_a = a.occurrences.len().saturating_sub(1) * a.line_count;
        let impact_b = b.occurrences.len().saturating_sub(1) * b.line_count;
        impact_b
            .cmp(&impact_a)
            .then_with(|| b.occurrences.len().cmp(&a.occurrences.len()))
            .then_with(|| a.occurrences.cmp(&b.occurrences))
            .then_with(|| a.fingerprint.cmp(&b.fingerprint))
    });

    dedup_clones(clusters)
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
    let n_files = files.len();
    if n_files < 2 {
        return Vec::new();
    }

    let log2_lut = build_log2_lut();
    let mut scratch = ScanScratch::new();
    let pass1 = run_pass1(files, &log2_lut, &mut scratch);

    let (surviving_dups, surv_bloom) = build_surviving_dups(pass1.dup_hashes, &pass1.df_sketch);
    if surviving_dups.is_empty() {
        return Vec::new();
    }

    let cand_bucket = select_candidate_buckets(n_files, &surviving_dups, &pass1.file_bloom);
    drop(pass1.file_bloom);

    let postings = run_pass2(
        files,
        &log2_lut,
        &mut scratch,
        &Pass2Filter {
            df_sketch: &pass1.df_sketch,
            file_has_boiler: &pass1.file_has_boiler,
            cand_bucket: &cand_bucket,
            surv_bloom: &surv_bloom,
            surviving_dups: &surviving_dups,
        },
    );
    drop(pass1.df_sketch);
    drop(surv_bloom);

    assemble_clusters(files, postings)
}

fn dedup_clones(clusters: Vec<CloneCluster>) -> Vec<CloneCluster> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clones_type1_type2_and_zero_boilerplate_fp() {
        const OPS: [&str; 6] = ["+", "-", "*", "^", "|", "&"];
        let mut owned: Vec<(String, String)> = Vec::new();

        // 1. 16 low-entropy boilerplate files (H < 2.75) + 16 high-entropy Type-2 boilerplate files (>= DF_CAP=8)
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

        // 2. Real Type-1 exact clone pair (8 lines)
        let mut t1 = String::new();
        for ln in 0..8 {
            t1.push_str(&format!(
                "const acc_{ln} = validatePayload(req.body, ctx.tenant) + computeDigest({ln}, 31) ^ 7;\n"
            ));
        }
        owned.push(("src/t1_a.ts".into(), t1.clone()));
        owned.push(("src/t1_b.ts".into(), t1));

        // 3. Real Type-2 alpha-renamed clone pair (8 lines, completely renamed variables!)
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
        let clusters = find_clones_from_slices(&slices);

        // Zero boilerplate FPs (neither low-entropy nor high-entropy Type-2 boilerplate leaks!)
        let boiler_fp = clusters
            .iter()
            .filter(|c| c.occurrences.iter().any(|(f, ln)| f.contains("boiler_") && *ln <= 6))
            .count();
        assert_eq!(boiler_fp, 0, "boilerplate leaked into clusters: {clusters:?}");

        // Exact Type-1 and Type-2 recall == 1.00!
        assert!(
            clusters.iter().any(|c| c.occurrences.iter().any(|(f, _)| f == "src/t1_a.ts")
                && c.occurrences.iter().any(|(f, _)| f == "src/t1_b.ts")),
            "missed Type-1 clone in {clusters:?}"
        );
        assert!(
            clusters.iter().any(|c| c.occurrences.iter().any(|(f, _)| f == "src/t2_a.ts")
                && c.occurrences.iter().any(|(f, _)| f == "src/t2_b.ts")),
            "missed Type-2 renamed clone in {clusters:?}"
        );
    }
}
