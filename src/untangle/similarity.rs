//! Unified code-similarity engine:
//! 1. Adaptive Type-1 & Type-2 clone detector (`< 500` exact, `>= 500` 2-pass Bloom+CM).
//! 2. Zero-copy SimiFlow fuzzy helper & capability clustering (SimHash + sparse LSH bands).

use crate::lex::{ident_byte, iter_idents, quoted_lit};
use crate::source::stem_of;
use crate::stem::analysis::{rapid_hash_u64, rapid_hash64};
use crate::types::{CloneCluster, RoleCluster};
use bstr::ByteSlice;
use fixedbitset::FixedBitSet;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use strsim::jaro_winkler;

const EXACT_CLONE_FILE_CAP: usize = 500;
const BLOOM_BITS: usize = 1 << 23;
const BLOOM_MASK: usize = BLOOM_BITS - 1;
const L2_BITS: usize = 1 << 22;
const L2_MASK: usize = L2_BITS - 1;
const CM_SLOTS: usize = 1 << 20;
const CM_MASK: usize = CM_SLOTS - 1;
const FB_BUCKETS: usize = 8192;
const FB_WORDS: usize = 16;
const SB_BITS: usize = 1 << 19;
const SB_MASK: usize = SB_BITS - 1;
const DF_BOILER_CAP: u8 = 8;

const SEED_CM: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_FB: u64 = 0xbf58_476d_1ce4_e5b9;
const SEED_L2: u64 = 0x94d0_49bb_1331_11eb;

#[derive(Clone, Copy)]
struct LineFeat { norm_hash: u64, valid: bool }

fn build_log2_lut() -> [f32; 256] {
    std::array::from_fn(|i| if i == 0 { 0.0 } else { (i as f32) * (i as f32).log2() })
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
fn is_ignored_line_prefix(trimmed: &[u8]) -> bool {
    if trimmed.len() < 12 { return true; }
    const PREFIXES: &[&[u8]] = &[
        b"//", b"#", b"/*", b"* ", b"import ", b"from ", b"use ", b"package ",
        b"pub const ", b"pub(crate) const ", b"pub(super) const ",
    ];
    if PREFIXES.iter().any(|p| trimmed.starts_with(p)) { return true; }
    let field_rest = [b"pub(crate) ".as_slice(), b"pub(super) ", b"pub "]
        .iter().find_map(|p| trimmed.strip_prefix(*p));
    field_rest.is_some_and(|rest| {
        rest.ends_with(b",") && rest.find_byte(b':').is_some()
            && rest.find_byte(b'=').is_none() && rest.find_byte(b'(').is_none()
    })
}

fn shannon_entropy(bytes: &[u8], lut: &[f32; 256]) -> f32 {
    let mut counts = [0u16; 128];
    for &byte in bytes { counts[(byte & 0x7f) as usize] += 1; }
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

fn renamer_new() -> AlphaRenamer {
    AlphaRenamer { hash: 0x99, var_slot: 0, local_map: [(0, 0); 16], local_len: 0, distinct_toks: 0, op_count: 0 }
}

fn renamer_intern(ren: &mut AlphaRenamer, tok_hash: u64) -> u32 {
    if let Some(&(_, slot)) = ren.local_map[..ren.local_len].iter().find(|&&(k, _)| k == tok_hash) {
        return slot;
    }
    let id = ren.var_slot;
    ren.var_slot += 1;
    ren.distinct_toks += 1;
    if ren.local_len < ren.local_map.len() {
        ren.local_map[ren.local_len] = (tok_hash, id);
        ren.local_len += 1;
    }
    id
}

fn renamer_feed_ident(ren: &mut AlphaRenamer, tok: &[u8]) {
    if is_clone_keyword(tok) {
        ren.hash = rapid_hash_u64(rapid_hash64(tok, 0x11), ren.hash);
        ren.distinct_toks += 1;
        return;
    }
    let id = renamer_intern(ren, rapid_hash64(tok, 0x77));
    ren.hash = rapid_hash_u64(0x1000_0000 | (id as u64), ren.hash);
}

fn renamer_feed_symbol(ren: &mut AlphaRenamer, byte: u8) {
    ren.hash = rapid_hash_u64(0x3000_0000 | (byte as u64), ren.hash);
    if b"+-*/^|&=<>?(.".contains(&byte) { ren.op_count += 1; }
}

fn advance_ident_end(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() && ident_byte(bytes[idx]) { idx += 1; }
    idx
}

fn step_type2_token(scanner: &mut AlphaRenamer, bytes: &[u8], idx: usize) -> usize {
    let byte = bytes[idx];
    if byte.is_ascii_whitespace() { return idx + 1; }
    if matches!(byte, b'"' | b'\'' | b'`') {
        scanner.hash = rapid_hash_u64(0x2000_0001, scanner.hash);
        return quoted_lit(bytes, idx, byte != b'`', false).0;
    }
    if byte.is_ascii_alphabetic() || byte == b'_' {
        let end = advance_ident_end(bytes, idx);
        renamer_feed_ident(scanner, &bytes[idx..end]);
        return end;
    }
    if byte.is_ascii_digit() {
        scanner.hash = rapid_hash_u64(0x2000_0000, scanner.hash);
        return advance_ident_end(bytes, idx);
    }
    renamer_feed_symbol(scanner, byte);
    idx + 1
}

fn scan_line_type2(raw: &[u8], lut: &[f32; 256]) -> LineFeat {
    let bytes = raw.trim_ascii();
    if is_ignored_line_prefix(bytes) || shannon_entropy(bytes, lut) < 2.75 {
        return LineFeat { norm_hash: 0, valid: false };
    }
    let (mut scanner, mut idx) = (renamer_new(), 0usize);
    while idx < bytes.len() { idx = step_type2_token(&mut scanner, bytes, idx); }
    LineFeat { norm_hash: scanner.hash, valid: scanner.distinct_toks >= 4 && scanner.op_count >= 2 }
}

#[inline]
fn collect_valid_kgrams(
    feats: &[LineFeat], out: &mut Vec<(u64, u32)>, mut on_kgram: impl FnMut(u64),
) {
    out.clear();
    for idx in 0..feats.len().saturating_sub(3) {
        if feats[idx..idx + 4].iter().all(|f| f.valid) {
            let hash = feats[idx..idx + 4].iter().fold(0x1337u64, |h, f| rapid_hash_u64(f.norm_hash, h));
            out.push((hash, idx as u32));
            on_kgram(hash);
        }
    }
}

fn apply_boiler_mask(
    n_lines: usize, raw: &[(u64, u32)], mask: &mut Vec<bool>, out: &mut Vec<(u64, u32)>,
    mut is_boiler: impl FnMut(u64) -> bool,
) {
    mask.clear();
    mask.resize(n_lines, false);
    for &(hash, pos) in raw {
        if is_boiler(hash) { mask[pos as usize..pos as usize + 4].fill(true); }
    }
    out.clear();
    for &(hash, pos) in raw {
        let pos_u = pos as usize;
        if !mask[pos_u..pos_u + 4].iter().any(|&b| b) { out.push((hash, pos)); }
    }
}

fn winnow_run(run: &[(u64, u32)], on_pick: &mut impl FnMut(u64, u32)) {
    let pick_min = |slice: &[(u64, u32)]| -> (u64, u32) {
        slice.iter().copied().fold(slice[0], |best, item| if item.0 <= best.0 { item } else { best })
    };
    if run.len() < 3 {
        let best = pick_min(run);
        on_pick(best.0, best.1);
        return;
    }
    let mut last_picked = u32::MAX;
    for win in run.windows(3) {
        let best = pick_min(win);
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
        while run_end < n && win_slice[run_end].1 == win_slice[run_end - 1].1 + 1 { run_end += 1; }
        winnow_run(&win_slice[run_start..run_end], &mut on_pick);
        run_start = run_end;
    }
}

fn scan_bytes_feats(bytes: &[u8], lut: &[f32; 256], out: &mut Vec<LineFeat>) {
    out.clear();
    out.extend(bytes.split(|&b| b == b'\n').map(|line| scan_line_type2(line.trim_suffix(b"\r"), lut)));
}

fn extract_line(bytes: &[u8], line_idx: usize) -> String {
    let Some(raw) = bytes.split(|&b| b == b'\n').nth(line_idx) else {
        return String::new();
    };
    let cleaned = raw.trim_suffix(b"\r").trim();
    cleaned.to_str_lossy().into_owned()
}

/// Fast single-pass exact clone detector for `< 500` files (zero `3.06 MiB` Bloom alloc).
fn find_clones_exact(files: &[(&str, &[u8])], log2_lut: &[f32; 256]) -> Vec<CloneCluster> {
    let mut df_counts: FxHashMap<u64, u32> = FxHashMap::default();
    let mut per_file = Vec::with_capacity(files.len());
    let (mut feats, mut seen_in_file) = (Vec::with_capacity(256), FxHashSet::default());
    for &(_, bytes) in files {
        scan_bytes_feats(bytes, log2_lut, &mut feats);
        let mut kgrams = Vec::new();
        seen_in_file.clear();
        collect_valid_kgrams(&feats, &mut kgrams, |hash| {
            if seen_in_file.insert(hash) { *df_counts.entry(hash).or_insert(0) += 1; }
        });
        per_file.push((feats.len(), kgrams));
    }

    let mut buckets: FxHashMap<u64, SmallVec<[(u32, u32); 2]>> = FxHashMap::default();
    let (mut boiler_mask, mut filtered, cap) = (Vec::new(), Vec::new(), DF_BOILER_CAP as u32);
    for (fid, (n_lines, raw_kgrams)) in per_file.into_iter().enumerate() {
        let is_boiler = |h: u64| df_counts.get(&h).copied().unwrap_or(0) >= cap;
        let win_slice: &[(u64, u32)] = if raw_kgrams.iter().any(|&(h, _)| is_boiler(h)) {
            apply_boiler_mask(n_lines, &raw_kgrams, &mut boiler_mask, &mut filtered, is_boiler);
            &filtered
        } else { &raw_kgrams };
        winnow_contiguous(win_slice, |hash, pos| {
            let df = df_counts.get(&hash).copied().unwrap_or(0);
            if (2..cap).contains(&df) {
                buckets.entry(hash).or_default().push((fid as u32, pos));
            }
        });
    }

    let postings = buckets.into_iter().filter(|(_, occs)| occs.len() >= 2)
        .flat_map(|(hash, occs)| occs.into_iter().map(move |(fid, pos)| (hash, fid, pos))).collect();
    assemble_clusters(files, postings)
}

#[inline]
fn cm4_slots(hash: u64) -> (usize, usize) {
    let hm = rapid_hash_u64(hash, SEED_CM);
    ((hm as usize) & CM_MASK, ((hm >> 22) as usize) & CM_MASK)
}

#[inline]
fn cm4_nibble(sketch: &[u8], pos: usize) -> u8 { (sketch[pos >> 1] >> ((pos & 1) * 4)) & 0x0f }

#[inline]
fn cm4_get(sketch: &[u8], hash: u64) -> u8 {
    let (p1, p2) = cm4_slots(hash);
    cm4_nibble(sketch, p1).min(cm4_nibble(sketch, p2))
}

#[inline]
fn cm4_inc_conservative(sketch: &mut [u8], hash: u64) -> u8 {
    let (p1, p2) = cm4_slots(hash);
    let (c1, c2) = (cm4_nibble(sketch, p1), cm4_nibble(sketch, p2));
    let min_c = c1.min(c2);
    let next_c = min_c.saturating_add(1).min(15);
    for (pos, cnt) in [(p1, c1), (p2, c2)] {
        if cnt == min_c && cnt < 15 {
            let shift = (pos & 1) * 4;
            sketch[pos >> 1] = (sketch[pos >> 1] & !(0x0f << shift)) | (next_c << shift);
        }
    }
    next_c
}

#[inline]
fn file_bloom_bits(hash: u64) -> [(usize, u64); 6] {
    let h_fb = rapid_hash_u64(hash, SEED_FB);
    std::array::from_fn(|slot| {
        let bit = ((h_fb >> (slot * 10)) as usize) & 1023;
        (bit >> 6, 1u64 << (bit & 63))
    })
}

#[inline]
fn stamp_table_insert(table: &mut [(u64, u32); 256], hash: u64, stamp: u32) -> bool {
    let mut slot = (hash as usize) & 255;
    for _ in 0..8 {
        if table[slot].1 != stamp {
            table[slot] = (hash, stamp);
            return true;
        }
        if table[slot].0 == hash { return false; }
        slot = (slot + 1) & 255;
    }
    true
}

#[inline]
fn bloom3_test_and_set(
    bits: &mut FixedBitSet, hash: u64, s1: u32, mul: u64, s2: u32, mask: usize,
) -> bool {
    let ps = [
        (hash as usize) & mask, ((hash >> s1) as usize) & mask,
        ((hash.wrapping_mul(mul) >> s2) as usize) & mask,
    ];
    if ps.iter().all(|&p| bits.contains(p)) { return true; }
    for pos in ps { bits.insert(pos); }
    false
}

struct BloomScratch {
    kg_tab: [(u64, u32); 256], wn_tab: [(u64, u32); 256],
    feats: Vec<LineFeat>, raw_kgrams: Vec<(u64, u32)>, kgrams: Vec<(u64, u32)>,
    boiler_mask: Vec<bool>, file_winnow_uniq: Vec<u64>,
}

impl BloomScratch {
    fn new() -> Self {
        Self {
            kg_tab: [(0, 0); 256], wn_tab: [(0, 0); 256],
            feats: Vec::with_capacity(256), raw_kgrams: Vec::with_capacity(256),
            kgrams: Vec::with_capacity(256), boiler_mask: Vec::with_capacity(256),
            file_winnow_uniq: Vec::with_capacity(256),
        }
    }

    fn filter_boiler(&mut self, df_sketch: &[u8]) {
        apply_boiler_mask(
            self.feats.len(), &self.raw_kgrams, &mut self.boiler_mask, &mut self.kgrams,
            |h| cm4_get(df_sketch, h) >= DF_BOILER_CAP,
        );
    }

    fn win_slice(&self, filtered: bool) -> &[(u64, u32)] {
        if filtered { &self.kgrams } else { &self.raw_kgrams }
    }
}

struct BloomPass1 {
    df_sketch: Vec<u8>, file_bloom: Vec<u64>, file_has_boiler: FixedBitSet, dup_hashes: Vec<u64>,
}

fn bloom_build_pass1_kgrams(
    scratch: &mut BloomScratch, df_sketch: &mut [u8], file_has_boiler: &mut FixedBitSet, fid: usize,
) -> bool {
    let (stamp, mut has_boiler) = ((fid as u32) + 1, false);
    collect_valid_kgrams(&scratch.feats, &mut scratch.raw_kgrams, |hash| {
        if stamp_table_insert(&mut scratch.kg_tab, hash, stamp)
            && cm4_inc_conservative(df_sketch, hash) >= DF_BOILER_CAP
        { has_boiler = true; }
    });
    if !has_boiler { return false; }
    file_has_boiler.insert(fid);
    scratch.filter_boiler(df_sketch);
    true
}

fn record_file_winnow(
    fid: usize, hashes: &[u64], df_sketch: &[u8], file_bloom: &mut [u64],
    l1_seen: &mut FixedBitSet, l2_dup: &mut FixedBitSet, dup_hashes: &mut Vec<u64>,
) {
    let fb_base = (fid & (FB_BUCKETS - 1)) * FB_WORDS;
    for &hash in hashes {
        for (word, mask) in file_bloom_bits(hash) { file_bloom[fb_base + word] |= mask; }
        let seen_before = bloom3_test_and_set(l1_seen, hash, 23, 0x517c_c1b7_2722_0a95, 41, BLOOM_MASK);
        if seen_before && (2..DF_BOILER_CAP).contains(&cm4_get(df_sketch, hash)) {
            let l2_key = rapid_hash_u64(hash, SEED_L2);
            if !bloom3_test_and_set(l2_dup, l2_key, 22, SEED_CM, 42, L2_MASK) {
                dup_hashes.push(hash);
            }
        }
    }
}

fn run_bloom_pass1(
    files: &[(&str, &[u8])], log2_lut: &[f32; 256], scratch: &mut BloomScratch,
) -> BloomPass1 {
    let (mut l1_seen, mut l2_dup) =
        (FixedBitSet::with_capacity(BLOOM_BITS), FixedBitSet::with_capacity(L2_BITS));
    let (mut file_bloom, mut df_sketch) = (vec![0u64; FB_BUCKETS * FB_WORDS], vec![0u8; CM_SLOTS / 2]);
    let (mut file_has_boiler, mut dup_hashes) =
        (FixedBitSet::with_capacity(files.len()), Vec::with_capacity(1024));

    for (fid, &(_, bytes)) in files.iter().enumerate() {
        let stamp = (fid as u32) + 1;
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let filtered = bloom_build_pass1_kgrams(scratch, &mut df_sketch, &mut file_has_boiler, fid);

        scratch.file_winnow_uniq.clear();
        let (wn_tab, uniq) = (&mut scratch.wn_tab, &mut scratch.file_winnow_uniq);
        let slice = if filtered { &scratch.kgrams } else { &scratch.raw_kgrams };
        winnow_contiguous(slice, |hash, _| {
            if stamp_table_insert(wn_tab, hash, stamp) { uniq.push(hash); }
        });
        record_file_winnow(
            fid, &scratch.file_winnow_uniq, &df_sketch, &mut file_bloom,
            &mut l1_seen, &mut l2_dup, &mut dup_hashes,
        );
    }
    BloomPass1 { df_sketch, file_bloom, file_has_boiler, dup_hashes }
}

fn bloom_build_pass2_kgrams(scratch: &mut BloomScratch, df_sketch: &[u8], may_boiler: bool) -> bool {
    let mut has_b = false;
    collect_valid_kgrams(&scratch.feats, &mut scratch.raw_kgrams, |hash| {
        if may_boiler && cm4_get(df_sketch, hash) >= DF_BOILER_CAP { has_b = true; }
    });
    if has_b { scratch.filter_boiler(df_sketch); }
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
    if surviving_dups.is_empty() { return Vec::new(); }

    let cand_bucket = select_candidate_buckets(files.len(), &surviving_dups, &pass1.file_bloom);
    drop(pass1.file_bloom);

    let mut postings = Vec::new();
    for (fid, &(_, bytes)) in files.iter().enumerate() {
        if !cand_bucket[fid & (FB_BUCKETS - 1)] { continue; }
        scan_bytes_feats(bytes, log2_lut, &mut scratch.feats);
        let may_boiler = pass1.file_has_boiler.contains(fid) || fid < 1024;
        let filtered = bloom_build_pass2_kgrams(&mut scratch, &pass1.df_sketch, may_boiler);
        winnow_contiguous(scratch.win_slice(filtered), |hash, pos| {
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
    n_files: usize, surviving_dups: &FxHashSet<u64>, file_bloom: &[u64],
) -> Vec<bool> {
    let (mut cand, active) = (vec![false; FB_BUCKETS], n_files.min(FB_BUCKETS));
    if surviving_dups.len() > 512 {
        cand[..active].fill(true);
        return cand;
    }
    let probes: Vec<[(usize, u64); 6]> = surviving_dups.iter().map(|&h| file_bloom_bits(h)).collect();
    for (bkt, flag) in cand.iter_mut().enumerate().take(active) {
        let slice = &file_bloom[bkt * FB_WORDS..(bkt + 1) * FB_WORDS];
        *flag = probes.iter().any(|bits| bits.iter().all(|&(w, m)| (slice[w] & m) != 0));
    }
    cand
}

fn collect_cluster_occ(
    files: &[(&str, &[u8])], group: &[(u64, u32, u32)],
) -> (Vec<(String, usize)>, (usize, usize)) {
    let (mut occ, mut last_fid, mut sample) = (Vec::new(), u32::MAX, (0usize, 0usize));
    for &(_, fid, line_idx) in group {
        if fid == last_fid { continue; }
        if occ.is_empty() { sample = (fid as usize, line_idx as usize); }
        occ.push((files[fid as usize].0.to_string(), (line_idx as usize) + 1));
        last_fid = fid;
    }
    (occ, sample)
}

fn dedup_clone_clusters(clusters: Vec<CloneCluster>) -> Vec<CloneCluster> {
    let mut out: Vec<CloneCluster> = Vec::new();
    for cluster in clusters {
        let overlaps = out.iter().any(|prev| {
            prev.occurrences.len() == cluster.occurrences.len()
                && prev.occurrences.iter().zip(&cluster.occurrences)
                    .all(|((pf, pl), (cf, cl))| pf == cf && pl.abs_diff(*cl) <= 4)
        });
        if overlaps { continue; }
        out.push(cluster);
        if out.len() >= 25 { break; }
    }
    out
}

fn assemble_clusters(files: &[(&str, &[u8])], mut postings: Vec<(u64, u32, u32)>) -> Vec<CloneCluster> {
    postings.sort_unstable_by_key(|&(h, fid, line)| (h, fid, line));
    let (mut clusters, mut idx) = (Vec::new(), 0usize);

    while idx < postings.len() {
        let (hash, start) = (postings[idx].0, idx);
        while idx < postings.len() && postings[idx].0 == hash { idx += 1; }
        let (mut occ, (sample_fid, sample_row)) = collect_cluster_occ(files, &postings[start..idx]);
        if occ.len() < 2 || (occ.len() as u8) >= DF_BOILER_CAP { continue; }
        occ.sort();
        let raw_sample = extract_line(files[sample_fid].1, sample_row);
        let sample_line = raw_sample.chars().take(100).collect();
        let fingerprint = format!("{:012x}", hash & 0x0000_ffff_ffff_ffff);
        clusters.push(CloneCluster { fingerprint, line_count: 4, sample_line, occurrences: occ });
    }

    clusters.sort_by(|a, b| {
        b.occurrences.len().cmp(&a.occurrences.len())
            .then_with(|| a.occurrences.cmp(&b.occurrences))
            .then_with(|| a.fingerprint.cmp(&b.fingerprint))
    });
    dedup_clone_clusters(clusters)
}

pub fn find_clones(source_rels: &[String], preloaded: &BTreeMap<String, Vec<u8>>) -> Vec<CloneCluster> {
    let pairs: Vec<(&str, &[u8])> = source_rels.iter()
        .filter_map(|rel| preloaded.get(rel).map(|b| (rel.as_str(), b.as_slice()))).collect();
    find_clones_from_slices(&pairs)
}

pub fn find_clones_from_slices(files: &[(&str, &[u8])]) -> Vec<CloneCluster> {
    if files.len() < 2 { return Vec::new(); }
    let log2_lut = build_log2_lut();
    if files.len() < EXACT_CLONE_FILE_CAP { find_clones_exact(files, &log2_lut) }
    else { find_clones_bloom(files, &log2_lut) }
}

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
pub struct SimiItem<'a> { pub name: Cow<'a, str>, pub body: &'a [u8] }
#[derive(Debug, Clone, Copy)]
pub struct SimiMatch { pub a: usize, pub b: usize }
#[derive(Debug, Clone)]
pub struct SimiStats { pub matches: Vec<SimiMatch>, pub lsh_candidates: usize, pub hot_buckets_capped: usize }

fn accum_simhash_bits(vec: &mut [f32; 64], hash: u64, weight: f32) {
    for (bit, acc) in vec.iter_mut().enumerate() {
        *acc += if (hash & (1u64 << bit)) != 0 { weight } else { -weight };
    }
}

fn simhash64_idf(tok_weights: &[(u64, f32)]) -> u64 {
    let mut vec = [0.0f32; 64];
    for &(hash, weight) in tok_weights { accum_simhash_bits(&mut vec, hash, weight); }
    vec.iter().enumerate().fold(0u64, |fp, (bit, &acc)| if acc > 0.0 { fp | (1u64 << bit) } else { fp })
}

fn weighted_jaccard(left: &[(u64, f32)], right: &[(u64, f32)]) -> f64 {
    let (mut i, mut j, mut inter, mut uni) = (0usize, 0usize, 0.0f64, 0.0f64);
    while i < left.len() && j < right.len() {
        let (lk, rk) = (left[i].0, right[j].0);
        if lk < rk {
            uni += left[i].1 as f64;
            i += 1;
        } else if lk > rk {
            uni += right[j].1 as f64;
            j += 1;
        } else {
            inter += left[i].1.min(right[j].1) as f64;
            uni += left[i].1.max(right[j].1) as f64;
            i += 1;
            j += 1;
        }
    }
    let tail = left[i..].iter().chain(&right[j..]);
    uni += tail.map(|&(_, w)| w as f64).sum::<f64>();
    if uni > 0.0 { inter / uni } else { 0.0 }
}

fn hash_domain_ident(raw: &[u8], stack_buf: &mut [u8; 64], filter_syntax: bool) -> Option<u64> {
    if raw.len() <= stack_buf.len() {
        for (dst, &src) in stack_buf[..raw.len()].iter_mut().zip(raw) {
            *dst = src.to_ascii_lowercase();
        }
        let lower = &stack_buf[..raw.len()];
        if filter_syntax && SYNTAX_STOP.contains(&lower) { return None; }
        return Some(rapid_hash64(lower, 0x333));
    }
    let owned: Vec<u8> = raw.iter().map(u8::to_ascii_lowercase).collect();
    Some(rapid_hash64(&owned, 0x333))
}

fn extract_domain_token_hashes(body: &[u8], filter_syntax: bool) -> Vec<u64> {
    let (mut toks, mut stack_buf) = (Vec::with_capacity(24), [0u8; 64]);
    for raw in iter_idents(body).filter(|r| r.len() >= 2) {
        if let Some(hash) = hash_domain_ident(raw, &mut stack_buf, filter_syntax) {
            toks.push(hash);
        }
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
        for &hash in &toks { *tok_df.entry(hash).or_insert(0) += 1; }
        raw_doc_toks.push(toks);
    }
    let (df_stop_cap, n_f) = ((n / 10).max(8) as u32, (n as f64) + 1.0);
    let (mut doc_vecs, mut fps) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for toks in raw_doc_toks {
        let weighted: Vec<(u64, f32)> = toks.into_iter().filter_map(|hash| {
            let df = tok_df.get(&hash).copied().unwrap_or(1);
            (df <= df_stop_cap).then(|| (hash, ((n_f / ((df as f64) + 1.0)).ln() + 1.0) as f32))
        }).collect();
        fps.push(if weighted.len() >= 3 { simhash64_idf(&weighted) } else { 0 });
        doc_vecs.push(weighted);
    }
    (doc_vecs, fps)
}

fn collect_pairwise_cands(fps: &[u64]) -> FxHashSet<(usize, usize)> {
    let mut cands = FxHashSet::default();
    for lo in 0..fps.len() {
        let fp_a = fps[lo];
        if fp_a == 0 { continue; }
        for (offset, &fp_b) in fps[lo + 1..].iter().enumerate() {
            if fp_b != 0 && (fp_a ^ fp_b).count_ones() <= HAMMING_MAX {
                cands.insert((lo, lo + 1 + offset));
            }
        }
    }
    cands
}

fn add_bucket_pairs(bucket: &[u32], fps: &[u64], cands: &mut FxHashSet<(usize, usize)>) {
    for (i, &u) in bucket.iter().enumerate() {
        for &v in &bucket[i + 1..] {
            let (lo, hi) = (u.min(v) as usize, u.max(v) as usize);
            if (fps[lo] ^ fps[hi]).count_ones() <= HAMMING_MAX {
                cands.insert((lo, hi));
            }
        }
    }
}

fn collect_candidates(fps: &[u64]) -> (FxHashSet<(usize, usize)>, usize) {
    if fps.len() <= DIRECT_PAIRWISE_MAX {
        return (collect_pairwise_cands(fps), 0);
    }

    let mut bands: SparseLshBands = FxHashMap::with_capacity_and_hasher(fps.len() * 4, Default::default());
    for (idx, &fp) in fps.iter().enumerate().filter(|&(_, &f)| f != 0) {
        for band_idx in 0..N_BANDS {
            let key = ((fp.rotate_left((band_idx * 4) as u32)) & 0xff) as u16;
            bands.entry(((band_idx as u16) << 8) | key).or_default().push(idx as u32);
        }
    }
    let (mut cands, mut hot_capped) = (FxHashSet::default(), 0usize);
    for bucket in bands.values().filter(|b| b.len() >= 2) {
        if bucket.len() > B_MAX {
            hot_capped += 1;
            continue;
        }
        add_bucket_pairs(bucket, fps, &mut cands);
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
            && (jaro_winkler(&items[lo].name, &items[hi].name) >= JARO_WINKLER_MIN
                || wj >= STRONG_BODY_JACCARD);
        ok.then_some(SimiMatch { a: lo, b: hi })
    }).collect();
    matches.sort_by_key(|m| (m.a, m.b));
    SimiStats { matches, lsh_candidates: lsh_cands.len(), hot_buckets_capped }
}

fn split_ident_lower(src: &str) -> Vec<String> {
    let (mut tokens, mut cur) = (Vec::new(), String::new());
    for ch in src.chars() {
        let camel = ch.is_ascii_uppercase() && cur.chars().last().is_some_and(|c| c.is_ascii_lowercase());
        if !ch.is_ascii_alphanumeric() {
            if !cur.is_empty() {
                tokens.push(cur.to_ascii_lowercase());
                cur.clear();
            }
        } else if camel {
            tokens.push(cur.to_ascii_lowercase());
            cur.clear();
            cur.push(ch);
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() { tokens.push(cur.to_ascii_lowercase()); }
    tokens
}

fn canonical_stem(rel: &str) -> String {
    let (path, stem) = (Path::new(rel), stem_of(Path::new(rel)));
    if matches!(stem, "index" | "mod" | "lib" | "main")
        && let Some(dir) = path.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str())
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
pub fn find_capability_clusters(
    source_rels: &[String], preloaded: &BTreeMap<String, Vec<u8>>,
) -> Vec<RoleCluster> {
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
    fn find_root(par: &mut [usize], mut x: usize) -> usize {
        while par[x] != x {
            par[x] = par[par[x]];
            x = par[x];
        }
        x
    }
    for mat in &stats.matches {
        let (ra, rb) = (find_root(&mut parent, mat.a), find_root(&mut parent, mat.b));
        if ra != rb { parent[rb] = ra; }
    }
    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (i, rel) in source_rels.iter().enumerate() {
        groups.entry(find_root(&mut parent, i)).or_default().push(rel.clone());
    }
    let mut clusters: Vec<RoleCluster> = groups.into_values().filter(|g| g.len() >= 2).map(|mut files| {
        files.sort();
        let cap = infer_capability_label(&files);
        let rec = format!(
            "Detected {} structurally similar `{cap}` modules; consolidate into one canonical module",
            files.len()
        );
        RoleCluster { recommendation: rec, capability: cap, files }
    }).collect();
    clusters.sort_by(|a, b| {
        b.files.len().cmp(&a.files.len()).then_with(|| a.capability.cmp(&b.capability))
            .then_with(|| a.files.cmp(&b.files))
    });
    clusters.truncate(20);
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_simi_items(owned: &[(String, String)]) -> Vec<SimiItem<'_>> {
        owned.iter().map(|(n, b)| SimiItem { name: Cow::Borrowed(n.as_str()), body: b.as_bytes() }).collect()
    }

    fn has_clone_pair(clusters: &[CloneCluster], a: &str, b: &str) -> bool {
        clusters.iter().any(|c| {
            c.occurrences.iter().any(|(f, _)| f == a) && c.occurrences.iter().any(|(f, _)| f == b)
        })
    }

    fn boiler_file_body(fid: usize) -> String {
        const OPS: [&str; 6] = ["+", "-", "*", "^", "|", "&"];
        let mut lines: Vec<String> = match fid {
            0..16 => (0..6).map(|_| "    a = 0; a = 0; a = 0; a = 0; a = 0;".into()).collect(),
            16..32 => (0..6).map(|ln| {
                format!("    this.fieldSlot_{fid}_{ln} = inputParam_{fid}_{ln}; return this.fieldSlot_{fid}_{ln};")
            }).collect(),
            _ => Vec::new(),
        };
        for ln in lines.len()..20 {
            let h = rapid_hash_u64(rapid_hash_u64(((fid as u64) << 16) | (ln as u64), 0xabc1), 0xdef2);
            let (o1, o2, c) = (OPS[(h as usize) % 6], OPS[((h >> 4) as usize) % 6], ((h >> 8) % 997) as usize);
            lines.push(format!("const uniq_{fid}_{ln} = calc_{fid}(in_{ln} {o1} {c}, ctx_{fid}) {o2} salt_{ln};"));
        }
        lines.join("\n")
    }

    fn build_boiler_fixture() -> Vec<(String, String)> {
        (0..48usize).map(|fid| (format!("src/boiler_{fid:02}.ts"), boiler_file_body(fid))).collect()
    }

    #[test]
    fn clones_type1_type2_no_boiler() {
        let mut owned = build_boiler_fixture();

        let gen8 = |fmt: fn(usize) -> String| -> String { (0..8).map(fmt).collect() };
        let t1 = gen8(|ln| format!("const acc_{ln} = validatePayload(req.body, ctx.tenant) + computeDigest({ln}, 31) ^ 7;\n"));
        let t2_a = gen8(|ln| format!("const userRec_{ln} = verifySignature(rawToken, 11) * normalizeEpoch(userRec, {ln}) + 19;\n"));
        let t2_b = gen8(|ln| format!("const acctObj_{ln} = verifySignature(jwtSlice, 11) * normalizeEpoch(acctObj, {ln}) + 19;\n"));
        owned.extend([
            ("src/t1_a.ts".into(), t1.clone()), ("src/t1_b.ts".into(), t1),
            ("src/t2_a.ts".into(), t2_a), ("src/t2_b.ts".into(), t2_b),
        ]);

        let slices: Vec<(&str, &[u8])> = owned.iter().map(|(p, b)| (p.as_str(), b.as_bytes())).collect();
        let lut = build_log2_lut();
        for (label, clusters) in [("exact", find_clones_from_slices(&slices)), ("bloom", find_clones_bloom(&slices, &lut))] {
            let boiler_fp = clusters.iter()
                .filter(|c| c.occurrences.iter().any(|(f, ln)| f.contains("boiler_") && *ln <= 6)).count();
            assert_eq!(boiler_fp, 0, "[{label}] boilerplate leaked: {clusters:?}");
            assert!(has_clone_pair(&clusters, "src/t1_a.ts", "src/t1_b.ts"), "[{label}] missed Type-1");
            assert!(has_clone_pair(&clusters, "src/t2_a.ts", "src/t2_b.ts"), "[{label}] missed Type-2");
        }
    }

    #[test]
    fn simiflow_idf_bmax32_precision() {
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
        owned.extend((50..1_000).map(|i| (
            format!("domainSpecificOp{i:04}"),
            format!("const return async await unique pipeline transform stage alpha_{} beta_{} gamma_{}",
                i * 3 + 1, i * 7 + 2, i * 13 + 3),
        )));
        let stats = find_similar_items(&to_simi_items(&owned), false);
        assert!(stats.lsh_candidates <= 150, "too many LSH candidates: {}", stats.lsh_candidates);
        let true_hits = stats.matches.iter().filter(|m| m.a < 50 && m.a % 2 == 0 && m.b == m.a + 1).count();
        assert!(true_hits >= 24, "expected >= 24/25 domain twins, got {true_hits}");
        assert_eq!(stats.matches.len(), true_hits);

        let mut skew: Vec<(String, String)> = (0..45)
            .map(|i| (format!("hotBucketTwin_{i}"), "merkle digest verify payload signature nonce".into())).collect();
        skew.extend((45..500).map(|i| (format!("distinctOp_{i}"), format!("tok_a_{i} tok_b_{i} tok_c_{i} tok_d_{i}"))));
        let skew_stats = find_similar_items(&to_simi_items(&skew), false);
        assert!(skew_stats.hot_buckets_capped >= N_BANDS && skew_stats.matches.is_empty());
    }

    #[test]
    fn simiflow_matches_auth_bodies() {
        let block = concat!(
            "    const normalized = raw.trim().toLowerCase();\n",
            "    const parts = normalized.split(':');\n",
            "    const checksum = parts.length * 31;\n",
            "    const output = checksum + normalized.length;\n",
        );
        let preloaded = BTreeMap::from([
            ("src/auth_v1.ts".into(), format!("export function authOne(raw: string) {{\n{block}    return output;\n}}\n").into_bytes()),
            ("src/auth_v2.ts".into(), format!("export function authTwo(raw: string) {{\n{block}    return output;\n}}\n").into_bytes()),
            ("src/app.py".into(), b"import os\n".to_vec()),
            ("src/main.go".into(), b"package main\nfunc main() {}\n".to_vec()),
        ]);
        let rels: Vec<String> = preloaded.keys().cloned().collect();
        let clusters = find_capability_clusters(&rels, &preloaded);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].files, vec!["src/auth_v1.ts", "src/auth_v2.ts"]);
    }
}
