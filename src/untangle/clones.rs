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

/// Zero-allocation stack-only fused Type-2 alpha-renaming scanner + `LOG2_LUT` Shannon entropy.
fn scan_line_type2(raw: &[u8], lut: &[f32; 256]) -> LineFeat {
    let mut start_trim = 0usize;
    while start_trim < raw.len() && raw[start_trim].is_ascii_whitespace() {
        start_trim += 1;
    }
    let mut end_trim = raw.len();
    while end_trim > start_trim && raw[end_trim - 1].is_ascii_whitespace() {
        end_trim -= 1;
    }
    let bytes = &raw[start_trim..end_trim];
    if is_ignored_line_prefix(bytes) {
        return LineFeat { norm_hash: 0, valid: false };
    }

    let mut counts = [0u16; 128];
    let mut total = 0usize;
    for &b in bytes {
        counts[(b & 0x7f) as usize] += 1;
        total += 1;
    }
    let mut sum_c_log_c = 0.0f32;
    for &c in &counts {
        if c > 0 {
            sum_c_log_c += lut[(c as usize).min(255)];
        }
    }
    let n_f = total as f32;
    let n_log_n = if total < 256 { lut[total] } else { n_f * n_f.log2() };
    let ent = (n_log_n - sum_c_log_c) / n_f;
    if ent < 2.75 {
        return LineFeat { norm_hash: 0, valid: false };
    }

    let mut h = 0x99u64;
    let mut i = 0usize;
    let mut var_slot = 0u32;
    let mut local_map: [(u64, u32); 16] = [(0, 0); 16];
    let mut local_len = 0usize;
    let mut distinct_toks = 0u16;
    let mut op_count = 0u16;

    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if matches!(b, b'"' | b'\'' | b'`') {
            let q = b;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                let c = bytes[i];
                i += 1;
                if c == q {
                    break;
                }
            }
            h = rapid_hash_u64(0x2000_0001u64, h);
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let tok = &bytes[start..i];
            if is_clone_keyword(tok) {
                let kh = rapid_hash64(tok, 0x11);
                h = rapid_hash_u64(kh, h);
                distinct_toks += 1;
            } else {
                let th = rapid_hash64(tok, 0x77);
                let mut found = None;
                for &(k, v) in &local_map[..local_len] {
                    if k == th {
                        found = Some(v);
                        break;
                    }
                }
                let id = match found {
                    Some(v) => v,
                    None => {
                        let v = var_slot;
                        var_slot += 1;
                        distinct_toks += 1;
                        if local_len < local_map.len() {
                            local_map[local_len] = (th, v);
                            local_len += 1;
                        }
                        v
                    }
                };
                h = rapid_hash_u64(0x1000_0000u64 | (id as u64), h);
            }
            continue;
        }
        if b.is_ascii_digit() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            h = rapid_hash_u64(0x2000_0000u64, h);
            continue;
        }
        h = rapid_hash_u64(0x3000_0000u64 | (b as u64), h);
        if matches!(
            b,
            b'+' | b'-' | b'*' | b'/' | b'^' | b'|' | b'&' | b'=' | b'<' | b'>' | b'?' | b'(' | b'.'
        ) {
            op_count += 1;
        }
        i += 1;
    }

    LineFeat {
        norm_hash: h,
        valid: distinct_toks >= 4 && op_count >= 2,
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
        let run = &win_slice[run_start..run_end];
        if run.len() < 3 {
            let mut best = run[0];
            for &item in &run[1..] {
                if item.0 <= best.0 {
                    best = item;
                }
            }
            on_pick(best.0, best.1);
        } else {
            let mut last_picked = u32::MAX;
            for w in 0..=run.len() - 3 {
                let mut best = run[w];
                if run[w + 1].0 <= best.0 {
                    best = run[w + 1];
                }
                if run[w + 2].0 <= best.0 {
                    best = run[w + 2];
                }
                if best.1 != last_picked {
                    last_picked = best.1;
                    on_pick(best.0, best.1);
                }
            }
        }
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
    let mut l1_seen = vec![0u64; BLOOM_BITS / 64]; // 1.00 MiB
    let mut l2_dup = vec![0u64; L2_BITS / 64]; // 0.50 MiB
    let mut df_sketch = vec![0u8; CM_SLOTS / 2]; // 0.50 MiB
    let mut file_bloom = vec![0u64; FB_BUCKETS * FB_WORDS]; // 1.00 MiB
    let mut file_has_boiler = vec![0u64; n_files.div_ceil(64)];
    let mut dup_hashes: Vec<u64> = Vec::with_capacity(1024);

    let mut kg_tab: [(u64, u32); 256] = [(0, 0); 256];
    let mut wn_tab: [(u64, u32); 256] = [(0, 0); 256];
    let mut feats: Vec<LineFeat> = Vec::with_capacity(256);
    let mut raw_kgrams: Vec<(u64, u32)> = Vec::with_capacity(256);
    let mut kgrams: Vec<(u64, u32)> = Vec::with_capacity(256);
    let mut boiler_mask: Vec<bool> = Vec::with_capacity(256);
    let mut file_winnow_uniq: Vec<u64> = Vec::with_capacity(256);

    // Pass 1: Fused Type-2 scan + CM4-DF + all-4-lines boilerplate mask + File-Bloom + 2L Bloom.
    for (fid, &(_, bytes)) in files.iter().enumerate() {
        let stamp = (fid as u32) + 1;
        scan_bytes_feats(bytes, &log2_lut, &mut feats);
        raw_kgrams.clear();
        let mut has_boiler_kgram = false;

        for i in 0..feats.len().saturating_sub(3) {
            if !(feats[i].valid && feats[i + 1].valid && feats[i + 2].valid && feats[i + 3].valid) {
                continue;
            }
            let h = kgram_hash(&feats, i);
            raw_kgrams.push((h, i as u32));

            let mut slot = (h as usize) & 255;
            let mut is_new = true;
            for _ in 0..8 {
                if kg_tab[slot].1 != stamp {
                    kg_tab[slot] = (h, stamp);
                    break;
                }
                if kg_tab[slot].0 == h {
                    is_new = false;
                    break;
                }
                slot = (slot + 1) & 255;
            }
            if is_new && cm4_inc_conservative(&mut df_sketch, h) >= DF_BOILER_CAP {
                has_boiler_kgram = true;
            }
        }

        let win_slice: &[(u64, u32)] = if has_boiler_kgram {
            file_has_boiler[fid >> 6] |= 1u64 << (fid & 63);
            boiler_mask.clear();
            boiler_mask.resize(feats.len(), false);
            for &(h, pos) in &raw_kgrams {
                if cm4_get(&df_sketch, h) >= DF_BOILER_CAP {
                    let i = pos as usize;
                    boiler_mask[i] = true;
                    boiler_mask[i + 1] = true;
                    boiler_mask[i + 2] = true;
                    boiler_mask[i + 3] = true;
                }
            }
            kgrams.clear();
            for &(h, pos) in &raw_kgrams {
                let i = pos as usize;
                if !(boiler_mask[i]
                    || boiler_mask[i + 1]
                    || boiler_mask[i + 2]
                    || boiler_mask[i + 3])
                {
                    kgrams.push((h, pos));
                }
            }
            &kgrams
        } else {
            &raw_kgrams
        };

        file_winnow_uniq.clear();
        winnow_contiguous(win_slice, |h, _| {
            let mut slot = (h as usize) & 255;
            let mut is_new = true;
            for _ in 0..8 {
                if wn_tab[slot].1 != stamp {
                    wn_tab[slot] = (h, stamp);
                    break;
                }
                if wn_tab[slot].0 == h {
                    is_new = false;
                    break;
                }
                slot = (slot + 1) & 255;
            }
            if is_new {
                file_winnow_uniq.push(h);
            }
        });

        let fb_base = (fid & (FB_BUCKETS - 1)) * FB_WORDS;
        for &h in &file_winnow_uniq {
            let h_fb = rapid_hash_u64(h, SEED_FB);
            for j in 0..6 {
                let b = ((h_fb >> (j * 10)) as usize) & 1023;
                file_bloom[fb_base + (b >> 6)] |= 1u64 << (b & 63);
            }

            let p1 = (h as usize) & BLOOM_MASK;
            let p2 = ((h >> 23) as usize) & BLOOM_MASK;
            let p3 = ((h.wrapping_mul(0x517c_c1b7_2722_0a95) >> 41) as usize) & BLOOM_MASK;
            let s1 = (l1_seen[p1 >> 6] & (1u64 << (p1 & 63))) != 0;
            let s2 = (l1_seen[p2 >> 6] & (1u64 << (p2 & 63))) != 0;
            let s3 = (l1_seen[p3 >> 6] & (1u64 << (p3 & 63))) != 0;
            if s1 && s2 && s3 {
                let df = cm4_get(&df_sketch, h);
                if (2..DF_BOILER_CAP).contains(&df) {
                    let h_l2 = rapid_hash_u64(h, SEED_L2);
                    let q1 = (h_l2 as usize) & L2_MASK;
                    let q2 = ((h_l2 >> 22) as usize) & L2_MASK;
                    let q3 = ((h_l2.wrapping_mul(SEED_CM) >> 42) as usize) & L2_MASK;
                    let d1 = (l2_dup[q1 >> 6] & (1u64 << (q1 & 63))) != 0;
                    let d2 = (l2_dup[q2 >> 6] & (1u64 << (q2 & 63))) != 0;
                    let d3 = (l2_dup[q3 >> 6] & (1u64 << (q3 & 63))) != 0;
                    if !(d1 && d2 && d3) {
                        l2_dup[q1 >> 6] |= 1u64 << (q1 & 63);
                        l2_dup[q2 >> 6] |= 1u64 << (q2 & 63);
                        l2_dup[q3 >> 6] |= 1u64 << (q3 & 63);
                        dup_hashes.push(h);
                    }
                }
            } else {
                l1_seen[p1 >> 6] |= 1u64 << (p1 & 63);
                l1_seen[p2 >> 6] |= 1u64 << (p2 & 63);
                l1_seen[p3 >> 6] |= 1u64 << (p3 & 63);
            }
        }
    }
    drop(l1_seen);
    drop(l2_dup);

    let mut surviving_dups: FxHashSet<u64> =
        FxHashSet::with_capacity_and_hasher(dup_hashes.len(), Default::default());
    let mut surv_bloom = vec![0u64; SB_BITS / 64];
    for h in dup_hashes {
        let df = cm4_get(&df_sketch, h);
        if (2..DF_BOILER_CAP).contains(&df) {
            surviving_dups.insert(h);
            let b1 = (h as usize) & SB_MASK;
            let b2 = ((h >> 19) as usize) & SB_MASK;
            surv_bloom[b1 >> 6] |= 1u64 << (b1 & 63);
            surv_bloom[b2 >> 6] |= 1u64 << (b2 & 63);
        }
    }
    if surviving_dups.is_empty() {
        return Vec::new();
    }

    // Adaptive File-Bloom candidate selection (skips non-matching buckets in Pass 2).
    let mut cand_bucket = vec![false; FB_BUCKETS];
    let active_buckets = n_files.min(FB_BUCKETS);
    if surviving_dups.len() <= 512 {
        #[derive(Clone, Copy)]
        struct FbProbe {
            w: [u8; 6],
            m: [u64; 6],
        }
        let probes: Vec<FbProbe> = surviving_dups
            .iter()
            .map(|&h| {
                let h_fb = rapid_hash_u64(h, SEED_FB);
                let mut w = [0u8; 6];
                let mut m = [0u64; 6];
                for j in 0..6 {
                    let b = ((h_fb >> (j * 10)) as usize) & 1023;
                    w[j] = (b >> 6) as u8;
                    m[j] = 1u64 << (b & 63);
                }
                FbProbe { w, m }
            })
            .collect();
        for (bkt, flag) in cand_bucket.iter_mut().enumerate().take(active_buckets) {
            let base = bkt * FB_WORDS;
            let slice = &file_bloom[base..base + FB_WORDS];
            for p in &probes {
                if (slice[(p.w[0] & 15) as usize] & p.m[0]) != 0
                    && (slice[(p.w[1] & 15) as usize] & p.m[1]) != 0
                    && (slice[(p.w[2] & 15) as usize] & p.m[2]) != 0
                    && (slice[(p.w[3] & 15) as usize] & p.m[3]) != 0
                    && (slice[(p.w[4] & 15) as usize] & p.m[4]) != 0
                    && (slice[(p.w[5] & 15) as usize] & p.m[5]) != 0
                {
                    *flag = true;
                    break;
                }
            }
        }
    } else {
        cand_bucket[..active_buckets].fill(true);
    }
    drop(file_bloom);

    // Pass 2 (Selective): re-scan only candidate buckets with `df_sketch` still alive!
    let mut postings: Vec<(u64, u32, u32)> = Vec::new();
    for (fid, &(_, bytes)) in files.iter().enumerate() {
        if !cand_bucket[fid & (FB_BUCKETS - 1)] {
            continue;
        }
        scan_bytes_feats(bytes, &log2_lut, &mut feats);
        raw_kgrams.clear();
        let may_have_boiler =
            (file_has_boiler[fid >> 6] & (1u64 << (fid & 63))) != 0 || fid < 1024;
        let win_slice: &[(u64, u32)] = if may_have_boiler {
            boiler_mask.clear();
            boiler_mask.resize(feats.len(), false);
            let mut has_boiler_kgram = false;
            for i in 0..feats.len().saturating_sub(3) {
                if !(feats[i].valid && feats[i + 1].valid && feats[i + 2].valid && feats[i + 3].valid) {
                    continue;
                }
                let h = kgram_hash(&feats, i);
                if cm4_get(&df_sketch, h) >= DF_BOILER_CAP {
                    has_boiler_kgram = true;
                    boiler_mask[i] = true;
                    boiler_mask[i + 1] = true;
                    boiler_mask[i + 2] = true;
                    boiler_mask[i + 3] = true;
                } else {
                    raw_kgrams.push((h, i as u32));
                }
            }
            if has_boiler_kgram {
                kgrams.clear();
                for &(h, pos) in &raw_kgrams {
                    let i = pos as usize;
                    if !(boiler_mask[i]
                        || boiler_mask[i + 1]
                        || boiler_mask[i + 2]
                        || boiler_mask[i + 3])
                    {
                        kgrams.push((h, pos));
                    }
                }
                &kgrams
            } else {
                &raw_kgrams
            }
        } else {
            for i in 0..feats.len().saturating_sub(3) {
                if !(feats[i].valid && feats[i + 1].valid && feats[i + 2].valid && feats[i + 3].valid) {
                    continue;
                }
                raw_kgrams.push((kgram_hash(&feats, i), i as u32));
            }
            &raw_kgrams
        };

        winnow_contiguous(win_slice, |h, pos| {
            let b1 = (h as usize) & SB_MASK;
            let b2 = ((h >> 19) as usize) & SB_MASK;
            if (surv_bloom[b1 >> 6] & (1u64 << (b1 & 63))) != 0
                && (surv_bloom[b2 >> 6] & (1u64 << (b2 & 63))) != 0
                && surviving_dups.contains(&h)
            {
                postings.push((h, fid as u32, pos));
            }
        });
    }
    drop(df_sketch);
    drop(surv_bloom);

    postings.sort_unstable_by_key(|&(h, fid, line)| (h, fid, line));

    let mut clusters: Vec<CloneCluster> = Vec::new();
    let mut idx = 0usize;
    while idx < postings.len() {
        let h = postings[idx].0;
        let start = idx;
        while idx < postings.len() && postings[idx].0 == h {
            idx += 1;
        }
        let mut occ: Vec<(String, usize)> = Vec::new();
        let mut last_fid = u32::MAX;
        let mut sample_fid = 0usize;
        let mut sample_line_idx = 0usize;
        for &(_, fid, line_idx) in &postings[start..idx] {
            if fid != last_fid {
                if occ.is_empty() {
                    sample_fid = fid as usize;
                    sample_line_idx = line_idx as usize;
                }
                occ.push((files[fid as usize].0.to_string(), (line_idx as usize) + 1));
                last_fid = fid;
            }
        }
        if occ.len() >= 2 && (occ.len() as u8) < DF_BOILER_CAP {
            occ.sort();
            let sample = extract_line(files[sample_fid].1, sample_line_idx);
            clusters.push(CloneCluster {
                fingerprint: format!("{:012x}", h & 0x0000_ffff_ffff_ffff),
                line_count: 4,
                sample_line: sample.chars().take(100).collect(),
                occurrences: occ,
            });
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
