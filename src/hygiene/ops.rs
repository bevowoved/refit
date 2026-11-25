use super::simd::{CsOp, DenseQkArgs, HygieneKernel, MAX_SPAN_LEN, RibbonSpan, TileScratch};
use super::weights::{AttnLayerWeights, CompactCsr, HeadKind, HeadPlan, ResolvedWeights};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

pub(super) const MAX_COLS: usize = 128;
pub(super) const SKETCH_DIM: usize = 40;
pub(super) const RIBBON_GUARD: usize = 6;
pub(super) const PEAK_WINDOW: usize = 32;

const BLOCK_SIZE: usize = 32;
const MAX_LANDMARKS: usize = 64;
const TOP_BLOCKS: usize = 4;
const SLAB_CAP: usize = 128;
const INDEX_SLOTS: usize = 1024;
const INDEX_WAYS: usize = 4;
pub(super) const FAR_INDEX_BYTES: usize =
    INDEX_SLOTS * INDEX_WAYS * std::mem::size_of::<(u16, u16)>() + INDEX_SLOTS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalOptions {
    pub approx_long: bool,
    pub approx_threshold: usize,
}

impl Default for EvalOptions {
    fn default() -> Self { Self { approx_long: false, approx_threshold: 1024 } }
}

#[derive(Debug, Clone, Default)]
pub struct SparseStats {
    pub attn_mode: &'static str,
    pub pairs_evaluated: usize,
    pub blocks: usize,
    pub far_hits: usize,
    pub index_bytes: usize,
    pub reused: bool,
    pub reuse_hits: usize,
    pub slab_reuses: usize,
}

pub(super) struct RingGate([f32; PEAK_WINDOW + 1]);
impl Default for RingGate { fn default() -> Self { Self([0.0; PEAK_WINDOW + 1]) } }
impl Deref for RingGate { type Target = [f32; PEAK_WINDOW + 1]; fn deref(&self) -> &Self::Target { &self.0 } }
impl DerefMut for RingGate { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 } }

pub(super) struct ByteLut {
    pub(super) map: [u8; 256],
    pub(super) fast_mask: [u8; 256],
    _pad: [u8; 3584],
}

pub(super) static BYTE_LUT: ByteLut = build_byte_lut();

const fn build_byte_lut() -> ByteLut {
    let (mut map, mut fast_mask, mut b) = ([127u8; 256], [0u8; 256], 0usize);
    while b < 128 {
        map[b] = if b == 0 { 32 } else { b as u8 };
        if b >= 0x20 { fast_mask[b] = 1; }
        b += 1;
    }
    ByteLut { map, fast_mask, _pad: [0u8; 3584] }
}

/// Map key vectors to far block indices via ternary hashing.
pub(super) struct FarIndex {
    slots: [[(u16, u16); INDEX_WAYS]; INDEX_SLOTS],
    counts: [u8; INDEX_SLOTS],
}

impl FarIndex {
    fn new() -> Self {
        Self { slots: [[(0, 0); INDEX_WAYS]; INDEX_SLOTS], counts: [0; INDEX_SLOTS] }
    }
    fn reset(&mut self) { self.counts.fill(0); }

    fn hash_row_lsh(vec: &[f32]) -> Option<(usize, u16)> {
        let (mut h1, mut h2, mut active) = (0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64, false);
        for (i, &v) in vec.iter().enumerate() {
            let trit: u64 = if v > 0.5 {
                active = true;
                2
            } else if v < -0.5 {
                active = true;
                1
            } else {
                0
            };
            h1 = (h1 ^ (trit ^ ((i as u64).wrapping_mul(3)))).wrapping_mul(0x0000_0100_0000_01b3);
            h2 = h2.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (trit << ((i & 15) * 2));
        }
        if !active { return None; }
        h1 ^= h1 >> 33;
        h2 ^= h2 >> 29;
        let slot = ((h1 ^ (h1 >> 17)) as usize) & (INDEX_SLOTS - 1);
        let tag = ((h2 ^ (h2 >> 16)) as u16).max(1);
        Some((slot, tag))
    }

    fn insert(&mut self, kv: &[f32], blk: u16) {
        let Some((slot, tag)) = Self::hash_row_lsh(kv) else { return };
        let cnt = self.counts[slot] as usize;
        if cnt < INDEX_WAYS && !self.slots[slot][..cnt].iter().any(|&(b, t)| b == blk && t == tag) {
            self.slots[slot][cnt] = (blk, tag);
            self.counts[slot] = (cnt + 1) as u8;
        }
    }

    fn lookup_far_blocks(&self, qv: &[f32], q: usize, blk_span: usize, out: &mut [u16; 2]) -> usize {
        let Some((slot, tag)) = Self::hash_row_lsh(qv) else { return 0 };
        let mut found = 0usize;
        for &(blk, t) in &self.slots[slot][..self.counts[slot] as usize] {
            let center = (blk as usize) * blk_span + (blk_span >> 1);
            if t == tag && center.abs_diff(q) > 31 && !out[..found].contains(&blk) {
                out[found] = blk;
                found += 1;
                if found == 2 { break; }
            }
        }
        found
    }
}

#[derive(Default)]
pub(super) struct IndexCache {
    blocks: Vec<[u16; TOP_BLOCKS]>,
    lens: Vec<u8>,
    n_rows: usize,
}

impl IndexCache {
    fn reset_for(&mut self, n: usize) {
        if self.blocks.len() < n {
            self.blocks.resize(n, [0; TOP_BLOCKS]);
            self.lens.resize(n, 0);
        }
        self.lens[..n].fill(0);
        self.n_rows = n;
    }
    fn store(&mut self, q: usize, sel: &[u16], len: usize) {
        let take = len.min(TOP_BLOCKS);
        self.blocks[q][..take].copy_from_slice(&sel[..take]);
        self.lens[q] = take as u8;
    }
    fn get(&self, q: usize) -> &[u16] {
        if q >= self.n_rows { &[] } else { &self.blocks[q][..self.lens[q] as usize] }
    }
}

/// Scratch workspace allocated on demand for sparse attention.
pub(super) struct SparseScratch {
    mu_k: Vec<f32>, var_k: Vec<f32>, mu_v: Vec<f32>,
    blk_cnt: Vec<u32>, blk_scores: Vec<(f64, u16)>,
    far_index: FarIndex, banded_cache: IndexCache, dense_cache: Vec<IndexCache>,
    slab_k: Vec<f32>, slab_v: Vec<f32>, slab_rows: [u32; SLAB_CAP], slab_len: usize,
    last_slab_blocks: [u16; TOP_BLOCKS], last_slab_cnt: usize,
    prefix_v: Vec<f64>, v_pos: Vec<bool>,
}

impl SparseScratch {
    fn new() -> Self {
        Self {
            mu_k: Vec::new(), var_k: Vec::new(), mu_v: Vec::new(),
            blk_cnt: Vec::new(), blk_scores: Vec::new(),
            far_index: FarIndex::new(), banded_cache: IndexCache::default(),
            dense_cache: Vec::new(),
            slab_k: Vec::new(), slab_v: Vec::new(), slab_rows: [0; SLAB_CAP], slab_len: 0,
            last_slab_blocks: [0; TOP_BLOCKS], last_slab_cnt: 0,
            prefix_v: Vec::new(), v_pos: Vec::new(),
        }
    }
    fn reset_slab(&mut self, h_dim: usize, v_dim: usize) {
        ensure_len(&mut self.slab_k, SLAB_CAP * h_dim);
        ensure_len(&mut self.slab_v, SLAB_CAP * v_dim);
        self.slab_len = 0;
        self.last_slab_cnt = 0;
    }
}

#[derive(Default)]
pub(super) struct EvalScratch {
    pub(super) grid: Vec<u8>,
    pub(super) row_lens: Vec<u16>,
    pub(super) rib_emb: Vec<f32>,
    pub(super) lex: Vec<f32>,
    pub(super) proj1: Vec<f32>,
    pub(super) pref1: Vec<f32>,
    pub(super) gates: Vec<f32>,
    pub(super) proj2: Vec<f32>,
    pub(super) pref2: Vec<f32>,
    pub(super) fused: Vec<f32>,
    pub(super) cells_soa: Vec<f32>,
    pub(super) acc_soa: Vec<f32>,

    pub(super) csum_p1: Vec<f32>,
    pub(super) base_p1: Vec<f32>,
    pub(super) csum_p2: Vec<f32>,
    pub(super) base_p2: Vec<f32>,
    pub(super) line_sum0: Vec<f32>,
    pub(super) line_sum1: Vec<f32>,
    pub(super) tile: TileScratch,
    pub(super) excl_tile: Vec<f32>,
    pub(super) saw_tile: Vec<f32>,
    pub(super) paren_tile: Vec<f32>,
    pub(super) hidden_tile: Vec<f32>,
    pub(super) stats_tile: Vec<f32>,

    pub(super) ctx_h_tile: Vec<f32>,
    pub(super) ctx_o_tile: Vec<f32>,
    pub(super) running_delta: Vec<f32>,
    pub(super) ring_w: Vec<f32>,
    pub(super) ring_g: RingGate,
    pub(super) prev1_raw: Vec<f32>,
    pub(super) prev2_raw: Vec<f32>,
    pub(super) mean_acc: Vec<f64>,
    pub(super) peak: Vec<f32>,

    pub(super) col_sums: Vec<f32>,
    pub(super) conv3: Vec<f32>,
    pub(super) conv7: Vec<f32>,
    pub(super) sketch: Vec<f32>,
    pub(super) deltas: Vec<f32>,
    pub(super) state: Vec<f32>,
    pub(super) q_buf: Vec<f32>,
    pub(super) k_buf: Vec<f32>,
    pub(super) v_buf: Vec<f32>,
    pub(super) merged: Vec<f32>,
    pub(super) attn_out: Vec<f32>,
    pub(super) gate: Vec<f32>,

    pub(super) logits: Vec<f64>,
    pub(super) keys: Vec<usize>,
    pub(super) k_tile: Vec<f64>,
    pub(super) logits_4: Vec<f64>,
    pub(super) q_nz: Vec<bool>,
    pub(super) k_nz: Vec<bool>,
    pub(super) sparse: Option<Box<SparseScratch>>,
}

impl EvalScratch {
    pub(super) fn new() -> Self { Self::default() }

    pub(super) fn ensure_span_buffers(&mut self, w: &ResolvedWeights) {
        let (emb_dim, pre_cols, hc_dim) = (w.tok_emb.cols, w.pre_conv3_w.cols, w.hc_dim);
        let p1 = w.pref1_proj.cols as usize;
        let g1 = w.gate1_w.cols as usize;
        let p2 = w.pref2_proj.cols as usize;
        let max_conv = w.conv3_w.cols.max(w.conv7_w.cols).max(w.sketch_w.cols);
        let ml_cols = w.ml_proj.cols as usize;
        let c0_h = w.ctx_layers[0].cmlp_w.cols as usize;
        let c1_h = w.ctx_layers[1].cmlp_w.cols as usize;
        let max_ctx_h = c0_h.max(c1_h);

        let sized: [(&mut Vec<f32>, usize); 27] = [
            (&mut self.rib_emb, (MAX_SPAN_LEN + 2) * emb_dim),
            (&mut self.lex, MAX_SPAN_LEN * pre_cols),
            (&mut self.proj1, MAX_SPAN_LEN * p1),
            (&mut self.pref1, MAX_SPAN_LEN * p1),
            (&mut self.gates, MAX_SPAN_LEN * g1),
            (&mut self.proj2, MAX_SPAN_LEN * p2),
            (&mut self.pref2, MAX_SPAN_LEN * p2),
            (&mut self.fused, MAX_SPAN_LEN * hc_dim),
            (&mut self.cells_soa, hc_dim * MAX_SPAN_LEN),
            (&mut self.acc_soa, max_conv * MAX_SPAN_LEN),
            (&mut self.csum_p1, p1),
            (&mut self.base_p1, p1),
            (&mut self.csum_p2, p2),
            (&mut self.base_p2, p2),
            (&mut self.line_sum0, hc_dim),
            (&mut self.line_sum1, hc_dim),
            (&mut self.excl_tile, 8 * ml_cols),
            (&mut self.saw_tile, 8 * w.ml_gate_w.cols as usize),
            (&mut self.paren_tile, 8 * w.ml_paren_w.cols),
            (&mut self.hidden_tile, 8 * w.lmlp1_w.cols as usize),
            (&mut self.stats_tile, 8 * w.lmlp2_w.cols as usize),
            (&mut self.ctx_h_tile, 8 * max_ctx_h),
            (&mut self.ctx_o_tile, 8 * w.state_dim),
            (&mut self.running_delta, ml_cols),
            (&mut self.ring_w, (PEAK_WINDOW + 1) * w.state_dim),
            (&mut self.prev1_raw, w.state_dim),
            (&mut self.prev2_raw, w.state_dim),
        ];
        for (buf, needed) in sized {
            ensure_len(buf, needed);
        }
        ensure_len(&mut self.peak, w.state_dim);
        if self.mean_acc.len() < w.state_dim {
            self.mean_acc.resize(w.state_dim, 0.0);
        }
    }

    pub(super) fn shrink_if_oversized(&mut self, num_lines: usize, max_lines: usize) {
        if num_lines <= max_lines { return; }
        drop_vec(&mut self.grid);
        drop_vec(&mut self.row_lens);
        for buf in [
            &mut self.col_sums, &mut self.conv3, &mut self.conv7, &mut self.sketch,
            &mut self.deltas, &mut self.state, &mut self.q_buf, &mut self.k_buf,
            &mut self.v_buf, &mut self.merged, &mut self.attn_out, &mut self.gate,
        ] { drop_vec(buf); }
        for buf in [&mut self.logits, &mut self.k_tile, &mut self.logits_4] { drop_vec(buf); }
        drop_vec(&mut self.keys);
        drop_vec(&mut self.q_nz);
        drop_vec(&mut self.k_nz);
        self.sparse = None;
    }
}

#[inline] fn ensure_len(buf: &mut Vec<f32>, needed: usize) { if buf.len() < needed { buf.resize(needed, 0.0); } }
#[inline] fn drop_vec<T>(buf: &mut Vec<T>) { buf.clear(); buf.shrink_to_fit(); }

#[inline]
fn swar_all_printable_ascii(chunk: [u8; 8]) -> bool {
    let word = u64::from_ne_bytes(chunk);
    (word & 0x8080_8080_8080_8080) == 0 && (word.wrapping_sub(0x2020_2020_2020_2020) & 0x8080_8080_8080_8080) == 0
}

pub(super) fn encode_grid_into(code: &str, grid: &mut Vec<u8>, row_lens: &mut Vec<u16>) -> usize {
    grid.clear();
    row_lens.clear();
    if code.is_empty() {
        grid.resize(MAX_COLS, 0);
        row_lens.push(0);
        return 1;
    }
    if let Some(n) = encode_grid_swar_ascii(code.as_bytes(), grid, row_lens) {
        return n;
    }
    encode_grid_general(code, grid, row_lens)
}

fn encode_grid_swar_ascii(bytes: &[u8], grid: &mut Vec<u8>, row_lens: &mut Vec<u16>) -> Option<usize> {
    let est = (bytes.len() / 40).clamp(1, 4096);
    grid.reserve(est * MAX_COLS);
    row_lens.reserve(est);
    grid.resize(MAX_COLS, 0);
    let (mut row_base, mut col, mut pos, len) = (0usize, 0usize, 0usize, bytes.len());
    while pos < len {
        if pos + 8 <= len
            && let Ok(chunk) = <[u8; 8]>::try_from(&bytes[pos..pos + 8])
            && swar_all_printable_ascii(chunk)
        {
            let take = (MAX_COLS - col).min(8);
            if take > 0 {
                grid[row_base + col..row_base + col + take].copy_from_slice(&chunk[..take]);
                col += take;
            }
            pos += 8;
            continue;
        }
        let b = bytes[pos];
        if BYTE_LUT.fast_mask[b as usize] != 0 {
            if col < MAX_COLS { grid[row_base + col] = BYTE_LUT.map[b as usize]; col += 1; }
            pos += 1;
        } else if b == b'\n' || (b == b'\r' && bytes.get(pos + 1) == Some(&b'\n')) {
            row_lens.push(col as u16);
            col = 0;
            row_base = grid.len();
            grid.resize(row_base + MAX_COLS, 0);
            pos += if b == b'\r' { 2 } else { 1 };
        } else {
            return None;
        }
    }
    if col > 0 || row_lens.is_empty() {
        row_lens.push(col as u16);
    } else {
        grid.truncate(row_base);
    }
    Some(row_lens.len())
}

fn expand_tabs(text: &str) -> Cow<'_, str> {
    if !text.as_bytes().contains(&b'\t') { return Cow::Borrowed(text); }
    let (mut out, mut col) = (String::with_capacity(text.len() + 32), 0usize);
    for ch in text.chars() {
        if ch == '\t' {
            let pad = 2 - (col % 2);
            out.push_str(&"  "[..pad]);
            col += pad;
        } else {
            out.push(ch);
            col = if ch == '\n' || ch == '\r' { 0 } else { col + 1 };
        }
    }
    Cow::Owned(out)
}

fn split_slices(text: &str) -> Vec<&str> {
    if text.is_empty() { return vec![""]; }
    let (mut parts, mut start, mut it) = (Vec::new(), 0usize, text.char_indices().peekable());
    while let Some((idx, ch)) = it.next() {
        if !matches!(ch, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'..='\x1e' | '\u{0085}' | '\u{2028}' | '\u{2029}') {
            continue;
        }
        parts.push(&text[start..idx]);
        let mut next = idx + ch.len_utf8();
        if ch == '\r' && it.peek().is_some_and(|&(_, c)| c == '\n') {
            it.next();
            next += 1;
        }
        start = next;
    }
    parts.push(&text[start..]);
    if parts.len() > 1 && parts.last().is_some_and(|s| s.is_empty()) { parts.pop(); }
    parts
}

fn encode_grid_general(code: &str, grid: &mut Vec<u8>, row_lens: &mut Vec<u16>) -> usize {
    grid.clear(); row_lens.clear();
    let expanded = expand_tabs(code);
    let lines = split_slices(&expanded);
    grid.resize(lines.len() * MAX_COLS, 0);
    row_lens.resize(lines.len(), 0);
    for (row, line) in lines.iter().enumerate() {
        let (base, mut count) = (row * MAX_COLS, 0usize);
        for (col, ch) in line.chars().take(MAX_COLS).enumerate() {
            let cp = ch as u32;
            grid[base + col] = if cp > 127 { 127 } else { BYTE_LUT.map[cp as usize] };
            count = col + 1;
        }
        row_lens[row] = count as u16;
    }
    lines.len()
}

#[inline]
pub(super) fn accumulate_row_csr(src: &[f32], weight: &CompactCsr, acc: &mut [f32]) {
    for &in_u16 in &weight.active_rows {
        let in_idx = in_u16 as usize;
        let iv = src[in_idx];
        if iv != 0.0 { weight.accum_row(in_idx, iv, acc); }
    }
}

#[inline]
fn fill_span_ribbon_emb(
    w: &ResolvedWeights,
    grid: &[u8],
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    rib_emb: &mut [f32],
) -> usize {
    let (emb_dim, span_len) = (w.tok_emb.cols, slen0 + paired_row1.map_or(0, |(_, s1)| s1));
    rib_emb[..emb_dim].fill(0.0);
    rib_emb[(span_len + 1) * emb_dim..(span_len + 2) * emb_dim].fill(0.0);
    let mut copy_row = |row: usize, slen: usize, off: usize| {
        let base = row * MAX_COLS;
        for col in 0..slen {
            let (dst, tb, cb) = ((1 + off + col) * emb_dim, grid[base + col] as usize * emb_dim, col * emb_dim);
            for ch in 0..emb_dim { rib_emb[dst + ch] = w.tok_emb.data[tb + ch] + w.col_emb.data[cb + ch]; }
        }
    };
    copy_row(row0, slen0, 0);
    if let Some((row1, slen1)) = paired_row1 { copy_row(row1, slen1, slen0); }
    span_len
}

#[inline]
fn run_pre_conv3(w: &ResolvedWeights, span_len: usize, rib_emb: &[f32], lex: &mut [f32]) {
    let (emb_dim, pre_cols) = (w.tok_emb.cols, w.pre_conv3_w.cols);
    for cell in 0..span_len {
        let dst = &mut lex[cell * pre_cols..(cell + 1) * pre_cols];
        dst.copy_from_slice(&w.pre_conv3_b);
        for tap in 0..w.pre_conv3_w.k_size {
            let tap_src = &rib_emb[(cell + tap) * emb_dim..(cell + tap + 1) * emb_dim];
            accumulate_row_csr(tap_src, &w.pre_conv3_w.taps[tap], dst);
        }
        for val in dst.iter_mut() { if *val < 0.0 { *val = 0.0; } }
    }
}

struct PrefixCfg { cell_lo: usize, cell_hi: usize, dim: usize, exclusive: bool, is_first: bool }

#[inline]
fn stream_line_prefix(
    proj: &[f32], out: &mut [f32], cfg: PrefixCfg, csum: &mut [f32], base_buf: &mut [f32],
) {
    if cfg.is_first {
        base_buf[..cfg.dim].fill(0.0);
    } else {
        base_buf[..cfg.dim].copy_from_slice(&csum[..cfg.dim]);
    }
    for cell in cfg.cell_lo..cfg.cell_hi {
        let off = cell * cfg.dim;
        if cfg.is_first && cell == cfg.cell_lo {
            csum[..cfg.dim].copy_from_slice(&proj[off..off + cfg.dim]);
        } else {
            for ch in 0..cfg.dim { csum[ch] += proj[off + ch]; }
        }
        for ch in 0..cfg.dim {
            let val = csum[ch] - base_buf[ch];
            out[off + ch] = if cfg.exclusive { val - proj[off + ch] } else { val };
        }
    }
}

#[inline]
fn scatter_fused_to_soa(
    w: &ResolvedWeights,
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    sc: &mut EvalScratch,
) -> [u64; 2] {
    let hc = w.hc_dim;
    let span_len = slen0 + paired_row1.map_or(0, |(_, s1)| s1);
    let cs_cols = w.col_sum_proj.cols as usize;
    let mut active = [0u64; 2];
    sc.line_sum0[..hc].fill(0.0);
    for cell in 0..slen0 {
        for ch in 0..hc {
            let val = sc.fused[cell * hc + ch];
            sc.cells_soa[ch * MAX_SPAN_LEN + cell] = val;
            sc.line_sum0[ch] += val;
            if val != 0.0 { active[ch >> 6] |= 1u64 << (ch & 63); }
        }
    }
    let dst0 = &mut sc.col_sums[row0 * cs_cols..(row0 + 1) * cs_cols];
    accumulate_row_csr(&sc.line_sum0[..hc], &w.col_sum_proj, dst0);
    if let Some((row1, _)) = paired_row1 {
        sc.line_sum1[..hc].fill(0.0);
        for cell in slen0..span_len {
            for ch in 0..hc {
                let val = sc.fused[cell * hc + ch];
                sc.cells_soa[ch * MAX_SPAN_LEN + cell] = val;
                sc.line_sum1[ch] += val;
                if val != 0.0 { active[ch >> 6] |= 1u64 << (ch & 63); }
            }
        }
        let dst1 = &mut sc.col_sums[row1 * cs_cols..(row1 + 1) * cs_cols];
        accumulate_row_csr(&sc.line_sum1[..hc], &w.col_sum_proj, dst1);
    }
    active
}

// SAFETY: Caller guarantees `K` target features and initialized span buffers.
#[inline(always)]
pub(super) unsafe fn eval_span_block01_with<K: HygieneKernel>(
    w: &ResolvedWeights,
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    is_first: bool,
    sc: &mut EvalScratch,
) {
    let span_len = fill_span_ribbon_emb(w, &sc.grid, row0, slen0, paired_row1, &mut sc.rib_emb);
    run_pre_conv3(w, span_len, &sc.rib_emb, &mut sc.lex);
    let (emb_dim, pre_cols) = (w.tok_emb.cols, w.pre_conv3_w.cols);
    let emb = &sc.rib_emb[emb_dim..(span_len + 1) * emb_dim];
    let p1 = w.pref1_proj.cols as usize;
    let g1 = w.gate1_w.cols as usize;
    let p2 = w.pref2_proj.cols as usize;
    let lex_s = &sc.lex[..span_len * pre_cols];

    // SAFETY: Caller verified `K` features; `emb`, `lex_s`, and `proj1` cover `span_len`.
    unsafe {
        K::sparse_parts(
            &[emb, lex_s], &w.pref1_ranges, span_len,
            CsOp::new(&w.pref1_proj, None, false), &mut sc.proj1[..span_len * p1], &mut sc.tile,
        );
    }
    let cfg_p1_a = PrefixCfg { cell_lo: 0, cell_hi: slen0, dim: p1, exclusive: true, is_first };
    stream_line_prefix(&sc.proj1, &mut sc.pref1, cfg_p1_a, &mut sc.csum_p1, &mut sc.base_p1);
    if paired_row1.is_some() {
        let cfg_p1_b = PrefixCfg {
            cell_lo: slen0, cell_hi: span_len, dim: p1, exclusive: true, is_first: false,
        };
        stream_line_prefix(&sc.proj1, &mut sc.pref1, cfg_p1_b, &mut sc.csum_p1, &mut sc.base_p1);
    }
    let pref1_s = &sc.pref1[..span_len * p1];
    // SAFETY: Caller verified `K` features; gate and prefix buffers cover `span_len`.
    unsafe {
        K::sparse_parts(
            &[emb, lex_s, pref1_s], &w.gate1_ranges, span_len,
            CsOp::new(&w.gate1_w, Some(&w.gate1_b), true), &mut sc.gates[..span_len * g1],
            &mut sc.tile,
        );
        K::linear_sparse(
            &sc.gates[..span_len * g1], span_len, g1,
            CsOp::new(&w.pref2_proj, None, false), &mut sc.proj2[..span_len * p2], &mut sc.tile,
        );
    }
    let cfg_p2_a = PrefixCfg { cell_lo: 0, cell_hi: slen0, dim: p2, exclusive: false, is_first };
    stream_line_prefix(&sc.proj2, &mut sc.pref2, cfg_p2_a, &mut sc.csum_p2, &mut sc.base_p2);
    if paired_row1.is_some() {
        let cfg_p2_b = PrefixCfg {
            cell_lo: slen0, cell_hi: span_len, dim: p2, exclusive: false, is_first: false,
        };
        stream_line_prefix(&sc.proj2, &mut sc.pref2, cfg_p2_b, &mut sc.csum_p2, &mut sc.base_p2);
    }
    let gates_s = &sc.gates[..span_len * g1];
    let pref2_s = &sc.pref2[..span_len * p2];
    // SAFETY: Caller verified `K` features; `fused` output holds `span_len * hc_dim`.
    unsafe {
        K::sparse_parts(
            &[emb, lex_s, gates_s, pref2_s], &w.col_fuse_ranges, span_len,
            CsOp::new(&w.col_fuse_w, Some(&w.col_fuse_b), true),
            &mut sc.fused[..span_len * w.hc_dim], &mut sc.tile,
        );
    }
    let active_mask = scatter_fused_to_soa(w, row0, slen0, paired_row1, sc);
    let span = RibbonSpan {
        cells_soa: &sc.cells_soa, active_mask, span_len,
        split_at: paired_row1.map(|(r1, _)| (slen0, row0, r1)), single_row: row0,
    };
    // SAFETY: Caller verified `K` features; SoA scratch uses `MAX_SPAN_LEN` stride.
    unsafe {
        K::ribbon_conv_span(
            &span, &w.conv3_w, &w.conv3_b, &w.conv3_pads, &mut sc.acc_soa, &mut sc.conv3,
        );
        K::ribbon_conv_span(
            &span, &w.conv7_w, &w.conv7_b, &w.conv7_pads, &mut sc.acc_soa, &mut sc.conv7,
        );
        K::ribbon_conv_span(
            &span, &w.sketch_w, &w.sketch_b, &w.sketch_pads, &mut sc.acc_soa, &mut sc.sketch,
        );
    }
}

pub(super) fn norm_sketch(sketch: &mut [f32], num_lines: usize, total_sketch: usize) {
    let total = num_lines * total_sketch;
    for val in &mut sketch[..total] { *val = val.clamp(0.0, 1.0); }
    let mut base = 0usize;
    while base < total {
        let end = (base + SKETCH_DIM).min(total);
        let mut sq_sum = 0.0f32;
        for &v in &sketch[base..end] {
            sq_sum += v * v;
        }
        let norm = (sq_sum + 1e-6f32).sqrt();
        for val in &mut sketch[base..end] {
            *val /= norm;
        }
        base += SKETCH_DIM;
    }
}

#[inline]
pub(super) fn axis_conv_row(deltas: &[f32], num_lines: usize, row: usize, w: &ResolvedWeights, dst: &mut [f32]) {
    let (conv, pad) = (&w.ml_paren_w, ((w.ml_paren_w.k_size - 1) >> 1) as isize);
    dst.copy_from_slice(&w.ml_paren_b);
    let lo = (pad - row as isize).max(0) as usize;
    let hi = (num_lines as isize - row as isize + pad).min(conv.k_size as isize) as usize;
    for tap in lo..hi {
        let tm = &conv.taps[tap];
        if tm.non_zero_count == 0 { continue; }
        let src_row = (row as isize + tap as isize - pad) as usize;
        accumulate_row_csr(&deltas[src_row * conv.rows..(src_row + 1) * conv.rows], tm, dst);
    }
    for val in dst.iter_mut() { if *val < 0.0 { *val = 0.0; } }
}

#[derive(Clone, Copy)]
struct HeadRef<'a> { layer: &'a AttnLayerWeights, head: &'a HeadPlan }

struct SoftmaxBuf<'a> {
    keys: &'a mut [usize],
    logits: &'a mut [f64],
    count: usize,
    max_l: f64,
}

impl SoftmaxBuf<'_> {
    #[inline]
    fn push(&mut self, key: usize, logit: f64) {
        self.keys[self.count] = key;
        self.logits[self.count] = logit;
        self.max_l = self.max_l.max(logit);
        self.count += 1;
    }
}

#[inline]
fn exp_sum_in_place(logits: &mut [f64], max_l: f64, init: f64) -> f64 {
    let mut sum = init;
    for lv in logits {
        let e = (*lv - max_l).exp();
        *lv = e;
        sum += e;
    }
    sum
}

#[inline]
fn accum_weighted_v(dst: &mut [f32], src: &[f32], aw: f64) {
    for (out, &val) in dst.iter_mut().zip(src.iter()) {
        *out += (aw * (val as f64)) as f32;
    }
}

#[inline]
fn accum_exact_v(
    hr: HeadRef<'_>,
    q: usize,
    keys: &[usize],
    weights: &[f64],
    exp_sum: f64,
    v_buf: &[f32],
    merged: &mut [f32],
) {
    let dst = q * hr.layer.v_tot + hr.head.v_off;
    let v_dim = hr.layer.v_dim;
    for (&k, &ew) in keys.iter().zip(weights.iter()) {
        let vb = k * hr.layer.v_tot + hr.head.v_off;
        accum_weighted_v(&mut merged[dst..dst + v_dim], &v_buf[vb..vb + v_dim], ew / exp_sum);
    }
}

fn attend_uniform_global(hr: HeadRef<'_>, n: usize, sc: &mut EvalScratch) {
    let (v_dim, wf) = (hr.layer.v_dim, ((1.0f64 / (n as f64 + hr.head.sink.exp())) as f32) as f64);
    let mut sum = vec![0.0f64; v_dim];
    for row in 0..n {
        let base = row * hr.layer.v_tot + hr.head.v_off;
        for (d, s) in sum.iter_mut().enumerate() {
            *s += sc.v_buf[base + d] as f64;
        }
    }
    for row in 0..n {
        let base = row * hr.layer.v_tot + hr.head.v_off;
        for (d, &s) in sum.iter().enumerate() {
            sc.merged[base + d] = (s * wf) as f32;
        }
    }
}

struct StaticBand<'a> { offsets: &'a [i8], rel_vals: &'a [f64], interior: &'a [f64], min_neg: usize, max_pos: usize }

fn attend_static_banded(hr: HeadRef<'_>, band: StaticBand<'_>, n: usize, sc: &mut EvalScratch) {
    let v_dim = hr.layer.v_dim;
    for q in 0..n {
        let dst = q * hr.layer.v_tot + hr.head.v_off;
        if q >= band.min_neg && q + band.max_pos < n {
            for (&off, &aw) in band.offsets.iter().zip(band.interior.iter()) {
                let vb = (q as isize + off as isize) as usize * hr.layer.v_tot + hr.head.v_off;
                accum_weighted_v(&mut sc.merged[dst..dst + v_dim], &sc.v_buf[vb..vb + v_dim], aw);
            }
            continue;
        }
        let mut sm = SoftmaxBuf { keys: &mut sc.keys, logits: &mut sc.logits, count: 0, max_l: hr.head.sink };
        for (&off, &rv) in band.offsets.iter().zip(band.rel_vals.iter()) {
            let k = q as isize + off as isize;
            if k >= 0 && k < n as isize {
                sm.push(k as usize, rv);
            }
        }
        let (count, max_l) = (sm.count, sm.max_l);
        let exp_sum = exp_sum_in_place(&mut sc.logits[..count], max_l, (hr.head.sink - max_l).exp());
        accum_exact_v(hr, q, &sc.keys[..count], &sc.logits[..count], exp_sum, &sc.v_buf, &mut sc.merged);
    }
}

fn attend_general_static_exact(hr: HeadRef<'_>, rel: &[f64; 65], n: usize, sc: &mut EvalScratch) {
    let v_dim = hr.layer.v_dim;
    for q in 0..n {
        let mut max_l = hr.head.sink;
        for (k, slot) in sc.logits.iter_mut().enumerate().take(n) {
            let rv = rel[((k as isize - q as isize).clamp(-32, 32) + 32) as usize];
            *slot = rv;
            max_l = max_l.max(rv);
        }
        let exp_sum = exp_sum_in_place(&mut sc.logits[..n], max_l, (hr.head.sink - max_l).exp());
        let dst = q * hr.layer.v_tot + hr.head.v_off;
        for (k, &ew) in sc.logits.iter().enumerate().take(n) {
            let vb = k * hr.layer.v_tot + hr.head.v_off;
            accum_weighted_v(&mut sc.merged[dst..dst + v_dim], &sc.v_buf[vb..vb + v_dim], ew / exp_sum);
        }
    }
}

/// Evaluate a static attention head with linear prefix sums.
fn attend_prefix(
    hr: HeadRef<'_>, rel: &[f64; 65], n: usize, sc: &mut EvalScratch, stats: &mut SparseStats,
) {
    let v_dim = hr.layer.v_dim;
    let sp = sc.sparse.get_or_insert_with(|| Box::new(SparseScratch::new()));
    if sp.prefix_v.len() < (n + 1) * v_dim { sp.prefix_v.resize((n + 1) * v_dim, 0.0); }
    sp.prefix_v[..v_dim].fill(0.0);
    for row in 0..n {
        let (src, prev, next) = (row * hr.layer.v_tot + hr.head.v_off, row * v_dim, (row + 1) * v_dim);
        for d in 0..v_dim { sp.prefix_v[next + d] = sp.prefix_v[prev + d] + (sc.v_buf[src + d] as f64); }
    }
    let total_off = n * v_dim;
    for q in 0..n {
        let (left_cnt, right_start) = (q.saturating_sub(31), (q + 32).min(n));
        let (right_cnt, band_len) = (n - right_start, right_start - left_cnt);
        let mut max_l = hr.head.sink;
        if left_cnt > 0 { max_l = max_l.max(rel[0]); }
        if right_cnt > 0 { max_l = max_l.max(rel[64]); }
        for (i, k) in (left_cnt..right_start).enumerate() {
            let rv = rel[(k as isize - q as isize + 32) as usize];
            sc.logits[i] = rv;
            max_l = max_l.max(rv);
        }
        let le = if left_cnt > 0 { (rel[0] - max_l).exp() } else { 0.0 };
        let re = if right_cnt > 0 { (rel[64] - max_l).exp() } else { 0.0 };
        let base_sum = (hr.head.sink - max_l).exp() + (left_cnt as f64) * le + (right_cnt as f64) * re;
        let exp_sum = exp_sum_in_place(&mut sc.logits[..band_len], max_l, base_sum);
        let dst = q * hr.layer.v_tot + hr.head.v_off;
        if left_cnt > 0 {
            let (aw, loff) = (le / exp_sum, left_cnt * v_dim);
            for d in 0..v_dim { sc.merged[dst + d] += (aw * sp.prefix_v[loff + d]) as f32; }
        }
        if right_cnt > 0 {
            let (aw, roff) = (re / exp_sum, right_start * v_dim);
            for d in 0..v_dim {
                let delta = sp.prefix_v[total_off + d] - sp.prefix_v[roff + d];
                sc.merged[dst + d] += (aw * delta) as f32;
            }
        }
        for (i, k) in (left_cnt..right_start).enumerate() {
            let vb = k * hr.layer.v_tot + hr.head.v_off;
            let aw = sc.logits[i] / exp_sum;
            accum_weighted_v(&mut sc.merged[dst..dst + v_dim], &sc.v_buf[vb..vb + v_dim], aw);
        }
        stats.pairs_evaluated += band_len + usize::from(left_cnt > 0) + usize::from(right_cnt > 0);
    }
}

fn fill_qk_nonzero_flags(hr: HeadRef<'_>, n: usize, sc: &mut EvalScratch) {
    sc.q_nz.resize(n, false);
    sc.k_nz.resize(n, false);
    for row in 0..n {
        let base = row * hr.layer.qk_tot + hr.head.qk_off;
        sc.q_nz[row] = sc.q_buf[base..base + hr.layer.h_dim].iter().any(|&v| v != 0.0);
        sc.k_nz[row] = sc.k_buf[base..base + hr.layer.h_dim].iter().any(|&v| v != 0.0);
    }
}

// SAFETY: Caller guarantees `K` target features and sized Q/K scratch slices.
#[inline(always)]
unsafe fn eval_banded_local_row<K: HygieneKernel>(
    hr: HeadRef<'_>, band: (&[i8], &[f64]), q: usize, n: usize, sc: &mut EvalScratch,
) -> (usize, f64) {
    let (q_active, q_base) = (sc.q_nz[q], q * hr.layer.qk_tot + hr.head.qk_off);
    let q_slice = &sc.q_buf[q_base..q_base + hr.layer.h_dim];
    let (mut count, mut max_l) = (0usize, hr.head.sink);
    for (&off, &rv) in band.0.iter().zip(band.1.iter()) {
        let k_i = q as isize + off as isize;
        if k_i < 0 || k_i >= n as isize { continue; }
        let k = k_i as usize;
        let qk = if q_active && sc.k_nz[k] {
            let kb = k * hr.layer.qk_tot + hr.head.qk_off;
            // SAFETY: Caller verified `K` features; query and key slices have `h_dim` elements.
            unsafe { K::dot_f32_as_f64(q_slice, &sc.k_buf[kb..kb + hr.layer.h_dim]) * hr.layer.inv_s }
        } else { 0.0 };
        let logit = rv + qk;
        sc.keys[count] = k;
        sc.logits[count] = logit;
        max_l = max_l.max(logit);
        count += 1;
    }
    (count, max_l)
}

// SAFETY: Caller guarantees `K` target features and `n`-row attention buffers.
#[inline(always)]
unsafe fn attend_banded_qk_with<K: HygieneKernel>(
    hr: HeadRef<'_>, band: (&[i8], &[f64]), n: usize, sc: &mut EvalScratch,
) {
    fill_qk_nonzero_flags(hr, n, sc);
    for q in 0..n {
        // SAFETY: Caller verified `K` features; scratch logits cover the local band.
        let (count, exp_sum) = unsafe {
            let (count, max_l) = eval_banded_local_row::<K>(hr, band, q, n, sc);
            let sum = (hr.head.sink - max_l).exp()
                + K::softmax_exp_slice(&mut sc.logits[..count], max_l);
            (count, sum)
        };
        accum_exact_v(hr, q, &sc.keys[..count], &sc.logits[..count], exp_sum, &sc.v_buf, &mut sc.merged);
    }
}

#[inline]
fn block_cumulant_score(q_slice: &[f32], mu_k: &[f32], var_k: &[f32], inv_s: f64) -> f64 {
    let (mut first, mut second) = (0.0f64, 0.0f64);
    for dim in 0..q_slice.len() {
        let qv = q_slice[dim] as f64;
        first += qv * (mu_k[dim] as f64);
        second += (qv * qv) * (var_k[dim] as f64);
    }
    first * inv_s + 0.5 * second * inv_s * inv_s
}

fn build_blocks_into(
    hr: HeadRef<'_>, n: usize, blk_span: usize, active_only: bool, sc: &mut EvalScratch,
) -> usize {
    let (num_blocks, h_dim, v_dim) = (n.div_ceil(blk_span), hr.layer.h_dim, hr.layer.v_dim);
    let sp = sc.sparse.get_or_insert_with(|| Box::new(SparseScratch::new()));
    ensure_len(&mut sp.mu_k, num_blocks * h_dim);
    ensure_len(&mut sp.var_k, num_blocks * h_dim);
    ensure_len(&mut sp.mu_v, num_blocks * v_dim);
    sp.mu_k[..num_blocks * h_dim].fill(0.0);
    sp.var_k[..num_blocks * h_dim].fill(0.0);
    sp.mu_v[..num_blocks * v_dim].fill(0.0);
    sp.blk_cnt.resize(num_blocks, 0);
    sp.blk_cnt[..num_blocks].fill(0);
    if sp.blk_scores.len() < num_blocks { sp.blk_scores.resize(num_blocks, (0.0, 0)); }

    for blk in 0..num_blocks {
        let (lo, hi, mut cnt) = (blk * blk_span, ((blk + 1) * blk_span).min(n), 0u32);
        let (mk, mv) = (&mut sp.mu_k[blk * h_dim..(blk + 1) * h_dim], &mut sp.mu_v[blk * v_dim..(blk + 1) * v_dim]);
        for row in lo..hi {
            if active_only && !(sc.k_nz[row] && sp.v_pos[row]) { continue; }
            cnt += 1;
            let (kb, vb) = (row * hr.layer.qk_tot + hr.head.qk_off, row * hr.layer.v_tot + hr.head.v_off);
            for d in 0..h_dim { mk[d] += sc.k_buf[kb + d]; }
            for d in 0..v_dim { mv[d] += sc.v_buf[vb + d]; }
        }
        sp.blk_cnt[blk] = cnt;
        if cnt == 0 { continue; }
        let inv_cnt = 1.0f32 / (cnt as f32);
        for v in mk.iter_mut() { *v *= inv_cnt; }
        for v in mv.iter_mut() { *v *= inv_cnt; }
        let vk = &mut sp.var_k[blk * h_dim..(blk + 1) * h_dim];
        for row in lo..hi {
            if active_only && !(sc.k_nz[row] && sp.v_pos[row]) { continue; }
            let kb = row * hr.layer.qk_tot + hr.head.qk_off;
            for d in 0..h_dim {
                let diff = sc.k_buf[kb + d] - mk[d];
                vk[d] += diff * diff * inv_cnt;
            }
        }
    }
    num_blocks
}

fn populate_far_index(hr: HeadRef<'_>, n: usize, blk_span: usize, active_v_pos: bool, sc: &mut EvalScratch) {
    let sp = sc.sparse.get_or_insert_with(|| Box::new(SparseScratch::new()));
    sp.far_index.reset();
    for row in 0..n {
        if !sc.k_nz[row] || (active_v_pos && !sp.v_pos[row]) { continue; }
        let kb = row * hr.layer.qk_tot + hr.head.qk_off;
        sp.far_index.insert(&sc.k_buf[kb..kb + hr.layer.h_dim], (row / blk_span) as u16);
    }
}

fn select_top_blocks(scores: &[(f64, u16)], k_sel: usize, out: &mut [u16; TOP_BLOCKS]) -> usize {
    let (mut picked, lim) = (0usize, k_sel.min(TOP_BLOCKS).min(scores.len()));
    for _ in 0..lim {
        let (mut best_blk, mut best_sc) = (u16::MAX, f64::NEG_INFINITY);
        for &(sc, blk) in scores {
            if sc > best_sc && !out[..picked].contains(&blk) {
                best_sc = sc;
                best_blk = blk;
            }
        }
        if best_blk == u16::MAX { break; }
        out[picked] = best_blk;
        picked += 1;
    }
    picked
}

fn merge_far_blocks(sel: &mut [u16; TOP_BLOCKS], mut sel_cnt: usize, k_sel: usize, far: &[u16]) -> usize {
    let cap = k_sel.min(TOP_BLOCKS);
    let mut overwrite = cap;
    for &fb in far {
        if sel[..sel_cnt].contains(&fb) { continue; }
        if sel_cnt < cap {
            sel[sel_cnt] = fb;
            sel_cnt += 1;
        } else if overwrite > 0 {
            overwrite -= 1;
            sel[overwrite] = fb;
        }
    }
    sel_cnt
}

struct SlabCfg<'a> { hr: HeadRef<'a>, n: usize, blk_span: usize, active_only: bool }

fn pack_slab_cached(
    cfg: &SlabCfg<'_>, sel: &[u16], bufs: (&[f32], &[f32], &[bool]),
    sp: &mut SparseScratch, stats: &mut SparseStats,
) {
    let cnt = sel.len().min(TOP_BLOCKS);
    if cnt == sp.last_slab_cnt && sp.last_slab_blocks[..cnt] == sel[..cnt] {
        stats.slab_reuses += 1;
        return;
    }
    sp.last_slab_blocks[..cnt].copy_from_slice(&sel[..cnt]);
    sp.last_slab_cnt = cnt;
    let (h_dim, v_dim, mut dst_row) = (cfg.hr.layer.h_dim, cfg.hr.layer.v_dim, 0usize);
    for &blk_u16 in &sel[..cnt] {
        let (lo, hi) = ((blk_u16 as usize) * cfg.blk_span, (((blk_u16 as usize) + 1) * cfg.blk_span).min(cfg.n));
        for row in lo..hi {
            if dst_row >= SLAB_CAP { break; }
            if cfg.active_only && !(bufs.2[row] && sp.v_pos[row]) { continue; }
            let kb = row * cfg.hr.layer.qk_tot + cfg.hr.head.qk_off;
            let vb = row * cfg.hr.layer.v_tot + cfg.hr.head.v_off;
            sp.slab_k[dst_row * h_dim..(dst_row + 1) * h_dim].copy_from_slice(&bufs.0[kb..kb + h_dim]);
            sp.slab_v[dst_row * v_dim..(dst_row + 1) * v_dim].copy_from_slice(&bufs.1[vb..vb + v_dim]);
            sp.slab_rows[dst_row] = row as u32;
            dst_row += 1;
        }
    }
    sp.slab_len = dst_row;
}

struct BandedFarCtx { q: usize, blk_span: usize, num_blocks: usize, reuse_cache: bool }

fn select_banded_far_blocks(
    q_slice: &[f32], layer: &AttnLayerWeights, ctx: BandedFarCtx,
    sp: &mut SparseScratch, stats: &mut SparseStats, sel: &mut [u16; TOP_BLOCKS],
) -> usize {
    let mut far_blks = [0u16; 2];
    let far_cnt = sp.far_index.lookup_far_blocks(q_slice, ctx.q, ctx.blk_span, &mut far_blks);
    stats.far_hits += far_cnt;
    if ctx.reuse_cache {
        let cached = sp.banded_cache.get(ctx.q);
        let take = cached.len().min(2);
        sel[..take].copy_from_slice(&cached[..take]);
        if take > 0 { stats.reused = true; stats.reuse_hits += 1; }
        return merge_far_blocks(sel, take, TOP_BLOCKS, &far_blks[..far_cnt]);
    }
    let (q_blk, h_dim, mut valid) = ((ctx.q / ctx.blk_span) as u16, layer.h_dim, 0usize);
    for blk in 0..ctx.num_blocks {
        if sp.blk_cnt[blk] == 0 || blk as u16 == q_blk { continue; }
        let (mu, var) = (&sp.mu_k[blk * h_dim..(blk + 1) * h_dim], &sp.var_k[blk * h_dim..(blk + 1) * h_dim]);
        sp.blk_scores[valid] = (block_cumulant_score(q_slice, mu, var, layer.inv_s), blk as u16);
        valid += 1;
    }
    stats.pairs_evaluated += valid;
    let top_n = select_top_blocks(&sp.blk_scores[..valid], TOP_BLOCKS, sel);
    let merged_n = merge_far_blocks(sel, top_n, TOP_BLOCKS, &far_blks[..far_cnt]);
    sp.banded_cache.store(ctx.q, sel, merged_n);
    merged_n
}

// SAFETY: Caller guarantees `K` target features and packed slab keys with stride `h_dim`.
#[inline(always)]
unsafe fn score_slab_banded<K: HygieneKernel>(
    q_slice: &[f32],
    q: usize,
    hr: HeadRef<'_>,
    sp: &SparseScratch,
    sm: &mut SoftmaxBuf<'_>,
    stats: &mut SparseStats,
) {
    let h_dim = hr.layer.h_dim;
    for s_idx in 0..sp.slab_len {
        let k = sp.slab_rows[s_idx] as usize;
        if k.abs_diff(q) <= 31 { continue; }
        let k_slice = &sp.slab_k[s_idx * h_dim..(s_idx + 1) * h_dim];
        // SAFETY: Caller verified `K` features; query and slab key slices have `h_dim` elements.
        let qk_far = unsafe { K::dot_f32_as_f64(q_slice, k_slice) * hr.layer.inv_s };
        stats.pairs_evaluated += 1;
        if qk_far > hr.head.sink && sm.count < sm.keys.len() {
            sm.keys[sm.count] = k;
            sm.logits[sm.count] = qk_far;
            sm.max_l = sm.max_l.max(qk_far);
            sm.count += 1;
        }
    }
}

/// Evaluate a far-horizon banded head with sparse block lookup.
// SAFETY: Caller guarantees `K` target features and sized attention scratch buffers.
#[inline(always)]
unsafe fn attend_banded_sparse<K: HygieneKernel>(
    hr: HeadRef<'_>, band: (&[i8], &[f64]), n: usize, reuse_cache: bool,
    sc: &mut EvalScratch, stats: &mut SparseStats,
) {
    fill_qk_nonzero_flags(hr, n, sc);
    let blk_span = n.div_ceil(MAX_LANDMARKS).max(BLOCK_SIZE);
    {
        let sp = sc.sparse.get_or_insert_with(|| Box::new(SparseScratch::new()));
        sp.v_pos.resize(n, false);
        for row in 0..n { sp.v_pos[row] = sc.v_buf[row * hr.layer.v_tot + hr.head.v_off] > 0.0; }
        sp.reset_slab(hr.layer.h_dim, hr.layer.v_dim);
    }
    let num_blocks = if reuse_cache {
        n.div_ceil(blk_span)
    } else {
        let nb = build_blocks_into(hr, n, blk_span, true, sc);
        stats.blocks += nb;
        if let Some(sp) = sc.sparse.as_mut() { sp.banded_cache.reset_for(n); }
        nb
    };
    populate_far_index(hr, n, blk_span, true, sc);
    stats.index_bytes = FAR_INDEX_BYTES;

    let slab_cfg = SlabCfg { hr, n, blk_span, active_only: true };
    let Some(mut sp) = sc.sparse.take() else { return };
    for q in 0..n {
        // SAFETY: Caller verified `K` features; local band fits inside scratch logits.
        let (count, max_l) = unsafe { eval_banded_local_row::<K>(hr, band, q, n, sc) };
        stats.pairs_evaluated += count;
        let mut sm = SoftmaxBuf { keys: &mut sc.keys, logits: &mut sc.logits, count, max_l };
        if sc.q_nz[q] {
            let q_base = q * hr.layer.qk_tot + hr.head.qk_off;
            let q_slice = &sc.q_buf[q_base..q_base + hr.layer.h_dim];
            let mut sel = [0u16; TOP_BLOCKS];
            let ctx = BandedFarCtx { q, blk_span, num_blocks, reuse_cache };
            let sel_cnt = select_banded_far_blocks(q_slice, hr.layer, ctx, &mut sp, stats, &mut sel);
            if sel_cnt > 0 {
                pack_slab_cached(&slab_cfg, &sel[..sel_cnt], (&sc.k_buf, &sc.v_buf, &sc.k_nz), &mut sp, stats);
                // SAFETY: Caller verified `K` features; `pack_slab_cached` packed `h_dim` rows.
                unsafe { score_slab_banded::<K>(q_slice, q, hr, &sp, &mut sm, stats); }
            }
        }
        let (cnt, ml) = (sm.count, sm.max_l);
        // SAFETY: Caller verified `K` features; `logits[..cnt]` holds all evaluated entries.
        let exp_sum = (hr.head.sink - ml).exp() + unsafe { K::softmax_exp_slice(&mut sc.logits[..cnt], ml) };
        accum_exact_v(hr, q, &sc.keys[..cnt], &sc.logits[..cnt], exp_sum, &sc.v_buf, &mut sc.merged);
    }
    sc.sparse = Some(sp);
}

#[inline]
fn dense_rel_bias(rel: &[f64; 65], all_zero: bool, q: usize, k: usize) -> f64 {
    if all_zero { 0.0 } else { rel[((k as isize - q as isize).clamp(-32, 32) + 32) as usize] }
}

#[inline]
fn dense_far_rel_bias(rel: &[f64; 65], all_zero: bool, b_lo: usize, b_hi: usize, k_lo: usize) -> f64 {
    if all_zero { 0.0 } else if b_hi <= k_lo || b_lo < k_lo { rel[0] } else { rel[64] }
}

struct DenseBlockCtx<'a> {
    rel: &'a [f64; 65], all_zero: bool, head_idx: usize, n: usize,
    blk_span: usize, num_blocks: usize, k_sel: usize, reuse_cross: bool,
}

fn select_dense_far_blocks(
    q_slice: &[f32], q: usize, layer: &AttnLayerWeights, ctx: &DenseBlockCtx<'_>,
    sp: &mut SparseScratch, stats: &mut SparseStats, sel: &mut [u16; TOP_BLOCKS],
) -> usize {
    if ctx.reuse_cross {
        let cached = sp.dense_cache[ctx.head_idx].get(q);
        let take = cached.len().min(ctx.k_sel);
        sel[..take].copy_from_slice(&cached[..take]);
        if take > 0 { stats.reused = true; stats.reuse_hits += 1; }
        return take;
    }
    let (k_lo, k_hi, mut valid) = (q.saturating_sub(32), (q + 33).min(ctx.n), 0usize);
    for blk in 0..ctx.num_blocks {
        let Some(score) = block_far_logit(q_slice, blk, (k_lo, k_hi), layer, ctx, sp) else {
            continue;
        };
        sp.blk_scores[valid] = (score, blk as u16);
        valid += 1;
    }
    stats.pairs_evaluated += valid;
    let top_n = select_top_blocks(&sp.blk_scores[..valid], ctx.k_sel, sel);
    let mut far_blks = [0u16; 2];
    let far_cnt = sp.far_index.lookup_far_blocks(q_slice, q, ctx.blk_span, &mut far_blks);
    stats.far_hits += far_cnt;
    let merged_n = merge_far_blocks(sel, top_n, ctx.k_sel, &far_blks[..far_cnt]);
    sp.dense_cache[ctx.head_idx].store(q, sel, merged_n);
    merged_n
}

fn block_far_logit(
    q_slice: &[f32], blk: usize, local: (usize, usize),
    layer: &AttnLayerWeights, ctx: &DenseBlockCtx<'_>, sp: &SparseScratch,
) -> Option<f64> {
    let (b_lo, b_hi) = (blk * ctx.blk_span, ((blk + 1) * ctx.blk_span).min(ctx.n));
    let overlap = b_hi.min(local.1).saturating_sub(b_lo.max(local.0));
    let rem = (b_hi - b_lo).saturating_sub(overlap);
    if rem == 0 {
        return None;
    }
    let h_dim = layer.h_dim;
    let mu = &sp.mu_k[blk * h_dim..(blk + 1) * h_dim];
    let var = &sp.var_k[blk * h_dim..(blk + 1) * h_dim];
    let rb = dense_far_rel_bias(ctx.rel, ctx.all_zero, b_lo, b_hi, local.0);
    Some((rem as f64).ln() + block_cumulant_score(q_slice, mu, var, layer.inv_s) + rb)
}

fn append_coarse_blocks(
    q_slice: &[f32], q: usize, layer: &AttnLayerWeights, ctx: &DenseBlockCtx<'_>,
    sel: &[u16], sp: &SparseScratch, sm: &mut SoftmaxBuf<'_>,
) -> usize {
    let (k_lo, k_hi, mut added) = (q.saturating_sub(32), (q + 33).min(ctx.n), 0usize);
    for blk in 0..ctx.num_blocks {
        if sel.contains(&(blk as u16)) || sm.count >= sm.keys.len() {
            continue;
        }
        let Some(logit) = block_far_logit(q_slice, blk, (k_lo, k_hi), layer, ctx, sp) else {
            continue;
        };
        sm.push(blk, logit);
        added += 1;
    }
    added
}

/// Evaluate a dense head with block summaries and exact local window.
// SAFETY: Caller guarantees `K` target features and sized attention scratch buffers.
#[inline(always)]
unsafe fn attend_dense_sparse<K: HygieneKernel>(
    hr: HeadRef<'_>, rel_cfg: (&[f64; 65], bool), ids: (usize, usize), n: usize,
    sc: &mut EvalScratch, stats: &mut SparseStats,
) {
    fill_qk_nonzero_flags(hr, n, sc);
    let blk_span = n.div_ceil(MAX_LANDMARKS).max(BLOCK_SIZE);
    let k_sel = (SLAB_CAP / blk_span).clamp(1, TOP_BLOCKS);
    let num_blocks = build_blocks_into(hr, n, blk_span, false, sc);
    stats.blocks += num_blocks;
    populate_far_index(hr, n, blk_span, false, sc);
    stats.index_bytes = FAR_INDEX_BYTES;

    let (h_dim, v_dim, (layer_idx, head_idx)) = (hr.layer.h_dim, hr.layer.v_dim, ids);
    let Some(mut sp) = sc.sparse.take() else { return };
    sp.reset_slab(h_dim, v_dim);
    if sp.dense_cache.len() <= head_idx { sp.dense_cache.resize_with(head_idx + 1, IndexCache::default); }
    let reuse_cross = layer_idx > 0 && sp.dense_cache[head_idx].n_rows == n;
    if !reuse_cross { sp.dense_cache[head_idx].reset_for(n); }
    let ctx = DenseBlockCtx {
        rel: rel_cfg.0, all_zero: rel_cfg.1, head_idx, n, blk_span, num_blocks, k_sel, reuse_cross,
    };
    let slab_cfg = SlabCfg { hr, n, blk_span, active_only: false };

    for q in 0..n {
        let q_base = q * hr.layer.qk_tot + hr.head.qk_off;
        let q_slice = &sc.q_buf[q_base..q_base + h_dim];
        let (k_lo, k_hi) = (q.saturating_sub(32), (q + 33).min(n));
        let mut sel = [0u16; TOP_BLOCKS];
        let sel_cnt = select_dense_far_blocks(q_slice, q, hr.layer, &ctx, &mut sp, stats, &mut sel);
        let mut sm = SoftmaxBuf { keys: &mut sc.keys, logits: &mut sc.logits, count: 0, max_l: hr.head.sink };

        for k in k_lo..k_hi {
            let kb = k * hr.layer.qk_tot + hr.head.qk_off;
            // SAFETY: Caller verified `K` features; local Q/K slices have `h_dim` elements.
            let dot = unsafe { K::dot_f32_as_f64(q_slice, &sc.k_buf[kb..kb + h_dim]) };
            let logit = dense_rel_bias(rel_cfg.0, rel_cfg.1, q, k) + dot * hr.layer.inv_s;
            sm.push(k, logit);
        }
        stats.pairs_evaluated += k_hi - k_lo;
        if sel_cnt > 0 {
            pack_slab_cached(&slab_cfg, &sel[..sel_cnt], (&sc.k_buf, &sc.v_buf, &sc.k_nz), &mut sp, stats);
            for s_idx in 0..sp.slab_len {
                let k = sp.slab_rows[s_idx] as usize;
                if k >= k_lo && k < k_hi { continue; }
                let key_vec = &sp.slab_k[s_idx * h_dim..(s_idx + 1) * h_dim];
                // SAFETY: Caller verified `K` features; slab key slice has `h_dim` elements.
                let dot = unsafe { K::dot_f32_as_f64(q_slice, key_vec) };
                let logit = dense_rel_bias(rel_cfg.0, rel_cfg.1, q, k) + dot * hr.layer.inv_s;
                sm.push(k, logit);
                stats.pairs_evaluated += 1;
            }
        }
        let exact_cnt = sm.count;
        let coarse_added = append_coarse_blocks(q_slice, q, hr.layer, &ctx, &sel[..sel_cnt], &sp, &mut sm);
        if reuse_cross { stats.pairs_evaluated += coarse_added; }

        let (cnt, ml) = (sm.count, sm.max_l);
        // SAFETY: Caller verified `K` features; `logits[..cnt]` holds all partition terms.
        let exp_sum = (hr.head.sink - ml).exp() + unsafe { K::softmax_exp_slice(&mut sc.logits[..cnt], ml) };
        accum_exact_v(hr, q, &sc.keys[..exact_cnt], &sc.logits[..exact_cnt], exp_sum, &sc.v_buf, &mut sc.merged);
        let dst = q * hr.layer.v_tot + hr.head.v_off;
        for idx in exact_cnt..cnt {
            let (aw, mb) = (sc.logits[idx] / exp_sum, sc.keys[idx] * v_dim);
            for d in 0..v_dim { sc.merged[dst + d] += (aw * (sp.mu_v[mb + d] as f64)) as f32; }
        }
    }
    sc.sparse = Some(sp);
}

// SAFETY: Caller guarantees `K` target features and `n`-row state buffers.
#[inline(always)]
pub(super) unsafe fn sink_attention_with<K: HygieneKernel>(
    layer: &AttnLayerWeights, layer_idx: usize, n: usize, state_dim: usize,
    sc: &mut EvalScratch, mut stats: Option<&mut SparseStats>,
) {
    let approx_active = stats.is_some();
    if layer.has_qk {
        ensure_len(&mut sc.q_buf, n * layer.qk_tot);
        ensure_len(&mut sc.k_buf, n * layer.qk_tot);
        if let (Some(wq), Some(wk)) = (&layer.wq, &layer.wk) {
            // SAFETY: Caller verified `K` features; `q_buf` and `k_buf` hold `n * qk_tot`.
            unsafe {
                let qk_len = n * layer.qk_tot;
                K::linear_sparse(
                    &sc.state, n, state_dim, CsOp::new(wq, None, false),
                    &mut sc.q_buf[..qk_len], &mut sc.tile,
                );
                K::linear_sparse(
                    &sc.state, n, state_dim, CsOp::new(wk, None, false),
                    &mut sc.k_buf[..qk_len], &mut sc.tile,
                );
            }
        }
    }
    ensure_len(&mut sc.v_buf, n * layer.v_tot);
    ensure_len(&mut sc.merged, n * layer.v_tot);
    ensure_len(&mut sc.attn_out, n * state_dim);
    sc.merged[..n * layer.v_tot].fill(0.0);
    let slot_cap = if approx_active { n.max(264) } else { n };
    if sc.logits.len() < slot_cap { sc.logits.resize(slot_cap, 0.0); }
    if sc.keys.len() < slot_cap { sc.keys.resize(slot_cap, 0); }

    // SAFETY: Caller verified `K` features; `v_buf` holds `n * v_tot` elements.
    unsafe {
        K::linear_sparse(
            &sc.state, n, state_dim, CsOp::new(&layer.wv, None, false),
            &mut sc.v_buf[..n * layer.v_tot], &mut sc.tile,
        );
    }

    if layer_idx == 0 && let Some(sp) = sc.sparse.as_mut() {
        for cache in &mut sp.dense_cache { cache.n_rows = 0; }
    }
    let mut banded_cache_ready = false;
    for (head_idx, head) in layer.heads.iter().enumerate() {
        let hr = HeadRef { layer, head };
        // SAFETY: Caller verified `K` features; projection buffers hold `n` rows.
        unsafe {
            dispatch_attn_head::<K>(
                hr, (layer_idx, head_idx), n, &mut banded_cache_ready, sc, stats.as_deref_mut(),
            );
        }
    }

    // SAFETY: Caller verified `K` features; `merged` and `attn_out` match layer dims.
    unsafe {
        K::linear_sparse(
            &sc.merged[..n * layer.v_tot], n, layer.v_tot,
            CsOp::new(&layer.wo, None, false), &mut sc.attn_out[..n * state_dim], &mut sc.tile,
        );
    }
}

// SAFETY: Caller guarantees `K` target features and sized projection buffers.
#[inline(always)]
unsafe fn dispatch_attn_head<K: HygieneKernel>(
    hr: HeadRef<'_>,
    ids: (usize, usize),
    n: usize,
    banded_cache_ready: &mut bool,
    sc: &mut EvalScratch,
    stats: Option<&mut SparseStats>,
) {
    match &hr.head.kind {
        HeadKind::UniformGlobal => attend_uniform_global(hr, n, sc),
        HeadKind::StaticBanded { offsets, rel_vals, interior_weights, min_neg, max_pos } => {
            let sb = StaticBand {
                offsets, rel_vals, interior: interior_weights,
                min_neg: *min_neg, max_pos: *max_pos,
            };
            attend_static_banded(hr, sb, n, sc);
        }
        HeadKind::BandedQk { offsets, rel_vals } => {
            let is_far_banded = offsets.iter().any(|&o| o.unsigned_abs() >= 31);
            if is_far_banded && let Some(st) = stats {
                st.attn_mode = "sparse";
                // SAFETY: Caller verified `K` features for sparse far-banded attention.
                unsafe {
                    attend_banded_sparse::<K>(
                        hr, (offsets, rel_vals), n, *banded_cache_ready, sc, st,
                    );
                }
                *banded_cache_ready = true;
            } else {
                // SAFETY: Caller verified `K` features for exact banded attention.
                unsafe { attend_banded_qk_with::<K>(hr, (offsets, rel_vals), n, sc); }
            }
        }
        HeadKind::DenseQk { rel, all_zero_rel } => {
            if let Some(st) = stats {
                st.attn_mode = "sparse";
                // SAFETY: Caller verified `K` features for sparse dense-QK attention.
                unsafe {
                    attend_dense_sparse::<K>(hr, (rel, *all_zero_rel), ids, n, sc, st);
                }
                return;
            }
            let args = DenseQkArgs {
                q_proj: &sc.q_buf, k_proj: &sc.k_buf, v_proj: &sc.v_buf, num_lines: n,
                qk_tot: hr.layer.qk_tot, qk_off: hr.head.qk_off, h_dim: hr.layer.h_dim,
                v_tot: hr.layer.v_tot, v_off: hr.head.v_off, v_dim: hr.layer.v_dim,
                inv_s: hr.layer.inv_s, sink: hr.head.sink, rel, all_zero_rel: *all_zero_rel,
            };
            // SAFETY: Caller verified `K` features; `DenseQkArgs` slices cover `n` rows.
            unsafe { K::dense_qk_head(&args, &mut sc.k_tile, &mut sc.logits_4, &mut sc.merged); }
        }
        HeadKind::GeneralStatic { rel } => {
            if let Some(st) = stats {
                if st.attn_mode.is_empty() {
                    st.attn_mode = "prefix";
                }
                attend_prefix(hr, rel, n, sc, st);
            } else {
                attend_general_static_exact(hr, rel, n, sc);
            }
        }
    }
}

#[inline]
pub(super) fn sigmoid_f64(val: f64) -> f64 { 1.0 / (1.0 + (-val.clamp(-30.0, 30.0)).exp()) }

#[cfg(test)]
pub(super) fn bench_dense_qk_two_layer(n: usize) -> (f64, SparseStats) {
    use super::simd::ScalarKernel;
    use crate::testutil::TestRng;
    let dim = 16usize;
    let id_csr = || CompactCsr {
        rows: dim as u16, cols: dim as u16, row_offsets: (0..=dim as u32).collect(),
        col_indices: (0..dim as u16).collect(), values: vec![1.0f32; dim],
        active_rows: (0..dim as u16).collect(), non_zero_count: dim as u32,
    };
    let mk = || AttnLayerWeights {
        wq: Some(id_csr()), wk: Some(id_csr()), wv: id_csr(), wo: id_csr(), has_qk: true,
        qk_tot: dim, h_dim: dim, v_tot: dim, v_dim: dim, inv_s: 1.0 / (dim as f64).sqrt(),
        heads: vec![HeadPlan {
            qk_off: 0, v_off: 0, sink: 2.0,
            kind: HeadKind::DenseQk { rel: [0.0; 65], all_zero_rel: true },
        }],
    };
    let (l1, l2, mut rng) = (mk(), mk(), TestRng::new(0x51A5_E001));
    let mut base = vec![0.0f32; n * dim];
    for slot in &mut base { *slot = (((rng.next_u64() & 0xFFFF) as f32) / 65535.0 - 0.5) * 0.8; }
    let gap = 1024.min(n.saturating_sub(24));
    for off in 0..8 {
        if 16 + gap + off < n {
            for d in 0..dim {
                let v = if (d + off) % 3 == 0 { 1.8 } else { -0.2 };
                base[(16 + off) * dim + d] = v;
                base[(16 + gap + off) * dim + d] = v;
            }
        }
    }
    let run_2l = |init: Vec<f32>, mut stats: Option<&mut SparseStats>| -> Vec<f32> {
        let mut sc = EvalScratch::new();
        sc.state = init;
        // SAFETY: `ScalarKernel` requires no CPU features.
        unsafe {
            sink_attention_with::<ScalarKernel>(&l1, 0, n, dim, &mut sc, stats.as_deref_mut());
            for i in 0..n * dim { sc.state[i] = 0.5 * sc.state[i] + 0.5 * sc.attn_out[i]; }
            sink_attention_with::<ScalarKernel>(&l2, 1, n, dim, &mut sc, stats);
        }
        sc.attn_out[..n * dim].to_vec()
    };
    let mut stats = SparseStats::default();
    let approx = run_2l(base.clone(), Some(&mut stats));
    let mae = if n <= 2048 {
        let exact = run_2l(base, None);
        let diff_sum: f64 = approx
            .iter()
            .zip(exact.iter())
            .map(|(&a, &b)| ((a - b) as f64).abs())
            .sum();
        diff_sum / ((n * dim) as f64)
    } else {
        0.0
    };
    (mae, stats)
}
