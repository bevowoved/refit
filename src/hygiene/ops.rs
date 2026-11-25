use super::simd::{
    DenseQkArgs, MAX_SPAN_LEN, dense_qk_head_scalar, linear_sparse_direct_csr,
    ribbon_conv_span_scalar, sparse_parts_direct_csr,
};
#[cfg(target_arch = "x86_64")]
use super::simd::{
    dense_qk_head_avx2, dot_f32_as_f64_avx2, linear_sparse_width_gated_avx2,
    ribbon_conv_span_avx2, softmax_exp_slice_avx2, sparse_parts_width_gated_avx2,
};
use super::weights::{AttnLayerWeights, CompactCsr, HeadKind, HeadPlan, ResolvedWeights};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

pub(super) const MAX_COLS: usize = 128;
pub(super) const SKETCH_DIM: usize = 40;
pub(super) const RIBBON_GUARD: usize = 6;
pub(super) const PEAK_WINDOW: usize = 32;

pub(super) struct RingGate([f32; PEAK_WINDOW + 1]);

impl Default for RingGate {
    fn default() -> Self {
        Self([0.0; PEAK_WINDOW + 1])
    }
}

impl Deref for RingGate {
    type Target = [f32; PEAK_WINDOW + 1];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RingGate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// 4 KB L1D-resident ASCII lookup table + SWAR helper for fast byte-grid encoding.
pub(super) struct ByteLut {
    pub(super) map: [u8; 256],
    pub(super) fast_mask: [u8; 256],
    _pad: [u8; 3584],
}

pub(super) static BYTE_LUT: ByteLut = build_byte_lut();

const fn build_byte_lut() -> ByteLut {
    let mut map = [127u8; 256];
    let mut fast_mask = [0u8; 256];
    let mut b = 0usize;
    while b < 128 {
        map[b] = if b == 0 { 32 } else { b as u8 };
        if b >= 0x20 {
            fast_mask[b] = 1;
        }
        b += 1;
    }
    ByteLut { map, fast_mask, _pad: [0u8; 3584] }
}

/// Reusable thread-local scratch arena for neural hygiene evaluation.
/// Large file-length buffers are trimmed by `shrink_if_oversized` after outlier files.
#[derive(Default)]
pub(super) struct EvalScratch {
    pub(super) grid: Vec<u8>,
    pub(super) row_lens: Vec<u16>,
    // Fixed span-local buffers (<= 256 cells, L1/L2 resident)
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
    #[cfg(target_arch = "x86_64")]
    pub(super) tile_in: Vec<[f32; 8]>,
    #[cfg(target_arch = "x86_64")]
    pub(super) tile_acc: Vec<[f32; 8]>,
    // 8-row streaming tile buffers for build_hidden & cmlp
    pub(super) excl_tile: Vec<f32>,
    pub(super) saw_tile: Vec<f32>,
    pub(super) paren_tile: Vec<f32>,
    pub(super) hidden_tile: Vec<f32>,
    pub(super) stats_tile: Vec<f32>,
    pub(super) ctx_h_tile: Vec<f32>,
    pub(super) ctx_o_tile: Vec<f32>,
    pub(super) running_delta: Vec<f32>,
    // 33-row L1-resident circular ring buffer for pool_doc
    pub(super) ring_w: Vec<f32>,
    pub(super) ring_g: RingGate,
    pub(super) prev1_raw: Vec<f32>,
    pub(super) prev2_raw: Vec<f32>,
    pub(super) mean_acc: Vec<f64>,
    pub(super) peak: Vec<f32>,
    // Per-file buffers (length proportional to num_lines)
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
}

impl EvalScratch {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn ensure_span_buffers(&mut self, w: &ResolvedWeights) {
        let emb_dim = w.tok_emb.cols;
        let pre_cols = w.pre_conv3_w.cols;
        let p1_cols = w.pref1_proj.cols as usize;
        let g1_cols = w.gate1_w.cols as usize;
        let p2_cols = w.pref2_proj.cols as usize;
        let hc_dim = w.hc_dim;
        let max_conv_out = w.conv3_w.cols.max(w.conv7_w.cols).max(w.sketch_w.cols);
        let ml_cols = w.ml_proj.cols as usize;
        let max_ctx_h = (w.ctx_layers[0].cmlp_w.cols as usize)
            .max(w.ctx_layers[1].cmlp_w.cols as usize);

        ensure_len(&mut self.rib_emb, (MAX_SPAN_LEN + 2) * emb_dim);
        ensure_len(&mut self.lex, MAX_SPAN_LEN * pre_cols);
        ensure_len(&mut self.proj1, MAX_SPAN_LEN * p1_cols);
        ensure_len(&mut self.pref1, MAX_SPAN_LEN * p1_cols);
        ensure_len(&mut self.gates, MAX_SPAN_LEN * g1_cols);
        ensure_len(&mut self.proj2, MAX_SPAN_LEN * p2_cols);
        ensure_len(&mut self.pref2, MAX_SPAN_LEN * p2_cols);
        ensure_len(&mut self.fused, MAX_SPAN_LEN * hc_dim);

        ensure_len(&mut self.cells_soa, hc_dim * MAX_SPAN_LEN);
        ensure_len(&mut self.acc_soa, max_conv_out * MAX_SPAN_LEN);
        ensure_len(&mut self.csum_p1, p1_cols);
        ensure_len(&mut self.base_p1, p1_cols);
        ensure_len(&mut self.csum_p2, p2_cols);
        ensure_len(&mut self.base_p2, p2_cols);
        ensure_len(&mut self.line_sum0, hc_dim);
        ensure_len(&mut self.line_sum1, hc_dim);

        ensure_len(&mut self.excl_tile, 8 * ml_cols);
        ensure_len(&mut self.saw_tile, 8 * w.ml_gate_w.cols as usize);
        ensure_len(&mut self.paren_tile, 8 * w.ml_paren_w.cols);
        ensure_len(&mut self.hidden_tile, 8 * w.lmlp1_w.cols as usize);
        ensure_len(&mut self.stats_tile, 8 * w.lmlp2_w.cols as usize);
        ensure_len(&mut self.ctx_h_tile, 8 * max_ctx_h);
        ensure_len(&mut self.ctx_o_tile, 8 * w.state_dim);
        ensure_len(&mut self.running_delta, ml_cols);

        ensure_len(&mut self.ring_w, (PEAK_WINDOW + 1) * w.state_dim);
        ensure_len(&mut self.prev1_raw, w.state_dim);
        ensure_len(&mut self.prev2_raw, w.state_dim);
        if self.mean_acc.len() < w.state_dim {
            self.mean_acc.resize(w.state_dim, 0.0);
        }
        ensure_len(&mut self.peak, w.state_dim);
    }

    /// Release oversized per-file allocations back to the allocator after files with `> max_lines`.
    pub(super) fn shrink_if_oversized(&mut self, num_lines: usize, max_lines: usize) {
        if num_lines <= max_lines {
            return;
        }
        drop_vec(&mut self.grid);
        drop_vec(&mut self.row_lens);

        for buf in [
            &mut self.col_sums,
            &mut self.conv3,
            &mut self.conv7,
            &mut self.sketch,
            &mut self.deltas,
            &mut self.state,
            &mut self.q_buf,
            &mut self.k_buf,
            &mut self.v_buf,
            &mut self.merged,
            &mut self.attn_out,
            &mut self.gate,
        ] {
            drop_vec(buf);
        }

        for buf in [&mut self.logits, &mut self.k_tile, &mut self.logits_4] {
            drop_vec(buf);
        }
        drop_vec(&mut self.keys);
        drop_vec(&mut self.q_nz);
        drop_vec(&mut self.k_nz);
    }
}

#[inline]
fn ensure_len(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
}

#[inline]
fn drop_vec<T>(buf: &mut Vec<T>) {
    buf.clear();
    buf.shrink_to_fit();
}

#[inline]
fn swar_all_printable_ascii(chunk: [u8; 8]) -> bool {
    let word = u64::from_ne_bytes(chunk);
    let high_bits = word & 0x8080_8080_8080_8080;
    if high_bits != 0 {
        return false;
    }
    let below_space = word.wrapping_sub(0x2020_2020_2020_2020) & 0x8080_8080_8080_8080;
    below_space == 0
}

/// Encode source text into a `num_lines x 128` byte grid + `row_lens`, using an 8-byte SWAR
/// printable-ASCII fast path and falling back to exact Python `splitlines` + 2-space tab expansion.
pub(super) fn encode_grid_into(code: &str, grid: &mut Vec<u8>, row_lens: &mut Vec<u16>) -> usize {
    grid.clear();
    row_lens.clear();
    if code.is_empty() {
        grid.resize(MAX_COLS, 0);
        row_lens.push(0);
        return 1;
    }

    if let Some(num_lines) = encode_grid_swar_ascii(code.as_bytes(), grid, row_lens) {
        return num_lines;
    }
    encode_grid_general(code, grid, row_lens)
}

fn encode_grid_swar_ascii(
    bytes: &[u8],
    grid: &mut Vec<u8>,
    row_lens: &mut Vec<u16>,
) -> Option<usize> {
    let est_lines = (bytes.len() / 40).clamp(1, 4096);
    grid.reserve(est_lines * MAX_COLS);
    row_lens.reserve(est_lines);

    grid.resize(MAX_COLS, 0);
    let mut row_base = 0usize;
    let mut col = 0usize;
    let mut pos = 0usize;
    let len = bytes.len();

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
            if col < MAX_COLS {
                grid[row_base + col] = BYTE_LUT.map[b as usize];
                col += 1;
            }
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
    if !text.as_bytes().contains(&b'\t') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 32);
    let mut col = 0usize;
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

fn is_break(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{0085}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

fn split_slices(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return vec![""];
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut it = text.char_indices().peekable();
    while let Some((idx, ch)) = it.next() {
        if !is_break(ch) {
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
    if parts.len() > 1 && parts.last().is_some_and(|s| s.is_empty()) {
        parts.pop();
    }
    parts
}

fn encode_grid_general(code: &str, grid: &mut Vec<u8>, row_lens: &mut Vec<u16>) -> usize {
    grid.clear();
    row_lens.clear();
    let expanded = expand_tabs(code);
    let lines = split_slices(&expanded);
    let num_lines = lines.len();
    grid.resize(num_lines * MAX_COLS, 0);
    row_lens.resize(num_lines, 0);

    for (row, line) in lines.iter().enumerate() {
        let base = row * MAX_COLS;
        let mut count = 0usize;
        for (col, ch) in line.chars().take(MAX_COLS).enumerate() {
            let cp = ch as u32;
            let mapped = if cp > 127 {
                127u8
            } else {
                BYTE_LUT.map[cp as usize]
            };
            grid[base + col] = mapped;
            count = col + 1;
        }
        row_lens[row] = count as u16;
    }
    num_lines
}

#[inline]
pub(super) fn accumulate_row_csr(src: &[f32], weight: &CompactCsr, acc: &mut [f32]) {
    for &in_u16 in &weight.active_rows {
        let in_idx = in_u16 as usize;
        let iv = src[in_idx];
        if iv == 0.0 {
            continue;
        }
        let lo = weight.row_offsets[in_idx] as usize;
        let hi = weight.row_offsets[in_idx + 1] as usize;
        for ptr in lo..hi {
            acc[weight.col_indices[ptr] as usize] += iv * weight.values[ptr];
        }
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
    let emb_dim = w.tok_emb.cols;
    let span_len = slen0 + paired_row1.map_or(0, |(_, s1)| s1);
    rib_emb[..emb_dim].fill(0.0);
    rib_emb[(span_len + 1) * emb_dim..(span_len + 2) * emb_dim].fill(0.0);

    let tok_data = &w.tok_emb.data;
    let col_data = &w.col_emb.data;
    let b0 = row0 * MAX_COLS;
    for col in 0..slen0 {
        let code = grid[b0 + col] as usize;
        let dst = (1 + col) * emb_dim;
        let tbase = code * emb_dim;
        let cbase = col * emb_dim;
        for ch in 0..emb_dim {
            rib_emb[dst + ch] = tok_data[tbase + ch] + col_data[cbase + ch];
        }
    }
    if let Some((row1, slen1)) = paired_row1 {
        let b1 = row1 * MAX_COLS;
        for col in 0..slen1 {
            let code = grid[b1 + col] as usize;
            let dst = (1 + slen0 + col) * emb_dim;
            let tbase = code * emb_dim;
            let cbase = col * emb_dim;
            for ch in 0..emb_dim {
                rib_emb[dst + ch] = tok_data[tbase + ch] + col_data[cbase + ch];
            }
        }
    }
    span_len
}

#[inline]
fn run_pre_conv3(w: &ResolvedWeights, span_len: usize, rib_emb: &[f32], lex: &mut [f32]) {
    let emb_dim = w.tok_emb.cols;
    let pre_cols = w.pre_conv3_w.cols;
    for cell in 0..span_len {
        let dst = &mut lex[cell * pre_cols..(cell + 1) * pre_cols];
        dst.copy_from_slice(&w.pre_conv3_b);
        for tap in 0..w.pre_conv3_w.k_size {
            let src = &rib_emb[(cell + tap) * emb_dim..(cell + tap + 1) * emb_dim];
            accumulate_row_csr(src, &w.pre_conv3_w.taps[tap], dst);
        }
        for val in dst.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
    }
}

#[inline]
fn stream_line_prefix(
    proj: &[f32],
    out: &mut [f32],
    cell_lo: usize,
    cell_hi: usize,
    dim: usize,
    exclusive: bool,
    is_first_line: bool,
    csum: &mut [f32],
    base_buf: &mut [f32],
) {
    if is_first_line {
        base_buf[..dim].fill(0.0);
    } else {
        base_buf[..dim].copy_from_slice(&csum[..dim]);
    }
    for cell in cell_lo..cell_hi {
        let off = cell * dim;
        if is_first_line && cell == cell_lo {
            csum[..dim].copy_from_slice(&proj[off..off + dim]);
        } else {
            for ch in 0..dim {
                csum[ch] += proj[off + ch];
            }
        }
        for ch in 0..dim {
            let val = csum[ch] - base_buf[ch];
            out[off + ch] = if exclusive { val - proj[off + ch] } else { val };
        }
    }
}

#[inline]
fn scatter_fused_to_soa(
    w: &ResolvedWeights,
    fused: &[f32],
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    cells_soa: &mut [f32],
    line_sum0: &mut [f32],
    line_sum1: &mut [f32],
    col_sums: &mut [f32],
) -> [u64; 2] {
    let hc = w.hc_dim;
    let span_len = slen0 + paired_row1.map_or(0, |(_, s1)| s1);
    let mut active_mask = [0u64; 2];
    line_sum0[..hc].fill(0.0);

    for cell in 0..slen0 {
        let src = cell * hc;
        for ch in 0..hc {
            let val = fused[src + ch];
            cells_soa[ch * MAX_SPAN_LEN + cell] = val;
            line_sum0[ch] += val;
            if val != 0.0 {
                active_mask[ch >> 6] |= 1u64 << (ch & 63);
            }
        }
    }
    let cs_cols = w.col_sum_proj.cols as usize;
    accumulate_row_csr(
        &line_sum0[..hc],
        &w.col_sum_proj,
        &mut col_sums[row0 * cs_cols..(row0 + 1) * cs_cols],
    );

    if let Some((row1, _)) = paired_row1 {
        line_sum1[..hc].fill(0.0);
        for cell in slen0..span_len {
            let src = cell * hc;
            for ch in 0..hc {
                let val = fused[src + ch];
                cells_soa[ch * MAX_SPAN_LEN + cell] = val;
                line_sum1[ch] += val;
                if val != 0.0 {
                    active_mask[ch >> 6] |= 1u64 << (ch & 63);
                }
            }
        }
        accumulate_row_csr(
            &line_sum1[..hc],
            &w.col_sum_proj,
            &mut col_sums[row1 * cs_cols..(row1 + 1) * cs_cols],
        );
    }
    active_mask
}

/// Process a single streamed Block 0+1 span (either paired Span 0+1 or single Span `>= 2`), scalar path.
pub(super) fn eval_span_block01_scalar(
    w: &ResolvedWeights,
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    is_first_line: bool,
    sc: &mut EvalScratch,
) {
    let span_len =
        fill_span_ribbon_emb(w, &sc.grid, row0, slen0, paired_row1, &mut sc.rib_emb);
    run_pre_conv3(w, span_len, &sc.rib_emb, &mut sc.lex);

    let emb_dim = w.tok_emb.cols;
    let emb = &sc.rib_emb[emb_dim..(span_len + 1) * emb_dim];
    let pre_cols = w.pre_conv3_w.cols;
    let p1_cols = w.pref1_proj.cols as usize;
    let g1_cols = w.gate1_w.cols as usize;
    let p2_cols = w.pref2_proj.cols as usize;

    sparse_parts_direct_csr(
        &[emb, &sc.lex[..span_len * pre_cols]],
        &w.pref1_ranges,
        span_len,
        &w.pref1_proj,
        None,
        false,
        &mut sc.proj1[..span_len * p1_cols],
    );
    stream_line_prefix(
        &sc.proj1,
        &mut sc.pref1,
        0,
        slen0,
        p1_cols,
        true,
        is_first_line,
        &mut sc.csum_p1,
        &mut sc.base_p1,
    );
    if paired_row1.is_some() {
        stream_line_prefix(
            &sc.proj1,
            &mut sc.pref1,
            slen0,
            span_len,
            p1_cols,
            true,
            false,
            &mut sc.csum_p1,
            &mut sc.base_p1,
        );
    }

    sparse_parts_direct_csr(
        &[emb, &sc.lex[..span_len * pre_cols], &sc.pref1[..span_len * p1_cols]],
        &w.gate1_ranges,
        span_len,
        &w.gate1_w,
        Some(&w.gate1_b),
        true,
        &mut sc.gates[..span_len * g1_cols],
    );
    linear_sparse_direct_csr(
        &sc.gates[..span_len * g1_cols],
        span_len,
        g1_cols,
        &w.pref2_proj,
        None,
        false,
        &mut sc.proj2[..span_len * p2_cols],
    );
    stream_line_prefix(
        &sc.proj2,
        &mut sc.pref2,
        0,
        slen0,
        p2_cols,
        false,
        is_first_line,
        &mut sc.csum_p2,
        &mut sc.base_p2,
    );
    if paired_row1.is_some() {
        stream_line_prefix(
            &sc.proj2,
            &mut sc.pref2,
            slen0,
            span_len,
            p2_cols,
            false,
            false,
            &mut sc.csum_p2,
            &mut sc.base_p2,
        );
    }

    sparse_parts_direct_csr(
        &[
            emb,
            &sc.lex[..span_len * pre_cols],
            &sc.gates[..span_len * g1_cols],
            &sc.pref2[..span_len * p2_cols],
        ],
        &w.col_fuse_ranges,
        span_len,
        &w.col_fuse_w,
        Some(&w.col_fuse_b),
        true,
        &mut sc.fused[..span_len * w.hc_dim],
    );

    let active_mask = scatter_fused_to_soa(
        w,
        &sc.fused,
        row0,
        slen0,
        paired_row1,
        &mut sc.cells_soa,
        &mut sc.line_sum0,
        &mut sc.line_sum1,
        &mut sc.col_sums,
    );
    let split_at = paired_row1.map(|(r1, _)| (slen0, row0, r1));
    ribbon_conv_span_scalar(
        &sc.cells_soa,
        active_mask,
        span_len,
        &w.conv3_w,
        &w.conv3_b,
        &w.conv3_pads,
        &mut sc.acc_soa,
        split_at,
        row0,
        &mut sc.conv3,
    );
    ribbon_conv_span_scalar(
        &sc.cells_soa,
        active_mask,
        span_len,
        &w.conv7_w,
        &w.conv7_b,
        &w.conv7_pads,
        &mut sc.acc_soa,
        split_at,
        row0,
        &mut sc.conv7,
    );
    ribbon_conv_span_scalar(
        &sc.cells_soa,
        active_mask,
        span_len,
        &w.sketch_w,
        &w.sketch_b,
        &w.sketch_pads,
        &mut sc.acc_soa,
        split_at,
        row0,
        &mut sc.sketch,
    );
}

/// Process a single streamed Block 0+1 span using hoisted AVX2 width-gated kernels.
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn eval_span_block01_avx2(
    w: &ResolvedWeights,
    row0: usize,
    slen0: usize,
    paired_row1: Option<(usize, usize)>,
    is_first_line: bool,
    sc: &mut EvalScratch,
) {
    let span_len =
        fill_span_ribbon_emb(w, &sc.grid, row0, slen0, paired_row1, &mut sc.rib_emb);
    run_pre_conv3(w, span_len, &sc.rib_emb, &mut sc.lex);

    let emb_dim = w.tok_emb.cols;
    let emb = &sc.rib_emb[emb_dim..(span_len + 1) * emb_dim];
    let pre_cols = w.pre_conv3_w.cols;
    let p1_cols = w.pref1_proj.cols as usize;
    let g1_cols = w.gate1_w.cols as usize;
    let p2_cols = w.pref2_proj.cols as usize;

    // SAFETY: AVX2 is enabled by this function's `#[target_feature]`.
    unsafe {
        sparse_parts_width_gated_avx2(
            &[emb, &sc.lex[..span_len * pre_cols]],
            &w.pref1_ranges,
            span_len,
            &w.pref1_proj,
            None,
            false,
            &mut sc.proj1[..span_len * p1_cols],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }
    stream_line_prefix(
        &sc.proj1,
        &mut sc.pref1,
        0,
        slen0,
        p1_cols,
        true,
        is_first_line,
        &mut sc.csum_p1,
        &mut sc.base_p1,
    );
    if paired_row1.is_some() {
        stream_line_prefix(
            &sc.proj1,
            &mut sc.pref1,
            slen0,
            span_len,
            p1_cols,
            true,
            false,
            &mut sc.csum_p1,
            &mut sc.base_p1,
        );
    }

    // SAFETY: AVX2 is enabled by this function's `#[target_feature]`.
    unsafe {
        sparse_parts_width_gated_avx2(
            &[emb, &sc.lex[..span_len * pre_cols], &sc.pref1[..span_len * p1_cols]],
            &w.gate1_ranges,
            span_len,
            &w.gate1_w,
            Some(&w.gate1_b),
            true,
            &mut sc.gates[..span_len * g1_cols],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
        linear_sparse_width_gated_avx2(
            &sc.gates[..span_len * g1_cols],
            span_len,
            g1_cols,
            &w.pref2_proj,
            None,
            false,
            &mut sc.proj2[..span_len * p2_cols],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }
    stream_line_prefix(
        &sc.proj2,
        &mut sc.pref2,
        0,
        slen0,
        p2_cols,
        false,
        is_first_line,
        &mut sc.csum_p2,
        &mut sc.base_p2,
    );
    if paired_row1.is_some() {
        stream_line_prefix(
            &sc.proj2,
            &mut sc.pref2,
            slen0,
            span_len,
            p2_cols,
            false,
            false,
            &mut sc.csum_p2,
            &mut sc.base_p2,
        );
    }

    // SAFETY: AVX2 is enabled by this function's `#[target_feature]`.
    unsafe {
        sparse_parts_width_gated_avx2(
            &[
                emb,
                &sc.lex[..span_len * pre_cols],
                &sc.gates[..span_len * g1_cols],
                &sc.pref2[..span_len * p2_cols],
            ],
            &w.col_fuse_ranges,
            span_len,
            &w.col_fuse_w,
            Some(&w.col_fuse_b),
            true,
            &mut sc.fused[..span_len * w.hc_dim],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }

    let active_mask = scatter_fused_to_soa(
        w,
        &sc.fused,
        row0,
        slen0,
        paired_row1,
        &mut sc.cells_soa,
        &mut sc.line_sum0,
        &mut sc.line_sum1,
        &mut sc.col_sums,
    );
    let split_at = paired_row1.map(|(r1, _)| (slen0, row0, r1));
    // SAFETY: AVX2 is enabled and `cells_soa`/`acc_soa` are sized for `MAX_SPAN_LEN`.
    unsafe {
        ribbon_conv_span_avx2(
            &sc.cells_soa,
            active_mask,
            span_len,
            &w.conv3_w,
            &w.conv3_b,
            &w.conv3_pads,
            &mut sc.acc_soa,
            split_at,
            row0,
            &mut sc.conv3,
        );
        ribbon_conv_span_avx2(
            &sc.cells_soa,
            active_mask,
            span_len,
            &w.conv7_w,
            &w.conv7_b,
            &w.conv7_pads,
            &mut sc.acc_soa,
            split_at,
            row0,
            &mut sc.conv7,
        );
        ribbon_conv_span_avx2(
            &sc.cells_soa,
            active_mask,
            span_len,
            &w.sketch_w,
            &w.sketch_b,
            &w.sketch_pads,
            &mut sc.acc_soa,
            split_at,
            row0,
            &mut sc.sketch,
        );
    }
}

pub(super) fn norm_sketch(sketch: &mut [f32], num_lines: usize, total_sketch: usize) {
    let total = num_lines * total_sketch;
    for val in &mut sketch[..total] {
        *val = val.clamp(0.0, 1.0);
    }
    let mut base = 0usize;
    while base < total {
        let end = (base + SKETCH_DIM).min(total);
        let mut norm_sq = 0.0f32;
        for &val in &sketch[base..end] {
            norm_sq += val * val;
        }
        let norm = (norm_sq + 1e-6f32).sqrt();
        for val in &mut sketch[base..end] {
            *val /= norm;
        }
        base += SKETCH_DIM;
    }
}

#[inline]
pub(super) fn axis_conv_row(
    deltas: &[f32],
    num_lines: usize,
    row: usize,
    w: &ResolvedWeights,
    dst: &mut [f32],
) {
    let conv = &w.ml_paren_w;
    let pad = ((conv.k_size - 1) >> 1) as isize;
    dst.copy_from_slice(&w.ml_paren_b);
    let lo = (pad - row as isize).max(0) as usize;
    let hi = (num_lines as isize - row as isize + pad).min(conv.k_size as isize) as usize;
    for tap in lo..hi {
        let tm = &conv.taps[tap];
        if tm.non_zero_count == 0 {
            continue;
        }
        let src_row = (row as isize + tap as isize - pad) as usize;
        let src = &deltas[src_row * conv.rows..(src_row + 1) * conv.rows];
        accumulate_row_csr(src, tm, dst);
    }
    for val in dst.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}

fn attend_uniform_global(
    v_proj: &[f32],
    n: usize,
    v_tot: usize,
    v_off: usize,
    v_dim: usize,
    sink: f64,
    merged: &mut [f32],
) {
    let weight = (1.0f64 / (n as f64 + sink.exp())) as f32;
    let mut sum = vec![0.0f64; v_dim];
    for row in 0..n {
        let base = row * v_tot + v_off;
        for d in 0..v_dim {
            sum[d] += v_proj[base + d] as f64;
        }
    }
    let wf = weight as f64;
    for row in 0..n {
        let base = row * v_tot + v_off;
        for d in 0..v_dim {
            merged[base + d] = (sum[d] * wf) as f32;
        }
    }
}

fn attend_static_banded(
    v_proj: &[f32],
    n: usize,
    v_tot: usize,
    v_off: usize,
    v_dim: usize,
    sink: f64,
    offsets: &[i8],
    rel_vals: &[f64],
    interior_weights: &[f64],
    min_neg: usize,
    max_pos: usize,
    logits: &mut [f64],
    keys: &mut [usize],
    merged: &mut [f32],
) {
    for q in 0..n {
        let dst = q * v_tot + v_off;
        if q >= min_neg && q + max_pos < n {
            for (&off, &aw) in offsets.iter().zip(interior_weights.iter()) {
                let k = (q as isize + off as isize) as usize;
                let vb = k * v_tot + v_off;
                for d in 0..v_dim {
                    merged[dst + d] += (aw * (v_proj[vb + d] as f64)) as f32;
                }
            }
            continue;
        }

        let mut count = 0usize;
        let mut max_l = sink;
        for (&off, &rv) in offsets.iter().zip(rel_vals.iter()) {
            let k_isize = q as isize + off as isize;
            if k_isize < 0 || k_isize >= n as isize {
                continue;
            }
            keys[count] = k_isize as usize;
            logits[count] = rv;
            max_l = max_l.max(rv);
            count += 1;
        }
        let mut exp_sum = (sink - max_l).exp();
        for lv in &mut logits[..count] {
            *lv = (*lv - max_l).exp();
            exp_sum += *lv;
        }
        for idx in 0..count {
            let aw = logits[idx] / exp_sum;
            let vb = keys[idx] * v_tot + v_off;
            for d in 0..v_dim {
                merged[dst + d] += (aw * (v_proj[vb + d] as f64)) as f32;
            }
        }
    }
}

fn attend_general_static(
    v_proj: &[f32],
    n: usize,
    v_tot: usize,
    v_off: usize,
    v_dim: usize,
    sink: f64,
    rel: &[f64; 65],
    logits: &mut [f64],
    merged: &mut [f32],
) {
    for q in 0..n {
        let mut max_l = sink;
        for k in 0..n {
            let dist = (k as isize - q as isize).clamp(-32, 32);
            let rv = rel[(dist + 32) as usize];
            logits[k] = rv;
            max_l = max_l.max(rv);
        }
        let mut exp_sum = (sink - max_l).exp();
        for lv in &mut logits[..n] {
            *lv = (*lv - max_l).exp();
            exp_sum += *lv;
        }
        let dst = q * v_tot + v_off;
        for k in 0..n {
            let aw = logits[k] / exp_sum;
            let vb = k * v_tot + v_off;
            for d in 0..v_dim {
                merged[dst + d] += (aw * (v_proj[vb + d] as f64)) as f32;
            }
        }
    }
}

fn fill_qk_nonzero_flags(
    q_buf: &[f32],
    k_buf: &[f32],
    n: usize,
    qk_tot: usize,
    qk_off: usize,
    h_dim: usize,
    q_nz: &mut Vec<bool>,
    k_nz: &mut Vec<bool>,
) {
    q_nz.resize(n, false);
    k_nz.resize(n, false);
    for row in 0..n {
        let base = row * qk_tot + qk_off;
        q_nz[row] = q_buf[base..base + h_dim].iter().any(|&v| v != 0.0);
        k_nz[row] = k_buf[base..base + h_dim].iter().any(|&v| v != 0.0);
    }
}

fn attend_banded_qk_scalar(
    layer: &AttnLayerWeights,
    head: &HeadPlan,
    offsets: &[i8],
    rel_vals: &[f64],
    n: usize,
    sc: &mut EvalScratch,
) {
    fill_qk_nonzero_flags(
        &sc.q_buf,
        &sc.k_buf,
        n,
        layer.qk_tot,
        head.qk_off,
        layer.h_dim,
        &mut sc.q_nz,
        &mut sc.k_nz,
    );
    for q in 0..n {
        let q_active = sc.q_nz[q];
        let q_base = q * layer.qk_tot + head.qk_off;
        let mut count = 0usize;
        let mut max_l = head.sink;
        for (&off, &rv) in offsets.iter().zip(rel_vals.iter()) {
            let k_isize = q as isize + off as isize;
            if k_isize < 0 || k_isize >= n as isize {
                continue;
            }
            let k = k_isize as usize;
            let qk = if q_active && sc.k_nz[k] {
                let k_base = k * layer.qk_tot + head.qk_off;
                let mut dot = 0.0f64;
                for d in 0..layer.h_dim {
                    dot += (sc.q_buf[q_base + d] as f64) * (sc.k_buf[k_base + d] as f64);
                }
                dot * layer.inv_s
            } else {
                0.0
            };
            let logit = rv + qk;
            sc.keys[count] = k;
            sc.logits[count] = logit;
            max_l = max_l.max(logit);
            count += 1;
        }
        let mut exp_sum = (head.sink - max_l).exp();
        for lv in &mut sc.logits[..count] {
            *lv = (*lv - max_l).exp();
            exp_sum += *lv;
        }
        let dst = q * layer.v_tot + head.v_off;
        for idx in 0..count {
            let aw = sc.logits[idx] / exp_sum;
            let vb = sc.keys[idx] * layer.v_tot + head.v_off;
            for d in 0..layer.v_dim {
                sc.merged[dst + d] += (aw * (sc.v_buf[vb + d] as f64)) as f32;
            }
        }
    }
}

// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn attend_banded_qk_avx2(
    layer: &AttnLayerWeights,
    head: &HeadPlan,
    offsets: &[i8],
    rel_vals: &[f64],
    n: usize,
    sc: &mut EvalScratch,
) {
    fill_qk_nonzero_flags(
        &sc.q_buf,
        &sc.k_buf,
        n,
        layer.qk_tot,
        head.qk_off,
        layer.h_dim,
        &mut sc.q_nz,
        &mut sc.k_nz,
    );
    for q in 0..n {
        let q_active = sc.q_nz[q];
        let q_base = q * layer.qk_tot + head.qk_off;
        let q_slice = &sc.q_buf[q_base..q_base + layer.h_dim];
        let mut count = 0usize;
        let mut max_l = head.sink;

        for (&off, &rv) in offsets.iter().zip(rel_vals.iter()) {
            let k_isize = q as isize + off as isize;
            if k_isize < 0 || k_isize >= n as isize {
                continue;
            }
            let k = k_isize as usize;
            let qk = if q_active && sc.k_nz[k] {
                let k_base = k * layer.qk_tot + head.qk_off;
                let k_slice = &sc.k_buf[k_base..k_base + layer.h_dim];
                // SAFETY: AVX2 is active and both slices have length `h_dim`.
                unsafe { dot_f32_as_f64_avx2(q_slice, k_slice) * layer.inv_s }
            } else {
                0.0
            };
            let logit = rv + qk;
            sc.keys[count] = k;
            sc.logits[count] = logit;
            max_l = max_l.max(logit);
            count += 1;
        }

        // SAFETY: AVX2 is active in this function.
        let exp_sum = (head.sink - max_l).exp()
            + unsafe { softmax_exp_slice_avx2(&mut sc.logits[..count], max_l) };
        let dst = q * layer.v_tot + head.v_off;
        for idx in 0..count {
            let aw = sc.logits[idx] / exp_sum;
            let vb = sc.keys[idx] * layer.v_tot + head.v_off;
            for d in 0..layer.v_dim {
                sc.merged[dst + d] += (aw * (sc.v_buf[vb + d] as f64)) as f32;
            }
        }
    }
}

pub(super) fn sink_attention_scalar(
    layer: &AttnLayerWeights,
    n: usize,
    state_dim: usize,
    sc: &mut EvalScratch,
) {
    if layer.has_qk {
        ensure_len(&mut sc.q_buf, n * layer.qk_tot);
        ensure_len(&mut sc.k_buf, n * layer.qk_tot);
        if let (Some(wq), Some(wk)) = (&layer.wq, &layer.wk) {
            linear_sparse_direct_csr(
                &sc.state,
                n,
                state_dim,
                wq,
                None,
                false,
                &mut sc.q_buf[..n * layer.qk_tot],
            );
            linear_sparse_direct_csr(
                &sc.state,
                n,
                state_dim,
                wk,
                None,
                false,
                &mut sc.k_buf[..n * layer.qk_tot],
            );
        }
    }
    ensure_len(&mut sc.v_buf, n * layer.v_tot);
    ensure_len(&mut sc.merged, n * layer.v_tot);
    ensure_len(&mut sc.attn_out, n * state_dim);
    sc.merged[..n * layer.v_tot].fill(0.0);
    if sc.logits.len() < n {
        sc.logits.resize(n, 0.0);
    }
    if sc.keys.len() < n {
        sc.keys.resize(n, 0);
    }

    linear_sparse_direct_csr(
        &sc.state,
        n,
        state_dim,
        &layer.wv,
        None,
        false,
        &mut sc.v_buf[..n * layer.v_tot],
    );

    for head in &layer.heads {
        match &head.kind {
            HeadKind::UniformGlobal => {
                attend_uniform_global(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    &mut sc.merged,
                );
            }
            HeadKind::StaticBanded {
                offsets,
                rel_vals,
                interior_weights,
                min_neg,
                max_pos,
            } => {
                attend_static_banded(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    offsets,
                    rel_vals,
                    interior_weights,
                    *min_neg,
                    *max_pos,
                    &mut sc.logits,
                    &mut sc.keys,
                    &mut sc.merged,
                );
            }
            HeadKind::BandedQk { offsets, rel_vals } => {
                attend_banded_qk_scalar(layer, head, offsets, rel_vals, n, sc);
            }
            HeadKind::DenseQk { rel, all_zero_rel } => {
                let args = DenseQkArgs {
                    q_proj: &sc.q_buf,
                    k_proj: &sc.k_buf,
                    v_proj: &sc.v_buf,
                    num_lines: n,
                    qk_tot: layer.qk_tot,
                    qk_off: head.qk_off,
                    h_dim: layer.h_dim,
                    v_tot: layer.v_tot,
                    v_off: head.v_off,
                    v_dim: layer.v_dim,
                    inv_s: layer.inv_s,
                    sink: head.sink,
                    rel,
                    all_zero_rel: *all_zero_rel,
                };
                dense_qk_head_scalar(&args, &mut sc.k_tile, &mut sc.logits_4, &mut sc.merged);
            }
            HeadKind::GeneralStatic { rel } => {
                attend_general_static(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    rel,
                    &mut sc.logits,
                    &mut sc.merged,
                );
            }
        }
    }

    linear_sparse_direct_csr(
        &sc.merged[..n * layer.v_tot],
        n,
        layer.v_tot,
        &layer.wo,
        None,
        false,
        &mut sc.attn_out[..n * state_dim],
    );
}

// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sink_attention_avx2(
    layer: &AttnLayerWeights,
    n: usize,
    state_dim: usize,
    sc: &mut EvalScratch,
) {
    if layer.has_qk {
        ensure_len(&mut sc.q_buf, n * layer.qk_tot);
        ensure_len(&mut sc.k_buf, n * layer.qk_tot);
        if let (Some(wq), Some(wk)) = (&layer.wq, &layer.wk) {
            // SAFETY: AVX2 is active; width gate automatically chooses direct CSR for cols > 128.
            unsafe {
                linear_sparse_width_gated_avx2(
                    &sc.state,
                    n,
                    state_dim,
                    wq,
                    None,
                    false,
                    &mut sc.q_buf[..n * layer.qk_tot],
                    &mut sc.tile_in,
                    &mut sc.tile_acc,
                );
                linear_sparse_width_gated_avx2(
                    &sc.state,
                    n,
                    state_dim,
                    wk,
                    None,
                    false,
                    &mut sc.k_buf[..n * layer.qk_tot],
                    &mut sc.tile_in,
                    &mut sc.tile_acc,
                );
            }
        }
    }
    ensure_len(&mut sc.v_buf, n * layer.v_tot);
    ensure_len(&mut sc.merged, n * layer.v_tot);
    ensure_len(&mut sc.attn_out, n * state_dim);
    sc.merged[..n * layer.v_tot].fill(0.0);
    if sc.logits.len() < n {
        sc.logits.resize(n, 0.0);
    }
    if sc.keys.len() < n {
        sc.keys.resize(n, 0);
    }

    // SAFETY: AVX2 is active in this function.
    unsafe {
        linear_sparse_width_gated_avx2(
            &sc.state,
            n,
            state_dim,
            &layer.wv,
            None,
            false,
            &mut sc.v_buf[..n * layer.v_tot],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }

    for head in &layer.heads {
        match &head.kind {
            HeadKind::UniformGlobal => {
                attend_uniform_global(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    &mut sc.merged,
                );
            }
            HeadKind::StaticBanded {
                offsets,
                rel_vals,
                interior_weights,
                min_neg,
                max_pos,
            } => {
                attend_static_banded(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    offsets,
                    rel_vals,
                    interior_weights,
                    *min_neg,
                    *max_pos,
                    &mut sc.logits,
                    &mut sc.keys,
                    &mut sc.merged,
                );
            }
            HeadKind::BandedQk { offsets, rel_vals } => {
                // SAFETY: AVX2 is active in this function.
                unsafe {
                    attend_banded_qk_avx2(layer, head, offsets, rel_vals, n, sc);
                }
            }
            HeadKind::DenseQk { rel, all_zero_rel } => {
                let args = DenseQkArgs {
                    q_proj: &sc.q_buf,
                    k_proj: &sc.k_buf,
                    v_proj: &sc.v_buf,
                    num_lines: n,
                    qk_tot: layer.qk_tot,
                    qk_off: head.qk_off,
                    h_dim: layer.h_dim,
                    v_tot: layer.v_tot,
                    v_off: head.v_off,
                    v_dim: layer.v_dim,
                    inv_s: layer.inv_s,
                    sink: head.sink,
                    rel,
                    all_zero_rel: *all_zero_rel,
                };
                // SAFETY: AVX2 is active in this function.
                unsafe {
                    dense_qk_head_avx2(&args, &mut sc.k_tile, &mut sc.logits_4, &mut sc.merged);
                }
            }
            HeadKind::GeneralStatic { rel } => {
                attend_general_static(
                    &sc.v_buf,
                    n,
                    layer.v_tot,
                    head.v_off,
                    layer.v_dim,
                    head.sink,
                    rel,
                    &mut sc.logits,
                    &mut sc.merged,
                );
            }
        }
    }

    // SAFETY: AVX2 is active in this function.
    unsafe {
        linear_sparse_width_gated_avx2(
            &sc.merged[..n * layer.v_tot],
            n,
            layer.v_tot,
            &layer.wo,
            None,
            false,
            &mut sc.attn_out[..n * state_dim],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }
}

#[inline]
pub(super) fn sigmoid_f64(val: f64) -> f64 {
    1.0 / (1.0 + (-val.clamp(-30.0, 30.0)).exp())
}
