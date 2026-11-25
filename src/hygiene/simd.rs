use super::weights::{CompactConv, CompactCsr, PadTap, PartRange};

/// Empirical L1D width gate: 8-line AVX2 tiling wins when `out_cols <= 128`.
#[cfg(target_arch = "x86_64")]
pub(super) const WIDE_OUT_GATE: usize = 128;
pub(super) const MAX_SPAN_LEN: usize = 256;
pub(super) const BK_ANTI_ALIAS: usize = 252;
pub(super) const STRIDE_ANTI_ALIAS: usize = 264;

/// Architecture-neutral scratch holder for 8-line AVX2 vertical tiling (`0` bytes off x86_64).
#[derive(Default)]
pub(super) struct TileScratch {
    #[cfg(target_arch = "x86_64")]
    pub(super) tile_in: Vec<[f32; 8]>,
    #[cfg(target_arch = "x86_64")]
    pub(super) tile_acc: Vec<[f32; 8]>,
}

#[derive(Clone, Copy)]
pub(super) struct CsOp<'a> {
    pub(super) weight: &'a CompactCsr,
    pub(super) bias: Option<&'a [f32]>,
    pub(super) relu: bool,
}

impl<'a> CsOp<'a> {
    #[inline]
    pub(super) const fn new(weight: &'a CompactCsr, bias: Option<&'a [f32]>, relu: bool) -> Self {
        Self { weight, bias, relu }
    }
}

pub(super) struct RibbonSpan<'a> {
    pub(super) cells_soa: &'a [f32],
    pub(super) active_mask: [u64; 2],
    pub(super) span_len: usize,
    pub(super) split_at: Option<(usize, usize, usize)>,
    pub(super) single_row: usize,
}

pub(super) struct DenseQkArgs<'a> {
    pub(super) q_proj: &'a [f32],
    pub(super) k_proj: &'a [f32],
    pub(super) v_proj: &'a [f32],
    pub(super) num_lines: usize,
    pub(super) qk_tot: usize,
    pub(super) qk_off: usize,
    pub(super) h_dim: usize,
    pub(super) v_tot: usize,
    pub(super) v_off: usize,
    pub(super) v_dim: usize,
    pub(super) inv_s: f64,
    pub(super) sink: f64,
    pub(super) rel: &'a [f64; 65],
    pub(super) all_zero_rel: bool,
}

/// Monomorphized scalar / AVX2 execution kernel eliminating all mirrored loops in `ops`/`model`.
pub(super) trait HygieneKernel {
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn linear_sparse(
        inp: &[f32], num_rows: usize, in_cols: usize, op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
    );
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn sparse_parts(
        parts: &[&[f32]], ranges: &[PartRange], num_rows: usize,
        op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
    );
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn ribbon_conv_span(
        span: &RibbonSpan<'_>, conv: &CompactConv, bias: &[f32],
        pad_taps: &[PadTap], acc_soa: &mut [f32], out: &mut [f32],
    );
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn dot_f32_as_f64(q_slice: &[f32], k_slice: &[f32]) -> f64;
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn softmax_exp_slice(row_l: &mut [f64], max_l: f64) -> f64;
    // SAFETY: Caller guarantees AVX2 when invoking `Avx2Kernel`.
    unsafe fn dense_qk_head(
        args: &DenseQkArgs<'_>, k_tile: &mut Vec<f64>, logits_4: &mut Vec<f64>, merged: &mut [f32],
    );
}

pub(super) struct ScalarKernel;

impl HygieneKernel for ScalarKernel {
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn linear_sparse(
        inp: &[f32], num_rows: usize, in_cols: usize,
        op: CsOp<'_>, out: &mut [f32], _tile: &mut TileScratch,
    ) { linear_sparse_direct_csr(inp, num_rows, in_cols, op, out); }
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn sparse_parts(
        parts: &[&[f32]], ranges: &[PartRange], num_rows: usize,
        op: CsOp<'_>, out: &mut [f32], _tile: &mut TileScratch,
    ) { sparse_parts_direct_csr(parts, ranges, num_rows, op, out); }
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn ribbon_conv_span(
        span: &RibbonSpan<'_>, conv: &CompactConv, bias: &[f32],
        pad_taps: &[PadTap], acc_soa: &mut [f32], out: &mut [f32],
    ) { ribbon_conv_span_with(span, conv, bias, pad_taps, acc_soa, out, accum_tap_soa_scalar); }
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn dot_f32_as_f64(q: &[f32], k: &[f32]) -> f64 { dot_f32_as_f64_scalar(q, k) }
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn softmax_exp_slice(row_l: &mut [f64], max_l: f64) -> f64 {
        softmax_exp_slice_scalar(row_l, max_l)
    }
    // SAFETY: Scalar fallback has no CPU feature requirements.
    #[inline(always)]
    unsafe fn dense_qk_head(
        args: &DenseQkArgs<'_>, k_tile: &mut Vec<f64>, logits_4: &mut Vec<f64>, merged: &mut [f32],
    ) {
        dense_qk_head_impl(
            args, k_tile, logits_4, merged, dot_bq4_scalar, softmax_exp_slice_scalar,
        );
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) struct Avx2Kernel;

#[cfg(target_arch = "x86_64")]
impl HygieneKernel for Avx2Kernel {
    // SAFETY: Caller guarantees AVX2 via enclosing `#[target_feature(enable = "avx2")]`.
    #[inline(always)]
    unsafe fn linear_sparse(
        inp: &[f32], num_rows: usize, in_cols: usize,
        op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
    ) {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { linear_sparse_width_gated_avx2(inp, num_rows, in_cols, op, out, tile); }
    }
    // SAFETY: Caller guarantees AVX2 via enclosing `#[target_feature(enable = "avx2")]`.
    #[inline(always)]
    unsafe fn sparse_parts(
        parts: &[&[f32]], ranges: &[PartRange], num_rows: usize,
        op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
    ) {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { sparse_parts_width_gated_avx2(parts, ranges, num_rows, op, out, tile); }
    }
    // SAFETY: Caller guarantees AVX2 and `MAX_SPAN_LEN`-strided SoA buffers.
    #[inline(always)]
    unsafe fn ribbon_conv_span(
        span: &RibbonSpan<'_>, conv: &CompactConv, bias: &[f32],
        pad_taps: &[PadTap], acc_soa: &mut [f32], out: &mut [f32],
    ) {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { ribbon_conv_span_avx2(span, conv, bias, pad_taps, acc_soa, out); }
    }
    // SAFETY: Caller guarantees AVX2 and valid slice bounds.
    #[inline(always)]
    unsafe fn dot_f32_as_f64(q: &[f32], k: &[f32]) -> f64 {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { dot_f32_as_f64_avx2(q, k) }
    }
    // SAFETY: Caller guarantees AVX2.
    #[inline(always)]
    unsafe fn softmax_exp_slice(row_l: &mut [f64], max_l: f64) -> f64 {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { softmax_exp_slice_avx2(row_l, max_l) }
    }
    // SAFETY: Caller guarantees AVX2.
    #[inline(always)]
    unsafe fn dense_qk_head(
        args: &DenseQkArgs<'_>, k_tile: &mut Vec<f64>, logits_4: &mut Vec<f64>, merged: &mut [f32],
    ) {
        // SAFETY: AVX2 active in caller boundary.
        unsafe { dense_qk_head_avx2(args, k_tile, logits_4, merged); }
    }
}

/// Pure scalar direct compact-CSR matrix multiply (`out_cols` arbitrary, zero transpose).
#[inline]
pub(super) fn linear_sparse_direct_csr(
    inp: &[f32], num_rows: usize, in_cols: usize, op: CsOp<'_>, out: &mut [f32],
) {
    let out_cols = op.weight.cols as usize;
    for row in 0..num_rows {
        let dst = &mut out[row * out_cols..(row + 1) * out_cols];
        init_row_bias(dst, op.bias);
        let base = row * in_cols;
        for &in_u16 in &op.weight.active_rows {
            let in_idx = in_u16 as usize;
            let iv = inp[base + in_idx];
            if iv != 0.0 { op.weight.accum_row(in_idx, iv, dst); }
        }
        if op.relu { relu_slice_in_place(dst); }
    }
}

#[inline]
pub(super) fn sparse_parts_direct_csr(
    parts: &[&[f32]], ranges: &[PartRange], num_rows: usize, op: CsOp<'_>, out: &mut [f32],
) {
    let out_cols = op.weight.cols as usize;
    for row in 0..num_rows {
        let dst = &mut out[row * out_cols..(row + 1) * out_cols];
        init_row_bias(dst, op.bias);
        for rg in ranges {
            let (src, row_base) = (parts[rg.part_idx], row * rg.dim);
            for &in_u16 in &op.weight.active_rows[rg.start..rg.end] {
                let in_idx = in_u16 as usize;
                let iv = src[row_base + (in_idx - rg.in_off)];
                if iv != 0.0 { op.weight.accum_row(in_idx, iv, dst); }
            }
        }
        if op.relu { relu_slice_in_place(dst); }
    }
}

#[inline]
fn init_row_bias(dst: &mut [f32], bias: Option<&[f32]>) {
    match bias { Some(b) => dst.copy_from_slice(b), None => dst.fill(0.0) }
}

#[inline]
fn relu_slice_in_place(dst: &mut [f32]) {
    for val in dst.iter_mut() { if *val < 0.0 { *val = 0.0; } }
}

#[inline]
fn dot_f32_as_f64_scalar(q: &[f32], k: &[f32]) -> f64 {
    q.iter().zip(k.iter()).fold(0.0f64, |acc, (&a, &b)| acc + (a as f64) * (b as f64))
}

#[inline]
fn softmax_exp_slice_scalar(row_l: &mut [f64], max_l: f64) -> f64 {
    let mut sum = 0.0f64;
    for lv in row_l.iter_mut() {
        *lv = (*lv - max_l).exp();
        sum += *lv;
    }
    sum
}

#[inline(always)]
fn ribbon_conv_span_with<F>(
    span: &RibbonSpan<'_>, conv: &CompactConv, bias: &[f32], pad_taps: &[PadTap],
    acc_soa: &mut [f32], out: &mut [f32], mut accum_tap: F,
) where F: FnMut(&RibbonSpan<'_>, &CompactCsr, usize, usize, usize, &mut [f32]) {
    if conv.non_zero_count == 0 { return; }
    let (out_dim, pad) = (conv.cols, (conv.k_size - 1) >> 1);
    for (oc, &bv) in bias.iter().enumerate().take(out_dim) {
        acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span.span_len].fill(bv);
    }
    for tap in 0..conv.k_size {
        let tm = &conv.taps[tap];
        if tm.non_zero_count == 0 { continue; }
        let shift = tap as isize - pad as isize;
        let col_lo = (-shift).max(0) as usize;
        let col_hi = (span.span_len as isize - shift).clamp(0, span.span_len as isize) as usize;
        apply_pad_cols(acc_soa, &pad_taps[tap], 0, col_lo.min(span.span_len));
        apply_pad_cols(acc_soa, &pad_taps[tap], col_hi, span.span_len);
        if col_lo < col_hi {
            accum_tap(span, tm, (col_lo as isize + shift) as usize, col_lo, col_hi, acc_soa);
        }
    }
    reduce_span_relu(acc_soa, out_dim, span, out);
}

fn accum_tap_soa_scalar(
    span: &RibbonSpan<'_>, tm: &CompactCsr, src_start: usize, col_lo: usize, col_hi: usize, acc_soa: &mut [f32],
) {
    let len = col_hi - col_lo;
    for &ch_u16 in &tm.active_rows {
        let ch = ch_u16 as usize;
        if (span.active_mask[ch >> 6] & (1u64 << (ch & 63))) == 0 { continue; }
        let src = &span.cells_soa[ch * MAX_SPAN_LEN + src_start..ch * MAX_SPAN_LEN + src_start + len];
        let (lo, hi) = (tm.row_offsets[ch] as usize, tm.row_offsets[ch + 1] as usize);
        for ptr in lo..hi {
            let (oc, wv) = (tm.col_indices[ptr] as usize, tm.values[ptr]);
            let dst = &mut acc_soa[oc * MAX_SPAN_LEN + col_lo..oc * MAX_SPAN_LEN + col_hi];
            for (d, &s) in dst.iter_mut().zip(src.iter()) { *d += s * wv; }
        }
    }
}

#[inline]
fn apply_pad_cols(acc_soa: &mut [f32], pad: &PadTap, lo: usize, hi: usize) {
    for col in lo..hi {
        for (&oc, &pv) in pad.cols.iter().zip(pad.vals.iter()) {
            acc_soa[oc as usize * MAX_SPAN_LEN + col] += pv;
        }
    }
}

#[inline]
fn reduce_span_relu(acc_soa: &[f32], out_dim: usize, span: &RibbonSpan<'_>, out: &mut [f32]) {
    let sum_pos = |s: &[f32]| s.iter().copied().filter(|&v| v > 0.0).sum::<f32>();
    if let Some((len0, row0, row1)) = span.split_at {
        let (d0, d1) = (row0 * out_dim, row1 * out_dim);
        for oc in 0..out_dim {
            let row_acc = &acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span.span_len];
            out[d0 + oc] = sum_pos(&row_acc[..len0]);
            out[d1 + oc] = sum_pos(&row_acc[len0..span.span_len]);
        }
    } else {
        let d0 = span.single_row * out_dim;
        for oc in 0..out_dim {
            let row_acc = &acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span.span_len];
            out[d0 + oc] = sum_pos(row_acc);
        }
    }
}

#[inline]
fn rel_bias_for(args: &DenseQkArgs<'_>, q_idx: usize, k_idx: usize) -> f64 {
    if args.all_zero_rel { return 0.0; }
    args.rel[((k_idx as isize - q_idx as isize).clamp(-32, 32) + 32) as usize]
}

fn accum_dense_v(args: &DenseQkArgs<'_>, q_idx: usize, weights: &[f64], exp_sum: f64, merged: &mut [f32]) {
    let dst = q_idx * args.v_tot + args.v_off;
    for (k_row, &ew) in weights.iter().enumerate().take(args.num_lines) {
        let (aw, vb) = (ew / exp_sum, k_row * args.v_tot + args.v_off);
        for d in 0..args.v_dim { merged[dst + d] += (aw * (args.v_proj[vb + d] as f64)) as f32; }
    }
}

struct TileRange { q_row: usize, k_start: usize, bq: usize, blen: usize }

#[inline(always)]
fn dense_qk_head_impl<D, S>(
    args: &DenseQkArgs<'_>, k_tile: &mut Vec<f64>, logits_4: &mut Vec<f64>, merged: &mut [f32],
    mut dot_bq4: D, mut softmax_row: S,
) where D: FnMut(&DenseQkArgs<'_>, &[[f64; 4]], &[f64], TileRange, &mut [f64]), S: FnMut(&mut [f64], f64) -> f64 {
    k_tile.resize(args.h_dim * STRIDE_ANTI_ALIAS, 0.0);
    logits_4.resize(4 * args.num_lines, 0.0);
    let mut q_pack = vec![[0.0f64; 4]; args.h_dim];
    let mut q_row = 0usize;
    while q_row < args.num_lines {
        let bq = (args.num_lines - q_row).min(4);
        for (d, slot) in q_pack.iter_mut().enumerate().take(args.h_dim) {
            for (lane, out) in slot.iter_mut().enumerate().take(bq) {
                *out = args.q_proj[(q_row + lane) * args.qk_tot + args.qk_off + d] as f64;
            }
        }
        logits_4[..bq * args.num_lines].fill(0.0);
        let mut k_start = 0usize;
        while k_start < args.num_lines {
            let blen = (args.num_lines - k_start).min(BK_ANTI_ALIAS);
            for local_k in 0..blen {
                let kb = (k_start + local_k) * args.qk_tot + args.qk_off;
                for d in 0..args.h_dim { k_tile[d * STRIDE_ANTI_ALIAS + local_k] = args.k_proj[kb + d] as f64; }
            }
            dot_bq4(args, &q_pack, k_tile, TileRange { q_row, k_start, bq, blen }, logits_4);
            k_start += blen;
        }
        for lane in 0..bq {
            let row_l = &mut logits_4[lane * args.num_lines..(lane + 1) * args.num_lines];
            let max_l = row_l.iter().copied().fold(args.sink, f64::max);
            let exp_sum = (args.sink - max_l).exp() + softmax_row(row_l, max_l);
            accum_dense_v(args, q_row + lane, row_l, exp_sum, merged);
        }
        q_row += bq;
    }
}

fn dot_bq4_scalar(args: &DenseQkArgs<'_>, q_pack: &[[f64; 4]], k_tile: &[f64], rg: TileRange, logits_4: &mut [f64]) {
    for local_k in 0..rg.blen {
        let k_row = rg.k_start + local_k;
        let mut acc = [0.0f64; 4];
        for (d, q_slot) in q_pack.iter().enumerate().take(args.h_dim) {
            let kv = k_tile[d * STRIDE_ANTI_ALIAS + local_k];
            for lane in 0..rg.bq { acc[lane] += q_slot[lane] * kv; }
        }
        for (lane, &a) in acc.iter().enumerate().take(rg.bq) {
            logits_4[lane * args.num_lines + k_row] = rel_bias_for(args, rg.q_row + lane, k_row) + a * args.inv_s;
        }
    }
}

/// Pure-AVX2 non-FMA Estrin degree-12 `exp(x)` on 4x `f64` with `x < -708.0 => 0.0` underflow mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn exp_f64x4_avx2_non_fma(
    orig_x: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    let (min_x, max_x) = (_mm256_set1_pd(-708.0), _mm256_set1_pd(709.0));
    let underflow = _mm256_cmp_pd::<_CMP_LT_OQ>(orig_x, min_x);
    let x = _mm256_min_pd(_mm256_max_pd(orig_x, min_x), max_x);
    let (log2e, magic) = (_mm256_set1_pd(std::f64::consts::LOG2_E), _mm256_set1_pd(6_755_399_441_055_744.0));
    let ln2_hi = _mm256_set1_pd(f64::from_bits(0x3fe6_2e42_fee0_0000));
    let ln2_lo = _mm256_set1_pd(1.908_214_929_270_587_7e-10);
    let dn_shifted = _mm256_add_pd(_mm256_mul_pd(x, log2e), magic);
    let dn = _mm256_sub_pd(dn_shifted, magic);
    let r = _mm256_sub_pd(_mm256_sub_pd(x, _mm256_mul_pd(dn, ln2_hi)), _mm256_mul_pd(dn, ln2_lo));
    let r2 = _mm256_mul_pd(r, r);
    let r4 = _mm256_mul_pd(r2, r2);
    let r8 = _mm256_mul_pd(r4, r4);
    let c = [
        1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5_040.0, 1.0 / 40_320.0,
        1.0 / 362_880.0, 1.0 / 3_628_800.0, 1.0 / 39_916_800.0, 1.0 / 479_001_600.0,
    ];
    let v = |i: usize| _mm256_set1_pd(c[i]);
    let q03 = _mm256_add_pd(
        _mm256_add_pd(v(0), _mm256_mul_pd(r, v(1))),
        _mm256_mul_pd(r2, _mm256_add_pd(v(2), _mm256_mul_pd(r, v(3)))),
    );
    let q47 = _mm256_add_pd(
        _mm256_add_pd(v(4), _mm256_mul_pd(r, v(5))),
        _mm256_mul_pd(r2, _mm256_add_pd(v(6), _mm256_mul_pd(r, v(7)))),
    );
    let q811 = _mm256_add_pd(
        _mm256_add_pd(v(8), _mm256_mul_pd(r, v(9))),
        _mm256_mul_pd(r2, _mm256_add_pd(v(10), _mm256_mul_pd(r, v(11)))),
    );
    let q_all = _mm256_add_pd(_mm256_add_pd(q03, _mm256_mul_pd(r4, q47)), _mm256_mul_pd(r8, q811));
    let poly = _mm256_add_pd(v(0), _mm256_mul_pd(r, q_all));
    let bias_sub = _mm256_set1_epi64x(0x4338_0000_0000_0000_i64 - 1023_i64);
    let exp_bits = _mm256_slli_epi64(_mm256_sub_epi64(_mm256_castpd_si256(dn_shifted), bias_sub), 52);
    let scale = _mm256_castsi256_pd(exp_bits);
    _mm256_andnot_pd(underflow, _mm256_mul_pd(poly, scale))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2, `src_ptr` readable for `len`, and `d0_ptr`/`d1_ptr` disjoint and writable.
unsafe fn axpy_pair_avx2(src_ptr: *const f32, d0_ptr: *mut f32, d1_ptr: *mut f32, wv0: f32, wv1: f32, len: usize) {
    use std::arch::x86_64::*;
    let (w0, w1, mut i) = (_mm256_set1_ps(wv0), _mm256_set1_ps(wv1), 0usize);
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`; disjoint validated span slices.
        unsafe {
            let sv = _mm256_loadu_ps(src_ptr.add(i));
            let d0 = _mm256_loadu_ps(d0_ptr.add(i));
            let d1 = _mm256_loadu_ps(d1_ptr.add(i));
            _mm256_storeu_ps(d0_ptr.add(i), _mm256_add_ps(d0, _mm256_mul_ps(sv, w0)));
            _mm256_storeu_ps(d1_ptr.add(i), _mm256_add_ps(d1, _mm256_mul_ps(sv, w1)));
        }
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len`; pointers disjoint and in bounds.
        unsafe { let sv = *src_ptr.add(i); *d0_ptr.add(i) += sv * wv0; *d1_ptr.add(i) += sv * wv1; }
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2, `src_ptr` readable and `d0_ptr` writable for `len`.
unsafe fn axpy_single_avx2(src_ptr: *const f32, d0_ptr: *mut f32, wv0: f32, len: usize) {
    use std::arch::x86_64::*;
    let (w0, mut i) = (_mm256_set1_ps(wv0), 0usize);
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`; pointers stay inside validated buffers.
        unsafe {
            let (sv, d0) = (_mm256_loadu_ps(src_ptr.add(i)), _mm256_loadu_ps(d0_ptr.add(i)));
            _mm256_storeu_ps(d0_ptr.add(i), _mm256_add_ps(d0, _mm256_mul_ps(sv, w0)));
        }
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len`; pointers stay in bounds.
        unsafe { *d0_ptr.add(i) += *src_ptr.add(i) * wv0; }
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and validated `cells_soa`/`acc_soa` lengths.
unsafe fn accum_tap_soa_avx2(
    span: &RibbonSpan<'_>, tm: &CompactCsr, src_start: usize, col_lo: usize, col_hi: usize, acc_soa: &mut [f32],
) {
    let (len, cells_ptr, acc_ptr) = (col_hi - col_lo, span.cells_soa.as_ptr(), acc_soa.as_mut_ptr());
    for &ch_u16 in &tm.active_rows {
        let ch = ch_u16 as usize;
        if (span.active_mask[ch >> 6] & (1u64 << (ch & 63))) == 0 { continue; }
        // SAFETY: Channel offset stays inside validated span buffer.
        let src_ptr = unsafe { cells_ptr.add(ch * MAX_SPAN_LEN + src_start) };
        let (mut ptr, hi) = (tm.row_offsets[ch] as usize, tm.row_offsets[ch + 1] as usize);
        while ptr + 1 < hi {
            let (oc0, oc1) = (tm.col_indices[ptr] as usize, tm.col_indices[ptr + 1] as usize);
            // SAFETY: `oc0 < oc1 < out_dim`, destination spans disjoint and in bounds.
            unsafe {
                let d0 = acc_ptr.add(oc0 * MAX_SPAN_LEN + col_lo);
                let d1 = acc_ptr.add(oc1 * MAX_SPAN_LEN + col_lo);
                axpy_pair_avx2(src_ptr, d0, d1, tm.values[ptr], tm.values[ptr + 1], len);
            }
            ptr += 2;
        }
        if ptr < hi {
            let oc0 = tm.col_indices[ptr] as usize;
            // SAFETY: `oc0 < out_dim` and `col_lo + len <= MAX_SPAN_LEN`.
            unsafe { axpy_single_avx2(src_ptr, acc_ptr.add(oc0 * MAX_SPAN_LEN + col_lo), tm.values[ptr], len); }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and `cells_soa`/`acc_soa` stride `MAX_SPAN_LEN`.
unsafe fn ribbon_conv_span_avx2(
    span: &RibbonSpan<'_>, conv: &CompactConv, bias: &[f32],
    pad_taps: &[PadTap], acc_soa: &mut [f32], out: &mut [f32],
) {
    assert!(
        span.span_len <= MAX_SPAN_LEN
            && span.cells_soa.len() >= conv.rows * MAX_SPAN_LEN
            && acc_soa.len() >= conv.cols * MAX_SPAN_LEN
    );
    ribbon_conv_span_with(span, conv, bias, pad_taps, acc_soa, out, |sp, tm, ss, lo, hi, acc| {
        // SAFETY: AVX2 active and buffer sizes asserted above.
        unsafe { accum_tap_soa_avx2(sp, tm, ss, lo, hi, acc); }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and `tile_in`/`tile_acc` sized for `op.weight`.
unsafe fn run_csr_tile8_avx2(op: CsOp<'_>, tile: &mut TileScratch, out: &mut [f32], row: usize) {
    use std::arch::x86_64::*;
    let out_cols = op.weight.cols as usize;
    match op.bias {
        Some(b) => for (s, &bv) in tile.tile_acc.iter_mut().zip(b.iter()).take(out_cols) { *s = [bv; 8]; },
        None => for s in tile.tile_acc.iter_mut().take(out_cols) { *s = [0.0; 8]; },
    }
    for &in_u16 in &op.weight.active_rows {
        let in_idx = in_u16 as usize;
        let iv = tile.tile_in[in_idx];
        if iv == [0.0; 8] { continue; }
        // SAFETY: `iv` is a stack `[f32; 8]` array.
        let inv = unsafe { _mm256_loadu_ps(iv.as_ptr()) };
        let (lo, hi) = (op.weight.row_offsets[in_idx] as usize, op.weight.row_offsets[in_idx + 1] as usize);
        for ptr in lo..hi {
            let (col, wv) = (op.weight.col_indices[ptr] as usize, _mm256_set1_ps(op.weight.values[ptr]));
            // SAFETY: `col < out_cols <= tile.tile_acc.len()`.
            unsafe {
                let acc = _mm256_loadu_ps(tile.tile_acc[col].as_ptr());
                _mm256_storeu_ps(tile.tile_acc[col].as_mut_ptr(), _mm256_add_ps(acc, _mm256_mul_ps(inv, wv)));
            }
        }
    }
    if op.relu {
        let zero = _mm256_setzero_ps();
        for slot in tile.tile_acc.iter_mut().take(out_cols) {
            // SAFETY: `slot` is a mutable `[f32; 8]`.
            unsafe { _mm256_storeu_ps(slot.as_mut_ptr(), _mm256_max_ps(_mm256_loadu_ps(slot.as_ptr()), zero)); }
        }
    }
    for lane in 0..8 {
        let dst = (row + lane) * out_cols;
        for (col, slot) in tile.tile_acc.iter().enumerate().take(out_cols) { out[dst + col] = slot[lane]; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on the executing x86_64 CPU.
unsafe fn linear_sparse_width_gated_avx2(
    inp: &[f32], num_rows: usize, in_cols: usize, op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
) {
    let out_cols = op.weight.cols as usize;
    if out_cols > WIDE_OUT_GATE || num_rows < 8 {
        return linear_sparse_direct_csr(inp, num_rows, in_cols, op, out);
    }
    if tile.tile_in.len() < in_cols { tile.tile_in.resize(in_cols, [0.0; 8]); }
    if tile.tile_acc.len() < out_cols { tile.tile_acc.resize(out_cols, [0.0; 8]); }
    let mut row = 0usize;
    while row + 8 <= num_rows {
        for &in_u16 in &op.weight.active_rows {
            let col = in_u16 as usize;
            let mut lanes = [0.0f32; 8];
            for (lane, slot) in lanes.iter_mut().enumerate() { *slot = inp[(row + lane) * in_cols + col]; }
            tile.tile_in[col] = lanes;
        }
        // SAFETY: AVX2 active and tile buffers sized above.
        unsafe { run_csr_tile8_avx2(op, tile, out, row); }
        row += 8;
    }
    if row < num_rows {
        linear_sparse_direct_csr(
            &inp[row * in_cols..num_rows * in_cols], num_rows - row, in_cols, op,
            &mut out[row * out_cols..num_rows * out_cols],
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on the executing x86_64 CPU.
unsafe fn sparse_parts_width_gated_avx2(
    parts: &[&[f32]], ranges: &[PartRange], num_rows: usize, op: CsOp<'_>, out: &mut [f32], tile: &mut TileScratch,
) {
    let (out_cols, in_cols) = (op.weight.cols as usize, op.weight.rows as usize);
    if out_cols > WIDE_OUT_GATE || num_rows < 8 {
        return sparse_parts_direct_csr(parts, ranges, num_rows, op, out);
    }
    if tile.tile_in.len() < in_cols { tile.tile_in.resize(in_cols, [0.0; 8]); }
    if tile.tile_acc.len() < out_cols { tile.tile_acc.resize(out_cols, [0.0; 8]); }
    let mut row = 0usize;
    while row + 8 <= num_rows {
        for rg in ranges {
            let src = parts[rg.part_idx];
            for &in_u16 in &op.weight.active_rows[rg.start..rg.end] {
                let (in_idx, local) = (in_u16 as usize, in_u16 as usize - rg.in_off);
                let mut lanes = [0.0f32; 8];
                for (lane, slot) in lanes.iter_mut().enumerate() { *slot = src[(row + lane) * rg.dim + local]; }
                tile.tile_in[in_idx] = lanes;
            }
        }
        // SAFETY: AVX2 active and tile buffers sized above.
        unsafe { run_csr_tile8_avx2(op, tile, out, row); }
        row += 8;
    }
    if row < num_rows {
        let mut tail_parts: [&[f32]; 5] = [&[]; 5];
        for rg in ranges { tail_parts[rg.part_idx] = &parts[rg.part_idx][row * rg.dim..num_rows * rg.dim]; }
        sparse_parts_direct_csr(
            &tail_parts[..parts.len()], ranges, num_rows - row, op, &mut out[row * out_cols..num_rows * out_cols],
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on the executing x86_64 CPU.
unsafe fn softmax_exp_slice_avx2(row_l: &mut [f64], max_l: f64) -> f64 {
    use std::arch::x86_64::*;
    let (vmax, mut vsum, mut i, len) = (_mm256_set1_pd(max_l), _mm256_setzero_pd(), 0usize, row_l.len());
    while i + 4 <= len {
        // SAFETY: `i + 4 <= len`; pointers stay inside `row_l` and AVX2 is active.
        unsafe {
            let exp_v = exp_f64x4_avx2_non_fma(_mm256_sub_pd(_mm256_loadu_pd(row_l.as_ptr().add(i)), vmax));
            _mm256_storeu_pd(row_l.as_mut_ptr().add(i), exp_v);
            vsum = _mm256_add_pd(vsum, exp_v);
        }
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` is a stack `[f64; 4]` array.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), vsum); }
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for lv in &mut row_l[i..] { *lv = (*lv - max_l).exp(); sum += *lv; }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and valid slice lengths.
unsafe fn dot_f32_as_f64_avx2(q_slice: &[f32], k_slice: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (len, mut acc, mut i) = (q_slice.len().min(k_slice.len()), _mm256_setzero_pd(), 0usize);
    while i + 4 <= len {
        // SAFETY: `i + 4 <= len`; both slices have `>= len` elements.
        let (qf, kf) = unsafe { (_mm_loadu_ps(q_slice.as_ptr().add(i)), _mm_loadu_ps(k_slice.as_ptr().add(i))) };
        acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_cvtps_pd(qf), _mm256_cvtps_pd(kf)));
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` is a stack `[f64; 4]` array.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc); }
    let mut dot = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < len { dot += (q_slice[i] as f64) * (k_slice[i] as f64); i += 1; }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and valid `q_pack`/`k_tile`/`logits_4` dimensions.
unsafe fn dot_bq4_avx2(
    args: &DenseQkArgs<'_>, q_pack: &[[f64; 4]], k_tile: &[f64],
    rg: TileRange, logits_4: &mut [f64],
) {
    use std::arch::x86_64::*;
    let (scale, mut tmp) = (_mm256_set1_pd(args.inv_s), [0.0f64; 4]);
    for local_k in 0..rg.blen {
        let (k_row, mut acc) = (rg.k_start + local_k, _mm256_setzero_pd());
        for (d, q_slot) in q_pack.iter().enumerate().take(args.h_dim) {
            // SAFETY: `q_slot` is a `[f64; 4]` element.
            let qv = unsafe { _mm256_loadu_pd(q_slot.as_ptr()) };
            let kv = _mm256_set1_pd(k_tile[d * STRIDE_ANTI_ALIAS + local_k]);
            acc = _mm256_add_pd(acc, _mm256_mul_pd(qv, kv));
        }
        // SAFETY: `tmp` is a stack `[f64; 4]` array.
        unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), _mm256_mul_pd(acc, scale)); }
        for (lane, &tv) in tmp.iter().enumerate().take(rg.bq) {
            logits_4[lane * args.num_lines + k_row] = rel_bias_for(args, rg.q_row + lane, k_row) + tv;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on the executing x86_64 CPU.
unsafe fn dense_qk_head_avx2(
    args: &DenseQkArgs<'_>, k_tile: &mut Vec<f64>, logits_4: &mut Vec<f64>, merged: &mut [f32],
) {
    dense_qk_head_impl(
        args, k_tile, logits_4, merged,
        |a, qp, kt, rg, l4| {
            // SAFETY: AVX2 active in this function.
            unsafe { dot_bq4_avx2(a, qp, kt, rg, l4); }
        },
        |row_l, max_l| {
            // SAFETY: AVX2 active in this function.
            unsafe { softmax_exp_slice_avx2(row_l, max_l) }
        },
    );
}
