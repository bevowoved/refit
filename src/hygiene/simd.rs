use super::weights::{CompactConv, CompactCsr, PadTap, PartRange};

/// Empirical L1D width gate: vertical 8-line AVX2 tiling wins when `out_cols <= 128`;
/// wider outputs (`attn1_wq [209x400]`, `ml_gate_w [8x514]`) use direct compact CSR.
pub(super) const WIDE_OUT_GATE: usize = 128;
pub(super) const MAX_SPAN_LEN: usize = 256;
pub(super) const BK_ANTI_ALIAS: usize = 252;
pub(super) const STRIDE_ANTI_ALIAS: usize = 264;

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

/// Pure scalar direct compact-CSR matrix multiply (`out_cols` arbitrary, zero transpose).
#[inline]
pub(super) fn linear_sparse_direct_csr(
    inp: &[f32],
    num_rows: usize,
    in_cols: usize,
    weight: &CompactCsr,
    bias: Option<&[f32]>,
    apply_relu: bool,
    out: &mut [f32],
) {
    let out_cols = weight.cols as usize;
    for row in 0..num_rows {
        let dst = &mut out[row * out_cols..(row + 1) * out_cols];
        init_row_bias(dst, bias);
        let base = row * in_cols;
        for &in_u16 in &weight.active_rows {
            let in_idx = in_u16 as usize;
            let iv = inp[base + in_idx];
            if iv == 0.0 {
                continue;
            }
            let lo = weight.row_offsets[in_idx] as usize;
            let hi = weight.row_offsets[in_idx + 1] as usize;
            for ptr in lo..hi {
                dst[weight.col_indices[ptr] as usize] += iv * weight.values[ptr];
            }
        }
        if apply_relu {
            relu_slice_in_place(dst);
        }
    }
}

/// Pure scalar multi-part compact-CSR matrix multiply using precomputed `PartRange` slices.
#[inline]
pub(super) fn sparse_parts_direct_csr(
    parts: &[&[f32]],
    ranges: &[PartRange],
    num_rows: usize,
    weight: &CompactCsr,
    bias: Option<&[f32]>,
    apply_relu: bool,
    out: &mut [f32],
) {
    let out_cols = weight.cols as usize;
    for row in 0..num_rows {
        let dst = &mut out[row * out_cols..(row + 1) * out_cols];
        init_row_bias(dst, bias);
        for rg in ranges {
            accum_part_row(parts[rg.part_idx], rg, row, weight, dst);
        }
        if apply_relu {
            relu_slice_in_place(dst);
        }
    }
}

#[inline]
fn init_row_bias(dst: &mut [f32], bias: Option<&[f32]>) {
    match bias {
        Some(b) => dst.copy_from_slice(b),
        None => dst.fill(0.0),
    }
}

#[inline]
fn relu_slice_in_place(dst: &mut [f32]) {
    for val in dst.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}

#[inline]
fn accum_part_row(src: &[f32], rg: &PartRange, row: usize, weight: &CompactCsr, dst: &mut [f32]) {
    let row_base = row * rg.dim;
    for &in_u16 in &weight.active_rows[rg.start..rg.end] {
        let in_idx = in_u16 as usize;
        let iv = src[row_base + (in_idx - rg.in_off)];
        if iv == 0.0 {
            continue;
        }
        let lo = weight.row_offsets[in_idx] as usize;
        let hi = weight.row_offsets[in_idx + 1] as usize;
        for ptr in lo..hi {
            dst[weight.col_indices[ptr] as usize] += iv * weight.values[ptr];
        }
    }
}

/// Scalar channel-major SoA ribbon convolution over a single span (`<= 256` cells).
pub(super) fn ribbon_conv_span_scalar(
    cells_soa: &[f32],
    active_mask: [u64; 2],
    span_len: usize,
    conv: &CompactConv,
    bias: &[f32],
    pad_taps: &[PadTap],
    acc_soa: &mut [f32],
    split_at: Option<(usize, usize, usize)>,
    single_row: usize,
    out: &mut [f32],
) {
    if conv.non_zero_count == 0 {
        return;
    }
    let out_dim = conv.cols;
    let pad = (conv.k_size - 1) >> 1;
    init_acc_soa(acc_soa, out_dim, span_len, bias);

    for tap in 0..conv.k_size {
        let tm = &conv.taps[tap];
        if tm.non_zero_count == 0 {
            continue;
        }
        let shift = tap as isize - pad as isize;
        let col_lo = (-shift).max(0) as usize;
        let col_hi = (span_len as isize - shift).clamp(0, span_len as isize) as usize;

        apply_pad_cols(acc_soa, &pad_taps[tap], 0, col_lo.min(span_len));
        apply_pad_cols(acc_soa, &pad_taps[tap], col_hi, span_len);
        if col_lo < col_hi {
            let src_start = (col_lo as isize + shift) as usize;
            accum_tap_soa_scalar(cells_soa, active_mask, tm, src_start, col_lo, col_hi, acc_soa);
        }
    }
    reduce_span_relu(acc_soa, out_dim, span_len, split_at, single_row, out);
}

#[inline]
fn init_acc_soa(acc_soa: &mut [f32], out_dim: usize, span_len: usize, bias: &[f32]) {
    for (oc, &bv) in bias.iter().enumerate().take(out_dim) {
        acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span_len].fill(bv);
    }
}

fn accum_tap_soa_scalar(
    cells_soa: &[f32],
    active_mask: [u64; 2],
    tm: &CompactCsr,
    src_start: usize,
    col_lo: usize,
    col_hi: usize,
    acc_soa: &mut [f32],
) {
    let len = col_hi - col_lo;
    for &ch_u16 in &tm.active_rows {
        let ch = ch_u16 as usize;
        if (active_mask[ch >> 6] & (1u64 << (ch & 63))) == 0 {
            continue;
        }
        let src = &cells_soa[ch * MAX_SPAN_LEN + src_start..ch * MAX_SPAN_LEN + src_start + len];
        let lo = tm.row_offsets[ch] as usize;
        let hi = tm.row_offsets[ch + 1] as usize;
        for ptr in lo..hi {
            let oc = tm.col_indices[ptr] as usize;
            let wv = tm.values[ptr];
            let dst = &mut acc_soa[oc * MAX_SPAN_LEN + col_lo..oc * MAX_SPAN_LEN + col_hi];
            for idx in 0..len {
                dst[idx] += src[idx] * wv;
            }
        }
    }
}

#[inline]
pub(super) fn apply_pad_cols(acc_soa: &mut [f32], pad: &PadTap, lo: usize, hi: usize) {
    for col in lo..hi {
        for (&oc, &pv) in pad.cols.iter().zip(pad.vals.iter()) {
            acc_soa[oc as usize * MAX_SPAN_LEN + col] += pv;
        }
    }
}

#[inline]
pub(super) fn reduce_span_relu(
    acc_soa: &[f32],
    out_dim: usize,
    span_len: usize,
    split_at: Option<(usize, usize, usize)>,
    single_row: usize,
    out: &mut [f32],
) {
    if let Some((len0, row0, row1)) = split_at {
        let (d0, d1) = (row0 * out_dim, row1 * out_dim);
        for oc in 0..out_dim {
            let row_acc = &acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span_len];
            out[d0 + oc] = sum_positive(&row_acc[..len0]);
            out[d1 + oc] = sum_positive(&row_acc[len0..span_len]);
        }
    } else {
        let d0 = single_row * out_dim;
        for oc in 0..out_dim {
            let row_acc = &acc_soa[oc * MAX_SPAN_LEN..oc * MAX_SPAN_LEN + span_len];
            out[d0 + oc] = sum_positive(row_acc);
        }
    }
}

#[inline]
fn sum_positive(slice: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &val in slice {
        if val > 0.0 {
            sum += val;
        }
    }
    sum
}

fn pack_q_block(args: &DenseQkArgs<'_>, q_row: usize, bq: usize, q_pack: &mut [[f64; 4]]) {
    for (d, slot) in q_pack.iter_mut().enumerate().take(args.h_dim) {
        for lane in 0..bq {
            slot[lane] = args.q_proj[(q_row + lane) * args.qk_tot + args.qk_off + d] as f64;
        }
    }
}

fn pack_k_block(args: &DenseQkArgs<'_>, k_start: usize, blen: usize, k_tile: &mut [f64]) {
    for local_k in 0..blen {
        let kb = (k_start + local_k) * args.qk_tot + args.qk_off;
        for d in 0..args.h_dim {
            k_tile[d * STRIDE_ANTI_ALIAS + local_k] = args.k_proj[kb + d] as f64;
        }
    }
}

#[inline]
fn rel_bias_for(args: &DenseQkArgs<'_>, q_idx: usize, k_idx: usize) -> f64 {
    if args.all_zero_rel {
        return 0.0;
    }
    let dist = (k_idx as isize - q_idx as isize).clamp(-32, 32);
    args.rel[(dist + 32) as usize]
}

fn accum_dense_v(args: &DenseQkArgs<'_>, q_idx: usize, weights: &[f64], exp_sum: f64, merged: &mut [f32]) {
    let dst = q_idx * args.v_tot + args.v_off;
    for (k_row, &ew) in weights.iter().enumerate().take(args.num_lines) {
        let aw = ew / exp_sum;
        let vb = k_row * args.v_tot + args.v_off;
        for d in 0..args.v_dim {
            merged[dst + d] += (aw * (args.v_proj[vb + d] as f64)) as f32;
        }
    }
}

fn dot_bq4_scalar(
    args: &DenseQkArgs<'_>,
    q_pack: &[[f64; 4]],
    k_tile: &[f64],
    q_row: usize,
    k_start: usize,
    bq: usize,
    blen: usize,
    logits_4: &mut [f64],
) {
    for local_k in 0..blen {
        let k_row = k_start + local_k;
        let mut acc = [0.0f64; 4];
        for (d, q_slot) in q_pack.iter().enumerate().take(args.h_dim) {
            let kv = k_tile[d * STRIDE_ANTI_ALIAS + local_k];
            for lane in 0..bq {
                acc[lane] += q_slot[lane] * kv;
            }
        }
        for lane in 0..bq {
            let rb = rel_bias_for(args, q_row + lane, k_row);
            logits_4[lane * args.num_lines + k_row] = rb + acc[lane] * args.inv_s;
        }
    }
}

/// Scalar `BQ=4 x BK=252` stride-264 dense QK attention kernel (live non-AVX2 fallback).
pub(super) fn dense_qk_head_scalar(
    args: &DenseQkArgs<'_>,
    k_tile: &mut Vec<f64>,
    logits_4: &mut Vec<f64>,
    merged: &mut [f32],
) {
    k_tile.resize(args.h_dim * STRIDE_ANTI_ALIAS, 0.0);
    logits_4.resize(4 * args.num_lines, 0.0);
    let mut q_pack = vec![[0.0f64; 4]; args.h_dim];

    let mut q_row = 0usize;
    while q_row < args.num_lines {
        let bq = (args.num_lines - q_row).min(4);
        pack_q_block(args, q_row, bq, &mut q_pack);
        logits_4[..bq * args.num_lines].fill(0.0);

        let mut k_start = 0usize;
        while k_start < args.num_lines {
            let blen = (args.num_lines - k_start).min(BK_ANTI_ALIAS);
            pack_k_block(args, k_start, blen, k_tile);
            dot_bq4_scalar(args, &q_pack, k_tile, q_row, k_start, bq, blen, logits_4);
            k_start += blen;
        }

        for lane in 0..bq {
            let row_l = &mut logits_4[lane * args.num_lines..(lane + 1) * args.num_lines];
            let max_l = row_l.iter().copied().fold(args.sink, f64::max);
            let mut exp_sum = (args.sink - max_l).exp();
            for lv in row_l.iter_mut() {
                *lv = (*lv - max_l).exp();
                exp_sum += *lv;
            }
            accum_dense_v(args, q_row + lane, row_l, exp_sum, merged);
        }
        q_row += bq;
    }
}

/// Pure-AVX2 non-FMA Estrin degree-12 `exp(x)` on 4x `f64` with `x < -708.0 => 0.0`
/// underflow mask (zero AVX-512DQ, zero FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn exp_f64x4_avx2_non_fma(
    orig_x: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    let min_x = _mm256_set1_pd(-708.0);
    let max_x = _mm256_set1_pd(709.0);
    let underflow_mask = _mm256_cmp_pd::<_CMP_LT_OQ>(orig_x, min_x);
    let x = _mm256_min_pd(_mm256_max_pd(orig_x, min_x), max_x);

    let log2e = _mm256_set1_pd(1.442_695_040_888_963_4);
    let ln2_hi = _mm256_set1_pd(f64::from_bits(0x3fe6_2e42_fee0_0000));
    let ln2_lo = _mm256_set1_pd(1.908_214_929_270_587_7e-10);
    let magic = _mm256_set1_pd(6_755_399_441_055_744.0);

    let z = _mm256_mul_pd(x, log2e);
    let dn_shifted = _mm256_add_pd(z, magic);
    let dn = _mm256_sub_pd(dn_shifted, magic);

    let r1 = _mm256_sub_pd(x, _mm256_mul_pd(dn, ln2_hi));
    let r = _mm256_sub_pd(r1, _mm256_mul_pd(dn, ln2_lo));

    let r2 = _mm256_mul_pd(r, r);
    let r4 = _mm256_mul_pd(r2, r2);
    let r8 = _mm256_mul_pd(r4, r4);

    let c1 = _mm256_set1_pd(1.0);
    let c2 = _mm256_set1_pd(0.5);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let c5 = _mm256_set1_pd(1.0 / 120.0);
    let c6 = _mm256_set1_pd(1.0 / 720.0);
    let c7 = _mm256_set1_pd(1.0 / 5_040.0);
    let c8 = _mm256_set1_pd(1.0 / 40_320.0);
    let c9 = _mm256_set1_pd(1.0 / 362_880.0);
    let c10 = _mm256_set1_pd(1.0 / 3_628_800.0);
    let c11 = _mm256_set1_pd(1.0 / 39_916_800.0);
    let c12 = _mm256_set1_pd(1.0 / 479_001_600.0);

    let p01 = _mm256_add_pd(c1, _mm256_mul_pd(r, c2));
    let p23 = _mm256_add_pd(c3, _mm256_mul_pd(r, c4));
    let q03 = _mm256_add_pd(p01, _mm256_mul_pd(r2, p23));

    let p45 = _mm256_add_pd(c5, _mm256_mul_pd(r, c6));
    let p67 = _mm256_add_pd(c7, _mm256_mul_pd(r, c8));
    let q47 = _mm256_add_pd(p45, _mm256_mul_pd(r2, p67));

    let p89 = _mm256_add_pd(c9, _mm256_mul_pd(r, c10));
    let p1011 = _mm256_add_pd(c11, _mm256_mul_pd(r, c12));
    let q811 = _mm256_add_pd(p89, _mm256_mul_pd(r2, p1011));

    let q_lo = _mm256_add_pd(q03, _mm256_mul_pd(r4, q47));
    let q_all = _mm256_add_pd(q_lo, _mm256_mul_pd(r8, q811));
    let poly = _mm256_add_pd(c1, _mm256_mul_pd(r, q_all));

    let bias_sub = _mm256_set1_epi64x(0x4338_0000_0000_0000_i64 - 1023_i64);
    let exp_bits =
        _mm256_slli_epi64(_mm256_sub_epi64(_mm256_castpd_si256(dn_shifted), bias_sub), 52);
    let scale = _mm256_castsi256_pd(exp_bits);
    let raw = _mm256_mul_pd(poly, scale);
    _mm256_andnot_pd(underflow_mask, raw)
}

/// 2-way paired non-FMA AVX2 `axpy`: loads `src` once for two disjoint output channels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2, `src_ptr` readable for `len`, and `d0_ptr`/`d1_ptr` disjoint and writable.
unsafe fn axpy_pair_avx2(
    src_ptr: *const f32,
    d0_ptr: *mut f32,
    d1_ptr: *mut f32,
    wv0: f32,
    wv1: f32,
    len: usize,
) {
    use std::arch::x86_64::*;

    let w0 = _mm256_set1_ps(wv0);
    let w1 = _mm256_set1_ps(wv1);
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`; all three pointers stay inside validated span slices.
        let (sv, d0, d1) = unsafe {
            (
                _mm256_loadu_ps(src_ptr.add(i)),
                _mm256_loadu_ps(d0_ptr.add(i)),
                _mm256_loadu_ps(d1_ptr.add(i)),
            )
        };
        let r0 = _mm256_add_ps(d0, _mm256_mul_ps(sv, w0));
        let r1 = _mm256_add_ps(d1, _mm256_mul_ps(sv, w1));
        // SAFETY: `d0_ptr` and `d1_ptr` are disjoint and `i + 8 <= len`.
        unsafe {
            _mm256_storeu_ps(d0_ptr.add(i), r0);
            _mm256_storeu_ps(d1_ptr.add(i), r1);
        }
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len`, pointers are disjoint and in bounds.
        unsafe {
            let sv = *src_ptr.add(i);
            *d0_ptr.add(i) += sv * wv0;
            *d1_ptr.add(i) += sv * wv1;
        }
        i += 1;
    }
}

/// Single-channel non-FMA AVX2 `axpy` tail for odd CSR row degree.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2, `src_ptr` readable for `len`, `d0_ptr` writable for `len`.
unsafe fn axpy_single_avx2(src_ptr: *const f32, d0_ptr: *mut f32, wv0: f32, len: usize) {
    use std::arch::x86_64::*;

    let w0 = _mm256_set1_ps(wv0);
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`, both loads stay within validated buffers.
        let (sv, d0) =
            unsafe { (_mm256_loadu_ps(src_ptr.add(i)), _mm256_loadu_ps(d0_ptr.add(i))) };
        let r0 = _mm256_add_ps(d0, _mm256_mul_ps(sv, w0));
        // SAFETY: `i + 8 <= len`, `d0_ptr` stays within `acc_soa`.
        unsafe {
            _mm256_storeu_ps(d0_ptr.add(i), r0);
        }
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len`, both pointers stay in bounds.
        unsafe {
            *d0_ptr.add(i) += *src_ptr.add(i) * wv0;
        }
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and validated `cells_soa`/`acc_soa` lengths.
unsafe fn accum_tap_soa_avx2(
    cells_soa: &[f32],
    active_mask: [u64; 2],
    tm: &CompactCsr,
    src_start: usize,
    col_lo: usize,
    col_hi: usize,
    acc_soa: &mut [f32],
) {
    let len = col_hi - col_lo;
    let cells_ptr = cells_soa.as_ptr();
    let acc_ptr = acc_soa.as_mut_ptr();

    for &ch_u16 in &tm.active_rows {
        let ch = ch_u16 as usize;
        if (active_mask[ch >> 6] & (1u64 << (ch & 63))) == 0 {
            continue;
        }
        // SAFETY: channel offset stays inside the validated span buffer.
        let src_ptr = unsafe { cells_ptr.add(ch * MAX_SPAN_LEN + src_start) };
        let lo = tm.row_offsets[ch] as usize;
        let hi = tm.row_offsets[ch + 1] as usize;
        let mut ptr = lo;

        while ptr + 1 < hi {
            let oc0 = tm.col_indices[ptr] as usize;
            let oc1 = tm.col_indices[ptr + 1] as usize;
            // SAFETY: `oc0 < oc1 < out_dim`, so destination spans are disjoint and in bounds.
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
            unsafe {
                let d0 = acc_ptr.add(oc0 * MAX_SPAN_LEN + col_lo);
                axpy_single_avx2(src_ptr, d0, tm.values[ptr], len);
            }
        }
    }
}

/// 2-way paired non-FMA AVX2 `axpy` over channel-major SoA ribbon span (`<= 256` cells).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available and `cells_soa`/`acc_soa` have stride `MAX_SPAN_LEN`.
pub(super) unsafe fn ribbon_conv_span_avx2(
    cells_soa: &[f32],
    active_mask: [u64; 2],
    span_len: usize,
    conv: &CompactConv,
    bias: &[f32],
    pad_taps: &[PadTap],
    acc_soa: &mut [f32],
    split_at: Option<(usize, usize, usize)>,
    single_row: usize,
    out: &mut [f32],
) {
    if conv.non_zero_count == 0 {
        return;
    }
    let out_dim = conv.cols;
    let pad = (conv.k_size - 1) >> 1;
    assert!(span_len <= MAX_SPAN_LEN);
    assert!(cells_soa.len() >= conv.rows * MAX_SPAN_LEN);
    assert!(acc_soa.len() >= out_dim * MAX_SPAN_LEN);
    init_acc_soa(acc_soa, out_dim, span_len, bias);

    for tap in 0..conv.k_size {
        let tm = &conv.taps[tap];
        if tm.non_zero_count == 0 {
            continue;
        }
        let shift = tap as isize - pad as isize;
        let col_lo = (-shift).max(0) as usize;
        let col_hi = (span_len as isize - shift).clamp(0, span_len as isize) as usize;

        apply_pad_cols(acc_soa, &pad_taps[tap], 0, col_lo.min(span_len));
        apply_pad_cols(acc_soa, &pad_taps[tap], col_hi, span_len);
        if col_lo < col_hi {
            let src_start = (col_lo as isize + shift) as usize;
            // SAFETY: AVX2 active and buffer sizes asserted above.
            unsafe {
                accum_tap_soa_avx2(cells_soa, active_mask, tm, src_start, col_lo, col_hi, acc_soa);
            }
        }
    }

    reduce_span_relu(acc_soa, out_dim, span_len, split_at, single_row, out);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn init_tile_acc(tile_acc: &mut [[f32; 8]], out_cols: usize, bias: Option<&[f32]>) {
    match bias {
        Some(b) => {
            for col in 0..out_cols {
                tile_acc[col] = [b[col]; 8];
            }
        }
        None => {
            for col in 0..out_cols {
                tile_acc[col] = [0.0; 8];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and `tile_in`/`tile_acc` sized for `weight`.
unsafe fn run_csr_tile8_avx2(
    weight: &CompactCsr,
    apply_relu: bool,
    tile_in: &[[f32; 8]],
    tile_acc: &mut [[f32; 8]],
    out: &mut [f32],
    row: usize,
) {
    use std::arch::x86_64::*;

    let out_cols = weight.cols as usize;
    for &in_u16 in &weight.active_rows {
        let in_idx = in_u16 as usize;
        let iv = tile_in[in_idx];
        if iv == [0.0; 8] {
            continue;
        }
        // SAFETY: `iv` is a valid 8-element `[f32; 8]` array on the stack.
        let inv = unsafe { _mm256_loadu_ps(iv.as_ptr()) };
        let lo = weight.row_offsets[in_idx] as usize;
        let hi = weight.row_offsets[in_idx + 1] as usize;
        for ptr in lo..hi {
            let col = weight.col_indices[ptr] as usize;
            let wv = _mm256_set1_ps(weight.values[ptr]);
            // SAFETY: `col < out_cols <= tile_acc.len()`, valid `[f32; 8]`.
            let acc = unsafe { _mm256_loadu_ps(tile_acc[col].as_ptr()) };
            let res = _mm256_add_ps(acc, _mm256_mul_ps(inv, wv));
            // SAFETY: `col < out_cols <= tile_acc.len()`, valid mutable `[f32; 8]`.
            unsafe {
                _mm256_storeu_ps(tile_acc[col].as_mut_ptr(), res);
            }
        }
    }

    if apply_relu {
        let zero = _mm256_setzero_ps();
        for slot in tile_acc.iter_mut().take(out_cols) {
            // SAFETY: `slot` is a valid mutable `[f32; 8]`.
            let acc = unsafe { _mm256_loadu_ps(slot.as_ptr()) };
            let clamped = _mm256_max_ps(acc, zero);
            // SAFETY: `slot` is a valid mutable `[f32; 8]`.
            unsafe {
                _mm256_storeu_ps(slot.as_mut_ptr(), clamped);
            }
        }
    }

    for lane in 0..8 {
        let dst = (row + lane) * out_cols;
        for col in 0..out_cols {
            out[dst + col] = tile_acc[col][lane];
        }
    }
}

/// Width-gated 8-line vertical non-FMA AVX2 `linear_sparse` (`out_cols <= 128` -> AVX2 tile,
/// `out_cols > 128` -> direct compact CSR with 0 regression on wide projections).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn linear_sparse_width_gated_avx2(
    inp: &[f32],
    num_rows: usize,
    in_cols: usize,
    weight: &CompactCsr,
    bias: Option<&[f32]>,
    apply_relu: bool,
    out: &mut [f32],
    tile_in: &mut Vec<[f32; 8]>,
    tile_acc: &mut Vec<[f32; 8]>,
) {
    let out_cols = weight.cols as usize;
    if out_cols > WIDE_OUT_GATE || num_rows < 8 {
        linear_sparse_direct_csr(inp, num_rows, in_cols, weight, bias, apply_relu, out);
        return;
    }

    if tile_in.len() < in_cols {
        tile_in.resize(in_cols, [0.0; 8]);
    }
    if tile_acc.len() < out_cols {
        tile_acc.resize(out_cols, [0.0; 8]);
    }

    let mut row = 0usize;
    while row + 8 <= num_rows {
        for &in_u16 in &weight.active_rows {
            let col = in_u16 as usize;
            let mut lanes = [0.0f32; 8];
            for (lane, slot) in lanes.iter_mut().enumerate() {
                *slot = inp[(row + lane) * in_cols + col];
            }
            tile_in[col] = lanes;
        }
        init_tile_acc(tile_acc, out_cols, bias);
        // SAFETY: AVX2 active and tile buffers resized above.
        unsafe {
            run_csr_tile8_avx2(weight, apply_relu, tile_in, tile_acc, out, row);
        }
        row += 8;
    }

    if row < num_rows {
        linear_sparse_direct_csr(
            &inp[row * in_cols..num_rows * in_cols],
            num_rows - row,
            in_cols,
            weight,
            bias,
            apply_relu,
            &mut out[row * out_cols..num_rows * out_cols],
        );
    }
}

/// Width-gated 8-line vertical non-FMA AVX2 multi-part `sparse_parts`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn sparse_parts_width_gated_avx2(
    parts: &[&[f32]],
    ranges: &[PartRange],
    num_rows: usize,
    weight: &CompactCsr,
    bias: Option<&[f32]>,
    apply_relu: bool,
    out: &mut [f32],
    tile_in: &mut Vec<[f32; 8]>,
    tile_acc: &mut Vec<[f32; 8]>,
) {
    let out_cols = weight.cols as usize;
    let in_cols = weight.rows as usize;
    if out_cols > WIDE_OUT_GATE || num_rows < 8 {
        sparse_parts_direct_csr(parts, ranges, num_rows, weight, bias, apply_relu, out);
        return;
    }

    if tile_in.len() < in_cols {
        tile_in.resize(in_cols, [0.0; 8]);
    }
    if tile_acc.len() < out_cols {
        tile_acc.resize(out_cols, [0.0; 8]);
    }

    let mut row = 0usize;
    while row + 8 <= num_rows {
        for rg in ranges {
            let src = parts[rg.part_idx];
            for &in_u16 in &weight.active_rows[rg.start..rg.end] {
                let in_idx = in_u16 as usize;
                let local = in_idx - rg.in_off;
                let mut lanes = [0.0f32; 8];
                for (lane, slot) in lanes.iter_mut().enumerate() {
                    *slot = src[(row + lane) * rg.dim + local];
                }
                tile_in[in_idx] = lanes;
            }
        }
        init_tile_acc(tile_acc, out_cols, bias);
        // SAFETY: AVX2 active and tile buffers resized above.
        unsafe {
            run_csr_tile8_avx2(weight, apply_relu, tile_in, tile_acc, out, row);
        }
        row += 8;
    }

    if row < num_rows {
        let mut tail_parts: [&[f32]; 5] = [&[]; 5];
        for rg in ranges {
            tail_parts[rg.part_idx] = &parts[rg.part_idx][row * rg.dim..num_rows * rg.dim];
        }
        sparse_parts_direct_csr(
            &tail_parts[..parts.len()],
            ranges,
            num_rows - row,
            weight,
            bias,
            apply_relu,
            &mut out[row * out_cols..num_rows * out_cols],
        );
    }
}

/// Vectorized AVX2 non-FMA softmax exp pass over `row_l` using Estrin degree-12 `exp_f64x4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn softmax_exp_slice_avx2(row_l: &mut [f64], max_l: f64) -> f64 {
    use std::arch::x86_64::*;

    let vmax = _mm256_set1_pd(max_l);
    let mut vsum = _mm256_setzero_pd();
    let mut i = 0usize;
    let len = row_l.len();

    while i + 4 <= len {
        // SAFETY: `i + 4 <= len`, pointer stays inside `row_l`.
        let raw = unsafe { _mm256_loadu_pd(row_l.as_ptr().add(i)) };
        let diff = _mm256_sub_pd(raw, vmax);
        // SAFETY: AVX2 is active in this function.
        let exp_v = unsafe { exp_f64x4_avx2_non_fma(diff) };
        // SAFETY: `i + 4 <= len`, mutable pointer stays inside `row_l`.
        unsafe {
            _mm256_storeu_pd(row_l.as_mut_ptr().add(i), exp_v);
        }
        vsum = _mm256_add_pd(vsum, exp_v);
        i += 4;
    }

    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` is a 4-element `[f64; 4]` array on the stack.
    unsafe {
        _mm256_storeu_pd(lanes.as_mut_ptr(), vsum);
    }
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < len {
        let ev = (row_l[i] - max_l).exp();
        row_l[i] = ev;
        sum += ev;
        i += 1;
    }
    sum
}

/// AVX2 non-FMA `f64` dot product for QK heads (`f32 -> f64` widening, zero FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available and slices have length `>= h_dim`.
pub(super) unsafe fn dot_f32_as_f64_avx2(q_slice: &[f32], k_slice: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = q_slice.len().min(k_slice.len());
    let mut acc = _mm256_setzero_pd();
    let mut i = 0usize;

    while i + 4 <= len {
        // SAFETY: `i + 4 <= len`, both slices have at least `len` elements.
        let (qf, kf) = unsafe {
            (
                _mm_loadu_ps(q_slice.as_ptr().add(i)),
                _mm_loadu_ps(k_slice.as_ptr().add(i)),
            )
        };
        let qd = _mm256_cvtps_pd(qf);
        let kd = _mm256_cvtps_pd(kf);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(qd, kd));
        i += 4;
    }

    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` is a valid 4-element `[f64; 4]` stack array.
    unsafe {
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    }
    let mut dot = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < len {
        dot += (q_slice[i] as f64) * (k_slice[i] as f64);
        i += 1;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 and valid `q_pack`/`k_tile`/`logits_4` dimensions.
unsafe fn dot_bq4_avx2(
    args: &DenseQkArgs<'_>,
    q_pack: &[[f64; 4]],
    k_tile: &[f64],
    q_row: usize,
    k_start: usize,
    bq: usize,
    blen: usize,
    logits_4: &mut [f64],
) {
    use std::arch::x86_64::*;

    let scale = _mm256_set1_pd(args.inv_s);
    let mut tmp = [0.0f64; 4];
    for local_k in 0..blen {
        let k_row = k_start + local_k;
        let mut acc = _mm256_setzero_pd();
        for (d, q_slot) in q_pack.iter().enumerate().take(args.h_dim) {
            // SAFETY: `q_slot` is a valid `[f64; 4]` element.
            let qv = unsafe { _mm256_loadu_pd(q_slot.as_ptr()) };
            let kv = _mm256_set1_pd(k_tile[d * STRIDE_ANTI_ALIAS + local_k]);
            acc = _mm256_add_pd(acc, _mm256_mul_pd(qv, kv));
        }
        let scaled = _mm256_mul_pd(acc, scale);
        // SAFETY: `tmp` is a valid `[f64; 4]` stack array.
        unsafe {
            _mm256_storeu_pd(tmp.as_mut_ptr(), scaled);
        }
        for lane in 0..bq {
            let rb = rel_bias_for(args, q_row + lane, k_row);
            logits_4[lane * args.num_lines + k_row] = rb + tmp[lane];
        }
    }
}

/// AVX2 `BQ=4 x BK=252` stride-264 dense QK attention kernel (4K-aliasing immune, non-FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
pub(super) unsafe fn dense_qk_head_avx2(
    args: &DenseQkArgs<'_>,
    k_tile: &mut Vec<f64>,
    logits_4: &mut Vec<f64>,
    merged: &mut [f32],
) {
    k_tile.resize(args.h_dim * STRIDE_ANTI_ALIAS, 0.0);
    logits_4.resize(4 * args.num_lines, 0.0);
    let mut q_pack = vec![[0.0f64; 4]; args.h_dim];

    let mut q_row = 0usize;
    while q_row < args.num_lines {
        let bq = (args.num_lines - q_row).min(4);
        pack_q_block(args, q_row, bq, &mut q_pack);
        logits_4[..bq * args.num_lines].fill(0.0);

        let mut k_start = 0usize;
        while k_start < args.num_lines {
            let blen = (args.num_lines - k_start).min(BK_ANTI_ALIAS);
            pack_k_block(args, k_start, blen, k_tile);
            // SAFETY: AVX2 active and slices sized above.
            unsafe {
                dot_bq4_avx2(args, &q_pack, k_tile, q_row, k_start, bq, blen, logits_4);
            }
            k_start += blen;
        }

        for lane in 0..bq {
            let row_l = &mut logits_4[lane * args.num_lines..(lane + 1) * args.num_lines];
            let max_l = row_l.iter().copied().fold(args.sink, f64::max);
            // SAFETY: AVX2 is active in this function.
            let exp_sum =
                (args.sink - max_l).exp() + unsafe { softmax_exp_slice_avx2(row_l, max_l) };
            accum_dense_v(args, q_row + lane, row_l, exp_sum, merged);
        }
        q_row += bq;
    }
}
