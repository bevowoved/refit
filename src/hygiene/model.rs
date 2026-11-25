use super::ops::{
    EvalScratch, MAX_COLS, PEAK_WINDOW, RIBBON_GUARD, axis_conv_row, encode_grid_into,
    eval_span_block01_scalar, norm_sketch, sigmoid_f64, sink_attention_scalar,
};
#[cfg(target_arch = "x86_64")]
use super::ops::{eval_span_block01_avx2, sink_attention_avx2};
use super::simd::{linear_sparse_direct_csr, sparse_parts_direct_csr};
#[cfg(target_arch = "x86_64")]
use super::simd::{linear_sparse_width_gated_avx2, sparse_parts_width_gated_avx2};
use super::weights::ResolvedWeights;
#[cfg(any(feature = "neural-v6", test))]
use super::weights::{CompactConv, CompactCsr};
use crate::types::{HygieneResult, Squiggle};
use std::cell::RefCell;

const PRIOR_LINES: f64 = 16.0;
const PEAK_MIN_GATE: f32 = 10.0;
const PEAK_PRIOR: f32 = 6.0;
const SPAN_CONT_WEIGHT: f32 = 0.25;
const SQUIGGLE_THRESHOLD: f32 = 0.55;
const SQUIGGLE_LOGIT_PREFILTER: f32 = 0.20067;
const SCRATCH_SHRINK_LINES: usize = 2048;

pub(super) const SQUIGGLE_SPEC: [(&str, &str, f64); 6] = [
    ("deep_nesting", "Deep nesting pyramid; flatten with guard clauses or helpers", 0.42),
    ("dense_line", "Wide line, packed statements, call train, or nested ternary", 0.10),
    ("cramped_layout", "Unbroken statement wall or ragged off-grid indentation", 0.08),
    ("duplicate_block", "Duplicated or structurally cloned multi-line block", 0.58),
    ("repetitive_structure", "Repetitive parallel equality or branch ladder", 0.34),
    ("comment_echo", "Comment echoes adjacent code or leaves commented-out code", 0.18),
];

thread_local! {
    static EVAL_SCRATCH: RefCell<EvalScratch> = RefCell::new(EvalScratch::new());
}

pub(super) fn score_v07_with_weights(content: &str, w: &ResolvedWeights) -> HygieneResult {
    EVAL_SCRATCH.with(|cell| {
        let mut sc_guard = cell.borrow_mut();
        let sc = &mut *sc_guard;
        sc.ensure_span_buffers(w);
        let num_lines = encode_grid_into(content, &mut sc.grid, &mut sc.row_lens);

        let res = dispatch_eval(w, num_lines, sc);
        sc.shrink_if_oversized(num_lines, SCRATCH_SHRINK_LINES);
        res
    })
}

#[cfg(test)]
pub(super) fn score_v07_scalar_only(content: &str, w: &ResolvedWeights) -> HygieneResult {
    EVAL_SCRATCH.with(|cell| {
        let mut sc_guard = cell.borrow_mut();
        let sc = &mut *sc_guard;
        sc.ensure_span_buffers(w);
        let num_lines = encode_grid_into(content, &mut sc.grid, &mut sc.row_lens);
        let res = eval_file_scalar(w, num_lines, sc);
        sc.shrink_if_oversized(num_lines, SCRATCH_SHRINK_LINES);
        res
    })
}

fn dispatch_eval(w: &ResolvedWeights, num_lines: usize, sc: &mut EvalScratch) -> HygieneResult {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: `is_x86_feature_detected!("avx2")` confirmed AVX2 support on this CPU.
        return unsafe { eval_file_avx2(w, num_lines, sc) };
    }
    eval_file_scalar(w, num_lines, sc)
}

fn prepare_file_buffers(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) {
    let cs_len = n * w.col_sum_proj.cols as usize;
    let c3_len = n * w.conv3_w.cols;
    let c7_len = n * w.conv7_w.cols;
    let sk_len = n * w.sketch_w.cols;
    let dl_len = n * w.ml_proj.cols as usize;
    let st_len = n * w.state_dim;

    resize_zero(&mut sc.col_sums, cs_len);
    resize_zero(&mut sc.conv3, c3_len);
    resize_zero(&mut sc.conv7, c7_len);
    resize_zero(&mut sc.sketch, sk_len);
    resize_zero(&mut sc.deltas, dl_len);
    resize_zero(&mut sc.state, st_len);
    resize_zero(&mut sc.gate, n);
}

#[inline]
fn resize_zero(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
    buf[..needed].fill(0.0);
}

fn eval_file_scalar(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) -> HygieneResult {
    prepare_file_buffers(w, n, sc);

    let mut first_nb: Option<(usize, usize)> = None;
    let mut paired_done = false;
    for row in 0..n {
        let raw_len = sc.row_lens[row] as usize;
        if raw_len == 0 {
            continue;
        }
        let slen = (raw_len + RIBBON_GUARD).min(MAX_COLS);
        if !paired_done {
            if let Some((row0, slen0)) = first_nb {
                eval_span_block01_scalar(w, row0, slen0, Some((row, slen)), true, sc);
                paired_done = true;
            } else {
                first_nb = Some((row, slen));
            }
        } else {
            eval_span_block01_scalar(w, row, slen, None, false, sc);
        }
    }
    if !paired_done
        && let Some((row0, slen0)) = first_nb
    {
        eval_span_block01_scalar(w, row0, slen0, None, true, sc);
    }

    build_state_scalar(w, n, sc);
    run_context_scalar(w, n, sc);
    finish_heads(w, n, sc)
}

// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eval_file_avx2(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) -> HygieneResult {
    prepare_file_buffers(w, n, sc);

    let mut first_nb: Option<(usize, usize)> = None;
    let mut paired_done = false;
    for row in 0..n {
        let raw_len = sc.row_lens[row] as usize;
        if raw_len == 0 {
            continue;
        }
        let slen = (raw_len + RIBBON_GUARD).min(MAX_COLS);
        if !paired_done {
            if let Some((row0, slen0)) = first_nb {
                // SAFETY: AVX2 is active in this function.
                unsafe {
                    eval_span_block01_avx2(w, row0, slen0, Some((row, slen)), true, sc);
                }
                paired_done = true;
            } else {
                first_nb = Some((row, slen));
            }
        } else {
            // SAFETY: AVX2 is active in this function.
            unsafe {
                eval_span_block01_avx2(w, row, slen, None, false, sc);
            }
        }
    }
    if !paired_done
        && let Some((row0, slen0)) = first_nb
    {
        // SAFETY: AVX2 is active in this function.
        unsafe {
            eval_span_block01_avx2(w, row0, slen0, None, true, sc);
        }
    }

    // SAFETY: AVX2 is active in this function.
    unsafe {
        build_state_avx2(w, n, sc);
        run_context_avx2(w, n, sc);
    }
    finish_heads(w, n, sc)
}

fn fill_ml_tile(w: &ResolvedWeights, n: usize, row_start: usize, b: usize, sc: &mut EvalScratch) {
    let ml_cols = w.ml_proj.cols as usize;
    let paren_cols = w.ml_paren_w.cols;

    for lane in 0..b {
        let row = row_start + lane;
        let d_off = row * ml_cols;
        let e_off = lane * ml_cols;
        for ch in 0..ml_cols {
            let d = sc.deltas[d_off + ch];
            let next = sc.running_delta[ch] + d;
            sc.excl_tile[e_off + ch] = next - d;
            sc.running_delta[ch] = next;
        }
        axis_conv_row(
            &sc.deltas,
            n,
            row,
            w,
            &mut sc.paren_tile[lane * paren_cols..(lane + 1) * paren_cols],
        );
    }
    linear_sparse_direct_csr(
        &sc.excl_tile[..b * ml_cols],
        b,
        ml_cols,
        &w.ml_gate_w,
        Some(&w.ml_gate_b),
        true,
        &mut sc.saw_tile[..b * w.ml_gate_w.cols as usize],
    );
}

fn copy_tile_into_state(w: &ResolvedWeights, row_start: usize, b: usize, sc: &mut EvalScratch) {
    let ls_cols = w.lmlp2_w.cols as usize;
    let sk_cols = w.sketch_w.cols;
    let dim = w.state_dim;
    for lane in 0..b {
        let row = row_start + lane;
        let dst = row * dim;
        let ls = lane * ls_cols;
        let sk = row * sk_cols;
        sc.state[dst..dst + ls_cols].copy_from_slice(&sc.stats_tile[ls..ls + ls_cols]);
        sc.state[dst + ls_cols..dst + dim].copy_from_slice(&sc.sketch[sk..sk + sk_cols]);
    }
}

fn build_state_scalar(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) {
    let cc = w.col_sum_proj.cols as usize;
    let c3c = w.conv3_w.cols;
    let c7c = w.conv7_w.cols;
    let ml_cols = w.ml_proj.cols as usize;
    let saw_cols = w.ml_gate_w.cols as usize;
    let paren_cols = w.ml_paren_w.cols;
    let h_cols = w.lmlp1_w.cols as usize;
    let ls_cols = w.lmlp2_w.cols as usize;

    norm_sketch(&mut sc.sketch, n, w.sketch_w.cols);
    sparse_parts_direct_csr(
        &[&sc.col_sums[..n * cc], &sc.conv3[..n * c3c], &sc.conv7[..n * c7c]],
        &w.ml_proj_ranges,
        n,
        &w.ml_proj,
        None,
        false,
        &mut sc.deltas[..n * ml_cols],
    );
    sc.running_delta[..ml_cols].fill(0.0);

    let mut row_start = 0usize;
    while row_start < n {
        let b = (n - row_start).min(8);
        fill_ml_tile(w, n, row_start, b, sc);
        let row_end = row_start + b;
        sparse_parts_direct_csr(
            &[
                &sc.col_sums[row_start * cc..row_end * cc],
                &sc.conv3[row_start * c3c..row_end * c3c],
                &sc.conv7[row_start * c7c..row_end * c7c],
                &sc.saw_tile[..b * saw_cols],
                &sc.paren_tile[..b * paren_cols],
            ],
            &w.lmlp1_ranges,
            b,
            &w.lmlp1_w,
            Some(&w.lmlp1_b),
            true,
            &mut sc.hidden_tile[..b * h_cols],
        );
        linear_sparse_direct_csr(
            &sc.hidden_tile[..b * h_cols],
            b,
            h_cols,
            &w.lmlp2_w,
            Some(&w.lmlp2_b),
            false,
            &mut sc.stats_tile[..b * ls_cols],
        );
        copy_tile_into_state(w, row_start, b, sc);
        row_start = row_end;
    }
}

// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn build_state_avx2(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) {
    let cc = w.col_sum_proj.cols as usize;
    let c3c = w.conv3_w.cols;
    let c7c = w.conv7_w.cols;
    let ml_cols = w.ml_proj.cols as usize;
    let saw_cols = w.ml_gate_w.cols as usize;
    let paren_cols = w.ml_paren_w.cols;
    let h_cols = w.lmlp1_w.cols as usize;
    let ls_cols = w.lmlp2_w.cols as usize;

    norm_sketch(&mut sc.sketch, n, w.sketch_w.cols);
    // SAFETY: AVX2 is active in this function.
    unsafe {
        sparse_parts_width_gated_avx2(
            &[&sc.col_sums[..n * cc], &sc.conv3[..n * c3c], &sc.conv7[..n * c7c]],
            &w.ml_proj_ranges,
            n,
            &w.ml_proj,
            None,
            false,
            &mut sc.deltas[..n * ml_cols],
            &mut sc.tile_in,
            &mut sc.tile_acc,
        );
    }
    sc.running_delta[..ml_cols].fill(0.0);

    let mut row_start = 0usize;
    while row_start < n {
        let b = (n - row_start).min(8);
        fill_ml_tile(w, n, row_start, b, sc);
        let row_end = row_start + b;
        // SAFETY: AVX2 is active in this function.
        unsafe {
            sparse_parts_width_gated_avx2(
                &[
                    &sc.col_sums[row_start * cc..row_end * cc],
                    &sc.conv3[row_start * c3c..row_end * c3c],
                    &sc.conv7[row_start * c7c..row_end * c7c],
                    &sc.saw_tile[..b * saw_cols],
                    &sc.paren_tile[..b * paren_cols],
                ],
                &w.lmlp1_ranges,
                b,
                &w.lmlp1_w,
                Some(&w.lmlp1_b),
                true,
                &mut sc.hidden_tile[..b * h_cols],
                &mut sc.tile_in,
                &mut sc.tile_acc,
            );
            linear_sparse_width_gated_avx2(
                &sc.hidden_tile[..b * h_cols],
                b,
                h_cols,
                &w.lmlp2_w,
                Some(&w.lmlp2_b),
                false,
                &mut sc.stats_tile[..b * ls_cols],
                &mut sc.tile_in,
                &mut sc.tile_acc,
            );
        }
        copy_tile_into_state(w, row_start, b, sc);
        row_start = row_end;
    }
}

fn run_context_scalar(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) {
    let dim = w.state_dim;
    for layer in &w.ctx_layers {
        sink_attention_scalar(&layer.attn, n, dim, sc);
        for idx in 0..n * dim {
            sc.state[idx] += sc.attn_out[idx];
        }
        let h_cols = layer.cmlp_w.cols as usize;
        let mut row_start = 0usize;
        while row_start < n {
            let b = (n - row_start).min(8);
            let lo = row_start * dim;
            let hi = (row_start + b) * dim;
            linear_sparse_direct_csr(
                &sc.state[lo..hi],
                b,
                dim,
                &layer.cmlp_w,
                Some(&layer.cmlp_b),
                true,
                &mut sc.ctx_h_tile[..b * h_cols],
            );
            linear_sparse_direct_csr(
                &sc.ctx_h_tile[..b * h_cols],
                b,
                h_cols,
                &layer.cmlp2_w,
                Some(&layer.cmlp2_b),
                false,
                &mut sc.ctx_o_tile[..b * dim],
            );
            for idx in 0..b * dim {
                sc.state[lo + idx] += sc.ctx_o_tile[idx];
            }
            row_start += b;
        }
    }
}

// SAFETY: Caller must guarantee AVX2 is available on the executing x86_64 CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn run_context_avx2(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) {
    let dim = w.state_dim;
    for layer in &w.ctx_layers {
        // SAFETY: AVX2 is active in this function.
        unsafe {
            sink_attention_avx2(&layer.attn, n, dim, sc);
        }
        for idx in 0..n * dim {
            sc.state[idx] += sc.attn_out[idx];
        }
        let h_cols = layer.cmlp_w.cols as usize;
        let mut row_start = 0usize;
        while row_start < n {
            let b = (n - row_start).min(8);
            let lo = row_start * dim;
            let hi = (row_start + b) * dim;
            // SAFETY: AVX2 is active in this function.
            unsafe {
                linear_sparse_width_gated_avx2(
                    &sc.state[lo..hi],
                    b,
                    dim,
                    &layer.cmlp_w,
                    Some(&layer.cmlp_b),
                    true,
                    &mut sc.ctx_h_tile[..b * h_cols],
                    &mut sc.tile_in,
                    &mut sc.tile_acc,
                );
                linear_sparse_width_gated_avx2(
                    &sc.ctx_h_tile[..b * h_cols],
                    b,
                    h_cols,
                    &layer.cmlp2_w,
                    Some(&layer.cmlp2_b),
                    false,
                    &mut sc.ctx_o_tile[..b * dim],
                    &mut sc.tile_in,
                    &mut sc.tile_acc,
                );
            }
            for idx in 0..b * dim {
                sc.state[lo + idx] += sc.ctx_o_tile[idx];
            }
            row_start += b;
        }
    }
}

fn finish_heads(w: &ResolvedWeights, n: usize, sc: &mut EvalScratch) -> HygieneResult {
    let dim = w.state_dim;
    let mut ranked = Vec::new();
    let mut total_squiggles = 0usize;
    let mut event_load = 0.0f64;
    let mut prev_line = [-10isize; 6];
    let mut span_len = [1usize; 6];
    let prob_span = (1.0f64 - SQUIGGLE_THRESHOLD as f64).max(1e-6);

    let mut sq_row = [0.0f32; 6];
    let mut gate_row = [0.0f32; 1];

    for row in 0..n {
        let slice = &sc.state[row * dim..(row + 1) * dim];
        linear_sparse_direct_csr(slice, 1, dim, &w.pool_gate_w, Some(&w.pool_gate_b), false, &mut gate_row);
        sc.gate[row] = sigmoid_f64(gate_row[0] as f64) as f32;

        linear_sparse_direct_csr(slice, 1, dim, &w.sq_w, Some(&w.sq_b), false, &mut sq_row);
        let max_logit = sq_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_logit < SQUIGGLE_LOGIT_PREFILTER {
            continue;
        }

        let (mut best_k, mut best_p) = (0usize, sigmoid_f64(sq_row[0] as f64) as f32);
        for (k, &logit) in sq_row.iter().enumerate().skip(1) {
            let prob_f32 = sigmoid_f64(logit as f64) as f32;
            if prob_f32 > best_p {
                best_p = prob_f32;
                best_k = k;
            }
        }
        if best_p < SQUIGGLE_THRESHOLD {
            continue;
        }

        total_squiggles += 1;
        let line_no = row + 1;
        let (kind, msg, sev) = SQUIGGLE_SPEC[best_k];
        let excess = (((best_p as f64) - SQUIGGLE_THRESHOLD as f64) / prob_span).clamp(0.0, 1.0);
        let conf_weight = 0.35 + 0.65 * excess;
        let span_mult = if (line_no as isize) - prev_line[best_k] <= 2 {
            span_len[best_k] += 1;
            let cur = span_len[best_k] as f64;
            0.25 * (cur / (cur - 1.0)).ln()
        } else {
            span_len[best_k] = 1;
            1.0
        };
        prev_line[best_k] = line_no as isize;
        event_load += sev * conf_weight * span_mult;

        let prob = ((best_p as f64) * 10000.0).round() / 10000.0;
        let sq = Squiggle {
            line: line_no,
            kind: kind.into(),
            probability: prob,
            message: msg.into(),
        };
        ranked.push((sev * prob, best_k, sq));
    }

    let doc_v = pool_doc_ring(n, dim, sc);
    let mut doc_h = vec![0.0f32; w.doc1_w.cols as usize];
    linear_sparse_direct_csr(&doc_v, 1, dim * 2, &w.doc1_w, Some(&w.doc1_b), true, &mut doc_h);
    let mut doc_l = [0.0f32; 1];
    linear_sparse_direct_csr(
        &doc_h,
        1,
        w.doc1_w.cols as usize,
        &w.doc2_w,
        Some(&w.doc2_b),
        false,
        &mut doc_l,
    );
    let score = (sigmoid_f64(doc_l[0] as f64) * 10000.0).round() / 10000.0;
    let rounded_load = (event_load * 10000.0).round() / 10000.0;

    HygieneResult {
        score,
        squiggles: rank_squiggles(ranked),
        cached: false,
        latency_us: 1,
        strict_score: None,
        veto_cap: None,
        total_squiggles: Some(total_squiggles),
        event_load: Some(rounded_load),
        triggered_vetoes: Vec::new(),
        multiscale_score: None,
        calibration_log: None,
    }
}

#[cfg(any(feature = "neural-v6", test))]
fn add_csr_counts(total: &mut u64, nnz: &mut u64, mat: &CompactCsr) {
    *total += (mat.rows as u64) * (mat.cols as u64);
    *nnz += mat.non_zero_count as u64;
}

#[cfg(any(feature = "neural-v6", test))]
fn add_conv_counts(total: &mut u64, nnz: &mut u64, conv: &CompactConv) {
    *total += (conv.k_size as u64) * (conv.rows as u64) * (conv.cols as u64);
    *nnz += conv.non_zero_count as u64;
}

/// Compute aggregate ternary weight zero ratio across all layers in `ResolvedWeights`.
#[cfg(any(feature = "neural-v6", test))]
pub(super) fn weight_zero_ratio(weights: &ResolvedWeights) -> f64 {
    let mut total = 0u64;
    let mut nnz = 0u64;
    for emb in [&weights.tok_emb, &weights.col_emb] {
        total += (emb.rows as u64) * (emb.cols as u64);
        nnz += emb.data.iter().filter(|&&val| val != 0.0).count() as u64;
    }
    for conv in [
        &weights.pre_conv3_w,
        &weights.conv3_w,
        &weights.conv7_w,
        &weights.sketch_w,
        &weights.ml_paren_w,
    ] {
        add_conv_counts(&mut total, &mut nnz, conv);
    }
    for mat in [
        &weights.pref1_proj,
        &weights.gate1_w,
        &weights.pref2_proj,
        &weights.col_fuse_w,
        &weights.col_sum_proj,
        &weights.ml_proj,
        &weights.ml_gate_w,
        &weights.lmlp1_w,
        &weights.lmlp2_w,
        &weights.sq_w,
        &weights.pool_gate_w,
        &weights.doc1_w,
        &weights.doc2_w,
    ] {
        add_csr_counts(&mut total, &mut nnz, mat);
    }
    for layer in &weights.ctx_layers {
        if let Some(ref wq) = layer.attn.wq {
            add_csr_counts(&mut total, &mut nnz, wq);
        }
        if let Some(ref wk) = layer.attn.wk {
            add_csr_counts(&mut total, &mut nnz, wk);
        }
        add_csr_counts(&mut total, &mut nnz, &layer.attn.wv);
        add_csr_counts(&mut total, &mut nnz, &layer.attn.wo);
        add_csr_counts(&mut total, &mut nnz, &layer.cmlp_w);
        add_csr_counts(&mut total, &mut nnz, &layer.cmlp2_w);
    }
    if total == 0 {
        return 0.0;
    }
    1.0 - (nnz as f64) / (total as f64)
}

/// 33-row L1-resident circular ring-buffer `pool_doc` (bit-exact with full `(n + 1) * dim` prefix).
fn pool_doc_ring(n: usize, dim: usize, sc: &mut EvalScratch) -> Vec<f32> {
    const RING_LEN: usize = PEAK_WINDOW + 1;
    let mut doc = vec![0.0f32; dim * 2];
    sc.mean_acc[..dim].fill(0.0);
    sc.ring_w[..dim].fill(0.0);
    sc.ring_g[0] = 0.0;
    sc.prev1_raw[..dim].fill(0.0);
    sc.prev2_raw[..dim].fill(0.0);
    sc.peak[..dim].fill(f32::NEG_INFINITY);
    let mut g_sum = PRIOR_LINES;

    for row in 0..n {
        let hi = row + 1;
        let hi_slot = hi % RING_LEN;
        let prev_slot = row % RING_LEN;
        let lo = hi.saturating_sub(PEAK_WINDOW);
        let lo_slot = lo % RING_LEN;

        let gv = sc.gate[row];
        let g64 = gv as f64;
        g_sum += g64;
        sc.ring_g[hi_slot] = sc.ring_g[prev_slot] + gv;
        let denom = (sc.ring_g[hi_slot] - sc.ring_g[lo_slot]).max(PEAK_MIN_GATE) + PEAK_PRIOR;

        let st_base = row * dim;
        let prev_w_base = prev_slot * dim;
        let hi_w_base = hi_slot * dim;
        let lo_w_base = lo_slot * dim;
        for ch in 0..dim {
            let st = sc.state[st_base + ch];
            sc.mean_acc[ch] += (st as f64) * g64;

            let raw = st.max(0.0) * gv;
            let ov = if n > 1 { raw.min(sc.prev1_raw[ch].max(sc.prev2_raw[ch])) } else { 0.0 };
            let pos = raw - (1.0 - SPAN_CONT_WEIGHT) * ov;
            sc.prev2_raw[ch] = sc.prev1_raw[ch];
            sc.prev1_raw[ch] = raw;

            let pw = sc.ring_w[prev_w_base + ch] + pos;
            sc.ring_w[hi_w_base + ch] = pw;
            let val = (pw - sc.ring_w[lo_w_base + ch]) / denom;
            sc.peak[ch] = sc.peak[ch].max(val);
        }
    }

    for ch in 0..dim {
        doc[ch] = (sc.mean_acc[ch] / g_sum) as f32;
    }
    doc[dim..].copy_from_slice(&sc.peak[..dim]);
    doc
}

fn rank_squiggles(mut ranked: Vec<(f64, usize, Squiggle)>) -> Vec<Squiggle> {
    let cmp = |a: &(f64, usize, Squiggle), b: &(f64, usize, Squiggle)| {
        b.0.total_cmp(&a.0).then_with(|| a.2.line.cmp(&b.2.line))
    };
    ranked.sort_by(cmp);

    let mut counts = [0usize; 6];
    let (mut selected, mut overflow) = (Vec::with_capacity(48), Vec::new());
    for item in ranked {
        if counts[item.1] < 4 {
            counts[item.1] += 1;
            selected.push(item);
        } else {
            overflow.push(item);
        }
    }

    let need = 48usize.saturating_sub(selected.len());
    selected.extend(overflow.into_iter().take(need));
    selected.sort_by(cmp);
    selected.truncate(48);
    selected.into_iter().map(|(_, _, sq)| sq).collect()
}
