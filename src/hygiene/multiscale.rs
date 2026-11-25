//! CH-V5/V6/V7 Multiscale Block/Module Scorer, Adaptive O(N) Attention
//! (DeepSeek NSA + COBS + LongCat LSA + 128 KiB Engram + mHC-lite/cmHC),
//! and Cross-Layer MLI IndexCache Reuse with Proof-Tier Calibration Audit.

use super::ops::sigmoid_f64;
use super::simd::{DenseQkArgs, dense_qk_head_scalar};
#[cfg(target_arch = "x86_64")]
use super::simd::{dense_qk_head_avx2, dot_f32_as_f64_avx2, softmax_exp_slice_avx2};
use crate::types::HygieneResult;
use serde::{Deserialize, Serialize};

/// Crossover line threshold between exact dense attention (`<= 512`) and O(N) NSA (`> 512`).
pub const ADAPTIVE_CROSSOVER_LINES: usize = 512;
/// Key/query attention head width matching v0.7.0 `attn1` (`h_dim = 40`).
pub const ATTN_H_DIM: usize = 40;
/// Value head width matching v0.7.0 `attn1` (`v_dim = 16`).
pub const ATTN_V_DIM: usize = 16;
/// COBS 2nd-order block compression span (`block = 32` lines).
pub const COBS_BLOCK_SIZE: usize = 32;
/// LongCat LSA selected top-k sparse blocks packed into a contiguous `128 x d` L1 slab.
pub const LSA_TOP_K_BLOCKS: usize = 4;
/// Local sliding window half-span covering `span = 2..12` on both sides.
pub const LOCAL_WINDOW_HALF: usize = 12;
/// Default cmHC interior contraction factor preventing vertex collapse on the Birkhoff polytope.
pub const CMHC_GAMMA: f32 = 0.05;
/// Hard ceiling for the O(1) Engram conditional memory table (`128 KiB`).
pub const ENGRAM_MAX_BYTES: usize = 128 * 1024;
/// Power-of-two slot count keeping `EngramTable` strictly within `128 KiB` (76 KiB resident).
pub const ENGRAM_SLOTS: usize = 1024;
const ENGRAM_MASK: usize = ENGRAM_SLOTS - 1;
/// Ternary sparsity crossover below which AVX-512 5-trit LUT outperforms compact CSR.
pub const SPARSITY_LUT_THRESHOLD: f64 = 0.85;

/// All `4! = 24` exact 4x4 permutation matrices for Birkhoff-von Neumann `mHC-lite`.
pub const PERM4: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

const DEFAULT_MHC_AFFINITIES: [[f32; 4]; 4] = [
    [0.4, 0.9, 0.5, 0.3],
    [0.3, 1.2, 0.6, 0.2],
    [0.2, 0.5, 1.0, 0.4],
    [0.1, 0.4, 0.3, 0.9],
];
const DEFAULT_H_POST: [f32; 4] = [0.15, 0.45, 0.25, 0.15];

/// Explicit Sparsity Crossover Guard: keep Width-Gated CSR for 96.75%-zero v0.7.0 weights;
/// enable AVX-512 5-trit LUT strictly when `weight_zero_ratio < 0.85`.
pub fn should_use_5trit_lut(weight_zero_ratio: f64) -> bool {
    weight_zero_ratio.is_finite() && weight_zero_ratio < SPARSITY_LUT_THRESHOLD
}

/// Selected adaptive attention branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchMode {
    DenseV3bStride264,
    NsaCobsLsaMhcLite,
}

impl BranchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            BranchMode::DenseV3bStride264 => "dense_v3b_stride264",
            BranchMode::NsaCobsLsaMhcLite => "nsa_cobs_lsa_mhc_lite",
        }
    }
}

/// Output of Birkhoff-von Neumann `mHC-lite` mixing with `cmHC` contraction.
#[derive(Debug, Clone, Copy)]
pub struct MhcMixResult {
    pub matrix: [[f32; 4]; 4],
    pub branch_weights: [f32; 4],
    pub max_ds_err: f32,
    pub sinkhorn_iters: u32,
    pub gamma: f32,
}

/// Exact Birkhoff-von Neumann `mHC-lite` 4x4 doubly-stochastic mixing (0 Sinkhorn iterations)
/// combined with `cmHC` interior contraction toward the Birkhoff barycenter.
pub fn mhc_lite_bvn_4x4(raw: [[f32; 4]; 4], gamma_cmhc: f32) -> MhcMixResult {
    let gamma = gamma_cmhc.clamp(0.0, 1.0) as f64;
    let mut logits = [0.0f64; 24];
    let mut max_logit = f64::NEG_INFINITY;
    for (idx, perm) in PERM4.iter().enumerate() {
        let trace_sum = (raw[0][perm[0] as usize] as f64)
            + (raw[1][perm[1] as usize] as f64)
            + (raw[2][perm[2] as usize] as f64)
            + (raw[3][perm[3] as usize] as f64);
        logits[idx] = trace_sum;
        if trace_sum > max_logit {
            max_logit = trace_sum;
        }
    }

    let mut theta = [0.0f64; 24];
    let mut sum_exp = 0.0f64;
    for idx in 0..24 {
        let exp_val = (logits[idx] - max_logit).exp();
        theta[idx] = exp_val;
        sum_exp += exp_val;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-30);

    let mut mat64 = [[0.0f64; 4]; 4];
    for (idx, perm) in PERM4.iter().enumerate() {
        let wt = theta[idx] * inv_sum;
        for row in 0..4 {
            mat64[row][perm[row] as usize] += wt;
        }
    }

    let keep = 1.0 - gamma;
    let bary = gamma * 0.25;
    let mut matrix = [[0.0f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            matrix[row][col] = (keep * mat64[row][col] + bary) as f32;
        }
    }

    let max_ds_err = doubly_stochastic_error(&matrix);
    let mut branch_weights = [0.0f32; 4];
    for col in 0..4 {
        let mut col_acc = 0.0f32;
        for row in 0..4 {
            col_acc += DEFAULT_H_POST[row] * matrix[row][col];
        }
        branch_weights[col] = col_acc;
    }

    MhcMixResult {
        matrix,
        branch_weights,
        max_ds_err,
        sinkhorn_iters: 0,
        gamma: gamma as f32,
    }
}

fn doubly_stochastic_error(matrix: &[[f32; 4]; 4]) -> f32 {
    let mut max_err = 0.0f32;
    for row in 0..4 {
        let row_sum: f32 = matrix[row].iter().sum();
        max_err = max_err.max((row_sum - 1.0).abs());
    }
    for col in 0..4 {
        let col_sum: f32 = (0..4).map(|row| matrix[row][col]).sum();
        max_err = max_err.max((col_sum - 1.0).abs());
    }
    max_err
}

/// Deterministic `<= 128 KiB` O(1) Engram conditional memory table with RMSNorm gating.
#[derive(Debug, Clone)]
pub struct EngramTable {
    values: Vec<[f32; ATTN_V_DIM]>,
    idiom_risk: Vec<f32>,
    idiom_sig: Vec<u64>,
}

const DEFAULT_IDIOMS: [(&str, f32); 15] = [
    ("todo!(", 0.90),
    ("unimplemented!(", 0.90),
    ("transmute", 0.85),
    (".unwrap()", 0.55),
    (concat!("#!", "[allow("), 0.90),
    (concat!("#", "[allow("), 0.75),
    ("serde(flatten)", 0.80),
    ("@ts-nocheck", 0.90),
    ("@ts-ignore", 0.80),
    ("eslint-disable", 0.80),
    ("except: pass", 0.85),
    ("as any", 0.65),
    ("catch {}", 0.75),
    ("panic!(", 0.55),
    ("unsafe {", 0.45),
];

impl EngramTable {
    pub fn new() -> Self {
        let mut tbl = Self {
            values: vec![[0.0f32; ATTN_V_DIM]; ENGRAM_SLOTS],
            idiom_risk: vec![0.0f32; ENGRAM_SLOTS],
            idiom_sig: vec![0u64; ENGRAM_SLOTS],
        };
        tbl.seed_default_idioms();
        assert!(tbl.memory_bytes() <= ENGRAM_MAX_BYTES);
        tbl
    }

    /// Exact total resident memory footprint (stack struct + heap buffers), bounded `<= 128 KiB`.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.values.capacity() * std::mem::size_of::<[f32; ATTN_V_DIM]>()
            + self.idiom_risk.capacity() * std::mem::size_of::<f32>()
            + self.idiom_sig.capacity() * std::mem::size_of::<u64>()
    }

    fn seed_default_idioms(&mut self) {
        for (idx, &(gram, risk)) in DEFAULT_IDIOMS.iter().enumerate() {
            let (slot, sig) = hash_ngram_pair(gram.as_bytes());
            self.idiom_sig[slot] = sig;
            self.idiom_risk[slot] = risk;
            let mut vec_val = [0.1f32; ATTN_V_DIM];
            vec_val[idx % ATTN_V_DIM] = 1.0;
            vec_val[(idx + 5) % ATTN_V_DIM] = risk;
            self.values[slot] = vec_val;
        }
    }

    /// Populate O(1) row-sketch key/value memory for neural Branch 4 attention.
    pub fn populate_row_kv(&mut self, num_lines: usize, h_dim: usize, v_dim: usize, k_buf: &[f32], v_buf: &[f32]) {
        for slot_vec in &mut self.values {
            slot_vec.fill(0.0);
        }
        let copy_dim = v_dim.min(ATTN_V_DIM);
        for row in 0..num_lines {
            let k_slice = &k_buf[row * h_dim..(row + 1) * h_dim];
            let slot = hash_row_sketch(k_slice) & ENGRAM_MASK;
            let v_base = row * v_dim;
            for dim in 0..copy_dim {
                let candidate = v_buf[v_base + dim];
                if candidate > self.values[slot][dim] {
                    self.values[slot][dim] = candidate;
                }
            }
        }
    }

    /// Query row-sketch O(1) Engram slot.
    #[inline]
    pub fn lookup_row_vec(&self, q_slice: &[f32]) -> &[f32; ATTN_V_DIM] {
        let slot = hash_row_sketch(q_slice) & ENGRAM_MASK;
        &self.values[slot]
    }

    /// O(1) RMSNorm-gated n-gram anti-pattern lookup against seeded idioms.
    pub fn query_idiom_gated(&self, gram: &str, ctx_vec: &[f32; ATTN_V_DIM]) -> f32 {
        let (slot, sig) = hash_ngram_pair(gram.as_bytes());
        if self.idiom_sig[slot] != sig {
            return 0.0;
        }
        let gate = rmsnorm_gate(ctx_vec, &self.values[slot]);
        self.idiom_risk[slot] * gate
    }

    /// Scan production lines with context-aware RMSNorm gating (suppressing `#[cfg(test)]` code).
    pub fn scan_source_idioms(&self, content: &str) -> (usize, f64) {
        let mut hits = 0usize;
        let mut risk_sum = 0.0f64;
        let mut test_tracker = TestBlockTracker::default();

        for raw_line in content.lines() {
            let in_test = test_tracker.step_line(raw_line);
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            let ctx_vec = line_context_vec(in_test);
            for &(gram, _) in &DEFAULT_IDIOMS {
                if trimmed.contains(gram) {
                    let gated = self.query_idiom_gated(gram, &ctx_vec);
                    if gated >= 0.35 {
                        hits += 1;
                        risk_sum += gated as f64;
                    }
                }
            }
        }
        (hits, risk_sum)
    }
}

impl Default for EngramTable {
    fn default() -> Self {
        Self::new()
    }
}

fn line_context_vec(in_test: bool) -> [f32; ATTN_V_DIM] {
    if in_test {
        [-1.0f32; ATTN_V_DIM]
    } else {
        [1.0f32; ATTN_V_DIM]
    }
}

/// RMSNorm context-aware gate `sigmoid(dot(RMSNorm(a), RMSNorm(b)) / sqrt(d))`.
#[inline]
pub fn rmsnorm_gate(local_vec: &[f32], engram_vec: &[f32]) -> f32 {
    let dim = local_vec.len().min(engram_vec.len());
    if dim == 0 {
        return 0.5;
    }
    let mut dot = 0.0f64;
    let mut sq_loc = 0.0f64;
    let mut sq_eng = 0.0f64;
    for idx in 0..dim {
        let loc_v = local_vec[idx] as f64;
        let eng_v = engram_vec[idx] as f64;
        dot += loc_v * eng_v;
        sq_loc += loc_v * loc_v;
        sq_eng += eng_v * eng_v;
    }
    let inv_dim = 1.0 / (dim as f64);
    let rms_loc = (sq_loc * inv_dim + 1e-6).sqrt();
    let rms_eng = (sq_eng * inv_dim + 1e-6).sqrt();
    let scaled = (dot / (rms_loc * rms_eng)) / (dim as f64).sqrt();
    sigmoid_f64(scaled) as f32
}

#[inline]
fn hash_ngram_pair(bytes: &[u8]) -> (usize, u64) {
    let mut h1 = 0xcbf2_9ce4_8422_2325u64;
    let mut h2 = 0x8422_2325_cbf2_9ce4u64;
    for &byte in bytes {
        h1 ^= byte as u64;
        h1 = h1.wrapping_mul(0x0100_0000_01b3);
        h2 = h2.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (byte as u64);
    }
    ((h1 as usize) & ENGRAM_MASK, h2 | 1)
}

#[inline]
fn hash_row_sketch(slice: &[f32]) -> usize {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (idx, &val) in slice.iter().enumerate() {
        if val > 0.35 {
            hash ^= (idx as u64).wrapping_mul(0x0100_0000_01b3);
            hash = hash.rotate_left(7);
        }
    }
    hash as usize
}

/// Attention inputs for adaptive dense / NSA+COBS+LSA dispatch.
pub struct AttnInputs<'a> {
    pub num_lines: usize,
    pub h_dim: usize,
    pub v_dim: usize,
    pub q_buf: &'a [f32],
    pub k_buf: &'a [f32],
    pub v_buf: &'a [f32],
    pub sink: f64,
}

/// Cross-layer sparse block index cache (`CH-V7` MLI / `IndexCache`).
#[derive(Debug, Clone, Default)]
pub struct IndexCache {
    pub num_lines: usize,
    pub block_size: usize,
    pub top_blocks: Vec<[usize; LSA_TOP_K_BLOCKS]>,
    pub picked_counts: Vec<usize>,
    pub reuse_hits: usize,
}

impl IndexCache {
    fn matches(&self, num_lines: usize) -> bool {
        self.num_lines == num_lines
            && self.block_size == COBS_BLOCK_SIZE
            && self.top_blocks.len() == num_lines
            && self.picked_counts.len() == num_lines
    }

    fn reset_for(&mut self, num_lines: usize) {
        self.num_lines = num_lines;
        self.block_size = COBS_BLOCK_SIZE;
        self.top_blocks.clear();
        self.top_blocks.resize(num_lines, [0usize; LSA_TOP_K_BLOCKS]);
        self.picked_counts.clear();
        self.picked_counts.resize(num_lines, 0);
    }
}

/// Result of one adaptive multiscale attention pass.
#[derive(Debug, Clone)]
pub struct AttnOutput {
    pub out: Vec<f32>,
    pub mode: BranchMode,
    pub pairs_evaluated: usize,
    pub mhc_ds_err: f32,
    pub mhc_sinkhorn_iters: u32,
    pub mhc_gamma: f32,
    pub mhc_min_entry: f32,
    pub mli_reused: bool,
}

fn min_matrix_entry(mat: &[[f32; 4]; 4]) -> f32 {
    let mut low = f32::INFINITY;
    for row in mat {
        for &val in row {
            if val < low {
                low = val;
            }
        }
    }
    low
}

/// Adaptive Attention Crossover:
/// - `N <= 512`: exact `CH-V3b` AVX2 `BQ=4 x BK=252` stride-264 dense attention (scalar fallback).
/// - `N > 512`: O(N) DeepSeek NSA 3-branch (COBS 2nd-order + LongCat LSA contiguous `128 x d`
///   L1 KV slab + local window `span = 2..12`) + Engram O(1) + `mHC-lite` + `IndexCache` MLI reuse.
pub fn run_adaptive_attention(
    inputs: &AttnInputs<'_>,
    engram: &mut EngramTable,
    index_cache: &mut IndexCache,
) -> AttnOutput {
    let mhc = mhc_lite_bvn_4x4(DEFAULT_MHC_AFFINITIES, CMHC_GAMMA);
    let min_entry = min_matrix_entry(&mhc.matrix);
    if inputs.num_lines == 0 {
        return AttnOutput {
            out: Vec::new(),
            mode: BranchMode::DenseV3bStride264,
            pairs_evaluated: 0,
            mhc_ds_err: mhc.max_ds_err,
            mhc_sinkhorn_iters: mhc.sinkhorn_iters,
            mhc_gamma: mhc.gamma,
            mhc_min_entry: min_entry,
            mli_reused: false,
        };
    }

    if inputs.num_lines <= ADAPTIVE_CROSSOVER_LINES {
        let out = run_dense_v3b_stride264(inputs);
        return AttnOutput {
            out,
            mode: BranchMode::DenseV3bStride264,
            pairs_evaluated: inputs.num_lines * inputs.num_lines,
            mhc_ds_err: mhc.max_ds_err,
            mhc_sinkhorn_iters: mhc.sinkhorn_iters,
            mhc_gamma: mhc.gamma,
            mhc_min_entry: min_entry,
            mli_reused: false,
        };
    }

    run_nsa_cobs_lsa_mhc(inputs, engram, index_cache, &mhc, min_entry)
}

fn run_dense_v3b_stride264(inputs: &AttnInputs<'_>) -> Vec<f32> {
    let rel_zero = [0.0f64; 65];
    let inv_s = (1.0f32 / (inputs.h_dim.max(1) as f32).sqrt()) as f64;
    let args = DenseQkArgs {
        q_proj: inputs.q_buf,
        k_proj: inputs.k_buf,
        v_proj: inputs.v_buf,
        num_lines: inputs.num_lines,
        qk_tot: inputs.h_dim,
        qk_off: 0,
        h_dim: inputs.h_dim,
        v_tot: inputs.v_dim,
        v_off: 0,
        v_dim: inputs.v_dim,
        inv_s,
        sink: inputs.sink,
        rel: &rel_zero,
        all_zero_rel: true,
    };
    let mut k_tile = Vec::new();
    let mut logits_4 = Vec::new();
    let mut merged = vec![0.0f32; inputs.num_lines * inputs.v_dim];
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: `is_x86_feature_detected!("avx2")` confirmed AVX2 on this CPU.
        unsafe {
            dense_qk_head_avx2(&args, &mut k_tile, &mut logits_4, &mut merged);
        }
        return merged;
    }
    dense_qk_head_scalar(&args, &mut k_tile, &mut logits_4, &mut merged);
    merged
}

struct CobsBlocks {
    num_blocks: usize,
    mu_k: Vec<f32>,
    var_k: Vec<f32>,
    mu_v: Vec<f32>,
}

fn build_cobs_blocks(inputs: &AttnInputs<'_>) -> CobsBlocks {
    let num_lines = inputs.num_lines;
    let h_dim = inputs.h_dim;
    let v_dim = inputs.v_dim;
    let num_blocks = num_lines.div_ceil(COBS_BLOCK_SIZE);
    let mut mu_k = vec![0.0f32; num_blocks * h_dim];
    let mut var_k = vec![0.0f32; num_blocks * h_dim];
    let mut mu_v = vec![0.0f32; num_blocks * v_dim];

    for blk in 0..num_blocks {
        let lo = blk * COBS_BLOCK_SIZE;
        let hi = (lo + COBS_BLOCK_SIZE).min(num_lines);
        let inv_cnt = 1.0f32 / ((hi - lo) as f32);
        let mk_slice = &mut mu_k[blk * h_dim..(blk + 1) * h_dim];
        let mv_slice = &mut mu_v[blk * v_dim..(blk + 1) * v_dim];
        accum_block_means(inputs, lo, hi, inv_cnt, mk_slice, mv_slice);
        let vk_slice = &mut var_k[blk * h_dim..(blk + 1) * h_dim];
        accum_block_var(inputs, lo, hi, inv_cnt, mk_slice, vk_slice);
    }

    CobsBlocks { num_blocks, mu_k, var_k, mu_v }
}

fn accum_block_means(
    inputs: &AttnInputs<'_>,
    lo: usize,
    hi: usize,
    inv_cnt: f32,
    mu_k_dst: &mut [f32],
    mu_v_dst: &mut [f32],
) {
    let h_dim = inputs.h_dim;
    let v_dim = inputs.v_dim;
    for row in lo..hi {
        let k_row = &inputs.k_buf[row * h_dim..(row + 1) * h_dim];
        for dim in 0..h_dim {
            mu_k_dst[dim] += k_row[dim] * inv_cnt;
        }
        let v_row = &inputs.v_buf[row * v_dim..(row + 1) * v_dim];
        for dim in 0..v_dim {
            mu_v_dst[dim] += v_row[dim] * inv_cnt;
        }
    }
}

fn accum_block_var(
    inputs: &AttnInputs<'_>,
    lo: usize,
    hi: usize,
    inv_cnt: f32,
    mu_k_row: &[f32],
    var_k_dst: &mut [f32],
) {
    let h_dim = inputs.h_dim;
    for row in lo..hi {
        let k_row = &inputs.k_buf[row * h_dim..(row + 1) * h_dim];
        for dim in 0..h_dim {
            let diff = k_row[dim] - mu_k_row[dim];
            var_k_dst[dim] += diff * diff * inv_cnt;
        }
    }
}

#[inline]
fn cobs_cumulant_score(q_slice: &[f32], mu_k: &[f32], var_k: &[f32], inv_s: f64) -> f64 {
    let mut first = 0.0f64;
    let mut second = 0.0f64;
    for dim in 0..q_slice.len() {
        let qv = q_slice[dim] as f64;
        first += qv * (mu_k[dim] as f64);
        second += (qv * qv) * (var_k[dim] as f64);
    }
    first * inv_s + 0.5 * second * inv_s * inv_s
}

fn score_all_blocks_and_select(
    q_slice: &[f32],
    cobs: &CobsBlocks,
    h_dim: usize,
    v_dim: usize,
    inv_s: f64,
    sink: f64,
    blk_scores: &mut [(f64, usize)],
    coarse_out: &mut [f32],
) -> ([usize; LSA_TOP_K_BLOCKS], usize) {
    let mut max_sc = sink;
    for blk in 0..cobs.num_blocks {
        let mu = &cobs.mu_k[blk * h_dim..(blk + 1) * h_dim];
        let var = &cobs.var_k[blk * h_dim..(blk + 1) * h_dim];
        let sc = cobs_cumulant_score(q_slice, mu, var, inv_s);
        blk_scores[blk] = (sc, blk);
        if sc > max_sc {
            max_sc = sc;
        }
    }

    coarse_out.fill(0.0);
    let mut exp_sum = (sink - max_sc).exp();
    for &(sc, blk) in blk_scores.iter().take(cobs.num_blocks) {
        let ew = (sc - max_sc).exp();
        exp_sum += ew;
        let v_base = blk * v_dim;
        for dim in 0..v_dim {
            coarse_out[dim] += (ew * (cobs.mu_v[v_base + dim] as f64)) as f32;
        }
    }
    let inv_sum = (1.0 / exp_sum) as f32;
    for val in coarse_out.iter_mut() {
        *val *= inv_sum;
    }

    let k_sel = LSA_TOP_K_BLOCKS.min(cobs.num_blocks);
    let mut top_b = [0usize; LSA_TOP_K_BLOCKS];
    let mut picked = 0usize;
    for slot in 0..k_sel {
        let mut best_blk = 0usize;
        let mut best_sc = f64::NEG_INFINITY;
        for &(sc, blk) in blk_scores.iter().take(cobs.num_blocks) {
            if sc > best_sc && !top_b[..picked].contains(&blk) {
                best_sc = sc;
                best_blk = blk;
            }
        }
        top_b[slot] = best_blk;
        picked += 1;
    }
    (top_b, picked)
}

fn coarse_from_cached_blocks(
    q_slice: &[f32],
    cobs: &CobsBlocks,
    cached_blocks: &[usize],
    h_dim: usize,
    v_dim: usize,
    inv_s: f64,
    sink: f64,
    coarse_out: &mut [f32],
) {
    let mut scores = [0.0f64; LSA_TOP_K_BLOCKS];
    let mut max_sc = sink;
    for (idx, &blk) in cached_blocks.iter().enumerate() {
        let mu = &cobs.mu_k[blk * h_dim..(blk + 1) * h_dim];
        let var = &cobs.var_k[blk * h_dim..(blk + 1) * h_dim];
        let sc = cobs_cumulant_score(q_slice, mu, var, inv_s);
        scores[idx] = sc;
        if sc > max_sc {
            max_sc = sc;
        }
    }
    coarse_out.fill(0.0);
    let mut exp_sum = (sink - max_sc).exp();
    for (idx, &blk) in cached_blocks.iter().enumerate() {
        let ew = (scores[idx] - max_sc).exp();
        exp_sum += ew;
        let v_base = blk * v_dim;
        for dim in 0..v_dim {
            coarse_out[dim] += (ew * (cobs.mu_v[v_base + dim] as f64)) as f32;
        }
    }
    let inv_sum = (1.0 / exp_sum) as f32;
    for val in coarse_out.iter_mut() {
        *val *= inv_sum;
    }
}

fn pack_lsa_slab(
    inputs: &AttnInputs<'_>,
    selected_blocks: &[usize],
    lsa_k: &mut [f32],
    lsa_v: &mut [f32],
) -> usize {
    let h_dim = inputs.h_dim;
    let v_dim = inputs.v_dim;
    let mut packed = 0usize;
    for &blk in selected_blocks {
        let lo = blk * COBS_BLOCK_SIZE;
        let hi = (lo + COBS_BLOCK_SIZE).min(inputs.num_lines);
        let cnt = hi - lo;
        lsa_k[packed * h_dim..(packed + cnt) * h_dim]
            .copy_from_slice(&inputs.k_buf[lo * h_dim..hi * h_dim]);
        lsa_v[packed * v_dim..(packed + cnt) * v_dim]
            .copy_from_slice(&inputs.v_buf[lo * v_dim..hi * v_dim]);
        packed += cnt;
    }
    packed
}

#[inline]
fn dot_f32_as_f64_scalar(left: &[f32], right: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for idx in 0..left.len() {
        acc += (left[idx] as f64) * (right[idx] as f64);
    }
    acc
}

fn attend_contiguous_slab_scalar(
    q_slice: &[f32],
    k_slab: &[f32],
    v_slab: &[f32],
    count: usize,
    h_dim: usize,
    v_dim: usize,
    inv_s: f64,
    sink: f64,
    logits: &mut [f64],
    out_vec: &mut [f32],
) {
    let mut max_l = sink;
    for idx in 0..count {
        let k_row = &k_slab[idx * h_dim..(idx + 1) * h_dim];
        let val = dot_f32_as_f64_scalar(q_slice, k_row) * inv_s;
        logits[idx] = val;
        if val > max_l {
            max_l = val;
        }
    }
    let mut exp_sum = (sink - max_l).exp();
    for val in &mut logits[..count] {
        *val = (*val - max_l).exp();
        exp_sum += *val;
    }
    out_vec.fill(0.0);
    for idx in 0..count {
        let aw = logits[idx] / exp_sum;
        let v_base = idx * v_dim;
        for dim in 0..v_dim {
            out_vec[dim] += (aw * (v_slab[v_base + dim] as f64)) as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on this x86_64 CPU.
unsafe fn attend_contiguous_slab_avx2(
    q_slice: &[f32],
    k_slab: &[f32],
    v_slab: &[f32],
    count: usize,
    h_dim: usize,
    v_dim: usize,
    inv_s: f64,
    sink: f64,
    logits: &mut [f64],
    out_vec: &mut [f32],
) {
    let mut max_l = sink;
    for idx in 0..count {
        let k_row = &k_slab[idx * h_dim..(idx + 1) * h_dim];
        // SAFETY: AVX2 is active in this function and both slices have length `h_dim`.
        let val = unsafe { dot_f32_as_f64_avx2(q_slice, k_row) } * inv_s;
        logits[idx] = val;
        if val > max_l {
            max_l = val;
        }
    }
    // SAFETY: AVX2 is active in this function.
    let exp_sum = (sink - max_l).exp() + unsafe { softmax_exp_slice_avx2(&mut logits[..count], max_l) };
    out_vec.fill(0.0);
    for idx in 0..count {
        let aw = logits[idx] / exp_sum;
        let v_base = idx * v_dim;
        for dim in 0..v_dim {
            out_vec[dim] += (aw * (v_slab[v_base + dim] as f64)) as f32;
        }
    }
}

fn attend_local_window_scalar(
    inputs: &AttnInputs<'_>,
    q_idx: usize,
    inv_s: f64,
    loc_logits: &mut [f64; LOCAL_WINDOW_HALF * 2 + 1],
    local_v: &mut [f32],
) -> usize {
    let l_lo = q_idx.saturating_sub(LOCAL_WINDOW_HALF);
    let l_hi = (q_idx + LOCAL_WINDOW_HALF + 1).min(inputs.num_lines);
    let count = l_hi - l_lo;
    let h_dim = inputs.h_dim;
    let v_dim = inputs.v_dim;
    let q_slice = &inputs.q_buf[q_idx * h_dim..(q_idx + 1) * h_dim];

    let mut max_l = inputs.sink;
    for (slot, k_idx) in (l_lo..l_hi).enumerate() {
        let k_row = &inputs.k_buf[k_idx * h_dim..(k_idx + 1) * h_dim];
        let val = dot_f32_as_f64_scalar(q_slice, k_row) * inv_s;
        loc_logits[slot] = val;
        if val > max_l {
            max_l = val;
        }
    }
    let mut exp_sum = (inputs.sink - max_l).exp();
    for val in &mut loc_logits[..count] {
        *val = (*val - max_l).exp();
        exp_sum += *val;
    }
    local_v.fill(0.0);
    for (slot, k_idx) in (l_lo..l_hi).enumerate() {
        let aw = loc_logits[slot] / exp_sum;
        let v_base = k_idx * v_dim;
        for dim in 0..v_dim {
            local_v[dim] += (aw * (inputs.v_buf[v_base + dim] as f64)) as f32;
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on this x86_64 CPU.
unsafe fn attend_local_window_avx2(
    inputs: &AttnInputs<'_>,
    q_idx: usize,
    inv_s: f64,
    loc_logits: &mut [f64; LOCAL_WINDOW_HALF * 2 + 1],
    local_v: &mut [f32],
) -> usize {
    let l_lo = q_idx.saturating_sub(LOCAL_WINDOW_HALF);
    let l_hi = (q_idx + LOCAL_WINDOW_HALF + 1).min(inputs.num_lines);
    let count = l_hi - l_lo;
    let h_dim = inputs.h_dim;
    let v_dim = inputs.v_dim;
    let q_slice = &inputs.q_buf[q_idx * h_dim..(q_idx + 1) * h_dim];

    let mut max_l = inputs.sink;
    for (slot, k_idx) in (l_lo..l_hi).enumerate() {
        let k_row = &inputs.k_buf[k_idx * h_dim..(k_idx + 1) * h_dim];
        // SAFETY: AVX2 is active in this function and both slices have length `h_dim`.
        let val = unsafe { dot_f32_as_f64_avx2(q_slice, k_row) } * inv_s;
        loc_logits[slot] = val;
        if val > max_l {
            max_l = val;
        }
    }
    // SAFETY: AVX2 is active in this function.
    let exp_sum = (inputs.sink - max_l).exp()
        + unsafe { softmax_exp_slice_avx2(&mut loc_logits[..count], max_l) };
    local_v.fill(0.0);
    for (slot, k_idx) in (l_lo..l_hi).enumerate() {
        let aw = loc_logits[slot] / exp_sum;
        let v_base = k_idx * v_dim;
        for dim in 0..v_dim {
            local_v[dim] += (aw * (inputs.v_buf[v_base + dim] as f64)) as f32;
        }
    }
    count
}

fn blend_mhc_row(
    branch_w: &[f32; 4],
    coarse_v: &[f32],
    fine_v: &[f32],
    local_v: &[f32],
    eng_vec: &[f32],
    alpha: f32,
    dst: &mut [f32],
) {
    for dim in 0..dst.len() {
        let eng_d = if dim < eng_vec.len() { eng_vec[dim] } else { 0.0 };
        dst[dim] = branch_w[0] * coarse_v[dim]
            + branch_w[1] * fine_v[dim]
            + branch_w[2] * local_v[dim]
            + branch_w[3] * (alpha * eng_d);
    }
}

struct NsaWorkspace {
    blk_scores: Vec<(f64, usize)>,
    lsa_k: Vec<f32>,
    lsa_v: Vec<f32>,
    lsa_logits: Vec<f64>,
    loc_logits: [f64; LOCAL_WINDOW_HALF * 2 + 1],
    coarse_v: Vec<f32>,
    fine_v: Vec<f32>,
    local_v: Vec<f32>,
}

impl NsaWorkspace {
    fn new(num_blocks: usize, h_dim: usize, v_dim: usize) -> Self {
        let slab_rows = LSA_TOP_K_BLOCKS * COBS_BLOCK_SIZE;
        Self {
            blk_scores: vec![(0.0f64, 0usize); num_blocks],
            lsa_k: vec![0.0f32; slab_rows * h_dim],
            lsa_v: vec![0.0f32; slab_rows * v_dim],
            lsa_logits: vec![0.0f64; slab_rows],
            loc_logits: [0.0f64; LOCAL_WINDOW_HALF * 2 + 1],
            coarse_v: vec![0.0f32; v_dim],
            fine_v: vec![0.0f32; v_dim],
            local_v: vec![0.0f32; v_dim],
        }
    }
}

fn run_nsa_cobs_lsa_mhc(
    inputs: &AttnInputs<'_>,
    engram: &mut EngramTable,
    index_cache: &mut IndexCache,
    mhc: &MhcMixResult,
    min_entry: f32,
) -> AttnOutput {
    let cobs = build_cobs_blocks(inputs);
    engram.populate_row_kv(inputs.num_lines, inputs.h_dim, inputs.v_dim, inputs.k_buf, inputs.v_buf);

    let reuse_indices = index_cache.matches(inputs.num_lines);
    if !reuse_indices {
        index_cache.reset_for(inputs.num_lines);
    } else {
        index_cache.reuse_hits += 1;
    }

    let mut work = NsaWorkspace::new(cobs.num_blocks, inputs.h_dim, inputs.v_dim);
    let mut out = vec![0.0f32; inputs.num_lines * inputs.v_dim];

    #[cfg(target_arch = "x86_64")]
    let pairs = if is_x86_feature_detected!("avx2") {
        // SAFETY: `is_x86_feature_detected!("avx2")` confirmed AVX2 on this CPU.
        unsafe {
            nsa_loop_avx2(
                inputs,
                &cobs,
                engram,
                index_cache,
                reuse_indices,
                &mhc.branch_weights,
                &mut work,
                &mut out,
            )
        }
    } else {
        nsa_loop_scalar(
            inputs,
            &cobs,
            engram,
            index_cache,
            reuse_indices,
            &mhc.branch_weights,
            &mut work,
            &mut out,
        )
    };
    #[cfg(not(target_arch = "x86_64"))]
    let pairs = nsa_loop_scalar(
        inputs,
        &cobs,
        engram,
        index_cache,
        reuse_indices,
        &mhc.branch_weights,
        &mut work,
        &mut out,
    );

    AttnOutput {
        out,
        mode: BranchMode::NsaCobsLsaMhcLite,
        pairs_evaluated: pairs,
        mhc_ds_err: mhc.max_ds_err,
        mhc_sinkhorn_iters: mhc.sinkhorn_iters,
        mhc_gamma: mhc.gamma,
        mhc_min_entry: min_entry,
        mli_reused: reuse_indices,
    }
}

fn nsa_loop_scalar(
    inputs: &AttnInputs<'_>,
    cobs: &CobsBlocks,
    engram: &EngramTable,
    index_cache: &mut IndexCache,
    reuse_indices: bool,
    branch_w: &[f32; 4],
    work: &mut NsaWorkspace,
    out: &mut [f32],
) -> usize {
    let inv_s = (1.0f32 / (inputs.h_dim.max(1) as f32).sqrt()) as f64;
    let mut pairs = 0usize;

    for q_idx in 0..inputs.num_lines {
        let q_slice = &inputs.q_buf[q_idx * inputs.h_dim..(q_idx + 1) * inputs.h_dim];
        let (top_b, picked) = if reuse_indices {
            let cnt = index_cache.picked_counts[q_idx];
            let blocks = index_cache.top_blocks[q_idx];
            coarse_from_cached_blocks(
                q_slice,
                cobs,
                &blocks[..cnt],
                inputs.h_dim,
                inputs.v_dim,
                inv_s,
                inputs.sink,
                &mut work.coarse_v,
            );
            pairs += cnt;
            (blocks, cnt)
        } else {
            let sel = score_all_blocks_and_select(
                q_slice,
                cobs,
                inputs.h_dim,
                inputs.v_dim,
                inv_s,
                inputs.sink,
                &mut work.blk_scores,
                &mut work.coarse_v,
            );
            index_cache.top_blocks[q_idx] = sel.0;
            index_cache.picked_counts[q_idx] = sel.1;
            pairs += cobs.num_blocks;
            sel
        };

        let lsa_cnt = pack_lsa_slab(inputs, &top_b[..picked], &mut work.lsa_k, &mut work.lsa_v);
        attend_contiguous_slab_scalar(
            q_slice,
            &work.lsa_k,
            &work.lsa_v,
            lsa_cnt,
            inputs.h_dim,
            inputs.v_dim,
            inv_s,
            inputs.sink,
            &mut work.lsa_logits,
            &mut work.fine_v,
        );
        pairs += lsa_cnt;

        let loc_cnt =
            attend_local_window_scalar(inputs, q_idx, inv_s, &mut work.loc_logits, &mut work.local_v);
        pairs += loc_cnt;

        let eng_vec = engram.lookup_row_vec(q_slice);
        let alpha = rmsnorm_gate(&work.local_v, eng_vec);
        let dst = &mut out[q_idx * inputs.v_dim..(q_idx + 1) * inputs.v_dim];
        blend_mhc_row(branch_w, &work.coarse_v, &work.fine_v, &work.local_v, eng_vec, alpha, dst);
    }
    pairs
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller guarantees AVX2 is available on this x86_64 CPU.
unsafe fn nsa_loop_avx2(
    inputs: &AttnInputs<'_>,
    cobs: &CobsBlocks,
    engram: &EngramTable,
    index_cache: &mut IndexCache,
    reuse_indices: bool,
    branch_w: &[f32; 4],
    work: &mut NsaWorkspace,
    out: &mut [f32],
) -> usize {
    let inv_s = (1.0f32 / (inputs.h_dim.max(1) as f32).sqrt()) as f64;
    let mut pairs = 0usize;

    for q_idx in 0..inputs.num_lines {
        let q_slice = &inputs.q_buf[q_idx * inputs.h_dim..(q_idx + 1) * inputs.h_dim];
        let (top_b, picked) = if reuse_indices {
            let cnt = index_cache.picked_counts[q_idx];
            let blocks = index_cache.top_blocks[q_idx];
            coarse_from_cached_blocks(
                q_slice,
                cobs,
                &blocks[..cnt],
                inputs.h_dim,
                inputs.v_dim,
                inv_s,
                inputs.sink,
                &mut work.coarse_v,
            );
            pairs += cnt;
            (blocks, cnt)
        } else {
            let sel = score_all_blocks_and_select(
                q_slice,
                cobs,
                inputs.h_dim,
                inputs.v_dim,
                inv_s,
                inputs.sink,
                &mut work.blk_scores,
                &mut work.coarse_v,
            );
            index_cache.top_blocks[q_idx] = sel.0;
            index_cache.picked_counts[q_idx] = sel.1;
            pairs += cobs.num_blocks;
            sel
        };

        let lsa_cnt = pack_lsa_slab(inputs, &top_b[..picked], &mut work.lsa_k, &mut work.lsa_v);
        // SAFETY: AVX2 is active in this function.
        unsafe {
            attend_contiguous_slab_avx2(
                q_slice,
                &work.lsa_k,
                &work.lsa_v,
                lsa_cnt,
                inputs.h_dim,
                inputs.v_dim,
                inv_s,
                inputs.sink,
                &mut work.lsa_logits,
                &mut work.fine_v,
            );
        }
        pairs += lsa_cnt;

        // SAFETY: AVX2 is active in this function.
        let loc_cnt = unsafe {
            attend_local_window_avx2(inputs, q_idx, inv_s, &mut work.loc_logits, &mut work.local_v)
        };
        pairs += loc_cnt;

        let eng_vec = engram.lookup_row_vec(q_slice);
        let alpha = rmsnorm_gate(&work.local_v, eng_vec);
        let dst = &mut out[q_idx * inputs.v_dim..(q_idx + 1) * inputs.v_dim];
        blend_mhc_row(branch_w, &work.coarse_v, &work.fine_v, &work.local_v, eng_vec, alpha, dst);
    }
    pairs
}

/// Optional `CH-V7` cross-file caller/churn ego-context metadata (from Stem/Genome/CSR graph).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EgoContext {
    pub fan_in: usize,
    pub fan_out: usize,
    pub churn_commits: usize,
    pub has_module_cycle: bool,
    pub neighbor_dirs: usize,
    pub external_field_readers: usize,
}

/// `CH-V5` Block/Function/Module structural metrics fusing Sonar cognitive complexity,
/// parameter/boolean arity, Ousterhout PIIR, and `ST-V4` deep-module rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockModuleMetrics {
    pub fn_count: usize,
    pub max_fn_loc: usize,
    pub max_cognitive: usize,
    pub max_params: usize,
    pub max_bool_params: usize,
    pub pub_items: usize,
    pub impl_loc: usize,
    pub piir: f64,
    pub field_leaks: usize,
    pub getter_leaks: usize,
    pub passthrough_fns: usize,
    pub shallow_module: bool,
    pub module_cycle: bool,
    pub tangle: bool,
    pub block_score: f64,
    pub delta_block: f64,
    pub structural_rules: Vec<String>,
}

#[derive(Default)]
struct TestBlockTracker {
    armed: bool,
    active: bool,
    depth: usize,
}

impl TestBlockTracker {
    fn step_line(&mut self, raw_line: &str) -> bool {
        let trimmed = raw_line.trim();
        if !self.active && trimmed.starts_with("#[cfg(test)]") {
            self.armed = true;
            return true;
        }
        let opens = raw_line.bytes().filter(|&byte| byte == b'{').count();
        let closes = raw_line.bytes().filter(|&byte| byte == b'}').count();
        if self.active {
            self.depth = self.depth.saturating_add(opens).saturating_sub(closes);
            if self.depth == 0 {
                self.active = false;
                self.armed = false;
            }
            return true;
        }
        if self.armed {
            if opens > 0 {
                self.depth = opens.saturating_sub(closes);
                self.active = self.depth > 0;
                self.armed = self.active;
            } else if trimmed.ends_with(';') {
                self.armed = false;
            }
            return true;
        }
        false
    }
}

#[derive(Default)]
struct BlockScanState {
    in_fn: bool,
    fn_depth: usize,
    fn_lines: usize,
    fn_cognitive: usize,
    fn_sig_has_self: bool,
    fn_body_field_ret: bool,
    fn_body_calls_only: bool,
    in_struct: bool,
    struct_depth: usize,
    pub_field_decls: usize,
}

impl BlockScanState {
    fn finish_fn(&mut self, metrics: &mut BlockModuleMetrics) {
        metrics.fn_count += 1;
        metrics.max_fn_loc = metrics.max_fn_loc.max(self.fn_lines);
        metrics.max_cognitive = metrics.max_cognitive.max(self.fn_cognitive);
        if self.fn_sig_has_self && self.fn_lines <= 4 && self.fn_body_field_ret {
            metrics.getter_leaks += 1;
        }
        if self.fn_lines <= 4 && (self.fn_body_calls_only || self.fn_body_field_ret) {
            metrics.passthrough_fns += 1;
        }
        self.in_fn = false;
        self.fn_depth = 0;
        self.fn_lines = 0;
        self.fn_cognitive = 0;
        self.fn_sig_has_self = false;
        self.fn_body_field_ret = false;
        self.fn_body_calls_only = false;
    }
}

fn is_pub_item_decl(trimmed: &str) -> bool {
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub async fn ")
        || trimmed.starts_with("pub unsafe fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("pub const ")
        || trimmed.starts_with("export function ")
        || trimmed.starts_with("export class ")
}

fn matching_paren_close(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (rel, byte) in text[open..].bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(open + rel);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_fn_param_arity(trimmed: &str) -> Option<(usize, usize, bool)> {
    let is_fn_line = trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub(crate) fn ")
        || trimmed.starts_with("pub(super) fn ")
        || trimmed.starts_with("async fn ")
        || trimmed.starts_with("pub async fn ")
        || trimmed.starts_with("unsafe fn ")
        || trimmed.starts_with("pub unsafe fn ")
        || trimmed.starts_with("func ")
        || trimmed.starts_with("function ")
        || trimmed.starts_with("export function ")
        || trimmed.starts_with("def ");
    if !is_fn_line {
        return None;
    }
    let open = trimmed.find('(')?;
    let close = matching_paren_close(trimmed, open)?;
    let params_str = trimmed[open + 1..close].trim();
    let has_self = params_str.contains("&self") || params_str.contains("&mut self");
    if params_str.is_empty() {
        return Some((0, 0, has_self));
    }
    let mut total_params = 0usize;
    let mut bool_params = 0usize;
    for raw_param in split_top_level_commas(params_str) {
        let seg = raw_param.trim();
        if seg.is_empty() || seg == "self" || seg == "&self" || seg == "&mut self" || seg == "mut self" {
            continue;
        }
        total_params += 1;
        if param_is_bool(seg) {
            bool_params += 1;
        }
    }
    Some((total_params, bool_params, has_self))
}

fn split_top_level_commas(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut angle = 0usize;
    let mut paren = 0usize;
    let mut brack = 0usize;
    for (idx, byte) in text.bytes().enumerate() {
        match byte {
            b'<' => angle += 1,
            b'>' => angle = angle.saturating_sub(1),
            b'(' => paren += 1,
            b')' => paren = paren.saturating_sub(1),
            b'[' => brack += 1,
            b']' => brack = brack.saturating_sub(1),
            b',' if angle == 0 && paren == 0 && brack == 0 => {
                parts.push(&text[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn param_is_bool(param: &str) -> bool {
    let ty_part = param.split_once(':').map_or(param, |(_, rhs)| rhs).trim();
    ty_part == "bool"
        || ty_part == "boolean"
        || ty_part.ends_with(" bool")
        || ty_part.starts_with("bool ")
}

fn line_cognitive_increment(trimmed: &str, nesting: usize) -> usize {
    let mut add = 0usize;
    let is_else_if = trimmed.starts_with("else if ") || trimmed.starts_with("} else if ");
    let is_else = trimmed.starts_with("else ") || trimmed.starts_with("} else {") || trimmed == "} else";
    let is_cf = trimmed.starts_with("if ")
        || trimmed.starts_with("for ")
        || trimmed.starts_with("while ")
        || trimmed.starts_with("loop ")
        || trimmed.starts_with("match ")
        || trimmed.starts_with("switch ");
    if is_else_if || is_else {
        add += 1;
    } else if is_cf {
        add += 1 + nesting;
    }
    let has_and = trimmed.contains("&&");
    let has_or = trimmed.contains("||");
    if has_and && has_or {
        add += 2;
    } else if has_and || has_or {
        add += 1;
    }
    add
}

fn check_pub_struct_field(trimmed: &str) -> bool {
    if trimmed.contains("PhantomData") {
        return false;
    }
    let is_field_vis = trimmed.starts_with("pub ") || trimmed.starts_with("pub(crate) ");
    let is_item_kw = trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("pub const ");
    (is_field_vis && !is_item_kw && trimmed.contains(':')) || trimmed.contains("{ pub ")
}

fn step_block_line(
    trimmed: &str,
    opens: usize,
    closes: usize,
    scan: &mut BlockScanState,
    metrics: &mut BlockModuleMetrics,
) {
    if is_pub_item_decl(trimmed) {
        metrics.pub_items += 1;
    }
    if (trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ")) && opens > 0 {
        scan.in_struct = true;
        scan.struct_depth = opens.saturating_sub(closes);
        if check_pub_struct_field(trimmed) {
            scan.pub_field_decls += 1;
        }
        if scan.struct_depth == 0 {
            scan.in_struct = false;
        }
    } else if scan.in_struct {
        if check_pub_struct_field(trimmed) {
            scan.pub_field_decls += 1;
        }
        scan.struct_depth = scan.struct_depth.saturating_add(opens).saturating_sub(closes);
        if scan.struct_depth == 0 {
            scan.in_struct = false;
        }
    }

    if !scan.in_fn
        && let Some((params, bools, has_self)) = parse_fn_param_arity(trimmed)
    {
        metrics.max_params = metrics.max_params.max(params);
        metrics.max_bool_params = metrics.max_bool_params.max(bools);
        if opens > 0 {
            scan.in_fn = true;
            scan.fn_depth = opens.saturating_sub(closes);
            scan.fn_lines = 1;
            scan.fn_sig_has_self = has_self;
            if trimmed.contains("&self.") && !trimmed.contains("PhantomData") {
                scan.fn_body_field_ret = true;
            }
            if scan.fn_depth == 0 {
                scan.finish_fn(metrics);
            }
        }
        return;
    }

    if scan.in_fn {
        scan.fn_lines += 1;
        let nesting = scan.fn_depth.saturating_sub(1);
        scan.fn_cognitive += line_cognitive_increment(trimmed, nesting);
        if trimmed.contains("&self.") && !trimmed.contains("PhantomData") {
            scan.fn_body_field_ret = true;
        }
        if trimmed.contains('(') && !trimmed.starts_with("if ") && !trimmed.starts_with("for ") {
            scan.fn_body_calls_only = true;
        }
        scan.fn_depth = scan.fn_depth.saturating_add(opens).saturating_sub(closes);
        if scan.fn_depth == 0 {
            scan.finish_fn(metrics);
        }
    }
}

/// Analyze `CH-V5` Block/Function/Module structure fusing Sonar cognitive complexity,
/// `max_fn_loc`, param/bool arity, Ousterhout PIIR, and `ST-V4` deep-module rules.
pub fn analyze_block_module(
    path: &str,
    content: &str,
    ego: Option<&EgoContext>,
) -> BlockModuleMetrics {
    let mut metrics = BlockModuleMetrics::default();
    if super::strict::is_test_only_path(path) {
        metrics.block_score = 1.0;
        return metrics;
    }

    let mut test_tracker = TestBlockTracker::default();
    let mut scan = BlockScanState::default();

    for raw_line in content.lines() {
        if test_tracker.step_line(raw_line) {
            continue;
        }
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        metrics.impl_loc += 1;
        let opens = raw_line.bytes().filter(|&byte| byte == b'{').count();
        let closes = raw_line.bytes().filter(|&byte| byte == b'}').count();
        step_block_line(trimmed, opens, closes, &mut scan, &mut metrics);
    }
    if scan.in_fn {
        scan.finish_fn(&mut metrics);
    }

    finalize_block_metrics(&mut metrics, scan.pub_field_decls, ego);
    metrics
}

fn finalize_block_metrics(
    metrics: &mut BlockModuleMetrics,
    pub_field_decls: usize,
    ego: Option<&EgoContext>,
) {
    let impl_loc = metrics.impl_loc.max(1) as f64;
    metrics.piir = (((metrics.pub_items as f64) * 3.0) / impl_loc).clamp(0.0, 1.0);
    metrics.piir = (metrics.piir * 10_000.0).round() / 10_000.0;

    let ext_readers = ego.map_or(0, |ctx| ctx.external_field_readers);
    let field_leak_hit = metrics.getter_leaks >= 2
        || (pub_field_decls > 0 && ext_readers >= 2)
        || (pub_field_decls > 0 && metrics.getter_leaks >= 1);
    if field_leak_hit {
        metrics.field_leaks = metrics.getter_leaks.max(pub_field_decls);
        metrics.structural_rules.push("field-leak".into());
    }

    let passthrough_ratio = if metrics.fn_count > 0 {
        (metrics.passthrough_fns as f64) / (metrics.fn_count as f64)
    } else {
        0.0
    };
    let surface_density = (metrics.pub_items as f64) / impl_loc;
    let shallow_by_cap = metrics.pub_items > 7 && (metrics.piir >= 0.45 || passthrough_ratio >= 0.50);
    let shallow_by_wrap =
        metrics.pub_items >= 4 && metrics.fn_count >= 3 && surface_density > 0.20 && passthrough_ratio >= 0.60;
    if shallow_by_cap || shallow_by_wrap {
        metrics.shallow_module = true;
        metrics.structural_rules.push("shallow-module".into());
    }

    if let Some(ctx) = ego {
        if ctx.has_module_cycle {
            metrics.module_cycle = true;
            metrics.structural_rules.push("module-cycle".into());
        }
        let deg = ctx.fan_in + ctx.fan_out;
        if ctx.fan_in >= 6 && ctx.fan_out >= 6 && ctx.neighbor_dirs >= 2 && deg > 0 {
            let instability = (ctx.fan_out as f64) / (deg as f64);
            if (0.35..=0.65).contains(&instability) {
                metrics.tangle = true;
                metrics.structural_rules.push("tangle".into());
            }
        }
    }

    let delta = compute_delta_block(metrics);
    metrics.delta_block = (delta * 10_000.0).round() / 10_000.0;
    let norm = sigmoid_f64(2.0);
    let raw_score = (sigmoid_f64(2.0 - delta) / norm).clamp(0.0, 1.0);
    metrics.block_score = (raw_score * 10_000.0).round() / 10_000.0;
}

fn compute_delta_block(metrics: &BlockModuleMetrics) -> f64 {
    let mut delta = 0.0f64;
    if metrics.max_fn_loc > 60 {
        delta += 0.015 * ((metrics.max_fn_loc - 60) as f64);
    }
    if metrics.max_cognitive > 15 {
        delta += 0.10 * ((metrics.max_cognitive - 15) as f64);
    }
    if metrics.max_params > 5 {
        delta += 0.25 * ((metrics.max_params - 5) as f64);
    }
    if metrics.max_bool_params >= 2 {
        delta += 0.35 * ((metrics.max_bool_params - 1) as f64);
    }
    if metrics.field_leaks > 0 {
        delta += 0.75;
    }
    if metrics.shallow_module {
        delta += 0.65 + 0.15 * (metrics.pub_items.saturating_sub(7) as f64);
    }
    if metrics.module_cycle {
        delta += 1.20;
    }
    if metrics.tangle {
        delta += 0.80;
    }
    delta
}

/// Structured `CH-V7` Proof-Tier Calibration Audit Log serialized into `HygieneResult::calibration_log`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAudit {
    pub proof_tier: String,
    pub branch_mode: String,
    pub block_score: f64,
    pub piir: f64,
    pub mhc_ds_err: f32,
    pub mhc_sinkhorn_iters: u32,
    pub mhc_gamma: f32,
    pub mhc_min_entry: f32,
    pub pairs_evaluated: usize,
    pub engram_hits: usize,
    pub engram_bytes: usize,
    pub mli_reused: bool,
    pub mli_reuse_hits: usize,
    pub raw_v07: f64,
    pub delta_block: f64,
    pub delta_cross: f64,
    pub delta_slop: f64,
    pub veto_cap: f64,
    pub final_score: f64,
    pub use_5trit_lut: bool,
    pub structural_rules: Vec<String>,
}

struct LineFeatureTensors {
    num_lines: usize,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
}

fn encode_line_qkv(content: &str) -> LineFeatureTensors {
    let lines: Vec<&str> = content.lines().collect();
    let num_lines = lines.len();
    let mut q_buf = vec![0.0f32; num_lines * ATTN_H_DIM];
    let mut k_buf = vec![0.0f32; num_lines * ATTN_H_DIM];
    let mut v_buf = vec![0.0f32; num_lines * ATTN_V_DIM];

    for (row, &raw_line) in lines.iter().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let q_row = &mut q_buf[row * ATTN_H_DIM..(row + 1) * ATTN_H_DIM];
        let k_row = &mut k_buf[row * ATTN_H_DIM..(row + 1) * ATTN_H_DIM];
        let v_row = &mut v_buf[row * ATTN_V_DIM..(row + 1) * ATTN_V_DIM];
        fill_line_feature_row(raw_line, trimmed, q_row, k_row, v_row);
    }

    LineFeatureTensors { num_lines, q_buf, k_buf, v_buf }
}

fn fill_line_feature_row(
    raw_line: &str,
    trimmed: &str,
    q_row: &mut [f32],
    k_row: &mut [f32],
    v_row: &mut [f32],
) {
    let bytes = trimmed.as_bytes();
    let mut norm_sq = 0.0f32;
    for win in bytes.windows(3) {
        let bucket = ((win[0] as usize) * 31 + (win[1] as usize) * 17 + (win[2] as usize))
            % ATTN_H_DIM;
        q_row[bucket] += 1.0;
        k_row[bucket] += 1.0;
    }
    let indent = raw_line.len().saturating_sub(raw_line.trim_start().len()) as f32;
    q_row[0] += indent * 0.125;
    k_row[0] += indent * 0.125;
    for val in q_row.iter() {
        norm_sq += val * val;
    }
    let inv_norm = 1.0 / (norm_sq + 1e-6).sqrt();
    for dim in 0..ATTN_H_DIM {
        q_row[dim] *= inv_norm;
        k_row[dim] *= inv_norm;
    }
    let len_norm = ((bytes.len() as f32) / 128.0).min(2.0);
    v_row[0] = len_norm;
    v_row[1] = (indent / 16.0).min(2.0);
    for (dim, val) in v_row.iter_mut().enumerate().skip(2) {
        *val = q_row[dim];
    }
}

fn build_layer2_qkv(base: &LineFeatureTensors, layer1_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut q2 = base.q_buf.clone();
    let mut k2 = base.k_buf.clone();
    for row in 0..base.num_lines {
        let dst = row * ATTN_H_DIM;
        let src = row * ATTN_V_DIM;
        for dim in 0..ATTN_V_DIM {
            let res_val = 0.25 * layer1_out[src + dim];
            q2[dst + dim] += res_val;
            k2[dst + dim] += res_val;
        }
    }
    (q2, k2)
}

#[inline]
fn logit_f64(prob: f64) -> f64 {
    let clamped = prob.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

/// Run full `CH-V5..V7` multiscale analysis and Proof-Tier Calibration on top of a `HygieneResult`.
/// Leaves canonical `v0.7.0` `result.score` 100% untouched; populates `multiscale_score` and `calibration_log`.
pub fn score_multiscale(
    path: &str,
    content: &str,
    result: &mut HygieneResult,
    ego: Option<&EgoContext>,
    weight_zero_ratio: f64,
) -> CalibrationAudit {
    let mut engram = EngramTable::new();
    let engram_bytes = engram.memory_bytes();
    let (engram_hits, engram_risk) = engram.scan_source_idioms(content);

    let tensors = encode_line_qkv(content);
    let mut index_cache = IndexCache::default();
    let layer1_in = AttnInputs {
        num_lines: tensors.num_lines,
        h_dim: ATTN_H_DIM,
        v_dim: ATTN_V_DIM,
        q_buf: &tensors.q_buf,
        k_buf: &tensors.k_buf,
        v_buf: &tensors.v_buf,
        sink: 1.5,
    };
    let layer1 = run_adaptive_attention(&layer1_in, &mut engram, &mut index_cache);

    let (q2, k2) = build_layer2_qkv(&tensors, &layer1.out);
    let layer2_in = AttnInputs {
        num_lines: tensors.num_lines,
        h_dim: ATTN_H_DIM,
        v_dim: ATTN_V_DIM,
        q_buf: &q2,
        k_buf: &k2,
        v_buf: &layer1.out,
        sink: 1.5,
    };
    let layer2 = run_adaptive_attention(&layer2_in, &mut engram, &mut index_cache);

    let block_metrics = analyze_block_module(path, content, ego);
    let total_pairs = layer1.pairs_evaluated.saturating_add(layer2.pairs_evaluated);
    let audit = build_calibration_audit(
        result,
        &block_metrics,
        ego,
        &layer2,
        total_pairs,
        engram_hits,
        engram_bytes,
        engram_risk,
        index_cache.reuse_hits,
        weight_zero_ratio,
    );

    if audit.veto_cap < 1.0 {
        result.veto_cap = Some(result.veto_cap.map_or(audit.veto_cap, |c| c.min(audit.veto_cap)));
        for rule in &block_metrics.structural_rules {
            if !result.triggered_vetoes.iter().any(|v| v == rule) {
                result.triggered_vetoes.push(rule.clone());
            }
        }
    }
    result.multiscale_score = Some(audit.final_score);
    result.calibration_log = serde_json::to_string(&audit).ok();
    audit
}

fn build_calibration_audit(
    result: &HygieneResult,
    block: &BlockModuleMetrics,
    ego: Option<&EgoContext>,
    layer2: &AttnOutput,
    pairs_evaluated: usize,
    engram_hits: usize,
    engram_bytes: usize,
    engram_risk: f64,
    mli_reuse_hits: usize,
    weight_zero_ratio: f64,
) -> CalibrationAudit {
    let raw_v07 = result.score;
    let delta_block = block.delta_block;
    let delta_cross = compute_delta_cross(ego);
    let delta_slop = compute_delta_slop(result, engram_hits, engram_risk);

    let mut veto_cap = result.veto_cap.unwrap_or(1.0);
    if block.module_cycle || block.field_leaks > 0 || block.shallow_module {
        veto_cap = veto_cap.min(0.48);
    }

    let strict_bound = result.strict_score.unwrap_or(raw_v07);
    let total_delta = delta_block + delta_cross + delta_slop;
    let calibrated = if total_delta == 0.0 && veto_cap >= 1.0 {
        strict_bound
    } else {
        let cal = sigmoid_f64(logit_f64(raw_v07) - total_delta).min(veto_cap).min(strict_bound);
        (cal * 10_000.0).round() / 10_000.0
    };

    let proof_tier = if veto_cap < 1.0 {
        "tier0_vetoed"
    } else if layer2.mode == BranchMode::DenseV3bStride264 {
        "tier1_exact_dense"
    } else {
        "tier2_nsa_mli_calibrated"
    };

    CalibrationAudit {
        proof_tier: proof_tier.into(),
        branch_mode: layer2.mode.as_str().into(),
        block_score: block.block_score,
        piir: block.piir,
        mhc_ds_err: layer2.mhc_ds_err,
        mhc_sinkhorn_iters: layer2.mhc_sinkhorn_iters,
        mhc_gamma: layer2.mhc_gamma,
        mhc_min_entry: layer2.mhc_min_entry,
        pairs_evaluated,
        engram_hits,
        engram_bytes,
        mli_reused: layer2.mli_reused,
        mli_reuse_hits,
        raw_v07,
        delta_block,
        delta_cross,
        delta_slop,
        veto_cap,
        final_score: calibrated,
        use_5trit_lut: should_use_5trit_lut(weight_zero_ratio),
        structural_rules: block.structural_rules.clone(),
    }
}

fn compute_delta_cross(ego: Option<&EgoContext>) -> f64 {
    let Some(ctx) = ego else { return 0.0 };
    let blast = (ctx.fan_in.saturating_mul(ctx.fan_out) as f64).sqrt();
    let churn_mult = 1.0 + ((ctx.churn_commits as f64) / 20.0).min(1.5);
    let raw = if blast > 6.0 { 0.08 * (blast - 6.0) * churn_mult } else { 0.0 };
    (raw * 10_000.0).round() / 10_000.0
}

fn compute_delta_slop(result: &HygieneResult, engram_hits: usize, engram_risk: f64) -> f64 {
    let mut delta = 0.0f64;
    if let Some(load) = result.event_load
        && load > 1.5
    {
        delta += 0.15 * (load - 1.5);
    }
    if engram_hits >= 2 {
        delta += 0.25 * engram_risk;
    }
    (delta * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hygiene::{HygieneScorer, model, weights};

    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next_u64(&mut self) -> u64 {
            let mut val = self.0;
            val ^= val << 13;
            val ^= val >> 7;
            val ^= val << 17;
            self.0 = val;
            val
        }

        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
        }
    }

    struct PlantedCloneDataset {
        num_lines: usize,
        q_buf: Vec<f32>,
        k_buf: Vec<f32>,
        v_buf: Vec<f32>,
        sink: f64,
        clone_src: usize,
        clone_dst: usize,
    }

    fn gen_planted_clone_dataset(num_lines: usize, seed: u64) -> PlantedCloneDataset {
        let mut rng = TestRng::new(seed);
        let mut q_buf = vec![0.0f32; num_lines * ATTN_H_DIM];
        let mut k_buf = vec![0.0f32; num_lines * ATTN_H_DIM];
        let mut v_buf = vec![0.0f32; num_lines * ATTN_V_DIM];

        for row in 0..num_lines {
            if row % 10 < 3 {
                continue;
            }
            let mut norm_q = 0.0f32;
            for dim in 0..ATTN_H_DIM {
                let sample = rng.next_f32() - 0.5;
                q_buf[row * ATTN_H_DIM + dim] = sample;
                k_buf[row * ATTN_H_DIM + dim] = sample + (rng.next_f32() - 0.5) * 0.15;
                norm_q += sample * sample;
            }
            let inv = 1.0 / (norm_q + 1e-6).sqrt();
            for dim in 0..ATTN_H_DIM {
                q_buf[row * ATTN_H_DIM + dim] *= inv;
                k_buf[row * ATTN_H_DIM + dim] *= inv;
            }
            for dim in 0..ATTN_V_DIM {
                v_buf[row * ATTN_V_DIM + dim] = rng.next_f32() * 0.4;
            }
        }

        let clone_src = num_lines / 8;
        let clone_dst = (clone_src + num_lines / 2 + 7).min(num_lines - 9);
        for off in 0..8 {
            let src_row = clone_src + off;
            let dst_row = clone_dst + off;
            for dim in 0..ATTN_H_DIM {
                let spike = if dim == (off * 4) % ATTN_H_DIM || dim == (off * 4 + 1) % ATTN_H_DIM {
                    4.5f32
                } else {
                    0.0
                };
                q_buf[src_row * ATTN_H_DIM + dim] = 0.0;
                k_buf[src_row * ATTN_H_DIM + dim] = spike;
                q_buf[dst_row * ATTN_H_DIM + dim] = spike;
                k_buf[dst_row * ATTN_H_DIM + dim] = 0.0;
            }
            for dim in 0..ATTN_V_DIM {
                v_buf[src_row * ATTN_V_DIM + dim] = 2.0;
                v_buf[dst_row * ATTN_V_DIM + dim] = 0.0;
            }
        }

        PlantedCloneDataset { num_lines, q_buf, k_buf, v_buf, sink: 1.5, clone_src, clone_dst }
    }

    fn clone_needle_recall(out: &[f32], ref_out: &[f32], clone_dst: usize) -> f32 {
        let mut num = 0.0f32;
        let mut den = 1e-6f32;
        for off in 0..8 {
            let row = clone_dst + off;
            for dim in 0..ATTN_V_DIM {
                num += out[row * ATTN_V_DIM + dim];
                den += ref_out[row * ATTN_V_DIM + dim];
            }
        }
        (num / den).clamp(0.0, 1.0)
    }

    fn frobenius_dist_from_barycenter(mat: &[[f32; 4]; 4]) -> f64 {
        let mut acc = 0.0f64;
        for row in mat {
            for &val in row {
                let diff = (val as f64) - 0.25;
                acc += diff * diff;
            }
        }
        acc.sqrt()
    }

    #[test]
    fn mhc_lite_bvn_doubly_stochastic_and_cmhc_contraction() {
        let test_mats: [[[f32; 4]; 4]; 3] = [
            DEFAULT_MHC_AFFINITIES,
            [
                [12.0, -3.0, 0.0, 0.0],
                [-2.0, 11.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, -1.0],
                [0.0, 0.0, -4.0, 9.0],
            ],
            [
                [-5.0, 4.0, 2.0, 1.0],
                [3.0, -6.0, 5.0, 0.5],
                [1.5, 2.5, -4.0, 6.0],
                [7.0, 0.2, 1.1, -3.0],
            ],
        ];

        for raw in test_mats {
            let uncontracted = mhc_lite_bvn_4x4(raw, 0.0);
            let contracted = mhc_lite_bvn_4x4(raw, CMHC_GAMMA);

            assert_eq!(uncontracted.sinkhorn_iters, 0);
            assert_eq!(contracted.sinkhorn_iters, 0);
            assert!(
                uncontracted.max_ds_err <= 1e-6,
                "uncontracted ds err {} > 1e-6",
                uncontracted.max_ds_err
            );
            assert!(
                contracted.max_ds_err <= 1e-6,
                "contracted ds err {} > 1e-6",
                contracted.max_ds_err
            );

            // Verify cmHC gamma = 0.05 interior contraction strictly bounds entries and contracts distance by 0.95
            for row in 0..4 {
                for col in 0..4 {
                    let val = contracted.matrix[row][col];
                    assert!(
                        val >= 0.01249 && val <= 0.96251,
                        "entry ({row},{col})={val} escaped cmHC interior"
                    );
                }
            }
            let dist_raw = frobenius_dist_from_barycenter(&uncontracted.matrix);
            let dist_cmhc = frobenius_dist_from_barycenter(&contracted.matrix);
            let expected = dist_raw * (1.0 - CMHC_GAMMA as f64);
            assert!(
                (dist_cmhc - expected).abs() <= 1e-5,
                "cmHC contraction mismatch: got {dist_cmhc}, expected {expected}"
            );

            // Readout branch weights form a convex distribution summing to 1.0
            let w_sum: f32 = contracted.branch_weights.iter().sum();
            assert!((w_sum - 1.0).abs() <= 1e-6, "branch_weights sum={w_sum}");
        }
    }

    #[test]
    fn nsa_cobs_lsa_long_range_clone_recall_and_mli_reuse() {
        let mut engram = EngramTable::new();
        assert!(
            engram.memory_bytes() <= ENGRAM_MAX_BYTES,
            "EngramTable {} bytes exceeds 128 KiB",
            engram.memory_bytes()
        );

        // 1. Short context N = 512 dispatches to exact dense CH-V3b stride-264
        let ds_short = gen_planted_clone_dataset(512, 1001);
        let in_short = AttnInputs {
            num_lines: ds_short.num_lines,
            h_dim: ATTN_H_DIM,
            v_dim: ATTN_V_DIM,
            q_buf: &ds_short.q_buf,
            k_buf: &ds_short.k_buf,
            v_buf: &ds_short.v_buf,
            sink: ds_short.sink,
        };
        let mut cache_short = IndexCache::default();
        let out_short = run_adaptive_attention(&in_short, &mut engram, &mut cache_short);
        assert_eq!(out_short.mode, BranchMode::DenseV3bStride264);
        assert!(!out_short.mli_reused);
        assert!(out_short.mhc_ds_err <= 1e-6);

        // 2. Long context N = 2048 dispatches to O(N) NSA+COBS+LSA+Engram+mHC-lite
        let ds_long = gen_planted_clone_dataset(2048, 31415 + 2048);
        let in_long = AttnInputs {
            num_lines: ds_long.num_lines,
            h_dim: ATTN_H_DIM,
            v_dim: ATTN_V_DIM,
            q_buf: &ds_long.q_buf,
            k_buf: &ds_long.k_buf,
            v_buf: &ds_long.v_buf,
            sink: ds_long.sink,
        };
        let ref_dense = run_dense_v3b_stride264(&in_long);
        let mut cache_long = IndexCache::default();

        // Layer 1: populates IndexCache
        let layer1 = run_adaptive_attention(&in_long, &mut engram, &mut cache_long);
        assert_eq!(layer1.mode, BranchMode::NsaCobsLsaMhcLite);
        assert!(!layer1.mli_reused);
        assert!(layer1.mhc_ds_err <= 1e-6);
        assert!(layer1.pairs_evaluated < ds_long.num_lines * ds_long.num_lines / 4);

        let recall_l1 = clone_needle_recall(&layer1.out, &ref_dense, ds_long.clone_dst);
        assert_eq!(recall_l1, 1.0, "expected 1.000 planted clone recall at N=2048, got {recall_l1}");

        // Verify COBS exact block hit for every planted clone row
        let expected_blk = ds_long.clone_src / COBS_BLOCK_SIZE;
        for off in 0..8 {
            let row = ds_long.clone_dst + off;
            let picked = cache_long.picked_counts[row];
            assert!(
                cache_long.top_blocks[row][..picked].contains(&expected_blk),
                "clone row {row} missed COBS target block {expected_blk}"
            );
        }

        // Layer 2: reuses IndexCache (MLI) with strictly fewer evaluated pairs and identical recall
        let layer2 = run_adaptive_attention(&in_long, &mut engram, &mut cache_long);
        assert_eq!(layer2.mode, BranchMode::NsaCobsLsaMhcLite);
        assert!(layer2.mli_reused);
        assert_eq!(cache_long.reuse_hits, 1);
        assert!(layer2.pairs_evaluated < layer1.pairs_evaluated);
        let recall_l2 = clone_needle_recall(&layer2.out, &ref_dense, ds_long.clone_dst);
        assert_eq!(recall_l2, 1.0, "MLI Layer 2 clone recall dropped to {recall_l2}");

        // Verify Engram stays strictly within 128 KiB after population
        assert!(engram.memory_bytes() <= ENGRAM_MAX_BYTES);
    }

    #[test]
    fn ch_v5_block_piir_and_sparsity_crossover_guard() {
        // 1. Sparsity Crossover Guard assertions (including live v0.7.0 weights check)
        assert!(!should_use_5trit_lut(0.9675));
        assert!(!should_use_5trit_lut(0.85));
        assert!(should_use_5trit_lut(0.80));
        let real_zero_ratio = model::weight_zero_ratio(weights::v07_weights());
        assert!(real_zero_ratio > 0.95, "expected v0.7.0 >95% zeros, got {real_zero_ratio}");
        assert!(!should_use_5trit_lut(real_zero_ratio));

        // 2. Clean deep module (including PhantomData immunity) scores cleanly
        let clean_src = concat!(
            "pub struct Ledger<T> {\n",
            "    total_cents: i64,\n",
            "    pub marker: std::marker::PhantomData<T>,\n",
            "}\n",
            "impl<T> Ledger<T> {\n",
            "    pub fn apply_credit(&mut self, amount_cents: i64) -> i64 {\n",
            "        if amount_cents <= 0 {\n",
            "            return self.total_cents;\n",
            "        }\n",
            "        self.total_cents += amount_cents;\n",
            "        self.total_cents\n",
            "    }\n",
            "}\n",
        );
        let clean_block = analyze_block_module("src/ledger.rs", clean_src, None);
        assert_eq!(clean_block.field_leaks, 0);
        assert!(!clean_block.shallow_module);
        assert_eq!(clean_block.block_score, 1.0);

        // 3. Shallow getter-leaking module with high arity, cognitive complexity, cycle & tangle
        let sloppy_src = concat!(
            "pub struct Bag { pub alpha: i32, pub beta: i32 }\n",
            "impl Bag {\n",
            "    pub fn get_a(&self) -> &i32 { &self.alpha }\n",
            "    pub fn get_b(&self) -> &i32 { &self.beta }\n",
            "}\n",
            "pub fn w1(val: i32) -> i32 { helper_one(val) }\n",
            "pub fn w2(val: i32) -> i32 { helper_two(val) }\n",
            "pub fn w3(val: i32) -> i32 { helper_three(val) }\n",
            "pub fn w4(val: i32) -> i32 { helper_four(val) }\n",
            "pub fn w5(val: i32) -> i32 { helper_five(val) }\n",
            "pub fn w6(val: i32) -> i32 { helper_six(val) }\n",
            "pub fn tangled(a_flag: bool, b_flag: bool, c_flag: bool, p1: i32, p2: i32, p3: i32) -> i32 {\n",
            "    if a_flag && b_flag || c_flag {\n",
            "        if p1 > 0 {\n",
            "            if p2 > 0 {\n",
            "                if p3 > 0 {\n",
            "                    return p1 + p2 + p3;\n",
            "                }\n",
            "            }\n",
            "        }\n",
            "    }\n",
            "    0\n",
            "}\n",
        );
        let ego = EgoContext {
            fan_in: 8,
            fan_out: 8,
            churn_commits: 18,
            has_module_cycle: true,
            neighbor_dirs: 3,
            external_field_readers: 3,
        };
        let sloppy_block = analyze_block_module("src/sloppy.rs", sloppy_src, Some(&ego));
        assert!(sloppy_block.field_leaks > 0);
        assert!(sloppy_block.shallow_module);
        assert!(sloppy_block.module_cycle);
        assert!(sloppy_block.tangle);
        assert!(sloppy_block.max_params >= 6);
        assert!(sloppy_block.max_bool_params >= 3);
        assert!(sloppy_block.block_score < 0.50, "got {}", sloppy_block.block_score);
        for rule in ["field-leak", "shallow-module", "module-cycle", "tangle"] {
            assert!(
                sloppy_block.structural_rules.iter().any(|item| item == rule),
                "missing `{rule}` in {:?}",
                sloppy_block.structural_rules
            );
        }

        // 4. End-to-end HygieneScorer::score_content_multiscale preserves v0.7.0 score & emits CalibrationAudit JSON
        let scorer = HygieneScorer::new();
        let base_only = scorer.score_content("src/sloppy.rs", sloppy_src);
        let multi = scorer.score_content_multiscale("src/sloppy.rs", sloppy_src, Some(&ego));
        assert_eq!(multi.score, base_only.score, "canonical v0.7.0 score must never change");
        let ms_score = multi.multiscale_score.expect("multiscale_score must be populated");
        assert!(ms_score <= 0.48, "expected hard structural cap <= 0.48, got {ms_score}");

        let raw_json = multi.calibration_log.expect("calibration_log JSON must be populated");
        let parsed: CalibrationAudit = serde_json::from_str(&raw_json).expect("valid CalibrationAudit JSON");
        assert_eq!(parsed.branch_mode, "dense_v3b_stride264");
        assert_eq!(parsed.proof_tier, "tier0_vetoed");
        assert!(!parsed.use_5trit_lut);
        assert!(parsed.mhc_ds_err <= 1e-6);
        assert!(parsed.engram_bytes <= ENGRAM_MAX_BYTES);
        assert!(parsed.piir > 0.0);
        assert_eq!(parsed.final_score, ms_score);

        // 5. Long synthetic source (>512 lines) triggers NSA+COBS+LSA+mHC-lite + MLI IndexCache reuse end-to-end
        let mut long_src = String::new();
        for idx in 0..560 {
            long_src.push_str(&format!("pub const ITEM_{idx}: usize = {idx};\n"));
        }
        let long_res = scorer.score_content_multiscale("src/long_consts.rs", &long_src, None);
        let long_log: CalibrationAudit =
            serde_json::from_str(long_res.calibration_log.as_deref().unwrap()).unwrap();
        assert_eq!(long_log.branch_mode, "nsa_cobs_lsa_mhc_lite");
        assert!(long_log.mli_reused);
        assert!(long_log.mhc_ds_err <= 1e-6);
    }
}
