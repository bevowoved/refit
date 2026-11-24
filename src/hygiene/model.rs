use super::ops::{
    ActiveSpans, AttnWeights, CompactCells, RIBBON_GUARD, RibbonCtx, active_spans, axis_conv,
    encode_grid, excl_prefix, fuse_cells, linear_sparse, norm_sketch, ribbon_conv,
    ribbon_positions, sigmoid_f64, sink_attention, sparse_parts,
};
use super::weights::{HygieneWeights, v07_weights};
use crate::types::{HygieneResult, Squiggle};
use std::collections::HashMap;

const PRIOR_LINES: f64 = 16.0;
const PEAK_WINDOW: usize = 32;
const PEAK_MIN_GATE: f32 = 10.0;
const PEAK_PRIOR: f32 = 6.0;
const SPAN_CONT_WEIGHT: f32 = 0.25;
const SQUIGGLE_THRESHOLD: f32 = 0.55;

const SQUIGGLE_SPEC: [(&str, &str, f64); 6] = [
    ("deep_nesting", "Deep nesting pyramid; flatten with guard clauses or helpers", 0.42),
    ("dense_line", "Wide line, packed statements, call train, or nested ternary", 0.10),
    ("cramped_layout", "Unbroken statement wall or ragged off-grid indentation", 0.08),
    ("duplicate_block", "Duplicated or structurally cloned multi-line block", 0.58),
    ("repetitive_structure", "Repetitive parallel equality or branch ladder", 0.34),
    ("comment_echo", "Comment echoes adjacent code or leaves commented-out code", 0.18),
];

pub(super) fn score_v07(content: &str) -> HygieneResult {
    let params = v07_weights();
    let (grid, num_lines) = encode_grid(content);
    let spans = active_spans(&grid, num_lines);
    let (col_sums, conv3, conv7, mut sketch) = encode_lines(params, &grid, &spans, num_lines);

    let hidden = build_hidden(params, &col_sums, &conv3, &conv7, num_lines);
    let lmlp2 = params.mat("lmlp2_w");
    let stats = linear_sparse(&hidden, num_lines, lmlp2, Some(params.vec("lmlp2_b")), false);

    let sk_cols = params.conv("sketch_w").cols;
    norm_sketch(&mut sketch, num_lines, sk_cols);

    let dim = lmlp2.cols + sk_cols;
    let mut state = vec![0.0f32; num_lines * dim];
    for row in 0..num_lines {
        let (dst, ls, sk) = (row * dim, row * lmlp2.cols, row * sk_cols);
        state[dst..dst + lmlp2.cols].copy_from_slice(&stats[ls..ls + lmlp2.cols]);
        state[dst + lmlp2.cols..dst + dim].copy_from_slice(&sketch[sk..sk + sk_cols]);
    }

    run_context(params, &mut state, num_lines);

    let sq_l = linear_sparse(&state, num_lines, params.mat("sq_w"), Some(params.vec("sq_b")), false);
    let g_l = linear_sparse(
        &state,
        num_lines,
        params.mat("pool_gate_w"),
        Some(params.vec("pool_gate_b")),
        false,
    );
    let gate: Vec<f32> = g_l.iter().map(|&v| sigmoid_f64(v as f64) as f32).collect();

    let doc_v = pool_doc(&state, &gate, num_lines, dim);
    let doc_h = linear_sparse(&doc_v, 1, params.mat("doc1_w"), Some(params.vec("doc1_b")), true);
    let doc_l = linear_sparse(&doc_h, 1, params.mat("doc2_w"), Some(params.vec("doc2_b")), false);
    let score = (sigmoid_f64(doc_l[0] as f64) * 10000.0).round() / 10000.0;

    HygieneResult {
        score,
        squiggles: extract_squiggles(&sq_l, num_lines),
        cached: false,
        latency_us: 1,
    }
}

fn encode_lines(
    p: &HygieneWeights,
    grid: &[u8],
    spans: &ActiveSpans,
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if spans.total == 0 {
        return (
            vec![0.0; n * p.mat("col_sum_proj").cols],
            vec![0.0; n * p.conv("conv3_w").cols],
            vec![0.0; n * p.conv("conv7_w").cols],
            vec![0.0; n * p.conv("sketch_w").cols],
        );
    }

    let cells = fuse_cells(p, grid, spans);
    let hc = p.mat("col_fuse_w").cols;
    let mut cell_sums = vec![0.0f32; n * hc];
    for (cell, &row) in spans.cell_rows.iter().enumerate().take(spans.total) {
        let (dst, src) = (row * hc, cell * hc);
        for ch in 0..hc {
            cell_sums[dst + ch] += cells[src + ch];
        }
    }

    let col_sums = linear_sparse(&cell_sums, n, p.mat("col_sum_proj"), None, false);
    let (rib_pos, rib_len) = ribbon_positions(spans, RIBBON_GUARD);
    let mut cell_at = vec![-1i32; rib_len];
    for (cell, &rp) in rib_pos.iter().enumerate() {
        cell_at[rp] = cell as i32;
    }

    let fuse_b = p.vec("col_fuse_b");
    let pad_fill: Vec<f32> = fuse_b.iter().map(|&v| v.max(0.0)).collect();
    let sc = compact_cells(&cells, spans.total, hc);
    let rctx = RibbonCtx {
        rib_pos: &rib_pos,
        cell_at: &cell_at,
        pad_fill: &pad_fill,
        cells: &sc,
        spans,
        num_lines: n,
    };

    let c3 = ribbon_conv(&rctx, p.conv("conv3_w"), p.vec("conv3_b"));
    let c7 = ribbon_conv(&rctx, p.conv("conv7_w"), p.vec("conv7_b"));
    let sk = ribbon_conv(&rctx, p.conv("sketch_w"), p.vec("sketch_b"));
    (col_sums, c3, c7, sk)
}

fn compact_cells(cells: &[f32], total: usize, hc: usize) -> CompactCells {
    let (mut offsets, mut chans, mut vals) = (vec![0; total + 1], Vec::new(), Vec::new());
    for (cell, off) in offsets.iter_mut().enumerate().take(total) {
        *off = chans.len();
        let base = cell * hc;
        for ch in 0..hc {
            let v = cells[base + ch];
            if v != 0.0 {
                chans.push(ch);
                vals.push(v);
            }
        }
    }
    offsets[total] = chans.len();
    CompactCells { offsets, chans, vals }
}

fn build_hidden(
    p: &HygieneWeights,
    col_sums: &[f32],
    c3: &[f32],
    c7: &[f32],
    n: usize,
) -> Vec<f32> {
    let cc = p.mat("col_sum_proj").cols;
    let c3c = p.conv("conv3_w").cols;
    let c7c = p.conv("conv7_w").cols;
    let ml_proj = p.mat("ml_proj");

    let deltas = sparse_parts(&[(col_sums, cc), (c3, c3c), (c7, c7c)], n, ml_proj, None, false);
    let saw = linear_sparse(
        &excl_prefix(&deltas, ml_proj.cols),
        n,
        p.mat("ml_gate_w"),
        Some(p.vec("ml_gate_b")),
        true,
    );
    let paren = axis_conv(&deltas, n, p.conv("ml_paren_w"), p.vec("ml_paren_b"));

    sparse_parts(
        &[
            (col_sums, cc),
            (c3, c3c),
            (c7, c7c),
            (&saw, p.mat("ml_gate_w").cols),
            (&paren, p.conv("ml_paren_w").cols),
        ],
        n,
        p.mat("lmlp1_w"),
        Some(p.vec("lmlp1_b")),
        true,
    )
}

fn run_context(p: &HygieneWeights, state: &mut [f32], n: usize) {
    for layer in ["1", "2"] {
        let weights = AttnWeights {
            wq: p.mats.get(&format!("attn{layer}_wq")),
            wk: p.mats.get(&format!("attn{layer}_wk")),
            wv: p.mat(&format!("attn{layer}_wv")),
            wo: p.mat(&format!("attn{layer}_wo")),
            b_rel: p.vec(&format!("attn{layer}_brel")),
            b_sink: p.vec(&format!("attn{layer}_bsink")),
        };
        let attn = sink_attention(state, n, &weights);
        for (s, a) in state.iter_mut().zip(attn.iter()) {
            *s += *a;
        }

        let h = linear_sparse(
            state,
            n,
            p.mat(&format!("cmlp{layer}_w")),
            Some(p.vec(&format!("cmlp{layer}_b"))),
            true,
        );
        let o = linear_sparse(
            &h,
            n,
            p.mat(&format!("cmlp{layer}_2_w")),
            Some(p.vec(&format!("cmlp{layer}_2_b"))),
            false,
        );
        for (s, c) in state.iter_mut().zip(o.iter()) {
            *s += *c;
        }
    }
}

fn pool_doc(state: &[f32], gate: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut doc = vec![0.0f32; dim * 2];
    let mut mean_acc = vec![0.0f64; dim];
    let mut g_sum = PRIOR_LINES;

    for (row, &gv) in gate.iter().enumerate().take(n) {
        let g = gv as f64;
        g_sum += g;
        let base = row * dim;
        for ch in 0..dim {
            mean_acc[ch] += (state[base + ch] as f64) * g;
        }
    }
    for ch in 0..dim {
        doc[ch] = (mean_acc[ch] / g_sum) as f32;
    }

    let pos = build_pos(state, gate, n, dim);
    let mut pref_w = vec![0.0f32; (n + 1) * dim];
    let mut pref_g = vec![0.0f32; n + 1];
    for row in 0..n {
        pref_g[row + 1] = pref_g[row] + gate[row];
        for ch in 0..dim {
            pref_w[(row + 1) * dim + ch] = pref_w[row * dim + ch] + pos[row * dim + ch];
        }
    }

    let mut peak = vec![f32::NEG_INFINITY; dim];
    for hi in 1..=n {
        let lo = hi.saturating_sub(PEAK_WINDOW);
        let denom = (pref_g[hi] - pref_g[lo]).max(PEAK_MIN_GATE) + PEAK_PRIOR;
        for ch in 0..dim {
            let val = (pref_w[hi * dim + ch] - pref_w[lo * dim + ch]) / denom;
            peak[ch] = peak[ch].max(val);
        }
    }
    doc[dim..].copy_from_slice(&peak);
    doc
}

fn build_pos(state: &[f32], gate: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut pos = vec![0.0f32; n * dim];
    for row in 0..n {
        for ch in 0..dim {
            let raw = state[row * dim + ch].max(0.0) * gate[row];
            let p1 = if row >= 1 { state[(row - 1) * dim + ch].max(0.0) * gate[row - 1] } else { 0.0 };
            let p2 = if row >= 2 { state[(row - 2) * dim + ch].max(0.0) * gate[row - 2] } else { 0.0 };
            let ov = if n > 1 { raw.min(p1.max(p2)) } else { 0.0 };
            pos[row * dim + ch] = raw - (1.0 - SPAN_CONT_WEIGHT) * ov;
        }
    }
    pos
}

fn extract_squiggles(sq_logits: &[f32], n: usize) -> Vec<Squiggle> {
    let nk = SQUIGGLE_SPEC.len();
    let mut ranked = Vec::new();

    for row in 0..n {
        let base = row * nk;
        let (mut best_k, mut best_p) = (0usize, sigmoid_f64(sq_logits[base] as f64) as f32);
        for k in 1..nk {
            let p = sigmoid_f64(sq_logits[base + k] as f64) as f32;
            if p > best_p {
                best_p = p;
                best_k = k;
            }
        }
        if best_p < SQUIGGLE_THRESHOLD {
            continue;
        }

        let (kind, msg, sev) = SQUIGGLE_SPEC[best_k];
        let prob = ((best_p as f64) * 10000.0).round() / 10000.0;
        let sq = Squiggle { line: row + 1, kind: kind.into(), probability: prob, message: msg.into() };
        ranked.push((sev * prob, kind, sq));
    }
    rank_squiggles(ranked)
}

fn rank_squiggles(mut ranked: Vec<(f64, &'static str, Squiggle)>) -> Vec<Squiggle> {
    let cmp = |a: &(f64, &str, Squiggle), b: &(f64, &str, Squiggle)| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.line.cmp(&b.2.line))
    };
    ranked.sort_by(cmp);

    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let (mut selected, mut overflow) = (Vec::with_capacity(48), Vec::new());
    for item in ranked {
        let cnt = counts.entry(item.1).or_insert(0);
        if *cnt < 4 {
            *cnt += 1;
            selected.push(item);
        } else {
            overflow.push(item);
        }
    }

    let need = 48usize.saturating_sub(selected.len());
    selected.extend(overflow.into_iter().take(need));
    selected.sort_by(cmp);
    selected.truncate(48);
    selected.into_iter().map(|(_, _, s)| s).collect()
}
