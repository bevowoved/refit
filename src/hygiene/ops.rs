use super::weights::{HygieneWeights, Sparse2D, SparseConv};
use std::borrow::Cow;

pub(super) const MAX_COLS: usize = 128;
pub(super) const MAX_REL_DIST: isize = 32;
pub(super) const SKETCH_DIM: usize = 40;
pub(super) const RIBBON_GUARD: usize = 6;

pub(super) struct ActiveSpans {
    pub(super) span_lens: Vec<usize>,
    pub(super) line_starts: Vec<usize>,
    pub(super) line_ids: Vec<usize>,
    pub(super) within_col: Vec<usize>,
    pub(super) cell_rows: Vec<usize>,
    pub(super) total: usize,
}

pub(super) struct CompactCells {
    pub(super) offsets: Vec<usize>,
    pub(super) chans: Vec<usize>,
    pub(super) vals: Vec<f32>,
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
        '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e'
            | '\u{0085}' | '\u{2028}' | '\u{2029}'
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

pub(super) fn encode_grid(code: &str) -> (Vec<u8>, usize) {
    let expanded = expand_tabs(code);
    let lines = split_slices(&expanded);
    let num_lines = lines.len();
    let mut grid = vec![0u8; num_lines * MAX_COLS];

    for (row, line) in lines.iter().enumerate() {
        let base = row * MAX_COLS;
        for (col, ch) in line.chars().take(MAX_COLS).enumerate() {
            let cp = ch as u32;
            grid[base + col] = if cp > 127 { 127 } else if cp == 0 { 32 } else { cp as u8 };
        }
    }
    (grid, num_lines)
}

pub(super) fn active_spans(grid: &[u8], num_lines: usize) -> ActiveSpans {
    let (mut nonblank, mut span_lens) = (Vec::new(), Vec::new());

    for row in 0..num_lines {
        let mut len = MAX_COLS;
        while len > 0 && grid[row * MAX_COLS + len - 1] == 0 {
            len -= 1;
        }
        if len > 0 {
            nonblank.push(row);
            span_lens.push((len + RIBBON_GUARD).min(MAX_COLS));
        }
    }

    let mut line_starts = vec![0usize; nonblank.len()];
    let mut total = 0usize;
    for (idx, &len) in span_lens.iter().enumerate() {
        line_starts[idx] = total;
        total += len;
    }

    let (mut line_ids, mut within_col, mut cell_rows) =
        (vec![0; total], vec![0; total], vec![0; total]);
    let mut line = 0usize;
    for cell in 0..total {
        while line + 1 < nonblank.len() && cell >= line_starts[line + 1] {
            line += 1;
        }
        line_ids[cell] = line;
        within_col[cell] = cell - line_starts[line];
        cell_rows[cell] = nonblank[line];
    }

    ActiveSpans { span_lens, line_starts, line_ids, within_col, cell_rows, total }
}

pub(super) fn ribbon_positions(spans: &ActiveSpans, guard: usize) -> (Vec<usize>, usize) {
    let mut starts = vec![0usize; spans.span_lens.len() + 1];
    starts[0] = guard;
    let mut running = 0usize;

    for (idx, &len) in spans.span_lens.iter().enumerate() {
        running += len + guard;
        starts[idx + 1] = running;
    }

    let mut pos = vec![0usize; spans.total];
    for cell in 0..spans.total {
        pos[cell] = starts[spans.line_ids[cell]] + spans.within_col[cell];
    }
    (pos, starts[starts.len() - 1] + guard)
}

#[inline]
pub(super) fn accumulate_row(src: &[f32], base: usize, w: &Sparse2D, acc: &mut [f32]) {
    for &in_idx in &w.active_rows {
        let in_val = src[base + in_idx];
        if in_val == 0.0 {
            continue;
        }
        for ptr in w.row_offsets[in_idx]..w.row_offsets[in_idx + 1] {
            acc[w.col_indices[ptr]] += in_val * w.values[ptr];
        }
    }
}

struct PartSlice<'a> {
    src: &'a [f32],
    dim: usize,
    in_off: usize,
    start: usize,
    end: usize,
}

fn plan_slices<'a>(parts: &[(&'a [f32], usize)], active: &[usize]) -> Vec<PartSlice<'a>> {
    let mut slices = Vec::with_capacity(parts.len());
    let (mut in_off, mut cursor) = (0usize, 0usize);

    for &(src, dim) in parts {
        let next_off = in_off + dim;
        let start = cursor;
        while cursor < active.len() && active[cursor] < next_off {
            cursor += 1;
        }
        if cursor > start {
            slices.push(PartSlice { src, dim, in_off, start, end: cursor });
        }
        in_off = next_off;
    }
    slices
}

#[inline]
fn accum_slice(sl: &PartSlice<'_>, row: usize, w: &Sparse2D, acc: &mut [f32]) {
    let row_base = row * sl.dim;
    for &in_idx in &w.active_rows[sl.start..sl.end] {
        let in_val = sl.src[row_base + (in_idx - sl.in_off)];
        if in_val == 0.0 {
            continue;
        }
        for ptr in w.row_offsets[in_idx]..w.row_offsets[in_idx + 1] {
            acc[w.col_indices[ptr]] += in_val * w.values[ptr];
        }
    }
}

pub(super) fn sparse_parts(
    parts: &[(&[f32], usize)],
    num_rows: usize,
    w: &Sparse2D,
    bias: Option<&[f32]>,
    relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_rows * w.cols];
    let mut acc = vec![0.0f32; w.cols];
    let slices = plan_slices(parts, &w.active_rows);

    for row in 0..num_rows {
        match bias {
            Some(b) => acc.copy_from_slice(b),
            None => acc.fill(0.0),
        }
        for sl in &slices {
            accum_slice(sl, row, w, &mut acc);
        }
        if relu {
            for v in &mut acc {
                *v = v.max(0.0);
            }
        }
        out[row * w.cols..(row + 1) * w.cols].copy_from_slice(&acc);
    }
    out
}

pub(super) fn linear_sparse(
    input: &[f32],
    num_rows: usize,
    w: &Sparse2D,
    bias: Option<&[f32]>,
    relu: bool,
) -> Vec<f32> {
    sparse_parts(&[(input, w.rows)], num_rows, w, bias, relu)
}

fn line_prefix(
    parts: &[(&[f32], usize)],
    n_cells: usize,
    proj: &Sparse2D,
    spans: &ActiveSpans,
    exclusive: bool,
) -> Vec<f32> {
    let projected = sparse_parts(parts, n_cells, proj, None, false);
    let dim = proj.cols;
    let mut csum = vec![0.0f32; projected.len()];

    if projected.len() >= dim {
        csum[..dim].copy_from_slice(&projected[..dim]);
        for idx in dim..projected.len() {
            csum[idx] = csum[idx - dim] + projected[idx];
        }
    }

    let mut out = vec![0.0f32; projected.len()];
    for cell in 0..spans.total {
        let start = spans.line_starts[spans.line_ids[cell]];
        for ch in 0..dim {
            let base = if start > 0 { csum[(start - 1) * dim + ch] } else { 0.0 };
            let val = csum[cell * dim + ch] - base;
            out[cell * dim + ch] = if exclusive { val - projected[cell * dim + ch] } else { val };
        }
    }
    out
}

pub(super) fn fuse_cells(params: &HygieneWeights, grid: &[u8], spans: &ActiveSpans) -> Vec<f32> {
    let n_cells = spans.total;
    let (tok_emb, col_emb) = (params.mat("tok_emb"), params.mat("col_emb"));
    let emb_dim = tok_emb.cols;

    let mut emb = vec![0.0f32; n_cells * emb_dim];
    for cell in 0..n_cells {
        let col = spans.within_col[cell];
        let code = grid[spans.cell_rows[cell] * MAX_COLS + col] as usize;
        for ch in 0..emb_dim {
            emb[cell * emb_dim + ch] =
                tok_emb.data[code * emb_dim + ch] + col_emb.data[col * emb_dim + ch];
        }
    }

    let pre_w = params.conv("pre_conv3_w");
    let lex = pre_conv(&emb, emb_dim, spans, pre_w, params.vec("pre_conv3_b"));

    let p1 = params.mat("pref1_proj");
    let pref1 = line_prefix(&[(&emb, emb_dim), (&lex, pre_w.cols)], n_cells, p1, spans, true);

    let g1 = params.mat("gate1_w");
    let gates = sparse_parts(
        &[(&emb, emb_dim), (&lex, pre_w.cols), (&pref1, p1.cols)],
        n_cells,
        g1,
        Some(params.vec("gate1_b")),
        true,
    );

    let p2 = params.mat("pref2_proj");
    let pref2 = line_prefix(&[(&gates, g1.cols)], n_cells, p2, spans, false);

    sparse_parts(
        &[(&emb, emb_dim), (&lex, pre_w.cols), (&gates, g1.cols), (&pref2, p2.cols)],
        n_cells,
        params.mat("col_fuse_w"),
        Some(params.vec("col_fuse_b")),
        true,
    )
}

fn pre_conv(
    emb: &[f32],
    emb_dim: usize,
    spans: &ActiveSpans,
    pre_w: &SparseConv,
    pre_b: &[f32],
) -> Vec<f32> {
    let n_cells = spans.total;
    let (rib_pos, rib_len) = ribbon_positions(spans, 1);
    let mut rib = vec![0.0f32; rib_len * emb_dim];

    for cell in 0..n_cells {
        let dst = rib_pos[cell] * emb_dim;
        rib[dst..dst + emb_dim].copy_from_slice(&emb[cell * emb_dim..(cell + 1) * emb_dim]);
    }

    let mut lex = vec![0.0f32; n_cells * pre_w.cols];
    let mut acc = vec![0.0f32; pre_w.cols];
    for cell in 0..n_cells {
        acc.copy_from_slice(pre_b);
        for (tap, weights) in pre_w.taps.iter().enumerate().take(pre_w.k_size) {
            accumulate_row(&rib, (rib_pos[cell] - 1 + tap) * emb_dim, weights, &mut acc);
        }
        for v in &mut acc {
            *v = v.max(0.0);
        }
        lex[cell * pre_w.cols..(cell + 1) * pre_w.cols].copy_from_slice(&acc);
    }
    lex
}

fn pad_taps(pad_fill: &[f32], conv_w: &SparseConv) -> Vec<(Vec<usize>, Vec<f32>)> {
    let mut out = Vec::with_capacity(conv_w.k_size);
    for tap in &conv_w.taps {
        let mut prod = vec![0.0f32; conv_w.cols];
        accumulate_row(pad_fill, 0, tap, &mut prod);
        let (mut cols, mut vals) = (Vec::new(), Vec::new());
        for (c, &v) in prod.iter().enumerate() {
            if v != 0.0 {
                cols.push(c);
                vals.push(v);
            }
        }
        out.push((cols, vals));
    }
    out
}

#[inline]
fn accum_tap(
    src_cell: i32,
    tap: &Sparse2D,
    pad: &(Vec<usize>, Vec<f32>),
    cells: &CompactCells,
    acc: &mut [f32],
) {
    if src_cell < 0 {
        for (&c, &v) in pad.0.iter().zip(pad.1.iter()) {
            acc[c] += v;
        }
        return;
    }

    let sc = src_cell as usize;
    for idx in cells.offsets[sc]..cells.offsets[sc + 1] {
        let (ch, iv) = (cells.chans[idx], cells.vals[idx]);
        for ptr in tap.row_offsets[ch]..tap.row_offsets[ch + 1] {
            acc[tap.col_indices[ptr]] += iv * tap.values[ptr];
        }
    }
}

pub(super) struct RibbonCtx<'a> {
    pub(super) rib_pos: &'a [usize],
    pub(super) cell_at: &'a [i32],
    pub(super) pad_fill: &'a [f32],
    pub(super) cells: &'a CompactCells,
    pub(super) spans: &'a ActiveSpans,
    pub(super) num_lines: usize,
}

pub(super) fn ribbon_conv(ctx: &RibbonCtx<'_>, conv_w: &SparseConv, bias: &[f32]) -> Vec<f32> {
    let out_dim = conv_w.cols;
    let mut out = vec![0.0f32; ctx.num_lines * out_dim];
    if conv_w.non_zero_count == 0 {
        return out;
    }

    let pads = pad_taps(ctx.pad_fill, conv_w);
    let pad = (conv_w.k_size - 1) >> 1;
    let mut acc = vec![0.0f32; out_dim];

    for cell in 0..ctx.spans.total {
        acc.copy_from_slice(bias);
        let rpos = ctx.rib_pos[cell];
        for tap in 0..conv_w.k_size {
            let tm = &conv_w.taps[tap];
            if tm.non_zero_count > 0 {
                accum_tap(ctx.cell_at[rpos - pad + tap], tm, &pads[tap], ctx.cells, &mut acc);
            }
        }
        let dst = ctx.spans.cell_rows[cell] * out_dim;
        for ch in 0..out_dim {
            if acc[ch] > 0.0 {
                out[dst + ch] += acc[ch];
            }
        }
    }
    out
}

pub(super) fn excl_prefix(deltas: &[f32], dim: usize) -> Vec<f32> {
    let mut running = vec![0.0f32; deltas.len()];
    let mut out = vec![0.0f32; deltas.len()];
    for idx in 0..deltas.len() {
        running[idx] = if idx >= dim { running[idx - dim] } else { 0.0 } + deltas[idx];
        out[idx] = running[idx] - deltas[idx];
    }
    out
}

pub(super) fn axis_conv(
    deltas: &[f32],
    num_lines: usize,
    conv_w: &SparseConv,
    bias: &[f32],
) -> Vec<f32> {
    let pad = ((conv_w.k_size - 1) >> 1) as isize;
    let mut out = vec![0.0f32; num_lines * conv_w.cols];
    let mut acc = vec![0.0f32; conv_w.cols];

    for row in 0..num_lines {
        acc.copy_from_slice(bias);
        let lo = (pad - row as isize).max(0) as usize;
        let hi = (num_lines as isize - row as isize + pad).min(conv_w.k_size as isize) as usize;
        for tap in lo..hi {
            let src = (row as isize + tap as isize - pad) as usize;
            accumulate_row(deltas, src * conv_w.rows, &conv_w.taps[tap], &mut acc);
        }
        for v in &mut acc {
            *v = v.max(0.0);
        }
        out[row * conv_w.cols..(row + 1) * conv_w.cols].copy_from_slice(&acc);
    }
    out
}

pub(super) fn norm_sketch(sketch: &mut [f32], num_lines: usize, total_sketch: usize) {
    for v in sketch.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let total = num_lines * total_sketch;
    let mut base = 0usize;
    while base < total {
        let end = (base + SKETCH_DIM).min(total);
        let mut norm_sq = 0.0f32;
        for &v in &sketch[base..end] {
            norm_sq += v * v;
        }
        let norm = (norm_sq + 1e-6f32).sqrt();
        for v in &mut sketch[base..end] {
            *v /= norm;
        }
        base += SKETCH_DIM;
    }
}

pub(super) struct AttnWeights<'a> {
    pub(super) wq: Option<&'a Sparse2D>,
    pub(super) wk: Option<&'a Sparse2D>,
    pub(super) wv: &'a Sparse2D,
    pub(super) wo: &'a Sparse2D,
    pub(super) b_rel: &'a [f32],
    pub(super) b_sink: &'a [f32],
}

struct HeadSpec<'a> {
    qk_off: usize,
    v_off: usize,
    rel: &'a [f32],
    sink: f64,
    banded: bool,
    head_qk: bool,
}

struct AttnBuf {
    logits: Vec<f64>,
    keys: Vec<usize>,
    merged: Vec<f32>,
}

struct HeadView<'a> {
    q: Option<&'a [f32]>,
    k: Option<&'a [f32]>,
    v: &'a [f32],
    n: usize,
    qk_tot: usize,
    h_dim: usize,
    v_tot: usize,
    v_dim: usize,
    inv_s: f32,
}

pub(super) fn sink_attention(state: &[f32], n: usize, w: &AttnWeights<'_>) -> Vec<f32> {
    let n_heads = w.b_sink.len();
    let has_qk = w.wq.is_some_and(|m| m.non_zero_count > 0)
        && w.wk.is_some_and(|m| m.non_zero_count > 0);
    let qk_tot = w.wq.map_or(0, |m| m.cols);
    let h_dim = if qk_tot > 0 { qk_tot / n_heads } else { 1 };
    let (v_tot, v_dim) = (w.wv.cols, w.wv.cols / n_heads);

    let q_buf = has_qk.then(|| linear_sparse(state, n, w.wq.unwrap(), None, false));
    let k_buf = has_qk.then(|| linear_sparse(state, n, w.wk.unwrap(), None, false));
    let v_proj = linear_sparse(state, n, w.wv, None, false);

    let view = HeadView {
        q: q_buf.as_deref(),
        k: k_buf.as_deref(),
        v: &v_proj,
        n,
        qk_tot,
        h_dim,
        v_tot,
        v_dim,
        inv_s: 1.0f32 / (h_dim as f32).sqrt(),
    };

    let mut buf = AttnBuf {
        logits: vec![0.0f64; n],
        keys: vec![0; n],
        merged: vec![0.0f32; n * v_tot],
    };
    let span = (2 * MAX_REL_DIST + 1) as usize;

    for h in 0..n_heads {
        let rel = &w.b_rel[h * span..(h + 1) * span];
        let (qk_off, v_off) = (h * h_dim, h * v_dim);
        let head_qk = has_qk
            && w.wq.is_some_and(|m| {
                m.col_indices.iter().any(|&c| c >= qk_off && c < qk_off + h_dim)
            });
        let banded = rel[0] <= -1e3 && rel[span - 1] <= -1e3;
        let sink = w.b_sink[h] as f64;

        if !banded && !head_qk && rel.iter().all(|&v| v == 0.0) {
            view.uniform_head(v_off, sink, &mut buf.merged);
        } else {
            let spec = HeadSpec { qk_off, v_off, rel, sink, banded, head_qk };
            view.attend_head(&spec, &mut buf);
        }
    }
    linear_sparse(&buf.merged, n, w.wo, None, false)
}

impl HeadView<'_> {
    fn uniform_head(&self, v_off: usize, sink: f64, merged: &mut [f32]) {
        let w = (1.0f64 / (self.n as f64 + sink.exp())) as f32;
        let mut sum = vec![0.0f64; self.v_dim];

        for row in 0..self.n {
            let base = row * self.v_tot + v_off;
            for d in 0..self.v_dim {
                sum[d] += self.v[base + d] as f64;
            }
        }
        for row in 0..self.n {
            let base = row * self.v_tot + v_off;
            for d in 0..self.v_dim {
                merged[base + d] = (sum[d] * w as f64) as f32;
            }
        }
    }

    fn attend_head(&self, spec: &HeadSpec<'_>, buf: &mut AttnBuf) {
        for q in 0..self.n {
            let (ks, ke) = if spec.banded {
                (
                    (q as isize - MAX_REL_DIST + 1).max(0) as usize,
                    (q as isize + MAX_REL_DIST).min(self.n as isize) as usize,
                )
            } else {
                (0, self.n)
            };

            let (mut count, mut max_l) = (0usize, spec.sink);
            for k in ks..ke {
                let d = (k as isize - q as isize).clamp(-MAX_REL_DIST, MAX_REL_DIST);
                let rb = spec.rel[(d + MAX_REL_DIST) as usize] as f64;
                if spec.banded && rb <= -1e3 {
                    continue;
                }
                let qk = if spec.head_qk { self.dot_head(q, k, spec.qk_off) } else { 0.0 };
                let logit = rb + qk;
                buf.keys[count] = k;
                buf.logits[count] = logit;
                max_l = max_l.max(logit);
                count += 1;
            }

            let mut exp_sum = (spec.sink - max_l).exp();
            for l in &mut buf.logits[..count] {
                *l = (*l - max_l).exp();
                exp_sum += *l;
            }

            let dst = q * self.v_tot + spec.v_off;
            for idx in 0..count {
                let aw = buf.logits[idx] / exp_sum;
                let vb = buf.keys[idx] * self.v_tot + spec.v_off;
                for d in 0..self.v_dim {
                    buf.merged[dst + d] += (aw * (self.v[vb + d] as f64)) as f32;
                }
            }
        }
    }

    #[inline]
    fn dot_head(&self, q: usize, k: usize, qk_off: usize) -> f64 {
        let (qp, kp) = (self.q.unwrap(), self.k.unwrap());
        let (qb, kb) = (q * self.qk_tot + qk_off, k * self.qk_tot + qk_off);
        let mut dot = 0.0f64;
        for d in 0..self.h_dim {
            dot += (qp[qb + d] as f64) * (kp[kb + d] as f64);
        }
        dot * (self.inv_s as f64)
    }
}

#[inline]
pub(super) fn sigmoid_f64(val: f64) -> f64 {
    1.0 / (1.0 + (-val.clamp(-30.0, 30.0)).exp())
}
