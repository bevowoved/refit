use crate::lex::ident_byte;
use crate::source::{dir_of, read_text, stem_of};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Runtime,
    TypeOnly,
    Dynamic,
    ModDecl,
    TestImport,
}

impl EdgeKind {
    #[inline]
    pub fn is_runtime_scc(self) -> bool {
        matches!(self, Self::Runtime | Self::Dynamic | Self::ModDecl)
    }
}

/// Compact Compressed Sparse Row directed graph using flat `u32` arrays.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
}

impl CsrGraph {
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut sorted: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|&(u, v)| (u as usize) < n && (v as usize) < n && u != v)
            .collect();
        sorted.sort_unstable();
        sorted.dedup();

        let mut deg = vec![0u32; n];
        for &(u, _) in &sorted {
            deg[u as usize] += 1;
        }
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + deg[i];
        }
        let mut cur = offsets.clone();
        let mut targets = vec![0u32; sorted.len()];
        for &(u, v) in &sorted {
            let pos = cur[u as usize] as usize;
            targets[pos] = v;
            cur[u as usize] += 1;
        }
        Self { n, offsets, targets }
    }

    /// David Pearce (2016) space-efficient iterative CSR strongly connected components.
    /// Reuses a single `rindex: Vec<u32>` for both discovery index and final component ID
    /// with an explicit heap stack (immune to recursion stack overflow).
    pub fn pearce_scc(&self) -> Vec<Vec<u32>> {
        let mut state = PearceState::new(self);
        for start in 0..self.n {
            if state.rindex[start] == 0 {
                state.visit_from(start);
            }
        }
        state.comps
    }
}

struct PearceState<'a> {
    graph: &'a CsrGraph,
    rindex: Vec<u32>,
    stack: Vec<u32>,
    call_stack: Vec<(u32, u32, bool)>,
    next_idx: u32,
    comp_id: u32,
    comps: Vec<Vec<u32>>,
}

impl<'a> PearceState<'a> {
    fn new(graph: &'a CsrGraph) -> Self {
        let n = graph.n;
        Self {
            graph,
            rindex: vec![0u32; n],
            stack: Vec::with_capacity(n.min(1024)),
            call_stack: Vec::with_capacity(n.min(256)),
            next_idx: 1,
            comp_id: (n as u32).saturating_add(1),
            comps: Vec::new(),
        }
    }

    fn visit_from(&mut self, start: usize) {
        self.rindex[start] = self.next_idx;
        self.next_idx += 1;
        self.call_stack.push((start as u32, self.graph.offsets[start], true));

        while let Some((node, edge_idx, is_root)) = self.call_stack.pop() {
            if let Some(finished_root) = self.advance_node(node, edge_idx, is_root) {
                self.finish_node(node, finished_root);
            }
        }
    }

    fn advance_node(&mut self, node: u32, mut edge_idx: u32, mut is_root: bool) -> Option<bool> {
        let node_u = node as usize;
        let end = self.graph.offsets[node_u + 1];
        while edge_idx < end {
            let neighbor = self.graph.targets[edge_idx as usize] as usize;
            edge_idx += 1;
            if self.rindex[neighbor] == 0 {
                self.call_stack.push((node, edge_idx, is_root));
                self.rindex[neighbor] = self.next_idx;
                self.next_idx += 1;
                self.call_stack.push((neighbor as u32, self.graph.offsets[neighbor], true));
                return None;
            }
            if self.rindex[neighbor] < self.rindex[node_u] {
                self.rindex[node_u] = self.rindex[neighbor];
                is_root = false;
            }
        }
        Some(is_root)
    }

    fn finish_node(&mut self, node: u32, is_root: bool) {
        let node_u = node as usize;
        if is_root {
            self.emit_component(node);
        } else {
            self.stack.push(node);
        }
        if let Some(parent) = self.call_stack.last_mut() {
            let parent_u = parent.0 as usize;
            if self.rindex[node_u] < self.rindex[parent_u] {
                self.rindex[parent_u] = self.rindex[node_u];
                parent.2 = false;
            }
        }
    }

    fn emit_component(&mut self, root: u32) {
        let root_u = root as usize;
        let mut comp = vec![root];
        let root_rindex = self.rindex[root_u];
        while let Some(&top) = self.stack.last() {
            if self.rindex[top as usize] < root_rindex {
                break;
            }
            self.stack.pop();
            self.rindex[top as usize] = self.comp_id;
            comp.push(top);
        }
        self.rindex[root_u] = self.comp_id;
        self.comp_id += 1;
        self.comps.push(comp);
    }
}

/// Compact per-module deep-module metrics extracted during lexing/parsing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleMetrics {
    pub loc: usize,
    pub exported_symbols: usize,
    pub total_fns: usize,
    pub passthrough_fns: usize,
    pub pub_fields: Vec<String>,
    pub field_reads: Vec<String>,
}

fn append_cycle_findings(
    source_rels: &[String],
    csr_edges: &[(u32, u32)],
    findings: &mut Vec<String>,
) {
    let csr = CsrGraph::from_edges(source_rels.len(), csr_edges);
    let mut cycles: Vec<Vec<&str>> = csr
        .pearce_scc()
        .into_iter()
        .filter(|comp| comp.len() >= 2)
        .map(|comp| {
            let mut files: Vec<&str> =
                comp.into_iter().map(|id| source_rels[id as usize].as_str()).collect();
            files.sort_unstable();
            files
        })
        .collect();
    cycles.sort();
    for cyc in cycles.into_iter().take(12) {
        let preview = cyc.iter().take(4).copied().collect::<Vec<_>>().join(" <-> ");
        findings.push(format!("[module-cycle] {preview} ({} files)", cyc.len()));
    }
}

fn append_tangle_findings(
    source_rels: &[String],
    fan_in: &[usize],
    fan_out: &[usize],
    neighbor_dirs: &[FxHashSet<String>],
    findings: &mut Vec<String>,
) {
    for (idx, rel) in source_rels.iter().enumerate() {
        let fi = fan_in[idx];
        let fo = fan_out[idx];
        if fi < 6 || fo < 6 || neighbor_dirs[idx].len() < 2 {
            continue;
        }
        let instability = (fo as f64) / ((fi + fo) as f64);
        if (0.35..=0.65).contains(&instability) {
            findings.push(format!(
                "[tangle] {rel}: fan_in={fi}, fan_out={fo}, instability={instability:.2} across {} dirs",
                neighbor_dirs[idx].len()
            ));
        }
    }
}

fn build_field_readers<'a>(
    source_rels: &[String],
    metrics: &'a BTreeMap<String, ModuleMetrics>,
) -> FxHashMap<&'a str, FxHashSet<usize>> {
    let mut field_readers: FxHashMap<&'a str, FxHashSet<usize>> = FxHashMap::default();
    for (idx, rel) in source_rels.iter().enumerate() {
        let Some(metric) = metrics.get(rel) else { continue };
        for field in &metric.field_reads {
            field_readers.entry(field.as_str()).or_default().insert(idx);
        }
    }
    field_readers
}

fn append_field_leak_findings(
    source_rels: &[String],
    metrics: &BTreeMap<String, ModuleMetrics>,
    findings: &mut Vec<String>,
) {
    let field_readers = build_field_readers(source_rels, metrics);
    let mut leak_count = 0usize;
    for (idx, rel) in source_rels.iter().enumerate() {
        if leak_count >= 12 {
            break;
        }
        let Some(metric) = metrics.get(rel) else { continue };
        for pub_field in &metric.pub_fields {
            let ext_readers = field_readers
                .get(pub_field.as_str())
                .map(|set| set.iter().filter(|&&reader| reader != idx).count())
                .unwrap_or(0);
            if ext_readers >= 2 {
                findings.push(format!(
                    "[field-leak] {rel}: public field `.{pub_field}` accessed directly by {ext_readers} external modules"
                ));
                leak_count += 1;
                break;
            }
        }
    }
}

fn append_shallow_module_findings(
    source_rels: &[String],
    metrics: &BTreeMap<String, ModuleMetrics>,
    findings: &mut Vec<String>,
) {
    let mut shallow_count = 0usize;
    for rel in source_rels {
        if shallow_count >= 12 {
            break;
        }
        let Some(metric) = metrics.get(rel) else { continue };
        if metric.exported_symbols < 4 || metric.total_fns < 3 || metric.loc == 0 {
            continue;
        }
        let surface_density = (metric.exported_symbols as f64) / (metric.loc as f64);
        let passthrough_ratio = (metric.passthrough_fns as f64) / (metric.total_fns as f64);
        if surface_density > 0.20 && passthrough_ratio >= 0.60 {
            findings.push(format!(
                "[shallow-module] {rel}: {} exports in {} LOC ({:.0}% passthrough wrappers)",
                metric.exported_symbols,
                metric.loc,
                passthrough_ratio * 100.0
            ));
            shallow_count += 1;
        }
    }
}

/// Evaluate `deslop-please` 4 deep-module architectural rules over runtime CSR edges & metrics:
/// 1. `module-cycle`: runtime SCC with >= 2 files (TypeOnly/TestImport excluded)
/// 2. `tangle`: toxic middle-band instability (`0.35 <= I <= 0.65`) with high cross-dir fan-in/out
/// 3. `field-leak`: public struct/class fields read directly by >= 2 external modules
/// 4. `shallow-module`: high export-to-LOC density with >= 60% passthrough wrappers
pub fn deep_module_findings(
    source_rels: &[String],
    runtime_edges: &[(String, String)],
    metrics: &BTreeMap<String, ModuleMetrics>,
) -> Vec<String> {
    let mut findings = Vec::new();
    let mut id_of: FxHashMap<&str, u32> =
        FxHashMap::with_capacity_and_hasher(source_rels.len(), Default::default());
    for (idx, rel) in source_rels.iter().enumerate() {
        id_of.insert(rel.as_str(), idx as u32);
    }

    let mut csr_edges = Vec::with_capacity(runtime_edges.len());
    let mut fan_in = vec![0usize; source_rels.len()];
    let mut fan_out = vec![0usize; source_rels.len()];
    let mut neighbor_dirs: Vec<FxHashSet<String>> = vec![FxHashSet::default(); source_rels.len()];

    for (src, dst) in runtime_edges {
        if let (Some(&src_id), Some(&dst_id)) = (id_of.get(src.as_str()), id_of.get(dst.as_str()))
            && src_id != dst_id
        {
            csr_edges.push((src_id, dst_id));
            fan_out[src_id as usize] += 1;
            fan_in[dst_id as usize] += 1;
            neighbor_dirs[src_id as usize].insert(dir_of(dst));
            neighbor_dirs[dst_id as usize].insert(dir_of(src));
        }
    }

    append_cycle_findings(source_rels, &csr_edges, &mut findings);
    append_tangle_findings(source_rels, &fan_in, &fan_out, &neighbor_dirs, &mut findings);
    append_field_leak_findings(source_rels, metrics, &mut findings);
    append_shallow_module_findings(source_rels, metrics, &mut findings);

    findings.truncate(40);
    findings
}

fn record_stem_hits(
    base: &str,
    tidx: usize,
    stem_to_srcs: &FxHashMap<&str, Vec<usize>>,
    matched_by_src: &mut [BTreeSet<usize>],
) {
    if let Some(srcs) = stem_to_srcs.get(base) {
        for &sidx in srcs {
            matched_by_src[sidx].insert(tidx);
        }
    }
}

fn match_single_test_stem(
    test_stem: &str,
    tidx: usize,
    stem_to_srcs: &FxHashMap<&str, Vec<usize>>,
    matched_by_src: &mut [BTreeSet<usize>],
) {
    if let Some(base) = test_stem.strip_prefix("test_") {
        record_stem_hits(base, tidx, stem_to_srcs, matched_by_src);
    }
    for suffix in ["_test", ".test", ".spec"] {
        if let Some(base) = test_stem.strip_suffix(suffix) {
            record_stem_hits(base, tidx, stem_to_srcs, matched_by_src);
        }
    }
    for (pos, byte) in test_stem.bytes().enumerate() {
        if byte == b'_' && pos + 1 < test_stem.len() {
            record_stem_hits(&test_stem[pos + 1..], tidx, stem_to_srcs, matched_by_src);
        }
    }
}

fn collect_unmatched_lookups<'a>(
    source_files: &'a [String],
    pymod_to_src: &'a FxHashMap<String, Vec<usize>>,
    matched_by_src: &[BTreeSet<usize>],
) -> (FxHashMap<&'a str, Vec<usize>>, FxHashMap<&'a str, Vec<usize>>) {
    let mut needed_stems: FxHashMap<&'a str, Vec<usize>> = FxHashMap::default();
    let mut needed_pymods: FxHashMap<&'a str, Vec<usize>> = FxHashMap::default();
    for (sidx, src) in source_files.iter().enumerate() {
        if !matched_by_src[sidx].is_empty() {
            continue;
        }
        let stem = stem_of(Path::new(src));
        if stem.len() >= 3 {
            needed_stems.entry(stem).or_default().push(sidx);
        }
    }
    for (pymod, srcs) in pymod_to_src {
        let unmatched: Vec<usize> = srcs
            .iter()
            .copied()
            .filter(|&sidx| {
                matched_by_src[sidx].is_empty() && stem_of(Path::new(&source_files[sidx])).len() >= 3
            })
            .collect();
        if !unmatched.is_empty() {
            needed_pymods.insert(pymod.as_str(), unmatched);
        }
    }
    (needed_stems, needed_pymods)
}

fn match_test_bodies_pass(
    repo_root: &Path,
    test_files: &[String],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    needed_pymods: &FxHashMap<&str, Vec<usize>>,
    matched_by_src: &mut [BTreeSet<usize>],
) {
    if needed_stems.is_empty() && needed_pymods.is_empty() {
        return;
    }
    let scan_one = |(tidx, tf): (usize, &String)| -> (usize, Vec<usize>) {
        let Some(txt) = read_text(repo_root, tf, preloaded) else {
            return (tidx, Vec::new());
        };
        let hits = scan_test_body_hits(&txt, needed_stems, needed_pymods);
        (tidx, hits)
    };

    let postings: Vec<(usize, Vec<usize>)> = if crate::platform::budget().is_constrained() {
        test_files.iter().enumerate().map(scan_one).collect()
    } else {
        test_files.par_iter().enumerate().map(scan_one).collect()
    };

    for (tidx, hits) in postings {
        for sidx in hits {
            matched_by_src[sidx].insert(tidx);
        }
    }
}

/// Inverted-posting test matcher replacing the O(|S|*|T|) body scan with zero retained test bodies.
pub fn match_tests_inverted(
    repo_root: &Path,
    source_files: &[String],
    test_files: &[String],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> HashMap<String, Vec<String>> {
    let mut stem_to_srcs: FxHashMap<&str, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(source_files.len(), Default::default());
    let mut pymod_to_src: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    let mut matched_by_src: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); source_files.len()];

    for (sidx, src) in source_files.iter().enumerate() {
        let stem = stem_of(Path::new(src));
        if !stem.is_empty() {
            stem_to_srcs.entry(stem).or_default().push(sidx);
        }
        if src.ends_with(".py") {
            let pymod = src.trim_end_matches(".py").replace('/', ".");
            pymod_to_src.entry(pymod).or_default().push(sidx);
        }
    }

    for (tidx, tf) in test_files.iter().enumerate() {
        match_single_test_stem(stem_of(Path::new(tf)), tidx, &stem_to_srcs, &mut matched_by_src);
    }

    let (needed_stems, needed_pymods) =
        collect_unmatched_lookups(source_files, &pymod_to_src, &matched_by_src);
    match_test_bodies_pass(
        repo_root,
        test_files,
        preloaded,
        &needed_stems,
        &needed_pymods,
        &mut matched_by_src,
    );

    let mut map = HashMap::with_capacity(source_files.len());
    for (sidx, src) in source_files.iter().enumerate() {
        let mut tests: Vec<String> = matched_by_src[sidx]
            .iter()
            .map(|&tidx| test_files[tidx].clone())
            .collect();
        tests.sort();
        map.insert(src.clone(), tests);
    }
    map
}

fn scan_slash_stem(
    body: &str,
    bytes: &[u8],
    start: usize,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    hit_set: &mut FxHashSet<usize>,
) -> usize {
    let mut end = start;
    while end < bytes.len() && ident_byte(bytes[end]) {
        end += 1;
    }
    if let Some(srcs) = needed_stems.get(&body[start..end]) {
        hit_set.extend(srcs.iter().copied());
    }
    end
}

fn record_dotted_pymod_hits(
    span: &str,
    needed_pymods: &FxHashMap<&str, Vec<usize>>,
    hit_set: &mut FxHashSet<usize>,
) {
    if let Some(srcs) = needed_pymods.get(span) {
        hit_set.extend(srcs.iter().copied());
    }
    let mut dot_scan = span;
    while let Some((head, _)) = dot_scan.rsplit_once('.') {
        if let Some(srcs) = needed_pymods.get(head) {
            hit_set.extend(srcs.iter().copied());
        }
        dot_scan = head;
    }
}

fn record_py_import_stem_hit(
    body: &str,
    bytes: &[u8],
    mut pos: usize,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    hit_set: &mut FxHashSet<usize>,
) {
    while pos < bytes.len() && bytes[pos] == b' ' {
        pos += 1;
    }
    let id_start = pos;
    while pos < bytes.len() && ident_byte(bytes[pos]) {
        pos += 1;
    }
    if pos > id_start
        && let Some(srcs) = needed_stems.get(&body[id_start..pos])
    {
        hit_set.extend(srcs.iter().copied());
    }
}

fn scan_ident_or_pymod(
    body: &str,
    bytes: &[u8],
    start: usize,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    needed_pymods: &FxHashMap<&str, Vec<usize>>,
    hit_set: &mut FxHashSet<usize>,
) -> usize {
    let mut end = start;
    while end < bytes.len() && (ident_byte(bytes[end]) || bytes[end] == b'.') {
        end += 1;
    }
    let span = &body[start..end];
    if span.contains('.') {
        record_dotted_pymod_hits(span, needed_pymods, hit_set);
    } else if matches!(span, "import" | "from") {
        record_py_import_stem_hit(body, bytes, end, needed_stems, hit_set);
    }
    end
}

fn scan_test_body_hits(
    body: &str,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    needed_pymods: &FxHashMap<&str, Vec<usize>>,
) -> Vec<usize> {
    let mut hit_set = FxHashSet::default();
    let bytes = body.as_bytes();
    let mut pos = 0usize;

    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte == b'/' && pos + 1 < bytes.len() && ident_byte(bytes[pos + 1]) {
            pos = scan_slash_stem(body, bytes, pos + 1, needed_stems, &mut hit_set);
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            pos = scan_ident_or_pymod(body, bytes, pos, needed_stems, needed_pymods, &mut hit_set);
        } else {
            pos += 1;
        }
    }

    hit_set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn reference_tarjan(g: &CsrGraph) -> Vec<Vec<u32>> {
        let n = g.n;
        let mut index = vec![0u32; n];
        let mut lowlink = vec![0u32; n];
        let mut on_stack = vec![false; n];
        let mut scc_stack = Vec::new();
        let mut call_stack = Vec::new();
        let mut cur_idx = 1u32;
        let mut out = Vec::new();

        for start in 0..n {
            if index[start] != 0 {
                continue;
            }
            index[start] = cur_idx;
            lowlink[start] = cur_idx;
            cur_idx += 1;
            scc_stack.push(start as u32);
            on_stack[start] = true;
            call_stack.push((start as u32, g.offsets[start]));

            while let Some((v, mut ei)) = call_stack.pop() {
                let vu = v as usize;
                let end = g.offsets[vu + 1];
                let mut pushed = false;
                while ei < end {
                    let w = g.targets[ei as usize] as usize;
                    ei += 1;
                    if index[w] == 0 {
                        call_stack.push((v, ei));
                        index[w] = cur_idx;
                        lowlink[w] = cur_idx;
                        cur_idx += 1;
                        scc_stack.push(w as u32);
                        on_stack[w] = true;
                        call_stack.push((w as u32, g.offsets[w]));
                        pushed = true;
                        break;
                    } else if on_stack[w] && index[w] < lowlink[vu] {
                        lowlink[vu] = index[w];
                    }
                }
                if !pushed {
                    if lowlink[vu] == index[vu] {
                        let mut comp = Vec::new();
                        while let Some(w) = scc_stack.pop() {
                            on_stack[w as usize] = false;
                            comp.push(w);
                            if w == v {
                                break;
                            }
                        }
                        out.push(comp);
                    }
                    if let Some(&(parent, _)) = call_stack.last() {
                        let pu = parent as usize;
                        if lowlink[vu] < lowlink[pu] {
                            lowlink[pu] = lowlink[vu];
                        }
                    }
                }
            }
        }
        out
    }

    fn canonicalize_sccs(mut comps: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        for c in &mut comps {
            c.sort_unstable();
        }
        comps.sort();
        comps
    }

    #[test]
    fn test_pearce_scc_exact_parity_dense_cyclic_and_deep_chain() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };

        for &(n, edge_mult) in &[(40usize, 1usize), (120, 3), (250, 6)] {
            let mut edges = Vec::with_capacity(n * edge_mult);
            for _ in 0..(n * edge_mult) {
                let u = (next() as usize % n) as u32;
                let v = (next() as usize % n) as u32;
                edges.push((u, v));
            }
            // Add explicit interlocking cycles
            for base in (0..n.saturating_sub(4)).step_by(17) {
                let b = base as u32;
                edges.push((b, b + 1));
                edges.push((b + 1, b + 2));
                edges.push((b + 2, b));
                edges.push((b + 1, b + 3));
                edges.push((b + 3, b + 1));
            }
            let csr = CsrGraph::from_edges(n, &edges);
            assert_eq!(
                canonicalize_sccs(csr.pearce_scc()),
                canonicalize_sccs(reference_tarjan(&csr)),
                "mismatch for n={n}, mult={edge_mult}"
            );
        }

        // Deep 1500-node chain with back-edge (verifies iterative stack safety)
        let n = 1500usize;
        let mut chain: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        chain.push(((n - 1) as u32, 0));
        let csr = CsrGraph::from_edges(n, &chain);
        let comps = csr.pearce_scc();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), n);
    }

    #[test]
    fn test_inverted_match_tests_and_deep_module_rules() {
        let dir = TempRepo::with_files(
            "inv_tests",
            &[
                ("pkg/alpha.py", "def run_alpha():\n    return 1\n"),
                ("pkg/beta_core.py", "def run_beta():\n    return 2\n"),
                ("pkg/orphan.py", "def iso():\n    return 0\n"),
                ("tests/test_alpha.py", "def test_a(): pass\n"),
                ("tests/test_smoke.py", "from pkg.beta_core import run_beta\ndef test_b(): assert run_beta() == 2\n"),
            ],
        );
        let srcs = vec!["pkg/alpha.py".into(), "pkg/beta_core.py".into(), "pkg/orphan.py".into()];
        let tests = vec!["tests/test_alpha.py".into(), "tests/test_smoke.py".into()];
        let matched = match_tests_inverted(dir.path(), &srcs, &tests, None);
        assert_eq!(matched["pkg/alpha.py"], vec!["tests/test_alpha.py"]);
        assert_eq!(matched["pkg/beta_core.py"], vec!["tests/test_smoke.py"]);
        assert!(matched["pkg/orphan.py"].is_empty());

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "src/model.rs".into(),
            ModuleMetrics {
                loc: 12,
                exported_symbols: 5,
                total_fns: 4,
                passthrough_fns: 3,
                pub_fields: vec!["raw_token".into()],
                field_reads: vec![],
            },
        );
        metrics.insert(
            "src/a.rs".into(),
            ModuleMetrics {
                loc: 50,
                exported_symbols: 1,
                total_fns: 2,
                passthrough_fns: 0,
                pub_fields: vec![],
                field_reads: vec!["raw_token".into()],
            },
        );
        metrics.insert(
            "src/b.rs".into(),
            ModuleMetrics {
                loc: 50,
                exported_symbols: 1,
                total_fns: 2,
                passthrough_fns: 0,
                pub_fields: vec![],
                field_reads: vec!["raw_token".into()],
            },
        );
        let findings = deep_module_findings(
            &["src/model.rs".into(), "src/a.rs".into(), "src/b.rs".into()],
            &[("src/a.rs".into(), "src/b.rs".into()), ("src/b.rs".into(), "src/a.rs".into())],
            &metrics,
        );
        assert!(findings.iter().any(|f| f.starts_with("[module-cycle]")));
        assert!(findings.iter().any(|f| f.starts_with("[field-leak]")));
        assert!(findings.iter().any(|f| f.starts_with("[shallow-module]")));
    }
}
