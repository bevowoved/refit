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
        let n = self.n;
        let mut rindex = vec![0u32; n];
        let mut stack: Vec<u32> = Vec::with_capacity(n.min(1024));
        let mut call_stack: Vec<(u32, u32, bool)> = Vec::with_capacity(n.min(256));
        let mut idx = 1u32;
        let mut c_id = (n as u32).saturating_add(1);
        let mut comps: Vec<Vec<u32>> = Vec::new();

        for start in 0..n {
            if rindex[start] != 0 {
                continue;
            }
            rindex[start] = idx;
            idx += 1;
            call_stack.push((start as u32, self.offsets[start], true));

            while let Some((v, mut ei, mut is_root)) = call_stack.pop() {
                let vu = v as usize;
                let end = self.offsets[vu + 1];
                let mut descended = false;
                while ei < end {
                    let w = self.targets[ei as usize] as usize;
                    ei += 1;
                    if rindex[w] == 0 {
                        call_stack.push((v, ei, is_root));
                        rindex[w] = idx;
                        idx += 1;
                        call_stack.push((w as u32, self.offsets[w], true));
                        descended = true;
                        break;
                    } else if rindex[w] < rindex[vu] {
                        rindex[vu] = rindex[w];
                        is_root = false;
                    }
                }
                if !descended {
                    if is_root {
                        let mut comp = vec![v];
                        let vr = rindex[vu];
                        while let Some(&top) = stack.last() {
                            if rindex[top as usize] >= vr {
                                stack.pop();
                                rindex[top as usize] = c_id;
                                comp.push(top);
                            } else {
                                break;
                            }
                        }
                        rindex[vu] = c_id;
                        c_id += 1;
                        comps.push(comp);
                    } else {
                        stack.push(v);
                    }
                    if let Some(parent) = call_stack.last_mut() {
                        let pu = parent.0 as usize;
                        if rindex[vu] < rindex[pu] {
                            rindex[pu] = rindex[vu];
                            parent.2 = false;
                        }
                    }
                }
            }
        }
        comps
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
        if let (Some(&u), Some(&v)) = (id_of.get(src.as_str()), id_of.get(dst.as_str()))
            && u != v
        {
            csr_edges.push((u, v));
            fan_out[u as usize] += 1;
            fan_in[v as usize] += 1;
            neighbor_dirs[u as usize].insert(dir_of(dst));
            neighbor_dirs[v as usize].insert(dir_of(src));
        }
    }

    // Rule 1: module-cycle
    let csr = CsrGraph::from_edges(source_rels.len(), &csr_edges);
    let mut cycles: Vec<Vec<&str>> = csr
        .pearce_scc()
        .into_iter()
        .filter(|c| c.len() >= 2)
        .map(|c| {
            let mut files: Vec<&str> = c.into_iter().map(|id| source_rels[id as usize].as_str()).collect();
            files.sort_unstable();
            files
        })
        .collect();
    cycles.sort();
    for cyc in cycles.into_iter().take(12) {
        let preview = cyc.iter().take(4).copied().collect::<Vec<_>>().join(" <-> ");
        findings.push(format!("[module-cycle] {preview} ({} files)", cyc.len()));
    }

    // Rule 2: tangle
    for (idx, rel) in source_rels.iter().enumerate() {
        let fi = fan_in[idx];
        let fo = fan_out[idx];
        if fi >= 6 && fo >= 6 && neighbor_dirs[idx].len() >= 2 {
            let instability = (fo as f64) / ((fi + fo) as f64);
            if (0.35..=0.65).contains(&instability) {
                findings.push(format!(
                    "[tangle] {rel}: fan_in={fi}, fan_out={fo}, instability={instability:.2} across {} dirs",
                    neighbor_dirs[idx].len()
                ));
            }
        }
    }

    // Rule 3: field-leak
    let mut field_readers: FxHashMap<&str, FxHashSet<usize>> = FxHashMap::default();
    for (idx, rel) in source_rels.iter().enumerate() {
        if let Some(m) = metrics.get(rel) {
            for f in &m.field_reads {
                field_readers.entry(f.as_str()).or_default().insert(idx);
            }
        }
    }
    let mut leak_count = 0usize;
    for (idx, rel) in source_rels.iter().enumerate() {
        if leak_count >= 12 {
            break;
        }
        let Some(m) = metrics.get(rel) else { continue };
        for pf in &m.pub_fields {
            let ext_readers = field_readers
                .get(pf.as_str())
                .map(|set| set.iter().filter(|&&r| r != idx).count())
                .unwrap_or(0);
            if ext_readers >= 2 {
                findings.push(format!(
                    "[field-leak] {rel}: public field `.{pf}` accessed directly by {ext_readers} external modules"
                ));
                leak_count += 1;
                break;
            }
        }
    }

    // Rule 4: shallow-module
    let mut shallow_count = 0usize;
    for rel in source_rels {
        if shallow_count >= 12 {
            break;
        }
        let Some(m) = metrics.get(rel) else { continue };
        if m.exported_symbols >= 4 && m.total_fns >= 3 && m.loc > 0 {
            let surface_density = (m.exported_symbols as f64) / (m.loc as f64);
            let passthrough_ratio = (m.passthrough_fns as f64) / (m.total_fns as f64);
            if surface_density > 0.20 && passthrough_ratio >= 0.60 {
                findings.push(format!(
                    "[shallow-module] {rel}: {} exports in {} LOC ({:.0}% passthrough wrappers)",
                    m.exported_symbols,
                    m.loc,
                    passthrough_ratio * 100.0
                ));
                shallow_count += 1;
            }
        }
    }

    findings.truncate(40);
    findings
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

    // Pass 1: O(|T|) filename stem matching (exact parity with `stem_matches`).
    for (tidx, tf) in test_files.iter().enumerate() {
        let ts = stem_of(Path::new(tf));
        if let Some(base) = ts.strip_prefix("test_")
            && let Some(srcs) = stem_to_srcs.get(base)
        {
            for &sidx in srcs {
                matched_by_src[sidx].insert(tidx);
            }
        }
        for suffix in ["_test", ".test", ".spec"] {
            if let Some(base) = ts.strip_suffix(suffix)
                && let Some(srcs) = stem_to_srcs.get(base)
            {
                for &sidx in srcs {
                    matched_by_src[sidx].insert(tidx);
                }
            }
        }
        for (pos, b) in ts.bytes().enumerate() {
            if b == b'_' && pos + 1 < ts.len() {
                let tail = &ts[pos + 1..];
                if let Some(srcs) = stem_to_srcs.get(tail) {
                    for &sidx in srcs {
                        matched_by_src[sidx].insert(tidx);
                    }
                }
            }
        }
    }

    // Build filtered inverted lookup for unmatched sources with stem.len() >= 3.
    let mut needed_stems: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    let mut needed_pymods: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    for (sidx, src) in source_files.iter().enumerate() {
        if !matched_by_src[sidx].is_empty() {
            continue;
        }
        let stem = stem_of(Path::new(src));
        if stem.len() >= 3 {
            needed_stems.entry(stem).or_default().push(sidx);
        }
    }
    for (pymod, srcs) in &pymod_to_src {
        let unmatched: Vec<usize> = srcs
            .iter()
            .copied()
            .filter(|&sidx| matched_by_src[sidx].is_empty() && stem_of(Path::new(&source_files[sidx])).len() >= 3)
            .collect();
        if !unmatched.is_empty() {
            needed_pymods.insert(pymod.as_str(), unmatched);
        }
    }

    // Pass 2: Stream each test file ONCE, extract only hit source IDs, drop body immediately.
    if !needed_stems.is_empty() || !needed_pymods.is_empty() {
        let scan_one = |(tidx, tf): (usize, &String)| -> (usize, Vec<usize>) {
            let Some(txt) = read_text(repo_root, tf, preloaded) else {
                return (tidx, Vec::new());
            };
            let hits = scan_test_body_hits(&txt, &needed_stems, &needed_pymods);
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

fn scan_test_body_hits(
    body: &str,
    needed_stems: &FxHashMap<&str, Vec<usize>>,
    needed_pymods: &FxHashMap<&str, Vec<usize>>,
) -> Vec<usize> {
    let mut hit_set = FxHashSet::default();
    let bytes = body.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && i + 1 < bytes.len() && ident_byte(bytes[i + 1]) {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && ident_byte(bytes[end]) {
                end += 1;
            }
            if let Some(srcs) = needed_stems.get(&body[start..end]) {
                hit_set.extend(srcs.iter().copied());
            }
            i = end;
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            let mut end = start;
            while end < bytes.len() && (ident_byte(bytes[end]) || bytes[end] == b'.') {
                end += 1;
            }
            let span = &body[start..end];
            if span.contains('.') {
                if let Some(srcs) = needed_pymods.get(span) {
                    hit_set.extend(srcs.iter().copied());
                }
                // Check prefix dotted modules (e.g. `pkg.root_core.fn_name` -> `pkg.root_core`).
                let mut dot_scan = span;
                while let Some((head, _)) = dot_scan.rsplit_once('.') {
                    if let Some(srcs) = needed_pymods.get(head) {
                        hit_set.extend(srcs.iter().copied());
                    }
                    dot_scan = head;
                }
            } else if matches!(span, "import" | "from") {
                let mut k = end;
                while k < bytes.len() && bytes[k] == b' ' {
                    k += 1;
                }
                let id_start = k;
                while k < bytes.len() && ident_byte(bytes[k]) {
                    k += 1;
                }
                if k > id_start
                    && let Some(srcs) = needed_stems.get(&body[id_start..k])
                {
                    hit_set.extend(srcs.iter().copied());
                }
            }
            i = end;
            continue;
        }
        i += 1;
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
