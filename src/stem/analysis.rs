//! Unified static-graph, deep-module architecture, clean-room git co-change NPMI,
//! and 16-byte proof-lattice witness analysis (merged `csr` + `genome` + `witness`).

use crate::lex::ident_byte;
use crate::platform::{read_zstd, write_zstd};
use crate::source::{dir_of, read_text, stem_of};
use crate::types::{CloneCluster, FileEgoContext};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use wait_timeout::ChildExt;

// ============================================================================
// 1. 16-Byte Proof-Lattice Witnesses & Portable RapidHash
// ============================================================================

pub const NO_FILE: u32 = u32::MAX;

pub const WITNESS_TEST_COUNT_MASK: u16 = 0x0FFF;
pub const WITNESS_DYNAMIC_REF: u16 = 1 << 12;
pub const WITNESS_GLOB_REEXPORT: u16 = 1 << 13;
pub const WITNESS_MACRO_ATTR: u16 = 1 << 14;
pub const WITNESS_EXTERNAL: u16 = 1 << 15;

pub const BOUNDARY_FFI_NO_MANGLE: u16 = 1 << 0;
pub const BOUNDARY_CTOR_INVENTORY: u16 = 1 << 1;
pub const BOUNDARY_MACRO_DERIVE: u16 = 1 << 2;
pub const BOUNDARY_PY_ALL: u16 = 1 << 3;
pub const BOUNDARY_FRAMEWORK_ROUTE: u16 = 1 << 4;
pub const BOUNDARY_ENTRYPOINT_OR_GLOB: u16 = 1 << 5;
pub const BOUNDARY_DYNAMIC_REF: u16 = 1 << 6;
pub const DEF_VIS_INTERNAL: u16 = 1 << 7;

pub const BOUNDARY_T4_MASK: u16 = BOUNDARY_FFI_NO_MANGLE
    | BOUNDARY_CTOR_INVENTORY
    | BOUNDARY_MACRO_DERIVE
    | BOUNDARY_PY_ALL
    | BOUNDARY_FRAMEWORK_ROUTE
    | BOUNDARY_ENTRYPOINT_OR_GLOB;

/// 16-byte `#[repr(C)]` 2-distinct-file witness replacing `HashMap<String, HashSet<String>>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefWitness {
    pub prod_file_a: u32,
    pub prod_file_b: u32,
    pub test_file: u32,
    pub prod_files_sat: u16,
    pub flags_and_test: u16,
}

const _: () = assert!(std::mem::size_of::<RefWitness>() == 16);

impl Default for RefWitness {
    fn default() -> Self {
        Self { prod_file_a: NO_FILE, prod_file_b: NO_FILE, test_file: NO_FILE, prod_files_sat: 0, flags_and_test: 0 }
    }
}

impl RefWitness {
    #[inline]
    pub fn record_prod(&mut self, fid: u32) {
        if self.prod_file_a == NO_FILE {
            self.prod_file_a = fid;
            self.prod_files_sat = 1;
        } else if self.prod_file_a != fid {
            if self.prod_file_b == NO_FILE {
                self.prod_file_b = fid;
                self.prod_files_sat = 2;
            } else if self.prod_file_b != fid {
                self.prod_files_sat = self.prod_files_sat.saturating_add(1);
            }
        }
    }

    #[inline]
    pub fn record_test(&mut self, fid: u32) {
        let cur = self.flags_and_test & WITNESS_TEST_COUNT_MASK;
        if self.test_file == NO_FILE {
            self.test_file = fid;
            self.set_test_count(cur.saturating_add(1));
        } else if self.test_file != fid {
            self.set_test_count(cur.saturating_add(1));
        }
    }

    #[inline] pub fn add_flag(&mut self, flag: u16) { self.flags_and_test |= flag & !WITNESS_TEST_COUNT_MASK; }
    #[inline] pub fn test_count(self) -> u16 { self.flags_and_test & WITNESS_TEST_COUNT_MASK }
    #[inline]
    fn set_test_count(&mut self, count: u16) {
        self.flags_and_test = (self.flags_and_test & !WITNESS_TEST_COUNT_MASK) | count.min(WITNESS_TEST_COUNT_MASK);
    }

    #[inline]
    pub fn has_cross_file_prod_caller(self, def_file: u32) -> bool {
        self.prod_files_sat >= 2
            || (self.prod_files_sat == 1 && self.prod_file_a != NO_FILE && self.prod_file_a != def_file)
    }

    #[inline]
    pub fn has_external_flag(self) -> bool {
        (self.flags_and_test & (WITNESS_GLOB_REEXPORT | WITNESS_MACRO_ATTR | WITNESS_EXTERNAL)) != 0
    }
    #[inline] pub fn has_dynamic_flag(self) -> bool { (self.flags_and_test & WITNESS_DYNAMIC_REF) != 0 }
}

/// Compact `u32` path interner eliminating repeated `String` clones across symbol/graph tables.
#[derive(Debug, Default)]
pub struct PathArena { paths: Vec<String>, by_path: FxHashMap<String, u32> }

impl PathArena {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            paths: Vec::with_capacity(cap),
            by_path: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    pub fn intern(&mut self, path: &str) -> u32 {
        if let Some(&id) = self.by_path.get(path) { return id; }
        let id = self.paths.len() as u32;
        let owned = path.to_string();
        self.paths.push(owned.clone());
        self.by_path.insert(owned, id);
        id
    }

    pub fn id_of(&self, path: &str) -> Option<u32> { self.by_path.get(path).copied() }
}

/// 4-Tier Deletion-Safety Proof Lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofTier { T1ProvenDead, T2LikelyDead, T3DynamicOrTestOnly, T4ExternalOrMacro }

impl ProofTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T1ProvenDead => "T1ProvenDead",
            Self::T2LikelyDead => "T2LikelyDead",
            Self::T3DynamicOrTestOnly => "T3DynamicOrTestOnly",
            Self::T4ExternalOrMacro => "T4ExternalOrMacro",
        }
    }
    #[inline] pub fn is_dead_candidate(self) -> bool { matches!(self, Self::T1ProvenDead | Self::T2LikelyDead) }
}

pub fn classify_symbol(
    def_file: u32, witness: RefWitness, intra_non_def_hits: u16, boundary_flags: u16,
) -> Option<ProofTier> {
    if witness.has_cross_file_prod_caller(def_file) { return None; }
    if (boundary_flags & BOUNDARY_T4_MASK) != 0 || witness.has_external_flag() {
        return Some(ProofTier::T4ExternalOrMacro);
    }
    let dyn_or_test = witness.test_count() > 0 || intra_non_def_hits > 0
        || (boundary_flags & BOUNDARY_DYNAMIC_REF) != 0 || witness.has_dynamic_flag();
    if dyn_or_test { return Some(ProofTier::T3DynamicOrTestOnly); }
    Some(if (boundary_flags & DEF_VIS_INTERNAL) != 0 { ProofTier::T1ProvenDead } else { ProofTier::T2LikelyDead })
}

/// Portable 128-bit multiply-XOR mixer (`rapidhash`-style) shared across STEM, clones & SimiFlow.
#[inline]
pub fn rapid_mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline]
pub fn rapid_hash64(bytes: &[u8], seed: u64) -> u64 {
    const S0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const S1: u64 = 0x8bb8_4b93_962e_acc9;
    const S2: u64 = 0x4b33_a62e_d433_d4a3;
    let mut h = seed ^ rapid_mum(seed ^ S0, S1) ^ (bytes.len() as u64);
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let chunk = u64::from_le_bytes([
            bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3],
            bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7],
        ]);
        h = rapid_mum(h ^ chunk ^ S1, S2);
        i += 8;
    }
    if i < bytes.len() {
        let mut tail = 0u64;
        let mut shift = 0u32;
        while i < bytes.len() {
            tail |= (bytes[i] as u64) << shift;
            shift += 8;
            i += 1;
        }
        h = rapid_mum(h ^ tail ^ S0, S2);
    }
    rapid_mum(h ^ (bytes.len() as u64), S1)
}

#[inline]
pub fn rapid_hash_u64(x: u64, seed: u64) -> u64 {
    const S0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const S1: u64 = 0x8bb8_4b93_962e_acc9;
    rapid_mum(x ^ seed ^ S0, x.rotate_left(31) ^ S1)
}

pub const SYM_HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

#[inline]
pub fn hash_symbol(name: &str) -> u64 {
    rapid_hash64(name.as_bytes(), SYM_HASH_SEED)
}

// ============================================================================
// 2. Compact CSR Graph, Pearce Iterative SCC, Ego Graph & Deep-Module Rules
// ============================================================================

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

    /// Build a CSR graph and mapped `u32` edge list from named nodes and edges.
    pub fn from_named_edges(nodes: &[String], edges: &[(String, String)]) -> (Self, Vec<(u32, u32)>) {
        let mut idx_map: FxHashMap<&str, u32> =
            FxHashMap::with_capacity_and_hasher(nodes.len(), Default::default());
        for (i, rel) in nodes.iter().enumerate() {
            idx_map.insert(rel.as_str(), i as u32);
        }
        let mut csr_edges = Vec::with_capacity(edges.len());
        for (src, dst) in edges {
            if let (Some(&u), Some(&v)) = (idx_map.get(src.as_str()), idx_map.get(dst.as_str()))
                && u != v
            {
                csr_edges.push((u, v));
            }
        }
        let csr = Self::from_edges(nodes.len(), &csr_edges);
        (csr, csr_edges)
    }

    /// David Pearce (2016) iterative CSR strongly connected components.
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

const COMMON_GENERIC_FIELDS: &[&str] = &[
    "id", "name", "path", "line", "kind", "status", "bytes", "data", "error",
    "value", "score", "count", "index", "key", "msg", "code", "len", "pos",
];

fn build_importers_of(source_rels: &[String], edges: &[(String, String)]) -> Vec<FxHashSet<usize>> {
    let id_of: FxHashMap<&str, usize> =
        source_rels.iter().enumerate().map(|(i, r)| (r.as_str(), i)).collect();
    let mut importers = vec![FxHashSet::default(); source_rels.len()];
    for (src, dst) in edges {
        if let (Some(&s), Some(&d)) = (id_of.get(src.as_str()), id_of.get(dst.as_str())) && s != d {
            importers[d].insert(s);
        }
    }
    importers
}

fn build_field_readers<'a>(
    source_rels: &[String], metrics: &'a BTreeMap<String, ModuleMetrics>,
) -> FxHashMap<&'a str, FxHashSet<usize>> {
    let mut field_readers: FxHashMap<&'a str, FxHashSet<usize>> = FxHashMap::default();
    for (idx, rel) in source_rels.iter().enumerate() {
        let Some(metric) = metrics.get(rel) else { continue };
        for field in &metric.field_reads { field_readers.entry(field.as_str()).or_default().insert(idx); }
    }
    field_readers
}

fn reader_refs_definer_type(
    definer_rel: &str, reader_rel: &str,
    defs_by_file: Option<&BTreeMap<String, Vec<String>>>,
    refs_by_file: Option<&BTreeMap<String, Vec<String>>>,
) -> bool {
    let (Some(defs_map), Some(refs_map)) = (defs_by_file, refs_by_file) else { return false };
    let (Some(defs), Some(refs)) = (defs_map.get(definer_rel), refs_map.get(reader_rel)) else { return false };
    defs.iter()
        .filter(|sym| sym.as_bytes().first().is_some_and(u8::is_ascii_uppercase))
        .any(|ty| refs.iter().any(|r| r == ty))
}

fn max_scoped_field_leak_for_file(
    idx: usize, rel: &str, source_rels: &[String], importers: &FxHashSet<usize>,
    field_readers: &FxHashMap<&str, FxHashSet<usize>>, arch: &ArchGraphRef<'_>,
) -> Option<(String, usize)> {
    let metric = arch.metrics.get(rel)?;
    let mut best: Option<(String, usize)> = None;
    for pub_field in &metric.pub_fields {
        let Some(readers) = field_readers.get(pub_field.as_str()) else { continue };
        let is_generic = COMMON_GENERIC_FIELDS.contains(&pub_field.as_str());
        let count = readers.iter().filter(|&&r| {
            r != idx && importers.contains(&r) && (!is_generic
                || reader_refs_definer_type(rel, &source_rels[r], arch.defs_by_file, arch.refs_by_file))
        }).count();
        if count > best.as_ref().map_or(0, |(_, c)| *c) { best = Some((pub_field.clone(), count)); }
    }
    best
}

/// Optional graph context passed into `build_ego_contexts` and `deep_module_findings_with_csr`.
pub struct ArchGraphRef<'a> {
    pub all_edges: &'a [(String, String)],
    pub runtime_edges: &'a [(String, String)],
    pub metrics: &'a BTreeMap<String, ModuleMetrics>,
    pub defs_by_file: Option<&'a BTreeMap<String, Vec<String>>>,
    pub refs_by_file: Option<&'a BTreeMap<String, Vec<String>>>,
}

/// Single-pass `O(V + E)` ego-graph context builder feeding `hygiene::strict` and audit triage.
pub fn build_ego_contexts(
    source_rels: &[String], sccs: &[Vec<u32>], arch: &ArchGraphRef<'_>,
    file_churn: &BTreeMap<String, u32>, clone_clusters: &[CloneCluster],
) -> FxHashMap<String, FileEgoContext> {
    let n = source_rels.len();
    let id_of: FxHashMap<&str, usize> =
        source_rels.iter().enumerate().map(|(i, r)| (r.as_str(), i)).collect();
    let (mut fan_in, mut fan_out) = (vec![0usize; n], vec![0usize; n]);
    let mut neighbor_dirs: Vec<FxHashSet<String>> = vec![FxHashSet::default(); n];
    for (src, dst) in arch.runtime_edges {
        if let (Some(&s), Some(&d)) = (id_of.get(src.as_str()), id_of.get(dst.as_str())) && s != d {
            fan_out[s] += 1;
            fan_in[d] += 1;
            neighbor_dirs[s].insert(dir_of(dst));
            neighbor_dirs[d].insert(dir_of(src));
        }
    }

    let mut in_cycle = vec![false; n];
    for comp in sccs.iter().filter(|c| c.len() >= 2) {
        for &id in comp { if let Some(slot) = in_cycle.get_mut(id as usize) { *slot = true; } }
    }
    let mut clone_hits = vec![0usize; n];
    for cluster in clone_clusters {
        for (file, _) in &cluster.occurrences {
            if let Some(&idx) = id_of.get(file.as_str()) { clone_hits[idx] += 1; }
        }
    }

    let importers_of = build_importers_of(source_rels, arch.all_edges);
    let field_readers = build_field_readers(source_rels, arch.metrics);
    let mut out = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, rel) in source_rels.iter().enumerate() {
        let (fi, fo, ndirs) = (fan_in[idx], fan_out[idx], neighbor_dirs[idx].len());
        let tangle = fi >= 6 && fo >= 6 && ndirs >= 2
            && (0.35..=0.65).contains(&((fo as f64) / ((fi + fo) as f64)));
        let external_field_readers = max_scoped_field_leak_for_file(
            idx, rel, source_rels, &importers_of[idx], &field_readers, arch,
        ).map_or(0, |(_, count)| count);
        out.insert(rel.clone(), FileEgoContext {
            fan_in: fi, fan_out: fo, neighbor_dirs: ndirs, has_module_cycle: in_cycle[idx],
            tangle, external_field_readers, clone_cluster_hits: clone_hits[idx],
            churn_commits: file_churn.get(rel).copied().unwrap_or(0),
        });
    }
    out
}

fn is_barrel_file(rel: &str) -> bool {
    let fname = rel.rsplit('/').next().unwrap_or(rel);
    matches!(fname, "mod.rs" | "lib.rs" | "index.ts" | "index.js" | "__init__.py")
}

/// Evaluate all 4 deep-module architectural rules reusing a prebuilt CSR SCC list & ego map.
pub fn deep_module_findings_with_csr(
    source_rels: &[String], sccs: &[Vec<u32>], arch: &ArchGraphRef<'_>,
    ego: &FxHashMap<String, FileEgoContext>,
) -> Vec<String> {
    let mut findings = Vec::new();
    let mut cycles: Vec<Vec<&str>> = sccs.iter().filter(|c| c.len() >= 2).map(|comp| {
        let mut files: Vec<&str> = comp.iter().map(|&id| source_rels[id as usize].as_str()).collect();
        files.sort_unstable();
        files
    }).collect();
    cycles.sort();
    for cyc in cycles.into_iter().take(12) {
        let preview = cyc.iter().take(4).copied().collect::<Vec<_>>().join(" <-> ");
        findings.push(format!("[module-cycle] {preview} ({} files)", cyc.len()));
    }
    for rel in source_rels {
        if let Some(ctx) = ego.get(rel) && ctx.tangle {
            let instab = (ctx.fan_out as f64) / ((ctx.fan_in + ctx.fan_out) as f64);
            findings.push(format!(
                "[tangle] {rel}: fan_in={}, fan_out={}, instability={instab:.2} across {} dirs",
                ctx.fan_in, ctx.fan_out, ctx.neighbor_dirs
            ));
        }
    }
    let importers_of = build_importers_of(source_rels, arch.all_edges);
    let field_readers = build_field_readers(source_rels, arch.metrics);
    let mut leak_count = 0usize;
    for (idx, rel) in source_rels.iter().enumerate() {
        if leak_count >= 12 { break; }
        if let Some((f, n)) = max_scoped_field_leak_for_file(
            idx, rel, source_rels, &importers_of[idx], &field_readers, arch,
        ) && n >= 2 {
            findings.push(format!(
                "[field-leak] {rel}: public field `.{f}` accessed directly by {n} external modules"
            ));
            leak_count += 1;
        }
    }
    let mut shallow_count = 0usize;
    for rel in source_rels {
        if shallow_count >= 12 { break; }
        let Some(m) = arch.metrics.get(rel).filter(|m| m.loc > 0) else { continue };
        let pr = if m.total_fns > 0 { (m.passthrough_fns as f64) / (m.total_fns as f64) } else { 0.0 };
        let trivial = !is_barrel_file(rel) && m.loc <= 30 && m.total_fns >= 1
            && m.exported_symbols >= 1 && m.passthrough_fns == m.total_fns;
        let dense = m.exported_symbols >= 4 && m.total_fns >= 3
            && (m.exported_symbols as f64) / (m.loc as f64) > 0.20 && pr >= 0.60;
        if trivial || dense {
            findings.push(format!(
                "[shallow-module] {rel}: {} exports in {} LOC ({:.0}% passthrough wrappers)",
                m.exported_symbols, m.loc, pr * 100.0
            ));
            shallow_count += 1;
        }
    }
    findings.truncate(40);
    findings
}

// ============================================================================
// 3. Inverted-Posting Test Matcher
// ============================================================================

type StrIndexMap<'a> = FxHashMap<&'a str, Vec<usize>>;

fn record_stem_hits(
    base: &str,
    tidx: usize,
    stem_to_srcs: &StrIndexMap<'_>,
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
    stem_to_srcs: &StrIndexMap<'_>,
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
) -> (StrIndexMap<'a>, StrIndexMap<'a>) {
    let mut needed_stems: StrIndexMap<'a> = FxHashMap::default();
    let mut needed_pymods: StrIndexMap<'a> = FxHashMap::default();
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

fn record_dotted_pymod_hits(
    span: &str,
    needed_pymods: &StrIndexMap<'_>,
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

fn scan_ident_or_pymod(
    body: &str,
    bytes: &[u8],
    start: usize,
    needed_stems: &StrIndexMap<'_>,
    needed_pymods: &StrIndexMap<'_>,
    hit_set: &mut FxHashSet<usize>,
) -> usize {
    let mut end = start;
    while end < bytes.len() && (ident_byte(bytes[end]) || bytes[end] == b'.') {
        end += 1;
    }
    let span = &body[start..end];
    if span.contains('.') {
        record_dotted_pymod_hits(span, needed_pymods, hit_set);
        return end;
    }
    if matches!(span, "import" | "from") {
        let mut p = end;
        while p < bytes.len() && bytes[p] == b' ' {
            p += 1;
        }
        let id_start = p;
        while p < bytes.len() && ident_byte(bytes[p]) {
            p += 1;
        }
        if p > id_start
            && let Some(srcs) = needed_stems.get(&body[id_start..p])
        {
            hit_set.extend(srcs.iter().copied());
        }
    }
    end
}

fn scan_test_body_hits(
    body: &str,
    needed_stems: &StrIndexMap<'_>,
    needed_pymods: &StrIndexMap<'_>,
) -> Vec<usize> {
    let mut hit_set = FxHashSet::default();
    let bytes = body.as_bytes();
    let mut pos = 0usize;

    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte == b'/' && pos + 1 < bytes.len() && ident_byte(bytes[pos + 1]) {
            let start = pos + 1;
            let mut end = start;
            while end < bytes.len() && ident_byte(bytes[end]) {
                end += 1;
            }
            if let Some(srcs) = needed_stems.get(&body[start..end]) {
                hit_set.extend(srcs.iter().copied());
            }
            pos = end;
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            pos = scan_ident_or_pymod(body, bytes, pos, needed_stems, needed_pymods, &mut hit_set);
        } else {
            pos += 1;
        }
    }
    hit_set.into_iter().collect()
}

/// Inverted-posting test matcher replacing the O(|S|*|T|) body scan with zero retained test bodies.
pub fn match_tests_inverted(
    repo_root: &Path,
    source_files: &[String],
    test_files: &[String],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> HashMap<String, Vec<String>> {
    let mut stem_to_srcs: StrIndexMap<'_> =
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
    if !needed_stems.is_empty() || !needed_pymods.is_empty() {
        let scan_one = |(tidx, tf): (usize, &String)| -> (usize, Vec<usize>) {
            let Some(txt) = read_text(repo_root, tf, preloaded) else {
                return (tidx, Vec::new());
            };
            (tidx, scan_test_body_hits(&txt, &needed_stems, &needed_pymods))
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

// ============================================================================
// 4. Clean-Room Streaming Git Co-Change NPMI + Worktree HEAD-OID Cache
// ============================================================================

pub const MAX_GIT_COMMITS: usize = 2500;
pub const BULK_COMMIT_LIMIT: usize = 40;
pub const MIN_COCHANGE_SUPPORT: u32 = 3;
const GENOME_CACHE_VERSION: u8 = 1;
const GIT_LOG_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CochangePair {
    pub file_a: String,
    pub file_b: String,
    pub npmi: f64,
    pub commits: u32,
}

#[derive(Debug, Clone, Default)]
pub struct GenomeSummary {
    pub pairs: Vec<CochangePair>,
    pub file_commits: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenomeCache {
    version: u8,
    head_oid: String,
    pairs: Vec<CochangePair>,
    file_commits: BTreeMap<String, u32>,
}

/// Returns `true` when `new_count` drops by more than 80% relative to `prev_count` (`>= 50`).
#[inline]
pub fn check_wipe_guard(prev_count: usize, new_count: usize) -> bool {
    prev_count >= 50 && new_count.saturating_mul(5) < prev_count
}

/// Lockfiles and churn-heavy artifacts excluded from NPMI co-change pairs.
pub fn is_ignored_cochange_path(path: &str) -> bool {
    let f = path.rsplit('/').next().unwrap_or(path);
    matches!(
        f,
        "Cargo.lock" | "pnpm-lock.yaml" | "package-lock.json" | "yarn.lock"
            | "bun.lock" | "bun.lockb" | "uv.lock" | "poetry.lock" | "Pipfile.lock"
            | "go.sum" | "flake.lock" | "CHANGELOG.md"
    ) || f.ends_with(".snap") || f.ends_with(".min.js") || f.ends_with(".min.css")
}

/// Exponential 14-day recency decay weight in `(0.0, 1.0]`.
pub fn recency_decay(age_secs: u64) -> f64 { (-(age_secs as f64) / (14.0 * 86_400.0)).exp() }

fn is_valid_oid(s: &str) -> bool { s.len() >= 40 && s.bytes().all(|b| b.is_ascii_hexdigit()) }

fn read_hex_oid(path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    is_valid_oid(trimmed).then(|| trimmed.to_string())
}

fn find_packed_ref_oid(common_dir: &Path, ref_name: &str) -> Option<String> {
    let raw = std::fs::read_to_string(common_dir.join("packed-refs")).ok()?;
    raw.lines().map(str::trim).filter(|t| !t.is_empty() && !t.starts_with(['#', '^'])).find_map(|t| {
        let mut parts = t.split_whitespace();
        let (oid, name) = (parts.next()?, parts.next()?);
        (name == ref_name && is_valid_oid(oid)).then(|| oid.to_string())
    })
}

/// Pure-Rust zero-subprocess git `HEAD` OID resolver (dirs, worktrees, detached HEAD, `packed-refs`).
pub fn resolve_git_head_oid(repo_root: &Path) -> Option<String> {
    let dot_git = repo_root.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else if dot_git.is_file() {
        let ptr = std::fs::read_to_string(&dot_git).ok()?;
        let p = PathBuf::from(ptr.trim().strip_prefix("gitdir:")?.trim());
        if p.is_absolute() { p } else { repo_root.join(p) }
    } else { return None };

    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head_trim = head.trim();
    let Some(ref_name) = head_trim.strip_prefix("ref:").map(str::trim) else {
        return is_valid_oid(head_trim).then(|| head_trim.to_string());
    };
    if let Some(oid) = read_hex_oid(&git_dir.join(ref_name)) { return Some(oid); }
    let common_dir = std::fs::read_to_string(git_dir.join("commondir")).ok().map(|raw| {
        let p = PathBuf::from(raw.trim());
        if p.is_absolute() { p } else { git_dir.join(p) }
    }).unwrap_or_else(|| git_dir.clone());
    read_hex_oid(&common_dir.join(ref_name)).or_else(|| find_packed_ref_oid(&common_dir, ref_name))
}

#[derive(Default)]
struct NpmiCounts {
    path_to_id: FxHashMap<String, usize>,
    id_to_path: Vec<String>,
    file_commits: Vec<u32>,
    churn_commits: Vec<u32>,
    co_counts: FxHashMap<u64, u32>,
    co_decay: FxHashMap<u64, f64>,
    valid_commits: u32,
}

impl NpmiCounts {
    fn intern_path(&mut self, path: &str) -> usize {
        if let Some(&existing) = self.path_to_id.get(path) { return existing; }
        let next_id = self.id_to_path.len();
        self.id_to_path.push(path.to_string());
        self.file_commits.push(0);
        self.churn_commits.push(0);
        self.path_to_id.insert(path.to_string(), next_id);
        next_id
    }

    fn record_commit(
        &mut self, raw: &[String], live_files: Option<&FxHashSet<&str>>,
        commit_ids: &mut Vec<usize>, decay_w: f64,
    ) {
        if raw.len() >= BULK_COMMIT_LIMIT { return; }
        commit_ids.clear();
        for path in raw {
            if is_ignored_cochange_path(path) || live_files.is_some_and(|l| !l.contains(path.as_str())) {
                continue;
            }
            commit_ids.push(self.intern_path(path));
        }
        commit_ids.sort_unstable();
        commit_ids.dedup();
        for &id in commit_ids.iter() { self.churn_commits[id] = self.churn_commits[id].saturating_add(1); }
        if commit_ids.len() < 2 { return; }
        self.valid_commits += 1;
        for &id in commit_ids.iter() { self.file_commits[id] += 1; }
        for i in 0..commit_ids.len() {
            for j in (i + 1)..commit_ids.len() {
                let key = ((commit_ids[i] as u64) << 32) | (commit_ids[j] as u64);
                *self.co_counts.entry(key).or_insert(0) += 1;
                *self.co_decay.entry(key).or_insert(0.0) += decay_w;
            }
        }
    }

    fn finalize(self, min_support: u32) -> GenomeSummary {
        let n_f = self.valid_commits.max(1) as f64;
        let mut ranked: Vec<(f64, CochangePair)> = self.co_counts.iter()
            .filter_map(|(&key, &c_xy)| {
                let pair = score_npmi_pair(&self, key, c_xy, min_support, n_f)?;
                let avg = self.co_decay.get(&key).copied().unwrap_or(c_xy as f64) / (c_xy as f64);
                Some((pair.npmi * (0.85 + 0.15 * avg.clamp(0.0, 1.0)), pair))
            })
            .collect();
        ranked.sort_by(|(sa, a), (sb, b)| {
            sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.commits.cmp(&a.commits))
                .then_with(|| a.file_a.cmp(&b.file_a))
                .then_with(|| a.file_b.cmp(&b.file_b))
        });
        let pairs = ranked.into_iter().map(|(_, p)| p).collect();
        let file_commits = self.id_to_path.into_iter().zip(self.churn_commits).filter(|(_, c)| *c > 0).collect();
        GenomeSummary { pairs, file_commits }
    }
}

fn score_npmi_pair(
    counts: &NpmiCounts, key: u64, c_xy: u32, min_support: u32, n_f: f64,
) -> Option<CochangePair> {
    if c_xy < min_support { return None; }
    let (idx_a, idx_b) = ((key >> 32) as usize, (key & 0xffff_ffff) as usize);
    let (p_xy, p_x, p_y) = (
        (c_xy as f64) / n_f, (counts.file_commits[idx_a] as f64) / n_f, (counts.file_commits[idx_b] as f64) / n_f,
    );
    if p_x <= 0.0 || p_y <= 0.0 || p_xy <= 0.0 { return None; }
    let npmi = ((p_xy / (p_x * p_y)).ln() / (-p_xy.ln()).max(1e-9)).clamp(-1.0, 1.0);
    if npmi <= 0.05 { return None; }
    let (pa, pb) = (&counts.id_to_path[idx_a], &counts.id_to_path[idx_b]);
    let (file_a, file_b) = if pa <= pb { (pa.clone(), pb.clone()) } else { (pb.clone(), pa.clone()) };
    Some(CochangePair { file_a, file_b, npmi, commits: c_xy })
}

#[cfg(test)]
pub fn compute_npmi_commits(
    commits: &[Vec<String>], live_files: Option<&FxHashSet<&str>>, min_support: u32,
) -> GenomeSummary {
    let (mut counts, mut commit_ids) = (NpmiCounts::default(), Vec::with_capacity(BULK_COMMIT_LIMIT));
    for raw in commits { counts.record_commit(raw, live_files, &mut commit_ids, 1.0); }
    counts.finalize(min_support)
}

fn parse_git_status_line(
    trimmed: &str, rename_map: &mut FxHashMap<String, String>, current: &mut Vec<String>,
) {
    let mut cols = trimmed.split('\t');
    let status = cols.next().unwrap_or("");
    if status.starts_with('R') {
        if let (Some(old_p), Some(new_p)) = (cols.next(), cols.next()) {
            let (old_n, new_n) = (old_p.replace('\\', "/"), new_p.replace('\\', "/"));
            let canon = rename_map.get(&new_n).cloned().unwrap_or(new_n);
            rename_map.insert(old_n, canon.clone());
            current.push(canon);
        }
    } else if let Some(path) = cols.next() {
        let norm = path.replace('\\', "/");
        current.push(rename_map.get(&norm).cloned().unwrap_or(norm));
    }
}

/// Stream up to 2,500 non-merge commits from `git log` into `NpmiCounts`, cached by `.git/HEAD` OID.
pub fn analyze_git_cochange(repo_root: &Path, source_rels: &[String]) -> GenomeSummary {
    let Some(head_oid) = resolve_git_head_oid(repo_root) else { return GenomeSummary::default() };
    let live: FxHashSet<&str> = source_rels.iter().map(String::as_str).collect();
    let cache_path = repo_root.join(".refit/cache/genome.zst");

    if let Some(cached) = read_zstd::<GenomeCache>(&cache_path)
        && cached.version == GENOME_CACHE_VERSION && cached.head_oid == head_oid
    {
        let pairs = if live.is_empty() {
            cached.pairs
        } else {
            cached.pairs.into_iter()
                .filter(|p| live.contains(p.file_a.as_str()) && live.contains(p.file_b.as_str()))
                .collect()
        };
        return GenomeSummary { pairs, file_commits: cached.file_commits };
    }

    let Ok(mut child) = Command::new("git")
        .args(["log", "--no-merges", "--name-status", "-M", "-n", "2500", "--format=COMMIT %ct"])
        .current_dir(repo_root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()
    else { return GenomeSummary::default() };
    let Some(stdout) = child.stdout.take() else { let _ = child.wait(); return GenomeSummary::default() };

    let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (started, mut reader) = (Instant::now(), BufReader::with_capacity(64 * 1024, stdout));
    let (mut line, mut current) = (String::new(), Vec::with_capacity(BULK_COMMIT_LIMIT));
    let (mut commit_ids, mut rename_map) = (Vec::with_capacity(BULK_COMMIT_LIMIT), FxHashMap::default());
    let (mut counts, mut current_decay, mut commit_count) = (NpmiCounts::default(), 1.0f64, 0usize);
    let live_opt = (!live.is_empty()).then_some(&live);

    while let Ok(n) = reader.read_line(&mut line) {
        if n == 0 || started.elapsed() >= GIT_LOG_TIMEOUT { break; }
        let trimmed = line.trim();
        if let Some(ts_str) = trimmed.strip_prefix("COMMIT ") {
            if !current.is_empty() {
                counts.record_commit(&current, live_opt, &mut commit_ids, current_decay);
                current.clear();
            }
            commit_count += 1;
            if commit_count > MAX_GIT_COMMITS { break; }
            current_decay = ts_str.trim().parse::<u64>()
                .map_or(1.0, |ts| recency_decay(now_secs.saturating_sub(ts)));
        } else if !trimmed.is_empty() {
            parse_git_status_line(trimmed, &mut rename_map, &mut current);
        }
        line.clear();
    }
    if !current.is_empty() { counts.record_commit(&current, live_opt, &mut commit_ids, current_decay); }
    drop(reader);
    if matches!(child.wait_timeout(Duration::from_secs(1)), Ok(None)) {
        let _ = child.kill();
        let _ = child.wait();
    }

    let summary = counts.finalize(MIN_COCHANGE_SUPPORT);
    let entry = GenomeCache {
        version: GENOME_CACHE_VERSION, head_oid,
        pairs: summary.pairs.iter().take(200).cloned().collect(),
        file_commits: summary.file_commits.clone(),
    };
    let _ = write_zstd(&cache_path, &entry);
    summary
}

pub fn format_genome_twins(summary: &GenomeSummary, limit: usize) -> Vec<String> {
    summary.pairs.iter().take(limit)
        .map(|p| format!("{} <-> {} (npmi={:.2}, commits={})", p.file_a, p.file_b, p.npmi, p.commits))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hygiene::HygieneScorer;
    use crate::testutil::{TempRepo, TestRng};
    use crate::untangle::{AuditScope, untangle_repo_with_scope};

    struct TarjanRef<'a> {
        graph: &'a CsrGraph, index: Vec<u32>, lowlink: Vec<u32>, on_stack: Vec<bool>,
        scc_stack: Vec<u32>, call_stack: Vec<(u32, u32)>, next_idx: u32, comps: Vec<Vec<u32>>,
    }

    impl<'a> TarjanRef<'a> {
        fn run(graph: &'a CsrGraph) -> Vec<Vec<u32>> {
            let n = graph.n;
            let mut st = Self {
                graph, index: vec![0; n], lowlink: vec![0; n], on_stack: vec![false; n],
                scc_stack: Vec::new(), call_stack: Vec::new(), next_idx: 1, comps: Vec::new(),
            };
            for start in 0..n { if st.index[start] == 0 { st.visit(start); } }
            st.comps
        }

        fn push_node(&mut self, node: usize) {
            self.index[node] = self.next_idx;
            self.lowlink[node] = self.next_idx;
            self.next_idx += 1;
            self.scc_stack.push(node as u32);
            self.on_stack[node] = true;
            self.call_stack.push((node as u32, self.graph.offsets[node]));
        }

        fn pop_scc(&mut self, root: u32) {
            let mut comp = Vec::new();
            while let Some(w) = self.scc_stack.pop() {
                self.on_stack[w as usize] = false;
                comp.push(w);
                if w == root { break; }
            }
            self.comps.push(comp);
        }

        fn visit(&mut self, start: usize) {
            self.push_node(start);
            while let Some((v, mut ei)) = self.call_stack.pop() {
                let (vu, end, mut descended) = (v as usize, self.graph.offsets[v as usize + 1], false);
                while ei < end {
                    let w = self.graph.targets[ei as usize] as usize;
                    ei += 1;
                    if self.index[w] == 0 {
                        self.call_stack.push((v, ei));
                        self.push_node(w);
                        descended = true;
                        break;
                    }
                    if self.on_stack[w] && self.index[w] < self.lowlink[vu] {
                        self.lowlink[vu] = self.index[w];
                    }
                }
                if descended { continue; }
                if self.lowlink[vu] == self.index[vu] { self.pop_scc(v); }
                if let Some(&(parent, _)) = self.call_stack.last() {
                    self.lowlink[parent as usize] = self.lowlink[parent as usize].min(self.lowlink[vu]);
                }
            }
        }
    }

    fn canonicalize_sccs(mut comps: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        for c in &mut comps { c.sort_unstable(); }
        comps.sort();
        comps
    }

    fn mk_metric(
        loc: usize, exp: usize, fns: usize, pass: usize, pubs: &[&str], reads: &[&str],
    ) -> ModuleMetrics {
        ModuleMetrics {
            loc, exported_symbols: exp, total_fns: fns, passthrough_fns: pass,
            pub_fields: pubs.iter().map(|s| (*s).into()).collect(),
            field_reads: reads.iter().map(|s| (*s).into()).collect(),
        }
    }

    #[test]
    fn test_pearce_scc_exact_parity_dense_cyclic_and_deep_chain() {
        let mut rng = TestRng::new(0x9e37_79b9_7f4a_7c15);
        for &(n, mult) in &[(40usize, 1usize), (120, 3), (250, 6)] {
            let mut edges: Vec<(u32, u32)> = (0..n * mult)
                .map(|_| (rng.next_usize(n) as u32, rng.next_usize(n) as u32)).collect();
            for base in (0..n.saturating_sub(4)).step_by(17) {
                let b = base as u32;
                edges.extend([(b, b + 1), (b + 1, b + 2), (b + 2, b), (b + 1, b + 3), (b + 3, b + 1)]);
            }
            let csr = CsrGraph::from_edges(n, &edges);
            assert_eq!(canonicalize_sccs(csr.pearce_scc()), canonicalize_sccs(TarjanRef::run(&csr)));
        }
        let n = 1500usize;
        let mut chain: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        chain.push(((n - 1) as u32, 0));
        let comps = CsrGraph::from_edges(n, &chain).pearce_scc();
        assert_eq!((comps.len(), comps[0].len()), (1, n));
    }

    #[test]
    fn test_csr_shallow_stub_and_scoped_field_leak() {
        let dir = TempRepo::with_files("inv_tests", &[
            ("pkg/alpha.py", "def run_alpha():\n    return 1\n"),
            ("pkg/beta_core.py", "def run_beta():\n    return 2\n"),
            ("pkg/orphan.py", "def iso():\n    return 0\n"),
            ("tests/test_alpha.py", "def test_a(): pass\n"),
            ("tests/test_smoke.py", "from pkg.beta_core import run_beta\ndef test_b(): assert run_beta() == 2\n"),
        ]);
        let srcs = vec!["pkg/alpha.py".into(), "pkg/beta_core.py".into(), "pkg/orphan.py".into()];
        let tests = vec!["tests/test_alpha.py".into(), "tests/test_smoke.py".into()];
        let matched = match_tests_inverted(dir.path(), &srcs, &tests, None);
        assert_eq!(matched["pkg/alpha.py"], vec!["tests/test_alpha.py"]);
        assert_eq!(matched["pkg/beta_core.py"], vec!["tests/test_smoke.py"]);
        assert!(matched["pkg/orphan.py"].is_empty());

        let metrics = BTreeMap::from([
            ("src/model.rs".into(), mk_metric(12, 5, 4, 3, &["raw_token", "line"], &[])),
            ("src/syntax_stub.rs".into(), mk_metric(6, 1, 1, 1, &[], &[])),
            ("src/sub/mod.rs".into(), mk_metric(6, 1, 1, 1, &[], &[])),
            ("src/a.rs".into(), mk_metric(50, 1, 2, 0, &[], &["raw_token", "line"])),
            ("src/b.rs".into(), mk_metric(50, 1, 2, 0, &[], &["raw_token", "line"])),
            ("src/unrelated.rs".into(), mk_metric(50, 1, 2, 0, &[], &["line"])),
        ]);
        let rels: Vec<String> = metrics.keys().cloned().collect();
        let edges = vec![
            ("src/a.rs".into(), "src/b.rs".into()), ("src/b.rs".into(), "src/a.rs".into()),
            ("src/a.rs".into(), "src/model.rs".into()), ("src/b.rs".into(), "src/model.rs".into()),
        ];
        let (csr, _) = CsrGraph::from_named_edges(&rels, &edges);
        let sccs = csr.pearce_scc();
        let arch = ArchGraphRef {
            all_edges: &edges, runtime_edges: &edges, metrics: &metrics,
            defs_by_file: None, refs_by_file: None,
        };
        let ego = build_ego_contexts(&rels, &sccs, &arch, &BTreeMap::new(), &[]);
        let findings = deep_module_findings_with_csr(&rels, &sccs, &arch, &ego);

        assert!(findings.iter().any(|f| f.starts_with("[module-cycle]")));
        assert!(findings.iter().any(|f| f.contains("[field-leak]") && f.contains(".raw_token")));
        assert!(!findings.iter().any(|f| f.contains(".line")), "generic .line leaked: {findings:?}");
        assert!(findings.iter().any(|f| f.contains("[shallow-module] src/syntax_stub.rs")));
        assert!(!findings.iter().any(|f| f.contains("src/sub/mod.rs")), "barrel flagged: {findings:?}");
        assert_eq!(ego["src/model.rs"].external_field_readers, 2);
        assert!(ego["src/a.rs"].has_module_cycle);
    }

    #[test]
    fn test_genome_npmi_hub_demotion_lockfile_filter_and_wipe_guard() {
        let mut bulk = vec!["src/hub.rs".into(), "Cargo.lock".into()];
        bulk.extend((0..45).map(|i| format!("src/bulk_{i}.rs")));
        let mut commits = vec![bulk];
        for c in 0..60 {
            let mut files = vec![
                "src/hub.rs".into(), format!("src/noise_{}.rs", c % 15),
                "Cargo.lock".into(), "pnpm-lock.yaml".into(),
            ];
            if c % 5 == 0 { files.extend(["src/route.rs".into(), "src/schema.rs".into()]); }
            if c % 6 == 1 { files.extend(["src/auth.rs".into(), "src/session.rs".into()]); }
            commits.push(files);
        }
        let summary = compute_npmi_commits(&commits, None, MIN_COCHANGE_SUPPORT);
        assert!(!summary.pairs.iter().any(|p| p.file_a.ends_with(".lock") || p.file_b.ends_with(".lock")));
        let top2: Vec<(&str, &str)> = summary.pairs.iter().take(2)
            .map(|p| (p.file_a.as_str(), p.file_b.as_str())).collect();
        assert!(top2.contains(&("src/route.rs", "src/schema.rs")) && top2.contains(&("src/auth.rs", "src/session.rs")));
        assert!(check_wipe_guard(100, 19) && !check_wipe_guard(100, 20) && !check_wipe_guard(40, 2));
        assert!(recency_decay(0) > 0.99 && recency_decay(14 * 86_400) < 0.40);
    }

    #[test]
    fn test_genome_head_cache_worktree_and_sustain_skip() {
        let fake_wt = TempRepo::new("wt_head");
        let oid = "0123456789abcdef0123456789abcdef01234567";
        fake_wt.write("common/packed-refs", format!("# pack-refs\n{oid} refs/heads/feat/wt\n"));
        fake_wt.write("wt_meta/commondir", "../common\n");
        fake_wt.write("wt_meta/HEAD", "ref: refs/heads/feat/wt\n");
        fake_wt.write("work/.git", format!("gitdir: {}\n", fake_wt.join("wt_meta").display()));
        assert_eq!(resolve_git_head_oid(&fake_wt.join("work")).as_deref(), Some(oid));

        let repo = TempRepo::git("genome_cache", "app", &[
            ("src/a.rs", "pub fn alpha_one() -> usize { 1 }\n"),
            ("src/b.rs", "pub fn beta_two() -> usize { 2 }\n"),
        ]);
        let head = resolve_git_head_oid(repo.path()).expect("real git HEAD OID");
        assert_eq!(head.len(), 40);

        let full = untangle_repo_with_scope(repo.path(), &HygieneScorer::new(), AuditScope::Full).unwrap();
        let cached = read_zstd::<GenomeCache>(&repo.join(".refit/cache/genome.zst")).unwrap();
        assert_eq!(cached.head_oid, head);
        assert!(cached.file_commits.get("src/a.rs").copied().unwrap_or(0) >= 1);

        let sustain = untangle_repo_with_scope(repo.path(), &HygieneScorer::new(), AuditScope::Sustain).unwrap();
        assert_eq!(sustain.total_source_files, full.total_source_files);
        assert!(sustain.genome_twins.is_empty() && sustain.capability_clusters.is_empty());
        assert!(sustain.deep_module_findings.is_empty() && sustain.doc_drift_findings.is_empty());
    }

    #[test]
    fn test_ref_witness_and_proof_lattice() {
        assert_eq!(std::mem::size_of::<RefWitness>(), 16);
        let mut w = RefWitness::default();
        assert!(!w.has_cross_file_prod_caller(0));
        w.record_prod(3); w.record_prod(3);
        assert_eq!(w.prod_files_sat, 1);
        assert!(!w.has_cross_file_prod_caller(3) && w.has_cross_file_prod_caller(4));
        w.record_prod(7);
        assert_eq!(w.prod_files_sat, 2);
        assert!(w.has_cross_file_prod_caller(3) && w.has_cross_file_prod_caller(7));
        w.record_test(10); w.record_test(10); w.record_test(11);
        assert_eq!(w.test_count(), 2);

        let empty = RefWitness::default();
        assert_eq!(classify_symbol(0, empty, 0, DEF_VIS_INTERNAL), Some(ProofTier::T1ProvenDead));
        assert_eq!(classify_symbol(0, empty, 0, 0), Some(ProofTier::T2LikelyDead));
        assert_eq!(classify_symbol(0, empty, 1, 0), Some(ProofTier::T3DynamicOrTestOnly));
        assert_eq!(classify_symbol(0, empty, 0, BOUNDARY_DYNAMIC_REF), Some(ProofTier::T3DynamicOrTestOnly));
        for flag in [
            BOUNDARY_FFI_NO_MANGLE, BOUNDARY_CTOR_INVENTORY, BOUNDARY_MACRO_DERIVE,
            BOUNDARY_PY_ALL, BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_ENTRYPOINT_OR_GLOB,
        ] {
            assert_eq!(classify_symbol(0, empty, 0, flag), Some(ProofTier::T4ExternalOrMacro));
        }
        let mut live = RefWitness::default();
        live.record_prod(9);
        assert_eq!(classify_symbol(0, live, 0, 0), None);
    }
}
