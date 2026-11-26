//! Unified static-graph, deep-module architecture, clean-room git co-change NPMI,
//! and 16-byte proof-lattice witness analysis (merged `csr` + `genome` + `witness`).

use crate::lex::{LangKind, LexCursor, LexKind, ident_byte};
use crate::platform::{par_map, read_zstd, write_zstd};
use crate::source::{dir_of, read_text, stem_of};
use crate::types::{CloneCluster, FileEgoContext};
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

pub const BOUNDARY_T4_MASK: u16 = BOUNDARY_FFI_NO_MANGLE | BOUNDARY_CTOR_INVENTORY
    | BOUNDARY_MACRO_DERIVE | BOUNDARY_PY_ALL | BOUNDARY_FRAMEWORK_ROUTE | BOUNDARY_ENTRYPOINT_OR_GLOB;

/// 16-byte `#[repr(C)]` 2-distinct-file witness replacing `HashMap<String, HashSet<String>>`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefWitness {
    pub prod_file_a: u32, pub prod_file_b: u32, pub test_file: u32,
    pub prod_files_sat: u16, pub flags_and_test: u16,
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
        if self.prod_file_a == NO_FILE { (self.prod_file_a, self.prod_files_sat) = (fid, 1); }
        else if self.prod_file_a != fid {
            if self.prod_file_b == NO_FILE { (self.prod_file_b, self.prod_files_sat) = (fid, 2); }
            else if self.prod_file_b != fid { self.prod_files_sat = self.prod_files_sat.saturating_add(1); }
        }
    }

    #[inline]
    pub fn record_test(&mut self, fid: u32) {
        if self.test_file == NO_FILE || self.test_file != fid {
            if self.test_file == NO_FILE { self.test_file = fid; }
            let next = self.test_count().saturating_add(1).min(WITNESS_TEST_COUNT_MASK);
            self.flags_and_test = (self.flags_and_test & !WITNESS_TEST_COUNT_MASK) | next;
        }
    }

    #[inline] pub fn add_flag(&mut self, flag: u16) { self.flags_and_test |= flag & !WITNESS_TEST_COUNT_MASK; }
    #[inline] pub fn test_count(self) -> u16 { self.flags_and_test & WITNESS_TEST_COUNT_MASK }

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
        Self { paths: Vec::with_capacity(cap), by_path: FxHashMap::with_capacity_and_hasher(cap, Default::default()) }
    }

    pub fn intern(&mut self, path: &str) -> u32 {
        if let Some(&id) = self.by_path.get(path) { return id; }
        let (id, owned) = (self.paths.len() as u32, path.to_string());
        self.paths.push(owned.clone());
        self.by_path.insert(owned, id);
        id
    }

    pub fn id_of(&self, path: &str) -> Option<u32> { self.by_path.get(path).copied() }
    pub fn path(&self, id: usize) -> &str { &self.paths[id] }
    pub fn into_paths(self) -> Vec<String> { self.paths }
}

/// 4-Tier Deletion-Safety Proof Lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofTier { T1ProvenDead, T2LikelyDead, T3DynamicOrTestOnly, T4ExternalOrMacro }

impl ProofTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T1ProvenDead => "T1ProvenDead", Self::T2LikelyDead => "T2LikelyDead",
            Self::T3DynamicOrTestOnly => "T3DynamicOrTestOnly", Self::T4ExternalOrMacro => "T4ExternalOrMacro",
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
    if witness.test_count() > 0 || intra_non_def_hits > 0
        || (boundary_flags & BOUNDARY_DYNAMIC_REF) != 0 || witness.has_dynamic_flag()
    {
        return Some(ProofTier::T3DynamicOrTestOnly);
    }
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
        let chunk = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap_or([0; 8]));
        h = rapid_mum(h ^ chunk ^ S1, S2);
        i += 8;
    }

    if i < bytes.len() {
        let (mut tail, mut shift) = (0u64, 0u32);
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
pub fn hash_symbol(name: &str) -> u64 { rapid_hash64(name.as_bytes(), SYM_HASH_SEED) }

// ============================================================================
// 2. Compact CSR Graph, Pearce Iterative SCC, Ego Graph & Deep-Module Rules
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind { Runtime, TypeOnly, Dynamic, ModDecl, TestImport }

impl EdgeKind {
    #[inline]
    pub fn is_runtime_scc(self) -> bool { matches!(self, Self::Runtime | Self::Dynamic | Self::ModDecl) }
}

/// Compact Compressed Sparse Row directed graph using flat `u32` arrays.
#[derive(Debug, Clone)]
pub struct CsrGraph { pub n: usize, pub offsets: Vec<u32>, pub targets: Vec<u32> }

impl CsrGraph {
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut sorted: Vec<(u32, u32)> = edges.iter().copied()
            .filter(|&(u, v)| (u as usize) < n && (v as usize) < n && u != v).collect();
        sorted.sort_unstable();
        sorted.dedup();
        let mut offsets = vec![0u32; n + 1];
        for &(u, _) in &sorted { offsets[u as usize + 1] += 1; }
        for i in 0..n { offsets[i + 1] += offsets[i]; }
        let (mut cur, mut targets) = (offsets.clone(), vec![0u32; sorted.len()]);
        for &(u, v) in &sorted {
            let pos = cur[u as usize] as usize;
            targets[pos] = v;
            cur[u as usize] += 1;
        }
        Self { n, offsets, targets }
    }

    /// Build a CSR graph and mapped `u32` edge list from named nodes and edges.
    pub fn from_named_edges(nodes: &[String], edges: &[(String, String)]) -> (Self, Vec<(u32, u32)>) {
        let idx_map = index_source_rels(nodes);
        let csr_edges: Vec<(u32, u32)> = edges.iter().filter_map(|(src, dst)| {
            let (&u, &v) = (idx_map.get(src.as_str())?, idx_map.get(dst.as_str())?);
            (u != v).then_some((u as u32, v as u32))
        }).collect();
        let csr = Self::from_edges(nodes.len(), &csr_edges);
        (csr, csr_edges)
    }

    /// David Pearce (2016) iterative CSR strongly connected components.
    pub fn pearce_scc(&self) -> Vec<Vec<u32>> {
        let mut state = PearceState::new(self);
        for start in 0..self.n { if state.rindex[start] == 0 { state.visit_from(start); } }
        state.comps
    }
}

struct PearceState<'a> {
    graph: &'a CsrGraph, rindex: Vec<u32>, stack: Vec<u32>,
    call_stack: Vec<(u32, u32, bool)>, next_idx: u32, comp_id: u32, comps: Vec<Vec<u32>>,
}

impl<'a> PearceState<'a> {
    fn new(graph: &'a CsrGraph) -> Self {
        let n = graph.n;
        Self {
            graph, rindex: vec![0u32; n], stack: Vec::with_capacity(n.min(1024)),
            call_stack: Vec::with_capacity(n.min(256)), next_idx: 1,
            comp_id: (n as u32).saturating_add(1), comps: Vec::new(),
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
        let (node_u, end) = (node as usize, self.graph.offsets[node as usize + 1]);
        while edge_idx < end {
            let neighbor = self.graph.targets[edge_idx as usize] as usize;
            edge_idx += 1;
            if self.rindex[neighbor] == 0 {
                self.push_unvisited(node, edge_idx, is_root, neighbor);
                return None;
            }
            is_root &= self.relax_lowlink(node_u, neighbor);
        }
        Some(is_root)
    }

    #[inline]
    fn relax_lowlink(&mut self, node_u: usize, neighbor: usize) -> bool {
        if self.rindex[neighbor] >= self.rindex[node_u] { return true; }
        self.rindex[node_u] = self.rindex[neighbor];
        false
    }

    fn push_unvisited(&mut self, node: u32, edge_idx: u32, is_root: bool, neighbor: usize) {
        self.call_stack.push((node, edge_idx, is_root));
        self.rindex[neighbor] = self.next_idx;
        self.next_idx += 1;
        self.call_stack.push((neighbor as u32, self.graph.offsets[neighbor], true));
    }

    fn finish_node(&mut self, node: u32, is_root: bool) {
        if is_root { self.emit_component(node); } else { self.stack.push(node); }
        let Some(parent) = self.call_stack.last_mut() else { return };
        let child_r = self.rindex[node as usize];
        let parent_r = &mut self.rindex[parent.0 as usize];
        if child_r < *parent_r {
            *parent_r = child_r;
            parent.2 = false;
        }
    }

    fn emit_component(&mut self, root: u32) {
        let (root_u, mut comp) = (root as usize, vec![root]);
        let root_rindex = self.rindex[root_u];
        while let Some(&top) = self.stack.last() {
            if self.rindex[top as usize] < root_rindex { break; }
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
    pub loc: usize, pub exported_symbols: usize, pub total_fns: usize, pub passthrough_fns: usize,
    pub pub_fields: Vec<String>, pub field_reads: Vec<String>,
}

const COMMON_GENERIC_FIELDS: &[&str] = &[
    "id", "name", "path", "line", "kind", "status", "bytes", "data", "error",
    "value", "score", "count", "index", "key", "msg", "code", "len", "pos",
];

/// Optional graph context passed into ego-graph and deep-module analysis.
pub struct ArchGraphRef<'a> {
    pub all_edges: &'a [(String, String)], pub runtime_edges: &'a [(String, String)],
    pub metrics: &'a BTreeMap<String, ModuleMetrics>,
    pub defs_by_file: Option<&'a BTreeMap<String, Vec<String>>>,
    pub refs_by_file: Option<&'a BTreeMap<String, Vec<String>>>,
    pub sources: Option<&'a BTreeMap<String, Vec<u8>>>,
}

fn index_source_rels(source_rels: &[String]) -> FxHashMap<&str, usize> {
    source_rels.iter()
        .enumerate()
        .map(|(idx, rel)| (rel.as_str(), idx))
        .collect()
}

struct RustStructFields<'a> { pub_fields: Vec<&'a str>, has_private: bool }

const TRIVIAL_DTO_TRAITS: &[&str] = &[
    "Default", "Display", "Debug", "Clone", "Copy", "PartialEq", "Eq",
    "PartialOrd", "Ord", "Hash", "From", "Into", "TryFrom", "AsRef", "Deref",
];

fn collect_rs_code_tokens(bytes: &[u8]) -> Vec<(LexKind, &str)> {
    let mut cursor = LexCursor::new(LangKind::Rust, bytes);
    let mut toks = Vec::new();
    let mut skip_attr_brackets = 0usize;
    let mut pending_hash = false;
    while let Some(tok) = cursor.next_token() {
        if matches!(tok.kind, LexKind::LineComment | LexKind::BlockComment | LexKind::Newline)
            || tok.flags.in_test()
        {
            continue;
        }
        if skip_attr_brackets > 0 {
            if tok.kind == LexKind::LBracket { skip_attr_brackets += 1; }
            else if tok.kind == LexKind::RBracket { skip_attr_brackets -= 1; }
            continue;
        }
        if pending_hash {
            if tok.kind == LexKind::Bang { continue; }
            pending_hash = false;
            if tok.kind == LexKind::LBracket { skip_attr_brackets = 1; continue; }
        }
        if tok.kind == LexKind::Hash { pending_hash = true; continue; }
        toks.push((tok.kind, tok.as_str()));
    }
    toks
}

fn parse_rs_struct_body<'a>(toks: &[(LexKind, &'a str)], start: usize) -> (RustStructFields<'a>, usize) {
    let (mut pub_fields, mut has_private) = (Vec::new(), false);
    let (mut i, mut depth, mut at_field_start) = (start, 0usize, true);
    while i < toks.len() {
        let (kind, text) = toks[i];
        if depth == 0 && kind == LexKind::RBrace { return (RustStructFields { pub_fields, has_private }, i + 1); }
        if depth == 0 && kind == LexKind::Comma { at_field_start = true; i += 1; continue; }
        if depth == 0 && at_field_start {
            at_field_start = false;
            if kind == LexKind::Ident && text == "pub" {
                let restricted = toks.get(i + 1).is_some_and(|(k, _)| *k == LexKind::LParen);
                if restricted {
                    has_private = true;
                } else if let Some((LexKind::Ident, name)) = toks.get(i + 1)
                    && toks.get(i + 2).is_some_and(|(k, _)| *k == LexKind::Colon)
                {
                    pub_fields.push(*name);
                }
            } else if kind == LexKind::Ident
                && toks.get(i + 1).is_some_and(|(k, _)| *k == LexKind::Colon)
            {
                has_private = true;
            }
        }
        match kind {
            LexKind::LBrace | LexKind::LParen | LexKind::LBracket | LexKind::Lt => depth += 1,
            LexKind::RBrace | LexKind::RParen | LexKind::RBracket | LexKind::Gt => depth = depth.saturating_sub(1),
            _ => {}
        }
        i += 1;
    }
    (RustStructFields { pub_fields, has_private }, i)
}

fn parse_rs_impl_behavior<'a>(toks: &[(LexKind, &'a str)], start: usize) -> (Option<&'a str>, bool, usize) {
    let (mut i, mut angle, mut in_where) = (start, 0usize, false);
    let (mut seen_for, mut trait_name, mut target_name) = (false, None, None);
    while i < toks.len() && !(angle == 0 && toks[i].0 == LexKind::LBrace) {
        let (kind, text) = toks[i];
        match kind {
            LexKind::Lt => angle += 1,
            LexKind::Gt => angle = angle.saturating_sub(1),
            LexKind::Ident if angle == 0 => {
                if text == "where" { in_where = true; }
                else if !in_where && text == "for" { seen_for = true; }
                else if !in_where && seen_for { target_name = Some(text); }
                else if !in_where {
                    if trait_name.is_none() || !seen_for { trait_name = Some(text); }
                    target_name = Some(text);
                }
            }
            _ => {}
        }
        i += 1;
    }
    if i >= toks.len() { return (None, false, i); }
    let (mut depth, mut has_fn) = (1usize, false);
    i += 1;
    while i < toks.len() && depth > 0 {
        match toks[i] {
            (LexKind::LBrace, _) => depth += 1,
            (LexKind::RBrace, _) => depth -= 1,
            (LexKind::Ident, "fn") if depth == 1 => has_fn = true,
            _ => {}
        }
        i += 1;
    }
    let trivial_trait = seen_for && trait_name.is_some_and(|t| TRIVIAL_DTO_TRAITS.contains(&t));
    (target_name, has_fn && !trivial_trait, i)
}

fn extract_rs_pure_dto_fields(bytes: &[u8]) -> FxHashSet<String> {
    let toks = collect_rs_code_tokens(bytes);
    let mut structs: Vec<(&str, RustStructFields<'_>)> = Vec::new();
    let mut behavioral: FxHashSet<&str> = FxHashSet::default();
    let (mut i, mut brace_depth) = (0usize, 0usize);
    while i < toks.len() {
        let (kind, text) = toks[i];
        if brace_depth == 0 && kind == LexKind::Ident && text == "pub"
            && toks.get(i + 1) == Some(&(LexKind::Ident, "struct"))
            && let Some(&(LexKind::Ident, sname)) = toks.get(i + 2)
        {
            let mut j = i + 3;
            while j < toks.len() && !matches!(toks[j].0, LexKind::LBrace | LexKind::LParen | LexKind::Semi) {
                j += 1;
            }
            if toks.get(j).is_some_and(|(k, _)| *k == LexKind::LBrace) {
                let (info, next_i) = parse_rs_struct_body(&toks, j + 1);
                structs.push((sname, info));
                i = next_i;
                continue;
            }
            i = (j + 1).min(toks.len());
            continue;
        }
        if brace_depth == 0 && kind == LexKind::Ident && text == "impl" {
            let (target, has_behavior, next_i) = parse_rs_impl_behavior(&toks, i + 1);
            if has_behavior && let Some(t) = target { behavioral.insert(t); }
            i = next_i;
            continue;
        }
        if kind == LexKind::LBrace { brace_depth += 1; }
        else if kind == LexKind::RBrace { brace_depth = brace_depth.saturating_sub(1); }
        i += 1;
    }
    let mut non_dto: FxHashSet<&str> = FxHashSet::default();
    let mut dto: FxHashSet<&str> = FxHashSet::default();
    for (name, info) in structs {
        let is_pure = !info.has_private && !behavioral.contains(name) && !info.pub_fields.is_empty();
        for field in info.pub_fields {
            if is_pure { dto.insert(field); } else { non_dto.insert(field); }
        }
    }
    dto.into_iter().filter(|f| !non_dto.contains(f)).map(str::to_string).collect()
}

struct ImportLeakCtx<'a> {
    source_rels: &'a [String], importers_of: Vec<FxHashSet<usize>>,
    field_readers: FxHashMap<&'a str, FxHashSet<usize>>,
    pure_dto_fields: Vec<FxHashSet<String>>,
}

fn build_leak_ctx<'a>(
    source_rels: &'a [String], id_of: &FxHashMap<&str, usize>, arch: &ArchGraphRef<'a>,
) -> ImportLeakCtx<'a> {
    let mut importers_of = vec![FxHashSet::default(); source_rels.len()];
    for (src, dst) in arch.all_edges {
        if let (Some(&s), Some(&d)) = (id_of.get(src.as_str()), id_of.get(dst.as_str())) && s != d {
            importers_of[d].insert(s);
        }
    }

    let mut field_readers: FxHashMap<&'a str, FxHashSet<usize>> = FxHashMap::default();
    let mut pure_dto_fields = vec![FxHashSet::default(); source_rels.len()];
    for (idx, rel) in source_rels.iter().enumerate() {
        if rel.ends_with(".rs") && let Some(bytes) = arch.sources.and_then(|m| m.get(rel)) {
            pure_dto_fields[idx] = extract_rs_pure_dto_fields(bytes);
        }
        let Some(metric) = arch.metrics.get(rel) else { continue };
        for field in &metric.field_reads {
            field_readers.entry(field.as_str()).or_default().insert(idx);
        }
    }
    ImportLeakCtx { source_rels, importers_of, field_readers, pure_dto_fields }
}

impl<'a> ImportLeakCtx<'a> {
    fn new(source_rels: &'a [String], id_of: &FxHashMap<&str, usize>, arch: &ArchGraphRef<'a>) -> Self {
        build_leak_ctx(source_rels, id_of, arch)
    }

    fn max_leak(&self, idx: usize, rel: &str, arch: &ArchGraphRef<'_>) -> Option<(String, usize)> {
        let (metric, importers) = (arch.metrics.get(rel)?, &self.importers_of[idx]);
        let mut best: Option<(String, usize)> = None;
        for pub_field in &metric.pub_fields {
            if self.pure_dto_fields[idx].contains(pub_field.as_str()) { continue; }
            let Some(readers) = self.field_readers.get(pub_field.as_str()) else { continue };
            let is_generic = COMMON_GENERIC_FIELDS.contains(&pub_field.as_str());
            let count = readers.iter().filter(|&&r| {
                r != idx && importers.contains(&r) && (!is_generic
                    || reader_refs_definer_type(rel, &self.source_rels[r], arch.defs_by_file, arch.refs_by_file))
            }).count();
            if count > best.as_ref().map_or(0, |(_, c)| *c) { best = Some((pub_field.clone(), count)); }
        }
        best
    }
}

fn reader_refs_definer_type(
    definer_rel: &str, reader_rel: &str,
    defs_by_file: Option<&BTreeMap<String, Vec<String>>>,
    refs_by_file: Option<&BTreeMap<String, Vec<String>>>,
) -> bool {
    let (Some(defs), Some(refs)) =
        (defs_by_file.and_then(|m| m.get(definer_rel)), refs_by_file.and_then(|m| m.get(reader_rel)))
    else { return false };
    defs.iter().any(|sym| {
        let is_type = sym.as_bytes().first().is_some_and(u8::is_ascii_uppercase);
        is_type && refs.iter().any(|r| r == sym)
    })
}

fn is_barrel_file(rel: &str) -> bool {
    matches!(rel.rsplit('/').next().unwrap_or(rel), "mod.rs" | "lib.rs" | "index.ts" | "index.js" | "__init__.py")
}

/// Single-pass `O(V + E)` ego-context builder and 4-rule deep-module evaluator.
pub fn analyze_ego_and_deep_modules(
    source_rels: &[String], sccs: &[Vec<u32>], arch: &ArchGraphRef<'_>,
    file_churn: &BTreeMap<String, u32>, clone_clusters: &[CloneCluster],
) -> (FxHashMap<String, FileEgoContext>, Vec<String>) {
    let id_of = index_source_rels(source_rels);
    let leak_ctx = ImportLeakCtx::new(source_rels, &id_of, arch);
    let leaks: Vec<_> = source_rels.iter()
        .enumerate()
        .map(|(i, rel)| leak_ctx.max_leak(i, rel, arch))
        .collect();
    let ego = build_ego_with_leaks(source_rels, &id_of, sccs, arch, file_churn, clone_clusters, &leaks);
    let findings = collect_deep_module_findings(source_rels, sccs, arch, &ego, &leaks);
    (ego, findings)
}

fn build_ego_with_leaks(
    source_rels: &[String], id_of: &FxHashMap<&str, usize>, sccs: &[Vec<u32>], arch: &ArchGraphRef<'_>,
    file_churn: &BTreeMap<String, u32>, clone_clusters: &[CloneCluster],
    leaks: &[Option<(String, usize)>],
) -> FxHashMap<String, FileEgoContext> {
    let n = source_rels.len();
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
    for &id in sccs.iter().filter(|c| c.len() >= 2).flatten() {
        if let Some(slot) = in_cycle.get_mut(id as usize) { *slot = true; }
    }

    let mut clone_hits = vec![0usize; n];
    for (file, _) in clone_clusters.iter().flat_map(|c| &c.occurrences) {
        if let Some(&idx) = id_of.get(file.as_str()) { clone_hits[idx] += 1; }
    }

    let mut out = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, rel) in source_rels.iter().enumerate() {
        let (fi, fo, ndirs) = (fan_in[idx], fan_out[idx], neighbor_dirs[idx].len());
        let tangle = fi >= 6 && fo >= 6 && ndirs >= 2
            && (0.35..=0.65).contains(&((fo as f64) / ((fi + fo) as f64)));
        let external_field_readers = leaks[idx].as_ref().map_or(0, |(_, count)| *count);
        out.insert(rel.clone(), FileEgoContext {
            fan_in: fi, fan_out: fo, neighbor_dirs: ndirs, has_module_cycle: in_cycle[idx],
            tangle, external_field_readers, clone_cluster_hits: clone_hits[idx],
            churn_commits: file_churn.get(rel).copied().unwrap_or(0),
        });
    }
    out
}

fn collect_cycle_findings(source_rels: &[String], sccs: &[Vec<u32>]) -> Vec<String> {
    let mut cycles: Vec<Vec<&str>> = sccs.iter().filter(|c| c.len() >= 2).map(|comp| {
        let mut files: Vec<&str> = comp.iter()
            .map(|&id| source_rels[id as usize].as_str())
            .collect();
        files.sort_unstable();
        files
    }).collect();
    cycles.sort();
    cycles.into_iter().take(12).map(|cyc| {
        format!("[module-cycle] {} ({} files)", cyc[..cyc.len().min(4)].join(" <-> "), cyc.len())
    }).collect()
}

fn collect_deep_module_findings(
    source_rels: &[String], sccs: &[Vec<u32>], arch: &ArchGraphRef<'_>,
    ego: &FxHashMap<String, FileEgoContext>, leaks: &[Option<(String, usize)>],
) -> Vec<String> {
    let mut findings = collect_cycle_findings(source_rels, sccs);

    for rel in source_rels {
        let Some(ctx) = ego.get(rel).filter(|c| c.tangle) else { continue };
        let instab = (ctx.fan_out as f64) / ((ctx.fan_in + ctx.fan_out) as f64);
        findings.push(format!(
            "[tangle] {rel}: fan_in={}, fan_out={}, instability={instab:.2} across {} dirs",
            ctx.fan_in, ctx.fan_out, ctx.neighbor_dirs
        ));
    }

    let leak_pairs = source_rels.iter()
        .zip(leaks)
        .filter_map(|(r, l)| {
            let (field, n) = l.as_ref()?;
            (*n >= 2).then_some((r, field, *n))
        })
        .take(12);
    for (rel, field, n) in leak_pairs {
        findings.push(format!("[field-leak] {rel}: public field `.{field}` accessed directly by {n} external modules"));
    }
    findings.extend(collect_shallow_findings(source_rels, arch).take(12));
    findings.truncate(40);
    findings
}

fn collect_shallow_findings<'a>(
    source_rels: &'a [String], arch: &'a ArchGraphRef<'_>,
) -> impl Iterator<Item = String> + 'a {
    source_rels.iter().filter_map(|rel| {
        let m = arch.metrics.get(rel).filter(|m| m.loc > 0)?;
        let pr = if m.total_fns > 0 { (m.passthrough_fns as f64) / (m.total_fns as f64) } else { 0.0 };
        let trivial = !is_barrel_file(rel) && m.loc <= 30 && m.total_fns >= 1
            && m.exported_symbols >= 1 && m.passthrough_fns == m.total_fns;
        let dense = m.exported_symbols >= 4 && m.total_fns >= 3
            && (m.exported_symbols as f64) / (m.loc as f64) > 0.20 && pr >= 0.60;
        (trivial || dense).then(|| format!(
            "[shallow-module] {rel}: {} exports in {} LOC ({:.0}% passthrough wrappers)",
            m.exported_symbols, m.loc, pr * 100.0
        ))
    })
}

// ============================================================================
// 3. Inverted-Posting Test Matcher
// ============================================================================

type StrIndexMap<'a> = FxHashMap<&'a str, Vec<usize>>;

fn extend_hits(map: &StrIndexMap<'_>, key: &str, hit_set: &mut FxHashSet<usize>) {
    if let Some(srcs) = map.get(key) { hit_set.extend(srcs.iter().copied()); }
}

fn match_single_test_stem(
    test_stem: &str, tidx: usize, stem_to_srcs: &StrIndexMap<'_>,
    matched_by_src: &mut [BTreeSet<usize>],
) {
    let mut hit = |base: &str| {
        if let Some(srcs) = stem_to_srcs.get(base) {
            for &sidx in srcs { matched_by_src[sidx].insert(tidx); }
        }
    };
    if let Some(base) = test_stem.strip_prefix("test_") { hit(base); }
    for suffix in ["_test", ".test", ".spec"] {
        if let Some(base) = test_stem.strip_suffix(suffix) { hit(base); }
    }
    for (pos, byte) in test_stem.bytes().enumerate() {
        if byte == b'_' && pos + 1 < test_stem.len() { hit(&test_stem[pos + 1..]); }
    }
}

fn collect_unmatched_lookups<'a>(
    source_files: &'a [String], pymod_to_src: &'a FxHashMap<String, Vec<usize>>,
    matched_by_src: &[BTreeSet<usize>],
) -> (StrIndexMap<'a>, StrIndexMap<'a>) {
    let (mut needed_stems, mut needed_pymods): (StrIndexMap<'a>, StrIndexMap<'a>) = Default::default();
    for (sidx, src) in source_files.iter().enumerate() {
        if !matched_by_src[sidx].is_empty() { continue; }
        let stem = stem_of(Path::new(src));
        if stem.len() >= 3 { needed_stems.entry(stem).or_default().push(sidx); }
    }
    for (pymod, srcs) in pymod_to_src {
        let unmatched: Vec<usize> = srcs.iter().copied()
            .filter(|&sidx| matched_by_src[sidx].is_empty() && stem_of(Path::new(&source_files[sidx])).len() >= 3)
            .collect();
        if !unmatched.is_empty() { needed_pymods.insert(pymod.as_str(), unmatched); }
    }
    (needed_stems, needed_pymods)
}

#[inline]
fn take_ident_end(bytes: &[u8], mut pos: usize, allow_dot: bool) -> usize {
    while pos < bytes.len() && (ident_byte(bytes[pos]) || (allow_dot && bytes[pos] == b'.')) { pos += 1; }
    pos
}

fn record_stem_at(
    body: &str, start: usize, needed_stems: &StrIndexMap<'_>, hit_set: &mut FxHashSet<usize>,
) -> usize {
    let end = take_ident_end(body.as_bytes(), start, false);
    if end > start { extend_hits(needed_stems, &body[start..end], hit_set); }
    end
}

fn scan_ident_or_pymod(
    body: &str, bytes: &[u8], start: usize, needed_stems: &StrIndexMap<'_>,
    needed_pymods: &StrIndexMap<'_>, hit_set: &mut FxHashSet<usize>,
) -> usize {
    let end = take_ident_end(bytes, start, true);
    let span = &body[start..end];
    if span.contains('.') {
        extend_hits(needed_pymods, span, hit_set);
        let mut dot_scan = span;
        while let Some((head, _)) = dot_scan.rsplit_once('.') {
            extend_hits(needed_pymods, head, hit_set);
            dot_scan = head;
        }
        return end;
    }
    if matches!(span, "import" | "from") {
        let skip = bytes[end..]
            .iter()
            .take_while(|&&b| b == b' ')
            .count();
        record_stem_at(body, end + skip, needed_stems, hit_set);
    }
    end
}

fn scan_test_body_hits(
    body: &str, needed_stems: &StrIndexMap<'_>, needed_pymods: &StrIndexMap<'_>,
) -> Vec<usize> {
    let (mut hit_set, bytes, mut pos) = (FxHashSet::default(), body.as_bytes(), 0usize);
    while pos < bytes.len() {
        match bytes[pos] {
            b'/' if pos + 1 < bytes.len() && ident_byte(bytes[pos + 1]) => {
                pos = record_stem_at(body, pos + 1, needed_stems, &mut hit_set);
            }
            b if b.is_ascii_alphabetic() || b == b'_' => {
                pos = scan_ident_or_pymod(body, bytes, pos, needed_stems, needed_pymods, &mut hit_set);
            }
            _ => pos += 1,
        }
    }
    hit_set.into_iter().collect()
}

/// Inverted-posting test matcher replacing the O(|S|*|T|) body scan with zero retained test bodies.
pub fn match_tests_inverted(
    repo_root: &Path, source_files: &[String], test_files: &[String],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> HashMap<String, Vec<String>> {
    let mut stem_to_srcs: StrIndexMap<'_> =
        FxHashMap::with_capacity_and_hasher(source_files.len(), Default::default());
    let mut pymod_to_src: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    let mut matched_by_src: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); source_files.len()];

    for (sidx, src) in source_files.iter().enumerate() {
        let stem = stem_of(Path::new(src));
        if !stem.is_empty() { stem_to_srcs.entry(stem).or_default().push(sidx); }
        if let Some(pymod) = src.strip_suffix(".py") {
            pymod_to_src.entry(pymod.replace('/', ".")).or_default().push(sidx);
        }
    }
    for (tidx, tf) in test_files.iter().enumerate() {
        match_single_test_stem(stem_of(Path::new(tf)), tidx, &stem_to_srcs, &mut matched_by_src);
    }

    let (needed_stems, needed_pymods) =
        collect_unmatched_lookups(source_files, &pymod_to_src, &matched_by_src);
    if !needed_stems.is_empty() || !needed_pymods.is_empty() {
        let postings = par_map(test_files, |tf| {
            read_text(repo_root, tf, preloaded)
                .map_or_else(Vec::new, |txt| scan_test_body_hits(&txt, &needed_stems, &needed_pymods))
        });
        for (tidx, hits) in postings.into_iter().enumerate() {
            for sidx in hits { matched_by_src[sidx].insert(tidx); }
        }
    }

    source_files.iter().enumerate().map(|(sidx, src)| {
        let mut tests: Vec<String> = matched_by_src[sidx].iter()
            .map(|&t| test_files[t].clone())
            .collect();
        tests.sort();
        (src.clone(), tests)
    }).collect()
}

// 4. Streaming Git Co-Change NPMI + Worktree HEAD-OID Cache

pub const MAX_GIT_COMMITS: usize = 2500;
pub const BULK_COMMIT_LIMIT: usize = 40;
pub const MIN_COCHANGE_SUPPORT: u32 = 3;
const GENOME_CACHE_VERSION: u8 = 1;
const GIT_LOG_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CochangePair { pub file_a: String, pub file_b: String, pub npmi: f64, pub commits: u32 }

#[derive(Debug, Clone, Default)]
pub struct GenomeSummary { pub pairs: Vec<CochangePair>, pub file_commits: BTreeMap<String, u32> }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenomeCache {
    version: u8, head_oid: String, pairs: Vec<CochangePair>, file_commits: BTreeMap<String, u32>,
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

fn resolve_rel_dir(base: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw.trim());
    if path.is_absolute() { path } else { base.join(path) }
}

fn find_packed_ref_oid(common_dir: &Path, ref_name: &str) -> Option<String> {
    let raw = std::fs::read_to_string(common_dir.join("packed-refs")).ok()?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(['#', '^']) { continue; }
        let mut parts = trimmed.split_whitespace();
        let (Some(oid), Some(name)) = (parts.next(), parts.next()) else { continue };
        if name == ref_name && is_valid_oid(oid) { return Some(oid.to_string()); }
    }
    None
}

/// Pure-Rust zero-subprocess git `HEAD` OID resolver (dirs, worktrees, detached HEAD, `packed-refs`).
pub fn resolve_git_head_oid(repo_root: &Path) -> Option<String> {
    let dot_git = repo_root.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else if dot_git.is_file() {
        let ptr = std::fs::read_to_string(&dot_git).ok()?;
        resolve_rel_dir(repo_root, ptr.trim().strip_prefix("gitdir:")?)
    } else {
        return None;
    };

    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head_trim = head.trim();
    let Some(ref_name) = head_trim.strip_prefix("ref:").map(str::trim) else {
        return is_valid_oid(head_trim).then(|| head_trim.to_string());
    };
    if let Some(oid) = read_hex_oid(&git_dir.join(ref_name)) { return Some(oid); }
    let common_dir = std::fs::read_to_string(git_dir.join("commondir"))
        .ok()
        .map(|raw| resolve_rel_dir(&git_dir, &raw))
        .unwrap_or_else(|| git_dir.clone());
    read_hex_oid(&common_dir.join(ref_name)).or_else(|| find_packed_ref_oid(&common_dir, ref_name))
}

#[derive(Default)]
struct NpmiCounts {
    arena: PathArena,
    file_commits: Vec<u32>,
    churn_commits: Vec<u32>,
    co_counts: FxHashMap<u64, u32>,
    co_decay: FxHashMap<u64, f64>,
    valid_commits: u32,
}

impl NpmiCounts {
    fn intern_path(&mut self, path: &str) -> usize {
        let id = self.arena.intern(path) as usize;
        if id == self.file_commits.len() {
            self.file_commits.push(0);
            self.churn_commits.push(0);
        }
        id
    }

    fn record_commit(
        &mut self, raw: &[String], live_files: Option<&FxHashSet<&str>>,
        commit_ids: &mut Vec<usize>, decay_w: f64,
    ) {
        if raw.len() >= BULK_COMMIT_LIMIT { return; }
        commit_ids.clear();
        for path in raw {
            let skip = is_ignored_cochange_path(path) || live_files.is_some_and(|l| !l.contains(path.as_str()));
            if !skip { commit_ids.push(self.intern_path(path)); }
        }
        commit_ids.sort_unstable();
        commit_ids.dedup();
        for &id in commit_ids.iter() { self.churn_commits[id] = self.churn_commits[id].saturating_add(1); }
        if commit_ids.len() < 2 { return; }

        self.valid_commits += 1;
        for &id in commit_ids.iter() { self.file_commits[id] += 1; }
        self.record_commit_pairs(commit_ids, decay_w);
    }

    fn record_commit_pairs(&mut self, commit_ids: &[usize], decay_w: f64) {
        for (i, &a) in commit_ids.iter().enumerate() {
            for &b in &commit_ids[i + 1..] {
                let key = ((a as u64) << 32) | (b as u64);
                *self.co_counts.entry(key).or_insert(0) += 1;
                *self.co_decay.entry(key).or_insert(0.0) += decay_w;
            }
        }
    }

    fn finalize(self, min_support: u32) -> GenomeSummary { build_genome_summary(self, min_support) }
}

fn build_genome_summary(counts: NpmiCounts, min_support: u32) -> GenomeSummary {
    let n_f = counts.valid_commits.max(1) as f64;
    let mut ranked: Vec<(f64, CochangePair)> = Vec::new();
    for (&key, &c_xy) in &counts.co_counts {
        let Some(pair) = score_npmi_pair(&counts, key, c_xy, min_support, n_f) else { continue };
        let avg = counts.co_decay.get(&key).copied().unwrap_or(c_xy as f64) / (c_xy as f64);
        ranked.push((pair.npmi * (0.85 + 0.15 * avg.clamp(0.0, 1.0)), pair));
    }

    ranked.sort_by(|(sa, a), (sb, b)| {
        sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.commits.cmp(&a.commits))
            .then_with(|| a.file_a.cmp(&b.file_a))
            .then_with(|| a.file_b.cmp(&b.file_b))
    });
    let pairs = ranked.into_iter().map(|(_, p)| p).collect();
    let file_commits = counts.arena.into_paths().into_iter()
        .zip(counts.churn_commits).filter(|(_, cnt)| *cnt > 0).collect();
    GenomeSummary { pairs, file_commits }
}

fn score_npmi_pair(
    counts: &NpmiCounts, key: u64, c_xy: u32, min_support: u32, n_f: f64,
) -> Option<CochangePair> {
    if c_xy < min_support { return None; }
    let (idx_a, idx_b) = ((key >> 32) as usize, (key & 0xffff_ffff) as usize);
    let p_xy = (c_xy as f64) / n_f;
    let (p_x, p_y) = ((counts.file_commits[idx_a] as f64) / n_f, (counts.file_commits[idx_b] as f64) / n_f);
    if p_x <= 0.0 || p_y <= 0.0 || p_xy <= 0.0 { return None; }
    let pmi = (p_xy / (p_x * p_y)).ln();
    let npmi = (pmi / (-p_xy.ln()).max(1e-9)).clamp(-1.0, 1.0);
    if npmi <= 0.05 { return None; }
    let (pa, pb) = (counts.arena.path(idx_a), counts.arena.path(idx_b));
    let (file_a, file_b) = if pa <= pb { (pa.into(), pb.into()) } else { (pb.into(), pa.into()) };
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
    let Some(first) = cols.next() else { return };
    if status.starts_with('R') && let Some(new_p) = cols.next() {
        let (old_n, new_n) = (first.replace('\\', "/"), new_p.replace('\\', "/"));
        let canon = rename_map.get(&new_n).cloned().unwrap_or(new_n);
        rename_map.insert(old_n, canon.clone());
        current.push(canon);
    } else {
        let norm = first.replace('\\', "/");
        current.push(rename_map.get(&norm).cloned().unwrap_or(norm));
    }
}

fn stream_git_cochange(
    repo_root: &Path, live_opt: Option<&FxHashSet<&str>>,
) -> Option<GenomeSummary> {
    let mut child = Command::new("git")
        .args(["log", "--no-merges", "--name-status", "-M", "-n", "2500", "--format=COMMIT %ct"])
        .current_dir(repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.wait();
        return None;
    };

    let mut reader = BufReader::with_capacity(64 * 1024, stdout);
    let counts = read_git_log_commits(&mut reader, live_opt);
    drop(reader);
    if matches!(child.wait_timeout(Duration::from_secs(1)), Ok(None)) {
        let _ = child.kill();
        let _ = child.wait();
    }
    Some(counts.finalize(MIN_COCHANGE_SUPPORT))
}

fn read_git_log_commits<R: BufRead>(reader: &mut R, live_opt: Option<&FxHashSet<&str>>) -> NpmiCounts {
    let now_secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let started = Instant::now();
    let (mut line, mut current) = (String::new(), Vec::with_capacity(BULK_COMMIT_LIMIT));
    let (mut commit_ids, mut rename_map) = (Vec::with_capacity(BULK_COMMIT_LIMIT), FxHashMap::default());
    let (mut counts, mut current_decay, mut commit_count) = (NpmiCounts::default(), 1.0f64, 0usize);

    while let Ok(n) = reader.read_line(&mut line) {
        if n == 0 || started.elapsed() >= GIT_LOG_TIMEOUT { break; }
        let trimmed = line.trim();
        let Some(ts_str) = trimmed.strip_prefix("COMMIT ") else {
            if !trimmed.is_empty() { parse_git_status_line(trimmed, &mut rename_map, &mut current); }
            line.clear();
            continue;
        };
        if !current.is_empty() {
            counts.record_commit(&current, live_opt, &mut commit_ids, current_decay);
            current.clear();
        }
        commit_count += 1;
        if commit_count > MAX_GIT_COMMITS { break; }
        current_decay = ts_str.trim().parse::<u64>()
            .map_or(1.0, |ts| recency_decay(now_secs.saturating_sub(ts)));
        line.clear();
    }

    if !current.is_empty() { counts.record_commit(&current, live_opt, &mut commit_ids, current_decay); }
    counts
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

    let Some(summary) = stream_git_cochange(repo_root, (!live.is_empty()).then_some(&live)) else {
        return GenomeSummary::default();
    };
    let entry = GenomeCache {
        version: GENOME_CACHE_VERSION,
        head_oid,
        pairs: summary.pairs[..summary.pairs.len().min(200)].to_vec(),
        file_commits: summary.file_commits.clone(),
    };
    let _ = write_zstd(&cache_path, &entry);
    summary
}

pub fn format_genome_twins(summary: &GenomeSummary, limit: usize) -> Vec<String> {
    summary.pairs.iter().take(limit).map(|p| {
        format!("{} <-> {} (npmi={:.2}, commits={})", p.file_a, p.file_b, p.npmi, p.commits)
    }).collect()
}

#[cfg(test)]
struct TarjanSt { idx: Vec<u32>, low: Vec<u32>, on: Vec<bool>, stk: Vec<u32>, next: u32, comps: Vec<Vec<u32>> }

#[cfg(test)]
fn tarjan_dfs(v: usize, g: &CsrGraph, st: &mut TarjanSt) {
    (st.idx[v], st.low[v]) = (st.next, st.next);
    st.next += 1;
    st.stk.push(v as u32);
    st.on[v] = true;
    let range = g.offsets[v] as usize..g.offsets[v + 1] as usize;
    for &w_u32 in &g.targets[range] {
        let w = w_u32 as usize;
        if st.idx[w] == 0 {
            tarjan_dfs(w, g, st);
            st.low[v] = st.low[v].min(st.low[w]);
        } else if st.on[w] {
            st.low[v] = st.low[v].min(st.idx[w]);
        }
    }
    if st.low[v] != st.idx[v] { return; }
    let mut comp = Vec::new();
    while let Some(w) = st.stk.pop() {
        st.on[w as usize] = false;
        comp.push(w);
        if w == v as u32 { break; }
    }
    st.comps.push(comp);
}

#[cfg(test)]
fn build_random_scc_edges(n: usize, mult: usize, rng: &mut crate::testutil::TestRng) -> Vec<(u32, u32)> {
    let mut edges: Vec<(u32, u32)> = (0..n * mult)
        .map(|_| (rng.next_usize(n) as u32, rng.next_usize(n) as u32))
        .collect();
    for base in (0..n.saturating_sub(4)).step_by(17) {
        let b = base as u32;
        edges.extend([(b, b + 1), (b + 1, b + 2), (b + 2, b), (b + 1, b + 3), (b + 3, b + 1)]);
    }
    edges
}

#[cfg(test)]
fn build_npmi_fixture() -> Vec<Vec<String>> {
    let mut bulk: Vec<String> = vec!["src/hub.rs".into(), "Cargo.lock".into()];
    bulk.extend((0..45).map(|i| format!("src/bulk_{i}.rs")));
    let mut commits = vec![bulk];
    for c in 0..60 {
        let noise = format!("src/noise_{}.rs", c % 15);
        let mut files = vec!["src/hub.rs".into(), noise, "Cargo.lock".into(), "pnpm-lock.yaml".into()];
        if c % 5 == 0 { files.extend(["src/route.rs".into(), "src/schema.rs".into()]); }
        if c % 6 == 1 { files.extend(["src/auth.rs".into(), "src/session.rs".into()]); }
        commits.push(files);
    }
    commits
}

#[cfg(test)]
fn verify_proof_tiers() {
    let empty = RefWitness::default();
    assert_eq!(classify_symbol(0, empty, 0, DEF_VIS_INTERNAL), Some(ProofTier::T1ProvenDead));
    assert_eq!(classify_symbol(0, empty, 0, 0), Some(ProofTier::T2LikelyDead));
    assert_eq!(classify_symbol(0, empty, 1, 0), Some(ProofTier::T3DynamicOrTestOnly));
    assert_eq!(classify_symbol(0, empty, 0, BOUNDARY_DYNAMIC_REF), Some(ProofTier::T3DynamicOrTestOnly));

    let t4_flags = [
        BOUNDARY_FFI_NO_MANGLE, BOUNDARY_CTOR_INVENTORY, BOUNDARY_MACRO_DERIVE,
        BOUNDARY_PY_ALL, BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_ENTRYPOINT_OR_GLOB,
    ];
    for flag in t4_flags {
        assert_eq!(classify_symbol(0, empty, 0, flag), Some(ProofTier::T4ExternalOrMacro));
    }

    let mut live = RefWitness::default();
    live.record_prod(9);
    assert_eq!(classify_symbol(0, live, 0, 0), None);
}

#[cfg(test)]
fn verify_worktree_head_oid() {
    let fake_wt = crate::testutil::TempRepo::new("wt_head");
    let oid = "0123456789abcdef0123456789abcdef01234567";
    fake_wt.write("common/packed-refs", format!("# pack-refs\n{oid} refs/heads/feat/wt\n"));
    fake_wt.write("wt_meta/commondir", "../common\n");
    fake_wt.write("wt_meta/HEAD", "ref: refs/heads/feat/wt\n");
    fake_wt.write("work/.git", format!("gitdir: {}\n", fake_wt.join("wt_meta").display()));
    assert_eq!(resolve_git_head_oid(&fake_wt.join("work")).as_deref(), Some(oid));
}

#[cfg(test)]
fn svec(items: &[&str]) -> Vec<String> { items.iter().map(|&s| s.to_string()).collect() }

#[cfg(test)]
fn mk_metric(loc: usize, exp: usize, fns: usize, pass: usize, pubs: &[&str], reads: &[&str]) -> ModuleMetrics {
    ModuleMetrics { loc, exported_symbols: exp, total_fns: fns, passthrough_fns: pass, pub_fields: svec(pubs), field_reads: svec(reads) }
}

#[cfg(test)]
fn verify_inv_test_match() {
    let dir = crate::testutil::TempRepo::with_files("inv_tests", &[
        ("pkg/alpha.py", "def run_alpha():\n    return 1\n"),
        ("pkg/beta_core.py", "def run_beta():\n    return 2\n"),
        ("pkg/orphan.py", "def iso():\n    return 0\n"),
        ("tests/test_alpha.py", "def test_a(): pass\n"),
        ("tests/test_smoke.py", "from pkg.beta_core import run_beta\ndef test_b(): assert run_beta() == 2\n"),
    ]);
    let srcs = svec(&["pkg/alpha.py", "pkg/beta_core.py", "pkg/orphan.py"]);
    let tests = svec(&["tests/test_alpha.py", "tests/test_smoke.py"]);
    let matched = match_tests_inverted(dir.path(), &srcs, &tests, None);
    let want_a = svec(&["tests/test_alpha.py"]);
    let want_b = svec(&["tests/test_smoke.py"]);
    assert_eq!((&matched["pkg/alpha.py"], &matched["pkg/beta_core.py"]), (&want_a, &want_b));
    assert!(matched["pkg/orphan.py"].is_empty());
}

#[cfg(test)]
fn verify_shallow_and_field_leak() {
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
    let sccs = CsrGraph::from_named_edges(&rels, &edges).0.pearce_scc();
    let arch = ArchGraphRef {
        all_edges: &edges, runtime_edges: &edges, metrics: &metrics,
        defs_by_file: None, refs_by_file: None, sources: None,
    };
    let (ego, findings) = analyze_ego_and_deep_modules(&rels, &sccs, &arch, &BTreeMap::new(), &[]);

    for (needle, want) in [
        ("[module-cycle]", true), (".raw_token", true), ("[shallow-module] src/syntax_stub.rs", true),
        (".line", false), ("src/sub/mod.rs", false),
    ] {
        assert_eq!(findings.iter().any(|f| f.contains(needle)), want, "{needle} in {findings:?}");
    }
    assert_eq!((ego["src/model.rs"].external_field_readers, ego["src/a.rs"].has_module_cycle), (2, true));
}

#[cfg(test)]
fn verify_rs_pure_dto_exemption() {
    let sources = BTreeMap::from([
        ("src/dto.rs".into(), concat!(
            "#[derive(Debug, Clone)]\n",
            "pub struct ReportDto {\n    pub summary_text: String,\n    pub metric_val: usize,\n}\n",
            "impl Default for ReportDto {\n    fn default() -> Self { Self { summary_text: String::new(), metric_val: 0 } }\n}\n",
        ).as_bytes().to_vec()),
        ("src/stateful.rs".into(),
            b"pub struct SessionStore {\n    pub raw_token: String,\n    secret_nonce: u64,\n}\n".to_vec()),
        ("src/engine.rs".into(), concat!(
            "pub struct Engine {\n    pub cache_budget: usize,\n}\n",
            "impl Engine {\n    pub fn recompute(&mut self) { self.cache_budget += 1; }\n}\n",
        ).as_bytes().to_vec()),
    ]);
    let metrics = BTreeMap::from([
        ("src/dto.rs".into(), mk_metric(8, 1, 1, 0, &["summary_text", "metric_val"], &[])),
        ("src/stateful.rs".into(), mk_metric(5, 1, 0, 0, &["raw_token"], &[])),
        ("src/engine.rs".into(), mk_metric(7, 1, 1, 0, &["cache_budget"], &[])),
        ("src/c1.rs".into(), mk_metric(40, 1, 2, 0, &[], &["summary_text", "raw_token", "cache_budget"])),
        ("src/c2.rs".into(), mk_metric(40, 1, 2, 0, &[], &["summary_text", "raw_token", "cache_budget"])),
    ]);
    let rels: Vec<String> = metrics.keys().cloned().collect();
    let edges = vec![
        ("src/c1.rs".into(), "src/dto.rs".into()), ("src/c2.rs".into(), "src/dto.rs".into()),
        ("src/c1.rs".into(), "src/stateful.rs".into()), ("src/c2.rs".into(), "src/stateful.rs".into()),
        ("src/c1.rs".into(), "src/engine.rs".into()), ("src/c2.rs".into(), "src/engine.rs".into()),
    ];
    let arch = ArchGraphRef {
        all_edges: &edges, runtime_edges: &edges, metrics: &metrics,
        defs_by_file: None, refs_by_file: None, sources: Some(&sources),
    };
    let (ego, findings) = analyze_ego_and_deep_modules(&rels, &[], &arch, &BTreeMap::new(), &[]);
    assert_eq!(ego["src/dto.rs"].external_field_readers, 0);
    assert!(!findings.iter().any(|f| f.contains("src/dto.rs")), "unexpected DTO leak: {findings:?}");
    assert!(findings.iter().any(|f| f.contains("[field-leak] src/stateful.rs") && f.contains(".raw_token")));
    assert!(findings.iter().any(|f| f.contains("[field-leak] src/engine.rs") && f.contains(".cache_budget")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hygiene::HygieneScorer;
    use crate::testutil::{TempRepo, TestRng};
    use crate::untangle::{AuditScope, untangle_repo_with_scope};

    fn tarjan_ref(graph: &CsrGraph) -> Vec<Vec<u32>> {
        let n = graph.n;
        let mut st = TarjanSt { idx: vec![0; n], low: vec![0; n], on: vec![false; n], stk: Vec::new(), next: 1, comps: Vec::new() };
        for v in 0..n { if st.idx[v] == 0 { tarjan_dfs(v, graph, &mut st); } }
        st.comps
    }

    fn canon_sccs(mut comps: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        for c in &mut comps { c.sort_unstable(); }
        comps.sort();
        comps
    }

    #[test]
    fn pearce_scc_dense_and_chain() {
        let mut rng = TestRng::new(0x9e37_79b9_7f4a_7c15);
        for &(n, mult) in &[(40usize, 1usize), (120, 3), (250, 6)] {
            let edges = build_random_scc_edges(n, mult, &mut rng);
            let csr = CsrGraph::from_edges(n, &edges);
            assert_eq!(canon_sccs(csr.pearce_scc()), canon_sccs(tarjan_ref(&csr)));
        }

        let n = 1500usize;
        let mut chain: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        chain.push(((n - 1) as u32, 0));
        let comps = CsrGraph::from_edges(n, &chain).pearce_scc();
        assert_eq!((comps.len(), comps[0].len()), (1, n));
    }

    #[test]
    fn csr_shallow_and_field_leak() {
        verify_inv_test_match();
        verify_shallow_and_field_leak();
        verify_rs_pure_dto_exemption();
    }

    #[test]
    fn genome_npmi_hub_and_wipe_guard() {
        let commits = build_npmi_fixture();
        let summary = compute_npmi_commits(&commits, None, MIN_COCHANGE_SUPPORT);
        let has_lock = summary.pairs.iter().any(|p| p.file_a.ends_with(".lock") || p.file_b.ends_with(".lock"));
        assert!(!has_lock);

        let top2: Vec<(&str, &str)> = summary.pairs.iter()
            .take(2)
            .map(|p| (p.file_a.as_str(), p.file_b.as_str()))
            .collect();
        for expected in [("src/route.rs", "src/schema.rs"), ("src/auth.rs", "src/session.rs")] {
            assert!(top2.contains(&expected), "missing {expected:?} in {top2:?}");
        }

        assert_eq!((check_wipe_guard(100, 19), check_wipe_guard(100, 20), check_wipe_guard(40, 2)), (true, false, false));
        let (d0, d14) = (recency_decay(0), recency_decay(14 * 86_400));
        assert!(d0 > 0.99 && d14 < 0.40);
    }

    #[test]
    fn genome_head_cache_and_worktree() {
        verify_worktree_head_oid();

        let repo = TempRepo::git("genome_cache", "app", &[
            ("src/a.rs", "pub fn alpha_one() -> usize { 1 }\n"), ("src/b.rs", "pub fn beta_two() -> usize { 2 }\n"),
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
    fn ref_witness_and_proof_lattice() {
        assert_eq!(std::mem::size_of::<RefWitness>(), 16);
        let mut w = RefWitness::default();
        assert!(!w.has_cross_file_prod_caller(0));
        w.record_prod(3);
        w.record_prod(3);
        assert_eq!(w.prod_files_sat, 1);
        assert!(!w.has_cross_file_prod_caller(3) && w.has_cross_file_prod_caller(4));

        w.record_prod(7);
        assert_eq!(w.prod_files_sat, 2);
        assert!(w.has_cross_file_prod_caller(3) && w.has_cross_file_prod_caller(7));
        for fid in [10, 10, 11] { w.record_test(fid); }
        assert_eq!(w.test_count(), 2);

        verify_proof_tiers();
    }
}
