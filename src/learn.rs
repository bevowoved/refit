use crate::lex::find_ascii_ci;
use crate::platform::write_atomic;
use crate::stem::{SymbolGraph, load_symbols};
use crate::types::LearnRule;
use anyhow::Result;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHasher};
use serde_json::{Value, json};
use smallvec::SmallVec;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const MAX_RULES: usize = 16;
const REWIND_HIT_THRESHOLD: u32 = 2;

/// Amortized break-even threshold in microseconds per file for activating `InvertedRewindIndex`.
pub const MIN_KERNEL_COST_US: u64 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Durability { Low, High }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum RewindTrigger {
    #[value(name = "new_dep_advisory", aliases = ["advisory", "dep"])]
    NewDepAdvisory,
    #[value(name = "new_smell_pattern", aliases = ["smell"])]
    NewSmellPattern,
    #[value(name = "new_invariant_rule", aliases = ["invariant"])]
    NewInvariantRule,
    #[value(name = "abi_changed", aliases = ["abi"])]
    AbiChanged,
}

impl RewindTrigger {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "advisory" | "dep" | "new_dep_advisory" => Self::NewDepAdvisory,
            "invariant" | "new_invariant_rule" => Self::NewInvariantRule,
            "abi" | "abi_changed" => Self::AbiChanged,
            _ => Self::NewSmellPattern,
        }
    }
    #[inline] pub fn include_1hop_rev_deps(self) -> bool { matches!(self, Self::NewDepAdvisory | Self::AbiChanged) }
    #[inline] pub fn skip_high_durability(self) -> bool { !matches!(self, Self::NewDepAdvisory | Self::NewInvariantRule) }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NewDepAdvisory => "new_dep_advisory", Self::NewSmellPattern => "new_smell_pattern",
            Self::NewInvariantRule => "new_invariant_rule", Self::AbiChanged => "abi_changed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewindEvent {
    trigger: RewindTrigger, pub token: String, pub token_hash: u32, pub rule_key: String,
}

impl RewindEvent {
    pub fn new(trigger: RewindTrigger, token: String, rule_key: String) -> Self {
        let token_hash = hash_token_u32(&token);
        Self { trigger, token, token_hash, rule_key }
    }

    pub fn trigger(&self) -> RewindTrigger { self.trigger }
}

#[inline] pub fn should_use_index(est_us_per_file: u64) -> bool { est_us_per_file >= MIN_KERNEL_COST_US }

#[inline]
pub fn hash_token_u32(token: &str) -> u32 {
    let mut h = FxHasher::default();
    token.hash(&mut h);
    let v = h.finish();
    (v ^ (v >> 32)) as u32
}

#[inline]
fn bloom_mask(tok: u32) -> u64 {
    let a = (tok.wrapping_mul(0x9E37_79B1) >> 26) & 63;
    let b = (tok.wrapping_mul(0x85EB_CA6B) >> 26) & 63;
    (1u64 << a) | (1u64 << b)
}

#[inline]
fn insert_sorted_unique(list: &mut SmallVec<[u32; 4]>, value: u32) {
    if list.is_empty() || list.last().copied().is_some_and(|last| last < value) {
        list.push(value);
        return;
    }
    if let Err(pos) = list.binary_search(&value) { list.insert(pos, value); }
}

/// Compact inverted signature index + 64-bit Bloom filter + 1-hop reverse-dep adjacency.
#[derive(Debug, Default)]
pub struct InvertedRewindIndex {
    postings: FxHashMap<u32, SmallVec<[u32; 4]>>, bloom: Vec<u64>,
    rev_deps: Vec<SmallVec<[u32; 4]>>, durability: Vec<Durability>,
    paths: Vec<String>, path_to_id: FxHashMap<String, u32>,
}

fn build_rewind_graph(
    snap: &SymbolGraph, durability_by_file: &FxHashMap<&str, Durability>,
) -> InvertedRewindIndex {
    let cap = snap.defs_by_file.len().max(snap.refs_by_file.len()).max(durability_by_file.len());
    let mut idx = InvertedRewindIndex {
        postings: FxHashMap::with_capacity_and_hasher(cap.min(4096), Default::default()),
        bloom: Vec::with_capacity(cap), rev_deps: Vec::with_capacity(cap),
        durability: Vec::with_capacity(cap), paths: Vec::with_capacity(cap),
        path_to_id: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
    };

    for rel in snap.defs_by_file.keys().chain(snap.refs_by_file.keys()).chain(snap.imports_by_file.keys()) {
        let dur = durability_by_file.get(rel.as_str()).copied().unwrap_or(Durability::Low);
        idx.register_file(rel, dur);
    }
    for (&rel, &dur) in durability_by_file { idx.register_file(rel, dur); }

    for (rel, syms) in snap.defs_by_file.iter().chain(snap.refs_by_file.iter()) {
        idx.index_file_syms(rel, syms);
    }
    for (dependent, dependency) in &snap.runtime_edges {
        idx.add_dep_by_path(dependent, dependency);
    }
    idx
}

impl InvertedRewindIndex {
    /// Build index directly from an in-memory `SymbolGraph`.
    pub fn from_symbol_graph(
        snap: &SymbolGraph, durability_by_file: &FxHashMap<&str, Durability>,
    ) -> Self {
        build_rewind_graph(snap, durability_by_file)
    }

    fn index_file_syms(&mut self, rel: &str, syms: &[String]) {
        let Some(&fid) = self.path_to_id.get(rel) else { return };
        let mask = &mut self.bloom[fid as usize];
        for sym in syms.iter().filter(|s| s.len() >= 3) {
            let tok = hash_token_u32(sym);
            *mask |= bloom_mask(tok);
            insert_sorted_unique(self.postings.entry(tok).or_default(), fid);
        }
    }

    pub fn register_file(&mut self, path: &str, durability: Durability) -> u32 {
        if let Some(&existing) = self.path_to_id.get(path) {
            self.durability[existing as usize] = durability;
            return existing;
        }

        let file_id = self.paths.len() as u32;
        self.paths.push(path.to_string());
        self.path_to_id.insert(path.to_string(), file_id);
        self.bloom.push(0);
        self.rev_deps.push(SmallVec::new());
        self.durability.push(durability);
        file_id
    }

    pub fn add_dep_by_path(&mut self, dependent: &str, dependency: &str) {
        let (Some(&user_id), Some(&dep_id)) =
            (self.path_to_id.get(dependent), self.path_to_id.get(dependency))
        else { return };
        if user_id != dep_id { insert_sorted_unique(&mut self.rev_deps[dep_id as usize], user_id); }
    }

    pub fn file_count(&self) -> usize { self.bloom.len() }

    fn is_skipped(&self, file_id: u32, skip_high: bool) -> bool {
        skip_high && self.durability.get(file_id as usize).is_some_and(|&d| d == Durability::High)
    }

    pub fn query_affected(
        &self, trigger_token: u32, include_1hop_rev_deps: bool, skip_high_durability: bool,
    ) -> Vec<u32> {
        let Some(cands) = self.postings.get(&trigger_token) else { return Vec::new() };
        let req = bloom_mask(trigger_token);
        let direct: Vec<u32> = cands.iter().copied()
            .filter(|&fid| self.bloom.get(fid as usize).is_some_and(|&m| (m & req) == req))
            .filter(|&fid| !self.is_skipped(fid, skip_high_durability))
            .collect();
        let mut affected = direct.clone();
        if include_1hop_rev_deps { self.append_1hop_rev(&direct, skip_high_durability, &mut affected); }
        affected.sort_unstable();
        affected.dedup();
        affected
    }

    fn append_1hop_rev(&self, direct: &[u32], skip_high: bool, out: &mut Vec<u32>) {
        let rev = direct.iter().flat_map(|&fid| &self.rev_deps[fid as usize]).copied();
        out.extend(rev.filter(|&fid| !self.is_skipped(fid, skip_high)));
    }

    pub fn query_event_paths(&self, event: &RewindEvent, est_us_per_file: u64) -> Vec<String> {
        if !should_use_index(est_us_per_file) || self.file_count() == 0 { return Vec::new(); }
        let ids = self.query_affected(
            event.token_hash,
            event.trigger.include_1hop_rev_deps(),
            event.trigger.skip_high_durability(),
        );
        ids.into_iter()
            .filter_map(|fid| self.paths.get(fid as usize).cloned())
            .collect()
    }
}

/// Query learned rules and optional SymbolGraph rewind impact.
pub fn query_rewind(repo_root: &Path, rewind_token: Option<&str>, trigger: RewindTrigger) -> Result<Value> {
    query_rewind_graph(repo_root, None, rewind_token, trigger)
}

/// Query learned rules and rewind impact reusing a cached `SymbolGraph`.
pub fn query_rewind_graph(
    repo_root: &Path, warm_graph: Option<&SymbolGraph>, rewind_token: Option<&str>, trigger: RewindTrigger,
) -> Result<Value> {
    let rules = LearnStore::open(repo_root)?.rules_snapshot();
    let Some(token) = rewind_token.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(json!({ "rules": rules, "rules_count": rules.len() }));
    };
    let owned;
    let snap = match warm_graph {
        Some(g) => g,
        None => { owned = load_symbols(repo_root, None); &owned }
    };

    let idx = InvertedRewindIndex::from_symbol_graph(snap, &FxHashMap::default());
    let ev = RewindEvent::new(trigger, token.to_string(), "manual_rewind".into());
    let affected = idx.query_event_paths(&ev, MIN_KERNEL_COST_US);

    Ok(json!({
        "rules": rules, "rules_count": rules.len(),
        "rewind": {
            "token": token, "trigger": trigger.as_str(),
            "indexed_files": idx.file_count(), "affected_files": affected,
        }
    }))
}

type DiagRow = (&'static str, &'static [&'static str], &'static str, &'static str);

const DIAG_RULES: &[DiagRow] = &[
    ("", &["abi"], "abi_pin", "Preserve exact exported signatures unless allow_abi_change is enabled."),
    ("asrf_apply", &["search block did not match", "ambiguous search"], "asrf_exact_context", "Copy <<<SEARCH>>> lines verbatim with 2-3 surrounding context lines."),
    ("", &["b006", "mutable default"], "ruff_b006", "Replace mutable default args ([]/{}) with None and init inside body."),
    ("", &["f401", "unused import"], "ruff_f401", "Remove every unused import cleanly."),
    ("", &["f841", "assigned to but never used"], "ruff_f841", "Remove unused local variables or bind intentionally with `_`."),
    ("", &["e501", "e702", "line too long", "dense_line"], "line_len", "Keep lines concise and split packed statements across lines."),
    ("", &["deep_nesting"], "deep_nesting", "Flatten nested pyramids using guard clauses and small helpers."),
    ("", &["duplicate_block", "repetitive_structure"], "dedup", "Extract repeated blocks into helpers and prefer lookup tables."),
    ("", &["cramped_layout"], "cramped_layout", "Separate logical phases and top-level definitions with blank lines."),
    ("gate3_ty", &["ty:"], "ty_types", "Add explicit type annotations and narrow Optional values before use."),
    ("gate3_5_invariants", &["builtin_no_secrets", "secret literal"], "invariants_security", "Never hardcode credentials or violate .refit/invariants.yml boundaries."),
    ("gate1_tests", &["assert", "pytest"], "test_boundary", "Trace exact boundary conditions (<= vs <, clamping, None guards) carefully."),
    ("gate4_kane", &["kane", "agent_info", "assert", "verify", "console", "locator", "bug_740", "policy"], "kane_live_fail", "Inspect Kane .evidence failure.yaml + console log, fix the UI/route regression, and re-verify with refit check --kane."),
];

/// Map a Kane failure diagnostic to suspected `(file, line, symbol)` spans & rewind-affected files.
pub fn rewind_kane_tokens(
    repo_root: &Path, warm_graph: Option<&SymbolGraph>, diagnostic: &str,
) -> (Vec<(String, usize, String)>, Vec<String>) {
    if let Ok(store) = LearnStore::open(repo_root) {
        let _ = store.record_failure("gate4_kane", "*", diagnostic);
    }
    let owned;
    let snap = match warm_graph {
        Some(g) => g,
        None => { owned = load_symbols(repo_root, None); &owned }
    };
    let mut tokens: Vec<String> = Vec::new();
    for raw in diagnostic.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))) {
        let clean = raw.trim_matches('-');
        if clean.len() >= 4 && clean.len() <= 48 && !tokens.iter().any(|t| t == clean) {
            tokens.push(clean.to_string());
        }
        if tokens.len() >= 24 { break; }
    }
    let mut spans: Vec<(String, usize, String)> = Vec::new();
    for (file, defs) in &snap.defs_by_file {
        for sym in defs {
            if tokens.iter().any(|t| t == sym || (sym.len() >= 5 && t.eq_ignore_ascii_case(sym)))
                && !spans.iter().any(|(f, _, s)| f == file && s == sym)
            {
                spans.push((file.clone(), 1, sym.clone()));
            }
        }
        if spans.len() >= 8 { break; }
    }
    let idx = InvertedRewindIndex::from_symbol_graph(snap, &FxHashMap::default());
    let mut affected: Vec<String> = Vec::new();
    for tok in spans.iter().map(|(_, _, s)| s.as_str()).chain(tokens.iter().map(String::as_str)).take(8) {
        let ev = RewindEvent::new(RewindTrigger::AbiChanged, tok.to_string(), "kane_live_fail".into());
        for f in idx.query_event_paths(&ev, MIN_KERNEL_COST_US) {
            if !affected.contains(&f) { affected.push(f); }
            if affected.len() >= 12 { break; }
        }
    }
    (spans, affected)
}

pub struct LearnStore {
    path: PathBuf,
    rules: Mutex<Vec<LearnRule>>,
}

impl LearnStore {
    pub fn open(repo_root: &Path) -> Result<Self> {
        let dir = repo_root.join(".refit");
        fs::create_dir_all(&dir)?;
        let path = dir.join("learn.jsonl");
        let loaded = fs::read_to_string(&path).map_or_else(|_| Vec::new(), |raw| {
            raw.lines()
                .filter_map(|l| serde_json::from_str::<LearnRule>(l.trim()).ok())
                .filter(|r| !r.key.ends_with("_generic"))
                .collect()
        });
        Ok(Self { path, rules: Mutex::new(loaded) })
    }

    pub fn rules_snapshot(&self) -> Vec<LearnRule> { self.rules.lock().clone() }

    fn with_rules_mut<R>(&self, f: impl FnOnce(&mut Vec<LearnRule>) -> (R, bool)) -> R {
        let mut guard = self.rules.lock();
        let (out, dirty) = f(&mut guard);
        let snap = dirty.then(|| guard.clone());
        drop(guard);
        if let Some(rules) = snap { let _ = self.flush_snapshot(&rules); }
        out
    }

    pub fn record_failure(&self, gate: &str, dir_prefix: &str, raw_diagnostic: &str) -> Option<RewindEvent> {
        let (key, advice) = classify_diag(gate, raw_diagnostic)?;
        let hits = self.with_rules_mut(|rules| (upsert_rule(rules, &key, gate, dir_prefix, advice), true));
        if hits != REWIND_HIT_THRESHOLD { return None; }
        let trigger = trigger_for_rule(&key, raw_diagnostic)?;
        Some(RewindEvent::new(trigger, extract_trigger_token(&key, raw_diagnostic), key))
    }

    pub fn credit_keys(&self, dir_prefix: &str, applied_keys: &[String]) {
        if applied_keys.is_empty() { return; }
        self.with_rules_mut(|rules| {
            let mut changed = false;
            for k in applied_keys { changed |= credit_one(rules, dir_prefix, k); }
            ((), changed)
        });
    }

    pub fn rule_suffix(&self, dir_prefix: &str) -> (String, Vec<String>) {
        format_rule_suffix(&self.rules.lock(), dir_prefix)
    }

    pub fn stats(&self) -> (usize, u32) {
        let guard = self.rules.lock();
        (guard.len(), guard.iter().map(|r| r.helpful).sum())
    }

    fn flush_snapshot(&self, rules: &[LearnRule]) -> Result<()> {
        let mut buf = String::new();
        for r in rules {
            buf.push_str(&serde_json::to_string(r)?);
            buf.push('\n');
        }
        write_atomic(&self.path, buf.as_bytes())
    }
}

fn format_rule_suffix(rules: &[LearnRule], dir_prefix: &str) -> (String, Vec<String>) {
    let mut cands: Vec<&LearnRule> = rules.iter()
        .filter(|r| !r.key.ends_with("_generic") && (r.dir_prefix == dir_prefix || r.hits >= 2))
        .collect();
    cands.sort_by(|a, b| b.hits.cmp(&a.hits).then_with(|| a.key.cmp(&b.key)));
    if cands.is_empty() { return (String::new(), Vec::new()); }

    let mut out = String::from("\nLearned repo rules from prior waves (obey strictly):\n");
    let mut keys = Vec::new();
    for rule in cands.into_iter().take(3) {
        let line = format!("- [{}] {}\n", rule.gate, rule.advice);
        if out.len() + line.len() > 420 { break; }
        out.push_str(&line);
        keys.push(rule.key.clone());
    }
    (out, keys)
}

fn upsert_rule(rules: &mut Vec<LearnRule>, key: &str, gate: &str, dir_prefix: &str, advice: String) -> u32 {
    if let Some(rule) = rules.iter_mut().find(|r| r.key == key && r.dir_prefix == dir_prefix) {
        rule.hits += 1;
        return rule.hits;
    }
    rules.push(LearnRule {
        key: key.to_string(), gate: gate.to_string(), dir_prefix: dir_prefix.to_string(),
        advice, hits: 1, helpful: 0,
    });
    prune_rules(rules);
    1
}

fn prune_rules(rules: &mut Vec<LearnRule>) {
    if rules.len() <= MAX_RULES {
        return;
    }
    rules.sort_by(|a, b| {
        let score_b = b.hits.saturating_add(b.helpful * 2);
        let score_a = a.hits.saturating_add(a.helpful * 2);
        score_b.cmp(&score_a).then_with(|| a.key.cmp(&b.key))
    });
    rules.truncate(MAX_RULES);
}

fn credit_one(rules: &mut [LearnRule], dir_prefix: &str, key: &str) -> bool {
    let exact = rules.iter().position(|r| r.key == key && r.dir_prefix == dir_prefix);
    let idx = exact.or_else(|| {
        rules.iter()
            .enumerate()
            .filter(|(_, r)| r.key == key && (r.dir_prefix == "*" || r.hits >= 2))
            .max_by_key(|(_, r)| r.hits)
            .map(|(i, _)| i)
    });
    if let Some(i) = idx {
        rules[i].helpful += 1;
        return true;
    }
    false
}

fn trigger_for_rule(key: &str, raw_diagnostic: &str) -> Option<RewindTrigger> {
    let raw_bytes = raw_diagnostic.as_bytes();
    let is_advisory = ["cve-", "advisory", "vulnerable"]
        .iter()
        .any(|n| find_ascii_ci(raw_bytes, n.as_bytes()).is_some());
    if is_advisory {
        return Some(RewindTrigger::NewDepAdvisory);
    }
    match key {
        "abi_pin" | "kane_live_fail" => Some(RewindTrigger::AbiChanged),
        "invariants_security" => Some(RewindTrigger::NewInvariantRule),
        "ruff_b006" | "ruff_f401" | "ruff_f841" | "line_len" | "deep_nesting" | "dedup"
        | "cramped_layout" | "ty_types" => Some(RewindTrigger::NewSmellPattern),
        _ => None,
    }
}

fn extract_trigger_token(key: &str, raw_diagnostic: &str) -> String {
    let mut parts = raw_diagnostic.split('`');
    let _ = parts.next();
    while let Some(span) = parts.next() {
        let _ = parts.next();
        let t = span.trim().trim_end_matches("()");
        if t.len() >= 3 && t.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return t.to_string();
        }
    }
    key.to_string()
}

fn classify_diag(gate: &str, raw: &str) -> Option<(String, String)> {
    let raw_bytes = raw.as_bytes();
    DIAG_RULES
        .iter()
        .find(|&&(rg, needles, _, _)| {
            (!rg.is_empty() && gate == rg)
                || needles.iter().any(|n| find_ascii_ci(raw_bytes, n.as_bytes()).is_some())
        })
        .map(|&(_, _, key, advice)| (key.to_string(), advice.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn open_store(tag: &str) -> (TempRepo, LearnStore) {
        let dir = TempRepo::new(tag);
        let store = LearnStore::open(dir.path()).unwrap();
        (dir, store)
    }

    fn record_batch(store: &LearnStore, dir: &str, items: &[(&str, &str)]) {
        for &(gate, msg) in items { store.record_failure(gate, dir, msg); }
    }

    #[test]
    fn learn_budget_keys_and_credit() {
        let (_dir, store) = open_store("learn_keys");
        record_batch(&store, "pkg/core", &[
            ("gate3_ruff", "B006 mutable default"),
            ("gate3_ty", "ty: invalid optional"),
            ("gate2_hygiene", "deep_nesting L14"),
            ("gate0_abi", "AbiPinViolation"),
        ]);
        let s1 = store.rule_suffix("pkg/core").0;
        let bullets = s1.lines().filter(|l| l.starts_with("- [")).count();
        assert!(s1 == store.rule_suffix("pkg/core").0 && s1.len() <= 420 && bullets <= 3);

        let (_dir2, s2) = open_store("learn_api");
        record_batch(&s2, "pkg/api", &[
            ("asrf_apply", "Hunk #0 SEARCH block did not match `pkg/api/mod_54.py`"),
            ("weird_gate", "transient socket hiccup"),
            ("gate3_ruff", "B006 mutable default"),
        ]);
        let (suffix, keys) = s2.rule_suffix("pkg/api");
        assert!(!suffix.contains("mod_54.py") && !suffix.contains("weird_gate"));
        assert!(keys.contains(&"asrf_exact_context".to_string()) && s2.stats().0 == 2);
        s2.credit_keys("pkg/api", &["asrf_exact_context".to_string()]);
        assert_eq!(s2.stats().1, 1);

        assert!(store.record_failure("gate3_ruff", "pkg/new", "B006 `bad_fn`").is_none());
        let ev = store.record_failure("gate3_ruff", "pkg/new", "B006 `bad_fn`").unwrap();
        assert_eq!((ev.trigger, ev.token.as_str()), (RewindTrigger::NewSmellPattern, "bad_fn"));
    }

    #[test]
    fn learn_max_rules_prune() {
        let (_dir, store) = open_store("learn_prune");
        for i in 0..24 {
            store.record_failure("gate3_ruff", &format!("dir_{i}"), "B006 mutable default");
        }
        assert_eq!(store.stats().0, MAX_RULES);
    }

    #[test]
    fn rewind_symgraph_and_1hop_deps() {
        let repo = TempRepo::with_files(
            "rewind_symgraph",
            &[
                ("src/a.rs", "pub fn legacy_parse() -> i32 { 1 }\n"),
                ("src/b.rs", "use crate::a::legacy_parse;\npub fn mid_call() -> i32 { legacy_parse() }\n"),
                ("src/c.rs", "use crate::b::mid_call;\npub fn top_call() -> i32 { mid_call() }\n"),
                ("src/vendor.rs", "use crate::a::legacy_parse;\npub fn vendor_wrap() -> i32 { legacy_parse() }\n"),
            ],
        );
        let snap = load_symbols(repo.path(), None);

        let dur_map: FxHashMap<&str, Durability> = [
            ("src/a.rs", Durability::High),
            ("src/b.rs", Durability::Low),
            ("src/c.rs", Durability::Low),
            ("src/vendor.rs", Durability::High),
        ]
        .into_iter()
        .collect();

        let idx = InvertedRewindIndex::from_symbol_graph(&snap, &dur_map);
        assert!(idx.file_count() >= 4);

        let ev_smell = RewindEvent::new(RewindTrigger::NewSmellPattern, "legacy_parse".into(), "smell".into());
        assert_eq!(idx.query_event_paths(&ev_smell, MIN_KERNEL_COST_US), vec!["src/b.rs"]);
        assert!(idx.query_event_paths(&ev_smell, MIN_KERNEL_COST_US - 1).is_empty());

        let ev_adv = RewindEvent::new(RewindTrigger::NewDepAdvisory, "legacy_parse".into(), "cve".into());
        let adv_paths = idx.query_event_paths(&ev_adv, MIN_KERNEL_COST_US);
        assert!(adv_paths.contains(&"src/b.rs".to_string()) && adv_paths.contains(&"src/c.rs".to_string()));
        assert!(adv_paths.contains(&"src/vendor.rs".to_string()));

        LearnStore::open(repo.path()).unwrap().record_failure("gate3_ruff", "src", "B006 mutable default");
        let val = query_rewind(repo.path(), Some("legacy_parse"), RewindTrigger::NewDepAdvisory).unwrap();
        assert_eq!((&val["rules_count"], &val["rewind"]["trigger"]), (&json!(1), &json!("new_dep_advisory")));
        let raw_aff = val["rewind"]["affected_files"].as_array().unwrap();
        let affected: Vec<&str> = raw_aff.iter().filter_map(Value::as_str).collect();
        assert!(affected.contains(&"src/b.rs") && affected.contains(&"src/c.rs"));
    }
}
