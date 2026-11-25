pub(crate) mod analysis;
mod parse;
pub(crate) mod sst;

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{mkdir_private, read_zstd, walk_files, write_zstd};
use crate::source::{
    RepoScan, RepoSnapshot, dir_of, entrypoint, ext_of, read_bytes, rel_norm, same_family,
    scan_files, stem_of,
};
use crate::types::DeadSymbol;
use analysis::{
    BOUNDARY_ENTRYPOINT_OR_GLOB, BOUNDARY_FFI_NO_MANGLE, BOUNDARY_MACRO_DERIVE, EdgeKind,
    ModuleMetrics, PathArena, RefWitness, WITNESS_DYNAMIC_REF, WITNESS_EXTERNAL,
    WITNESS_GLOB_REEXPORT, WITNESS_MACRO_ATTR, check_wipe_guard, classify_symbol, hash_symbol,
};
use parse::{FileSymbols, parse_symbols};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const CACHE_VERSION: &str = "0.6.0";

const STEM_SQL: &str = concat!(
    "WITH uniq AS (SELECT name FROM symbols GROUP BY name ",
    "HAVING COUNT(DISTINCT file) = 1 AND LENGTH(name) >= 5) ",
    "SELECT DISTINCT r.file AS src, s.file AS dst ",
    "FROM refs r JOIN symbols s ON r.name = s.name ",
    "JOIN uniq u ON u.name = s.name WHERE r.file != s.file ",
    "AND r.file NOT LIKE 'tests/%' AND r.file NOT LIKE '%/tests/%' ",
    "AND r.file NOT LIKE '%_test.%' AND r.file NOT LIKE '%.test.%' ",
    "AND r.file NOT LIKE '%.spec.%' AND s.file NOT LIKE 'tests/%' ",
    "AND s.file NOT LIKE '%/tests/%' AND s.file NOT LIKE '%_test.%' ",
    "AND s.file NOT LIKE '%.test.%' AND s.file NOT LIKE '%.spec.%' LIMIT 2000"
);

#[derive(Debug, Clone)]
pub struct EdgeScan {
    pub used_stem: bool,
    pub index_ms: u64,
    pub edges: Vec<(String, String)>,
    pub runtime_edges: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolGraph {
    pub defs_by_file: BTreeMap<String, Vec<String>>,
    pub refs_by_file: BTreeMap<String, Vec<String>>,
    pub imports_by_file: BTreeMap<String, Vec<String>>,
    pub edges: Vec<(String, String)>,
    pub runtime_edges: Vec<(String, String)>,
    pub fan_in: HashMap<String, usize>,
    pub fan_out: HashMap<String, usize>,
    pub dead_symbols: Vec<DeadSymbol>,
    pub metrics_by_file: BTreeMap<String, ModuleMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SymbolCache {
    version: String,
    files: BTreeMap<String, FileSymbols>,
    #[serde(default)]
    test_hashes: BTreeMap<String, (String, Vec<u64>)>,
}

#[derive(Serialize)]
struct SymbolCacheRef<'a> {
    version: &'a str,
    files: &'a BTreeMap<String, FileSymbols>,
    test_hashes: &'a BTreeMap<String, (String, Vec<u64>)>,
}

struct TsConfigScope {
    pkg_dir: String,
    aliases: Vec<(String, String)>,
}

struct LookupTables {
    all_rels: FxHashSet<String>,
    mod_to_rel: FxHashMap<String, String>,
    stem_to_rels: FxHashMap<String, Vec<String>>,
    go_pkg_rels: FxHashMap<String, Vec<String>>,
    go_full_pkg_rels: FxHashMap<String, Vec<String>>,
    ts_scopes: Vec<TsConfigScope>,
    rs_crate_roots: Vec<String>,
}

pub fn locate_stem(hint: Option<&Path>) -> Option<PathBuf> {
    if let Some(h) = hint.filter(|p| p.exists()) { return Some(h.to_path_buf()); }
    if let Ok(env_bin) = std::env::var("REFIT_STEM_BIN") {
        let p = PathBuf::from(env_bin);
        if p.exists() { return Some(p); }
    }
    let bin = if cfg!(windows) { "stem.exe" } else { "stem" };
    let sibling = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(bin)));
    sibling.into_iter().chain(std::iter::once(PathBuf::from(".tools").join(bin))).find(|c| c.exists())
}

pub fn scan_edges(
    repo_root: &Path, stem_bin_hint: Option<&Path>, preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> EdgeScan {
    let started = Instant::now();
    let (mut disc, _, _, _) = discover_internal(repo_root, preloaded, None);
    let mut edge_set: HashSet<(String, String)> = disc.edges.into_iter().collect();
    let mut rt_set: HashSet<(String, String)> = disc.runtime_edges.into_iter().collect();
    let want_ext = stem_bin_hint.is_some() || std::env::var("REFIT_USE_STEM_BIN").as_deref() == Ok("1");
    if want_ext && query_ext(repo_root, stem_bin_hint, &mut edge_set) {
        disc.used_stem = true;
        rt_set.extend(edge_set.iter().cloned());
    }
    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    let mut runtime_edges: Vec<(String, String)> = rt_set.into_iter().collect();
    edges.sort();
    runtime_edges.sort();
    EdgeScan { used_stem: disc.used_stem, index_ms: started.elapsed().as_millis() as u64, edges, runtime_edges }
}

fn query_ext(repo_root: &Path, hint: Option<&Path>, edge_set: &mut HashSet<(String, String)>) -> bool {
    let Some(stem_bin) = locate_stem(hint) else { return false };
    let stem_home = repo_root.join(".refit/stem-home");
    if mkdir_private(&stem_home).is_err() { return false; }

    let mut cmd = Command::new(&stem_bin);
    cmd.arg("mcp").current_dir(repo_root).env_clear()
        .env("HOME", &stem_home).env("USERPROFILE", &stem_home)
        .env("STEM_MCP_TOOLS", "defs,refs,callers,impls,symbols,sql,brief")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
    for k in ["PATH", "SYSTEMROOT", "TEMP", "TMP"] {
        if let Ok(v) = std::env::var(k) { cmd.env(k, v); }
    }
    let Ok(mut child) = cmd.spawn() else { return false };
    let init = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": { "name": "refit", "version": "0.5.0" } }
    });
    let call = json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": { "name": "sql", "arguments": { "query": STEM_SQL } }
    });
    if let Some(mut stdin) = child.stdin.take() { let _ = writeln!(stdin, "{init}\n{call}"); }
    let Ok(out) = child.wait_with_output() else { return false };
    if !out.status.success() { return false; }

    let mut used = false;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { continue };
        let id = v["id"].as_u64().unwrap_or(0);
        used |= id == 1 && v.get("result").is_some();
        if id == 2 && let Some(rows) = v["result"]["structuredContent"]["rows"].as_array() {
            for row in rows {
                if let (Some(s), Some(d)) = (row["src"].as_str(), row["dst"].as_str()) {
                    edge_set.insert((s.replace('\\', "/"), d.replace('\\', "/")));
                }
            }
        }
    }
    used
}

#[cfg(test)]
pub fn index_repo(repo_root: &Path, preloaded: Option<&BTreeMap<String, Vec<u8>>>) -> (EdgeScan, SymbolGraph) {
    let (disc, _, _, snap) = discover_internal(repo_root, preloaded, None);
    (disc, snap)
}

/// Snapshot-backed repository indexing sharing the single `RepoSnapshot` tree walk and `ts_configs`.
pub fn index_repo_snap(repo_root: &Path, snap: &RepoSnapshot) -> (EdgeScan, SymbolGraph) {
    let (disc, _, _, graph) = discover_internal(repo_root, Some(&snap.preloaded), Some(snap));
    (disc, graph)
}

pub fn load_symbols(repo_root: &Path, preloaded: Option<&BTreeMap<String, Vec<u8>>>) -> SymbolGraph {
    discover_internal(repo_root, preloaded, None).3
}

fn file_edge_lists(edges: &[(String, String)], norm: &str) -> (Vec<String>, Vec<String>) {
    let deps = edges.iter().filter(|(s, _)| s == norm).map(|(_, d)| d.clone()).collect();
    let dependents = edges.iter().filter(|(_, d)| d == norm).map(|(s, _)| s.clone()).collect();
    (deps, dependents)
}

pub fn query_symbols(repo_root: &Path, symbol: Option<&str>, file: Option<&str>) -> Value {
    if symbol.is_none() && let Some(f) = file {
        let norm = f.replace('\\', "/");
        if let Some(hit) = sst::query_file_sst(repo_root, CACHE_VERSION, &norm) {
            let (deps, dependents) = file_edge_lists(&hit.edges, &norm);
            let fan_in = hit.runtime_edges.iter().filter(|(_, d)| d == &norm).count();
            let fan_out = hit.runtime_edges.iter().filter(|(s, _)| s == &norm).count();
            return json!({
                "file": norm, "defs": hit.sym.defs, "imports": hit.sym.imports,
                "deps": deps, "dependents": dependents, "fan_in": fan_in, "fan_out": fan_out
            });
        }
    }
    query_symbol_graph(&load_symbols(repo_root, None), symbol, file)
}

pub fn query_symbol_graph(snap: &SymbolGraph, symbol: Option<&str>, file: Option<&str>) -> Value {
    if let Some(sym) = symbol {
        let definers: Vec<String> = snap.defs_by_file.iter()
            .filter(|(_, defs)| defs.iter().any(|d| d == sym)).map(|(f, _)| f.clone()).collect();
        let callers: Vec<String> = snap.refs_by_file.iter()
            .filter(|(f, refs)| !definers.contains(*f) && refs.iter().any(|r| r == sym))
            .map(|(f, _)| f.clone()).collect();
        return json!({
            "symbol": sym, "definers": definers, "callers": callers,
            "is_dead": !definers.is_empty() && callers.is_empty()
        });
    }
    if let Some(f) = file {
        let norm = f.replace('\\', "/");
        let (deps, dependents) = file_edge_lists(&snap.edges, &norm);
        return json!({
            "file": norm,
            "defs": snap.defs_by_file.get(&norm).cloned().unwrap_or_default(),
            "imports": snap.imports_by_file.get(&norm).cloned().unwrap_or_default(),
            "deps": deps, "dependents": dependents,
            "fan_in": snap.fan_in.get(&norm).copied().unwrap_or(0),
            "fan_out": snap.fan_out.get(&norm).copied().unwrap_or(0)
        });
    }
    json!({
        "indexed_files": snap.defs_by_file.len(), "total_edges": snap.edges.len(),
        "dead_symbols_count": snap.dead_symbols.len(),
        "dead_symbols": snap.dead_symbols.iter().take(40).collect::<Vec<_>>(),
        "top_fan_in": top_fan(&snap.fan_in, 10), "top_fan_out": top_fan(&snap.fan_out, 10)
    })
}

fn top_fan(map: &HashMap<String, usize>, limit: usize) -> Vec<(String, usize)> {
    let mut items: Vec<(String, usize)> = map.iter().map(|(k, &v)| (k.clone(), v)).collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items.truncate(limit);
    items
}

#[cfg(test)]
pub fn index_stats(repo_root: &Path) -> (EdgeScan, usize, usize) {
    let (d, p, c, _) = discover_internal(repo_root, None, None);
    (d, p, c)
}

fn resolve_tsconfig_base(rel_dir: &str, base_url: &str) -> String {
    let clean = base_url.trim_start_matches("./").trim_matches('/');
    if rel_dir.is_empty() {
        clean.to_string()
    } else if clean.is_empty() || clean == "." {
        rel_dir.to_string()
    } else {
        format!("{rel_dir}/{clean}")
    }
}

fn map_tsconfig_target(effective_base: &str, raw_target: &str) -> String {
    if effective_base.is_empty() || effective_base == "." {
        raw_target.to_string()
    } else if raw_target.is_empty() {
        format!("{effective_base}/")
    } else {
        join_rel(effective_base, raw_target)
    }
}

fn apply_tsconfig_paths(
    paths: &serde_json::Map<String, Value>, effective_base: &str, aliases: &mut Vec<(String, String)>,
) {
    for (alias, targets) in paths {
        let prefix = alias.trim_end_matches('*');
        let Some(arr) = (!prefix.is_empty()).then(|| targets.as_array()).flatten() else { continue };
        aliases.retain(|(existing, _)| existing != prefix);
        for target in arr.iter().filter_map(Value::as_str) {
            let raw = target.trim_start_matches("./").trim_end_matches('*');
            aliases.push((prefix.to_string(), map_tsconfig_target(effective_base, raw)));
        }
    }
}

fn load_tsconfig_resolved(repo_root: &Path, cfg_path: &Path, depth: usize) -> (String, Vec<(String, String)>) {
    if depth > 8 { return (String::new(), Vec::new()); }
    let Ok(raw) = std::fs::read_to_string(cfg_path) else { return (String::new(), Vec::new()) };
    let Ok(val) = serde_json::from_str::<Value>(&strip_jsonc(&raw)) else { return (String::new(), Vec::new()) };

    let cfg_dir = cfg_path.parent().unwrap_or(repo_root);
    let (mut base, mut aliases) = if let Some(ext_rel) = val["extends"].as_str() {
        let mut parent = cfg_dir.join(ext_rel);
        if !parent.exists() && parent.extension().is_none() { parent.set_extension("json"); }
        load_tsconfig_resolved(repo_root, &dunce::canonicalize(&parent).unwrap_or(parent), depth + 1)
    } else {
        (String::new(), Vec::new())
    };
    let canon_root = dunce::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    let canon_dir = dunce::canonicalize(cfg_dir).unwrap_or_else(|_| cfg_dir.to_path_buf());
    let rel_dir = rel_norm(&canon_root, &canon_dir).or_else(|| rel_norm(repo_root, cfg_dir)).unwrap_or_default();
    if let Some(base_url) = val["compilerOptions"]["baseUrl"].as_str() {
        base = resolve_tsconfig_base(&rel_dir, base_url);
    }
    if let Some(paths) = val["compilerOptions"]["paths"].as_object() {
        let eff = if base.is_empty() { rel_dir.as_str() } else { base.as_str() };
        apply_tsconfig_paths(paths, eff, &mut aliases);
    }
    (base, aliases)
}

fn discover_ts_scopes(repo_root: &Path, pre_ts_configs: Option<&[PathBuf]>) -> Vec<TsConfigScope> {
    let mut cfg_paths = Vec::new();
    if let Some(found) = pre_ts_configs {
        cfg_paths.extend(found.iter().cloned());
    } else {
        let root_cfg = repo_root.join("tsconfig.json");
        if root_cfg.exists() { cfg_paths.push(root_cfg); }
        for p in walk_files(repo_root, None) {
            if let Some(name) = p.file_name().and_then(|s| s.to_str())
                && matches!(name, "tsconfig.json" | "jsconfig.json") && !cfg_paths.contains(&p)
            {
                cfg_paths.push(p);
            }
        }
    }
    let mut scopes = Vec::new();
    for cfg in cfg_paths {
        let (_, aliases) = load_tsconfig_resolved(repo_root, &cfg, 0);
        if aliases.is_empty() { continue; }
        let pkg_dir = cfg.parent().and_then(|d| rel_norm(repo_root, d)).unwrap_or_default();
        scopes.push(TsConfigScope { pkg_dir, aliases });
    }
    scopes.sort_by(|a, b| b.pkg_dir.len().cmp(&a.pkg_dir.len()).then_with(|| a.pkg_dir.cmp(&b.pkg_dir)));
    scopes
}

fn parse_go_work_dirs(work_text: &str, mod_dirs: &mut Vec<PathBuf>) {
    let mut in_use = false;
    for line in work_text.lines().map(str::trim) {
        if line.starts_with("use (") { in_use = true; continue; }
        if in_use && line.starts_with(')') { in_use = false; continue; }
        let cand = if in_use { Some(line) } else { line.strip_prefix("use ") };
        if let Some(raw) = cand {
            let cleaned = raw.split("//").next().unwrap_or("").trim().trim_matches('"').trim_start_matches("./");
            if !cleaned.is_empty() { mod_dirs.push(PathBuf::from(cleaned)); }
        }
    }
}

fn read_go_mod_path(go_mod_path: &Path) -> Option<String> {
    let txt = std::fs::read_to_string(go_mod_path).ok()?;
    txt.lines().map(str::trim).find_map(|line| {
        let mod_path = line.strip_prefix("module ")?.split_whitespace().next()?.trim_matches('"');
        (!mod_path.is_empty()).then(|| mod_path.to_string())
    })
}

fn discover_go_modules(repo_root: &Path) -> Vec<(String, String)> {
    let mut mod_dirs = Vec::new();
    if repo_root.join("go.mod").exists() { mod_dirs.push(PathBuf::new()); }
    if let Ok(work) = std::fs::read_to_string(repo_root.join("go.work")) {
        parse_go_work_dirs(&work, &mut mod_dirs);
    }
    mod_dirs.into_iter().filter_map(|sub| {
        let gm = if sub.as_os_str().is_empty() { repo_root.join("go.mod") } else { repo_root.join(&sub).join("go.mod") };
        read_go_mod_path(&gm).map(|m| (sub.to_string_lossy().replace('\\', "/"), m))
    }).collect()
}

fn join_rel(base_dir: &str, rel_import: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in base_dir.split('/').chain(rel_import.split('/')) {
        match seg {
            "" | "." => {}
            ".." => { parts.pop(); }
            other => parts.push(other),
        }
    }
    let mut joined = parts.join("/");
    if rel_import.ends_with('/') && !joined.is_empty() { joined.push('/'); }
    joined
}

fn register_py_init_files(all_files: &[(String, String, Option<PathBuf>)], mod_to_rel: &mut FxHashMap<String, String>) {
    for (rel, ext, _) in all_files {
        if ext == "py" && let Some(pkg) = rel.strip_suffix("/__init__.py") {
            mod_to_rel.insert(pkg.replace('/', "."), rel.clone());
            if let Some(s) = pkg.strip_prefix("src/") { mod_to_rel.insert(s.replace('/', "."), rel.clone()); }
        }
    }
}

fn register_py_source(rel: &str, mod_to_rel: &mut FxHashMap<String, String>) {
    mod_to_rel.insert(rel.trim_end_matches(".py").replace('/', "."), rel.to_string());
    if let Some(stripped) = rel.strip_prefix("src/") {
        mod_to_rel.entry(stripped.trim_end_matches(".py").replace('/', ".")).or_insert_with(|| rel.to_string());
    }
}

fn register_rs_source(rel: &str, mod_to_rel: &mut FxHashMap<String, String>, crate_roots: &mut FxHashSet<String>) {
    if let Some(mod_dir) = rel.strip_suffix("/mod.rs") { mod_to_rel.insert(mod_dir.to_string(), rel.to_string()); }
    if rel == "src/lib.rs" || rel == "src/main.rs" || rel.starts_with("src/") {
        crate_roots.insert("src".to_string());
    } else if let Some(p) = rel.strip_suffix("/src/lib.rs").or_else(|| rel.strip_suffix("/src/main.rs")) {
        crate_roots.insert(format!("{p}/src"));
    }
}

fn go_full_pkg_path(rel_dir: &str, mod_dir: &str, mod_path: &str) -> Option<String> {
    if mod_dir.is_empty() {
        return Some(if rel_dir.is_empty() { mod_path.to_string() } else { format!("{mod_path}/{rel_dir}") });
    }
    if rel_dir == mod_dir { return Some(mod_path.to_string()); }
    rel_dir.strip_prefix(&format!("{mod_dir}/")).map(|sub| format!("{mod_path}/{sub}"))
}

fn register_go_source(rel: &str, path: &Path, go_mods: &[(String, String)], tbl: &mut LookupTables) {
    let rel_dir = dir_of(rel);
    if let Some(dir) = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
        tbl.go_pkg_rels.entry(dir.to_string()).or_default().push(rel.to_string());
    }
    for (mod_dir, mod_path) in go_mods {
        if let Some(full_pkg) = go_full_pkg_path(&rel_dir, mod_dir, mod_path) {
            tbl.go_full_pkg_rels.entry(full_pkg).or_default().push(rel.to_string());
        }
    }
}

fn build_tables(
    repo_root: &Path, sources: &[(String, String, Option<PathBuf>)],
    all_files: &[(String, String, Option<PathBuf>)], pre_ts_configs: Option<&[PathBuf]>,
) -> LookupTables {
    let go_mods = discover_go_modules(repo_root);
    let mut tbl = LookupTables {
        all_rels: FxHashSet::with_capacity_and_hasher(sources.len(), Default::default()),
        mod_to_rel: FxHashMap::with_capacity_and_hasher(sources.len() * 2, Default::default()),
        stem_to_rels: FxHashMap::default(), go_pkg_rels: FxHashMap::default(),
        go_full_pkg_rels: FxHashMap::default(),
        ts_scopes: discover_ts_scopes(repo_root, pre_ts_configs), rs_crate_roots: Vec::new(),
    };
    let mut crate_roots = FxHashSet::default();
    register_py_init_files(all_files, &mut tbl.mod_to_rel);
    for (rel, ext, _) in sources {
        tbl.all_rels.insert(rel.clone());
        let path = Path::new(rel);
        if ext == "py" {
            register_py_source(rel, &mut tbl.mod_to_rel);
        } else {
            let no_ext = rel.rsplit_once('.').map_or(rel.as_str(), |(h, _)| h);
            tbl.mod_to_rel.insert(no_ext.to_string(), rel.clone());
            if ext == "rs" {
                register_rs_source(rel, &mut tbl.mod_to_rel, &mut crate_roots);
            } else if ext == "go" {
                register_go_source(rel, path, &go_mods, &mut tbl);
            }
        }
        let stem = stem_of(path);
        if !stem.is_empty() { tbl.stem_to_rels.entry(stem.to_string()).or_default().push(rel.clone()); }
    }
    let mut roots: Vec<String> = crate_roots.into_iter().collect();
    roots.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    tbl.rs_crate_roots = roots;
    tbl
}

pub(crate) fn should_use_sst(file_count: usize, est_json_bytes: u64) -> bool {
    let forced = std::env::var("REFIT_FORCE_SST").ok().is_some_and(|v| {
        let t = v.trim();
        !t.is_empty() && !matches!(t.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off")
    });
    forced
        || (cfg!(feature = "sst-storage")
            && (file_count >= sst::SST_MIN_FILES || est_json_bytes >= sst::SST_MIN_INDEX_BYTES))
}

fn estimate_symbol_json_bytes(files: &BTreeMap<String, FileSymbols>) -> u64 {
    files
        .iter()
        .map(|(rel, f)| {
            let sym_bytes: usize = f.defs.iter().map(|s| s.len() + 3).sum::<usize>()
                + f.imports.iter().map(|s| s.len() + 3).sum::<usize>()
                + f.refs_tokens.iter().map(|s| s.len() + 3).sum::<usize>();
            (rel.len() + sym_bytes + 128) as u64
        })
        .sum()
}

fn load_cached_symbols(repo_root: &Path) -> SymbolCache {
    let cache_path = repo_root.join(".refit/cache/symbol_index.zst");
    let sst_path = repo_root.join(".refit/cache/sym.sst");
    if cache_path.is_file()
        && let Some(idx) =
            read_zstd::<SymbolCache>(&cache_path).filter(|idx| idx.version == CACHE_VERSION)
    {
        return idx;
    }
    if sst_path.is_file()
        && let Some(files) = sst::load_symbol_sst(&sst_path, CACHE_VERSION)
    {
        return SymbolCache {
            version: CACHE_VERSION.to_string(),
            files,
            test_hashes: BTreeMap::new(),
        };
    }
    SymbolCache::default()
}

fn persist_symbol_caches(
    repo_root: &Path,
    new_files: &BTreeMap<String, FileSymbols>,
    new_test_hashes: &BTreeMap<String, (String, Vec<u64>)>,
    edges: &[(String, String)],
    runtime_edges: &[(String, String)],
    prev_len: usize,
) {
    let cache_path = repo_root.join(".refit/cache/symbol_index.zst");
    let sst_path = repo_root.join(".refit/cache/sym.sst");
    if check_wipe_guard(prev_len, new_files.len()) {
        tracing::warn!(
            prev_files = prev_len,
            new_files = new_files.len(),
            "index-wipe guard triggered (>80% files vanished); keeping previous symbol index cache"
        );
        return;
    }
    let est_bytes = estimate_symbol_json_bytes(new_files);
    if should_use_sst(new_files.len(), est_bytes) {
        if sst::save_symbol_sst(
            &sst_path,
            CACHE_VERSION,
            new_files,
            edges,
            runtime_edges,
            prev_len,
        )
        .is_ok_and(|written| written)
        {
            let _ = std::fs::remove_file(&cache_path);
        }
    } else {
        let next_ref = SymbolCacheRef {
            version: CACHE_VERSION,
            files: new_files,
            test_hashes: new_test_hashes,
        };
        if write_zstd(&cache_path, &next_ref).is_ok() {
            let _ = std::fs::remove_file(&sst_path);
        }
    }
}

fn discover_internal(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    repo_snap: Option<&RepoSnapshot>,
) -> (EdgeScan, usize, usize, SymbolGraph) {
    let started = Instant::now();
    let prev = load_cached_symbols(repo_root);

    let scan: RepoScan = match repo_snap {
        Some(s) => s.to_repo_scan(),
        None => scan_files(repo_root, preloaded),
    };
    let tables = build_tables(
        repo_root,
        &scan.sources,
        &scan.all,
        repo_snap.map(|s| s.ts_configs.as_slice()),
    );

    let parse_one = |(rel, ext, _): &(String, String, Option<PathBuf>)| -> Option<(String, FileSymbols, bool)> {
        let bytes = read_bytes(repo_root, rel, preloaded)?;
        let hash = MvccStore::hash_bytes(&bytes);
        if let Some(hit) = prev.files.get(rel).filter(|f| f.blake3 == hash) {
            return Some((rel.clone(), hit.clone(), false));
        }
        let entry = parse_symbols(ext, &String::from_utf8_lossy(&bytes), hash);
        Some((rel.clone(), entry, true))
    };

    let parsed_rows: Vec<(String, FileSymbols, bool)> = if crate::platform::budget().is_constrained() {
        scan.sources.iter().filter_map(parse_one).collect()
    } else {
        scan.sources.par_iter().filter_map(parse_one).collect()
    };

    let (mut parsed_n, mut cached_n) = (0usize, 0usize);
    let mut new_files = BTreeMap::new();
    for (rel, entry, was_parsed) in parsed_rows {
        if was_parsed {
            parsed_n += 1;
        } else {
            cached_n += 1;
        }
        new_files.insert(rel, entry);
    }

    let (edges, runtime_edges, glob_reexport_targets) = collect_edges(&new_files, &tables);
    let (new_test_hashes, test_changed) =
        collect_test_hashes_cached(repo_root, preloaded, &scan.tests, &prev.test_hashes);
    let want_sst = should_use_sst(new_files.len(), estimate_symbol_json_bytes(&new_files));
    let cache_exists = repo_root.join(".refit/cache/symbol_index.zst").is_file();
    let sst_exists = repo_root.join(".refit/cache/sym.sst").is_file();
    let format_mismatch = if want_sst { !sst_exists || cache_exists } else { !cache_exists || sst_exists };
    if parsed_n > 0 || test_changed || prev.files.len() != new_files.len() || format_mismatch {
        persist_symbol_caches(
            repo_root,
            &new_files,
            &new_test_hashes,
            &edges,
            &runtime_edges,
            prev.files.len(),
        );
    }
    let snap = build_graph(
        new_files,
        &new_test_hashes,
        &edges,
        &runtime_edges,
        &glob_reexport_targets,
    );

    (
        EdgeScan {
            used_stem: false,
            index_ms: started.elapsed().as_millis() as u64,
            edges,
            runtime_edges,
        },
        parsed_n,
        cached_n,
        snap,
    )
}

fn collect_file_edges(
    rel: &str,
    entry: &FileSymbols,
    tbl: &LookupTables,
    edge_set: &mut FxHashSet<(String, String)>,
    rt_set: &mut FxHashSet<(String, String)>,
    glob_targets: &mut FxHashSet<String>,
) {
    let src_dir = dir_of(rel);
    let src_ext = ext_of(rel);

    for imp in &entry.typed_imports {
        if imp.kind == EdgeKind::TestImport {
            continue;
        }
        for target in resolve_import(&imp.target, rel, &src_dir, &src_ext, tbl) {
            if target == rel || !tbl.all_rels.contains(&target) {
                continue;
            }
            edge_set.insert((rel.to_string(), target.clone()));
            if imp.kind.is_runtime_scc() {
                rt_set.insert((rel.to_string(), target));
            }
        }
    }

    for glob in &entry.glob_reexports {
        for target in resolve_import(glob, rel, &src_dir, &src_ext, tbl) {
            if target != rel {
                glob_targets.insert(target);
            }
        }
    }
}

type EdgeTuple = (Vec<(String, String)>, Vec<(String, String)>, FxHashSet<String>);

fn collect_edges(files: &BTreeMap<String, FileSymbols>, tbl: &LookupTables) -> EdgeTuple {
    let mut edge_set: FxHashSet<(String, String)> = FxHashSet::default();
    let mut rt_set: FxHashSet<(String, String)> = FxHashSet::default();
    let mut glob_targets: FxHashSet<String> = FxHashSet::default();

    for (rel, entry) in files {
        collect_file_edges(rel, entry, tbl, &mut edge_set, &mut rt_set, &mut glob_targets);
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    let mut runtime_edges: Vec<(String, String)> = rt_set.into_iter().collect();
    edges.sort();
    runtime_edges.sort();
    (edges, runtime_edges, glob_targets)
}

fn compute_fan_counts(
    runtime_edges: &[(String, String)],
) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let (mut fan_in, mut fan_out) = (HashMap::new(), HashMap::new());
    for (src, dst) in runtime_edges {
        *fan_out.entry(src.clone()).or_insert(0) += 1;
        *fan_in.entry(dst.clone()).or_insert(0) += 1;
    }
    (fan_in, fan_out)
}

fn record_def_boundary_witnesses(
    entry: &FileSymbols,
    is_glob: bool,
    witnesses: &mut FxHashMap<u64, RefWitness>,
) {
    for (idx, sym) in entry.defs.iter().enumerate() {
        let flags = entry.def_meta.get(idx).map_or(0, |meta| meta.boundary_flags);
        let has_ffi = (flags & BOUNDARY_FFI_NO_MANGLE) != 0;
        let has_macro = (flags & BOUNDARY_MACRO_DERIVE) != 0;
        if !is_glob && !has_ffi && !has_macro {
            continue;
        }
        let witness = witnesses.entry(hash_symbol(sym)).or_default();
        if is_glob {
            witness.add_flag(WITNESS_GLOB_REEXPORT);
        }
        if has_ffi {
            witness.add_flag(WITNESS_EXTERNAL);
        }
        if has_macro {
            witness.add_flag(WITNESS_MACRO_ATTR);
        }
    }
}

fn record_source_witnesses(
    files: &BTreeMap<String, FileSymbols>,
    glob_targets: &FxHashSet<String>,
    arena: &mut PathArena,
    witnesses: &mut FxHashMap<u64, RefWitness>,
) {
    for (rel, entry) in files {
        let fid = arena.intern(rel);
        for tok in &entry.refs_tokens {
            witnesses.entry(hash_symbol(tok)).or_default().record_prod(fid);
        }
        for tok in &entry.test_refs_tokens {
            witnesses.entry(hash_symbol(tok)).or_default().record_test(fid);
        }
        record_def_boundary_witnesses(entry, glob_targets.contains(rel), witnesses);
        for attr_sym in &entry.attr_refs {
            witnesses
                .entry(hash_symbol(attr_sym))
                .or_default()
                .add_flag(WITNESS_DYNAMIC_REF);
        }
    }
}

type TestHashMap = BTreeMap<String, (String, Vec<u64>)>;

fn collect_test_hashes_cached(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    tests: &[(String, String, Option<PathBuf>)],
    prev_test_hashes: &TestHashMap,
) -> (TestHashMap, bool) {
    let scan_test = |(rel, ext, _): &(String, String, Option<PathBuf>)| -> Option<(String, String, Vec<u64>, bool)> {
        let bytes = read_bytes(repo_root, rel, preloaded)?;
        let hash = MvccStore::hash_bytes(&bytes);
        if let Some((prev_h, cached_ids)) = prev_test_hashes.get(rel)
            && prev_h == &hash
        {
            return Some((rel.clone(), hash, cached_ids.clone(), false));
        }
        let parsed = parse_symbols(ext, &String::from_utf8_lossy(&bytes), String::new());
        let mut hashes: Vec<u64> = parsed
            .refs_tokens
            .iter()
            .chain(parsed.test_refs_tokens.iter())
            .map(|sym| hash_symbol(sym))
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        Some((rel.clone(), hash, hashes, true))
    };

    let rows: Vec<(String, String, Vec<u64>, bool)> = if crate::platform::budget().is_constrained() {
        tests.iter().filter_map(scan_test).collect()
    } else {
        tests.par_iter().filter_map(scan_test).collect()
    };

    let mut changed = rows.len() != prev_test_hashes.len();
    let mut map = BTreeMap::new();
    for (rel, hash, ids, was_parsed) in rows {
        changed |= was_parsed;
        map.insert(rel, (hash, ids));
    }
    (map, changed)
}

fn collect_dead_symbols_for_file(
    rel: &str,
    fid: u32,
    entry: &FileSymbols,
    glob_targets: &FxHashSet<String>,
    witnesses: &FxHashMap<u64, RefWitness>,
    dead: &mut Vec<DeadSymbol>,
) {
    let is_boundary_file = entrypoint(rel)
        || rel.ends_with("layout.tsx")
        || rel.ends_with("layout.ts")
        || glob_targets.contains(rel);
    let file_boundary = if is_boundary_file { BOUNDARY_ENTRYPOINT_OR_GLOB } else { 0 };

    for (idx, sym) in entry.defs.iter().enumerate() {
        let meta = entry.def_meta.get(idx).copied().unwrap_or_default();
        let witness = witnesses.get(&hash_symbol(sym)).copied().unwrap_or_default();
        let combined_flags = meta.boundary_flags | file_boundary;
        if let Some(tier) = classify_symbol(fid, witness, meta.intra_non_def_hits, combined_flags)
            && tier.is_dead_candidate()
        {
            dead.push(DeadSymbol {
                file: rel.to_string(),
                symbol: sym.clone(),
                proof_tier: Some(tier.as_str().to_string()),
            });
        }
    }
}

fn build_graph(
    files: BTreeMap<String, FileSymbols>,
    test_hashes: &TestHashMap,
    edges: &[(String, String)],
    runtime_edges: &[(String, String)],
    glob_targets: &FxHashSet<String>,
) -> SymbolGraph {
    let (fan_in, fan_out) = compute_fan_counts(runtime_edges);
    let mut arena = PathArena::with_capacity(files.len() + test_hashes.len());
    let mut witnesses: FxHashMap<u64, RefWitness> =
        FxHashMap::with_capacity_and_hasher(files.len() * 16 + 64, Default::default());
    record_source_witnesses(&files, glob_targets, &mut arena, &mut witnesses);
    for (trel, (_, hashes)) in test_hashes {
        let tid = arena.intern(trel);
        for &hash in hashes { witnesses.entry(hash).or_default().record_test(tid); }
    }
    let (mut defs_by, mut refs_by, mut imps_by, mut metrics_by, mut dead) =
        (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), Vec::new());
    for (rel, entry) in files {
        let fid = arena.id_of(&rel).unwrap_or(analysis::NO_FILE);
        collect_dead_symbols_for_file(&rel, fid, &entry, glob_targets, &witnesses, &mut dead);
        defs_by.insert(rel.clone(), entry.defs);
        refs_by.insert(rel.clone(), entry.refs_tokens);
        imps_by.insert(rel.clone(), entry.imports);
        metrics_by.insert(rel, entry.metrics);
    }
    SymbolGraph {
        defs_by_file: defs_by, refs_by_file: refs_by, imports_by_file: imps_by,
        edges: edges.to_vec(), runtime_edges: runtime_edges.to_vec(),
        fan_in, fan_out, dead_symbols: dead, metrics_by_file: metrics_by,
    }
}

fn strip_js_ext(p: &str) -> &str {
    [".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs", ".mts", ".cts"]
        .iter().find_map(|ext| p.strip_suffix(ext)).unwrap_or(p)
}

fn resolve_relative_import(imp: &str, src_dir: &str, tbl: &LookupTables) -> Option<String> {
    if !imp.starts_with("./") && !imp.starts_with("../") { return None; }
    let base = join_rel(src_dir, strip_js_ext(imp));
    tbl.mod_to_rel.get(&base)
        .or_else(|| tbl.mod_to_rel.get(&format!("{base}/index"))).cloned()
        .or_else(|| tbl.all_rels.contains(&base).then_some(base))
}

fn resolve_ts_scope_import(imp: &str, src_dir: &str, tbl: &LookupTables) -> Option<String> {
    for scope in &tbl.ts_scopes {
        let in_scope = scope.pkg_dir.is_empty() || src_dir == scope.pkg_dir
            || src_dir.starts_with(&format!("{}/", scope.pkg_dir));
        if !in_scope { continue; }
        for (alias, target_prefix) in &scope.aliases {
            let Some(rest) = imp.strip_prefix(alias.as_str()) else { continue };
            let mapped = format!("{target_prefix}{}", strip_js_ext(rest));
            if let Some(hit) = tbl.mod_to_rel.get(&mapped)
                .or_else(|| tbl.mod_to_rel.get(&format!("{mapped}/index")))
            { return Some(hit.clone()); }
        }
    }
    None
}

fn resolve_rs_import(imp: &str, src_rel: &str, src_dir: &str, tbl: &LookupTables) -> Option<String> {
    if let Some(sym) = imp.strip_prefix("super::") {
        let parent = dir_of(src_dir);
        let cand = if parent.is_empty() { sym.into() } else { format!("{parent}/{sym}") };
        if let Some(hit) = tbl.mod_to_rel.get(&cand) { return Some(hit.clone()); }
    }
    if !src_dir.is_empty() && let Some(hit) = tbl.mod_to_rel.get(&format!("{src_dir}/{imp}")) {
        return Some(hit.clone());
    }
    let root = tbl.rs_crate_roots.iter().find(|r| src_rel.starts_with(&format!("{r}/")))?;
    tbl.mod_to_rel.get(&format!("{root}/{imp}")).cloned()
}

fn resolve_import(
    imp: &str, src_rel: &str, src_dir: &str, src_ext: &str, tbl: &LookupTables,
) -> Vec<String> {
    if let Some(hit) = resolve_relative_import(imp, src_dir, tbl) { return vec![hit]; }
    if src_ext == "py" && imp.starts_with('.') && let Some(hit) = resolve_pyrel(imp, src_dir, tbl) {
        return vec![hit];
    }
    if let Some(hit) = resolve_ts_scope_import(imp, src_dir, tbl) { return vec![hit]; }
    if src_ext == "rs" && let Some(hit) = resolve_rs_import(imp, src_rel, src_dir, tbl) {
        return vec![hit];
    }
    if src_ext == "go"
        && let Some(files) = tbl.go_full_pkg_rels.get(imp)
            .or_else(|| tbl.go_pkg_rels.get(imp.rsplit('/').next().unwrap_or(imp)))
    {
        return files.clone();
    }
    if let Some(hit) = tbl.mod_to_rel.get(imp) { return vec![hit.clone()]; }
    resolve_stem(imp, src_dir, src_ext, tbl)
}

fn resolve_pyrel(imp: &str, src_dir: &str, tbl: &LookupTables) -> Option<String> {
    let dots = imp.bytes().take_while(|&b| b == b'.').count();
    let mut parts: Vec<&str> = src_dir.split('/').filter(|s| !s.is_empty()).collect();
    for _ in 1..dots { parts.pop(); }
    parts.extend(imp[dots..].split('.').filter(|s| !s.is_empty()));
    tbl.mod_to_rel.get(&parts.join(".")).cloned()
}

fn resolve_stem(imp: &str, src_dir: &str, src_ext: &str, tbl: &LookupTables) -> Vec<String> {
    let key = imp.strip_prefix("super::").unwrap_or(imp);
    let Some(cands) = tbl.stem_to_rels.get(key) else { return Vec::new() };
    if !imp.starts_with("super::") && let Some(same) = cands.iter().find(|c| dir_of(c) == src_dir) {
        return vec![same.clone()];
    }
    let matches: Vec<&String> = cands.iter().filter(|c| same_family(src_ext, &ext_of(c))).collect();
    if matches.len() == 1 { vec![matches[0].clone()] } else { Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn has_edge(edges: &[(String, String)], src: &str, dst: &str) -> bool {
        edges.iter().any(|(s, d)| s == src && d == dst)
    }

    #[test]
    fn test_symbol_index_polyglot_and_cache() {
        let leaf = concat!(
            "pub fn base_fn() -> i32 { 1 }\n#[cfg(test)]\nmod tests {\n",
            "    use crate::top::top_fn;\n    #[test] fn t() { assert_eq!(top_fn(), 1); }\n}\n",
        );
        let dir = TempRepo::with_files(
            "sym_all",
            &[
                ("tsconfig.json", r#"{"compilerOptions":{"paths":{"@/*":["src/*"]}}}"#),
                ("src/core/auth.ts", "export function verifyToken() { return true; }\nexport function unusedGhost() { return false; }\n"),
                ("src/ui/login.tsx", "import { verifyToken } from '../core/auth';\nexport default function Login() { return verifyToken(); }\n"),
                ("src/ui/profile.tsx", "import { verifyToken } from '@/core/auth';\nexport function Profile() { return verifyToken(); }\n"),
                ("src/leaf.rs", leaf),
                ("src/top.rs", "use crate::leaf::base_fn;\npub fn top_fn() -> i32 { base_fn() }\n"),
            ],
        );

        let (d1, p1, c1) = index_stats(dir.path());
        assert_eq!((p1, c1), (5, 0));
        assert!(d1.index_ms < 500);

        let snap = load_symbols(dir.path(), None);
        for (src, dst, want) in [
            ("src/ui/login.tsx", "src/core/auth.ts", true),
            ("src/ui/profile.tsx", "src/core/auth.ts", true),
            ("src/top.rs", "src/leaf.rs", true),
            ("src/leaf.rs", "src/top.rs", false),
        ] {
            assert_eq!(has_edge(&snap.edges, src, dst), want, "{src} -> {dst}");
        }
        assert!(snap.dead_symbols.iter().any(|d| d.symbol == "unusedGhost"));

        let (_, p2, c2) = index_stats(dir.path());
        assert_eq!((p2, c2), (0, 5));

        dir.write("src/top.rs", "use crate::leaf::base_fn;\npub fn top_fn() -> i32 { base_fn() + 1 }\n");
        let (_, p3, c3) = index_stats(dir.path());
        assert_eq!((p3, c3), (1, 4));
    }

    #[test]
    fn test_t1_t4_boundaries_recursive_self_call_and_monorepo_resolution() {
        let dir = TempRepo::with_files(
            "stem_v2_v4",
            &[
                ("tsconfig.base.json", r#"{"compilerOptions":{"paths":{"@shared/*":["packages/shared/src/*"]}}}"#),
                ("packages/web/tsconfig.json", r#"{"extends":"../../tsconfig.base.json","compilerOptions":{"paths":{"@web/*":["src/*"]}}}"#),
                ("packages/shared/src/types.ts", "export interface UserDto { id: string }\nexport function liveShared() { return 1; }\n"),
                ("packages/web/src/a.ts", "import type { UserDto } from '@shared/types';\nimport { liveShared } from '@shared/types';\nexport function runWeb(u: UserDto) { return liveShared() + u.id.length; }\n"),
                ("packages/web/src/b.ts", "import type { runWeb } from '@web/a';\nexport const keepTypeEdgeOutOfScc = 1;\n"),
                (
                    "src/rules.rs",
                    concat!(
                        "use crate::sub::{\n",
                        "    mod_helper,\n",
                        "};\n",
                        "#[unsafe(no_mangle)]\n",
                        "pub extern \"C\" fn abi_export() -> i32 { 1 }\n",
                        "#[ctor::ctor]\n",
                        "pub fn ctor_hook() {}\n",
                        "pub(crate) fn recursive_dead(n: u32) -> u32 {\n",
                        "    if n == 0 { 0 } else { recursive_dead(n - 1) }\n",
                        "}\n",
                        "pub(crate) fn intra_helper() -> i32 { mod_helper() }\n",
                        "pub fn entry_caller() -> i32 { intra_helper() }\n",
                        "pub(crate) fn test_only_helper() -> i32 { 42 }\n",
                        "#[cfg(test)]\n",
                        "mod tests {\n",
                        "    fn chk(buf: [u8; 16]) -> usize { buf.len() + super::test_only_helper() as usize }\n",
                        "}\n",
                    ),
                ),
                ("src/sub/mod.rs", "pub fn mod_helper() -> i32 { 7 }\n"),
                ("src/pymod/api.py", "__all__ = ['public_exported']\ndef public_exported():\n    return 1\ndef dead_py_helper():\n    return 2\n"),
                ("go.mod", "module example.com/acme\n\ngo 1.24\n"),
                ("pkg/store/store.go", "package store\n//go:linkname LinkedHook runtime.hook\nfunc LinkedHook() int { return 1 }\nfunc UnusedGoFn() int { return 2 }\n"),
                ("pkg/api/handler.go", "package api\nimport \"example.com/acme/pkg/store\"\nfunc ServeHTTP() int { return store.LinkedHook() }\n"),
            ],
        );

        let (disc, snap) = index_repo(dir.path(), None);
        // Multi-line Rust `use crate::sub::{ mod_helper };` -> `src/sub/mod.rs`
        assert!(has_edge(&disc.runtime_edges, "src/rules.rs", "src/sub/mod.rs"));
        // tsconfig extends `@shared/*` -> Runtime edge
        assert!(has_edge(&disc.runtime_edges, "packages/web/src/a.ts", "packages/shared/src/types.ts"));
        // TS `import type` is NOT in runtime_edges (cannot create runtime SCC cycles!)
        assert!(!has_edge(&disc.runtime_edges, "packages/web/src/b.ts", "packages/web/src/a.ts"));
        // Go go.mod full module path import resolution
        assert!(has_edge(&disc.runtime_edges, "pkg/api/handler.go", "pkg/store/store.go"));

        let dead_names: HashMap<&str, Option<&str>> = snap
            .dead_symbols
            .iter()
            .map(|d| (d.symbol.as_str(), d.proof_tier.as_deref()))
            .collect();

        // Recursive self-call with no other callers MUST still be caught as T1ProvenDead!
        assert_eq!(dead_names.get("recursive_dead"), Some(&Some("T1ProvenDead")));
        assert_eq!(dead_names.get("dead_py_helper"), Some(&Some("T2LikelyDead")));
        assert_eq!(dead_names.get("UnusedGoFn"), Some(&Some("T2LikelyDead")));

        // T3 (intra-file + inline cfg(test) with [u8; 16]!) and T4 boundaries MUST NOT be flagged dead!
        for safe in ["intra_helper", "test_only_helper", "abi_export", "ctor_hook", "public_exported", "LinkedHook", "ServeHTTP"] {
            assert!(!dead_names.contains_key(safe), "false dead symbol: {safe}");
        }
    }
}
