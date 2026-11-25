pub(crate) mod analysis;
mod parse;

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{
    ensure_cachedir_tag, env_trimmed, env_truthy, mkdir_private, par_map, parse_go_block_lines,
    read_zstd, walk_files, write_atomic, write_zstd,
};
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
use anyhow::Result;
use parse::{FileSymbols, parse_symbols};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const CACHE_VERSION: &str = "0.6.0";
const SST_MAGIC: &[u8; 8] = b"RFST0004";
const SST_MIN_FILES: usize = 500;
const SST_MIN_INDEX_BYTES: u64 = 4 * 1024 * 1024;
const RESTART_INTERVAL: usize = 8;
const TARGET_BLOCK_BYTES: usize = 8 * 1024;
const MAX_BLOCK_BYTES: usize = 16 * 1024;
const MAX_BLOCK_KEYS: usize = 256;
const MAX_DECOMP_BLOCK_BYTES: usize = 16 * 1024 * 1024;
const SST_HEADER_LEN: usize = 30;
const VERSION_KEY: &[u8] = b"\0__version__";
const EDGES_KEY: &[u8] = b"\0__edges__";

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

struct TsConfigScope { pkg_dir: String, aliases: Vec<(String, String)> }

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
    if let Some(p) = env_trimmed("REFIT_STEM_BIN").map(PathBuf::from).filter(|p| p.exists()) {
        return Some(p);
    }
    let bin = if cfg!(windows) { "stem.exe" } else { "stem" };
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(bin)));
    if let Some(p) = sibling.filter(|c| c.exists()) {
        return Some(p);
    }
    let local = PathBuf::from(".tools").join(bin);
    local.exists().then_some(local)
}

pub fn scan_edges(
    repo_root: &Path, stem_bin_hint: Option<&Path>, preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> EdgeScan {
    let started = Instant::now();
    let (mut disc, _, _, _) = discover_internal(repo_root, preloaded, None);
    let mut edge_set: HashSet<(String, String)> = disc.edges.into_iter().collect();
    let mut rt_set: HashSet<(String, String)> = disc.runtime_edges.into_iter().collect();
    let want_ext = stem_bin_hint.is_some() || env_truthy("REFIT_USE_STEM_BIN");
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
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": { "name": "refit", "version": "0.5.0" }
        }
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

fn file_query_json(
    norm: &str, defs: &[String], imports: &[String], edges: &[(String, String)],
    fan_in: usize, fan_out: usize,
) -> Value {
    let deps: Vec<String> = edges
        .iter()
        .filter(|(s, _)| s == norm)
        .map(|(_, d)| d.clone())
        .collect();
    let dependents: Vec<String> = edges
        .iter()
        .filter(|(_, d)| d == norm)
        .map(|(s, _)| s.clone())
        .collect();
    json!({
        "file": norm, "defs": defs, "imports": imports,
        "deps": deps, "dependents": dependents, "fan_in": fan_in, "fan_out": fan_out
    })
}

pub fn query_symbols(repo_root: &Path, symbol: Option<&str>, file: Option<&str>) -> Value {
    if symbol.is_none() && let Some(f) = file {
        let norm = f.replace('\\', "/");
        if let Some(hit) = query_file_sst(repo_root, CACHE_VERSION, &norm) {
            let fan_in = hit.runtime_edges.iter().filter(|(_, d)| d == &norm).count();
            let fan_out = hit.runtime_edges.iter().filter(|(s, _)| s == &norm).count();
            return file_query_json(
                &norm, &hit.sym.defs, &hit.sym.imports, &hit.edges, fan_in, fan_out,
            );
        }
    }
    query_symbol_graph(&load_symbols(repo_root, None), symbol, file)
}

pub fn query_symbol_graph(snap: &SymbolGraph, symbol: Option<&str>, file: Option<&str>) -> Value {
    if let Some(sym) = symbol {
        let definers: Vec<String> = snap
            .defs_by_file
            .iter()
            .filter(|(_, defs)| defs.iter().any(|d| d == sym))
            .map(|(f, _)| f.clone())
            .collect();
        let callers: Vec<String> = snap
            .refs_by_file
            .iter()
            .filter(|(f, refs)| !definers.contains(*f) && refs.iter().any(|r| r == sym))
            .map(|(f, _)| f.clone())
            .collect();
        return json!({
            "symbol": sym, "definers": definers, "callers": callers,
            "is_dead": !definers.is_empty() && callers.is_empty()
        });
    }
    if let Some(f) = file {
        let norm = f.replace('\\', "/");
        let defs = snap.defs_by_file.get(&norm).map(Vec::as_slice).unwrap_or(&[]);
        let imports = snap.imports_by_file.get(&norm).map(Vec::as_slice).unwrap_or(&[]);
        let fan_in = snap.fan_in.get(&norm).copied().unwrap_or(0);
        let fan_out = snap.fan_out.get(&norm).copied().unwrap_or(0);
        return file_query_json(&norm, defs, imports, &snap.edges, fan_in, fan_out);
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
    let Ok(raw) = fs::read_to_string(cfg_path) else { return (String::new(), Vec::new()) };
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
    for raw in parse_go_block_lines(work_text, "use") {
        let no_comment = raw.split("//").next().unwrap_or("");
        let cleaned = no_comment.trim().trim_matches('"').trim_start_matches("./");
        if !cleaned.is_empty() {
            mod_dirs.push(PathBuf::from(cleaned));
        }
    }
}

fn read_go_mod_path(go_mod_path: &Path) -> Option<String> {
    let txt = fs::read_to_string(go_mod_path).ok()?;
    txt.lines().map(str::trim).find_map(|line| {
        let mod_path = line.strip_prefix("module ")?.split_whitespace().next()?.trim_matches('"');
        (!mod_path.is_empty()).then(|| mod_path.to_string())
    })
}

fn discover_go_modules(repo_root: &Path) -> Vec<(String, String)> {
    let mut mod_dirs = Vec::new();
    if repo_root.join("go.mod").exists() { mod_dirs.push(PathBuf::new()); }
    if let Ok(work) = fs::read_to_string(repo_root.join("go.work")) {
        parse_go_work_dirs(&work, &mut mod_dirs);
    }
    mod_dirs.into_iter().filter_map(|sub| {
        let gm = if sub.as_os_str().is_empty() {
            repo_root.join("go.mod")
        } else {
            repo_root.join(&sub).join("go.mod")
        };
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

fn register_py_path(
    mod_to_rel: &mut FxHashMap<String, String>, stem: &str, rel: &str, force_src: bool,
) {
    mod_to_rel.insert(stem.replace('/', "."), rel.to_string());
    if let Some(s) = stem.strip_prefix("src/") {
        let key = s.replace('/', ".");
        if force_src || !mod_to_rel.contains_key(&key) {
            mod_to_rel.insert(key, rel.to_string());
        }
    }
}

fn register_py_init_files(
    all_files: &[(String, String, Option<PathBuf>)], mod_to_rel: &mut FxHashMap<String, String>,
) {
    for (rel, ext, _) in all_files {
        if ext == "py" && let Some(pkg) = rel.strip_suffix("/__init__.py") {
            register_py_path(mod_to_rel, pkg, rel, true);
        }
    }
}

fn register_rs_source(
    rel: &str, mod_to_rel: &mut FxHashMap<String, String>, crate_roots: &mut FxHashSet<String>,
) {
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
            register_py_path(&mut tbl.mod_to_rel, rel.trim_end_matches(".py"), rel, false);
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

fn cache_paths(repo_root: &Path) -> (PathBuf, PathBuf) {
    (repo_root.join(".refit/cache/symbol_index.zst"), repo_root.join(".refit/cache/sym.sst"))
}

pub(crate) fn should_use_sst(file_count: usize, est_json_bytes: u64) -> bool {
    env_truthy("REFIT_FORCE_SST")
        || (cfg!(feature = "sst-storage")
            && (file_count >= SST_MIN_FILES || est_json_bytes >= SST_MIN_INDEX_BYTES))
}

fn estimate_symbol_json_bytes(files: &BTreeMap<String, FileSymbols>) -> u64 {
    files.iter().map(|(rel, f)| {
        let sym_bytes: usize = f.defs.iter().map(|s| s.len() + 3).sum::<usize>()
            + f.imports.iter().map(|s| s.len() + 3).sum::<usize>()
            + f.refs_tokens.iter().map(|s| s.len() + 3).sum::<usize>();
        (rel.len() + sym_bytes + 128) as u64
    }).sum()
}

fn load_cached_symbols(repo_root: &Path) -> SymbolCache {
    let (cache_path, sst_path) = cache_paths(repo_root);
    if cache_path.is_file()
        && let Some(idx) = read_zstd::<SymbolCache>(&cache_path).filter(|i| i.version == CACHE_VERSION)
    {
        return idx;
    }
    if sst_path.is_file() && let Some(files) = load_symbol_sst(&sst_path, CACHE_VERSION) {
        return SymbolCache { version: CACHE_VERSION.into(), files, test_hashes: BTreeMap::new() };
    }
    SymbolCache::default()
}

fn persist_symbol_caches(
    repo_root: &Path, new_files: &BTreeMap<String, FileSymbols>,
    new_test_hashes: &BTreeMap<String, (String, Vec<u64>)>,
    edges: &[(String, String)], runtime_edges: &[(String, String)], prev_len: usize,
) {
    let (cache_path, sst_path) = cache_paths(repo_root);
    if check_wipe_guard(prev_len, new_files.len()) {
        tracing::warn!(
            prev_files = prev_len, new_files = new_files.len(),
            "index-wipe guard triggered (>80% files vanished); keeping previous symbol index cache"
        );
        return;
    }
    if should_use_sst(new_files.len(), estimate_symbol_json_bytes(new_files)) {
        if save_symbol_sst(&sst_path, CACHE_VERSION, new_files, edges, runtime_edges, prev_len)
            .is_ok_and(|written| written)
        {
            let _ = fs::remove_file(&cache_path);
        }
    } else {
        let next_ref = SymbolCacheRef {
            version: CACHE_VERSION, files: new_files, test_hashes: new_test_hashes,
        };
        if write_zstd(&cache_path, &next_ref).is_ok() { let _ = fs::remove_file(&sst_path); }
    }
}

fn discover_internal(
    repo_root: &Path, preloaded: Option<&BTreeMap<String, Vec<u8>>>, repo_snap: Option<&RepoSnapshot>,
) -> (EdgeScan, usize, usize, SymbolGraph) {
    let started = Instant::now();
    let prev = load_cached_symbols(repo_root);
    let scan: RepoScan = match repo_snap {
        Some(s) => s.to_repo_scan(),
        None => scan_files(repo_root, preloaded),
    };
    let tables = build_tables(
        repo_root, &scan.sources, &scan.all, repo_snap.map(|s| s.ts_configs.as_slice()),
    );

    let parsed_rows = par_map(&scan.sources, |(rel, ext, _)| {
        let bytes = read_bytes(repo_root, rel, preloaded)?;
        let hash = MvccStore::hash_bytes(&bytes);
        if let Some(hit) = prev.files.get(rel).filter(|f| f.blake3 == hash) {
            return Some((rel.clone(), hit.clone(), false));
        }
        Some((rel.clone(), parse_symbols(ext, &String::from_utf8_lossy(&bytes), hash), true))
    });

    let (mut parsed_n, mut cached_n, mut new_files) = (0usize, 0usize, BTreeMap::new());
    for (rel, entry, was_parsed) in parsed_rows.into_iter().flatten() {
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
    let (cache_path, sst_path) = cache_paths(repo_root);
    let (cache_exists, sst_exists) = (cache_path.is_file(), sst_path.is_file());
    let format_mismatch = if want_sst { !sst_exists || cache_exists } else { !cache_exists || sst_exists };
    if parsed_n > 0 || test_changed || prev.files.len() != new_files.len() || format_mismatch {
        persist_symbol_caches(repo_root, &new_files, &new_test_hashes, &edges, &runtime_edges, prev.files.len());
    }
    let snap = build_graph(new_files, &new_test_hashes, &edges, &runtime_edges, &glob_reexport_targets);
    let edge_scan = EdgeScan {
        used_stem: false, index_ms: started.elapsed().as_millis() as u64, edges, runtime_edges,
    };
    (edge_scan, parsed_n, cached_n, snap)
}

fn collect_file_edges(
    rel: &str, entry: &FileSymbols, tbl: &LookupTables,
    edge_set: &mut FxHashSet<(String, String)>, rt_set: &mut FxHashSet<(String, String)>,
    glob_targets: &mut FxHashSet<String>,
) {
    let (src_dir, src_ext) = (dir_of(rel), ext_of(rel));
    for imp in entry.typed_imports.iter().filter(|i| i.kind != EdgeKind::TestImport) {
        for target in resolve_import(&imp.target, rel, &src_dir, &src_ext, tbl) {
            if target == rel || !tbl.all_rels.contains(&target) { continue; }
            edge_set.insert((rel.to_string(), target.clone()));
            if imp.kind.is_runtime_scc() { rt_set.insert((rel.to_string(), target)); }
        }
    }
    for glob in &entry.glob_reexports {
        for target in resolve_import(glob, rel, &src_dir, &src_ext, tbl) {
            if target != rel { glob_targets.insert(target); }
        }
    }
}

type EdgeTuple = (Vec<(String, String)>, Vec<(String, String)>, FxHashSet<String>);

fn collect_edges(files: &BTreeMap<String, FileSymbols>, tbl: &LookupTables) -> EdgeTuple {
    let (mut edge_set, mut rt_set, mut glob_targets) =
        (FxHashSet::default(), FxHashSet::default(), FxHashSet::default());
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
    entry: &FileSymbols, is_glob: bool, witnesses: &mut FxHashMap<u64, RefWitness>,
) {
    for (idx, sym) in entry.defs.iter().enumerate() {
        let flags = entry.def_meta.get(idx).map_or(0, |meta| meta.boundary_flags);
        let (has_ffi, has_macro) =
            ((flags & BOUNDARY_FFI_NO_MANGLE) != 0, (flags & BOUNDARY_MACRO_DERIVE) != 0);
        if !is_glob && !has_ffi && !has_macro { continue; }
        let witness = witnesses.entry(hash_symbol(sym)).or_default();
        if is_glob { witness.add_flag(WITNESS_GLOB_REEXPORT); }
        if has_ffi { witness.add_flag(WITNESS_EXTERNAL); }
        if has_macro { witness.add_flag(WITNESS_MACRO_ATTR); }
    }
}

fn record_source_witnesses(
    files: &BTreeMap<String, FileSymbols>, glob_targets: &FxHashSet<String>,
    arena: &mut PathArena, witnesses: &mut FxHashMap<u64, RefWitness>,
) {
    for (rel, entry) in files {
        let fid = arena.intern(rel);
        for tok in &entry.refs_tokens { witnesses.entry(hash_symbol(tok)).or_default().record_prod(fid); }
        for tok in &entry.test_refs_tokens { witnesses.entry(hash_symbol(tok)).or_default().record_test(fid); }
        record_def_boundary_witnesses(entry, glob_targets.contains(rel), witnesses);
        for attr_sym in &entry.attr_refs {
            witnesses.entry(hash_symbol(attr_sym)).or_default().add_flag(WITNESS_DYNAMIC_REF);
        }
    }
}

type TestHashMap = BTreeMap<String, (String, Vec<u64>)>;

fn collect_test_hashes_cached(
    repo_root: &Path, preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    tests: &[(String, String, Option<PathBuf>)], prev_test_hashes: &TestHashMap,
) -> (TestHashMap, bool) {
    let rows = par_map(tests, |(rel, ext, _)| {
        let bytes = read_bytes(repo_root, rel, preloaded)?;
        let hash = MvccStore::hash_bytes(&bytes);
        if let Some((prev_h, cached_ids)) = prev_test_hashes.get(rel) && prev_h == &hash {
            return Some((rel.clone(), hash, cached_ids.clone(), false));
        }
        let parsed = parse_symbols(ext, &String::from_utf8_lossy(&bytes), String::new());
        let mut hashes: Vec<u64> = parsed.refs_tokens.iter()
            .chain(parsed.test_refs_tokens.iter()).map(|s| hash_symbol(s)).collect();
        hashes.sort_unstable();
        hashes.dedup();
        Some((rel.clone(), hash, hashes, true))
    });

    let (mut changed, mut map) = (false, BTreeMap::new());
    for (rel, hash, ids, was_parsed) in rows.into_iter().flatten() {
        changed |= was_parsed;
        map.insert(rel, (hash, ids));
    }
    changed |= map.len() != prev_test_hashes.len();
    (map, changed)
}

fn collect_dead_symbols_for_file(
    rel: &str, fid: u32, entry: &FileSymbols, glob_targets: &FxHashSet<String>,
    witnesses: &FxHashMap<u64, RefWitness>, dead: &mut Vec<DeadSymbol>,
) {
    let is_boundary_file = entrypoint(rel) || rel.ends_with("layout.tsx")
        || rel.ends_with("layout.ts") || glob_targets.contains(rel);
    let file_boundary = if is_boundary_file { BOUNDARY_ENTRYPOINT_OR_GLOB } else { 0 };
    for (idx, sym) in entry.defs.iter().enumerate() {
        let meta = entry.def_meta.get(idx).copied().unwrap_or_default();
        let witness = witnesses.get(&hash_symbol(sym)).copied().unwrap_or_default();
        if let Some(tier) =
            classify_symbol(fid, witness, meta.intra_non_def_hits, meta.boundary_flags | file_boundary)
            && tier.is_dead_candidate()
        {
            dead.push(DeadSymbol {
                file: rel.to_string(), symbol: sym.clone(), proof_tier: Some(tier.as_str().into()),
            });
        }
    }
}

fn build_graph(
    files: BTreeMap<String, FileSymbols>, test_hashes: &TestHashMap,
    edges: &[(String, String)], runtime_edges: &[(String, String)], glob_targets: &FxHashSet<String>,
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

// ============================================================================
// Zstd front-coded SST index (`k = 8`)
// ============================================================================

#[cfg(test)]
fn pkg_id_for_rel(rel: &str) -> String {
    let mut parts = rel.split('/');
    let Some(first) = parts.next() else { return "root".into() };
    if matches!(first, "packages" | "crates" | "apps" | "libs" | "services")
        && let Some(second) = parts.next() && !second.is_empty()
    {
        return format!("{first}/{second}");
    }
    if parts.next().is_some() { first.to_string() } else { "root".into() }
}

#[cfg(test)]
fn package_abi_changed(prev: Option<&[u8; 32]>, next: &[u8; 32]) -> bool { prev != Some(next) }

#[inline]
fn read_le_u32(buf: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(off..off + 4)?.try_into().ok()?))
}

#[inline]
fn read_le_u64(buf: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(buf.get(off..off + 8)?.try_into().ok()?))
}

#[inline]
fn write_varint(out: &mut Vec<u8>, mut val: u32) {
    while val >= 0x80 {
        out.push(((val as u8) & 0x7F) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline]
fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let (mut out, mut shift) = (0u32, 0u32);
    while *pos < buf.len() && shift < 32 {
        let b = buf[*pos];
        *pos += 1;
        out |= ((b & 0x7F) as u32) << shift;
        if (b & 0x80) == 0 { return Some(out); }
        shift += 7;
    }
    None
}

fn encode_front_coded_block(entries: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut data = Vec::with_capacity(entries.len().saturating_mul(24).max(64));
    let mut restarts = Vec::with_capacity((entries.len() / RESTART_INTERVAL) + 1);
    let mut prev: &[u8] = b"";
    for (idx, &(key, val)) in entries.iter().enumerate() {
        let shared = if idx % RESTART_INTERVAL == 0 {
            restarts.push(data.len() as u16);
            0
        } else {
            let mut n = 0usize;
            while n < prev.len() && n < key.len() && prev[n] == key[n] {
                n += 1;
            }
            n
        };
        let suffix = &key[shared..];
        write_varint(&mut data, shared as u32);
        write_varint(&mut data, suffix.len() as u32);
        data.extend_from_slice(suffix);
        write_varint(&mut data, val.len() as u32);
        data.extend_from_slice(val);
        prev = key;
    }
    for &off in &restarts { data.extend_from_slice(&off.to_le_bytes()); }
    data.extend_from_slice(&(restarts.len() as u16).to_le_bytes());
    data.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    data
}

fn block_tail_meta(block: &[u8]) -> Option<(usize, usize, usize)> {
    let tail = block.len();
    if tail < 4 { return None; }
    let n_keys = u16::from_le_bytes([block[tail - 2], block[tail - 1]]) as usize;
    let n_rest = u16::from_le_bytes([block[tail - 4], block[tail - 3]]) as usize;
    let rest_bytes = n_rest.checked_mul(2)?.checked_add(4)?;
    (n_rest > 0 && rest_bytes <= tail).then_some((n_keys, n_rest, tail - rest_bytes))
}

#[inline]
fn read_fc_entry<'a>(block: &'a [u8], pos: &mut usize, cur: &mut Vec<u8>) -> Option<&'a [u8]> {
    let shared = read_varint(block, pos)? as usize;
    let slen = read_varint(block, pos)? as usize;
    let suffix = block.get(*pos..pos.checked_add(slen)?)?;
    *pos += slen;
    let vlen = read_varint(block, pos)? as usize;
    let val = block.get(*pos..pos.checked_add(vlen)?)?;
    *pos += vlen;
    if shared > cur.len() { return None; }
    cur.truncate(shared);
    cur.extend_from_slice(suffix);
    Some(val)
}

fn lookup_front_coded_block<'a>(block: &'a [u8], target: &[u8]) -> Option<&'a [u8]> {
    let (n_keys, n_rest, rest_start) = block_tail_meta(block)?;
    let rest_off = |i: usize| -> Option<usize> {
        let o = rest_start.checked_add(i.checked_mul(2)?)?;
        let b = block.get(o..o + 2)?;
        Some(u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let (mut lo, mut hi) = (0usize, n_rest);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let mut p = rest_off(mid)?;
        let _shared = read_varint(block, &mut p)?;
        let slen = read_varint(block, &mut p)? as usize;
        if block.get(p..p.checked_add(slen)?)? <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (mut pos, mut cur) = (rest_off(lo)?, Vec::with_capacity(64));
    for _ in 0..n_keys.saturating_sub(lo * RESTART_INTERVAL).min(RESTART_INTERVAL) {
        let val = read_fc_entry(block, &mut pos, &mut cur)?;
        match cur.as_slice().cmp(target) {
            Ordering::Equal => return Some(val),
            Ordering::Greater => return None,
            Ordering::Less => {}
        }
    }
    None
}

fn decode_front_coded_block(block: &[u8], out: &mut Vec<(Vec<u8>, Vec<u8>)>) -> Option<()> {
    let (n_keys, _, _) = block_tail_meta(block)?;
    let (mut pos, mut cur) = (0usize, Vec::with_capacity(64));
    for _ in 0..n_keys {
        let val = read_fc_entry(block, &mut pos, &mut cur)?;
        out.push((cur.clone(), val.to_vec()));
    }
    Some(())
}

#[inline]
fn blake3_tag(chunks: &[&[u8]]) -> u32 {
    let mut h = blake3::Hasher::new();
    for chunk in chunks { h.update(chunk); }
    let d = h.finalize();
    let b = d.as_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn sst_block_tag(offset: u64, comp_len: u32, raw_len: u32, comp: &[u8]) -> u32 {
    blake3_tag(&[&offset.to_le_bytes(), &comp_len.to_le_bytes(), &raw_len.to_le_bytes(), comp])
}

struct BlockAnchor { first_key: Vec<u8>, offset: u64, comp_len: u32, raw_len: u32, tag: u32 }

fn parse_sst_anchors(footer: &[u8], index_offset: u64, block_count: usize) -> Option<Vec<BlockAnchor>> {
    let (mut pos, mut anchors) = (0usize, Vec::with_capacity(block_count.min(4096)));
    for _ in 0..block_count {
        let klen = read_varint(footer, &mut pos)? as usize;
        let first_key = footer.get(pos..pos.checked_add(klen)?)?.to_vec();
        pos += klen;
        let m = footer.get(pos..pos + 20)?;
        pos += 20;
        let (offset, comp_len) = (read_le_u64(m, 0)?, read_le_u32(m, 8)?);
        let (raw_len, tag) = (read_le_u32(m, 12)?, read_le_u32(m, 16)?);
        let end = offset.checked_add(comp_len as u64)?;
        if offset < SST_HEADER_LEN as u64 || end > index_offset || raw_len as usize > MAX_DECOMP_BLOCK_BYTES {
            return None;
        }
        anchors.push(BlockAnchor { first_key, offset, comp_len, raw_len, tag });
    }
    (pos == footer.len()).then_some(anchors)
}

fn next_sst_block_end(entries: &[(&[u8], &[u8])], idx: usize) -> usize {
    let (mut est, mut end) = (0usize, idx);
    while end < entries.len() && (end - idx) < MAX_BLOCK_KEYS {
        let sz = entries[end].0.len() + entries[end].1.len() + 8;
        if end > idx && (est >= TARGET_BLOCK_BYTES || est + sz > MAX_BLOCK_BYTES) { break; }
        est += sz;
        end += 1;
    }
    end
}

struct SstIndex {
    file: File,
    anchors: Vec<BlockAnchor>,
    entry_count: usize,
    blocks_decoded: usize,
    last_block: Option<(usize, Vec<u8>)>,
}

impl SstIndex {
    fn write_atomic(path: &Path, entries: &[(&[u8], &[u8])], prev_entries: usize) -> Result<bool> {
        if check_wipe_guard(prev_entries, entries.len()) {
            tracing::warn!(prev_entries, new_entries = entries.len(), "SST wipe guard triggered; preserving prior SST");
            return Ok(false);
        }
        let mut raw = vec![0u8; SST_HEADER_LEN];
        let (mut anchors, mut idx) = (Vec::new(), 0usize);
        while idx < entries.len() {
            let end = next_sst_block_end(entries, idx);
            let block = encode_front_coded_block(&entries[idx..end]);
            if block.len() > MAX_DECOMP_BLOCK_BYTES {
                tracing::warn!(block_len = block.len(), "SST block exceeds cap; skipping write");
                return Ok(false);
            }
            let comp = zstd::bulk::compress(&block, 1)?;
            let (offset, comp_len, raw_len) =
                (raw.len() as u64, u32::try_from(comp.len())?, u32::try_from(block.len())?);
            let tag = sst_block_tag(offset, comp_len, raw_len, &comp);
            anchors.push(BlockAnchor { first_key: entries[idx].0.to_vec(), offset, comp_len, raw_len, tag });
            raw.extend_from_slice(&comp);
            idx = end;
        }
        let index_offset = raw.len() as u64;
        for a in &anchors {
            write_varint(&mut raw, a.first_key.len() as u32);
            raw.extend_from_slice(&a.first_key);
            raw.extend_from_slice(&a.offset.to_le_bytes());
            for word in [a.comp_len, a.raw_len, a.tag] { raw.extend_from_slice(&word.to_le_bytes()); }
        }
        raw[0..8].copy_from_slice(SST_MAGIC);
        raw[8] = 1;
        raw[10..14].copy_from_slice(&(anchors.len() as u32).to_le_bytes());
        raw[14..18].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        raw[18..26].copy_from_slice(&index_offset.to_le_bytes());
        let ftag = blake3_tag(&[&raw[0..26], &raw[index_offset as usize..]]);
        raw[26..30].copy_from_slice(&ftag.to_le_bytes());
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            ensure_cachedir_tag(parent);
        }
        write_atomic(path, &raw)?;
        Ok(true)
    }

    fn open(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let file_len = file.metadata().ok()?.len();
        if file_len < SST_HEADER_LEN as u64 { return None; }
        let mut hdr = [0u8; SST_HEADER_LEN];
        file.read_exact(&mut hdr).ok()?;
        if &hdr[0..8] != SST_MAGIC || hdr[8] != 1 { return None; }
        let (block_count, entry_count) = (read_le_u32(&hdr, 10)? as usize, read_le_u32(&hdr, 14)? as usize);
        let (index_offset, expected_tag) = (read_le_u64(&hdr, 18)?, read_le_u32(&hdr, 26)?);
        let footer_len = file_len.checked_sub(index_offset)?;
        if index_offset < SST_HEADER_LEN as u64 || footer_len > MAX_DECOMP_BLOCK_BYTES as u64 {
            return None;
        }
        let mut footer = vec![0u8; usize::try_from(footer_len).ok()?];
        file.seek(SeekFrom::Start(index_offset)).ok()?;
        file.read_exact(&mut footer).ok()?;
        if blake3_tag(&[&hdr[0..26], &footer]) != expected_tag { return None; }
        let anchors = parse_sst_anchors(&footer, index_offset, block_count)?;
        Some(Self { file, anchors, entry_count, blocks_decoded: 0, last_block: None })
    }

    fn decode_block(&mut self, block_idx: usize) -> Option<&[u8]> {
        if !self.last_block.as_ref().is_some_and(|(idx, _)| *idx == block_idx) {
            let a = &self.anchors[block_idx];
            let mut comp = vec![0u8; a.comp_len as usize];
            self.file.seek(SeekFrom::Start(a.offset)).ok()?;
            self.file.read_exact(&mut comp).ok()?;
            if sst_block_tag(a.offset, a.comp_len, a.raw_len, &comp) != a.tag { return None; }
            let raw = zstd::bulk::decompress(&comp, a.raw_len as usize).ok()?;
            if raw.len() != a.raw_len as usize { return None; }
            self.blocks_decoded += 1;
            self.last_block = Some((block_idx, raw));
        }
        self.last_block.as_ref().map(|(_, buf)| buf.as_slice())
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let block_idx = match self.anchors.binary_search_by(|a| a.first_key.as_slice().cmp(key)) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(pos) => pos - 1,
        };
        lookup_front_coded_block(self.decode_block(block_idx)?, key).map(<[u8]>::to_vec)
    }

    fn decode_all(&mut self) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::with_capacity(self.entry_count);
        for idx in 0..self.anchors.len() {
            decode_front_coded_block(self.decode_block(idx)?, &mut out)?;
        }
        (out.len() == self.entry_count).then_some(out)
    }

    #[cfg(test)]
    fn blocks_decoded(&self) -> usize { self.blocks_decoded }
}

fn load_symbol_sst(path: &Path, expected_version: &str) -> Option<BTreeMap<String, FileSymbols>> {
    let mut sst = SstIndex::open(path)?;
    if sst.get(VERSION_KEY).as_deref() != Some(expected_version.as_bytes()) { return None; }
    let mut files = BTreeMap::new();
    for (key, val) in sst.decode_all()? {
        if let Some(rel) = key.strip_prefix(b"f:") {
            files.insert(String::from_utf8(rel.to_vec()).ok()?, serde_json::from_slice(&val).ok()?);
        }
    }
    Some(files)
}

fn save_symbol_sst(
    path: &Path, version: &str, files: &BTreeMap<String, FileSymbols>,
    edges: &[(String, String)], runtime_edges: &[(String, String)], prev_entries: usize,
) -> Result<bool> {
    let mut encoded = Vec::with_capacity(files.len() + 2);
    encoded.push((VERSION_KEY.to_vec(), version.as_bytes().to_vec()));
    encoded.push((EDGES_KEY.to_vec(), serde_json::to_vec(&(edges, runtime_edges))?));
    for (rel, sym) in files {
        encoded.push((format!("f:{rel}").into_bytes(), serde_json::to_vec(sym)?));
    }
    encoded.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    let refs: Vec<(&[u8], &[u8])> = encoded.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    SstIndex::write_atomic(path, &refs, prev_entries)
}

struct FileSstLookup { sym: FileSymbols, edges: Vec<(String, String)>, runtime_edges: Vec<(String, String)> }

fn query_file_sst(repo_root: &Path, expected_version: &str, norm_rel: &str) -> Option<FileSstLookup> {
    let (_, sst_path) = cache_paths(repo_root);
    let mut sst = sst_path.is_file().then(|| SstIndex::open(&sst_path))??;
    if sst.get(VERSION_KEY).as_deref() != Some(expected_version.as_bytes()) { return None; }
    let sym: FileSymbols = serde_json::from_slice(&sst.get(format!("f:{norm_rel}").as_bytes())?).ok()?;
    let disk_bytes = fs::read(repo_root.join(norm_rel)).ok()?;
    if MvccStore::hash_bytes(&disk_bytes) != sym.blake3 { return None; }
    let (edges, runtime_edges) = serde_json::from_slice(&sst.get(EDGES_KEY)?).ok()?;
    Some(FileSstLookup { sym, edges, runtime_edges })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn has_edge(edges: &[(String, String)], src: &str, dst: &str) -> bool {
        edges.iter().any(|(s, d)| s == src && d == dst)
    }

    #[test]
    fn symbol_index_polyglot_cache() {
        let leaf = concat!(
            "pub fn base_fn() -> i32 { 1 }\n#[cfg(test)]\nmod tests {\n",
            "    use crate::top::top_fn;\n    #[test] fn t() { assert_eq!(top_fn(), 1); }\n}\n",
        );
        let auth = "export function verifyToken() { return true; }\nexport function unusedGhost() { return false; }\n";
        let login = "import { verifyToken } from '../core/auth';\nexport default function Login() { return verifyToken(); }\n";
        let profile = "import { verifyToken } from '@/core/auth';\nexport function Profile() { return verifyToken(); }\n";
        let dir = TempRepo::with_files("sym_all", &[
            ("tsconfig.json", r#"{"compilerOptions":{"paths":{"@/*":["src/*"]}}}"#),
            ("src/core/auth.ts", auth),
            ("src/ui/login.tsx", login),
            ("src/ui/profile.tsx", profile),
            ("src/leaf.rs", leaf),
            ("src/top.rs", "use crate::leaf::base_fn;\npub fn top_fn() -> i32 { base_fn() }\n"),
        ]);

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
    fn dead_symbol_tiers_monorepo() {
        let web_a = concat!(
            "import type { UserDto } from '@shared/types';\n",
            "import { liveShared } from '@shared/types';\n",
            "export function runWeb(u: UserDto) { return liveShared() + u.id.length; }\n",
        );
        let rules = concat!(
            "use crate::sub::{\n    mod_helper,\n};\n",
            "#[unsafe(no_mangle)]\npub extern \"C\" fn abi_export() -> i32 { 1 }\n",
            "#[ctor::ctor]\npub fn ctor_hook() {}\n",
            "pub(crate) fn recursive_dead(n: u32) -> u32 {\n",
            "    if n == 0 { 0 } else { recursive_dead(n - 1) }\n}\n",
            "pub(crate) fn intra_helper() -> i32 { mod_helper() }\n",
            "pub fn entry_caller() -> i32 { intra_helper() }\n",
            "pub(crate) fn test_only_helper() -> i32 { 42 }\n",
            "#[cfg(test)]\nmod tests {\n",
            "    fn chk(buf: [u8; 16]) -> usize { buf.len() + super::test_only_helper() as usize }\n}\n",
        );
        let store_go = concat!(
            "package store\n//go:linkname LinkedHook runtime.hook\n",
            "func LinkedHook() int { return 1 }\nfunc UnusedGoFn() int { return 2 }\n",
        );
        let handler_go = concat!(
            "package api\nimport \"example.com/acme/pkg/store\"\n",
            "func ServeHTTP() int { return store.LinkedHook() }\n",
        );
        let dir = TempRepo::with_files("stem_v2_v4", &[
            ("tsconfig.base.json", r#"{"compilerOptions":{"paths":{"@shared/*":["packages/shared/src/*"]}}}"#),
            ("packages/web/tsconfig.json", r#"{"extends":"../../tsconfig.base.json","compilerOptions":{"paths":{"@web/*":["src/*"]}}}"#),
            ("packages/shared/src/types.ts", "export interface UserDto { id: string }\nexport function liveShared() { return 1; }\n"),
            ("packages/web/src/a.ts", web_a),
            ("packages/web/src/b.ts", "import type { runWeb } from '@web/a';\nexport const keepTypeEdgeOutOfScc = 1;\n"),
            ("src/rules.rs", rules),
            ("src/sub/mod.rs", "pub fn mod_helper() -> i32 { 7 }\n"),
            ("src/pymod/api.py", "__all__ = ['public_exported']\ndef public_exported():\n    return 1\ndef dead_py_helper():\n    return 2\n"),
            ("go.mod", "module example.com/acme\n\ngo 1.24\n"),
            ("pkg/store/store.go", store_go),
            ("pkg/api/handler.go", handler_go),
        ]);

        let (disc, snap) = index_repo(dir.path(), None);
        assert!(has_edge(&disc.runtime_edges, "src/rules.rs", "src/sub/mod.rs"));
        assert!(has_edge(&disc.runtime_edges, "packages/web/src/a.ts", "packages/shared/src/types.ts"));
        assert!(!has_edge(&disc.runtime_edges, "packages/web/src/b.ts", "packages/web/src/a.ts"));
        assert!(has_edge(&disc.runtime_edges, "pkg/api/handler.go", "pkg/store/store.go"));

        let dead_names: HashMap<&str, Option<&str>> =
            snap.dead_symbols.iter().map(|d| (d.symbol.as_str(), d.proof_tier.as_deref())).collect();
        assert_eq!(dead_names.get("recursive_dead"), Some(&Some("T1ProvenDead")));
        assert_eq!(dead_names.get("dead_py_helper"), Some(&Some("T2LikelyDead")));
        assert_eq!(dead_names.get("UnusedGoFn"), Some(&Some("T2LikelyDead")));
        for safe in [
            "intra_helper", "test_only_helper", "abi_export", "ctor_hook",
            "public_exported", "LinkedHook", "ServeHTTP",
        ] {
            assert!(!dead_names.contains_key(safe), "false dead symbol: {safe}");
        }
    }

    #[test]
    fn sst_zstd_lazy_and_threshold() {
        assert!(!should_use_sst(499, 4096));
        assert_eq!(should_use_sst(500, 4096), cfg!(feature = "sst-storage"));
        assert_eq!(pkg_id_for_rel("packages/core/src/lib.rs"), "packages/core");
        assert_eq!(pkg_id_for_rel("src/lib.rs"), "src");
        assert_eq!(pkg_id_for_rel("lib.rs"), "root");
        let (h1, h2) = ([1u8; 32], [2u8; 32]);
        assert!(
            package_abi_changed(None, &h1)
                && !package_abi_changed(Some(&h1), &h1)
                && package_abi_changed(Some(&h1), &h2)
        );

        fn as_pairs(entries: &[(String, String)]) -> Vec<(&str, &str)> {
            entries.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
        }
        let small: Vec<_> = (0..20)
            .map(|i| (format!("src/s_{i:02}.rs"), format!("pub fn small_fn_{i}() -> i32 {{ {i} }}\n")))
            .collect();
        let small_repo = TempRepo::with_files("sst_small", &as_pairs(&small));
        let _ = index_repo(small_repo.path(), None);
        assert!(small_repo.join(".refit/cache/symbol_index.zst").is_file());
        assert!(!small_repo.join(".refit/cache/sym.sst").exists());
        assert!(!should_use_sst(20, 4096));

        let mut big = vec![("src/m_000.rs".to_string(), "pub fn sym_000() -> i32 { 0 }\n".to_string())];
        for i in 1..520 {
            let p = i - 1;
            let body = format!(
                "use crate::m_{p:03}::sym_{p:03};\npub fn sym_{i:03}() -> i32 {{ sym_{p:03}() + {i} }}\n"
            );
            big.push((format!("src/m_{i:03}.rs"), body));
        }
        let big_repo = TempRepo::with_files("sst_big", &as_pairs(&big));
        let (disc, snap) = index_repo(big_repo.path(), None);
        let (zst_path, sst_path) = cache_paths(big_repo.path());
        if cfg!(feature = "sst-storage") {
            assert!(should_use_sst(520, 4096) && sst_path.is_file() && !zst_path.exists());
        } else {
            let loaded = read_zstd::<Value>(&zst_path).unwrap();
            let files: BTreeMap<String, FileSymbols> =
                serde_json::from_value(loaded["files"].clone()).unwrap();
            assert!(save_symbol_sst(&sst_path, "0.6.0", &files, &disc.edges, &disc.runtime_edges, 0).unwrap());
            let _ = fs::remove_file(&zst_path);
        }

        let sst_bytes = fs::read(&sst_path).unwrap();
        assert_eq!(&sst_bytes[0..8], SST_MAGIC);
        let mut idx = SstIndex::open(&sst_path).unwrap();
        let decoded = idx.decode_all().unwrap();
        let raw_json: usize = decoded.iter().map(|(k, v)| k.len() + v.len()).sum();
        assert!(
            (sst_bytes.len() as f64) < (raw_json as f64) * 0.35,
            "sst={}B raw={}B", sst_bytes.len(), raw_json
        );

        let mut lazy = SstIndex::open(&sst_path).unwrap();
        assert_eq!(lazy.blocks_decoded(), 0);
        let sym_250: FileSymbols = serde_json::from_slice(&lazy.get(b"f:src/m_250.rs").unwrap()).unwrap();
        assert_eq!((lazy.blocks_decoded(), sym_250.defs), (1, vec!["sym_250".to_string()]));

        let q_sst = query_symbols(big_repo.path(), None, Some("src/m_250.rs"));
        assert_eq!(q_sst["defs"], json!(snap.defs_by_file["src/m_250.rs"]));
        assert_eq!(q_sst["imports"], json!(snap.imports_by_file["src/m_250.rs"]));
        assert_eq!(
            (&q_sst["fan_in"], &q_sst["fan_out"]),
            (&json!(snap.fan_in["src/m_250.rs"]), &json!(snap.fan_out["src/m_250.rs"]))
        );

        let mut tampered = sst_bytes;
        let mid = SST_HEADER_LEN + (tampered.len() - SST_HEADER_LEN) / 3;
        tampered[mid] ^= 0x01;
        fs::write(&sst_path, &tampered).unwrap();
        assert!(load_symbol_sst(&sst_path, "0.6.0").is_none());
        assert_eq!(query_symbols(big_repo.path(), None, Some("src/m_250.rs")), q_sst);
    }
}
