pub(crate) mod csr;
pub(crate) mod genome;
mod parse;
mod strip;
pub(crate) mod witness;

pub(crate) use strip::strip_noise;

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{mkdir_private, read_zstd, walk_files, write_zstd};
use crate::source::{dir_of, entrypoint, ext_of, read_bytes, rel_norm, same_family, scan_files, stem_of};
use crate::types::DeadSymbol;
use csr::{EdgeKind, ModuleMetrics};
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
use witness::{
    BOUNDARY_ENTRYPOINT_OR_GLOB, BOUNDARY_FFI_NO_MANGLE, BOUNDARY_MACRO_DERIVE, PathArena,
    RefWitness, WITNESS_DYNAMIC_REF, WITNESS_EXTERNAL, WITNESS_GLOB_REEXPORT, WITNESS_MACRO_ATTR,
    classify_symbol, hash_symbol,
};

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
}

#[derive(Serialize)]
struct SymbolCacheRef<'a> {
    version: &'a str,
    files: &'a BTreeMap<String, FileSymbols>,
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
    if let Some(h) = hint.filter(|p| p.exists()) {
        return Some(h.to_path_buf());
    }
    if let Ok(env_bin) = std::env::var("REFIT_STEM_BIN") {
        let p = PathBuf::from(env_bin);
        if p.exists() {
            return Some(p);
        }
    }

    let bin = if cfg!(windows) { "stem.exe" } else { "stem" };
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(bin)));
    sibling
        .into_iter()
        .chain(std::iter::once(PathBuf::from(".tools").join(bin)))
        .find(|c| c.exists())
}

pub fn scan_edges(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> EdgeScan {
    let started = Instant::now();
    let (mut disc, _, _, _) = discover_internal(repo_root, preloaded);
    let mut edge_set: HashSet<(String, String)> = disc.edges.into_iter().collect();
    let mut rt_set: HashSet<(String, String)> = disc.runtime_edges.into_iter().collect();

    let want_ext = stem_bin_hint.is_some()
        || std::env::var("REFIT_USE_STEM_BIN").as_deref() == Ok("1");
    if want_ext && query_ext(repo_root, stem_bin_hint, &mut edge_set) {
        disc.used_stem = true;
        rt_set.extend(edge_set.iter().cloned());
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    let mut runtime_edges: Vec<(String, String)> = rt_set.into_iter().collect();
    edges.sort();
    runtime_edges.sort();
    EdgeScan {
        used_stem: disc.used_stem,
        index_ms: started.elapsed().as_millis() as u64,
        edges,
        runtime_edges,
    }
}

fn query_ext(
    repo_root: &Path,
    hint: Option<&Path>,
    edge_set: &mut HashSet<(String, String)>,
) -> bool {
    let Some(stem_bin) = locate_stem(hint) else { return false };
    let stem_home = repo_root.join(".refit/stem-home");
    if mkdir_private(&stem_home).is_err() {
        return false;
    }

    let mut cmd = Command::new(&stem_bin);
    cmd.arg("mcp")
        .current_dir(repo_root)
        .env_clear()
        .env("HOME", &stem_home)
        .env("USERPROFILE", &stem_home)
        .env("STEM_MCP_TOOLS", "defs,refs,callers,impls,symbols,sql,brief")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for k in ["PATH", "SYSTEMROOT", "TEMP", "TMP"] {
        if let Ok(v) = std::env::var(k) {
            cmd.env(k, v);
        }
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
    if let Some(mut stdin) = child.stdin.take() {
        let _ = writeln!(stdin, "{init}\n{call}");
    }

    let Ok(out) = child.wait_with_output() else { return false };
    if !out.status.success() {
        return false;
    }

    let mut used = false;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { continue };
        let id = v["id"].as_u64().unwrap_or(0);
        used |= id == 1 && v.get("result").is_some();
        if id == 2
            && let Some(rows) = v["result"]["structuredContent"]["rows"].as_array()
        {
            for row in rows {
                if let (Some(s), Some(d)) = (row["src"].as_str(), row["dst"].as_str()) {
                    edge_set.insert((s.replace('\\', "/"), d.replace('\\', "/")));
                }
            }
        }
    }
    used
}

pub fn index_repo(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (EdgeScan, SymbolGraph) {
    let (disc, _, _, snap) = discover_internal(repo_root, preloaded);
    (disc, snap)
}

pub fn load_symbols(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> SymbolGraph {
    discover_internal(repo_root, preloaded).3
}

pub fn query_symbols(repo_root: &Path, symbol: Option<&str>, file: Option<&str>) -> Value {
    let snap = load_symbols(repo_root, None);
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
        let deps: Vec<String> = snap
            .edges
            .iter()
            .filter(|(s, _)| s == &norm)
            .map(|(_, d)| d.clone())
            .collect();
        let dependents: Vec<String> = snap
            .edges
            .iter()
            .filter(|(_, d)| d == &norm)
            .map(|(s, _)| s.clone())
            .collect();
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
        "indexed_files": snap.defs_by_file.len(),
        "total_edges": snap.edges.len(),
        "dead_symbols_count": snap.dead_symbols.len(),
        "dead_symbols": snap.dead_symbols.iter().take(40).collect::<Vec<_>>(),
        "top_fan_in": top_fan(&snap.fan_in, 10),
        "top_fan_out": top_fan(&snap.fan_out, 10)
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
    let (d, p, c, _) = discover_internal(repo_root, None);
    (d, p, c)
}

fn load_tsconfig_resolved(
    repo_root: &Path,
    cfg_path: &Path,
    depth: usize,
) -> (String, Vec<(String, String)>) {
    if depth > 8 {
        return (String::new(), Vec::new());
    }
    let Ok(raw) = std::fs::read_to_string(cfg_path) else {
        return (String::new(), Vec::new());
    };
    let cleaned = strip_jsonc(&raw);
    let Ok(v) = serde_json::from_str::<Value>(&cleaned) else {
        return (String::new(), Vec::new());
    };

    let cfg_dir = cfg_path.parent().unwrap_or(repo_root);
    let (mut base, mut aliases) = if let Some(ext_rel) = v["extends"].as_str() {
        let mut parent = cfg_dir.join(ext_rel);
        if !parent.exists() && parent.extension().is_none() {
            parent.set_extension("json");
        }
        let canon_parent = dunce::canonicalize(&parent).unwrap_or(parent);
        load_tsconfig_resolved(repo_root, &canon_parent, depth + 1)
    } else {
        (String::new(), Vec::new())
    };

    let canon_root = dunce::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    let canon_dir = dunce::canonicalize(cfg_dir).unwrap_or_else(|_| cfg_dir.to_path_buf());
    let rel_dir = rel_norm(&canon_root, &canon_dir)
        .or_else(|| rel_norm(repo_root, cfg_dir))
        .unwrap_or_default();
    if let Some(b) = v["compilerOptions"]["baseUrl"].as_str() {
        let clean_b = b.trim_start_matches("./").trim_matches('/');
        base = if rel_dir.is_empty() {
            clean_b.to_string()
        } else if clean_b.is_empty() || clean_b == "." {
            rel_dir.clone()
        } else {
            format!("{rel_dir}/{clean_b}")
        };
    }

    if let Some(paths) = v["compilerOptions"]["paths"].as_object() {
        let effective_base = if base.is_empty() { rel_dir.as_str() } else { base.as_str() };
        for (alias, targets) in paths {
            let prefix = alias.trim_end_matches('*');
            let Some(arr) = (!prefix.is_empty()).then(|| targets.as_array()).flatten() else {
                continue;
            };
            aliases.retain(|(p, _)| p != prefix);
            for t in arr.iter().filter_map(Value::as_str) {
                let raw_t = t.trim_start_matches("./").trim_end_matches('*');
                let mapped = if effective_base.is_empty() || effective_base == "." {
                    raw_t.to_string()
                } else if raw_t.is_empty() {
                    format!("{effective_base}/")
                } else {
                    join_rel(effective_base, raw_t)
                };
                aliases.push((prefix.to_string(), mapped));
            }
        }
    }

    (base, aliases)
}

fn discover_ts_scopes(repo_root: &Path) -> Vec<TsConfigScope> {
    let mut cfg_paths = Vec::new();
    let root_cfg = repo_root.join("tsconfig.json");
    if root_cfg.exists() {
        cfg_paths.push(root_cfg);
    }
    for p in walk_files(repo_root, None) {
        if let Some(name) = p.file_name().and_then(|s| s.to_str())
            && matches!(name, "tsconfig.json" | "jsconfig.json")
            && !cfg_paths.contains(&p)
        {
            cfg_paths.push(p);
        }
    }

    let mut scopes = Vec::new();
    for cfg in cfg_paths {
        let (_, aliases) = load_tsconfig_resolved(repo_root, &cfg, 0);
        if aliases.is_empty() {
            continue;
        }
        let pkg_dir = cfg
            .parent()
            .and_then(|d| rel_norm(repo_root, d))
            .unwrap_or_default();
        scopes.push(TsConfigScope { pkg_dir, aliases });
    }
    // Longest `pkg_dir` first for most-specific package scope matching.
    scopes.sort_by(|a, b| b.pkg_dir.len().cmp(&a.pkg_dir.len()).then_with(|| a.pkg_dir.cmp(&b.pkg_dir)));
    scopes
}

fn discover_go_modules(repo_root: &Path) -> Vec<(String, String)> {
    let mut mod_dirs = Vec::new();
    if repo_root.join("go.mod").exists() {
        mod_dirs.push(PathBuf::new());
    }
    if let Ok(work) = std::fs::read_to_string(repo_root.join("go.work")) {
        let mut in_use = false;
        for line in work.lines().map(str::trim) {
            if line.starts_with("use (") {
                in_use = true;
                continue;
            }
            if in_use && line.starts_with(')') {
                in_use = false;
                continue;
            }
            let cand = if in_use {
                Some(line)
            } else {
                line.strip_prefix("use ")
            };
            if let Some(raw) = cand {
                let cleaned = raw
                    .split("//")
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_start_matches("./");
                if !cleaned.is_empty() {
                    mod_dirs.push(PathBuf::from(cleaned));
                }
            }
        }
    }

    let mut out = Vec::new();
    for sub in mod_dirs {
        let gm = if sub.as_os_str().is_empty() {
            repo_root.join("go.mod")
        } else {
            repo_root.join(&sub).join("go.mod")
        };
        let Ok(txt) = std::fs::read_to_string(&gm) else { continue };
        for line in txt.lines().map(str::trim) {
            if let Some(rest) = line.strip_prefix("module ") {
                let mod_path = rest.split_whitespace().next().unwrap_or("").trim_matches('"');
                if !mod_path.is_empty() {
                    let dir_str = sub.to_string_lossy().replace('\\', "/");
                    out.push((dir_str, mod_path.to_string()));
                }
                break;
            }
        }
    }
    out
}

fn join_rel(base_dir: &str, rel_import: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in base_dir.split('/').chain(rel_import.split('/')) {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    let trailing_slash = rel_import.ends_with('/');
    let mut joined = parts.join("/");
    if trailing_slash && !joined.is_empty() {
        joined.push('/');
    }
    joined
}

fn build_tables(
    repo_root: &Path,
    sources: &[(String, String, Option<PathBuf>)],
    all_files: &[(String, String, Option<PathBuf>)],
) -> LookupTables {
    let go_mods = discover_go_modules(repo_root);
    let mut tbl = LookupTables {
        all_rels: FxHashSet::with_capacity_and_hasher(sources.len(), Default::default()),
        mod_to_rel: FxHashMap::with_capacity_and_hasher(sources.len() * 2, Default::default()),
        stem_to_rels: FxHashMap::default(),
        go_pkg_rels: FxHashMap::default(),
        go_full_pkg_rels: FxHashMap::default(),
        ts_scopes: discover_ts_scopes(repo_root),
        rs_crate_roots: Vec::new(),
    };

    let mut crate_roots = FxHashSet::default();
    for (rel, ext, _) in all_files {
        if ext == "py" && rel.ends_with("/__init__.py") {
            let pkg = rel.trim_end_matches("/__init__.py");
            tbl.mod_to_rel.insert(pkg.replace('/', "."), rel.clone());
            if let Some(stripped) = pkg.strip_prefix("src/") {
                tbl.mod_to_rel.insert(stripped.replace('/', "."), rel.clone());
            }
        }
    }

    for (rel, ext, _) in sources {
        tbl.all_rels.insert(rel.clone());
        let p = Path::new(rel);
        if ext == "py" {
            let py_mod = rel.trim_end_matches(".py").replace('/', ".");
            tbl.mod_to_rel.insert(py_mod, rel.clone());
            if let Some(stripped) = rel.strip_prefix("src/") {
                let src_mod = stripped.trim_end_matches(".py").replace('/', ".");
                tbl.mod_to_rel.entry(src_mod).or_insert_with(|| rel.clone());
            }
        } else {
            let no_ext = rel.rsplit_once('.').map_or(rel.as_str(), |(a, _)| a);
            tbl.mod_to_rel.insert(no_ext.to_string(), rel.clone());
            if ext == "rs" {
                if let Some(mod_dir) = rel.strip_suffix("/mod.rs") {
                    tbl.mod_to_rel.insert(mod_dir.to_string(), rel.clone());
                }
                if rel == "src/lib.rs" || rel == "src/main.rs" {
                    crate_roots.insert("src".to_string());
                } else if let Some(prefix) = rel.strip_suffix("/src/lib.rs").or_else(|| rel.strip_suffix("/src/main.rs")) {
                    crate_roots.insert(format!("{prefix}/src"));
                } else if rel.starts_with("src/") {
                    crate_roots.insert("src".to_string());
                }
            }
            if ext == "go" {
                let rel_dir = dir_of(rel);
                if let Some(dir) = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()) {
                    tbl.go_pkg_rels.entry(dir.to_string()).or_default().push(rel.clone());
                }
                for (mod_dir, mod_path) in &go_mods {
                    let full_pkg = if mod_dir.is_empty() {
                        if rel_dir.is_empty() {
                            Some(mod_path.clone())
                        } else {
                            Some(format!("{mod_path}/{rel_dir}"))
                        }
                    } else if rel_dir == *mod_dir {
                        Some(mod_path.clone())
                    } else {
                        rel_dir
                            .strip_prefix(&format!("{mod_dir}/"))
                            .map(|sub| format!("{mod_path}/{sub}"))
                    };
                    if let Some(fp) = full_pkg {
                        tbl.go_full_pkg_rels.entry(fp).or_default().push(rel.clone());
                    }
                }
            }
        }
        let stem = stem_of(p);
        if !stem.is_empty() {
            tbl.stem_to_rels.entry(stem.to_string()).or_default().push(rel.clone());
        }
    }

    let mut roots: Vec<String> = crate_roots.into_iter().collect();
    roots.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    tbl.rs_crate_roots = roots;
    tbl
}

fn discover_internal(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (EdgeScan, usize, usize, SymbolGraph) {
    let started = Instant::now();
    let cache_path = repo_root.join(".refit/cache/symbol_index.zst");
    let prev: SymbolCache = read_zstd(&cache_path)
        .filter(|idx: &SymbolCache| idx.version == CACHE_VERSION)
        .unwrap_or_default();

    let scan = scan_files(repo_root, preloaded);
    let tables = build_tables(repo_root, &scan.sources, &scan.all);

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

    if parsed_n > 0 || prev.files.len() != new_files.len() {
        if genome::check_wipe_guard(prev.files.len(), new_files.len()) {
            tracing::warn!(
                prev_files = prev.files.len(),
                new_files = new_files.len(),
                "index-wipe guard triggered (>80% files vanished); keeping previous symbol index cache"
            );
        } else {
            let next_ref = SymbolCacheRef {
                version: CACHE_VERSION,
                files: &new_files,
            };
            let _ = write_zstd(&cache_path, &next_ref);
        }
    }

    let (edges, runtime_edges, glob_reexport_targets) = collect_edges(&new_files, &tables);
    let snap = build_graph(
        repo_root,
        preloaded,
        new_files,
        &scan.tests,
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

fn collect_edges(
    files: &BTreeMap<String, FileSymbols>,
    tbl: &LookupTables,
) -> (Vec<(String, String)>, Vec<(String, String)>, FxHashSet<String>) {
    let mut edge_set: FxHashSet<(String, String)> = FxHashSet::default();
    let mut rt_set: FxHashSet<(String, String)> = FxHashSet::default();
    let mut glob_targets: FxHashSet<String> = FxHashSet::default();

    for (rel, entry) in files {
        let src_dir = dir_of(rel);
        let src_ext = ext_of(rel);

        for ti in &entry.typed_imports {
            if ti.kind == EdgeKind::TestImport {
                continue;
            }
            for target in resolve_import(&ti.target, rel, &src_dir, &src_ext, tbl) {
                if target != *rel && tbl.all_rels.contains(&target) {
                    edge_set.insert((rel.clone(), target.clone()));
                    if ti.kind.is_runtime_scc() {
                        rt_set.insert((rel.clone(), target));
                    }
                }
            }
        }

        for glob in &entry.glob_reexports {
            for target in resolve_import(glob, rel, &src_dir, &src_ext, tbl) {
                if target != *rel {
                    glob_targets.insert(target);
                }
            }
        }
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    let mut runtime_edges: Vec<(String, String)> = rt_set.into_iter().collect();
    edges.sort();
    runtime_edges.sort();
    (edges, runtime_edges, glob_targets)
}

fn build_graph(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    files: BTreeMap<String, FileSymbols>,
    tests: &[(String, String, Option<PathBuf>)],
    edges: &[(String, String)],
    runtime_edges: &[(String, String)],
    glob_targets: &FxHashSet<String>,
) -> SymbolGraph {
    let (mut fan_in, mut fan_out) = (HashMap::new(), HashMap::new());
    for (s, d) in runtime_edges {
        *fan_out.entry(s.clone()).or_insert(0) += 1;
        *fan_in.entry(d.clone()).or_insert(0) += 1;
    }

    let mut arena = PathArena::with_capacity(files.len() + tests.len());
    let mut witnesses: FxHashMap<u64, RefWitness> =
        FxHashMap::with_capacity_and_hasher(files.len() * 16 + 64, Default::default());

    for (rel, entry) in &files {
        let fid = arena.intern(rel);
        for tok in &entry.refs_tokens {
            witnesses.entry(hash_symbol(tok)).or_default().record_prod(fid);
        }
        for tok in &entry.test_refs_tokens {
            witnesses.entry(hash_symbol(tok)).or_default().record_test(fid);
        }
        let is_glob = glob_targets.contains(rel);
        for (idx, sym) in entry.defs.iter().enumerate() {
            let flags = entry.def_meta.get(idx).map_or(0, |m| m.boundary_flags);
            if is_glob || (flags & (BOUNDARY_FFI_NO_MANGLE | BOUNDARY_MACRO_DERIVE)) != 0 {
                let w = witnesses.entry(hash_symbol(sym)).or_default();
                if is_glob {
                    w.add_flag(WITNESS_GLOB_REEXPORT);
                }
                if (flags & BOUNDARY_FFI_NO_MANGLE) != 0 {
                    w.add_flag(WITNESS_EXTERNAL);
                }
                if (flags & BOUNDARY_MACRO_DERIVE) != 0 {
                    w.add_flag(WITNESS_MACRO_ATTR);
                }
            }
        }
        for attr_sym in &entry.attr_refs {
            witnesses
                .entry(hash_symbol(attr_sym))
                .or_default()
                .add_flag(WITNESS_DYNAMIC_REF);
        }
    }

    // Stream test files and fold their tokens directly into 16B RefWitnesses (0 retained test bodies).
    let scan_test = |(rel, ext, _): &(String, String, Option<PathBuf>)| -> Option<(String, Vec<u64>)> {
        let bytes = read_bytes(repo_root, rel, preloaded)?;
        let parsed = parse_symbols(ext, &String::from_utf8_lossy(&bytes), String::new());
        let mut hashes: Vec<u64> = parsed
            .refs_tokens
            .iter()
            .chain(parsed.test_refs_tokens.iter())
            .map(|s| hash_symbol(s))
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        Some((rel.clone(), hashes))
    };

    let test_hashes: Vec<(String, Vec<u64>)> = if crate::platform::budget().is_constrained() {
        tests.iter().filter_map(scan_test).collect()
    } else {
        tests.par_iter().filter_map(scan_test).collect()
    };

    for (trel, hashes) in test_hashes {
        let tid = arena.intern(&trel);
        for h in hashes {
            witnesses.entry(h).or_default().record_test(tid);
        }
    }

    let (mut defs_by, mut refs_by, mut imps_by, mut metrics_by) =
        (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    let mut dead = Vec::new();
    for (rel, entry) in files {
        let fid = arena.id_of(&rel).unwrap_or(witness::NO_FILE);
        let file_boundary = if entrypoint(&rel)
            || rel.ends_with("layout.tsx")
            || rel.ends_with("layout.ts")
            || glob_targets.contains(&rel)
        {
            BOUNDARY_ENTRYPOINT_OR_GLOB
        } else {
            0
        };

        for (idx, sym) in entry.defs.iter().enumerate() {
            let meta = entry.def_meta.get(idx).copied().unwrap_or_default();
            let w = witnesses.get(&hash_symbol(sym)).copied().unwrap_or_default();
            if let Some(tier) =
                classify_symbol(fid, w, meta.intra_non_def_hits, meta.boundary_flags | file_boundary)
                && tier.is_dead_candidate()
            {
                dead.push(DeadSymbol {
                    file: rel.clone(),
                    symbol: sym.clone(),
                    proof_tier: Some(tier.as_str().to_string()),
                });
            }
        }
        defs_by.insert(rel.clone(), entry.defs);
        refs_by.insert(rel.clone(), entry.refs_tokens);
        imps_by.insert(rel.clone(), entry.imports);
        metrics_by.insert(rel, entry.metrics);
    }

    SymbolGraph {
        defs_by_file: defs_by,
        refs_by_file: refs_by,
        imports_by_file: imps_by,
        edges: edges.to_vec(),
        runtime_edges: runtime_edges.to_vec(),
        fan_in,
        fan_out,
        dead_symbols: dead,
        metrics_by_file: metrics_by,
    }
}

fn strip_js_ext(p: &str) -> &str {
    for ext in [".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs", ".mts", ".cts"] {
        if let Some(s) = p.strip_suffix(ext) {
            return s;
        }
    }
    p
}

fn resolve_import(
    imp: &str,
    src_rel: &str,
    src_dir: &str,
    src_ext: &str,
    tbl: &LookupTables,
) -> Vec<String> {
    if imp.starts_with("./") || imp.starts_with("../") {
        let base = join_rel(src_dir, strip_js_ext(imp));
        for cand in [base.clone(), format!("{base}/index")] {
            if let Some(hit) = tbl.mod_to_rel.get(&cand) {
                return vec![hit.clone()];
            }
        }
        if tbl.all_rels.contains(&base) {
            return vec![base];
        }
    }

    if src_ext == "py"
        && imp.starts_with('.')
        && let Some(hit) = resolve_pyrel(imp, src_dir, tbl)
    {
        return vec![hit];
    }

    // Per-package tsconfig.json paths/extends (longest matching package scope first).
    for scope in &tbl.ts_scopes {
        let in_scope = scope.pkg_dir.is_empty()
            || src_dir == scope.pkg_dir
            || src_dir.starts_with(&format!("{}/", scope.pkg_dir));
        if !in_scope {
            continue;
        }
        for (alias, target_prefix) in &scope.aliases {
            if let Some(rest) = imp.strip_prefix(alias.as_str()) {
                let mapped = format!("{target_prefix}{}", strip_js_ext(rest));
                if let Some(hit) = tbl
                    .mod_to_rel
                    .get(&mapped)
                    .or_else(|| tbl.mod_to_rel.get(&format!("{mapped}/index")))
                {
                    return vec![hit.clone()];
                }
            }
        }
    }

    if src_ext == "rs" {
        if let Some(sym) = imp.strip_prefix("super::") {
            let parent = dir_of(src_dir);
            let cand = if parent.is_empty() { sym.into() } else { format!("{parent}/{sym}") };
            if let Some(hit) = tbl.mod_to_rel.get(&cand) {
                return vec![hit.clone()];
            }
        }
        // Resolve `use crate::foo::bar` / `mod foo;` against enclosing crate's `src/` root and `src_dir`.
        if !src_dir.is_empty() {
            let sub = format!("{src_dir}/{imp}");
            if let Some(hit) = tbl.mod_to_rel.get(&sub) {
                return vec![hit.clone()];
            }
        }
        if let Some(root) = tbl
            .rs_crate_roots
            .iter()
            .find(|r| src_rel.starts_with(&format!("{r}/")))
        {
            let cand = format!("{root}/{imp}");
            if let Some(hit) = tbl.mod_to_rel.get(&cand) {
                return vec![hit.clone()];
            }
        }
    }

    if src_ext == "go" {
        if let Some(files) = tbl.go_full_pkg_rels.get(imp) {
            return files.clone();
        }
        let last = imp.rsplit('/').next().unwrap_or(imp);
        if let Some(files) = tbl.go_pkg_rels.get(last) {
            return files.clone();
        }
    }

    if let Some(hit) = tbl.mod_to_rel.get(imp) {
        return vec![hit.clone()];
    }

    resolve_stem(imp, src_dir, src_ext, tbl)
}

fn resolve_pyrel(imp: &str, src_dir: &str, tbl: &LookupTables) -> Option<String> {
    let dots = imp.bytes().take_while(|&b| b == b'.').count();
    let mut parts: Vec<&str> = src_dir.split('/').filter(|s| !s.is_empty()).collect();
    for _ in 1..dots {
        parts.pop();
    }
    parts.extend(imp[dots..].split('.').filter(|s| !s.is_empty()));
    tbl.mod_to_rel.get(&parts.join(".")).cloned()
}

fn resolve_stem(imp: &str, src_dir: &str, src_ext: &str, tbl: &LookupTables) -> Vec<String> {
    let key = imp.strip_prefix("super::").unwrap_or(imp);
    let Some(cands) = tbl.stem_to_rels.get(key) else {
        return Vec::new();
    };

    if !imp.starts_with("super::")
        && let Some(same) = cands.iter().find(|c| dir_of(c) == src_dir)
    {
        return vec![same.clone()];
    }

    let matches: Vec<&String> = cands
        .iter()
        .filter(|c| same_family(src_ext, &ext_of(c)))
        .collect();
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
