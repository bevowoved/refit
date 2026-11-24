mod parse;
mod strip;

pub(crate) use strip::strip_noise;

use crate::lex::strip_jsonc;
use crate::mvcc::MvccStore;
use crate::platform::{mkdir_private, read_zstd, write_zstd};
use crate::source::{dir_of, entrypoint, ext_of, read_bytes, same_family, scan_files, stem_of};
use crate::types::DeadSymbol;
use parse::{FileSymbols, parse_symbols};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

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
}

#[derive(Debug, Clone, Default)]
pub struct SymbolGraph {
    pub defs_by_file: BTreeMap<String, Vec<String>>,
    pub refs_by_file: BTreeMap<String, Vec<String>>,
    pub imports_by_file: BTreeMap<String, Vec<String>>,
    pub edges: Vec<(String, String)>,
    pub fan_in: HashMap<String, usize>,
    pub fan_out: HashMap<String, usize>,
    pub dead_symbols: Vec<DeadSymbol>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SymbolCache {
    version: String,
    files: BTreeMap<String, FileSymbols>,
}

struct LookupTables {
    all_rels: HashSet<String>,
    mod_to_rel: HashMap<String, String>,
    stem_to_rels: HashMap<String, Vec<String>>,
    go_pkg_rels: HashMap<String, Vec<String>>,
    ts_aliases: Vec<(String, String)>,
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

    let want_ext = stem_bin_hint.is_some()
        || std::env::var("REFIT_USE_STEM_BIN").as_deref() == Ok("1");
    if want_ext && query_ext(repo_root, stem_bin_hint, &mut edge_set) {
        disc.used_stem = true;
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    edges.sort();
    EdgeScan {
        used_stem: disc.used_stem,
        index_ms: started.elapsed().as_millis() as u64,
        edges,
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

fn ts_aliases(repo_root: &Path) -> Vec<(String, String)> {
    let Ok(raw) = std::fs::read_to_string(repo_root.join("tsconfig.json")) else {
        return Vec::new();
    };
    let cleaned = strip_jsonc(&raw);
    let Ok(v) = serde_json::from_str::<Value>(&cleaned) else {
        return Vec::new();
    };

    let base = v["compilerOptions"]["baseUrl"]
        .as_str()
        .unwrap_or("")
        .trim_start_matches("./")
        .trim_matches('/');
    let Some(paths) = v["compilerOptions"]["paths"].as_object() else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (alias, targets) in paths {
        let prefix = alias.trim_end_matches('*');
        let Some(arr) = (!prefix.is_empty()).then(|| targets.as_array()).flatten() else {
            continue;
        };
        for t in arr.iter().filter_map(Value::as_str) {
            let raw_t = t.trim_start_matches("./").trim_end_matches('*');
            let mapped = if base.is_empty() || base == "." {
                raw_t.to_string()
            } else if raw_t.is_empty() {
                format!("{base}/")
            } else {
                format!("{base}/{raw_t}")
            };
            out.push((prefix.to_string(), mapped));
        }
    }
    out
}

fn join_rel(base_dir: &str, rel_import: &str) -> String {
    let mut parts: Vec<&str> = base_dir
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    for seg in rel_import.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

fn build_tables(repo_root: &Path, sources: &[(String, String, Option<PathBuf>)]) -> LookupTables {
    let mut tbl = LookupTables {
        all_rels: HashSet::with_capacity(sources.len()),
        mod_to_rel: HashMap::with_capacity(sources.len()),
        stem_to_rels: HashMap::new(),
        go_pkg_rels: HashMap::new(),
        ts_aliases: ts_aliases(repo_root),
    };

    for (rel, ext, _) in sources {
        tbl.all_rels.insert(rel.clone());
        let p = Path::new(rel);
        if ext == "py" {
            tbl.mod_to_rel
                .insert(rel.trim_end_matches(".py").replace('/', "."), rel.clone());
        } else {
            let no_ext = rel.rsplit_once('.').map_or(rel.as_str(), |(a, _)| a);
            tbl.mod_to_rel.insert(no_ext.to_string(), rel.clone());
            if ext == "go"
                && let Some(dir) = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str())
            {
                tbl.go_pkg_rels.entry(dir.to_string()).or_default().push(rel.clone());
            }
        }
        let stem = stem_of(p);
        if !stem.is_empty() {
            tbl.stem_to_rels.entry(stem.to_string()).or_default().push(rel.clone());
        }
    }
    tbl
}

fn discover_internal(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (EdgeScan, usize, usize, SymbolGraph) {
    let started = Instant::now();
    let cache_path = repo_root.join(".refit/cache/symbol_index.zst");
    let prev: SymbolCache = read_zstd(&cache_path)
        .filter(|idx: &SymbolCache| idx.version == "0.5.0")
        .unwrap_or_default();

    let scan = scan_files(repo_root, preloaded);
    let tables = build_tables(repo_root, &scan.sources);

    let parsed_rows: Vec<(String, FileSymbols, bool)> = scan
        .sources
        .par_iter()
        .filter_map(|(rel, ext, _)| {
            let bytes = read_bytes(repo_root, rel, preloaded)?;
            let hash = MvccStore::hash_bytes(&bytes);
            if let Some(hit) = prev.files.get(rel).filter(|f| f.blake3 == hash) {
                return Some((rel.clone(), hit.clone(), false));
            }
            let entry = parse_symbols(ext, &String::from_utf8_lossy(&bytes), hash);
            Some((rel.clone(), entry, true))
        })
        .collect();

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
        let next = SymbolCache { version: "0.5.0".into(), files: new_files.clone() };
        let _ = write_zstd(&cache_path, &next);
    }

    let edges = collect_edges(&new_files, &tables);
    let snap = build_graph(repo_root, preloaded, new_files, &scan.tests, &edges);

    (
        EdgeScan {
            used_stem: false,
            index_ms: started.elapsed().as_millis() as u64,
            edges,
        },
        parsed_n,
        cached_n,
        snap,
    )
}

fn collect_edges(
    files: &BTreeMap<String, FileSymbols>,
    tbl: &LookupTables,
) -> Vec<(String, String)> {
    let mut def_to_files: HashMap<&str, Vec<&str>> = HashMap::new();
    for (rel, entry) in files {
        for sym in &entry.defs {
            def_to_files.entry(sym.as_str()).or_default().push(rel.as_str());
        }
    }

    let mut edge_set = HashSet::new();
    for (rel, entry) in files {
        let src_dir = dir_of(rel);
        let src_ext = ext_of(rel);

        for imp in &entry.imports {
            for target in resolve_import(imp, &src_dir, &src_ext, tbl) {
                if target != *rel {
                    edge_set.insert((rel.clone(), target));
                }
            }
        }

        if src_ext == "rs" {
            continue;
        }
        for tok in entry.refs_tokens.iter().filter(|t| t.len() >= 6) {
            let Some(defs) = def_to_files.get(tok.as_str()).filter(|d| d.len() == 1) else {
                continue;
            };
            let target = defs[0];
            if target != rel && same_family(&src_ext, &ext_of(target)) {
                edge_set.insert((rel.clone(), target.to_string()));
            }
        }
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    edges.sort();
    edges
}

fn build_graph(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    files: BTreeMap<String, FileSymbols>,
    tests: &[(String, String, Option<PathBuf>)],
    edges: &[(String, String)],
) -> SymbolGraph {
    let (mut fan_in, mut fan_out) = (HashMap::new(), HashMap::new());
    for (s, d) in edges {
        *fan_out.entry(s.clone()).or_insert(0) += 1;
        *fan_in.entry(d.clone()).or_insert(0) += 1;
    }

    let mut all_refs: HashMap<String, HashSet<String>> = HashMap::new();
    for (rel, entry) in &files {
        for r in &entry.refs_tokens {
            all_refs.entry(r.clone()).or_default().insert(rel.clone());
        }
    }

    let test_refs: Vec<(String, Vec<String>)> = tests
        .par_iter()
        .filter_map(|(rel, ext, _)| {
            let bytes = read_bytes(repo_root, rel, preloaded)?;
            let parsed = parse_symbols(ext, &String::from_utf8_lossy(&bytes), String::new());
            Some((rel.clone(), parsed.refs_tokens))
        })
        .collect();
    for (trel, toks) in test_refs {
        for r in toks {
            all_refs.entry(r).or_default().insert(trel.clone());
        }
    }

    let (mut defs_by, mut refs_by, mut imps_by) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    let mut dead = Vec::new();
    for (rel, entry) in files {
        if !entrypoint(&rel) {
            for d in &entry.defs {
                let callers = all_refs
                    .get(d)
                    .map(|set| set.iter().filter(|f| *f != &rel).count())
                    .unwrap_or(0);
                if callers == 0 {
                    dead.push(DeadSymbol { file: rel.clone(), symbol: d.clone() });
                }
            }
        }
        defs_by.insert(rel.clone(), entry.defs);
        refs_by.insert(rel.clone(), entry.refs_tokens);
        imps_by.insert(rel, entry.imports);
    }

    SymbolGraph {
        defs_by_file: defs_by,
        refs_by_file: refs_by,
        imports_by_file: imps_by,
        edges: edges.to_vec(),
        fan_in,
        fan_out,
        dead_symbols: dead,
    }
}

fn resolve_import(imp: &str, src_dir: &str, src_ext: &str, tbl: &LookupTables) -> Vec<String> {
    if imp.starts_with("./") || imp.starts_with("../") {
        let base = join_rel(src_dir, imp.trim_end_matches(".js").trim_end_matches(".ts"));
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

    for (alias, target_prefix) in &tbl.ts_aliases {
        if let Some(rest) = imp.strip_prefix(alias.as_str()) {
            let mapped = format!("{target_prefix}{rest}");
            if let Some(hit) = tbl
                .mod_to_rel
                .get(&mapped)
                .or_else(|| tbl.mod_to_rel.get(&format!("{mapped}/index")))
            {
                return vec![hit.clone()];
            }
        }
    }

    if src_ext == "rs"
        && let Some(sym) = imp.strip_prefix("super::")
    {
        let parent = dir_of(src_dir);
        let cand = if parent.is_empty() { sym.into() } else { format!("{parent}/{sym}") };
        if let Some(hit) = tbl.mod_to_rel.get(&cand) {
            return vec![hit.clone()];
        }
    }

    if src_ext == "go"
        && let Some(files) = tbl.go_pkg_rels.get(imp)
    {
        return files.clone();
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
}
