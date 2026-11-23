use crate::mvcc::MvccStore;
use crate::platform::{mkdir_private, read_zstd, walk_files, write_zstd};
use crate::types::DeadSymbol;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileSymbols {
    blake3: String,
    defs: Vec<String>,
    refs_tokens: Vec<String>,
    imports: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SymbolCache {
    version: String,
    files: BTreeMap<String, FileSymbols>,
}

pub fn locate_stem(hint: Option<&Path>) -> Option<PathBuf> {
    if let Some(h) = hint {
        if h.exists() {
            return Some(h.to_path_buf());
        }
    }
    if let Ok(env_bin) = std::env::var("REFIT_STEM_BIN") {
        let p = PathBuf::from(env_bin);
        if p.exists() {
            return Some(p);
        }
    }
    let exe_sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(stem_binary())));
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from(".tools").join(stem_binary())];
    if let Some(s) = exe_sibling {
        candidates.insert(0, s);
    }
    candidates.into_iter().find(|c| c.exists())
}

fn stem_binary() -> &'static str {
    if cfg!(windows) { "stem.exe" } else { "stem" }
}

pub fn scan_edges(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> EdgeScan {
    let started = Instant::now();
    let (mut disc, _parsed, _cached, _snap) =
        discover_internal(repo_root, preloaded);
    let mut edge_set: HashSet<(String, String)> = disc.edges.into_iter().collect();

    let want_external_stem = stem_bin_hint.is_some()
        || std::env::var("REFIT_USE_STEM_BIN").as_deref() == Ok("1");

    if want_external_stem {
        if let Some(stem_bin) = locate_stem(stem_bin_hint) {
            let stem_home = repo_root.join(".refit/stem-home");
            if mkdir_private(&stem_home).is_ok() {
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

                if let Ok(mut child) = cmd.spawn() {
                    // Scoped SQL excludes test paths without matching words like `attestation`/`latest`/`backtest`.
                    let reqs = concat!(
                        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"refit\",\"version\":\"0.5.0\"}}}\n",
                        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"sql\",\"arguments\":{\"query\":\"WITH uniq AS (SELECT name FROM symbols GROUP BY name HAVING COUNT(DISTINCT file) = 1 AND LENGTH(name) >= 5) SELECT DISTINCT r.file AS src, s.file AS dst FROM refs r JOIN symbols s ON r.name = s.name JOIN uniq u ON u.name = s.name WHERE r.file != s.file AND r.file NOT LIKE 'tests/%' AND r.file NOT LIKE '%/tests/%' AND r.file NOT LIKE '%_test.%' AND r.file NOT LIKE '%.test.%' AND r.file NOT LIKE '%.spec.%' AND s.file NOT LIKE 'tests/%' AND s.file NOT LIKE '%/tests/%' AND s.file NOT LIKE '%_test.%' AND s.file NOT LIKE '%.test.%' AND s.file NOT LIKE '%.spec.%' LIMIT 2000\"}}}\n"
                    );
                    if let Some(mut stdin) = child.stdin.take() {
                        let _ = stdin.write_all(reqs.as_bytes());
                        let _ = stdin.flush();
                    }
                    if let Ok(output) = child.wait_with_output() {
                        if output.status.success() {
                            let text = String::from_utf8_lossy(&output.stdout);
                            for line in text.lines() {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                                    let id = v["id"].as_u64().unwrap_or(0);
                                    if id == 1 && v.get("result").is_some() {
                                        disc.used_stem = true;
                                    }
                                    if id == 2 {
                                        if let Some(rows) = v["result"]["structuredContent"]["rows"].as_array() {
                                            for row in rows {
                                                if let (Some(src), Some(dst)) = (row["src"].as_str(), row["dst"].as_str()) {
                                                    edge_set.insert((src.replace('\\', "/"), dst.replace('\\', "/")));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    edges.sort();

    EdgeScan {
        used_stem: disc.used_stem,
        index_ms: started.elapsed().as_millis() as u64,
        edges,
    }
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

pub fn query_symbols(
    repo_root: &Path,
    symbol: Option<&str>,
    file: Option<&str>,
) -> serde_json::Value {
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
            "symbol": sym,
            "definers": definers,
            "callers": callers,
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
            "deps": deps,
            "dependents": dependents,
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

fn clean_jsonc(raw: &str) -> String {
    let mut no_comments = String::with_capacity(raw.len());
    let b = raw.as_bytes();
    let mut i = 0usize;
    let mut in_str = false;
    while i < b.len() {
        if in_str {
            no_comments.push(b[i] as char);
            if b[i] == b'\\' && i + 1 < b.len() {
                i += 1;
                no_comments.push(b[i] as char);
            } else if b[i] == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b[i] == b'"' {
            in_str = true;
            no_comments.push('"');
            i += 1;
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            i += 2;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            no_comments.push(b[i] as char);
            i += 1;
        }
    }
    // Strip trailing commas before `}` or `]` outside strings.
    let mut out = String::with_capacity(no_comments.len());
    let nb = no_comments.as_bytes();
    let mut j = 0usize;
    in_str = false;
    while j < nb.len() {
        if in_str {
            out.push(nb[j] as char);
            if nb[j] == b'\\' && j + 1 < nb.len() {
                j += 1;
                out.push(nb[j] as char);
            } else if nb[j] == b'"' {
                in_str = false;
            }
            j += 1;
            continue;
        }
        if nb[j] == b'"' {
            in_str = true;
            out.push('"');
            j += 1;
        } else if nb[j] == b',' {
            let mut k = j + 1;
            while k < nb.len() && nb[k].is_ascii_whitespace() {
                k += 1;
            }
            if k < nb.len() && (nb[k] == b'}' || nb[k] == b']') {
                j += 1;
            } else {
                out.push(',');
                j += 1;
            }
        } else {
            out.push(nb[j] as char);
            j += 1;
        }
    }
    out
}

fn ts_aliases(repo_root: &Path) -> Vec<(String, String)> {
    let mut aliases = Vec::new();
    let Ok(raw) = std::fs::read_to_string(repo_root.join("tsconfig.json")) else {
        return aliases;
    };
    let cleaned = clean_jsonc(&raw);
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned) else {
        return aliases;
    };
    let base_url = v["compilerOptions"]["baseUrl"]
        .as_str()
        .unwrap_or("")
        .trim_start_matches("./")
        .trim_matches('/');
    if let Some(paths) = v["compilerOptions"]["paths"].as_object() {
        for (alias, targets) in paths {
            let a_prefix = alias.trim_end_matches('*').to_string();
            if a_prefix.is_empty() {
                continue;
            }
            if let Some(arr) = targets.as_array() {
                for item in arr {
                    if let Some(t) = item.as_str() {
                        let t_raw = t.trim_start_matches("./").trim_end_matches('*');
                        let t_prefix = if base_url.is_empty() || base_url == "." {
                            t_raw.to_string()
                        } else if t_raw.is_empty() {
                            format!("{base_url}/")
                        } else {
                            format!("{base_url}/{t_raw}")
                        };
                        aliases.push((a_prefix.clone(), t_prefix));
                    }
                }
            }
        }
    }
    aliases
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

fn discover_internal(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (EdgeScan, usize, usize, SymbolGraph) {
    let started = Instant::now();
    let cache_path = repo_root.join(".refit/cache/symbol_index.zst");
    let prev_index: SymbolCache = read_zstd(&cache_path)
        .filter(|idx: &SymbolCache| idx.version == "0.5.0")
        .unwrap_or_default();

    let tsconfig_aliases = ts_aliases(repo_root);
    let mut all_rel_set: HashSet<String> = HashSet::new();
    let mut module_to_rel: HashMap<String, String> = HashMap::new();
    let mut stem_to_rels: HashMap<String, Vec<String>> = HashMap::new();
    let mut go_pkg_to_rels: HashMap<String, Vec<String>> = HashMap::new();
    let mut src_files: Vec<(String, String, Option<PathBuf>)> = Vec::new();
    let mut test_files: Vec<(String, String, Option<PathBuf>)> = Vec::new();

    let mut register_file = |rel_str: String, ext: String, abs: Option<PathBuf>| {
        let is_test = rel_str.starts_with("tests/")
            || rel_str.contains("/test_")
            || rel_str.starts_with("test_")
            || rel_str.ends_with("_test.go")
            || rel_str.contains(".test.")
            || rel_str.contains(".spec.");
        if rel_str.ends_with("__init__.py") {
            return;
        }
        if is_test {
            test_files.push((rel_str, ext, abs));
            return;
        }
        all_rel_set.insert(rel_str.clone());
        let p = Path::new(&rel_str);
        if ext == "py" {
            module_to_rel.insert(rel_str.trim_end_matches(".py").replace('/', "."), rel_str.clone());
        } else {
            // Register extensionless path for TS/JS/Rust/Go resolution (e.g. `src/core/auth`).
            let no_ext = rel_str
                .rsplit_once('.')
                .map(|(a, _)| a.to_string())
                .unwrap_or_else(|| rel_str.clone());
            module_to_rel.insert(no_ext, rel_str.clone());
            if ext == "go" {
                if let Some(parent_name) = p
                    .parent()
                    .and_then(|d| d.file_name())
                    .and_then(|s| s.to_str())
                {
                    go_pkg_to_rels
                        .entry(parent_name.to_string())
                        .or_default()
                        .push(rel_str.clone());
                }
            }
        }
        if let Some(st) = p.file_stem().and_then(|s| s.to_str()) {
            stem_to_rels.entry(st.to_string()).or_default().push(rel_str.clone());
        }
        src_files.push((rel_str, ext, abs));
    };

    if let Some(map) = preloaded {
        for rel_str in map.keys() {
            let ext = Path::new(rel_str)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if matches!(ext.as_str(), "py" | "rs" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go") {
                register_file(rel_str.clone(), ext, None);
            }
        }
    } else {
        for path in walk_files(repo_root, None) {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(ext.as_str(), "py" | "rs" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go") {
                continue;
            }
            let Ok(rel) = path.strip_prefix(repo_root) else {
                continue;
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            register_file(rel_str, ext, Some(path));
        }
    }

    let results: Vec<(String, FileSymbols, bool)> = src_files
        .par_iter()
        .filter_map(|(rel, ext, abs_opt)| {
            let owned;
            let bytes: &[u8] = if let Some(b) = preloaded.and_then(|m| m.get(rel)) {
                b.as_slice()
            } else {
                owned = std::fs::read(abs_opt.as_ref()?).ok()?;
                owned.as_slice()
            };
            let hash = MvccStore::hash_bytes(bytes);
            if let Some(prev) = prev_index.files.get(rel) {
                if prev.blake3 == hash {
                    return Some((rel.clone(), prev.clone(), false));
                }
            }
            let content = String::from_utf8_lossy(bytes);
            let entry = parse_symbols(ext, &content, hash);
            Some((rel.clone(), entry, true))
        })
        .collect();

    let mut parsed_count = 0usize;
    let mut cached_count = 0usize;
    let mut new_files: BTreeMap<String, FileSymbols> = BTreeMap::new();
    for (rel, entry, was_parsed) in results {
        if was_parsed {
            parsed_count += 1;
        } else {
            cached_count += 1;
        }
        new_files.insert(rel, entry);
    }

    let mut def_to_files: HashMap<String, Vec<String>> = HashMap::new();
    for (rel, entry) in &new_files {
        for sym in &entry.defs {
            def_to_files.entry(sym.clone()).or_default().push(rel.clone());
        }
    }

    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    for (rel, entry) in &new_files {
        let src_dir = Path::new(rel)
            .parent()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        let src_ext = Path::new(rel)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        for imp in &entry.imports {
            for target in resolve_import(
                imp,
                &src_dir,
                &src_ext,
                &all_rel_set,
                &module_to_rel,
                &stem_to_rels,
                &go_pkg_to_rels,
                &tsconfig_aliases,
            ) {
                if target != *rel {
                    edge_set.insert((rel.clone(), target));
                }
            }
        }

        // Only emit ref-token edges for uniquely defined, same-language, non-generic symbols.
        if src_ext != "rs" {
            for tok in &entry.refs_tokens {
                if tok.len() < 6 {
                    continue;
                }
                if let Some(definers) = def_to_files.get(tok) {
                    if definers.len() == 1 {
                        let target = &definers[0];
                        let dst_ext = Path::new(target)
                            .extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        if target != rel && same_family(&src_ext, &dst_ext) {
                            edge_set.insert((rel.clone(), target.clone()));
                        }
                    }
                }
            }
        }
    }

    if parsed_count > 0 || prev_index.files.len() != new_files.len() {
        let next_idx = SymbolCache {
            version: "0.5.0".into(),
            files: new_files.clone(),
        };
        let _ = write_zstd(&cache_path, &next_idx);
    }

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    edges.sort();

    // Build SymbolGraph with fan-in, fan-out, and dead exported symbols.
    let mut fan_in: HashMap<String, usize> = HashMap::new();
    let mut fan_out: HashMap<String, usize> = HashMap::new();
    for (s, d) in &edges {
        *fan_out.entry(s.clone()).or_insert(0) += 1;
        *fan_in.entry(d.clone()).or_insert(0) += 1;
    }

    let mut all_external_refs: HashMap<String, HashSet<String>> = HashMap::new();
    for (rel, entry) in &new_files {
        for r in &entry.refs_tokens {
            all_external_refs
                .entry(r.clone())
                .or_default()
                .insert(rel.clone());
        }
    }
    // Include test-file references so test-covered exports are never falsely reported as dead symbols.
    let test_refs: Vec<(String, Vec<String>)> = test_files
        .par_iter()
        .filter_map(|(rel, ext, abs_opt)| {
            let owned;
            let bytes: &[u8] = if let Some(b) = preloaded.and_then(|m| m.get(rel)) {
                b.as_slice()
            } else {
                owned = std::fs::read(abs_opt.as_ref()?).ok()?;
                owned.as_slice()
            };
            let content = String::from_utf8_lossy(bytes);
            let parsed = parse_symbols(ext, &content, String::new());
            Some((rel.clone(), parsed.refs_tokens))
        })
        .collect();
    for (trel, toks) in test_refs {
        for r in toks {
            all_external_refs.entry(r).or_default().insert(trel.clone());
        }
    }

    let mut defs_by_file = BTreeMap::new();
    let mut refs_by_file = BTreeMap::new();
    let mut imports_by_file = BTreeMap::new();
    let mut dead_symbols = Vec::new();
    for (rel, entry) in new_files {
        for d in &entry.defs {
            let ext_callers = all_external_refs
                .get(d)
                .map(|set| set.iter().filter(|f| *f != &rel).count())
                .unwrap_or(0);
            if ext_callers == 0 && !entrypoint(&rel) {
                dead_symbols.push(DeadSymbol {
                    file: rel.clone(),
                    symbol: d.clone(),
                });
            }
        }
        defs_by_file.insert(rel.clone(), entry.defs);
        refs_by_file.insert(rel.clone(), entry.refs_tokens);
        imports_by_file.insert(rel, entry.imports);
    }

    let snap = SymbolGraph {
        defs_by_file,
        refs_by_file,
        imports_by_file,
        edges: edges.clone(),
        fan_in,
        fan_out,
        dead_symbols,
    };

    (
        EdgeScan {
            used_stem: false,
            index_ms: started.elapsed().as_millis() as u64,
            edges,
        },
        parsed_count,
        cached_count,
        snap,
    )
}

fn entrypoint(rel: &str) -> bool {
    rel.ends_with("main.rs")
        || rel.ends_with("lib.rs")
        || rel.ends_with("main.py")
        || rel.ends_with("main.go")
        || rel.ends_with("index.ts")
        || rel.ends_with("index.js")
        || rel.ends_with("page.tsx")
        || rel.ends_with("route.ts")
}

fn same_family(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let js = |e: &str| matches!(e, "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs");
    js(a) && js(b)
}

fn resolve_import(
    imp: &str,
    src_dir: &str,
    src_ext: &str,
    all_rels: &HashSet<String>,
    module_to_rel: &HashMap<String, String>,
    stem_to_rels: &HashMap<String, Vec<String>>,
    go_pkg_to_rels: &HashMap<String, Vec<String>>,
    tsconfig_aliases: &[(String, String)],
) -> Vec<String> {
    // 1) Relative TS/JS (`./foo`, `../bar`) or Python relative (`.foo`, `..bar`).
    if imp.starts_with("./") || imp.starts_with("../") {
        let base = join_rel(src_dir, imp.trim_end_matches(".js").trim_end_matches(".ts"));
        for candidate in [base.clone(), format!("{base}/index")] {
            if let Some(hit) = module_to_rel.get(&candidate) {
                return vec![hit.clone()];
            }
        }
        if all_rels.contains(&base) {
            return vec![base];
        }
    }
    if src_ext == "py" && imp.starts_with('.') {
        let dots = imp.chars().take_while(|&c| c == '.').count();
        let rest = &imp[dots..];
        let mut dir_parts: Vec<&str> = src_dir.split('/').filter(|s| !s.is_empty()).collect();
        for _ in 1..dots {
            dir_parts.pop();
        }
        if !rest.is_empty() {
            for seg in rest.split('.') {
                if !seg.is_empty() {
                    dir_parts.push(seg);
                }
            }
        }
        let dotted = dir_parts.join(".");
        if let Some(hit) = module_to_rel.get(&dotted) {
            return vec![hit.clone()];
        }
    }
    // 2) tsconfig.json path aliases (`@/lib/auth` -> `src/lib/auth`).
    for (alias, target_prefix) in tsconfig_aliases {
        if let Some(rest) = imp.strip_prefix(alias.as_str()) {
            let mapped = format!("{target_prefix}{rest}");
            if let Some(hit) = module_to_rel
                .get(&mapped)
                .or_else(|| module_to_rel.get(&format!("{mapped}/index")))
            {
                return vec![hit.clone()];
            }
        }
    }
    // 3) Rust `super::foo` resolves against parent directory first.
    if src_ext == "rs" {
        if let Some(super_sym) = imp.strip_prefix("super::") {
            let parent_dir = Path::new(src_dir)
                .parent()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            let candidate = if parent_dir.is_empty() {
                super_sym.to_string()
            } else {
                format!("{parent_dir}/{super_sym}")
            };
            if let Some(hit) = module_to_rel.get(&candidate) {
                return vec![hit.clone()];
            }
        }
    }
    // 4) Go package directory mapping (`github.com/org/app/pkg/auth` -> all `.go` files in `auth/`).
    if src_ext == "go" {
        if let Some(pkg_files) = go_pkg_to_rels.get(imp) {
            return pkg_files.clone();
        }
    }
    // 5) Exact module path lookup (`pkg.core.mod_00` or `src/utils`).
    if let Some(hit) = module_to_rel.get(imp) {
        return vec![hit.clone()];
    }
    // 6) Rust `crate::foo` / `self::foo` or stem fallback.
    let lookup_key = imp.strip_prefix("super::").unwrap_or(imp);
    if let Some(cands) = stem_to_rels.get(lookup_key) {
        if !imp.starts_with("super::") {
            if let Some(same_dir) = cands.iter().find(|c| {
                let d = Path::new(c)
                    .parent()
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_default();
                d == src_dir
            }) {
                return vec![same_dir.clone()];
            }
        }
        let lang_matches: Vec<&String> = cands
            .iter()
            .filter(|c| {
                let e = Path::new(c)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                same_family(src_ext, &e)
            })
            .collect();
        if lang_matches.len() == 1 {
            return vec![lang_matches[0].clone()];
        }
    }
    Vec::new()
}

/// Return the end offset if `bytes[i]` starts a Rust char literal instead of a lifetime.
fn char_end(bytes: &[u8], i: usize) -> Option<usize> {
    if i + 2 >= bytes.len() || bytes[i] != b'\'' {
        return None;
    }
    if bytes[i + 1] == b'\\' {
        let mut p = i + 2;
        if p < bytes.len() {
            p += 1;
        }
        while p < bytes.len() && p < i + 14 && bytes[p] != b'\'' && bytes[p] != b'\n' {
            p += 1;
        }
        if p < bytes.len() && bytes[p] == b'\'' {
            return Some(p + 1);
        }
        return None;
    }
    // Handle single ASCII OR multi-byte UTF-8 char literal (`'é'`, `'🚀'`).
    let tail = std::str::from_utf8(&bytes[i + 1..]).ok()?;
    let ch = tail.chars().next()?;
    let next = i + 1 + ch.len_utf8();
    if next < bytes.len() && bytes[next] == b'\'' && ch != '\n' {
        Some(next + 1)
    } else {
        None
    }
}

/// Strip comments and Rust `#[cfg(test)]` items in one pass while keeping string contents intact.
pub(crate) fn strip_noise(ext: &str, content: &str) -> String {
    let is_rs = ext == "rs";
    let is_py = ext == "py";
    let is_js_or_go = matches!(ext, "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go");
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut i = 0usize;

    // Track `#[cfg(test)]` item skipping in Rust with exact brace tracking outside strings/comments.
    let mut pending_cfg_test = false;
    let mut skip_cfg_depth: i32 = 0;
    let mut skip_cfg_entered_brace = false;

    while i < bytes.len() {
        let b = bytes[i];
        let skipping = pending_cfg_test && (skip_cfg_entered_brace || skip_cfg_depth > 0);

        // 1) Python triple-quoted docstrings & `#` comments.
        if is_py {
            if b == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if (b == b'"' || b == b'\'')
                && i + 2 < bytes.len()
                && bytes[i + 1] == b
                && bytes[i + 2] == b
            {
                let q = b;
                i += 3;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'\n' {
                        out.push('\n');
                    }
                    if i + 2 < bytes.len() && bytes[i] == q && bytes[i + 1] == q && bytes[i + 2] == q {
                        i += 3;
                        break;
                    }
                    i += 1;
                }
                continue;
            }
            if b == b'"' || b == b'\'' {
                let q = b;
                out.push(q as char);
                i += 1;
                while i < bytes.len() && bytes[i] != b'\n' {
                    let c = bytes[i];
                    out.push(c as char);
                    if c == b'\\' && i + 1 < bytes.len() {
                        i += 1;
                        out.push(bytes[i] as char);
                        i += 1;
                        continue;
                    }
                    i += 1;
                    if c == q {
                        break;
                    }
                }
                continue;
            }
            let ch = content[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }

        // 2) Line comment `//`
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // 3) Block comment `/*... */` (nested in Rust)
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'\n' {
                    if !skipping {
                        out.push('\n');
                    }
                    i += 1;
                } else if is_rs && bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }

        // 4) Rust `#[cfg(test)]` attribute trigger (at top or nested item boundary).
        if is_rs && !skipping && b == b'#' && content[i..].starts_with("#[cfg(test)]") {
            pending_cfg_test = true;
            skip_cfg_depth = 0;
            skip_cfg_entered_brace = false;
            i += "#[cfg(test)]".len();
            continue;
        }

        // 5) Rust raw strings `r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`
        if is_rs && (b == b'r' || ((b == b'b' || b == b'c') && i + 1 < bytes.len() && bytes[i + 1] == b'r')) {
            let prev_is_ident = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if !prev_is_ident {
                let start_r = if b == b'r' { i } else { i + 1 };
                let mut p = start_r + 1;
                let mut hashes = 0usize;
                while p < bytes.len() && bytes[p] == b'#' {
                    hashes += 1;
                    p += 1;
                }
                if p < bytes.len() && bytes[p] == b'"' {
                    let start = i;
                    p += 1;
                    while p < bytes.len() {
                        if bytes[p] == b'"' {
                            let mut m = 0usize;
                            while m < hashes && p + 1 + m < bytes.len() && bytes[p + 1 + m] == b'#' {
                                m += 1;
                            }
                            if m == hashes {
                                p += 1 + hashes;
                                break;
                            }
                        }
                        p += 1;
                    }
                    if !pending_cfg_test {
                        out.push_str(&content[start..p]);
                    }
                    i = p;
                    continue;
                }
            }
        }

        // 6) Rust char literals vs lifetimes
        if is_rs && b == b'\'' {
            if let Some(end) = char_end(bytes, i) {
                if !pending_cfg_test {
                    out.push_str(&content[i..end]);
                }
                i = end;
                continue;
            }
            if !pending_cfg_test {
                out.push('\'');
            }
            i += 1;
            continue;
        }

        // 7) Double-quoted, single-quoted (non-Rust), or backtick (JS/TS/Go) strings
        if b == b'"' || (!is_rs && b == b'\'') || (is_js_or_go && b == b'`') {
            let q = b;
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' && q != b'`' {
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                let done = bytes[i] == q;
                i += 1;
                if done {
                    break;
                }
            }
            if !pending_cfg_test {
                out.push_str(&content[start..i]);
            }
            continue;
        }

        // 8) Track Rust `#[cfg(test)]` item end (`;` before any `{`, or matching `}`).
        if is_rs && pending_cfg_test {
            if b == b'{' {
                skip_cfg_depth += 1;
                skip_cfg_entered_brace = true;
            } else if b == b'}' && skip_cfg_entered_brace {
                skip_cfg_depth -= 1;
                if skip_cfg_depth <= 0 {
                    pending_cfg_test = false;
                    skip_cfg_entered_brace = false;
                    skip_cfg_depth = 0;
                }
            } else if b == b';' && !skip_cfg_entered_brace {
                pending_cfg_test = false;
            } else if b == b'\n' {
                out.push('\n');
            }
            i += 1;
            continue;
        }

        let ch = content[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_noise(name: &str) -> bool {
    name.len() < 3
        || matches!(
            name,
            "new" | "get" | "set" | "run" | "main" | "init" | "default" | "Error" | "Result"
                | "Option" | "String" | "Vec" | "len" | "test" | "self" | "this" | "true"
                | "false" | "None" | "Some" | "Ok" | "Err" | "from" | "into" | "clone" | "fmt"
                | "pub" | "return" | "const" | "let" | "var" | "function" | "export" | "import"
                | "class" | "def" | "async" | "await" | "func" | "type" | "struct" | "enum"
                | "trait" | "impl" | "package" | "crate" | "super" | "mod" | "use" | "for"
                | "while" | "match" | "else" | "elif" | "pass" | "break" | "continue"
        )
}

fn first_ident(s: &str) -> Option<String> {
    let trimmed = s.trim_start();
    let ident: String = trimmed
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() || ident.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
        None
    } else {
        Some(ident)
    }
}

fn quoted_path(s: &str) -> Option<String> {
    let q = s.find(['"', '\''])?;
    let quote = s.as_bytes()[q] as char;
    let rest = &s[q + 1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn parse_symbols(ext: &str, content: &str, blake3: String) -> FileSymbols {
    let cleaned = strip_noise(ext, content);
    let mut defs: HashSet<String> = HashSet::new();
    let mut imports: HashSet<String> = HashSet::new();
    let mut refs_tokens: HashSet<String> = HashSet::new();
    let mut in_go_import_block = false;

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Extract polyglot imports.
        match ext {
            "py" => {
                if let Some(rest) = trimmed.strip_prefix("from ") {
                    if let Some((mod_part, after_imp)) = rest.split_once(" import ") {
                        let mod_clean = mod_part.trim();
                        if mod_clean.chars().all(|c| c == '.') && !mod_clean.is_empty() {
                            // `from. import foo, bar` -> `.foo`, `.bar`
                            for item in after_imp.trim_matches(['(', ')']).split(',') {
                                if let Some(sym) = first_ident(item) {
                                    imports.insert(format!("{mod_clean}{sym}"));
                                }
                            }
                        } else if !mod_clean.is_empty() {
                            imports.insert(mod_clean.to_string());
                        }
                    } else if let Some(mod_part) = rest.split_whitespace().next() {
                        imports.insert(mod_part.to_string());
                    }
                } else if let Some(rest) = trimmed.strip_prefix("import ") {
                    for item in rest.split(',') {
                        let mod_part = item.trim().split_whitespace().next().unwrap_or("");
                        if !mod_part.is_empty() {
                            imports.insert(mod_part.to_string());
                        }
                    }
                }
            }
            "rs" => {
                // Catch `use crate::foo`, `use crate::{a, b}`, `use super::foo`, `mod foo;`, AND inline `crate::foo::bar`!
                let mut scan = trimmed;
                while let Some(pos) = scan.find("crate::") {
                    let rest = &scan[pos + 7..];
                    if let Some(inner) = rest.strip_prefix('{') {
                        if let Some(close) = inner.find('}') {
                            for item in inner[..close].split(',') {
                                if let Some(id) = first_ident(item) {
                                    imports.insert(id);
                                }
                            }
                        }
                    } else if let Some(id) = first_ident(rest) {
                        imports.insert(id);
                    }
                    scan = rest;
                }
                if let Some(rest) = trimmed.strip_prefix("use super::") {
                    if let Some(inner) = rest.strip_prefix('{') {
                        if let Some(close) = inner.find('}') {
                            for item in inner[..close].split(',') {
                                if let Some(id) = first_ident(item) {
                                    imports.insert(format!("super::{id}"));
                                }
                            }
                        }
                    } else if let Some(id) = first_ident(rest) {
                        imports.insert(format!("super::{id}"));
                    }
                } else if let Some(rest) = trimmed.strip_prefix("use self::") {
                    if let Some(id) = first_ident(rest) {
                        imports.insert(id);
                    }
                } else if let Some(rest) = trimmed.strip_prefix("mod ").or_else(|| trimmed.strip_prefix("pub mod ")) {
                    if trimmed.ends_with(';') {
                        if let Some(id) = first_ident(rest) {
                            imports.insert(id);
                        }
                    }
                }
            }
            "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => {
                if let Some(pos) = trimmed.find(" from ") {
                    if let Some(path) = quoted_path(&trimmed[pos + 6..]) {
                        imports.insert(path);
                    }
                } else if trimmed.starts_with("import ") && (trimmed.contains('"') || trimmed.contains('\'')) {
                    if let Some(path) = quoted_path(trimmed) {
                        imports.insert(path);
                    }
                }
                if let Some(pos) = trimmed.find("require(") {
                    if let Some(path) = quoted_path(&trimmed[pos + 8..]) {
                        imports.insert(path);
                    }
                }
            }
            "go" => {
                if trimmed.starts_with("import (") {
                    in_go_import_block = true;
                } else if in_go_import_block {
                    if trimmed.starts_with(')') {
                        in_go_import_block = false;
                    } else if let Some(path) = quoted_path(trimmed) {
                        if let Some(last) = path.rsplit('/').next() {
                            imports.insert(last.to_string());
                        }
                    }
                } else if trimmed.starts_with("import ") {
                    if let Some(path) = quoted_path(trimmed) {
                        if let Some(last) = path.rsplit('/').next() {
                            imports.insert(last.to_string());
                        }
                    }
                }
            }
            _ => {}
        }

        // Extract top-level exported symbol definitions (avoiding indented Rust `impl` methods).
        let is_top_level = !line.starts_with(' ') && !line.starts_with('\t');
        let def_cand = match ext {
            "py" if is_top_level => trimmed
                .strip_prefix("async def ")
                .or_else(|| trimmed.strip_prefix("def "))
                .or_else(|| trimmed.strip_prefix("class "))
                .and_then(first_ident),
            "rs" if is_top_level => trimmed
                .strip_prefix("pub async fn ")
                .or_else(|| trimmed.strip_prefix("pub fn "))
                .or_else(|| trimmed.strip_prefix("pub(crate) async fn "))
                .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
                .or_else(|| trimmed.strip_prefix("pub struct "))
                .or_else(|| trimmed.strip_prefix("pub(crate) struct "))
                .or_else(|| trimmed.strip_prefix("pub enum "))
                .or_else(|| trimmed.strip_prefix("pub(crate) enum "))
                .or_else(|| trimmed.strip_prefix("pub trait "))
                .or_else(|| trimmed.strip_prefix("pub(crate) trait "))
                .or_else(|| trimmed.strip_prefix("pub type "))
                .or_else(|| trimmed.strip_prefix("pub(crate) type "))
                .or_else(|| trimmed.strip_prefix("pub const "))
                .or_else(|| trimmed.strip_prefix("pub(crate) const "))
                .and_then(first_ident),
            "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => trimmed
                .strip_prefix("export default async function ")
                .or_else(|| trimmed.strip_prefix("export default function "))
                .or_else(|| trimmed.strip_prefix("export default class "))
                .or_else(|| trimmed.strip_prefix("export async function "))
                .or_else(|| trimmed.strip_prefix("export function "))
                .or_else(|| trimmed.strip_prefix("export class "))
                .or_else(|| trimmed.strip_prefix("export interface "))
                .or_else(|| trimmed.strip_prefix("export type "))
                .or_else(|| trimmed.strip_prefix("export const "))
                .and_then(first_ident),
            "go" => {
                if let Some(rest) = trimmed.strip_prefix("func ") {
                    let after_recv = if rest.starts_with('(') {
                        rest.find(')').map(|p| &rest[p + 1..]).unwrap_or("")
                    } else {
                        rest
                    };
                    first_ident(after_recv)
                        .filter(|id| id.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false))
                } else if let Some(rest) = trimmed.strip_prefix("type ") {
                    first_ident(rest)
                        .filter(|id| id.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(sym) = def_cand {
            if !sym.starts_with('_') && !is_noise(&sym) {
                defs.insert(sym);
            }
        }

        // Collect all unique identifier tokens without the old 256-token cap; preserve Rust lifetimes.
        let tb = trimmed.as_bytes();
        let mut ti = 0usize;
        let mut cur = String::new();
        let flush_cur = |cur: &mut String, refs: &mut HashSet<String>| {
            if !cur.is_empty() {
                if !is_noise(cur)
                    && !cur.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true)
                {
                    refs.insert(std::mem::take(cur));
                } else {
                    cur.clear();
                }
            }
        };
        while ti < tb.len() {
            let b = tb[ti];
            if ext == "rs" && b == b'\'' {
                flush_cur(&mut cur, &mut refs_tokens);
                if let Some(end) = char_end(tb, ti) {
                    ti = end;
                } else {
                    ti += 1;
                }
                continue;
            }
            if b == b'"' || (ext != "rs" && b == b'\'') || b == b'`' {
                flush_cur(&mut cur, &mut refs_tokens);
                let q = b;
                ti += 1;
                while ti < tb.len() {
                    if tb[ti] == b'\\' && q != b'`' {
                        ti = (ti + 2).min(tb.len());
                        continue;
                    }
                    let done = tb[ti] == q;
                    ti += 1;
                    if done {
                        break;
                    }
                }
                continue;
            }
            if b.is_ascii_alphanumeric() || b == b'_' {
                cur.push(b as char);
            } else {
                flush_cur(&mut cur, &mut refs_tokens);
            }
            ti += 1;
        }
        flush_cur(&mut cur, &mut refs_tokens);
    }

    for d in &defs {
        refs_tokens.remove(d);
    }

    let mut defs_v: Vec<String> = defs.into_iter().collect();
    let mut refs_v: Vec<String> = refs_tokens.into_iter().collect();
    let mut imps_v: Vec<String> = imports.into_iter().collect();
    defs_v.sort();
    refs_v.sort();
    imps_v.sort();

    FileSymbols {
        blake3,
        defs: defs_v,
        refs_tokens: refs_v,
        imports: imps_v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;

    #[test]
    fn probe_56_symbol_cache() {
        let dir = std::env::temp_dir().join(format!("refit_inc_sym_{}_{}", std::process::id(), now_ms()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/core_math.rs"), "pub fn compute_alpha_score(x: i32) -> i32 { x + 1 }\n").unwrap();
        std::fs::write(dir.join("src/service.rs"), "use crate::core_math::compute_alpha_score;\npub fn invoke_svc(x: i32) -> i32 { compute_alpha_score(x) }\n").unwrap();
        std::fs::write(dir.join("src/widget.ts"), "export function renderWidgetBanner() { return 1; }\n").unwrap();
        std::fs::write(dir.join("src/page.tsx"), "import { renderWidgetBanner } from './widget';\nexport function HomePage() { return renderWidgetBanner(); }\n").unwrap();

        let (d1, parsed1, cached1) = index_stats(&dir);
        assert_eq!(parsed1, 4);
        assert_eq!(cached1, 0);
        assert!(d1.index_ms < 500, "zero-spawn incremental index must be sub-500ms, got {}ms", d1.index_ms);
        assert!(d1.edges.contains(&("src/service.rs".into(), "src/core_math.rs".into())));
        assert!(d1.edges.contains(&("src/page.tsx".into(), "src/widget.ts".into())));

        let (_d2, parsed2, cached2) = index_stats(&dir);
        assert_eq!(parsed2, 0);
        assert_eq!(cached2, 4);

        std::fs::write(dir.join("src/core_math.rs"), "pub fn compute_alpha_score(x: i32) -> i32 { x + 2 }\n").unwrap();
        let (_d3, parsed3, cached3) = index_stats(&dir);
        assert_eq!(parsed3, 1);
        assert_eq!(cached3, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_80_stem_polyglot() {
        let dir = std::env::temp_dir().join(format!("refit_stem80_{}_{}", std::process::id(), now_ms()));
        std::fs::create_dir_all(dir.join("src/core")).unwrap();
        std::fs::create_dir_all(dir.join("src/ui")).unwrap();
        std::fs::write(
            dir.join("tsconfig.json"),
            r#"{"compilerOptions":{"paths":{"@/*":["src/*"]}}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("src/core/auth.ts"), "export function verifyToken() { return true; }\nexport function unusedGhost() { return false; }\n").unwrap();
        std::fs::write(dir.join("src/ui/login.tsx"), "import { verifyToken } from '../core/auth';\nexport default function Login() { return verifyToken(); }\n").unwrap();
        std::fs::write(dir.join("src/ui/profile.tsx"), "import { verifyToken } from '@/core/auth';\nexport function Profile() { return verifyToken(); }\n").unwrap();

        // Rust acyclic pair where leaf mentions parent ONLY inside #[cfg(test)] -> must NOT create a cycle!
        std::fs::write(dir.join("src/leaf.rs"), "pub fn base_fn() -> i32 { 1 }\n#[cfg(test)]\nmod tests {\n    use crate::top::top_fn;\n    #[test] fn t() { assert_eq!(top_fn(), 1); }\n}\n").unwrap();
        std::fs::write(dir.join("src/top.rs"), "use crate::leaf::base_fn;\npub fn top_fn() -> i32 { base_fn() }\n").unwrap();

        let snap = load_symbols(&dir, None);
        assert!(snap.edges.contains(&("src/ui/login.tsx".into(), "src/core/auth.ts".into())), "edges={:?}", snap.edges);
        assert!(snap.edges.contains(&("src/ui/profile.tsx".into(), "src/core/auth.ts".into())), "edges={:?}", snap.edges);
        assert!(snap.edges.contains(&("src/top.rs".into(), "src/leaf.rs".into())));
        assert!(!snap.edges.contains(&("src/leaf.rs".into(), "src/top.rs".into())), "cfg(test) leaked cycle: {:?}", snap.edges);
        assert!(snap.dead_symbols.iter().any(|d| d.symbol == "unusedGhost"), "dead={:?}", snap.dead_symbols);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
