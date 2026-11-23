use crate::platform::create_private_dir;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct StemDiscovery {
    pub used_stem: bool,
    pub index_ms: u64,
    pub edges: Vec<(String, String)>,
    #[allow(dead_code)]
    pub symbol_counts: HashMap<String, usize>,
}

pub fn resolve_stem_bin(hint: Option<&Path>) -> Option<PathBuf> {
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
        .and_then(|p| p.parent().map(|d| d.join(stem_exe_name())));
    let mut candidates: Vec<PathBuf> = vec![
        PathBuf::from(".tools").join(stem_exe_name()),
        PathBuf::from("D:/cerebras/refit/.tools").join(stem_exe_name()),
    ];
    if let Some(s) = exe_sibling {
        candidates.insert(0, s);
    }
    for c in candidates {
        if c.exists() {
            return Some(c);
        }
    }
    None
}

fn stem_exe_name() -> &'static str {
    if cfg!(windows) { "stem.exe" } else { "stem" }
}

/// Runs proxysoul/stem v0.4.0 `stem mcp` inside an airtight 0700 isolated HOME with env_clear()
/// (zero access to host ~/.stem/.env, OAuth tokens, or CEREBRAS_API_KEY), queries Stem's
/// deterministic SQL fact store via read-only `sql` tool, and merges Python import edges.
pub fn discover_repo(repo_root: &Path, stem_bin_hint: Option<&Path>) -> StemDiscovery {
    let mut used_stem = false;
    let mut index_ms = 0u64;
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    let mut symbol_counts: HashMap<String, usize> = HashMap::new();

    if let Some(stem_bin) = resolve_stem_bin(stem_bin_hint) {
        let stem_home = repo_root.join(".refit/stem-home");
        if create_private_dir(&stem_home).is_ok() {
            let started = Instant::now();
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
                let reqs = concat!(
                    "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"refit\",\"version\":\"0.3.0\"}}}\n",
                    "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"sql\",\"arguments\":{\"query\":\"SELECT DISTINCT r.file AS src, s.file AS dst FROM refs r JOIN symbols s ON r.name = s.name WHERE r.file != s.file LIMIT 2000\"}}}\n",
                    "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"sql\",\"arguments\":{\"query\":\"SELECT file, COUNT(*) AS cnt FROM symbols GROUP BY file\"}}}\n"
                );
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(reqs.as_bytes());
                    let _ = stdin.flush();
                }
                if let Ok(output) = child.wait_with_output() {
                    index_ms = started.elapsed().as_millis() as u64;
                    if output.status.success() {
                        let text = String::from_utf8_lossy(&output.stdout);
                        for line in text.lines() {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                                let id = v["id"].as_u64().unwrap_or(0);
                                if id == 1 && v.get("result").is_some() {
                                    used_stem = true;
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
                                if id == 3 {
                                    if let Some(rows) = v["result"]["structuredContent"]["rows"].as_array() {
                                        for row in rows {
                                            if let (Some(f), Some(c)) = (row["file"].as_str(), row["cnt"].as_u64()) {
                                                symbol_counts.insert(f.replace('\\', "/"), c as usize);
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

    // Merge deterministic Python import-edge analysis for 100% sound wave ordering
    extract_python_import_edges(repo_root, &mut edge_set, &mut symbol_counts);

    let mut edges: Vec<(String, String)> = edge_set.into_iter().collect();
    edges.sort();

    StemDiscovery {
        used_stem,
        index_ms,
        edges,
        symbol_counts,
    }
}

fn extract_python_import_edges(
    repo_root: &Path,
    edges: &mut HashSet<(String, String)>,
    symbol_counts: &mut HashMap<String, usize>,
) {
    let mut module_to_rel: HashMap<String, String> = HashMap::new();
    let mut py_files: Vec<String> = Vec::new();

    for entry in WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".refit" | ".venv" | ".git" | "__pycache__" | ".pytest_cache" | ".ruff_cache" | "target" | "node_modules"
            )
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("py") {
            continue;
        }
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.starts_with("tests/") || rel_str.ends_with("__init__.py") {
            continue;
        }
        let dotted = rel_str.trim_end_matches(".py").replace('/', ".");
        module_to_rel.insert(dotted, rel_str.clone());
        py_files.push(rel_str);
    }

    for rel in &py_files {
        let abs = repo_root.join(rel);
        let Ok(content) = std::fs::read_to_string(&abs) else {
            continue;
        };
        let mut syms = 0usize;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("def ") || trimmed.starts_with("class ") {
                syms += 1;
            }
            if let Some(rest) = trimmed.strip_prefix("from ") {
                if let Some(mod_part) = rest.split_whitespace().next() {
                    if let Some(target) = module_to_rel.get(mod_part) {
                        if target != rel {
                            edges.insert((rel.clone(), target.clone()));
                        }
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("import ") {
                for item in rest.split(',') {
                    let mod_part = item.trim().split_whitespace().next().unwrap_or("");
                    if let Some(target) = module_to_rel.get(mod_part) {
                        if target != rel {
                            edges.insert((rel.clone(), target.clone()));
                        }
                    }
                }
            }
        }
        symbol_counts.entry(rel.clone()).or_insert(syms);
    }
}
