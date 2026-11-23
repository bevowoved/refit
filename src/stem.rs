use crate::platform::{create_private_dir, run_sandboxed_cmd};
use rusqlite::{Connection, OpenFlags};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct StemDiscovery {
    pub used_stem: bool,
    pub index_ms: u64,
    pub edges: Vec<(String, String)>,
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

/// Runs proxysoul/stem v0.4.0 inside an airtight 0700 isolated HOME with env_clear()
/// (zero access to host ~/.stem/.env, OAuth tokens, or CEREBRAS_API_KEY), reads
/// facts.db in strict SQLITE_OPEN_READ_ONLY mode, and merges deterministic Python import edges.
pub fn discover_repo(repo_root: &Path, stem_bin_hint: Option<&Path>) -> StemDiscovery {
    let mut used_stem = false;
    let mut index_ms = 0u64;
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    let mut symbol_counts: HashMap<String, usize> = HashMap::new();

    if let Some(stem_bin) = resolve_stem_bin(stem_bin_hint) {
        let stem_home = repo_root.join(".refit/stem-home");
        if create_private_dir(&stem_home).is_ok() {
            let mut env_allow = vec![
                ("HOME".to_string(), stem_home.display().to_string()),
                ("USERPROFILE".to_string(), stem_home.display().to_string()),
                (
                    "STEM_MCP_TOOLS".to_string(),
                    "defs,refs,callers,impls,symbols,sql,brief".to_string(),
                ),
            ];
            for k in ["PATH", "SYSTEMROOT", "TEMP", "TMP"] {
                if let Ok(v) = std::env::var(k) {
                    env_allow.push((k.to_string(), v));
                }
            }

            let args = vec!["index".to_string(), repo_root.display().to_string()];
            if let Ok(res) = run_sandboxed_cmd(
                &stem_bin.display().to_string(),
                &args,
                repo_root,
                &env_allow,
                true, // strict env_clear()!
                20_000,
            ) {
                index_ms = res.wall_ms;
                if res.exit_code == 0 {
                    used_stem = true;
                    if let Some(db_path) = find_facts_db(&stem_home) {
                        read_facts_db(&db_path, &mut edge_set, &mut symbol_counts);
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

fn find_facts_db(stem_home: &Path) -> Option<PathBuf> {
    for entry in WalkDir::new(stem_home).into_iter().flatten() {
        if entry.file_type().is_file() && entry.file_name() == "facts.db" {
            return Some(entry.path().to_path_buf());
        }
    }
    None
}

fn read_facts_db(
    db_path: &Path,
    edges: &mut HashSet<(String, String)>,
    symbol_counts: &mut HashMap<String, usize>,
) {
    let Ok(conn) = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) else {
        return;
    };
    let _ = conn.execute_batch("PRAGMA query_only = ON;");

    let mut tables = HashSet::new();
    if let Ok(mut stmt) = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'") {
        if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
            for t in rows.flatten() {
                tables.insert(t);
            }
        }
    }

    if tables.contains("files") && tables.contains("symbols") {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT f.path, COUNT(s.id) FROM files f LEFT JOIN symbols s ON s.file_id = f.id GROUP BY f.path",
        ) {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
                for (path, cnt) in rows.flatten() {
                    let norm = path.replace('\\', "/");
                    symbol_counts.insert(norm, cnt.max(0) as usize);
                }
            }
        }
    }

    if tables.contains("files") && tables.contains("symbols") && tables.contains("refs") {
        let q = "SELECT DISTINCT f_src.path, f_dst.path \
                 FROM refs r \
                 JOIN files f_src ON r.file_id = f_src.id \
                 JOIN symbols s ON r.symbol_id = s.id \
                 JOIN files f_dst ON s.file_id = f_dst.id \
                 WHERE f_src.path != f_dst.path";
        if let Ok(mut stmt) = conn.prepare(q) {
            if let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) {
                for (src, dst) in rows.flatten() {
                    edges.insert((src.replace('\\', "/"), dst.replace('\\', "/")));
                }
            }
        }
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
                            // rel depends on target
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
