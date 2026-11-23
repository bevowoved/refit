use crate::graph::{build_dag, hash_abi, supported_ext};
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::{
    find_secret, sandbox_env, now_ms, read_zstd, run_cmd,
    walk_files, write_zstd,
};
use crate::types::{DagPlan, GateVerdict, ModuleUnit};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

static STAGE_SEQ: AtomicU64 = AtomicU64::new(1);
static STAGE_HARDLINKS_TOTAL: AtomicU64 = AtomicU64::new(0);
static STAGE_COPIED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CARGO_SLOT_POOL: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
static CARGO_EPHEMERAL_SEQ: AtomicU64 = AtomicU64::new(100);

pub fn stage_hardlinks() -> u64 {
    STAGE_HARDLINKS_TOTAL.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn stage_copies() -> u64 {
    STAGE_COPIED_TOTAL.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn detect_langs(repo_root: &Path) -> Vec<String> {
    read_tree(repo_root).1
}

struct CargoSlot {
    slot: usize,
    pooled: bool,
    dir: PathBuf,
}

impl CargoSlot {
    fn acquire(repo_root: &Path) -> Self {
        let pool = CARGO_SLOT_POOL.get_or_init(|| Mutex::new((0..8).rev().collect()));
        let (slot, pooled) = {
            let mut g = pool.lock();
            if let Some(s) = g.pop() {
                (s, true)
            } else {
                (CARGO_EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed) as usize, false)
            }
        };
        let root = stages_dir(repo_root).join("cargo-slots");
        let dir = root.join(format!("slot-{slot}"));
        let _ = fs::create_dir_all(&dir);
        Self { slot, pooled, dir }
    }
}

impl Drop for CargoSlot {
    fn drop(&mut self) {
        if self.pooled {
            if let Some(pool) = CARGO_SLOT_POOL.get() {
                pool.lock().push(self.slot);
            }
        } else {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

fn stages_dir(repo_root: &Path) -> PathBuf {
    repo_root
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.join(".refit-stages"))
        .unwrap_or_else(|| std::env::temp_dir().join("refit-stages"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainInfo {
    pub language: String,
    pub canonical_linter: String,
    pub canonical_typecheck_lsp: String,
    pub canonical_test_runner: String,
}

pub fn detect_tools(langs: &[String]) -> Vec<ToolchainInfo> {
    langs
        .iter()
        .map(|l| match l.as_str() {
            "python" => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "ruff (format + check)".into(),
                canonical_typecheck_lsp: "ty".into(),
                canonical_test_runner: "pytest".into(),
            },
            "rust" => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "rustfmt + cargo clippy".into(),
                canonical_typecheck_lsp: "cargo check".into(),
                canonical_test_runner: "cargo test".into(),
            },
            "typescript" => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "biome / eslint".into(),
                canonical_typecheck_lsp: "tsc --noEmit".into(),
                canonical_test_runner: "vitest / node --test".into(),
            },
            "javascript" => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "biome / eslint".into(),
                canonical_typecheck_lsp: "node --check".into(),
                canonical_test_runner: "node --test".into(),
            },
            "go" => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "gofmt + go vet".into(),
                canonical_typecheck_lsp: "gopls / go vet".into(),
                canonical_test_runner: "go test ./...".into(),
            },
            _ => ToolchainInfo {
                language: l.clone(),
                canonical_linter: "refit-untangle".into(),
                canonical_typecheck_lsp: "host-lsp".into(),
                canonical_test_runner: "repo-test".into(),
            },
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub repo_root: PathBuf,
    pub python_bin: String,
    pub pytest_bin: String,
    pub ruff_bin: String,
    pub ty_bin: String,
    pub min_hygiene_target: f64,
    pub strict_polyglot_tools: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStats {
    pub tests_total: usize,
    pub tests_failed: usize,
    pub ruff_errors: usize,
    pub ty_errors: usize,
    pub hygiene_median: f64,
    pub hygiene_p10: f64,
    pub per_file_diagnostics: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanCache {
    pub version: String,
    pub repo_fingerprint: String,
    #[serde(default)]
    pub fast_mode: bool,
    #[serde(default)]
    pub per_file_hashes: BTreeMap<String, String>,
    pub detected_languages: Vec<String>,
    pub polyglot_toolchains: Vec<ToolchainInfo>,
    pub plan: DagPlan,
    pub audit: AuditStats,
}

fn read_tree(
    repo_root: &Path,
) -> (
    String,
    Vec<String>,
    Vec<String>,
    BTreeMap<String, Vec<u8>>,
    BTreeMap<String, String>,
) {
    let mut langs = BTreeSet::new();
    let mut all_supported: Vec<(String, PathBuf)> = Vec::new();
    let mut source_rels: Vec<String> = Vec::new();

    for path in walk_files(repo_root, None) {
        let Some(ext_os) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        let ext = ext_os.to_ascii_lowercase();
        if !supported_ext(&ext) {
            continue;
        }
        match ext.as_str() {
            "py" => {
                langs.insert("python".to_string());
            }
            "ts" | "tsx" => {
                langs.insert("typescript".to_string());
            }
            "js" | "jsx" | "mjs" | "cjs" => {
                langs.insert("javascript".to_string());
            }
            "go" => {
                langs.insert("go".to_string());
            }
            "rs" => {
                langs.insert("rust".to_string());
            }
            _ => {}
        }
        let Ok(rel_path) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel = rel_path.to_string_lossy().replace('\\', "/");
        if !rel.ends_with("__init__.py")
            && !rel.starts_with("tests/")
            && !rel.contains("/test_")
            && !rel.starts_with("test_")
            && !rel.ends_with("_test.go")
            && !rel.contains(".test.")
            && !rel.contains(".spec.")
        {
            source_rels.push(rel.clone());
        }
        all_supported.push((rel, path));
    }
    if langs.is_empty() {
        langs.insert("polyglot".to_string());
    }
    all_supported.sort_by(|a, b| a.0.cmp(&b.0));
    source_rels.sort();

    let loaded: Vec<(String, Vec<u8>, String, String)> = all_supported
        .par_iter()
        .map(|(rel, abs)| {
            let bytes = fs::read(abs).unwrap_or_default();
            let h = MvccStore::hash_bytes(&bytes);
            let digest_line = format!("{rel}:{}:{h}", bytes.len());
            (rel.clone(), bytes, h, digest_line)
        })
        .collect();

    let mut preloaded = BTreeMap::new();
    let mut per_file_hashes = BTreeMap::new();
    let mut digest_lines = Vec::with_capacity(loaded.len());
    for (rel, bytes, hash, dline) in loaded {
        preloaded.insert(rel.clone(), bytes);
        per_file_hashes.insert(rel, hash);
        digest_lines.push(dline);
    }
    let fp = MvccStore::hash_bytes(digest_lines.join("\n").as_bytes());
    (fp, langs.into_iter().collect(), source_rels, preloaded, per_file_hashes)
}

pub fn scan_repo(
    repo_root: &Path,
    use_cache: bool,
    scorer: &HygieneScorer,
) -> Result<(ScanCache, bool)> {
    scan_cached(repo_root, use_cache, false, scorer)
}

pub fn scan_cached(
    repo_root: &Path,
    use_cache: bool,
    skip_tests: bool,
    scorer: &HygieneScorer,
) -> Result<(ScanCache, bool)> {
    let cache_path = repo_root.join(".refit/cache/scan_cache.zst");
    let (fingerprint, detected_languages, source_rels, preloaded, per_file_hashes) =
        read_tree(repo_root);
    let prev_cached = read_zstd::<ScanCache>(&cache_path)
        .filter(|c| c.version == "0.5.0");

    if use_cache {
        if let Some(ref cached) = prev_cached {
            if cached.repo_fingerprint == fingerprint && (!cached.fast_mode || skip_tests) {
                return Ok((cached.clone(), true));
            }
        }
    }

    let polyglot_toolchains = detect_tools(&detected_languages);
    let gate_cfg = read_gates(repo_root);

    let changed_rels: Vec<String> = if let Some(ref prev) = prev_cached {
        source_rels
            .iter()
            .filter(|rel| prev.per_file_hashes.get(*rel) != per_file_hashes.get(*rel))
            .cloned()
            .collect()
    } else {
        source_rels.clone()
    };

    let (plan_res, audit_res) = std::thread::scope(|s| {
        let h_plan = s.spawn(|| build_dag(repo_root, None, scorer, Some(&preloaded)));
        let h_audit = s.spawn(|| {
            let is_partial =
                use_cache && skip_tests && prev_cached.is_some() && !changed_rels.is_empty();
            let audit_targets = if is_partial {
                changed_rels.as_slice()
            } else {
                source_rels.as_slice()
            };
            let mut fresh = audit_files(
                &gate_cfg,
                audit_targets,
                scorer,
                Some(&preloaded),
                skip_tests,
            )?;
            // Merge unchanged files' cached diagnostics AND recompute full-repo aggregates.
            if is_partial {
                if let Some(ref prev) = prev_cached {
                    let changed_set: BTreeSet<&String> = changed_rels.iter().collect();
                    for rel in &source_rels {
                        if !changed_set.contains(rel) {
                            if let Some(old_d) = prev.audit.per_file_diagnostics.get(rel) {
                                fresh
                                    .per_file_diagnostics
                                    .entry(rel.clone())
                                    .or_insert_with(|| old_d.clone());
                            }
                        }
                    }
                    // Recompute full-repo hygiene quantiles (BLAKE3 cache hits in ~1us for unchanged files).
                    let (full_med, full_p10, _) =
                        scorer.score_files(repo_root, &source_rels, Some(&preloaded));
                    fresh.hygiene_median = full_med;
                    fresh.hygiene_p10 = full_p10;
                    fresh.ruff_errors = fresh
                        .per_file_diagnostics
                        .values()
                        .flat_map(|v| v.iter())
                        .filter(|d| d.starts_with("ruff "))
                        .count();
                    let changed_has_py = changed_rels.iter().any(|r| r.ends_with(".py"));
                    if !changed_has_py {
                        fresh.ty_errors = prev.audit.ty_errors;
                    } else {
                        // Recompute the full Python ty error count when a .py file changes.
                        let all_py: Vec<String> = source_rels
                            .iter()
                            .filter(|r| r.ends_with(".py"))
                            .cloned()
                            .collect();
                        if !all_py.is_empty() {
                            let mut ty_args = vec!["check".to_string(), "--error-on-warning".to_string()];
                            ty_args.extend(all_py);
                            let base_env = sandbox_env(&[]);
                            if let Ok(ty_o) = run_cmd(&gate_cfg.ty_bin, &ty_args, repo_root, &base_env, true, 20_000) {
                                fresh.ty_errors = if ty_o.exit_code == 0 {
                                    0
                                } else {
                                    parse_ty(&ty_o.stdout, &ty_o.stderr).len()
                                };
                            }
                        }
                    }
                    fresh.tests_total = prev.audit.tests_total;
                    fresh.tests_failed = prev.audit.tests_failed;
                }
            }
            Ok::<AuditStats, anyhow::Error>(fresh)
        });
        (
            h_plan.join().expect("dag plan thread"),
            h_audit.join().expect("full audit thread"),
        )
    });

    scorer.flush_cache();

    let bundle = ScanCache {
        version: "0.5.0".into(),
        repo_fingerprint: fingerprint,
        fast_mode: skip_tests,
        per_file_hashes,
        detected_languages,
        polyglot_toolchains,
        plan: plan_res?,
        audit: audit_res?,
    };
    let _ = write_zstd(&cache_path, &bundle);
    Ok((bundle, false))
}

pub fn read_gates(repo_root: &Path) -> GateConfig {
    let user_local = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(|h| PathBuf::from(h).join(".local/bin"))
        .unwrap_or_default();

    let mut venv_candidates: Vec<PathBuf> = vec![repo_root.join(".venv"), repo_root.join("venv")];
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if !venv.trim().is_empty() {
            venv_candidates.push(PathBuf::from(venv));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().take(4) {
            venv_candidates.push(anc.join(".venv"));
        }
    }
    venv_candidates.push(PathBuf::from(".venv"));

    let mut python_bin = "python".to_string();
    let mut pytest_bin = "pytest".to_string();
    for v in &venv_candidates {
        let py = if cfg!(windows) {
            v.join("Scripts/python.exe")
        } else {
            v.join("bin/python")
        };
        let pt = if cfg!(windows) {
            v.join("Scripts/pytest.exe")
        } else {
            v.join("bin/pytest")
        };
        if py.exists() {
            python_bin = py.display().to_string();
        }
        if pt.exists() {
            pytest_bin = pt.display().to_string();
            break;
        }
    }

    let ruff_bin = resolve_tool("ruff", &user_local);
    let ty_bin = resolve_tool("ty", &user_local);
    let strict_polyglot_tools = std::env::var("REFIT_STRICT_TOOLS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    GateConfig {
        repo_root: repo_root.to_path_buf(),
        python_bin,
        pytest_bin,
        ruff_bin,
        ty_bin,
        min_hygiene_target: 0.85,
        strict_polyglot_tools,
    }
}

fn resolve_tool(name: &str, user_local: &Path) -> String {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let local = user_local.join(&exe);
    if local.exists() {
        return local.display().to_string();
    }
    name.to_string()
}

struct StageGuard(PathBuf);
impl Drop for StageGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn sweep_stages(repo_root: &Path, max_age_secs: u64) -> usize {
    let stage_root_parent = stages_dir(repo_root);
    let Ok(rd) = fs::read_dir(&stage_root_parent) else {
        return 0;
    };
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(max_age_secs))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut swept = 0usize;
    for entry in rd.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        // Only sweep per-tx stage directories (`tx-*`); never delete warm `cargo-slots/` caches!
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("tx-") {
            continue;
        }
        let is_old = fs::metadata(&p)
            .and_then(|m| m.modified())
            .map(|m| m <= cutoff)
            .unwrap_or(false);
        if is_old && fs::remove_dir_all(&p).is_ok() {
            swept += 1;
        }
    }
    swept
}

#[cfg(test)]
pub fn encode_lsp(payload: &serde_json::Value) -> Vec<u8> {
    let body = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

#[cfg(test)]
pub fn decode_lsp(raw: &[u8]) -> Option<(serde_json::Value, usize)> {
    let sep = b"\r\n\r\n";
    let hdr_end = raw.windows(4).position(|w| w == sep)?;
    let hdr = std::str::from_utf8(&raw[..hdr_end]).ok()?;
    let len = hdr.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case("content-length") {
            v.trim().parse::<usize>().ok()
        } else {
            None
        }
    })?;
    let body_start = hdr_end + 4;
    let body_end = body_start + len;
    if raw.len() < body_end {
        return None;
    }
    let val: serde_json::Value = serde_json::from_slice(&raw[body_start..body_end]).ok()?;
    Some((val, body_end))
}

fn materialize_stage(
    repo_root: &Path,
    candidate_files: &BTreeMap<String, String>,
    deleted_files: &[String],
) -> Result<StageGuard> {
    let seq = STAGE_SEQ.fetch_add(1, Ordering::Relaxed);
    if seq == 1 || seq.is_multiple_of(64) {
        let _ = sweep_stages(repo_root, 1800);
    }
    let stage_root_parent = stages_dir(repo_root);
    let stage_dir = stage_root_parent.join(format!("tx-{}-{seq}", now_ms()));
    fs::create_dir_all(&stage_dir)?;
    let deleted_set: BTreeSet<String> = deleted_files
        .iter()
        .map(|s| s.replace('\\', "/"))
        .collect();

    let to_link: Vec<(PathBuf, PathBuf)> = walk_files(repo_root, None)
        .into_iter()
        .filter_map(|path| {
            let rel = path.strip_prefix(repo_root).ok()?;
            if rel.as_os_str().is_empty() {
                return None;
            }
            let rel_norm = rel.to_string_lossy().replace('\\', "/");
            if candidate_files.contains_key(&rel_norm) || deleted_set.contains(&rel_norm) {
                return None;
            }
            let dst = stage_dir.join(rel);
            Some((path, dst))
        })
        .collect();

    let mut parents: BTreeSet<PathBuf> = BTreeSet::new();
    for (_, dst) in &to_link {
        if let Some(p) = dst.parent() {
            parents.insert(p.to_path_buf());
        }
    }
    for p in parents {
        let _ = fs::create_dir_all(&p);
    }

    let (hardlinked, copied): (u64, u64) = to_link
        .par_iter()
        .map(|(src, dst)| {
            if fs::hard_link(src, dst).is_ok() {
                (1u64, 0u64)
            } else if fs::copy(src, dst).is_ok() {
                (0u64, 1u64)
            } else {
                (0u64, 0u64)
            }
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    if hardlinked > 0 {
        STAGE_HARDLINKS_TOTAL.fetch_add(hardlinked, Ordering::Relaxed);
    }
    if copied > 0 {
        STAGE_COPIED_TOTAL.fetch_add(copied, Ordering::Relaxed);
    }

    for (rel, content) in candidate_files {
        let dst = stage_dir.join(rel);
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(&dst, content.as_bytes())
            .with_context(|| format!("write staged candidate {}", dst.display()))?;
    }

    Ok(StageGuard(stage_dir))
}

#[derive(Debug, Clone)]
struct GuardRule {
    id: String,
    languages: Vec<String>,
    paths: Vec<String>,
    forbid: Option<String>,
    require: Option<String>,
    forbid_import: Option<String>,
    message: String,
}

fn parse_list(rest: &str) -> Vec<String> {
    let inner = rest.trim().trim_matches(['[', ']']);
    inner
        .split(',')
        .map(|s| s.trim().trim_matches(['"', '\'']).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_rules(text: &str) -> Vec<GuardRule> {
    let mut rules: Vec<GuardRule> = Vec::new();
    let mut cur: Option<GuardRule> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line == "rules:" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("- id:") {
            if let Some(prev) = cur.take() {
                rules.push(prev);
            }
            cur = Some(GuardRule {
                id: rest.trim().trim_matches(['"', '\'']).to_string(),
                languages: Vec::new(),
                paths: Vec::new(),
                forbid: None,
                require: None,
                forbid_import: None,
                message: "architectural invariant violated".to_string(),
            });
            continue;
        }
        if let Some(ref mut r) = cur {
            if let Some(rest) = line.strip_prefix("languages:") {
                r.languages = parse_list(rest)
                    .into_iter()
                    .map(|s| s.to_ascii_lowercase())
                    .collect();
            } else if let Some(rest) = line.strip_prefix("paths:") {
                r.paths = parse_list(rest);
            } else if let Some(rest) = line.strip_prefix("forbid_import:") {
                r.forbid_import = Some(rest.trim().trim_matches(['"', '\'']).to_string());
            } else if let Some(rest) = line.strip_prefix("forbid:") {
                r.forbid = Some(rest.trim().trim_matches(['"', '\'']).to_string());
            } else if let Some(rest) = line.strip_prefix("require:") {
                r.require = Some(rest.trim().trim_matches(['"', '\'']).to_string());
            } else if let Some(rest) = line.strip_prefix("message:") {
                r.message = rest.trim().trim_matches(['"', '\'']).to_string();
            }
        }
    }
    if let Some(last) = cur {
        rules.push(last);
    }
    rules
}

fn is_comment(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with('#')
        || trimmed.starts_with("/*")
        || trimmed == "*"
        || trimmed.starts_with("* ")
        || trimmed.starts_with("--")
}

fn is_import(trimmed: &str, in_go_import_block: bool) -> bool {
    in_go_import_block
        || trimmed.starts_with("import ")
        || trimmed.starts_with("from ")
        || trimmed.starts_with("use ")
        || trimmed.starts_with("pub use ")
        || trimmed.starts_with("pub(crate) use ")
        || trimmed.contains("require(")
        || (trimmed.starts_with("export ") && trimmed.contains(" from "))
}

fn has_import(line: &str, forbid_imp: &str) -> bool {
    let bytes = line.as_bytes();
    for (pos, _) in line.match_indices(forbid_imp) {
        let left_ok = pos == 0 || !(bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_');
        let end = pos + forbid_imp.len();
        let right_ok = end == bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern.ends_with('/') {
        return text.starts_with(pattern);
    }
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return text == prefix || text.starts_with(&format!("{prefix}/"));
    }
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            // Collapse `**` into multi-segment wildcard, single `*` matches any non-slash or fallback.
            let is_double = pi + 1 < p.len() && p[pi + 1] == '*';
            if is_double {
                pi += 2;
                star_p = Some((pi, true));
                star_t = ti;
            } else {
                pi += 1;
                star_p = Some((pi, false));
                star_t = ti;
            }
        } else if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if let Some((sp, is_double)) = star_p {
            if !is_double && t[star_t] == '/' {
                return false;
            }
            star_t += 1;
            ti = star_t;
            pi = sp;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn match_globs(rel: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let norm = rel.replace('\\', "/");
    patterns.iter().any(|p| {
        let p_norm = p.replace('\\', "/");
        if p_norm.contains('*') || p_norm.contains('?') {
            glob_match(&p_norm, &norm)
        } else if p_norm.ends_with('/') {
            norm.starts_with(&p_norm)
        } else {
            norm == p_norm || norm.starts_with(&format!("{p_norm}/"))
        }
    })
}

pub fn check_rules(
    repo_root: &Path,
    candidate_files: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (rel, content) in candidate_files {
        for (idx, line) in content.lines().enumerate() {
            if let Some(secret_kind) = find_secret(line) {
                violations.push(format!(
                    "Gate 3.5 invariant [builtin_no_secrets:{secret_kind}] `{rel}` L{}: hardcoded secret literal forbidden",
                    idx + 1
                ));
            }
        }
    }

    let inv_path = repo_root.join(".refit/invariants.yml");
    let Ok(yml) = fs::read_to_string(&inv_path) else {
        return violations;
    };
    let rules = parse_rules(&yml);
    for (rel, content) in candidate_files {
        let ext = Path::new(rel)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        for rule in &rules {
            if !rule.languages.is_empty() && !rule.languages.iter().any(|l| l == &ext) {
                continue;
            }
            if !match_globs(rel, &rule.paths) {
                continue;
            }
            if let Some(ref pat) = rule.forbid {
                if !pat.is_empty() {
                    for (idx, line) in content.lines().enumerate() {
                        let t = line.trim();
                        if is_comment(t) {
                            continue;
                        }
                        if t.contains(pat.as_str()) {
                            violations.push(format!(
                                "Gate 3.5 invariant [{}] `{rel}` L{}: {}",
                                rule.id,
                                idx + 1,
                                rule.message
                            ));
                        }
                    }
                }
            }
            if let Some(ref forbid_imp) = rule.forbid_import {
                if !forbid_imp.is_empty() {
                    let mut in_go_imp = false;
                    for (idx, line) in content.lines().enumerate() {
                        let t = line.trim();
                        if ext == "go" && t.starts_with("import (") {
                            in_go_imp = true;
                            continue;
                        }
                        if in_go_imp && t.starts_with(')') {
                            in_go_imp = false;
                            continue;
                        }
                        if !is_comment(t)
                            && is_import(t, in_go_imp)
                            && has_import(t, forbid_imp.as_str())
                        {
                            violations.push(format!(
                                "Gate 3.5 import-boundary [{}] `{rel}` L{}: forbidden import `{forbid_imp}` ({})",
                                rule.id,
                                idx + 1,
                                rule.message
                            ));
                        }
                    }
                }
            }
            if let Some(ref req) = rule.require {
                if !req.is_empty() && !content.contains(req.as_str()) {
                    violations.push(format!(
                        "Gate 3.5 invariant [{}] `{rel}` L1: missing required pattern `{req}` ({})",
                        rule.id, rule.message
                    ));
                }
            }
        }
    }
    violations
}

/// Check balanced braces and strings across Rust, TypeScript, JavaScript, JSX, and Go.
pub fn check_braces(rel_path: &str, source: &str) -> std::result::Result<(), String> {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(ext.as_str(), "rs" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go") {
        return Ok(());
    }
    let is_rs = ext == "rs";
    let is_go = ext == "go";
    let is_js_like = matches!(ext.as_str(), "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs");
    let is_jsx = matches!(ext.as_str(), "tsx" | "jsx");

    let bytes = source.as_bytes();
    let mut stack: Vec<(u8, usize)> = Vec::new();
    // Track nested JS/TS template literal `${... }` expression brace depths.
    let mut tpl_brace_depths: Vec<usize> = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut prev_sig: u8 = b';';
    let mut prev_word = String::new();

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Line comment //
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment /*... */ (nested in Rust)
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start_line = line;
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'\n' {
                    line += 1;
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
            if depth > 0 {
                return Err(format!("unterminated block comment at L{start_line}"));
            }
            continue;
        }
        // Rust raw strings: r"...", r#"..."#, br#"..."#, cr#"..."#
        if is_rs && (b == b'r' || ((b == b'b' || b == b'c') && i + 1 < bytes.len() && bytes[i + 1] == b'r')) {
            let start_r = if b == b'r' { i } else { i + 1 };
            let prev_is_ident = i > 0
                && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if !prev_is_ident {
                let mut p = start_r + 1;
                let mut hashes = 0usize;
                while p < bytes.len() && bytes[p] == b'#' {
                    hashes += 1;
                    p += 1;
                }
                if p < bytes.len() && bytes[p] == b'"' {
                    let start_line = line;
                    p += 1;
                    let mut closed = false;
                    while p < bytes.len() {
                        if bytes[p] == b'\n' {
                            line += 1;
                        }
                        if bytes[p] == b'"' {
                            let mut matched = 0usize;
                            while matched < hashes && p + 1 + matched < bytes.len() && bytes[p + 1 + matched] == b'#' {
                                matched += 1;
                            }
                            if matched == hashes {
                                i = p + 1 + hashes;
                                closed = true;
                                break;
                            }
                        }
                        p += 1;
                    }
                    if !closed {
                        return Err(format!("unterminated Rust raw string at L{start_line}"));
                    }
                    prev_sig = b'"';
                    prev_word.clear();
                    continue;
                }
            }
        }
        // JS/TS regex literal /.../flags after punctuation OR expression-starter keywords (`return`, `case`, `typeof`, etc.)
        let after_regex_kw = is_js_like
            && matches!(
                prev_word.as_str(),
                "return" | "case" | "throw" | "in" | "of" | "typeof" | "void" | "delete" | "yield" | "await" | "new"
            );
        if is_js_like
            && b == b'/'
            && i + 1 < bytes.len()
            && bytes[i + 1] != b'/'
            && bytes[i + 1] != b'*'
            && (after_regex_kw
                || matches!(
                    prev_sig,
                    b'(' | b'[' | b'{' | b',' | b';' | b':' | b'=' | b'!' | b'&' | b'|' | b'?' | b'+' | b'-' | b'~'
                ))
        {
            let mut p = i + 1;
            let mut in_class = false;
            let mut closed = false;
            while p < bytes.len() && bytes[p] != b'\n' {
                if bytes[p] == b'\\' {
                    p += 2;
                    continue;
                }
                if bytes[p] == b'[' {
                    in_class = true;
                } else if bytes[p] == b']' {
                    in_class = false;
                } else if bytes[p] == b'/' && !in_class {
                    p += 1;
                    while p < bytes.len() && bytes[p].is_ascii_alphabetic() {
                        p += 1;
                    }
                    closed = true;
                    break;
                }
                p += 1;
            }
            if closed {
                i = p;
                prev_sig = b'0';
                prev_word.clear();
                continue;
            }
        }
        // JS/TS template literal backtick with `${...}` expression support
        if is_js_like && b == b'`' {
            i += 1;
            scan_template(bytes, &mut i, &mut line, &mut stack, &mut tpl_brace_depths)?;
            prev_sig = b'`';
            prev_word.clear();
            continue;
        }
        // Resume JS/TS template literal when closing a `${...}` expression brace
        if is_js_like && b == b'}' && tpl_brace_depths.last() == Some(&stack.len()) {
            tpl_brace_depths.pop();
            i += 1;
            scan_template(bytes, &mut i, &mut line, &mut stack, &mut tpl_brace_depths)?;
            prev_sig = b'`';
            prev_word.clear();
            continue;
        }
        // JSX text content between `>` and `<` / `{` (allows single- or multi-line apostrophes like `Don't` without swallowing TS comparisons)
        if is_jsx && b == b'>' && prev_sig != b'=' {
            i += 1;
            prev_sig = b'>';
            prev_word.clear();
            let mut look = i;
            let mut look_lines = 0usize;
            let mut saw_alpha = false;
            while look < bytes.len() && bytes[look] != b'<' && bytes[look] != b'{' {
                if bytes[look] == b'\n' {
                    look_lines += 1;
                }
                if bytes[look].is_ascii_alphabetic() {
                    saw_alpha = true;
                }
                if matches!(bytes[look], b';' | b')' | b']' | b'}' | b'?' | b'=' | b'"' | b'`') {
                    saw_alpha = false;
                    break;
                }
                look += 1;
            }
            if saw_alpha && look < bytes.len() && (bytes[look] == b'<' || bytes[look] == b'{') {
                line += look_lines;
                i = look;
            }
            continue;
        }
        // Quoted strings ("...", Go `...`, and char/single-quoted '...')
        if b == b'"' || (is_go && b == b'`') || b == b'\'' {
            if b == b'\'' && is_rs {
                // Distinguish Rust char literal ('x', 'é', '🚀', '\n', '\x7f', '\u{...}') from lifetime ('a / 'static).
                let mut is_char_lit = false;
                if i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                    let mut p = i + 2;
                    if p < bytes.len() {
                        p += 1;
                    }
                    while p < bytes.len() && p < i + 14 && bytes[p] != b'\'' && bytes[p] != b'\n' {
                        p += 1;
                    }
                    is_char_lit = p < bytes.len() && bytes[p] == b'\'';
                } else if let Ok(tail) = std::str::from_utf8(&bytes[i + 1..]) {
                    if let Some(ch) = tail.chars().next() {
                        let next = i + 1 + ch.len_utf8();
                        is_char_lit = next < bytes.len() && bytes[next] == b'\'' && ch != '\n';
                    }
                }
                if !is_char_lit {
                    i += 1;
                    prev_sig = b'\'';
                    prev_word.clear();
                    continue;
                }
            }
            let quote = b;
            let start_line = line;
            i += 1;
            let mut closed = false;
            while i < bytes.len() {
                if bytes[i] == b'\\' && quote != b'`' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'\n' {
                    line += 1;
                    if quote != b'`' && !is_rs {
                        return Err(format!("unterminated string literal at L{start_line}"));
                    }
                }
                if bytes[i] == quote {
                    i += 1;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return Err(format!("unterminated string literal at L{start_line}"));
            }
            prev_sig = quote;
            prev_word.clear();
            continue;
        }
        // Track identifier words for JS/TS keyword-followed regex literals (`return /{/g;`).
        if is_js_like && (b.is_ascii_alphabetic() || b == b'_') {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            prev_word = String::from_utf8_lossy(&bytes[start..i]).into_owned();
            prev_sig = bytes[i - 1];
            continue;
        }
        prev_word.clear();

        match b {
            b'{' | b'(' | b'[' => {
                stack.push((b, line));
                prev_sig = b;
            }
            b'}' | b')' | b']' => {
                let expected = match b {
                    b'}' => b'{',
                    b')' => b'(',
                    _ => b'[',
                };
                match stack.pop() {
                    Some((open, _)) if open == expected => {}
                    Some((open, open_line)) => {
                        return Err(format!(
                            "mismatched `{}` at L{line} (expected close for `{}` from L{open_line})",
                            b as char, open as char
                        ));
                    }
                    None => {
                        return Err(format!("unexpected closing `{}` at L{line}", b as char));
                    }
                }
                prev_sig = b;
            }
            _ => {
                prev_sig = b;
            }
        }
        i += 1;
    }
    if !tpl_brace_depths.is_empty() {
        return Err("unterminated JS/TS template literal interpolation".into());
    }
    if let Some((open, open_line)) = stack.pop() {
        return Err(format!("unclosed `{}` opened at L{open_line}", open as char));
    }
    Ok(())
}

fn scan_template(
    bytes: &[u8],
    i: &mut usize,
    line: &mut usize,
    stack: &[(u8, usize)],
    tpl_brace_depths: &mut Vec<usize>,
) -> std::result::Result<(), String> {
    let start_line = *line;
    while *i < bytes.len() {
        if bytes[*i] == b'\\' {
            *i += 2;
            continue;
        }
        if bytes[*i] == b'\n' {
            *line += 1;
        }
        if bytes[*i] == b'$' && *i + 1 < bytes.len() && bytes[*i + 1] == b'{' {
            *i += 2;
            tpl_brace_depths.push(stack.len());
            return Ok(());
        }
        if bytes[*i] == b'`' {
            *i += 1;
            return Ok(());
        }
        *i += 1;
    }
    Err(format!("unterminated template literal at L{start_line}"))
}

#[cfg(test)]
pub fn verify_edits(
    cfg: &GateConfig,
    unit: &ModuleUnit,
    candidate_files: &BTreeMap<String, String>,
    hygiene: &HygieneScorer,
) -> Result<GateVerdict> {
    run_gates(cfg, unit, candidate_files, &[], hygiene)
}

pub fn run_gates(
    cfg: &GateConfig,
    unit: &ModuleUnit,
    candidate_files: &BTreeMap<String, String>,
    deleted_files: &[String],
    hygiene: &HygieneScorer,
) -> Result<GateVerdict> {
    let stage = materialize_stage(&cfg.repo_root, candidate_files, deleted_files)?;
    let stage_root = &stage.0;
    let base_sandbox_env = sandbox_env(&[]);

    let mut verdict = GateVerdict {
        passed: false,
        failed_gate: None,
        gate0_syntax_ok: false,
        gate0_abi_ok: true,
        gate0_ms: 0,
        gate1_tests_passed: 0,
        gate1_tests_failed: 0,
        gate1_ms: 0,
        gate1_cpu_ms: 0,
        gate1_peak_rss_bytes: 0,
        gate2_score_before: unit.baseline_hygiene,
        gate2_score_after: unit.baseline_hygiene,
        gate2_ms: 0,
        gate3_ruff_errors: 0,
        gate3_ty_errors: 0,
        gate3_ms: 0,
        formatted_files: 0,
        gate1_skipped_reason: None,
        diagnostics: Vec::new(),
        final_contents: BTreeMap::new(),
        deleted_files: deleted_files.to_vec(),
    };

    let staged_rel_paths: Vec<String> = candidate_files.keys().cloned().collect();
    let py_paths: Vec<String> = staged_rel_paths
        .iter()
        .filter(|p| p.ends_with(".py"))
        .cloned()
        .collect();
    let js_paths: Vec<String> = staged_rel_paths
        .iter()
        .filter(|p| p.ends_with(".js") || p.ends_with(".mjs") || p.ends_with(".cjs"))
        .cloned()
        .collect();
    let rs_paths: Vec<String> = staged_rel_paths
        .iter()
        .filter(|p| p.ends_with(".rs"))
        .cloned()
        .collect();
    let ts_paths: Vec<String> = staged_rel_paths
        .iter()
        .filter(|p| p.ends_with(".ts") || p.ends_with(".tsx"))
        .cloned()
        .collect();
    let go_paths: Vec<String> = staged_rel_paths
        .iter()
        .filter(|p| p.ends_with(".go"))
        .cloned()
        .collect();

    let has_rs = !rs_paths.is_empty();
    let has_ts = !ts_paths.is_empty();
    let has_go = !go_paths.is_empty();

    // Acquire exclusive cargo target slot OUTSIDE repo if Rust manifest is present.
    let cargo_slot = if has_rs && stage_root.join("Cargo.toml").exists() {
        Some(CargoSlot::acquire(&cfg.repo_root))
    } else {
        None
    };

    let g0_t = Instant::now();
    // 1) Pure-Rust sub-millisecond structural lexer for Rust, TS/TSX, JS/JSX, and Go.
    for (rel, content) in candidate_files {
        if let Err(err) = check_braces(rel, content) {
            verdict.gate0_syntax_ok = false;
            verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
            verdict.failed_gate = Some("gate0_syntax".into());
            verdict
                .diagnostics
                .push(format!("Gate 0 polyglot syntax error in `{rel}`: {err}"));
            return Ok(verdict);
        }
    }

    // 2) Run ALL applicable language Gate 0 checks in parallel.
    let gate0_err: Option<(String, String)> = std::thread::scope(|s| {
        let h_py = if !py_paths.is_empty() {
            Some(s.spawn(|| -> Option<(String, String)> {
                let mut py_args = vec![
                    "-c".to_string(),
                    "import ast, sys\nfor p in sys.argv[1:]:\n    ast.parse(open(p, encoding='utf-8').read(), filename=p)".to_string(),
                ];
                py_args.extend(py_paths.iter().cloned());
                match run_cmd(&cfg.python_bin, &py_args, stage_root, &base_sandbox_env, true, 5_000) {
                    Ok(syn) if syn.exit_code == 0 && !syn.timed_out => None,
                    Ok(syn) => Some((
                        "gate0_syntax".into(),
                        format!("Gate 0 syntax error: {}", first_lines(&syn.stderr, 4)),
                    )),
                    Err(e) => Some(("gate0_syntax".into(), format!("Gate 0 python spawn error: {e}"))),
                }
            }))
        } else {
            None
        };
        let h_js = if !js_paths.is_empty() {
            Some(s.spawn(|| -> Option<(String, String)> {
                js_paths.par_iter().find_map_any(|js| {
                    match run_cmd("node", &["--check".into(), js.clone()], stage_root, &base_sandbox_env, true, 5_000) {
                        Ok(out) if out.exit_code == 0 && !out.timed_out => None,
                        Ok(out) if out.timed_out => Some(("gate0_syntax".into(), format!("Gate 0 node --check timed out for `{js}`"))),
                        Ok(out) => Some((
                            "gate0_syntax".into(),
                            format!("Gate 0 node --check failed for `{js}`: {}", first_lines(&out.stderr, 4)),
                        )),
                        Err(e) if cfg.strict_polyglot_tools => Some(("gate0_tool_missing".into(), format!("node missing: {e}"))),
                        Err(_) => None,
                    }
                })
            }))
        } else {
            None
        };
        let h_rs = if let Some(ref slot) = cargo_slot {
            let slot_target = slot.dir.display().to_string();
            Some(s.spawn(move || -> Option<(String, String)> {
                let env = sandbox_env(&[("CARGO_TARGET_DIR".to_string(), slot_target)]);
                match run_cmd("cargo", &["check".into(), "--quiet".into()], stage_root, &env, true, 30_000) {
                    Ok(out) if out.exit_code == 0 && !out.timed_out => None,
                    Ok(out) if out.timed_out => Some(("gate0_rust_check".into(), "Gate 0 cargo check timed out".into())),
                    Ok(out) => Some((
                        "gate0_rust_check".into(),
                        format!("Gate 0 cargo check failed: {}", first_lines(&out.stderr, 6)),
                    )),
                    Err(e) if cfg.strict_polyglot_tools => Some(("gate0_tool_missing".into(), format!("cargo missing: {e}"))),
                    Err(_) => None,
                }
            }))
        } else {
            None
        };
        let h_ts = if has_ts && stage_root.join("tsconfig.json").exists() {
            Some(s.spawn(|| -> Option<(String, String)> {
                match run_cmd("tsc", &["--noEmit".into()], stage_root, &base_sandbox_env, true, 20_000) {
                    Ok(out) if out.exit_code == 0 && !out.timed_out => None,
                    Ok(out) if out.timed_out => Some(("gate0_ts_check".into(), "Gate 0 tsc --noEmit timed out".into())),
                    Ok(out) => Some((
                        "gate0_ts_check".into(),
                        format!("Gate 0 tsc --noEmit failed: {}", first_lines(&out.stdout, 6)),
                    )),
                    Err(e) if cfg.strict_polyglot_tools => Some(("gate0_tool_missing".into(), format!("tsc missing: {e}"))),
                    Err(_) => None,
                }
            }))
        } else {
            None
        };
        let h_go = if has_go && stage_root.join("go.mod").exists() {
            Some(s.spawn(|| -> Option<(String, String)> {
                match run_cmd("go", &["vet".into(), "./...".into()], stage_root, &base_sandbox_env, true, 20_000) {
                    Ok(out) if out.exit_code == 0 && !out.timed_out => None,
                    Ok(out) if out.timed_out => Some(("gate0_go_vet".into(), "Gate 0 go vet timed out".into())),
                    Ok(out) => Some((
                        "gate0_go_vet".into(),
                        format!("Gate 0 go vet failed: {}", first_lines(&out.stderr, 6)),
                    )),
                    Err(e) if cfg.strict_polyglot_tools => Some(("gate0_tool_missing".into(), format!("go missing: {e}"))),
                    Err(_) => None,
                }
            }))
        } else {
            None
        };
        for h in [h_py, h_js, h_rs, h_ts, h_go].into_iter().flatten() {
            if let Some(err) = h.join().ok().flatten() {
                return Some(err);
            }
        }
        None
    });
    if let Some((gate, msg)) = gate0_err {
        verdict.gate0_syntax_ok = false;
        verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
        verdict.failed_gate = Some(gate);
        verdict.diagnostics.push(msg);
        return Ok(verdict);
    }
    verdict.gate0_syntax_ok = true;

    if !unit.allow_abi_change {
        for (rel, content) in candidate_files {
            if let Some(expected_abi) = unit.abi_hash.get(rel) {
                let actual_abi = hash_abi(rel, content);
                if &actual_abi != expected_abi {
                    verdict.gate0_abi_ok = false;
                    verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
                    verdict.failed_gate = Some("gate0_abi".into());
                    verdict.diagnostics.push(format!(
                        "Gate 0 AbiPinViolation in `{rel}`: exported signature names changed"
                    ));
                    return Ok(verdict);
                }
            }
        }
    }
    verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;

    let inv_violations = check_rules(&cfg.repo_root, candidate_files);
    if !inv_violations.is_empty() {
        verdict.failed_gate = Some("gate3_5_invariants".into());
        verdict.diagnostics.extend(inv_violations.into_iter().take(8));
        return Ok(verdict);
    }

    // 3) Deterministic Pre-Pass Formatters.
    let g3_start = Instant::now();
    if !py_paths.is_empty() {
        let mut fmt_args = vec!["format".to_string(), "--quiet".to_string()];
        fmt_args.extend(py_paths.iter().cloned());
        if let Ok(out) = run_cmd(&cfg.ruff_bin, &fmt_args, stage_root, &base_sandbox_env, true, 8_000) {
            if out.exit_code == 0 {
                verdict.formatted_files += py_paths.len();
            }
        }
    }
    if has_rs {
        // Detect Rust edition from Cargo.toml (don't hardcode 2024 for edition 2018/2021 crates).
        let edition = fs::read_to_string(stage_root.join("Cargo.toml"))
            .ok()
            .and_then(|s| {
                s.lines().find_map(|l| {
                    let t = l.trim();
                    t.strip_prefix("edition")
                        .and_then(|r| r.split_once('='))
                        .map(|(_, v)| v.trim().trim_matches(['"', '\'']).to_string())
                })
            })
            .filter(|e| matches!(e.as_str(), "2015" | "2018" | "2021" | "2024"))
            .unwrap_or_else(|| "2021".to_string());
        let mut fmt_args = vec!["--edition".to_string(), edition];
        fmt_args.extend(rs_paths.iter().cloned());
        if let Ok(out) = run_cmd("rustfmt", &fmt_args, stage_root, &base_sandbox_env, true, 8_000) {
            if out.exit_code == 0 {
                verdict.formatted_files += rs_paths.len();
            }
        }
    }
    if has_go {
        let mut fmt_args = vec!["-w".to_string()];
        fmt_args.extend(go_paths.iter().cloned());
        if let Ok(out) = run_cmd("gofmt", &fmt_args, stage_root, &base_sandbox_env, true, 8_000) {
            if out.exit_code == 0 {
                verdict.formatted_files += go_paths.len();
            }
        }
    }
    if has_ts || !js_paths.is_empty() {
        let mut fmt_args = vec!["format".to_string(), "--write".to_string()];
        fmt_args.extend(ts_paths.iter().cloned());
        fmt_args.extend(js_paths.iter().cloned());
        if let Ok(out) = run_cmd("biome", &fmt_args, stage_root, &base_sandbox_env, true, 8_000) {
            if out.exit_code == 0 {
                verdict.formatted_files += ts_paths.len() + js_paths.len();
            }
        }
    }

    let mut working_texts: BTreeMap<String, String> = BTreeMap::new();
    let mut ruff_diags = Vec::new();
    let mut ty_diags = Vec::new();
    let mut poly_lint_diags = Vec::new();
    let mut pt_exit = 0i32;
    let mut pt_stdout = String::new();
    let mut ran_any_tests = false;

    if !py_paths.is_empty() {
        let mut ruff_args = vec![
            "check".to_string(),
            "--fix".to_string(),
            "--output-format=json".to_string(),
        ];
        ruff_args.extend(py_paths.iter().cloned());
        match run_cmd(&cfg.ruff_bin, &ruff_args, stage_root, &base_sandbox_env, true, 8_000) {
            Ok(ruff_out) => {
                ruff_diags = parse_ruff(&ruff_out.stdout);
                verdict.gate3_ruff_errors = ruff_diags.len();
            }
            Err(e) if cfg.strict_polyglot_tools => {
                poly_lint_diags.push(format!("ruff unavailable (strict mode): {e}"));
            }
            Err(_) => {}
        }
    }

    // Read post-format / post-ruff-fix candidate files from stage.
    for rel in &staged_rel_paths {
        let txt = fs::read_to_string(stage_root.join(rel))?;
        working_texts.insert(rel.clone(), txt);
    }

    // Re-verify ABI pin & Gate 3.5 invariants AFTER formatter / ruff --fix mutations.
    if !unit.allow_abi_change {
        for (rel, content) in &working_texts {
            if let Some(expected_abi) = unit.abi_hash.get(rel) {
                if &hash_abi(rel, content) != expected_abi {
                    verdict.gate0_abi_ok = false;
                    verdict.failed_gate = Some("gate0_abi".into());
                    verdict.diagnostics.push(format!(
                        "Gate 0 AbiPinViolation after formatter/autofix in `{rel}`"
                    ));
                    return Ok(verdict);
                }
            }
        }
    }
    let post_fix_inv = check_rules(&cfg.repo_root, &working_texts);
    if !post_fix_inv.is_empty() {
        verdict.failed_gate = Some("gate3_5_invariants".into());
        verdict.diagnostics.extend(post_fix_inv.into_iter().take(8));
        return Ok(verdict);
    }

    // 4) Run Gate 1 (Sandboxed Impact Tests) & Gate 3 (Static Analysis) across ALL detected languages.
    if !py_paths.is_empty() {
        let py_has_tests = !unit.test_files.is_empty() || stage_root.join("tests").exists();
        let (ty_res, pt_res, g1_elapsed) = std::thread::scope(|s| {
            let h_ty = s.spawn(|| {
                let mut args = vec!["check".to_string(), "--error-on-warning".to_string()];
                args.extend(py_paths.iter().cloned());
                run_cmd(&cfg.ty_bin, &args, stage_root, &base_sandbox_env, true, 10_000)
            });
            let h_pt = if py_has_tests {
                Some(s.spawn(|| {
                    let t = Instant::now();
                    let mut args = vec!["-q".to_string(), "--tb=short".to_string()];
                    if unit.test_files.is_empty() {
                        args.push("tests".to_string());
                    } else {
                        args.extend(unit.test_files.iter().cloned());
                    }
                    let env = sandbox_env(&[(
                        "PYTHONPATH".to_string(),
                        stage_root.display().to_string(),
                    )]);
                    let res = run_cmd(&cfg.pytest_bin, &args, stage_root, &env, true, 15_000);
                    (res, t.elapsed().as_millis() as u64)
                }))
            } else {
                None
            };
            let r_ty = h_ty.join().expect("ty thread");
            let (r_pt, g1_ms) = match h_pt {
                Some(h) => {
                    let (r, ms) = h.join().expect("pytest thread");
                    (Some(r), ms)
                }
                None => (None, 0),
            };
            (r_ty, r_pt, g1_ms)
        });

        match ty_res {
            Ok(ty_out) => {
                ty_diags = if ty_out.exit_code == 0 {
                    Vec::new()
                } else {
                    parse_ty(&ty_out.stdout, &ty_out.stderr)
                };
                verdict.gate3_ty_errors = ty_diags.len();
            }
            Err(e) if cfg.strict_polyglot_tools => {
                ty_diags.push(format!("ty unavailable (strict mode): {e}"));
                verdict.gate3_ty_errors = 1;
            }
            Err(_) => {}
        }

        if let Some(pt_r) = pt_res {
            match pt_r {
                Ok(pt_out) => {
                    ran_any_tests = true;
                    let (passed, failed) = parse_pytest(&pt_out.stdout, &pt_out.stderr, pt_out.exit_code);
                    verdict.gate1_tests_passed += passed;
                    verdict.gate1_tests_failed += failed;
                    verdict.gate1_ms += g1_elapsed;
                    verdict.gate1_cpu_ms += pt_out.cpu_ms;
                    verdict.gate1_peak_rss_bytes = verdict.gate1_peak_rss_bytes.max(pt_out.peak_rss_bytes);
                    pt_exit = pt_out.exit_code;
                    pt_stdout = pt_out.stdout;
                }
                Err(e) if cfg.strict_polyglot_tools => {
                    verdict.gate1_tests_failed += 1;
                    pt_exit = 1;
                    pt_stdout.push_str(&format!("pytest unavailable (strict mode): {e}"));
                }
                Err(_) => {}
            }
        }
    }

    // Polyglot Gate 1 + Gate 3 for Rust (`cargo test` + `cargo clippy`), Go (`go test`), and JS/TS (`node --test`/`vitest` + `biome lint`).
    if let Some(ref slot) = cargo_slot {
        let env = sandbox_env(&[("CARGO_TARGET_DIR".to_string(), slot.dir.display().to_string())]);
        let g1_t = Instant::now();
        match run_cmd("cargo", &["test".into(), "--quiet".into()], stage_root, &env, true, 45_000) {
            Ok(test_out) => {
                ran_any_tests = true;
                verdict.gate1_ms += g1_t.elapsed().as_millis() as u64;
                verdict.gate1_cpu_ms += test_out.cpu_ms;
                verdict.gate1_peak_rss_bytes = verdict.gate1_peak_rss_bytes.max(test_out.peak_rss_bytes);
                if test_out.exit_code != 0 || test_out.timed_out {
                    verdict.gate1_tests_failed += 1;
                    pt_exit = 1;
                    pt_stdout.push_str(&format!(
                        "cargo test failed: {}",
                        first_lines(&format!("{}\n{}", test_out.stdout, test_out.stderr), 8)
                    ));
                } else {
                    verdict.gate1_tests_passed += 1;
                }
            }
            Err(e) if cfg.strict_polyglot_tools => {
                verdict.gate1_tests_failed += 1;
                pt_exit = 1;
                pt_stdout.push_str(&format!("cargo test unavailable (strict mode): {e}"));
            }
            Err(_) => {}
        }
        if let Ok(clip_out) = run_cmd(
            "cargo",
            &["clippy".into(), "--no-deps".into(), "--quiet".into(), "--".into(), "-D".into(), "warnings".into()],
            stage_root,
            &env,
            true,
            30_000,
        ) {
            if clip_out.exit_code != 0 && !clip_out.timed_out {
                let msg = format!("{}\n{}", clip_out.stdout, clip_out.stderr);
                if !msg.contains("no such command") {
                    poly_lint_diags.push(format!("cargo clippy: {}", first_lines(&msg, 4)));
                }
            }
        }
    }
    if has_go && stage_root.join("go.mod").exists() {
        let g1_t = Instant::now();
        match run_cmd("go", &["test".into(), "./...".into()], stage_root, &base_sandbox_env, true, 30_000) {
            Ok(go_out) => {
                ran_any_tests = true;
                verdict.gate1_ms += g1_t.elapsed().as_millis() as u64;
                if go_out.exit_code != 0 || go_out.timed_out {
                    verdict.gate1_tests_failed += 1;
                    pt_exit = 1;
                    pt_stdout.push_str(&format!(
                        "go test failed: {}",
                        first_lines(&format!("{}\n{}", go_out.stdout, go_out.stderr), 6)
                    ));
                } else {
                    verdict.gate1_tests_passed += 1;
                }
            }
            Err(e) if cfg.strict_polyglot_tools => {
                verdict.gate1_tests_failed += 1;
                pt_exit = 1;
                pt_stdout.push_str(&format!("go test unavailable (strict mode): {e}"));
            }
            Err(_) => {}
        }
    }
    if (has_ts || !js_paths.is_empty()) && !unit.test_files.is_empty() {
        let js_tests: Vec<String> = unit
            .test_files
            .iter()
            .filter(|t| t.ends_with(".js") || t.ends_with(".mjs") || t.ends_with(".cjs"))
            .cloned()
            .collect();
        let ts_tests: Vec<String> = unit
            .test_files
            .iter()
            .filter(|t| t.ends_with(".ts") || t.ends_with(".tsx"))
            .cloned()
            .collect();
        if !js_tests.is_empty() {
            let mut args = vec!["--test".to_string()];
            args.extend(js_tests);
            let g1_t = Instant::now();
            match run_cmd("node", &args, stage_root, &base_sandbox_env, true, 15_000) {
                Ok(n_out) => {
                    ran_any_tests = true;
                    verdict.gate1_ms += g1_t.elapsed().as_millis() as u64;
                    if n_out.exit_code != 0 || n_out.timed_out {
                        verdict.gate1_tests_failed += 1;
                        pt_exit = 1;
                        pt_stdout.push_str(&format!(
                            "node --test failed: {}",
                            first_lines(&format!("{}\n{}", n_out.stdout, n_out.stderr), 6)
                        ));
                    } else {
                        verdict.gate1_tests_passed += 1;
                    }
                }
                Err(e) if cfg.strict_polyglot_tools => {
                    verdict.gate1_tests_failed += 1;
                    pt_exit = 1;
                    pt_stdout.push_str(&format!("node unavailable (strict mode): {e}"));
                }
                Err(_) => {}
            }
        }
        if !ts_tests.is_empty() {
            let mut v_args = vec!["run".to_string()];
            v_args.extend(ts_tests);
            let g1_t = Instant::now();
            if let Ok(v_out) = run_cmd("vitest", &v_args, stage_root, &base_sandbox_env, true, 25_000) {
                ran_any_tests = true;
                verdict.gate1_ms += g1_t.elapsed().as_millis() as u64;
                if v_out.exit_code != 0 || v_out.timed_out {
                    verdict.gate1_tests_failed += 1;
                    pt_exit = 1;
                    pt_stdout.push_str(&format!(
                        "vitest failed: {}",
                        first_lines(&format!("{}\n{}", v_out.stdout, v_out.stderr), 6)
                    ));
                } else {
                    verdict.gate1_tests_passed += 1;
                }
            }
        }
    }
    if has_ts || !js_paths.is_empty() {
        let mut check_args = vec!["lint".to_string()];
        check_args.extend(ts_paths.iter().cloned());
        check_args.extend(js_paths.iter().cloned());
        match run_cmd("biome", &check_args, stage_root, &base_sandbox_env, true, 10_000) {
            Ok(b_out) if b_out.exit_code != 0 && !b_out.timed_out => {
                poly_lint_diags.push(format!(
                    "biome lint: {}",
                    first_lines(&format!("{}\n{}", b_out.stdout, b_out.stderr), 4)
                ));
            }
            Err(e) if cfg.strict_polyglot_tools => {
                poly_lint_diags.push(format!("biome unavailable (strict mode): {e}"));
            }
            _ => {}
        }
    }
    verdict.gate3_ms = g3_start.elapsed().as_millis() as u64;
    if !ran_any_tests {
        verdict.gate1_skipped_reason = Some("no_matching_tests_or_manifest_found".into());
    }

    let g2_t = Instant::now();
    let mut score_sum = 0.0f64;
    let mut hygiene_hints = Vec::new();
    for (rel, txt) in &working_texts {
        let h = hygiene.score_content(rel, txt);
        score_sum += h.score;
        for sq in h.squiggles.iter().take(6) {
            hygiene_hints.push(format!(
                "codehygiene `{rel}` L{}: {} ({:.2}) {}",
                sq.line, sq.kind, sq.probability, sq.message
            ));
        }
    }
    // Pure-delete transactions (removing dead files) pass Gate 2 cleanly instead of collapsing to 0.0.
    verdict.gate2_score_after = if working_texts.is_empty() {
        cfg.min_hygiene_target.max(unit.baseline_hygiene)
    } else {
        score_sum / (working_texts.len() as f64)
    };
    verdict.gate2_ms = g2_t.elapsed().as_millis() as u64;

    if pt_exit != 0 || verdict.gate1_tests_failed > 0 {
        verdict.failed_gate = Some("gate1_tests".into());
        verdict.diagnostics.push(format!(
            "Gate 1 tests failed ({} failed): {}",
            verdict.gate1_tests_failed,
            first_lines(&pt_stdout, 8)
        ));
    }
    if verdict.gate3_ruff_errors > 0 || !poly_lint_diags.is_empty() {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate3_ruff".into());
        }
        verdict.diagnostics.extend(ruff_diags.into_iter().take(6));
        verdict.diagnostics.extend(poly_lint_diags.into_iter().take(4));
    }
    if verdict.gate3_ty_errors > 0 {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate3_ty".into());
        }
        verdict.diagnostics.extend(ty_diags.into_iter().take(6));
    }
    let hygiene_ok = verdict.gate2_score_after >= cfg.min_hygiene_target
        || (verdict.gate2_score_after >= verdict.gate2_score_before + 0.12 && verdict.gate2_score_after >= 0.72);
    if !hygiene_ok {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate2_hygiene".into());
        }
        verdict.diagnostics.push(format!(
            "Gate 2 codehygiene too low: {:.3} (before {:.3}, target >= {:.2})",
            verdict.gate2_score_after, verdict.gate2_score_before, cfg.min_hygiene_target
        ));
        verdict.diagnostics.extend(hygiene_hints.into_iter().take(6));
    }

    if verdict.failed_gate.is_none() {
        verdict.passed = true;
        for (rel, txt) in working_texts {
            verdict.final_contents.insert(rel, txt.into_bytes());
        }
    }

    Ok(verdict)
}

pub fn audit_repo(
    cfg: &GateConfig,
    rel_files: &[String],
    hygiene: &HygieneScorer,
) -> Result<AuditStats> {
    audit_files(cfg, rel_files, hygiene, None, false)
}

pub fn audit_files(
    cfg: &GateConfig,
    rel_files: &[String],
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    skip_tests: bool,
) -> Result<AuditStats> {
    let mut per_file_diagnostics: HashMap<String, Vec<String>> = HashMap::new();
    let py_files: Vec<String> = rel_files
        .iter()
        .filter(|f| f.ends_with(".py"))
        .cloned()
        .collect();
    let has_rs = rel_files.iter().any(|f| f.ends_with(".rs"));
    let has_ts = rel_files.iter().any(|f| f.ends_with(".ts") || f.ends_with(".tsx"));
    let has_go = rel_files.iter().any(|f| f.ends_with(".go"));
    let base_env = sandbox_env(&[]);

    let (hygiene_median, hygiene_p10, per_file_h, py_audit, poly_diags) = std::thread::scope(|s| {
        let h_hyg = s.spawn(|| hygiene.score_files(&cfg.repo_root, rel_files, preloaded));
        let h_py = if !py_files.is_empty() {
            Some(s.spawn(|| -> Result<(usize, usize, Vec<String>, Vec<(String, String)>, Vec<String>)> {
                let (pt_res, ruff_res, ty_res) = std::thread::scope(|s2| {
                    let t_pt = if !skip_tests && cfg.repo_root.join("tests").exists() {
                        Some(s2.spawn(|| {
                            let env = sandbox_env(&[(
                                "PYTHONPATH".to_string(),
                                cfg.repo_root.display().to_string(),
                            )]);
                            run_cmd(
                                &cfg.pytest_bin,
                                &["-q".into(), "--tb=line".into()],
                                &cfg.repo_root,
                                &env,
                                true,
                                30_000,
                            )
                        }))
                    } else {
                        None
                    };
                    let t_ruff = s2.spawn(|| {
                        let mut args = vec!["check".to_string(), "--output-format=json".to_string()];
                        args.extend(py_files.iter().cloned());
                        run_cmd(&cfg.ruff_bin, &args, &cfg.repo_root, &base_env, true, 15_000)
                    });
                    let t_ty = s2.spawn(|| {
                        let mut args = vec!["check".to_string(), "--error-on-warning".to_string()];
                        args.extend(py_files.iter().cloned());
                        run_cmd(&cfg.ty_bin, &args, &cfg.repo_root, &base_env, true, 20_000)
                    });
                    (
                        t_pt.map(|h| h.join().expect("pytest audit thread")),
                        t_ruff.join().expect("ruff audit thread"),
                        t_ty.join().expect("ty audit thread"),
                    )
                });

                let (passed, failed, failed_lines) = if let Some(pr) = pt_res {
                    let pt = pr?;
                    let (p, f) = parse_pytest(&pt.stdout, &pt.stderr, pt.exit_code);
                    let fl: Vec<String> = pt
                        .stdout
                        .lines()
                        .filter(|l| l.starts_with("FAILED "))
                        .map(String::from)
                        .collect();
                    (p, f, fl)
                } else {
                    (0, 0, Vec::new())
                };

                let ruff_out = ruff_res?;
                let ruff_items = ruff_map(&ruff_out.stdout, &cfg.repo_root);

                let ty_out = ty_res?;
                let ty_lines = if ty_out.exit_code == 0 {
                    Vec::new()
                } else {
                    parse_ty(&ty_out.stdout, &ty_out.stderr)
                };

                Ok((passed, failed, failed_lines, ruff_items, ty_lines))
            }))
        } else {
            None
        };

        // In --fast mode, skip heavy multi-second whole-repo compiler builds (cargo check/tsc/go vet);
        // run them in full audit mode with sandboxed env.
        let h_poly = s.spawn(|| -> Vec<(String, String)> {
            let mut out = Vec::new();
            if skip_tests {
                return out;
            }
            std::thread::scope(|sp| {
                let h_rs = if has_rs && cfg.repo_root.join("Cargo.toml").exists() {
                    Some(sp.spawn(|| {
                        let slot = CargoSlot::acquire(&cfg.repo_root);
                        let env = sandbox_env(&[(
                            "CARGO_TARGET_DIR".to_string(),
                            slot.dir.display().to_string(),
                        )]);
                        run_cmd("cargo", &["check".into(), "--quiet".into()], &cfg.repo_root, &env, true, 30_000)
                            .ok()
                            .filter(|o| o.exit_code != 0 && !o.timed_out)
                            .map(|o| ("Cargo.toml".to_string(), format!("cargo check: {}", first_lines(&o.stderr, 4))))
                    }))
                } else {
                    None
                };
                let h_ts = if has_ts && cfg.repo_root.join("tsconfig.json").exists() {
                    Some(sp.spawn(|| {
                        run_cmd("tsc", &["--noEmit".into()], &cfg.repo_root, &base_env, true, 20_000)
                            .ok()
                            .filter(|o| o.exit_code != 0 && !o.timed_out)
                            .map(|o| ("tsconfig.json".to_string(), format!("tsc: {}", first_lines(&o.stdout, 4))))
                    }))
                } else {
                    None
                };
                let h_go = if has_go && cfg.repo_root.join("go.mod").exists() {
                    Some(sp.spawn(|| {
                        run_cmd("go", &["vet".into(), "./...".into()], &cfg.repo_root, &base_env, true, 20_000)
                            .ok()
                            .filter(|o| o.exit_code != 0 && !o.timed_out)
                            .map(|o| ("go.mod".to_string(), format!("go vet: {}", first_lines(&o.stderr, 4))))
                    }))
                } else {
                    None
                };
                for h in [h_rs, h_ts, h_go].into_iter().flatten() {
                    if let Some(item) = h.join().ok().flatten() {
                        out.push(item);
                    }
                }
            });
            out
        });

        let (med, p10, map) = h_hyg.join().expect("hygiene audit thread");
        let py_res = h_py.map(|h| h.join().expect("python audit thread"));
        let p_diags = h_poly.join().expect("polyglot audit thread");
        (med, p10, map, py_res, p_diags)
    });
    for (file_k, msg) in poly_diags {
        per_file_diagnostics.entry(file_k).or_default().push(msg);
    }

    let mut tests_total = 0usize;
    let mut tests_failed = 0usize;
    let mut ruff_errors = 0usize;
    let mut ty_errors = 0usize;

    if let Some(res) = py_audit {
        let (passed, failed, failed_lines, ruff_items, ty_lines) = res?;
        tests_total = passed + failed;
        tests_failed = failed;
        for line in failed_lines {
            for rel in &py_files {
                let stem = Path::new(rel).file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if !stem.is_empty() && line.contains(&format!("test_{stem}")) {
                    per_file_diagnostics
                        .entry(rel.clone())
                        .or_default()
                        .push(format!("pytest {line}"));
                }
            }
        }
        ruff_errors = ruff_items.len();
        for (file_rel, msg) in ruff_items {
            per_file_diagnostics.entry(file_rel).or_default().push(msg);
        }
        ty_errors = ty_lines.len();
        for msg in &ty_lines {
            for rel in &py_files {
                if msg.contains(rel) || msg.replace('\\', "/").contains(rel) {
                    per_file_diagnostics
                        .entry(rel.clone())
                        .or_default()
                        .push(msg.clone());
                }
            }
        }
    }

    for (rel, hr) in &per_file_h {
        for sq in hr.squiggles.iter().take(6) {
            per_file_diagnostics.entry(rel.clone()).or_default().push(format!(
                "codehygiene L{}: {} ({:.2})",
                sq.line, sq.kind, sq.probability
            ));
        }
    }
    for diags in per_file_diagnostics.values_mut() {
        diags.sort();
        diags.dedup();
    }

    Ok(AuditStats {
        tests_total,
        tests_failed,
        ruff_errors,
        ty_errors,
        hygiene_median,
        hygiene_p10,
        per_file_diagnostics,
    })
}

fn parse_pytest(stdout: &str, stderr: &str, exit_code: i32) -> (usize, usize) {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines().rev() {
        let l = line.trim();
        if l.contains(" passed") || l.contains(" failed") || l.contains(" error") {
            for part in l.split(',') {
                let toks: Vec<&str> = part.split_whitespace().collect();
                for w in toks.windows(2) {
                    if let Ok(n) = w[0].parse::<usize>() {
                        if w[1].starts_with("passed") {
                            passed = passed.max(n);
                        } else if w[1].starts_with("failed") || w[1].starts_with("error") {
                            failed = failed.max(n);
                        }
                    }
                }
            }
            if passed > 0 || failed > 0 {
                break;
            }
        }
    }
    if exit_code != 0 && failed == 0 {
        failed = 1;
    }
    (passed, failed)
}

fn parse_ruff(stdout: &str) -> Vec<String> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    arr.into_iter()
        .map(|v| {
            let fname = v["filename"]
                .as_str()
                .and_then(|s| s.replace('\\', "/").rsplit('/').next().map(|n| n.to_string()))
                .unwrap_or_default();
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            format!("ruff {fname} {code} L{line}: {msg}")
        })
        .collect()
}

fn ruff_map(stdout: &str, repo_root: &Path) -> Vec<(String, String)> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    let root_norm = repo_root.display().to_string().replace('\\', "/");
    arr.into_iter()
        .map(|v| {
            let fname = v["filename"].as_str().unwrap_or("").replace('\\', "/");
            let rel = fname
                .strip_prefix(&format!("{root_norm}/"))
                .unwrap_or(&fname)
                .to_string();
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            (rel, format!("ruff {code} L{line}: {msg}"))
        })
        .collect()
}

fn parse_ty(stdout: &str, stderr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines() {
        let t = line.trim();
        if t.starts_with("error[") || t.starts_with("warning[") || t.contains("--> ") {
            out.push(format!("ty: {t}"));
        }
    }
    if out.is_empty() {
        let summary = first_lines(&combined, 4);
        if !summary.is_empty() {
            out.push(format!("ty: {summary}"));
        }
    }
    out
}

fn first_lines(s: &str, max: usize) -> String {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::dag_dir;

    #[test]
    fn probe_34_rust_ts_audit() {
        let base = std::env::temp_dir().join(format!(
            "refit_poly_audit_{}_{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(base.join("src")).unwrap();
        fs::write(base.join("src/lib.rs"), "pub fn calc(x: i32) -> i32 {\n    x * 2\n}\n").unwrap();
        fs::write(base.join("src/util.js"), "export function calc(x) {\n  return x * 2;\n}\n").unwrap();
        fs::write(
            base.join("src/Card.tsx"),
            "export function Card({ title }: { title: string }) {\n  return <section>{title}</section>;\n}\n",
        )
        .unwrap();

        let langs = detect_langs(&base);
        assert!(langs.contains(&"rust".to_string()));
        assert!(langs.contains(&"javascript".to_string()));
        assert!(langs.contains(&"typescript".to_string()));
        let chains = detect_tools(&langs);
        assert_eq!(chains.len(), 3);

        let cfg = read_gates(&base);
        let rels = vec!["src/lib.rs".into(), "src/util.js".into(), "src/Card.tsx".into()];
        let audit = audit_repo(&cfg, &rels, &HygieneScorer::new()).unwrap();
        assert_eq!(audit.tests_failed, 0);
        assert_eq!(audit.ruff_errors, 0);
        assert_eq!(audit.ty_errors, 0);
        assert!(audit.hygiene_median >= 0.85, "got {}", audit.hygiene_median);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn probe_38_stage_hardlink() {
        let sandbox_parent = std::env::temp_dir().join(format!(
            "refit_stage_test_{}_{}",
            std::process::id(),
            now_ms()
        ));
        let repo = sandbox_parent.join("target_codebase");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/keep.py"), "VALUE = 1\n").unwrap();
        fs::write(repo.join("src/edit.py"), "VALUE = 2\n").unwrap();

        let before_links = stage_hardlinks();
        let mut candidate = BTreeMap::new();
        candidate.insert("src/edit.py".to_string(), "VALUE = 99\n".to_string());

        let stage_path = {
            let guard = materialize_stage(&repo, &candidate, &[]).unwrap();
            assert!(!guard.0.starts_with(&repo));
            assert!(stage_hardlinks() > before_links);
            assert!(!repo.join(".refit-stages").exists());
            guard.0.clone()
        };

        assert!(!stage_path.exists());
        let _ = fs::remove_dir_all(&sandbox_parent);
    }

    #[test]
    fn probe_39_stage_inode() {
        let sandbox_parent = std::env::temp_dir().join(format!(
            "refit_inode_test_{}_{}",
            std::process::id(),
            now_ms()
        ));
        let repo = sandbox_parent.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/edit.py"), "ORIG = 1\n").unwrap();

        let mut candidate = BTreeMap::new();
        candidate.insert("src/edit.py".to_string(), "CAND = 2\n".to_string());
        let guard = materialize_stage(&repo, &candidate, &[]).unwrap();
        fs::write(guard.0.join("src/edit.py"), "MUTATED_IN_STAGE = 3\n").unwrap();
        assert_eq!(fs::read_to_string(repo.join("src/edit.py")).unwrap(), "ORIG = 1\n");
        drop(guard);
        let _ = fs::remove_dir_all(&sandbox_parent);
    }

    #[test]
    fn probe_45_scan_cache() {
        let dir = std::env::temp_dir().join(format!("refit_scancache_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/lib.rs"), "pub fn a() -> i32 { 1 }\n").unwrap();

        let scorer = HygieneScorer::new();
        let (b1, hit1) = scan_repo(&dir, true, &scorer).unwrap();
        assert!(!hit1);
        let (b2, hit2) = scan_repo(&dir, true, &scorer).unwrap();
        assert!(hit2);
        assert_eq!(b1.repo_fingerprint, b2.repo_fingerprint);

        fs::write(dir.join("src/lib.rs"), "pub fn a() -> i32 { 2 }\n").unwrap();
        let (b3, hit3) = scan_repo(&dir, true, &scorer).unwrap();
        assert!(!hit3);
        assert_ne!(b1.repo_fingerprint, b3.repo_fingerprint);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_46_scan_mtime() {
        let dir = std::env::temp_dir().join(format!("refit_mtime_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        let f = dir.join("src/app.ts");
        let body = "export const V = 42;\n";
        fs::write(&f, body).unwrap();

        let scorer = HygieneScorer::new();
        let (_, hit1) = scan_repo(&dir, true, &scorer).unwrap();
        assert!(!hit1);
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&f, body).unwrap();
        let (_, hit2) = scan_repo(&dir, true, &scorer).unwrap();
        assert!(hit2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_52_brace_check() {
        assert!(check_braces("src/lib.rs", "pub fn f<'a>(s: &'a str) -> char { let _ = \"}\"; '{' }\n").is_ok());
        assert!(check_braces("src/lib.rs", "pub fn broken() { let x = 1;\n").is_err());
        assert!(check_braces("src/app.tsx", "export const C = () => <div>{\"}\"}</div>;\n").is_ok());
        assert!(check_braces("src/app.ts", "export function bad(x: number { return x; }\n").is_err());
        assert!(check_braces("pkg/svc.go", "package p\nfunc Ok() string { return \"{\" }\n").is_ok());
        assert!(check_braces("pkg/svc.go", "package p\nfunc Bad() { /* unclosed\n").is_err());
    }

    #[test]
    fn probe_53_gate0_polyglot() {
        let dir = std::env::temp_dir().join(format!("refit_poly_g0_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        let orig = "export function doubleVal(x: number): number {\n  return x * 2;\n}\n";
        fs::write(dir.join("src/math.ts"), orig).unwrap();

        let scorer = HygieneScorer::new();
        let cfg = read_gates(&dir);
        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &scorer).unwrap();
        let unit = &plan.units[0];

        let mut bad = BTreeMap::new();
        bad.insert("src/math.ts".into(), "export function doubleVal(x: number): number {\n  return x * 2;\n".into());
        let v_bad = verify_edits(&cfg, unit, &bad, &scorer).unwrap();
        assert!(!v_bad.passed);
        assert_eq!(v_bad.failed_gate.as_deref(), Some("gate0_syntax"));

        let mut good = BTreeMap::new();
        good.insert("src/math.ts".into(), "export function doubleVal(x: number): number {\n  const out = x * 2;\n  return out;\n}\n".into());
        let v_good = verify_edits(&cfg, unit, &good, &scorer).unwrap();
        assert!(v_good.passed);
        assert!(v_good.gate0_syntax_ok);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_58_gate35_rules() {
        let dir = std::env::temp_dir().join(format!("refit_inv58_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join(".refit")).unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join(".refit/invariants.yml"),
            "rules:\n- id: no_std_mutex_unwrap\n  languages: [rs]\n  forbid: \".lock().unwrap()\"\n  message: \"use parking_lot::Mutex instead of std Mutex unwrap\"\n",
        )
        .unwrap();
        fs::write(dir.join("src/lock.rs"), "pub fn read_val() -> i32 {\n    1\n}\n").unwrap();

        let scorer = HygieneScorer::new();
        let cfg = read_gates(&dir);
        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &scorer).unwrap();
        let unit = &plan.units[0];

        let mut bad_inv = BTreeMap::new();
        bad_inv.insert("src/lock.rs".into(), "pub fn read_val() -> i32 {\n    let _g = m.lock().unwrap();\n    1\n}\n".into());
        let v1 = verify_edits(&cfg, unit, &bad_inv, &scorer).unwrap();
        assert!(!v1.passed);
        assert_eq!(v1.failed_gate.as_deref(), Some("gate3_5_invariants"));
        assert!(v1.diagnostics.iter().any(|d| d.contains("no_std_mutex_unwrap") && d.contains("L2")));

        let mut bad_sec = BTreeMap::new();
        let fake_tok = ["csk", "-", "1234567890abcdef"].concat();
        bad_sec.insert("src/lock.rs".into(), format!("pub fn read_val() -> i32 {{\n    let _k = \"{fake_tok}\";\n    1\n}}\n"));
        let v2 = verify_edits(&cfg, unit, &bad_sec, &scorer).unwrap();
        assert!(!v2.passed);
        assert_eq!(v2.failed_gate.as_deref(), Some("gate3_5_invariants"));
        assert!(v2.diagnostics.iter().any(|d| d.contains("builtin_no_secrets")));

        let mut clean = BTreeMap::new();
        clean.insert("src/lock.rs".into(), "pub fn read_val() -> i32 {\n    let out = 2;\n    out\n}\n".into());
        let v3 = verify_edits(&cfg, unit, &clean, &scorer).unwrap();
        assert!(v3.passed);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_71_stage_gc() {
        let parent = std::env::temp_dir().join(format!("refit_stg71_{}_{}", std::process::id(), now_ms()));
        let repo = parent.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/a.ts"), "export function ok(x: number) { return x; }\n").unwrap();
        let stale_stage = parent.join(".refit-stages/tx-orphan-old");
        fs::create_dir_all(&stale_stage).unwrap();
        fs::write(stale_stage.join("junk.txt"), "stale").unwrap();
        let _ = stage_copies();

        let swept = sweep_stages(&repo, 0);
        assert!(swept >= 1);
        assert!(!stale_stage.exists());
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn probe_72_lsp_and_cache() {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "capabilities": {} }
        });
        let framed = encode_lsp(&msg);
        let (decoded, consumed) = decode_lsp(&framed).unwrap();
        assert_eq!(consumed, framed.len());
        assert_eq!(decoded["method"], "initialize");

        let dir = std::env::temp_dir().join(format!("refit_inc72_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/a.rs"), "pub fn a() -> i32 { 1 }\n").unwrap();
        fs::write(dir.join("src/b.rs"), "pub fn b() -> i32 { 2 }\n").unwrap();
        let scorer = HygieneScorer::open_persistent(&dir);
        let (b1, hit1) = scan_cached(&dir, true, true, &scorer).unwrap();
        assert!(!hit1);
        assert!(b1.fast_mode);
        assert_eq!(b1.per_file_hashes.len(), 2);

        let (_b2, hit2) = scan_cached(&dir, true, true, &scorer).unwrap();
        assert!(hit2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_91_lexer_syntax() {
        assert!(check_braces("src/App.tsx", "export const C = () => <div>Don't panic {1 + 2}</div>;\n").is_ok());
        assert!(check_braces("src/re.js", "export const r = /[{(]/g;\n").is_ok());
        assert!(check_braces("src/tpl.ts", "export const s = (x: number) => `val=${foo(\"}\") + x}`;\n").is_ok());
        assert!(check_braces("src/raw.rs", "pub fn f() -> &'static str { let _b = b'{'; r#\"unbalanced }\"# }\n").is_ok());
        assert!(check_braces("pkg/raw.go", "package p\nfunc Raw() string { return `unbalanced {\\` }\n").is_ok());
    }

    #[test]
    fn probe_92_gate35_imports() {
        let dir = std::env::temp_dir().join(format!("refit_inv92_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join(".refit")).unwrap();
        fs::write(
            dir.join(".refit/invariants.yml"),
            "rules:\n- id: ui_no_db\n  paths: [src/ui/]\n  forbid_import: \"src/db\"\n  message: \"UI layer cannot import DB directly\"\n- id: no_unwrap\n  languages: [rs]\n  forbid: \".unwrap()\"\n  message: \"avoid unwrap\"\n",
        )
        .unwrap();

        let mut cand = BTreeMap::new();
        // Do not flag `.unwrap()` inside a comment.
        cand.insert("src/core/ok.rs".into(), "// never call .unwrap() in prod\npub fn ok() -> i32 { 1 }\n".into());
        // UI importing DB MUST fail import-boundary rule!
        cand.insert("src/ui/page.ts".into(), "import { query } from '../src/db/pool';\nexport const p = query;\n".into());
        // AWS key MUST fail builtin_no_secrets!
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        cand.insert("src/cfg.py".into(), format!("KEY = \"{aws}\"\n"));

        let v = check_rules(&dir, &cand);
        assert!(!v.iter().any(|m| m.contains("src/core/ok.rs")), "comment triggered false positive: {v:?}");
        assert!(v.iter().any(|m| m.contains("ui_no_db") && m.contains("src/ui/page.ts")), "missing import boundary: {v:?}");
        assert!(v.iter().any(|m| m.contains("aws_access_key_id")), "missing AWS secret check: {v:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_93_cargo_slots() {
        let dir = std::env::temp_dir().join(format!("refit_slot93_{}_{}", std::process::id(), now_ms()));
        let repo = dir.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let g1 = CargoSlot::acquire(&repo);
        let g2 = CargoSlot::acquire(&repo);
        assert_ne!(g1.slot, g2.slot);
        assert!(!g1.dir.starts_with(&repo));
        drop(g1);
        drop(g2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_98_lexer_slot_del() {
        let rs = "/* outer /* nested { */ still comment */\npub fn chars() -> (char, char) { ('é', '🚀') }\n";
        assert!(check_braces("src/u.rs", rs).is_ok());

        let ts = "export function rx() {\n  return /{/g;\n}\n";
        assert!(check_braces("src/r.ts", ts).is_ok());

        let jsx = "export function Banner() {\n  return <p>\n    Don't panic\n  </p>;\n}\n";
        assert!(check_braces("src/b.tsx", jsx).is_ok());

        // `sweep_stages` must NEVER delete `cargo-slots/`!
        let parent = std::env::temp_dir().join(format!("refit_g98_{}_{}", std::process::id(), now_ms()));
        let repo = parent.join("repo");
        fs::create_dir_all(&repo).unwrap();
        let slot = CargoSlot::acquire(&repo);
        let orphan = parent.join(".refit-stages/tx-dead-1");
        fs::create_dir_all(&orphan).unwrap();
        let swept = sweep_stages(&repo, 0);
        assert_eq!(swept, 1);
        assert!(!orphan.exists());
        assert!(slot.dir.exists(), "cargo-slots must survive orphan stage sweeps");
        drop(slot);

        // Pure-delete tx must pass Gate 2 instead of collapsing to 0.0.
        fs::write(repo.join("dead.rs"), "pub fn unused() {}\n").unwrap();
        let unit = ModuleUnit {
            id: "del_dead".into(),
            files: vec!["dead.rs".into()],
            layer: 0,
            wave: 0,
            deps: vec![],
            dependents: vec![],
            test_files: vec![],
            loc: 1,
            base_blake3: BTreeMap::new(),
            abi_hash: BTreeMap::new(),
            baseline_hygiene: 0.90,
            allow_abi_change: true,
        };
        let cfg = read_gates(&repo);
        let scorer = HygieneScorer::new();
        let v = run_gates(&cfg, &unit, &BTreeMap::new(), &["dead.rs".into()], &scorer).unwrap();
        assert!(v.passed, "pure-delete must pass Gate 2: {:?}", v.diagnostics);
        assert!(v.gate2_score_after >= 0.85);
        let _ = fs::remove_dir_all(&parent);
    }
}
