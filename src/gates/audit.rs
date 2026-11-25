use super::runners::{
    CmdSpec, detect_cold_js_linter, detect_cold_rust_checker, detect_cold_ts_checker, first_lines,
    merged_rust_cargo_env, parse_pytest, parse_ty, ruff_map,
};
use super::stage::{
    CargoSlot, ensure_cache_dirs_tagged, polyglot_cache_env, py_worktree_cache_env,
};
use crate::graph::build_dag;
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::{
    FileBuf, StatEntry, budget, collect_os_memory, ensure_cachedir_tag, env_trimmed, env_truthy,
    file_mtime_size, load_stat_cache, now_unix_ns, par_join2, par_join3, par_map, read_zstd,
    run_cmd, sandbox_env, save_stat_cache, stat_cache_fresh, walk_files, write_zstd,
};
use crate::source::{ext_lang, ext_str, rel_norm, source_rel, stem_of, supported_ext};
use crate::types::{DagPlan, SandboxResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

const TOOLCHAINS: [(&str, &str, &str, &str); 5] = [
    ("python", "ruff (format + check)", "ty", "pytest"),
    ("rust", "rustfmt + cargo clippy", "cargo check", "cargo test"),
    ("typescript", "biome / eslint", "tsc --noEmit", "vitest / node --test"),
    ("javascript", "biome / eslint", "node --check", "node --test"),
    ("go", "gofmt + go vet", "gopls / go vet", "go test ./..."),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainInfo {
    pub language: String,
    pub canonical_linter: String,
    pub canonical_typecheck_lsp: String,
    pub canonical_test_runner: String,
}

pub fn detect_tools(langs: &[String]) -> Vec<ToolchainInfo> {
    let ((_, _, ts_tc), (_, _, js_lint), (_, _, rs_tc)) =
        (detect_cold_ts_checker(), detect_cold_js_linter(false), detect_cold_rust_checker());
    langs.iter().map(|l| {
        let (_, lint, tc, test) = TOOLCHAINS.iter().find(|&&(k, _, _, _)| k == l)
            .copied().unwrap_or(("", "refit-untangle", "host-lsp", "repo-test"));
        let canonical_linter = match l.as_str() {
            "typescript" | "javascript" if js_lint.starts_with("oxlint") => js_lint,
            _ => lint,
        }.into();
        let canonical_typecheck_lsp = match l.as_str() {
            "typescript" => ts_tc,
            "rust" => rs_tc,
            _ => tc,
        }.into();
        ToolchainInfo { language: l.clone(), canonical_linter, canonical_typecheck_lsp, canonical_test_runner: test.into() }
    }).collect()
}

#[cfg(test)]
pub fn detect_langs(repo_root: &Path) -> Vec<String> { read_tree(repo_root).langs }

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub repo_root: PathBuf, pub python_bin: String, pub pytest_bin: String,
    pub ruff_bin: String, pub ty_bin: String, pub min_hygiene_target: f64,
    pub strict_polyglot_tools: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStats {
    pub tests_total: usize, pub tests_failed: usize, pub ruff_errors: usize, pub ty_errors: usize,
    pub hygiene_median: f64, pub hygiene_p10: f64, pub per_file_diagnostics: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanCache {
    pub version: String, pub repo_fingerprint: String,
    #[serde(default)] pub fast_mode: bool,
    #[serde(default)] pub per_file_hashes: BTreeMap<String, String>,
    pub detected_languages: Vec<String>, pub polyglot_toolchains: Vec<ToolchainInfo>,
    pub plan: DagPlan, pub audit: AuditStats,
}

struct TreeMeta {
    fingerprint: String, langs: Vec<String>, source_rels: Vec<String>,
    supported: Vec<(String, PathBuf)>, hashes: BTreeMap<String, String>,
}

fn collect_tree_paths(repo_root: &Path) -> (Vec<String>, Vec<(String, PathBuf)>, Vec<String>) {
    let (mut langs, mut supported, mut source_rels) = (BTreeSet::new(), Vec::new(), Vec::new());
    for path in walk_files(repo_root, None) {
        let ext = ext_str(&path);
        if !supported_ext(ext) { continue; }
        if let Some(lang) = ext_lang(ext) { langs.insert(lang.to_string()); }
        let Some(rel) = rel_norm(repo_root, &path) else { continue };
        if source_rel(&rel) { source_rels.push(rel.clone()); }
        supported.push((rel, path));
    }
    if langs.is_empty() { langs.insert("polyglot".into()); }
    supported.sort_by(|a, b| a.0.cmp(&b.0));
    source_rels.sort();
    (langs.into_iter().collect(), supported, source_rels)
}

fn resolve_file_hash(
    rel: &str, abs: &Path, prev_stats: &BTreeMap<String, StatEntry>, now_ns: u64, fb: &mut FileBuf,
) -> (u64, String, Option<StatEntry>) {
    if let Some((mtime_ns, size)) = file_mtime_size(abs) {
        if let Some(prev) = prev_stats.get(rel) && stat_cache_fresh(prev, mtime_ns, size, now_ns) {
            return (size, prev.blake3.clone(), None);
        }
        if let Ok((len, hash)) = fb.hash_file_blake3(abs) {
            return (len, hash.clone(), Some(StatEntry { mtime_ns, size: len, blake3: hash }));
        }
    }
    let bytes = fs::read(abs).unwrap_or_default();
    (bytes.len() as u64, MvccStore::hash_bytes(&bytes), None)
}

fn hash_supported_entries(
    supported: &[(String, PathBuf)],
    prev_stats: &BTreeMap<String, StatEntry>,
    now_ns: u64,
) -> Vec<(String, u64, String, Option<StatEntry>)> {
    thread_local! {
        static BUF: RefCell<FileBuf> = RefCell::new(FileBuf::new());
    }
    par_map(supported, |(rel, abs)| {
        let (len, h, upd) = BUF.with(|cell| {
            resolve_file_hash(rel, abs, prev_stats, now_ns, &mut cell.borrow_mut())
        });
        (rel.clone(), len, h, upd)
    })
}

fn read_tree(repo_root: &Path) -> TreeMeta {
    let (langs, supported, source_rels) = collect_tree_paths(repo_root);
    let stat_path = repo_root.join(".refit/cache/stat.bin");
    let (prev_stats, now_ns) = (load_stat_cache(&stat_path), now_unix_ns());
    let (mut next_stats, mut stat_dirty, mut hashes) = (BTreeMap::new(), false, BTreeMap::new());
    let mut fp = blake3::Hasher::new();

    for (idx, (rel, len, hash, updated)) in
        hash_supported_entries(&supported, &prev_stats, now_ns).into_iter().enumerate()
    {
        if idx > 0 {
            fp.update(b"\n");
        }
        fp.update(format!("{rel}:{len}:{hash}").as_bytes());
        if let Some(entry) = updated {
            stat_dirty = true;
            next_stats.insert(rel.clone(), entry);
        } else if let Some(prev) = prev_stats.get(&rel) {
            next_stats.insert(rel.clone(), prev.clone());
        }
        hashes.insert(rel, hash);
    }
    if stat_dirty || next_stats.len() != prev_stats.len() {
        let _ = save_stat_cache(&stat_path, &next_stats);
    }
    TreeMeta { fingerprint: fp.finalize().to_hex().to_string(), langs, source_rels, supported, hashes }
}

pub fn scan_repo(repo_root: &Path, use_cache: bool, scorer: &HygieneScorer) -> Result<(ScanCache, bool)> {
    scan_cached(repo_root, use_cache, false, scorer)
}

pub fn scan_cached(
    repo_root: &Path,
    use_cache: bool,
    skip_tests: bool,
    scorer: &HygieneScorer,
) -> Result<(ScanCache, bool)> {
    let cache_dir = repo_root.join(".refit/cache");
    let cache_path = cache_dir.join("scan_cache.zst");
    let snap = read_tree(repo_root);
    let prev = read_zstd::<ScanCache>(&cache_path).filter(|c| c.version == env!("CARGO_PKG_VERSION"));
    if use_cache
        && let Some(ref c) = prev
        && c.repo_fingerprint == snap.fingerprint
        && (!c.fast_mode || skip_tests)
    {
        return Ok((c.clone(), true));
    }

    ensure_cachedir_tag(&cache_dir);
    let preloaded_owned: Option<BTreeMap<String, Vec<u8>>> = (!budget().is_constrained()).then(|| {
        par_map(&snap.supported, |(rel, abs)| (rel.clone(), fs::read(abs).unwrap_or_default()))
            .into_iter()
            .collect()
    });
    let preloaded = preloaded_owned.as_ref();
    let (polyglot_toolchains, gate_cfg) = (detect_tools(&snap.langs), read_gates(repo_root));
    let changed: Vec<String> = match prev {
        Some(ref p) => snap
            .source_rels
            .iter()
            .filter(|r| p.per_file_hashes.get(*r) != snap.hashes.get(*r))
            .cloned()
            .collect(),
        None => snap.source_rels.clone(),
    };
    let run_audit = || {
        let partial = use_cache && skip_tests && prev.is_some() && !changed.is_empty();
        let targets = if partial { &changed } else { &snap.source_rels };
        let mut fresh = audit_files(&gate_cfg, targets, scorer, preloaded, skip_tests)?;
        if let Some(ref p) = prev.filter(|_| partial) {
            merge_partial(&gate_cfg, scorer, &snap, preloaded, &changed, p, &mut fresh);
        }
        Ok::<AuditStats, anyhow::Error>(fresh)
    };
    // Pre-warm hygiene once before `build_dag` so `dag_from` hits warm cache with zero contention.
    let (audit_res, plan_res) = par_join2(
        run_audit,
        || build_dag(repo_root, None, scorer, preloaded),
    );
    drop(preloaded_owned);
    scorer.flush_cache();
    collect_os_memory();
    let bundle = ScanCache {
        version: env!("CARGO_PKG_VERSION").into(), repo_fingerprint: snap.fingerprint, fast_mode: skip_tests,
        per_file_hashes: snap.hashes, detected_languages: snap.langs, polyglot_toolchains,
        plan: plan_res?, audit: audit_res?,
    };
    let _ = write_zstd(&cache_path, &bundle);
    Ok((bundle, false))
}

fn merge_partial(
    gate_cfg: &GateConfig, scorer: &HygieneScorer, snap: &TreeMeta,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>, changed: &[String], prev: &ScanCache, fresh: &mut AuditStats,
) {
    let changed_set: BTreeSet<&String> = changed.iter().collect();
    for rel in snap.source_rels.iter().filter(|r| !changed_set.contains(r)) {
        if let Some(old) = prev.audit.per_file_diagnostics.get(rel) {
            fresh.per_file_diagnostics.entry(rel.clone()).or_insert_with(|| old.clone());
        }
    }
    let (med, p10, _) = scorer.score_files(&gate_cfg.repo_root, &snap.source_rels, preloaded);
    (fresh.hygiene_median, fresh.hygiene_p10) = (med, p10);
    fresh.ruff_errors = fresh
        .per_file_diagnostics
        .values()
        .flatten()
        .filter(|d| d.starts_with("ruff "))
        .count();
    fresh.ty_errors = if changed.iter().any(|r| r.ends_with(".py")) {
        recount_ty(&gate_cfg.repo_root, gate_cfg, &snap.source_rels).unwrap_or(fresh.ty_errors)
    } else {
        prev.audit.ty_errors
    };
    (fresh.tests_total, fresh.tests_failed) = (prev.audit.tests_total, prev.audit.tests_failed);
}

fn py_rels(rels: &[String]) -> Vec<String> {
    rels.iter()
        .filter(|r| r.ends_with(".py"))
        .cloned()
        .collect()
}

fn recount_ty(repo_root: &Path, cfg: &GateConfig, source_rels: &[String]) -> Option<usize> {
    let all_py = py_rels(source_rels);
    if all_py.is_empty() { return None; }
    let mut args = vec!["check".into(), "--error-on-warning".into()];
    args.extend(all_py);
    let o = run_cmd(&cfg.ty_bin, &args, repo_root, &sandbox_env(&py_worktree_cache_env(repo_root)), true, 20_000).ok()?;
    Some(if o.exit_code == 0 { 0 } else { parse_ty(&o.stdout, &o.stderr).len() })
}

pub fn read_gates(repo_root: &Path) -> GateConfig {
    let user_local = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))
        .map(|h| PathBuf::from(h).join(".local/bin")).unwrap_or_default();
    let mut venvs = vec![repo_root.join(".venv"), repo_root.join("venv")];
    if let Some(v) = env_trimmed("VIRTUAL_ENV") {
        venvs.push(PathBuf::from(v));
    }
    if let Ok(exe) = std::env::current_exe() {
        venvs.extend(exe.ancestors().take(4).map(|a| a.join(".venv")));
    }
    venvs.push(PathBuf::from(".venv"));
    let (py_rel, pt_rel) = if cfg!(windows) {
        ("Scripts/python.exe", "Scripts/pytest.exe")
    } else {
        ("bin/python", "bin/pytest")
    };
    let (mut python_bin, mut pytest_bin) = ("python".to_string(), "pytest".to_string());
    for v in &venvs {
        let (py, pt) = (v.join(py_rel), v.join(pt_rel));
        if py.exists() {
            python_bin = py.display().to_string();
        }
        if pt.exists() {
            pytest_bin = pt.display().to_string();
            break;
        }
    }
    let strict = env_truthy("REFIT_STRICT_TOOLS");
    GateConfig {
        repo_root: repo_root.to_path_buf(), python_bin, pytest_bin,
        ruff_bin: resolve_tool("ruff", &user_local), ty_bin: resolve_tool("ty", &user_local),
        min_hygiene_target: 0.85, strict_polyglot_tools: strict,
    }
}

fn resolve_tool(name: &str, user_local: &Path) -> String {
    let local = user_local.join(if cfg!(windows) { format!("{name}.exe") } else { name.into() });
    if local.exists() { local.display().to_string() } else { name.into() }
}

pub fn audit_repo(cfg: &GateConfig, rel_files: &[String], hygiene: &HygieneScorer) -> Result<AuditStats> {
    audit_files(cfg, rel_files, hygiene, None, false)
}

type PyAudit = (usize, usize, Vec<String>, Vec<(String, String)>, Vec<String>);

pub fn audit_files(
    cfg: &GateConfig,
    rel_files: &[String],
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    skip_tests: bool,
) -> Result<AuditStats> {
    let py_files = py_rels(rel_files);
    ensure_cache_dirs_tagged(&cfg.repo_root);
    let base_env = sandbox_env(&polyglot_cache_env(&cfg.repo_root));

    let ((med, p10, per_h), py_audit, poly_diags) = par_join3(
        || hygiene.score_files(&cfg.repo_root, rel_files, preloaded),
        || (!py_files.is_empty()).then(|| audit_python(cfg, &py_files, &base_env, skip_tests)),
        || audit_polyglot(cfg, rel_files, &base_env, skip_tests),
    );

    let mut stats = AuditStats { hygiene_median: med, hygiene_p10: p10, ..AuditStats::default() };
    for (file_k, msg) in poly_diags {
        stats.per_file_diagnostics.entry(file_k).or_default().push(msg);
    }
    if let Some(res) = py_audit {
        merge_py(res?, &py_files, &mut stats);
    }

    for (rel, hr) in &per_h {
        for sq in hr.squiggles.iter().take(6) {
            let msg = format!("codehygiene L{}: {} ({:.2})", sq.line, sq.kind, sq.probability);
            stats.per_file_diagnostics.entry(rel.clone()).or_default().push(msg);
        }
    }
    for diags in stats.per_file_diagnostics.values_mut() {
        diags.sort();
        diags.dedup();
    }
    Ok(stats)
}

fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

fn opt_tool<T: Default>(
    res: Result<SandboxResult>,
    strict: bool,
    parse: impl FnOnce(SandboxResult) -> T,
) -> Result<T> {
    match res {
        Ok(out) => Ok(parse(out)),
        Err(e) if !strict && is_not_found(&e) => Ok(T::default()),
        Err(e) => Err(e),
    }
}

fn audit_python(
    cfg: &GateConfig,
    py_files: &[String],
    base_env: &[(String, String)],
    skip_tests: bool,
) -> Result<PyAudit> {
    let run_pt = || {
        let mut py_extra = py_worktree_cache_env(&cfg.repo_root);
        py_extra.push(("PYTHONPATH".into(), cfg.repo_root.display().to_string()));
        let env = sandbox_env(&py_extra);
        let args = ["-q".into(), "--tb=line".into()];
        run_cmd(&cfg.pytest_bin, &args, &cfg.repo_root, &env, true, 30_000)
    };
    let run_rf = || {
        let mut args = vec!["check".into(), "--output-format=json".into()];
        args.extend(py_files.iter().cloned());
        run_cmd(&cfg.ruff_bin, &args, &cfg.repo_root, base_env, true, 15_000)
    };
    let run_ty_cmd = || {
        let mut args = vec!["check".into(), "--error-on-warning".into()];
        args.extend(py_files.iter().cloned());
        run_cmd(&cfg.ty_bin, &args, &cfg.repo_root, base_env, true, 20_000)
    };

    let want_pt = !skip_tests && cfg.repo_root.join("tests").exists();
    let (pt_res, ruff_res, ty_res) = par_join3(|| want_pt.then(run_pt), run_rf, run_ty_cmd);
    let strict = cfg.strict_polyglot_tools;

    let (passed, failed, failed_lines) = match pt_res {
        Some(res) => opt_tool(res, strict, |pt| {
            let (p, f) = parse_pytest(&pt.stdout, &pt.stderr, pt.exit_code);
            let fl = pt
                .stdout
                .lines()
                .filter(|l| l.starts_with("FAILED "))
                .map(String::from)
                .collect();
            (p, f, fl)
        })?,
        None => Default::default(),
    };
    let ruff_items = opt_tool(ruff_res, strict, |o| ruff_map(&o.stdout, &cfg.repo_root))?;
    let ty_lines = opt_tool(ty_res, strict, |o| {
        if o.exit_code == 0 { Vec::new() } else { parse_ty(&o.stdout, &o.stderr) }
    })?;
    Ok((passed, failed, failed_lines, ruff_items, ty_lines))
}

fn audit_polyglot(
    cfg: &GateConfig,
    rel_files: &[String],
    base_env: &[(String, String)],
    skip_tests: bool,
) -> Vec<(String, String)> {
    if skip_tests {
        return Vec::new();
    }
    let root = &cfg.repo_root;
    let has_ext = |exts: &[&str]| rel_files.iter().any(|f| exts.iter().any(|e| f.ends_with(e)));
    let (run_rs, run_ts, run_go) = (
        has_ext(&[".rs"]) && root.join("Cargo.toml").exists(),
        has_ext(&[".ts", ".tsx"]) && root.join("tsconfig.json").exists(),
        has_ext(&[".go"]) && root.join("go.mod").exists(),
    );
    let (rs, ts, go) = par_join3(
        || run_rs.then(|| probe_cargo(root)).flatten(),
        || {
            run_ts.then(|| {
                let (prog, args, label) = detect_cold_ts_checker();
                poly_probe(&CmdSpec::new(prog, &args, root, base_env, 20_000), "tsconfig.json", label, true)
            }).flatten()
        },
        || {
            run_go.then(|| {
                let args = ["vet".into(), "./...".into()];
                poly_probe(&CmdSpec::new("go", &args, root, base_env, 20_000), "go.mod", "go vet", false)
            }).flatten()
        },
    );
    let mut findings = Vec::new();
    findings.extend(rs);
    findings.extend(ts);
    findings.extend(go);
    findings
}

fn probe_cargo(root: &Path) -> Option<(String, String)> {
    let slot = CargoSlot::acquire(root);
    let env = merged_rust_cargo_env(&slot.dir, slot.rust_cache_env(root, false));
    let (prog, args, label) = detect_cold_rust_checker();
    if prog == "rust-glancer" {
        return poly_probe(&CmdSpec::new(prog, &args, root, &env, 20_000), "Cargo.toml", label, false);
    }
    let cargo_args = ["check".into(), "--quiet".into()];
    poly_probe(&CmdSpec::new("cargo", &cargo_args, root, &env, 30_000), "Cargo.toml", "cargo check", false)
}

fn poly_probe(cmd: &CmdSpec<'_>, file: &str, lbl: &str, use_out: bool) -> Option<(String, String)> {
    let o = cmd.run().ok()?;
    if o.exit_code == 0 || o.timed_out { return None; }
    let text = if use_out { &o.stdout } else { &o.stderr };
    Some((file.into(), format!("{lbl}: {}", first_lines(text, 4))))
}

fn merge_py((passed, failed, failed_lines, ruff_items, ty_lines): PyAudit, py_files: &[String], stats: &mut AuditStats) {
    (stats.tests_total, stats.tests_failed, stats.ruff_errors, stats.ty_errors) =
        (passed + failed, failed, ruff_items.len(), ty_lines.len());
    for line in failed_lines {
        for rel in py_files {
            let stem = stem_of(Path::new(rel));
            if !stem.is_empty() && line.contains(&format!("test_{stem}")) {
                stats.per_file_diagnostics.entry(rel.clone()).or_default().push(format!("pytest {line}"));
            }
        }
    }
    for (file_rel, msg) in ruff_items {
        stats.per_file_diagnostics.entry(file_rel).or_default().push(msg);
    }
    for msg in &ty_lines {
        for rel in py_files {
            if msg.contains(rel) || msg.replace('\\', "/").contains(rel) {
                stats.per_file_diagnostics.entry(rel.clone()).or_default().push(msg.clone());
            }
        }
    }
}
