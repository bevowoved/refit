use super::stage::{CargoSlot, py_worktree_cache_env};
use crate::platform::{budget, run_cmd, sandbox_env};
use crate::types::{GateVerdict, SandboxResult};
use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;

static PYTEST_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-9]+)[ \t]+(passed|failed|error)").unwrap());
static LINE_TOK_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^L[0-9]+:?$").unwrap());
static LOC_TOK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([^:]+):[0-9]+(?::[0-9]*)?$").unwrap());

pub(super) struct CmdSpec<'a> {
    pub(super) prog: &'a str, args: &'a [String], cwd: &'a Path,
    env: &'a [(String, String)], timeout_ms: u64,
}

impl<'a> CmdSpec<'a> {
    pub(super) fn new(
        prog: &'a str, args: &'a [String], cwd: &'a Path, env: &'a [(String, String)], timeout_ms: u64,
    ) -> Self { Self { prog, args, cwd, env, timeout_ms } }

    pub(super) fn run(&self) -> Result<SandboxResult> {
        run_cmd(self.prog, self.args, self.cwd, self.env, true, self.timeout_ms)
    }
}

#[derive(Default)]
pub(super) struct RunAccum {
    pub(super) ruff_diags: Vec<String>, pub(super) ty_diags: Vec<String>,
    pub(super) poly_lint: Vec<String>, pub(super) pt_exit: i32,
    pub(super) pt_stdout: String, pub(super) ran_tests: bool,
}

pub(super) fn cmd_args(head: &[&str], tail: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(head.len() + tail.len());
    args.extend(head.iter().map(|&s| s.to_string()));
    args.extend(tail.iter().cloned());
    args
}

pub(super) fn py_test_env(repo_root: &Path, cwd: &Path) -> Vec<(String, String)> {
    let mut extra = py_worktree_cache_env(repo_root);
    extra.push(("PYTHONPATH".into(), cwd.display().to_string()));
    sandbox_env(&extra)
}

pub(super) fn check_g0(
    gate: &str, label: &str, cmd: &CmdSpec<'_>, strict: bool, use_stdout: bool,
) -> Option<(String, String)> {
    match cmd.run() {
        Ok(out) if out.exit_code == 0 && !out.timed_out => None,
        Ok(out) if out.timed_out => Some((gate.into(), format!("Gate 0 {label} timed out"))),
        Ok(out) => {
            let text = if use_stdout { &out.stdout } else { &out.stderr };
            Some((gate.into(), format!("Gate 0 {label} failed: {}", first_lines(text, 6))))
        }
        Err(e) if strict => Some(("gate0_tool_missing".into(), format!("{} missing: {e}", cmd.prog))),
        Err(_) => None,
    }
}

pub(super) fn run_rust_check(
    repo_root: &Path, cwd: &Path, slot: &CargoSlot, staged: bool, strict: bool,
) -> Option<(String, String)> {
    let env = merged_rust_cargo_env(&slot.dir, slot.rust_cache_env(repo_root, staged));
    let (prog, args, label) = detect_cold_rust_checker();
    if prog == "rust-glancer"
        && let Ok(out) = CmdSpec::new(prog, &args, cwd, &env, 20_000).run()
        && !out.timed_out
    {
        if out.exit_code == 0 { return None; }
        let combined = first_pair(&out.stdout, &out.stderr, 6);
        if !combined.to_ascii_lowercase().contains("unrecognized") {
            return Some(("gate0_rust_check".into(), format!("Gate 0 {label} failed: {combined}")));
        }
    }
    let cargo_args = cmd_args(&["check", "--quiet"], &[]);
    let spec = CmdSpec::new("cargo", &cargo_args, cwd, &env, 30_000);
    check_g0("gate0_rust_check", "cargo check", &spec, strict, false)
}

pub(super) fn probe_poly_lang(
    lang: &str, root: &Path, base_env: &[(String, String)],
) -> Option<(String, String)> {
    match lang {
        "rs" => {
            let slot = CargoSlot::acquire(root);
            let env = merged_rust_cargo_env(&slot.dir, slot.rust_cache_env(root, false));
            let (prog, args, label) = detect_cold_rust_checker();
            let ms = if prog == "rust-glancer" { 20_000 } else { 30_000 };
            poly_probe(&CmdSpec::new(prog, &args, root, &env, ms), "Cargo.toml", label, false)
        }
        "ts" => {
            let (prog, args, label) = detect_cold_ts_checker();
            poly_probe(&CmdSpec::new(prog, &args, root, base_env, 20_000), "tsconfig.json", label, true)
        }
        _ => {
            let args = cmd_args(&["vet", "./..."], &[]);
            poly_probe(&CmdSpec::new("go", &args, root, base_env, 20_000), "go.mod", "go vet", false)
        }
    }
}

fn poly_probe(cmd: &CmdSpec<'_>, file: &str, lbl: &str, use_out: bool) -> Option<(String, String)> {
    let o = cmd.run().ok()?;
    if o.exit_code == 0 || o.timed_out { return None; }
    let text = if use_out { &o.stdout } else { &o.stderr };
    Some((file.into(), format!("{lbl}: {}", first_lines(text, 4))))
}

pub(super) fn run_fmt(
    prog: &str, head: &[&str], files: &[String], cwd: &Path, env: &[(String, String)],
) -> usize {
    if files.is_empty() { return 0; }
    let args = cmd_args(head, files);
    match run_cmd(prog, &args, cwd, env, true, 8_000) {
        Ok(out) if out.exit_code == 0 => files.len(),
        _ => 0,
    }
}

pub(super) fn rust_edition(stage_root: &Path) -> String {
    let doc = fs::read_to_string(stage_root.join("Cargo.toml")).ok()
        .and_then(|s| toml::from_str::<toml::Value>(&s).ok());
    let ed = doc.as_ref().and_then(|d| {
        d.get("package").and_then(|p| p.get("edition")).and_then(toml::Value::as_str)
            .or_else(|| d.get("workspace")?.get("package")?.get("edition")?.as_str())
    });
    ed.filter(|e| matches!(*e, "2015" | "2018" | "2021" | "2024")).unwrap_or("2021").into()
}

pub(super) fn poly_test(
    label: &str, cmd: &CmdSpec<'_>, strict: bool, max_lines: usize,
    verdict: &mut GateVerdict, acc: &mut RunAccum,
) {
    let t = Instant::now();
    match cmd.run() {
        Ok(out) => {
            acc.ran_tests = true;
            verdict.gate1_ms += t.elapsed().as_millis() as u64;
            verdict.gate1_cpu_ms += out.cpu_ms;
            verdict.gate1_peak_rss_bytes = verdict.gate1_peak_rss_bytes.max(out.peak_rss_bytes);
            if out.exit_code != 0 || out.timed_out {
                verdict.gate1_tests_failed += 1;
                acc.pt_exit = 1;
                acc.pt_stdout.push_str(&format!("{label} failed: {}", first_pair(&out.stdout, &out.stderr, max_lines)));
            } else {
                verdict.gate1_tests_passed += 1;
            }
        }
        Err(e) if strict => {
            verdict.gate1_tests_failed += 1;
            acc.pt_exit = 1;
            acc.pt_stdout.push_str(&format!("{label} unavailable (strict mode): {e}"));
        }
        Err(_) => {}
    }
}

pub(super) fn parse_pytest(stdout: &str, stderr: &str, exit_code: i32) -> (usize, usize) {
    let (mut passed, mut failed) = (0usize, 0usize);
    let err_rev = stderr.lines().rev();
    for line in err_rev.chain(stdout.lines().rev()) {
        for cap in PYTEST_RE.captures_iter(line) {
            if let Ok(n) = cap[1].parse::<usize>() {
                if &cap[2] == "passed" { passed = passed.max(n); } else { failed = failed.max(n); }
            }
        }
        if passed > 0 || failed > 0 { break; }
    }
    if exit_code != 0 && failed == 0 { failed = 1; }
    (passed, failed)
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RuffDiag { filename: String, code: Option<String>, message: String, location: RuffLoc }

#[derive(Deserialize, Default)]
struct RuffLoc { #[serde(default)] row: u64 }

fn map_ruff_json<T>(stdout: &str, mut f: impl FnMut(&str, &str, u64, &str) -> T) -> Vec<T> {
    let Ok(arr) = serde_json::from_str::<Vec<RuffDiag>>(stdout.trim()) else { return Vec::new() };
    arr.iter()
        .map(|d| {
            let code = d.code.as_deref().unwrap_or("RUFF");
            f(&d.filename, code, d.location.row, &d.message)
        })
        .collect()
}

pub(super) fn parse_ruff(stdout: &str) -> Vec<String> {
    map_ruff_json(stdout, |raw, code, line, msg| {
        format!("ruff {} {code} L{line}: {msg}", raw.rsplit(['/', '\\']).next().unwrap_or(""))
    })
}

pub(super) fn ruff_map(stdout: &str, repo_root: &Path) -> Vec<(String, String)> {
    map_ruff_json(stdout, |raw, code, line, msg| {
        let rel = crate::platform::rel_to_root(repo_root, Path::new(raw)).unwrap_or_else(|| raw.replace('\\', "/"));
        (rel, format!("ruff {code} L{line}: {msg}"))
    })
}

pub(super) fn parse_ty(stdout: &str, stderr: &str) -> Vec<String> {
    let mut out: Vec<String> = stdout.lines().chain(stderr.lines()).map(str::trim)
        .filter(|t| t.starts_with("error[") || t.starts_with("warning[") || t.contains("--> "))
        .map(|t| format!("ty: {t}")).collect();
    if out.is_empty() {
        let summary = first_pair(stdout, stderr, 4);
        if !summary.is_empty() { out.push(format!("ty: {summary}")); }
    }
    out
}

pub(super) fn first_lines(s: &str, max: usize) -> String { first_pair(s, "", max) }

pub(super) fn first_pair(a: &str, b: &str, max: usize) -> String {
    let picked: Vec<&str> = a.lines()
        .chain(b.lines())
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect();
    picked.join(" | ")
}

pub(super) fn tool_on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    let dirs: Vec<PathBuf> = std::env::split_paths(&path).collect();
    super::stage::find_tool_in_dirs(&dirs, bin).is_some()
}

pub(super) fn oxlint_supports_type_aware() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        tool_on_path("oxlint")
            && run_cmd("oxlint", &["--help".into()], &std::env::temp_dir(), &sandbox_env(&[]), true, 3_000)
                .is_ok_and(|o| o.stdout.contains("--type-aware") || o.stderr.contains("--type-aware"))
    })
}

pub(super) fn select_ts_checker(has_tsrs: bool, has_ox_ta: bool) -> (&'static str, Vec<String>, &'static str) {
    if has_tsrs { ("tsrs", Vec::new(), "tsrs") }
    else if has_ox_ta { ("oxlint", cmd_args(&["--type-aware", "--type-check"], &[]), "oxlint --type-aware") }
    else { ("tsc", cmd_args(&["--noEmit"], &[]), "tsc --noEmit") }
}

pub(super) fn select_js_linter(
    has_ox: bool, has_ox_ta: bool, has_tsconfig: bool,
) -> (&'static str, Vec<String>, &'static str) {
    if has_ox && has_ox_ta && has_tsconfig { ("oxlint", cmd_args(&["--type-aware"], &[]), "oxlint --type-aware") }
    else if has_ox { ("oxlint", Vec::new(), "oxlint") }
    else { ("biome", cmd_args(&["lint"], &[]), "biome lint") }
}

pub(super) fn select_rust_checker(has_rg: bool) -> (&'static str, Vec<String>, &'static str) {
    if has_rg { ("rust-glancer", cmd_args(&["check"], &[]), "rust-glancer check") }
    else { ("cargo", cmd_args(&["check", "--quiet"], &[]), "cargo check") }
}

pub(super) fn detect_cold_ts_checker() -> (&'static str, Vec<String>, &'static str) {
    select_ts_checker(tool_on_path("tsrs"), oxlint_supports_type_aware())
}

pub(super) fn detect_cold_js_linter(has_tsconfig: bool) -> (&'static str, Vec<String>, &'static str) {
    select_js_linter(tool_on_path("oxlint"), oxlint_supports_type_aware(), has_tsconfig)
}

pub(super) fn detect_cold_rust_checker() -> (&'static str, Vec<String>, &'static str) {
    select_rust_checker(tool_on_path("rust-glancer"))
}

pub(super) fn rust_cargo_env_for(target_dir: &Path, constrained: bool) -> Vec<(String, String)> {
    let mut extra = vec![("CARGO_TARGET_DIR".into(), target_dir.display().to_string())];
    if constrained {
        for (k, v) in [("CARGO_BUILD_JOBS", "1"), ("CARGO_INCREMENTAL", "0"), ("CARGO_PROFILE_DEV_CODEGEN_UNITS", "1")] {
            extra.push((k.into(), v.into()));
        }
    }
    sandbox_env(&extra)
}

pub(super) fn merged_rust_cargo_env(target_dir: &Path, extra: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut env = rust_cargo_env_for(target_dir, budget().is_constrained());
    for (k, v) in extra {
        if let Some(entry) = env.iter_mut().find(|(ek, _)| ek.eq_ignore_ascii_case(&k)) {
            entry.1 = v;
        } else {
            env.push((k, v));
        }
    }
    env
}

fn normalize_diag_token(tok: &str) -> &str {
    if LINE_TOK_RE.is_match(tok) { return ""; }
    LOC_TOK_RE.captures(tok).and_then(|c| c.get(1)).map_or(tok, |m| m.as_str())
}

fn normalize_diag_key(diag: &str) -> String {
    let toks: Vec<&str> = diag.split_whitespace()
        .map(normalize_diag_token)
        .filter(|t| !t.is_empty())
        .collect();
    toks.join(" ")
}

/// Compute net-new diagnostics introduced after an edit relative to a baseline snapshot.
pub fn net_new_diagnostics(baseline: &[String], current: &[String]) -> Vec<String> {
    if baseline.is_empty() { return current.to_vec(); }
    let mut budget_map: HashMap<String, usize> = HashMap::with_capacity(baseline.len());
    for diag in baseline { *budget_map.entry(normalize_diag_key(diag)).or_insert(0) += 1; }
    current
        .iter()
        .filter(|diag| match budget_map.get_mut(&normalize_diag_key(diag)) {
            Some(rem) if *rem > 0 => { *rem -= 1; false }
            _ => true,
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruff_map_path_norm() {
        let mk = |f: &str, c: &str, m: &str, r: u64| {
            serde_json::json!({ "filename": f, "code": c, "message": m, "location": { "row": r } })
        };
        let json_out = serde_json::Value::Array(vec![
            mk(r"\\?\d:\cerebras\refit\src\mod_a.py", "F401", "unused import", 3),
            mk(r"D:\cerebras\refit\pkg\sub\b.py", "E501", "line too long", 12),
        ]).to_string();
        assert_eq!(ruff_map(&json_out, Path::new(r"D:\cerebras\refit")), vec![
            ("src/mod_a.py".into(), "ruff F401 L3: unused import".into()),
            ("pkg/sub/b.py".into(), "ruff E501 L12: line too long".into()),
        ]);
    }

    #[test]
    fn cold_lsp_net_new_diagnostics() {
        let baseline: Vec<String> = vec![
            "ruff app.py F401 L3: unused import".into(),
            "src/view.ts:10:5 error TS2322: type mismatch".into(),
        ];
        let current: Vec<String> = vec![
            "ruff app.py F401 L7: unused import".into(),
            "ruff app.py F401 L15: unused import".into(),
            "src/view.ts:14:5 error TS2322: type mismatch".into(),
            "ruff app.py B006 L22: mutable default".into(),
        ];

        assert_eq!(net_new_diagnostics(&baseline, &current),
            vec!["ruff app.py F401 L15: unused import", "ruff app.py B006 L22: mutable default"]);
        assert_eq!(net_new_diagnostics(&[], &current), current);
        assert!(net_new_diagnostics(&baseline, &baseline).is_empty());

        for (tsrs, ox_ta, exp) in [(true, true, "tsrs"), (false, true, "oxlint"), (false, false, "tsc")] {
            assert_eq!(select_ts_checker(tsrs, ox_ta).0, exp);
        }
        for (ox, ta, ts, exp) in [(true, true, true, "oxlint --type-aware"), (true, false, true, "oxlint"), (false, false, true, "biome lint")] {
            assert_eq!(select_js_linter(ox, ta, ts).2, exp);
        }
        for (rg, exp) in [(true, "rust-glancer"), (false, "cargo")] {
            assert_eq!(select_rust_checker(rg).0, exp);
        }

        let has_var = |env: &[(String, String)], k: &str, v: &str| env.iter().any(|(ek, ev)| ek == k && ev == v);
        let tight = rust_cargo_env_for(Path::new("/tmp/cargo-slot"), true);
        for (k, v) in [("CARGO_BUILD_JOBS", "1"), ("CARGO_INCREMENTAL", "0"), ("CARGO_PROFILE_DEV_CODEGEN_UNITS", "1")] {
            assert!(has_var(&tight, k, v));
        }
        assert!(!has_var(&rust_cargo_env_for(Path::new("/tmp/cargo-slot"), false), "CARGO_BUILD_JOBS", "1"));
    }
}
