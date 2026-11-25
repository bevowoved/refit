use crate::platform::{budget, run_cmd, sandbox_env};
use crate::types::{GateVerdict, SandboxResult};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

pub(super) struct CmdSpec<'a> {
    pub(super) prog: &'a str,
    args: &'a [String],
    cwd: &'a Path,
    env: &'a [(String, String)],
    timeout_ms: u64,
}

impl<'a> CmdSpec<'a> {
    pub(super) fn new(
        prog: &'a str,
        args: &'a [String],
        cwd: &'a Path,
        env: &'a [(String, String)],
        timeout_ms: u64,
    ) -> Self {
        Self { prog, args, cwd, env, timeout_ms }
    }

    pub(super) fn run(&self) -> Result<SandboxResult> {
        run_cmd(self.prog, self.args, self.cwd, self.env, true, self.timeout_ms)
    }
}

#[derive(Default)]
pub(super) struct RunAccum {
    pub(super) ruff_diags: Vec<String>,
    pub(super) ty_diags: Vec<String>,
    pub(super) poly_lint: Vec<String>,
    pub(super) pt_exit: i32,
    pub(super) pt_stdout: String,
    pub(super) ran_tests: bool,
}

pub(super) fn check_g0(
    gate: &str,
    label: &str,
    cmd: &CmdSpec<'_>,
    strict: bool,
    use_stdout: bool,
) -> Option<(String, String)> {
    match cmd.run() {
        Ok(out) if out.exit_code == 0 && !out.timed_out => None,
        Ok(out) if out.timed_out => Some((gate.into(), format!("Gate 0 {label} timed out"))),
        Ok(out) => {
            let text = if use_stdout { &out.stdout } else { &out.stderr };
            Some((gate.into(), format!("Gate 0 {label} failed: {}", first_lines(text, 6))))
        }
        Err(e) if strict => {
            Some(("gate0_tool_missing".into(), format!("{} missing: {e}", cmd.prog)))
        }
        Err(_) => None,
    }
}

pub(super) fn run_fmt(
    prog: &str,
    args: &[String],
    cwd: &Path,
    env: &[(String, String)],
    count: usize,
) -> usize {
    match run_cmd(prog, args, cwd, env, true, 8_000) {
        Ok(out) if out.exit_code == 0 => count,
        _ => 0,
    }
}

pub(super) fn rust_edition(stage_root: &Path) -> String {
    let Ok(src) = fs::read_to_string(stage_root.join("Cargo.toml")) else {
        return "2021".into();
    };
    let found = src.lines().find_map(|l| {
        let rest = l.trim().strip_prefix("edition")?;
        let (_, v) = rest.split_once('=')?;
        Some(v.trim().trim_matches(['"', '\'']).to_string())
    });
    found
        .filter(|e| matches!(e.as_str(), "2015" | "2018" | "2021" | "2024"))
        .unwrap_or_else(|| "2021".into())
}

pub(super) fn poly_test(
    label: &str,
    cmd: &CmdSpec<'_>,
    strict: bool,
    max_lines: usize,
    verdict: &mut GateVerdict,
    acc: &mut RunAccum,
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
                let msg = first_pair(&out.stdout, &out.stderr, max_lines);
                acc.pt_stdout.push_str(&format!("{label} failed: {msg}"));
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
    let rev_err = stderr.lines().rev();
    let rev_out = stdout.lines().rev();

    for line in rev_err.chain(rev_out) {
        let l = line.trim();
        if !l.contains(" passed") && !l.contains(" failed") && !l.contains(" error") {
            continue;
        }
        count_summary(l, &mut passed, &mut failed);
        if passed > 0 || failed > 0 {
            break;
        }
    }

    if exit_code != 0 && failed == 0 {
        failed = 1;
    }
    (passed, failed)
}

fn count_summary(line: &str, passed: &mut usize, failed: &mut usize) {
    for part in line.split(',') {
        let toks: Vec<&str> = part.split_whitespace().collect();
        for w in toks.windows(2) {
            let Ok(n) = w[0].parse::<usize>() else { continue };
            if w[1].starts_with("passed") {
                *passed = (*passed).max(n);
            } else if w[1].starts_with("failed") || w[1].starts_with("error") {
                *failed = (*failed).max(n);
            }
        }
    }
}

pub(super) fn parse_ruff(stdout: &str) -> Vec<String> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    arr.into_iter()
        .map(|v| {
            let raw = v["filename"].as_str().unwrap_or("");
            let fname = raw.rsplit(['/', '\\']).next().unwrap_or("");
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            format!("ruff {fname} {code} L{line}: {msg}")
        })
        .collect()
}

pub(super) fn ruff_map(stdout: &str, repo_root: &Path) -> Vec<(String, String)> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };

    arr.into_iter()
        .map(|v| {
            let raw = v["filename"].as_str().unwrap_or("");
            let rel = crate::platform::rel_to_root(repo_root, Path::new(raw))
                .unwrap_or_else(|| raw.replace('\\', "/"));
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            (rel, format!("ruff {code} L{line}: {msg}"))
        })
        .collect()
}

pub(super) fn parse_ty(stdout: &str, stderr: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let t = line.trim();
        if t.starts_with("error[") || t.starts_with("warning[") || t.contains("--> ") {
            out.push(format!("ty: {t}"));
        }
    }
    if out.is_empty() {
        let summary = first_pair(stdout, stderr, 4);
        if !summary.is_empty() {
            out.push(format!("ty: {summary}"));
        }
    }
    out
}

pub(super) fn first_lines(s: &str, max: usize) -> String {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect::<Vec<_>>()
        .join(" | ")
}

pub(super) fn first_pair(a: &str, b: &str, max: usize) -> String {
    a.lines()
        .chain(b.lines())
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Fast zero-spawn `PATH` lookup for cold low-RAM native Rust verifiers.
pub(super) fn tool_on_path(bin: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    let exts: &[&str] = if cfg!(windows) {
        &["", ".exe", ".cmd", ".bat"]
    } else {
        &[""]
    };
    for dir in std::env::split_paths(&path_var) {
        for ext in exts {
            let candidate = if ext.is_empty() {
                dir.join(bin)
            } else {
                dir.join(format!("{bin}{ext}"))
            };
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

pub(super) fn oxlint_supports_type_aware() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        if !tool_on_path("oxlint") {
            return false;
        }
        let cwd = std::env::temp_dir();
        let args = vec!["--help".into()];
        let env = sandbox_env(&[]);
        run_cmd("oxlint", &args, &cwd, &env, true, 3_000)
            .is_ok_and(|out| out.stdout.contains("--type-aware") || out.stderr.contains("--type-aware"))
    })
}

pub(super) fn select_ts_checker(
    has_tsrs: bool,
    has_oxlint_type_aware: bool,
) -> (&'static str, Vec<String>, &'static str) {
    if has_tsrs {
        ("tsrs", Vec::new(), "tsrs")
    } else if has_oxlint_type_aware {
        (
            "oxlint",
            vec!["--type-aware".into(), "--type-check".into()],
            "oxlint --type-aware",
        )
    } else {
        ("tsc", vec!["--noEmit".into()], "tsc --noEmit")
    }
}

pub(super) fn select_js_linter(
    has_oxlint: bool,
    has_oxlint_type_aware: bool,
    has_tsconfig: bool,
) -> (&'static str, Vec<String>, &'static str) {
    if has_oxlint && has_oxlint_type_aware && has_tsconfig {
        ("oxlint", vec!["--type-aware".into()], "oxlint --type-aware")
    } else if has_oxlint {
        ("oxlint", Vec::new(), "oxlint")
    } else {
        ("biome", vec!["lint".into()], "biome lint")
    }
}

pub(super) fn select_rust_checker(
    has_rust_glancer: bool,
) -> (&'static str, Vec<String>, &'static str) {
    if has_rust_glancer {
        ("rust-glancer", vec!["check".into()], "rust-glancer check")
    } else {
        ("cargo", vec!["check".into(), "--quiet".into()], "cargo check")
    }
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
        extra.push(("CARGO_BUILD_JOBS".into(), "1".into()));
        extra.push(("CARGO_INCREMENTAL".into(), "0".into()));
        extra.push(("CARGO_PROFILE_DEV_CODEGEN_UNITS".into(), "1".into()));
    }
    sandbox_env(&extra)
}

pub(super) fn rust_cargo_env(target_dir: &Path) -> Vec<(String, String)> {
    rust_cargo_env_for(target_dir, budget().is_constrained())
}

pub(super) fn merged_rust_cargo_env(
    target_dir: &Path,
    extra: Vec<(String, String)>,
) -> Vec<(String, String)> {
    let mut env = rust_cargo_env(target_dir);
    for (k, v) in extra {
        if let Some(entry) = env.iter_mut().find(|(ek, _)| ek == &k) {
            entry.1 = v;
        } else {
            env.push((k, v));
        }
    }
    env
}

fn normalize_diag_token(tok: &str) -> String {
    let no_colon = tok.trim_end_matches(':');
    if let Some(rest) = no_colon.strip_prefix('L')
        && !rest.is_empty()
        && rest.bytes().all(|byte| byte.is_ascii_digit())
    {
        return String::new();
    }
    let mut parts = tok.split(':');
    let head = parts.next().unwrap_or(tok);
    let second = parts.next();
    let third = parts.next();
    let is_num = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if second.is_some_and(is_num) && third.is_none_or(|s| s.is_empty() || is_num(s)) {
        return head.to_string();
    }
    tok.to_string()
}

fn normalize_diag_key(diag: &str) -> String {
    diag.split_whitespace()
        .map(normalize_diag_token)
        .filter(|tok| !tok.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute net-new diagnostics introduced after an edit relative to a baseline snapshot.
/// Uses line-shift-tolerant multiset subtraction so pre-existing lint debt is never blamed.
pub fn net_new_diagnostics(baseline: &[String], current: &[String]) -> Vec<String> {
    if baseline.is_empty() {
        return current.to_vec();
    }
    let mut budget_map: HashMap<String, usize> = HashMap::with_capacity(baseline.len());
    for diag in baseline {
        *budget_map.entry(normalize_diag_key(diag)).or_insert(0) += 1;
    }

    let mut net_new = Vec::new();
    for diag in current {
        let key = normalize_diag_key(diag);
        if let Some(rem) = budget_map.get_mut(&key)
            && *rem > 0
        {
            *rem -= 1;
            continue;
        }
        net_new.push(diag.clone());
    }
    net_new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruff_map_normalizes_verbatim_and_drive_case() {
        let json_out = serde_json::json!([
            {
                "filename": r"\\?\d:\cerebras\refit\src\mod_a.py",
                "code": "F401",
                "message": "unused import",
                "location": { "row": 3 }
            },
            {
                "filename": r"D:\cerebras\refit\pkg\sub\b.py",
                "code": "E501",
                "message": "line too long",
                "location": { "row": 12 }
            }
        ])
        .to_string();

        let mapped = ruff_map(&json_out, Path::new(r"D:\cerebras\refit"));
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].0, "src/mod_a.py");
        assert_eq!(mapped[0].1, "ruff F401 L3: unused import");
        assert_eq!(mapped[1].0, "pkg/sub/b.py");
        assert_eq!(mapped[1].1, "ruff E501 L12: line too long");
    }

    #[test]
    fn cold_lsp_diff_net_new_diagnostics() {
        let baseline = vec![
            "ruff app.py F401 L3: unused import".to_string(),
            "src/view.ts:10:5 error TS2322: type mismatch".to_string(),
        ];
        let current = vec![
            "ruff app.py F401 L7: unused import".to_string(),
            "ruff app.py F401 L15: unused import".to_string(),
            "src/view.ts:14:5 error TS2322: type mismatch".to_string(),
            "ruff app.py B006 L22: mutable default".to_string(),
        ];

        let diff = net_new_diagnostics(&baseline, &current);
        assert_eq!(
            diff,
            vec![
                "ruff app.py F401 L15: unused import",
                "ruff app.py B006 L22: mutable default",
            ]
        );
        assert_eq!(net_new_diagnostics(&[], &current), current);
        assert!(net_new_diagnostics(&baseline, &baseline).is_empty());

        assert_eq!(select_ts_checker(true, true).0, "tsrs");
        assert_eq!(select_ts_checker(false, true).0, "oxlint");
        assert_eq!(select_ts_checker(false, false).0, "tsc");

        assert_eq!(select_js_linter(true, true, true).2, "oxlint --type-aware");
        assert_eq!(select_js_linter(true, false, true).2, "oxlint");
        assert_eq!(select_js_linter(false, false, true).2, "biome lint");

        assert_eq!(select_rust_checker(true).0, "rust-glancer");
        assert_eq!(select_rust_checker(false).0, "cargo");

        let tight = rust_cargo_env_for(Path::new("/tmp/cargo-slot"), true);
        let has_var = |env: &[(String, String)], key: &str, val: &str| {
            env.iter().any(|(ek, ev)| ek == key && ev == val)
        };
        assert!(has_var(&tight, "CARGO_BUILD_JOBS", "1"));
        assert!(has_var(&tight, "CARGO_INCREMENTAL", "0"));
        assert!(has_var(&tight, "CARGO_PROFILE_DEV_CODEGEN_UNITS", "1"));

        let normal = rust_cargo_env_for(Path::new("/tmp/cargo-slot"), false);
        assert!(!has_var(&normal, "CARGO_BUILD_JOBS", "1"));
    }
}
