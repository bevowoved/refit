use crate::platform::{budget, run_cmd, sandbox_env};
use crate::types::{GateVerdict, SandboxResult};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
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
    let found = toml::from_str::<toml::Value>(&src).ok().and_then(|v| {
        let pkg = v
            .get("package")
            .and_then(|p| p.get("edition"))
            .and_then(toml::Value::as_str);
        let ws = v
            .get("workspace")
            .and_then(|w| w.get("package"))
            .and_then(|p| p.get("edition"))
            .and_then(toml::Value::as_str);
        pkg.or(ws).map(str::to_string)
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

fn map_ruff_json<T>(
    stdout: &str,
    mut f: impl FnMut(&str, &str, u64, &str) -> T,
) -> Vec<T> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    arr.iter()
        .map(|v| {
            f(
                v["filename"].as_str().unwrap_or(""),
                v["code"].as_str().unwrap_or("RUFF"),
                v["location"]["row"].as_u64().unwrap_or(0),
                v["message"].as_str().unwrap_or(""),
            )
        })
        .collect()
}

pub(super) fn parse_ruff(stdout: &str) -> Vec<String> {
    map_ruff_json(stdout, |raw, code, line, msg| {
        let fname = raw.rsplit(['/', '\\']).next().unwrap_or("");
        format!("ruff {fname} {code} L{line}: {msg}")
    })
}

pub(super) fn ruff_map(stdout: &str, repo_root: &Path) -> Vec<(String, String)> {
    map_ruff_json(stdout, |raw, code, line, msg| {
        let rel = crate::platform::rel_to_root(repo_root, Path::new(raw))
            .unwrap_or_else(|| raw.replace('\\', "/"));
        (rel, format!("ruff {code} L{line}: {msg}"))
    })
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
    first_pair(s, "", max)
}

pub(super) fn first_pair(a: &str, b: &str, max: usize) -> String {
    a.lines()
        .chain(b.lines())
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect::<Vec<_>>()
        .join(" | ")
}

pub(super) fn find_tool_in_dirs(dirs: &[PathBuf], bin: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) { &[".exe", ".cmd", ".bat", ""] } else { &[""] };
    dirs.iter().find_map(|dir| {
        exts.iter().find_map(|ext| {
            let p = if ext.is_empty() { dir.join(bin) } else { dir.join(format!("{bin}{ext}")) };
            p.is_file().then_some(p)
        })
    })
}

pub(super) fn find_tool_on_path(bin: &str) -> Option<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(&std::env::var_os("PATH")?).collect();
    find_tool_in_dirs(&dirs, bin)
}

pub(super) fn tool_on_path(bin: &str) -> bool {
    find_tool_on_path(bin).is_some()
}

pub(super) fn oxlint_supports_type_aware() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        tool_on_path("oxlint")
            && run_cmd("oxlint", &["--help".into()], &std::env::temp_dir(), &sandbox_env(&[]), true, 3_000)
                .is_ok_and(|o| o.stdout.contains("--type-aware") || o.stderr.contains("--type-aware"))
    })
}

pub(super) fn select_ts_checker(
    has_tsrs: bool,
    has_oxlint_type_aware: bool,
) -> (&'static str, Vec<String>, &'static str) {
    if has_tsrs {
        ("tsrs", Vec::new(), "tsrs")
    } else if has_oxlint_type_aware {
        ("oxlint", vec!["--type-aware".into(), "--type-check".into()], "oxlint --type-aware")
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

pub(super) fn select_rust_checker(has_rg: bool) -> (&'static str, Vec<String>, &'static str) {
    if has_rg {
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
        for (k, v) in [
            ("CARGO_BUILD_JOBS", "1"),
            ("CARGO_INCREMENTAL", "0"),
            ("CARGO_PROFILE_DEV_CODEGEN_UNITS", "1"),
        ] {
            extra.push((k.into(), v.into()));
        }
    }
    sandbox_env(&extra)
}

pub(super) fn merged_rust_cargo_env(
    target_dir: &Path,
    extra: Vec<(String, String)>,
) -> Vec<(String, String)> {
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

fn normalize_diag_token(tok: &str) -> String {
    let is_num = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if tok.trim_end_matches(':').strip_prefix('L').is_some_and(is_num) {
        return String::new();
    }
    let mut parts = tok.split(':');
    let head = parts.next().unwrap_or(tok);
    if parts.next().is_some_and(is_num) && parts.next().is_none_or(|s| s.is_empty() || is_num(s)) {
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
pub fn net_new_diagnostics(baseline: &[String], current: &[String]) -> Vec<String> {
    if baseline.is_empty() {
        return current.to_vec();
    }
    let mut budget_map: HashMap<String, usize> = HashMap::with_capacity(baseline.len());
    for diag in baseline {
        *budget_map.entry(normalize_diag_key(diag)).or_insert(0) += 1;
    }
    current
        .iter()
        .filter(|diag| {
            let key = normalize_diag_key(diag);
            match budget_map.get_mut(&key) {
                Some(rem) if *rem > 0 => {
                    *rem -= 1;
                    false
                }
                _ => true,
            }
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
        ])
        .to_string();

        let mapped = ruff_map(&json_out, Path::new(r"D:\cerebras\refit"));
        let expected = [
            ("src/mod_a.py", "ruff F401 L3: unused import"),
            ("pkg/sub/b.py", "ruff E501 L12: line too long"),
        ];
        assert_eq!(mapped.len(), expected.len());
        for ((got_p, got_m), (exp_p, exp_m)) in mapped.iter().zip(expected) {
            assert_eq!((got_p.as_str(), got_m.as_str()), (exp_p, exp_m));
        }
    }

    #[test]
    fn cold_lsp_diff_net_new_diagnostics() {
        let baseline: Vec<String> = [
            "ruff app.py F401 L3: unused import",
            "src/view.ts:10:5 error TS2322: type mismatch",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let current: Vec<String> = [
            "ruff app.py F401 L7: unused import",
            "ruff app.py F401 L15: unused import",
            "src/view.ts:14:5 error TS2322: type mismatch",
            "ruff app.py B006 L22: mutable default",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        assert_eq!(
            net_new_diagnostics(&baseline, &current),
            vec![
                "ruff app.py F401 L15: unused import",
                "ruff app.py B006 L22: mutable default",
            ]
        );
        assert_eq!(net_new_diagnostics(&[], &current), current);
        assert!(net_new_diagnostics(&baseline, &baseline).is_empty());

        for (tsrs, ox_ta, exp) in [(true, true, "tsrs"), (false, true, "oxlint"), (false, false, "tsc")] {
            assert_eq!(select_ts_checker(tsrs, ox_ta).0, exp);
        }
        for (ox, ta, ts, exp) in [
            (true, true, true, "oxlint --type-aware"),
            (true, false, true, "oxlint"),
            (false, false, true, "biome lint"),
        ] {
            assert_eq!(select_js_linter(ox, ta, ts).2, exp);
        }
        for (rg, exp) in [(true, "rust-glancer"), (false, "cargo")] {
            assert_eq!(select_rust_checker(rg).0, exp);
        }

        let has_var = |env: &[(String, String)], k: &str, v: &str| {
            env.iter().any(|(ek, ev)| ek == k && ev == v)
        };
        let tight = rust_cargo_env_for(Path::new("/tmp/cargo-slot"), true);
        for (k, v) in [
            ("CARGO_BUILD_JOBS", "1"),
            ("CARGO_INCREMENTAL", "0"),
            ("CARGO_PROFILE_DEV_CODEGEN_UNITS", "1"),
        ] {
            assert!(has_var(&tight, k, v));
        }
        let normal = rust_cargo_env_for(Path::new("/tmp/cargo-slot"), false);
        assert!(!has_var(&normal, "CARGO_BUILD_JOBS", "1"));
    }
}
