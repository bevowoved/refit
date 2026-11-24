use crate::platform::run_cmd;
use crate::types::{GateVerdict, SandboxResult};
use anyhow::Result;
use std::fs;
use std::path::Path;
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
}
