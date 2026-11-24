use crate::platform::{pwsh_utf8_preamble, run_cmd, sandbox_env, strip_utf8_bom, trim_crlf};
use crate::source::quantile;
use crate::types::{BenchReport, SandboxResult};
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

pub fn bench_python(
    python_bin: &str,
    bench_script: &Path,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchReport> {
    let env = sandbox_env(&[]);
    let script = bench_script.display().to_string();

    bench_pairs(before_repo, after_repo, pairs, |repo| {
        let args = vec![
            script.clone(),
            "--repo".into(),
            repo.display().to_string(),
            "--iterations".into(),
            "120".into(),
        ];
        run_cmd(python_bin, &args, repo, &env, true, 30_000)
            .with_context(|| format!("run bench on {}", repo.display()))
    })
}

fn find_unquoted_shell_meta(cmd: &str) -> Option<char> {
    let (mut in_single, mut in_double, mut escaped) = (false, false, false);
    for ch in cmd.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && !in_single {
            escaped = true;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if !in_single && !in_double && matches!(ch, '&' | '|' | '^' | '%' | ';' | '`') {
            return Some(ch);
        }
    }
    None
}

fn is_pwsh_exe(exe: &str) -> bool {
    let base = exe.rsplit(['/', '\\']).next().unwrap_or(exe);
    base.eq_ignore_ascii_case("pwsh")
        || base.eq_ignore_ascii_case("pwsh.exe")
        || base.eq_ignore_ascii_case("powershell")
        || base.eq_ignore_ascii_case("powershell.exe")
}

fn is_blocked_shell_exe(exe: &str) -> bool {
    let lower = exe.to_ascii_lowercase();
    let base = lower.rsplit(['/', '\\']).next().unwrap_or(&lower);
    base == "cmd"
        || base == "cmd.exe"
        || base.ends_with(".bat")
        || base.ends_with(".cmd")
        || (lower.contains("windowsapps") && lower.contains("microsoft.powershell_"))
}

/// Tokenize and validate a benchmark command template directly via `shlex::split`
/// without `cmd /C` (`CVE-2024-24576` BatBadBut mitigation).
pub fn parse_bench_cmd(cmd_template: &str, repo: &Path) -> Result<(String, Vec<String>)> {
    if cmd_template.bytes().any(|b| matches!(b, 0 | b'\r' | b'\n')) {
        anyhow::bail!("bench --cmd rejected: contains NUL or CR/LF control byte");
    }
    if let Some(meta) = find_unquoted_shell_meta(cmd_template) {
        anyhow::bail!(
            "bench --cmd rejected: unquoted shell metacharacter `{meta}` is disallowed (CVE-2024-24576)"
        );
    }

    let tokens = shlex::split(cmd_template)
        .ok_or_else(|| anyhow::anyhow!("bench --cmd rejected: unbalanced quotes or invalid escapes"))?;
    let Some((first, rest)) = tokens.split_first() else {
        anyhow::bail!("bench --cmd rejected: empty command template");
    };

    let repo_str = dunce::simplified(repo).display().to_string();
    let exe = first.replace("{repo}", &repo_str);
    if exe.is_empty() || is_blocked_shell_exe(&exe) {
        anyhow::bail!(
            "bench --cmd rejected: disallowed batch/shell executable `{exe}` (CVE-2024-24576)"
        );
    }

    let mut args: Vec<String> = rest.iter().map(|t| t.replace("{repo}", &repo_str)).collect();

    // When an explicit `pwsh` hook is invoked with `-Command`/`-c`, prepend BOM-less UTF-8
    // console encoding so headless Windows pipes do not lock to OEM ibm437/CP936.
    if is_pwsh_exe(&exe) {
        let mut i = 0usize;
        while i + 1 < args.len() {
            if args[i].eq_ignore_ascii_case("-command") || args[i].eq_ignore_ascii_case("-c") {
                let preamble = pwsh_utf8_preamble();
                if !args[i + 1].contains("UTF8Encoding") {
                    args[i + 1] = format!("{preamble}{}", args[i + 1]);
                }
                break;
            }
            i += 1;
        }
    }

    Ok((exe, args))
}

/// Run alternating OS benchmarks using a direct tokenized command template with `{repo}` substitution.
pub fn bench_command(
    cmd_template: &str,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchReport> {
    let env = sandbox_env(&[]);

    bench_pairs(before_repo, after_repo, pairs, |repo| {
        let (exe, args) = parse_bench_cmd(cmd_template, repo)?;
        run_cmd(&exe, &args, repo, &env, true, 30_000)
            .with_context(|| format!("run generic bench `{exe}` on {}", repo.display()))
    })
}

#[derive(Default)]
struct Samples {
    cpu: Vec<f64>,
    wall: Vec<f64>,
    rss: Vec<f64>,
}

fn bench_pairs(
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
    mut exec: impl FnMut(&Path) -> Result<SandboxResult>,
) -> Result<BenchReport> {
    let (mut before, mut after) = (Samples::default(), Samples::default());
    let mut checksum: Option<String> = None;

    for _ in 0..pairs {
        for (is_after, repo) in [(false, before_repo), (true, after_repo)] {
            let res = exec(repo)?;
            if res.exit_code != 0 {
                anyhow::bail!("bench failed on {}: {}", repo.display(), res.stderr);
            }
            verify_digest(&res.stdout, &mut checksum)?;

            let target = if is_after { &mut after } else { &mut before };
            target.cpu.push(if res.cpu_ms > 0 { res.cpu_ms } else { res.wall_ms } as f64);
            target.wall.push(res.wall_ms as f64);
            target.rss.push((res.peak_rss_bytes as f64) / (1024.0 * 1024.0));
        }
    }

    let b_cpu = quantile(&mut before.cpu, 0.5).max(1.0);
    let a_cpu = quantile(&mut after.cpu, 0.5).max(1.0);
    let b_rss = quantile(&mut before.rss, 0.5).max(1.0);
    let a_rss = quantile(&mut after.rss, 0.5).max(1.0);
    let (cpu_ratio, rss_ratio) = (a_cpu / b_cpu, a_rss / b_rss);

    Ok(BenchReport {
        pairs,
        before_cpu_ms_median: b_cpu,
        after_cpu_ms_median: a_cpu,
        cpu_ratio,
        before_wall_ms_median: quantile(&mut before.wall, 0.5).max(1.0),
        after_wall_ms_median: quantile(&mut after.wall, 0.5).max(1.0),
        before_rss_mb_median: b_rss,
        after_rss_mb_median: a_rss,
        rss_ratio,
        passed_anti_regression: cpu_ratio <= 1.05 && rss_ratio <= 1.05,
    })
}

fn verify_digest(stdout: &str, expected: &mut Option<String>) -> Result<()> {
    let cleaned = strip_utf8_bom(stdout);
    let mut rev = cleaned.lines().rev().map(trim_crlf);
    let Some(line) = rev.find(|l| l.trim().starts_with('{')) else { return Ok(()) };
    let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { return Ok(()) };
    let Some(chk) = v.get("checksum").map(Value::to_string) else { return Ok(()) };

    if let Some(prev) = expected {
        if prev != &chk {
            anyhow::bail!("Benchmark checksum mismatch (Q7): before={prev} after={chk}");
        }
    } else {
        *expected = Some(chk);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn bench_shlex_direct_and_cve_2024_24576_guard() {
        let repo = Path::new(r"D:\My Repos\proj");

        // Quoted JSON printf command (bench_smoke) parses cleanly without `cmd /C`.
        let (exe, args) = parse_bench_cmd(r#"printf "{\"checksum\":\"gate2-ok\"}\n""#, repo).unwrap();
        assert_eq!(exe, "printf");
        assert_eq!(args, vec![r#"{"checksum":"gate2-ok"}\n"#]);

        // `{repo}` substitution happens per token AFTER shlex::split so Windows backslashes & spaces stay intact.
        let (exe2, args2) = parse_bench_cmd("python3 bench.py --repo {repo} --flag", repo).unwrap();
        assert_eq!(exe2, "python3");
        assert_eq!(args2, vec!["bench.py", "--repo", r"D:\My Repos\proj", "--flag"]);

        // Quoted semicolons inside Python `-c` are allowed; unquoted shell metachars are rejected.
        let (py, py_args) =
            parse_bench_cmd(r#"python3 -c "import json; print(json.dumps({'checksum':'ok'}))""#, repo)
                .unwrap();
        assert_eq!(py, "python3");
        assert_eq!(py_args.len(), 2);

        // Explicit pwsh hook prepends BOM-less UTF-8 console encoding preamble.
        let (pwsh, pwsh_args) =
            parse_bench_cmd(r#"pwsh -NoProfile -Command "Write-Output ok""#, repo).unwrap();
        assert_eq!(pwsh, "pwsh");
        assert!(pwsh_args[2].starts_with(pwsh_utf8_preamble()));

        // CVE-2024-24576 / shell injection vectors MUST be rejected.
        for bad in [
            "cargo & calc.exe",
            "ruff | sh",
            "echo %PATH%",
            "tool^inject",
            "echo ok; rm -rf /",
            "echo `whoami`",
            "echo\0evil",
            "echo\r\nevil",
            "run.bat --repo {repo}",
            "script.CMD arg",
            "cmd.exe /C dir",
            r#"C:\Users\x\AppData\Local\Microsoft\WindowsApps\Microsoft.PowerShell_8wekyb3d8bbwe\pwsh.exe -c ok"#,
            "unterminated \"quote",
            "   ",
        ] {
            assert!(parse_bench_cmd(bad, repo).is_err(), "must reject `{bad}`");
        }

        // Digest verification handles UTF-8 BOM and CRLF.
        let mut chk = None;
        verify_digest("\u{FEFF}{\"checksum\":\"gate2-ok\"}\r\n", &mut chk).unwrap();
        assert_eq!(chk.as_deref(), Some("\"gate2-ok\""));
        verify_digest("{\"checksum\":\"gate2-ok\"}\n", &mut chk).unwrap();
        assert!(verify_digest("{\"checksum\":\"drift\"}\n", &mut chk).is_err());

        // End-to-end direct command execution (no `cmd /C` or `sh -c`).
        let a = TempRepo::new("bench_before");
        let b = TempRepo::new("bench_after");
        let rep = bench_command("git --version", a.path(), b.path(), 1).unwrap();
        assert_eq!(rep.pairs, 1);
    }
}
