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

const BOOTSTRAP_RESAMPLES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0xC0DE_5EED_2026_0004;

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

fn quantile_nearest(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((vals.len() as f64 - 1.0) * q).round() as usize;
    vals[idx.min(vals.len() - 1)]
}

/// Deterministic bootstrap 95% confidence interval `(median, ci_lo, ci_hi)` over paired CPU ratios.
pub fn bootstrap_ratio_ci(ratios: &[f64], resamples: usize) -> (f64, f64, f64) {
    if ratios.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let mut base = ratios.to_vec();
    let median = quantile_nearest(&mut base, 0.5);
    let mut rng = XorShift64::new(BOOTSTRAP_SEED);
    let mut boot = Vec::with_capacity(resamples);
    let mut sample = vec![0.0f64; ratios.len()];

    for _ in 0..resamples {
        for slot in &mut sample {
            *slot = ratios[rng.next_usize(ratios.len())];
        }
        boot.push(quantile_nearest(&mut sample, 0.5));
    }
    let lo = quantile_nearest(&mut boot, 0.025);
    let hi = quantile_nearest(&mut boot, 0.975);
    (median, lo, hi)
}

fn assign_wilcoxon_ranks(sorted_nz: &[f64]) -> Vec<f64> {
    let count = sorted_nz.len();
    let mut ranks = vec![0.0f64; count];
    let mut i = 0usize;
    while i < count {
        let mut j = i + 1;
        while j < count && (sorted_nz[j].abs() - sorted_nz[i].abs()).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        for slot in &mut ranks[i..j] {
            *slot = avg_rank;
        }
        i = j;
    }
    ranks
}

fn exact_wilcoxon_small(ranks: &[f64], w_min: f64) -> f64 {
    let count = ranks.len();
    let total = 1usize << count;
    let mut le_count = 0usize;
    for mask in 0..total {
        let mut sum = 0.0f64;
        for (bit, &rank) in ranks.iter().enumerate() {
            if ((mask >> bit) & 1) != 0 {
                sum += rank;
            }
        }
        if sum <= w_min + 1e-9 {
            le_count += 1;
        }
    }
    ((2.0 * (le_count as f64)) / (total as f64)).min(1.0)
}

fn normal_approx_wilcoxon(count: usize, w_min: f64) -> f64 {
    let mu = (count * (count + 1)) as f64 / 4.0;
    let sigma = ((count * (count + 1) * (2 * count + 1)) as f64 / 24.0).sqrt();
    let z_score = ((w_min - mu).abs() - 0.5).max(0.0) / sigma.max(1e-9);
    let t_val = 1.0 / (1.0 + 0.2316419 * z_score);
    let poly = t_val
        * (0.319381530
            + t_val
                * (-0.356563782
                    + t_val * (1.781477937 + t_val * (-1.821255978 + t_val * 1.330274429))));
    let pdf = (-0.5 * z_score * z_score).exp() / (2.0 * std::f64::consts::PI).sqrt();
    (2.0 * pdf * poly).clamp(0.0, 1.0)
}

/// Exact two-sided Wilcoxon signed-rank `(W_min, p_value)` for `m <= 16`, normal approx above.
pub fn exact_wilcoxon_p(diffs: &[f64]) -> (f64, f64) {
    let mut nz: Vec<f64> = diffs.iter().copied().filter(|d| d.abs() > 1e-12).collect();
    if nz.is_empty() {
        return (0.0, 1.0);
    }
    nz.sort_by(|lhs, rhs| {
        lhs.abs()
            .partial_cmp(&rhs.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ranks = assign_wilcoxon_ranks(&nz);
    let (mut w_pos, mut w_neg) = (0.0f64, 0.0f64);
    for (idx, &diff) in nz.iter().enumerate() {
        if diff > 0.0 {
            w_pos += ranks[idx];
        } else {
            w_neg += ranks[idx];
        }
    }
    let w_min = w_pos.min(w_neg);
    let p_val = if nz.len() <= 16 {
        exact_wilcoxon_small(&ranks, w_min)
    } else {
        normal_approx_wilcoxon(nz.len(), w_min)
    };
    (w_min, p_val)
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
    let use_abba = pairs >= 4;

    for pair_idx in 0..pairs {
        let order = if use_abba && pair_idx % 2 == 1 {
            [(true, after_repo), (false, before_repo)]
        } else {
            [(false, before_repo), (true, after_repo)]
        };
        for (is_after, repo) in order {
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

    let paired_ratios: Vec<f64> = before
        .cpu
        .iter()
        .zip(&after.cpu)
        .map(|(&b_val, &a_val)| a_val.max(1.0) / b_val.max(1.0))
        .collect();
    let paired_diffs: Vec<f64> = before
        .cpu
        .iter()
        .zip(&after.cpu)
        .map(|(&b_val, &a_val)| a_val - b_val)
        .collect();
    let (cpu_ratio_ci_lo, cpu_ratio_ci_hi, wilcoxon_p) = if paired_ratios.is_empty() {
        (None, None, None)
    } else {
        let (_, lo, hi) = bootstrap_ratio_ci(&paired_ratios, BOOTSTRAP_RESAMPLES);
        let (_, p_val) = exact_wilcoxon_p(&paired_diffs);
        (Some(lo), Some(hi), Some(p_val))
    };

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
        cpu_ratio_ci_lo,
        cpu_ratio_ci_hi,
        wilcoxon_p,
        abba_mode: Some(use_abba),
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
        assert_eq!(rep.abba_mode, Some(false));
    }

    #[test]
    fn bench_abba_bootstrap_ci_and_exact_wilcoxon() {
        let before = Path::new("before_repo");
        let after = Path::new("after_repo");

        // 1. Verify ABBA interleaved schedule for pairs >= 4 (6 pairs => 3 exact ABBA quartets).
        let mut order: Vec<&'static str> = Vec::new();
        let mut step = 0u64;
        let drift_rep = bench_pairs(before, after, 6, |repo| {
            let tag = if repo == before { "A" } else { "B" };
            order.push(tag);
            step += 1;
            // Identical baseline + linear thermal drift (+10 ms/step): ABBA cancels drift!
            let cpu = 1_000 + step * 10;
            Ok(SandboxResult {
                exit_code: 0,
                timed_out: false,
                stdout: "{\"checksum\":\"same\"}\n".into(),
                stderr: String::new(),
                wall_ms: cpu,
                cpu_ms: cpu,
                peak_rss_bytes: 32 * 1024 * 1024,
            })
        })
        .unwrap();
        assert_eq!(
            order,
            vec!["A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B", "A"]
        );
        assert_eq!(drift_rep.abba_mode, Some(true));
        assert!((drift_rep.cpu_ratio - 1.0).abs() <= 0.02, "ratio={}", drift_rep.cpu_ratio);
        assert_eq!(drift_rep.wilcoxon_p, Some(1.0), "ABBA must cancel symmetric linear drift");
        assert!(drift_rep.cpu_ratio_ci_lo.unwrap() <= 1.0 && drift_rep.cpu_ratio_ci_hi.unwrap() >= 1.0);

        // 2. Pairs < 4 preserves classic AB order for compatibility.
        let mut short_order: Vec<&'static str> = Vec::new();
        let short_rep = bench_pairs(before, after, 2, |repo| {
            short_order.push(if repo == before { "A" } else { "B" });
            Ok(SandboxResult {
                exit_code: 0,
                timed_out: false,
                stdout: String::new(),
                stderr: String::new(),
                wall_ms: 50,
                cpu_ms: 50,
                peak_rss_bytes: 16 * 1024 * 1024,
            })
        })
        .unwrap();
        assert_eq!(short_order, vec!["A", "B", "A", "B"]);
        assert_eq!(short_rep.abba_mode, Some(false));

        // 3. Clear 1.5x slowdown over 8 ABBA pairs -> tight deterministic Bootstrap 95% CI & exact p.
        let slow_rep = bench_pairs(before, after, 8, |repo| {
            let cpu = if repo == before { 100 } else { 150 };
            Ok(SandboxResult {
                exit_code: 0,
                timed_out: false,
                stdout: String::new(),
                stderr: String::new(),
                wall_ms: cpu,
                cpu_ms: cpu,
                peak_rss_bytes: 16 * 1024 * 1024,
            })
        })
        .unwrap();
        assert!(!slow_rep.passed_anti_regression);
        assert_eq!(slow_rep.cpu_ratio_ci_lo, Some(1.5));
        assert_eq!(slow_rep.cpu_ratio_ci_hi, Some(1.5));
        assert!((slow_rep.wilcoxon_p.unwrap() - (2.0 / 256.0)).abs() < 1e-12);

        // 4. Deterministic Bootstrap CI reproducibility + exact Wilcoxon textbook vectors.
        let ratios = [0.95, 1.02, 0.98, 1.04, 1.01, 0.97, 1.03, 1.00];
        let b1 = bootstrap_ratio_ci(&ratios, 2_000);
        let b2 = bootstrap_ratio_ci(&ratios, 2_000);
        assert_eq!(b1, b2, "bootstrap CI must be bit-deterministic");
        assert!(b1.1 <= b1.0 && b1.0 <= b1.2);

        let (w6, p6) = exact_wilcoxon_p(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(w6, 0.0);
        assert!((p6 - 0.03125).abs() < 1e-12, "exact m=6 p={p6}");
        let (_, p_sym) = exact_wilcoxon_p(&[1.0, -1.0, 2.0, -2.0]);
        assert_eq!(p_sym, 1.0);
        assert_eq!(exact_wilcoxon_p(&[0.0, 0.0]), (0.0, 1.0));
        let large_pos: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        assert!(exact_wilcoxon_p(&large_pos).1 < 0.001);

        // 5. Additive serde backward compatibility (legacy BenchReport JSON without optional fields).
        let legacy_json = r#"{
            "pairs": 3, "before_cpu_ms_median": 10.0, "after_cpu_ms_median": 10.0,
            "cpu_ratio": 1.0, "before_rss_mb_median": 20.0, "after_rss_mb_median": 20.0,
            "rss_ratio": 1.0, "passed_anti_regression": true
        }"#;
        let parsed: BenchReport = serde_json::from_str(legacy_json).unwrap();
        assert_eq!((parsed.cpu_ratio_ci_lo, parsed.wilcoxon_p, parsed.abba_mode), (None, None, None));
    }
}
