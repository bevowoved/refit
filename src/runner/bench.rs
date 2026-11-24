use crate::platform::{run_cmd, sandbox_env};
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

/// Run alternating OS benchmarks using a command template with `{repo}` substitution.
pub fn bench_command(
    cmd_template: &str,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchReport> {
    let env = sandbox_env(&[]);

    bench_pairs(before_repo, after_repo, pairs, |repo| {
        let expanded = cmd_template.replace("{repo}", &repo.display().to_string());
        let (prog, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
        run_cmd(prog, &[flag.into(), expanded], repo, &env, true, 30_000)
            .with_context(|| format!("run generic bench on {}", repo.display()))
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
    let mut rev = stdout.lines().rev();
    let Some(line) = rev.find(|l| l.trim().starts_with('{')) else { return Ok(()) };
    let Ok(v) = serde_json::from_str::<Value>(line) else { return Ok(()) };
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
