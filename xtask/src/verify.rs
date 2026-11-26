//! Pure-Rust S1-S14 cross-OS verification harness (`cargo xtask verify`).

use anyhow::{Context, Result, bail};
use clap::Args;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::fs::{self, FileTimes};
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};
use walkdir::WalkDir;

#[derive(Args)]
pub struct VerifyArgs {
    #[arg(long)]
    pub worktree: Option<PathBuf>,
    #[arg(long)]
    pub baseline_repo: Option<PathBuf>,
    #[arg(long)]
    pub bin: Option<PathBuf>,
    #[arg(long)]
    pub clones_dir: Option<PathBuf>,
    #[arg(long)]
    pub out_json: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    pub scenarios: Vec<String>,
    #[arg(long)]
    pub skip_build: bool,
    #[arg(long)]
    pub skip_linux: bool,
    #[arg(long)]
    pub skip_clones: bool,
    #[arg(long)]
    pub offline_synthetic: bool,
}

#[derive(Serialize)]
struct ScenarioOut {
    id: String,
    name: String,
    status: String,
    duration_ms: u64,
    details: String,
    metrics: Map<String, Value>,
    errors: Vec<String>,
}

struct CmdOut {
    code: i32,
    stdout: String,
    stderr: String,
    elapsed_ms: u64,
}

struct Ctx {
    wt: PathBuf,
    base_repo: Option<PathBuf>,
    bin: PathBuf,
    clones_dir: PathBuf,
    enabled: Option<HashSet<String>>,
    skip_build: bool,
    skip_linux: bool,
    skip_clones: bool,
    offline_synthetic: bool,
    results: Vec<ScenarioOut>,
}

impl Ctx {
    fn want(&self, id: &str) -> bool {
        self.enabled.as_ref().is_none_or(|s| s.contains(&id.to_ascii_uppercase()))
    }

    fn record(
        &mut self,
        id: &str,
        name: &str,
        started: Instant,
        metrics: Map<String, Value>,
        details: String,
        errors: Vec<String>,
    ) {
        let status = if errors.is_empty() { "PASS" } else { "FAIL" };
        let ms = started.elapsed().as_millis() as u64;
        println!("[{status:<4}] {id:<3} {name} ({ms} ms) :: {details}");
        for e in &errors {
            println!("       ERR: {e}");
        }
        self.results.push(ScenarioOut {
            id: id.into(),
            name: name.into(),
            status: status.into(),
            duration_ms: ms,
            details,
            metrics,
            errors,
        });
    }

    fn skip(&mut self, id: &str, name: &str, reason: &str) {
        println!("[SKIP] {id:<3} {name} :: {reason}");
        self.results.push(ScenarioOut {
            id: id.into(),
            name: name.into(),
            status: "SKIP".into(),
            duration_ms: 0,
            details: reason.into(),
            metrics: Map::new(),
            errors: Vec::new(),
        });
    }
}

fn run_cmd(
    prog: &str, args: &[&str], cwd: &Path, envs: &[(&str, &str)], stdin_txt: Option<&str>,
) -> Result<CmdOut> {
    let t0 = Instant::now();
    let mut cmd = Command::new(prog);
    cmd.args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stdin_txt.is_some() { cmd.stdin(Stdio::piped()); } else { cmd.stdin(Stdio::null()); }
    for (k, v) in envs { cmd.env(k, v); }
    let mut child = cmd.spawn().with_context(|| format!("spawn `{prog}`"))?;
    if let Some(txt) = stdin_txt
        && let Some(mut sin) = child.stdin.take()
    {
        let _ = sin.write_all(txt.as_bytes());
    }
    let out = child.wait_with_output()?;
    Ok(CmdOut {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        elapsed_ms: t0.elapsed().as_millis() as u64,
    })
}

fn parse_test_counts(txt: &str) -> (usize, usize, usize) {
    let (mut pass, mut fail, mut ign) = (0usize, 0usize, 0usize);
    for line in txt.lines() {
        let Some(rest) = line.split("test result:").nth(1) else { continue };
        let mut prev_num: Option<usize> = None;
        for tok in rest.split_whitespace() {
            let clean = tok.trim_end_matches([';', '.']);
            if let Ok(n) = clean.parse::<usize>() {
                prev_num = Some(n);
                continue;
            }
            if let Some(n) = prev_num.take() {
                match clean {
                    "passed" => pass += n,
                    "failed" => fail += n,
                    "ignored" => ign += n,
                    _ => {}
                }
            }
        }
    }
    (pass, fail, ign)
}

#[cfg(windows)]
fn to_wsl_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    if s.len() >= 2 && s.as_bytes()[1] == b':' {
        format!("/mnt/{}{}", (s.as_bytes()[0] as char).to_ascii_lowercase(), &s[2..])
    } else {
        s
    }
}

#[cfg(windows)]
fn run_wsl_bash(script: &str) -> Result<CmdOut> {
    let t0 = Instant::now();
    let mut child = Command::new("wsl.exe")
        .args(["-e", "bash", "-s"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn wsl.exe")?;
    if let Some(mut sin) = child.stdin.take() {
        sin.write_all(script.replace("\r\n", "\n").as_bytes())?;
    }
    let out = child.wait_with_output()?;
    Ok(CmdOut {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        elapsed_ms: t0.elapsed().as_millis() as u64,
    })
}

fn rs_files(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .map(|e| e.into_path())
        .collect()
}

fn s1_builds(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S1") { return Ok(()); }
    if ctx.skip_build { ctx.skip("S1", "5-Way Builds, Clippy & 0 #[allow(", "skipped via --skip-build"); return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    let checks: &[(&str, &[&str])] = &[
        ("host_default", &["clippy", "--all-targets", "--", "-D", "warnings"]),
        ("host_no_default", &["clippy", "--all-targets", "--no-default-features", "--", "-D", "warnings"]),
        ("host_full", &["clippy", "--all-targets", "--features", "full", "--", "-D", "warnings"]),
        ("xtask_clippy", &["clippy", "--package", "xtask", "--all-targets", "--", "-D", "warnings"]),
        ("host_release_full", &["build", "--release", "--features", "full"]),
    ];
    for &(lbl, args) in checks {
        let r = run_cmd("cargo", args, &ctx.wt, &[], None)?;
        m.insert(lbl.into(), json!(r.code));
        if r.code != 0 {
            let lines: Vec<&str> = r.stderr.lines().collect();
            let tail = &lines[lines.len().saturating_sub(6)..];
            errs.push(format!("{lbl} failed (exit={}): {}", r.code, tail.join(" | ")));
        }
    }
    let mut suppress = 0usize;
    for f in rs_files(&ctx.wt.join("src")) {
        let txt = fs::read_to_string(&f)?;
        suppress += txt.lines().filter(|l| l.contains("#[allow(") || l.contains("#[expect(")).count();
    }
    m.insert("suppress_count".into(), json!(suppress));
    if suppress != 0 { errs.push(format!("found {suppress} #[allow(/#[expect( in src/")); }

    #[cfg(windows)]
    if !ctx.skip_linux {
        let wsl_wt = to_wsl_path(&ctx.wt);
        let sh = format!(
            "set -euo pipefail\nexport PATH=\"$HOME/.cargo/bin:$PATH\"\nexport CARGO_TARGET_DIR=/tmp/refit-wsl-target\ncd \"{wsl_wt}\"\n\
             cargo clippy --all-targets --no-default-features --features alloc-jemalloc -- -D warnings\n\
             cargo clippy --all-targets --no-default-features --features \"alloc-jemalloc,full\" -- -D warnings\n"
        );
        let r = run_wsl_bash(&sh)?;
        m.insert("wsl_clippy_exit".into(), json!(r.code));
        if r.code != 0 { errs.push(format!("WSL2 clippy failed (exit={})", r.code)); }
    }
    ctx.record("S1", "5-Way Builds, Clippy & 0 #[allow(", t0, m, format!("suppress={suppress}, checks={}", checks.len()), errs);
    Ok(())
}

fn s2_tests(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S2") { return Ok(()); }
    if ctx.skip_build { ctx.skip("S2", "Full Test Suite (Host+Linux)", "skipped via --skip-build"); return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    let r = run_cmd("cargo", &["test", "--features", "full"], &ctx.wt, &[], None)?;
    let (p, f, i) = parse_test_counts(&format!("{}\n{}", r.stdout, r.stderr));
    m.insert("host_passed".into(), json!(p));
    m.insert("host_failed".into(), json!(f));
    m.insert("host_ignored".into(), json!(i));
    if r.code != 0 || f > 0 { errs.push(format!("host cargo test failed ({p}p/{f}f)")); }
    if p < 85 { errs.push(format!("host passed tests {p} < 85")); }
    if i != 0 { errs.push(format!("host ignored tests {i} != 0")); }

    #[cfg(windows)]
    if !ctx.skip_linux {
        let wsl_wt = to_wsl_path(&ctx.wt);
        let sh = format!(
            "set -euo pipefail\nexport PATH=\"$HOME/.cargo/bin:$PATH\"\nexport CARGO_TARGET_DIR=/tmp/refit-wsl-target\ncd \"{wsl_wt}\"\n\
             cargo test --no-default-features --features \"alloc-jemalloc,full\"\n"
        );
        let rw = run_wsl_bash(&sh)?;
        let (wp, wf, wi) = parse_test_counts(&format!("{}\n{}", rw.stdout, rw.stderr));
        m.insert("wsl_passed".into(), json!(wp));
        m.insert("wsl_failed".into(), json!(wf));
        if rw.code != 0 || wf > 0 || wi != 0 { errs.push(format!("WSL2 cargo test failed ({wp}p/{wf}f/{wi}i)")); }
    }
    ctx.record("S2", "Full Test Suite (Host+Linux)", t0, m, format!("host={p}p/{f}f"), errs);
    Ok(())
}

fn s3_self_audit(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S3") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    let files = rs_files(&ctx.wt.join("src"));
    let loc: usize = files
        .iter()
        .map(|f| fs::read_to_string(f).map_or(0, |s| s.lines().count()))
        .sum();
    m.insert("rs_files".into(), json!(files.len()));
    m.insert("total_rs_loc".into(), json!(loc));
    if !(46..=48).contains(&files.len()) { errs.push(format!("src/ .rs count {} outside 46..=48", files.len())); }
    if loc > 35_298 { errs.push(format!("src/ LOC {loc} > 35,298 ceiling")); }

    let bh = [("HTTP_PROXY", "http://127.0.0.1:1"), ("HTTPS_PROXY", "http://127.0.0.1:1")];
    let wt_s = ctx.wt.to_string_lossy().into_owned();
    let r = run_cmd(ctx.bin.to_str().unwrap(), &["audit", &wt_s], &ctx.wt, &bh, None)?;
    if r.code != 0 {
        errs.push(format!("refit audit exited {}: {}", r.code, r.stderr));
    } else {
        let v: Value = serde_json::from_str(&r.stdout)?;
        let aud = &v["audit"];
        let scc = aud["scc_count"].as_u64().unwrap_or(99);
        let clones = aud["clone_clusters"].as_array().map_or(99, Vec::len);
        let med = aud["hygiene_median"].as_f64().unwrap_or(0.0);
        let p10 = aud["hygiene_p10"].as_f64().unwrap_or(0.0);
        m.insert("scc_count".into(), json!(scc));
        m.insert("clone_clusters".into(), json!(clones));
        m.insert("hygiene_median".into(), json!(med));
        m.insert("hygiene_p10".into(), json!(p10));
        if scc != 0 { errs.push(format!("scc_count={scc} != 0")); }
        if clones != 0 { errs.push(format!("clone_clusters={clones} != 0")); }
        if med < 0.82 { errs.push(format!("hygiene_median={med:.4} < 0.82")); }
        if p10 < 0.80 { errs.push(format!("hygiene_p10={p10:.4} < 0.80")); }
    }
    ctx.record("S3", "refit Self-Audit", t0, m, format!("files={}, loc={loc}", files.len()), errs);
    Ok(())
}

fn s4_parity(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S4") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        let r = run_cmd("cargo", &["test", "--features", "full", "parity_vs_js_and_scalar"], &ctx.wt, &[], None)?;
        m.insert("unit_parity_exit".into(), json!(r.code));
        if r.code != 0 { errs.push("parity_vs_js_and_scalar failed".into()); }
    }
    if let Some(ref base) = ctx.base_repo
        && base.is_dir()
    {
        let had_refit = base.join(".refit").exists();
        let base_s = base.to_string_lossy().into_owned();
        let r = run_cmd(ctx.bin.to_str().unwrap(), &["scan", &base_s, "--no-cache"], &ctx.wt, &[], None)?;
        if !had_refit { let _ = fs::remove_dir_all(base.join(".refit")); }
        m.insert("baseline_scan_exit".into(), json!(r.code));
        if r.code != 0 { errs.push(format!("baseline scan failed: {}", r.stderr)); }
    }
    ctx.record("S4", "Canonical Neural Parity", t0, m, "worker.mjs + AVX2/scalar parity".into(), errs);
    Ok(())
}

fn s5_approx_long(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S5") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        for tn in ["sparse_far_clone_recall", "sparse_keeps_local_squiggles", "dense_qk_sparse_and_reuse"] {
            let r = run_cmd("cargo", &["test", "--features", "full", tn], &ctx.wt, &[], None)?;
            if r.code != 0 { errs.push(format!("test {tn} failed")); }
        }
    }
    let td = tempfile::tempdir()?;
    fs::create_dir_all(td.path().join("src"))?;
    let block_lines = [
        "pub fn evaluate_ledger_span(entries: &[u64], floor_val: u64, rate_ppm: u64) -> u64 {",
        "    let mut acc = entries.iter().copied().fold(0u64, u64::wrapping_add);",
        "    let scaled = acc.saturating_mul(23).saturating_add(floor_val);",
        "    let bounded = scaled.clamp(250, 250_000);",
        "    let blended = (bounded * 5) / 4 + entries.len() as u64;",
        "    let levy = (blended * rate_ppm) / 100_000;",
        "    let margin = blended.saturating_sub(levy + floor_val / 8);",
        "    let digest = margin.rotate_left(7) ^ 0x3C3C_5678;",
        "    let shard = (digest % 991).saturating_add(margin);",
        "    let buffer = (shard / 5).clamp(32, 8192);",
        "    let diff = shard.saturating_sub(buffer).wrapping_add(bounded);",
        "    let tag = (diff ^ digest).rotate_right(5);",
        "    acc = acc.wrapping_add(tag).saturating_sub(bounded);",
        "    acc ^ digest",
        "}",
    ];
    let block = format!("{}\n\n", block_lines.join("\n"));

    let mut big = String::with_capacity(48 * 1024);
    for i in 0..30 {
        let v = i * 7 + 3;
        big.push_str(&format!(
            "pub fn pre_{i}(x: u64) -> u64 {{\n    x.wrapping_add({v})\n}}\n"
        ));
    }
    big.push_str(&block);
    for i in 0..260 {
        let (mul, xor_k) = (i * 11 + 5, i * 13 + 9);
        big.push_str(&format!(
            "pub fn mid_{i}(a_{i}: u64) -> u64 {{\n    \
             let r_{i} = a_{i}.wrapping_mul({mul}) ^ {xor_k};\n    r_{i}\n}}\n"
        ));
    }
    big.push_str(&block.replace("evaluate_ledger_span", "evaluate_ledger_tail"));
    for i in 0..80 {
        let v = i * 5 + 2;
        big.push_str(&format!(
            "pub fn epi_{i}(z: u64) -> u64 {{\n    z.saturating_sub({v})\n}}\n"
        ));
    }
    fs::write(td.path().join("src/far_clone.rs"), big)?;

    let root = td.path().to_string_lossy().into_owned();
    let bin = ctx.bin.to_str().unwrap();
    let r1 = run_cmd(bin, &["audit", &root], td.path(), &[], None)?;
    let r2 = run_cmd(bin, &["audit", &root, "--approx-long"], td.path(), &[], None)?;
    let r3 = run_cmd(bin, &["audit", &root], td.path(), &[], None)?;
    if r1.code != 0 || r2.code != 0 || r3.code != 0 {
        errs.push("CLI --approx-long run failed".into());
    } else {
        let s1 = serde_json::from_str::<Value>(&r1.stdout)?["audit"]["hygiene_median"].as_f64().unwrap_or(0.0);
        let s2 = serde_json::from_str::<Value>(&r2.stdout)?["audit"]["hygiene_median"].as_f64().unwrap_or(0.0);
        let s3 = serde_json::from_str::<Value>(&r3.stdout)?["audit"]["hygiene_median"].as_f64().unwrap_or(0.0);
        m.insert("default_before".into(), json!(s1));
        m.insert("approx_long".into(), json!(s2));
        m.insert("default_after".into(), json!(s3));
        if (s1 - s3).abs() > 0.0 { errs.push(format!("cache contamination {s1} -> {s3}")); }
        if s1 - s2 < 0.08 { errs.push(format!("--approx-long drop too small: {s1} -> {s2}")); }
    }
    ctx.record("S5", "--approx-long Neural Verification", t0, m, "cache isolation + far-clone recall".into(), errs);
    Ok(())
}

fn s6_research(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S6") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        let r = run_cmd("cargo", &["test", "--features", "full", "crates_io_etag_304_and_hops"], &ctx.wt, &[], None)?;
        if r.code != 0 { errs.push("crates_io_etag_304_and_hops failed".into()); }
    }
    let td = tempfile::tempdir()?;
    fs::create_dir_all(td.path().join("src"))?;
    fs::write(td.path().join("src/lib.rs"), "pub fn ok(x: i32) -> i32 { x + 1 }\n")?;
    fs::write(td.path().join("Cargo.toml"), "[package]\nname=\"s6\"\nversion=\"0.1.0\"\nrust-version=\"1.100\"\n[dependencies]\nlazy_static=\"1.5.0\"\nserde_yaml=\"0.9.34\"\n")?;
    fs::write(td.path().join("Cargo.lock"), "version = 3\n\n[[package]]\nname = \"lazy_static\"\nversion = \"1.5.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n\n[[package]]\nname = \"serde_yaml\"\nversion = \"0.9.34\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n")?;
    let env = [("HTTP_PROXY", "http://127.0.0.1:1"), ("HTTPS_PROXY", "http://127.0.0.1:1"), ("KEENABLE_API_KEY", ""), ("EXA_API_KEY", ""), ("FIRECRAWL_API_KEY", "")];
    let root = td.path().to_string_lossy().into_owned();
    let bin = ctx.bin.to_str().unwrap();
    let roff = run_cmd(bin, &["audit", &root], td.path(), &env, None)?;
    let rres = run_cmd(bin, &["audit", &root, "--research"], td.path(), &env, None)?;
    if roff.code == 0 {
        let v: Value = serde_json::from_str(&roff.stdout)?;
        if v["audit"].get("dep_advisories").is_some() { errs.push("offline audit emitted dep_advisories".into()); }
    } else { errs.push("offline audit failed".into()); }
    if rres.code == 0 {
        let v: Value = serde_json::from_str(&rres.stdout)?;
        let advs = v["audit"]["dep_advisories"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        m.insert("advisories".into(), json!(advs.len()));
        if !advs.iter().any(|a| a["kind"] == "std_replacement" && a["name"] == "lazy_static") {
            errs.push("missing lazy_static advisory".into());
        }
        if !advs.iter().any(|a| a["kind"] == "rustsec_unmaintained" && a["name"] == "serde_yaml") {
            errs.push("missing serde_yaml advisory".into());
        }
    } else {
        errs.push("audit --research failed offline".into());
    }
    ctx.record("S6", "Offline vs --research Advisories", t0, m, "offline omission + RustSec/std advisories".into(), errs);
    Ok(())
}

fn s7_storage(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S7") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        for tn in ["sst_zstd_lazy_and_threshold", "stat_bin_migrates_stale"] {
            let r = run_cmd("cargo", &["test", "--features", "full", tn], &ctx.wt, &[], None)?;
            if r.code != 0 { errs.push(format!("{tn} failed")); }
        }
    }
    let td = tempfile::tempdir()?;
    fs::create_dir_all(td.path().join("src"))?;
    for i in 0..12 { fs::write(td.path().join(format!("src/m_{i}.rs")), format!("pub fn f_{i}(x: i32) -> i32 {{ x + {i} }}\n"))?; }
    let root = td.path().to_string_lossy().into_owned();
    let bin = ctx.bin.to_str().unwrap();
    let r1 = run_cmd(bin, &["scan", &root, "--no-cache"], td.path(), &[], None)?;
    let cache = td.path().join(".refit/cache");
    if r1.code != 0 || !cache.join("symbol_index.zst").exists() || cache.join("sym.sst").exists() {
        errs.push("small repo did not write symbol_index.zst exclusively".into());
    }
    let r2 = run_cmd(bin, &["scan", &root, "--no-cache"], td.path(), &[("REFIT_FORCE_SST", "1")], None)?;
    let sst = cache.join("sym.sst");
    let magic = fs::read(&sst)
        .ok()
        .and_then(|b| b.get(..8).map(|s| String::from_utf8_lossy(s).into_owned()))
        .unwrap_or_default();
    m.insert("sst_magic".into(), json!(magic));
    if r2.code != 0 || magic != "RFST0004" || cache.join("symbol_index.zst").exists() {
        errs.push(format!("forced SST check failed (magic={magic})"));
    }
    ctx.record("S7", "Adaptive Storage (zst vs RFST0004)", t0, m, format!("magic={magic}"), errs);
    Ok(())
}

fn s8_router(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S8") { return Ok(()); }
    if ctx.skip_build { ctx.skip("S8", "Search Router Failover & Race", "skipped via --skip-build"); return Ok(()); }
    let t0 = Instant::now();
    let mut errs = Vec::new();
    for tn in ["router_failover_and_race", "router_429_aimd_and_bound"] {
        let r = run_cmd("cargo", &["test", "--features", "full", tn], &ctx.wt, &[], None)?;
        if r.code != 0 { errs.push(format!("{tn} failed")); }
    }
    ctx.record("S8", "Search Router Failover & Race", t0, Map::new(), "432 failover + 429 AIMD".into(), errs);
    Ok(())
}

fn s9_gc(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S9") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        let r = run_cmd("cargo", &["test", "--features", "full", "gates::stage::"], &ctx.wt, &[], None)?;
        if r.code != 0 { errs.push("gates::stage:: failed".into()); }
    }
    let td = tempfile::tempdir()?;
    let rust_cache = td.path().join(".refit/cache/rust");
    fs::create_dir_all(&rust_cache)?;
    let tag = td.path().join(".refit/cache/CACHEDIR.TAG");
    fs::write(&tag, "Signature: 8a477f597d28d172789f06886806bc55")?;
    let blob = vec![0u8; 512 * 1024];
    let now = SystemTime::now();
    for i in 0..5u64 {
        let p = rust_cache.join(format!("blob_{i}.bin"));
        fs::write(&p, &blob)?;
        let f = fs::OpenOptions::new().write(true).open(&p)?;
        let mtime = now.checked_sub(Duration::from_secs((10 - i) * 60)).unwrap_or(now);
        let _ = f.set_times(FileTimes::new().set_modified(mtime));
    }
    let root = td.path().to_string_lossy().into_owned();
    let r = run_cmd(ctx.bin.to_str().unwrap(), &["gc", &root, "--cache-max-mb", "1"], td.path(), &[], None)?;
    let mut rem = 0u64;
    for entry in WalkDir::new(&rust_cache).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() {
            rem += entry.metadata().map_or(0, |md| md.len());
        }
    }
    m.insert("remaining_bytes".into(), json!(rem));
    m.insert("tag_ok".into(), json!(tag.exists()));
    if r.code != 0 || !tag.exists() || rem > 1024 * 1024 {
        errs.push(format!("refit gc failed: exit={} tag={} rem={rem}", r.code, tag.exists()));
    }
    ctx.record("S9", "Build Cache & refit gc", t0, m, format!("rem={rem}B, tag={}", tag.exists()), errs);
    Ok(())
}

fn s10_mcp(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S10") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    if !ctx.skip_build {
        for flt in ["mcp::", "epoch_single_publish_on_commit", "learn::"] {
            let r = run_cmd("cargo", &["test", "--features", "full", flt], &ctx.wt, &[], None)?;
            if r.code != 0 { errs.push(format!("test {flt} failed")); }
        }
    }
    let td = tempfile::tempdir()?;
    fs::create_dir_all(td.path().join("src"))?;
    fs::write(td.path().join("src/a.rs"), "pub fn legacy_parse(v: i32) -> i32 { v + 1 }\n")?;
    fs::write(td.path().join("src/b.rs"), "use crate::a::legacy_parse;\npub fn call_parse(v: i32) -> i32 { legacy_parse(v) }\n")?;
    fs::write(td.path().join("src/c.rs"), "use crate::b::call_parse;\npub fn entry(v: i32) -> i32 { call_parse(v) }\n")?;
    let root = td.path().to_string_lossy().into_owned();
    let bin = ctx.bin.to_str().unwrap();

    let msgs = concat!(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2026-07-28\",\"capabilities\":{},\"clientInfo\":{\"name\":\"v\",\"version\":\"1\"}}}\n",
        "[{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}]\n",
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\",\"params\":{\"_meta\":{\"protocolVersion\":\"2026-07-28\"}}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"_meta\":{\"protocolVersion\":\"2026-07-28\"},\"params\":{\"name\":\"refit_scan\",\"arguments\":{\"fast\":true}}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"_meta\":{\"protocolVersion\":\"2026-07-28\"},\"params\":{\"name\":\"refit_scan\",\"arguments\":{\"fast\":true}}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"_meta\":{\"protocolVersion\":\"2026-07-28\"},\"params\":{\"name\":\"refit_read\",\"arguments\":{\"path\":\"src/a.rs\",\"start_line\":1,\"end_line\":5}}}\n"
    );
    let r = run_cmd(bin, &["mcp", &root], td.path(), &[], Some(msgs))?;
    let vals: Vec<Value> = r.stdout
        .lines()
        .filter_map(|l| serde_json::from_str(l.trim()).ok())
        .collect();
    m.insert("stdio_responses".into(), json!(vals.len()));
    if r.code != 0 || vals.len() < 6 { errs.push(format!("MCP stdio failed ({} responses)", vals.len())); }
    if !vals.iter().any(|v| v["error"]["code"].as_i64() == Some(-32600)) { errs.push("missing -32600 batch rejection".into()); }
    if let Some(t) = vals.iter().find(|v| v["id"] == 3) {
        let n = t["result"]["tools"].as_array().map_or(0, Vec::len);
        m.insert("stdio_tools".into(), json!(n));
        if n != 12 { errs.push(format!("expected 12 MCP tools, got {n}")); }
    }
    if let Some(s2) = vals.iter().find(|v| v["id"] == 5) {
        let txt = s2["result"]["content"][0]["text"].as_str().unwrap_or("{}");
        let inner: Value = serde_json::from_str(txt).unwrap_or_default();
        if inner["epoch_cache_hit"].as_bool() != Some(true) { errs.push("2nd refit_scan missed epoch_cache_hit".into()); }
    }
    let lrn = run_cmd(bin, &["learn", &root, "--rewind", "legacy_parse", "--trigger", "advisory"], td.path(), &[], None)?;
    if lrn.code != 0 || !lrn.stdout.contains("src/b.rs") { errs.push("learn --rewind missed src/b.rs".into()); }

    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut srv = Command::new(bin)
        .args(["mcp", &root, "--http", &format!("127.0.0.1:{port}")])
        .current_dir(td.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(3)).build()?;
    let mut http_ok = false;
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(120));
        if let Ok(resp) = client.post(format!("http://127.0.0.1:{port}/mcp"))
            .header("MCP-Protocol-Version", "2026-07-28")
            .header("Mcp-Method", "tools/list")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"protocolVersion":"2026-07-28"}}}))
            .send()
            && let Ok(v) = resp.json::<Value>()
            && v["result"]["tools"].as_array().is_some_and(|a| a.len() == 12)
        {
            http_ok = true;
            break;
        }
    }
    let _ = srv.kill();
    let _ = srv.wait();
    m.insert("http_ok".into(), json!(http_ok));
    if !http_ok { errs.push(format!("HTTP /mcp probe failed on port {port}")); }
    ctx.record("S10", "MCP 2026-07-28 Stdio+HTTP & Rewind", t0, m, format!("stdio={}, http={http_ok}", vals.len()), errs);
    Ok(())
}

fn s11_resilience(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S11") { return Ok(()); }
    if ctx.skip_build { ctx.skip("S11", "Crash & Security Resilience", "skipped via --skip-build"); return Ok(()); }
    let t0 = Instant::now();
    let mut errs = Vec::new();
    for flt in ["wal_crash_points_7_of_7", "mvcc::", "platform::paths::"] {
        let r = run_cmd("cargo", &["test", "--features", "full", flt], &ctx.wt, &[], None)?;
        if r.code != 0 { errs.push(format!("{flt} failed")); }
    }
    ctx.record("S11", "Crash & Security Resilience", t0, Map::new(), "WAL 7/7 + paths".into(), errs);
    Ok(())
}

fn s12_clones(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S12") { return Ok(()); }
    if ctx.skip_clones { ctx.skip("S12", "Real Cloned-Repo Matrix", "skipped via --skip-clones"); return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    let specs: &[(&str, bool, &str, &str)] = &[
        ("uv", true, "Resolver", "crates/uv-resolver/src/lib.rs"),
        ("hono", false, "Hono", "src/hono.ts"),
        ("gin", false, "Context", "context.go"),
    ];
    let bh = [("HTTP_PROXY", "http://127.0.0.1:1"), ("HTTPS_PROXY", "http://127.0.0.1:1"), ("KEENABLE_API_KEY", ""), ("EXA_API_KEY", ""), ("FIRECRAWL_API_KEY", "")];
    let bin = ctx.bin.to_str().unwrap();
    let mut tested = 0usize;
    for &(name, expect_sst, token, sample) in specs {
        let dir = ctx.clones_dir.join(name);
        if ctx.offline_synthetic || !dir.join(".git").exists() { continue; }
        tested += 1;
        let _ = fs::remove_dir_all(dir.join(".refit"));
        let ds = dir.to_string_lossy().into_owned();
        let r_scan = run_cmd(bin, &["scan", &ds, "--no-cache"], &dir, &bh, None)?;
        let has_sst = dir.join(".refit/cache/sym.sst").exists();
        let has_zst = dir.join(".refit/cache/symbol_index.zst").exists();
        if r_scan.code != 0 || (expect_sst && (!has_sst || has_zst)) || (!expect_sst && (has_sst || !has_zst)) {
            errs.push(format!("[{name}] scan/storage mismatch exit={} sst={has_sst} zst={has_zst}", r_scan.code));
        }
        let r_aud = run_cmd(bin, &["audit", &ds], &dir, &bh, None)?;
        let r_app = run_cmd(bin, &["audit", &ds, "--approx-long"], &dir, &bh, None)?;
        let r_sus = run_cmd(bin, &["sustain", &ds, "--dry-run"], &dir, &bh, None)?;
        let r_lrn = run_cmd(bin, &["learn", &ds, "--rewind", token, "--trigger", "advisory"], &dir, &bh, None)?;
        let r_gc = run_cmd(bin, &["gc", &ds, "--cache-max-mb", "600"], &dir, &[], None)?;
        let mcp_in = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"2026-07-28\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":\"s12\",\"version\":\"1\"}}}}}}\n\
             {{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"name\":\"refit_symbols\",\"arguments\":{{\"file\":\"{sample}\"}}}}}}\n"
        );
        let r_mcp = run_cmd(bin, &["mcp", &ds], &dir, &[], Some(&mcp_in))?;
        for (lbl, c) in [("audit", r_aud.code), ("approx", r_app.code), ("sustain", r_sus.code), ("learn", r_lrn.code), ("gc", r_gc.code), ("mcp", r_mcp.code)] {
            if c != 0 { errs.push(format!("[{name}] {lbl} failed (exit={c})")); }
        }
        m.insert(name.into(), json!({"scan_ms": r_scan.elapsed_ms, "audit_ms": r_aud.elapsed_ms, "has_sst": has_sst}));
        let _ = fs::remove_dir_all(dir.join(".refit"));
    }
    if tested == 0 {
        ctx.skip("S12", "Real Cloned-Repo Matrix", "no local clones found in clones_dir");
        return Ok(());
    }
    ctx.record("S12", "Real Cloned-Repo Matrix (uv/hono/gin)", t0, m, format!("repos={tested}"), errs);
    Ok(())
}

fn s13_linux_bench(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S13") { return Ok(()); }
    if ctx.skip_linux { ctx.skip("S13", "Linux 1-vCPU Perf & Peak RSS", "skipped via --skip-linux"); return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());

    let bench_body = |wt_path: &str| format!(
        r#"set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/refit-wsl-target
export REFIT_MAX_CPUS=1
export REFIT_BUDGET_MB=512
cd "{wt_path}"
BENCH_REPO=/tmp/refit-bench-repo
rm -rf "$BENCH_REPO" && mkdir -p "$BENCH_REPO"
cp -a Cargo.toml Cargo.lock src "$BENCH_REPO/"
git -C "$BENCH_REPO" init -q -b main && git -C "$BENCH_REPO" config user.email b@t && git -C "$BENCH_REPO" config user.name B
git -C "$BENCH_REPO" add -A && git -C "$BENCH_REPO" commit -q -m init
cargo build --release --no-default-features --features alloc-jemalloc >/dev/null 2>&1 && cp "$CARGO_TARGET_DIR/release/refit" /tmp/refit-jem
cargo build --release >/dev/null 2>&1 && cp "$CARGO_TARGET_DIR/release/refit" /tmp/refit-mim
cargo build --release --no-default-features --features "alloc-jemalloc,full" >/dev/null 2>&1 && cp "$CARGO_TARGET_DIR/release/refit" /tmp/refit-jem-full
measure_ms() {{
  local times=() i t0 t1
  for i in 1 2 3 4 5; do
    t0=$(date +%s%N)
    taskset -c 0 "$@" >/dev/null 2>&1
    t1=$(date +%s%N)
    times+=("$(( (t1 - t0) / 1000 ))")
  done
  printf '%s\n' "${{times[@]}}" | sort -n | sed -n '3p' | awk '{{printf "%.2f", $1/1000.0}}'
}}
measure() {{
  local lbl="$1" bin="$2"
  rm -rf "$BENCH_REPO/.refit"
  local scan_kb aud_kb mcp_kb aud_ms sus_ms
  taskset -c 0 /usr/bin/time -f "%M" -o /tmp/refit-rss.tmp "$bin" scan "$BENCH_REPO" --fast --no-cache >/dev/null 2>&1
  scan_kb=$(cat /tmp/refit-rss.tmp)
  rm -rf "$BENCH_REPO/.refit"
  taskset -c 0 /usr/bin/time -f "%M" -o /tmp/refit-rss.tmp "$bin" audit "$BENCH_REPO" >/dev/null 2>&1
  aud_kb=$(cat /tmp/refit-rss.tmp)
  taskset -c 0 "$bin" audit "$BENCH_REPO" >/dev/null 2>&1
  aud_ms=$(measure_ms "$bin" audit "$BENCH_REPO")
  sus_ms=$(measure_ms "$bin" sustain "$BENCH_REPO" --dry-run)
  printf '%s\n' \
    '{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2026-07-28","capabilities":{{}},"clientInfo":{{"name":"b","version":"1"}}}}}}' \
    '{{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{{"_meta":{{"protocolVersion":"2026-07-28"}}}}}}' \
    '{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"refit_scan","arguments":{{"fast":true}}}}}}' \
    '{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"refit_symbols","arguments":{{"file":"src/main.rs"}}}}}}' \
    '{{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{{"name":"refit_read","arguments":{{"path":"src/main.rs","start_line":1,"end_line":40}}}}}}' \
    | taskset -c 0 /usr/bin/time -f "%M" -o /tmp/refit-rss.tmp "$bin" mcp "$BENCH_REPO" >/dev/null 2>&1
  mcp_kb=$(cat /tmp/refit-rss.tmp)
  echo "METRIC:${{lbl}}:${{aud_ms}}:${{sus_ms}}:${{aud_kb}}:${{scan_kb}}:${{mcp_kb}}"
}}
measure jem /tmp/refit-jem
measure mim /tmp/refit-mim
measure jem_full /tmp/refit-jem-full
rm -rf "$BENCH_REPO" /tmp/refit-rss.tmp /tmp/refit-jem /tmp/refit-mim /tmp/refit-jem-full
"#
    );

    #[cfg(windows)]
    let r = run_wsl_bash(&bench_body(&to_wsl_path(&ctx.wt)))?;
    #[cfg(target_os = "linux")]
    let r = run_cmd("bash", &["-c", &bench_body(&ctx.wt.to_string_lossy())], &ctx.wt, &[], None)?;
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        ctx.skip("S13", "Linux 1-vCPU Perf & Peak RSS", "taskset 1-vCPU bench runs on Linux/WSL2");
        return Ok(());
    }

    #[cfg(any(windows, target_os = "linux"))]
    {
        if r.code != 0 {
            errs.push(format!("S13 bench failed (exit={}): {}", r.code, r.stderr));
        } else {
            let mut count = 0usize;
            for line in r.stdout.lines().filter(|l| l.starts_with("METRIC:")) {
                let p: Vec<&str> = line.trim().split(':').collect();
                if p.len() < 7 { continue; }
                count += 1;
                let lbl = p[1];
                let aud_ms = p[2].parse::<f64>().unwrap_or(999.0);
                let sus_ms = p[3].parse::<f64>().unwrap_or(999.0);
                let (aud_mb, scan_mb, mcp_mb) = (
                    p[4].parse::<f64>().unwrap_or(99999.0) / 1024.0,
                    p[5].parse::<f64>().unwrap_or(99999.0) / 1024.0,
                    p[6].parse::<f64>().unwrap_or(99999.0) / 1024.0,
                );
                m.insert(lbl.into(), json!({"audit_warm_ms": aud_ms, "sustain_ms": sus_ms, "audit_cold_mib": aud_mb, "scan_cold_mib": scan_mb, "mcp_mib": mcp_mb}));
                let (max_aud, max_scan, max_mcp) = if lbl == "mim" { (60.0, 64.0, 42.0) } else { (34.0, 34.0, 24.0) };
                if aud_ms > 75.0 { errs.push(format!("[{lbl}] audit_warm {aud_ms:.2}ms > 75ms")); }
                if sus_ms > 36.0 { errs.push(format!("[{lbl}] sustain {sus_ms:.2}ms > 36ms")); }
                if aud_mb > max_aud { errs.push(format!("[{lbl}] audit_cold {aud_mb:.2}MiB > {max_aud}MiB")); }
                if scan_mb > max_scan { errs.push(format!("[{lbl}] scan_cold {scan_mb:.2}MiB > {max_scan}MiB")); }
                if mcp_mb > max_mcp { errs.push(format!("[{lbl}] mcp {mcp_mb:.2}MiB > {max_mcp}MiB")); }
            }
            if count < 3 { errs.push(format!("expected 3 variant metrics, got {count}")); }
        }
        ctx.record("S13", "Linux 1-vCPU Perf & Peak RSS", t0, m, "jem / mim / jem_full measured".into(), errs);
        Ok(())
    }
}

fn has_dep_key(toml_txt: &str, name: &str) -> bool {
    let sp = format!("{name} ");
    let eq = format!("{name}=");
    toml_txt.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with(&sp) || t.starts_with(&eq)
    })
}

fn s14_guardrails(ctx: &mut Ctx) -> Result<()> {
    if !ctx.want("S14") { return Ok(()); }
    let t0 = Instant::now();
    let (mut errs, mut m) = (Vec::new(), Map::new());
    let toml_txt = fs::read_to_string(ctx.wt.join("Cargo.toml"))?;
    let lock_txt = fs::read_to_string(ctx.wt.join("Cargo.lock"))?;

    for banned in ["roaring", "lz4_flex", "ringbuf", "empryo"] {
        let hit = toml_txt.to_ascii_lowercase().contains(banned) || lock_txt.to_ascii_lowercase().contains(banned);
        if hit { errs.push(format!("banned crate `{banned}` found in Cargo manifests")); }
    }
    for no_direct in ["icu_normalizer", "unicode-ident", "base64", "heck"] {
        if has_dep_key(&toml_txt, no_direct) {
            errs.push(format!("disallowed direct dep `{no_direct}` in Cargo.toml"));
        }
    }
    for req in ["semver", "reflink-copy"] {
        if !has_dep_key(&toml_txt, req) {
            errs.push(format!("required dep `{req}` missing from Cargo.toml"));
        }
    }
    for dead in ["mcp-v2", "neural-v6", "full-v7"] {
        if toml_txt.lines().any(|l| l.trim_start().starts_with(&format!("{dead} "))) {
            errs.push(format!("retired feature `{dead}` still present"));
        }
    }
    // Single-version active graph check across all 3 desktop/server targets!
    let mut dup_total = 0usize;
    for target in ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu", "aarch64-apple-darwin"] {
        let rdup = run_cmd(
            "cargo",
            &["tree", "--locked", "--features", "full", "--target", target, "-e", "normal", "--duplicates"],
            &ctx.wt,
            &[],
            None,
        )?;
        let dup_lines: Vec<&str> = rdup.stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        dup_total += dup_lines.len();
        if !dup_lines.is_empty() {
            errs.push(format!("[{target}] active multi-version duplicates: {}", dup_lines.join(" | ")));
        }
    }
    m.insert("active_duplicate_lines".into(), json!(dup_total));
    ctx.record("S14", "Dependency, License & 0-Duplicate Guardrails", t0, m, format!("dup_lines={dup_total}"), errs);
    Ok(())
}

pub fn run(default_root: &Path, args: VerifyArgs) -> Result<()> {
    let wt = args.worktree.unwrap_or_else(|| default_root.to_path_buf());
    let exe_name = if cfg!(windows) { "refit.exe" } else { "refit" };
    let bin = args.bin.unwrap_or_else(|| {
        let base = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| wt.join("target"));
        base.join("release").join(exe_name)
    });
    let clones_dir = args.clones_dir.unwrap_or_else(|| {
        let parent = wt.parent().and_then(Path::parent).unwrap_or(default_root);
        parent.join("_test_clones")
    });
    let enabled = if args.scenarios.is_empty() {
        None
    } else {
        let set = args.scenarios
            .into_iter()
            .map(|s| s.trim().to_ascii_uppercase())
            .collect();
        Some(set)
    };
    let mut ctx = Ctx {
        wt,
        base_repo: args.baseline_repo,
        bin,
        clones_dir,
        enabled,
        skip_build: args.skip_build,
        skip_linux: args.skip_linux,
        skip_clones: args.skip_clones,
        offline_synthetic: args.offline_synthetic,
        results: Vec::new(),
    };

    println!("=== refit Cross-OS Verification Harness (S1-S14) ===");
    let steps: &[fn(&mut Ctx) -> Result<()>] = &[
        s1_builds, s2_tests, s3_self_audit, s4_parity, s5_approx_long, s6_research, s7_storage,
        s8_router, s9_gc, s10_mcp, s11_resilience, s12_clones, s13_linux_bench, s14_guardrails,
    ];
    for step in steps {
        step(&mut ctx)?;
    }

    let passed = ctx.results.iter().filter(|r| r.status == "PASS").count();
    let failed = ctx.results.iter().filter(|r| r.status == "FAIL").count();
    let skipped = ctx.results.iter().filter(|r| r.status == "SKIP").count();
    if let Some(out_path) = args.out_json {
        if let Some(p) = out_path.parent() { fs::create_dir_all(p)?; }
        let rep = json!({
            "summary": {"total": ctx.results.len(), "passed": passed, "failed": failed, "skipped": skipped, "overall": if failed == 0 { "PASS" } else { "FAIL" }},
            "scenarios": ctx.results,
        });
        fs::write(&out_path, serde_json::to_string_pretty(&rep)?)?;
    }
    println!("\nTOTAL: {} | PASS: {passed} | FAIL: {failed} | SKIP: {skipped}", ctx.results.len());
    if failed > 0 { bail!("{failed} verification scenario(s) failed"); }
    Ok(())
}
