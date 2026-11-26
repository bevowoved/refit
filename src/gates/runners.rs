use super::stage::{CargoSlot, py_worktree_cache_env};
use crate::platform::{
    SecretTier, budget, env_trimmed, find_secret_tiered, new_http, now_ms, read_git_head_fast,
    read_zip_entry_capped, redact_secrets, run_cmd, sandbox_env, write_atomic,
};
use crate::types::{GateVerdict, SandboxResult};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
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

// ---------------------------------------------------------------------------
// Gate 4: Capability-Negotiated, Upgrade-Proof TestMu AI Kane CLI Engine (V4)
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct KaneCaps: u16 {
        const BUG_DETECTION      = 1 << 0; // >= 0.5.0
        const REMOTE_GRID        = 1 << 1; // >= 0.8.4
        const NO_ADAPTIVE_HEAL   = 1 << 2; // >= 0.8.5 (retires --retry)
        const V1_STREAM_TEE      = 1 << 3; // >= 0.8.17
        const TESTRUN_AGENT_FLAG = 1 << 4; // >= 0.8.21
        const REAL_DEVICE_TARGET = 1 << 5; // >= 0.9.0
        const POLICY_AND_EXPLORE = 1 << 6; // >= 0.10.0
        const REAL_DEV_TUNNEL    = 1 << 7; // >= 0.10.3
        const BUGS_MEMORY        = 1 << 8; // >= 0.10.7
        const RED_AGENT_INFO     = 1 << 9; // >= 0.10.8
    }
}

/// Single-source-of-truth SARIF 2.1.0 & LearnStore contract table for Gate 4 Kane findings.
pub const KANE_SARIF_RULES: &[(&str, &str, &str)] = &[
    ("kane/step_failure",         "error",   "gate4_kane_step"),
    ("kane/product_bug_740",      "error",   "gate4_kane_bug_740"),
    ("kane/policy_fail",          "error",   "gate4_kane_policy"),
    ("kane/policy_errored",       "warning", "gate4_kane_policy_err"),
    ("kane/unresolved_vars",      "warning", "gate4_kane_vars"),
    ("kane/doc_lie",              "error",   "gate4_kane_doc_lie"),
    ("kane/evidence_secret_leak", "error",   "gate4_kane_secret"),
    ("kane/schema_version_drift", "note",    "gate4_kane_schema"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KaneSurface { Run, TestMd, TestRun, Cover, BugsList, EvidenceValidate, Explore }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KaneRequest {
    pub spec: String,
    pub surface: Option<KaneSurface>,
    pub url: Option<String>,
    pub policy: Option<String>,
    pub target: Option<String>,
    pub app: Option<String>,
    pub tunnel_name: Option<String>,
    pub remote: bool,
    pub headless: bool,
    pub bug_detection: bool,
    pub no_adaptive_heal: bool,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedStep {
    pub step: u64,
    #[serde(skip_serializing_if = "Option::is_none")] pub heading: Option<String>,
    pub remark: String,
    #[serde(default)] pub is_agent_info: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyVerdict {
    pub result: String,
    pub exit_code: i32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub failed_gate_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub errored_gate_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSpanHint { pub file: String, pub line: usize, pub symbol: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaneReceipt {
    pub ok: bool,
    pub verdict: String,
    pub surface: KaneSurface,
    pub exit_code: i32,
    #[serde(skip_serializing_if = "Option::is_none")] pub cli_version: Option<String>,
    pub duration_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")] pub credits_consumed: Option<f64>,
    pub replay_steps: u64,
    pub author_steps: u64,
    pub adaptive_healed: bool,
    pub confirmed_bug_740: bool,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub test_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub evidence_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub evidence_serve_cmd: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub failed_steps: Vec<FailedStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub evidence_excerpts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub evidence_secret_leaks: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub unresolved_vars: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub policy_verdict: Option<PolicyVerdict>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub final_state: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub suspected_spans: Vec<CodeSpanHint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub rewind_affected_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")] pub advisories: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub git_head: Option<String>,
    pub recorded_ms: u64,
}

struct KaneRegexes { ansi: Regex, evid_bt: Regex, evid_bare: Regex, semver: Regex }

static KANE_RE: LazyLock<KaneRegexes> = LazyLock::new(|| KaneRegexes {
    ansi: Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").expect("ansi re"),
    evid_bt: Regex::new(r"`kane-cli evidence serve ([^`\r\n]+)`").expect("evid bt re"),
    evid_bare: Regex::new(r"kane-cli evidence serve ([^\r\n`]+?\.evidence)").expect("evid re"),
    semver: Regex::new(r"([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)").expect("ver re"),
});

fn val_code_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn caps_for_version(ver_raw: &str) -> KaneCaps {
    let Some(m) = KANE_RE.semver.captures(ver_raw).and_then(|c| c.get(1)) else {
        return KaneCaps::all();
    };
    let Ok(v) = semver::Version::parse(m.as_str()) else { return KaneCaps::all(); };
    let core = (v.major, v.minor, v.patch);
    let mut c = KaneCaps::empty();
    if core >= (0, 5, 0)  { c |= KaneCaps::BUG_DETECTION; }
    if core >= (0, 8, 4)  { c |= KaneCaps::REMOTE_GRID; }
    if core >= (0, 8, 5)  { c |= KaneCaps::NO_ADAPTIVE_HEAL; }
    if core >= (0, 8, 17) { c |= KaneCaps::V1_STREAM_TEE; }
    if core >= (0, 8, 21) { c |= KaneCaps::TESTRUN_AGENT_FLAG; }
    if core >= (0, 9, 0)  { c |= KaneCaps::REAL_DEVICE_TARGET; }
    if core >= (0, 10, 0) { c |= KaneCaps::POLICY_AND_EXPLORE; }
    if core >= (0, 10, 3) { c |= KaneCaps::REAL_DEV_TUNNEL; }
    if core >= (0, 10, 7) { c |= KaneCaps::BUGS_MEMORY; }
    if core >= (0, 10, 8) { c |= KaneCaps::RED_AGENT_INFO; }
    c
}

pub fn resolve_kane_bin() -> String {
    if let Some(ov) = env_trimmed("REFIT_KANE_BIN") { return ov; }
    #[cfg(windows)]
    {
        if let Some(appdata) = env_trimmed("APPDATA") {
            let npm_cmd = PathBuf::from(appdata).join("npm").join("kane-cli.cmd");
            if npm_cmd.is_file() { return npm_cmd.display().to_string(); }
        }
    }
    "kane-cli".into()
}

pub fn probe_kane_caps(bin: &str) -> (Option<String>, KaneCaps) {
    static CACHED: OnceLock<(Option<String>, KaneCaps)> = OnceLock::new();
    let probe = |prog: &str| -> (Option<String>, KaneCaps) {
        let env = [("NO_COLOR".into(), "1".into())];
        if let Ok(out) = run_cmd(prog, &["--version".into()], &std::env::temp_dir(), &env, false, 4_000)
            && out.exit_code == 0
        {
            let text = format!("{} {}", out.stdout, out.stderr);
            if let Some(m) = KANE_RE.semver.captures(&text).and_then(|c| c.get(1)) {
                let s = m.as_str().to_string();
                let caps = caps_for_version(&s);
                return (Some(s), caps);
            }
        }
        (None, KaneCaps::all())
    };
    if env_trimmed("REFIT_KANE_BIN").is_some() { return probe(bin); }
    CACHED.get_or_init(|| probe(bin)).clone()
}

pub fn infer_kane_surface(spec: &str, explicit: Option<KaneSurface>) -> KaneSurface {
    if let Some(s) = explicit { return s; }
    let t = spec.trim();
    if t.eq_ignore_ascii_case("cover") { return KaneSurface::Cover; }
    if t.eq_ignore_ascii_case("bugs") || t.starts_with("bugs:") { return KaneSurface::BugsList; }
    if t.ends_with(".evidence") || t.starts_with("validate:") { return KaneSurface::EvidenceValidate; }
    if t.starts_with("explore:") { return KaneSurface::Explore; }
    if t.starts_with("tags:") || t.starts_with("match:") || t.eq_ignore_ascii_case("from-context") {
        return KaneSurface::TestRun;
    }
    if t.ends_with("_test.md") || t.ends_with(".test.md") { return KaneSurface::TestMd; }
    KaneSurface::Run
}

fn push_opt_flag(args: &mut Vec<String>, flag: &str, val: Option<&str>) {
    if let Some(v) = val.filter(|s| !s.is_empty()) {
        args.extend([flag.into(), v.into()]);
    }
}

fn push_testrun_args(args: &mut Vec<String>, spec: &str, req: &KaneRequest, caps: KaneCaps) {
    args.extend(["testrun".into(), "run".into()]);
    if let Some(tag) = spec.strip_prefix("tags:") {
        args.extend(["--tags".into(), tag.into()]);
    } else if let Some(pat) = spec.strip_prefix("match:") {
        args.extend(["--match".into(), pat.into()]);
    } else if spec.eq_ignore_ascii_case("from-context") {
        args.push("--from-context".into());
    } else {
        args.push(spec.into());
    }
    if caps.contains(KaneCaps::TESTRUN_AGENT_FLAG) { args.push("--agent".into()); }
    if req.remote && caps.contains(KaneCaps::REMOTE_GRID) {
        args.push("--remote".into());
    } else if req.headless {
        args.push("--headless".into());
    }
    if caps.contains(KaneCaps::POLICY_AND_EXPLORE) { push_opt_flag(args, "--policy", req.policy.as_deref()); }
}

fn push_device_flags(args: &mut Vec<String>, req: &KaneRequest, caps: KaneCaps) {
    let is_real_dev = req.target.as_deref().is_some_and(|t| {
        t.eq_ignore_ascii_case("android") || t.eq_ignore_ascii_case("ios")
    });
    if !is_real_dev || caps.contains(KaneCaps::REAL_DEVICE_TARGET) {
        push_opt_flag(args, "--target", req.target.as_deref());
    }
    push_opt_flag(args, "--app", req.app.as_deref());
    if is_real_dev && caps.contains(KaneCaps::REAL_DEV_TUNNEL) {
        push_opt_flag(args, "--tunnel-name", req.tunnel_name.as_deref());
    }
}

pub fn build_kane_args(req: &KaneRequest, caps: KaneCaps) -> (KaneSurface, Vec<String>) {
    let surface = infer_kane_surface(&req.spec, req.surface);
    let mut args: Vec<String> = Vec::new();
    let spec = req.spec.trim();
    match surface {
        KaneSurface::Run => {
            args.extend(["run".into(), spec.into(), "--agent".into()]);
            if req.headless { args.push("--headless".into()); }
            push_opt_flag(&mut args, "--url", req.url.as_deref());
            if req.timeout_secs > 0 { args.extend(["--timeout".into(), req.timeout_secs.to_string()]); }
            if req.bug_detection && caps.contains(KaneCaps::BUG_DETECTION) { args.push("--bug-detection".into()); }
            if req.no_adaptive_heal && caps.contains(KaneCaps::NO_ADAPTIVE_HEAL) { args.push("--no-adaptive-heal".into()); }
        }
        KaneSurface::TestMd => {
            args.extend(["testmd".into(), "run".into(), spec.into(), "--agent".into(), "--on-lock-conflict".into(), "wait".into()]);
            if req.headless { args.push("--headless".into()); }
            push_opt_flag(&mut args, "--url", req.url.as_deref());
            if caps.contains(KaneCaps::POLICY_AND_EXPLORE) { push_opt_flag(&mut args, "--policy", req.policy.as_deref()); }
            if req.timeout_secs > 0 { args.extend(["--timeout".into(), req.timeout_secs.to_string()]); }
            if req.no_adaptive_heal && caps.contains(KaneCaps::NO_ADAPTIVE_HEAL) { args.push("--no-adaptive-heal".into()); }
        }
        KaneSurface::TestRun => push_testrun_args(&mut args, spec, req, caps),
        KaneSurface::Cover => args.extend(["cover".into(), "--mode".into(), "agent".into()]),
        KaneSurface::BugsList => {
            args.extend(["bugs".into(), "list".into()]);
            if let Some(f) = spec.strip_prefix("bugs:").filter(|s| !s.is_empty()) { args.push(f.into()); }
            args.push("--agent".into());
        }
        KaneSurface::EvidenceValidate => {
            let p = spec.strip_prefix("validate:").unwrap_or(spec);
            args.extend(["evidence".into(), "validate".into(), p.into(), "--json".into()]);
        }
        KaneSurface::Explore => {
            let target_url = spec.strip_prefix("explore:").filter(|s| !s.is_empty())
                .or(req.url.as_deref()).unwrap_or(spec);
            args.extend(["explore".into(), "--url".into(), target_url.into(), "--agent".into()]);
        }
    }
    if matches!(surface, KaneSurface::Run | KaneSurface::TestMd | KaneSurface::TestRun) {
        push_device_flags(&mut args, req, caps);
    }
    (surface, args)
}

pub async fn probe_url_reachable(url: &str) -> std::result::Result<u16, String> {
    let client = new_http("refit-kane-preflight", 3, 2).map_err(|e| e.to_string())?;
    let resp = match client.head(url).send().await {
        Ok(r) if matches!(r.status().as_u16(), 405 | 501) => {
            client.get(url).send().await.map_err(|e| e.to_string())?
        }
        Ok(r) => r,
        Err(_) => client.get(url).send().await.map_err(|e| e.to_string())?,
    };
    let code = resp.status().as_u16();
    if code >= 500 { Err(format!("start URL `{url}` returned HTTP {code}")) } else { Ok(code) }
}

fn extract_evidence_path(
    stderr: &str, session_dir: Option<&str>, evidence_id: Option<&str>, cwd: &Path,
) -> Option<PathBuf> {
    let clean = KANE_RE.ansi.replace_all(stderr, "");
    if let Some(m) = KANE_RE.evid_bt.captures(&clean).and_then(|c| c.get(1)) {
        let p = PathBuf::from(m.as_str().trim());
        return Some(if p.is_absolute() { p } else { cwd.join(p) });
    }
    if let Some(m) = KANE_RE.evid_bare.captures(&clean).and_then(|c| c.get(1)) {
        let p = PathBuf::from(m.as_str().trim());
        return Some(if p.is_absolute() { p } else { cwd.join(p) });
    }
    if let Some(eid) = evidence_id {
        let p = cwd.join(".testmuai/evidence").join(format!("{eid}.evidence"));
        if p.is_file() { return Some(p); }
    }
    if let Some(sd) = session_dir {
        let dir = Path::new(sd).join("evidence");
        if let Ok(rd) = fs::read_dir(dir) {
            for ent in rd.flatten() {
                if ent.path().extension().and_then(|e| e.to_str()) == Some("evidence") {
                    return Some(ent.path());
                }
            }
        }
    }
    None
}

fn inspect_evidence_zip(pack: &Path) -> (Vec<String>, Vec<String>) {
    const MAX_PACK: u64 = 8 * 1024 * 1024;
    const MAX_ENTRY: usize = 256 * 1024;
    let Ok(meta) = fs::metadata(pack) else { return (Vec::new(), Vec::new()); };
    if meta.len() > MAX_PACK { return (Vec::new(), Vec::new()); }
    let Ok(bytes) = fs::read(pack) else { return (Vec::new(), Vec::new()); };
    let Ok(archive) = rawzip::ZipArchive::from_slice(bytes.as_slice()) else { return (Vec::new(), Vec::new()); };
    let (mut excerpts, mut leaks, mut entries) = (Vec::new(), Vec::new(), archive.entries());
    for _ in 0..128 {
        let Ok(Some(entry)) = entries.next_entry() else { break };
        let name = String::from_utf8_lossy(entry.file_path().as_bytes()).into_owned();
        let wanted = name.ends_with("failure.yaml") || name.ends_with("result.yaml")
            || name.ends_with("-console.ndjson") || name.ends_with(".har");
        if !wanted { continue; }
        let Ok(raw) = read_zip_entry_capped(&archive, &entry, MAX_ENTRY) else { continue };
        if let Some((ln, kind, SecretTier::High)) = find_secret_tiered(&raw) {
            leaks.push(format!("{name}:{ln}:{kind}"));
        }
        let text = redact_secrets(String::from_utf8_lossy(&raw).into_owned());
        for line in text.lines() {
            let tr = line.trim();
            if tr.is_empty() { continue; }
            let hit = name.ends_with("failure.yaml")
                || tr.contains("AGENT_INFO") || tr.contains("error") || tr.contains("Error")
                || tr.contains("failed") || tr.contains("740") || tr.contains("rca_category");
            if hit && excerpts.len() < 12 {
                excerpts.push(format!("[{name}] {}", tr.chars().take(220).collect::<String>()));
            }
        }
    }
    (excerpts, leaks)
}

#[derive(Default)]
struct NdjsonAccum {
    cli_version: Option<String>,
    session_dir: Option<String>,
    evidence_id: Option<String>,
    test_url: Option<String>,
    terminal_status: Option<String>,
    terminal_seen: bool,
    remote_exit: Option<i32>,
    session_paused: bool,
    duration_s: f64,
    credits_sum: f64,
    has_credits: bool,
    replay_steps: u64,
    author_steps: u64,
    adaptive_healed: bool,
    confirmed_bug_740: bool,
    summary: String,
    failed_steps: Vec<FailedStep>,
    unresolved_vars: Vec<String>,
    policy_verdict: Option<PolicyVerdict>,
    final_state: Option<Value>,
    advisories: Vec<String>,
}

fn collect_gate_ids(pols: &[Value], field: &str) -> Vec<String> {
    pols.iter()
        .filter_map(|p| p.get(field).and_then(Value::as_array))
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn reduce_verdict_event(v: &Value, acc: &mut NdjsonAccum) {
    let result = v.get("result").and_then(Value::as_str).unwrap_or("fail").to_string();
    let exit_code = v.get("exit_code").and_then(Value::as_i64).unwrap_or(1) as i32;
    let pols = v.get("policies").and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
    acc.policy_verdict = Some(PolicyVerdict {
        result, exit_code,
        failed_gate_ids: collect_gate_ids(pols, "failed_gate_ids"),
        errored_gate_ids: collect_gate_ids(pols, "errored_gate_ids"),
    });
}

fn reduce_run_end(surface: KaneSurface, v: &Value, acc: &mut NdjsonAccum) {
    if let Some(c) = v.get("credits_consumed").or_else(|| v.get("credits")).and_then(Value::as_f64) {
        acc.credits_sum += c;
        acc.has_credits = true;
    }
    if let Some(d) = v.get("duration").or_else(|| v.get("duration_s")).and_then(Value::as_f64) {
        acc.duration_s = d;
    }
    if let Some(u) = v.get("test_url").or_else(|| v.get("share_url")).and_then(Value::as_str) {
        acc.test_url = Some(u.to_string());
    }
    if let Some(sd) = v.get("session_dir").and_then(Value::as_str) {
        acc.session_dir = Some(sd.to_string());
    }
    if let Some(s) = v.get("summary").or_else(|| v.get("one_liner")).or_else(|| v.get("reason"))
        .and_then(Value::as_str).filter(|s| !s.is_empty())
    {
        acc.summary = s.to_string();
    }
    let meta_740 = v.get("per_flow_metadata").and_then(Value::as_array).is_some_and(|arr| {
        arr.iter().any(|m| m.get("result_code").and_then(val_code_str).as_deref() == Some("740"))
    });
    if v.get("result_code").and_then(val_code_str).as_deref() == Some("740")
        || meta_740 || v.pointer("/verdict/confirmed").and_then(Value::as_bool) == Some(true)
    {
        acc.confirmed_bug_740 = true;
    }
    if let Some(fs_val) = v.get("final_state").or_else(|| v.get("variables_out")) {
        acc.final_state = Some(fs_val.clone());
    } else if let Some(fu) = v.get("final_url").and_then(Value::as_str) {
        acc.final_state = Some(json!({ "final_url": fu }));
    }
    if surface == KaneSurface::Run {
        acc.terminal_status = v.get("status").and_then(Value::as_str).map(str::to_string);
        acc.terminal_seen = true;
    }
}

fn reduce_testrun_event(ev_type: &str, v: &Value, acc: &mut NdjsonAccum) {
    match ev_type {
        "testrun_plan" => {
            let valid = v.get("valid").and_then(Value::as_bool).unwrap_or(true);
            let empty = v.get("members").and_then(Value::as_array).is_some_and(Vec::is_empty);
            if !valid || empty {
                acc.terminal_status = Some("refused".into());
                acc.summary = "testrun plan invalid or empty".into();
            }
        }
        "testrun_authored_member_end" | "testrun_member_end" => {
            let st = v.get("status").and_then(Value::as_str).unwrap_or("");
            if matches!(st, "failed" | "broken") {
                let remark = v.pointer("/failure/message").or_else(|| v.get("error"))
                    .and_then(Value::as_str).unwrap_or("testrun member failed").to_string();
                let step = v.pointer("/failure/step_index").and_then(Value::as_u64).unwrap_or(0);
                let heading = v.get("path").and_then(Value::as_str).map(str::to_string);
                acc.failed_steps.push(FailedStep { step, heading, remark, is_agent_info: false });
            }
        }
        "testrun_summary" => {
            if let Some(d) = v.get("duration_s").and_then(Value::as_f64) { acc.duration_s = d; }
            if let Some(eid) = v.pointer("/execution/evidence_id").and_then(Value::as_str) {
                acc.evidence_id = Some(eid.to_string());
            }
        }
        "testrun_done" => {
            acc.terminal_status = v.get("overall_status").and_then(Value::as_str).map(str::to_string);
            acc.terminal_seen = true;
        }
        "remote_done" => {
            acc.remote_exit = v.get("exit").and_then(Value::as_i64).map(|x| x as i32);
            if let Some(st) = v.get("status").and_then(Value::as_str) { acc.terminal_status = Some(st.to_string()); }
            acc.terminal_seen = true;
        }
        _ => {}
    }
}

fn reduce_untyped_event(v: Value, acc: &mut NdjsonAccum) {
    if v.get("step").is_some() {
        let st = v.get("status").and_then(Value::as_str).unwrap_or("");
        let remark = v.get("remark").and_then(Value::as_str).unwrap_or("");
        if st == "failed" || remark.contains("AGENT_INFO") {
            let step = v.get("step").and_then(Value::as_u64).unwrap_or(0);
            acc.failed_steps.push(FailedStep {
                step, heading: None, remark: remark.to_string(),
                is_agent_info: remark.contains("AGENT_INFO"),
            });
        }
    } else if let Some(ok) = v.get("ok").and_then(Value::as_bool) {
        acc.terminal_status = Some(if ok { "passed".into() } else { "failed".into() });
        acc.terminal_seen = true;
        acc.final_state = Some(v);
    }
}

fn reduce_one_event(surface: KaneSurface, v: Value, acc: &mut NdjsonAccum) {
    if let Some(ver) = v.get("v").and_then(Value::as_u64)
        && ver > 1 && !acc.advisories.iter().any(|a| a.starts_with("schema_version_drift:"))
    {
        acc.advisories.push(format!("schema_version_drift: saw wire v={ver} (supported v=1)"));
    }
    let ev_type = v.get("type").and_then(Value::as_str).unwrap_or("");
    match ev_type {
        "stream_start" => {
            if let Some(cv) = v.get("cli_version").and_then(Value::as_str) { acc.cli_version = Some(cv.to_string()); }
            if let Some(sd) = v.get("session_dir").and_then(Value::as_str) { acc.session_dir = Some(sd.to_string()); }
        }
        "warning" => {
            for item in v.get("variables").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                if !acc.unresolved_vars.iter().any(|x| x == item) { acc.unresolved_vars.push(item.to_string()); }
            }
        }
        "verdict" => reduce_verdict_event(&v, acc),
        "session_paused" => acc.session_paused = true,
        "step_end" => {
            let kind = v.get("kind").and_then(Value::as_str).unwrap_or("");
            if v.get("status").and_then(Value::as_str) == Some("failed") || kind == "AGENT_INFO" {
                let remark = v.get("remark").or_else(|| v.get("message")).and_then(Value::as_str).unwrap_or(kind).to_string();
                let step = v.get("step").and_then(Value::as_u64).unwrap_or(0);
                let heading = v.get("heading").and_then(Value::as_str).map(str::to_string);
                acc.failed_steps.push(FailedStep { step, heading, remark, is_agent_info: kind == "AGENT_INFO" });
            }
        }
        "run_end" => reduce_run_end(surface, &v, acc),
        "test_md_summary" => {
            if let Some(st) = v.get("steps") {
                acc.replay_steps = st.get("replay_decisions").and_then(Value::as_u64).unwrap_or(acc.replay_steps);
                acc.author_steps = st.get("author_decisions").and_then(Value::as_u64).unwrap_or(acc.author_steps);
            }
            if v.pointer("/adaptive_heal/triggered").and_then(Value::as_bool) == Some(true) { acc.adaptive_healed = true; }
        }
        "test_md_done" => {
            acc.terminal_status = v.get("overall_status").and_then(Value::as_str).map(str::to_string);
            acc.terminal_seen = true;
            if let Some(d) = v.get("duration_s").and_then(Value::as_f64) { acc.duration_s = d; }
            if let Some(u) = v.get("share_url").and_then(Value::as_str) { acc.test_url = Some(u.to_string()); }
        }
        "testrun_plan" | "testrun_authored_member_end" | "testrun_member_end"
        | "testrun_summary" | "testrun_done" | "remote_done" => reduce_testrun_event(ev_type, &v, acc),
        "done" => {
            acc.terminal_status = v.get("status").or_else(|| v.get("state")).and_then(Value::as_str).map(str::to_string);
            acc.terminal_seen = true;
        }
        "error" | "remote_error" => {
            let msg = v.get("message").or_else(|| v.get("remedy")).and_then(Value::as_str).unwrap_or("kane error");
            if acc.summary.is_empty() { acc.summary = msg.to_string(); }
        }
        "" => reduce_untyped_event(v, acc),
        _ => {}
    }
}

fn reduce_ndjson_stream(surface: KaneSurface, raw: &str, acc: &mut NdjsonAccum) {
    for line in raw.lines() {
        let tr = line.trim();
        if tr.starts_with('{') && let Ok(v) = serde_json::from_str::<Value>(tr) {
            reduce_one_event(surface, v, acc);
        }
    }
}

pub fn parse_kane_ndjson(
    surface: KaneSurface, stdout: &str, stderr: &str, exit_code: i32, timed_out: bool, cwd: &Path,
) -> KaneReceipt {
    let mut acc = NdjsonAccum::default();
    reduce_ndjson_stream(surface, stdout, &mut acc);
    if !acc.terminal_seen
        && let Some(sd) = acc.session_dir.clone()
        && let Ok(tee) = fs::read_to_string(Path::new(&sd).join("events.ndjson"))
    {
        reduce_ndjson_stream(surface, &tee, &mut acc);
    }
    let ev_path = extract_evidence_path(stderr, acc.session_dir.as_deref(), acc.evidence_id.as_deref(), cwd);
    let (evidence_excerpts, evidence_secret_leaks) = ev_path
        .as_deref().filter(|p| p.is_file()).map(inspect_evidence_zip).unwrap_or_default();
    let eff_exit = acc.remote_exit.unwrap_or(exit_code);
    let verdict = if acc.session_paused {
        "session_paused"
    } else if acc.policy_verdict.as_ref().is_some_and(|p| p.result == "errored") {
        "policy_errored"
    } else if acc.policy_verdict.as_ref().is_some_and(|p| p.result == "fail") {
        "policy_failed"
    } else if acc.confirmed_bug_740 {
        "bug_740"
    } else if timed_out || eff_exit == 3 {
        "timeout"
    } else if eff_exit == 2 || acc.terminal_status.as_deref() == Some("refused") {
        "did_not_start"
    } else if eff_exit == 0 && !matches!(acc.terminal_status.as_deref(), Some("failed" | "broken")) {
        "passed"
    } else {
        "failed"
    };
    let summary = if !acc.summary.is_empty() {
        acc.summary
    } else if acc.replay_steps > 0 && acc.author_steps == 0 && verdict == "passed" {
        format!("cached replay passed ({} replay / 0 author steps, 0 LLM credits)", acc.replay_steps)
    } else if let Some(f) = acc.failed_steps.last() {
        f.remark.clone()
    } else {
        format!("kane {surface:?} finished with verdict `{verdict}` (exit {eff_exit})")
    };
    let (evidence_path, evidence_serve_cmd) = match ev_path {
        Some(p) => {
            let s = p.display().to_string();
            (Some(s.clone()), Some(format!("kane-cli evidence serve \"{s}\"")))
        }
        None => (None, None),
    };
    KaneReceipt {
        ok: verdict == "passed" && evidence_secret_leaks.is_empty(),
        verdict: verdict.into(),
        surface,
        exit_code: eff_exit,
        cli_version: acc.cli_version,
        duration_s: acc.duration_s,
        credits_consumed: acc.has_credits.then_some(acc.credits_sum),
        replay_steps: acc.replay_steps,
        author_steps: acc.author_steps,
        adaptive_healed: acc.adaptive_healed,
        confirmed_bug_740: acc.confirmed_bug_740,
        summary,
        test_url: acc.test_url,
        evidence_path,
        evidence_serve_cmd,
        failed_steps: acc.failed_steps,
        evidence_excerpts,
        evidence_secret_leaks,
        unresolved_vars: acc.unresolved_vars,
        policy_verdict: acc.policy_verdict,
        final_state: acc.final_state,
        suspected_spans: Vec::new(),
        rewind_affected_files: Vec::new(),
        advisories: acc.advisories,
        git_head: read_git_head_fast(cwd),
        recorded_ms: now_ms(),
    }
}

pub fn run_kane(repo_root: &Path, req: &KaneRequest) -> KaneReceipt {
    let bin = resolve_kane_bin();
    let (probed_ver, caps) = probe_kane_caps(&bin);
    let (surface, args) = build_kane_args(req, caps);
    let timeout_ms = if req.timeout_secs > 0 { req.timeout_secs.saturating_mul(1_000) } else { 180_000 };
    let env = vec![
        ("KANE_CLI_USER_AGENT".into(), "refit".into()),
        ("NO_COLOR".into(), "1".into()),
    ];
    let mut receipt = match run_cmd(&bin, &args, repo_root, &env, false, timeout_ms) {
        Ok(res) => parse_kane_ndjson(surface, &res.stdout, &res.stderr, res.exit_code, res.timed_out, repo_root),
        Err(e) => KaneReceipt {
            ok: false, verdict: "did_not_start".into(), surface, exit_code: 2,
            cli_version: probed_ver.clone(), duration_s: 0.0, credits_consumed: None,
            replay_steps: 0, author_steps: 0, adaptive_healed: false, confirmed_bug_740: false,
            summary: format!("failed to launch `{bin}`: {e}"),
            test_url: None, evidence_path: None, evidence_serve_cmd: None,
            failed_steps: Vec::new(), evidence_excerpts: Vec::new(), evidence_secret_leaks: Vec::new(),
            unresolved_vars: Vec::new(), policy_verdict: None, final_state: None,
            suspected_spans: Vec::new(), rewind_affected_files: Vec::new(), advisories: Vec::new(),
            git_head: read_git_head_fast(repo_root), recorded_ms: now_ms(),
        },
    };
    if receipt.cli_version.is_none() { receipt.cli_version = probed_ver; }
    if let Ok(raw) = serde_json::to_vec_pretty(&receipt) {
        let _ = write_atomic(&repo_root.join(".refit/kane_last.json"), &raw);
    }
    receipt
}

pub fn read_kane_status(repo_root: &Path) -> Value {
    let read_j = |name: &str| {
        fs::read_to_string(repo_root.join(".refit").join(name))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
    };
    json!({
        "ok": true,
        "git_head": read_git_head_fast(repo_root),
        "last": read_j("kane_last.json"),
        "state": read_j("kane_state.json"),
    })
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

    fn build_store_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let (mut out, mut cd) = (Vec::new(), Vec::new());
        for &(name, data) in entries {
            let off = out.len() as u32;
            let mut crc = flate2::Crc::new();
            crc.update(data);
            let sum = crc.sum();
            let nb = name.as_bytes();
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&sum.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(nb);
            out.extend_from_slice(data);

            cd.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&sum.to_le_bytes());
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            cd.extend_from_slice(&[0u8; 12]);
            cd.extend_from_slice(&off.to_le_bytes());
            cd.extend_from_slice(nb);
        }
        let cd_off = out.len() as u32;
        let cd_len = cd.len() as u32;
        let n = entries.len() as u16;
        out.extend_from_slice(&cd);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn kane_v4_contracts_caps_and_evidence_secret_audit() {
        let td = tempfile::tempdir().unwrap();
        // 1. Official run.ndjson polymorphic int result_code (100) + string per_flow_metadata ("100")
        let run_stream = concat!(
            "{\"v\":1,\"type\":\"stream_start\",\"cli_version\":\"0.10.9\"}\n",
            "{\"v\":1,\"type\":\"run_end\",\"status\":\"passed\",\"result_code\":100,\"credits_consumed\":11.5,\"duration\":4.2,\"final_state\":{\"page_heading\":\"refit\"},\"per_flow_metadata\":[{\"result_code\":\"100\"}]}\n",
        );
        let r1 = parse_kane_ndjson(KaneSurface::Run, run_stream, "", 0, false, td.path());
        assert!(r1.ok && r1.verdict == "passed" && r1.cli_version.as_deref() == Some("0.10.9") && r1.credits_consumed == Some(11.5));

        // 2. 0-credit testmd replay + policy verdict disambiguation (exit 3 policy_errored vs exit 1 policy_failed)
        let replay = concat!(
            "{\"v\":1,\"type\":\"run_end\",\"status\":\"passed\",\"result_code\":100,\"summary\":\"\",\"final_url\":\"http://localhost:8080\"}\n",
            "{\"v\":1,\"type\":\"test_md_summary\",\"steps\":{\"replay_decisions\":2,\"author_decisions\":0},\"adaptive_heal\":{\"triggered\":false}}\n",
            "{\"v\":1,\"type\":\"test_md_done\",\"overall_status\":\"passed\",\"duration_s\":1.9}\n",
        );
        let r2 = parse_kane_ndjson(KaneSurface::TestMd, replay, "", 0, false, td.path());
        assert!(r2.ok && r2.replay_steps == 2 && r2.author_steps == 0 && r2.credits_consumed.is_none() && r2.summary.contains("0 LLM credits"));

        let pol_err = concat!(
            "{\"v\":1,\"type\":\"verdict\",\"result\":\"errored\",\"exit_code\":3,\"policies\":[{\"id\":\"p1\",\"errored_gate_ids\":[\"g1\"]}]}\n",
            "{\"v\":1,\"type\":\"testrun_done\",\"overall_status\":\"passed\"}\n",
        );
        let r3 = parse_kane_ndjson(KaneSurface::TestRun, pol_err, "", 3, false, td.path());
        assert!(!r3.ok && r3.verdict == "policy_errored" && r3.policy_verdict.as_ref().unwrap().errored_gate_ids == vec!["g1"]);

        // 3. 0.10.8 AGENT_INFO + bug 740 + v2 drift + ANSI space-path .evidence ZIP secret catch + redaction
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        let pack_dir = td.path().join("my run");
        fs::create_dir_all(&pack_dir).unwrap();
        let pack = pack_dir.join("hud 01.evidence");
        let zip = build_store_zip(&[
            ("failure.yaml", format!("rca_category: ui_bug\nleak: {aws}\n").as_bytes()),
            ("tests/t1/logs/0-console.ndjson", b"{\"level\":\"error\",\"text\":\"TypeError in renderHud\"}\n"),
        ]);
        fs::write(&pack, zip).unwrap();
        let stderr = format!("\x1b[32mevidence:\x1b[0m view locally with `kane-cli evidence serve {}`\r\n", pack.display());
        let bug_stream = concat!(
            "non-json banner\n",
            "{\"v\":2,\"type\":\"step_end\",\"step\":3,\"kind\":\"AGENT_INFO\",\"status\":\"failed\",\"remark\":\"button click broke\"}\n",
            "{\"v\":2,\"type\":\"run_end\",\"status\":\"failed\",\"result_code\":\"740\",\"credits\":2.5,\"verdict\":{\"confirmed\":true}}\n",
        );
        let r4 = parse_kane_ndjson(KaneSurface::Run, bug_stream, &stderr, 1, false, td.path());
        assert!(!r4.ok && r4.verdict == "bug_740" && r4.confirmed_bug_740 && r4.credits_consumed == Some(2.5));
        assert!(r4.failed_steps.iter().any(|s| s.is_agent_info));
        assert!(!r4.evidence_secret_leaks.is_empty() && r4.evidence_excerpts.iter().all(|e| !e.contains(&aws)));
        assert!(r4.advisories.iter().any(|a| a.contains("schema_version_drift")));

        // 4. Capability negotiation: 0.8.18 strips --policy & testrun --agent; 0.10.9 guards --tunnel-name to real devices
        let req = KaneRequest {
            spec: "tags:smoke".into(), policy: Some("p.yaml".into()),
            target: Some("chrome".into()), tunnel_name: Some("tun1".into()), headless: true, ..Default::default()
        };
        let (_, args_old) = build_kane_args(&req, caps_for_version("0.8.18"));
        assert!(!args_old.contains(&"--policy".to_string()) && !args_old.contains(&"--agent".to_string()));
        let (_, args_new) = build_kane_args(&req, caps_for_version("0.10.9"));
        assert!(args_new.contains(&"--policy".to_string()) && args_new.contains(&"--agent".to_string()) && !args_new.contains(&"--tunnel-name".to_string()));
        assert!(!KANE_SARIF_RULES.is_empty());
    }
}
