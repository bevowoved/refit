use crate::graph::compute_python_abi_hash;
use crate::hygiene::HygieneScorer;
use crate::platform::{now_ms, run_sandboxed_cmd};
use crate::types::{GateVerdict, ModuleUnit};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use walkdir::WalkDir;

static STAGE_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub repo_root: PathBuf,
    pub python_bin: String,
    pub pytest_bin: String,
    pub ruff_bin: String,
    pub ty_bin: String,
    pub min_hygiene_target: f64,
}

#[derive(Debug, Clone)]
pub struct FullAuditResult {
    pub tests_total: usize,
    pub tests_failed: usize,
    pub ruff_errors: usize,
    pub ty_errors: usize,
    pub hygiene_median: f64,
    pub hygiene_p10: f64,
    pub per_file_diagnostics: HashMap<String, Vec<String>>,
}

pub fn detect_gate_config(repo_root: &Path) -> GateConfig {
    let user_local = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(|h| PathBuf::from(h).join(".local/bin"))
        .unwrap_or_default();

    let venv_candidates = [
        repo_root.join(".venv"),
        PathBuf::from("D:/cerebras/refit/.venv"),
        PathBuf::from(".venv"),
    ];

    let mut python_bin = "python".to_string();
    let mut pytest_bin = "pytest".to_string();
    for v in &venv_candidates {
        let py = if cfg!(windows) {
            v.join("Scripts/python.exe")
        } else {
            v.join("bin/python")
        };
        let pt = if cfg!(windows) {
            v.join("Scripts/pytest.exe")
        } else {
            v.join("bin/pytest")
        };
        if py.exists() {
            python_bin = py.display().to_string();
        }
        if pt.exists() {
            pytest_bin = pt.display().to_string();
            break;
        }
    }

    let ruff_bin = resolve_tool("ruff", &user_local);
    let ty_bin = resolve_tool("ty", &user_local);

    GateConfig {
        repo_root: repo_root.to_path_buf(),
        python_bin,
        pytest_bin,
        ruff_bin,
        ty_bin,
        min_hygiene_target: 0.85,
    }
}

fn resolve_tool(name: &str, user_local: &Path) -> String {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let local = user_local.join(&exe);
    if local.exists() {
        return local.display().to_string();
    }
    name.to_string()
}

struct StageGuard(PathBuf);
impl Drop for StageGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Copies the lightweight target repo source/tests into `.refit/stages/tx-<id>/` (< 3 ms for 10K LOC)
/// and overlays candidate files so K parallel workers verify with ZERO working-tree interference.
fn materialize_stage(
    repo_root: &Path,
    candidate_files: &BTreeMap<String, String>,
) -> Result<StageGuard> {
    let seq = STAGE_SEQ.fetch_add(1, Ordering::Relaxed);
    let stage_dir = repo_root
        .join(".refit/stages")
        .join(format!("tx-{}-{seq}", now_ms()));
    fs::create_dir_all(&stage_dir)?;

    for entry in WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".refit" | ".venv" | ".git" | "__pycache__" | ".pytest_cache" | ".ruff_cache" | "target" | "node_modules" | ".tools"
            )
        })
        .flatten()
    {
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        if rel.as_os_str().is_empty() {
            continue;
        }
        let dst = stage_dir.join(rel);
        if entry.file_type().is_dir() {
            let _ = fs::create_dir_all(&dst);
        } else if entry.file_type().is_file() {
            if let Some(p) = dst.parent() {
                let _ = fs::create_dir_all(p);
            }
            let _ = fs::copy(path, &dst);
        }
    }

    for (rel, content) in candidate_files {
        let dst = stage_dir.join(rel);
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(&dst, content.as_bytes())
            .with_context(|| format!("write staged candidate {}", dst.display()))?;
    }

    Ok(StageGuard(stage_dir))
}

pub fn verify_candidate(
    cfg: &GateConfig,
    unit: &ModuleUnit,
    candidate_files: &BTreeMap<String, String>,
    hygiene: &HygieneScorer,
) -> Result<GateVerdict> {
    let stage = materialize_stage(&cfg.repo_root, candidate_files)?;
    let stage_root = &stage.0;

    let mut verdict = GateVerdict {
        passed: false,
        failed_gate: None,
        gate0_syntax_ok: false,
        gate0_abi_ok: true,
        gate0_ms: 0,
        gate1_tests_passed: 0,
        gate1_tests_failed: 0,
        gate1_ms: 0,
        gate1_cpu_ms: 0,
        gate1_peak_rss_bytes: 0,
        gate2_score_before: unit.baseline_hygiene,
        gate2_score_after: unit.baseline_hygiene,
        gate2_ms: 0,
        gate3_ruff_errors: 0,
        gate3_ty_errors: 0,
        gate3_ms: 0,
        diagnostics: Vec::new(),
        final_contents: BTreeMap::new(),
    };

    let staged_rel_paths: Vec<String> = candidate_files.keys().cloned().collect();

    // ---------------- GATE 0: Syntax + Exported ABI Pin ----------------
    let g0_t = Instant::now();
    let mut py_args = vec![
        "-c".to_string(),
        "import ast, sys\nfor p in sys.argv[1:]:\n    ast.parse(open(p, encoding='utf-8').read(), filename=p)".to_string(),
    ];
    for rel in &staged_rel_paths {
        py_args.push(rel.clone());
    }
    let syn = run_sandboxed_cmd(&cfg.python_bin, &py_args, stage_root, &[], false, 5_000)?;
    verdict.gate0_syntax_ok = syn.exit_code == 0 && !syn.timed_out;
    if !verdict.gate0_syntax_ok {
        verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
        verdict.failed_gate = Some("gate0_syntax".into());
        verdict
            .diagnostics
            .push(format!("Gate 0 syntax error: {}", first_lines(&syn.stderr, 4)));
        return Ok(verdict);
    }

    if !unit.allow_abi_change {
        for (rel, content) in candidate_files {
            if let Some(expected_abi) = unit.abi_hash.get(rel) {
                let actual_abi = compute_python_abi_hash(content);
                if &actual_abi != expected_abi {
                    verdict.gate0_abi_ok = false;
                    verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
                    verdict.failed_gate = Some("gate0_abi".into());
                    verdict.diagnostics.push(format!(
                        "Gate 0 AbiPinViolation in `{rel}`: public def/class signature names changed"
                    ));
                    return Ok(verdict);
                }
            }
        }
    }
    verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;

    // ---------------- GATE 3 Auto-fix + Zero-Error/Zero-Warning Check ----------------
    // Run ruff --fix first inside stage so formatting/import order is clean before Gate 1 & Gate 2
    let g3_t = Instant::now();
    let mut ruff_fix_args = vec![
        "check".to_string(),
        "--fix".to_string(),
        "--exit-zero".to_string(),
        "--quiet".to_string(),
    ];
    ruff_fix_args.extend(staged_rel_paths.iter().cloned());
    let _ = run_sandboxed_cmd(&cfg.ruff_bin, &ruff_fix_args, stage_root, &[], false, 8_000);

    // Re-read possibly auto-fixed bytes
    let mut working_texts: BTreeMap<String, String> = BTreeMap::new();
    for rel in &staged_rel_paths {
        let txt = fs::read_to_string(stage_root.join(rel))?;
        working_texts.insert(rel.clone(), txt);
    }

    let mut ruff_check_args = vec![
        "check".to_string(),
        "--output-format=json".to_string(),
    ];
    ruff_check_args.extend(staged_rel_paths.iter().cloned());
    let ruff_res = run_sandboxed_cmd(&cfg.ruff_bin, &ruff_check_args, stage_root, &[], false, 8_000)?;
    let ruff_diags = parse_ruff_json(&ruff_res.stdout);
    verdict.gate3_ruff_errors = ruff_diags.len();

    let mut ty_args = vec![
        "check".to_string(),
        "--error-on-warning".to_string(),
    ];
    ty_args.extend(staged_rel_paths.iter().cloned());
    let ty_res = run_sandboxed_cmd(&cfg.ty_bin, &ty_args, stage_root, &[], false, 10_000)?;
    let ty_diags = if ty_res.exit_code == 0 {
        Vec::new()
    } else {
        parse_ty_lines(&ty_res.stdout, &ty_res.stderr)
    };
    verdict.gate3_ty_errors = ty_diags.len();
    verdict.gate3_ms = g3_t.elapsed().as_millis() as u64;

    // ---------------- GATE 1: Sandboxed Impact-Cone Unit Tests ----------------
    let g1_t = Instant::now();
    let mut pt_args = vec!["-q".to_string(), "--tb=short".to_string()];
    if unit.test_files.is_empty() {
        pt_args.push("tests".to_string());
    } else {
        pt_args.extend(unit.test_files.iter().cloned());
    }
    let pt_env = vec![("PYTHONPATH".to_string(), stage_root.display().to_string())];
    let pt_res = run_sandboxed_cmd(&cfg.pytest_bin, &pt_args, stage_root, &pt_env, false, 15_000)?;
    let (passed, failed) = parse_pytest_summary(&pt_res.stdout, &pt_res.stderr, pt_res.exit_code);
    verdict.gate1_tests_passed = passed;
    verdict.gate1_tests_failed = failed;
    verdict.gate1_ms = g1_t.elapsed().as_millis() as u64;
    verdict.gate1_cpu_ms = pt_res.cpu_ms;
    verdict.gate1_peak_rss_bytes = pt_res.peak_rss_bytes;

    // ---------------- GATE 2: Warm codehygiene v0.7.0 Maintainability ----------------
    let g2_t = Instant::now();
    let mut score_sum = 0.0f64;
    let mut hygiene_hints = Vec::new();
    for (rel, txt) in &working_texts {
        let h = hygiene.score_content(rel, txt);
        score_sum += h.score;
        for sq in h.squiggles.iter().take(4) {
            hygiene_hints.push(format!(
                "codehygiene `{rel}` L{}: {} ({:.2}) {}",
                sq.line, sq.kind, sq.probability, sq.message
            ));
        }
    }
    verdict.gate2_score_after = score_sum / (working_texts.len().max(1) as f64);
    verdict.gate2_ms = g2_t.elapsed().as_millis() as u64;

    // Evaluate pass/fail gates in strict order
    if pt_res.exit_code != 0 || verdict.gate1_tests_failed > 0 {
        verdict.failed_gate = Some("gate1_tests".into());
        verdict.diagnostics.push(format!(
            "Gate 1 pytest failed ({failed} failed): {}",
            first_lines(&pt_res.stdout, 8)
        ));
    }
    if verdict.gate3_ruff_errors > 0 {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate3_ruff".into());
        }
        verdict.diagnostics.extend(ruff_diags.into_iter().take(6));
    }
    if verdict.gate3_ty_errors > 0 {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate3_ty".into());
        }
        verdict.diagnostics.extend(ty_diags.into_iter().take(6));
    }
    let hygiene_ok = verdict.gate2_score_after >= cfg.min_hygiene_target
        || (verdict.gate2_score_after >= verdict.gate2_score_before + 0.12 && verdict.gate2_score_after >= 0.72);
    if !hygiene_ok {
        if verdict.failed_gate.is_none() {
            verdict.failed_gate = Some("gate2_hygiene".into());
        }
        verdict.diagnostics.push(format!(
            "Gate 2 codehygiene too low: {:.3} (before {:.3}, target >= {:.2})",
            verdict.gate2_score_after, verdict.gate2_score_before, cfg.min_hygiene_target
        ));
        verdict.diagnostics.extend(hygiene_hints.into_iter().take(4));
    }

    if verdict.failed_gate.is_none() {
        verdict.passed = true;
        for (rel, txt) in working_texts {
            verdict.final_contents.insert(rel, txt.into_bytes());
        }
    }

    Ok(verdict)
}

pub fn run_full_repo_audit(
    cfg: &GateConfig,
    rel_files: &[String],
    hygiene: &HygieneScorer,
) -> Result<FullAuditResult> {
    let mut per_file_diagnostics: HashMap<String, Vec<String>> = HashMap::new();

    // 1. Full pytest suite
    let pt_env = vec![("PYTHONPATH".to_string(), cfg.repo_root.display().to_string())];
    let pt = run_sandboxed_cmd(
        &cfg.pytest_bin,
        &["-q".into(), "--tb=line".into()],
        &cfg.repo_root,
        &pt_env,
        false,
        30_000,
    )?;
    let (passed, failed) = parse_pytest_summary(&pt.stdout, &pt.stderr, pt.exit_code);
    for line in pt.stdout.lines() {
        if line.starts_with("FAILED ") {
            // Map test_mod_XX -> pkg/.../mod_XX.py
            for rel in rel_files {
                let stem = Path::new(rel).file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if !stem.is_empty() && line.contains(&format!("test_{stem}")) {
                    per_file_diagnostics
                        .entry(rel.clone())
                        .or_default()
                        .push(format!("pytest {line}"));
                }
            }
        }
    }

    // 2. Full ruff check
    let mut ruff_args = vec!["check".to_string(), "--output-format=json".to_string()];
    ruff_args.extend(rel_files.iter().cloned());
    let ruff_res = run_sandboxed_cmd(&cfg.ruff_bin, &ruff_args, &cfg.repo_root, &[], false, 15_000)?;
    let ruff_items = parse_ruff_json_with_files(&ruff_res.stdout, &cfg.repo_root);
    let ruff_errors = ruff_items.len();
    for (file_rel, msg) in ruff_items {
        per_file_diagnostics.entry(file_rel).or_default().push(msg);
    }

    // 3. Full ty check --error-on-warning
    let mut ty_args = vec!["check".to_string(), "--error-on-warning".to_string()];
    ty_args.extend(rel_files.iter().cloned());
    let ty_res = run_sandboxed_cmd(&cfg.ty_bin, &ty_args, &cfg.repo_root, &[], false, 20_000)?;
    let ty_lines = if ty_res.exit_code == 0 {
        Vec::new()
    } else {
        parse_ty_lines(&ty_res.stdout, &ty_res.stderr)
    };
    let ty_errors = ty_lines.len();
    for msg in &ty_lines {
        for rel in rel_files {
            if msg.contains(rel) || msg.replace('\\', "/").contains(rel) {
                per_file_diagnostics
                    .entry(rel.clone())
                    .or_default()
                    .push(msg.clone());
            }
        }
    }

    // 4. Full codehygiene scan
    let (hygiene_median, hygiene_p10, per_file_h) = hygiene.score_repo(&cfg.repo_root, rel_files);
    for (rel, hr) in &per_file_h {
        for sq in hr.squiggles.iter().take(4) {
            per_file_diagnostics.entry(rel.clone()).or_default().push(format!(
                "codehygiene L{}: {} ({:.2})",
                sq.line, sq.kind, sq.probability
            ));
        }
    }

    Ok(FullAuditResult {
        tests_total: passed + failed,
        tests_failed: failed,
        ruff_errors,
        ty_errors,
        hygiene_median,
        hygiene_p10,
        per_file_diagnostics,
    })
}

fn parse_pytest_summary(stdout: &str, stderr: &str, exit_code: i32) -> (usize, usize) {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines().rev() {
        let l = line.trim();
        if l.contains(" passed") || l.contains(" failed") || l.contains(" error") {
            for part in l.split(',') {
                let toks: Vec<&str> = part.split_whitespace().collect();
                for w in toks.windows(2) {
                    if let Ok(n) = w[0].parse::<usize>() {
                        if w[1].starts_with("passed") {
                            passed = passed.max(n);
                        } else if w[1].starts_with("failed") || w[1].starts_with("error") {
                            failed = failed.max(n);
                        }
                    }
                }
            }
            if passed > 0 || failed > 0 {
                break;
            }
        }
    }
    if exit_code != 0 && failed == 0 {
        failed = 1;
    }
    (passed, failed)
}

fn parse_ruff_json(stdout: &str) -> Vec<String> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    arr.into_iter()
        .map(|v| {
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            format!("ruff {code} L{line}: {msg}")
        })
        .collect()
}

fn parse_ruff_json_with_files(stdout: &str, repo_root: &Path) -> Vec<(String, String)> {
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    let root_norm = repo_root.display().to_string().replace('\\', "/");
    arr.into_iter()
        .map(|v| {
            let fname = v["filename"].as_str().unwrap_or("").replace('\\', "/");
            let rel = fname
                .strip_prefix(&format!("{root_norm}/"))
                .unwrap_or(&fname)
                .to_string();
            let code = v["code"].as_str().unwrap_or("RUFF");
            let msg = v["message"].as_str().unwrap_or("");
            let line = v["location"]["row"].as_u64().unwrap_or(0);
            (rel, format!("ruff {code} L{line}: {msg}"))
        })
        .collect()
}

fn parse_ty_lines(stdout: &str, stderr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines() {
        let t = line.trim();
        if t.starts_with("error[") || t.starts_with("warning[") || t.contains("--> ") {
            out.push(format!("ty: {t}"));
        }
    }
    if out.is_empty() {
        let summary = first_lines(&combined, 4);
        if !summary.is_empty() {
            out.push(format!("ty: {summary}"));
        }
    }
    out
}

fn first_lines(s: &str, max: usize) -> String {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .take(max)
        .collect::<Vec<_>>()
        .join(" | ")
}
