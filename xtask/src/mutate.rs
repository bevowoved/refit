use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use walkdir::WalkDir;

struct Mutant {
    id: &'static str,
    file: &'static str,
    target: &'static str,
    replacement: &'static str,
    test: &'static str,
    claim: &'static str,
}

const MUTANTS: &[Mutant] = &[
    Mutant {
        id: "M1_mvcc_skip_cas",
        file: "src/mvcc.rs",
        target: "if let Some(conflict) = check_cas(&store.repo_root, expected_hashes, &mut old_blobs)? {",
        replacement: "if false { let conflict = check_cas(&store.repo_root, expected_hashes, &mut old_blobs)?.unwrap();",
        test: "mvcc_cas_and_rollback",
        claim: "MVCC commit rejects stale CAS hash",
    },
    Mutant {
        id: "M2_sparse_drop_far_gate",
        file: "src/hygiene/ops.rs",
        target: "if qk_far > hr.head.sink && sm.count < sm.keys.len() {",
        replacement: "if false && qk_far > hr.head.sink && sm.count < sm.keys.len() {",
        test: "sparse_far_clone_recall",
        claim: "Sparse long-file recall detects distant clones",
    },
    Mutant {
        id: "M3_uv_allow_symlink",
        file: "src/gates/stage.rs",
        target: "\"symlink\" => anyhow::bail!(\"UV_LINK_MODE=symlink is forbidden because `uv cache clean` breaks virtualenvs\"),",
        replacement: "\"symlink\" => Ok(\"symlink\"),",
        test: "stage_cascade_and_uv_mode",
        claim: "Stage rejects UV_LINK_MODE=symlink",
    },
    Mutant {
        id: "M4_path_skip_canonical",
        file: "src/platform/paths.rs",
        target: "if !verify_canonical_in_root(&root, &cur) {",
        replacement: "if false && !verify_canonical_in_root(&root, &cur) {",
        test: "cross_os_paths",
        claim: "Path guard blocks symlink escape outside repo",
    },
    Mutant {
        id: "M5_sst_skip_footer_tag",
        file: "src/stem.rs",
        target: "if blake3_tag(&[&hdr[0..26], &footer]) != expected_tag { return None; }",
        replacement: "if false && blake3_tag(&[&hdr[0..26], &footer]) != expected_tag { return None; }",
        test: "sst_zstd_lazy_and_threshold",
        claim: "SST reader rejects tampered footer tag",
    },
    Mutant {
        id: "M6_mrtr_skip_hmac",
        file: "src/mcp.rs",
        target: "if blake3::keyed_hash(&self.key, &payload_bytes) != blake3::Hash::from_bytes(sig_arr) {",
        replacement: "if false && blake3::keyed_hash(&self.key, &payload_bytes) != blake3::Hash::from_bytes(sig_arr) {",
        test: "resume_token_sign_and_expire",
        claim: "MCP resume token rejects forged HMAC",
    },
    Mutant {
        id: "M7_mcp_allow_batch",
        file: "src/mcp.rs",
        target: "if parsed.is_array() {",
        replacement: "if false && parsed.is_array() {",
        test: "http_discover_and_headers",
        claim: "MCP HTTP rejects JSON-RPC batch arrays",
    },
    Mutant {
        id: "M8_research_drop_432",
        file: "src/research.rs",
        target: "if is_quota_exhausted(code) {",
        replacement: "if false && is_quota_exhausted(code) {",
        test: "router_failover_and_race",
        claim: "Search router marks HTTP 432 permanently open",
    },
];

#[derive(Serialize)]
struct MutantResult {
    id: &'static str,
    file: &'static str,
    test: &'static str,
    claim: &'static str,
    status: &'static str,
    exit_code: i32,
    duration_ms: u64,
}

#[derive(Serialize)]
struct MutateReport {
    total: usize,
    killed: usize,
    survived: usize,
    overall: &'static str,
    mutants: Vec<MutantResult>,
}

fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    for entry in WalkDir::new(src) {
        let entry = entry?;
        let rel = entry.path().strip_prefix(src)?;
        let out = dst.join(rel);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&out)?;
        } else if entry.file_type().is_file() {
            if let Some(p) = out.parent() {
                fs::create_dir_all(p)?;
            }
            fs::copy(entry.path(), &out)?;
        }
    }
    Ok(())
}

pub fn run(repo_root: &Path, out_json: Option<&Path>) -> Result<()> {
    let tmp = tempfile::Builder::new().prefix("refit_t14_").tempdir()?;
    let tmp_root = tmp.path();
    let target_dir = std::env::temp_dir().join("refit_t14_cargo_target");
    fs::create_dir_all(&target_dir)?;

    for item in ["Cargo.toml", "Cargo.lock", "src", "skills", "tests", "vendor", "xtask"] {
        let src = repo_root.join(item);
        if src.is_dir() {
            copy_tree(&src, &tmp_root.join(item))?;
        } else if src.is_file() {
            fs::copy(&src, tmp_root.join(item))?;
        }
    }

    println!("=== T14 Mutation Sensitivity Suite ({} Faults) ===", MUTANTS.len());
    let warm = Command::new("cargo")
        .args(["test", "--offline", "--features", "full", "--bin", "refit", "--no-run"])
        .current_dir(tmp_root)
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .context("warm-build T14 scratch copy")?;
    if !warm.status.success() {
        bail!(
            "clean baseline test build failed in T14 scratch copy:\n{}",
            String::from_utf8_lossy(&warm.stderr)
        );
    }

    let mut results = Vec::with_capacity(MUTANTS.len());
    let mut killed = 0usize;

    for m in MUTANTS {
        let path: PathBuf = tmp_root.join(m.file);
        let orig = fs::read_to_string(&path)?;
        if !orig.contains(m.target) {
            bail!("anchor `{}` missing in {}", m.id, m.file);
        }
        let mutated = orig.replacen(m.target, m.replacement, 1);
        let started = Instant::now();
        fs::write(&path, &mutated)?;
        let out = Command::new("cargo")
            .args(["test", "--offline", "--features", "full", "--bin", "refit", m.test])
            .current_dir(tmp_root)
            .env("CARGO_TARGET_DIR", &target_dir)
            .output();
        fs::write(&path, &orig)?;
        let out = out.with_context(|| format!("run cargo test for {}", m.id))?;
        let ms = started.elapsed().as_millis() as u64;
        let code = out.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&out.stdout);
        if stdout.contains("running 0 tests") {
            bail!("mutation test `{}` matched 0 tests for {}", m.test, m.id);
        }
        let is_killed = !out.status.success();
        if is_killed {
            killed += 1;
        }
        let status = if is_killed { "KILLED" } else { "SURVIVED" };
        println!("[{status:<8}] {:<26} test={:<30} ({ms} ms)", m.id, m.test);
        results.push(MutantResult {
            id: m.id,
            file: m.file,
            test: m.test,
            claim: m.claim,
            status,
            exit_code: code,
            duration_ms: ms,
        });
    }

    let report = MutateReport {
        total: MUTANTS.len(),
        killed,
        survived: MUTANTS.len() - killed,
        overall: if killed == MUTANTS.len() { "PASS" } else { "FAIL" },
        mutants: results,
    };
    if let Some(out_path) = out_json {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, serde_json::to_string_pretty(&report)?)?;
        println!("T14 RESULT: {killed}/{} KILLED -> {}", MUTANTS.len(), out_path.display());
    } else {
        println!("T14 RESULT: {killed}/{} KILLED", MUTANTS.len());
    }
    if killed != MUTANTS.len() {
        bail!("T14 mutation sensitivity failed: {killed}/{} killed", MUTANTS.len());
    }
    Ok(())
}
