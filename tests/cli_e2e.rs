use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("refit_e2e_{tag}_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, rel: &str, body: &str) {
        let p = self.0.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, body).unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn refit_bin() -> &'static str {
    env!("CARGO_BIN_EXE_refit")
}

fn run_refit(args: &[&str], envs: &[(&str, &str)]) -> (i32, String, String) {
    let mut cmd = Command::new(refit_bin());
    cmd.args(args).env("NO_COLOR", "1");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn refit binary");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn sample_repo(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    dir.write(
        "Cargo.toml",
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    );
    dir.write(
        "src/lib.rs",
        "pub mod helper;\npub fn run(x: i32) -> i32 { helper::add_one(x) }\n",
    );
    dir.write(
        "src/helper.rs",
        "pub fn add_one(x: i32) -> i32 { x + 1 }\npub fn unused_dead_fn() -> i32 { 99 }\n",
    );
    // Include one >1200-line file to verify --approx-long cache isolation.
    let mut big = String::from("pub fn top_clone(v: i32) -> i32 {\n    let a = v + 1;\n    let b = a * 2;\n    b - 3\n}\n");
    for i in 0..420 {
        big.push_str(&format!(
            "pub fn step_{i}(x: i32) -> i32 {{\n    let y = x.wrapping_add({i});\n    y.wrapping_mul(3)\n}}\n"
        ));
    }
    big.push_str("pub fn bot_clone(v: i32) -> i32 {\n    let a = v + 1;\n    let b = a * 2;\n    b - 3\n}\n");
    dir.write("src/long_mod.rs", &big);
    dir
}

#[test]
fn cli_scan_audit_approx_long_isolation() {
    let repo = sample_repo("scan_audit");
    let root = repo.path().to_str().unwrap();

    let (code, out, err) = run_refit(&["scan", root, "--fast"], &[]);
    assert_eq!(code, 0, "scan failed: {err}");
    let scan: Value = serde_json::from_str(&out).expect("scan JSON");
    assert!(scan["units"].as_u64().unwrap_or(0) >= 3);
    assert_eq!(scan["scc_count"].as_u64(), Some(0));

    let (code, out_exact, err) = run_refit(&["audit", root], &[]);
    assert_eq!(code, 0, "audit exact failed: {err}");
    let exact: Value = serde_json::from_str(&out_exact).expect("audit exact JSON");
    let exact_p10 = exact["audit"]["hygiene_p10"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&exact_p10));

    let (code, out_approx, err) = run_refit(&["audit", root, "--approx-long"], &[]);
    assert_eq!(code, 0, "audit approx-long failed: {err}");
    let approx: Value = serde_json::from_str(&out_approx).expect("audit approx JSON");
    assert!(approx["audit"]["hygiene_p10"].as_f64().is_some());

    // Exact audit after approx-long must reproduce exact p10 (cache key isolation).
    let (code, out_exact2, err) = run_refit(&["audit", root], &[]);
    assert_eq!(code, 0, "audit exact repeat failed: {err}");
    let exact2: Value = serde_json::from_str(&out_exact2).expect("audit exact2 JSON");
    assert!(
        (exact2["audit"]["hygiene_p10"].as_f64().unwrap() - exact_p10).abs() < 1e-6,
        "approx-long poisoned exact cache"
    );

    // Offline --research must degrade cleanly without network access.
    let (code, out_res, err) = run_refit(
        &["audit", root, "--research"],
        &[("REFIT_SEARCH_PROVIDERS", "keenable"), ("KEENABLE_API_KEY", "")],
    );
    assert_eq!(code, 0, "audit --research offline failed: {err}");
    let res_val: Value = serde_json::from_str(&out_res).expect("audit research JSON");
    assert!(res_val["dependency_research"].is_array());
}

#[test]
fn cli_sustain_learn_gc_and_legacy_cache_migration() {
    let repo = sample_repo("sustain_gc");
    let root = repo.path().to_str().unwrap();

    // Plant stale v1 stat cache magic ("RFST0001") and verify scan auto-migrates to RFST0002.
    let cache_dir = repo.path().join(".refit/cache");
    fs::create_dir_all(&cache_dir).unwrap();
    let stat_bin = cache_dir.join("stat.bin");
    fs::write(&stat_bin, b"RFST0001legacy_payload").unwrap();

    let (code, _, err) = run_refit(&["scan", root, "--fast"], &[]);
    assert_eq!(code, 0, "scan after legacy cache failed: {err}");
    let migrated = fs::read(&stat_bin).unwrap();
    assert!(
        migrated.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]),
        "expected zstd v2 stat cache after migration, got {:?}",
        &migrated[..migrated.len().min(8)]
    );

    // Sustain dry-run lists all 4 governance artifacts without writing them.
    let (code, out, err) = run_refit(&["sustain", root, "--dry-run"], &[]);
    assert_eq!(code, 0, "sustain --dry-run failed: {err}");
    let sus: Value = serde_json::from_str(&out).expect("sustain JSON");
    let files = sus["files_written"].as_array().expect("sustain files_written");
    assert!(files.len() >= 4);
    assert!(!repo.path().join("AGENTS.md").exists());

    #[cfg(feature = "ste100")]
    {
        let (code, out, err) = run_refit(&["sustain", root, "--dry-run", "--ste100"], &[]);
        assert_eq!(code, 0, "sustain --ste100 failed: {err}");
        let ste: Value = serde_json::from_str(&out).expect("sustain ste100 JSON");
        assert!(ste["ste100"].is_object());
    }

    // Learn rewind returns structured impact report.
    let (code, out, err) = run_refit(&["learn", root, "--rewind", "add_one"], &[]);
    assert_eq!(code, 0, "learn --rewind failed: {err}");
    let lrn: Value = serde_json::from_str(&out).expect("learn JSON");
    assert!(lrn["rules"].is_array());
    assert!(lrn["rewind"].is_object());

    // GC preserves CACHEDIR.TAG.
    let (code, out, err) = run_refit(&["gc", root, "--cache-max-mb", "1"], &[]);
    assert_eq!(code, 0, "gc failed: {err}");
    let _: Value = serde_json::from_str(&out).expect("gc JSON");
    assert!(cache_dir.join("CACHEDIR.TAG").exists());
}

#[test]
fn cli_mcp_stdio_roundtrip_and_batch_reject() {
    let repo = sample_repo("mcp_stdio");
    let root = repo.path().to_str().unwrap();

    let mut child = Command::new(refit_bin())
        .args(["mcp", root])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn refit mcp");

    {
        let stdin = child.stdin.as_mut().unwrap();
        let reqs = [
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"1.0"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
            r#"[{"jsonrpc":"2.0","id":3,"method":"ping"}]"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"refit_scan","arguments":{"fast":true}}}"#,
        ];
        for line in reqs {
            writeln!(stdin, "{line}").unwrap();
        }
    }

    let output = child.wait_with_output().expect("wait refit mcp");
    assert!(
        output.status.success(),
        "mcp exit {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut responses = Vec::new();
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        responses.push(serde_json::from_str::<Value>(line).expect("valid JSON-RPC line"));
    }
    assert_eq!(responses.len(), 4, "expected 4 responses, got {responses:?}");

    let init = responses.iter().find(|r| r["id"] == 1).expect("init response");
    assert!(init["result"]["serverInfo"]["name"].as_str().is_some());

    let list = responses.iter().find(|r| r["id"] == 2).expect("tools/list response");
    let tools = list["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 17, "expected 17 MCP tools, got {}", tools.len());

    let batch_err = responses.iter().find(|r| r["error"]["code"] == -32600);
    assert!(batch_err.is_some(), "batch array must return -32600: {responses:?}");

    let scan = responses.iter().find(|r| r["id"] == 4).expect("refit_scan response");
    assert!(scan["error"].is_null(), "refit_scan failed: {scan:?}");
}
