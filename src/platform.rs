pub mod budget;
pub mod epoch;
pub mod io;
pub mod memgate;
pub mod paths;
mod proc;
mod redact;
pub mod sieve;

pub use crate::net::backoff_ms;
#[cfg(test)]
use crate::net::{is_quota_exhausted, is_retryable, retry_ms};
pub use budget::{allocator_name, budget, collect_os_memory, init_allocator, under_memory_pressure};
pub use epoch::EpochRegistry;
pub use io::{SidecarLock, ensure_cachedir_tag, has_cachedir_tag};
pub use memgate::{MemGate, TangoCreditRing};
pub use paths::{
    check_rel, check_target_in_repo, normalize_rel, pwsh_utf8_preamble, rel_to_root,
    strip_utf8_bom, trim_crlf,
};
pub use proc::{run_cmd, self_metrics};
pub use redact::{find_secret, redact_secrets};
pub use sieve::SieveCache;

use crate::net::build_http;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);
static CLOCK_ANCHOR: OnceLock<(u64, Instant)> = OnceLock::new();

const SKIP_DIRS: &[&str] = &[
    ".refit", ".venv", "venv", ".git", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", ".tox", "target", "node_modules", ".tools",
    "dist", "build", ".next", ".nuxt", ".terraform", "vendor", "coverage",
    "out", "tmp", ".cache",
];

const SANDBOX_VARS: &[&str] = &[
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
    "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA",
    "APPDATA", "LANG", "LC_ALL", "RUSTUP_HOME", "CARGO_HOME", "GOPATH",
    "GOROOT", "GOCACHE", "VIRTUAL_ENV", "NODE_PATH", "RUSTC_WRAPPER",
    "SCCACHE_DIR", "CARGO_BUILD_RUSTC_WRAPPER",
];

pub fn os_tag() -> String {
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{}-{arch}", std::env::consts::OS)
}

#[inline]
pub fn now_ms() -> u64 {
    let (epoch_ms, started) = CLOCK_ANCHOR.get_or_init(|| {
        let base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        (base, Instant::now())
    });
    epoch_ms.saturating_add(started.elapsed().as_millis() as u64)
}

#[inline]
pub fn skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

pub fn walk_files(repo_root: &Path, max_depth: Option<usize>) -> Vec<PathBuf> {
    walk_tree(repo_root, max_depth, false)
}

pub fn walk_tree(
    repo_root: &Path,
    max_depth: Option<usize>,
    include_hidden_configs: bool,
) -> Vec<PathBuf> {
    let mut wb = WalkBuilder::new(repo_root);
    wb.hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .ignore(true)
        .parents(false)
        .require_git(false)
        .add_custom_ignore_filename(".refitignore")
        .filter_entry(move |e| keep_entry(e, include_hidden_configs));

    if let Some(d) = max_depth {
        wb.max_depth(Some(d));
    }

    let mut out: Vec<PathBuf> = wb
        .build()
        .flatten()
        .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
        .map(|e| e.into_path())
        .collect();
    out.sort();
    out
}

fn keep_entry(e: &ignore::DirEntry, include_hidden: bool) -> bool {
    if e.depth() == 0 || !e.file_type().is_some_and(|ft| ft.is_dir()) {
        return true;
    }

    let name = e.file_name().to_string_lossy();
    if skip_dir(&name) || has_cachedir_tag(e.path()) {
        return false;
    }
    if name.starts_with('.') {
        return include_hidden && name == ".github";
    }
    true
}

pub fn sandbox_env(extra: &[(String, String)]) -> Vec<(String, String)> {
    let mut env = Vec::with_capacity(SANDBOX_VARS.len() + extra.len() + 1);
    for &k in SANDBOX_VARS {
        if let Ok(v) = std::env::var(k) {
            env.push((k.to_string(), v));
        }
    }
    if std::env::var_os("MSYSTEM").is_some() {
        env.push(("MSYS_NO_PATHCONV".to_string(), "1".to_string()));
    }

    for (k, v) in extra {
        if let Some(slot) = env.iter_mut().find(|(ek, _)| ek.eq_ignore_ascii_case(k)) {
            slot.1 = v.clone();
        } else {
            env.push((k.clone(), v.clone()));
        }
    }
    env
}

pub fn char_prefix(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

pub fn new_http(user_agent: &str, timeout_secs: u64, pool_max: usize) -> Result<reqwest::Client> {
    build_http(user_agent, 15, 600, pool_max, Some(timeout_secs))
}

pub fn stream_http(
    user_agent: &str,
    connect_timeout_secs: u64,
    pool_idle_secs: u64,
    pool_max: usize,
) -> Result<reqwest::Client> {
    build_http(user_agent, connect_timeout_secs, pool_idle_secs, pool_max, None)
}

pub fn write_zstd<T: Serialize>(path: &Path, val: &T) -> Result<()> {
    io::write_zstd_stream(path, val)
}

pub fn read_zstd<T: DeserializeOwned>(path: &Path) -> Option<T> {
    io::read_zstd_stream(path)
}

pub fn mkdir_private(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("create private directory {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("chmod 0700 {}", path.display()))?;
    }
    Ok(())
}

pub fn write_atomic(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent dir {}", parent.display()))?;

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = target.file_name().and_then(|s| s.to_str()).unwrap_or("blob");
    let tmp_path = parent.join(format!(".refit.tmp.{}.{seq}.{name}", std::process::id()));

    let mut f = File::create(&tmp_path)
        .with_context(|| format!("create temp file {}", tmp_path.display()))?;
    f.write_all(bytes)?;
    io::sync_file_barrier(&f)?;
    drop(f);

    replace_file(&tmp_path, target, parent)
}

pub(crate) fn replace_file_pub(src_path: &Path, dst_path: &Path, parent: &Path) -> Result<()> {
    replace_file(src_path, dst_path, parent)
}

#[cfg(windows)]
fn wide_path(p: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    let mut v: Vec<u16> = p.as_os_str().encode_wide().collect();
    v.push(0);
    v
}

#[cfg(windows)]
fn replace_file(src_path: &Path, dst_path: &Path, _parent: &Path) -> Result<()> {
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    let src = wide_path(src_path);
    let dst = wide_path(dst_path);

    let mut last_err = None;
    for attempt in 0..5 {
        // SAFETY: `src` and `dst` are valid NUL-terminated UTF-16 buffers alive for this call.
        if unsafe { MoveFileExW(src.as_ptr(), dst.as_ptr(), flags) } != 0 {
            return Ok(());
        }
        last_err = Some(std::io::Error::last_os_error());
        std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
    }

    let _ = fs::remove_file(src_path);
    anyhow::bail!("MoveFileExW failed for {}: {:?}", dst_path.display(), last_err)
}

#[cfg(unix)]
fn replace_file(src_path: &Path, dst_path: &Path, parent: &Path) -> Result<()> {
    fs::rename(src_path, dst_path).with_context(|| {
        let _ = fs::remove_file(src_path);
        format!("atomic rename {} -> {}", src_path.display(), dst_path.display())
    })?;
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use std::time::Duration;

    #[test]
    fn test_http_retry_and_backoff() {
        assert!(new_http(concat!("refit-probe/", env!("CARGO_PKG_VERSION")), 30, 16).is_ok());
        assert!(stream_http(concat!("refit-stream/", env!("CARGO_PKG_VERSION")), 15, 600, 16).is_ok());

        for code in [200, 201, 204, 400, 401, 403, 404, 422, 432] {
            assert!(!is_retryable(code), "code={code}");
        }
        assert!(is_quota_exhausted(432) && !is_quota_exhausted(429));
        for code in [408, 429, 500, 502, 503, 504, 529] {
            assert!(is_retryable(code), "code={code}");
        }

        for (raw, want) in [("3", Some(3_000)), ("1.5", Some(1_500)), ("9999", Some(30_000)), ("-5", None), ("invalid", None)] {
            assert_eq!(retry_ms(raw, 30_000), want, "raw={raw}");
        }
        let soon = httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(5));
        assert!((2_000..=6_500).contains(&retry_ms(&soon, 30_000).unwrap()));
        let far = httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(3600));
        assert_eq!(retry_ms(&far, 15_000), Some(15_000));

        for (att, lo, hi) in [(0, 100, 200), (2, 400, 800), (9, 5_000, 10_000)] {
            assert!((lo..=hi).contains(&backoff_ms(att, 200, 10_000, None, 1)));
        }
        assert_eq!(backoff_ms(0, 200, 10_000, Some(4_500), 1), 4_500);
        assert_ne!(backoff_ms(2, 200, 10_000, None, 11), backoff_ms(2, 200, 10_000, None, 97));
    }

    #[test]
    fn test_atomic_io_clock_and_cmd() {
        let dir = TempRepo::new("atomic_io");
        let target = dir.join("state.json");
        for i in 0..25 {
            let payload = format!("{{\"rev\":{i}}}");
            write_atomic(&target, payload.as_bytes()).unwrap();
            assert_eq!(dir.read("state.json"), payload);
        }

        let mut prev = now_ms();
        for _ in 0..10_000 {
            let cur = now_ms();
            assert!(cur >= prev);
            prev = cur;
        }

        let cwd = std::env::temp_dir();
        let (prog, flag, slow_cmd) = if cfg!(windows) {
            ("cmd", "/C", "ping -n 5 127.0.0.1 >nul")
        } else {
            ("sh", "-c", "sleep 3")
        };
        let run = |script: &str, ms: u64| run_cmd(prog, &[flag.into(), script.into()], &cwd, &[], false, ms).unwrap();
        let ok = run("echo ok", 5_000);
        assert!(ok.exit_code == 0 && !ok.timed_out && ok.stdout.to_lowercase().contains("ok"));
        let slow = run(slow_cmd, 150);
        assert!(slow.timed_out && slow.exit_code == -1);
    }

    #[test]
    fn test_walk_pruning_and_ignores() {
        let dir = TempRepo::with_files(
            "walk_all",
            &[
                (".gitignore", "ignored_gen/\n*.generated.py\n"),
                (".refitignore", "custom_skip/\n"),
                ("src/keep.rs", "pub fn keep() {}\n"),
                (".git/HEAD", "ref: refs/heads/main\n"),
                ("node_modules/pkg/index.js", "module.exports = 1;\n"),
                ("target/debug/bin.rs", "fn main() {}\n"),
                (".refit/cas/blob", "x"),
                ("ignored_gen/junk.py", "x = 1\n"),
                ("src/auto.generated.py", "x = 2\n"),
                ("custom_skip/skip.ts", "export const x = 1;\n"),
            ],
        );

        let rels: Vec<String> = walk_files(dir.path(), None)
            .iter()
            .filter_map(|p| crate::source::rel_norm(dir.path(), p))
            .collect();
        assert_eq!(rels, vec![".gitignore", ".refitignore", "src/keep.rs"]);
    }

    #[test]
    fn test_secret_detection_and_redaction() {
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        let gh = ["ghp_", "1234567890abcdefghij123456"].concat();
        let db = ["postgres://", "admin:sup3rs3cr3t@db.internal:5432/prod"].concat();
        let slack = ["xoxa-", "1234567890abcdef1234"].concat();
        let pem = [
            "-----BEGIN RSA ", "PRIVATE KEY-----\nMIIEpAIBAAKCAQEASuperSecretBodyPayload==\n",
            "-----END RSA PRIVATE KEY-----",
        ]
        .concat();
        let multi_uri = "docs = 'postgres://localhost/db' prod = 'postgres://admin:p4ssw0rd@db.prod:5432/app'";

        let cases = [
            (format!("key = \"{aws}\""), Some("aws_access_key_id")),
            (format!("tok = '{gh}'"), Some("github_pat")),
            (db.clone(), Some("database_credential_url")),
            (pem.clone(), Some("pem_private_key")),
            (multi_uri.into(), Some("database_credential_url")),
            ("let normal = \"hello_world\";".into(), None),
        ];
        for (sample, want) in &cases {
            assert_eq!(find_secret(sample), *want, "sample={sample}");
        }

        let red = redact_secrets(format!("a={aws} b={gh} c={db} s={slack} p={pem}"));
        for leak in [&aws, &gh, &slack, &"sup3rs3cr3t".to_string(), &"SuperSecretBodyPayload".to_string()] {
            assert!(!red.contains(leak.as_str()), "leaked {leak}");
        }
        assert!(red.contains("REDACTED_AWS_") && red.contains("REDACTED_GITHUB_") && red.contains("REDACTED_SLACK_"));
    }
}
