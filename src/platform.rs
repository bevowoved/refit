use crate::types::{SandboxResult, SelfMetrics};
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);
static CLOCK_ANCHOR: OnceLock<(u64, Instant)> = OnceLock::new();

pub fn os_tag() -> String {
    let os = match std::env::consts::OS {
        "macos" => "macos",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{os}-{arch}")
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
    matches!(
        name,
        ".refit"
            | ".venv"
            | "venv"
            | ".git"
            | "__pycache__"
            | ".pytest_cache"
            | ".ruff_cache"
            | ".mypy_cache"
            | ".tox"
            | "target"
            | "node_modules"
            | ".tools"
            | "dist"
            | "build"
            | ".next"
            | ".nuxt"
            | ".terraform"
            | "vendor"
            | "coverage"
            | "out"
            | "tmp"
            | ".cache"
    )
}

/// Check that `rel` stays inside the repository and never targets `.git` or `.refit`.
pub fn check_rel(rel: &str) -> bool {
    if rel.is_empty() || rel.contains('\0') {
        return false;
    }
    let norm = rel.replace('\\', "/");
    let p = Path::new(&norm);
    if p.is_absolute() {
        return false;
    }
    let mut has_normal = false;
    for comp in p.components() {
        match comp {
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return false,
            std::path::Component::Normal(seg) => {
                let s = seg.to_string_lossy();
                if s == ".git" || s == ".refit" {
                    return false;
                }
                has_normal = true;
            }
            std::path::Component::CurDir => {}
        }
    }
    has_normal
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
        .filter_entry(move |e| {
            if e.depth() == 0 {
                return true;
            }
            let is_dir = e.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
            if !is_dir {
                return true;
            }
            let name = e.file_name().to_string_lossy();
            if skip_dir(&name) {
                return false;
            }
            if name.starts_with('.') {
                return include_hidden_configs && name == ".github";
            }
            true
        });
    if let Some(d) = max_depth {
        wb.max_depth(Some(d));
    }
    let mut out: Vec<PathBuf> = wb
        .build()
        .flatten()
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .map(|e| e.into_path())
        .collect();
    out.sort();
    out
}

/// Build a minimal environment allowlist for sandboxed commands (`clear_env = true`).
pub fn sandbox_env(extra: &[(String, String)]) -> Vec<(String, String)> {
    const ALLOWED_VARS: &[&str] = &[
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
        "LANG",
        "LC_ALL",
        "RUSTUP_HOME",
        "CARGO_HOME",
        "GOPATH",
        "GOROOT",
        "GOCACHE",
        "VIRTUAL_ENV",
        "NODE_PATH",
    ];
    let mut env: Vec<(String, String)> = Vec::with_capacity(ALLOWED_VARS.len() + extra.len());
    for &k in ALLOWED_VARS {
        if let Ok(v) = std::env::var(k) {
            env.push((k.to_string(), v));
        }
    }
    for (k, v) in extra {
        if let Some(existing) = env.iter_mut().find(|(ek, _)| ek.eq_ignore_ascii_case(k)) {
            existing.1 = v.clone();
        } else {
            env.push((k.clone(), v.clone()));
        }
    }
    env
}

/// Detect hardcoded secret tokens, PEM keys, or credentialed DB URIs in a line.
pub fn find_secret(line: &str) -> Option<&'static str> {
    if line.contains("-----BEGIN ") && line.contains("PRIVATE KEY-----") {
        return Some("pem_private_key");
    }
    const PREFIX_RULES: &[(&str, usize, &str)] = &[
        ("csk-", 12, "cerebras_api_key"),
        ("keen_", 12, "keenable_api_key"),
        ("sk-ant-", 16, "anthropic_api_key"),
        ("sk-proj-", 16, "openai_api_key"),
        ("sk_live_", 16, "stripe_live_key"),
        ("rk_live_", 16, "stripe_restricted_key"),
        ("ghp_", 20, "github_pat"),
        ("gho_", 20, "github_oauth"),
        ("github_pat_", 20, "github_fine_grained_pat"),
        ("xoxb-", 16, "slack_bot_token"),
        ("xoxp-", 16, "slack_user_token"),
        ("xoxa-", 16, "slack_app_token"),
        ("AKIA", 16, "aws_access_key_id"),
        ("ASIA", 16, "aws_sts_key_id"),
        ("sk-", 24, "generic_sk_secret"),
    ];
    for &(prefix, min_tail, kind) in PREFIX_RULES {
        let mut rest = line;
        while let Some(pos) = rest.find(prefix) {
            let valid_boundary = pos == 0
                || !rest[..pos]
                    .chars()
                    .next_back()
                    .map(|c| c.is_ascii_alphanumeric() || c == '_')
                    .unwrap_or(false);
            let tail = &rest[pos + prefix.len()..];
            if valid_boundary {
                let token_len = if prefix == "AKIA" || prefix == "ASIA" {
                    tail.chars()
                        .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                        .count()
                } else {
                    tail.chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                        .count()
                };
                if token_len >= min_tail {
                    return Some(kind);
                }
            }
            rest = tail;
        }
    }
    for scheme in [
        "postgres://",
        "postgresql://",
        "mysql://",
        "mongodb://",
        "mongodb+srv://",
        "redis://",
    ] {
        let mut rest = line;
        while let Some(pos) = rest.find(scheme) {
            let after = &rest[pos + scheme.len()..];
            let authority = after
                .split(['/', ' ', '"', '\'', ')', ']'])
                .next()
                .unwrap_or("");
            if let Some((creds, host)) = authority.split_once('@') {
                if !host.is_empty()
                    && creds.contains(':')
                    && !creds.ends_with(':')
                    && !creds.contains("${")
                    && !creds.contains("REDACTED")
                {
                    return Some("database_credential_url");
                }
            }
            rest = after;
        }
    }
    None
}

pub fn redact_secrets(mut s: String) -> String {
    const REDACT_PREFIXES: &[(&str, &str, usize)] = &[
        ("csk-", "REDACTED_CSK_", 8),
        ("keen_", "REDACTED_KEEN_", 8),
        ("sk-ant-", "REDACTED_ANTHROPIC_", 12),
        ("sk-proj-", "REDACTED_OPENAI_", 12),
        ("sk_live_", "REDACTED_STRIPE_", 12),
        ("rk_live_", "REDACTED_STRIPE_", 12),
        ("ghp_", "REDACTED_GITHUB_", 16),
        ("gho_", "REDACTED_GITHUB_", 16),
        ("github_pat_", "REDACTED_GITHUB_", 16),
        ("xoxb-", "REDACTED_SLACK_", 12),
        ("xoxp-", "REDACTED_SLACK_", 12),
        ("xoxa-", "REDACTED_SLACK_", 12),
        ("AKIA", "REDACTED_AWS_", 16),
        ("ASIA", "REDACTED_AWS_", 16),
        ("sk-", "REDACTED_SK_", 24),
    ];
    for &(prefix, replacement, min_tail) in REDACT_PREFIXES {
        if s.contains(prefix) {
            s = redact_prefix(&s, prefix, replacement, min_tail);
        }
    }
    for scheme in [
        "postgres://",
        "postgresql://",
        "mysql://",
        "mongodb://",
        "mongodb+srv://",
        "redis://",
    ] {
        if s.contains(scheme) {
            s = redact_uris(&s, scheme);
        }
    }
    if s.contains("PRIVATE KEY-----") {
        s = redact_pem(&s);
    }
    s
}

fn redact_pem(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(begin) = rest.find("-----BEGIN ") {
        let after_begin = &rest[begin..];
        let Some(hdr_end_rel) = after_begin.find("PRIVATE KEY-----") else {
            out.push_str(&rest[..begin + 11]);
            rest = &rest[begin + 11..];
            continue;
        };
        out.push_str(&rest[..begin]);
        out.push_str("[REDACTED_PEM_PRIVATE_KEY]");
        let body_start = begin + hdr_end_rel + "PRIVATE KEY-----".len();
        let tail = &rest[body_start..];
        if let Some(end_rel) = tail.find("-----END ") {
            let after_end = &tail[end_rel..];
            if let Some(close_rel) = after_end.find("PRIVATE KEY-----") {
                rest = &tail[end_rel + close_rel + "PRIVATE KEY-----".len()..];
                continue;
            }
        }
        rest = "";
        break;
    }
    out.push_str(rest);
    out.replace("PRIVATE KEY-----", "REDACTED_PRIVATE_KEY-----")
}

fn redact_prefix(input: &str, prefix: &str, replacement: &str, min_tail: usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find(prefix) {
        let valid_boundary = pos == 0
            || !rest[..pos]
                .chars()
                .next_back()
                .map(|c| c.is_ascii_alphanumeric() || c == '_')
                .unwrap_or(false);
        let after = &rest[pos + prefix.len()..];
        let skip = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
            .count();
        if valid_boundary && skip >= min_tail {
            out.push_str(&rest[..pos]);
            out.push_str(replacement);
            rest = &after[skip..];
        } else {
            out.push_str(&rest[..pos + prefix.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn redact_uris(input: &str, scheme: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find(scheme) {
        out.push_str(&rest[..pos + scheme.len()]);
        let after = &rest[pos + scheme.len()..];
        let end = after
            .find(['/', ' ', '"', '\'', ')', ']', '\n', '\r'])
            .unwrap_or(after.len());
        let auth = &after[..end];
        if let Some((creds, host)) = auth.split_once('@') {
            if creds.contains(':') && !host.is_empty() {
                out.push_str("REDACTED:REDACTED@");
                out.push_str(host);
                rest = &after[end..];
                continue;
            }
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

pub fn char_prefix(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

pub fn new_http(user_agent: &str, timeout_secs: u64, pool_max: usize) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(user_agent)
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .connect_timeout(Duration::from_secs(15))
        .pool_max_idle_per_host(pool_max)
        .pool_idle_timeout(Some(Duration::from_secs(600)))
        .http2_keep_alive_interval(Some(Duration::from_secs(15)))
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .http2_adaptive_window(true)
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .context("build HTTP/2 keep-alive client")
}

pub fn stream_http(
    user_agent: &str,
    connect_timeout_secs: u64,
    pool_idle_secs: u64,
    pool_max: usize,
) -> Result<reqwest::Client> {
    // No global total-request timeout: long-lived SSE streams are guarded by per-chunk stall watchdogs.
    reqwest::Client::builder()
        .user_agent(user_agent)
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .pool_max_idle_per_host(pool_max)
        .pool_idle_timeout(Some(Duration::from_secs(pool_idle_secs)))
        .http2_keep_alive_interval(Some(Duration::from_secs(15)))
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .http2_adaptive_window(true)
        .build()
        .context("build HTTP/2 streaming keep-alive client")
}

#[inline]
pub fn is_retryable(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
}

pub fn retry_ms(header_val: &str, max_cap_ms: u64) -> Option<u64> {
    let trimmed = header_val.trim();
    if let Ok(secs) = trimmed.parse::<f64>() {
        if secs.is_finite() && secs >= 0.0 {
            let ms = (secs * 1000.0).round() as u64;
            return Some(ms.min(max_cap_ms));
        }
    }
    if let Ok(target) = httpdate::parse_http_date(trimmed) {
        let wait_ms = target
            .duration_since(SystemTime::now())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        return Some(wait_ms.min(max_cap_ms));
    }
    None
}

pub fn backoff_ms(
    attempt: u32,
    base_ms: u64,
    cap_ms: u64,
    retry_after_ms: Option<u64>,
    jitter_seed: u64,
) -> u64 {
    if let Some(ra) = retry_after_ms {
        return ra.min(cap_ms);
    }
    let exp = base_ms
        .saturating_mul(1u64.checked_shl(attempt.min(10)).unwrap_or(1024))
        .min(cap_ms);
    let half = exp / 2;
    let span = (exp - half).max(1);
    let mix = jitter_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(attempt as u64);
    half + (mix % span)
}

pub fn write_zstd<T: Serialize>(path: &Path, val: &T) -> Result<()> {
    let raw = serde_json::to_vec(val)?;
    let compressed = zstd::stream::encode_all(raw.as_slice(), 3)?;
    write_atomic(path, &compressed)
}

pub fn read_zstd<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let raw = fs::read(path).ok()?;
    let decoded = zstd::stream::decode_all(raw.as_slice()).ok()?;
    serde_json::from_slice(&decoded).ok()
}

pub fn mkdir_private(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("create private directory {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o700);
        fs::set_permissions(path, perms)
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
    let pid = std::process::id();
    let tmp_name = format!(
        ".refit.tmp.{pid}.{seq}.{}",
        target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("blob")
    );
    let tmp_path = parent.join(tmp_name);

    {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("create temp file {}", tmp_path.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write temp file {}", tmp_path.display()))?;
        f.sync_data()
            .with_context(|| format!("sync temp file {}", tmp_path.display()))?;
    }

    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{
            MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
        };
        let src: Vec<u16> = tmp_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let dst: Vec<u16> = target
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut last_err = None;
        for attempt in 0..5 {
            let ok = unsafe {
                MoveFileExW(
                    src.as_ptr(),
                    dst.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            };
            if ok != 0 {
                return Ok(());
            }
            last_err = Some(std::io::Error::last_os_error());
            std::thread::sleep(Duration::from_millis(10 * (attempt + 1)));
        }
        let _ = fs::remove_file(&tmp_path);
        anyhow::bail!(
            "MoveFileExW failed for {}: {:?}",
            target.display(),
            last_err
        );
    }

    #[cfg(unix)]
    {
        fs::rename(&tmp_path, target).with_context(|| {
            let _ = fs::remove_file(&tmp_path);
            format!("atomic rename {} -> {}", tmp_path.display(), target.display())
        })?;
        // Sync parent dir on Unix so rename directory entry survives sudden power loss.
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

#[cfg(windows)]
pub fn self_metrics() -> SelfMetrics {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    unsafe {
        let proc_h = GetCurrentProcess();
        let mut pmc: PROCESS_MEMORY_COUNTERS = zeroed();
        pmc.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let rss_bytes = if GetProcessMemoryInfo(proc_h, &mut pmc, pmc.cb) != 0 {
            pmc.WorkingSetSize as u64
        } else {
            0
        };

        let mut creation: FILETIME = zeroed();
        let mut exit: FILETIME = zeroed();
        let mut kernel: FILETIME = zeroed();
        let mut user: FILETIME = zeroed();
        let cpu_ms = if GetProcessTimes(proc_h, &mut creation, &mut exit, &mut kernel, &mut user) != 0 {
            let k = ((kernel.dwHighDateTime as u64) << 32) | (kernel.dwLowDateTime as u64);
            let u = ((user.dwHighDateTime as u64) << 32) | (user.dwLowDateTime as u64);
            (k + u) / 10_000
        } else {
            0
        };

        SelfMetrics { rss_bytes, cpu_ms }
    }
}

#[cfg(target_os = "linux")]
pub fn self_metrics() -> SelfMetrics {
    let rss_bytes = fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0);
    let cpu_ms = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let ms = (ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000;
            ms.max(0) as u64
        })
        .unwrap_or(0);
    SelfMetrics { rss_bytes, cpu_ms }
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn self_metrics() -> SelfMetrics {
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let ms = (ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000;
            SelfMetrics {
                rss_bytes: u.max_rss().max(0) as u64,
                cpu_ms: ms.max(0) as u64,
            }
        })
        .unwrap_or_default()
}

pub fn run_cmd(
    program: &str,
    args: &[String],
    cwd: &Path,
    env_overrides: &[(String, String)],
    clear_env: bool,
    timeout_ms: u64,
) -> Result<SandboxResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if clear_env {
        cmd.env_clear();
    }
    for (k, v) in env_overrides {
        cmd.env(k, v);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                let _ = nix::unistd::setpgid(
                    nix::unistd::Pid::from_raw(0),
                    nix::unistd::Pid::from_raw(0),
                );
                Ok(())
            });
        }
    }

    #[cfg(unix)]
    let (unix_cpu_before, _) = unix_rusage();

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn sandboxed command `{program}` in {}", cwd.display()))?;

    #[cfg(windows)]
    let job_handle = attach_job(&child);

    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();

    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = out_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = err_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    });

    // Event-driven wait_timeout (WaitForSingleObject on Windows / SIGCHLD on Unix): zero 10ms polling waste.
    let deadline = Duration::from_millis(timeout_ms.max(100));
    let mut timed_out = false;
    let exit_code = match wait_timeout::ChildExt::wait_timeout(&mut child, deadline)? {
        Some(status) => status.code().unwrap_or(-1),
        None => {
            timed_out = true;
            #[cfg(windows)]
            {
                if let Some(job) = job_handle {
                    unsafe {
                        windows_sys::Win32::System::JobObjects::TerminateJobObject(job, 1);
                    }
                }
                let _ = child.kill();
            }
            #[cfg(unix)]
            {
                let pgid = nix::unistd::Pid::from_raw(child.id() as i32);
                let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
                if wait_timeout::ChildExt::wait_timeout(&mut child, Duration::from_millis(100))
                    .ok()
                    .flatten()
                    .is_none()
                {
                    let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
                    let _ = child.kill();
                }
            }
            let _ = child.wait();
            -1
        }
    };

    let wall_ms = started.elapsed().as_millis() as u64;
    let (cpu_ms, peak_rss_bytes) = {
        #[cfg(windows)]
        {
            close_job(job_handle)
        }
        #[cfg(unix)]
        {
            let (after_cpu, rss) = unix_rusage();
            (after_cpu.saturating_sub(unix_cpu_before), rss)
        }
    };

    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();

    Ok(SandboxResult {
        exit_code,
        timed_out,
        stdout,
        stderr,
        wall_ms,
        cpu_ms,
        peak_rss_bytes,
    })
}

#[cfg(windows)]
fn attach_job(child: &std::process::Child) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use std::mem::{size_of, zeroed};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            CloseHandle(job);
            return None;
        }
        let proc_h = child.as_raw_handle();
        if AssignProcessToJobObject(job, proc_h) == 0 {
            CloseHandle(job);
            return None;
        }
        Some(job)
    }
}

#[cfg(windows)]
fn close_job(
    job_handle: Option<windows_sys::Win32::Foundation::HANDLE>,
) -> (u64, u64) {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };

    let Some(job) = job_handle else {
        return (0, 0);
    };
    unsafe {
        let mut acct: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = zeroed();
        let cpu_ms = if QueryInformationJobObject(
            job,
            JobObjectBasicAccountingInformation,
            &mut acct as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
            std::ptr::null_mut(),
        ) != 0
        {
            let ticks = (acct.TotalUserTime + acct.TotalKernelTime).max(0) as u64;
            ticks / 10_000
        } else {
            0
        };

        let mut ext: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        let peak_rss = if QueryInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &mut ext as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        ) != 0
        {
            ext.PeakJobMemoryUsed as u64
        } else {
            0
        };

        CloseHandle(job);
        (cpu_ms, peak_rss)
    }
}

#[cfg(unix)]
fn unix_rusage() -> (u64, u64) {
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_CHILDREN)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let cpu_ms = ((ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000)
                .max(0) as u64;
            let raw_rss = u.max_rss().max(0) as u64;
            #[cfg(target_os = "linux")]
            let rss_bytes = raw_rss * 1024;
            #[cfg(not(target_os = "linux"))]
            let rss_bytes = raw_rss;
            (cpu_ms, rss_bytes)
        })
        .unwrap_or((0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_08_http_client_h2() {
        let client = new_http("refit-probe/0.4.0", 30, 16);
        assert!(client.is_ok());
    }

    #[test]
    fn probe_09_retry_status() {
        for ok_code in [200, 201, 204, 400, 401, 403, 404, 422] {
            assert!(!is_retryable(ok_code), "must not retry {ok_code}");
        }
        for retry_code in [408, 429, 500, 502, 503, 504, 529] {
            assert!(is_retryable(retry_code), "must retry {retry_code}");
        }
    }

    #[test]
    fn probe_10_retry_after_cap() {
        assert_eq!(retry_ms("3", 30_000), Some(3_000));
        assert_eq!(retry_ms("1.5", 30_000), Some(1_500));
        assert_eq!(retry_ms("9999", 30_000), Some(30_000));
        assert_eq!(retry_ms("-5", 30_000), None);
        assert_eq!(retry_ms("invalid", 30_000), None);
    }

    #[test]
    fn probe_11_retry_backoff() {
        let d0 = backoff_ms(0, 200, 10_000, None, 1);
        let d2 = backoff_ms(2, 200, 10_000, None, 1);
        let d9 = backoff_ms(9, 200, 10_000, None, 1);
        let ra = backoff_ms(0, 200, 10_000, Some(4_500), 1);
        assert!((100..=200).contains(&d0));
        assert!((400..=800).contains(&d2));
        assert!((5_000..=10_000).contains(&d9));
        assert_eq!(ra, 4_500);
        let s1 = backoff_ms(2, 200, 10_000, None, 11);
        let s2 = backoff_ms(2, 200, 10_000, None, 97);
        assert_ne!(s1, s2);
    }

    #[test]
    fn probe_40_write_atomic() {
        let dir = std::env::temp_dir().join(format!("refit_atomic_{}_{}", std::process::id(), now_ms()));
        let target = dir.join("state.json");
        for i in 0..25 {
            let payload = format!("{{\"rev\":{i}}}");
            write_atomic(&target, payload.as_bytes()).unwrap();
            assert_eq!(fs::read_to_string(&target).unwrap(), payload);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_47_walk_prune() {
        let dir = std::env::temp_dir().join(format!("refit_walk_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::create_dir_all(dir.join("node_modules/pkg")).unwrap();
        fs::create_dir_all(dir.join("target/debug")).unwrap();
        fs::create_dir_all(dir.join(".refit/cas")).unwrap();
        fs::write(dir.join("src/a.rs"), "pub fn a() {}\n").unwrap();
        fs::write(dir.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(dir.join("node_modules/pkg/index.js"), "module.exports = 1;\n").unwrap();
        fs::write(dir.join("target/debug/bin.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.join(".refit/cas/blob"), "x").unwrap();

        let files = walk_files(&dir, None);
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("a.rs"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_48_monotonic_ms() {
        let mut prev = now_ms();
        for _ in 0..10_000 {
            let cur = now_ms();
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    fn probe_50_cmd_timeout() {
        let cwd = std::env::temp_dir();
        #[cfg(windows)]
        let (prog, ok_args, slow_args) = (
            "cmd",
            vec!["/C".into(), "echo ok".into()],
            vec!["/C".into(), "ping -n 5 127.0.0.1 >nul".into()],
        );
        #[cfg(unix)]
        let (prog, ok_args, slow_args) = (
            "sh",
            vec!["-c".into(), "echo ok".into()],
            vec!["-c".into(), "sleep 3".into()],
        );

        let r_ok = run_cmd(prog, &ok_args, &cwd, &[], false, 5_000).unwrap();
        assert_eq!(r_ok.exit_code, 0);
        assert!(!r_ok.timed_out);
        assert!(r_ok.stdout.to_lowercase().contains("ok"));

        let r_to = run_cmd(prog, &slow_args, &cwd, &[], false, 150).unwrap();
        assert!(r_to.timed_out);
        assert_eq!(r_to.exit_code, -1);
    }

    #[test]
    fn probe_64_retry_http_date() {
        let future = SystemTime::now() + Duration::from_secs(5);
        let http_date = httpdate::fmt_http_date(future);
        let parsed = retry_ms(&http_date, 30_000).unwrap();
        assert!((2_000..=6_500).contains(&parsed), "expected ~5s, got {parsed}ms");

        let far = SystemTime::now() + Duration::from_secs(3600);
        assert_eq!(retry_ms(&httpdate::fmt_http_date(far), 15_000), Some(15_000));
        assert!(stream_http("refit-stream/0.4.0", 15, 600, 16).is_ok());
    }

    #[test]
    fn probe_76_walk_ignore() {
        let dir = std::env::temp_dir().join(format!("refit_gi76_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("ignored_gen")).unwrap();
        fs::create_dir_all(dir.join("custom_skip")).unwrap();
        fs::write(dir.join(".gitignore"), "ignored_gen/\n*.generated.py\n").unwrap();
        fs::write(dir.join(".refitignore"), "custom_skip/\n").unwrap();
        fs::write(dir.join("src/keep.rs"), "pub fn keep() {}\n").unwrap();
        fs::write(dir.join("ignored_gen/junk.py"), "x = 1\n").unwrap();
        fs::write(dir.join("src/auto.generated.py"), "x = 2\n").unwrap();
        fs::write(dir.join("custom_skip/skip.ts"), "export const x = 1;\n").unwrap();

        let files = walk_files(&dir, None);
        let rels: Vec<String> = files
            .iter()
            .filter_map(|p| p.strip_prefix(&dir).ok())
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .collect();
        assert!(rels.contains(&"src/keep.rs".to_string()), "rels={rels:?}");
        assert!(!rels.iter().any(|r| r.contains("ignored_gen")), "rels={rels:?}");
        assert!(!rels.iter().any(|r| r.contains("generated.py")), "rels={rels:?}");
        assert!(!rels.iter().any(|r| r.contains("custom_skip")), "rels={rels:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_77_secret_families() {
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        let gh = ["ghp_", "1234567890abcdefghij123456"].concat();
        let db = ["postgres://", "admin:sup3rs3cr3t@db.internal:5432/prod"].concat();
        let pem = ["-----BEGIN RSA ", "PRIVATE KEY-----"].concat();
        assert_eq!(find_secret(&format!("key = \"{aws}\"")), Some("aws_access_key_id"));
        assert_eq!(find_secret(&format!("tok = '{gh}'")), Some("github_pat"));
        assert_eq!(find_secret(&db), Some("database_credential_url"));
        assert_eq!(find_secret(&pem), Some("pem_private_key"));
        assert_eq!(find_secret("let normal = \"hello_world\";"), None);

        let redacted = redact_secrets(format!("a={aws} b={gh} c={db}"));
        assert!(!redacted.contains(&aws));
        assert!(!redacted.contains(&gh));
        assert!(!redacted.contains("sup3rs3cr3t"));
        assert!(redacted.contains("REDACTED_AWS_"));
        assert!(redacted.contains("REDACTED_GITHUB_"));
    }

    #[test]
    fn probe_99_pem_and_uri() {
        let slack_app = ["xoxa-", "1234567890abcdef1234"].concat();
        let pem_full = [
            "-----BEGIN RSA ",
            "PRIVATE KEY-----\nMIIEpAIBAAKCAQEASuperSecretBodyPayload==\n-----END RSA ",
            "PRIVATE KEY-----",
        ]
        .concat();
        let red = redact_secrets(format!("t={slack_app} pem={pem_full}"));
        assert!(!red.contains("SuperSecretBodyPayload"), "PEM body leaked: {red}");
        assert!(!red.contains(&slack_app), "xoxa- token leaked: {red}");
        assert!(red.contains("REDACTED_SLACK_"));

        // Non-credentialed URI earlier on same line must NOT hide a credentialed URI later on the line.
        let line = "docs = 'postgres://localhost/db' prod = 'postgres://admin:p4ssw0rd@db.prod:5432/app'";
        assert_eq!(find_secret(line), Some("database_credential_url"));
    }
}
