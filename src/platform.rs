pub mod budget;
pub mod epoch;
pub mod memgate;
pub mod paths;
mod proc;
mod redact;
pub mod rewind;
pub mod sieve;

pub use crate::net::backoff_ms;
#[cfg(test)]
use crate::net::{is_quota_exhausted, is_retryable, retry_ms};
pub use budget::{allocator_name, budget, collect_os_memory, init_allocator, under_memory_pressure};
pub use epoch::EpochRegistry;
pub use memgate::{MemGate, TangoCreditRing};
pub use paths::{
    check_rel, check_target_in_repo, normalize_rel, pwsh_utf8_preamble, rel_to_root,
    strip_utf8_bom, trim_crlf,
};
pub use proc::{run_cmd, self_metrics};
pub use redact::{find_secret, redact_secrets};
pub use rewind::{
    Durability, InvertedRewindIndex, MIN_KERNEL_COST_US, RewindEvent, RewindTrigger,
    should_use_index,
};
pub use sieve::SieveCache;

use crate::net::build_http;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const FILE_BUF_SIZE: usize = 64 * 1024;
#[cfg(target_os = "linux")]
const FADV_SEQ_MIN: u64 = 64 * 1024;
#[cfg(target_os = "linux")]
const FADV_DONTNEED_MIN: u64 = 1024 * 1024;
const ZSTD_STREAM_WLOG: u32 = 18;
const STAT_MAGIC: &[u8; 8] = b"RFST0002";
const STAT_MAX_DECODE_BYTES: u64 = 64 * 1024 * 1024;
const RACY_GIT_NS: u64 = 2_000_000_000;
const CACHEDIR_TAG_BODY: &[u8] = b"Signature: 8a477f597d28d172789f06886806bc55\n";

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

/// Reusable O(1) 64 KiB scratch buffer for streaming file I/O.
pub struct FileBuf {
    buf: Vec<u8>,
}

impl FileBuf {
    pub fn new() -> Self {
        Self { buf: vec![0u8; FILE_BUF_SIZE] }
    }

    pub fn hash_file_blake3(&mut self, path: &Path) -> Result<(u64, String)> {
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
        advise_sequential_if_medium(&f, file_len);

        let mut hasher = blake3::Hasher::new();
        let mut total = 0u64;
        loop {
            let n = f.read(&mut self.buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&self.buf[..n]);
            total += n as u64;
        }
        advise_dontneed_if_large(&f, total);
        Ok((total, hasher.finalize().to_hex().to_string()))
    }
}

impl Default for FileBuf {
    fn default() -> Self {
        Self::new()
    }
}

/// Write the standard 43-byte `CACHEDIR.TAG` marker inside `dir`.
pub fn ensure_cachedir_tag(dir: &Path) {
    let tag = dir.join("CACHEDIR.TAG");
    if tag.exists() {
        return;
    }
    let _ = fs::create_dir_all(dir);
    let _ = fs::write(&tag, CACHEDIR_TAG_BODY);
}

#[inline]
pub fn has_cachedir_tag(dir: &Path) -> bool {
    let tag = dir.join("CACHEDIR.TAG");
    let Ok(mut f) = File::open(&tag) else { return false };
    let mut head = [0u8; 43];
    f.read_exact(&mut head).is_ok() && head.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55")
}

fn advise_sequential_if_medium(file: &File, len: u64) {
    #[cfg(target_os = "linux")]
    if (FADV_SEQ_MIN..FADV_DONTNEED_MIN).contains(&len) {
        let _ = nix::fcntl::posix_fadvise(
            file,
            0,
            0,
            nix::fcntl::PosixFadviseAdvice::POSIX_FADV_SEQUENTIAL,
        );
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, len);
}

fn advise_dontneed_if_large(file: &File, len: u64) {
    #[cfg(target_os = "linux")]
    if len >= FADV_DONTNEED_MIN {
        let _ = nix::fcntl::posix_fadvise(
            file,
            0,
            0,
            nix::fcntl::PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        );
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, len);
}

/// Cross-OS durability barrier (`fdatasync` on Linux, `F_FULLFSYNC` on macOS, `sync_data` on Win).
pub fn sync_file_barrier(file: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    if nix::fcntl::fcntl(file, nix::fcntl::FcntlArg::F_FULLFSYNC).is_ok() {
        return Ok(());
    }
    file.sync_data()
}

fn write_atomic_with<T>(target: &Path, write_fn: impl FnOnce(&mut File) -> Result<T>) -> Result<T> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent dir {}", parent.display()))?;

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nonce = u32::from_le_bytes(os_random_bytes::<4>().unwrap_or([0; 4]));
    let name = target.file_name().and_then(|s| s.to_str()).unwrap_or("blob");
    let tmp_path = parent.join(format!(".refit.tmp.{}.{seq}.{nonce:08x}.{name}", std::process::id()));

    let outcome = (|| -> Result<T> {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("create temp file {}", tmp_path.display()))?;
        let res = write_fn(&mut f)?;
        sync_file_barrier(&f)?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        advise_dontneed_if_large(&f, len);
        Ok(res)
    })();

    match outcome {
        Ok(val) => {
            replace_file(&tmp_path, target, parent)?;
            Ok(val)
        }
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            Err(err)
        }
    }
}

pub fn write_zstd<T: Serialize>(path: &Path, val: &T) -> Result<()> {
    write_atomic_with(path, |f| {
        let mut enc = zstd::stream::Encoder::new(f, 1)?;
        let _ = enc.window_log(ZSTD_STREAM_WLOG);
        serde_json::to_writer(&mut enc, val)?;
        enc.finish()?;
        Ok(())
    })
}

pub fn read_zstd<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let f = File::open(path).ok()?;
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    advise_sequential_if_medium(&f, len);
    let dec = zstd::stream::Decoder::new(&f).ok()?;
    let out = serde_json::from_reader(dec).ok();
    advise_dontneed_if_large(&f, len);
    out
}

/// Stream-compress an existing file on disk into `dst` without loading the whole file in RAM.
pub fn compress_file_zstd(src: &Path, dst: &Path, level: i32) -> Result<(u64, u64)> {
    write_atomic_with(dst, |writer| {
        let mut reader = File::open(src).with_context(|| format!("open {}", src.display()))?;
        let src_len = reader.metadata().map(|m| m.len()).unwrap_or(0);
        advise_sequential_if_medium(&reader, src_len);

        let mut enc = zstd::stream::Encoder::new(&mut *writer, level)?;
        let _ = enc.window_log(ZSTD_STREAM_WLOG);
        let mut buf = vec![0u8; FILE_BUF_SIZE];
        let mut raw_total = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            enc.write_all(&buf[..n])?;
            raw_total += n as u64;
        }
        advise_dontneed_if_large(&reader, raw_total);
        enc.finish()?;
        let out_len = writer.metadata().map(|m| m.len()).unwrap_or(0);
        Ok((raw_total, out_len))
    })
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

pub fn os_random_bytes<const N: usize>() -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    getrandom::fill(&mut buf).map_err(|err| anyhow::anyhow!("OS CSPRNG failed: {err}"))?;
    Ok(buf)
}

pub fn write_atomic(target: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic_with(target, |f| Ok(f.write_all(bytes)?))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatEntry {
    pub mtime_ns: u64,
    pub size: u64,
    pub blake3: String,
}

pub fn file_mtime_size(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos() as u64;
    Some((mtime_ns, meta.len()))
}

pub fn now_unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Racy-git 2-second guard: only trust a stat cache hit when `now - mtime_ns >= 2s`.
#[inline]
pub fn stat_cache_fresh(entry: &StatEntry, mtime_ns: u64, size: u64, now_ns: u64) -> bool {
    entry.size == size
        && entry.mtime_ns == mtime_ns
        && now_ns.saturating_sub(mtime_ns) >= RACY_GIT_NS
}

pub fn load_stat_cache(path: &Path) -> BTreeMap<String, StatEntry> {
    let Ok(raw) = fs::read(path) else { return BTreeMap::new() };
    let Ok(dec) = zstd::stream::Decoder::new(raw.as_slice()) else { return BTreeMap::new() };
    let mut out = Vec::new();
    if dec.take(STAT_MAX_DECODE_BYTES).read_to_end(&mut out).is_err() {
        return BTreeMap::new();
    }
    decode_stat_map(&out).unwrap_or_default()
}

pub fn save_stat_cache(path: &Path, map: &BTreeMap<String, StatEntry>) -> Result<()> {
    if let Some(p) = path.parent() {
        ensure_cachedir_tag(p);
    }
    let raw = encode_stat_map(map);
    let compressed = zstd::bulk::compress(&raw, 1)?;
    write_atomic(path, &compressed)
}

fn encode_stat_map(map: &BTreeMap<String, StatEntry>) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 112);
    out.extend_from_slice(STAT_MAGIC);
    out.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (rel, e) in map {
        let rb = rel.as_bytes();
        let rlen = rb.len().min(u16::MAX as usize);
        let hb = e.blake3.as_bytes();
        let hlen = hb.len().min(u16::MAX as usize);
        out.extend_from_slice(&(rlen as u16).to_le_bytes());
        out.extend_from_slice(&rb[..rlen]);
        out.extend_from_slice(&e.mtime_ns.to_le_bytes());
        out.extend_from_slice(&e.size.to_le_bytes());
        out.extend_from_slice(&(hlen as u16).to_le_bytes());
        out.extend_from_slice(&hb[..hlen]);
    }
    out
}

fn decode_stat_map(buf: &[u8]) -> Option<BTreeMap<String, StatEntry>> {
    if buf.len() < 12 || &buf[..8] != STAT_MAGIC {
        return None;
    }
    let count = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
    let mut pos = 12usize;
    let mut map = BTreeMap::new();
    for _ in 0..count {
        let rlen = u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?) as usize;
        pos += 2;
        let rel = std::str::from_utf8(buf.get(pos..pos + rlen)?).ok()?.to_string();
        pos += rlen;
        let mtime_ns = u64::from_le_bytes(buf.get(pos..pos + 8)?.try_into().ok()?);
        pos += 8;
        let size = u64::from_le_bytes(buf.get(pos..pos + 8)?.try_into().ok()?);
        pos += 8;
        let hlen = u16::from_le_bytes(buf.get(pos..pos + 2)?.try_into().ok()?) as usize;
        pos += 2;
        let blake3 = std::str::from_utf8(buf.get(pos..pos + hlen)?).ok()?.to_string();
        pos += hlen;
        map.insert(rel, StatEntry { mtime_ns, size, blake3 });
    }
    Some(map)
}

/// Cross-OS sidecar lock (`File::lock()` with `O_EXCL` stale-reaper fallback for OverlayFS/WSL2).
pub enum SidecarLock {
    OsFile(File),
    ExclLease(PathBuf),
}

impl SidecarLock {
    pub fn acquire(lock_path: &Path) -> Result<Self> {
        let file = open_lock_file(lock_path)?;
        match file.lock() {
            Ok(()) => Ok(Self::OsFile(file)),
            Err(err) if is_lock_unsupported(&err) => acquire_excl_fallback(lock_path),
            Err(err) => Err(err).with_context(|| format!("acquire lock {}", lock_path.display())),
        }
    }

    pub fn try_acquire(lock_path: &Path) -> Result<Option<Self>> {
        let file = open_lock_file(lock_path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self::OsFile(file))),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(err)) if is_lock_unsupported(&err) => {
                try_acquire_excl_once(lock_path)
            }
            Err(fs::TryLockError::Error(err)) => Err(err.into()),
        }
    }
}

impl Drop for SidecarLock {
    fn drop(&mut self) {
        match self {
            Self::OsFile(f) => { let _ = f.unlock(); }
            Self::ExclLease(p) => { let _ = fs::remove_file(p); }
        }
    }
}

fn open_lock_file(lock_path: &Path) -> Result<File> {
    if let Some(p) = lock_path.parent() {
        fs::create_dir_all(p).with_context(|| format!("create lock dir {}", p.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
        .with_context(|| format!("open sidecar lock {}", lock_path.display()))
}

fn is_lock_unsupported(err: &io::Error) -> bool {
    #[cfg(unix)]
    if let Some(code) = err.raw_os_error()
        && matches!(
            nix::errno::Errno::from_raw(code),
            nix::errno::Errno::EOPNOTSUPP
                | nix::errno::Errno::ENOTSUP
                | nix::errno::Errno::ENOLCK
                | nix::errno::Errno::ENOSYS
        )
    {
        return true;
    }
    err.kind() == io::ErrorKind::Unsupported
}

fn try_acquire_excl_once(lock_path: &Path) -> Result<Option<SidecarLock>> {
    let name = lock_path.file_name().and_then(|s| s.to_str()).unwrap_or("refit.lock");
    let lease = lock_path.with_file_name(format!("{name}.excl"));
    match OpenOptions::new().create_new(true).write(true).open(&lease) {
        Ok(mut f) => {
            let _ = writeln!(f, "{} {}", std::process::id(), now_ms());
            Ok(Some(SidecarLock::ExclLease(lease)))
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            reap_stale_excl(&lease);
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

fn acquire_excl_fallback(lock_path: &Path) -> Result<SidecarLock> {
    for attempt in 0..200u64 {
        if let Some(lock) = try_acquire_excl_once(lock_path)? {
            return Ok(lock);
        }
        std::thread::sleep(Duration::from_millis((5 + attempt).min(50)));
    }
    anyhow::bail!("timed out waiting for O_EXCL fallback lock {}", lock_path.display())
}

fn reap_stale_excl(lease: &Path) {
    #[cfg(unix)]
    if let Ok(raw) = fs::read_to_string(lease)
        && let Some(pid) = raw.split_whitespace().next().and_then(|s| s.parse::<i32>().ok())
        && pid > 0
        && matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Ok(()) | Err(nix::errno::Errno::EPERM)
        )
    {
        return;
    }
    let Ok(meta) = fs::metadata(lease) else { return };
    if meta.modified().ok().and_then(|m| m.elapsed().ok()).is_some_and(|a| a > Duration::from_secs(60)) {
        let _ = fs::remove_file(lease);
    }
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

    #[test]
    fn stat_bin_v1_lz4_auto_invalidates_to_v2_zstd() {
        let tmp = TempRepo::new("io_stat_v2");
        let cache = tmp.join(".refit/cache/stat.bin");

        // 1. Stale RFST0001 LZ4-prepended bytes must safely return empty map (zero panic).
        let mut fake_v1_lz4 = vec![12, 0, 0, 0];
        fake_v1_lz4.extend_from_slice(b"RFST0001\x00\x00\x00\x00");
        write_atomic(&cache, &fake_v1_lz4).unwrap();
        assert!(load_stat_cache(&cache).is_empty());

        // Even Zstd-compressed RFST0001 bytes must be rejected by STAT_MAGIC = RFST0002.
        let zst_v1 = zstd::bulk::compress(b"RFST0001\x00\x00\x00\x00", 1).unwrap();
        write_atomic(&cache, &zst_v1).unwrap();
        assert!(load_stat_cache(&cache).is_empty());

        // 2. Round-trip v2 Zstd stat.bin + 2-second racy-git guard.
        let old_ns = 1_000_000_000_000u64;
        let now_ns = old_ns + 5_000_000_000;
        let mut map = BTreeMap::new();
        map.insert("src/a.rs".into(), StatEntry { mtime_ns: old_ns, size: 128, blake3: "abcd".into() });
        save_stat_cache(&cache, &map).unwrap();

        assert!(has_cachedir_tag(&tmp.join(".refit/cache")));
        let on_disk = fs::read(&cache).unwrap();
        assert_eq!(&on_disk[..4], &[0x28, 0xB5, 0x2F, 0xFD], "stat.bin must be Zstd-compressed");
        let loaded = load_stat_cache(&cache);
        let e = loaded.get("src/a.rs").unwrap();
        assert!(stat_cache_fresh(e, old_ns, 128, now_ns));
        assert!(!stat_cache_fresh(e, old_ns, 128, old_ns + 500_000_000));
        assert!(!stat_cache_fresh(e, old_ns, 129, now_ns));
    }

    #[test]
    fn file_buf_hash_and_sidecar_lock() {
        let tmp = TempRepo::with_files("io_buf", &[("big.txt", &"hello\n".repeat(4000))]);
        let p = tmp.join("big.txt");
        let mut fb = FileBuf::new();
        let (len, hex) = fb.hash_file_blake3(&p).unwrap();
        let raw = fs::read(&p).unwrap();
        assert_eq!(len, raw.len() as u64);
        assert_eq!(hex, blake3::hash(&raw).to_hex().to_string());

        let lp = tmp.join(".refit/test.lock");
        let lk = SidecarLock::acquire(&lp).unwrap();
        assert!(SidecarLock::try_acquire(&lp).unwrap().is_none());
        drop(lk);
        assert!(SidecarLock::try_acquire(&lp).unwrap().is_some());
    }
}
