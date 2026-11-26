pub mod paths;
mod proc;

pub use crate::net::backoff_ms;
#[cfg(test)]
use crate::net::{is_quota_exhausted, is_retryable, retry_ms};
pub use paths::{
    check_rel, check_target_in_repo, normalize_rel, pwsh_utf8_preamble, rel_to_root,
    strip_utf8_bom, trim_crlf,
};
pub use proc::{
    MemGate, allocator_name, budget, collect_os_memory, env_parse, env_trimmed, env_truthy,
    init_allocator, par_join2, par_join3, par_map, run_cmd, self_metrics, under_memory_pressure,
};

use crate::lex::left_bound;
use crate::net::build_http;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use anyhow::{Context, Result, bail};
use flate2::read::DeflateDecoder;
use ignore::WalkBuilder;
use rustc_hash::{FxBuildHasher, FxHasher};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const EPOCH_SLOTS: usize = 64;
const FILE_BUF_SIZE: usize = 64 * 1024;
#[cfg(target_os = "linux")] const FADV_SEQ_MIN: u64 = 64 * 1024;
#[cfg(target_os = "linux")] const FADV_DONTNEED_MIN: u64 = 1024 * 1024;
const ZSTD_STREAM_WLOG: u32 = 18;
/// Binary stat cache format header bytes.
const STAT_MAGIC: &[u8; 8] = b"RFST0003";
const STAT_MAX_DECODE_BYTES: u64 = 64 * 1024 * 1024;
const RACY_GIT_NS: u64 = 2_000_000_000;
const CACHEDIR_TAG_BODY: &[u8] = b"Signature: 8a477f597d28d172789f06886806bc55\n";

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
    let arch = match std::env::consts::ARCH { "x86_64" => "x64", "aarch64" => "arm64", other => other };
    format!("{}-{arch}", std::env::consts::OS)
}

#[inline]
pub fn now_ms() -> u64 {
    let (epoch_ms, started) = CLOCK_ANCHOR.get_or_init(|| {
        let base = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |dur| dur.as_millis() as u64);
        (base, Instant::now())
    });
    epoch_ms.saturating_add(started.elapsed().as_millis() as u64)
}

#[inline] pub fn skip_dir(name: &str) -> bool { SKIP_DIRS.contains(&name) }
pub fn walk_files(repo_root: &Path, max_depth: Option<usize>) -> Vec<PathBuf> {
    walk_tree(repo_root, max_depth, false)
}

pub fn walk_tree(repo_root: &Path, max_depth: Option<usize>, include_hidden_configs: bool) -> Vec<PathBuf> {
    let mut wb = WalkBuilder::new(repo_root);
    wb.hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .ignore(true)
        .parents(false)
        .require_git(false)
        .add_custom_ignore_filename(".refitignore")
        .max_depth(max_depth)
        .filter_entry(move |entry| keep_entry(entry, include_hidden_configs));
    let mut out: Vec<PathBuf> = wb.build().flatten()
        .filter(|entry| entry.file_type().is_some_and(|ft| ft.is_file()))
        .map(ignore::DirEntry::into_path).collect();
    out.sort();
    out
}

fn keep_entry(entry: &ignore::DirEntry, include_hidden: bool) -> bool {
    if entry.depth() == 0 || !entry.file_type().is_some_and(|ft| ft.is_dir()) { return true; }
    let name = entry.file_name().to_string_lossy();
    if skip_dir(&name) || has_cachedir_tag(entry.path()) { return false; }
    !name.starts_with('.') || (include_hidden && name == ".github")
}

pub fn sandbox_env(extra: &[(String, String)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = SANDBOX_VARS.iter()
        .filter_map(|&key| std::env::var(key).ok().map(|val| (key.to_string(), val)))
        .collect();
    if std::env::var_os("MSYSTEM").is_some() {
        env.push(("MSYS_NO_PATHCONV".into(), "1".into()));
    }
    for (key, val) in extra {
        if let Some(slot) = env.iter_mut().find(|(ek, _)| ek.eq_ignore_ascii_case(key)) {
            slot.1 = val.clone();
        } else {
            env.push((key.clone(), val.clone()));
        }
    }
    env
}

pub fn char_prefix(text: &str, max_chars: usize) -> &str {
    text.char_indices().nth(max_chars).map_or(text, |(idx, _)| &text[..idx])
}

pub fn new_http(user_agent: &str, timeout_secs: u64, pool_max: usize) -> Result<reqwest::Client> {
    build_http(user_agent, 15, 600, pool_max, Some(timeout_secs))
}

pub fn stream_http(
    user_agent: &str, connect_timeout_secs: u64, pool_idle_secs: u64, pool_max: usize,
) -> Result<reqwest::Client> {
    build_http(user_agent, connect_timeout_secs, pool_idle_secs, pool_max, None)
}

fn stream_read_chunks(
    path: &Path, buf: &mut [u8], mut on_chunk: impl FnMut(&[u8]) -> Result<()>,
) -> Result<u64> {
    let mut reader = File::open(path).with_context(|| format!("open {}", path.display()))?;
    advise_file(&reader, reader.metadata().map_or(0, |meta| meta.len()), false);
    let mut total = 0u64;
    loop {
        let read_n = reader.read(buf)?;
        if read_n == 0 { break; }
        on_chunk(&buf[..read_n])?;
        total += read_n as u64;
    }
    advise_file(&reader, total, true);
    Ok(total)
}

/// Reusable O(1) 64 KiB scratch buffer for streaming file I/O.
pub struct FileBuf { buf: Vec<u8> }

impl FileBuf {
    pub fn new() -> Self { Self { buf: vec![0u8; FILE_BUF_SIZE] } }

    pub fn hash_file_blake3(&mut self, path: &Path) -> Result<(u64, String)> {
        let mut hasher = blake3::Hasher::new();
        let total = stream_read_chunks(path, &mut self.buf, |chunk| {
            hasher.update(chunk);
            Ok(())
        })?;
        Ok((total, hasher.finalize().to_hex().to_string()))
    }
}

impl Default for FileBuf { fn default() -> Self { Self::new() } }

/// Write the standard 43-byte `CACHEDIR.TAG` marker inside `dir`.
pub fn ensure_cachedir_tag(dir: &Path) {
    let tag = dir.join("CACHEDIR.TAG");
    if tag.exists() { return; }
    let _ = fs::create_dir_all(dir);
    let _ = fs::write(&tag, CACHEDIR_TAG_BODY);
}

#[inline]
pub fn has_cachedir_tag(dir: &Path) -> bool {
    let Ok(mut file) = File::open(dir.join("CACHEDIR.TAG")) else { return false };
    let mut head = [0u8; 43];
    file.read_exact(&mut head).is_ok()
        && head.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55")
}

fn advise_file(file: &File, len: u64, dontneed: bool) {
    cfg_select! {
        target_os = "linux" => {
            use rustix::fs::{Advice, fadvise};
            let adv = if dontneed && len >= FADV_DONTNEED_MIN {
                Some(Advice::DontNeed)
            } else if !dontneed && (FADV_SEQ_MIN..FADV_DONTNEED_MIN).contains(&len) {
                Some(Advice::Sequential)
            } else { None };
            if let Some(hint) = adv { let _ = fadvise(file, 0, None, hint); }
        }
        _ => { let _ = (file, len, dontneed); }
    }
}

/// Cross-OS durability barrier (`fdatasync` on Linux, `F_FULLFSYNC` on macOS, `sync_data` on Win).
pub fn sync_file_barrier(file: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    if rustix::fs::fcntl_fullfsync(file).is_ok() { return Ok(()); }
    file.sync_data()
}

fn persist_with_retry(mut tmp: tempfile::NamedTempFile, target: &Path) -> Result<()> {
    let mut delay_ms = 5u64;
    for attempt in 0..7 {
        match tmp.persist(target) {
            Ok(_) => {
                #[cfg(unix)]
                if let Some(parent) = target.parent() && let Ok(dir) = File::open(parent) {
                    let _ = dir.sync_all();
                }
                return Ok(());
            }
            Err(err) if cfg!(windows) && err.error.raw_os_error() == Some(5) && attempt < 6 => {
                tmp = err.file;
                std::thread::sleep(Duration::from_millis(delay_ms));
                delay_ms = (delay_ms * 2).min(160);
            }
            Err(err) => return Err(err.error).with_context(|| format!("atomic persist {}", target.display())),
        }
    }
    unreachable!()
}

fn write_atomic_with<T>(target: &Path, write_fn: impl FnOnce(&mut File) -> Result<T>) -> Result<T> {
    let parent = target
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create parent dir {}", parent.display()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create temp file in {}", parent.display()))?;
    let res = write_fn(tmp.as_file_mut())?;
    sync_file_barrier(tmp.as_file())?;
    advise_file(tmp.as_file(), tmp.as_file().metadata().map_or(0, |meta| meta.len()), true);
    persist_with_retry(tmp, target)?;
    Ok(res)
}

pub fn write_zstd<T: Serialize>(path: &Path, val: &T) -> Result<()> {
    write_atomic_with(path, |file| {
        let mut enc = zstd::stream::Encoder::new(file, 1)?;
        let _ = enc.window_log(ZSTD_STREAM_WLOG);
        serde_json::to_writer(&mut enc, val)?;
        enc.finish()?;
        Ok(())
    })
}

pub fn read_zstd<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let file = File::open(path).ok()?;
    let len = file.metadata().map_or(0, |meta| meta.len());
    advise_file(&file, len, false);
    let out = serde_json::from_reader(zstd::stream::Decoder::new(&file).ok()?).ok();
    advise_file(&file, len, true);
    out
}

/// Stream-compress an existing file on disk into `dst` without loading the whole file in RAM.
pub fn compress_file_zstd(src: &Path, dst: &Path, level: i32) -> Result<(u64, u64)> {
    write_atomic_with(dst, |writer| {
        let mut enc = zstd::stream::Encoder::new(&mut *writer, level)?;
        let _ = enc.window_log(ZSTD_STREAM_WLOG);
        let mut buf = vec![0u8; FILE_BUF_SIZE];
        let raw_total = stream_read_chunks(src, &mut buf, |chunk| Ok(enc.write_all(chunk)?))?;
        enc.finish()?;
        Ok((raw_total, writer.metadata().map_or(0, |meta| meta.len())))
    })
}

/// Read and CRC-verify a single `STORE` or `DEFLATE` ZIP entry with an uncompressed byte cap.
pub fn read_zip_entry_capped(
    archive: &rawzip::ZipSliceArchive<&[u8]>,
    entry: &rawzip::ZipFileHeaderRecord<'_>,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    let local = archive.get_entry(entry.wayfinder()).context("read ZIP entry")?;
    let mut out = Vec::new();
    match entry.compression_method() {
        rawzip::CompressionMethod::STORE => {
            if local.data().len() > max_bytes {
                bail!("stored ZIP entry exceeds {max_bytes} byte cap");
            }
            local.verifying_reader(local.data()).read_to_end(&mut out).context("verify stored ZIP entry")?;
        }
        rawzip::CompressionMethod::DEFLATE => {
            if entry.uncompressed_size_hint() > max_bytes as u64 {
                bail!("deflated ZIP entry exceeds {max_bytes} byte zip-bomb cap");
            }
            let res = local
                .verifying_reader(DeflateDecoder::new(local.data()))
                .take((max_bytes + 1) as u64)
                .read_to_end(&mut out);
            if out.len() > max_bytes {
                bail!("deflated ZIP entry exceeds {max_bytes} byte zip-bomb cap");
            }
            res.context("inflate ZIP entry")?;
        }
        other => bail!("unsupported ZIP compression method {other:?}"),
    }
    Ok(out)
}

/// Read the active Git `HEAD` commit hash directly from `.git/HEAD` (0 subprocesses, worktree-safe).
pub fn read_git_head_fast(repo_root: &Path) -> Option<String> {
    let dot_git = repo_root.join(".git");
    let git_dir = if dot_git.is_file() {
        let raw = fs::read_to_string(&dot_git).ok()?;
        let rest = raw.trim().strip_prefix("gitdir:")?.trim();
        let p = PathBuf::from(rest);
        if p.is_absolute() { p } else { repo_root.join(p) }
    } else {
        dot_git
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let trimmed = head.trim();
    if let Some(ref_rel) = trimmed.strip_prefix("ref:") {
        let ref_rel = ref_rel.trim();
        if let Ok(sha) = fs::read_to_string(git_dir.join(ref_rel)) {
            let s = sha.trim();
            if !s.is_empty() { return Some(s.to_string()); }
        }
        let commondir = fs::read_to_string(git_dir.join("commondir")).ok().map_or_else(
            || git_dir.clone(),
            |c| {
                let p = PathBuf::from(c.trim());
                if p.is_absolute() { p } else { git_dir.join(p) }
            },
        );
        if let Ok(sha) = fs::read_to_string(commondir.join(ref_rel)) {
            let s = sha.trim();
            if !s.is_empty() { return Some(s.to_string()); }
        }
        let packed = fs::read_to_string(commondir.join("packed-refs")).ok()?;
        for line in packed.lines() {
            if let Some((sha, name)) = line.trim().split_once(' ')
                && name == ref_rel && !sha.starts_with('#')
            {
                return Some(sha.to_string());
            }
        }
        return None;
    }
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

pub fn mkdir_private(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("create private directory {}", path.display()))?;
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
    write_atomic_with(target, |file| Ok(file.write_all(bytes)?))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatEntry { pub mtime_ns: u64, pub size: u64, pub blake3: String }

pub fn file_mtime_size(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let mtime_ns = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos() as u64;
    Some((mtime_ns, meta.len()))
}

pub fn now_unix_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |dur| dur.as_nanos() as u64)
}

/// Trust a stat cache hit when at least 2 seconds elapsed since mtime.
#[inline]
pub fn stat_cache_fresh(entry: &StatEntry, mtime_ns: u64, size: u64, now_ns: u64) -> bool {
    entry.size == size && entry.mtime_ns == mtime_ns && now_ns.saturating_sub(mtime_ns) >= RACY_GIT_NS
}

pub fn load_stat_cache(path: &Path) -> BTreeMap<String, StatEntry> {
    let Ok(raw) = fs::read(path) else { return BTreeMap::new() };
    let Ok(dec) = zstd::stream::Decoder::new(raw.as_slice()) else { return BTreeMap::new() };
    let mut out = Vec::new();
    if dec.take(STAT_MAX_DECODE_BYTES).read_to_end(&mut out).is_err() { return BTreeMap::new(); }
    decode_stat_map(&out).unwrap_or_default()
}

pub fn save_stat_cache(path: &Path, map: &BTreeMap<String, StatEntry>) -> Result<()> {
    if let Some(parent) = path.parent() { ensure_cachedir_tag(parent); }
    write_atomic(path, &zstd::bulk::compress(&encode_stat_map(map), 1)?)
}

fn encode_stat_map(map: &BTreeMap<String, StatEntry>) -> Vec<u8> {
    let mut out = Vec::with_capacity(STAT_MAGIC.len() + map.len() * 96);
    out.extend_from_slice(STAT_MAGIC);
    if let Ok(body) = postcard::to_allocvec(map) { out.extend_from_slice(&body); }
    out
}

fn decode_stat_map(buf: &[u8]) -> Option<BTreeMap<String, StatEntry>> {
    postcard::from_bytes(buf.strip_prefix(STAT_MAGIC)?).ok()
}

/// Cross-OS sidecar lock (`File::lock()` with `O_EXCL` stale-reaper fallback for OverlayFS/WSL2).
pub enum SidecarLock { OsFile(File), ExclLease(PathBuf) }

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
            Self::OsFile(file) => { let _ = file.unlock(); }
            Self::ExclLease(path) => { let _ = fs::remove_file(path); }
        }
    }
}

fn open_lock_file(lock_path: &Path) -> Result<File> {
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create lock dir {}", parent.display()))?;
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
    if let Some(code) = err.raw_os_error() {
        let unsupported = code == libc::EOPNOTSUPP
            || code == libc::ENOTSUP
            || code == libc::ENOLCK
            || code == libc::ENOSYS;
        if unsupported {
            return true;
        }
    }
    err.kind() == io::ErrorKind::Unsupported
}

fn try_acquire_excl_once(lock_path: &Path) -> Result<Option<SidecarLock>> {
    let name = lock_path.file_name().and_then(|seg| seg.to_str()).unwrap_or("refit.lock");
    let lease = lock_path.with_file_name(format!("{name}.excl"));
    let mut opts = OpenOptions::new();
    opts.create_new(true).write(true);
    match opts.open(&lease) {
        Ok(mut file) => {
            let _ = writeln!(file, "{} {}", std::process::id(), now_ms());
            Ok(Some(SidecarLock::ExclLease(lease)))
        }
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            reap_stale_excl(&lease);
            Ok(None)
        }
        Err(err) => Err(err.into()),
    }
}

fn acquire_excl_fallback(lock_path: &Path) -> Result<SidecarLock> {
    for attempt in 0..200u64 {
        if let Some(lock) = try_acquire_excl_once(lock_path)? { return Ok(lock); }
        std::thread::sleep(Duration::from_millis((5 + attempt).min(50)));
    }
    anyhow::bail!("timed out waiting for O_EXCL fallback lock {}", lock_path.display())
}

fn reap_stale_excl(lease: &Path) {
    #[cfg(unix)]
    if let Ok(raw) = fs::read_to_string(lease) {
        let token = raw.split_whitespace().next();
        if let Some(pid) = token
            .and_then(|tok| tok.parse::<i32>().ok())
            .and_then(rustix::process::Pid::from_raw)
            && matches!(
                rustix::process::test_kill_process(pid),
                Ok(()) | Err(rustix::io::Errno::PERM)
            )
        {
            return;
        }
    }
    let elapsed = fs::metadata(lease)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|mt| mt.elapsed().ok());
    if elapsed.is_some_and(|age| age > Duration::from_secs(60)) {
        let _ = fs::remove_file(lease);
    }
}

#[derive(Clone, Copy)]
struct TupleWeighter(usize);

impl<K, V> quick_cache::Weighter<K, (V, usize)> for TupleWeighter {
    #[inline] fn weight(&self, _key: &K, val: &(V, usize)) -> u64 { val.1.max(self.0) as u64 }
}

/// Thread-safe byte- and entry-bounded cache backed by `quick_cache` + `FxBuildHasher`.
pub struct SieveCache<K, V> {
    inner: quick_cache::sync::Cache<K, (V, usize), TupleWeighter, FxBuildHasher>,
    max_bytes: usize,
}

impl<K: Eq + Hash + Clone, V: Clone> SieveCache<K, V> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        let (max_entries, max_bytes) = (max_entries.max(1), max_bytes.max(1));
        let opts = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(max_entries).weight_capacity(max_bytes as u64)
            .hot_allocation(1.0).build().expect("valid cache options");
        let inner = quick_cache::sync::Cache::with_options(
            opts, TupleWeighter((max_bytes / max_entries).max(1)), FxBuildHasher,
            quick_cache::sync::DefaultLifecycle::default(),
        );
        Self { inner, max_bytes }
    }

    pub fn get(&self, key: &K) -> Option<V> { self.inner.get(key).map(|(val, _)| val) }

    pub fn insert(&self, key: K, val: V, weight_bytes: usize) -> Option<V> {
        let weight = weight_bytes.max(1);
        let prev = self.inner.remove(&key).map(|(_, (old, _))| old);
        if weight <= self.max_bytes { self.inner.insert(key, (val, weight)); }
        prev
    }

    pub fn get_or_insert_with(&self, key: K, weight_bytes: usize, make: impl FnOnce() -> V) -> V {
        let weight = weight_bytes.max(1);
        if weight > self.max_bytes { return self.get(&key).unwrap_or_else(make); }
        let Ok((val, _)) = self
            .inner
            .get_or_insert_with(&key, || Ok::<_, std::convert::Infallible>((make(), weight)));
        val
    }

    pub fn remove(&self, key: &K) -> Option<V> { self.inner.remove(key).map(|(_, (val, _))| val) }
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn snapshot(&self) -> Vec<(K, V)> { self.inner.iter().map(|(key, (val, _))| (key, val)).collect() }
}

/// Lock-free monotonic epoch registry (`global_epoch` + 64 atomic key slots).
pub struct EpochRegistry { global_epoch: AtomicU64, slots: [AtomicU64; EPOCH_SLOTS] }

impl EpochRegistry {
    pub fn new() -> Self {
        Self { global_epoch: AtomicU64::new(1), slots: std::array::from_fn(|_| AtomicU64::new(1)) }
    }

    pub fn global_arc() -> std::sync::Arc<Self> { Self::global_arc_slot().clone() }

    fn global_arc_slot() -> &'static std::sync::Arc<Self> {
        static GLOBAL: OnceLock<std::sync::Arc<EpochRegistry>> = OnceLock::new();
        GLOBAL.get_or_init(|| std::sync::Arc::new(Self::new()))
    }

    #[inline] pub fn global_epoch(&self) -> u64 { self.global_epoch.load(Ordering::Acquire) }

    #[inline]
    fn slot_idx(key: &str) -> usize {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        (hasher.finish() as usize) & (EPOCH_SLOTS - 1)
    }

    #[inline] pub fn slot_epoch(&self, key: &str) -> u64 {
        self.slots[Self::slot_idx(key)].load(Ordering::Acquire)
    }

    /// Single-authority monotonic publish (`+1` exact) for ABI cutoff or single-key updates.
    #[inline]
    #[cfg(test)]
    pub fn publish(&self, key: &str, _head_rev: u64) -> u64 {
        let next = self.global_epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        self.slots[Self::slot_idx(key)].fetch_max(next, Ordering::Release);
        next
    }

    /// Single-authority monotonic batch publish called by `MvccStore::commit_tx` (`+1` exact).
    pub fn publish_batch(&self, module_id: &str, files: &[String], _head_rev: u64) -> u64 {
        let next = self.global_epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        self.slots[Self::slot_idx(module_id)].fetch_max(next, Ordering::Release);
        for rel in files { self.slots[Self::slot_idx(rel)].fetch_max(next, Ordering::Release); }
        next
    }
}

impl Default for EpochRegistry { fn default() -> Self { Self::new() } }

/// Extract trimmed body lines from Go `keyword ( ... )` blocks and single-line `keyword ...` entries.
pub fn parse_go_block_lines<'a>(txt: &'a str, kw: &str) -> Vec<&'a str> {
    let (mut out, mut in_block) = (Vec::new(), false);
    for raw in txt.lines() {
        let line = raw.trim();
        if in_block {
            if line.starts_with(')') { in_block = false; }
            else if !line.is_empty() && !line.starts_with("//") { out.push(line); }
            continue;
        }
        let Some(rest) = line.strip_prefix(kw) else { continue };
        let after = rest.trim_start();
        if let Some(body) = after.strip_prefix('(') {
            if let Some(end) = body.find(')') {
                let inner = body[..end].trim();
                if !inner.is_empty() && !inner.starts_with("//") { out.push(inner); }
            } else {
                in_block = true;
            }
        } else if !after.is_empty() && rest.starts_with(|ch: char| ch.is_ascii_whitespace()) {
            out.push(after);
        }
    }
    out
}

const PEM_MARKER: &str = "PRIVATE KEY-----";
const URI_SCHEMES: &[&[u8]] = &[
    b"postgres", b"postgresql", b"mysql", b"mongodb", b"mongodb+srv", b"redis",
    b"rediss", b"amqp", b"amqps", b"mssql", b"ftp", b"sftp", b"ldap", b"smtp",
];
// Compact rule table: prefix|kind|redact_tag|detect_min|redact_min|mode
// mode: 0=alnum_-, 1=UPPER+dig, 2=hex, 3=b64/url, 4=alnum+digit_req
const SECRET_TABLE_A: &str = concat!(
    "sk-ant-api03-|anthropic_api_key|REDACTED_ANTHROPIC_|32|16|0;",
    "sk-ant-admin-|anthropic_admin_key|REDACTED_ANTHROPIC_|32|16|0;",
    "sk-ant-usr-|anthropic_user_key|REDACTED_ANTHROPIC_|32|16|0;",
    "sk-ant-|anthropic_api_key|REDACTED_ANTHROPIC_|16|12|0;",
    "sk-proj-|openai_api_key|REDACTED_OPENAI_|16|12|0;",
    "sk-svcacct-|openai_service_key|REDACTED_OPENAI_|24|16|0;",
    "sk-admin-|openai_admin_key|REDACTED_OPENAI_|24|16|0;",
    "sk-or-v1-|openrouter_api_key|REDACTED_OPENROUTER_|48|24|2;",
    "sk-lf-|langfuse_secret_key|REDACTED_LANGFUSE_|24|16|0;",
    "sk-|generic_sk_secret|REDACTED_SK_|24|24|4;sk_live_|stripe_live_key|REDACTED_STRIPE_|16|12|0;",
    "rk_live_|stripe_restricted_key|REDACTED_STRIPE_|16|12|0;",
    "sk.eyJ|mapbox_secret_jwt|REDACTED_MAPBOX_|32|24|3;",
    "ghp_|github_pat|REDACTED_GITHUB_|20|16|0;gho_|github_oauth|REDACTED_GITHUB_|20|16|0;",
    "ghu_|github_u2s_token|REDACTED_GITHUB_|20|16|0;ghs_|github_s2s_token|REDACTED_GITHUB_|20|16|0;",
    "ghr_|github_refresh_token|REDACTED_GITHUB_|20|16|0;",
    "github_pat_|github_fine_grained_pat|REDACTED_GITHUB_|20|16|0;",
    "glpat-|gitlab_pat|REDACTED_GITLAB_|20|16|0;gldt-|gitlab_deploy_token|REDACTED_GITLAB_|20|16|0;",
    "glrt-|gitlab_runner_token|REDACTED_GITLAB_|20|16|0;glcbt-|gitlab_job_token|REDACTED_GITLAB_|20|16|0;",
    "glsa_|grafana_sa_token|REDACTED_GRAFANA_|28|20|0;glc_|grafana_cloud_token|REDACTED_GRAFANA_|32|24|3;",
    "xoxb-|slack_bot_token|REDACTED_SLACK_|16|12|0;xoxp-|slack_user_token|REDACTED_SLACK_|16|12|0;",
    "xoxa-|slack_app_token|REDACTED_SLACK_|16|12|0;xoxr-|slack_refresh_token|REDACTED_SLACK_|16|12|0;",
    "xoxs-|slack_legacy_token|REDACTED_SLACK_|16|12|0;xapp-|slack_app_level|REDACTED_SLACK_|24|16|0;",
    "AKIA|aws_access_key_id|REDACTED_AWS_|16|16|1;ASIA|aws_sts_key_id|REDACTED_AWS_|16|16|1;",
    "ABIA|aws_sts_svc_key|REDACTED_AWS_|16|16|1;ACCA|aws_ctx_key_id|REDACTED_AWS_|16|16|1;",
    "AKCp|jfrog_api_key|REDACTED_JFROG_|48|32|0;AIza|gcp_api_key|REDACTED_GCP_|35|24|0;",
    "ya29.|gcp_oauth_token|REDACTED_GCP_|32|24|3;1//0|gcp_oauth_refresh|REDACTED_GCP_|32|24|3;",
    "dop_v1_|digitalocean_pat|REDACTED_DO_|64|32|2;doo_v1_|digitalocean_oauth|REDACTED_DO_|64|32|2;",
    "dor_v1_|digitalocean_refresh|REDACTED_DO_|64|32|2;hvs.|vault_service_token|REDACTED_VAULT_|20|16|0;",
    "hvb.|vault_batch_token|REDACTED_VAULT_|20|16|0;hvr.|vault_recovery_token|REDACTED_VAULT_|20|16|0;",
    "dp.st.|doppler_svc_token|REDACTED_DOPPLER_|32|20|0;dp.sa.|doppler_sa_token|REDACTED_DOPPLER_|32|20|0;",
    "dp.pt.|doppler_pers_token|REDACTED_DOPPLER_|32|20|0;dp.ct.|doppler_cli_token|REDACTED_DOPPLER_|32|20|0;",
    "shpat_|shopify_admin_pat|REDACTED_SHOPIFY_|32|20|2;shpca_|shopify_custom_app|REDACTED_SHOPIFY_|32|20|2;",
    "shppa_|shopify_private_app|REDACTED_SHOPIFY_|32|20|2;shpss_|shopify_shared_sec|REDACTED_SHOPIFY_|32|20|2;",
    "bkua_|buildkite_user_tok|REDACTED_BK_|32|20|0;bkct_|buildkite_cluster|REDACTED_BK_|32|20|0;",
    "bkar_|buildkite_agent|REDACTED_BK_|32|20|0;bkaj_|buildkite_job|REDACTED_BK_|32|20|0;",
    "bktx_|buildkite_exch|REDACTED_BK_|32|20|0;csk-|cerebras_api_key|REDACTED_CSK_|12|8|0;",
    "gsk_|groq_api_key|REDACTED_GROQ_|32|20|0;hf_|huggingface_token|REDACTED_HF_|30|20|4;",
);
const SECRET_TABLE_B: &str = concat!(
    "keen_|keenable_api_key|REDACTED_KEEN_|12|8|0;xai-|xai_api_key|REDACTED_XAI_|32|20|0;",
    "r8_|replicate_token|REDACTED_REPLICATE_|32|20|4;fc-|firecrawl_api_key|REDACTED_FC_|24|16|2;",
    "tvly-|tavily_api_key|REDACTED_TAVILY_|24|16|0;pplx-|perplexity_api_key|REDACTED_PPLX_|32|20|0;",
    "fw_|fireworks_api_key|REDACTED_FW_|24|16|4;lsv2_tr_|langsmith_trace_key|REDACTED_LS_|24|16|0;",
    "lsv2_sk_|langsmith_svc_key|REDACTED_LS_|24|16|0;lsv2_pt_|langsmith_pat|REDACTED_LS_|24|16|0;",
    "whsec_|stripe_webhook_secret|REDACTED_STRIPE_|24|16|0;pdl_live_|paddle_live_key|REDACTED_PADDLE_|24|16|0;",
    "pdl_sdbx_|paddle_sandbox_key|REDACTED_PADDLE_|24|16|0;sq0csp-|square_oauth_secret|REDACTED_SQ_|32|20|0;",
    "sq0atb-|square_access_token|REDACTED_SQ_|22|16|0;SG.|sendgrid_api_key|REDACTED_SG_|40|24|3;",
    "xkeysib-|brevo_api_key|REDACTED_BREVO_|40|24|0;re_|resend_api_key|REDACTED_RESEND_|24|16|4;",
    "sec-c-|pubnub_secret_key|REDACTED_PUBNUB_|32|20|0;tskey-api-|tailscale_api_key|REDACTED_TS_|20|16|0;",
    "tskey-auth-|tailscale_auth_key|REDACTED_TS_|20|16|0;ops_|1password_svc_token|REDACTED_1P_|32|20|3;",
    "npm_|npm_access_token|REDACTED_NPM_|32|20|0;pypi-|pypi_api_token|REDACTED_PYPI_|32|20|0;",
    "cio_|crates_io_token|REDACTED_CRATES_|24|16|0;dckr_pat_|docker_hub_pat|REDACTED_DOCKER_|24|16|0;",
    "dckr_oat_|docker_hub_oat|REDACTED_DOCKER_|24|16|0;hex_|hex_pm_token|REDACTED_HEX_|32|20|2;",
    "CLOJARS_|clojars_token|REDACTED_CLOJARS_|40|24|0;packagist_ack_|packagist_key|REDACTED_PKG_|24|16|0;",
    "packagist_acs_|packagist_secret|REDACTED_PKG_|24|16|0;squ_|sonarqube_user_tok|REDACTED_SONAR_|32|20|2;",
    "sqp_|sonarqube_proj_tok|REDACTED_SONAR_|32|20|2;lin_api_|linear_api_key|REDACTED_LINEAR_|32|20|0;",
    "ntn_|notion_api_token|REDACTED_NOTION_|32|20|0;figd_|figma_pat|REDACTED_FIGMA_|32|20|0;",
    "sntrys_|sentry_auth_token|REDACTED_SENTRY_|32|20|3;sntryu_|sentry_user_token|REDACTED_SENTRY_|32|20|2;",
    "phx_|posthog_personal_key|REDACTED_PH_|32|20|0;phs_|posthog_secret_key|REDACTED_PH_|32|20|0;",
    "NRAK-|newrelic_user_key|REDACTED_NR_|24|16|1;NRII-|newrelic_insert_key|REDACTED_NR_|24|16|0;",
    "pat-na1-|hubspot_pat_na|REDACTED_HS_|32|20|0;pat-eu1-|hubspot_pat_eu|REDACTED_HS_|32|20|0;",
    "pat-in1-|hubspot_pat_in|REDACTED_HS_|32|20|0;sb_secret_|supabase_secret_key|REDACTED_SB_|24|16|0;",
    "nt_live_|neon_api_key|REDACTED_NEON_|24|16|0;dapi|databricks_pat|REDACTED_DBX_|32|24|2;",
    "LTAI|alibaba_access_key|REDACTED_ALI_|16|16|0;AccountKey=|azure_storage_key|REDACTED_AZ_|40|24|3;",
    "SharedAccessSignature|azure_sas_token|REDACTED_AZ_|20|16|3;API-|octopus_api_key|REDACTED_OCTO_|20|16|1;",
    "AGE-SECRET-KEY-1|age_secret_key|REDACTED_AGE_|40|24|1;nsec1|nostr_secret_key|REDACTED_NOSTR_|50|32|0;",
    "xprv|bip32_ext_privkey|REDACTED_XPRV_|80|40|0;PuTTY-User-Key-File-|putty_private_key|REDACTED_PPK_|4|4|0;",
    "hooks.slack.com/services/|slack_webhook_url|REDACTED_SLACK_|20|16|3;",
    "discord.com/api/webhooks/|discord_webhook_url|REDACTED_DISCORD_|20|16|3;",
);
const GENERIC_KEYS: &[&str] = &[
    "api_key", "apikey", "secret_key", "client_secret",
    "access_key", "auth_token", "private_key", "password", "passwd",
];

struct SecretRule { pfx: &'static str, kind: &'static str, tag: &'static str, dmin: usize, rmin: usize, mode: u8 }
struct SecretSlab { ac: AhoCorasick, rules: Vec<SecretRule> }
struct SecretEngine { slabs: [SecretSlab; 2], generic_ac: AhoCorasick, hash: String, total: usize }

fn parse_slab(raw: &'static str, extras: &[&'static str]) -> SecretSlab {
    let rules: Vec<SecretRule> = raw.split(';').filter(|r| !r.is_empty()).map(|row| {
        let mut it = row.split('|');
        SecretRule {
            pfx: it.next().unwrap_or(""), kind: it.next().unwrap_or(""), tag: it.next().unwrap_or(""),
            dmin: it.next().and_then(|v| v.parse().ok()).unwrap_or(16),
            rmin: it.next().and_then(|v| v.parse().ok()).unwrap_or(12),
            mode: it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
        }
    }).collect();
    let ac = AhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest)
        .build(rules.iter().map(|r| r.pfx).chain(extras.iter().copied()))
        .expect("secret slab ac");
    SecretSlab { ac, rules }
}

static SECRET_ENGINE: LazyLock<SecretEngine> = LazyLock::new(|| {
    let s0 = parse_slab(SECRET_TABLE_A, &["://", "-----BEGIN "]);
    let s1 = parse_slab(SECRET_TABLE_B, &[]);
    let total = s0.rules.len() + s1.rules.len() + URI_SCHEMES.len() + 2;
    let mut h = blake3::Hasher::new();
    h.update(SECRET_TABLE_A.as_bytes());
    h.update(SECRET_TABLE_B.as_bytes());
    let generic_ac = AhoCorasickBuilder::new().ascii_case_insensitive(true)
        .build(GENERIC_KEYS).expect("generic secret ac");
    SecretEngine { slabs: [s0, s1], generic_ac, hash: h.finalize().to_hex()[..16].into(), total }
});

/// Return `(rule_count, rules_blake3_16)` for zero-recompute secret caches.
pub fn secret_rule_meta() -> (usize, &'static str) { (SECRET_ENGINE.total, &SECRET_ENGINE.hash) }

fn token_len(bytes: &[u8], mode: u8) -> usize {
    let n = bytes.iter().take_while(|&&c| match mode {
        1 => c.is_ascii_uppercase() || c.is_ascii_digit(),
        2 => c.is_ascii_hexdigit(),
        3 => c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'+' | b'/' | b'='),
        4 => c.is_ascii_alphanumeric(),
        _ => c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'),
    }).count();
    if mode == 4 && !bytes[..n].iter().any(u8::is_ascii_digit) { 0 } else { n }
}

fn line_slice(bytes: &[u8], pos: usize) -> &[u8] {
    let lo = bytes[..pos].iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let hi = bytes[pos..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |i| pos + i);
    &bytes[lo..hi]
}

fn has_ignore_tag(line: &[u8]) -> bool {
    const TAGS: &[&[u8]] = &[
        b"gitleaks:allow", b"pragma: allowlist secret", b"trufflehog:ignore", b"refit:ignore",
    ];
    TAGS.iter().any(|t| line.windows(t.len()).any(|w| w.eq_ignore_ascii_case(t)))
}

fn uri_creds_span(bytes: &[u8], start: usize, end: usize) -> Option<(usize, usize)> {
    if !URI_SCHEMES.iter().any(|s| start >= s.len() && bytes[start - s.len()..start].eq_ignore_ascii_case(s)) {
        return None;
    }
    let after = &bytes[end..];
    let stop = after.iter().position(|&b| matches!(b, b'/' | b' ' | b'"' | b'\'' | b')' | b']' | b'\n' | b'\r'))
        .unwrap_or(after.len());
    let at = after[..stop].iter().position(|&b| b == b'@')?;
    let (creds, host) = (&after[..at], &after[at + 1..stop]);
    if host.is_empty() || !creds.contains(&b':') || creds.ends_with(b":") { return None; }
    if creds.windows(2).any(|w| w == b"${") || creds.windows(8).any(|w| w == b"REDACTED") { return None; }
    Some((at + 1, stop))
}

fn shannon_entropy(bytes: &[u8]) -> (f64, usize) {
    let mut freq = [0u32; 128];
    for &b in bytes { if (b as usize) < 128 { freq[b as usize] += 1; } }
    let (len, mut ent, mut uniq) = (bytes.len() as f64, 0.0, 0usize);
    for c in freq {
        if c > 0 { uniq += 1; let p = (c as f64) / len; ent -= p * p.log2(); }
    }
    (ent, uniq)
}

fn find_generic_entropy(bytes: &[u8]) -> Option<usize> {
    const PLACEHOLDERS: &[&[u8]] = &[
        b"example", b"redacted", b"dummy", b"placeholder", b"your_", b"changeme", b"xxxx", b"0000", b"${", b"{{",
    ];
    for mat in SECRET_ENGINE.generic_ac.find_iter(bytes) {
        if !left_bound(bytes, mat.start()) { continue; }
        let rest = &bytes[mat.end()..bytes.len().min(mat.end() + 140)];
        let Some(eq) = rest.iter().position(|&b| matches!(b, b'=' | b':' | b'\n')) else { continue };
        if rest[eq] == b'\n' || eq > 12 { continue; }
        let after = rest[eq + 1..].iter().skip_while(|&&b| matches!(b, b' ' | b'\t')).copied().collect::<Vec<u8>>();
        let Some(&q) = after.first().filter(|&&b| matches!(b, b'"' | b'\'')) else { continue };
        let Some(end) = after[1..].iter().position(|&b| b == q) else { continue };
        let val = &after[1..1 + end];
        if val.len() < 20 || val.iter().any(|b| b.is_ascii_whitespace() || *b == b'/') { continue; }
        let (has_d, has_a) = (val.iter().any(u8::is_ascii_digit), val.iter().any(u8::is_ascii_alphabetic));
        if !has_d || !has_a || PLACEHOLDERS.iter().any(|p| val.windows(p.len()).any(|w| w.eq_ignore_ascii_case(p))) {
            continue;
        }
        let (ent, uniq) = shannon_entropy(val);
        if ent >= 3.8 && uniq >= 12 && !has_ignore_tag(line_slice(bytes, mat.start())) { return Some(mat.start()); }
    }
    None
}

const BASE62_ALPHABET: &[u8; 62] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn ieee_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn encode_base62_6(mut val: u32) -> [u8; 6] {
    let mut out = [b'0'; 6];
    for slot in out.iter_mut().rev() {
        *slot = BASE62_ALPHABET[(val % 62) as usize];
        val /= 62;
    }
    out
}

/// Offline IEEE CRC32-Base62 checksum verifier for 36-char GitHub and npm tokens.
pub fn verify_token_checksum(prefix: &str, body: &[u8]) -> Option<bool> {
    if !matches!(prefix, "ghp_" | "gho_" | "ghu_" | "ghs_" | "ghr_" | "npm_") {
        return None;
    }
    if body.len() != 36 || !body.iter().all(u8::is_ascii_alphanumeric) {
        return None;
    }
    let expected = encode_base62_6(ieee_crc32(&body[..30]));
    Some(&body[30..36] == expected.as_slice())
}

fn invisible_unicode_kind(ch: char) -> Option<&'static str> {
    match ch {
        '\u{E0000}'..='\u{E007F}' => Some("unicode_tag_injection"),
        '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' => Some("unicode_bidi_override"),
        _ => None,
    }
}

/// Detect invisible Unicode Tag prompt-injection or BiDi Trojan Source overrides (1-based line).
pub fn find_invisible_unicode(bytes: &[u8]) -> Option<(usize, &'static str)> {
    memchr::memchr2(0xE2, 0xF3, bytes)?;
    for (idx, line) in bytes.split(|&b| b == b'\n').enumerate() {
        if memchr::memchr2(0xE2, 0xF3, line).is_none() { continue; }
        let kind = line.utf8_chunks()
            .flat_map(|chunk| chunk.valid().chars())
            .find_map(invisible_unicode_kind);
        if let Some(k) = kind && !has_ignore_tag(line) {
            return Some((idx + 1, k));
        }
    }
    None
}

/// Strip invisible Unicode Tag (`U+E0000..=U+E007F`) and BiDi override/isolate codepoints.
pub fn sanitize_invisible_unicode(text: &str) -> (String, usize) {
    if memchr::memchr2(0xE2, 0xF3, text.as_bytes()).is_none() {
        return (text.to_string(), 0);
    }
    let mut out = String::with_capacity(text.len());
    let mut stripped = 0usize;
    for ch in text.chars() {
        if invisible_unicode_kind(ch).is_some() { stripped += 1; } else { out.push(ch); }
    }
    (out, stripped)
}

/// Severity tier for detected secrets and security hazards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretTier { High, Warn }

impl SecretTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Warn => "warn",
        }
    }
}

fn match_slabs(bytes: &[u8]) -> Option<(usize, &'static str)> {
    for (si, slab) in SECRET_ENGINE.slabs.iter().enumerate() {
        let base = slab.rules.len();
        for mat in slab.ac.find_iter(bytes) {
            let pid = mat.pattern().as_usize();
            if si == 0 && pid == base + 1 {
                let ln = line_slice(bytes, mat.start());
                if ln.windows(PEM_MARKER.len()).any(|w| w == PEM_MARKER.as_bytes())
                    && !has_ignore_tag(ln)
                {
                    return Some((mat.start(), "pem_private_key"));
                }
            } else if si == 0 && pid == base {
                if uri_creds_span(bytes, mat.start(), mat.end()).is_some()
                    && !has_ignore_tag(line_slice(bytes, mat.start()))
                {
                    return Some((mat.start(), "database_credential_url"));
                }
            } else {
                let r = &slab.rules[pid];
                let tlen = token_len(&bytes[mat.end()..], r.mode);
                if left_bound(bytes, mat.start()) && tlen >= r.dmin
                    && !has_ignore_tag(line_slice(bytes, mat.start()))
                {
                    let body = &bytes[mat.end()..mat.end() + tlen];
                    if verify_token_checksum(r.pfx, body) == Some(false) { continue; }
                    return Some((mat.start(), r.kind));
                }
            }
        }
    }
    None
}

fn line_number_at(bytes: &[u8], pos: usize) -> usize {
    bytes[..pos].iter().filter(|&&b| b == b'\n').count() + 1
}

/// Single-pass 2-slab AVX2 Fat-Teddy SIMD detection across 120+ prefix/URI/PEM rules.
pub fn find_secret(line: &str) -> Option<&'static str> { match_slabs(line.as_bytes()).map(|(_, k)| k) }

/// Tiered binary-safe secret & invisible-Unicode scanner (`High` blocks, `Warn` advises).
pub fn find_secret_tiered(bytes: &[u8]) -> Option<(usize, &'static str, SecretTier)> {
    if bytes[..bytes.len().min(8000)].contains(&0) { return None; }
    if let Some((pos, kind)) = match_slabs(bytes) {
        return Some((line_number_at(bytes, pos), kind, SecretTier::High));
    }
    if let Some((line, kind)) = find_invisible_unicode(bytes) {
        return Some((line, kind, SecretTier::High));
    }
    let pos = find_generic_entropy(bytes)?;
    Some((line_number_at(bytes, pos), "high_entropy_secret", SecretTier::Warn))
}

/// Binary-safe zero-copy secret scan with Tier-4 high-entropy generic secret detection.
#[cfg_attr(not(test), expect(dead_code))]
pub fn find_secret_in_bytes(bytes: &[u8]) -> Option<(usize, &'static str)> {
    find_secret_tiered(bytes).map(|(line, kind, _)| (line, kind))
}

fn consume_pem_block(text: &str, end: usize) -> Option<usize> {
    let tail_start = end + text[end..].find(PEM_MARKER)? + PEM_MARKER.len();
    let tail = &text[tail_start..];
    let Some(end_rel) = tail.find("-----END ") else { return Some(text.len()) };
    let Some(close_rel) = tail[end_rel..].find(PEM_MARKER) else { return Some(text.len()) };
    Some(tail_start + end_rel + close_rel + PEM_MARKER.len())
}

/// Single-pass SIMD secret redaction across all secret prefixes, URIs, PEM blocks, and invisible Unicode.
pub fn redact_secrets(text: String) -> String {
    let text = if memchr::memchr2(0xE2, 0xF3, text.as_bytes()).is_some() {
        sanitize_invisible_unicode(&text).0
    } else {
        text
    };
    let bytes = text.as_bytes();
    if SECRET_ENGINE.slabs.iter().all(|s| s.ac.find(bytes).is_none()) && !text.contains(PEM_MARKER) {
        return text;
    }
    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    for (si, slab) in SECRET_ENGINE.slabs.iter().enumerate() {
        let base = slab.rules.len();
        for mat in slab.ac.find_iter(bytes) {
            let pid = mat.pattern().as_usize();
            if si == 0 && pid == base + 1 {
                if let Some(next_pos) = consume_pem_block(&text, mat.end()) {
                    hits.push((mat.start(), next_pos, "[REDACTED_PEM_PRIVATE_KEY]".into()));
                }
            } else if si == 0 && pid == base {
                if let Some((host_off, stop)) = uri_creds_span(bytes, mat.start(), mat.end()) {
                    let host = &text[mat.end() + host_off..mat.end() + stop];
                    hits.push((mat.end(), mat.end() + stop, format!("REDACTED:REDACTED@{host}")));
                }
            } else {
                let r = &slab.rules[pid];
                let skip = token_len(&bytes[mat.end()..], 0);
                if left_bound(bytes, mat.start()) && skip >= r.rmin {
                    hits.push((mat.start(), mat.end() + skip, r.tag.into()));
                }
            }
        }
    }
    hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    let (mut out, mut cursor) = (String::with_capacity(text.len()), 0usize);
    for (s, e, rep) in hits {
        if s < cursor { continue; }
        out.push_str(&text[cursor..s]);
        out.push_str(&rep);
        cursor = e;
    }
    out.push_str(&text[cursor..]);
    if out.contains(PEM_MARKER) { out.replace(PEM_MARKER, "REDACTED_PRIVATE_KEY-----") } else { out }
}

#[cfg(test)]
use crate::testutil::{TempRepo, TestRng};

#[cfg(test)]
#[test]
fn atomic_io_and_clock() {
    let dir = TempRepo::with_files("plat_io", &[
        (".gitignore", "ignored_gen/\n*.generated.py\n"), (".refitignore", "custom_skip/\n"),
        ("src/keep.rs", "pub fn keep() {}\n"), (".git/HEAD", "ref: refs/heads/main\n"),
        ("node_modules/pkg/index.js", "module.exports = 1;\n"), ("target/debug/bin.rs", "fn main() {}\n"),
        (".refit/cas/blob", "x"), ("ignored_gen/junk.py", "x = 1\n"),
        ("src/auto.generated.py", "x = 2\n"), ("custom_skip/skip.ts", "export const x = 1;\n"),
    ]);
    let rels: Vec<String> = walk_files(dir.path(), None).iter()
        .filter_map(|p| crate::source::rel_norm(dir.path(), p)).collect();
    assert_eq!(rels, vec![".gitignore", ".refitignore", "src/keep.rs"]);

    let target = dir.join("state.json");
    for rev in 0..25 {
        let payload = format!("{{\"rev\":{rev}}}");
        write_atomic(&target, payload.as_bytes()).unwrap();
        assert_eq!(dir.read("state.json"), payload);
    }

    let mut prev = now_ms();
    for _ in 0..10_000 { let cur = now_ms(); assert!(cur >= prev); prev = cur; }

    let big = dir.join("big.txt");
    fs::write(&big, "hello\n".repeat(4000)).unwrap();
    let (len, hex) = FileBuf::new().hash_file_blake3(&big).unwrap();
    let raw = fs::read(&big).unwrap();
    assert_eq!((len, hex), (raw.len() as u64, blake3::hash(&raw).to_hex().to_string()));
}

#[cfg(test)]
#[test]
fn sidecar_lock_and_cmd() {
    let dir = TempRepo::new("plat_lock");
    let lp = dir.join(".refit/test.lock");
    let lk = SidecarLock::acquire(&lp).unwrap();
    assert!(SidecarLock::try_acquire(&lp).unwrap().is_none());
    drop(lk);

    let lk2 = SidecarLock::try_acquire(&lp).unwrap().expect("lock reacquired");
    assert!(SidecarLock::try_acquire(&lp).unwrap().is_none());
    drop(lk2);

    let cwd = std::env::temp_dir();
    let (prog, flag, slow_cmd) = if cfg!(windows) {
        ("cmd", "/C", "ping -n 5 127.0.0.1 >nul")
    } else {
        ("sh", "-c", "sleep 3")
    };
    let run = |cmd: &str, ms: u64| {
        run_cmd(prog, &[flag.into(), cmd.into()], &cwd, &[], false, ms).unwrap()
    };

    let ok = run("echo ok", 5_000);
    let ok_out = ok.stdout.trim().to_ascii_lowercase();
    assert_eq!((ok.exit_code, ok.timed_out, ok_out.as_str()), (0, false, "ok"));

    let slow = run(slow_cmd, 150);
    assert_eq!((slow.timed_out, slow.exit_code), (true, -1));
}

#[cfg(test)]
#[test]
fn http_and_retry_logic() {
    let _c1 = new_http(concat!("refit-probe/", env!("CARGO_PKG_VERSION")), 30, 16).unwrap();
    let _c2 = stream_http(concat!("refit-stream/", env!("CARGO_PKG_VERSION")), 15, 600, 16)
        .unwrap();
    for code in [200, 201, 204, 400, 401, 403, 404, 422, 432] {
        assert!(!is_retryable(code), "code={code}");
    }
    assert!(is_quota_exhausted(432) && !is_quota_exhausted(429));
    for code in [408, 429, 500, 502, 503, 504, 529] {
        assert!(is_retryable(code), "code={code}");
    }

    let cases = [
        ("3", Some(3_000)),
        ("1.5", Some(1_500)),
        ("9999", Some(30_000)),
        ("-5", None),
        ("bad", None),
    ];
    for (raw, want) in cases {
        assert_eq!(retry_ms(raw, 30_000), want);
    }

    let soon = httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(5));
    assert!((2_000..=6_500).contains(&retry_ms(&soon, 30_000).unwrap()));
    let far = httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(3600));
    assert_eq!(retry_ms(&far, 15_000), Some(15_000));
    assert_eq!(backoff_ms(0, 200, 10_000, Some(4_500), 1), 4_500);
    assert_ne!(
        backoff_ms(2, 200, 10_000, None, 11),
        backoff_ms(2, 200, 10_000, None, 97)
    );
}

#[cfg(test)]
fn make_valid_crc_token(prefix: &str, entropy30: &str) -> String {
    assert_eq!(entropy30.len(), 30);
    let tail = encode_base62_6(ieee_crc32(entropy30.as_bytes()));
    format!("{prefix}{entropy30}{}", std::str::from_utf8(&tail).unwrap())
}

#[cfg(test)]
#[test]
fn secret_detection_redaction() {
    let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
    let gh = make_valid_crc_token("ghp_", "1234567890abcdefghij1234567890");
    let gh_bad = ["ghp_", "1234567890abcdefghij1234567890", "000000"].concat();
    let npm_ok = make_valid_crc_token("npm_", "abcdefghij1234567890ABCDEFGHIJ");
    let npm_bad = ["npm_", "abcdefghij1234567890ABCDEFGHIJ", "zzzzzz"].concat();
    let db = ["postgres://", "admin:sup3rs3cr3t@db.internal:5432/prod"].concat();
    let slack = ["xoxa-", "1234567890abcdef1234"].concat();
    let age = ["AGE-SECRET-KEY-1", "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"].concat();
    let pem = concat!(
        "-----BEGIN RSA ",
        "PRIVATE KEY-----\nMIIEpAIBAAKCAQEASuperSecretBodyPayload==\n",
        "-----END RSA PRIVATE KEY-----",
    ).to_string();
    let multi = format!("docs = 'postgres://localhost/db' prod = '{}'",
        ["postgres://", "admin:p4ssw0rd@db.prod:5432/app"].concat());

    assert_eq!(ieee_crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(verify_token_checksum("ghp_", &gh.as_bytes()[4..]), Some(true));
    assert_eq!(verify_token_checksum("ghp_", &gh_bad.as_bytes()[4..]), Some(false));
    assert_eq!(verify_token_checksum("npm_", &npm_ok.as_bytes()[4..]), Some(true));
    assert_eq!(verify_token_checksum("npm_", &npm_bad.as_bytes()[4..]), Some(false));
    assert_eq!(verify_token_checksum("xoxa-", &gh.as_bytes()[4..]), None);

    for (sample, want) in [
        (format!("key = \"{aws}\""), Some("aws_access_key_id")),
        (format!("tok = '{gh}'"), Some("github_pat")),
        (format!("tok = '{gh_bad}'"), None),
        (format!("tok = '{npm_ok}'"), Some("npm_access_token")),
        (format!("tok = '{npm_bad}'"), None),
        (format!("k = \"{age}\""), Some("age_secret_key")),
        (format!("tok = '{gh}' # gitleaks:allow"), None),
        (db.clone(), Some("database_credential_url")),
        (pem.clone(), Some("pem_private_key")),
        (multi, Some("database_credential_url")),
        ("let normal = \"hello_world\";".into(), None),
    ] {
        assert_eq!(find_secret(&sample), want, "sample={sample}");
    }
    assert!(secret_rule_meta().0 >= 130);
    assert_eq!(find_secret_in_bytes(b"\x00\x01AKIAIOSFODNN7EXAMPLE"), None);
    let ent_line = ["api_key = \"", "aB9xQ2mK7pZ4vL8nR3tW6yC1eH5j\""].concat();
    assert_eq!(find_secret_in_bytes(ent_line.as_bytes()), Some((1, "high_entropy_secret")));
    assert_eq!(
        find_secret_tiered(ent_line.as_bytes()),
        Some((1, "high_entropy_secret", SecretTier::Warn))
    );
    assert_eq!(
        find_secret_tiered(format!("tok = '{gh}'").as_bytes()),
        Some((1, "github_pat", SecretTier::High))
    );

    let tag_inj = "ok\nhidden \u{E0001}\u{E0041} text\n";
    assert_eq!(find_invisible_unicode(tag_inj.as_bytes()), Some((2, "unicode_tag_injection")));
    assert_eq!(find_secret_tiered(tag_inj.as_bytes()), Some((2, "unicode_tag_injection", SecretTier::High)));
    let bidi = "let access = \"\u{202E}admin\";\n";
    assert_eq!(find_invisible_unicode(bidi.as_bytes()), Some((1, "unicode_bidi_override")));
    let bidi_ignored = "let access = \"\u{202E}admin\"; // refit:ignore\n";
    assert_eq!(find_invisible_unicode(bidi_ignored.as_bytes()), None);
    assert_eq!(sanitize_invisible_unicode(tag_inj), ("ok\nhidden  text\n".into(), 2));
    assert_eq!(sanitize_invisible_unicode("clean ascii"), ("clean ascii".into(), 0));

    let red = redact_secrets(format!("a={aws} b={gh} c={db} s={slack} p={pem}"));
    for leak in [aws.as_str(), gh.as_str(), slack.as_str(), "sup3rs3cr3t", "SuperSecretBodyPayload"] {
        assert!(!red.contains(leak), "leaked {leak}");
    }
    for tag in ["REDACTED_AWS_", "REDACTED_GITHUB_", "REDACTED_SLACK_"] { assert!(red.contains(tag)); }
}

#[cfg(test)]
#[test]
fn stat_bin_migrates_stale() {
    let tmp = TempRepo::new("io_stat_mig");
    let cache = tmp.join(".refit/cache/stat.bin");
    let mut stale_lz4 = vec![12, 0, 0, 0];
    stale_lz4.extend_from_slice(b"RFST0001\x00\x00\x00\x00");
    let stale_zst = zstd::bulk::compress(b"RFST0001\x00\x00\x00\x00", 1).unwrap();
    for payload in [&stale_lz4, &stale_zst] {
        write_atomic(&cache, payload).unwrap();
        assert!(load_stat_cache(&cache).is_empty());
    }

    let mtime = 1_000_000_000_000u64;
    let entry = StatEntry { mtime_ns: mtime, size: 128, blake3: "abcd".into() };
    save_stat_cache(&cache, &BTreeMap::from([("src/a.rs".into(), entry)])).unwrap();
    assert!(has_cachedir_tag(&tmp.join(".refit/cache")));
    assert_eq!(&fs::read(&cache).unwrap()[..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    let loaded = load_stat_cache(&cache);
    let ent = loaded.get("src/a.rs").unwrap();
    assert!(!stat_cache_fresh(ent, mtime, 128, mtime + RACY_GIT_NS - 1));
    assert!(stat_cache_fresh(ent, mtime, 128, mtime + RACY_GIT_NS));
    assert!(!stat_cache_fresh(ent, mtime, 129, mtime + RACY_GIT_NS));
}

#[cfg(test)]
fn sieve_step(
    cache: &SieveCache<u32, u32>, model: &mut BTreeMap<u32, (u32, usize)>, rng: &mut TestRng, step: u32,
) {
    let key = rng.next_usize(20) as u32;
    match rng.next_usize(10) {
        0..=4 => {
            let weight = 1 + rng.next_usize(320);
            let had = model.get(&key).map(|item| item.0);
            assert_eq!(cache.insert(key, step, weight), had, "step={step} k={key}");
            if weight > 256 {
                model.remove(&key);
                assert_eq!(cache.get(&key), None);
            } else {
                model.insert(key, (step, weight));
                assert_eq!(cache.get(&key), Some(step));
            }
        }
        5..=7 => assert_eq!(cache.get(&key), model.get(&key).map(|item| item.0)),
        _ => assert_eq!(cache.remove(&key), model.remove(&key).map(|item| item.0)),
    }
    let entries: Vec<(u32, (u32, usize))> = cache.inner.iter().collect();
    let exact_w: usize = entries.iter().map(|&(_, (_, weight))| weight).sum();
    assert!(exact_w <= cache.max_bytes && cache.len() <= 8);
    for &(ck, (cv, cw)) in &entries { assert_eq!(model.get(&ck), Some(&(cv, cw))); }
    model.retain(|mk, _| cache.inner.peek(mk).is_some());
}

#[cfg(test)]
#[test]
fn sieve_model_and_byte_cap() {
    let hot = SieveCache::<&str, u32>::new(3, 1024);
    for (key, val) in [("hot", 1), ("c1", 2), ("c2", 3)] { hot.insert(key, val, 100); }
    assert_eq!(hot.get(&"hot"), Some(1));
    hot.insert("fresh", 4, 100);
    assert_eq!((hot.len(), hot.get(&"hot"), hot.get(&"fresh")), (3, Some(1), Some(4)));

    let cache = SieveCache::<u32, u32>::new(8, 256);
    let (mut model, mut rng) = (BTreeMap::new(), TestRng::new(0x51E7_CAFE));
    for step in 0..1_000u32 { sieve_step(&cache, &mut model, &mut rng, step); }
}

#[cfg(test)]
#[test]
fn epoch_single_publish_on_commit() {
    let tmp = TempRepo::with_files("epoch_mvcc", &[("src/a.rs", "pub fn a() -> i32 { 1 }\n")]);
    let reg = std::sync::Arc::new(EpochRegistry::new());
    let store = crate::mvcc::MvccStore::open_with_epoch(tmp.path(), "ep", reg.clone()).unwrap();
    let ep0 = reg.global_epoch();
    let exp = BTreeMap::from([("src/a.rs".into(), tmp.hash("src/a.rs"))]);
    let next = BTreeMap::from([("src/a.rs".into(), b"pub fn a() -> i32 { 2 }\n".to_vec())]);
    let out = store.commit_tx("mod_a", &exp, &next, &[]).unwrap();
    std::assert_matches!(out, crate::mvcc::CommitOutcome::Committed { head_rev: 1, .. });

    let ep1 = reg.global_epoch();
    assert_eq!((ep1, reg.slot_epoch("src/a.rs"), reg.slot_epoch("mod_a")), (ep0 + 1, ep1, ep1));
    let def = crate::mvcc::MvccStore::open(tmp.path(), "ep-def").unwrap();
    assert!(std::sync::Arc::ptr_eq(def.epoch_registry(), &EpochRegistry::global_arc()));
}
