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
use anyhow::{Context, Result};
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

const DB_SCHEMES: &[&str] = &[
    "postgres://", "postgresql://", "mysql://", "mongodb://", "mongodb+srv://", "redis://",
];

// (prefix, kind, redact_tag, detect_min, redact_min, upper_only)
const SECRET_SPECS: &[(&str, &str, &str, usize, usize, bool)] = &[
    ("csk-", "cerebras_api_key", "REDACTED_CSK_", 12, 8, false),
    ("keen_", "keenable_api_key", "REDACTED_KEEN_", 12, 8, false),
    ("sk-ant-", "anthropic_api_key", "REDACTED_ANTHROPIC_", 16, 12, false),
    ("sk-proj-", "openai_api_key", "REDACTED_OPENAI_", 16, 12, false),
    ("sk_live_", "stripe_live_key", "REDACTED_STRIPE_", 16, 12, false),
    ("rk_live_", "stripe_restricted_key", "REDACTED_STRIPE_", 16, 12, false),
    ("ghp_", "github_pat", "REDACTED_GITHUB_", 20, 16, false),
    ("gho_", "github_oauth", "REDACTED_GITHUB_", 20, 16, false),
    ("github_pat_", "github_fine_grained_pat", "REDACTED_GITHUB_", 20, 16, false),
    ("xoxb-", "slack_bot_token", "REDACTED_SLACK_", 16, 12, false),
    ("xoxp-", "slack_user_token", "REDACTED_SLACK_", 16, 12, false),
    ("xoxa-", "slack_app_token", "REDACTED_SLACK_", 16, 12, false),
    ("AKIA", "aws_access_key_id", "REDACTED_AWS_", 16, 16, true),
    ("ASIA", "aws_sts_key_id", "REDACTED_AWS_", 16, 16, true),
    ("sk-", "generic_sk_secret", "REDACTED_SK_", 24, 24, false),
];

const PEM_MARKER: &str = "PRIVATE KEY-----";
const DB_BASE: usize = SECRET_SPECS.len();
const PEM_IDX: usize = DB_BASE + DB_SCHEMES.len();

static SECRET_AC: LazyLock<AhoCorasick> = LazyLock::new(|| {
    let patterns = SECRET_SPECS.iter().map(|spec| spec.0)
        .chain(DB_SCHEMES.iter().copied()).chain(std::iter::once("-----BEGIN "));
    AhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest).build(patterns)
        .expect("secret aho-corasick")
});

fn secret_token_len(tail: &str, upper_only: bool) -> usize {
    tail.bytes().take_while(|byte| {
        if upper_only { byte.is_ascii_uppercase() || byte.is_ascii_digit() }
        else { byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') }
    }).count()
}

fn db_creds_span(after: &str) -> Option<(usize, &str, &str)> {
    let end = after.find(['/', ' ', '"', '\'', ')', ']', '\n', '\r']).unwrap_or(after.len());
    let (creds, host) = after[..end].split_once('@')?;
    (!host.is_empty() && creds.contains(':')).then_some((end, creds, host))
}

/// Single-pass SIMD secret detection across all 22 prefix/URI/PEM patterns.
pub fn find_secret(line: &str) -> Option<&'static str> {
    let bytes = line.as_bytes();
    for mat in SECRET_AC.find_iter(line) {
        let pid = mat.pattern().as_usize();
        if pid == PEM_IDX {
            if line[mat.end()..].contains(PEM_MARKER) { return Some("pem_private_key"); }
        } else if pid < DB_BASE {
            let (_, kind, _, dmin, _, upper) = SECRET_SPECS[pid];
            if left_bound(bytes, mat.start()) && secret_token_len(&line[mat.end()..], upper) >= dmin {
                return Some(kind);
            }
        } else if let Some((_, creds, _)) = db_creds_span(&line[mat.end()..])
            && !creds.ends_with(':') && !creds.contains("${") && !creds.contains("REDACTED")
        {
            return Some("database_credential_url");
        }
    }
    None
}

fn consume_pem_block(text: &str, end: usize) -> Option<usize> {
    let tail_start = end + text[end..].find(PEM_MARKER)? + PEM_MARKER.len();
    let tail = &text[tail_start..];
    let Some(end_rel) = tail.find("-----END ") else { return Some(text.len()) };
    let Some(close_rel) = tail[end_rel..].find(PEM_MARKER) else { return Some(text.len()) };
    Some(tail_start + end_rel + close_rel + PEM_MARKER.len())
}

/// Single-pass SIMD secret redaction across all secret prefixes, DB URIs, and PEM blocks.
pub fn redact_secrets(text: String) -> String {
    if SECRET_AC.find(&text).is_none() && !text.contains(PEM_MARKER) { return text; }
    let bytes = text.as_bytes();
    let (mut out, mut cursor) = (String::with_capacity(text.len()), 0usize);

    for mat in SECRET_AC.find_iter(&text) {
        if mat.start() < cursor { continue; }
        let pid = mat.pattern().as_usize();
        if pid == PEM_IDX {
            if let Some(next_pos) = consume_pem_block(&text, mat.end()) {
                out.push_str(&text[cursor..mat.start()]);
                out.push_str("[REDACTED_PEM_PRIVATE_KEY]");
                cursor = next_pos;
            }
        } else if pid < DB_BASE {
            let (_, _, tag, _, rmin, _) = SECRET_SPECS[pid];
            let skip = secret_token_len(&text[mat.end()..], false);
            if left_bound(bytes, mat.start()) && skip >= rmin {
                out.push_str(&text[cursor..mat.start()]);
                out.push_str(tag);
                cursor = mat.end() + skip;
            }
        } else if let Some((span_end, _, host)) = db_creds_span(&text[mat.end()..]) {
            out.push_str(&text[cursor..mat.end()]);
            out.push_str("REDACTED:REDACTED@");
            out.push_str(host);
            cursor = mat.end() + span_end;
        }
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
    ]);
    let rels: Vec<String> = walk_files(dir.path(), None)
        .iter()
        .filter_map(|path| crate::source::rel_norm(dir.path(), path))
        .collect();
    assert_eq!(rels, vec![".gitignore", ".refitignore", "src/keep.rs"]);

    let target = dir.join("state.json");
    for rev in 0..25 {
        let payload = format!("{{\"rev\":{rev}}}");
        write_atomic(&target, payload.as_bytes()).unwrap();
        assert_eq!(dir.read("state.json"), payload);
    }

    let mut prev = now_ms();
    for _ in 0..10_000 {
        let cur = now_ms();
        assert!(cur >= prev);
        prev = cur;
    }

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
#[test]
fn secret_detection_redaction() {
    let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
    let gh = ["ghp_", "1234567890abcdefghij123456"].concat();
    let db = ["postgres://", "admin:sup3rs3cr3t@db.internal:5432/prod"].concat();
    let slack = ["xoxa-", "1234567890abcdef1234"].concat();
    let pem = [
        "-----BEGIN RSA ",
        "PRIVATE KEY-----\nMIIEpAIBAAKCAQEASuperSecretBodyPayload==\n",
        "-----END RSA PRIVATE KEY-----",
    ]
    .concat();
    let multi =
        "docs = 'postgres://localhost/db' prod = 'postgres://admin:p4ssw0rd@db.prod:5432/app'";

    let samples = [
        (format!("key = \"{aws}\""), Some("aws_access_key_id")),
        (format!("tok = '{gh}'"), Some("github_pat")),
        (db.clone(), Some("database_credential_url")),
        (pem.clone(), Some("pem_private_key")),
        (multi.into(), Some("database_credential_url")),
        ("let normal = \"hello_world\";".into(), None),
    ];
    for (sample, want) in samples {
        assert_eq!(find_secret(&sample), want);
    }

    let red = redact_secrets(format!("a={aws} b={gh} c={db} s={slack} p={pem}"));
    let leaks = [
        aws.as_str(),
        gh.as_str(),
        slack.as_str(),
        "sup3rs3cr3t",
        "SuperSecretBodyPayload",
    ];
    for leak in leaks {
        assert!(!red.contains(leak), "leaked {leak}");
    }
    for tag in ["REDACTED_AWS_", "REDACTED_GITHUB_", "REDACTED_SLACK_"] {
        assert!(red.contains(tag));
    }
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
