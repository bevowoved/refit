use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const FILE_BUF_SIZE: usize = 64 * 1024;
#[cfg(target_os = "linux")]
pub const FADV_SEQ_MIN: u64 = 64 * 1024;
#[cfg(target_os = "linux")]
pub const FADV_DONTNEED_MIN: u64 = 1024 * 1024;
const ZSTD_STREAM_WLOG: u32 = 18;
const STAT_MAGIC: &[u8; 8] = b"RFST0001";
const RACY_GIT_NS: u64 = 2_000_000_000;
const CACHEDIR_TAG_BODY: &[u8] = b"Signature: 8a477f597d28d172789f06886806bc55\n";

static STREAM_TMP_SEQ: AtomicU64 = AtomicU64::new(1);

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

/// Apply `POSIX_FADV_SEQUENTIAL` strictly for medium files `[64 KiB, 1 MiB)`.
pub fn advise_sequential_if_medium(file: &File, len: u64) {
    #[cfg(target_os = "linux")]
    {
        if (FADV_SEQ_MIN..FADV_DONTNEED_MIN).contains(&len) {
            let _ = nix::fcntl::posix_fadvise(
                file,
                0,
                0,
                nix::fcntl::PosixFadviseAdvice::POSIX_FADV_SEQUENTIAL,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, len);
}

/// Evict page-cache pages strictly for large files (`>= 1 MiB`).
/// Callers writing files MUST call `sync_file_barrier` BEFORE calling this!
pub fn advise_dontneed_if_large(file: &File, len: u64) {
    #[cfg(target_os = "linux")]
    {
        if len >= FADV_DONTNEED_MIN {
            let _ = nix::fcntl::posix_fadvise(
                file,
                0,
                0,
                nix::fcntl::PosixFadviseAdvice::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, len);
}

/// Cross-OS durability barrier (`fdatasync` on Linux, `F_FULLFSYNC` on macOS, `sync_data` on Win).
pub fn sync_file_barrier(file: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        if nix::fcntl::fcntl(file, nix::fcntl::FcntlArg::F_FULLFSYNC).is_ok() {
            return Ok(());
        }
    }
    file.sync_data()
}

/// Stream a JSON value directly into a temporary Zstd file (`windowLog <= 18`) and atomically swap.
pub fn write_zstd_stream<T: Serialize>(path: &Path, val: &T) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create zstd parent {}", parent.display()))?;

    let seq = STREAM_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("cache");
    let tmp = parent.join(format!(".refit.zst.{}.{seq}.{name}", std::process::id()));

    let written_len = {
        let f = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        let mut enc = zstd::stream::Encoder::new(f, 1)?;
        let _ = enc.window_log(ZSTD_STREAM_WLOG);
        serde_json::to_writer(&mut enc, val)?;
        let f = enc.finish()?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        sync_file_barrier(&f)?;
        advise_dontneed_if_large(&f, len);
        len
    };
    let _ = written_len;
    super::replace_file_pub(&tmp, path, parent)
}

/// Stream-decode a JSON Zstd file with `O(1)` read overhead.
pub fn read_zstd_stream<T: DeserializeOwned>(path: &Path) -> Option<T> {
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
    let parent = dst
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let seq = STREAM_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let name = dst.file_name().and_then(|s| s.to_str()).unwrap_or("out.zst");
    let tmp = parent.join(format!(".refit.zst.{}.{seq}.{name}", std::process::id()));

    let (raw_len, zst_len) = {
        let mut reader = File::open(src).with_context(|| format!("open {}", src.display()))?;
        let src_len = reader.metadata().map(|m| m.len()).unwrap_or(0);
        advise_sequential_if_medium(&reader, src_len);

        let writer = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        let mut enc = zstd::stream::Encoder::new(writer, level)?;
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

        let out_file = enc.finish()?;
        let out_len = out_file.metadata().map(|m| m.len()).unwrap_or(0);
        sync_file_barrier(&out_file)?;
        advise_dontneed_if_large(&out_file, out_len);
        (raw_total, out_len)
    };

    super::replace_file_pub(&tmp, dst, parent)?;
    Ok((raw_len, zst_len))
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
    let Ok(dec) = lz4_flex::decompress_size_prepended(&raw) else { return BTreeMap::new() };
    decode_stat_map(&dec).unwrap_or_default()
}

pub fn save_stat_cache(path: &Path, map: &BTreeMap<String, StatEntry>) -> Result<()> {
    if let Some(p) = path.parent() {
        ensure_cachedir_tag(p);
    }
    let raw = encode_stat_map(map);
    let compressed = lz4_flex::compress_prepend_size(&raw);
    super::write_atomic(path, &compressed)
}

fn encode_stat_map(map: &BTreeMap<String, StatEntry>) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 112);
    out.extend_from_slice(STAT_MAGIC);
    out.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (rel, e) in map {
        let rb = rel.as_bytes();
        let hb = e.blake3.as_bytes();
        out.extend_from_slice(&(rb.len().min(u16::MAX as usize) as u16).to_le_bytes());
        out.extend_from_slice(&rb[..rb.len().min(u16::MAX as usize)]);
        out.extend_from_slice(&e.mtime_ns.to_le_bytes());
        out.extend_from_slice(&e.size.to_le_bytes());
        out.extend_from_slice(&(hb.len().min(u16::MAX as usize) as u16).to_le_bytes());
        out.extend_from_slice(&hb[..hb.len().min(u16::MAX as usize)]);
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
        if let Some(p) = lock_path.parent() {
            fs::create_dir_all(p)
                .with_context(|| format!("create lock dir {}", p.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .with_context(|| format!("open sidecar lock {}", lock_path.display()))?;

        match file.lock() {
            Ok(()) => Ok(Self::OsFile(file)),
            Err(err) if is_lock_unsupported(&err) => acquire_excl_fallback(lock_path),
            Err(err) => Err(err).with_context(|| format!("acquire lock {}", lock_path.display())),
        }
    }

    pub fn try_acquire(lock_path: &Path) -> Result<Option<Self>> {
        if let Some(p) = lock_path.parent() {
            fs::create_dir_all(p)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;

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
            Self::OsFile(f) => {
                let _ = f.unlock();
            }
            Self::ExclLease(p) => {
                let _ = fs::remove_file(p);
            }
        }
    }
}

fn is_lock_unsupported(err: &io::Error) -> bool {
    if err.kind() == io::ErrorKind::Unsupported {
        return true;
    }
    #[cfg(unix)]
    {
        if let Some(code) = err.raw_os_error() {
            let errno = nix::errno::Errno::from_raw(code);
            if errno == nix::errno::Errno::EOPNOTSUPP
                || errno == nix::errno::Errno::ENOTSUP
                || errno == nix::errno::Errno::ENOLCK
                || errno == nix::errno::Errno::ENOSYS
            {
                return true;
            }
        }
    }
    false
}

fn excl_lease_path(lock_path: &Path) -> PathBuf {
    let name = lock_path.file_name().and_then(|s| s.to_str()).unwrap_or("refit.lock");
    lock_path.with_file_name(format!("{name}.excl"))
}

fn try_acquire_excl_once(lock_path: &Path) -> Result<Option<SidecarLock>> {
    let lease = excl_lease_path(lock_path);
    match OpenOptions::new().create_new(true).write(true).open(&lease) {
        Ok(mut f) => {
            let _ = writeln!(f, "{} {}", std::process::id(), super::now_ms());
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

fn excl_holder_alive(lease: &Path) -> bool {
    #[cfg(unix)]
    {
        let Ok(raw) = fs::read_to_string(lease) else { return false };
        let Some(pid) = raw.split_whitespace().next().and_then(|s| s.parse::<i32>().ok()) else {
            return false;
        };
        if pid <= 0 {
            return false;
        }
        return matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Ok(()) | Err(nix::errno::Errno::EPERM)
        );
    }
    #[cfg(not(unix))]
    {
        let _ = lease;
        false
    }
}

fn reap_stale_excl(lease: &Path) {
    if excl_holder_alive(lease) {
        return;
    }
    let Ok(meta) = fs::metadata(lease) else { return };
    let stale = meta
        .modified()
        .ok()
        .and_then(|m| m.elapsed().ok())
        .is_some_and(|age| age > Duration::from_secs(60));
    if stale {
        let _ = fs::remove_file(lease);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn stat_cache_racy_git_guard_and_lz4_roundtrip() {
        let tmp = TempRepo::new("io_stat");
        let cache = tmp.join(".refit/cache/stat.bin");
        let old_ns = 1_000_000_000_000u64;
        let now_ns = old_ns + 5_000_000_000;
        let mut map = BTreeMap::new();
        map.insert("src/a.rs".into(), StatEntry { mtime_ns: old_ns, size: 128, blake3: "abcd".into() });
        save_stat_cache(&cache, &map).unwrap();

        assert!(has_cachedir_tag(&tmp.join(".refit/cache")));
        let loaded = load_stat_cache(&cache);
        let e = loaded.get("src/a.rs").unwrap();
        assert!(stat_cache_fresh(e, old_ns, 128, now_ns));
        // Within 2-second racy-git window must reject stat cache hit!
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
