#[cfg(test)]
use super::runners::find_tool_in_dirs;
use super::runners::find_tool_on_path;
use crate::platform::{
    SidecarLock, budget, ensure_cachedir_tag, env_parse, env_truthy, file_mtime_size, now_ms,
    par_map, walk_files,
};
use crate::source::{rel_norm, work_dir};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::mem::DropGuard;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const DEFAULT_CACHE_CAP_BYTES: u64 = 600 * 1024 * 1024;
pub const BLAZE_MIN_MEM_BYTES: u64 = 16 * 1024 * 1024 * 1024;

static STAGE_SEQ: AtomicU64 = AtomicU64::new(1);
static STAGE_HARDLINKS_TOTAL: AtomicU64 = AtomicU64::new(0);
static STAGE_COPIED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CARGO_SLOT_POOL: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
static CARGO_EPHEMERAL_SEQ: AtomicU64 = AtomicU64::new(100);
static RUST_WRAPPER_CACHE: OnceLock<Option<(RustCacheBackend, PathBuf)>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneOutcome { Reflinked, HardLinked, Copied }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustCacheBackend { Mbx, Kache }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheGcStats { pub evicted_entries: usize, pub freed_bytes: u64, pub remaining_bytes: u64 }

pub fn stage_hardlinks() -> u64 { STAGE_HARDLINKS_TOTAL.load(Ordering::Relaxed) }
#[cfg(test)] pub fn stage_copies() -> u64 { STAGE_COPIED_TOTAL.load(Ordering::Relaxed) }

/// Zero-spawn process-wide PATH probe (`mbx` then `kache`) cached once via `OnceLock`.
pub fn detect_rust_wrapper() -> Option<(RustCacheBackend, &'static Path)> {
    RUST_WRAPPER_CACHE.get_or_init(|| {
        find_tool_on_path("mbx").map(|p| (RustCacheBackend::Mbx, p))
            .or_else(|| find_tool_on_path("kache").map(|p| (RustCacheBackend::Kache, p)))
    }).as_ref().map(|(b, p)| (*b, p.as_path()))
}

fn is_copy_fallback_err(err: &io::Error, include_emlink: bool) -> bool {
    if matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::AlreadyExists) { return false; }
    if matches!(err.kind(), io::ErrorKind::PermissionDenied | io::ErrorKind::Unsupported | io::ErrorKind::CrossesDevices) {
        return true;
    }
    let Some(raw) = err.raw_os_error() else { return false };
    let unix_match = matches!(raw, 1 | 18 | 22 | 38 | 45 | 78 | 95) || (include_emlink && raw == 31);
    let win_match = cfg!(windows)
        && (matches!(raw, 5 | 17 | 50 | 87) || (include_emlink && raw == 1142));
    unix_match || win_match
}

/// Same-FS copy cascade: `reflink` (when `build-cache` enabled) -> `hard_link` -> verified `copy`.
pub fn clone_or_copy(src: &Path, dst: &Path) -> Result<CloneOutcome> {
    clone_or_copy_cascade(src, dst, try_reflink_native, |s, d| fs::hard_link(s, d))
}

fn clone_or_copy_cascade<R, H>(src: &Path, dst: &Path, reflink_fn: R, hardlink_fn: H) -> Result<CloneOutcome>
where R: FnOnce(&Path, &Path) -> io::Result<()>, H: FnOnce(&Path, &Path) -> io::Result<()> {
    match reflink_fn(src, dst) {
        Ok(()) => return Ok(CloneOutcome::Reflinked),
        Err(err) if is_copy_fallback_err(&err, false) => {}
        Err(err) => anyhow::bail!("clone_or_copy reflink failed for {} -> {}: {err}", src.display(), dst.display()),
    }
    match hardlink_fn(src, dst) {
        Ok(()) => return Ok(CloneOutcome::HardLinked),
        Err(err) if is_copy_fallback_err(&err, true) => {}
        Err(err) => anyhow::bail!("clone_or_copy hard_link failed for {} -> {}: {err}", src.display(), dst.display()),
    }
    verified_copy(src, dst).with_context(|| format!("clone_or_copy copy failed for {} -> {}", src.display(), dst.display()))?;
    Ok(CloneOutcome::Copied)
}

fn try_reflink_native(src: &Path, dst: &Path) -> io::Result<()> {
    #[cfg(feature = "build-cache")]
    { reflink_copy::reflink(src, dst) }
    #[cfg(not(feature = "build-cache"))]
    { let _ = (src, dst); Err(io::Error::new(io::ErrorKind::Unsupported, "build-cache feature disabled")) }
}

fn verified_copy(src: &Path, dst: &Path) -> io::Result<u64> {
    let (expected, copied) = (fs::metadata(src)?.len(), fs::copy(src, dst)?);
    if copied != expected {
        let _ = fs::remove_file(dst);
        return Err(io::Error::other(format!("short copy {} -> {} ({copied} != {expected})", src.display(), dst.display())));
    }
    Ok(copied)
}

/// Ensure exact 43-byte `CACHEDIR.TAG` exists in `.refit/cache`, `.refit-stages`, and `.refit-tx`.
pub fn ensure_cache_dirs_tagged(repo: &Path) -> [PathBuf; 3] {
    let dirs = [
        repo.join(".refit").join("cache"),
        work_dir(repo, ".refit-stages", "refit-stages"),
        work_dir(repo, ".refit-tx", "refit-tx"),
    ];
    for dir in &dirs { ensure_cachedir_tag(dir); }
    dirs
}

/// Evict oldest-mtime entries in `dir` until remaining payload `<= max_bytes`.
pub fn prune_cache_lru(dir: &Path, max_bytes: u64) -> Result<CacheGcStats> {
    prune_cache_dirs_lru(&[dir.to_path_buf()], max_bytes)
}

/// Tag and LRU-prune `.refit/cache`, `.refit-stages`, and `.refit-tx` under one cap.
pub fn prune_repo_caches(repo: &Path, max_bytes: u64) -> Result<CacheGcStats> {
    prune_cache_dirs_lru(&ensure_cache_dirs_tagged(repo), max_bytes)
}

fn prune_cache_dirs_lru(dirs: &[PathBuf], max_bytes: u64) -> Result<CacheGcStats> {
    let mut entries: Vec<(u64, PathBuf, bool, u64)> = Vec::new();
    for dir in dirs { collect_lru_entries(dir, &mut entries)?; }
    let mut total: u64 = entries.iter().map(|e| e.3).sum();
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let (mut evicted_entries, mut freed_bytes) = (0usize, 0u64);
    for (_, path, is_dir, size) in entries {
        if total <= max_bytes { break; }
        let ok = if is_dir { fs::remove_dir_all(&path).is_ok() } else { fs::remove_file(&path).is_ok() };
        if ok {
            evicted_entries += 1;
            freed_bytes = freed_bytes.saturating_add(size);
            total = total.saturating_sub(size);
        }
    }
    Ok(CacheGcStats { evicted_entries, freed_bytes, remaining_bytes: total })
}

fn is_fresh_stage_tx(meta: &fs::Metadata) -> bool {
    let Ok(mtime) = meta.modified() else {
        return false;
    };
    let Ok(age) = mtime.elapsed() else {
        return false;
    };
    age.as_secs() < 600
}

fn is_protected_stage_entry(path: &Path, name: &str, meta: &fs::Metadata) -> bool {
    match name {
        "CACHEDIR.TAG" => true,
        "cargo-slots" => cargo_slots_has_active_lock(path),
        _ => name.starts_with("tx-") && meta.is_dir() && is_fresh_stage_tx(meta),
    }
}

fn cargo_slots_has_active_lock(slots_dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(slots_dir) else { return false };
    rd.flatten().any(|item| {
        let lossy = item.file_name().to_string_lossy().into_owned();
        lossy.starts_with("slot-") && lossy.ends_with(".lock")
            && !matches!(SidecarLock::try_acquire(&item.path()), Ok(Some(_)))
    })
}

fn collect_lru_entries(dir: &Path, out: &mut Vec<(u64, PathBuf, bool, u64)>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let rd = fs::read_dir(dir).with_context(|| format!("read cache dir {}", dir.display()))?;
    for item in rd.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        let path = item.path();
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        if is_protected_stage_entry(&path, &name, &meta) {
            continue;
        }
        let is_dir = meta.is_dir();
        let (fb, fsize) = file_mtime_size(&path).unwrap_or((0, meta.len()));
        let (size, mtime_ns) = if is_dir { measure_dir_tree(&path, fb) } else { (fsize, fb) };
        out.push((mtime_ns, path, is_dir, size));
    }
    Ok(())
}

fn measure_dir_tree(root: &Path, fallback_mtime: u64) -> (u64, u64) {
    let mut stack = vec![root.to_path_buf()];
    let (mut total, mut newest, mut saw) = (0u64, 0u64, false);
    while let Some(cur) = stack.pop() {
        let Ok(rd) = fs::read_dir(&cur) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            let Ok(meta) = fs::symlink_metadata(&path) else { continue };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                stack.push(path);
            } else if let Some((mtime_ns, len)) = file_mtime_size(&path) {
                saw = true;
                total = total.saturating_add(len);
                newest = newest.max(mtime_ns);
            }
        }
    }
    (total, if saw { newest } else { fallback_mtime })
}

/// Validate `UV_LINK_MODE`, explicitly rejecting `symlink` (`Err`).
pub fn validate_uv_link_mode(mode: &str) -> Result<&'static str> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "hardlink" => Ok("hardlink"),
        "clone" => Ok("clone"),
        "copy" => Ok("copy"),
        "symlink" => anyhow::bail!("UV_LINK_MODE=symlink is forbidden because `uv cache clean` breaks virtualenvs"),
        other => anyhow::bail!("unsupported UV_LINK_MODE `{other}`"),
    }
}

fn resolve_uv_link_mode(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|raw| validate_uv_link_mode(raw).ok())
        .unwrap_or(if cfg!(target_os = "macos") { "clone" } else { "hardlink" })
}

/// Python worktree cache environment (`UV_CACHE_DIR` + safe `UV_LINK_MODE`).
pub fn py_worktree_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let inherited = std::env::var("UV_LINK_MODE").ok();
    vec![
        ("UV_CACHE_DIR".into(), repo_root.join(".refit/cache/uv-cache").display().to_string()),
        ("UV_LINK_MODE".into(), resolve_uv_link_mode(inherited.as_deref()).into()),
        ("MALLOC_ARENA_MAX".into(), "1".into()),
        ("MIMALLOC_PURGE_DELAY".into(), "0".into()),
    ]
}

/// Combined JS/TS + Python worktree cache environment using only real tool-consumed env vars.
pub fn polyglot_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let cache_dir = repo_root.join(".refit").join("cache");
    let mut env = vec![
        ("BUN_INSTALL_GLOBAL_STORE".into(), "1".into()),
        ("BUN_INSTALL_CACHE_DIR".into(), cache_dir.join("bun-cache").display().to_string()),
        ("BUN_INSTALL_LINKER".into(), "isolated".into()),
        ("PNPM_STORE_DIR".into(), cache_dir.join("pnpm-store").display().to_string()),
    ];
    env.extend(py_worktree_cache_env(repo_root));
    env
}

fn build_rust_cache_env_core(
    repo_root: &Path, slot_dir: &Path, ephemeral_stage: bool,
    wrapper: Option<(RustCacheBackend, &Path)>, effective_mem_bytes: u64,
    constrained: bool, blaze_opt_in: bool,
) -> Vec<(String, String)> {
    let build_dir = slot_dir.join("build-dir");
    let _ = fs::create_dir_all(&build_dir);
    let mut env = vec![
        ("CARGO_TARGET_DIR".into(), slot_dir.display().to_string()),
        ("CARGO_BUILD_BUILD_DIR".into(), build_dir.display().to_string()),
        ("MALLOC_ARENA_MAX".into(), "1".into()),
        ("MIMALLOC_PURGE_DELAY".into(), "0".into()),
    ];
    if ephemeral_stage || constrained { env.push(("CARGO_INCREMENTAL".into(), "0".into())); }
    if let Some((backend, bin)) = wrapper {
        let cache_root = repo_root.join(".refit/cache/rust");
        env.push(("RUSTC_WRAPPER".into(), bin.display().to_string()));
        env.push(("MBX_EXCLUDE_OUT_DIR".into(), "1".into()));
        match backend {
            RustCacheBackend::Mbx => {
                env.push(("MBX_CACHE_DIR".into(), cache_root.display().to_string()));
                if cfg!(any(target_os = "linux", target_os = "macos"))
                    && effective_mem_bytes >= BLAZE_MIN_MEM_BYTES && blaze_opt_in
                {
                    env.push(("MBX_BLAZE".into(), "1".into()));
                }
            }
            RustCacheBackend::Kache => env.push(("KACHE_DIR".into(), cache_root.display().to_string())),
        }
    }
    env
}

/// Build Rust gate/audit cache env: activates `mbx`/`kache` ONLY when installed on `PATH`.
pub fn build_rust_cache_env(repo_root: &Path, slot_dir: &Path, ephemeral_stage: bool) -> Vec<(String, String)> {
    ensure_cache_dirs_tagged(repo_root);
    let b = budget();
    build_rust_cache_env_core(
        repo_root, slot_dir, ephemeral_stage, detect_rust_wrapper(),
        b.effective_mem_bytes, b.is_constrained(), env_truthy("MBX_BLAZE"),
    )
}

pub(super) struct CargoSlot {
    #[cfg(test)] pub(super) slot: usize,
    #[cfg(not(test))] slot: usize,
    pooled: bool, pub(super) dir: PathBuf, _lock: Option<SidecarLock>,
}

fn max_cargo_slots() -> usize {
    env_parse::<usize>("REFIT_CARGO_SLOTS").map_or(2, |n| n.clamp(1, 2))
}

impl CargoSlot {
    pub(super) fn acquire(repo_root: &Path) -> Self {
        let cap = max_cargo_slots();
        let pool = CARGO_SLOT_POOL.get_or_init(|| Mutex::new((0..cap).rev().collect()));
        let slots_root = stages_dir(repo_root).join("cargo-slots");
        ensure_cachedir_tag(&stages_dir(repo_root));
        ensure_cachedir_tag(&slots_root);
        let (mut skipped, mut chosen) = (Vec::new(), None::<(usize, SidecarLock)>);
        while let Some(s) = pool.lock().pop() {
            if let Ok(Some(lk)) = SidecarLock::try_acquire(&slots_root.join(format!("slot-{s}.lock"))) {
                chosen = Some((s, lk));
                break;
            }
            skipped.push(s);
        }
        if !skipped.is_empty() { pool.lock().extend(skipped); }
        let (slot, pooled, lock) = match chosen {
            Some((s, lk)) => (s, true, Some(lk)),
            None => {
                let s = CARGO_EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed) as usize;
                let lk = SidecarLock::try_acquire(&slots_root.join(format!("slot-{s}.lock")))
                    .ok()
                    .flatten();
                (s, false, lk)
            }
        };
        let dir = slots_root.join(format!("slot-{slot}"));
        let _ = fs::create_dir_all(&dir);
        Self { slot, pooled, dir, _lock: lock }
    }

    pub(super) fn rust_cache_env(&self, repo_root: &Path, ephemeral_stage: bool) -> Vec<(String, String)> {
        build_rust_cache_env(repo_root, &self.dir, ephemeral_stage)
    }
}

impl Drop for CargoSlot {
    fn drop(&mut self) {
        self._lock = None;
        if self.pooled {
            if let Some(pool) = CARGO_SLOT_POOL.get() { pool.lock().push(self.slot); }
        } else {
            let _ = fs::remove_dir_all(&self.dir);
            let _ = fs::remove_file(self.dir.with_extension("lock"));
        }
    }
}

pub(super) fn stages_dir(repo_root: &Path) -> PathBuf { work_dir(repo_root, ".refit-stages", "refit-stages") }

pub fn sweep_stages(repo_root: &Path, max_age_secs: u64) -> usize {
    let Ok(rd) = fs::read_dir(stages_dir(repo_root)) else { return 0 };
    let cutoff = SystemTime::now().checked_sub(Duration::from_secs(max_age_secs)).unwrap_or(UNIX_EPOCH);
    rd.flatten().filter(|entry| {
        let p = entry.path();
        p.is_dir() && entry.file_name().to_string_lossy().starts_with("tx-")
            && fs::metadata(&p).and_then(|m| m.modified()).is_ok_and(|m| m <= cutoff)
            && fs::remove_dir_all(&p).is_ok()
    }).count()
}

#[cfg(test)]
pub fn encode_lsp(payload: &serde_json::Value) -> Vec<u8> {
    let body = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

#[cfg(test)]
pub fn decode_lsp(raw: &[u8]) -> Option<(serde_json::Value, usize)> {
    let hdr_end = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let hdr_text = std::str::from_utf8(&raw[..hdr_end]).ok()?;
    let len: usize = hdr_text.lines().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        if k.trim().eq_ignore_ascii_case("content-length") {
            v.trim().parse().ok()
        } else {
            None
        }
    })?;
    let end = hdr_end + 4 + len;
    if raw.len() < end {
        return None;
    }
    let val = serde_json::from_slice(&raw[hdr_end + 4..end]).ok()?;
    Some((val, end))
}

fn link_or_copy_file(src: &Path, dst: &Path) -> Result<(u64, u64)> {
    match clone_or_copy(src, dst) {
        Ok(CloneOutcome::Reflinked | CloneOutcome::HardLinked) => Ok((1, 0)),
        Ok(CloneOutcome::Copied) => Ok((0, 1)),
        Err(err) => anyhow::bail!("stage materialize failed for {} -> {}: {err}", src.display(), dst.display()),
    }
}

fn link_or_copy_pairs(pairs: &[(PathBuf, PathBuf)]) -> Result<(u64, u64)> {
    par_map(pairs, |(s, d)| link_or_copy_file(s, d))
        .into_iter()
        .try_fold((0u64, 0u64), |(hl, cp), item| {
            let (dh, dc) = item?;
            Ok((hl + dh, cp + dc))
        })
}

pub(super) fn materialize_stage(
    repo_root: &Path, candidates: &BTreeMap<String, String>, deleted: &[String],
) -> Result<DropGuard<PathBuf, impl FnOnce(PathBuf) + use<>>> {
    let seq = STAGE_SEQ.fetch_add(1, Ordering::Relaxed);
    ensure_cache_dirs_tagged(repo_root);
    if seq == 1 || seq.is_multiple_of(64) {
        let _ = sweep_stages(repo_root, 1800);
        let _ = prune_cache_lru(&repo_root.join(".refit/cache"), DEFAULT_CACHE_CAP_BYTES);
    }
    let stage_dir = stages_dir(repo_root).join(format!("tx-{}-{seq}", now_ms()));
    fs::create_dir_all(&stage_dir)?;
    let guard = DropGuard::new(stage_dir, |dir| { let _ = fs::remove_dir_all(dir); });
    let del_set: BTreeSet<String> = deleted.iter().map(|s| s.replace('\\', "/")).collect();
    let to_link: Vec<(PathBuf, PathBuf)> = walk_files(repo_root, None).into_iter().filter_map(|path| {
        let rel = rel_norm(repo_root, &path)?;
        (!candidates.contains_key(&rel) && !del_set.contains(&rel)).then(|| (path, guard.join(&rel)))
    }).collect();
    for p in to_link.iter().filter_map(|(_, d)| d.parent()).collect::<BTreeSet<_>>() {
        fs::create_dir_all(p).with_context(|| format!("create staged parent dir {}", p.display()))?;
    }
    let (hl, cp) = link_or_copy_pairs(&to_link)?;
    if hl > 0 { STAGE_HARDLINKS_TOTAL.fetch_add(hl, Ordering::Relaxed); }
    if cp > 0 { STAGE_COPIED_TOTAL.fetch_add(cp, Ordering::Relaxed); }
    for (rel, content) in candidates {
        let dst = guard.join(rel);
        if let Some(p) = dst.parent() { fs::create_dir_all(p)?; }
        fs::write(&dst, content.as_bytes()).with_context(|| format!("write staged candidate {}", dst.display()))?;
    }
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::has_cachedir_tag;
    use crate::testutil::TempRepo;
    use std::io::Read;

    fn env_val<'a>(env: &'a [(String, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    fn stage_copy_failure_propagates_and_sentinel_guard() {
        let tmp = TempRepo::with_files("stage_err", &[("src/a.rs", "pub fn ok() -> u8 { 42 }\n")]);
        let dst = tmp.join("out_stage");
        let occupied = dst.join("occupied_dir");
        fs::create_dir_all(&occupied).unwrap();

        for (s, d) in [(tmp.join("src/missing.rs"), dst.join("target.rs")), (tmp.join("src/a.rs"), occupied)] {
            let err = link_or_copy_pairs(&[(s, d)]).unwrap_err();
            assert!(err.to_string().contains("stage materialize failed"));
        }

        let stage = materialize_stage(tmp.path(), &BTreeMap::new(), &[]).unwrap();
        let mut buf = [0xFFu8; 64];
        let mut file = fs::File::open(stage.join("src/a.rs")).unwrap();
        let n = file.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"pub fn ok() -> u8 { 42 }\n");
        assert!(buf[n..].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn stage_cache_orch_clone_lru_uv_and_conditional_wrapper() {
        let tmp = TempRepo::with_files("stage_cache_orch", &[("src/a.rs", "pub fn x() -> u8 { 7 }\n")]);
        let src = tmp.join("src/a.rs");

        for (idx, code) in [1i32, 18, 38].into_iter().enumerate() {
            let raw_err = |_: &Path, _: &Path| Err(io::Error::from_raw_os_error(code));
            let perm = |_: &Path, _: &Path| Err(io::Error::new(io::ErrorKind::PermissionDenied, "EPERM"));
            let hl = clone_or_copy_cascade(&src, &tmp.join(&format!("hl_{idx}.rs")), raw_err, |s, d| fs::hard_link(s, d)).unwrap();
            assert_eq!(hl, CloneOutcome::HardLinked);
            let dst_cp = tmp.join(&format!("cp_{idx}.rs"));
            let cp = clone_or_copy_cascade(&src, &dst_cp, perm, raw_err).unwrap();
            assert_eq!((cp, fs::read(&dst_cp).unwrap()), (CloneOutcome::Copied, fs::read(&src).unwrap()));
        }
        assert!(clone_or_copy(&tmp.join("missing.rs"), &tmp.join("never.rs")).is_err());

        assert!(validate_uv_link_mode("symlink").is_err());
        assert_eq!(validate_uv_link_mode("hardlink").unwrap(), "hardlink");
        assert!(matches!(resolve_uv_link_mode(Some("symlink")), "hardlink" | "clone"));
        let poly = polyglot_cache_env(tmp.path());
        assert!(env_val(&poly, "UV_CACHE_DIR").is_some());
        assert!(matches!(env_val(&poly, "UV_LINK_MODE"), Some("hardlink" | "clone" | "copy")));
        for dead in ["REFIT_JS_RELINKER", "REFIT_BUN_LINKER_ARGS"] {
            assert!(env_val(&poly, dead).is_none());
        }

        for d in &ensure_cache_dirs_tagged(tmp.path()) {
            assert!(has_cachedir_tag(d));
        }
        let slot = CargoSlot::acquire(tmp.path());
        fs::write(slot.dir.join("active.bin"), vec![1u8; 4096]).unwrap();
        let cache_dir = tmp.join(".refit/cache");
        fs::write(cache_dir.join("old.bin"), vec![2u8; 4096]).unwrap();
        std::thread::sleep(Duration::from_millis(25));
        fs::write(cache_dir.join("new.bin"), vec![3u8; 1024]).unwrap();

        let gc = prune_repo_caches(tmp.path(), 2048).unwrap();
        assert!(gc.evicted_entries >= 1 && gc.remaining_bytes <= 2048);
        assert!(has_cachedir_tag(&cache_dir) && slot.dir.join("active.bin").exists());
        assert!(!cache_dir.join("old.bin").exists());

        let (gb32, gb8) = (32u64 << 30, 8u64 << 30);
        let absent = build_rust_cache_env_core(tmp.path(), &slot.dir, true, None, gb32, false, true);
        for (k, exp) in [
            ("RUSTC_WRAPPER", None),
            ("MBX_CACHE_DIR", None),
            ("MBX_BLAZE", None),
            ("REFIT_RUST_CACHE_KEY", None),
            ("CARGO_INCREMENTAL", Some("0")),
        ] {
            assert_eq!(env_val(&absent, k), exp);
        }

        let bin_dir = tmp.join("fake_bin");
        fs::create_dir_all(&bin_dir).unwrap();
        fs::write(bin_dir.join(if cfg!(windows) { "mbx.exe" } else { "mbx" }), b"stub").unwrap();
        let mbx_path = find_tool_in_dirs(&[bin_dir], "mbx").unwrap();
        let wrap = Some((RustCacheBackend::Mbx, mbx_path.as_path()));
        let with_mbx = build_rust_cache_env_core(tmp.path(), &slot.dir, false, wrap, gb32, false, true);
        let expect_blaze = cfg!(any(target_os = "linux", target_os = "macos")).then_some("1");
        assert_eq!(env_val(&with_mbx, "RUSTC_WRAPPER"), Some(mbx_path.to_str().unwrap()));
        assert_eq!(env_val(&with_mbx, "MBX_EXCLUDE_OUT_DIR"), Some("1"));
        assert_eq!(env_val(&with_mbx, "MBX_BLAZE"), expect_blaze);
        assert!(env_val(&with_mbx, "MBX_CACHE_DIR").is_some());

        let low_mem = build_rust_cache_env_core(tmp.path(), &slot.dir, false, wrap, gb8, false, true);
        assert!(env_val(&low_mem, "MBX_BLAZE").is_none());
        drop(slot);
    }
}
