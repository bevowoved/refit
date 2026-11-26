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
const ALLOC_ENV: [(&str, &str); 2] = [("MALLOC_ARENA_MAX", "1"), ("MIMALLOC_PURGE_DELAY", "0")];

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

pub(super) fn find_tool_in_dirs(dirs: &[PathBuf], bin: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) { &[".exe", ".cmd", ".bat", ""] } else { &[""] };
    dirs.iter().find_map(|dir| {
        exts.iter().find_map(|ext| {
            let p = if ext.is_empty() { dir.join(bin) } else { dir.join(format!("{bin}{ext}")) };
            p.is_file().then_some(p)
        })
    })
}

fn find_tool_on_path(bin: &str) -> Option<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(&std::env::var_os("PATH")?).collect();
    find_tool_in_dirs(&dirs, bin)
}

/// Cached PATH probe for `mbx` or `kache`.
pub fn detect_rust_wrapper() -> Option<(RustCacheBackend, &'static Path)> {
    RUST_WRAPPER_CACHE.get_or_init(|| {
        find_tool_on_path("mbx").map(|path| (RustCacheBackend::Mbx, path))
            .or_else(|| find_tool_on_path("kache").map(|path| (RustCacheBackend::Kache, path)))
    }).as_ref().map(|(backend, path)| (*backend, path.as_path()))
}

fn is_copy_fallback_err(err: &io::Error, include_emlink: bool) -> bool {
    if matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::AlreadyExists) { return false; }
    if matches!(
        err.kind(),
        io::ErrorKind::PermissionDenied | io::ErrorKind::Unsupported | io::ErrorKind::CrossesDevices
    ) { return true; }
    let Some(raw) = err.raw_os_error() else { return false };
    let unix_match = matches!(raw, 1 | 18 | 22 | 38 | 45 | 78 | 95) || (include_emlink && raw == 31);
    let win_match = cfg!(windows) && (matches!(raw, 5 | 17 | 50 | 87) || (include_emlink && raw == 1142));
    unix_match || win_match
}

/// Same-FS copy cascade: `reflink` (when `build-cache` enabled) -> `hard_link` -> verified `copy`.
pub fn clone_or_copy(src: &Path, dst: &Path) -> Result<CloneOutcome> {
    clone_or_copy_cascade(src, dst, try_reflink_native, |from, to| fs::hard_link(from, to))
}

fn try_link_step(res: io::Result<()>, emlink: bool, label: &str, src: &Path, dst: &Path) -> Result<bool> {
    match res {
        Ok(()) => Ok(true),
        Err(err) if is_copy_fallback_err(&err, emlink) => Ok(false),
        Err(err) => {
            anyhow::bail!("clone_or_copy {label} failed for {} -> {}: {err}", src.display(), dst.display())
        }
    }
}

fn clone_or_copy_cascade<R, H>(src: &Path, dst: &Path, reflink_fn: R, hardlink_fn: H) -> Result<CloneOutcome>
where R: FnOnce(&Path, &Path) -> io::Result<()>, H: FnOnce(&Path, &Path) -> io::Result<()> {
    if try_link_step(reflink_fn(src, dst), false, "reflink", src, dst)? {
        return Ok(CloneOutcome::Reflinked);
    }
    if try_link_step(hardlink_fn(src, dst), true, "hard_link", src, dst)? {
        return Ok(CloneOutcome::HardLinked);
    }
    verified_copy(src, dst)
        .with_context(|| format!("clone_or_copy copy failed for {} -> {}", src.display(), dst.display()))?;
    Ok(CloneOutcome::Copied)
}

fn try_reflink_native(src: &Path, dst: &Path) -> io::Result<()> {
    cfg_select! {
        feature = "build-cache" => reflink_copy::reflink(src, dst),
        _ => {
            let _ = (src, dst);
            Err(io::Error::new(io::ErrorKind::Unsupported, "build-cache feature disabled"))
        }
    }
}

fn verified_copy(src: &Path, dst: &Path) -> io::Result<u64> {
    let (expected, copied) = (fs::metadata(src)?.len(), fs::copy(src, dst)?);
    if copied != expected {
        let _ = fs::remove_file(dst);
        let msg = format!("short copy {} -> {} ({copied} != {expected})", src.display(), dst.display());
        return Err(io::Error::other(msg));
    }
    Ok(copied)
}

/// Ensure exact 43-byte `CACHEDIR.TAG` exists in `.refit/cache`, `.refit-stages`, and `.refit-tx`.
pub fn ensure_cache_dirs_tagged(repo: &Path) -> [PathBuf; 3] {
    let dirs = [
        repo.join(".refit/cache"),
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
    let mut entries = Vec::new();
    for dir in dirs { collect_lru_entries(dir, &mut entries)?; }
    let mut total: u64 = entries.iter().map(|item| item.3).sum();
    entries.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
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
    let Ok(mtime) = meta.modified() else { return false };
    mtime.elapsed().is_ok_and(|age| age.as_secs() < 600)
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
    if !dir.exists() { return Ok(()); }
    let rd = fs::read_dir(dir).with_context(|| format!("read cache dir {}", dir.display()))?;
    for item in rd.flatten() {
        let (name, path) = (item.file_name().to_string_lossy().into_owned(), item.path());
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        if is_protected_stage_entry(&path, &name, &meta) { continue; }
        let is_dir = meta.is_dir();
        let (fb, fsize) = file_mtime_size(&path).unwrap_or((0, meta.len()));
        let (size, mtime_ns) = if is_dir { measure_dir_tree(&path, fb) } else { (fsize, fb) };
        out.push((mtime_ns, path, is_dir, size));
    }
    Ok(())
}

fn measure_dir_tree(root: &Path, fallback_mtime: u64) -> (u64, u64) {
    let (mut total, mut newest, mut saw) = (0u64, 0u64, false);
    let walker = walkdir::WalkDir::new(root).min_depth(1);
    for entry in walker.into_iter().flatten() {
        let ftype = entry.file_type();
        if !ftype.is_symlink() && !ftype.is_dir()
            && let Some((mtime_ns, len)) = file_mtime_size(entry.path())
        {
            saw = true;
            total = total.saturating_add(len);
            newest = newest.max(mtime_ns);
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
    requested.and_then(|raw| validate_uv_link_mode(raw).ok())
        .unwrap_or(if cfg!(target_os = "macos") { "clone" } else { "hardlink" })
}

fn push_kv(env: &mut Vec<(String, String)>, pairs: &[(&str, &str)]) {
    env.extend(pairs.iter().map(|&(key, val)| (key.into(), val.into())));
}

/// Python worktree cache environment (`UV_CACHE_DIR` + safe `UV_LINK_MODE`).
pub fn py_worktree_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let inherited = std::env::var("UV_LINK_MODE").ok();
    let mut env = vec![
        ("UV_CACHE_DIR".into(), repo_root.join(".refit/cache/uv-cache").display().to_string()),
        ("UV_LINK_MODE".into(), resolve_uv_link_mode(inherited.as_deref()).into()),
    ];
    push_kv(&mut env, &ALLOC_ENV);
    env
}

/// Combined JS/TS + Python worktree cache environment using only real tool-consumed env vars.
pub fn polyglot_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let cache_dir = repo_root.join(".refit/cache");
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
    ];
    push_kv(&mut env, &ALLOC_ENV);
    if ephemeral_stage || constrained { env.push(("CARGO_INCREMENTAL".into(), "0".into())); }
    if let Some((backend, bin)) = wrapper {
        let cache_root = repo_root.join(".refit/cache/rust").display().to_string();
        env.push(("RUSTC_WRAPPER".into(), bin.display().to_string()));
        env.push(("MBX_EXCLUDE_OUT_DIR".into(), "1".into()));
        match backend {
            RustCacheBackend::Mbx => {
                env.push(("MBX_CACHE_DIR".into(), cache_root));
                if cfg!(any(target_os = "linux", target_os = "macos"))
                    && effective_mem_bytes >= BLAZE_MIN_MEM_BYTES && blaze_opt_in
                {
                    env.push(("MBX_BLAZE".into(), "1".into()));
                }
            }
            RustCacheBackend::Kache => env.push(("KACHE_DIR".into(), cache_root)),
        }
    }
    env
}

/// Build Rust gate/audit cache env: activates `mbx`/`kache` ONLY when installed on `PATH`.
pub fn build_rust_cache_env(
    repo_root: &Path, slot_dir: &Path, ephemeral_stage: bool,
) -> Vec<(String, String)> {
    ensure_cache_dirs_tagged(repo_root);
    let bdg = budget();
    build_rust_cache_env_core(
        repo_root, slot_dir, ephemeral_stage, detect_rust_wrapper(),
        bdg.effective_mem_bytes, bdg.is_constrained(), env_truthy("MBX_BLAZE"),
    )
}

pub(super) struct CargoSlot {
    #[cfg(test)] pub(super) slot: usize,
    #[cfg(not(test))] slot: usize,
    pooled: bool, pub(super) dir: PathBuf, _lock: Option<SidecarLock>,
}

fn max_cargo_slots() -> usize { env_parse::<usize>("REFIT_CARGO_SLOTS").map_or(2, |cap| cap.clamp(1, 2)) }

impl CargoSlot {
    pub(super) fn acquire(repo_root: &Path) -> Self {
        let cap = max_cargo_slots();
        let pool = CARGO_SLOT_POOL.get_or_init(|| Mutex::new((0..cap).rev().collect()));
        let slots_root = stages_dir(repo_root).join("cargo-slots");
        ensure_cachedir_tag(&stages_dir(repo_root));
        ensure_cachedir_tag(&slots_root);

        let (mut skipped, mut chosen) = (Vec::new(), None::<(usize, SidecarLock)>);
        while let Some(slot_id) = pool.lock().pop() {
            let lock_path = slots_root.join(format!("slot-{slot_id}.lock"));
            if let Ok(Some(lk)) = SidecarLock::try_acquire(&lock_path) {
                chosen = Some((slot_id, lk));
                break;
            }
            skipped.push(slot_id);
        }
        if !skipped.is_empty() {
            pool.lock().extend(skipped);
        }

        let (slot, pooled, lock) = match chosen {
            Some((slot_id, lk)) => (slot_id, true, Some(lk)),
            None => {
                let slot_id = CARGO_EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed) as usize;
                let lock_path = slots_root.join(format!("slot-{slot_id}.lock"));
                let lk = SidecarLock::try_acquire(&lock_path).ok().flatten();
                (slot_id, false, lk)
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
        let path = entry.path();
        path.is_dir() && entry.file_name().to_string_lossy().starts_with("tx-")
            && fs::metadata(&path).and_then(|meta| meta.modified()).is_ok_and(|mt| mt <= cutoff)
            && fs::remove_dir_all(&path).is_ok()
    }).count()
}

#[cfg(test)]
pub fn encode_lsp(payload: &serde_json::Value) -> Vec<u8> {
    let body = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

#[cfg(test)]
pub fn decode_lsp(raw: &[u8]) -> Option<(serde_json::Value, usize)> {
    let hdr_end = raw.array_windows::<4>().position(|win| win == b"\r\n\r\n")?;
    let header = std::str::from_utf8(&raw[..hdr_end]).ok()?;
    let len: usize = header.lines().find_map(|line| {
        let (key, val) = line.split_once(':')?;
        if !key.trim().eq_ignore_ascii_case("content-length") {
            return None;
        }
        val.trim().parse().ok()
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
        Err(err) => {
            anyhow::bail!("stage materialize failed for {} -> {}: {err}", src.display(), dst.display())
        }
    }
}

fn link_or_copy_pairs(pairs: &[(PathBuf, PathBuf)]) -> Result<(u64, u64)> {
    par_map(pairs, |(src, dst)| link_or_copy_file(src, dst)).into_iter().try_fold((0u64, 0u64), |acc, item| {
        let (dh, dc) = item?;
        Ok((acc.0 + dh, acc.1 + dc))
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
    let del_set: BTreeSet<String> = deleted.iter().map(|rel| rel.replace('\\', "/")).collect();
    let to_link: Vec<(PathBuf, PathBuf)> = walk_files(repo_root, None).into_iter().filter_map(|path| {
        let rel = rel_norm(repo_root, &path)?;
        (!candidates.contains_key(&rel) && !del_set.contains(&rel)).then(|| (path, guard.join(&rel)))
    }).collect();
    for parent in to_link.iter().filter_map(|(_, dst)| dst.parent()).collect::<BTreeSet<_>>() {
        fs::create_dir_all(parent).with_context(|| format!("create staged parent dir {}", parent.display()))?;
    }
    let (hl, cp) = link_or_copy_pairs(&to_link)?;
    if hl > 0 { STAGE_HARDLINKS_TOTAL.fetch_add(hl, Ordering::Relaxed); }
    if cp > 0 { STAGE_COPIED_TOTAL.fetch_add(cp, Ordering::Relaxed); }
    for (rel, content) in candidates {
        let dst = guard.join(rel);
        if let Some(parent) = dst.parent() { fs::create_dir_all(parent)?; }
        fs::write(&dst, content.as_bytes())
            .with_context(|| format!("write staged candidate {}", dst.display()))?;
    }
    Ok(guard)
}

#[cfg(test)]
use crate::platform::has_cachedir_tag;
#[cfg(test)]
use crate::testutil::TempRepo;
#[cfg(test)]
use std::io::Read;

#[cfg(test)]
fn env_val<'a>(env: &'a [(String, String)], key: &str) -> Option<&'a str> {
    env.iter().find(|(ek, _)| ek == key).map(|(_, val)| val.as_str())
}

#[cfg(test)]
#[test]
fn stage_copy_err_and_read() {
    let tmp = TempRepo::with_files("stage_err", &[("src/a.rs", "pub fn ok() -> u8 { 42 }\n")]);
    let occupied = tmp.join("out_stage/occupied_dir");
    fs::create_dir_all(&occupied).unwrap();
    for (src, dst) in [
        (tmp.join("src/missing.rs"), tmp.join("out_stage/target.rs")),
        (tmp.join("src/a.rs"), occupied),
    ] {
        let err = link_or_copy_pairs(&[(src, dst)]).unwrap_err();
        assert!(err.to_string().contains("stage materialize failed"));
    }

    let stage = materialize_stage(tmp.path(), &BTreeMap::new(), &[]).unwrap();
    let mut buf = [0xFFu8; 64];
    let mut file = fs::File::open(stage.join("src/a.rs")).unwrap();
    let read_n = file.read(&mut buf).unwrap();
    assert_eq!(&buf[..read_n], b"pub fn ok() -> u8 { 42 }\n");
    assert!(buf[read_n..].iter().all(|&byte| byte == 0xFF));
}

#[cfg(test)]
#[test]
fn stage_cascade_and_uv_mode() {
    let tmp = TempRepo::with_files(
        "stage_cache_orch",
        &[("src/a.rs", "pub fn x() -> u8 { 7 }\n")],
    );
    let src = tmp.join("src/a.rs");

    for (idx, code) in [1i32, 18, 38].into_iter().enumerate() {
        let raw_err = |_: &Path, _: &Path| Err(io::Error::from_raw_os_error(code));
        let perm = |_: &Path, _: &Path| {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "EPERM"))
        };
        let hl_dst = tmp.join(&format!("hl_{idx}.rs"));
        let hl = clone_or_copy_cascade(&src, &hl_dst, raw_err, |from, to| fs::hard_link(from, to))
            .unwrap();
        assert_eq!(hl, CloneOutcome::HardLinked);

        let dst_cp = tmp.join(&format!("cp_{idx}.rs"));
        let cp = clone_or_copy_cascade(&src, &dst_cp, perm, raw_err).unwrap();
        assert_eq!(
            (cp, fs::read(&dst_cp).unwrap()),
            (CloneOutcome::Copied, fs::read(&src).unwrap())
        );
    }
    assert!(clone_or_copy(&tmp.join("missing.rs"), &tmp.join("never.rs")).is_err());

    assert!(validate_uv_link_mode("symlink").is_err());
    assert_eq!(validate_uv_link_mode("hardlink").unwrap(), "hardlink");
    std::assert_matches!(resolve_uv_link_mode(Some("symlink")), "hardlink" | "clone");

    let poly = polyglot_cache_env(tmp.path());
    assert!(env_val(&poly, "UV_CACHE_DIR").is_some());
    std::assert_matches!(env_val(&poly, "UV_LINK_MODE"), Some("hardlink" | "clone" | "copy"));
    for dead in ["REFIT_JS_RELINKER", "REFIT_BUN_LINKER_ARGS"] {
        assert!(env_val(&poly, dead).is_none());
    }
}

#[cfg(test)]
#[test]
fn stage_cache_gc_and_wrapper() {
    let tmp = TempRepo::with_files(
        "stage_cache_gc",
        &[("src/a.rs", "pub fn x() -> u8 { 7 }\n")],
    );
    for dir in &ensure_cache_dirs_tagged(tmp.path()) {
        assert!(has_cachedir_tag(dir));
    }

    let slot = CargoSlot::acquire(tmp.path());
    let cache_dir = tmp.join(".refit/cache");
    fs::write(slot.dir.join("active.bin"), vec![1u8; 4096]).unwrap();
    fs::write(cache_dir.join("old.bin"), vec![2u8; 4096]).unwrap();
    std::thread::sleep(Duration::from_millis(25));
    fs::write(cache_dir.join("new.bin"), vec![3u8; 1024]).unwrap();

    let gc = prune_repo_caches(tmp.path(), 2048).unwrap();
    assert!(gc.evicted_entries >= 1 && gc.remaining_bytes <= 2048);
    assert!(has_cachedir_tag(&cache_dir) && slot.dir.join("active.bin").exists());
    assert!(!cache_dir.join("old.bin").exists());

    let (gb32, gb8) = (32u64 << 30, 8u64 << 30);
    let absent = build_rust_cache_env_core(tmp.path(), &slot.dir, true, None, gb32, false, true);
    let absent_checks = [
        ("RUSTC_WRAPPER", None),
        ("MBX_CACHE_DIR", None),
        ("MBX_BLAZE", None),
        ("REFIT_RUST_CACHE_KEY", None),
        ("CARGO_INCREMENTAL", Some("0")),
    ];
    for (key, exp) in absent_checks {
        assert_eq!(env_val(&absent, key), exp);
    }

    let bin_dir = tmp.join("fake_bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let bin_name = if cfg!(windows) { "mbx.exe" } else { "mbx" };
    fs::write(bin_dir.join(bin_name), b"stub").unwrap();
    let mbx_path = find_tool_in_dirs(&[bin_dir], "mbx").unwrap();
    let wrap = Some((RustCacheBackend::Mbx, mbx_path.as_path()));
    let with_mbx =
        build_rust_cache_env_core(tmp.path(), &slot.dir, false, wrap, gb32, false, true);
    let expect_blaze = cfg!(any(target_os = "linux", target_os = "macos")).then_some("1");
    assert_eq!(env_val(&with_mbx, "RUSTC_WRAPPER"), Some(mbx_path.to_str().unwrap()));
    assert_eq!(env_val(&with_mbx, "MBX_EXCLUDE_OUT_DIR"), Some("1"));
    assert_eq!(env_val(&with_mbx, "MBX_BLAZE"), expect_blaze);
    assert!(env_val(&with_mbx, "MBX_CACHE_DIR").is_some());

    let low_mem =
        build_rust_cache_env_core(tmp.path(), &slot.dir, false, wrap, gb8, false, true);
    assert!(env_val(&low_mem, "MBX_BLAZE").is_none());
    drop(slot);
}

#[cfg(test)]
#[test]
fn stage_slots_survive_gc_race() {
    use std::sync::{Arc, Barrier};
    let tmp = TempRepo::with_files("slot_gc_race", &[("src/a.rs", "pub fn ok() {}\n")]);
    let (root, acquired, release) =
        (Arc::new(tmp.path().to_path_buf()), Arc::new(Barrier::new(4)), Arc::new(Barrier::new(4)));

    // Three slots guarantee at least one ephemeral slot (> max_cargo_slots = 2).
    let workers: Vec<_> = (0..3).map(|idx| {
        let (rp, b1, b2) = (Arc::clone(&root), Arc::clone(&acquired), Arc::clone(&release));
        std::thread::spawn(move || {
            let slot = CargoSlot::acquire(&rp);
            let marker = slot.dir.join(format!("active_{idx}.bin"));
            fs::write(&marker, vec![idx as u8 + 1; 2048]).unwrap();
            b1.wait();
            b2.wait();
            assert!(marker.exists(), "active slot {idx} evicted during concurrent GC");
            drop(slot);
        })
    }).collect();

    acquired.wait();
    for _ in 0..5 { let _ = prune_repo_caches(&root, 0).unwrap(); }
    release.wait();
    for worker in workers { worker.join().unwrap(); }
    assert!(has_cachedir_tag(&root.join(".refit/cache")));
}
