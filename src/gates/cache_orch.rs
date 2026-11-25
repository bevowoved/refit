use crate::platform::{budget, ensure_cachedir_tag};
use crate::source::work_dir;
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Hard disk-budget cap (`0.6 GB` = `600 MiB`) for build, stage, and tx caches.
pub const DEFAULT_CACHE_CAP_BYTES: u64 = 600 * 1024 * 1024;
/// Minimum host/container RAM (`16 GiB`) required before opt-in `MBX_BLAZE=1` is allowed.
pub const BLAZE_MIN_MEM_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Canonical `Bun` isolated-linker install flags for cross-worktree store sharing.
pub const BUN_ISOLATED_LINKER_ARGS: &[&str] = &["--linker", "isolated"];

static RUSTC_VV_CACHE: OnceLock<(String, String)> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneOutcome {
    Reflinked,
    HardLinked,
    Copied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustCacheBackend {
    Mbx,
    Kache,
    CargoBuildDir,
}

impl RustCacheBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mbx => "mbx",
            Self::Kache => "kache",
            Self::CargoBuildDir => "cargo-build-dir",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKeyInputs {
    pub source_blake3: String,
    pub lockfile_blake3: String,
    pub rustc_vv: String,
    pub target_triple: String,
    pub profile: String,
    pub features: Vec<String>,
    pub rustflags: String,
    pub build_rs_env: Vec<(String, String)>,
    pub proc_macro_dylib_blake3: String,
    pub remap_path_prefix: String,
}

impl CacheKeyInputs {
    /// Compute a canonical 10-case domain-separated BLAKE3 hex digest.
    /// Features and `build.rs` env pairs are sorted canonically; `OUT_DIR` env entries
    /// and `OUT_DIR`-rooted paths are excluded so worktree cache keys stay portable.
    pub fn digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"refit-rust-cache-key-v1\0");
        hash_field(&mut hasher, b"src", &self.source_blake3);
        hash_field(&mut hasher, b"lock", &self.lockfile_blake3);
        hash_field(&mut hasher, b"rustc", &self.rustc_vv);
        hash_field(&mut hasher, b"target", &self.target_triple);
        hash_field(&mut hasher, b"profile", &self.profile);
        hash_sorted_list(&mut hasher, b"feat", &self.features);
        hash_field(&mut hasher, b"rustflags", &self.rustflags);
        hash_build_rs_env(&mut hasher, &self.build_rs_env);
        hash_field(&mut hasher, b"pmacro", &self.proc_macro_dylib_blake3);
        hash_field(&mut hasher, b"remap", &self.remap_path_prefix);
        hasher.finalize().to_hex().to_string()
    }

    pub fn for_repo(repo_root: &Path, profile: &str) -> Self {
        let (rustc_vv, target_triple) = cached_rustc_vv();
        let source_blake3 = hash_path_if_cacheable(&repo_root.join("Cargo.toml"));
        let lockfile_blake3 = hash_path_if_cacheable(&repo_root.join("Cargo.lock"));
        let rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
        let build_rs_env = collect_build_rs_env();

        Self {
            source_blake3,
            lockfile_blake3,
            rustc_vv,
            target_triple,
            profile: profile.to_string(),
            features: Vec::new(),
            rustflags,
            build_rs_env,
            proc_macro_dylib_blake3: "none".to_string(),
            remap_path_prefix: "/refit-build".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RustCachePlan {
    pub backend: RustCacheBackend,
    pub wrapper_bin: Option<PathBuf>,
    pub target_dir: PathBuf,
    pub build_dir: PathBuf,
    pub cache_key: String,
    pub incremental: bool,
    pub blaze_enabled: bool,
}

impl RustCachePlan {
    pub fn for_slot(repo_root: &Path, slot_dir: &Path, ephemeral_stage: bool) -> Self {
        let detected = detect_rust_wrapper_in(&env_path_dirs());
        let (backend, wrapper_bin) = match detected {
            Some((kind, bin)) => (kind, Some(bin)),
            None => (RustCacheBackend::CargoBuildDir, None),
        };
        let profile = if ephemeral_stage { "stage-check" } else { "audit-check" };
        let cache_key = CacheKeyInputs::for_repo(repo_root, profile).digest_hex();
        let opt_in = env_truthy("MBX_BLAZE");
        let blaze_enabled = allow_mbx_blaze(budget().effective_mem_bytes, opt_in);
        let incremental = !ephemeral_stage && !budget().is_constrained();
        let build_dir = slot_dir.join("build-dir");

        Self {
            backend,
            wrapper_bin,
            target_dir: slot_dir.to_path_buf(),
            build_dir,
            cache_key,
            incremental,
            blaze_enabled,
        }
    }

    pub fn to_env(&self, repo_root: &Path) -> Vec<(String, String)> {
        let cache_root = repo_root.join(".refit").join("cache").join("rust");
        let mut env = vec![
            ("CARGO_TARGET_DIR".into(), self.target_dir.display().to_string()),
            ("CARGO_BUILD_BUILD_DIR".into(), self.build_dir.display().to_string()),
            ("REFIT_RUST_CACHE_BACKEND".into(), self.backend.as_str().into()),
            ("REFIT_RUST_CACHE_KEY".into(), self.cache_key.clone()),
            ("MALLOC_ARENA_MAX".into(), "1".into()),
            ("MIMALLOC_PURGE_DELAY".into(), "0".into()),
            ("MBX_BLAZE".into(), if self.blaze_enabled { "1".into() } else { "0".into() }),
        ];
        if !self.incremental {
            env.push(("CARGO_INCREMENTAL".into(), "0".into()));
        }
        if let Some(bin) = &self.wrapper_bin {
            env.push(("RUSTC_WRAPPER".into(), bin.display().to_string()));
            env.push(("MBX_EXCLUDE_OUT_DIR".into(), "1".into()));
            match self.backend {
                RustCacheBackend::Mbx => {
                    env.push(("MBX_CACHE_DIR".into(), cache_root.display().to_string()));
                }
                RustCacheBackend::Kache => {
                    env.push(("KACHE_DIR".into(), cache_root.display().to_string()));
                }
                RustCacheBackend::CargoBuildDir => {}
            }
        }
        env
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheGcStats {
    pub evicted_entries: usize,
    pub freed_bytes: u64,
    pub remaining_bytes: u64,
}

/// Classify non-fatal reflink errors (`EPERM` / `PermissionDenied` / `ENOTSUP` / `EOPNOTSUPP` /
/// `EXDEV` / `ENOSYS` / `Unsupported`) that must trigger fallback rather than aborting.
pub fn is_reflink_fallback_err(err: &io::Error) -> bool {
    if matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::AlreadyExists) {
        return false;
    }
    if matches!(
        err.kind(),
        io::ErrorKind::PermissionDenied
            | io::ErrorKind::Unsupported
            | io::ErrorKind::CrossesDevices
    ) {
        return true;
    }
    is_fallback_os_code(err.raw_os_error(), false)
}

/// Classify non-fatal `hard_link` errors that should fall back to verified `fs::copy`.
pub fn is_hardlink_fallback_err(err: &io::Error) -> bool {
    if matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::AlreadyExists) {
        return false;
    }
    if matches!(
        err.kind(),
        io::ErrorKind::PermissionDenied
            | io::ErrorKind::Unsupported
            | io::ErrorKind::CrossesDevices
    ) {
        return true;
    }
    is_fallback_os_code(err.raw_os_error(), true)
}

fn is_fallback_os_code(code: Option<i32>, include_emlink: bool) -> bool {
    let Some(raw) = code else { return false };
    // POSIX errno codes across Linux/macOS/BSD: EPERM(1), EXDEV(18), EINVAL(22),
    // ENOSYS(38/78), ENOTSUP/EOPNOTSUPP(45/95), plus EMLINK(31) for link caps.
    if matches!(raw, 1 | 18 | 22 | 38 | 45 | 78 | 95) {
        return true;
    }
    if include_emlink && raw == 31 {
        return true;
    }
    #[cfg(windows)]
    {
        // ERROR_ACCESS_DENIED(5), ERROR_NOT_SAME_DEVICE(17), ERROR_NOT_SUPPORTED(50),
        // ERROR_INVALID_PARAMETER(87), ERROR_TOO_MANY_LINKS(1142).
        if matches!(raw, 5 | 17 | 50 | 87) || (include_emlink && raw == 1142) {
            return true;
        }
    }
    false
}

/// Same-FS copy cascade: `reflink` (when `build-cache` enabled) -> `hard_link` -> verified `copy`.
pub fn clone_or_copy(src: &Path, dst: &Path) -> Result<CloneOutcome> {
    clone_or_copy_with(src, dst, try_reflink_native, try_hardlink_native)
}

fn try_hardlink_native(src: &Path, dst: &Path) -> io::Result<()> {
    fs::hard_link(src, dst)
}

/// Testable cascade seam accepting injectable reflink and hardlink hooks.
pub fn clone_or_copy_with<R, H>(
    src: &Path,
    dst: &Path,
    reflink_fn: R,
    hardlink_fn: H,
) -> Result<CloneOutcome>
where
    R: FnOnce(&Path, &Path) -> io::Result<()>,
    H: FnOnce(&Path, &Path) -> io::Result<()>,
{
    match reflink_fn(src, dst) {
        Ok(()) => return Ok(CloneOutcome::Reflinked),
        Err(err) if is_reflink_fallback_err(&err) => {}
        Err(err) => {
            anyhow::bail!(
                "clone_or_copy reflink failed for {} -> {}: {err}",
                src.display(),
                dst.display()
            );
        }
    }
    match hardlink_fn(src, dst) {
        Ok(()) => return Ok(CloneOutcome::HardLinked),
        Err(err) if is_hardlink_fallback_err(&err) => {}
        Err(err) => {
            anyhow::bail!(
                "clone_or_copy hard_link failed for {} -> {}: {err}",
                src.display(),
                dst.display()
            );
        }
    }
    verified_copy(src, dst).with_context(|| {
        format!(
            "clone_or_copy verified copy failed for {} -> {}",
            src.display(),
            dst.display()
        )
    })?;
    Ok(CloneOutcome::Copied)
}

fn try_reflink_native(src: &Path, dst: &Path) -> io::Result<()> {
    #[cfg(feature = "build-cache")]
    {
        reflink_copy::reflink(src, dst)
    }
    #[cfg(not(feature = "build-cache"))]
    {
        let _ = (src, dst);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "build-cache feature disabled",
        ))
    }
}

fn verified_copy(src: &Path, dst: &Path) -> io::Result<u64> {
    let expected = fs::metadata(src)?.len();
    let copied = fs::copy(src, dst)?;
    if copied != expected {
        let _ = fs::remove_file(dst);
        return Err(io::Error::other(format!(
            "short copy for {} -> {} (copied {copied} != expected {expected})",
            src.display(),
            dst.display()
        )));
    }
    Ok(copied)
}

/// Returns `true` when `path` targets Cargo's build-script `OUT_DIR` or `target/*/build/*/out`,
/// which MUST be excluded from cross-worktree Rust build-cache keys (`mbx#633`).
pub fn is_out_dir_excluded(path: &Path) -> bool {
    let raw = path.to_string_lossy().replace('\\', "/");
    let segs: Vec<&str> = raw.split('/').filter(|seg| !seg.is_empty()).collect();
    if segs.iter().any(|seg| matches!(*seg, "OUT_DIR" | "$OUT_DIR" | "${OUT_DIR}")) {
        return true;
    }
    if let Some(env_out) = std::env::var_os("OUT_DIR") {
        let out_norm = Path::new(&env_out).to_string_lossy().replace('\\', "/");
        if !out_norm.is_empty() && raw.starts_with(out_norm.as_str()) {
            return true;
        }
    }
    has_cargo_build_out_segments(&segs)
}

fn has_cargo_build_out_segments(segs: &[&str]) -> bool {
    for (idx, seg) in segs.iter().enumerate() {
        if *seg != "build" || idx + 2 >= segs.len() {
            continue;
        }
        if segs[idx + 2] != "out" {
            continue;
        }
        let in_cargo_root = segs[..idx].iter().any(|anc| {
            matches!(*anc, "target" | "cargo-slots" | "build-dir" | ".refit-stages")
                || anc.starts_with("slot-")
        });
        if in_cargo_root {
            return true;
        }
    }
    false
}

/// Build Rust gate/audit cache environment using `mbx` / `kache` or Cargo 1.100 separated build-dir.
pub fn build_rust_cache_env(
    repo_root: &Path,
    slot_dir: &Path,
    ephemeral_stage: bool,
) -> Vec<(String, String)> {
    ensure_cache_dirs_tagged(repo_root);
    let plan = RustCachePlan::for_slot(repo_root, slot_dir, ephemeral_stage);
    let _ = fs::create_dir_all(&plan.build_dir);
    plan.to_env(repo_root)
}

/// Strict `indexable-inc/blaze` counterexample guard:
/// ALWAYS `false` on Windows or when `effective_mem_bytes < 16 GiB`, and strictly opt-in.
pub fn allow_mbx_blaze(effective_mem_bytes: u64, opt_in_env: bool) -> bool {
    allow_mbx_blaze_for_os(std::env::consts::OS, effective_mem_bytes, opt_in_env)
}

/// OS-parameterized seam for `allow_mbx_blaze` so tests can verify Linux/macOS/Windows rules.
pub fn allow_mbx_blaze_for_os(
    target_os: &str,
    effective_mem_bytes: u64,
    opt_in_env: bool,
) -> bool {
    let supported_os = matches!(target_os, "linux" | "macos");
    supported_os && effective_mem_bytes >= BLAZE_MIN_MEM_BYTES && opt_in_env
}

/// Return the three canonical refit cache directories:
/// `.refit/cache/`, `.refit-stages/`, and `.refit-tx/`.
pub fn cache_dirs_for_repo(repo: &Path) -> [PathBuf; 3] {
    [
        repo.join(".refit").join("cache"),
        work_dir(repo, ".refit-stages", "refit-stages"),
        work_dir(repo, ".refit-tx", "refit-tx"),
    ]
}

/// Ensure exact 43-byte-signature `CACHEDIR.TAG` exists in `.refit/cache/`, `.refit-stages/`,
/// and `.refit-tx/`.
pub fn ensure_cache_dirs_tagged(repo: &Path) -> [PathBuf; 3] {
    let dirs = cache_dirs_for_repo(repo);
    for dir in &dirs {
        ensure_cachedir_tag(dir);
    }
    dirs
}

/// Evict oldest-mtime entries (files/subdirs) in `dir` until remaining payload `<= max_bytes`.
/// Never deletes `CACHEDIR.TAG`.
pub fn prune_cache_lru(dir: &Path, max_bytes: u64) -> Result<CacheGcStats> {
    prune_cache_dirs_lru(&[dir.to_path_buf()], max_bytes)
}

/// Tag and LRU-prune `.refit/cache/`, `.refit-stages/`, and `.refit-tx/` under one hard cap.
pub fn prune_repo_caches(repo: &Path, max_bytes: u64) -> Result<CacheGcStats> {
    let dirs = ensure_cache_dirs_tagged(repo);
    prune_cache_dirs_lru(&dirs, max_bytes)
}

struct LruEntry {
    path: PathBuf,
    is_dir: bool,
    size: u64,
    mtime_ns: u64,
}

fn prune_cache_dirs_lru(dirs: &[PathBuf], max_bytes: u64) -> Result<CacheGcStats> {
    let mut entries = Vec::new();
    for dir in dirs {
        collect_lru_entries(dir, &mut entries)?;
    }

    let mut total: u64 = entries.iter().map(|entry| entry.size).sum();
    entries.sort_by(|left, right| {
        left.mtime_ns
            .cmp(&right.mtime_ns)
            .then_with(|| left.path.cmp(&right.path))
    });

    let mut evicted_entries = 0usize;
    let mut freed_bytes = 0u64;
    for entry in entries {
        if total <= max_bytes {
            break;
        }
        let removed = if entry.is_dir {
            fs::remove_dir_all(&entry.path).is_ok()
        } else {
            fs::remove_file(&entry.path).is_ok()
        };
        if removed {
            evicted_entries += 1;
            freed_bytes = freed_bytes.saturating_add(entry.size);
            total = total.saturating_sub(entry.size);
        }
    }

    Ok(CacheGcStats {
        evicted_entries,
        freed_bytes,
        remaining_bytes: total,
    })
}

fn collect_lru_entries(dir: &Path, out: &mut Vec<LruEntry>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let rd = fs::read_dir(dir).with_context(|| format!("read cache dir {}", dir.display()))?;
    for item in rd.flatten() {
        if item.file_name().to_string_lossy() == "CACHEDIR.TAG" {
            continue;
        }
        let path = item.path();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        let is_dir = meta.is_dir();
        let (size, mtime_ns) = if is_dir {
            measure_dir_tree(&path, system_time_ns(meta.modified().ok()))
        } else {
            (meta.len(), system_time_ns(meta.modified().ok()))
        };
        out.push(LruEntry { path, is_dir, size, mtime_ns });
    }
    Ok(())
}

fn measure_dir_tree(root: &Path, fallback_mtime: u64) -> (u64, u64) {
    let mut stack = vec![root.to_path_buf()];
    let mut total_bytes = 0u64;
    let mut newest_file_mtime = 0u64;
    let mut saw_file = false;

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
            } else {
                saw_file = true;
                total_bytes = total_bytes.saturating_add(meta.len());
                newest_file_mtime = newest_file_mtime.max(system_time_ns(meta.modified().ok()));
            }
        }
    }
    let mtime_ns = if saw_file { newest_file_mtime } else { fallback_mtime };
    (total_bytes, mtime_ns)
}

fn system_time_ns(ts: Option<SystemTime>) -> u64 {
    ts.and_then(|val| val.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |dur| dur.as_nanos() as u64)
}

/// JS/TS worktree cache environment (`Bun` global store + isolated linker, `aube`/`nub` probe,
/// and same-FS `PNPM_STORE_DIR` inside `.refit/cache/pnpm-store`).
pub fn js_worktree_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    js_worktree_cache_env_with(repo_root, &env_path_dirs())
}

pub fn js_worktree_cache_env_with(repo_root: &Path, search_dirs: &[PathBuf]) -> Vec<(String, String)> {
    let cache_dir = repo_root.join(".refit").join("cache");
    let pnpm_store = cache_dir.join("pnpm-store");
    let bun_cache = cache_dir.join("bun-cache");
    let mut env = vec![
        ("BUN_INSTALL_GLOBAL_STORE".into(), "1".into()),
        ("BUN_INSTALL_CACHE_DIR".into(), bun_cache.display().to_string()),
        ("BUN_INSTALL_LINKER".into(), "isolated".into()),
        ("REFIT_BUN_LINKER_ARGS".into(), BUN_ISOLATED_LINKER_ARGS.join(" ")),
        ("PNPM_STORE_DIR".into(), pnpm_store.display().to_string()),
    ];
    if let Some((tool, bin)) = detect_js_relinker_in(search_dirs) {
        env.push(("REFIT_JS_RELINKER".into(), tool.into()));
        env.push(("REFIT_JS_RELINKER_BIN".into(), bin.display().to_string()));
    }
    env
}

/// OS default safe `UV_LINK_MODE`: `"clone"` on macOS APFS, `"hardlink"` on Linux/Windows.
pub fn default_uv_link_mode() -> &'static str {
    if cfg!(target_os = "macos") {
        "clone"
    } else {
        "hardlink"
    }
}

/// Validate a requested `UV_LINK_MODE`, explicitly rejecting `symlink` (`Err`)
/// because `uv cache clean` breaks active virtualenvs.
pub fn validate_uv_link_mode(mode: &str) -> Result<&'static str> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "hardlink" => Ok("hardlink"),
        "clone" => Ok("clone"),
        "copy" => Ok("copy"),
        "symlink" => anyhow::bail!(
            "UV_LINK_MODE=symlink is forbidden because `uv cache clean` breaks active virtualenvs"
        ),
        other => anyhow::bail!("unsupported UV_LINK_MODE `{other}`"),
    }
}

/// Resolve a safe `UV_LINK_MODE`, overriding forbidden `symlink` or unknown values to OS default.
pub fn safe_uv_link_mode(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|raw| validate_uv_link_mode(raw).ok())
        .unwrap_or_else(default_uv_link_mode)
}

/// Python worktree cache environment (`UV_CACHE_DIR` in `.refit/cache/uv-cache` + safe
/// `UV_LINK_MODE`, explicitly overriding `UV_LINK_MODE=symlink`, zero idle daemons).
pub fn py_worktree_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let inherited = std::env::var("UV_LINK_MODE").ok();
    py_worktree_cache_env_with(repo_root, inherited.as_deref())
}

pub fn py_worktree_cache_env_with(
    repo_root: &Path,
    requested_link_mode: Option<&str>,
) -> Vec<(String, String)> {
    let uv_cache = repo_root.join(".refit").join("cache").join("uv-cache");
    let link_mode = safe_uv_link_mode(requested_link_mode);
    vec![
        ("UV_CACHE_DIR".into(), uv_cache.display().to_string()),
        ("UV_LINK_MODE".into(), link_mode.into()),
        ("MALLOC_ARENA_MAX".into(), "1".into()),
        ("MIMALLOC_PURGE_DELAY".into(), "0".into()),
    ]
}

/// Combined JS/TS + Python worktree cache environment plus strict `MBX_BLAZE` guard.
pub fn polyglot_cache_env(repo_root: &Path) -> Vec<(String, String)> {
    let mut env = js_worktree_cache_env(repo_root);
    env.extend(py_worktree_cache_env(repo_root));
    let opt_in = env_truthy("MBX_BLAZE");
    let blaze_val = if allow_mbx_blaze(budget().effective_mem_bytes, opt_in) {
        "1"
    } else {
        "0"
    };
    env.push(("MBX_BLAZE".into(), blaze_val.into()));
    env
}

pub fn detect_rust_wrapper_in(dirs: &[PathBuf]) -> Option<(RustCacheBackend, PathBuf)> {
    if let Some(bin) = find_bin_in_dirs(dirs, "mbx") {
        return Some((RustCacheBackend::Mbx, bin));
    }
    if let Some(bin) = find_bin_in_dirs(dirs, "kache") {
        return Some((RustCacheBackend::Kache, bin));
    }
    None
}

pub fn detect_js_relinker_in(dirs: &[PathBuf]) -> Option<(&'static str, PathBuf)> {
    if let Some(bin) = find_bin_in_dirs(dirs, "aube") {
        return Some(("aube", bin));
    }
    if let Some(bin) = find_bin_in_dirs(dirs, "nub") {
        return Some(("nub", bin));
    }
    None
}

fn env_path_dirs() -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|val| std::env::split_paths(&val).collect())
        .unwrap_or_default()
}

fn find_bin_in_dirs(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    let candidates: Vec<String> = if cfg!(windows) {
        vec![format!("{name}.exe"), format!("{name}.cmd"), name.to_string()]
    } else {
        vec![name.to_string()]
    };
    for dir in dirs {
        for cand in &candidates {
            let full = dir.join(cand);
            if full.is_file() {
                return Some(full);
            }
        }
    }
    None
}

fn env_truthy(var: &str) -> bool {
    std::env::var(var)
        .is_ok_and(|val| matches!(val.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

fn hash_field(hasher: &mut blake3::Hasher, tag: &[u8], value: &str) {
    hasher.update(tag);
    hasher.update(b":");
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
    hasher.update(b"\0");
}

fn hash_sorted_list(hasher: &mut blake3::Hasher, tag: &[u8], items: &[String]) {
    let mut sorted: Vec<&str> = items.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.dedup();
    hasher.update(tag);
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for item in sorted {
        hash_field(hasher, b"i", item);
    }
}

fn hash_build_rs_env(hasher: &mut blake3::Hasher, pairs: &[(String, String)]) {
    let mut sorted: Vec<(&str, &str)> = pairs
        .iter()
        .filter(|(key, val)| key != "OUT_DIR" && !is_out_dir_excluded(Path::new(val)))
        .map(|(key, val)| (key.as_str(), val.as_str()))
        .collect();
    sorted.sort_unstable();
    hasher.update(b"build_rs_env");
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for (key, val) in sorted {
        hash_field(hasher, b"k", key);
        hash_field(hasher, b"v", val);
    }
}

fn hash_path_if_cacheable(path: &Path) -> String {
    if is_out_dir_excluded(path) {
        return "excluded_out_dir".to_string();
    }
    match fs::read(path) {
        Ok(bytes) => blake3::hash(&bytes).to_hex().to_string(),
        Err(_) => "none".to_string(),
    }
}

fn collect_build_rs_env() -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for (key, val) in std::env::vars() {
        let relevant = key.starts_with("CARGO_CFG_") || key.starts_with("DEP_");
        if relevant && key != "OUT_DIR" && !is_out_dir_excluded(Path::new(&val)) {
            pairs.push((key, val));
        }
    }
    pairs.sort();
    pairs
}

fn cached_rustc_vv() -> (String, String) {
    RUSTC_VV_CACHE
        .get_or_init(|| {
            let fallback_triple = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);
            let Ok(out) = Command::new("rustc").arg("-vV").output() else {
                return (format!("rustc-unknown-{fallback_triple}"), fallback_triple);
            };
            if !out.status.success() {
                return (format!("rustc-unknown-{fallback_triple}"), fallback_triple);
            }
            let raw = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let triple = raw
                .lines()
                .find_map(|line| line.strip_prefix("host:"))
                .map(|host| host.trim().to_string())
                .unwrap_or(fallback_triple);
            (raw, triple)
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::has_cachedir_tag;
    use crate::testutil::TempRepo;
    use std::time::Duration;

    fn sample_key_inputs() -> CacheKeyInputs {
        CacheKeyInputs {
            source_blake3: "src_hash_aaa".into(),
            lockfile_blake3: "lock_hash_bbb".into(),
            rustc_vv: "rustc 1.100.0 (0ef1bc386 2026-11-09)".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            profile: "dev".into(),
            features: vec!["alloc-mimalloc".into(), "build-cache".into()],
            rustflags: "-C target-cpu=native".into(),
            build_rs_env: vec![
                ("DEP_ZSTD_INCLUDE".into(), "/usr/include".into()),
                ("CARGO_CFG_UNIX".into(), "1".into()),
            ],
            proc_macro_dylib_blake3: "pmacro_hash_ccc".into(),
            remap_path_prefix: "/refit-build".into(),
        }
    }

    #[test]
    fn fault_inject_eperm_exdev_enosys_permission_denied_cascade() {
        let tmp = TempRepo::with_files("cache_clone", &[("src.bin", "payload-verify-0123456789\n")]);
        let src = tmp.join("src.bin");
        let expected = fs::read(&src).unwrap();

        let reflink_errs = [
            ("EPERM", io::Error::from_raw_os_error(1)),
            ("EXDEV", io::Error::from_raw_os_error(18)),
            ("ENOSYS", io::Error::from_raw_os_error(38)),
            ("EOPNOTSUPP", io::Error::from_raw_os_error(95)),
            ("ENOTSUP", io::Error::from_raw_os_error(45)),
            ("PermissionDenied", io::Error::new(io::ErrorKind::PermissionDenied, "lxc apparmor")),
            ("Unsupported", io::Error::new(io::ErrorKind::Unsupported, "no ficlone")),
        ];

        // 1. Reflink fails with each non-fatal trigger -> falls back cleanly to HardLinked.
        for (label, err) in reflink_errs {
            let dst = tmp.join(&format!("hl_{label}.bin"));
            let outcome =
                clone_or_copy_with(&src, &dst, |_, _| Err(err), try_hardlink_native).unwrap();
            assert_eq!(outcome, CloneOutcome::HardLinked, "case {label}");
            assert_eq!(fs::read(&dst).unwrap(), expected);
        }

        // 2. Both reflink AND hard_link fail with EPERM/EXDEV/ENOSYS/PermissionDenied -> verified Copied.
        let dual_errs = [
            ("dual_eperm", io::Error::from_raw_os_error(1), io::Error::from_raw_os_error(1)),
            ("dual_exdev", io::Error::from_raw_os_error(18), io::Error::from_raw_os_error(18)),
            ("dual_enosys", io::Error::from_raw_os_error(38), io::Error::from_raw_os_error(38)),
            (
                "dual_perm",
                io::Error::new(io::ErrorKind::PermissionDenied, "lxc"),
                io::Error::new(io::ErrorKind::PermissionDenied, "readonly-link"),
            ),
        ];
        for (label, r_err, h_err) in dual_errs {
            let dst = tmp.join(&format!("{label}.bin"));
            let outcome = clone_or_copy_with(&src, &dst, |_, _| Err(r_err), |_, _| Err(h_err)).unwrap();
            assert_eq!(outcome, CloneOutcome::Copied, "case {label}");
            assert_eq!(fs::read(&dst).unwrap(), expected);
        }

        // 3. Real native clone_or_copy succeeds on local FS (with or without build-cache feature).
        let native_dst = tmp.join("native.bin");
        let native_out = clone_or_copy(&src, &native_dst).unwrap();
        assert!(matches!(
            native_out,
            CloneOutcome::Reflinked | CloneOutcome::HardLinked | CloneOutcome::Copied
        ));
        assert_eq!(fs::read(&native_dst).unwrap(), expected);

        // 4. Fatal NotFound error NEVER drops silently.
        let missing = tmp.join("does_not_exist.bin");
        let err = clone_or_copy(&missing, &tmp.join("never.bin")).unwrap_err();
        assert!(err.to_string().contains("clone_or_copy"));
    }

    #[test]
    fn ten_case_invalidation_matrix_and_out_dir_exclusion() {
        let base = sample_key_inputs();
        let base_hex = base.digest_hex();
        assert_eq!(base_hex.len(), 64);

        // Canonical ordering of features & build_rs_env must produce the exact same key.
        let mut reordered = base.clone();
        reordered.features.reverse();
        reordered.build_rs_env.reverse();
        assert_eq!(reordered.digest_hex(), base_hex);

        // Mutating ANY of the 10 inputs MUST change the digest.
        let mut mutations: Vec<(&str, CacheKeyInputs)> = Vec::new();
        let mut c1 = base.clone();
        c1.source_blake3 = "src_hash_mut".into();
        mutations.push(("1_source_blake3", c1));

        let mut c2 = base.clone();
        c2.lockfile_blake3 = "lock_hash_mut".into();
        mutations.push(("2_lockfile_blake3", c2));

        let mut c3 = base.clone();
        c3.rustc_vv = "rustc 1.101.0".into();
        mutations.push(("3_rustc_vv", c3));

        let mut c4 = base.clone();
        c4.target_triple = "aarch64-apple-darwin".into();
        mutations.push(("4_target_triple", c4));

        let mut c5 = base.clone();
        c5.profile = "release".into();
        mutations.push(("5_profile", c5));

        let mut c6 = base.clone();
        c6.features.push("ste100".into());
        mutations.push(("6_features", c6));

        let mut c7 = base.clone();
        c7.rustflags = "-D warnings".into();
        mutations.push(("7_rustflags", c7));

        let mut c8 = base.clone();
        c8.build_rs_env.push(("DEP_OPENSSL_ROOT".into(), "/opt/ssl".into()));
        mutations.push(("8_build_rs_env", c8));

        let mut c9 = base.clone();
        c9.proc_macro_dylib_blake3 = "pmacro_hash_mut".into();
        mutations.push(("9_proc_macro_dylib_blake3", c9));

        let mut c10 = base.clone();
        c10.remap_path_prefix = "/alt-build".into();
        mutations.push(("10_remap_path_prefix", c10));

        assert_eq!(mutations.len(), 10);
        for (name, mutated) in mutations {
            assert_ne!(mutated.digest_hex(), base_hex, "mutation {name} failed to invalidate key");
        }

        // OUT_DIR env entries and target/*/build/*/out paths MUST be excluded.
        let mut with_out_env = base.clone();
        with_out_env.build_rs_env.push((
            "OUT_DIR".into(),
            "/wt-a/target/debug/build/serde-1234/out".into(),
        ));
        with_out_env.build_rs_env.push((
            "DEP_GENERATED_PATH".into(),
            "target/release/build/ring-9999/out/asm.rs".into(),
        ));
        assert_eq!(with_out_env.digest_hex(), base_hex);

        assert!(is_out_dir_excluded(Path::new("target/debug/build/foo-123/out/bindings.rs")));
        assert!(is_out_dir_excluded(Path::new(r"target\release\build\bar-456\out\gen.rs")));
        assert!(is_out_dir_excluded(Path::new(
            "target/x86_64-unknown-linux-gnu/debug/build/baz-789/out"
        )));
        assert!(is_out_dir_excluded(Path::new(".refit-stages/cargo-slots/slot-0/debug/build/x/out/a.rs")));
        assert!(is_out_dir_excluded(Path::new("/tmp/OUT_DIR/generated.rs")));
        assert!(!is_out_dir_excluded(Path::new("src/gates/cache_orch.rs")));
        assert!(!is_out_dir_excluded(Path::new("target/debug/deps/librefit.rlib")));
    }

    #[test]
    fn lru_gc_enforces_cap_and_preserves_43_byte_cachedir_tag() {
        let tmp = TempRepo::nested("cache_lru", "repo");
        let dirs = ensure_cache_dirs_tagged(tmp.path());
        for dir in &dirs {
            assert!(has_cachedir_tag(dir), "missing tag in {}", dir.display());
            let raw = fs::read(dir.join("CACHEDIR.TAG")).unwrap();
            assert!(
                raw.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55"),
                "43-byte signature mismatch in {}",
                dir.display()
            );
            assert_eq!(raw.len(), 44);
        }

        // Populate .refit/cache with 6 MiB across oldest/middle/newest files + subdirs.
        let cache_dir = &dirs[0];
        let chunk_2m = vec![0xA5u8; 2 * 1024 * 1024];
        let old_file = cache_dir.join("01_oldest.bin");
        let mid_dir = cache_dir.join("02_middle_dir");
        let new_file = cache_dir.join("03_newest.bin");

        fs::write(&old_file, &chunk_2m).unwrap();
        fs::create_dir_all(&mid_dir).unwrap();
        fs::write(mid_dir.join("blob.bin"), &chunk_2m).unwrap();
        fs::write(&new_file, &chunk_2m).unwrap();

        let now = SystemTime::now();
        fs::OpenOptions::new()
            .write(true)
            .open(&old_file)
            .unwrap()
            .set_modified(now - Duration::from_secs(300))
            .unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(mid_dir.join("blob.bin"))
            .unwrap()
            .set_modified(now - Duration::from_secs(120))
            .unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&new_file)
            .unwrap()
            .set_modified(now)
            .unwrap();

        let cap_4m = 4 * 1024 * 1024;
        let stats = prune_cache_lru(cache_dir, cap_4m).unwrap();
        assert_eq!(stats.evicted_entries, 1);
        assert_eq!(stats.freed_bytes, 2 * 1024 * 1024);
        assert!(stats.remaining_bytes <= cap_4m);
        assert!(!old_file.exists(), "oldest entry should be evicted");
        assert!(mid_dir.exists() && new_file.exists(), "newer entries should remain");
        assert!(has_cachedir_tag(cache_dir), "CACHEDIR.TAG must never be deleted");

        // Tighter cap (2 MiB) evicts middle subdir too, still preserving newest + CACHEDIR.TAG.
        let stats2 = prune_cache_lru(cache_dir, 2 * 1024 * 1024).unwrap();
        assert_eq!(stats2.evicted_entries, 1);
        assert!(!mid_dir.exists());
        assert!(new_file.exists());
        assert!(has_cachedir_tag(cache_dir));
    }

    #[test]
    fn wrapper_detection_js_py_envs_and_uv_symlink_ban() {
        let tmp = TempRepo::new("cache_env");
        let bin_dir = tmp.join("fake_bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let ext = if cfg!(windows) { ".exe" } else { "" };

        // Empty PATH -> None (CargoBuildDir fallback).
        assert!(detect_rust_wrapper_in(std::slice::from_ref(&bin_dir)).is_none());

        // kache detected when mbx absent.
        fs::write(bin_dir.join(format!("kache{ext}")), b"").unwrap();
        let (kind_k, _) = detect_rust_wrapper_in(std::slice::from_ref(&bin_dir)).unwrap();
        assert_eq!(kind_k, RustCacheBackend::Kache);

        // mbx takes precedence over kache.
        fs::write(bin_dir.join(format!("mbx{ext}")), b"").unwrap();
        let (kind_m, _) = detect_rust_wrapper_in(std::slice::from_ref(&bin_dir)).unwrap();
        assert_eq!(kind_m, RustCacheBackend::Mbx);

        // aube / nub probe + Bun globalStore + isolated linker + same-FS PNPM_STORE_DIR.
        fs::write(bin_dir.join(format!("nub{ext}")), b"").unwrap();
        let js_env = js_worktree_cache_env_with(tmp.path(), std::slice::from_ref(&bin_dir));
        let js_map: std::collections::BTreeMap<&str, &str> =
            js_env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(js_map.get("BUN_INSTALL_GLOBAL_STORE"), Some(&"1"));
        assert_eq!(js_map.get("BUN_INSTALL_LINKER"), Some(&"isolated"));
        assert_eq!(js_map.get("REFIT_BUN_LINKER_ARGS"), Some(&"--linker isolated"));
        assert!(js_map.get("PNPM_STORE_DIR").is_some_and(|v| v.contains("pnpm-store")));
        assert_eq!(js_map.get("REFIT_JS_RELINKER"), Some(&"nub"));

        fs::write(bin_dir.join(format!("aube{ext}")), b"").unwrap();
        let (relink_name, _) = detect_js_relinker_in(std::slice::from_ref(&bin_dir)).unwrap();
        assert_eq!(relink_name, "aube");

        // UV_LINK_MODE=symlink MUST be explicitly rejected AND overridden to OS-safe default!
        assert!(validate_uv_link_mode("symlink").is_err());
        assert!(validate_uv_link_mode("SYMLINK").is_err());
        assert_eq!(validate_uv_link_mode("hardlink").unwrap(), "hardlink");
        assert_eq!(validate_uv_link_mode("clone").unwrap(), "clone");
        assert_eq!(safe_uv_link_mode(Some("symlink")), default_uv_link_mode());

        let py_env = py_worktree_cache_env_with(tmp.path(), Some("symlink"));
        let py_map: std::collections::BTreeMap<&str, &str> =
            py_env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert!(py_map.get("UV_CACHE_DIR").is_some_and(|v| v.contains("uv-cache")));
        assert_ne!(py_map.get("UV_LINK_MODE"), Some(&"symlink"));
        assert_eq!(py_map.get("UV_LINK_MODE"), Some(&default_uv_link_mode()));
    }

    #[test]
    fn blaze_guard_rejects_windows_and_sub_16gib() {
        let gib = 1024 * 1024 * 1024u64;
        // Windows ALWAYS false regardless of RAM or opt-in.
        assert!(!allow_mbx_blaze_for_os("windows", 64 * gib, true));
        assert!(!allow_mbx_blaze_for_os("windows", 16 * gib, true));
        if cfg!(target_os = "windows") {
            assert!(!allow_mbx_blaze(64 * gib, true));
        }

        // Sub-16 GiB (including 256 MiB containers) ALWAYS false on Linux & macOS.
        assert!(!allow_mbx_blaze_for_os("linux", 256 * 1024 * 1024, true));
        assert!(!allow_mbx_blaze_for_os("linux", 16 * gib - 1, true));
        assert!(!allow_mbx_blaze_for_os("macos", 8 * gib, true));

        // Without opt-in ALWAYS false even on 64 GiB Linux/macOS.
        assert!(!allow_mbx_blaze_for_os("linux", 64 * gib, false));
        assert!(!allow_mbx_blaze_for_os("macos", 64 * gib, false));

        // Strictly true ONLY on Linux/macOS with >= 16 GiB AND opt-in.
        assert!(allow_mbx_blaze_for_os("linux", 16 * gib, true));
        assert!(allow_mbx_blaze_for_os("macos", 32 * gib, true));
    }
}
