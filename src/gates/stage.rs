use super::cache_orch::{
    CloneOutcome, DEFAULT_CACHE_CAP_BYTES, build_rust_cache_env, clone_or_copy,
    ensure_cache_dirs_tagged, prune_cache_lru,
};
use crate::platform::{SidecarLock, ensure_cachedir_tag, now_ms, walk_files};
use crate::source::{rel_norm, work_dir};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::mem::DropGuard;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

static STAGE_SEQ: AtomicU64 = AtomicU64::new(1);
static STAGE_HARDLINKS_TOTAL: AtomicU64 = AtomicU64::new(0);
static STAGE_COPIED_TOTAL: AtomicU64 = AtomicU64::new(0);
static CARGO_SLOT_POOL: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
static CARGO_EPHEMERAL_SEQ: AtomicU64 = AtomicU64::new(100);

pub fn stage_hardlinks() -> u64 {
    STAGE_HARDLINKS_TOTAL.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn stage_copies() -> u64 {
    STAGE_COPIED_TOTAL.load(Ordering::Relaxed)
}

pub(super) struct CargoSlot {
    #[cfg(test)]
    pub(super) slot: usize,
    #[cfg(not(test))]
    slot: usize,
    pooled: bool,
    pub(super) dir: PathBuf,
    _lock: Option<SidecarLock>,
}

fn max_cargo_slots() -> usize {
    std::env::var("REFIT_CARGO_SLOTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(2, |n| n.clamp(1, 2))
}

impl CargoSlot {
    pub(super) fn acquire(repo_root: &Path) -> Self {
        let cap = max_cargo_slots();
        let pool = CARGO_SLOT_POOL.get_or_init(|| Mutex::new((0..cap).rev().collect()));
        let slots_root = stages_dir(repo_root).join("cargo-slots");
        ensure_cachedir_tag(&stages_dir(repo_root));
        ensure_cachedir_tag(&slots_root);

        let mut skipped = Vec::new();
        let mut chosen: Option<(usize, SidecarLock)> = None;
        while let Some(s) = pool.lock().pop() {
            let lock_path = slots_root.join(format!("slot-{s}.lock"));
            if let Ok(Some(lk)) = SidecarLock::try_acquire(&lock_path) {
                chosen = Some((s, lk));
                break;
            }
            skipped.push(s);
        }
        if !skipped.is_empty() {
            let mut g = pool.lock();
            for s in skipped {
                g.push(s);
            }
        }

        let (slot, pooled, lock) = match chosen {
            Some((s, lk)) => (s, true, Some(lk)),
            None => (CARGO_EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed) as usize, false, None),
        };
        let dir = slots_root.join(format!("slot-{slot}"));
        let _ = fs::create_dir_all(&dir);
        Self { slot, pooled, dir, _lock: lock }
    }

    pub(super) fn rust_cache_env(
        &self,
        repo_root: &Path,
        ephemeral_stage: bool,
    ) -> Vec<(String, String)> {
        build_rust_cache_env(repo_root, &self.dir, ephemeral_stage)
    }
}

impl Drop for CargoSlot {
    fn drop(&mut self) {
        if self.pooled {
            if let Some(pool) = CARGO_SLOT_POOL.get() {
                pool.lock().push(self.slot);
            }
        } else {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

pub(super) fn stages_dir(repo_root: &Path) -> PathBuf {
    work_dir(repo_root, ".refit-stages", "refit-stages")
}

pub fn sweep_stages(repo_root: &Path, max_age_secs: u64) -> usize {
    let Ok(rd) = fs::read_dir(stages_dir(repo_root)) else { return 0 };
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(max_age_secs))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut swept = 0usize;

    for entry in rd.flatten() {
        let p = entry.path();
        if !p.is_dir() || !entry.file_name().to_string_lossy().starts_with("tx-") {
            continue;
        }
        let old = fs::metadata(&p).and_then(|m| m.modified()).is_ok_and(|m| m <= cutoff);
        if old && fs::remove_dir_all(&p).is_ok() {
            swept += 1;
        }
    }
    swept
}

#[cfg(test)]
pub fn encode_lsp(payload: &serde_json::Value) -> Vec<u8> {
    let body = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

#[cfg(test)]
pub fn decode_lsp(raw: &[u8]) -> Option<(serde_json::Value, usize)> {
    let hdr_end = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let hdr = std::str::from_utf8(&raw[..hdr_end]).ok()?;
    let len: usize = hdr.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if !k.trim().eq_ignore_ascii_case("content-length") {
            return None;
        }
        v.trim().parse().ok()
    })?;

    let end = hdr_end + 4 + len;
    if raw.len() < end {
        return None;
    }
    Some((serde_json::from_slice(&raw[hdr_end + 4..end]).ok()?, end))
}

fn link_or_copy_file(src: &Path, dst: &Path) -> Result<(u64, u64)> {
    match clone_or_copy(src, dst) {
        Ok(CloneOutcome::Reflinked | CloneOutcome::HardLinked) => Ok((1, 0)),
        Ok(CloneOutcome::Copied) => Ok((0, 1)),
        Err(err) => anyhow::bail!(
            "stage materialize failed for {} -> {}: {err}",
            src.display(),
            dst.display()
        ),
    }
}

fn link_or_copy_pairs(pairs: &[(PathBuf, PathBuf)]) -> Result<(u64, u64)> {
    pairs
        .par_iter()
        .try_fold(
            || (0u64, 0u64),
            |(hl, cp), (src, dst)| {
                let (dh, dc) = link_or_copy_file(src, dst)?;
                Ok((hl + dh, cp + dc))
            },
        )
        .try_reduce(|| (0u64, 0u64), |a, b| Ok((a.0 + b.0, a.1 + b.1)))
}

pub(super) fn materialize_stage(
    repo_root: &Path,
    candidates: &BTreeMap<String, String>,
    deleted: &[String],
) -> Result<DropGuard<PathBuf, impl FnOnce(PathBuf)>> {
    let seq = STAGE_SEQ.fetch_add(1, Ordering::Relaxed);
    ensure_cache_dirs_tagged(repo_root);
    if seq == 1 || seq.is_multiple_of(64) {
        let _ = sweep_stages(repo_root, 1800);
        let _ = prune_cache_lru(&stages_dir(repo_root), DEFAULT_CACHE_CAP_BYTES);
    }

    let stage_dir = stages_dir(repo_root).join(format!("tx-{}-{seq}", now_ms()));
    fs::create_dir_all(&stage_dir)?;
    let guard = DropGuard::new(stage_dir, |dir| {
        let _ = fs::remove_dir_all(dir);
    });
    let del_set: BTreeSet<String> = deleted.iter().map(|s| s.replace('\\', "/")).collect();

    let to_link: Vec<(PathBuf, PathBuf)> = walk_files(repo_root, None)
        .into_iter()
        .filter_map(|path| {
            let rel = rel_norm(repo_root, &path)?;
            if candidates.contains_key(&rel) || del_set.contains(&rel) {
                return None;
            }
            Some((path, guard.join(&rel)))
        })
        .collect();

    let parents: BTreeSet<&Path> = to_link.iter().filter_map(|(_, d)| d.parent()).collect();
    for p in parents {
        fs::create_dir_all(p)
            .with_context(|| format!("create staged parent dir {}", p.display()))?;
    }

    let (hl, cp) = link_or_copy_pairs(&to_link)?;

    if hl > 0 {
        STAGE_HARDLINKS_TOTAL.fetch_add(hl, Ordering::Relaxed);
    }
    if cp > 0 {
        STAGE_COPIED_TOTAL.fetch_add(cp, Ordering::Relaxed);
    }

    for (rel, content) in candidates {
        let dst = guard.join(rel);
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(&dst, content.as_bytes())
            .with_context(|| format!("write staged candidate {}", dst.display()))?;
    }
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use std::io::Read;

    #[test]
    fn stage_copy_failure_propagates_and_sentinel_guard() {
        let tmp = TempRepo::with_files("stage_err", &[("src/a.rs", "pub fn ok() -> u8 { 42 }\n")]);
        let dst_dir = tmp.join("out_stage");
        fs::create_dir_all(&dst_dir).unwrap();

        // Missing source: BOTH hard_link and copy must fail and propagate Err (not silent (0, 0)).
        let missing = tmp.join("src/does_not_exist.rs");
        let err = link_or_copy_pairs(&[(missing, dst_dir.join("target.rs"))]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("stage materialize failed"), "unexpected error: {msg}");

        // Destination collision with an existing directory: BOTH hard_link and copy fail -> Err.
        let occupied = dst_dir.join("occupied_dir");
        fs::create_dir_all(&occupied).unwrap();
        let err2 = link_or_copy_pairs(&[(tmp.join("src/a.rs"), occupied)]).unwrap_err();
        assert!(err2.to_string().contains("stage materialize failed"));

        // 0xFF blanking sentinel verification on successful materialization.
        let empty = BTreeMap::new();
        let stage = materialize_stage(tmp.path(), &empty, &[]).unwrap();
        let staged_file = stage.join("src/a.rs");
        let mut buf = [0xFFu8; 64];
        let n = fs::File::open(&staged_file).unwrap().read(&mut buf).unwrap();
        assert!(n > 0 && &buf[..n] == b"pub fn ok() -> u8 { 42 }\n");
        assert!(buf[n..].iter().all(|&b| b == 0xFF));
    }
}
