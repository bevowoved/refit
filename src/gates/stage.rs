use crate::platform::{now_ms, walk_files};
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
}

impl CargoSlot {
    pub(super) fn acquire(repo_root: &Path) -> Self {
        let pool = CARGO_SLOT_POOL.get_or_init(|| Mutex::new((0..8).rev().collect()));
        let (slot, pooled) = match pool.lock().pop() {
            Some(s) => (s, true),
            None => (CARGO_EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed) as usize, false),
        };
        let dir = stages_dir(repo_root).join("cargo-slots").join(format!("slot-{slot}"));
        let _ = fs::create_dir_all(&dir);
        Self { slot, pooled, dir }
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
    match fs::hard_link(src, dst) {
        Ok(()) => Ok((1, 0)),
        Err(link_err) => match fs::copy(src, dst) {
            Ok(_) => Ok((0, 1)),
            Err(copy_err) => anyhow::bail!(
                "stage materialize failed for {} -> {} (hard_link: {link_err}; copy: {copy_err})",
                src.display(),
                dst.display()
            ),
        },
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
    if seq == 1 || seq.is_multiple_of(64) {
        let _ = sweep_stages(repo_root, 1800);
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
