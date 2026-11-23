use crate::platform::{write_atomic, now_ms};
use crate::types::{WalFileEntry, WalRecord};
use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static TX_SEQ: AtomicU64 = AtomicU64::new(1);
static WAL_TIMELINE_CACHE: OnceLock<DashMap<PathBuf, (u64, Vec<WalEvent>)>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEvent {
    pub rev: u64,
    pub tx_id: String,
    pub module_id: String,
    pub state: String,
    pub files: Vec<String>,
    pub ts_ms: u64,
    pub reason: Option<String>,
    pub loc_delta: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: String,
    pub pre_hash: String,
    pub post_hash: String,
    pub added: usize,
    pub removed: usize,
    pub unified_diff: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxDiffReport {
    pub tx_id: String,
    pub rev: u64,
    pub module_id: String,
    pub state: String,
    pub files: Vec<FileDiff>,
}

#[derive(Debug, Clone)]
pub struct WalRecovery {
    pub rolled_back_txs: usize,
    pub committed_modules: HashMap<String, BTreeMap<String, String>>,
    pub head_rev: u64,
}

#[derive(Debug, Clone)]
pub enum CommitOutcome {
    Committed {
        head_rev: u64,
        tx_id: String,
        new_hashes: BTreeMap<String, String>,
    },
    CasConflict {
        path: String,
        expected: String,
        actual: String,
    },
}

struct WalLock(File);
impl Drop for WalLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

pub struct MvccStore {
    repo_root: PathBuf,
    cas_dir: PathBuf,
    wal_path: PathBuf,
    lock_path: PathBuf,
    commit_lock: Mutex<u64>,
}

impl MvccStore {
    pub fn open(repo_root: &Path, run_id: &str) -> Result<Self> {
        let refit_dir = repo_root.join(".refit");
        let cas_dir = refit_dir.join("cas");
        fs::create_dir_all(&cas_dir)
            .with_context(|| format!("create CAS dir {}", cas_dir.display()))?;
        let wal_path = refit_dir.join("wal.jsonl");
        let lock_path = refit_dir.join("wal.lock");

        // Recover persisted max head_rev on open so MCP & multi-process commits advance monotonically.
        let persisted_rev = wal_head(&wal_path);

        let store = Self {
            repo_root: repo_root.to_path_buf(),
            cas_dir,
            wal_path,
            lock_path,
            commit_lock: Mutex::new(persisted_rev),
        };

        if !store.wal_path.exists() {
            let _flock = store.lock_wal()?;
            if !store.wal_path.exists() {
                store.append_wal(&WalRecord::RunStart {
                    run_id: run_id.to_string(),
                    ts_ms: now_ms(),
                    repo_root: repo_root.display().to_string(),
                })?;
            }
        }

        Ok(store)
    }

    fn lock_wal(&self) -> Result<WalLock> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&self.lock_path)
            .with_context(|| format!("open WAL lock {}", self.lock_path.display()))?;
        file.lock()
            .with_context(|| format!("acquire OS lock {}", self.lock_path.display()))?;
        Ok(WalLock(file))
    }

    pub fn hash_bytes(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    pub fn hash_file(path: &Path) -> Result<String> {
        let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Ok(Self::hash_bytes(&data))
    }

    pub fn store_cas(&self, bytes: &[u8]) -> Result<String> {
        let hex = Self::hash_bytes(bytes);
        let obj = self.cas_dir.join(&hex);
        if !obj.exists() {
            write_atomic(&obj, bytes)?;
        }
        Ok(hex)
    }

    pub fn load_cas(&self, hash: &str) -> Result<Vec<u8>> {
        let obj = self.cas_dir.join(hash);
        fs::read(&obj).with_context(|| format!("load CAS blob {hash}"))
    }

    fn append_wal(&self, record: &WalRecord) -> Result<()> {
        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)
            .with_context(|| format!("open WAL {}", self.wal_path.display()))?;
        file.write_all(line.as_bytes())?;
        file.sync_data()?;
        Ok(())
    }

    pub fn recover_wal(&self) -> Result<WalRecovery> {
        let mut head_rev = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        if !self.wal_path.exists() {
            return Ok(WalRecovery {
                rolled_back_txs: 0,
                committed_modules: HashMap::new(),
                head_rev: 0,
            });
        }

        let file = fs::File::open(&self.wal_path)?;
        let reader = BufReader::new(file);

        let mut prepared: Vec<(String, String, Vec<WalFileEntry>)> = Vec::new();
        let mut committed_txs: HashSet<String> = HashSet::new();
        let mut rolled_back_ids: HashSet<String> = HashSet::new();
        let mut committed_modules: HashMap<String, BTreeMap<String, String>> = HashMap::new();
        let mut max_rev = 0u64;

        for line in reader.lines() {
            let Ok(line) = line else { break };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<WalRecord>(trimmed) else {
                continue;
            };
            match rec {
                WalRecord::Prepared {
                    tx_id,
                    module_id,
                    files,
                    ..
                } => {
                    prepared.push((tx_id, module_id, files));
                }
                WalRecord::Committed {
                    tx_id,
                    module_id,
                    head_rev: rev,
                    files,
                    ..
                } => {
                    committed_txs.insert(tx_id);
                    max_rev = max_rev.max(rev);
                    let mut map = BTreeMap::new();
                    for f in files {
                        if !f.deleted {
                            map.insert(f.path, f.new_blake3);
                        }
                    }
                    committed_modules.insert(module_id, map);
                }
                WalRecord::Rollback { tx_id, .. } => {
                    rolled_back_ids.insert(tx_id);
                }
                WalRecord::RunStart { .. } => {}
            }
        }

        let mut rolled_back_txs = 0usize;
        for (tx_id, module_id, files) in prepared {
            if !committed_txs.contains(&tx_id) && !rolled_back_ids.contains(&tx_id) {
                for f in &files {
                    let abs = self.repo_root.join(&f.path);
                    if f.created {
                        let _ = fs::remove_file(&abs);
                    } else if let Ok(old_bytes) = self.load_cas(&f.old_blake3) {
                        write_atomic(&abs, &old_bytes)?;
                    }
                }
                let _ = self.append_wal(&WalRecord::Rollback {
                    tx_id,
                    module_id,
                    gate: "wal_recovery".to_string(),
                    reason: "restored PREPARED transaction from CAS after crash".to_string(),
                    ts_ms: now_ms(),
                });
                rolled_back_txs += 1;
            }
        }

        *head_rev = max_rev;
        Ok(WalRecovery {
            rolled_back_txs,
            committed_modules,
            head_rev: max_rev,
        })
    }

    #[cfg(test)]
    pub fn commit_edits(
        &self,
        module_id: &str,
        expected_hashes: &BTreeMap<String, String>,
        new_files: &BTreeMap<String, Vec<u8>>,
    ) -> Result<CommitOutcome> {
        self.commit_tx(module_id, expected_hashes, new_files, &[])
    }

    /// Commit file edits, creates, and deletes atomically with an OS WAL lock and rollback guard.
    pub fn commit_tx(
        &self,
        module_id: &str,
        expected_hashes: &BTreeMap<String, String>,
        new_files: &BTreeMap<String, Vec<u8>>,
        deleted_files: &[String],
    ) -> Result<CommitOutcome> {
        let mut rev_guard = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        // Refresh head_rev in case another process committed while we waited on OS lock.
        *rev_guard = (*rev_guard).max(wal_head(&self.wal_path));

        let mut old_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (rel_path, expected) in expected_hashes {
            let abs = self.repo_root.join(rel_path);
            if expected.is_empty() {
                // Empty expected hash enforces CAS-absent check for newly created files.
                if abs.exists() {
                    let actual = fs::read(&abs)
                        .map(|b| Self::hash_bytes(&b))
                        .unwrap_or_else(|_| "exists".into());
                    return Ok(CommitOutcome::CasConflict {
                        path: rel_path.clone(),
                        expected: "<absent>".into(),
                        actual,
                    });
                }
                continue;
            }
            let current = fs::read(&abs)
                .with_context(|| format!("CAS read failed for {}", abs.display()))?;
            let actual = Self::hash_bytes(&current);
            if &actual != expected {
                return Ok(CommitOutcome::CasConflict {
                    path: rel_path.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
            old_blobs.insert(rel_path.clone(), current);
        }

        let mut entries = Vec::with_capacity(new_files.len() + deleted_files.len());
        let mut new_hashes = BTreeMap::new();
        let mut loc_delta: i64 = 0;

        for (rel_path, new_bytes) in new_files {
            let abs = self.repo_root.join(rel_path);
            let (old_hash, old_lines, created) = if let Some(old) = old_blobs.get(rel_path) {
                let l = count_lines(old);
                (self.store_cas(old)?, l, false)
            } else if abs.is_file() {
                let existing = fs::read(&abs)?;
                let l = count_lines(&existing);
                let h = self.store_cas(&existing)?;
                old_blobs.insert(rel_path.clone(), existing);
                (h, l, false)
            } else {
                (self.store_cas(&[])?, 0, true)
            };
            let new_hash = self.store_cas(new_bytes)?;
            let new_lines = count_lines(new_bytes);
            loc_delta += new_lines - old_lines;
            new_hashes.insert(rel_path.clone(), new_hash.clone());
            entries.push(WalFileEntry {
                path: rel_path.clone(),
                old_blake3: old_hash,
                new_blake3: new_hash,
                created,
                deleted: false,
            });
        }

        let mut del_originals: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for rel_path in deleted_files {
            if new_files.contains_key(rel_path) {
                continue;
            }
            let abs = self.repo_root.join(rel_path);
            if let Some(old) = old_blobs.get(rel_path).cloned().or_else(|| fs::read(&abs).ok()) {
                let old_lines = count_lines(&old);
                let old_hash = self.store_cas(&old)?;
                let empty_hash = self.store_cas(&[])?;
                loc_delta -= old_lines;
                del_originals.insert(rel_path.clone(), old);
                entries.push(WalFileEntry {
                    path: rel_path.clone(),
                    old_blake3: old_hash,
                    new_blake3: empty_hash,
                    created: false,
                    deleted: true,
                });
            }
        }

        let tx_id = format!("tx-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));

        self.append_wal(&WalRecord::Prepared {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            ts_ms: now_ms(),
            files: entries.clone(),
        })?;

        let mut written_paths: Vec<(String, bool)> = Vec::with_capacity(new_files.len());
        for (rel_path, new_bytes) in new_files {
            let abs = self.repo_root.join(rel_path);
            let was_created = !old_blobs.contains_key(rel_path);
            if let Err(err) = write_atomic(&abs, new_bytes) {
                for (prev_rel, prev_created) in written_paths.iter().rev() {
                    let prev_abs = self.repo_root.join(prev_rel);
                    if *prev_created {
                        let _ = fs::remove_file(&prev_abs);
                    } else if let Some(orig_bytes) = old_blobs.get(prev_rel) {
                        let _ = write_atomic(&prev_abs, orig_bytes);
                    }
                }
                let _ = self.append_wal(&WalRecord::Rollback {
                    tx_id: tx_id.clone(),
                    module_id: module_id.to_string(),
                    gate: "mvcc_commit_io".to_string(),
                    reason: format!(
                        "auto-restored {} file(s) after write failure on `{rel_path}`: {err}",
                        written_paths.len()
                    ),
                    ts_ms: now_ms(),
                });
                return Err(err);
            }
            written_paths.push((rel_path.clone(), was_created));
        }

        let mut removed_paths: Vec<String> = Vec::new();
        for (rel_path, orig_bytes) in &del_originals {
            let abs = self.repo_root.join(rel_path);
            if abs.exists() {
                if let Err(err) = fs::remove_file(&abs) {
                    // Restore both removed and written files on delete IO failure.
                    for prev_del in removed_paths.iter().rev() {
                        if let Some(b) = del_originals.get(prev_del) {
                            let _ = write_atomic(&self.repo_root.join(prev_del), b);
                        }
                    }
                    let _ = write_atomic(&abs, orig_bytes);
                    for (prev_rel, prev_created) in written_paths.iter().rev() {
                        let prev_abs = self.repo_root.join(prev_rel);
                        if *prev_created {
                            let _ = fs::remove_file(&prev_abs);
                        } else if let Some(orig) = old_blobs.get(prev_rel) {
                            let _ = write_atomic(&prev_abs, orig);
                        }
                    }
                    let _ = self.append_wal(&WalRecord::Rollback {
                        tx_id: tx_id.clone(),
                        module_id: module_id.to_string(),
                        gate: "mvcc_commit_io".to_string(),
                        reason: format!("auto-restored after delete failure on `{rel_path}`: {err}"),
                        ts_ms: now_ms(),
                    });
                    return Err(err.into());
                }
                removed_paths.push(rel_path.clone());
            }
        }

        *rev_guard += 1;
        let head_rev = *rev_guard;
        self.append_wal(&WalRecord::Committed {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            head_rev,
            ts_ms: now_ms(),
            loc_delta: Some(loc_delta),
            files: entries,
        })?;

        Ok(CommitOutcome::Committed {
            head_rev,
            tx_id,
            new_hashes,
        })
    }

    pub fn record_rollback(&self, module_id: &str, gate: &str, reason: &str) -> Result<()> {
        let _flock = self.lock_wal()?;
        let tx_id = format!("rb-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));
        self.append_wal(&WalRecord::Rollback {
            tx_id,
            module_id: module_id.to_string(),
            gate: gate.to_string(),
            reason: reason.chars().take(240).collect(),
            ts_ms: now_ms(),
        })
    }

    pub fn head_rev(&self) -> u64 {
        *self.commit_lock.lock()
    }

    /// Prune CAS blobs not referenced by the last `keep_last_revs` commits or active `PREPARED` records.
    pub fn gc_cas(&self, keep_last_revs: usize) -> Result<usize> {
        let _guard = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        if !self.wal_path.exists() {
            return Ok(0);
        }
        let file = fs::File::open(&self.wal_path)?;
        let reader = BufReader::new(file);
        let mut committed_records: Vec<(u64, Vec<WalFileEntry>)> = Vec::new();
        let mut prepared_hashes: HashSet<String> = HashSet::new();
        let mut committed_ids: HashSet<String> = HashSet::new();
        let mut pending_prepared: Vec<(String, Vec<WalFileEntry>)> = Vec::new();

        for line in reader.lines().map_while(Result::ok) {
            let Ok(rec) = serde_json::from_str::<WalRecord>(line.trim()) else {
                continue;
            };
            match rec {
                WalRecord::Committed {
                    tx_id,
                    head_rev,
                    files,
                    ..
                } => {
                    committed_ids.insert(tx_id);
                    committed_records.push((head_rev, files));
                }
                WalRecord::Prepared { tx_id, files, .. } => {
                    pending_prepared.push((tx_id, files));
                }
                WalRecord::Rollback { tx_id, .. } => {
                    committed_ids.insert(tx_id);
                }
                WalRecord::RunStart { .. } => {}
            }
        }
        for (tx_id, files) in pending_prepared {
            if !committed_ids.contains(&tx_id) {
                for f in files {
                    prepared_hashes.insert(f.old_blake3);
                    prepared_hashes.insert(f.new_blake3);
                }
            }
        }
        committed_records.sort_by_key(|(rev, _)| *rev);
        let skip = committed_records.len().saturating_sub(keep_last_revs);
        let mut retained = prepared_hashes;
        for (_, files) in committed_records.into_iter().skip(skip) {
            for f in files {
                retained.insert(f.old_blake3);
                retained.insert(f.new_blake3);
            }
        }
        let mut pruned = 0usize;
        if let Ok(rd) = fs::read_dir(&self.cas_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_file() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !name.starts_with(".refit.tmp.") && !retained.contains(&name) && fs::remove_file(&p).is_ok() {
                        pruned += 1;
                    }
                }
            }
        }
        Ok(pruned)
    }

    fn storage_paths(repo_root: &Path) -> (PathBuf, PathBuf) {
        let primary_wal = repo_root.join(".refit/wal.jsonl");
        let primary_cas = repo_root.join(".refit/cas");
        if primary_wal.exists() || primary_cas.exists() {
            return (primary_wal, primary_cas);
        }
        let alt_wal = repo_root.join(".refit/mvcc/wal.jsonl");
        let alt_cas = repo_root.join(".refit/mvcc/cas");
        if alt_wal.exists() || alt_cas.exists() {
            return (alt_wal, alt_cas);
        }
        (primary_wal, primary_cas)
    }

    fn read_blob(cas_dir: &Path, hash: &str) -> Option<String> {
        let bytes = fs::read(cas_dir.join(hash)).ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn wal_timeline(repo_root: &Path) -> Result<Vec<WalEvent>> {
        let (wal_path, cas_dir) = Self::storage_paths(repo_root);
        if !wal_path.exists() {
            return Ok(Vec::new());
        }
        let wal_len = fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
        let cache = WAL_TIMELINE_CACHE.get_or_init(DashMap::new);
        if let Some(hit) = cache.get(&wal_path) {
            if hit.0 == wal_len {
                return Ok(hit.1.clone());
            }
        }
        let file = fs::File::open(&wal_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<WalRecord>(trimmed) else {
                continue;
            };
            match rec {
                WalRecord::Committed {
                    tx_id,
                    module_id,
                    head_rev,
                    ts_ms,
                    loc_delta: stored_delta,
                    files,
                } => {
                    let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
                    let loc_delta = if let Some(d) = stored_delta {
                        d
                    } else {
                        let mut d = 0i64;
                        for f in &files {
                            let old_l = Self::read_blob(&cas_dir, &f.old_blake3)
                                .map(|s| s.lines().count() as i64)
                                .unwrap_or(0);
                            let new_l = Self::read_blob(&cas_dir, &f.new_blake3)
                                .map(|s| s.lines().count() as i64)
                                .unwrap_or(0);
                            d += new_l - old_l;
                        }
                        d
                    };
                    entries.push(WalEvent {
                        rev: head_rev,
                        tx_id,
                        module_id,
                        state: "COMMITTED".into(),
                        files: paths,
                        ts_ms,
                        reason: None,
                        loc_delta,
                    });
                }
                WalRecord::Rollback {
                    tx_id,
                    module_id,
                    gate,
                    reason,
                    ts_ms,
                } => {
                    entries.push(WalEvent {
                        rev: 0,
                        tx_id,
                        module_id,
                        state: "ROLLED_BACK".into(),
                        files: Vec::new(),
                        ts_ms,
                        reason: Some(format!("{gate}: {reason}")),
                        loc_delta: 0,
                    });
                }
                WalRecord::RunStart { .. } | WalRecord::Prepared { .. } => {}
            }
        }
        cache.insert(wal_path, (wal_len, entries.clone()));
        Ok(entries)
    }

    pub fn diff_tx(repo_root: &Path, target_tx_id: &str) -> Result<Option<TxDiffReport>> {
        let (wal_path, cas_dir) = Self::storage_paths(repo_root);
        if !wal_path.exists() {
            return Ok(None);
        }
        let file = fs::File::open(&wal_path)?;
        let reader = BufReader::new(file);

        let mut matched: Option<(u64, String, String, Vec<WalFileEntry>)> = None;
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let Ok(rec) = serde_json::from_str::<WalRecord>(line.trim()) else {
                continue;
            };
            match rec {
                WalRecord::Committed {
                    tx_id,
                    module_id,
                    head_rev,
                    files,
                    ..
                } if tx_id == target_tx_id => {
                    matched = Some((head_rev, module_id, "COMMITTED".into(), files));
                }
                WalRecord::Prepared {
                    tx_id,
                    module_id,
                    files,
                    ..
                } if tx_id == target_tx_id && matched.is_none() => {
                    matched = Some((0, module_id, "PREPARED".into(), files));
                }
                _ => {}
            }
        }
        let Some((rev, module_id, state, wal_files)) = matched else {
            return Ok(None);
        };
        let mut diffs = Vec::with_capacity(wal_files.len());
        for wf in wal_files {
            let before = Self::read_blob(&cas_dir, &wf.old_blake3).unwrap_or_default();
            let after = Self::read_blob(&cas_dir, &wf.new_blake3).unwrap_or_default();
            let (added, removed, unified_diff) = make_diff(&wf.path, &before, &after);
            diffs.push(FileDiff {
                path: wf.path,
                pre_hash: wf.old_blake3,
                post_hash: wf.new_blake3,
                added,
                removed,
                unified_diff,
            });
        }
        Ok(Some(TxDiffReport {
            tx_id: target_tx_id.to_string(),
            rev,
            module_id,
            state,
            files: diffs,
        }))
    }
}

fn count_lines(bytes: &[u8]) -> i64 {
    String::from_utf8_lossy(bytes).lines().count() as i64
}

fn wal_head(wal_path: &Path) -> u64 {
    let Ok(file) = fs::File::open(wal_path) else {
        return 0;
    };
    let reader = BufReader::new(file);
    let mut max_rev = 0u64;
    for line in reader.lines().map_while(Result::ok) {
        if let Ok(WalRecord::Committed { head_rev, .. }) = serde_json::from_str::<WalRecord>(line.trim()) {
            max_rev = max_rev.max(head_rev);
        }
    }
    max_rev
}

fn make_diff(path: &str, before: &str, after: &str) -> (usize, usize, String) {
    let diff = TextDiff::from_lines(before, after);
    let mut added = 0usize;
    let mut removed = 0usize;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }
    let unified = diff
        .unified_diff()
        .context_radius(2)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    (added, removed, unified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn probe_35_mvcc_cas() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_cas_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let fpath = tmp.join("mod_a.py");
        fs::write(&fpath, b"def f() -> int:\n    return 1\n").unwrap();
        let real_hash = MvccStore::hash_file(&fpath).unwrap();
        let store = MvccStore::open(&tmp, "cas-probe").unwrap();

        let mut bad_exp = BTreeMap::new();
        bad_exp.insert("mod_a.py".into(), "deadbeef".into());
        let mut new_map = BTreeMap::new();
        new_map.insert("mod_a.py".into(), b"def f() -> int:\n    return 2\n".to_vec());

        match store.commit_edits("mod_a", &bad_exp, &new_map).unwrap() {
            CommitOutcome::CasConflict { path, expected, actual } => {
                assert_eq!(path, "mod_a.py");
                assert_eq!(expected, "deadbeef");
                assert_eq!(actual, real_hash);
            }
            other => panic!("expected CasConflict, got {other:?}"),
        }
        assert_eq!(fs::read(&fpath).unwrap(), b"def f() -> int:\n    return 1\n");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn probe_36_wal_recovery() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_wal_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let fpath = tmp.join("mod_a.py");
        fs::write(&fpath, b"def f() -> int:\n    return 1\n").unwrap();
        let h0 = MvccStore::hash_file(&fpath).unwrap();
        let store = MvccStore::open(&tmp, "wal-probe").unwrap();

        let mut good_exp = BTreeMap::new();
        good_exp.insert("mod_a.py".into(), h0);
        let mut new_map = BTreeMap::new();
        new_map.insert("mod_a.py".into(), b"def f() -> int:\n    return 2\n".to_vec());

        let h1 = match store.commit_edits("mod_a", &good_exp, &new_map).unwrap() {
            CommitOutcome::Committed { head_rev, new_hashes, .. } => {
                assert_eq!(head_rev, 1);
                new_hashes["mod_a.py"].clone()
            }
            other => panic!("expected Committed, got {other:?}"),
        };

        let torn = b"def f() -> int:\n    return 999_TORN\n";
        let torn_hash = store.store_cas(torn).unwrap();
        store
            .append_wal(&WalRecord::Prepared {
                tx_id: "tx-crash-sim".into(),
                module_id: "mod_a".into(),
                ts_ms: now_ms(),
                files: vec![WalFileEntry {
                    path: "mod_a.py".into(),
                    old_blake3: h1.clone(),
                    new_blake3: torn_hash,
                    created: false,
                    deleted: false,
                }],
            })
            .unwrap();
        let mut wal_f = OpenOptions::new().append(true).open(&store.wal_path).unwrap();
        wal_f.write_all(b"{\"kind\":\"committed\",\"tx_id\":\"trunca").unwrap();
        fs::write(&fpath, torn).unwrap();

        let rep = store.recover_wal().unwrap();
        assert_eq!(rep.rolled_back_txs, 1);
        assert_eq!(rep.head_rev, 1);
        assert_eq!(fs::read(&fpath).unwrap(), b"def f() -> int:\n    return 2\n");
        assert_eq!(rep.committed_modules["mod_a"]["mod_a.py"], h1);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn probe_37_mvcc_parallel() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_conc_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let store = Arc::new(MvccStore::open(&tmp, "conc-probe").unwrap());

        for i in 0..8 {
            fs::write(tmp.join(format!("m{i}.py")), format!("X = {i}\n")).unwrap();
        }
        std::thread::scope(|s| {
            for i in 0..8 {
                let st = store.clone();
                let root = tmp.clone();
                s.spawn(move || {
                    let rel = format!("m{i}.py");
                    let h0 = MvccStore::hash_file(&root.join(&rel)).unwrap();
                    let mut exp = BTreeMap::new();
                    exp.insert(rel.clone(), h0);
                    let mut next = BTreeMap::new();
                    next.insert(rel.clone(), format!("X = {}\n", i + 100).into_bytes());
                    let res = st.commit_edits(&format!("m{i}"), &exp, &next).unwrap();
                    assert!(matches!(res, CommitOutcome::Committed { .. }));
                });
            }
        });

        assert_eq!(store.head_rev(), 8);
        for i in 0..8 {
            assert_eq!(fs::read_to_string(tmp.join(format!("m{i}.py"))).unwrap(), format!("X = {}\n", i + 100));
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn probe_62_wal_diff() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_diff_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let fpath = tmp.join("calc.rs");
        fs::write(&fpath, "pub fn calc(x: i32) -> i32 {\n    x + 1\n}\n").unwrap();
        let h0 = MvccStore::hash_file(&fpath).unwrap();
        let store = MvccStore::open(&tmp, "diff-probe").unwrap();

        let mut exp = BTreeMap::new();
        exp.insert("calc.rs".into(), h0);
        let mut next = BTreeMap::new();
        next.insert(
            "calc.rs".into(),
            b"pub fn calc(x: i32) -> i32 {\n    let doubled = x * 2;\n    doubled + 1\n}\n".to_vec(),
        );
        let tx_id = match store.commit_edits("calc", &exp, &next).unwrap() {
            CommitOutcome::Committed { tx_id, .. } => tx_id,
            other => panic!("expected Committed, got {other:?}"),
        };
        store.record_rollback("bad_mod", "gate0_syntax", "missing brace").unwrap();

        let timeline = MvccStore::wal_timeline(&tmp).unwrap();
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].tx_id, tx_id);
        assert_eq!(timeline[0].state, "COMMITTED");
        assert_eq!(timeline[0].loc_delta, 1);
        assert_eq!(timeline[1].state, "ROLLED_BACK");

        let diff = MvccStore::diff_tx(&tmp, &tx_id).unwrap().unwrap();
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].removed, 1);
        assert_eq!(diff.files[0].added, 2);
        assert!(diff.files[0].unified_diff.contains("-    x + 1"));
        assert!(diff.files[0].unified_diff.contains("+    let doubled = x * 2;"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn probe_66_mvcc_rollback() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_rb_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let f1 = tmp.join("a.py");
        fs::write(&f1, b"A = 1\n").unwrap();
        fs::create_dir_all(tmp.join("z_dir")).unwrap();

        let store = MvccStore::open(&tmp, "rb-probe").unwrap();
        let mut exp = BTreeMap::new();
        exp.insert("a.py".into(), MvccStore::hash_file(&f1).unwrap());
        let mut next = BTreeMap::new();
        next.insert("a.py".into(), b"A = 999\n".to_vec());
        next.insert("z_dir".into(), b"impossible".to_vec());

        assert!(store.commit_edits("scc_pair", &exp, &next).is_err());
        assert_eq!(fs::read(&f1).unwrap(), b"A = 1\n");
        let tl = MvccStore::wal_timeline(&tmp).unwrap();
        assert!(tl.iter().any(|e| e.state == "ROLLED_BACK"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn probe_67_diff_hunks() {
        let mut before = String::new();
        let mut after = String::new();
        for i in 1..=20 {
            before.push_str(&format!("line_{i}\n"));
            if i == 2 {
                after.push_str("line_2_modified\n");
            } else if i == 19 {
                after.push_str("line_19_modified\n");
            } else {
                after.push_str(&format!("line_{i}\n"));
            }
        }
        let (added, removed, udiff) = make_diff("demo.rs", &before, &after);
        assert_eq!(added, 2);
        assert_eq!(removed, 2);
        assert_eq!(udiff.matches("@@ ").count(), 2);
    }

    #[test]
    fn probe_88_mvcc_ops_gc() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc88_{}_{}", std::process::id(), now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("old.py"), b"OLD = 1\n").unwrap();

        let s1 = MvccStore::open(&tmp, "r1").unwrap();
        let mut exp = BTreeMap::new();
        exp.insert("old.py".into(), MvccStore::hash_file(&tmp.join("old.py")).unwrap());
        let mut created = BTreeMap::new();
        created.insert("new_mod.py".into(), b"NEW = 2\n".to_vec());
        let r1 = s1
            .commit_tx("split", &exp, &created, &["old.py".into()])
            .unwrap();
        match r1 {
            CommitOutcome::Committed { head_rev, .. } => assert_eq!(head_rev, 1),
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!tmp.join("old.py").exists());
        assert!(tmp.join("new_mod.py").exists());

        // Re-open store (simulating fresh MCP call): head_rev MUST continue monotonically at 2!
        let s2 = MvccStore::open(&tmp, "r2").unwrap();
        assert_eq!(s2.head_rev(), 1);
        let mut exp2 = BTreeMap::new();
        exp2.insert("new_mod.py".into(), MvccStore::hash_file(&tmp.join("new_mod.py")).unwrap());
        let mut upd2 = BTreeMap::new();
        upd2.insert("new_mod.py".into(), b"NEW = 3\n".to_vec());
        match s2.commit_edits("upd", &exp2, &upd2).unwrap() {
            CommitOutcome::Committed { head_rev, .. } => assert_eq!(head_rev, 2),
            other => panic!("expected rev 2, got {other:?}"),
        }

        // CAS GC keeping only last 1 rev should prune rev-1-only blobs.
        let pruned = s2.gc_cas(1).unwrap();
        assert!(pruned >= 1, "expected >=1 pruned CAS blob, got {pruned}");
        let _ = fs::remove_dir_all(&tmp);
    }
}
