mod diff;
mod wal;

pub use wal::{CommitOutcome, TxDiffReport, WalEvent, WalRecovery};

use crate::platform::{EpochRegistry, now_ms, write_atomic};
use crate::source::count_lines;
use crate::types::{WalFileEntry, WalRecord};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use wal::{
    WalHaltError, WalScanStats, append_framed_record, append_jsonl_record, file_uses_frames,
    truncate_torn_tail, visit_wal, visit_wal_stats, wal_head, wal_next_seq,
};

static TX_SEQ: AtomicU64 = AtomicU64::new(1);

pub struct MvccStore {
    repo_root: PathBuf,
    cas_dir: PathBuf,
    wal_path: PathBuf,
    lock_path: PathBuf,
    commit_lock: Mutex<u64>,
    wal_cursor: Mutex<(u64, u64)>,
    epoch: Arc<EpochRegistry>,
}

struct TxBatch<'a> {
    tx_id: &'a str,
    module_id: &'a str,
    new_files: &'a BTreeMap<String, Vec<u8>>,
    old_blobs: &'a BTreeMap<String, Vec<u8>>,
    del_blobs: &'a BTreeMap<String, Vec<u8>>,
}

impl MvccStore {
    pub fn open(repo_root: &Path, run_id: &str) -> Result<Self> {
        Self::open_with_epoch(repo_root, run_id, EpochRegistry::global_arc())
    }

    pub fn open_with_epoch(
        repo_root: &Path,
        run_id: &str,
        epoch: Arc<EpochRegistry>,
    ) -> Result<Self> {
        let refit_dir = repo_root.join(".refit");
        let cas_dir = refit_dir.join("cas");
        fs::create_dir_all(&cas_dir)
            .with_context(|| format!("create CAS dir {}", cas_dir.display()))?;

        let wal_path = refit_dir.join("wal.jsonl");
        let lock_path = refit_dir.join("wal.lock");
        let persisted_rev = wal_head(&wal_path);

        let store = Self {
            repo_root: repo_root.to_path_buf(),
            cas_dir,
            wal_path,
            lock_path,
            commit_lock: Mutex::new(persisted_rev),
            wal_cursor: Mutex::new((0, 0)),
            epoch,
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

    #[cfg(test)]
    pub(crate) fn epoch_registry(&self) -> &Arc<EpochRegistry> {
        &self.epoch
    }

    fn lock_wal(&self) -> Result<crate::platform::SidecarLock> {
        crate::platform::SidecarLock::acquire(&self.lock_path)
            .with_context(|| format!("acquire OS lock {}", self.lock_path.display()))
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

    pub fn load_cas_verified(&self, hash: &str) -> Result<Vec<u8>> {
        let bytes = fs::read(self.cas_dir.join(hash)).map_err(|err| {
            WalHaltError::UnrecoverableRollback(format!("missing CAS blob `{hash}`: {err}"))
        })?;
        let actual = Self::hash_bytes(&bytes);
        if actual != hash {
            return Err(WalHaltError::UnrecoverableRollback(format!(
                "corrupt CAS blob `{hash}` (actual `{actual}`)"
            ))
            .into());
        }
        Ok(bytes)
    }

    fn append_wal(&self, record: &WalRecord) -> Result<()> {
        let payload = serde_json::to_string(record)?;
        let use_framed = cfg!(feature = "wal-crc32c") || file_uses_frames(&self.wal_path);
        if use_framed {
            let mut cursor = self.wal_cursor.lock();
            let disk_len = fs::metadata(&self.wal_path).map(|meta| meta.len()).unwrap_or(0);
            let seq = if disk_len == cursor.0 && cursor.1 > 0 {
                cursor.1
            } else {
                wal_next_seq(&self.wal_path)?
            };
            let new_len = append_framed_record(&self.wal_path, seq, payload.as_bytes())?;
            *cursor = (new_len, seq + 1);
        } else {
            let _ = append_jsonl_record(&self.wal_path, payload.as_bytes())?;
        }
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

        let (prepared, closed, committed_modules, max_rev, stats) = scan_recovery(&self.wal_path)?;
        tracing::debug!(
            valid_records = stats.valid_records,
            file_len = stats.file_len,
            truncated = stats.truncated_bytes,
            has_framed = stats.has_framed,
            "WAL recovery scan completed"
        );
        if stats.truncated_bytes > 0 {
            truncate_torn_tail(&self.wal_path, stats.valid_pos)?;
        }
        *self.wal_cursor.lock() = (stats.valid_pos as u64, stats.next_seq.max(1));

        let mut rolled_back_txs = 0usize;
        let mut recovered_paths = Vec::new();
        for (tx_id, module_id, files) in prepared {
            if closed.contains(&tx_id) {
                continue;
            }
            restore_prepared(self, &files)?;
            recovered_paths.extend(files.into_iter().map(|f| f.path));
            self.append_wal(&WalRecord::Rollback {
                tx_id,
                module_id,
                gate: "wal_recovery".into(),
                reason: "restored PREPARED transaction from CAS after crash".into(),
                ts_ms: now_ms(),
            })?;
            rolled_back_txs += 1;
        }

        if rolled_back_txs > 0 {
            self.epoch.publish_batch("wal_recovery", &recovered_paths, max_rev);
        }
        *head_rev = max_rev;
        Ok(WalRecovery { rolled_back_txs, committed_modules, head_rev: max_rev })
    }

    pub fn commit_tx(
        &self,
        module_id: &str,
        expected_hashes: &BTreeMap<String, String>,
        new_files: &BTreeMap<String, Vec<u8>>,
        deleted_files: &[String],
    ) -> Result<CommitOutcome> {
        let mut rev_guard = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        *rev_guard = (*rev_guard).max(wal_head(&self.wal_path));

        let mut old_blobs = BTreeMap::new();
        if let Some(conflict) = check_cas(&self.repo_root, expected_hashes, &mut old_blobs)? {
            return Ok(conflict);
        }

        let mut entries = Vec::with_capacity(new_files.len() + deleted_files.len());
        let mut new_hashes = BTreeMap::new();
        let mut loc_delta = 0i64;
        stage_writes(self, new_files, &mut old_blobs, &mut entries, &mut new_hashes, &mut loc_delta)?;

        let mut del_blobs = BTreeMap::new();
        stage_deletes(self, new_files, deleted_files, &old_blobs, &mut del_blobs, &mut entries, &mut loc_delta)?;

        let tx_id = format!("tx-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));
        self.append_wal(&WalRecord::Prepared {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            ts_ms: now_ms(),
            files: entries.clone(),
        })?;

        let batch = TxBatch {
            tx_id: &tx_id,
            module_id,
            new_files,
            old_blobs: &old_blobs,
            del_blobs: &del_blobs,
        };
        apply_io(self, &batch)?;

        *rev_guard += 1;
        let head_rev = *rev_guard;
        let committed_paths: Vec<String> = entries.iter().map(|e| e.path.clone()).collect();
        self.append_wal(&WalRecord::Committed {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            head_rev,
            ts_ms: now_ms(),
            loc_delta: Some(loc_delta),
            files: entries,
        })?;
        let _ = self.epoch.publish_batch(module_id, &committed_paths, head_rev);
        Ok(CommitOutcome::Committed { head_rev, tx_id, new_hashes })
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

    pub fn gc_cas(&self, keep_last_revs: usize) -> Result<usize> {
        let _guard = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        if !self.wal_path.exists() {
            return Ok(0);
        }

        let retained = collect_retained(&self.wal_path, keep_last_revs)?;
        Ok(prune_cas(&self.cas_dir, &retained))
    }

    pub fn wal_timeline(repo_root: &Path) -> Result<Vec<WalEvent>> {
        diff::wal_timeline(repo_root)
    }

    pub fn diff_tx(repo_root: &Path, target_tx: &str) -> Result<Option<TxDiffReport>> {
        diff::diff_tx(repo_root, target_tx)
    }
}

type RecoveryScan = (
    Vec<(String, String, Vec<WalFileEntry>)>,
    HashSet<String>,
    HashMap<String, BTreeMap<String, String>>,
    u64,
    WalScanStats,
);

fn scan_recovery(wal_path: &Path) -> Result<RecoveryScan> {
    let mut prepared = Vec::new();
    let mut closed = HashSet::new();
    let mut committed = HashMap::new();
    let mut max_rev = 0u64;

    let stats = visit_wal_stats(wal_path, |rec| match rec {
        WalRecord::Prepared { tx_id, module_id, files, .. } => {
            prepared.push((tx_id, module_id, files));
        }
        WalRecord::Committed { tx_id, module_id, head_rev, files, .. } => {
            closed.insert(tx_id);
            max_rev = max_rev.max(head_rev);
            let map: BTreeMap<String, String> = files
                .into_iter()
                .filter(|f| !f.deleted)
                .map(|f| (f.path, f.new_blake3))
                .collect();
            committed.insert(module_id, map);
        }
        WalRecord::Rollback { tx_id, .. } => {
            closed.insert(tx_id);
        }
        WalRecord::RunStart { .. } => {}
    })?;
    Ok((prepared, closed, committed, max_rev, stats))
}

fn restore_prepared(store: &MvccStore, files: &[WalFileEntry]) -> Result<()> {
    for entry in files {
        let abs = store.repo_root.join(&entry.path);
        if entry.created {
            if let Err(err) = fs::remove_file(&abs)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                return Err(WalHaltError::UnrecoverableRollback(format!(
                    "remove created `{}`: {err}",
                    entry.path
                ))
                .into());
            }
            continue;
        }
        let old_bytes = store.load_cas_verified(&entry.old_blake3)?;
        write_atomic(&abs, &old_bytes).map_err(|err| {
            WalHaltError::UnrecoverableRollback(format!("restore `{}`: {err}", entry.path))
        })?;
    }
    Ok(())
}

fn check_cas(
    repo_root: &Path,
    expected: &BTreeMap<String, String>,
    old_blobs: &mut BTreeMap<String, Vec<u8>>,
) -> Result<Option<CommitOutcome>> {
    for (rel, exp) in expected {
        let abs = repo_root.join(rel);
        if exp.is_empty() && abs.exists() {
            let actual = fs::read(&abs)
                .map_or_else(|_| "exists".into(), |b| MvccStore::hash_bytes(&b));
            return Ok(Some(CommitOutcome::CasConflict {
                path: rel.clone(),
                expected: "<absent>".into(),
                actual,
            }));
        }
        if exp.is_empty() {
            continue;
        }

        let cur = fs::read(&abs)
            .with_context(|| format!("CAS read failed for {}", abs.display()))?;
        let actual = MvccStore::hash_bytes(&cur);
        if &actual != exp {
            return Ok(Some(CommitOutcome::CasConflict {
                path: rel.clone(),
                expected: exp.clone(),
                actual,
            }));
        }
        old_blobs.insert(rel.clone(), cur);
    }
    Ok(None)
}

fn stage_writes(
    store: &MvccStore,
    new_files: &BTreeMap<String, Vec<u8>>,
    old_blobs: &mut BTreeMap<String, Vec<u8>>,
    entries: &mut Vec<WalFileEntry>,
    new_hashes: &mut BTreeMap<String, String>,
    loc_delta: &mut i64,
) -> Result<()> {
    for (rel, bytes) in new_files {
        let abs = store.repo_root.join(rel);
        let (old_hash, old_l, created) = if let Some(old) = old_blobs.get(rel) {
            (store.store_cas(old)?, count_lines(old) as i64, false)
        } else if abs.is_file() {
            let existing = fs::read(&abs)?;
            let l = count_lines(&existing) as i64;
            let h = store.store_cas(&existing)?;
            old_blobs.insert(rel.clone(), existing);
            (h, l, false)
        } else {
            (store.store_cas(&[])?, 0, true)
        };

        let new_hash = store.store_cas(bytes)?;
        *loc_delta += count_lines(bytes) as i64 - old_l;
        new_hashes.insert(rel.clone(), new_hash.clone());
        entries.push(WalFileEntry {
            path: rel.clone(),
            old_blake3: old_hash,
            new_blake3: new_hash,
            created,
            deleted: false,
        });
    }
    Ok(())
}

fn stage_deletes(
    store: &MvccStore,
    new_files: &BTreeMap<String, Vec<u8>>,
    deleted: &[String],
    old_blobs: &BTreeMap<String, Vec<u8>>,
    del_blobs: &mut BTreeMap<String, Vec<u8>>,
    entries: &mut Vec<WalFileEntry>,
    loc_delta: &mut i64,
) -> Result<()> {
    for rel in deleted {
        if new_files.contains_key(rel) {
            continue;
        }
        let Some(old) = old_blobs
            .get(rel)
            .cloned()
            .or_else(|| fs::read(store.repo_root.join(rel)).ok())
        else {
            continue;
        };

        let old_hash = store.store_cas(&old)?;
        let empty_hash = store.store_cas(&[])?;
        *loc_delta -= count_lines(&old) as i64;
        del_blobs.insert(rel.clone(), old);
        entries.push(WalFileEntry {
            path: rel.clone(),
            old_blake3: old_hash,
            new_blake3: empty_hash,
            created: false,
            deleted: true,
        });
    }
    Ok(())
}

fn apply_io(store: &MvccStore, batch: &TxBatch<'_>) -> Result<()> {
    let mut written = Vec::with_capacity(batch.new_files.len());
    for (rel, bytes) in batch.new_files {
        let created = !batch.old_blobs.contains_key(rel);
        if let Err(err) = write_atomic(&store.repo_root.join(rel), bytes) {
            let msg = format!("write failure on `{rel}`: {err}");
            rollback_io(store, batch, &written, &[], msg)?;
            return Err(err);
        }
        written.push((rel.clone(), created));
    }

    let mut removed = Vec::new();
    for rel in batch.del_blobs.keys() {
        let abs = store.repo_root.join(rel);
        if !abs.exists() {
            continue;
        }
        if let Err(err) = fs::remove_file(&abs) {
            let msg = format!("delete failure on `{rel}`: {err}");
            rollback_io(store, batch, &written, &removed, msg)?;
            return Err(err.into());
        }
        removed.push(rel.clone());
    }
    Ok(())
}

fn rollback_written_file(root: &Path, rel: &str, created: bool, orig: Option<&Vec<u8>>) -> Result<()> {
    let abs = root.join(rel);
    if created {
        return match fs::remove_file(&abs) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.into()),
        };
    }
    let Some(bytes) = orig else {
        return Err(WalHaltError::UnrecoverableRollback(format!(
            "missing pre-image in memory for `{rel}`"
        ))
        .into());
    };
    write_atomic(&abs, bytes)
}

fn rollback_io(
    store: &MvccStore,
    batch: &TxBatch<'_>,
    written: &[(String, bool)],
    removed: &[String],
    reason: String,
) -> Result<()> {
    let root = &store.repo_root;
    let mut first_err: Option<anyhow::Error> = None;

    for rel in removed.iter().rev() {
        if let Some(orig) = batch.del_blobs.get(rel)
            && let Err(err) = write_atomic(&root.join(rel), orig)
            && first_err.is_none()
        {
            first_err = Some(err.context(format!("rollback resurrect `{rel}`")));
        }
    }

    for (rel, created) in written.iter().rev() {
        let orig = batch.old_blobs.get(rel);
        if let Err(err) = rollback_written_file(root, rel, *created, orig)
            && first_err.is_none()
        {
            first_err = Some(err.context(format!("rollback restore `{rel}`")));
        }
    }

    if let Some(rb_err) = first_err {
        return Err(WalHaltError::UnrecoverableRollback(format!(
            "rollback I/O failed during `{reason}`: {rb_err}"
        ))
        .into());
    }

    store.append_wal(&WalRecord::Rollback {
        tx_id: batch.tx_id.to_string(),
        module_id: batch.module_id.to_string(),
        gate: "mvcc_commit_io".into(),
        reason,
        ts_ms: now_ms(),
    })?;
    Ok(())
}

fn collect_retained(wal_path: &Path, keep_last_revs: usize) -> Result<HashSet<String>> {
    let (mut committed, mut pending, mut closed) = (Vec::new(), Vec::new(), HashSet::new());
    visit_wal(wal_path, |rec| match rec {
        WalRecord::Committed { tx_id, head_rev, files, .. } => {
            closed.insert(tx_id);
            committed.push((head_rev, files));
        }
        WalRecord::Prepared { tx_id, files, .. } => pending.push((tx_id, files)),
        WalRecord::Rollback { tx_id, .. } => {
            closed.insert(tx_id);
        }
        WalRecord::RunStart { .. } => {}
    })?;

    let mut retained = HashSet::new();
    for (tx_id, files) in pending {
        if !closed.contains(&tx_id) {
            keep_hashes(&files, &mut retained);
        }
    }

    committed.sort_by_key(|(rev, _)| *rev);
    let skip = committed.len().saturating_sub(keep_last_revs);
    for (_, files) in committed.into_iter().skip(skip) {
        keep_hashes(&files, &mut retained);
    }
    Ok(retained)
}

fn keep_hashes(files: &[WalFileEntry], retained: &mut HashSet<String>) {
    for f in files {
        retained.insert(f.old_blake3.clone());
        retained.insert(f.new_blake3.clone());
    }
}

fn prune_cas(cas_dir: &Path, retained: &HashSet<String>) -> usize {
    let Ok(rd) = fs::read_dir(cas_dir) else { return 0 };
    let mut pruned = 0usize;
    for entry in rd.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if p.is_file() && !name.starts_with(".refit.tmp.") && !retained.contains(&name) {
            pruned += usize::from(fs::remove_file(&p).is_ok());
        }
    }
    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TempRepo, commit_file, map1};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;

    const MOD_A: &str = "def f() -> int:\n    return 1\n";

    fn open_repo(tag: &str, run: &str, files: &[(&str, &str)]) -> (TempRepo, MvccStore) {
        let dir = TempRepo::with_files(tag, files);
        let store = MvccStore::open(dir.path(), run).unwrap();
        (dir, store)
    }

    #[test]
    fn test_mvcc_cas_recovery_and_rollback() {
        let (dir, store) = open_repo("mvcc_cas", "cas-probe", &[("mod_a.py", MOD_A)]);
        let real_hash = dir.hash("mod_a.py");
        let next = "def f() -> int:\n    return 2\n";

        match store.commit_tx("mod_a", &map1("mod_a.py", "deadbeef".into()), &map1("mod_a.py", next.as_bytes().to_vec()), &[]).unwrap() {
            CommitOutcome::CasConflict { path, expected, actual } => {
                assert_eq!((path.as_str(), expected.as_str(), actual), ("mod_a.py", "deadbeef", real_hash));
            }
            other => panic!("expected CasConflict, got {other:?}"),
        }
        assert_eq!(dir.read("mod_a.py"), MOD_A);

        let (rev, _, h1) = commit_file(&store, dir.path(), "mod_a", "mod_a.py", next);
        assert_eq!(rev, 1);

        let torn = b"def f() -> int:\n    return 999_TORN\n";
        let entry = WalFileEntry {
            path: "mod_a.py".into(), old_blake3: h1.clone(),
            new_blake3: store.store_cas(torn).unwrap(), created: false, deleted: false,
        };
        store.append_wal(&WalRecord::Prepared {
            tx_id: "tx-crash-sim".into(), module_id: "mod_a".into(), ts_ms: now_ms(), files: vec![entry],
        }).unwrap();
        let mut opts = OpenOptions::new();
        let mut wal_f = opts.append(true).open(&store.wal_path).unwrap();
        wal_f.write_all(b"{\"kind\":\"committed\",\"tx_id\":\"trunca").unwrap();
        dir.write("mod_a.py", torn);

        let rep = store.recover_wal().unwrap();
        assert_eq!((rep.rolled_back_txs, rep.head_rev), (1, 1));
        assert_eq!((dir.read("mod_a.py").as_str(), rep.committed_modules["mod_a"]["mod_a.py"].as_str()), (next, h1.as_str()));

        fs::create_dir_all(dir.join("z_dir")).unwrap();
        let bad_io = BTreeMap::from([("mod_a.py".into(), b"A = 999\n".to_vec()), ("z_dir".into(), b"impossible".to_vec())]);
        assert!(store.commit_tx("scc_pair", &map1("mod_a.py", h1), &bad_io, &[]).is_err());
        assert_eq!(dir.read("mod_a.py"), next);
        let tl = MvccStore::wal_timeline(dir.path()).unwrap();
        assert!(tl.iter().any(|e| e.state == "ROLLED_BACK"));
    }

    #[test]
    fn test_mvcc_parallel_commits() {
        let (dir, raw) = open_repo("mvcc_par", "conc-probe", &[]);
        let store = Arc::new(raw);
        for i in 0..8 {
            dir.write(&format!("m{i}.py"), format!("X = {i}\n"));
        }

        std::thread::scope(|s| {
            for i in 0..8 {
                let (st, root) = (store.clone(), dir.path().to_path_buf());
                s.spawn(move || {
                    let _ = commit_file(&st, &root, &format!("m{i}"), &format!("m{i}.py"), format!("X = {}\n", i + 100));
                });
            }
        });

        assert_eq!(store.head_rev(), 8);
        for i in 0..8 {
            assert_eq!(dir.read(&format!("m{i}.py")), format!("X = {}\n", i + 100));
        }
    }

    #[test]
    fn test_wal_timeline_and_diff_hunks() {
        let src = "pub fn calc(x: i32) -> i32 {\n    x + 1\n}\n";
        let (dir, store) = open_repo("mvcc_diff", "diff-probe", &[("calc.rs", src)]);
        let body = "pub fn calc(x: i32) -> i32 {\n    let doubled = x * 2;\n    doubled + 1\n}\n";
        let (_, tx_id, _) = commit_file(&store, dir.path(), "calc", "calc.rs", body);
        store.record_rollback("bad_mod", "gate0_syntax", "missing brace").unwrap();

        let tl = MvccStore::wal_timeline(dir.path()).unwrap();
        assert_eq!((tl.len(), tl[0].tx_id.as_str(), tl[0].loc_delta), (2, tx_id.as_str(), 1));
        assert_eq!((tl[0].state.as_str(), tl[1].state.as_str()), ("COMMITTED", "ROLLED_BACK"));

        let diff = MvccStore::diff_tx(dir.path(), &tx_id).unwrap().unwrap();
        assert_eq!((diff.files.len(), diff.files[0].removed, diff.files[0].added), (1, 1, 2));
        assert!(diff.files[0].unified_diff.contains("+    let doubled = x * 2;"));

        let (mut before, mut after) = (String::new(), String::new());
        for i in 1..=20 {
            before.push_str(&format!("line_{i}\n"));
            after.push_str(&match i {
                2 => "line_2_modified\n".into(),
                19 => "line_19_modified\n".into(),
                _ => format!("line_{i}\n"),
            });
        }
        let (added, removed, udiff) = diff::make_diff("demo.rs", &before, &after);
        assert_eq!((added, removed, udiff.matches("@@ ").count()), (2, 2, 2));
    }

    #[test]
    fn test_mvcc_create_delete_and_gc() {
        let (dir, s1) = open_repo("mvcc_gc", "r1", &[("old.py", "OLD = 1\n")]);
        let r1 = s1.commit_tx("split", &map1("old.py", dir.hash("old.py")), &map1("new_mod.py", b"NEW = 2\n".to_vec()), &["old.py".into()]).unwrap();
        assert!(matches!(r1, CommitOutcome::Committed { head_rev: 1, .. }));
        assert!(!dir.has("old.py") && dir.has("new_mod.py"));

        let s2 = MvccStore::open(dir.path(), "r2").unwrap();
        assert_eq!(s2.head_rev(), 1);
        assert_eq!(commit_file(&s2, dir.path(), "upd", "new_mod.py", "NEW = 3\n").0, 2);
        assert!(s2.gc_cas(1).unwrap() >= 1);
    }

    #[test]
    fn wal_crash_points_7_of_7() {
        // 0. RFC 3720 Castagnoli CRC32C verification (slice-by-8 across short and long buffers).
        assert_eq!(wal::crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(wal::crc32c(b""), 0x0000_0000);
        let long_buf: Vec<u8> = (0..137u8).collect();
        let expected_long = {
            let mut crc = 0xFFFF_FFFFu32;
            for &b in &long_buf {
                crc ^= b as u32;
                for _ in 0..8 {
                    crc = if (crc & 1) != 0 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 };
                }
            }
            !crc
        };
        assert_eq!(wal::crc32c(&long_buf), expected_long);

        // CP-1: Crash after store_cas writes blobs, before Prepared WAL append.
        {
            let (dir, store) = open_repo("cp1_cas_only", "cp1", &[("a.rs", "pub fn v() -> i32 { 1 }\n")]);
            let pre_hash = dir.hash("a.rs");
            let _orphan = store.store_cas(b"pub fn v() -> i32 { 2 }\n").unwrap();
            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (0, 0));
            assert_eq!(dir.hash("a.rs"), pre_hash);
            assert!(store.gc_cas(1).unwrap() >= 1);
        }

        // CP-2: Crash mid-write of Prepared record (truncated partial tail) + all-zero EOF sentinel.
        {
            let (dir, store) = open_repo("cp2_torn_prep", "cp2", &[("a.rs", "pub fn v() -> i32 { 1 }\n")]);
            let pre_hash = dir.hash("a.rs");
            let prep = WalRecord::Prepared {
                tx_id: "tx-torn-prep".into(),
                module_id: "a".into(),
                ts_ms: now_ms(),
                files: vec![],
            };
            let raw_frame = wal::encode_wal_frame(2, serde_json::to_string(&prep).unwrap().as_bytes()).unwrap();
            {
                let mut f = OpenOptions::new().append(true).open(&store.wal_path).unwrap();
                f.write_all(&raw_frame[..13]).unwrap();
            }
            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (0, 0));
            assert_eq!(dir.hash("a.rs"), pre_hash);
            // All-zero EOF sentinel auto-truncates cleanly.
            {
                let mut f = OpenOptions::new().append(true).open(&store.wal_path).unwrap();
                f.write_all(&[0u8; 64]).unwrap();
            }
            let rec_z = store.recover_wal().unwrap();
            assert_eq!(rec_z.rolled_back_txs, 0);
            // Subsequent commit heals cleanly and advances head_rev.
            let (rev, _, _) = commit_file(&store, dir.path(), "a", "a.rs", "pub fn v() -> i32 { 2 }\n");
            assert_eq!(rev, 1);
        }

        // CP-3: Crash after Prepared fsynced, before apply_io mutates any file.
        {
            let (dir, store) = open_repo("cp3_prep_only", "cp3", &[("a.rs", "let x = 1;\n")]);
            let h_old = store.store_cas(b"let x = 1;\n").unwrap();
            let h_new = store.store_cas(b"let x = 2;\n").unwrap();
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-cp3".into(),
                module_id: "a".into(),
                ts_ms: now_ms(),
                files: vec![WalFileEntry {
                    path: "a.rs".into(),
                    old_blake3: h_old.clone(),
                    new_blake3: h_new,
                    created: false,
                    deleted: false,
                }],
            }).unwrap();
            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (1, 0));
            assert_eq!(dir.hash("a.rs"), h_old);
        }

        // CP-4: Crash mid-apply_io in a 4-file batch (2 edits + 1 create + 1 delete).
        {
            let (dir, store) = open_repo(
                "cp4_mid_batch",
                "cp4",
                &[("e1.rs", "E1 = 1\n"), ("e2.rs", "E2 = 1\n"), ("del.rs", "DEL = 1\n")],
            );
            let h_e1 = store.store_cas(b"E1 = 1\n").unwrap();
            let h_e2 = store.store_cas(b"E2 = 1\n").unwrap();
            let h_del = store.store_cas(b"DEL = 1\n").unwrap();
            let h_empty = store.store_cas(b"").unwrap();
            let h_new = store.store_cas(b"NEW = 2\n").unwrap();
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-cp4".into(),
                module_id: "batch4".into(),
                ts_ms: now_ms(),
                files: vec![
                    WalFileEntry { path: "e1.rs".into(), old_blake3: h_e1.clone(), new_blake3: h_new.clone(), created: false, deleted: false },
                    WalFileEntry { path: "e2.rs".into(), old_blake3: h_e2.clone(), new_blake3: h_new.clone(), created: false, deleted: false },
                    WalFileEntry { path: "created.rs".into(), old_blake3: h_empty.clone(), new_blake3: h_new, created: true, deleted: false },
                    WalFileEntry { path: "del.rs".into(), old_blake3: h_del.clone(), new_blake3: h_empty, created: false, deleted: true },
                ],
            }).unwrap();
            // Simulate partial disk mutation: e1 written, created written, e2 & del untouched.
            dir.write("e1.rs", "E1 = CORRUPT\n");
            dir.write("created.rs", "CREATED = 99\n");

            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (1, 0));
            assert_eq!(dir.hash("e1.rs"), h_e1);
            assert_eq!(dir.hash("e2.rs"), h_e2);
            assert_eq!(dir.hash("del.rs"), h_del);
            assert!(!dir.has("created.rs"));
        }

        // CP-5: Crash after ALL apply_io writes and deletes succeed, before Committed WAL append.
        {
            let (dir, store) = open_repo("cp5_pre_commit", "cp5", &[("keep.rs", "K=1\n"), ("gone.rs", "G=1\n")]);
            let h_k = store.store_cas(b"K=1\n").unwrap();
            let h_g = store.store_cas(b"G=1\n").unwrap();
            let h_e = store.store_cas(b"").unwrap();
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-cp5".into(),
                module_id: "m5".into(),
                ts_ms: now_ms(),
                files: vec![
                    WalFileEntry { path: "keep.rs".into(), old_blake3: h_k.clone(), new_blake3: store.store_cas(b"K=2\n").unwrap(), created: false, deleted: false },
                    WalFileEntry { path: "gone.rs".into(), old_blake3: h_g.clone(), new_blake3: h_e, created: false, deleted: true },
                ],
            }).unwrap();
            dir.write("keep.rs", "K=2\n");
            fs::remove_file(dir.join("gone.rs")).unwrap();

            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (1, 0));
            assert_eq!(dir.hash("keep.rs"), h_k);
            assert_eq!(dir.hash("gone.rs"), h_g);
        }

        // CP-6: Crash mid-write of Committed record (truncated tail after Prepared + full disk writes).
        {
            let (dir, store) = open_repo("cp6_torn_commit", "cp6", &[("m.rs", "M=1\n")]);
            let h_old = store.store_cas(b"M=1\n").unwrap();
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-cp6".into(),
                module_id: "m6".into(),
                ts_ms: now_ms(),
                files: vec![WalFileEntry {
                    path: "m.rs".into(),
                    old_blake3: h_old.clone(),
                    new_blake3: store.store_cas(b"M=2\n").unwrap(),
                    created: false,
                    deleted: false,
                }],
            }).unwrap();
            dir.write("m.rs", "M=2\n");
            {
                let mut f = OpenOptions::new().append(true).open(&store.wal_path).unwrap();
                f.write_all(b"{\"kind\":\"committed\",\"tx_id\":\"tx-cp6\"").unwrap();
            }
            let rec = store.recover_wal().unwrap();
            assert_eq!((rec.rolled_back_txs, rec.head_rev), (1, 0));
            assert_eq!(dir.hash("m.rs"), h_old);
            assert_eq!(commit_file(&store, dir.path(), "m6", "m.rs", "M=3\n").0, 1);
        }

        // CP-7: Double-crash during recover_wal + leftover .refit.tmp.* files -> idempotent recovery.
        {
            let (dir, store) = open_repo("cp7_double_crash", "cp7", &[("a.rs", "A=1\n"), ("b.rs", "B=1\n")]);
            let h_a = store.store_cas(b"A=1\n").unwrap();
            let h_b = store.store_cas(b"B=1\n").unwrap();
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-cp7".into(),
                module_id: "m7".into(),
                ts_ms: now_ms(),
                files: vec![
                    WalFileEntry { path: "a.rs".into(), old_blake3: h_a.clone(), new_blake3: store.store_cas(b"A=2\n").unwrap(), created: false, deleted: false },
                    WalFileEntry { path: "b.rs".into(), old_blake3: h_b.clone(), new_blake3: store.store_cas(b"B=2\n").unwrap(), created: false, deleted: false },
                ],
            }).unwrap();
            // Simulate partial restore during crashed first recovery + leftover .refit.tmp.* files.
            dir.write("a.rs", "A=1\n");
            dir.write("b.rs", "B=2\n");
            dir.write(".refit.tmp.99999.1.b.rs", "LEFTOVER");
            fs::write(store.cas_dir.join(".refit.tmp.99999.2.blob"), b"LEFTOVER").unwrap();

            let r1 = store.recover_wal().unwrap();
            assert_eq!(r1.rolled_back_txs, 1);
            assert_eq!((dir.hash("a.rs"), dir.hash("b.rs")), (h_a, h_b));
            let r2 = store.recover_wal().unwrap();
            assert_eq!(r2.rolled_back_txs, 0);
            assert_eq!(commit_file(&store, dir.path(), "m7", "a.rs", "A=3\n").0, 1);
        }

        // Strict PM-V7 clean-halt vectors: mid-file 1-bit flip, line fusion, seq gap, missing/corrupt CAS, legacy upgrade.
        {
            let dir = TempRepo::new("pmv7_clean_halt");
            let wal = dir.join("framed.wal");
            let r1 = serde_json::to_vec(&WalRecord::RunStart { run_id: "r".into(), ts_ms: 1, repo_root: ".".into() }).unwrap();
            let r2 = serde_json::to_vec(&WalRecord::Rollback { tx_id: "t1".into(), module_id: "m".into(), gate: "g".into(), reason: "ok".into(), ts_ms: 2 }).unwrap();
            let r3 = serde_json::to_vec(&WalRecord::Rollback { tx_id: "t2".into(), module_id: "m".into(), gate: "g".into(), reason: "ok".into(), ts_ms: 3 }).unwrap();
            wal::append_framed_record(&wal, 1, &r1).unwrap();
            wal::append_framed_record(&wal, 2, &r2).unwrap();
            wal::append_framed_record(&wal, 3, &r3).unwrap();

            // Mid-file 1-bit flip -> Clean-Halt MidFileCorruption.
            let mut flipped = fs::read(&wal).unwrap();
            let mid = flipped.len() / 2;
            flipped[mid] ^= 0x01;
            assert!(matches!(
                wal::scan_wal_bytes(&flipped, &mut |_| {}),
                Err(WalHaltError::MidFileCorruption(_))
            ));

            // Frame line fusion (truncated frame #2 directly glued to valid frame #3) -> Clean-Halt.
            let f1 = wal::encode_wal_frame(1, &r1).unwrap();
            let f2 = wal::encode_wal_frame(2, &r2).unwrap();
            let f3 = wal::encode_wal_frame(3, &r3).unwrap();
            let mut fused = Vec::new();
            fused.extend_from_slice(&f1);
            fused.extend_from_slice(&f2[..f2.len() - 4]);
            fused.extend_from_slice(&f3);
            assert!(matches!(
                wal::scan_wal_bytes(&fused, &mut |_| {}),
                Err(WalHaltError::MidFileCorruption(_))
            ));

            // Monotonic sequence gap (1 -> 3) -> Clean-Halt.
            let mut gapped = Vec::new();
            gapped.extend_from_slice(&f1);
            gapped.extend_from_slice(&f3);
            assert!(matches!(
                wal::scan_wal_bytes(&gapped, &mut |_| {}),
                Err(WalHaltError::MidFileCorruption(_))
            ));

            // Legacy JSONL seamlessly upgraded to RFWL framed records.
            let mut upgraded = Vec::new();
            upgraded.extend_from_slice(&r1);
            upgraded.push(b'\n');
            upgraded.extend_from_slice(&f2);
            upgraded.extend_from_slice(&f3);
            let mut count = 0usize;
            let stats = wal::scan_wal_bytes(&upgraded, &mut |_| count += 1).unwrap();
            assert_eq!((count, stats.valid_records, stats.next_seq, stats.has_framed), (3, 3, 4, true));
        }

        // Missing or corrupt pre-image CAS blob -> Clean-Halt UnrecoverableRollback (NEVER fake Rollback!).
        {
            let (dir, store) = open_repo("pmv7_bad_cas", "cas", &[("lost.rs", "ORIG\n")]);
            store.append_wal(&WalRecord::Prepared {
                tx_id: "tx-missing-cas".into(),
                module_id: "lost".into(),
                ts_ms: now_ms(),
                files: vec![WalFileEntry {
                    path: "lost.rs".into(),
                    old_blake3: "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000".into(),
                    new_blake3: store.store_cas(b"NEW\n").unwrap(),
                    created: false,
                    deleted: false,
                }],
            }).unwrap();
            let err = store.recover_wal().unwrap_err();
            assert!(matches!(
                err.downcast_ref::<WalHaltError>(),
                Some(WalHaltError::UnrecoverableRollback(_))
            ));
            let tl = MvccStore::wal_timeline(dir.path()).unwrap();
            assert!(tl.is_empty(), "must NEVER append fake Rollback on missing CAS: {tl:?}");

            // Now place a corrupt CAS blob at the expected hash -> still Clean-Halt!
            fs::write(
                store.cas_dir.join("deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000"),
                b"CORRUPTED_BYTES",
            ).unwrap();
            let err2 = store.recover_wal().unwrap_err();
            assert!(matches!(
                err2.downcast_ref::<WalHaltError>(),
                Some(WalHaltError::UnrecoverableRollback(_))
            ));
        }
    }
}
