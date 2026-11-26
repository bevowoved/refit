mod diff;
mod wal;

pub use wal::{CommitOutcome, TxDiffReport, WalEvent, WalRecovery};

use crate::platform::{EpochRegistry, SidecarLock, now_ms, write_atomic};
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

    pub fn open_with_epoch(repo_root: &Path, run_id: &str, epoch: Arc<EpochRegistry>) -> Result<Self> {
        let refit_dir = repo_root.join(".refit");
        let cas_dir = refit_dir.join("cas");
        fs::create_dir_all(&cas_dir).with_context(|| format!("create CAS dir {}", cas_dir.display()))?;

        let (wal_path, lock_path) = (refit_dir.join("wal.jsonl"), refit_dir.join("wal.lock"));
        let persisted_rev = wal_head(&wal_path);
        let store = Self {
            repo_root: repo_root.to_path_buf(), cas_dir, wal_path, lock_path,
            commit_lock: Mutex::new(persisted_rev), wal_cursor: Mutex::new((0, 0)), epoch,
        };
        if !store.wal_path.exists() {
            let _flock = store.lock_wal()?;
            if !store.wal_path.exists() {
                store.append_wal(&WalRecord::RunStart {
                    run_id: run_id.to_string(), ts_ms: now_ms(), repo_root: repo_root.display().to_string(),
                })?;
            }
        }
        Ok(store)
    }

    #[cfg(test)]
    pub(crate) fn epoch_registry(&self) -> &Arc<EpochRegistry> { &self.epoch }

    fn lock_wal(&self) -> Result<SidecarLock> {
        SidecarLock::acquire(&self.lock_path)
            .with_context(|| format!("acquire OS lock {}", self.lock_path.display()))
    }

    pub fn hash_bytes(bytes: &[u8]) -> String { blake3::hash(bytes).to_hex().to_string() }

    pub fn hash_file(path: &Path) -> Result<String> {
        Ok(Self::hash_bytes(&fs::read(path).with_context(|| format!("read {}", path.display()))?))
    }

    pub fn store_cas(&self, bytes: &[u8]) -> Result<String> {
        let hex = Self::hash_bytes(bytes);
        let obj = self.cas_dir.join(&hex);
        if !obj.exists() { write_atomic(&obj, bytes)?; }
        Ok(hex)
    }

    pub fn load_cas_verified(&self, hash: &str) -> Result<Vec<u8>> {
        let bytes = fs::read(self.cas_dir.join(hash)).map_err(|err| {
            WalHaltError::UnrecoverableRollback(format!("missing CAS blob `{hash}`: {err}"))
        })?;
        let actual = Self::hash_bytes(&bytes);
        if actual != hash {
            return Err(WalHaltError::UnrecoverableRollback(
                format!("corrupt CAS blob `{hash}` (actual `{actual}`)"),
            ).into());
        }
        Ok(bytes)
    }

    fn append_wal(&self, record: &WalRecord) -> Result<()> {
        let payload = serde_json::to_string(record)?;
        if cfg!(feature = "wal-crc32c") || file_uses_frames(&self.wal_path) {
            let mut cursor = self.wal_cursor.lock();
            let disk_len = fs::metadata(&self.wal_path).map_or(0, |meta| meta.len());
            let seq = if disk_len == cursor.0 && cursor.1 > 0 {
                cursor.1
            } else { wal_next_seq(&self.wal_path)? };
            let new_len = append_framed_record(&self.wal_path, seq, payload.as_bytes())?;
            *cursor = (new_len, seq + 1);
        } else {
            let _ = append_jsonl_record(&self.wal_path, payload.as_bytes())?;
        }
        Ok(())
    }

    pub fn recover_wal(&self) -> Result<WalRecovery> { recover_store(self) }

    pub fn commit_tx(
        &self, module_id: &str, expected_hashes: &BTreeMap<String, String>,
        new_files: &BTreeMap<String, Vec<u8>>, deleted_files: &[String],
    ) -> Result<CommitOutcome> {
        execute_commit(self, module_id, expected_hashes, new_files, deleted_files)
    }

    pub fn record_rollback(&self, module_id: &str, gate: &str, reason: &str) -> Result<()> {
        let _flock = self.lock_wal()?;
        let tx_id = format!("rb-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));
        self.append_wal(&rollback_record(tx_id, module_id.into(), gate, reason.chars().take(240).collect()))
    }

    pub fn head_rev(&self) -> u64 { *self.commit_lock.lock() }

    pub fn gc_cas(&self, keep_last_revs: usize) -> Result<usize> {
        let _guard = self.commit_lock.lock();
        let _flock = self.lock_wal()?;
        if !self.wal_path.exists() { return Ok(0); }
        Ok(prune_cas(&self.cas_dir, &collect_retained(&self.wal_path, keep_last_revs)?))
    }

    pub fn wal_timeline(repo_root: &Path) -> Result<Vec<WalEvent>> { diff::wal_timeline(repo_root) }
    pub fn diff_tx(repo_root: &Path, target_tx: &str) -> Result<Option<TxDiffReport>> {
        diff::diff_tx(repo_root, target_tx)
    }
}

fn rollback_record(tx_id: String, module_id: String, gate: &str, reason: String) -> WalRecord {
    WalRecord::Rollback { tx_id, module_id, gate: gate.into(), reason, ts_ms: now_ms() }
}

fn wal_file_entry(
    path: String, old_blake3: String, new_blake3: String, created: bool, deleted: bool,
) -> WalFileEntry {
    WalFileEntry { path, old_blake3, new_blake3, created, deleted }
}

fn remove_if_exists(abs: &Path) -> std::io::Result<()> {
    match fs::remove_file(abs) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn rollback_unclosed_txs(
    store: &MvccStore, prepared: Vec<(String, String, Vec<WalFileEntry>)>,
    closed: &HashSet<String>, max_rev: u64,
) -> Result<usize> {
    let (mut rolled_back, mut paths) = (0usize, Vec::new());
    for (tx_id, module_id, files) in prepared {
        if closed.contains(&tx_id) { continue; }
        restore_prepared(store, &files)?;
        paths.extend(files.into_iter().map(|entry| entry.path));
        let reason = "restored PREPARED transaction from CAS after crash".into();
        store.append_wal(&rollback_record(tx_id, module_id, "wal_recovery", reason))?;
        rolled_back += 1;
    }
    if rolled_back > 0 { store.epoch.publish_batch("wal_recovery", &paths, max_rev); }
    Ok(rolled_back)
}

fn recover_store(store: &MvccStore) -> Result<WalRecovery> {
    let mut head_rev = store.commit_lock.lock();
    let _flock = store.lock_wal()?;
    if !store.wal_path.exists() {
        return Ok(WalRecovery { rolled_back_txs: 0, committed_modules: HashMap::new(), head_rev: 0 });
    }
    let (prepared, closed, committed_modules, max_rev, stats) = scan_recovery(&store.wal_path)?;
    tracing::debug!(
        valid_records = stats.valid_records, file_len = stats.file_len,
        truncated = stats.truncated_bytes, has_framed = stats.has_framed, "WAL recovery scan completed"
    );
    if stats.truncated_bytes > 0 { truncate_torn_tail(&store.wal_path, stats.valid_pos)?; }
    *store.wal_cursor.lock() = (stats.valid_pos as u64, stats.next_seq.max(1));
    let rolled_back_txs = rollback_unclosed_txs(store, prepared, &closed, max_rev)?;
    *head_rev = max_rev;
    Ok(WalRecovery { rolled_back_txs, committed_modules, head_rev: max_rev })
}

fn execute_commit(
    store: &MvccStore, module_id: &str, expected_hashes: &BTreeMap<String, String>,
    new_files: &BTreeMap<String, Vec<u8>>, deleted_files: &[String],
) -> Result<CommitOutcome> {
    let mut rev_guard = store.commit_lock.lock();
    let _flock = store.lock_wal()?;
    *rev_guard = (*rev_guard).max(wal_head(&store.wal_path));

    let mut old_blobs = BTreeMap::new();
    if let Some(conflict) = check_cas(&store.repo_root, expected_hashes, &mut old_blobs)? {
        return Ok(conflict);
    }

    let (entries, new_hashes, del_blobs, loc_delta) =
        stage_tx_entries(store, new_files, deleted_files, &mut old_blobs)?;

    let tx_id = format!("tx-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));
    store.append_wal(&WalRecord::Prepared {
        tx_id: tx_id.clone(), module_id: module_id.into(), ts_ms: now_ms(), files: entries.clone(),
    })?;

    let batch = TxBatch {
        tx_id: &tx_id, module_id, new_files, old_blobs: &old_blobs, del_blobs: &del_blobs,
    };
    apply_io(store, &batch)?;

    *rev_guard += 1;
    let head_rev = *rev_guard;
    let committed_paths: Vec<String> = entries.iter().map(|entry| entry.path.clone()).collect();
    store.append_wal(&WalRecord::Committed {
        tx_id: tx_id.clone(), module_id: module_id.into(), head_rev, ts_ms: now_ms(),
        loc_delta: Some(loc_delta), files: entries,
    })?;
    let _ = store.epoch.publish_batch(module_id, &committed_paths, head_rev);
    Ok(CommitOutcome::Committed { head_rev, tx_id, new_hashes })
}

type RecoveryScan = (
    Vec<(String, String, Vec<WalFileEntry>)>,
    HashSet<String>,
    HashMap<String, BTreeMap<String, String>>,
    u64,
    WalScanStats,
);

fn scan_recovery(wal_path: &Path) -> Result<RecoveryScan> {
    let (mut prepared, mut closed, mut committed, mut max_rev) =
        (Vec::new(), HashSet::new(), HashMap::new(), 0u64);
    let stats = visit_wal_stats(wal_path, |rec| match rec {
        WalRecord::Prepared { tx_id, module_id, files, .. } => prepared.push((tx_id, module_id, files)),
        WalRecord::Committed { tx_id, module_id, head_rev, files, .. } => {
            closed.insert(tx_id);
            max_rev = max_rev.max(head_rev);
            let map = files.into_iter().filter(|ent| !ent.deleted)
                .map(|ent| (ent.path, ent.new_blake3)).collect();
            committed.insert(module_id, map);
        }
        WalRecord::Rollback { tx_id, .. } => { closed.insert(tx_id); }
        WalRecord::RunStart { .. } => {}
    })?;
    Ok((prepared, closed, committed, max_rev, stats))
}

fn restore_prepared(store: &MvccStore, files: &[WalFileEntry]) -> Result<()> {
    for entry in files {
        let abs = store.repo_root.join(&entry.path);
        if entry.created {
            remove_if_exists(&abs).map_err(|err| {
                WalHaltError::UnrecoverableRollback(format!("remove created `{}`: {err}", entry.path))
            })?;
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
    repo_root: &Path, expected: &BTreeMap<String, String>, old_blobs: &mut BTreeMap<String, Vec<u8>>,
) -> Result<Option<CommitOutcome>> {
    for (rel, exp) in expected {
        let abs = repo_root.join(rel);
        if exp.is_empty() {
            if abs.exists() {
                let actual = fs::read(&abs)
                    .map_or_else(|_| "exists".into(), |raw| MvccStore::hash_bytes(&raw));
                return Ok(Some(CommitOutcome::CasConflict {
                    path: rel.clone(), expected: "<absent>".into(), actual,
                }));
            }
            continue;
        }
        let cur = fs::read(&abs).with_context(|| format!("CAS read failed for {}", abs.display()))?;
        let actual = MvccStore::hash_bytes(&cur);
        if &actual != exp {
            return Ok(Some(CommitOutcome::CasConflict { path: rel.clone(), expected: exp.clone(), actual }));
        }
        old_blobs.insert(rel.clone(), cur);
    }
    Ok(None)
}

type StagedTx = (Vec<WalFileEntry>, BTreeMap<String, String>, BTreeMap<String, Vec<u8>>, i64);

fn stage_tx_entries(
    store: &MvccStore, new_files: &BTreeMap<String, Vec<u8>>, deleted: &[String],
    old_blobs: &mut BTreeMap<String, Vec<u8>>,
) -> Result<StagedTx> {
    let mut entries = Vec::with_capacity(new_files.len() + deleted.len());
    let (mut new_hashes, mut del_blobs, mut loc_delta) = (BTreeMap::new(), BTreeMap::new(), 0i64);
    let mut empty_hash: Option<String> = None;
    let mut get_empty = || -> Result<String> {
        if let Some(ref hex) = empty_hash { return Ok(hex.clone()); }
        let hex = store.store_cas(&[])?;
        empty_hash = Some(hex.clone());
        Ok(hex)
    };

    for (rel, bytes) in new_files {
        let abs = store.repo_root.join(rel);
        let (old_hash, old_lines, created) = if let Some(old) = old_blobs.get(rel) {
            (store.store_cas(old)?, count_lines(old) as i64, false)
        } else if abs.is_file() {
            let existing = fs::read(&abs)?;
            let (hex, lines) = (store.store_cas(&existing)?, count_lines(&existing) as i64);
            old_blobs.insert(rel.clone(), existing);
            (hex, lines, false)
        } else {
            (get_empty()?, 0, true)
        };
        let new_hash = store.store_cas(bytes)?;
        loc_delta += count_lines(bytes) as i64 - old_lines;
        new_hashes.insert(rel.clone(), new_hash.clone());
        entries.push(wal_file_entry(rel.clone(), old_hash, new_hash, created, false));
    }

    for rel in deleted {
        if new_files.contains_key(rel) { continue; }
        let Some(old) = old_blobs.get(rel).cloned()
            .or_else(|| fs::read(store.repo_root.join(rel)).ok()) else { continue };
        let empty_hash = get_empty()?;
        entries.push(wal_file_entry(rel.clone(), store.store_cas(&old)?, empty_hash, false, true));
        loc_delta -= count_lines(&old) as i64;
        del_blobs.insert(rel.clone(), old);
    }
    Ok((entries, new_hashes, del_blobs, loc_delta))
}

fn apply_io(store: &MvccStore, batch: &TxBatch<'_>) -> Result<()> {
    let mut written = Vec::with_capacity(batch.new_files.len());
    for (rel, bytes) in batch.new_files {
        let created = !batch.old_blobs.contains_key(rel);
        if let Err(err) = write_atomic(&store.repo_root.join(rel), bytes) {
            rollback_io(store, batch, &written, &[], format!("write failure on `{rel}`: {err}"))?;
            return Err(err);
        }
        written.push((rel.clone(), created));
    }

    let mut removed = Vec::new();
    for rel in batch.del_blobs.keys() {
        let abs = store.repo_root.join(rel);
        if !abs.exists() { continue; }
        if let Err(err) = fs::remove_file(&abs) {
            rollback_io(store, batch, &written, &removed, format!("delete failure on `{rel}`: {err}"))?;
            return Err(err.into());
        }
        removed.push(rel.clone());
    }
    Ok(())
}

fn rollback_written_file(root: &Path, rel: &str, created: bool, orig: Option<&Vec<u8>>) -> Result<()> {
    let abs = root.join(rel);
    if created { return Ok(remove_if_exists(&abs)?); }
    let Some(bytes) = orig else {
        let msg = format!("missing pre-image in memory for `{rel}`");
        return Err(WalHaltError::UnrecoverableRollback(msg).into());
    };
    write_atomic(&abs, bytes)
}

fn rollback_io(
    store: &MvccStore, batch: &TxBatch<'_>, written: &[(String, bool)], removed: &[String], reason: String,
) -> Result<()> {
    let root = &store.repo_root;
    let mut first_err: Option<anyhow::Error> = None;
    let mut record_err = |res: Result<()>, action: &str, rel: &str| {
        if let Err(err) = res && first_err.is_none() {
            first_err = Some(err.context(format!("rollback {action} `{rel}`")));
        }
    };

    for rel in removed.iter().rev() {
        if let Some(orig) = batch.del_blobs.get(rel) {
            record_err(write_atomic(&root.join(rel), orig), "resurrect", rel);
        }
    }
    for (rel, created) in written.iter().rev() {
        record_err(rollback_written_file(root, rel, *created, batch.old_blobs.get(rel)), "restore", rel);
    }
    if let Some(rb_err) = first_err {
        return Err(WalHaltError::UnrecoverableRollback(
            format!("rollback I/O failed during `{reason}`: {rb_err}"),
        ).into());
    }
    store.append_wal(&rollback_record(batch.tx_id.into(), batch.module_id.into(), "mvcc_commit_io", reason))
}

fn collect_retained(wal_path: &Path, keep_last_revs: usize) -> Result<HashSet<String>> {
    let (mut committed, mut pending, mut closed) = (Vec::new(), Vec::new(), HashSet::new());
    visit_wal(wal_path, |rec| match rec {
        WalRecord::Committed { tx_id, head_rev, files, .. } => {
            closed.insert(tx_id);
            committed.push((head_rev, files));
        }
        WalRecord::Prepared { tx_id, files, .. } => pending.push((tx_id, files)),
        WalRecord::Rollback { tx_id, .. } => { closed.insert(tx_id); }
        WalRecord::RunStart { .. } => {}
    })?;

    let mut retained = HashSet::new();
    for (tx_id, files) in pending {
        if !closed.contains(&tx_id) { keep_hashes(&files, &mut retained); }
    }
    committed.sort_by_key(|(rev, _)| *rev);
    let skip = committed.len().saturating_sub(keep_last_revs);
    for (_, files) in committed.into_iter().skip(skip) { keep_hashes(&files, &mut retained); }
    Ok(retained)
}

fn keep_hashes(files: &[WalFileEntry], retained: &mut HashSet<String>) {
    for entry in files {
        retained.insert(entry.old_blake3.clone());
        retained.insert(entry.new_blake3.clone());
    }
}

fn prune_cas(cas_dir: &Path, retained: &HashSet<String>) -> usize {
    let Ok(rd) = fs::read_dir(cas_dir) else { return 0 };
    rd.flatten().filter(|entry| {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        path.is_file() && !name.starts_with(".refit.tmp.") && !retained.contains(&name)
            && fs::remove_file(&path).is_ok()
    }).count()
}

#[cfg(test)]
use crate::testutil::{TempRepo, TestRng, commit_file, map1};
#[cfg(test)]
use std::fs::OpenOptions;
#[cfg(test)]
use std::io::Write;

#[cfg(test)]
const MOD_A: &str = "def f() -> int:\n    return 1\n";

#[cfg(test)]
fn open_repo(tag: &str, run: &str, files: &[(&str, &str)]) -> (TempRepo, MvccStore) {
    let dir = TempRepo::with_files(tag, files);
    let store = MvccStore::open(dir.path(), run).unwrap();
    (dir, store)
}

#[cfg(test)]
fn read_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(root).unwrap().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != ".refit" && path.is_file() {
            out.insert(name, fs::read(&path).unwrap());
        }
    }
    out
}

#[cfg(test)]
type PrepSpec<'a> = (&'a str, &'a [u8], &'a [u8], bool, bool);

#[cfg(test)]
fn stage_prep(st: &MvccStore, tx_id: &str, mod_id: &str, specs: &[PrepSpec<'_>]) -> Vec<String> {
    let mut old_hashes = Vec::with_capacity(specs.len());
    let mut entries = Vec::with_capacity(specs.len());
    for &(path, old_b, new_b, created, deleted) in specs {
        let (old_h, new_h) = (st.store_cas(old_b).unwrap(), st.store_cas(new_b).unwrap());
        old_hashes.push(old_h.clone());
        entries.push(wal_file_entry(path.into(), old_h, new_h, created, deleted));
    }
    st.append_wal(&WalRecord::Prepared {
        tx_id: tx_id.into(), module_id: mod_id.into(), ts_ms: now_ms(), files: entries,
    }).unwrap();
    old_hashes
}

#[cfg(test)]
fn recover_check(
    st: &MvccStore, dir: &TempRepo, tail: Option<&[u8]>, rolled: usize, head_rev: u64,
    checks: &[(&str, &str)],
) -> WalRecovery {
    if let Some(raw) = tail {
        let mut opts = OpenOptions::new();
        let mut wal = opts.append(true).open(&st.wal_path).unwrap();
        wal.write_all(raw).unwrap();
    }
    let rec = st.recover_wal().unwrap();
    assert_eq!((rec.rolled_back_txs, rec.head_rev), (rolled, head_rev));
    for &(rel, want) in checks { assert_eq!(dir.hash(rel), want); }
    rec
}

#[cfg(test)]
fn model_step(
    dir: &TempRepo, reg: &Arc<EpochRegistry>, store: &mut MvccStore,
    model: &mut BTreeMap<String, Vec<u8>>, commits: &mut u64, rng: &mut TestRng, step: u64,
) {
    let names = ["f0.rs", "f1.rs", "f2.rs", "f3.rs", "f4.rs"];
    let name = names[rng.next_usize(names.len())].to_string();
    let op = rng.next_usize(10);
    if op == 0 && let Some(live) = model.keys().next().cloned() {
        let bad = if step.is_multiple_of(2) { "badhash".into() } else { String::new() };
        let res = store.commit_tx("cas", &map1(&live, bad), &map1(&live, b"x\n".to_vec()), &[]);
        std::assert_matches!(res.unwrap(), CommitOutcome::CasConflict { .. });
    } else if op == 1 {
        *store = MvccStore::open_with_epoch(dir.path(), "reopen", reg.clone()).unwrap();
        let rec = store.recover_wal().unwrap();
        assert_eq!((rec.rolled_back_txs, rec.head_rev), (0, *commits));
    } else if op <= 3 && model.contains_key(&name) {
        let exp = map1(&name, MvccStore::hash_bytes(&model[&name]));
        let res = store.commit_tx("del", &exp, &BTreeMap::new(), std::slice::from_ref(&name)).unwrap();
        *commits += 1;
        std::assert_matches!(res, CommitOutcome::Committed { head_rev, .. } if head_rev == *commits);
        model.remove(&name);
    } else {
        let prev = model.get(&name).map_or_else(String::new, |raw| MvccStore::hash_bytes(raw));
        let body = format!("pub const V: u64 = {step};\n").into_bytes();
        let res = store.commit_tx("put", &map1(&name, prev), &map1(&name, body.clone()), &[]).unwrap();
        *commits += 1;
        std::assert_matches!(res, CommitOutcome::Committed { head_rev, .. } if head_rev == *commits);
        model.insert(name, body);
    }
    assert_eq!((store.head_rev(), reg.global_epoch()), (*commits, 1 + *commits));
    assert_eq!(read_tree(dir.path()), *model);
}

#[cfg(test)]
#[test]
fn mvcc_model_and_crash_matrix() {
    let dir = TempRepo::new("mvcc_model");
    let reg = Arc::new(EpochRegistry::new());
    let mut store = MvccStore::open_with_epoch(dir.path(), "model", reg.clone()).unwrap();
    let (mut model, mut commits, mut rng) = (BTreeMap::new(), 0u64, TestRng::new(0x4D56_CC01));
    for step in 0..32u64 {
        model_step(&dir, &reg, &mut store, &mut model, &mut commits, &mut rng, step);
    }

    let (par_dir, par_st) = open_repo("mvcc_par", "par", &[]);
    let shared = Arc::new(par_st);
    for idx in 0..8 { par_dir.write(&format!("m{idx}.py"), format!("X = {idx}\n")); }
    std::thread::scope(|scope| {
        for idx in 0..8 {
            let (st, root) = (Arc::clone(&shared), par_dir.path().to_path_buf());
            scope.spawn(move || {
                let body = format!("X = {}\n", idx + 100);
                let _ = commit_file(&st, &root, &format!("m{idx}"), &format!("m{idx}.py"), body);
            });
        }
    });
    assert_eq!(shared.head_rev(), 8);
    for idx in 0..8 {
        assert_eq!(par_dir.read(&format!("m{idx}.py")), format!("X = {}\n", idx + 100));
    }
}

#[cfg(test)]
#[test]
fn mvcc_cas_and_rollback() {
    let (dir, store) = open_repo("mvcc_cas", "cas", &[("mod_a.py", MOD_A)]);
    let (real_h, next) = (dir.hash("mod_a.py"), "def f() -> int:\n    return 2\n");
    let conflict = store.commit_tx(
        "mod_a", &map1("mod_a.py", "deadbeef".into()),
        &map1("mod_a.py", next.as_bytes().to_vec()), &[],
    ).unwrap();
    std::assert_matches!(
        conflict,
        CommitOutcome::CasConflict { ref path, ref expected, ref actual }
            if path == "mod_a.py" && expected == "deadbeef" && actual == &real_h
    );
    assert_eq!(dir.read("mod_a.py"), MOD_A);

    let (rev, tx_id, h1) = commit_file(&store, dir.path(), "mod_a", "mod_a.py", next);
    assert_eq!(rev, 1);
    let torn = b"def f() -> int:\n    return 999\n";
    stage_prep(&store, "tx-crash", "mod_a", &[("mod_a.py", next.as_bytes(), torn, false, false)]);
    dir.write("mod_a.py", torn);
    let tail = b"{\"kind\":\"committed\",\"tx_id\":\"trunca";
    let rep = recover_check(&store, &dir, Some(tail), 1, 1, &[("mod_a.py", &h1)]);
    assert_eq!(dir.read("mod_a.py"), next);
    assert_eq!(rep.committed_modules["mod_a"]["mod_a.py"], h1);

    fs::create_dir_all(dir.join("z_dir")).unwrap();
    let bad = BTreeMap::from([("mod_a.py".into(), b"A = 9\n".to_vec()), ("z_dir".into(), b"x".to_vec())]);
    assert!(store.commit_tx("pair", &map1("mod_a.py", h1), &bad, &[]).is_err());
    assert_eq!(dir.read("mod_a.py"), next);

    store.record_rollback("bad", "gate0", "syntax").unwrap();
    let tl = MvccStore::wal_timeline(dir.path()).unwrap();
    let last_state = tl.last().unwrap().state.as_str();
    assert_eq!(
        (tl[0].tx_id.as_str(), tl[0].state.as_str(), last_state),
        (tx_id.as_str(), "COMMITTED", "ROLLED_BACK")
    );
    let diff_rep = MvccStore::diff_tx(dir.path(), &tx_id).unwrap().unwrap();
    assert_eq!((diff_rep.files.len(), diff_rep.files[0].removed, diff_rep.files[0].added), (1, 1, 1));
    let before: String = (1..=20).map(|idx| format!("line_{idx}\n")).collect();
    let after: String = (1..=20).map(|idx| match idx {
        2 => "line_2_x\n".into(), 19 => "line_19_x\n".into(), _ => format!("line_{idx}\n"),
    }).collect();
    assert_eq!(diff::make_diff("demo.rs", &before, &after).2.matches("@@ ").count(), 2);

    let (gdir, s1) = open_repo("mvcc_gc", "r1", &[("old.py", "OLD = 1\n")]);
    let r1 = s1.commit_tx(
        "split", &map1("old.py", gdir.hash("old.py")),
        &map1("new.py", b"NEW = 2\n".to_vec()), &["old.py".into()],
    ).unwrap();
    std::assert_matches!(r1, CommitOutcome::Committed { head_rev: 1, .. });
    assert!(!gdir.has("old.py") && gdir.has("new.py"));
    let s2 = MvccStore::open(gdir.path(), "r2").unwrap();
    let rev2 = commit_file(&s2, gdir.path(), "upd", "new.py", "NEW = 3\n").0;
    assert_eq!((s2.head_rev(), rev2, s2.gc_cas(1).unwrap()), (2, 2, 2));
}

#[cfg(test)]
#[test]
fn wal_crash_points_7_of_7() {
    assert_eq!((wal::crc32c(b"123456789"), wal::crc32c(b"")), (0xE306_9283, 0));
    let long_buf: Vec<u8> = (0..137u8).collect();
    let ref_crc = !long_buf.iter().fold(0xFFFF_FFFFu32, |mut crc, &byte| {
        crc ^= u32::from(byte);
        for _ in 0..8 { crc = if (crc & 1) != 0 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 }; }
        crc
    });
    assert_eq!(wal::crc32c(&long_buf), ref_crc);

    let (d1, s1) = open_repo("cp1", "cp1", &[("a.rs", "pub fn v() -> i32 { 1 }\n")]);
    let h1 = d1.hash("a.rs");
    s1.store_cas(b"pub fn v() -> i32 { 2 }\n").unwrap();
    recover_check(&s1, &d1, None, 0, 0, &[("a.rs", &h1)]);
    assert_eq!(s1.gc_cas(1).unwrap(), 1);

    let (d2, s2) = open_repo("cp2", "cp2", &[("a.rs", "pub fn v() -> i32 { 1 }\n")]);
    let prep_raw = serde_json::to_vec(&WalRecord::Prepared {
        tx_id: "tx-torn".into(), module_id: "a".into(), ts_ms: now_ms(), files: vec![],
    }).unwrap();
    let frame = wal::encode_wal_frame(2, &prep_raw).unwrap();
    let h2 = d2.hash("a.rs");
    recover_check(&s2, &d2, Some(&frame[..13]), 0, 0, &[("a.rs", &h2)]);
    recover_check(&s2, &d2, Some(&[0u8; 64]), 0, 0, &[]);
    assert_eq!(commit_file(&s2, d2.path(), "a", "a.rs", "pub fn v() -> i32 { 2 }\n").0, 1);

    let torn = &b"{\"kind\":\"committed\",\"tx_id\":\"tx-cp6\""[..];
    for (tag, mutate, tail) in [("cp3", false, None), ("cp6", true, Some(torn))] {
        let (dir, st) = open_repo(tag, tag, &[("m.rs", "M=1\n")]);
        let spec = [("m.rs", b"M=1\n".as_slice(), b"M=2\n".as_slice(), false, false)];
        let hs = stage_prep(&st, &format!("tx-{tag}"), tag, &spec);
        if mutate { dir.write("m.rs", "M=2\n"); }
        recover_check(&st, &dir, tail, 1, 0, &[("m.rs", &hs[0])]);
        if tail.is_some() { assert_eq!(commit_file(&st, dir.path(), tag, "m.rs", "M=3\n").0, 1); }
    }

    let (d4, s4) = open_repo(
        "cp4", "cp4", &[("e1.rs", "E1=1\n"), ("e2.rs", "E2=1\n"), ("del.rs", "D=1\n")],
    );
    let h4 = stage_prep(&s4, "tx-cp4", "b4", &[
        ("e1.rs", b"E1=1\n", b"N=2\n", false, false), ("e2.rs", b"E2=1\n", b"N=2\n", false, false),
        ("new.rs", b"", b"N=2\n", true, false), ("del.rs", b"D=1\n", b"", false, true),
    ]);
    d4.write_all(&[("e1.rs", "CORRUPT\n"), ("new.rs", "TMP\n")]);
    recover_check(&s4, &d4, None, 1, 0, &[("e1.rs", &h4[0]), ("e2.rs", &h4[1]), ("del.rs", &h4[3])]);
    assert!(!d4.has("new.rs"));

    let (d5, s5) = open_repo("cp5", "cp5", &[("k.rs", "K=1\n"), ("g.rs", "G=1\n")]);
    let h5 = stage_prep(&s5, "tx-cp5", "m5", &[
        ("k.rs", b"K=1\n", b"K=2\n", false, false), ("g.rs", b"G=1\n", b"", false, true),
    ]);
    d5.write("k.rs", "K=2\n");
    fs::remove_file(d5.join("g.rs")).unwrap();
    recover_check(&s5, &d5, None, 1, 0, &[("k.rs", &h5[0]), ("g.rs", &h5[1])]);

    let (d7, s7) = open_repo("cp7", "cp7", &[("a.rs", "A=1\n"), ("b.rs", "B=1\n")]);
    let h7 = stage_prep(&s7, "tx-cp7", "m7", &[
        ("a.rs", b"A=1\n", b"A=2\n", false, false), ("b.rs", b"B=1\n", b"B=2\n", false, false),
    ]);
    d7.write_all(&[("b.rs", "B=2\n"), (".refit.tmp.99.1.b.rs", "TMP")]);
    fs::write(s7.cas_dir.join(".refit.tmp.99.2.blob"), b"TMP").unwrap();
    recover_check(&s7, &d7, None, 1, 0, &[("a.rs", &h7[0]), ("b.rs", &h7[1])]);
    let rec7 = s7.recover_wal().unwrap().rolled_back_txs;
    assert_eq!((rec7, commit_file(&s7, d7.path(), "m7", "a.rs", "A=3\n").0), (0, 1));

    let recs = [
        WalRecord::RunStart { run_id: "r".into(), ts_ms: 1, repo_root: ".".into() },
        rollback_record("t1".into(), "m".into(), "g", "ok".into()),
        rollback_record("t2".into(), "m".into(), "g", "ok".into()),
    ];
    let raw: Vec<Vec<u8>> = recs
        .iter()
        .map(|rec| serde_json::to_vec(rec).unwrap())
        .collect();
    let fr: Vec<Vec<u8>> = raw.iter().enumerate()
        .map(|(idx, buf)| wal::encode_wal_frame((idx + 1) as u64, buf).unwrap()).collect();
    let mut flipped = [&fr[0][..], &fr[1][..], &fr[2][..]].concat();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x01;
    let fused = [&fr[0][..], &fr[1][..fr[1].len() - 4], &fr[2][..]].concat();
    let gapped = [&fr[0][..], &fr[2][..]].concat();
    for bad in [&flipped, &fused, &gapped] {
        let res = wal::scan_wal_bytes(bad, &mut |_| {});
        std::assert_matches!(res, Err(WalHaltError::MidFileCorruption(_)));
    }
    let upgraded = [&raw[0][..], b"\n", &fr[1][..], &fr[2][..]].concat();
    let mut seen = 0usize;
    let stats = wal::scan_wal_bytes(&upgraded, &mut |_| seen += 1).unwrap();
    assert_eq!((seen, stats.valid_records, stats.next_seq, stats.has_framed), (3, 3, 4, true));

    let (cdir, cst) = open_repo("cas_halt", "cas", &[("lost.rs", "ORIG\n")]);
    let missing = "deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000";
    let new_h = cst.store_cas(b"NEW\n").unwrap();
    let entry = wal_file_entry("lost.rs".into(), missing.into(), new_h, false, false);
    cst.append_wal(&WalRecord::Prepared {
        tx_id: "tx-m".into(), module_id: "lost".into(), ts_ms: now_ms(), files: vec![entry],
    }).unwrap();
    let is_halt = |err: anyhow::Error| {
        matches!(err.downcast_ref::<WalHaltError>(), Some(WalHaltError::UnrecoverableRollback(_)))
    };
    assert!(is_halt(cst.recover_wal().unwrap_err()));
    assert!(MvccStore::wal_timeline(cdir.path()).unwrap().is_empty());
    fs::write(cst.cas_dir.join(missing), b"BAD").unwrap();
    assert!(is_halt(cst.recover_wal().unwrap_err()));
}
