use crate::platform::{atomic_replace_file, now_ms};
use crate::types::{WalFileEntry, WalRecord};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

static TX_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct RecoveryReport {
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

pub struct MvccStore {
    repo_root: PathBuf,
    cas_dir: PathBuf,
    wal_path: PathBuf,
    commit_lock: Mutex<u64>,
}

impl MvccStore {
    pub fn open(repo_root: &Path, run_id: &str) -> Result<Self> {
        let refit_dir = repo_root.join(".refit");
        let cas_dir = refit_dir.join("cas");
        fs::create_dir_all(&cas_dir)
            .with_context(|| format!("create CAS dir {}", cas_dir.display()))?;
        let wal_path = refit_dir.join("wal.jsonl");

        let store = Self {
            repo_root: repo_root.to_path_buf(),
            cas_dir,
            wal_path,
            commit_lock: Mutex::new(0),
        };

        if !store.wal_path.exists() {
            store.append_wal(&WalRecord::RunStart {
                run_id: run_id.to_string(),
                ts_ms: now_ms(),
                repo_root: repo_root.display().to_string(),
            })?;
        }

        Ok(store)
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
            atomic_replace_file(&obj, bytes)?;
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

    pub fn recover_from_wal(&self) -> Result<RecoveryReport> {
        let mut head_rev = self.commit_lock.lock().unwrap();
        if !self.wal_path.exists() {
            return Ok(RecoveryReport {
                rolled_back_txs: 0,
                committed_modules: HashMap::new(),
                head_rev: 0,
            });
        }

        let file = fs::File::open(&self.wal_path)?;
        let reader = BufReader::new(file);

        let mut prepared: Vec<(String, String, Vec<WalFileEntry>)> = Vec::new();
        let mut committed_txs: HashSet<String> = HashSet::new();
        let mut committed_modules: HashMap<String, BTreeMap<String, String>> = HashMap::new();
        let mut max_rev = 0u64;

        for line in reader.lines() {
            let Ok(line) = line else { break };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<WalRecord>(trimmed) else {
                // Ignore incomplete truncated tail line after SIGKILL
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
                        map.insert(f.path, f.new_blake3);
                    }
                    committed_modules.insert(module_id, map);
                }
                WalRecord::RunStart { .. } | WalRecord::Rollback { .. } => {}
            }
        }

        let mut rolled_back_txs = 0usize;
        for (tx_id, module_id, files) in prepared {
            if !committed_txs.contains(&tx_id) {
                for f in &files {
                    if let Ok(old_bytes) = self.load_cas(&f.old_blake3) {
                        let abs = self.repo_root.join(&f.path);
                        atomic_replace_file(&abs, &old_bytes)?;
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
        Ok(RecoveryReport {
            rolled_back_txs,
            committed_modules,
            head_rev: max_rev,
        })
    }

    pub fn commit_module(
        &self,
        module_id: &str,
        expected_hashes: &BTreeMap<String, String>,
        new_files: &BTreeMap<String, Vec<u8>>,
    ) -> Result<CommitOutcome> {
        let mut rev_guard = self.commit_lock.lock().unwrap();

        // 1. Strict CAS check before writing anything to disk
        let mut old_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (rel_path, expected) in expected_hashes {
            let abs = self.repo_root.join(rel_path);
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

        // 2. Persist both old & new blobs to CAS before WAL PREPARED
        let mut entries = Vec::with_capacity(new_files.len());
        let mut new_hashes = BTreeMap::new();
        for (rel_path, new_bytes) in new_files {
            let old_hash = if let Some(old) = old_blobs.get(rel_path) {
                self.store_cas(old)?
            } else {
                let abs = self.repo_root.join(rel_path);
                if abs.exists() {
                    let existing = fs::read(&abs)?;
                    self.store_cas(&existing)?
                } else {
                    self.store_cas(&[])?
                }
            };
            let new_hash = self.store_cas(new_bytes)?;
            new_hashes.insert(rel_path.clone(), new_hash.clone());
            entries.push(WalFileEntry {
                path: rel_path.clone(),
                old_blake3: old_hash,
                new_blake3: new_hash,
            });
        }

        let tx_id = format!("tx-{}-{}", now_ms(), TX_SEQ.fetch_add(1, Ordering::Relaxed));

        // 3. Durable WAL PREPARED
        self.append_wal(&WalRecord::Prepared {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            ts_ms: now_ms(),
            files: entries.clone(),
        })?;

        // 4. Atomic temp-rename in deterministic sorted path order
        for (rel_path, new_bytes) in new_files {
            let abs = self.repo_root.join(rel_path);
            atomic_replace_file(&abs, new_bytes)?;
        }

        // 5. Advance rev and durable WAL COMMITTED
        *rev_guard += 1;
        let head_rev = *rev_guard;
        self.append_wal(&WalRecord::Committed {
            tx_id: tx_id.clone(),
            module_id: module_id.to_string(),
            head_rev,
            ts_ms: now_ms(),
            files: entries,
        })?;

        Ok(CommitOutcome::Committed {
            head_rev,
            tx_id,
            new_hashes,
        })
    }

    pub fn record_rollback(&self, module_id: &str, gate: &str, reason: &str) -> Result<()> {
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
        *self.commit_lock.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvcc_cas_and_crash_recovery_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("refit_mvcc_test_{}", now_ms()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let fpath = tmp.join("mod_a.py");
        fs::write(&fpath, b"def f() -> int:\n    return 1\n").unwrap();
        let h0 = MvccStore::hash_file(&fpath).unwrap();

        let store = MvccStore::open(&tmp, "test-run").unwrap();

        // 1. CAS conflict must leave disk untouched
        let mut bad_exp = BTreeMap::new();
        bad_exp.insert("mod_a.py".into(), "deadbeef".into());
        let mut new_map = BTreeMap::new();
        new_map.insert("mod_a.py".into(), b"def f() -> int:\n    return 2\n".to_vec());
        match store.commit_module("mod_a", &bad_exp, &new_map).unwrap() {
            CommitOutcome::CasConflict { .. } => {}
            other => panic!("expected CasConflict, got {other:?}"),
        }
        assert_eq!(fs::read(&fpath).unwrap(), b"def f() -> int:\n    return 1\n");

        // 2. Valid commit advances rev
        let mut good_exp = BTreeMap::new();
        good_exp.insert("mod_a.py".into(), h0);
        let h1 = match store.commit_module("mod_a", &good_exp, &new_map).unwrap() {
            CommitOutcome::Committed { head_rev, new_hashes, .. } => {
                assert_eq!(head_rev, 1);
                new_hashes["mod_a.py"].clone()
            }
            other => panic!("expected Committed, got {other:?}"),
        };

        // 3. Simulate SIGKILL mid-transaction: PREPARED written + torn file on disk without COMMITTED
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
                }],
            })
            .unwrap();
        fs::write(&fpath, torn).unwrap();

        // 4. Recovery restores exact committed bytes and reports mod_a as committed
        let rep = store.recover_from_wal().unwrap();
        assert_eq!(rep.rolled_back_txs, 1);
        assert_eq!(rep.head_rev, 1);
        assert_eq!(fs::read(&fpath).unwrap(), b"def f() -> int:\n    return 2\n");
        assert_eq!(rep.committed_modules["mod_a"]["mod_a.py"], h1);

        let _ = fs::remove_dir_all(&tmp);
    }
}
