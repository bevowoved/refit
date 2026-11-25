use super::framed_wal::{WalScanStats, scan_wal_file};
use crate::types::WalRecord;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

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

/// Stream every valid `WalRecord` from either binary `RFWL` framed WAL or legacy JSONL.
pub fn visit_wal(path: &Path, on_rec: impl FnMut(WalRecord)) -> Result<()> {
    let _ = scan_wal_file(path, on_rec)?;
    Ok(())
}

/// Scan every valid `WalRecord` and return framing / torn-tail statistics.
pub fn visit_wal_stats(path: &Path, on_rec: impl FnMut(WalRecord)) -> Result<WalScanStats> {
    scan_wal_file(path, on_rec)
}

pub fn wal_head(wal_path: &Path) -> u64 {
    let mut max_rev = 0u64;
    let _ = visit_wal(wal_path, |rec| {
        if let WalRecord::Committed { head_rev, .. } = rec {
            max_rev = max_rev.max(head_rev);
        }
    });
    max_rev
}

pub fn storage_paths(repo_root: &Path) -> (PathBuf, PathBuf) {
    let wal = repo_root.join(".refit/wal.jsonl");
    let cas = repo_root.join(".refit/cas");
    if wal.exists() || cas.exists() {
        return (wal, cas);
    }

    let alt_wal = repo_root.join(".refit/mvcc/wal.jsonl");
    let alt_cas = repo_root.join(".refit/mvcc/cas");
    if alt_wal.exists() || alt_cas.exists() {
        return (alt_wal, alt_cas);
    }
    (wal, cas)
}

pub fn read_blob(cas_dir: &Path, hash: &str) -> Option<String> {
    let bytes = fs::read(cas_dir.join(hash)).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}
