use super::wal::{FileDiff, TxDiffReport, WalEvent, read_blob, storage_paths, visit_wal};
use crate::source::count_lines;
use crate::types::{WalFileEntry, WalRecord};
use anyhow::Result;
use dashmap::DashMap;
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static TIMELINE_CACHE: OnceLock<DashMap<PathBuf, (u64, Vec<WalEvent>)>> = OnceLock::new();

pub fn wal_timeline(repo_root: &Path) -> Result<Vec<WalEvent>> {
    let (wal_path, cas_dir) = storage_paths(repo_root);
    if !wal_path.exists() {
        return Ok(Vec::new());
    }

    let wal_len = fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    let cache = TIMELINE_CACHE.get_or_init(DashMap::new);
    if let Some(hit) = cache.get(&wal_path).filter(|e| e.0 == wal_len) {
        return Ok(hit.1.clone());
    }

    let mut events = Vec::new();
    visit_wal(&wal_path, |rec| match rec {
        WalRecord::Committed {
            tx_id,
            module_id,
            head_rev,
            ts_ms,
            loc_delta,
            files,
        } => {
            let delta = loc_delta.unwrap_or_else(|| calc_delta(&cas_dir, &files));
            let paths = files.into_iter().map(|f| f.path).collect();
            events.push(WalEvent {
                rev: head_rev,
                tx_id,
                module_id,
                state: "COMMITTED".into(),
                files: paths,
                ts_ms,
                reason: None,
                loc_delta: delta,
            });
        }
        WalRecord::Rollback {
            tx_id,
            module_id,
            gate,
            reason,
            ts_ms,
        } => {
            events.push(WalEvent {
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
    })?;

    cache.insert(wal_path, (wal_len, events.clone()));
    Ok(events)
}

fn calc_delta(cas_dir: &Path, files: &[WalFileEntry]) -> i64 {
    let mut delta = 0i64;
    for f in files {
        let old_l = fs::read(cas_dir.join(&f.old_blake3))
            .map(|b| count_lines(&b) as i64)
            .unwrap_or(0);
        let new_l = fs::read(cas_dir.join(&f.new_blake3))
            .map(|b| count_lines(&b) as i64)
            .unwrap_or(0);
        delta += new_l - old_l;
    }
    delta
}

pub fn diff_tx(repo_root: &Path, target_tx: &str) -> Result<Option<TxDiffReport>> {
    let (wal_path, cas_dir) = storage_paths(repo_root);
    if !wal_path.exists() {
        return Ok(None);
    }

    let mut matched: Option<(u64, String, String, Vec<WalFileEntry>)> = None;
    visit_wal(&wal_path, |rec| match rec {
        WalRecord::Committed {
            tx_id,
            module_id,
            head_rev,
            files,
            ..
        } if tx_id == target_tx => {
            matched = Some((head_rev, module_id, "COMMITTED".into(), files));
        }
        WalRecord::Prepared {
            tx_id,
            module_id,
            files,
            ..
        } if tx_id == target_tx && matched.is_none() => {
            matched = Some((0, module_id, "PREPARED".into(), files));
        }
        _ => {}
    })?;

    let Some((rev, module_id, state, wal_files)) = matched else {
        return Ok(None);
    };

    let mut diffs = Vec::with_capacity(wal_files.len());
    for wf in wal_files {
        let before = read_blob(&cas_dir, &wf.old_blake3).unwrap_or_default();
        let after = read_blob(&cas_dir, &wf.new_blake3).unwrap_or_default();
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
        tx_id: target_tx.to_string(),
        rev,
        module_id,
        state,
        files: diffs,
    }))
}

pub fn make_diff(path: &str, before: &str, after: &str) -> (usize, usize, String) {
    let diff = TextDiff::from_lines(before, after);
    let (mut added, mut removed) = (0usize, 0usize);
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
