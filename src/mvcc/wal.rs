//! Unified MVCC WAL storage: binary `RFWL` Castagnoli CRC32C (`0xE3069283`) framing,
//! legacy JSONL auto-detection, torn-tail recovery, and clean-halt corruption detection.

use crate::platform::sync_file_barrier;
use crate::types::WalRecord;
use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use memchr::{memchr, memmem};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const WAL_MAGIC: &[u8; 4] = b"RFWL";
pub const WAL_HEADER_LEN: usize = 20; // magic(4) + len(4) + crc32c(4) + seq(8)
pub const WAL_MAX_FRAME_LEN: usize = 4 * 1024 * 1024;
const CRC32C_POLY: u32 = 0x82F6_3B78;

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
    Committed { head_rev: u64, tx_id: String, new_hashes: BTreeMap<String, String> },
    CasConflict { path: String, expected: String, actual: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalHaltError {
    MidFileCorruption(usize),
    UnrecoverableRollback(String),
}

impl fmt::Display for WalHaltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MidFileCorruption(off) => write!(f, "WAL clean-halt: mid-file corruption at byte offset {off}"),
            Self::UnrecoverableRollback(detail) => write!(f, "WAL clean-halt: unrecoverable rollback ({detail})"),
        }
    }
}

impl std::error::Error for WalHaltError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WalScanStats {
    pub valid_records: usize,
    pub valid_pos: usize,
    pub file_len: usize,
    pub truncated_bytes: usize,
    pub next_seq: u64,
    pub has_framed: bool,
}

enum StepOutcome {
    Record { rec: WalRecord, next_cursor: usize, framed_seq: Option<u64> },
    Skip(usize),
    StopAtTornTail,
}

/// Compact pure-Rust Castagnoli CRC32C (`0x82F63B78`; RFC 3720 check `0xE3069283`).
#[inline]
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32C_POLY & mask);
        }
    }
    !crc
}

/// Encode `[Magic "RFWL" 4B | Len u32 | CRC32C 4B | Seq u64 | JSON payload | '\n']`.
pub fn encode_wal_frame(seq: u64, payload: &[u8]) -> Result<Vec<u8>> {
    if payload.len() > WAL_MAX_FRAME_LEN {
        anyhow::bail!("WAL frame payload {} exceeds 4 MiB limit ({WAL_MAX_FRAME_LEN})", payload.len());
    }
    let mut frame = Vec::with_capacity(WAL_HEADER_LEN + payload.len() + 1);
    frame.extend_from_slice(WAL_MAGIC);
    frame.write_u32::<LittleEndian>(payload.len() as u32)?;
    frame.write_u32::<LittleEndian>(0)?; // placeholder for CRC32C
    frame.write_u64::<LittleEndian>(seq)?;
    frame.extend_from_slice(payload);
    let checksum = crc32c(&frame[12..]);
    LittleEndian::write_u32(&mut frame[8..12], checksum);
    frame.push(b'\n');
    Ok(frame)
}

fn append_wal_bytes(wal_path: &Path, bytes: &[u8]) -> Result<u64> {
    let existed = wal_path.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .with_context(|| format!("open WAL {}", wal_path.display()))?;
    file.write_all(bytes)?;
    sync_file_barrier(&file)?;
    let new_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    #[cfg(unix)]
    if !existed
        && let Some(parent) = wal_path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
    #[cfg(not(unix))]
    let _ = existed;
    Ok(new_len)
}

pub fn append_framed_record(wal_path: &Path, seq: u64, payload: &[u8]) -> Result<u64> {
    append_wal_bytes(wal_path, &encode_wal_frame(seq, payload)?)
}

pub fn append_jsonl_record(wal_path: &Path, payload: &[u8]) -> Result<u64> {
    let mut line = Vec::with_capacity(payload.len() + 1);
    line.extend_from_slice(payload);
    line.push(b'\n');
    append_wal_bytes(wal_path, &line)
}

pub fn truncate_torn_tail(wal_path: &Path, valid_pos: usize) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(wal_path)
        .with_context(|| format!("open WAL for truncate {}", wal_path.display()))?;
    file.set_len(valid_pos as u64)?;
    sync_file_barrier(&file)?;
    Ok(())
}

pub fn file_uses_frames(wal_path: &Path) -> bool {
    let Ok(mut file) = File::open(wal_path) else { return false };
    let mut head = [0u8; 4];
    file.read_exact(&mut head).is_ok() && &head == WAL_MAGIC
}

#[inline]
fn is_all_zero(slice: &[u8]) -> bool {
    slice.iter().all(|&b| b == 0)
}

fn parse_frame_header(buf: &[u8], pos: usize) -> Option<(usize, u32, u64)> {
    let hdr = buf.get(pos..pos + WAL_HEADER_LEN)?;
    if &hdr[0..4] != WAL_MAGIC {
        return None;
    }
    Some((
        LittleEndian::read_u32(&hdr[4..8]) as usize,
        LittleEndian::read_u32(&hdr[8..12]),
        LittleEndian::read_u64(&hdr[12..20]),
    ))
}

fn is_valid_framed_at(buf: &[u8], pos: usize) -> bool {
    let Some((len, expected_crc, _)) = parse_frame_header(buf, pos) else { return false };
    let end = pos.saturating_add(WAL_HEADER_LEN).saturating_add(len);
    len <= WAL_MAX_FRAME_LEN && end < buf.len() && buf[end] == b'\n' && crc32c(&buf[pos + 12..end]) == expected_crc
}

fn has_valid_record_after(buf: &[u8], from: usize) -> bool {
    let start = from.saturating_add(1);
    if start >= buf.len() {
        return false;
    }
    let tail = &buf[start..];
    if memmem::find_iter(tail, WAL_MAGIC).any(|rel| is_valid_framed_at(buf, start + rel)) {
        return true;
    }
    for rel in memchr::memchr_iter(b'{', tail).take(64) {
        let pos = start + rel;
        let line_end = memchr(b'\n', &buf[pos..]).map_or(buf.len(), |off| pos + off);
        let trimmed = buf[pos..line_end].trim_ascii();
        if !trimmed.is_empty() && serde_json::from_slice::<WalRecord>(trimmed).is_ok() {
            return true;
        }
    }
    false
}

fn parse_framed_step(
    buf: &[u8],
    cursor: usize,
    valid_records: usize,
    last_framed_seq: u64,
) -> std::result::Result<StepOutcome, WalHaltError> {
    if cursor + WAL_HEADER_LEN + 1 > buf.len() {
        return Ok(StepOutcome::StopAtTornTail);
    }
    let Some((len, expected_crc, seq)) = parse_frame_header(buf, cursor) else {
        return Err(WalHaltError::MidFileCorruption(cursor));
    };
    let frame_end = cursor.saturating_add(WAL_HEADER_LEN).saturating_add(len);
    if len > WAL_MAX_FRAME_LEN || frame_end >= buf.len() {
        return if has_valid_record_after(buf, cursor) {
            Err(WalHaltError::MidFileCorruption(cursor))
        } else {
            Ok(StepOutcome::StopAtTornTail)
        };
    }
    if crc32c(&buf[cursor + 12..frame_end]) != expected_crc || buf[frame_end] != b'\n' {
        return if frame_end + 1 < buf.len() && !is_all_zero(&buf[frame_end + 1..]) {
            Err(WalHaltError::MidFileCorruption(cursor))
        } else {
            Ok(StepOutcome::StopAtTornTail)
        };
    }
    let seq_ok = if last_framed_seq > 0 { seq == last_framed_seq + 1 } else { seq == 1 || seq == (valid_records as u64) + 1 };
    if !seq_ok {
        return Err(WalHaltError::MidFileCorruption(cursor));
    }
    let rec = serde_json::from_slice::<WalRecord>(&buf[cursor + WAL_HEADER_LEN..frame_end])
        .map_err(|_| WalHaltError::MidFileCorruption(cursor))?;
    Ok(StepOutcome::Record { rec, next_cursor: frame_end + 1, framed_seq: Some(seq) })
}

fn parse_jsonl_step(buf: &[u8], cursor: usize) -> std::result::Result<StepOutcome, WalHaltError> {
    let Some(rel_nl) = memchr(b'\n', &buf[cursor..]) else {
        return if has_valid_record_after(buf, cursor) {
            Err(WalHaltError::MidFileCorruption(cursor))
        } else {
            Ok(StepOutcome::StopAtTornTail)
        };
    };
    let line_end = cursor + rel_nl;
    let trimmed = buf[cursor..line_end].trim_ascii();
    if trimmed.is_empty() {
        return Ok(StepOutcome::Skip(line_end + 1));
    }
    if let Ok(rec) = serde_json::from_slice::<WalRecord>(trimmed) {
        return Ok(StepOutcome::Record { rec, next_cursor: line_end + 1, framed_seq: None });
    }
    let has_trailing = line_end + 1 < buf.len() && !is_all_zero(&buf[line_end + 1..]);
    let fused = memmem::find(trimmed, b"}{").is_some() || memmem::find(trimmed, WAL_MAGIC).is_some();
    if has_trailing || fused || has_valid_record_after(buf, cursor) {
        return Err(WalHaltError::MidFileCorruption(cursor));
    }
    Ok(StepOutcome::StopAtTornTail)
}

/// Scan dual-format WAL bytes (`RFWL` binary frames and legacy JSONL seamlessly).
pub fn scan_wal_bytes(
    buf: &[u8],
    on_rec: &mut impl FnMut(WalRecord),
) -> std::result::Result<WalScanStats, WalHaltError> {
    let (mut cursor, mut valid_pos, mut valid_records) = (0usize, 0usize, 0usize);
    let (mut last_framed_seq, mut has_framed) = (0u64, false);
    while cursor < buf.len() && !is_all_zero(&buf[cursor..]) {
        let step = if buf[cursor..].starts_with(WAL_MAGIC) {
            parse_framed_step(buf, cursor, valid_records, last_framed_seq)?
        } else if buf[cursor] == b'{' {
            parse_jsonl_step(buf, cursor)?
        } else if matches!(buf[cursor], b'\n' | b'\r' | b' ' | b'\t') {
            StepOutcome::Skip(cursor + 1)
        } else if has_valid_record_after(buf, cursor) {
            return Err(WalHaltError::MidFileCorruption(cursor));
        } else {
            StepOutcome::StopAtTornTail
        };
        match step {
            StepOutcome::Record { rec, next_cursor, framed_seq } => {
                on_rec(rec);
                cursor = next_cursor;
                valid_pos = cursor;
                valid_records += 1;
                if let Some(seq) = framed_seq {
                    last_framed_seq = seq;
                    has_framed = true;
                }
            }
            StepOutcome::Skip(next_cursor) => {
                cursor = next_cursor;
                valid_pos = cursor;
            }
            StepOutcome::StopAtTornTail => break,
        }
    }
    let next_seq = if has_framed { last_framed_seq + 1 } else { (valid_records as u64) + 1 };
    Ok(WalScanStats {
        valid_records,
        valid_pos,
        file_len: buf.len(),
        truncated_bytes: buf.len().saturating_sub(valid_pos),
        next_seq,
        has_framed,
    })
}

pub fn visit_wal_stats(path: &Path, mut on_rec: impl FnMut(WalRecord)) -> Result<WalScanStats> {
    if !path.exists() {
        return Ok(WalScanStats { next_seq: 1, ..WalScanStats::default() });
    }
    let buf = fs::read(path).with_context(|| format!("read WAL {}", path.display()))?;
    scan_wal_bytes(&buf, &mut on_rec).map_err(anyhow::Error::new)
}

pub fn visit_wal(path: &Path, on_rec: impl FnMut(WalRecord)) -> Result<()> {
    let _ = visit_wal_stats(path, on_rec)?;
    Ok(())
}

pub fn wal_next_seq(path: &Path) -> Result<u64> {
    Ok(visit_wal_stats(path, |_| {})?.next_seq.max(1))
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
    let (wal, cas) = (repo_root.join(".refit/wal.jsonl"), repo_root.join(".refit/cas"));
    if wal.exists() || cas.exists() {
        return (wal, cas);
    }
    let (alt_wal, alt_cas) = (repo_root.join(".refit/mvcc/wal.jsonl"), repo_root.join(".refit/mvcc/cas"));
    if alt_wal.exists() || alt_cas.exists() { (alt_wal, alt_cas) } else { (wal, cas) }
}

pub fn read_blob(cas_dir: &Path, hash: &str) -> Option<String> {
    fs::read(cas_dir.join(hash)).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
}
