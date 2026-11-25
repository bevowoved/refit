//! PM-V7 CRC32C binary-framed WAL (`RFWL`) with dual-format legacy JSONL auto-detection,
//! all-zero EOF sentinel handling, monotonic sequence verification, and clean-halt errors.

use crate::platform::sync_file_barrier;
use crate::types::WalRecord;
use anyhow::{Context, Result};
use memchr::{memchr, memmem};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub const WAL_MAGIC: &[u8; 4] = b"RFWL";
pub const WAL_HEADER_LEN: usize = 20; // magic(4) + len(4) + crc32c(4) + seq(8)
pub const WAL_MAX_FRAME_LEN: usize = 4 * 1024 * 1024;
const CRC32C_POLY: u32 = 0x82F6_3B78;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalHaltError {
    MidFileCorruption(usize),
    UnrecoverableRollback(String),
}

impl fmt::Display for WalHaltError {
    fn fmt(&self, fmt_out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MidFileCorruption(offset) => {
                write!(fmt_out, "WAL clean-halt: mid-file corruption at byte offset {offset}")
            }
            Self::UnrecoverableRollback(detail) => {
                write!(fmt_out, "WAL clean-halt: unrecoverable rollback ({detail})")
            }
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
    Record {
        rec: WalRecord,
        next_cursor: usize,
        framed_seq: Option<u64>,
    },
    Skip(usize),
    StopAtTornTail,
}

const fn make_crc32c_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut idx = 0usize;
    while idx < 256 {
        let mut crc = idx as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ CRC32C_POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        tables[0][idx] = crc;
        idx += 1;
    }

    let mut slice = 1usize;
    while slice < 8 {
        let mut entry = 0usize;
        while entry < 256 {
            let prev = tables[slice - 1][entry];
            tables[slice][entry] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            entry += 1;
        }
        slice += 1;
    }
    tables
}

static CRC32C_TABLES: [[u32; 256]; 8] = make_crc32c_tables();

/// Pure-Rust slice-by-8 Castagnoli CRC32C (`0x82F63B78`; RFC 3720 check `0xE3069283`).
#[inline]
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ crc;
        let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        crc = CRC32C_TABLES[7][(lo & 0xFF) as usize]
            ^ CRC32C_TABLES[6][((lo >> 8) & 0xFF) as usize]
            ^ CRC32C_TABLES[5][((lo >> 16) & 0xFF) as usize]
            ^ CRC32C_TABLES[4][(lo >> 24) as usize]
            ^ CRC32C_TABLES[3][(hi & 0xFF) as usize]
            ^ CRC32C_TABLES[2][((hi >> 8) & 0xFF) as usize]
            ^ CRC32C_TABLES[1][((hi >> 16) & 0xFF) as usize]
            ^ CRC32C_TABLES[0][(hi >> 24) as usize];
    }
    for &byte in chunks.remainder() {
        let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLES[0][idx];
    }
    !crc
}

/// Encode `[Magic "RFWL" 4B | Len u32 | CRC32C 4B | Seq u64 | JSON payload | '\n']`.
pub fn encode_wal_frame(seq: u64, payload: &[u8]) -> Result<Vec<u8>> {
    if payload.len() > WAL_MAX_FRAME_LEN {
        anyhow::bail!(
            "WAL frame payload {} exceeds 4 MiB limit ({WAL_MAX_FRAME_LEN})",
            payload.len()
        );
    }
    let mut body = Vec::with_capacity(8 + payload.len());
    body.extend_from_slice(&seq.to_le_bytes());
    body.extend_from_slice(payload);
    let checksum = crc32c(&body);

    let mut frame = Vec::with_capacity(WAL_HEADER_LEN + payload.len() + 1);
    frame.extend_from_slice(WAL_MAGIC);
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&checksum.to_le_bytes());
    frame.extend_from_slice(&body);
    frame.push(b'\n');
    Ok(frame)
}

fn sync_parent_dir_if_new(wal_path: &Path, existed: bool) {
    if existed {
        return;
    }
    #[cfg(unix)]
    if let Some(parent) = wal_path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
    #[cfg(not(unix))]
    let _ = wal_path;
}

/// Append a binary `RFWL` CRC32C frame with cross-OS durability barrier.
pub fn append_framed_record(wal_path: &Path, seq: u64, payload: &[u8]) -> Result<u64> {
    let existed = wal_path.exists();
    let frame = encode_wal_frame(seq, payload)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .with_context(|| format!("open WAL {}", wal_path.display()))?;
    file.write_all(&frame)?;
    sync_file_barrier(&file)?;
    let new_len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    sync_parent_dir_if_new(wal_path, existed);
    Ok(new_len)
}

/// Append a legacy JSONL WAL record with cross-OS durability barrier.
pub fn append_jsonl_record(wal_path: &Path, payload: &[u8]) -> Result<u64> {
    let existed = wal_path.exists();
    let mut line = Vec::with_capacity(payload.len() + 1);
    line.extend_from_slice(payload);
    line.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(wal_path)
        .with_context(|| format!("open WAL {}", wal_path.display()))?;
    file.write_all(&line)?;
    sync_file_barrier(&file)?;
    let new_len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    sync_parent_dir_if_new(wal_path, existed);
    Ok(new_len)
}

/// Auto-truncate a torn or zero-filled EOF tail and flush durability barrier.
pub fn truncate_torn_tail(wal_path: &Path, valid_pos: usize) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(wal_path)
        .with_context(|| format!("open WAL for truncate {}", wal_path.display()))?;
    file.set_len(valid_pos as u64)?;
    sync_file_barrier(&file)?;
    Ok(())
}

/// Detect whether an existing WAL file begins with binary `RFWL` framing.
pub fn file_uses_frames(wal_path: &Path) -> bool {
    let Ok(mut file) = File::open(wal_path) else {
        return false;
    };
    let mut head = [0u8; 4];
    file.read_exact(&mut head).is_ok() && &head == WAL_MAGIC
}

#[inline]
fn is_all_zero(slice: &[u8]) -> bool {
    slice.iter().all(|&byte| byte == 0)
}

fn trim_ascii_slice(mut slice: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = slice.split_first() {
        if first.is_ascii_whitespace() {
            slice = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = slice.split_last() {
        if last.is_ascii_whitespace() {
            slice = rest;
        } else {
            break;
        }
    }
    slice
}

fn parse_frame_header(buf: &[u8], pos: usize) -> Option<(usize, u32, u64)> {
    let hdr = buf.get(pos..pos + WAL_HEADER_LEN)?;
    if &hdr[0..4] != WAL_MAGIC {
        return None;
    }
    let len = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let crc = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
    let seq = u64::from_le_bytes([
        hdr[12], hdr[13], hdr[14], hdr[15], hdr[16], hdr[17], hdr[18], hdr[19],
    ]);
    Some((len, crc, seq))
}

fn is_valid_framed_at(buf: &[u8], pos: usize) -> bool {
    let Some((len, expected_crc, _)) = parse_frame_header(buf, pos) else {
        return false;
    };
    let end = pos.saturating_add(WAL_HEADER_LEN).saturating_add(len);
    len <= WAL_MAX_FRAME_LEN
        && end < buf.len()
        && buf[end] == b'\n'
        && crc32c(&buf[pos + 12..end]) == expected_crc
}

fn has_valid_record_after(buf: &[u8], from: usize) -> bool {
    let start = from.saturating_add(1);
    if start >= buf.len() {
        return false;
    }
    let tail = &buf[start..];
    for rel in memmem::find_iter(tail, WAL_MAGIC) {
        if is_valid_framed_at(buf, start + rel) {
            return true;
        }
    }

    let mut checked = 0usize;
    for rel in memchr::memchr_iter(b'{', tail) {
        checked += 1;
        if checked > 64 {
            break;
        }
        let pos = start + rel;
        let line_end = memchr(b'\n', &buf[pos..]).map_or(buf.len(), |off| pos + off);
        let trimmed = trim_ascii_slice(&buf[pos..line_end]);
        if !trimmed.is_empty() && serde_json::from_slice::<WalRecord>(trimmed).is_ok() {
            return true;
        }
    }
    false
}

#[inline]
fn check_monotonic_seq(seq: u64, valid_records: usize, last_framed_seq: u64) -> bool {
    if last_framed_seq > 0 {
        seq == last_framed_seq + 1
    } else {
        seq == 1 || seq == (valid_records as u64) + 1
    }
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
        if has_valid_record_after(buf, cursor) {
            return Err(WalHaltError::MidFileCorruption(cursor));
        }
        return Ok(StepOutcome::StopAtTornTail);
    }

    let actual_crc = crc32c(&buf[cursor + 12..frame_end]);
    let nl_ok = buf[frame_end] == b'\n';
    let seq_ok = check_monotonic_seq(seq, valid_records, last_framed_seq);
    if actual_crc != expected_crc || !nl_ok {
        if frame_end + 1 < buf.len() && !is_all_zero(&buf[frame_end + 1..]) {
            return Err(WalHaltError::MidFileCorruption(cursor));
        }
        return Ok(StepOutcome::StopAtTornTail);
    }
    if !seq_ok {
        return Err(WalHaltError::MidFileCorruption(cursor));
    }

    let payload = &buf[cursor + WAL_HEADER_LEN..frame_end];
    let rec = serde_json::from_slice::<WalRecord>(payload)
        .map_err(|_| WalHaltError::MidFileCorruption(cursor))?;
    Ok(StepOutcome::Record {
        rec,
        next_cursor: frame_end + 1,
        framed_seq: Some(seq),
    })
}

fn has_fused_tail(trimmed: &[u8]) -> bool {
    memmem::find(trimmed, b"}{").is_some() || memmem::find(trimmed, WAL_MAGIC).is_some()
}

fn parse_jsonl_step(buf: &[u8], cursor: usize) -> std::result::Result<StepOutcome, WalHaltError> {
    let Some(rel_nl) = memchr(b'\n', &buf[cursor..]) else {
        if has_valid_record_after(buf, cursor) {
            return Err(WalHaltError::MidFileCorruption(cursor));
        }
        return Ok(StepOutcome::StopAtTornTail);
    };

    let line_end = cursor + rel_nl;
    let trimmed = trim_ascii_slice(&buf[cursor..line_end]);
    if trimmed.is_empty() {
        return Ok(StepOutcome::Skip(line_end + 1));
    }
    if let Ok(rec) = serde_json::from_slice::<WalRecord>(trimmed) {
        return Ok(StepOutcome::Record {
            rec,
            next_cursor: line_end + 1,
            framed_seq: None,
        });
    }
    let has_trailing = line_end + 1 < buf.len() && !is_all_zero(&buf[line_end + 1..]);
    if has_trailing || has_fused_tail(trimmed) || has_valid_record_after(buf, cursor) {
        return Err(WalHaltError::MidFileCorruption(cursor));
    }
    Ok(StepOutcome::StopAtTornTail)
}

/// Scan dual-format WAL bytes (`RFWL` binary frames and legacy JSONL seamlessly).
/// Clean-halts with `WalHaltError::MidFileCorruption` on any mid-file corruption or line fusion.
pub fn scan_wal_bytes(
    buf: &[u8],
    on_rec: &mut impl FnMut(WalRecord),
) -> std::result::Result<WalScanStats, WalHaltError> {
    let (mut cursor, mut valid_pos, mut valid_records) = (0usize, 0usize, 0usize);
    let (mut last_framed_seq, mut has_framed) = (0u64, false);

    while cursor < buf.len() {
        if is_all_zero(&buf[cursor..]) {
            break;
        }
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

    let next_seq = if has_framed {
        last_framed_seq + 1
    } else {
        (valid_records as u64) + 1
    };
    Ok(WalScanStats {
        valid_records,
        valid_pos,
        file_len: buf.len(),
        truncated_bytes: buf.len().saturating_sub(valid_pos),
        next_seq,
        has_framed,
    })
}

/// Scan a dual-format WAL file from disk.
pub fn scan_wal_file(path: &Path, mut on_rec: impl FnMut(WalRecord)) -> Result<WalScanStats> {
    if !path.exists() {
        return Ok(WalScanStats {
            next_seq: 1,
            ..WalScanStats::default()
        });
    }
    let buf = fs::read(path).with_context(|| format!("read WAL {}", path.display()))?;
    scan_wal_bytes(&buf, &mut on_rec).map_err(anyhow::Error::new)
}

/// Determine the next dense sequence number for appending a new framed record.
pub fn wal_next_seq(path: &Path) -> Result<u64> {
    let stats = scan_wal_file(path, |_| {})?;
    Ok(stats.next_seq.max(1))
}
