//! ST-V3 front-coded (`k = 8`) 4-16 KiB SST blocks with embedded `u16` sparse restart index,
//! hybrid Roaring/sparse compressed postings, `10k`-LRU frozen-package offload, and atomic SST flush.

use super::genome::check_wipe_guard;
use super::parse::FileSymbols;
use crate::graph::{InterfaceCutoff, check_interface_cutoff};
use crate::mvcc::framed_wal::crc32c;
use crate::platform::io::sync_file_barrier;
use crate::platform::{EpochRegistry, ensure_cachedir_tag, replace_file_pub};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub const SST_MAGIC: &[u8; 8] = b"RFST0003";
pub const RESTART_INTERVAL: usize = 8;
pub const TARGET_BLOCK_BYTES: usize = 8 * 1024;
pub const MAX_BLOCK_BYTES: usize = 16 * 1024;
pub const MAX_BLOCK_KEYS: usize = 256;
pub const ROARING_THRESHOLD: usize = 4_096;
pub const DEFAULT_MAX_FROZEN_ENTRIES: usize = 10_000;

const MODE_KEYS_ONLY: u8 = 0;
const MODE_KEY_VALUE: u8 = 1;
const SST_HEADER_LEN: usize = 30;
const VERSION_KEY: &[u8] = b"\0__version__";

static SST_TMP_SEQ: AtomicU64 = AtomicU64::new(1);

/// Encode an unsigned 32-bit integer as LEB128 varint.
#[inline]
pub fn write_varint(out: &mut Vec<u8>, mut val: u32) {
    while val >= 0x80 {
        out.push(((val as u8) & 0x7F) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

/// Decode a LEB128 varint from `buf` starting at `pos`, returning `None` on truncation/overflow.
#[inline]
pub fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let mut out = 0u32;
    let mut shift = 0u32;
    while *pos < buf.len() {
        let byte = buf[*pos];
        *pos += 1;
        if shift >= 32 {
            return None;
        }
        out |= ((byte & 0x7F) as u32) << shift;
        if (byte & 0x80) == 0 {
            return Some(out);
        }
        shift += 7;
    }
    None
}

#[inline]
fn common_prefix_len(lhs: &[u8], rhs: &[u8]) -> usize {
    let limit = lhs.len().min(rhs.len());
    let mut shared = 0usize;
    while shared < limit && lhs[shared] == rhs[shared] {
        shared += 1;
    }
    shared
}

/// Encode sorted `(key, value)` entries into a front-coded (`k = 8`) block with `u16` restart tail.
pub fn encode_front_coded_block(entries: &[(&[u8], &[u8])], key_only: bool) -> Vec<u8> {
    let mut data = Vec::with_capacity(entries.len().saturating_mul(18).max(64));
    let mut restarts: Vec<u16> = Vec::with_capacity((entries.len() / RESTART_INTERVAL) + 1);
    let mut prev: &[u8] = b"";

    for (idx, &(key, val)) in entries.iter().enumerate() {
        let shared = if idx % RESTART_INTERVAL == 0 {
            restarts.push(data.len() as u16);
            0usize
        } else {
            common_prefix_len(prev, key)
        };
        let suffix = &key[shared..];
        write_varint(&mut data, shared as u32);
        write_varint(&mut data, suffix.len() as u32);
        data.extend_from_slice(suffix);
        if !key_only {
            write_varint(&mut data, val.len() as u32);
            data.extend_from_slice(val);
        }
        prev = key;
    }

    for &off in &restarts {
        data.extend_from_slice(&off.to_le_bytes());
    }
    data.extend_from_slice(&(restarts.len() as u16).to_le_bytes());
    data.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    data
}

fn block_tail_meta(block: &[u8]) -> Option<(usize, usize, usize)> {
    if block.len() < 4 {
        return None;
    }
    let tail = block.len();
    let n_keys = u16::from_le_bytes([block[tail - 2], block[tail - 1]]) as usize;
    let n_rest = u16::from_le_bytes([block[tail - 4], block[tail - 3]]) as usize;
    let rest_bytes = n_rest.checked_mul(2)?.checked_add(4)?;
    if n_rest == 0 || rest_bytes > tail {
        return None;
    }
    Some((n_keys, n_rest, tail - rest_bytes))
}

#[inline]
fn read_restart_offset(block: &[u8], rest_start: usize, idx: usize) -> Option<usize> {
    let off = rest_start.checked_add(idx.checked_mul(2)?)?;
    let bytes = block.get(off..off + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]) as usize)
}

fn read_restart_key(block: &[u8], offset: usize) -> Option<&[u8]> {
    let mut pos = offset;
    let _shared = read_varint(block, &mut pos)?;
    let slen = read_varint(block, &mut pos)? as usize;
    block.get(pos..pos.checked_add(slen)?)
}

fn binary_search_restarts(block: &[u8], rest_start: usize, n_rest: usize, target: &[u8]) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = n_rest;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let off = read_restart_offset(block, rest_start, mid)?;
        let key = read_restart_key(block, off)?;
        if key <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// Point-lookup `target` inside a front-coded block via `u16` sparse restart binary search.
pub fn lookup_front_coded_block<'a>(block: &'a [u8], target: &[u8], key_only: bool) -> Option<&'a [u8]> {
    let (n_keys, n_rest, rest_start) = block_tail_meta(block)?;
    let lo = binary_search_restarts(block, rest_start, n_rest, target)?;
    let mut pos = read_restart_offset(block, rest_start, lo)?;
    let mut cur = Vec::with_capacity(64);
    let max_steps = n_keys.saturating_sub(lo * RESTART_INTERVAL).min(RESTART_INTERVAL);

    for _ in 0..max_steps {
        let shared = read_varint(block, &mut pos)? as usize;
        let slen = read_varint(block, &mut pos)? as usize;
        let suffix = block.get(pos..pos.checked_add(slen)?)?;
        pos += slen;
        let val = if key_only {
            &[][..]
        } else {
            let vlen = read_varint(block, &mut pos)? as usize;
            let vslice = block.get(pos..pos.checked_add(vlen)?)?;
            pos += vlen;
            vslice
        };
        if shared > cur.len() {
            return None;
        }
        cur.truncate(shared);
        cur.extend_from_slice(suffix);
        match cur.as_slice().cmp(target) {
            Ordering::Equal => return Some(val),
            Ordering::Greater => return None,
            Ordering::Less => {}
        }
    }
    None
}

fn decode_front_coded_block(
    block: &[u8],
    key_only: bool,
    out: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Option<()> {
    let (n_keys, _, _) = block_tail_meta(block)?;
    let mut pos = 0usize;
    let mut cur = Vec::with_capacity(64);

    for _ in 0..n_keys {
        let shared = read_varint(block, &mut pos)? as usize;
        let slen = read_varint(block, &mut pos)? as usize;
        let suffix = block.get(pos..pos.checked_add(slen)?)?;
        pos += slen;
        let val = if key_only {
            Vec::new()
        } else {
            let vlen = read_varint(block, &mut pos)? as usize;
            let vslice = block.get(pos..pos.checked_add(vlen)?)?;
            pos += vlen;
            vslice.to_vec()
        };
        if shared > cur.len() {
            return None;
        }
        cur.truncate(shared);
        cur.extend_from_slice(suffix);
        out.push((cur.clone(), val));
    }
    Some(())
}

#[derive(Debug, Clone)]
struct BlockAnchor {
    first_key: Vec<u8>,
    offset: u32,
    len: u32,
}

/// Immutable front-coded SST table with block-level binary search and CRC32C integrity check.
#[derive(Debug, Clone)]
pub struct SstTable {
    raw: Vec<u8>,
    anchors: Vec<BlockAnchor>,
    key_only: bool,
    entry_count: usize,
}

fn next_block_end(entries: &[(&[u8], &[u8])], start: usize, key_only: bool) -> usize {
    let mut est_bytes = 0usize;
    let mut end = start;
    while end < entries.len() && (end - start) < MAX_BLOCK_KEYS {
        let (key, val) = entries[end];
        let entry_est = key.len() + if key_only { 4 } else { val.len() + 8 };
        if end > start && (est_bytes >= TARGET_BLOCK_BYTES || est_bytes + entry_est > MAX_BLOCK_BYTES) {
            break;
        }
        est_bytes += entry_est;
        end += 1;
    }
    end
}

fn write_sst_header(
    raw: &mut [u8],
    key_only: bool,
    block_count: usize,
    entry_count: usize,
    index_offset: u64,
    checksum: u32,
) {
    raw[0..8].copy_from_slice(SST_MAGIC);
    raw[8] = if key_only { MODE_KEYS_ONLY } else { MODE_KEY_VALUE };
    raw[9] = 0;
    raw[10..14].copy_from_slice(&(block_count as u32).to_le_bytes());
    raw[14..18].copy_from_slice(&(entry_count as u32).to_le_bytes());
    raw[18..26].copy_from_slice(&index_offset.to_le_bytes());
    raw[26..30].copy_from_slice(&checksum.to_le_bytes());
}

fn parse_sst_anchors(raw: &[u8], index_offset: usize, block_count: usize) -> Option<Vec<BlockAnchor>> {
    let mut pos = index_offset;
    let mut anchors = Vec::with_capacity(block_count.min(4096));
    for _ in 0..block_count {
        let klen = read_varint(raw, &mut pos)? as usize;
        let first_key = raw.get(pos..pos.checked_add(klen)?)?.to_vec();
        pos += klen;
        let off_b = raw.get(pos..pos + 4)?;
        pos += 4;
        let len_b = raw.get(pos..pos + 4)?;
        pos += 4;
        let offset = u32::from_le_bytes([off_b[0], off_b[1], off_b[2], off_b[3]]);
        let len = u32::from_le_bytes([len_b[0], len_b[1], len_b[2], len_b[3]]);
        let end = (offset as usize).checked_add(len as usize)?;
        if end > index_offset {
            return None;
        }
        anchors.push(BlockAnchor { first_key, offset, len });
    }
    Some(anchors)
}

impl SstTable {
    /// Build a front-coded SST from sorted `(key, value)` pairs.
    pub fn build(entries: &[(&[u8], &[u8])]) -> Self {
        let key_only = entries.iter().all(|(_, val)| val.is_empty());
        let mut raw = vec![0u8; SST_HEADER_LEN];
        let mut anchors = Vec::new();
        let mut idx = 0usize;

        while idx < entries.len() {
            let end = next_block_end(entries, idx, key_only);
            let block = encode_front_coded_block(&entries[idx..end], key_only);
            let offset = raw.len() as u32;
            let len = block.len() as u32;
            anchors.push(BlockAnchor {
                first_key: entries[idx].0.to_vec(),
                offset,
                len,
            });
            raw.extend_from_slice(&block);
            idx = end;
        }

        let index_offset = raw.len() as u64;
        for anchor in &anchors {
            write_varint(&mut raw, anchor.first_key.len() as u32);
            raw.extend_from_slice(&anchor.first_key);
            raw.extend_from_slice(&anchor.offset.to_le_bytes());
            raw.extend_from_slice(&anchor.len.to_le_bytes());
        }

        let checksum = crc32c(&raw[SST_HEADER_LEN..]);
        write_sst_header(&mut raw, key_only, anchors.len(), entries.len(), index_offset, checksum);
        Self {
            raw,
            anchors,
            key_only,
            entry_count: entries.len(),
        }
    }

    /// Decode and verify an SST image from raw bytes.
    pub fn from_bytes(raw: Vec<u8>) -> Option<Self> {
        if raw.len() < SST_HEADER_LEN || &raw[0..8] != SST_MAGIC {
            return None;
        }
        let key_only = match raw[8] {
            MODE_KEYS_ONLY => true,
            MODE_KEY_VALUE => false,
            _ => return None,
        };
        let block_count = u32::from_le_bytes([raw[10], raw[11], raw[12], raw[13]]) as usize;
        let entry_count = u32::from_le_bytes([raw[14], raw[15], raw[16], raw[17]]) as usize;
        let index_offset = u64::from_le_bytes([
            raw[18], raw[19], raw[20], raw[21], raw[22], raw[23], raw[24], raw[25],
        ]) as usize;
        let expected_crc = u32::from_le_bytes([raw[26], raw[27], raw[28], raw[29]]);

        if index_offset < SST_HEADER_LEN || index_offset > raw.len() {
            return None;
        }
        if crc32c(&raw[SST_HEADER_LEN..]) != expected_crc {
            return None;
        }
        let anchors = parse_sst_anchors(&raw, index_offset, block_count)?;
        Some(Self { raw, anchors, key_only, entry_count })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }

    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Point-lookup `key` across SST blocks in `O(log blocks + log restarts)`.
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        if self.anchors.is_empty() {
            return None;
        }
        let block_idx = match self.anchors.binary_search_by(|a| a.first_key.as_slice().cmp(key)) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(pos) => pos - 1,
        };
        let anchor = &self.anchors[block_idx];
        let start = anchor.offset as usize;
        let end = start + (anchor.len as usize);
        lookup_front_coded_block(&self.raw[start..end], key, self.key_only)
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Decode all `(key, value)` entries in sorted order.
    pub fn decode_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::with_capacity(self.entry_count);
        for anchor in &self.anchors {
            let start = anchor.offset as usize;
            let end = start + (anchor.len as usize);
            if decode_front_coded_block(&self.raw[start..end], self.key_only, &mut out).is_none() {
                break;
            }
        }
        out
    }
}

/// Atomic SST flush protocol: `.sst.tmp` -> `sync_file_barrier` -> atomic rename + `> 80%` wipe guard.
pub fn write_sst_atomic(
    path: &Path,
    table: &SstTable,
    prev_entries: usize,
    new_entries: usize,
) -> Result<bool> {
    if check_wipe_guard(prev_entries, new_entries) {
        tracing::warn!(
            prev_entries,
            new_entries,
            "SST index-wipe guard triggered (>80% drop); preserving prior SST"
        );
        return Ok(false);
    }

    let parent = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create SST dir {}", parent.display()))?;
    ensure_cachedir_tag(parent);

    let seq = SST_TMP_SEQ.fetch_add(1, AtomicOrdering::Relaxed);
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("sym.sst");
    let tmp_path = parent.join(format!(".{name}.{}.{seq}.sst.tmp", std::process::id()));

    {
        let mut file = File::create(&tmp_path)
            .with_context(|| format!("create SST temp {}", tmp_path.display()))?;
        file.write_all(table.as_bytes())?;
        sync_file_barrier(&file)?;
    }
    replace_file_pub(&tmp_path, path, parent)?;
    Ok(true)
}

/// Read and verify an SST file from disk.
pub fn read_sst_file(path: &Path) -> Option<SstTable> {
    let raw = fs::read(path).ok()?;
    SstTable::from_bytes(raw)
}

#[cfg(not(feature = "sst-storage"))]
#[derive(Debug, Clone, Default)]
pub struct CompactBitset {
    chunks: BTreeMap<u16, Box<[u64; 1024]>>,
    count: usize,
}

#[cfg(not(feature = "sst-storage"))]
impl CompactBitset {
    fn insert(&mut self, doc_id: u32) -> bool {
        let hi = (doc_id >> 16) as u16;
        let lo = (doc_id & 0xFFFF) as usize;
        let word = lo >> 6;
        let mask = 1u64 << (lo & 63);
        let chunk = self.chunks.entry(hi).or_insert_with(|| Box::new([0u64; 1024]));
        if (chunk[word] & mask) != 0 {
            return false;
        }
        chunk[word] |= mask;
        self.count += 1;
        true
    }

    fn contains(&self, doc_id: u32) -> bool {
        let hi = (doc_id >> 16) as u16;
        let lo = (doc_id & 0xFFFF) as usize;
        let Some(chunk) = self.chunks.get(&hi) else {
            return false;
        };
        (chunk[lo >> 6] & (1u64 << (lo & 63))) != 0
    }

    fn len(&self) -> usize {
        self.count
    }

    fn serialized_size_bytes(&self) -> usize {
        self.chunks.len().saturating_mul(8_196)
    }
}

/// Hybrid compressed postings: sorted `Vec<u32>` for `< 4096` docs, auto-promoting to
/// `roaring::RoaringBitmap` (under `sst-storage`) or chunked 64K-bit container fallback at `>= 4096`.
#[derive(Debug, Clone)]
pub enum HybridPostings {
    Sparse(Vec<u32>),
    #[cfg(feature = "sst-storage")]
    Roaring(roaring::RoaringBitmap),
    #[cfg(not(feature = "sst-storage"))]
    Bitset(CompactBitset),
}

impl Default for HybridPostings {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridPostings {
    pub fn new() -> Self {
        Self::Sparse(Vec::new())
    }

    pub fn insert(&mut self, doc_id: u32) -> bool {
        match self {
            Self::Sparse(vec) => {
                match vec.binary_search(&doc_id) {
                    Ok(_) => false,
                    Err(pos) => {
                        vec.insert(pos, doc_id);
                        if vec.len() >= ROARING_THRESHOLD {
                            self.promote_to_dense();
                        }
                        true
                    }
                }
            }
            #[cfg(feature = "sst-storage")]
            Self::Roaring(bm) => bm.insert(doc_id),
            #[cfg(not(feature = "sst-storage"))]
            Self::Bitset(bs) => bs.insert(doc_id),
        }
    }

    fn promote_to_dense(&mut self) {
        let Self::Sparse(vec) = self else {
            return;
        };
        #[cfg(feature = "sst-storage")]
        {
            let mut bm = roaring::RoaringBitmap::new();
            for &doc in vec.iter() {
                bm.insert(doc);
            }
            *self = Self::Roaring(bm);
        }
        #[cfg(not(feature = "sst-storage"))]
        {
            let mut bs = CompactBitset::default();
            for &doc in vec.iter() {
                bs.insert(doc);
            }
            *self = Self::Bitset(bs);
        }
    }

    pub fn contains(&self, doc_id: u32) -> bool {
        match self {
            Self::Sparse(vec) => vec.binary_search(&doc_id).is_ok(),
            #[cfg(feature = "sst-storage")]
            Self::Roaring(bm) => bm.contains(doc_id),
            #[cfg(not(feature = "sst-storage"))]
            Self::Bitset(bs) => bs.contains(doc_id),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Sparse(vec) => vec.len(),
            #[cfg(feature = "sst-storage")]
            Self::Roaring(bm) => bm.len() as usize,
            #[cfg(not(feature = "sst-storage"))]
            Self::Bitset(bs) => bs.len(),
        }
    }

    pub fn is_dense(&self) -> bool {
        !matches!(self, Self::Sparse(_))
    }

    pub fn serialized_size_bytes(&self) -> usize {
        match self {
            Self::Sparse(vec) => 4 + vec.len().saturating_mul(2),
            #[cfg(feature = "sst-storage")]
            Self::Roaring(bm) => bm.serialized_size(),
            #[cfg(not(feature = "sst-storage"))]
            Self::Bitset(bs) => bs.serialized_size_bytes(),
        }
    }
}

/// Compact resident summary kept for frozen/evicted packages in `FrozenPkgCache`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSummary {
    pub pkg_id: String,
    pub interface_blake3: String,
    pub exported_count: u32,
    pub file_count: u32,
    pub epoch: u64,
}

/// Full hot-package symbol detail offloaded once LRU capacity (`max_entries`) is exceeded.
#[derive(Debug, Clone)]
pub struct PkgDetail {
    pub pkg_id: String,
    pub interface_blake3: String,
    pub exported_symbols: Vec<String>,
    pub file_ids: HybridPostings,
    pub epoch: u64,
}

impl PkgDetail {
    pub fn to_summary(&self) -> ExportSummary {
        ExportSummary {
            pkg_id: self.pkg_id.clone(),
            interface_blake3: self.interface_blake3.clone(),
            exported_count: self.exported_symbols.len() as u32,
            file_count: self.file_ids.len() as u32,
            epoch: self.epoch,
        }
    }
}

struct HotPkgSlot {
    tick: u64,
    detail: PkgDetail,
}

/// `rust-glancer` bounded LRU (`max_entries = 10_000`) keeping only compact `ExportSummary`
/// resident for cold/frozen packages while offloading full symbol details.
pub struct FrozenPkgCache {
    max_entries: usize,
    tick: u64,
    hot: FxHashMap<String, HotPkgSlot>,
    lru_order: BTreeSet<(u64, String)>,
    summaries: FxHashMap<String, ExportSummary>,
    evictions: u64,
}

impl FrozenPkgCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            tick: 0,
            hot: FxHashMap::default(),
            lru_order: BTreeSet::new(),
            summaries: FxHashMap::default(),
            evictions: 0,
        }
    }

    pub fn with_default_cap() -> Self {
        Self::new(DEFAULT_MAX_FROZEN_ENTRIES)
    }

    pub fn global() -> &'static Mutex<Self> {
        static GLOBAL_CACHE: OnceLock<Mutex<FrozenPkgCache>> = OnceLock::new();
        GLOBAL_CACHE.get_or_init(|| Mutex::new(FrozenPkgCache::with_default_cap()))
    }

    /// Insert or update a hot package; returns `true` when `interface_blake3` changed.
    pub fn put_package(&mut self, detail: PkgDetail) -> bool {
        let pkg_id = detail.pkg_id.clone();
        let changed = self
            .summaries
            .get(&pkg_id)
            .is_none_or(|prev| prev.interface_blake3 != detail.interface_blake3);
        let summary = detail.to_summary();
        self.summaries.insert(pkg_id.clone(), summary);

        if let Some(old) = self.hot.remove(&pkg_id) {
            self.lru_order.remove(&(old.tick, pkg_id.clone()));
        }
        self.tick = self.tick.saturating_add(1);
        let now_tick = self.tick;
        self.lru_order.insert((now_tick, pkg_id.clone()));
        self.hot.insert(pkg_id, HotPkgSlot { tick: now_tick, detail });

        while self.hot.len() > self.max_entries {
            let Some((_, victim)) = self.lru_order.pop_first() else {
                break;
            };
            self.hot.remove(&victim);
            self.evictions = self.evictions.saturating_add(1);
        }
        changed
    }

    /// Touch and fetch a hot package detail, returning `None` if already offloaded to frozen summary.
    pub fn get_hot(&mut self, pkg_id: &str) -> Option<&PkgDetail> {
        let old_tick = self.hot.get(pkg_id)?.tick;
        self.lru_order.remove(&(old_tick, pkg_id.to_string()));
        self.tick = self.tick.saturating_add(1);
        let next_tick = self.tick;
        self.lru_order.insert((next_tick, pkg_id.to_string()));
        let slot = self.hot.get_mut(pkg_id)?;
        slot.tick = next_tick;
        Some(&slot.detail)
    }

    /// Always-resident compact `ExportSummary` lookup (works for both hot and offloaded packages).
    pub fn summary(&self, pkg_id: &str) -> Option<&ExportSummary> {
        self.summaries.get(pkg_id)
    }

    /// Salsa `interface_blake3` early-cutoff refresh wired to `graph/abi.rs::hash_abi` and `EpochRegistry`.
    pub fn refresh_package_interface(
        &mut self,
        pkg_id: &str,
        files: &[(&str, &str)],
        registry: &EpochRegistry,
    ) -> InterfaceCutoff {
        let prev = self.summaries.get(pkg_id).map(|sum| sum.interface_blake3.as_str());
        let cutoff = check_interface_cutoff(pkg_id, files, prev, registry);
        let entry = self.summaries.entry(pkg_id.to_string()).or_insert_with(|| ExportSummary {
            pkg_id: pkg_id.to_string(),
            interface_blake3: cutoff.interface_blake3.clone(),
            exported_count: 0,
            file_count: files.len() as u32,
            epoch: cutoff.epoch,
        });
        entry.interface_blake3 = cutoff.interface_blake3.clone();
        entry.file_count = files.len() as u32;
        entry.epoch = cutoff.epoch;
        cutoff
    }

    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    pub fn summary_len(&self) -> usize {
        self.summaries.len()
    }

    pub fn evicted_count(&self) -> u64 {
        self.evictions
    }
}

/// Derive a package partition key (`PkgId`) from a repository-relative path.
pub fn pkg_id_for_rel(rel: &str) -> String {
    let mut parts = rel.split('/');
    let Some(first) = parts.next() else {
        return "root".into();
    };
    if matches!(first, "packages" | "crates" | "apps" | "libs" | "services")
        && let Some(second) = parts.next()
        && !second.is_empty()
    {
        return format!("{first}/{second}");
    }
    if parts.next().is_some() {
        first.to_string()
    } else {
        "root".into()
    }
}

/// Load cached `FileSymbols` from a front-coded `sym.sst` if version matches.
pub fn load_symbol_sst(path: &Path, expected_version: &str) -> Option<BTreeMap<String, FileSymbols>> {
    let table = read_sst_file(path)?;
    if table.get(VERSION_KEY) != Some(expected_version.as_bytes()) {
        return None;
    }
    let mut files = BTreeMap::new();
    for (key, val) in table.decode_all() {
        if key.as_slice() == VERSION_KEY {
            continue;
        }
        let rel = String::from_utf8(key).ok()?;
        let sym: FileSymbols = serde_json::from_slice(&val).ok()?;
        files.insert(rel, sym);
    }
    Some(files)
}

/// Persist `FileSymbols` into a front-coded `sym.sst` with atomic flush and `> 80%` wipe guard.
pub fn save_symbol_sst(
    path: &Path,
    version: &str,
    files: &BTreeMap<String, FileSymbols>,
    prev_entries: usize,
) -> Result<bool> {
    let mut encoded: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(files.len() + 1);
    encoded.push((VERSION_KEY.to_vec(), version.as_bytes().to_vec()));
    for (rel, sym) in files {
        let bytes = serde_json::to_vec(sym)?;
        encoded.push((rel.as_bytes().to_vec(), bytes));
    }
    encoded.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let refs: Vec<(&[u8], &[u8])> = encoded
        .iter()
        .map(|(key, val)| (key.as_slice(), val.as_slice()))
        .collect();
    let table = SstTable::build(&refs);
    let written = write_sst_atomic(path, &table, prev_entries, files.len())?;
    if written && (!table.contains(VERSION_KEY) || table.entry_count() != refs.len()) {
        anyhow::bail!("SST self-check failed for {}", path.display());
    }
    Ok(written)
}

/// Refresh `FrozenPkgCache` summaries & hybrid postings from the latest indexed `FileSymbols`.
pub fn update_frozen_pkg_cache(files: &BTreeMap<String, FileSymbols>) {
    let mut by_pkg: BTreeMap<String, (Vec<String>, HybridPostings, blake3::Hasher)> = BTreeMap::new();
    for (idx, (rel, sym)) in files.iter().enumerate() {
        let pkg_id = pkg_id_for_rel(rel);
        let entry = by_pkg
            .entry(pkg_id)
            .or_insert_with(|| (Vec::new(), HybridPostings::new(), blake3::Hasher::new()));
        entry.0.extend(sym.defs.iter().cloned());
        entry.1.insert(idx as u32);
        entry.2.update(rel.as_bytes());
        entry.2.update(sym.blake3.as_bytes());
    }

    let reg = EpochRegistry::global();
    let mut cache = FrozenPkgCache::global().lock();
    for (pkg_id, (exported_symbols, file_ids, hasher)) in by_pkg {
        let iface_hash = hasher.finalize().to_hex().to_string();
        let prev_epoch = reg.try_read(&pkg_id).map_or_else(|| reg.global_epoch(), |(ep, _)| ep);
        let detail = PkgDetail {
            pkg_id: pkg_id.clone(),
            interface_blake3: iface_hash,
            exported_symbols,
            file_ids,
            epoch: prev_epoch,
        };
        let changed = cache.put_package(detail);
        if changed {
            let _ = reg.publish(&pkg_id, cache.summary_len() as u64);
        }
        if let Some(hot) = cache.get_hot(&pkg_id) {
            tracing::trace!(
                pkg = %hot.pkg_id,
                dense = hot.file_ids.is_dense(),
                postings_bytes = hot.file_ids.serialized_size_bytes(),
                has_zero = hot.file_ids.contains(0),
                "hot package refreshed"
            );
        }
        if let Some(sum) = cache.summary(&pkg_id) {
            tracing::trace!(
                pkg = %sum.pkg_id,
                exports = sum.exported_count,
                files = sum.file_count,
                epoch = sum.epoch,
                "frozen package summary resident"
            );
        }
    }
    tracing::trace!(
        hot = cache.hot_len(),
        summaries = cache.summary_len(),
        evictions = cache.evicted_count(),
        "frozen package cache stats"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn make_synthetic_paths(count: usize) -> Vec<String> {
        let mut paths: Vec<String> = (0..count)
            .map(|idx| format!("packages/pkg_{:02}/src/sub_{:02}/mod_{:04}.rs", idx % 40, idx % 10, idx))
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn sst_front_coded_10x_shrinkage_roaring_and_10k_lru() {
        let tmp = TempRepo::new("sst_v3_all");
        let paths = make_synthetic_paths(5_000);

        // 1. Compare JSON index size vs front-coded (k = 8) SST size (>= 8x, typically > 11x shrinkage).
        let mut json_buf = String::from("{\"version\":\"0.6.0\",\"files\":{");
        for (idx, path) in paths.iter().enumerate() {
            if idx > 0 {
                json_buf.push(',');
            }
            json_buf.push_str(&format!(
                "\"{path}\":{{\"blake3\":\"{:064x}\",\"defs\":[\"fn_{idx}\",\"ty_{idx}\"],\"imports\":[\"mod_{}\"]}}",
                idx as u64,
                idx % 64
            ));
        }
        json_buf.push_str("}}");

        let key_entries: Vec<(&[u8], &[u8])> = paths.iter().map(|p| (p.as_bytes(), &b""[..])).collect();
        let table = SstTable::build(&key_entries);
        let shrinkage = (json_buf.len() as f64) / (table.as_bytes().len().max(1) as f64);
        assert!(
            shrinkage >= 8.0,
            "expected >= 8x shrinkage vs JSON, got {shrinkage:.2}x (json={}B, sst={}B)",
            json_buf.len(),
            table.as_bytes().len()
        );

        // Atomic write + reload + 100% lookup parity across all 5,000 keys.
        let sst_path = tmp.join(".refit/cache/sym.sst");
        assert!(write_sst_atomic(&sst_path, &table, 0, paths.len()).unwrap());
        let loaded = read_sst_file(&sst_path).expect("load written SST");
        assert_eq!(loaded.entry_count(), paths.len());
        for path in &paths {
            assert!(loaded.contains(path.as_bytes()), "missing key {path}");
        }
        assert!(!loaded.contains(b"packages/pkg_99/src/nonexistent.rs"));

        // Tamper detection via CRC32C.
        let mut tampered = loaded.as_bytes().to_vec();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0x01;
        assert!(SstTable::from_bytes(tampered).is_none(), "CRC32C must reject tampered SST");

        // > 80% wipe guard refuses overwriting 100 entries with 10 entries.
        let small = SstTable::build(&key_entries[..10]);
        assert!(!write_sst_atomic(&sst_path, &small, 100, 10).unwrap());
        assert_eq!(read_sst_file(&sst_path).unwrap().entry_count(), paths.len());

        // Key-value roundtrip parity.
        let kv_vals: Vec<String> = (0..128).map(|i| format!("payload_value_{i:04}")).collect();
        let kv_refs: Vec<(&[u8], &[u8])> = paths[..128]
            .iter()
            .zip(&kv_vals)
            .map(|(k, v)| (k.as_bytes(), v.as_bytes()))
            .collect();
        let kv_table = SstTable::build(&kv_refs);
        for (k, v) in &kv_refs {
            assert_eq!(kv_table.get(k), Some(*v));
        }
        assert_eq!(kv_table.decode_all().len(), 128);

        // 2. HybridPostings: sparse < 4096 -> dense (Roaring / CompactBitset) at >= 4096.
        let mut postings = HybridPostings::new();
        for doc in 0..4_095u32 {
            assert!(postings.insert(doc));
        }
        assert!(!postings.is_dense());
        assert_eq!(postings.len(), 4_095);
        assert!(postings.insert(4_095));
        assert!(postings.is_dense(), "must promote to dense postings at 4096 docs");
        for doc in 4_096..6_000u32 {
            assert!(postings.insert(doc));
        }
        assert_eq!(postings.len(), 6_000);
        assert!(postings.contains(0) && postings.contains(4_095) && postings.contains(5_999));
        assert!(!postings.contains(6_000));
        let raw_u32_bytes = 6_000 * 4;
        assert!(
            postings.serialized_size_bytes() * 2 <= raw_u32_bytes,
            "dense postings {}B should be >= 2x smaller than u32 vec {}B",
            postings.serialized_size_bytes(),
            raw_u32_bytes
        );

        // 3. FrozenPkgCache 10k-LRU offload keeping compact ExportSummary resident for cold packages.
        let mut lru = FrozenPkgCache::with_default_cap();
        for idx in 0..10_050usize {
            let pkg_id = format!("packages/p_{idx:05}");
            let mut fids = HybridPostings::new();
            fids.insert(idx as u32);
            lru.put_package(PkgDetail {
                pkg_id,
                interface_blake3: format!("{idx:064x}"),
                exported_symbols: vec![format!("sym_{idx}")],
                file_ids: fids,
                epoch: 1,
            });
            // Keep p_00000 hot by touching it right before the 10k LRU boundary is reached.
            if idx == DEFAULT_MAX_FROZEN_ENTRIES - 1 {
                assert!(lru.get_hot("packages/p_00000").is_some());
            }
        }
        assert_eq!(lru.hot_len(), DEFAULT_MAX_FROZEN_ENTRIES);
        assert_eq!(lru.evicted_count(), 50);
        assert_eq!(lru.summary_len(), 10_050);
        // p_00000 was touched so it stays hot; p_00001 was evicted but its ExportSummary stays resident!
        assert!(lru.get_hot("packages/p_00000").is_some());
        assert!(lru.get_hot("packages/p_00001").is_none());
        let cold_sum = lru.summary("packages/p_00001").expect("evicted package keeps ExportSummary");
        assert_eq!(cold_sum.exported_count, 1);
        assert_eq!(cold_sum.file_count, 1);

        // 4. Salsa interface_blake3 early cutoff wired to graph/abi.rs::hash_abi and EpochRegistry.
        let reg = EpochRegistry::new();
        let v1_body = "pub fn compute(x: i32) -> i32 { x + 1 }\n";
        let v2_same_abi = "pub fn compute(x: i32) -> i32 { let y = x * 2; y - x + 1 }\n";
        let v3_changed_abi = "pub fn compute(x: i32, scale: i32) -> i32 { x * scale }\n";

        let c1 = lru.refresh_package_interface("packages/calc", &[("src/calc.rs", v1_body)], &reg);
        assert!(c1.changed);
        let c2 = lru.refresh_package_interface("packages/calc", &[("src/calc.rs", v2_same_abi)], &reg);
        assert!(!c2.changed, "body-only edit must trigger Salsa early cutoff");
        assert_eq!(c2.epoch, c1.epoch, "unchanged ABI must not bump EpochRegistry");
        assert_eq!(c2.interface_blake3, c1.interface_blake3);

        let c3 = lru.refresh_package_interface("packages/calc", &[("src/calc.rs", v3_changed_abi)], &reg);
        assert!(c3.changed, "signature change must invalidate interface_blake3");
        assert!(c3.epoch > c2.epoch, "changed ABI must publish new epoch");
        assert_ne!(c3.interface_blake3, c2.interface_blake3);
    }
}
