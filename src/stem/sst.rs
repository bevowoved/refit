//! Leaf Zstd-compressed front-coded (`k = 8`) SST index (`RFST0004`) with lazy single-block
//! point-lookup, streaming BLAKE3 authentication, and cycle-free package ABI helpers.

use super::genome::check_wipe_guard;
use super::parse::FileSymbols;
use crate::platform::{ensure_cachedir_tag, write_atomic};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const SST_MAGIC: &[u8; 8] = b"RFST0004";
pub const SST_MIN_FILES: usize = 500;
pub const SST_MIN_INDEX_BYTES: u64 = 4 * 1024 * 1024;

const RESTART_INTERVAL: usize = 8;
const TARGET_BLOCK_BYTES: usize = 8 * 1024;
const MAX_BLOCK_BYTES: usize = 16 * 1024;
const MAX_BLOCK_KEYS: usize = 256;
const MAX_DECOMP_BLOCK_BYTES: usize = 16 * 1024 * 1024;
const SST_HEADER_LEN: usize = 30;
const VERSION_KEY: &[u8] = b"\0__version__";
const EDGES_KEY: &[u8] = b"\0__edges__";

/// Cycle-free package partition key derived from a repository-relative path.
#[cfg(test)]
pub fn pkg_id_for_rel(rel: &str) -> String {
    let mut parts = rel.split('/');
    let Some(first) = parts.next() else { return "root".into() };
    if matches!(first, "packages" | "crates" | "apps" | "libs" | "services")
        && let Some(second) = parts.next()
        && !second.is_empty()
    {
        return format!("{first}/{second}");
    }
    if parts.next().is_some() { first.to_string() } else { "root".into() }
}

/// Cycle-free plain `[u8; 32]` package ABI change check (zero `crate::graph` import).
#[cfg(test)]
pub fn package_abi_changed(prev: Option<&[u8; 32]>, next: &[u8; 32]) -> bool {
    prev != Some(next)
}

#[inline]
fn write_varint(out: &mut Vec<u8>, mut val: u32) {
    while val >= 0x80 {
        out.push(((val as u8) & 0x7F) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline]
fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let (mut out, mut shift) = (0u32, 0u32);
    while *pos < buf.len() && shift < 32 {
        let b = buf[*pos];
        *pos += 1;
        out |= ((b & 0x7F) as u32) << shift;
        if (b & 0x80) == 0 {
            return Some(out);
        }
        shift += 7;
    }
    None
}

fn encode_front_coded_block(entries: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut data = Vec::with_capacity(entries.len().saturating_mul(24).max(64));
    let mut restarts = Vec::with_capacity((entries.len() / RESTART_INTERVAL) + 1);
    let mut prev: &[u8] = b"";
    for (idx, &(key, val)) in entries.iter().enumerate() {
        let shared = if idx % RESTART_INTERVAL == 0 {
            restarts.push(data.len() as u16);
            0
        } else {
            prev.iter().zip(key).take_while(|(a, b)| a == b).count()
        };
        let suffix = &key[shared..];
        write_varint(&mut data, shared as u32);
        write_varint(&mut data, suffix.len() as u32);
        data.extend_from_slice(suffix);
        write_varint(&mut data, val.len() as u32);
        data.extend_from_slice(val);
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
    let tail = block.len();
    if tail < 4 {
        return None;
    }
    let n_keys = u16::from_le_bytes([block[tail - 2], block[tail - 1]]) as usize;
    let n_rest = u16::from_le_bytes([block[tail - 4], block[tail - 3]]) as usize;
    let rest_bytes = n_rest.checked_mul(2)?.checked_add(4)?;
    (n_rest > 0 && rest_bytes <= tail).then_some((n_keys, n_rest, tail - rest_bytes))
}

#[inline]
fn read_fc_entry<'a>(block: &'a [u8], pos: &mut usize, cur: &mut Vec<u8>) -> Option<&'a [u8]> {
    let shared = read_varint(block, pos)? as usize;
    let slen = read_varint(block, pos)? as usize;
    let suffix = block.get(*pos..pos.checked_add(slen)?)?;
    *pos += slen;
    let vlen = read_varint(block, pos)? as usize;
    let val = block.get(*pos..pos.checked_add(vlen)?)?;
    *pos += vlen;
    if shared > cur.len() {
        return None;
    }
    cur.truncate(shared);
    cur.extend_from_slice(suffix);
    Some(val)
}

fn lookup_front_coded_block<'a>(block: &'a [u8], target: &[u8]) -> Option<&'a [u8]> {
    let (n_keys, n_rest, rest_start) = block_tail_meta(block)?;
    let rest_off = |i: usize| -> Option<usize> {
        let o = rest_start.checked_add(i.checked_mul(2)?)?;
        let b = block.get(o..o + 2)?;
        Some(u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let (mut lo, mut hi) = (0usize, n_rest);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let mut p = rest_off(mid)?;
        let _shared = read_varint(block, &mut p)?;
        let slen = read_varint(block, &mut p)? as usize;
        if block.get(p..p.checked_add(slen)?)? <= target { lo = mid; } else { hi = mid; }
    }
    let mut pos = rest_off(lo)?;
    let mut cur = Vec::with_capacity(64);
    for _ in 0..n_keys.saturating_sub(lo * RESTART_INTERVAL).min(RESTART_INTERVAL) {
        let val = read_fc_entry(block, &mut pos, &mut cur)?;
        match cur.as_slice().cmp(target) {
            Ordering::Equal => return Some(val),
            Ordering::Greater => return None,
            Ordering::Less => {}
        }
    }
    None
}

fn decode_front_coded_block(block: &[u8], out: &mut Vec<(Vec<u8>, Vec<u8>)>) -> Option<()> {
    let (n_keys, _, _) = block_tail_meta(block)?;
    let (mut pos, mut cur) = (0usize, Vec::with_capacity(64));
    for _ in 0..n_keys {
        let val = read_fc_entry(block, &mut pos, &mut cur)?;
        out.push((cur.clone(), val.to_vec()));
    }
    Some(())
}

#[inline]
fn blake3_tag(chunks: &[&[u8]]) -> u32 {
    let mut h = blake3::Hasher::new();
    for chunk in chunks {
        h.update(chunk);
    }
    let d = h.finalize();
    let b = d.as_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

struct BlockAnchor {
    first_key: Vec<u8>,
    offset: u64,
    comp_len: u32,
    raw_len: u32,
    tag: u32,
}

fn parse_sst_anchors(footer: &[u8], index_offset: u64, block_count: usize) -> Option<Vec<BlockAnchor>> {
    let mut pos = 0usize;
    let mut anchors = Vec::with_capacity(block_count.min(4096));
    for _ in 0..block_count {
        let klen = read_varint(footer, &mut pos)? as usize;
        let first_key = footer.get(pos..pos.checked_add(klen)?)?.to_vec();
        pos += klen;
        let m = footer.get(pos..pos + 20)?;
        pos += 20;
        let offset = u64::from_le_bytes(m[0..8].try_into().ok()?);
        let comp_len = u32::from_le_bytes(m[8..12].try_into().ok()?);
        let raw_len = u32::from_le_bytes(m[12..16].try_into().ok()?);
        let tag = u32::from_le_bytes(m[16..20].try_into().ok()?);
        let end = offset.checked_add(comp_len as u64)?;
        if offset < SST_HEADER_LEN as u64 || end > index_offset || raw_len as usize > MAX_DECOMP_BLOCK_BYTES {
            return None;
        }
        anchors.push(BlockAnchor { first_key, offset, comp_len, raw_len, tag });
    }
    (pos == footer.len()).then_some(anchors)
}

/// Lazy disk-backed Zstd front-coded SST index (`RFST0004`).
pub struct SstIndex {
    file: File,
    anchors: Vec<BlockAnchor>,
    entry_count: usize,
    blocks_decoded: usize,
    last_block: Option<(usize, Vec<u8>)>,
}

impl SstIndex {
    /// Atomically write sorted `(key, value)` entries in Zstd-compressed front-coded (`k = 8`) 8–16 KiB blocks.
    pub fn write_atomic(path: &Path, entries: &[(&[u8], &[u8])], prev_entries: usize) -> Result<bool> {
        if check_wipe_guard(prev_entries, entries.len()) {
            tracing::warn!(prev_entries, new_entries = entries.len(), "SST wipe guard triggered; preserving prior SST");
            return Ok(false);
        }
        let mut raw = vec![0u8; SST_HEADER_LEN];
        let (mut anchors, mut idx) = (Vec::new(), 0usize);
        while idx < entries.len() {
            let (mut est, mut end) = (0usize, idx);
            while end < entries.len() && (end - idx) < MAX_BLOCK_KEYS {
                let sz = entries[end].0.len() + entries[end].1.len() + 8;
                if end > idx && (est >= TARGET_BLOCK_BYTES || est + sz > MAX_BLOCK_BYTES) {
                    break;
                }
                est += sz;
                end += 1;
            }
            let block = encode_front_coded_block(&entries[idx..end]);
            let comp = zstd::bulk::compress(&block, 1)?;
            let (offset, comp_len, raw_len) = (raw.len() as u64, u32::try_from(comp.len())?, u32::try_from(block.len())?);
            let tag = blake3_tag(&[&offset.to_le_bytes(), &comp_len.to_le_bytes(), &raw_len.to_le_bytes(), &comp]);
            anchors.push(BlockAnchor { first_key: entries[idx].0.to_vec(), offset, comp_len, raw_len, tag });
            raw.extend_from_slice(&comp);
            idx = end;
        }
        let index_offset = raw.len() as u64;
        for a in &anchors {
            write_varint(&mut raw, a.first_key.len() as u32);
            raw.extend_from_slice(&a.first_key);
            raw.extend_from_slice(&a.offset.to_le_bytes());
            raw.extend_from_slice(&a.comp_len.to_le_bytes());
            raw.extend_from_slice(&a.raw_len.to_le_bytes());
            raw.extend_from_slice(&a.tag.to_le_bytes());
        }
        raw[0..8].copy_from_slice(SST_MAGIC);
        raw[8] = 1;
        raw[10..14].copy_from_slice(&(anchors.len() as u32).to_le_bytes());
        raw[14..18].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        raw[18..26].copy_from_slice(&index_offset.to_le_bytes());
        let ftag = blake3_tag(&[&raw[0..26], &raw[index_offset as usize..]]);
        raw[26..30].copy_from_slice(&ftag.to_le_bytes());
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            ensure_cachedir_tag(parent);
        }
        write_atomic(path, &raw)?;
        Ok(true)
    }

    /// Open SST by reading ONLY the 30-byte header + compact anchor footer (`O(blocks)` RAM).
    pub fn open(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let file_len = file.metadata().ok()?.len();
        if file_len < SST_HEADER_LEN as u64 {
            return None;
        }
        let mut hdr = [0u8; SST_HEADER_LEN];
        file.read_exact(&mut hdr).ok()?;
        if &hdr[0..8] != SST_MAGIC || hdr[8] != 1 {
            return None;
        }
        let block_count = u32::from_le_bytes(hdr[10..14].try_into().ok()?) as usize;
        let entry_count = u32::from_le_bytes(hdr[14..18].try_into().ok()?) as usize;
        let index_offset = u64::from_le_bytes(hdr[18..26].try_into().ok()?);
        let expected_tag = u32::from_le_bytes(hdr[26..30].try_into().ok()?);
        if index_offset < SST_HEADER_LEN as u64 || index_offset > file_len {
            return None;
        }
        let mut footer = vec![0u8; usize::try_from(file_len - index_offset).ok()?];
        file.seek(SeekFrom::Start(index_offset)).ok()?;
        file.read_exact(&mut footer).ok()?;
        if blake3_tag(&[&hdr[0..26], &footer]) != expected_tag {
            return None;
        }
        let anchors = parse_sst_anchors(&footer, index_offset, block_count)?;
        Some(Self { file, anchors, entry_count, blocks_decoded: 0, last_block: None })
    }

    fn decode_block(&mut self, block_idx: usize) -> Option<&[u8]> {
        if !self.last_block.as_ref().is_some_and(|(idx, _)| *idx == block_idx) {
            let a = &self.anchors[block_idx];
            let mut comp = vec![0u8; a.comp_len as usize];
            self.file.seek(SeekFrom::Start(a.offset)).ok()?;
            self.file.read_exact(&mut comp).ok()?;
            let tag = blake3_tag(&[&a.offset.to_le_bytes(), &a.comp_len.to_le_bytes(), &a.raw_len.to_le_bytes(), &comp]);
            if tag != a.tag {
                return None;
            }
            let raw = zstd::bulk::decompress(&comp, a.raw_len as usize).ok()?;
            if raw.len() != a.raw_len as usize {
                return None;
            }
            self.blocks_decoded += 1;
            self.last_block = Some((block_idx, raw));
        }
        self.last_block.as_ref().map(|(_, buf)| buf.as_slice())
    }

    /// Lazy single-block point-lookup: binary-search anchors -> seek -> read & Zstd-decompress 1 block.
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let block_idx = match self.anchors.binary_search_by(|a| a.first_key.as_slice().cmp(key)) {
            Ok(exact) => exact,
            Err(0) => return None,
            Err(pos) => pos - 1,
        };
        lookup_front_coded_block(self.decode_block(block_idx)?, key).map(<[u8]>::to_vec)
    }

    /// Full sequential scan verifying every block's BLAKE3 tag before returning decoded entries.
    pub fn decode_all(&mut self) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::with_capacity(self.entry_count);
        for idx in 0..self.anchors.len() {
            decode_front_coded_block(self.decode_block(idx)?, &mut out)?;
        }
        (out.len() == self.entry_count).then_some(out)
    }

    #[cfg(test)]
    pub fn blocks_decoded(&self) -> usize {
        self.blocks_decoded
    }
}

pub fn load_symbol_sst(path: &Path, expected_version: &str) -> Option<BTreeMap<String, FileSymbols>> {
    let mut sst = SstIndex::open(path)?;
    if sst.get(VERSION_KEY).as_deref() != Some(expected_version.as_bytes()) {
        return None;
    }
    let mut files = BTreeMap::new();
    for (key, val) in sst.decode_all()? {
        if let Some(rel) = key.strip_prefix(b"f:") {
            files.insert(String::from_utf8(rel.to_vec()).ok()?, serde_json::from_slice(&val).ok()?);
        }
    }
    Some(files)
}

pub fn save_symbol_sst(
    path: &Path,
    version: &str,
    files: &BTreeMap<String, FileSymbols>,
    edges: &[(String, String)],
    runtime_edges: &[(String, String)],
    prev_entries: usize,
) -> Result<bool> {
    let mut encoded = Vec::with_capacity(files.len() + 2);
    encoded.push((VERSION_KEY.to_vec(), version.as_bytes().to_vec()));
    encoded.push((EDGES_KEY.to_vec(), serde_json::to_vec(&(edges, runtime_edges))?));
    for (rel, sym) in files {
        encoded.push((format!("f:{rel}").into_bytes(), serde_json::to_vec(sym)?));
    }
    encoded.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    let refs: Vec<(&[u8], &[u8])> = encoded.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    SstIndex::write_atomic(path, &refs, prev_entries)
}

pub struct FileSstLookup {
    pub sym: FileSymbols,
    pub edges: Vec<(String, String)>,
    pub runtime_edges: Vec<(String, String)>,
}

/// Lazy single-file point-lookup from `.refit/cache/sym.sst` without decoding the other `N - 1` files.
pub fn query_file_sst(repo_root: &Path, expected_version: &str, norm_rel: &str) -> Option<FileSstLookup> {
    let sst_path = repo_root.join(".refit/cache/sym.sst");
    let mut sst = sst_path.is_file().then(|| SstIndex::open(&sst_path))??;
    if sst.get(VERSION_KEY).as_deref() != Some(expected_version.as_bytes()) {
        return None;
    }
    let sym: FileSymbols = serde_json::from_slice(&sst.get(format!("f:{norm_rel}").as_bytes())?).ok()?;
    let disk_bytes = fs::read(repo_root.join(norm_rel)).ok()?;
    if blake3::hash(&disk_bytes).to_hex().as_str() != sym.blake3 {
        return None;
    }
    let (edges, runtime_edges) = serde_json::from_slice(&sst.get(EDGES_KEY)?).ok()?;
    Some(FileSstLookup { sym, edges, runtime_edges })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stem::{index_repo, query_symbols, should_use_sst};
    use crate::testutil::TempRepo;

    #[test]
    fn sst_zstd_lazy_lookup_and_threshold_gate() {
        assert_eq!(pkg_id_for_rel("packages/core/src/lib.rs"), "packages/core");
        assert_eq!(pkg_id_for_rel("src/lib.rs"), "src");
        assert_eq!(pkg_id_for_rel("lib.rs"), "root");
        let (h1, h2) = ([1u8; 32], [2u8; 32]);
        assert!(package_abi_changed(None, &h1) && !package_abi_changed(Some(&h1), &h1) && package_abi_changed(Some(&h1), &h2));

        let small: Vec<_> = (0..20).map(|i| (format!("src/s_{i:02}.rs"), format!("pub fn small_fn_{i}() -> i32 {{ {i} }}\n"))).collect();
        let srefs: Vec<_> = small.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let small_repo = TempRepo::with_files("sst_small", &srefs);
        let _ = index_repo(small_repo.path(), None);
        assert!(small_repo.join(".refit/cache/symbol_index.zst").is_file() && !small_repo.join(".refit/cache/sym.sst").exists());
        assert!(!should_use_sst(20, 4096));

        let mut big = vec![("src/m_000.rs".to_string(), "pub fn sym_000() -> i32 { 0 }\n".to_string())];
        for i in 1..520 {
            let p = i - 1;
            big.push((format!("src/m_{i:03}.rs"), format!("use crate::m_{p:03}::sym_{p:03};\npub fn sym_{i:03}() -> i32 {{ sym_{p:03}() + {i} }}\n")));
        }
        let brefs: Vec<_> = big.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let big_repo = TempRepo::with_files("sst_big", &brefs);
        let (disc, snap) = index_repo(big_repo.path(), None);
        let (sst_path, zst_path) = (big_repo.join(".refit/cache/sym.sst"), big_repo.join(".refit/cache/symbol_index.zst"));
        if cfg!(feature = "sst-storage") {
            assert!(should_use_sst(520, 4096) && sst_path.is_file() && !zst_path.exists());
        } else {
            let loaded = crate::platform::read_zstd::<serde_json::Value>(&zst_path).unwrap();
            let files: BTreeMap<String, FileSymbols> = serde_json::from_value(loaded["files"].clone()).unwrap();
            assert!(save_symbol_sst(&sst_path, "0.6.0", &files, &disc.edges, &disc.runtime_edges, 0).unwrap());
            let _ = fs::remove_file(&zst_path);
        }

        let sst_bytes = fs::read(&sst_path).unwrap();
        assert_eq!(&sst_bytes[0..8], SST_MAGIC);
        let raw_json: usize = SstIndex::open(&sst_path).unwrap().decode_all().unwrap().iter().map(|(k, v)| k.len() + v.len()).sum();
        assert!((sst_bytes.len() as f64) < (raw_json as f64) * 0.35, "sst={}B raw={}B", sst_bytes.len(), raw_json);

        let mut lazy = SstIndex::open(&sst_path).unwrap();
        assert_eq!(lazy.blocks_decoded(), 0);
        let sym_250: FileSymbols = serde_json::from_slice(&lazy.get(b"f:src/m_250.rs").unwrap()).unwrap();
        assert_eq!((lazy.blocks_decoded(), sym_250.defs), (1, vec!["sym_250".to_string()]));

        let q_sst = query_symbols(big_repo.path(), None, Some("src/m_250.rs"));
        assert_eq!(q_sst["defs"], serde_json::json!(snap.defs_by_file["src/m_250.rs"]));
        assert_eq!(q_sst["imports"], serde_json::json!(snap.imports_by_file["src/m_250.rs"]));
        assert_eq!((&q_sst["fan_in"], &q_sst["fan_out"]), (&serde_json::json!(snap.fan_in["src/m_250.rs"]), &serde_json::json!(snap.fan_out["src/m_250.rs"])));

        let mut tampered = sst_bytes;
        let mid = SST_HEADER_LEN + (tampered.len() - SST_HEADER_LEN) / 3;
        tampered[mid] ^= 0x01;
        fs::write(&sst_path, &tampered).unwrap();
        assert!(load_symbol_sst(&sst_path, "0.6.0").is_none());
        assert_eq!(query_symbols(big_repo.path(), None, Some("src/m_250.rs")), q_sst);
    }
}
