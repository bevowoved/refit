use anyhow::{Context, Result, anyhow, bail};
use constant_time_eq::constant_time_eq;
use flate2::read::DeflateDecoder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// Embedded 8 KB ternary weights from `brianbell-x/codehygiene v0.7.0`.
static WEIGHTS_NPZ: &[u8] = include_bytes!("../../vendor/codehygiene/weights.npz");
static WEIGHTS_CACHE: OnceLock<ResolvedWeights> = OnceLock::new();

pub(super) const MAX_NPZ_ARCHIVE_BYTES: usize = 512 * 1024;
pub(super) const MAX_NPY_ENTRY_BYTES: usize = 2 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_ZIP_ENTRIES: usize = 32;
const MAX_REL_DIST: isize = 32;
const REL_SPAN: usize = 65;
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

pub(super) struct Dense2D {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) data: Vec<f32>,
}

pub(super) struct CompactCsr {
    pub(super) rows: u16,
    pub(super) cols: u16,
    pub(super) row_offsets: Vec<u32>,
    pub(super) col_indices: Vec<u16>,
    pub(super) values: Vec<f32>,
    pub(super) active_rows: Vec<u16>,
    pub(super) non_zero_count: u32,
}

pub(super) struct CompactConv {
    pub(super) k_size: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) taps: Vec<CompactCsr>,
    pub(super) non_zero_count: u32,
}

pub(super) struct PadTap {
    pub(super) cols: Vec<u16>,
    pub(super) vals: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(super) struct PartRange {
    pub(super) part_idx: usize,
    pub(super) dim: usize,
    pub(super) in_off: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

pub(super) enum HeadKind {
    UniformGlobal,
    StaticBanded {
        offsets: Vec<i8>,
        rel_vals: Vec<f64>,
        interior_weights: Vec<f64>,
        min_neg: usize,
        max_pos: usize,
    },
    BandedQk {
        offsets: Vec<i8>,
        rel_vals: Vec<f64>,
    },
    DenseQk {
        rel: [f64; REL_SPAN],
        all_zero_rel: bool,
    },
    GeneralStatic {
        rel: [f64; REL_SPAN],
    },
}

pub(super) struct HeadPlan {
    pub(super) qk_off: usize,
    pub(super) v_off: usize,
    pub(super) sink: f64,
    pub(super) kind: HeadKind,
}

pub(super) struct AttnLayerWeights {
    pub(super) wq: Option<CompactCsr>,
    pub(super) wk: Option<CompactCsr>,
    pub(super) wv: CompactCsr,
    pub(super) wo: CompactCsr,
    pub(super) has_qk: bool,
    pub(super) qk_tot: usize,
    pub(super) h_dim: usize,
    pub(super) v_tot: usize,
    pub(super) v_dim: usize,
    pub(super) inv_s: f64,
    pub(super) heads: Vec<HeadPlan>,
}

pub(super) struct ContextLayerWeights {
    pub(super) attn: AttnLayerWeights,
    pub(super) cmlp_w: CompactCsr,
    pub(super) cmlp_b: Vec<f32>,
    pub(super) cmlp2_w: CompactCsr,
    pub(super) cmlp2_b: Vec<f32>,
}

pub(super) struct ResolvedWeights {
    pub(super) hash_prefix: String,
    pub(super) tok_emb: Dense2D,
    pub(super) col_emb: Dense2D,
    pub(super) pre_conv3_w: CompactConv,
    pub(super) pre_conv3_b: Vec<f32>,
    pub(super) pref1_proj: CompactCsr,
    pub(super) pref1_ranges: Vec<PartRange>,
    pub(super) gate1_w: CompactCsr,
    pub(super) gate1_b: Vec<f32>,
    pub(super) gate1_ranges: Vec<PartRange>,
    pub(super) pref2_proj: CompactCsr,
    pub(super) col_fuse_w: CompactCsr,
    pub(super) col_fuse_b: Vec<f32>,
    pub(super) col_fuse_ranges: Vec<PartRange>,
    pub(super) col_sum_proj: CompactCsr,
    pub(super) conv3_w: CompactConv,
    pub(super) conv3_b: Vec<f32>,
    pub(super) conv3_pads: Vec<PadTap>,
    pub(super) conv7_w: CompactConv,
    pub(super) conv7_b: Vec<f32>,
    pub(super) conv7_pads: Vec<PadTap>,
    pub(super) sketch_w: CompactConv,
    pub(super) sketch_b: Vec<f32>,
    pub(super) sketch_pads: Vec<PadTap>,
    pub(super) ml_proj: CompactCsr,
    pub(super) ml_proj_ranges: Vec<PartRange>,
    pub(super) ml_gate_w: CompactCsr,
    pub(super) ml_gate_b: Vec<f32>,
    pub(super) ml_paren_w: CompactConv,
    pub(super) ml_paren_b: Vec<f32>,
    pub(super) lmlp1_w: CompactCsr,
    pub(super) lmlp1_b: Vec<f32>,
    pub(super) lmlp1_ranges: Vec<PartRange>,
    pub(super) lmlp2_w: CompactCsr,
    pub(super) lmlp2_b: Vec<f32>,
    pub(super) ctx_layers: [ContextLayerWeights; 2],
    pub(super) sq_w: CompactCsr,
    pub(super) sq_b: Vec<f32>,
    pub(super) pool_gate_w: CompactCsr,
    pub(super) pool_gate_b: Vec<f32>,
    pub(super) doc1_w: CompactCsr,
    pub(super) doc1_b: Vec<f32>,
    pub(super) doc2_w: CompactCsr,
    pub(super) doc2_b: Vec<f32>,
    pub(super) state_dim: usize,
    pub(super) hc_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct WeightsManifest {
    pub(super) version: String,
    pub(super) arch: String,
    pub(super) file: String,
    pub(super) blake3: String,
}

pub(super) fn v07_weights() -> &'static ResolvedWeights {
    WEIGHTS_CACHE.get_or_init(|| {
        load_weights_from_npz(WEIGHTS_NPZ).expect("embedded v0.7.0 weights.npz must be valid")
    })
}

pub(super) fn default_weights_hash_prefix() -> &'static str {
    &v07_weights().hash_prefix
}

pub(super) fn load_verified_override(repo_root: &Path) -> Result<Option<Arc<ResolvedWeights>>> {
    let dir = repo_root.join(".refit").join("weights");
    let manifest_path = dir.join("hygiene.json");
    let default_npz_path = dir.join("hygiene.npz");

    if !manifest_path.exists() {
        if default_npz_path.exists() {
            bail!(
                "unpinned local weights override `{}` found without `{}` manifest",
                default_npz_path.display(),
                manifest_path.display()
            );
        }
        return Ok(None);
    }

    let manifest_bytes = read_capped_file(&manifest_path, MAX_MANIFEST_BYTES)
        .with_context(|| format!("read weights manifest {}", manifest_path.display()))?;
    let manifest: WeightsManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse weights manifest {}", manifest_path.display()))?;

    if manifest.version.trim().is_empty() {
        bail!("weights manifest missing non-empty `version`");
    }
    if manifest.arch != "v07" {
        bail!("unsupported weights arch `{}` (expected `v07`)", manifest.arch);
    }
    validate_override_filename(&manifest.file)?;

    let expected_hex = manifest.blake3.trim().to_ascii_lowercase();
    if expected_hex.len() != 64 || !expected_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("weights manifest `blake3` must be a 64-char hex digest");
    }

    let npz_path = dir.join(&manifest.file);
    let npz_bytes = read_capped_file(&npz_path, MAX_NPZ_ARCHIVE_BYTES)
        .with_context(|| format!("read weights archive {}", npz_path.display()))?;

    let actual_hex = blake3::hash(&npz_bytes).to_hex().to_string();
    if !constant_time_eq(actual_hex.as_bytes(), expected_hex.as_bytes()) {
        bail!(
            "weights digest mismatch for `{}`: expected {}, got {}",
            npz_path.display(),
            expected_hex,
            actual_hex
        );
    }

    let resolved = load_weights_from_npz(&npz_bytes)
        .with_context(|| format!("unpack verified weights {}", npz_path.display()))?;
    Ok(Some(Arc::new(resolved)))
}

fn validate_override_filename(file: &str) -> Result<()> {
    let bad = file.is_empty()
        || file.contains('/')
        || file.contains('\\')
        || file.contains("..")
        || file.contains('\0');
    if bad {
        bail!("unsafe weights filename `{file}` in manifest");
    }
    Ok(())
}

fn read_capped_file(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut out = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut out)?;
    if out.len() > limit {
        bail!("file `{}` exceeds safety cap of {} bytes", path.display(), limit);
    }
    Ok(out)
}

fn le_u16(buf: &[u8], off: usize) -> Result<u16> {
    let slice = buf
        .get(off..off.checked_add(2).ok_or_else(|| anyhow!("u16 offset overflow"))?)
        .ok_or_else(|| anyhow!("truncated u16 at offset {off}"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn le_u32(buf: &[u8], off: usize) -> Result<u32> {
    let slice = buf
        .get(off..off.checked_add(4).ok_or_else(|| anyhow!("u32 offset overflow"))?)
        .ok_or_else(|| anyhow!("truncated u32 at offset {off}"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

enum NpyData {
    U8(Vec<u8>),
    F32(Vec<f32>),
    Str(String),
}

fn parse_npy(bytes: &[u8]) -> Result<NpyData> {
    if bytes.get(0..6) != Some(b"\x93NUMPY".as_slice()) {
        bail!("invalid .npy magic header");
    }
    let major = *bytes.get(6).ok_or_else(|| anyhow!("truncated .npy version"))?;
    let (hdr_len, prefix_len) = match major {
        1 => (le_u16(bytes, 8)? as usize, 10usize),
        2 | 3 => (le_u32(bytes, 8)? as usize, 12usize),
        other => bail!("unsupported .npy version {other}"),
    };
    let data_off = prefix_len
        .checked_add(hdr_len)
        .ok_or_else(|| anyhow!(".npy header length overflow"))?;
    let hdr_bytes = bytes
        .get(8..data_off)
        .ok_or_else(|| anyhow!("truncated .npy header"))?;
    let hdr = String::from_utf8_lossy(hdr_bytes);
    let descr = hdr
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .unwrap_or("");
    let body = bytes
        .get(data_off..)
        .ok_or_else(|| anyhow!("truncated .npy body"))?;

    if descr.ends_with("u1") {
        Ok(NpyData::U8(body.to_vec()))
    } else if descr == "<f4" {
        if !body.len().is_multiple_of(4) {
            bail!("misaligned <f4 .npy payload ({} bytes)", body.len());
        }
        let vals = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(NpyData::F32(vals))
    } else if descr.starts_with("<U") {
        if !body.len().is_multiple_of(4) {
            bail!("misaligned <U .npy payload ({} bytes)", body.len());
        }
        let text: String = body
            .as_chunks::<4>()
            .0
            .iter()
            .filter_map(|c| char::from_u32(u32::from_le_bytes(*c)))
            .filter(|&c| c != '\0')
            .collect();
        Ok(NpyData::Str(text))
    } else {
        bail!("unsupported .npy descr `{descr}`");
    }
}

fn inflate_entry_capped(method: u16, raw: &[u8]) -> Result<Vec<u8>> {
    match method {
        0 => {
            if raw.len() > MAX_NPY_ENTRY_BYTES {
                bail!("stored NPZ entry exceeds {} byte cap", MAX_NPY_ENTRY_BYTES);
            }
            Ok(raw.to_vec())
        }
        8 => {
            let mut out = Vec::new();
            DeflateDecoder::new(raw)
                .take((MAX_NPY_ENTRY_BYTES + 1) as u64)
                .read_to_end(&mut out)
                .context("inflate NPZ entry")?;
            if out.len() > MAX_NPY_ENTRY_BYTES {
                bail!(
                    "deflated NPZ entry exceeds {} byte zip-bomb cap",
                    MAX_NPY_ENTRY_BYTES
                );
            }
            Ok(out)
        }
        other => bail!("unsupported ZIP compression method {other}"),
    }
}

pub(super) fn load_weights_from_npz(bytes: &[u8]) -> Result<ResolvedWeights> {
    if bytes.len() > MAX_NPZ_ARCHIVE_BYTES {
        bail!(
            "weights archive size {} exceeds {} byte cap",
            bytes.len(),
            MAX_NPZ_ARCHIVE_BYTES
        );
    }
    if bytes.len() < 22 {
        bail!("weights archive too small ({} bytes)", bytes.len());
    }

    let hash_hex = blake3::hash(bytes).to_hex().to_string();
    let hash_prefix = hash_hex[..16].to_string();

    let lo = bytes.len().saturating_sub(65557);
    let hi = bytes.len() - 22;
    let eocd = (lo..=hi)
        .rev()
        .find(|&pos| le_u32(bytes, pos).ok() == Some(0x0605_4b50))
        .ok_or_else(|| anyhow!("missing ZIP EOCD in weights.npz"))?;

    let count = le_u16(bytes, eocd + 10)? as usize;
    if count == 0 || count > MAX_ZIP_ENTRIES {
        bail!("invalid NPZ entry count {count}");
    }
    let mut dir = le_u32(bytes, eocd + 16)? as usize;

    let mut trits: Option<Vec<u8>> = None;
    let mut vec_buf: Option<Vec<f32>> = None;
    let mut meta: Option<String> = None;

    for _ in 0..count {
        if le_u32(bytes, dir)? != 0x0201_4b50 {
            bail!("invalid ZIP central directory signature at offset {dir}");
        }
        let method = le_u16(bytes, dir + 10)?;
        let comp = le_u32(bytes, dir + 20)? as usize;
        let nlen = le_u16(bytes, dir + 28)? as usize;
        let extra = le_u16(bytes, dir + 30)? as usize;
        let comment = le_u16(bytes, dir + 32)? as usize;
        let local = le_u32(bytes, dir + 42)? as usize;

        let name_end = dir
            .checked_add(46)
            .and_then(|v| v.checked_add(nlen))
            .ok_or_else(|| anyhow!("ZIP name offset overflow"))?;
        let name_bytes = bytes
            .get(dir + 46..name_end)
            .ok_or_else(|| anyhow!("truncated ZIP entry name"))?;
        let name = String::from_utf8_lossy(name_bytes);

        if le_u32(bytes, local)? != 0x0403_4b50 {
            bail!("invalid ZIP local header signature at offset {local}");
        }
        let local_nlen = le_u16(bytes, local + 26)? as usize;
        let local_extra = le_u16(bytes, local + 28)? as usize;
        let data_start = local
            .checked_add(30)
            .and_then(|v| v.checked_add(local_nlen))
            .and_then(|v| v.checked_add(local_extra))
            .ok_or_else(|| anyhow!("ZIP local data offset overflow"))?;
        let data_end = data_start
            .checked_add(comp)
            .ok_or_else(|| anyhow!("ZIP compressed size overflow"))?;
        let comp_slice = bytes
            .get(data_start..data_end)
            .ok_or_else(|| anyhow!("truncated ZIP entry payload"))?;

        let unzipped = inflate_entry_capped(method, comp_slice)?;
        match (name.strip_suffix(".npy").unwrap_or(&name), parse_npy(&unzipped)?) {
            ("trits", NpyData::U8(v)) => trits = Some(v),
            ("vecs", NpyData::F32(v)) => vec_buf = Some(v),
            ("meta", NpyData::Str(s)) => meta = Some(s),
            _ => {}
        }

        dir = name_end
            .checked_add(extra)
            .and_then(|v| v.checked_add(comment))
            .ok_or_else(|| anyhow!("ZIP central directory advance overflow"))?;
    }

    let trits = trits.ok_or_else(|| anyhow!("missing `trits.npy` in archive"))?;
    let vec_buf = vec_buf.ok_or_else(|| anyhow!("missing `vecs.npy` in archive"))?;
    let meta = meta.ok_or_else(|| anyhow!("missing `meta.npy` in archive"))?;
    unpack_resolved(&trits, &vec_buf, &meta, hash_prefix)
}

fn make_compact_csr(dense: &[f32], rows: usize, cols: usize) -> Result<CompactCsr> {
    let rows_u16 = u16::try_from(rows).map_err(|_| anyhow!("CSR rows {rows} > u16::MAX"))?;
    let cols_u16 = u16::try_from(cols).map_err(|_| anyhow!("CSR cols {cols} > u16::MAX"))?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("CSR shape overflow"))?;
    if dense.len() < expected {
        bail!("CSR dense slice len {} < expected {}", dense.len(), expected);
    }

    let mut row_offsets = vec![0u32; rows + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut active_rows = Vec::new();

    for row in 0..rows {
        row_offsets[row] = col_indices.len() as u32;
        let base = row * cols;
        for col in 0..cols {
            let val = dense[base + col];
            if val != 0.0 {
                col_indices.push(col as u16);
                values.push(val);
            }
        }
        if col_indices.len() as u32 > row_offsets[row] {
            active_rows.push(row as u16);
        }
    }

    let non_zero_count = u32::try_from(col_indices.len())
        .map_err(|_| anyhow!("CSR nnz exceeds u32::MAX"))?;
    row_offsets[rows] = non_zero_count;
    Ok(CompactCsr {
        rows: rows_u16,
        cols: cols_u16,
        row_offsets,
        col_indices,
        values,
        active_rows,
        non_zero_count,
    })
}

fn make_compact_conv(
    dense: &[f32],
    k_size: usize,
    rows: usize,
    cols: usize,
) -> Result<CompactConv> {
    let span = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("conv shape overflow"))?;
    let total = k_size
        .checked_mul(span)
        .ok_or_else(|| anyhow!("conv total overflow"))?;
    if dense.len() < total {
        bail!("conv dense len {} < expected {}", dense.len(), total);
    }

    let mut taps = Vec::with_capacity(k_size);
    let mut non_zero_count = 0u32;
    for tap in 0..k_size {
        let csr = make_compact_csr(&dense[tap * span..(tap + 1) * span], rows, cols)?;
        non_zero_count = non_zero_count
            .checked_add(csr.non_zero_count)
            .ok_or_else(|| anyhow!("conv nnz overflow"))?;
        taps.push(csr);
    }
    Ok(CompactConv { k_size, rows, cols, taps, non_zero_count })
}

fn parse_meta(entry: &str) -> Result<(String, Vec<usize>, usize)> {
    let mut parts = entry.split(':');
    let name = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("empty tensor name in meta"))?
        .to_string();
    let dims_str = parts.next().ok_or_else(|| anyhow!("missing dims for `{name}`"))?;
    let mut dims = Vec::new();
    for raw_dim in dims_str.split(',') {
        let d: usize = raw_dim
            .trim()
            .parse()
            .with_context(|| format!("invalid dim `{raw_dim}` for `{name}`"))?;
        dims.push(d);
    }
    let count_str = parts.next().ok_or_else(|| anyhow!("missing count for `{name}`"))?;
    let count: usize = count_str
        .trim()
        .parse()
        .with_context(|| format!("invalid count `{count_str}` for `{name}`"))?;
    Ok((name, dims, count))
}

fn decode_trits(trit_buf: &[u8]) -> Result<Vec<i8>> {
    let total = trit_buf
        .len()
        .checked_mul(5)
        .ok_or_else(|| anyhow!("trit buffer length overflow"))?;
    let mut flat = vec![0i8; total];
    for (byte_idx, &bv) in trit_buf.iter().enumerate() {
        for slot in 0..5 {
            let digit = (bv / POW3[slot]) % 3;
            flat[byte_idx * 5 + slot] = ((digit as i8 + 1) % 3) - 1;
        }
    }
    Ok(flat)
}

fn unpack_tensor(trits: &[i8], count: usize, scale: Option<&Vec<f32>>) -> Result<Vec<f32>> {
    if trits.len() < count {
        bail!("truncated trit stream: need {count}, have {}", trits.len());
    }
    if let Some(sv) = scale
        && sv.is_empty()
    {
        bail!("empty scale vector in weights");
    }
    let mut dense = vec![0.0f32; count];
    for idx in 0..count {
        let t = trits[idx];
        if t != 0 {
            dense[idx] = scale.map_or(t as f32, |sv| (t as f32) * sv[idx % sv.len()]);
        }
    }
    Ok(dense)
}

struct RawUnpacked {
    dense_embs: HashMap<String, Dense2D>,
    mats: HashMap<String, CompactCsr>,
    convs: HashMap<String, CompactConv>,
    vecs: HashMap<String, Vec<f32>>,
}

impl RawUnpacked {
    fn take_emb(&mut self, key: &str) -> Result<Dense2D> {
        self.dense_embs
            .remove(key)
            .ok_or_else(|| anyhow!("missing embedding `{key}`"))
    }

    fn take_mat(&mut self, key: &str) -> Result<CompactCsr> {
        self.mats
            .remove(key)
            .ok_or_else(|| anyhow!("missing matrix `{key}`"))
    }

    fn take_conv(&mut self, key: &str) -> Result<CompactConv> {
        self.convs
            .remove(key)
            .ok_or_else(|| anyhow!("missing conv `{key}`"))
    }

    fn take_vec(&mut self, key: &str) -> Result<Vec<f32>> {
        self.vecs
            .remove(key)
            .ok_or_else(|| anyhow!("missing vector `{key}`"))
    }
}

fn unpack_raw(trit_buf: &[u8], vec_buf: &[f32], meta_str: &str) -> Result<RawUnpacked> {
    let (trit_part, vec_part) = meta_str
        .split_once('|')
        .ok_or_else(|| anyhow!("missing '|' separator in weights meta"))?;

    let mut raw_vecs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut vec_off = 0usize;
    for entry in vec_part.split(" ; ") {
        let (name, _, count) = parse_meta(entry)?;
        let end = vec_off
            .checked_add(count)
            .ok_or_else(|| anyhow!("vec offset overflow for `{name}`"))?;
        let slice = vec_buf
            .get(vec_off..end)
            .ok_or_else(|| anyhow!("truncated vecs buffer for `{name}`"))?;
        raw_vecs.insert(name, slice.to_vec());
        vec_off = end;
    }

    let flat_trits = decode_trits(trit_buf)?;
    let mut dense_embs = HashMap::new();
    let mut mats = HashMap::new();
    let mut convs = HashMap::new();
    let mut byte_off = 0usize;

    for entry in trit_part.split(" ; ") {
        let (name, shape, packed) = parse_meta(entry)?;
        let mut n_elems = 1usize;
        for &dim in &shape {
            n_elems = n_elems
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("tensor shape product overflow for `{name}`"))?;
        }
        let trit_start = byte_off
            .checked_mul(5)
            .ok_or_else(|| anyhow!("trit start overflow for `{name}`"))?;
        let skey = if name == "col_sum_proj" {
            "col_sum_s".to_string()
        } else {
            format!("{}_s", name.strip_suffix("_w").unwrap_or(&name))
        };
        let trit_slice = flat_trits
            .get(trit_start..)
            .ok_or_else(|| anyhow!("trit offset out of bounds for `{name}`"))?;
        let dense = unpack_tensor(trit_slice, n_elems, raw_vecs.get(&skey))?;
        byte_off = byte_off
            .checked_add(packed)
            .ok_or_else(|| anyhow!("trit byte offset overflow for `{name}`"))?;

        match shape.as_slice() {
            [k, r, c] => {
                convs.insert(name, make_compact_conv(&dense, *k, *r, *c)?);
            }
            [r, c] => {
                if name == "tok_emb" || name == "col_emb" {
                    dense_embs.insert(name, Dense2D { rows: *r, cols: *c, data: dense });
                } else {
                    mats.insert(name, make_compact_csr(&dense, *r, *c)?);
                }
            }
            _ => bail!("unsupported tensor rank {} for `{name}`", shape.len()),
        }
    }

    let vecs = raw_vecs.into_iter().filter(|(k, _)| !k.ends_with("_s")).collect();
    Ok(RawUnpacked { dense_embs, mats, convs, vecs })
}

fn precompute_part_ranges(dims: &[usize], active_rows: &[u16]) -> Vec<PartRange> {
    let mut ranges = Vec::with_capacity(dims.len());
    let mut in_off = 0usize;
    let mut cursor = 0usize;
    for (part_idx, &dim) in dims.iter().enumerate() {
        let next_off = in_off + dim;
        let start = cursor;
        while cursor < active_rows.len() && (active_rows[cursor] as usize) < next_off {
            cursor += 1;
        }
        if cursor > start {
            ranges.push(PartRange { part_idx, dim, in_off, start, end: cursor });
        }
        in_off = next_off;
    }
    ranges
}

fn precompute_pad_taps(pad_fill: &[f32], conv: &CompactConv) -> Vec<PadTap> {
    conv.taps
        .iter()
        .map(|tap| {
            let mut prod = vec![0.0f32; conv.cols];
            for &r_u16 in &tap.active_rows {
                let row = r_u16 as usize;
                let iv = pad_fill[row];
                if iv == 0.0 {
                    continue;
                }
                let lo = tap.row_offsets[row] as usize;
                let hi = tap.row_offsets[row + 1] as usize;
                for ptr in lo..hi {
                    prod[tap.col_indices[ptr] as usize] += iv * tap.values[ptr];
                }
            }
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for (col, &val) in prod.iter().enumerate() {
                if val != 0.0 {
                    cols.push(col as u16);
                    vals.push(val);
                }
            }
            PadTap { cols, vals }
        })
        .collect()
}

fn build_attn_layer(
    raw: &mut RawUnpacked,
    layer: &str,
    state_dim: usize,
) -> Result<AttnLayerWeights> {
    let wq = raw.mats.remove(&format!("attn{layer}_wq"));
    let wk = raw.mats.remove(&format!("attn{layer}_wk"));
    let wv = raw.take_mat(&format!("attn{layer}_wv"))?;
    let wo = raw.take_mat(&format!("attn{layer}_wo"))?;
    let b_rel = raw.take_vec(&format!("attn{layer}_brel"))?;
    let b_sink = raw.take_vec(&format!("attn{layer}_bsink"))?;

    let n_heads = b_sink.len();
    if n_heads == 0 {
        bail!("attn{layer} has 0 heads");
    }
    if b_rel.len() != n_heads * REL_SPAN {
        bail!(
            "attn{layer}_brel len {} != n_heads ({n_heads}) * {REL_SPAN}",
            b_rel.len()
        );
    }
    if wv.rows as usize != state_dim || wo.cols as usize != state_dim {
        bail!("attn{layer} wv/wo dim mismatch with state_dim {state_dim}");
    }
    if wv.cols as usize != wo.rows as usize || !(wv.cols as usize).is_multiple_of(n_heads) {
        bail!("attn{layer} wv/wo head dim mismatch");
    }

    let has_qk = wq.as_ref().is_some_and(|m| m.non_zero_count > 0)
        && wk.as_ref().is_some_and(|m| m.non_zero_count > 0);
    let qk_tot = wq.as_ref().map_or(0, |m| m.cols as usize);
    if has_qk {
        let wq_ref = wq.as_ref().ok_or_else(|| anyhow!("missing wq"))?;
        let wk_ref = wk.as_ref().ok_or_else(|| anyhow!("missing wk"))?;
        if wq_ref.rows as usize != state_dim
            || wk_ref.rows as usize != state_dim
            || wq_ref.cols != wk_ref.cols
            || !qk_tot.is_multiple_of(n_heads)
        {
            bail!("attn{layer} wq/wk shape mismatch");
        }
    }
    let h_dim = if qk_tot > 0 { qk_tot / n_heads } else { 1 };
    let v_tot = wv.cols as usize;
    let v_dim = v_tot / n_heads;
    let inv_s = (1.0f32 / (h_dim as f32).sqrt()) as f64;

    let mut heads = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let rel_slice = &b_rel[h * REL_SPAN..(h + 1) * REL_SPAN];
        let qk_off = h * h_dim;
        let v_off = h * v_dim;
        let sink = b_sink[h] as f64;
        let head_qk = has_qk
            && wq.as_ref().is_some_and(|m| {
                m.col_indices
                    .iter()
                    .any(|&c| (c as usize) >= qk_off && (c as usize) < qk_off + h_dim)
            });
        let banded = rel_slice[0] <= -1e3 && rel_slice[REL_SPAN - 1] <= -1e3;
        let all_zero = rel_slice.iter().all(|&v| v == 0.0);

        let kind = classify_head(rel_slice, sink, banded, head_qk, all_zero);
        heads.push(HeadPlan { qk_off, v_off, sink, kind });
    }

    Ok(AttnLayerWeights {
        wq,
        wk,
        wv,
        wo,
        has_qk,
        qk_tot,
        h_dim,
        v_tot,
        v_dim,
        inv_s,
        heads,
    })
}

fn classify_head(
    rel_slice: &[f32],
    sink: f64,
    banded: bool,
    head_qk: bool,
    all_zero: bool,
) -> HeadKind {
    if !banded && !head_qk && all_zero {
        return HeadKind::UniformGlobal;
    }
    if banded {
        let mut offsets = Vec::new();
        let mut rel_vals = Vec::new();
        let mut min_neg = 0usize;
        let mut max_pos = 0usize;
        for d in (-MAX_REL_DIST + 1)..MAX_REL_DIST {
            let rv = rel_slice[(d + MAX_REL_DIST) as usize] as f64;
            if rv <= -1e3 {
                continue;
            }
            offsets.push(d as i8);
            rel_vals.push(rv);
            if d < 0 {
                min_neg = min_neg.max((-d) as usize);
            } else {
                max_pos = max_pos.max(d as usize);
            }
        }
        if head_qk {
            return HeadKind::BandedQk { offsets, rel_vals };
        }
        let mut max_l = sink;
        for &rv in &rel_vals {
            max_l = max_l.max(rv);
        }
        let mut exp_sum = (sink - max_l).exp();
        let mut exps = Vec::with_capacity(rel_vals.len());
        for &rv in &rel_vals {
            let e = (rv - max_l).exp();
            exp_sum += e;
            exps.push(e);
        }
        let interior_weights = exps.into_iter().map(|e| e / exp_sum).collect();
        return HeadKind::StaticBanded {
            offsets,
            rel_vals,
            interior_weights,
            min_neg,
            max_pos,
        };
    }

    let mut rel = [0.0f64; REL_SPAN];
    for (dst, &src) in rel.iter_mut().zip(rel_slice.iter()) {
        *dst = src as f64;
    }
    if head_qk {
        HeadKind::DenseQk { rel, all_zero_rel: all_zero }
    } else {
        HeadKind::GeneralStatic { rel }
    }
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!("tensor `{name}` shape mismatch: got {actual}, expected {expected}");
    }
    Ok(())
}

fn build_context_layer(
    raw: &mut RawUnpacked,
    layer: &str,
    state_dim: usize,
) -> Result<ContextLayerWeights> {
    let attn = build_attn_layer(raw, layer, state_dim)?;
    let cmlp_w = raw.take_mat(&format!("cmlp{layer}_w"))?;
    let cmlp_b = raw.take_vec(&format!("cmlp{layer}_b"))?;
    let cmlp2_w = raw.take_mat(&format!("cmlp{layer}_2_w"))?;
    let cmlp2_b = raw.take_vec(&format!("cmlp{layer}_2_b"))?;

    check_len(&format!("cmlp{layer}_w.rows"), cmlp_w.rows as usize, state_dim)?;
    check_len(&format!("cmlp{layer}_b"), cmlp_b.len(), cmlp_w.cols as usize)?;
    check_len(&format!("cmlp{layer}_2_w.rows"), cmlp2_w.rows as usize, cmlp_w.cols as usize)?;
    check_len(&format!("cmlp{layer}_2_w.cols"), cmlp2_w.cols as usize, state_dim)?;
    check_len(&format!("cmlp{layer}_2_b"), cmlp2_b.len(), state_dim)?;

    Ok(ContextLayerWeights { attn, cmlp_w, cmlp_b, cmlp2_w, cmlp2_b })
}

fn take_linear(
    raw: &mut RawUnpacked,
    w_key: &str,
    b_key: &str,
    expected_rows: usize,
    expected_cols: Option<usize>,
) -> Result<(CompactCsr, Vec<f32>)> {
    let mat = raw.take_mat(w_key)?;
    let bias = raw.take_vec(b_key)?;
    check_len(&format!("{w_key}.rows"), mat.rows as usize, expected_rows)?;
    if let Some(cols) = expected_cols {
        check_len(&format!("{w_key}.cols"), mat.cols as usize, cols)?;
    }
    check_len(b_key, bias.len(), mat.cols as usize)?;
    Ok((mat, bias))
}

fn unpack_resolved(
    trit_buf: &[u8],
    vec_buf: &[f32],
    meta_str: &str,
    hash_prefix: String,
) -> Result<ResolvedWeights> {
    let mut raw = unpack_raw(trit_buf, vec_buf, meta_str)?;

    let tok_emb = raw.take_emb("tok_emb")?;
    let col_emb = raw.take_emb("col_emb")?;
    check_len("tok_emb.rows", tok_emb.rows, 128)?;
    check_len("col_emb.rows", col_emb.rows, 128)?;
    check_len("col_emb.cols", col_emb.cols, tok_emb.cols)?;
    let emb_dim = tok_emb.cols;

    let pre_conv3_w = raw.take_conv("pre_conv3_w")?;
    let pre_conv3_b = raw.take_vec("pre_conv3_b")?;
    check_len("pre_conv3_w.k_size", pre_conv3_w.k_size, 3)?;
    check_len("pre_conv3_w.rows", pre_conv3_w.rows, emb_dim)?;
    check_len("pre_conv3_b", pre_conv3_b.len(), pre_conv3_w.cols)?;

    let pref1_proj = raw.take_mat("pref1_proj")?;
    check_len("pref1_proj.rows", pref1_proj.rows as usize, emb_dim + pre_conv3_w.cols)?;
    let pref1_ranges = precompute_part_ranges(&[emb_dim, pre_conv3_w.cols], &pref1_proj.active_rows);

    let g1_in = emb_dim + pre_conv3_w.cols + pref1_proj.cols as usize;
    let (gate1_w, gate1_b) = take_linear(&mut raw, "gate1_w", "gate1_b", g1_in, None)?;
    let gate1_ranges = precompute_part_ranges(
        &[emb_dim, pre_conv3_w.cols, pref1_proj.cols as usize],
        &gate1_w.active_rows,
    );

    let pref2_proj = raw.take_mat("pref2_proj")?;
    check_len("pref2_proj.rows", pref2_proj.rows as usize, gate1_w.cols as usize)?;

    let fuse_in = emb_dim + pre_conv3_w.cols + gate1_w.cols as usize + pref2_proj.cols as usize;
    let (col_fuse_w, col_fuse_b) = take_linear(&mut raw, "col_fuse_w", "col_fuse_b", fuse_in, None)?;
    let hc_dim = col_fuse_w.cols as usize;
    if hc_dim == 0 || hc_dim > 128 {
        bail!("unsupported col_fuse_w.cols={hc_dim} (expected 1..=128)");
    }
    let col_fuse_ranges = precompute_part_ranges(
        &[emb_dim, pre_conv3_w.cols, gate1_w.cols as usize, pref2_proj.cols as usize],
        &col_fuse_w.active_rows,
    );

    let pad_fill: Vec<f32> = col_fuse_b.iter().map(|&v| v.max(0.0)).collect();
    let col_sum_proj = raw.take_mat("col_sum_proj")?;
    check_len("col_sum_proj.rows", col_sum_proj.rows as usize, hc_dim)?;

    let conv3_w = raw.take_conv("conv3_w")?;
    let conv3_b = raw.take_vec("conv3_b")?;
    let conv7_w = raw.take_conv("conv7_w")?;
    let conv7_b = raw.take_vec("conv7_b")?;
    let sketch_w = raw.take_conv("sketch_w")?;
    let sketch_b = raw.take_vec("sketch_b")?;
    for (lbl, conv, bias) in [
        ("conv3", &conv3_w, &conv3_b),
        ("conv7", &conv7_w, &conv7_b),
        ("sketch", &sketch_w, &sketch_b),
    ] {
        check_len(&format!("{lbl}_w.rows"), conv.rows, hc_dim)?;
        check_len(&format!("{lbl}_b"), bias.len(), conv.cols)?;
        if conv.k_size == 0 || conv.k_size.is_multiple_of(2) || conv.k_size > 13 {
            bail!("{lbl}_w invalid k_size={}", conv.k_size);
        }
    }

    let conv3_pads = precompute_pad_taps(&pad_fill, &conv3_w);
    let conv7_pads = precompute_pad_taps(&pad_fill, &conv7_w);
    let sketch_pads = precompute_pad_taps(&pad_fill, &sketch_w);

    let ml_proj = raw.take_mat("ml_proj")?;
    let ml_dims = [col_sum_proj.cols as usize, conv3_w.cols, conv7_w.cols];
    check_len("ml_proj.rows", ml_proj.rows as usize, ml_dims.iter().sum())?;
    let ml_proj_ranges = precompute_part_ranges(&ml_dims, &ml_proj.active_rows);

    let (ml_gate_w, ml_gate_b) =
        take_linear(&mut raw, "ml_gate_w", "ml_gate_b", ml_proj.cols as usize, None)?;

    let ml_paren_w = raw.take_conv("ml_paren_w")?;
    let ml_paren_b = raw.take_vec("ml_paren_b")?;
    check_len("ml_paren_w.rows", ml_paren_w.rows, ml_proj.cols as usize)?;
    check_len("ml_paren_b", ml_paren_b.len(), ml_paren_w.cols)?;

    let lmlp1_dims = [
        col_sum_proj.cols as usize,
        conv3_w.cols,
        conv7_w.cols,
        ml_gate_w.cols as usize,
        ml_paren_w.cols,
    ];
    let (lmlp1_w, lmlp1_b) =
        take_linear(&mut raw, "lmlp1_w", "lmlp1_b", lmlp1_dims.iter().sum(), None)?;
    let lmlp1_ranges = precompute_part_ranges(&lmlp1_dims, &lmlp1_w.active_rows);

    let (lmlp2_w, lmlp2_b) =
        take_linear(&mut raw, "lmlp2_w", "lmlp2_b", lmlp1_w.cols as usize, None)?;

    let state_dim = lmlp2_w.cols as usize + sketch_w.cols;
    let ctx1 = build_context_layer(&mut raw, "1", state_dim)?;
    let ctx2 = build_context_layer(&mut raw, "2", state_dim)?;

    let (sq_w, sq_b) = take_linear(&mut raw, "sq_w", "sq_b", state_dim, Some(6))?;
    let (pool_gate_w, pool_gate_b) =
        take_linear(&mut raw, "pool_gate_w", "pool_gate_b", state_dim, Some(1))?;
    let (doc1_w, doc1_b) = take_linear(&mut raw, "doc1_w", "doc1_b", state_dim * 2, None)?;
    let (doc2_w, doc2_b) =
        take_linear(&mut raw, "doc2_w", "doc2_b", doc1_w.cols as usize, Some(1))?;

    Ok(ResolvedWeights {
        hash_prefix,
        tok_emb,
        col_emb,
        pre_conv3_w,
        pre_conv3_b,
        pref1_proj,
        pref1_ranges,
        gate1_w,
        gate1_b,
        gate1_ranges,
        pref2_proj,
        col_fuse_w,
        col_fuse_b,
        col_fuse_ranges,
        col_sum_proj,
        conv3_w,
        conv3_b,
        conv3_pads,
        conv7_w,
        conv7_b,
        conv7_pads,
        sketch_w,
        sketch_b,
        sketch_pads,
        ml_proj,
        ml_proj_ranges,
        ml_gate_w,
        ml_gate_b,
        ml_paren_w,
        ml_paren_b,
        lmlp1_w,
        lmlp1_b,
        lmlp1_ranges,
        lmlp2_w,
        lmlp2_b,
        ctx_layers: [ctx1, ctx2],
        sq_w,
        sq_b,
        pool_gate_w,
        pool_gate_b,
        doc1_w,
        doc1_b,
        doc2_w,
        doc2_b,
        state_dim,
        hc_dim,
    })
}
