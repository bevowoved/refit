use flate2::read::DeflateDecoder;
use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

/// Embedded 8 KB ternary weights from `brianbell-x/codehygiene v0.7.0`.
static WEIGHTS_NPZ: &[u8] = include_bytes!("../../vendor/codehygiene/weights.npz");
static WEIGHTS_CACHE: OnceLock<HygieneWeights> = OnceLock::new();

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

pub(super) struct Sparse2D {
    pub(super) data: Vec<f32>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_offsets: Vec<usize>,
    pub(super) col_indices: Vec<usize>,
    pub(super) values: Vec<f32>,
    pub(super) active_rows: Vec<usize>,
    pub(super) non_zero_count: usize,
}

pub(super) struct SparseConv {
    pub(super) k_size: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) taps: Vec<Sparse2D>,
    pub(super) non_zero_count: usize,
}

pub(super) struct HygieneWeights {
    pub(super) mats: HashMap<String, Sparse2D>,
    pub(super) convs: HashMap<String, SparseConv>,
    pub(super) vecs: HashMap<String, Vec<f32>>,
}

impl HygieneWeights {
    pub(super) fn mat(&self, k: &str) -> &Sparse2D {
        self.mats.get(k).unwrap_or_else(|| panic!("missing mat `{k}`"))
    }

    pub(super) fn conv(&self, k: &str) -> &SparseConv {
        self.convs.get(k).unwrap_or_else(|| panic!("missing conv `{k}`"))
    }

    pub(super) fn vec(&self, k: &str) -> &[f32] {
        self.vecs.get(k).unwrap_or_else(|| panic!("missing vec `{k}`"))
    }
}

pub(super) fn v07_weights() -> &'static HygieneWeights {
    WEIGHTS_CACHE.get_or_init(|| unpack_npz(WEIGHTS_NPZ))
}

fn le_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

enum NpyData {
    U8(Vec<u8>),
    F32(Vec<f32>),
    Str(String),
}

fn parse_npy(bytes: &[u8]) -> NpyData {
    let major = bytes[6];
    let hdr_len = if major == 1 { le_u16(bytes, 8) as usize } else { le_u32(bytes, 8) as usize };
    let data_off = (if major == 1 { 10 } else { 12 }) + hdr_len;
    let hdr = String::from_utf8_lossy(&bytes[8..data_off]);
    let descr = hdr
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .unwrap_or("");
    let body = &bytes[data_off..];

    if descr.ends_with("u1") {
        NpyData::U8(body.to_vec())
    } else if descr == "<f4" {
        let vals = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        NpyData::F32(vals)
    } else if descr.starts_with("<U") {
        let s: String = body
            .as_chunks::<4>()
            .0
            .iter()
            .filter_map(|c| char::from_u32(u32::from_le_bytes(*c)))
            .filter(|&c| c != '\0')
            .collect();
        NpyData::Str(s)
    } else {
        panic!("unsupported npy descr `{descr}`");
    }
}

fn unpack_npz(bytes: &[u8]) -> HygieneWeights {
    let lo = bytes.len().saturating_sub(65557);
    let hi = bytes.len().saturating_sub(22);
    let eocd = (lo..=hi)
        .rev()
        .find(|&p| le_u32(bytes, p) == 0x06054b50)
        .expect("valid weights.npz EOCD");

    let count = le_u16(bytes, eocd + 10) as usize;
    let mut dir = le_u32(bytes, eocd + 16) as usize;
    let (mut trits, mut vec_buf, mut meta) = (Vec::new(), Vec::new(), String::new());

    for _ in 0..count {
        let method = le_u16(bytes, dir + 10);
        let comp = le_u32(bytes, dir + 20) as usize;
        let nlen = le_u16(bytes, dir + 28) as usize;
        let skip = 46 + nlen + le_u16(bytes, dir + 30) as usize + le_u16(bytes, dir + 32) as usize;
        let local = le_u32(bytes, dir + 42) as usize;
        let name = String::from_utf8_lossy(&bytes[dir + 46..dir + 46 + nlen]);

        let start = local + 30 + le_u16(bytes, local + 26) as usize
            + le_u16(bytes, local + 28) as usize;
        let unzipped = inflate_entry(method, &bytes[start..start + comp]);
        match (name.strip_suffix(".npy").unwrap_or(&name), parse_npy(&unzipped)) {
            ("trits", NpyData::U8(v)) => trits = v,
            ("vecs", NpyData::F32(v)) => vec_buf = v,
            ("meta", NpyData::Str(s)) => meta = s,
            _ => {}
        }
        dir += skip;
    }
    unpack_weights(&trits, &vec_buf, &meta)
}

fn inflate_entry(method: u16, raw: &[u8]) -> Vec<u8> {
    if method != 8 {
        return raw.to_vec();
    }
    let mut out = Vec::new();
    DeflateDecoder::new(raw).read_to_end(&mut out).expect("inflate");
    out
}

fn make_sparse2d(data: Vec<f32>, rows: usize, cols: usize) -> Sparse2D {
    let mut row_offsets = vec![0usize; rows + 1];
    let (mut col_indices, mut values, mut active_rows) = (Vec::new(), Vec::new(), Vec::new());

    for row in 0..rows {
        row_offsets[row] = col_indices.len();
        for col in 0..cols {
            let val = data[row * cols + col];
            if val != 0.0 {
                col_indices.push(col);
                values.push(val);
            }
        }
        if col_indices.len() > row_offsets[row] {
            active_rows.push(row);
        }
    }

    let non_zero_count = col_indices.len();
    row_offsets[rows] = non_zero_count;
    Sparse2D { data, rows, cols, row_offsets, col_indices, values, active_rows, non_zero_count }
}

fn make_conv(dense: &[f32], k_size: usize, rows: usize, cols: usize) -> SparseConv {
    let span = rows * cols;
    let mut taps = Vec::with_capacity(k_size);
    let mut nz = 0usize;

    for tap in 0..k_size {
        let s = make_sparse2d(dense[tap * span..(tap + 1) * span].to_vec(), rows, cols);
        nz += s.non_zero_count;
        taps.push(s);
    }
    SparseConv { k_size, rows, cols, taps, non_zero_count: nz }
}

fn parse_meta(entry: &str) -> (String, Vec<usize>, usize) {
    let mut parts = entry.split(':');
    let name = parts.next().unwrap_or("").to_string();
    let dims = parts
        .next()
        .unwrap_or("")
        .split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect();
    let count_str = parts.next().unwrap_or("0").trim();
    let count = count_str.parse().unwrap_or(0);
    (name, dims, count)
}

fn decode_trits(trit_buf: &[u8]) -> Vec<i8> {
    let mut flat = vec![0i8; trit_buf.len() * 5];
    for (byte_idx, &bv) in trit_buf.iter().enumerate() {
        for slot in 0..5 {
            let digit = (bv / POW3[slot]) % 3;
            flat[byte_idx * 5 + slot] = ((digit as i8 + 1) % 3) - 1;
        }
    }
    flat
}

fn unpack_weights(trit_buf: &[u8], vec_buf: &[f32], meta_str: &str) -> HygieneWeights {
    let (trit_part, vec_part) = meta_str.split_once('|').expect("meta '|' separator");
    let mut raw_vecs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut vec_off = 0usize;

    for entry in vec_part.split(" ; ") {
        let (name, _, count) = parse_meta(entry);
        raw_vecs.insert(name, vec_buf[vec_off..vec_off + count].to_vec());
        vec_off += count;
    }

    let flat_trits = decode_trits(trit_buf);
    let (mut mats, mut convs, mut byte_off) = (HashMap::new(), HashMap::new(), 0usize);

    for entry in trit_part.split(" ; ") {
        let (name, shape, packed) = parse_meta(entry);
        let n_elems: usize = shape.iter().product();
        let skey = if name == "col_sum_proj" {
            "col_sum_s".into()
        } else {
            format!("{}_s", name.strip_suffix("_w").unwrap_or(&name))
        };
        let dense = unpack_tensor(&flat_trits[byte_off * 5..], n_elems, raw_vecs.get(&skey));
        byte_off += packed;

        if shape.len() == 3 {
            convs.insert(name, make_conv(&dense, shape[0], shape[1], shape[2]));
        } else {
            mats.insert(name, make_sparse2d(dense, shape[0], shape[1]));
        }
    }

    let vecs = raw_vecs.into_iter().filter(|(k, _)| !k.ends_with("_s")).collect();
    HygieneWeights { mats, convs, vecs }
}

fn unpack_tensor(trits: &[i8], count: usize, scale: Option<&Vec<f32>>) -> Vec<f32> {
    let mut dense = vec![0.0f32; count];
    for idx in 0..count {
        let t = trits[idx];
        if t != 0 {
            dense[idx] = scale.map_or(t as f32, |sv| (t as f32) * sv[idx % sv.len()]);
        }
    }
    dense
}
