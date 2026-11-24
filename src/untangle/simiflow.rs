//! SimiFlow fuzzy helper & capability clustering engine:
//! Stage 1: Corpus `df > 10%` stop-token filter + IDF weighting
//! Stage 2: IDF-weighted 64-bit `SimHash`
//! Stage 3: 16 overlapping 8-bit LSH bands capped at `B_MAX = 32` + Hamming `<= 16`
//! Stage 4: Exact IDF-Weighted Jaccard (`>= 0.55`) + `strsim::jaro_winkler`

use crate::source::stem_of;
use crate::stem::witness::rapid_hash64;
use crate::types::RoleCluster;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::path::Path;
use strsim::jaro_winkler;

const N_BANDS: usize = 16;
const B_MAX: usize = 32;
const HAMMING_MAX: u32 = 16;
const WEIGHTED_JACCARD_MIN: f64 = 0.55;
const JARO_WINKLER_MIN: f64 = 0.80;
const STRONG_BODY_JACCARD: f64 = 0.85;

const SYNTAX_STOP: &[&str] = &[
    "const", "let", "var", "return", "export", "import", "from", "function", "async", "await",
    "pub", "fn", "struct", "enum", "impl", "trait", "mod", "use", "crate", "self", "super",
    "def", "class", "pass", "none", "true", "false", "func", "package", "type", "interface",
    "string", "number", "boolean", "void", "usize", "u32", "u64", "i32", "i64", "f64", "bool",
    "if", "else", "elif", "for", "while", "match", "break", "continue", "in", "of", "new",
];

#[derive(Debug, Clone)]
pub struct SimiItem {
    pub name: String,
    pub body: String,
}

#[derive(Debug, Clone, Copy)]
pub struct SimiMatch {
    pub a: usize,
    pub b: usize,
}

#[derive(Debug, Clone)]
pub struct SimiStats {
    pub matches: Vec<SimiMatch>,
    pub lsh_candidates: usize,
    pub hot_buckets_capped: usize,
}

fn simhash64_idf(tok_weights: &[(u64, f32)]) -> u64 {
    let mut v = [0.0f32; 64];
    for &(h, w) in tok_weights {
        for (bit, acc) in v.iter_mut().enumerate() {
            if (h & (1u64 << bit)) != 0 {
                *acc += w;
            } else {
                *acc -= w;
            }
        }
    }
    let mut fp = 0u64;
    for (bit, &acc) in v.iter().enumerate() {
        if acc > 0.0 {
            fp |= 1u64 << bit;
        }
    }
    fp
}

fn weighted_jaccard(a: &[(u64, f32)], b: &[(u64, f32)]) -> f64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut inter = 0.0f64;
    let mut uni = 0.0f64;
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            inter += a[i].1.min(b[j].1) as f64;
            uni += a[i].1.max(b[j].1) as f64;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            uni += a[i].1 as f64;
            i += 1;
        } else {
            uni += b[j].1 as f64;
            j += 1;
        }
    }
    while i < a.len() {
        uni += a[i].1 as f64;
        i += 1;
    }
    while j < b.len() {
        uni += b[j].1 as f64;
        j += 1;
    }
    if uni > 0.0 { inter / uni } else { 0.0 }
}

fn extract_domain_token_hashes(body: &str, filter_syntax: bool) -> Vec<u64> {
    let mut toks = Vec::with_capacity(24);
    for raw in body.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if raw.len() < 2 || raw.as_bytes()[0].is_ascii_digit() {
            continue;
        }
        let lower = raw.to_ascii_lowercase();
        if filter_syntax && SYNTAX_STOP.contains(&lower.as_str()) {
            continue;
        }
        toks.push(rapid_hash64(lower.as_bytes(), 0x333));
    }
    toks.sort_unstable();
    toks.dedup();
    toks
}

/// Core 4-stage SimiFlow matcher (`df > 10%` stop filter + IDF SimHash + 16-band LSH `B_max=32`
/// + exact Weighted Jaccard `>= 0.55` + `strsim::jaro_winkler >= 0.80`).
pub fn find_similar_items(items: &[SimiItem], filter_syntax: bool) -> SimiStats {
    let n = items.len();
    if n < 2 {
        return SimiStats {
            matches: Vec::new(),
            lsh_candidates: 0,
            hot_buckets_capped: 0,
        };
    }

    let df_stop_cap = (n / 10).max(8) as u32;
    let mut tok_df: FxHashMap<u64, u32> =
        FxHashMap::with_capacity_and_hasher(n * 8 + 64, Default::default());
    let mut raw_doc_toks: Vec<Vec<u64>> = Vec::with_capacity(n);

    for item in items {
        let toks = extract_domain_token_hashes(&item.body, filter_syntax);
        for &h in &toks {
            *tok_df.entry(h).or_insert(0) += 1;
        }
        raw_doc_toks.push(toks);
    }

    let mut bands: Vec<Vec<Vec<usize>>> = (0..N_BANDS).map(|_| vec![Vec::new(); 256]).collect();
    let mut doc_vecs: Vec<Vec<(u64, f32)>> = Vec::with_capacity(n);
    let mut fps: Vec<u64> = Vec::with_capacity(n);
    let n_f = (n as f64) + 1.0;

    for (idx, toks) in raw_doc_toks.into_iter().enumerate() {
        let mut weighted: Vec<(u64, f32)> = Vec::with_capacity(toks.len());
        for h in toks {
            let df = tok_df.get(&h).copied().unwrap_or(1);
            if df > df_stop_cap {
                continue;
            }
            let idf = ((n_f / ((df as f64) + 1.0)).ln() + 1.0) as f32;
            weighted.push((h, idf));
        }
        let fp = if weighted.len() >= 3 {
            let f = simhash64_idf(&weighted);
            for (b, table) in bands.iter_mut().enumerate() {
                let key = ((f.rotate_left((b * 4) as u32)) & 0xff) as usize;
                table[key].push(idx);
            }
            f
        } else {
            0
        };
        fps.push(fp);
        doc_vecs.push(weighted);
    }

    let mut lsh_cands: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut hot_buckets_capped = 0usize;
    for table in &bands {
        for bucket in table {
            if bucket.len() < 2 {
                continue;
            }
            if bucket.len() > B_MAX {
                hot_buckets_capped += 1;
                continue;
            }
            for i in 0..bucket.len() {
                let a = bucket[i];
                for &c in &bucket[i + 1..] {
                    let (lo, hi) = if a < c { (a, c) } else { (c, a) };
                    if (fps[lo] ^ fps[hi]).count_ones() <= HAMMING_MAX {
                        lsh_cands.insert((lo, hi));
                    }
                }
            }
        }
    }

    let mut matched_pairs: Vec<SimiMatch> = Vec::new();
    for &(lo, hi) in &lsh_cands {
        let wj = weighted_jaccard(&doc_vecs[lo], &doc_vecs[hi]);
        if wj >= WEIGHTED_JACCARD_MIN {
            let jw = jaro_winkler(&items[lo].name, &items[hi].name);
            if jw >= JARO_WINKLER_MIN || wj >= STRONG_BODY_JACCARD {
                matched_pairs.push(SimiMatch { a: lo, b: hi });
            }
        }
    }
    matched_pairs.sort_by_key(|m| (m.a, m.b));

    SimiStats {
        matches: matched_pairs,
        lsh_candidates: lsh_cands.len(),
        hot_buckets_capped,
    }
}

fn split_ident_lower(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        let camel = ch.is_ascii_uppercase()
            && cur.chars().last().is_some_and(|c| c.is_ascii_lowercase());
        if !ch.is_ascii_alphanumeric() {
            if !cur.is_empty() {
                tokens.push(cur.to_ascii_lowercase());
                cur.clear();
            }
        } else if camel {
            tokens.push(cur.to_ascii_lowercase());
            cur.clear();
            cur.push(ch);
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        tokens.push(cur.to_ascii_lowercase());
    }
    tokens
}

fn canonical_stem(rel: &str) -> String {
    let p = Path::new(rel);
    let stem = stem_of(p);
    if matches!(stem, "index" | "mod" | "lib" | "main")
        && let Some(dir) = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str())
    {
        return dir.to_ascii_lowercase();
    }
    stem.to_ascii_lowercase()
}

fn infer_capability_label(files: &[String]) -> String {
    let mut counts: FxHashMap<String, usize> = FxHashMap::default();
    for rel in files {
        let st = canonical_stem(rel);
        for tok in split_ident_lower(&st) {
            if tok.len() >= 2 && !tok.bytes().all(|b| b.is_ascii_digit()) && !matches!(tok.as_str(), "v1" | "v2" | "v3") {
                *counts.entry(tok).or_insert(0) += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(k, _)| k)
        .unwrap_or_else(|| "shared_helper".to_string())
}

/// Discover duplicated capability/helper modules via SimiFlow IDF SimHash + LSH + Weighted Jaccard.
pub fn find_capability_clusters(
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
) -> Vec<RoleCluster> {
    if source_rels.len() < 2 {
        return Vec::new();
    }
    let items: Vec<SimiItem> = source_rels
        .iter()
        .map(|rel| {
            let body = preloaded
                .get(rel)
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default();
            SimiItem {
                name: canonical_stem(rel),
                body,
            }
        })
        .collect();

    let stats = find_similar_items(&items, true);
    tracing::debug!(
        lsh_candidates = stats.lsh_candidates,
        hot_buckets_capped = stats.hot_buckets_capped,
        matches = stats.matches.len(),
        "simiflow capability clustering complete"
    );
    if stats.matches.is_empty() {
        return Vec::new();
    }

    // Union-Find over matched file pairs into deduplicated capability clusters.
    let mut parent: Vec<usize> = (0..source_rels.len()).collect();
    fn find_root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for m in &stats.matches {
        let ra = find_root(&mut parent, m.a);
        let rb = find_root(&mut parent, m.b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for i in 0..source_rels.len() {
        let r = find_root(&mut parent, i);
        groups.entry(r).or_default().push(source_rels[i].clone());
    }

    let mut clusters: Vec<RoleCluster> = groups
        .into_values()
        .filter(|g| g.len() >= 2)
        .map(|mut files| {
            files.sort();
            let cap = infer_capability_label(&files);
            let msg = format!(
                "Detected {} structurally similar `{cap}` modules; consolidate into one canonical module",
                files.len()
            );
            RoleCluster {
                capability: cap,
                files,
                recommendation: msg,
            }
        })
        .collect();
    clusters.sort_by(|a, b| {
        b.files
            .len()
            .cmp(&a.files.len())
            .then_with(|| a.capability.cmp(&b.capability))
            .then_with(|| a.files.cmp(&b.files))
    });
    clusters.truncate(20);
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simiflow_idf_bmax32_and_zero_false_positives() {
        const DOMAINS: [(&str, &str); 25] = [
            ("AuthToken", "header bearer jwt signature expiry issuer audience claims decode parse slice trim"),
            ("DateEpoch", "timestamp millis timezone offset isoformat calendar leapyear weekday utc epoch parse clamp"),
            ("CsvEscape", "delimiter quote comma newline backslash buffer append row column field escape parser"),
            ("PathCanon", "segment dotdot slash root normalize symlink prefix mount drive relative join clean"),
            ("CacheKey", "namespace version shard bucket digest prefix suffix ttl evict lookup entry key"),
            ("SlugTitle", "unicode ascii hyphen lowercase collapse whitespace diacritic url permalink title slug trim"),
            ("CidrMatch", "ipv4 ipv6 subnet mask prefixlen octets bitwise network broadcast range cidr route"),
            ("MimeSniff", "magic bytes header charset subtype boundary multipart contenttype media sniff detect table"),
            ("SemverCmp", "major minor patch prerelease buildmeta caret tilde comparator range version order parse"),
            ("RetryWait", "backoff jitter exponential cap attempt deadline timeout sleep schedule delay factor limit"),
            ("RateLimit", "tokenbucket refill capacity burst window quota throttle counter sliding clock permit allow"),
            ("CsrfGuard", "nonce cookie origin referer double submit secret hmac constanttime compare token guard"),
            ("HtmlClean", "tag attribute script iframe sanitize entity strip whitespace dom fragment node filter"),
            ("YamlQuote", "scalar indent block fold anchor alias colon boolean null string quote emitter"),
            ("UrlEncode", "percent hex query fragment reserved unreserved plus space decode encode uri param"),
            ("HashChunk", "block digest state update finalize merkle leaf branch padding buffer stream chunk"),
            ("ByteOrder", "endian swap little big network host u16 u32 u64 slice read write"),
            ("LineWrap", "column width break indent hyphen word margin paragraph wrap measure glyph line"),
            ("AnsiStrip", "escape bracket csi sgr color reset cursor sequence terminal strip control byte"),
            ("GlobMatch", "wildcard star question bracket negate recursive segment directory pattern glob match path"),
            ("SqlBind", "placeholder parameter dialect escape quote prepared query tuple value bind statement exec"),
            ("XmlEntity", "ampersand lt gt apos quot numeric codepoint cdata namespace element decode entity"),
            ("HexDigest", "nibble lookup lower upper encode decode byte pair buffer validate length checksum"),
            ("UuidParse", "variant version hyphen canonical bytes clockseq node timestamp random parse format uuid"),
            ("DiffHunk", "context insert delete line range header patch offset apply merge hunk diff"),
        ];

        let mut items = Vec::with_capacity(1_000);
        for (i, &(dom, vocab)) in DOMAINS.iter().enumerate() {
            items.push(SimiItem {
                name: format!("normalize{dom}V1"),
                body: format!("const return async await ctx logger {vocab} verify_{i}"),
            });
            items.push(SimiItem {
                name: format!("normalize{dom}V2"),
                body: format!("const return async await ctx logger {vocab} check_{i}"),
            });
        }
        for i in 50..1_000 {
            items.push(SimiItem {
                name: format!("domainSpecificOp{i:04}"),
                body: format!(
                    "const return async await unique pipeline transform stage alpha_{} beta_{} gamma_{}",
                    i * 3 + 1,
                    i * 7 + 2,
                    i * 13 + 3
                ),
            });
        }

        let stats = find_similar_items(&items, false);
        assert!(stats.lsh_candidates <= 150, "too many LSH candidates: {}", stats.lsh_candidates);
        let true_hits = stats
            .matches
            .iter()
            .filter(|m| m.a < 50 && m.a % 2 == 0 && m.b == m.a + 1)
            .count();
        assert!(true_hits >= 24, "expected >= 24/25 domain twins, got {true_hits}");
        assert_eq!(
            stats.matches.len(),
            true_hits,
            "expected 0 false positives, got {}",
            stats.matches.len().saturating_sub(true_hits)
        );

        // Verify hot-bucket skew cap (`B_max = 32`) triggers when 45 docs (df <= 10% of 500)
        // collide into the same LSH bucket (`45 > B_MAX = 32`).
        let mut skew_items = Vec::with_capacity(500);
        for i in 0..45 {
            skew_items.push(SimiItem {
                name: format!("hotBucketTwin_{i}"),
                body: "merkle digest verify payload signature nonce".into(),
            });
        }
        for i in 45..500 {
            skew_items.push(SimiItem {
                name: format!("distinctOp_{i}"),
                body: format!("tok_a_{i} tok_b_{i} tok_c_{i} tok_d_{i}"),
            });
        }
        let skew_stats = find_similar_items(&skew_items, false);
        assert!(skew_stats.hot_buckets_capped >= N_BANDS);
        assert_eq!(skew_stats.matches.len(), 0);
    }

    #[test]
    fn test_simiflow_matches_identical_4line_auth_bodies() {
        let block = concat!(
            "    const normalized = raw.trim().toLowerCase();\n",
            "    const parts = normalized.split(':');\n",
            "    const checksum = parts.length * 31;\n",
            "    const output = checksum + normalized.length;\n",
        );
        let a1 = format!("export function authOne(raw: string) {{\n{block}    return output;\n}}\n");
        let a2 = format!("export function authTwo(raw: string) {{\n{block}    return output;\n}}\n");
        let mut preloaded = BTreeMap::new();
        preloaded.insert("src/auth_v1.ts".to_string(), a1.into_bytes());
        preloaded.insert("src/auth_v2.ts".to_string(), a2.into_bytes());
        preloaded.insert("src/app.py".to_string(), b"import os\n".to_vec());
        preloaded.insert("src/main.go".to_string(), b"package main\nfunc main() {}\n".to_vec());
        let rels: Vec<String> = preloaded.keys().cloned().collect();

        let clusters = find_capability_clusters(&rels, &preloaded);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].files, vec!["src/auth_v1.ts", "src/auth_v2.ts"]);
    }
}
