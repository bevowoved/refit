use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::{os_tag, check_rel, char_prefix, walk_files};
use crate::stem::{scan_edges, strip_noise, EdgeScan};
use crate::types::{AsrfHunk, DagPlan, ModuleUnit, PatchOp};
use anyhow::{Context, Result};
use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct AppliedPatch {
    pub updated: BTreeMap<String, String>,
    pub created: Vec<String>,
    pub deleted: Vec<String>,
    pub renamed: Vec<(String, String)>,
}

pub fn supported_ext(ext: &str) -> bool {
    matches!(ext, "py" | "rs" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "go")
}

pub fn hash_abi(rel_path: &str, source: &str) -> String {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "py" => python_abi(source),
        "rs" => rust_abi(source),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => ts_abi(source),
        "go" => go_abi(source),
        _ => MvccStore::hash_bytes(b""),
    }
}

/// Compute the Python exported ABI hash across multi-line definitions, classes, and `__all__`.
pub fn python_abi(source: &str) -> String {
    let mut sigs: Vec<String> = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut current_class: Option<String> = None;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let indent = line.bytes().take_while(|&b| b == b' ' || b == b'\t').count();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }
        if indent == 0 {
            if let Some(rest) = trimmed.strip_prefix("class ") {
                let cname: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !cname.is_empty() && !cname.starts_with('_') {
                    sigs.push(format!("class {cname}"));
                    current_class = Some(cname);
                } else {
                    current_class = None;
                }
                i += 1;
                continue;
            }
            if !trimmed.starts_with('@') {
                current_class = None;
            }
            if trimmed.starts_with("__all__") {
                let mut all_sig = trimmed.to_string();
                while paren_delta(&all_sig) > 0 && i + 1 < lines.len() {
                    i += 1;
                    all_sig.push(' ');
                    all_sig.push_str(lines[i].trim());
                }
                sigs.push(norm_sig(&all_sig));
                i += 1;
                continue;
            }
        }

        let is_def = trimmed.starts_with("def ") || trimmed.starts_with("async def ");
        if is_def && (indent == 0 || (indent <= 4 && current_class.is_some())) {
            let fn_rest = trimmed
                .strip_prefix("async def ")
                .or_else(|| trimmed.strip_prefix("def "))
                .unwrap_or("");
            let is_private = fn_rest.starts_with('_') && !fn_rest.starts_with("__init__");
            if !is_private {
                let mut sig = strip_hash(trimmed).to_string();
                let mut depth = paren_delta(&sig);
                while (depth > 0 || !def_closed(&sig)) && i + 1 < lines.len() {
                    i += 1;
                    sig.push(' ');
                    sig.push_str(strip_hash(lines[i].trim()));
                    depth = paren_delta(&sig);
                }
                if let Some(colon) = def_colon(&sig) {
                    sig.truncate(colon);
                }
                let norm = norm_params(&sig);
                if indent > 0 {
                    if let Some(ref cls) = current_class {
                        sigs.push(format!("{cls}.{norm}"));
                    }
                } else {
                    sigs.push(norm);
                }
            }
        }
        i += 1;
    }
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn strip_hash(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_q: Option<u8> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_q {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == q {
                in_q = None;
            }
        } else if b == b'"' || b == b'\'' {
            in_q = Some(b);
        } else if b == b'#' {
            return line[..i].trim_end();
        }
        i += 1;
    }
    line
}

fn paren_delta(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut d = 0i32;
    let mut in_q: Option<u8> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_q {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == q {
                in_q = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' | b'\'' => in_q = Some(b),
            b'(' | b'[' | b'{' => d += 1,
            b')' | b']' | b'}' => d -= 1,
            _ => {}
        }
        i += 1;
    }
    d
}

fn def_colon(sig: &str) -> Option<usize> {
    let bytes = sig.as_bytes();
    let mut depth = 0i32;
    let mut saw_close_paren = false;
    let mut in_q: Option<u8> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_q {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == q {
                in_q = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' | b'\'' => in_q = Some(b),
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if b == b')' && depth == 0 {
                    saw_close_paren = true;
                }
            }
            b':' if depth == 0 && saw_close_paren => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

fn def_closed(sig: &str) -> bool {
    def_colon(sig).is_some()
}

/// Normalize signature spacing so line wrapping does not change the ABI hash.
fn norm_sig(sig: &str) -> String {
    let mut s = collapse_ws(sig);
    for (from, to) in [
        ("( ", "("),
        (" )", ")"),
        ("< ", "<"),
        (" >", ">"),
        ("[ ", "["),
        (" ]", "]"),
        (" ,", ","),
        (",)", ")"),
        (",>", ">"),
    ] {
        while s.contains(from) {
            s = s.replace(from, to);
        }
    }
    s.trim().to_string()
}

/// Find the top-level body delimiter (`{`, `;`, or `=>`) outside `()`, `[]`, `<>`, and strings.
fn sig_end(sig: &str, allow_semi: bool, stop_arrow: bool) -> Option<usize> {
    let b = sig.as_bytes();
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut angle = 0i32;
    let mut in_q: Option<u8> = None;
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = in_q {
            if c == b'\\' && i + 1 < b.len() {
                i += 2;
                continue;
            }
            if c == q {
                in_q = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' || c == b'`' {
            in_q = Some(c);
            i += 1;
            continue;
        }
        if c == b'-' && i + 1 < b.len() && b[i + 1] == b'>' {
            i += 2;
            continue;
        }
        if c == b'=' && i + 1 < b.len() && b[i + 1] == b'>' {
            if stop_arrow && paren == 0 && bracket == 0 && angle == 0 {
                return Some(i);
            }
            i += 2;
            continue;
        }
        match c {
            b'(' => paren += 1,
            b')' => paren = (paren - 1).max(0),
            b'[' => bracket += 1,
            b']' => bracket = (bracket - 1).max(0),
            b'<' => angle += 1,
            b'>' => angle = (angle - 1).max(0),
            b'{' if paren == 0 && bracket == 0 && angle == 0 => {
                // Skip Go inline `interface{}` / `struct{}` when immediately followed by `}`.
                if i + 1 < b.len() && b[i + 1] == b'}' {
                    i += 2;
                    continue;
                }
                return Some(i);
            }
            b';' if allow_semi && paren == 0 && bracket == 0 && angle == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Compute the Rust exported ABI hash for `pub` and `pub(crate)` items outside `#[cfg(test)]`.
fn rust_abi(source: &str) -> String {
    let cleaned = strip_noise("rs", source);
    let mut sigs = Vec::new();
    let lines: Vec<&str> = cleaned.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        let is_pub = t.starts_with("pub fn ")
            || t.starts_with("pub async fn ")
            || t.starts_with("pub(crate) fn ")
            || t.starts_with("pub(crate) async fn ")
            || t.starts_with("pub struct ")
            || t.starts_with("pub(crate) struct ")
            || t.starts_with("pub enum ")
            || t.starts_with("pub(crate) enum ")
            || t.starts_with("pub trait ")
            || t.starts_with("pub(crate) trait ")
            || t.starts_with("pub type ")
            || t.starts_with("pub(crate) type ")
            || t.starts_with("pub const ")
            || t.starts_with("pub(crate) const ");
        if is_pub {
            let mut sig = t.to_string();
            while sig_end(&sig, true, false).is_none() && i + 1 < lines.len() {
                i += 1;
                sig.push(' ');
                sig.push_str(lines[i].trim());
            }
            let end = sig_end(&sig, true, false).unwrap_or(sig.len());
            sigs.push(norm_sig(&sig[..end]));
        }
        i += 1;
    }
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

/// Compute the TypeScript and JavaScript exported ABI hash.
fn ts_abi(source: &str) -> String {
    let mut sigs = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        if t.starts_with("export ") {
            let mut sig = t.to_string();
            if t.starts_with("export {") {
                while !sig.contains('}') && i + 1 < lines.len() {
                    i += 1;
                    sig.push(' ');
                    sig.push_str(lines[i].trim());
                }
                let head = sig.split(';').next().unwrap_or(&sig);
                sigs.push(norm_sig(head));
            } else if t.starts_with("export function ")
                || t.starts_with("export async function ")
                || t.starts_with("export default function ")
                || t.starts_with("export default async function ")
                || t.starts_with("export class ")
                || t.starts_with("export default class ")
                || t.starts_with("export interface ")
                || t.starts_with("export type ")
                || t.starts_with("export const ")
                || t.starts_with("export *")
            {
                while sig_end(&sig, true, true).is_none() && i + 1 < lines.len() {
                    i += 1;
                    sig.push(' ');
                    sig.push_str(lines[i].trim());
                }
                let end = sig_end(&sig, true, true).unwrap_or(sig.len());
                sigs.push(norm_sig(&sig[..end]));
            }
        }
        i += 1;
    }
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

/// Compute the Go exported ABI hash for functions, types, and receiver methods.
fn go_abi(source: &str) -> String {
    let mut sigs = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        let exported = if let Some(rest) = t.strip_prefix("func ") {
            let after_recv = if rest.starts_with('(') {
                rest.find(')').map(|p| rest[p + 1..].trim_start()).unwrap_or("")
            } else {
                rest.trim_start()
            };
            after_recv
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        } else if let Some(rest) = t.strip_prefix("type ") {
            rest.trim_start()
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        } else {
            false
        };
        if exported {
            let mut sig = t.to_string();
            while sig_end(&sig, false, false).is_none()
                && i + 1 < lines.len()
                && sig.contains('(')
                && paren_delta(&sig) > 0
            {
                i += 1;
                sig.push(' ');
                sig.push_str(lines[i].trim());
            }
            let end = sig_end(&sig, false, false).unwrap_or(sig.len());
            sigs.push(norm_sig(&sig[..end]));
        }
        i += 1;
    }
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn norm_params(sig: &str) -> String {
    if sig.starts_with("class ") {
        let head = sig.split('(').next().unwrap_or(sig).trim();
        return head.to_string();
    }
    let Some(open) = sig.find('(') else {
        return sig.trim().to_string();
    };
    let Some(close) = sig.rfind(')') else {
        return sig.trim().to_string();
    };
    let prefix = sig[..open].trim();
    let params_raw = &sig[open + 1..close];
    let mut names = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for ch in params_raw.chars() {
        match ch {
            '[' | '(' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            ']' | ')' | '}' => {
                depth = depth.saturating_sub(1);
                cur.push(ch);
            }
            ',' if depth == 0 => {
                let name = param_name(&cur);
                if !name.is_empty() {
                    names.push(name);
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    let last = param_name(&cur);
    if !last.is_empty() {
        names.push(last);
    }
    format!("{prefix}({})", names.join(", "))
}

fn param_name(param: &str) -> String {
    let p = param.trim();
    if p.is_empty() {
        return String::new();
    }
    let before_eq = p.split('=').next().unwrap_or(p).trim();
    let before_colon = before_eq.split(':').next().unwrap_or(before_eq).trim();
    before_colon.to_string()
}

#[cfg(test)]
pub fn dag_dir(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    hygiene: &HygieneScorer,
) -> Result<DagPlan> {
    build_dag(repo_root, stem_bin_hint, hygiene, None)
}

pub fn build_dag(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<DagPlan> {
    let discovery = scan_edges(repo_root, stem_bin_hint, preloaded);
    dag_from(repo_root, discovery, hygiene, preloaded)
}

pub fn dag_from(
    repo_root: &Path,
    discovery: EdgeScan,
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<DagPlan> {

    let mut source_files: Vec<String> = Vec::new();
    let mut test_files: Vec<String> = Vec::new();

    if let Some(map) = preloaded {
        for rel_str in map.keys() {
            let ext = Path::new(rel_str)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !supported_ext(&ext) || rel_str.ends_with("__init__.py") {
                continue;
            }
            if test_path(rel_str) {
                test_files.push(rel_str.clone());
            } else {
                source_files.push(rel_str.clone());
            }
        }
    } else {
        for path in walk_files(repo_root, None) {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !supported_ext(&ext) {
                continue;
            }
            let Ok(rel) = path.strip_prefix(repo_root) else {
                continue;
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if rel_str.ends_with("__init__.py") {
                continue;
            }
            if test_path(&rel_str) {
                test_files.push(rel_str);
            } else {
                source_files.push(rel_str);
            }
        }
    }

    source_files.sort();
    test_files.sort();

    let test_contents: Vec<(String, String)> = test_files
        .par_iter()
        .map(|tf| {
            let txt = if let Some(b) = preloaded.and_then(|m| m.get(tf)) {
                String::from_utf8_lossy(b).into_owned()
            } else {
                fs::read_to_string(repo_root.join(tf)).unwrap_or_default()
            };
            (tf.clone(), txt)
        })
        .collect();

    let mut mod_to_tests: HashMap<String, Vec<String>> = HashMap::new();
    for src in &source_files {
        let stem = Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let mut matched: BTreeSet<String> = test_files
            .iter()
            .filter(|t| {
                let ts = Path::new(t).file_stem().and_then(|s| s.to_str()).unwrap_or("");
                ts == format!("test_{stem}")
                    || ts.ends_with(&format!("_{stem}"))
                    || ts == format!("{stem}_test")
                    || ts == format!("{stem}.test")
                    || ts == format!("{stem}.spec")
            })
            .cloned()
            .collect();
        if matched.is_empty() && stem.len() >= 3 {
            let py_mod = src.trim_end_matches(".py").replace('/', ".");
            for (tf, body) in &test_contents {
                if body.contains(&py_mod)
                    || body.contains(&format!("import {stem}"))
                    || body.contains(&format!("from {stem}"))
                    || body.contains(&format!("/{stem}"))
                {
                    matched.insert(tf.clone());
                }
            }
        }
        mod_to_tests.insert(src.clone(), matched.into_iter().collect());
    }

    let mut g: DiGraph<String, ()> = DiGraph::new();
    let mut file_to_idx: HashMap<String, NodeIndex> = HashMap::new();
    for f in &source_files {
        let idx = g.add_node(f.clone());
        file_to_idx.insert(f.clone(), idx);
    }
    for (src, dst) in &discovery.edges {
        if let (Some(&u), Some(&v)) = (file_to_idx.get(src), file_to_idx.get(dst)) {
            if u != v && !g.contains_edge(u, v) {
                g.add_edge(u, v, ());
            }
        }
    }

    let sccs = tarjan_scc(&g);
    let mut file_to_scc: HashMap<String, usize> = HashMap::new();
    let mut scc_files_list: Vec<Vec<String>> = Vec::with_capacity(sccs.len());
    for (scc_idx, comp) in sccs.iter().enumerate() {
        let mut files: Vec<String> = comp.iter().map(|&ni| g[ni].clone()).collect();
        files.sort();
        for f in &files {
            file_to_scc.insert(f.clone(), scc_idx);
        }
        scc_files_list.push(files);
    }

    let n_sccs = scc_files_list.len();
    let mut scc_deps: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_sccs];
    let mut scc_dependents: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_sccs];

    for (src, dst) in &discovery.edges {
        if let (Some(&su), Some(&sv)) = (file_to_scc.get(src), file_to_scc.get(dst)) {
            if su != sv {
                scc_deps[su].insert(sv);
                scc_dependents[sv].insert(su);
            }
        }
    }

    let mut wave_of: Vec<usize> = vec![0; n_sccs];
    let mut in_deg: Vec<usize> = scc_deps.iter().map(|d| d.len()).collect();
    let mut queue: std::collections::VecDeque<usize> = in_deg
        .iter()
        .enumerate()
        .filter_map(|(i, &deg)| if deg == 0 { Some(i) } else { None })
        .collect();

    while let Some(u) = queue.pop_front() {
        let uw = wave_of[u];
        for &dep_of_u in &scc_dependents[u] {
            wave_of[dep_of_u] = wave_of[dep_of_u].max(uw + 1);
            in_deg[dep_of_u] = in_deg[dep_of_u].saturating_sub(1);
            if in_deg[dep_of_u] == 0 {
                queue.push_back(dep_of_u);
            }
        }
    }

    // Disambiguate colliding single-file stems (`index.ts`, `route.ts`, `mod.rs`) via canonical relative path.
    let mut stem_counts: HashMap<String, usize> = HashMap::new();
    for files in &scc_files_list {
        if files.len() == 1 {
            let st = Path::new(&files[0])
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("mod")
                .to_string();
            *stem_counts.entry(st).or_insert(0) += 1;
        }
    }

    let mut scc_ids: Vec<String> = scc_files_list
        .iter()
        .map(|files| {
            if files.len() == 1 {
                let rel = &files[0];
                let st = Path::new(rel)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mod");
                if stem_counts.get(st).copied().unwrap_or(0) > 1 {
                    rel.rsplit_once('.')
                        .map(|(a, _)| a.to_string())
                        .unwrap_or_else(|| rel.clone())
                } else {
                    st.to_string()
                }
            } else {
                let joined: Vec<String> = files
                    .iter()
                    .take(4)
                    .map(|f| {
                        Path::new(f)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("m")
                            .to_string()
                    })
                    .collect();
                if files.len() > 4 {
                    let h = MvccStore::hash_bytes(files.join(",").as_bytes());
                    format!("scc_{}_{}", joined.join("_"), &h[..6])
                } else {
                    format!("scc_{}", joined.join("_"))
                }
            }
        })
        .collect();
    // Final uniqueness pass for same-dir multi-extension files (e.g. `src/utils.ts` + `src/utils.py`) or colliding SCC names.
    let mut id_counts: HashMap<String, usize> = HashMap::new();
    for id in &scc_ids {
        *id_counts.entry(id.clone()).or_insert(0) += 1;
    }
    for i in 0..scc_ids.len() {
        if id_counts.get(&scc_ids[i]).copied().unwrap_or(0) > 1 {
            let files = &scc_files_list[i];
            if files.len() == 1 {
                scc_ids[i] = files[0].clone();
            } else {
                let h = MvccStore::hash_bytes(files.join(",").as_bytes());
                scc_ids[i] = format!("{}_{}", scc_ids[i], &h[..6]);
            }
        }
    }

    let built_units: Result<Vec<ModuleUnit>> = (0..n_sccs)
        .into_par_iter()
        .map(|i| {
            let files = &scc_files_list[i];
            let mut base_blake3 = BTreeMap::new();
            let mut abi_hash = BTreeMap::new();
            let mut unit_tests = BTreeSet::new();
            let mut unit_loc = 0usize;
            let mut score_sum = 0.0f64;

            for rel in files {
                let content = if let Some(b) = preloaded.and_then(|m| m.get(rel)) {
                    String::from_utf8_lossy(b).into_owned()
                } else {
                    let abs = repo_root.join(rel);
                    fs::read_to_string(&abs)
                        .with_context(|| format!("read source file {}", abs.display()))?
                };
                let loc = content.lines().count();
                unit_loc += loc;
                base_blake3.insert(rel.clone(), MvccStore::hash_bytes(content.as_bytes()));
                abi_hash.insert(rel.clone(), hash_abi(rel, &content));
                let h = hygiene.score_content(rel, &content);
                score_sum += h.score;
                if let Some(ts) = mod_to_tests.get(rel) {
                    for t in ts {
                        unit_tests.insert(t.clone());
                    }
                }
            }

            let wave = wave_of[i];
            let layer = infer_layer(&files[0], wave);
            let deps = scc_deps[i].iter().map(|&d| scc_ids[d].clone()).collect();
            let dependents = scc_dependents[i]
                .iter()
                .map(|&d| scc_ids[d].clone())
                .collect();

            Ok(ModuleUnit {
                id: scc_ids[i].clone(),
                files: files.clone(),
                layer,
                wave,
                deps,
                dependents,
                test_files: unit_tests.into_iter().collect(),
                loc: unit_loc,
                base_blake3,
                abi_hash,
                baseline_hygiene: score_sum / (files.len().max(1) as f64),
                // Guard ABI on giant accidental SCCs (>4 files); permit small 2-4 file mutual cycles.
                allow_abi_change: files.len() > 1 && files.len() <= 4,
            })
        })
        .collect();
    let mut units = built_units?;
    let total_loc: usize = units.iter().map(|u| u.loc).sum();
    let max_wave: usize = units.iter().map(|u| u.wave).max().unwrap_or(0);

    units.sort_by(|a, b| {
        let a_first = a.files.first().map(|s| s.as_str()).unwrap_or("");
        let b_first = b.files.first().map(|s| s.as_str()).unwrap_or("");
        let a_ext = Path::new(a_first).extension().and_then(|s| s.to_str()).unwrap_or("");
        let b_ext = Path::new(b_first).extension().and_then(|s| s.to_str()).unwrap_or("");
        let a_dir = Path::new(a_first).parent().and_then(|p| p.to_str()).unwrap_or("");
        let b_dir = Path::new(b_first).parent().and_then(|p| p.to_str()).unwrap_or("");
        a.wave
            .cmp(&b.wave)
            .then(a.layer.cmp(&b.layer))
            .then(a_ext.cmp(b_ext))
            .then(a_dir.cmp(b_dir))
            .then(a.id.cmp(&b.id))
    });

    Ok(DagPlan {
        repo_root: repo_root.display().to_string(),
        host_os: os_tag(),
        stem_used: discovery.used_stem,
        stem_index_ms: discovery.index_ms,
        total_files: source_files.len(),
        total_loc,
        scc_count: units.iter().filter(|u| u.files.len() > 1).count(),
        wave_count: if units.is_empty() { 0 } else { max_wave + 1 },
        units,
    })
}

fn test_path(rel_str: &str) -> bool {
    rel_str.starts_with("tests/")
        || rel_str.contains("/test_")
        || rel_str.starts_with("test_")
        || rel_str.ends_with("_test.go")
        || rel_str.contains(".test.")
        || rel_str.contains(".spec.")
}

/// Assign the module layer from folder conventions or the Kahn wave depth.
fn infer_layer(rel_path: &str, wave: usize) -> usize {
    if rel_path.contains("/core/") || rel_path.starts_with("core/") {
        0
    } else if rel_path.contains("/engine/") || rel_path.starts_with("engine/") {
        1
    } else if rel_path.contains("/services/") || rel_path.starts_with("services/") {
        2
    } else {
        wave.min(3)
    }
}

fn parse_diags(diagnostics: &[String]) -> HashSet<usize> {
    let mut lines = HashSet::new();
    for d in diagnostics {
        let bytes = d.as_bytes();
        // Match Python `line 123` / `Line 123`
        let lower = d.to_ascii_lowercase();
        let mut scan = lower.as_str();
        while let Some(pos) = scan.find("line ") {
            let after = &scan[pos + 5..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<usize>() {
                if n > 0 && n < 100_000 {
                    lines.insert(n);
                }
            }
            scan = after;
        }
        // Match word-bounded `L123` and file-line `:123` (skipping URL ports after `://`).
        for i in 0..bytes.len() {
            let is_l = bytes[i] == b'L'
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric());
            let is_colon = bytes[i] == b':'
                && !(i + 2 < bytes.len() && bytes[i + 1] == b'/' && bytes[i + 2] == b'/')
                && !d[..i].ends_with("http")
                && !d[..i].ends_with("https")
                && !d[..i].ends_with("127.0.0.1")
                && !d[..i].ends_with("localhost");
            if is_l || is_colon {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                let ok_end = j == bytes.len() || !bytes[j].is_ascii_alphabetic();
                if j > i + 1 && ok_end {
                    if let Ok(n) = d[i + 1..j].parse::<usize>() {
                        if n > 0 && n < 100_000 {
                            lines.insert(n);
                        }
                    }
                }
            }
        }
    }
    lines
}

fn failed_test(test_src: &str, diagnostics: &[String], max_chars: usize) -> String {
    // Locate failing test names across Python (`::test_`), Rust (`probe_`/`test_`), Go (`FAIL: Test`), and JS/TS (`test`/`it`).
    let mut failing_names: Vec<String> = Vec::new();
    for d in diagnostics {
        for marker in ["::test_", "::probe_", "FAIL: Test", "test "] {
            let mut rest = d.as_str();
            while let Some(pos) = rest.find(marker) {
                let tail = match marker {
                    "::test_" | "::probe_" => &rest[pos + 2..],
                    "FAIL: Test" => &rest[pos + 6..],
                    _ => &rest[pos + marker.len()..],
                };
                let name: String = tail
                    .split("::")
                    .last()
                    .unwrap_or(tail)
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if name.len() >= 4 && !failing_names.contains(&name) {
                    failing_names.push(name);
                }
                rest = &rest[pos + marker.len()..];
            }
        }
    }
    if !failing_names.is_empty() {
        let src_lines: Vec<&str> = test_src.lines().collect();
        let mut out = String::new();
        for fname in &failing_names {
            let needles = [
                format!("def {fname}"),
                format!("fn {fname}"),
                format!("func {fname}"),
                format!("'{fname}'"),
                format!("\"{fname}\""),
            ];
            if let Some(start) = src_lines
                .iter()
                .position(|l| needles.iter().any(|n| l.contains(n.as_str())))
            {
                let end = (start + 45).min(src_lines.len());
                out.push_str(&src_lines[start..end].join("\n"));
                out.push('\n');
                if out.chars().count() >= max_chars {
                    break;
                }
            }
        }
        if !out.is_empty() {
            return char_prefix(&out, max_chars).to_string();
        }
    }
    char_prefix(test_src, max_chars).to_string()
}

pub fn build_packet(
    repo_root: &Path,
    unit: &ModuleUnit,
    diagnostics: &[String],
    learn_suffix: &str,
    max_chars: usize,
) -> Result<String> {
    let mut out = String::new();
    out.push_str(&format!(
        "Refactor unit `{}` (wave {}, layer {}, baseline hygiene {:.2}).\n",
        unit.id, unit.wave, unit.layer, unit.baseline_hygiene
    ));

    let unique_diags: BTreeSet<&str> = diagnostics
        .iter()
        .map(|d| d.trim())
        .filter(|d| !d.is_empty())
        .collect();
    if !unique_diags.is_empty() {
        out.push_str("Current Gate 1/2/3 diagnostics to fix completely:\n");
        for d in unique_diags.iter().take(36) {
            out.push_str("- ");
            out.push_str(d);
            out.push('\n');
        }
    }
    if unit.files.len() > 1 {
        out.push_str("IMPORTANT: This cyclic SCC unit has multiple files. Emit <<<REFIT_PATCH>>> blocks to fix ALL defects in EVERY file below.\n");
    }

    let has_test_fail = unique_diags
        .iter()
        .any(|d| d.to_lowercase().contains("pytest") || d.contains("FAILED") || d.contains("assert") || d.contains("Gate 1"));
    if has_test_fail {
        for tf in unit.test_files.iter().take(2) {
            if let Ok(tc) = fs::read_to_string(repo_root.join(tf)) {
                let snippet = failed_test(&tc, diagnostics, 2400);
                out.push_str(&format!("\nRelevant failing test `{tf}`:\n```\n{snippet}\n```\n"));
            }
        }
    }

    let diag_lines = parse_diags(diagnostics);
    for rel in &unit.files {
        let content = fs::read_to_string(repo_root.join(rel))
            .with_context(|| format!("read {rel}"))?;
        let sliced = slice_anchored(
            &content,
            max_chars / unit.files.len().max(1),
            &diag_lines,
        );
        out.push_str(&format!("\nFile `{rel}`:\n```\n{sliced}\n```\n"));
    }

    if !learn_suffix.is_empty() {
        out.push_str(learn_suffix);
    }

    out.push_str(
        "\nEmit ONLY raw <<<REFIT_PATCH path=\"...\">>> blocks directly in your reply. Every <<<SEARCH>>> block must match uniquely. Preserve exported signatures unless allow_abi_change is enabled.\n",
    );
    Ok(out)
}

fn top_decl(line: &str) -> bool {
    !line.starts_with(' ')
        && !line.starts_with('\t')
        && (line.starts_with("def ")
            || line.starts_with("async def ")
            || line.starts_with("class ")
            || line.starts_with("pub fn ")
            || line.starts_with("pub async fn ")
            || line.starts_with("fn ")
            || line.starts_with("export ")
            || line.starts_with("function ")
            || line.starts_with("func "))
}

#[cfg(test)]
fn slice_source(content: &str, budget_chars: usize) -> String {
    slice_anchored(content, budget_chars, &HashSet::new())
}

/// Anchors micro-slicing directly around parsed diagnostic line numbers ±25 lines so diagnostic code is NEVER hidden.
fn slice_anchored(
    content: &str,
    budget_chars: usize,
    diag_lines: &HashSet<usize>,
) -> String {
    if content.chars().count() <= budget_chars {
        return content.to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    let mut blocks: Vec<(usize, usize, bool)> = Vec::new();
    let mut header_end = 0;
    while header_end < lines.len() {
        if top_decl(lines[header_end]) && header_end > 0 {
            break;
        }
        header_end += 1;
    }
    blocks.push((0, header_end, true));
    let mut i = header_end;

    while i < lines.len() {
        let start = i;
        i += 1;
        while i < lines.len() {
            if top_decl(lines[i]) {
                break;
            }
            i += 1;
        }
        let slice = &lines[start..i];
        let touches_diag = diag_lines
            .iter()
            .any(|&dl| dl >= start.saturating_sub(25) + 1 && dl <= i + 25);
        let is_hot = touches_diag
            || slice.iter().any(|l| {
                let indent = l.len().saturating_sub(l.trim_start().len());
                indent >= 12
                    || l.chars().count() > 88
                    || l.contains("=[]")
                    || l.contains("={}")
                    || l.contains("None")
                    || l.trim().starts_with("elif ")
                    || l.contains("TODO")
                    || l.contains("unwrap()")
            });
        blocks.push((start, i, is_hot));
    }

    let mut out = String::new();
    let mut out_chars = 0usize;
    let mut omitted = 0usize;
    for (start, end, hot) in blocks {
        let chunk = lines[start..end].join("\n");
        let chunk_chars = chunk.chars().count();
        if hot || out_chars + chunk_chars <= budget_chars {
            if omitted > 0 {
                let marker = format!("\n// ... ({omitted} clean lines omitted) ...\n");
                out_chars += marker.chars().count();
                out.push_str(&marker);
                omitted = 0;
            }
            out.push_str(&chunk);
            out.push('\n');
            out_chars += chunk_chars + 1;
        } else {
            if let Some(first) = lines.get(start) {
                if top_decl(first) {
                    let stub = format!("// stub: {}\n", char_prefix(first.trim(), 120));
                    out_chars += stub.chars().count();
                    out.push_str(&stub);
                }
            }
            omitted += end.saturating_sub(start);
        }
    }
    out
}

/// Parses raw ASRF hunks supporting `op="edit"|"create"|"delete"|"rename"` and `new_path="..."`.
pub fn parse_patches(content: &str) -> Vec<AsrfHunk> {
    let mut hunks = Vec::new();
    let norm = content.replace("\r\n", "\n");
    let mut cursor = 0usize;

    while let Some(rel_start) = norm[cursor..].find("<<<REFIT_PATCH") {
        let start = cursor + rel_start;
        let Some(header_end_rel) = norm[start..].find(">>>") else {
            break;
        };
        let header = &norm[start..start + header_end_rel];
        let path = extract_attr(header, "path=");
        let op_raw = extract_attr(header, "op=");
        let new_path_raw = extract_attr(header, "new_path=");
        let op = match op_raw.to_ascii_lowercase().as_str() {
            "create" => PatchOp::Create,
            "delete" => PatchOp::Delete,
            "rename" | "move" => PatchOp::Rename,
            _ => PatchOp::Edit,
        };
        let new_path = if new_path_raw.is_empty() {
            None
        } else {
            Some(new_path_raw.replace('\\', "/"))
        };
        let after_header = start + header_end_rel + 3;
        let Some(end_rel_from_hdr) = norm[after_header..].find("<<<END>>>") else {
            break;
        };
        let block_body = &norm[after_header..after_header + end_rel_from_hdr];
        let end_pos = after_header + end_rel_from_hdr;

        let (search, replace) = if let (Some(sr), Some(rr)) =
            (block_body.find("<<<SEARCH>>>"), block_body.find("<<<REPLACE>>>"))
        {
            if rr >= sr + "<<<SEARCH>>>".len() {
                let s_body = &block_body[sr + "<<<SEARCH>>>".len()..rr];
                let r_body = &block_body[rr + "<<<REPLACE>>>".len()..];
                (
                    trim_newline(s_body),
                    trim_newline(r_body),
                )
            } else {
                cursor = end_pos + "<<<END>>>".len();
                continue;
            }
        } else if let Some(rr) = block_body.find("<<<REPLACE>>>") {
            // Allow concise `op="create"` blocks with only `<<<REPLACE>>>...<<<END>>>`.
            let r_body = &block_body[rr + "<<<REPLACE>>>".len()..];
            (String::new(), trim_newline(r_body))
        } else if matches!(op, PatchOp::Delete | PatchOp::Rename) {
            // Allow concise `op="delete"` / `op="rename"` without empty SEARCH/REPLACE markers.
            (String::new(), String::new())
        } else {
            cursor = end_pos + "<<<END>>>".len();
            continue;
        };

        let valid = match op {
            PatchOp::Edit => !path.is_empty() && !search.is_empty(),
            PatchOp::Create => !path.is_empty(),
            PatchOp::Delete => !path.is_empty(),
            PatchOp::Rename => !path.is_empty() && new_path.is_some(),
        };
        if valid {
            hunks.push(AsrfHunk {
                path: path.replace('\\', "/"),
                op,
                new_path,
                search,
                replace,
            });
        }

        cursor = end_pos + "<<<END>>>".len();
    }

    hunks
}

fn extract_attr(header: &str, key: &str) -> String {
    // Require leading whitespace boundary so `path=` NEVER matches `new_path=`.
    for (pos, _) in header.match_indices(key) {
        if pos > 0 && !header.as_bytes()[pos - 1].is_ascii_whitespace() {
            continue;
        }
        let rest = header[pos + key.len()..].trim_start();
        if let Some(first) = rest.chars().next() {
            if first == '"' || first == '\'' {
                let inner = &rest[1..];
                if let Some(end) = inner.find(first) {
                    return inner[..end].to_string();
                }
            }
        }
        return rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c| c == '"' || c == '\'')
            .to_string();
    }
    String::new()
}

fn trim_newline(s: &str) -> String {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.to_string()
}

#[cfg(test)]
pub fn apply_edits(
    base_files: &BTreeMap<String, String>,
    hunks: &[AsrfHunk],
) -> Result<BTreeMap<String, String>> {
    Ok(apply_patches(base_files, hunks)?.updated)
}

/// Applies edit/create/delete/rename hunks, enforces repo-relative path safety, AND enforces unique-match safety.
pub fn apply_patches(
    base_files: &BTreeMap<String, String>,
    hunks: &[AsrfHunk],
) -> Result<AppliedPatch> {
    if hunks.is_empty() {
        anyhow::bail!("No <<<REFIT_PATCH>>> blocks found in assistant output");
    }
    let mut patch = AppliedPatch {
        updated: base_files.clone(),
        created: Vec::new(),
        deleted: Vec::new(),
        renamed: Vec::new(),
    };

    for (idx, h) in hunks.iter().enumerate() {
        if !check_rel(&h.path) {
            anyhow::bail!("Hunk #{idx} unsafe path outside repository: `{}`", h.path);
        }
        if let Some(ref np) = h.new_path {
            if !check_rel(np) {
                anyhow::bail!("Hunk #{idx} unsafe new_path outside repository: `{np}`");
            }
        }
        match h.op {
            PatchOp::Create => {
                let next = h.replace.replace("\r\n", "\n");
                patch.updated.insert(h.path.clone(), next);
                if !patch.created.contains(&h.path) {
                    patch.created.push(h.path.clone());
                }
                patch.deleted.retain(|d| d != &h.path);
                continue;
            }
            PatchOp::Delete => {
                let target_key = match_target(&patch.updated, &h.path, idx, false)?;
                patch.updated.remove(&target_key);
                patch.created.retain(|c| c != &target_key);
                if !patch.deleted.contains(&target_key) {
                    patch.deleted.push(target_key);
                }
                continue;
            }
            PatchOp::Rename => {
                let target_key = match_target(&patch.updated, &h.path, idx, false)?;
                let dest = h
                    .new_path
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Hunk #{idx} rename missing new_path"))?;
                let mut body = patch
                    .updated
                    .remove(&target_key)
                    .unwrap_or_default()
                    .replace("\r\n", "\n");
                if !h.search.is_empty() {
                    body = apply_hunk(&body, &h.search, &h.replace, idx, &target_key)?;
                }
                if target_key != dest {
                    if !patch.deleted.contains(&target_key) {
                        patch.deleted.push(target_key.clone());
                    }
                    patch.deleted.retain(|d| d != &dest);
                    if !patch.created.contains(&dest) {
                        patch.created.push(dest.clone());
                    }
                    patch.renamed.push((target_key, dest.clone()));
                }
                patch.updated.insert(dest, body);
                continue;
            }
            PatchOp::Edit => {
                let target_key = match_target(&patch.updated, &h.path, idx, true)?;
                let cur = patch
                    .updated
                    .get(&target_key)
                    .cloned()
                    .unwrap_or_default()
                    .replace("\r\n", "\n");
                let next = apply_hunk(&cur, &h.search, &h.replace, idx, &target_key)?;
                patch.updated.insert(target_key, next);
            }
        }
    }
    Ok(patch)
}

fn match_target(
    updated: &BTreeMap<String, String>,
    path: &str,
    idx: usize,
    allow_single_fallback: bool,
) -> Result<String> {
    if updated.contains_key(path) {
        Ok(path.to_string())
    } else if let Some(k) = updated.keys().find(|k| {
        k.ends_with(&format!("/{path}")) || path.ends_with(&format!("/{k}"))
    }) {
        Ok(k.clone())
    } else if allow_single_fallback && updated.len() == 1 {
        Ok(updated.keys().next().unwrap().clone())
    } else {
        anyhow::bail!("Hunk #{idx} targets unknown path `{path}`");
    }
}

fn apply_hunk(
    cur: &str,
    search: &str,
    replace: &str,
    idx: usize,
    target_key: &str,
) -> Result<String> {
    let search_norm = search.replace("\r\n", "\n");
    let replace_norm = replace.replace("\r\n", "\n");

    // Check uniqueness on exact match too so duplicate lines never silently patch the wrong site.
    let exact_matches: Vec<usize> = cur.match_indices(&search_norm).map(|(p, _)| p).collect();
    if exact_matches.len() > 1 {
        anyhow::bail!(
            "Hunk #{idx} ambiguous SEARCH block matched {} locations in `{target_key}`; include 2-3 surrounding context lines",
            exact_matches.len()
        );
    }
    if let Some(&pos) = exact_matches.first() {
        let mut next = String::with_capacity(cur.len() + replace_norm.len());
        next.push_str(&cur[..pos]);
        next.push_str(&replace_norm);
        next.push_str(&cur[pos + search_norm.len()..]);
        Ok(next)
    } else if let Some(next) = replace_trimmed(cur, &search_norm, &replace_norm) {
        Ok(next)
    } else if let Some(next) = replace_elided(cur, &search_norm, &replace_norm) {
        Ok(next)
    } else if let Some(next) = replace_ws(cur, &search_norm, &replace_norm) {
        Ok(next)
    } else {
        anyhow::bail!(
            "Hunk #{idx} SEARCH block did not match `{target_key}` (first line: `{}`)",
            search_norm.lines().next().unwrap_or("")
        );
    }
}

fn is_elision(line: &str) -> bool {
    let t = line.trim().to_ascii_lowercase();
    t.contains("... existing code ...")
        || t.contains("... unchanged ...")
        || t == "// ..."
        || t == "# ..."
        || t == "/* ... */"
}

fn find_block(doc_lines: &[&str], needle: &[&str], from_idx: usize) -> Option<usize> {
    if needle.is_empty() || from_idx + needle.len() > doc_lines.len() {
        return None;
    }
    let mut found: Option<usize> = None;
    for i in from_idx..=doc_lines.len() - needle.len() {
        let ok = needle
            .iter()
            .enumerate()
            .all(|(j, nl)| doc_lines[i + j].trim() == nl.trim());
        if ok {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found
}

fn replace_elided(doc: &str, search: &str, replace: &str) -> Option<String> {
    let s_lines: Vec<&str> = search.lines().collect();
    let r_lines: Vec<&str> = replace.lines().collect();
    let s_mark = s_lines.iter().position(|l| is_elision(l))?;
    let r_mark = r_lines.iter().position(|l| is_elision(l))?;
    if s_lines[s_mark + 1..].iter().any(|l| is_elision(l))
        || r_lines[r_mark + 1..].iter().any(|l| is_elision(l))
    {
        return None;
    }
    let s_head = &s_lines[..s_mark];
    let s_tail = &s_lines[s_mark + 1..];
    if s_head.is_empty() || s_tail.is_empty() {
        return None;
    }
    let doc_lines: Vec<&str> = doc.lines().collect();
    let head_start = find_block(&doc_lines, s_head, 0)?;
    let head_end = head_start + s_head.len();
    let tail_start = find_block(&doc_lines, s_tail, head_end)?;
    let tail_end = tail_start + s_tail.len();

    let mut out_lines: Vec<String> = Vec::with_capacity(doc_lines.len() + r_lines.len());
    for l in &doc_lines[..head_start] {
        out_lines.push((*l).to_string());
    }
    for l in &r_lines[..r_mark] {
        out_lines.push((*l).to_string());
    }
    for l in &doc_lines[head_end..tail_start] {
        out_lines.push((*l).to_string());
    }
    for l in &r_lines[r_mark + 1..] {
        out_lines.push((*l).to_string());
    }
    for l in &doc_lines[tail_end..] {
        out_lines.push((*l).to_string());
    }
    let mut joined = out_lines.join("\n");
    if doc.ends_with('\n') {
        joined.push('\n');
    }
    Some(joined)
}

fn collapse_ws(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn replace_ws(doc: &str, search: &str, replace: &str) -> Option<String> {
    let doc_lines: Vec<&str> = doc.lines().collect();
    let s_norm: Vec<String> = search.lines().map(collapse_ws).collect();
    if s_norm.is_empty() || s_norm.len() > doc_lines.len() {
        return None;
    }
    let doc_norm: Vec<String> = doc_lines.iter().map(|l| collapse_ws(l)).collect();
    let mut match_idx: Option<usize> = None;
    for i in 0..=doc_norm.len() - s_norm.len() {
        if (0..s_norm.len()).all(|j| doc_norm[i + j] == s_norm[j]) {
            if match_idx.is_some() {
                return None;
            }
            match_idx = Some(i);
        }
    }
    let start = match_idx?;
    let mut out_lines: Vec<String> = Vec::with_capacity(doc_lines.len());
    for l in &doc_lines[..start] {
        out_lines.push((*l).to_string());
    }
    for l in replace.lines() {
        out_lines.push(l.to_string());
    }
    for l in &doc_lines[start + s_norm.len()..] {
        out_lines.push((*l).to_string());
    }
    let mut joined = out_lines.join("\n");
    if doc.ends_with('\n') {
        joined.push('\n');
    }
    Some(joined)
}

pub fn impacted_tests(plan: &DagPlan, changed_files: &[String]) -> Vec<String> {
    let by_id: HashMap<&str, &ModuleUnit> = plan.units.iter().map(|u| (u.id.as_str(), u)).collect();
    let changed_set: HashSet<String> = changed_files
        .iter()
        .map(|s| s.replace('\\', "/"))
        .collect();

    let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut visited: HashSet<String> = HashSet::new();
    for u in &plan.units {
        if u.files.iter().any(|f| changed_set.contains(f)) && visited.insert(u.id.clone()) {
            queue.push_back(u.id.clone());
        }
    }

    let mut tests: BTreeSet<String> = BTreeSet::new();
    while let Some(uid) = queue.pop_front() {
        if let Some(u) = by_id.get(uid.as_str()) {
            for t in &u.test_files {
                tests.insert(t.clone());
            }
            for dep in &u.dependents {
                if visited.insert(dep.clone()) {
                    queue.push_back(dep.clone());
                }
            }
        }
    }
    tests.into_iter().collect()
}

pub fn critical_depths(units: &[ModuleUnit]) -> HashMap<String, usize> {
    let by_id: HashMap<&str, &ModuleUnit> = units.iter().map(|u| (u.id.as_str(), u)).collect();
    let mut memo: HashMap<String, usize> = HashMap::with_capacity(units.len());

    fn dfs(
        id: &str,
        by_id: &HashMap<&str, &ModuleUnit>,
        memo: &mut HashMap<String, usize>,
        visiting: &mut HashSet<String>,
    ) -> usize {
        if let Some(&d) = memo.get(id) {
            return d;
        }
        if !visiting.insert(id.to_string()) {
            return 0;
        }
        let max_child = by_id
            .get(id)
            .map(|u| {
                u.dependents
                    .iter()
                    .map(|dep| 1 + dfs(dep, by_id, memo, visiting))
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        visiting.remove(id);
        memo.insert(id.to_string(), max_child);
        max_child
    }

    let mut visiting = HashSet::new();
    for u in units {
        dfs(&u.id, &by_id, &mut memo, &mut visiting);
    }
    memo
}

/// Build exported ABI context for the direct upstream dependencies of `unit`.
pub fn deps_context(
    unit: &ModuleUnit,
    all_units: &[ModuleUnit],
    repo_root: &Path,
) -> String {
    if unit.deps.is_empty() {
        return String::new();
    }
    let by_id: HashMap<&str, &ModuleUnit> = all_units.iter().map(|u| (u.id.as_str(), u)).collect();
    let mut dep_units: Vec<ModuleUnit> = Vec::new();
    for dep_id in &unit.deps {
        if let Some(u) = by_id.get(dep_id.as_str()) {
            dep_units.push((*u).clone());
        }
    }
    abi_summary("UPSTREAM DEPENDENCY ABI (read-only imports used by this unit):", &dep_units, repo_root, 2400)
}

#[cfg(test)]
pub fn wave_context(wave_units: &[ModuleUnit], repo_root: &Path) -> String {
    abi_summary("WAVE SHARED ABI PREFIX (read-only across wave siblings):", wave_units, repo_root, 1200)
}

fn abi_summary(
    header: &str,
    units: &[ModuleUnit],
    repo_root: &Path,
    max_chars: usize,
) -> String {
    let mut rels: BTreeSet<String> = BTreeSet::new();
    for u in units {
        for f in &u.files {
            rels.insert(f.clone());
        }
    }
    let rel_vec: Vec<String> = rels.into_iter().collect();
    let lines: Vec<String> = rel_vec
        .par_iter()
        .filter_map(|rel| {
            let src = fs::read_to_string(repo_root.join(rel)).ok()?;
            let mut sigs: Vec<String> = Vec::new();
            for l in src.lines() {
                let t = l.trim();
                if top_decl(l)
                    || t.starts_with("pub struct ")
                    || t.starts_with("pub enum ")
                    || t.starts_with("export interface ")
                    || t.starts_with("export type ")
                {
                    let head = t.split('{').next().unwrap_or(t).trim();
                    sigs.push(char_prefix(head, 96).to_string());
                    if sigs.len() >= 6 {
                        break;
                    }
                }
            }
            if sigs.is_empty() {
                None
            } else {
                Some(format!("- {rel}: {}", sigs.join(" | ")))
            }
        })
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    let mut out = format!("{header}\n");
    for line in lines {
        if out.chars().count() + line.chars().count() + 1 > max_chars {
            break;
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn replace_trimmed(doc: &str, search: &str, replace: &str) -> Option<String> {
    let doc_lines: Vec<&str> = doc.lines().collect();
    let s_lines: Vec<&str> = search.lines().collect();
    if s_lines.is_empty() || s_lines.len() > doc_lines.len() {
        return None;
    }
    let mut match_idx: Option<usize> = None;
    for i in 0..=doc_lines.len() - s_lines.len() {
        let ok = s_lines
            .iter()
            .enumerate()
            .all(|(j, sl)| doc_lines[i + j].trim() == sl.trim());
        if ok {
            if match_idx.is_some() {
                return None;
            }
            match_idx = Some(i);
        }
    }
    let start = match_idx?;
    let mut out_lines: Vec<String> = Vec::new();
    for l in &doc_lines[..start] {
        out_lines.push((*l).to_string());
    }
    for l in replace.lines() {
        out_lines.push(l.to_string());
    }
    for l in &doc_lines[start + s_lines.len()..] {
        out_lines.push((*l).to_string());
    }
    let mut joined = out_lines.join("\n");
    if doc.ends_with('\n') {
        joined.push('\n');
    }
    Some(joined)
}

pub fn units_interfere(a: &ModuleUnit, b: &ModuleUnit) -> bool {
    if a.id == b.id {
        return true;
    }
    let a_files: HashSet<&String> = a.files.iter().collect();
    if b.files.iter().any(|f| a_files.contains(f)) {
        return true;
    }
    a.deps.contains(&b.id) || b.deps.contains(&a.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;

    #[test]
    fn probe_17_diag_sort_dedup() {
        let dir = std::env::temp_dir().join(format!("refit_slice_d_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.py"), "def f(x):\n    return x\n").unwrap();
        let unit = ModuleUnit {
            id: "a".into(),
            files: vec!["a.py".into()],
            layer: 0,
            wave: 0,
            deps: vec![],
            dependents: vec![],
            test_files: vec![],
            loc: 2,
            base_blake3: BTreeMap::new(),
            abi_hash: BTreeMap::new(),
            baseline_hygiene: 0.70,
            allow_abi_change: false,
        };
        let p1 = build_packet(&dir, &unit, &["z_err".into(), "a_err".into(), "z_err".into()], "", 4000).unwrap();
        let p2 = build_packet(&dir, &unit, &["a_err".into(), "z_err".into()], "", 4000).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.matches("z_err").count(), 1);
        assert!(p1.find("a_err").unwrap() < p1.find("z_err").unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_18_slice_anchors() {
        let mut src = String::from("import os\n\n");
        for i in 0..25 {
            src.push_str(&format!("def clean_{i}(x: int) -> int:\n    return x + {i}\n\n"));
        }
        src.push_str("def hot_target(items=[]):\n    if items:\n        if len(items) > 1:\n            return items[0]\n    return None\n");
        let sliced = slice_source(&src, 400);
        assert!(sliced.len() < src.len());
        assert!(sliced.contains("def hot_target(items=[]):"));
        assert!(sliced.contains("// stub: def clean_"));
    }

    #[test]
    fn probe_19_slice_utf8() {
        let mut src = String::from("# こんにちは世界 🚀 café résumé\n");
        for i in 0..40 {
            src.push_str(&format!("def fn_{i}():\n    return '🚀日本語テスト_{i}'\n\n"));
        }
        let sliced = slice_source(&src, 120);
        assert!(!sliced.is_empty());
        assert_eq!(char_prefix("🚀日本語", 2), "🚀日");
    }

    #[test]
    fn probe_25_polyglot_dag() {
        let dir = std::env::temp_dir().join(format!("refit_poly_dag_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/a.py"), "def py_fn(x):\n    return x\n").unwrap();
        fs::write(dir.join("src/b.rs"), "pub fn rs_fn(x: i32) -> i32 { x }\n").unwrap();
        fs::write(dir.join("src/c.ts"), "export function tsFn(x: number) { return x; }\n").unwrap();
        fs::write(dir.join("src/D.tsx"), "export function Card() { return <div/>; }\n").unwrap();
        fs::write(dir.join("src/e.js"), "export function jsFn(x) { return x; }\n").unwrap();
        fs::write(dir.join("src/f.go"), "package main\nfunc GoFn(x int) int { return x }\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        assert_eq!(plan.total_files, 6);
        assert_eq!(plan.units.len(), 6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_26_polyglot_abi() {
        assert_ne!(
            hash_abi("a.rs", "pub fn calc(x: i32) -> i32 { x }"),
            hash_abi("a.rs", "pub fn calc_v2(x: i32) -> i32 { x }")
        );
        assert_ne!(
            hash_abi("b.ts", "export function render(id: string) { return id; }"),
            hash_abi("b.ts", "export function renderCard(id: string) { return id; }")
        );
        assert_ne!(
            hash_abi("c.go", "package p\nfunc Build(x int) int { return x }"),
            hash_abi("c.go", "package p\nfunc BuildAll(x int) int { return x }")
        );
    }

    #[test]
    fn probe_27_python_abi() {
        let untyped = "def compute(a, b=1):\n    return a + b\n";
        let typed = "def compute(a: int, b: int = 1) -> int:\n    return a + b\n";
        let renamed = "def compute(x: int, b: int = 1) -> int:\n    return x + b\n";
        assert_eq!(python_abi(untyped), python_abi(typed));
        assert_ne!(python_abi(untyped), python_abi(renamed));
    }

    #[test]
    fn probe_28_tarjan_scc() {
        let dir = std::env::temp_dir().join(format!("refit_scc_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("pkg")).unwrap();
        fs::write(dir.join("pkg/a.py"), "from pkg.b import fb\ndef fa():\n    return fb()\n").unwrap();
        fs::write(dir.join("pkg/b.py"), "from pkg.a import fa\ndef fb():\n    return fa()\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        assert_eq!(plan.scc_count, 1);
        assert_eq!(plan.units.len(), 1);
        assert_eq!(plan.units[0].files.len(), 2);
        assert!(plan.units[0].allow_abi_change);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_29_kahn_waves() {
        let dir = std::env::temp_dir().join(format!("refit_kahn_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("pkg")).unwrap();
        fs::write(dir.join("pkg/leaf.py"), "def base():\n    return 1\n").unwrap();
        fs::write(dir.join("pkg/mid.py"), "from pkg.leaf import base\ndef mid():\n    return base()\n").unwrap();
        fs::write(dir.join("pkg/top.py"), "from pkg.mid import mid\ndef top():\n    return mid()\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        let w = |id: &str| plan.units.iter().find(|u| u.id == id).unwrap().wave;
        assert_eq!(w("leaf"), 0);
        assert_eq!(w("mid"), 1);
        assert_eq!(w("top"), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_30_unit_disjoint() {
        let mk = |id: &str, file: &str, deps: Vec<&str>| ModuleUnit {
            id: id.into(),
            files: vec![file.into()],
            layer: 0,
            wave: 0,
            deps: deps.into_iter().map(String::from).collect(),
            dependents: vec![],
            test_files: vec![],
            loc: 1,
            base_blake3: BTreeMap::new(),
            abi_hash: BTreeMap::new(),
            baseline_hygiene: 0.9,
            allow_abi_change: false,
        };
        let a = mk("a", "a.py", vec![]);
        let b = mk("b", "b.py", vec!["a"]);
        let c = mk("c", "c.py", vec![]);
        assert!(units_interfere(&a, &b));
        assert!(!units_interfere(&a, &c));
    }

    #[test]
    fn probe_31_patch_apply() {
        let mut base = BTreeMap::new();
        base.insert("pkg/a.py".into(), "x = 1\r\ny = 2\r\n".into());
        let patch = "<<<REFIT_PATCH path=\"a.py\">>>\r\n<<<SEARCH>>>\r\nx = 1\r\n<<<REPLACE>>>\r\nx = 10\r\n<<<END>>>\r\n<<<REFIT_PATCH path=\"pkg/a.py\">>>\r\n<<<SEARCH>>>\r\ny = 2\r\n<<<REPLACE>>>\r\ny = 20\r\n<<<END>>>";
        let hunks = parse_patches(patch);
        assert_eq!(hunks.len(), 2);
        let out = apply_edits(&base, &hunks).unwrap();
        assert_eq!(out["pkg/a.py"], "x = 10\ny = 20\n");
    }

    #[test]
    fn probe_32_patch_ambiguous() {
        let doc = "    val = 1\n    val = 1\n";
        assert!(replace_trimmed(doc, "val = 1", "val = 2").is_none());
    }

    #[test]
    fn probe_33_patch_invalid() {
        let mut base = BTreeMap::new();
        base.insert("a.py".into(), "x = 1\n".into());
        base.insert("b.py".into(), "y = 2\n".into());
        assert!(apply_edits(&base, &[]).is_err());
        let hunks = parse_patches("<<<REFIT_PATCH path=\"ghost.py\">>>\n<<<SEARCH>>>\nx = 1\n<<<REPLACE>>>\nx = 2\n<<<END>>>");
        assert!(apply_edits(&base, &hunks).is_err());
    }

    #[test]
    fn probe_54_wave_cluster() {
        let dir = std::env::temp_dir().join(format!("refit_kvcluster_{}_{}", std::process::id(), crate::platform::now_ms()));
        fs::create_dir_all(dir.join("alpha")).unwrap();
        fs::create_dir_all(dir.join("beta")).unwrap();
        fs::write(dir.join("alpha/z_mod.rs"), "pub fn z() -> i32 { 1 }\n").unwrap();
        fs::write(dir.join("alpha/a_mod.py"), "def a():\n    return 1\n").unwrap();
        fs::write(dir.join("beta/b_mod.py"), "def b():\n    return 2\n").unwrap();
        fs::write(dir.join("alpha/c_mod.rs"), "pub fn c() -> i32 { 3 }\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        let exts: Vec<&str> = plan
            .units
            .iter()
            .map(|u| Path::new(&u.files[0]).extension().and_then(|s| s.to_str()).unwrap_or(""))
            .collect();
        assert_eq!(exts, vec!["py", "py", "rs", "rs"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_57_elision_patch() {
        let mut base = BTreeMap::new();
        base.insert(
            "src/calc.rs".into(),
            "pub fn run(x: i32) -> i32 {\n    let step1 = x + 1;\n    let untouched_middle = step1 * 42;\n    untouched_middle\n}\n"
                .into(),
        );
        let patch = "<<<REFIT_PATCH path=\"src/calc.rs\">>>\n<<<SEARCH>>>\npub fn run(x: i32) -> i32 {\n// ... existing code ...\n    untouched_middle\n}\n<<<REPLACE>>>\npub fn run(x: i32) -> i32 {\n    let _checked = x >= 0;\n// ... existing code ...\n    untouched_middle + 1\n}\n<<<END>>>";
        let hunks = parse_patches(patch);
        let out = apply_edits(&base, &hunks).unwrap();
        assert!(out["src/calc.rs"].contains("let untouched_middle = step1 * 42;"));
        assert!(out["src/calc.rs"].contains("let _checked = x >= 0;"));
        assert!(out["src/calc.rs"].contains("untouched_middle + 1"));

        let mut amb = BTreeMap::new();
        amb.insert("src/dup.py".into(), "def dup():\n    return 1\ndef dup():\n    return 2\n".into());
        let bad_patch = "<<<REFIT_PATCH path=\"src/dup.py\">>>\n<<<SEARCH>>>\ndef dup():\n# ... existing code ...\n    return 2\n<<<REPLACE>>>\ndef dup():\n# ... existing code ...\n    return 3\n<<<END>>>";
        assert!(apply_edits(&amb, &parse_patches(bad_patch)).is_err());
    }

    #[test]
    fn probe_59_wave_abi_ctx() {
        let dir = std::env::temp_dir().join(format!("refit_wabi_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/a.py"), "def alpha(x: int) -> int:\n    return x\n").unwrap();
        fs::write(dir.join("src/b.rs"), "pub fn beta(y: i32) -> i32 { y }\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        let fwd = wave_context(&plan.units, &dir);
        let mut rev_units = plan.units.clone();
        rev_units.reverse();
        let rev = wave_context(&rev_units, &dir);
        assert_eq!(fwd, rev);
        assert!(fwd.chars().count() <= 1200);
        assert!(fwd.contains("def alpha"));
        assert!(fwd.contains("pub fn beta"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_70_critical_path() {
        let dir = std::env::temp_dir().join(format!("refit_cp70_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("pkg")).unwrap();
        fs::create_dir_all(dir.join("tests")).unwrap();
        fs::write(dir.join("pkg/root_core.py"), "def core_v():\n    return 1\n").unwrap();
        fs::write(dir.join("pkg/mid_layer.py"), "from pkg.root_core import core_v\ndef mid_v():\n    return core_v()\n").unwrap();
        fs::write(dir.join("pkg/top_api.py"), "from pkg.mid_layer import mid_v\ndef top_v():\n    return mid_v()\n").unwrap();
        fs::write(dir.join("pkg/isolated.py"), "def iso():\n    return 0\n").unwrap();
        fs::write(dir.join("tests/test_smoke_suite.py"), "from pkg.root_core import core_v\ndef test_c():\n    assert core_v() == 1\n").unwrap();

        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        let depths = critical_depths(&plan.units);
        assert_eq!(depths["root_core"], 2);
        assert_eq!(depths["mid_layer"], 1);
        assert_eq!(depths["top_api"], 0);
        assert_eq!(depths["isolated"], 0);

        let root_unit = plan.units.iter().find(|u| u.id == "root_core").unwrap();
        let iso_unit = plan.units.iter().find(|u| u.id == "isolated").unwrap();
        assert!(root_unit.test_files.contains(&"tests/test_smoke_suite.py".to_string()));
        assert!(iso_unit.test_files.is_empty());

        let cone_root = impacted_tests(&plan, &["pkg/root_core.py".into()]);
        assert_eq!(cone_root, vec!["tests/test_smoke_suite.py".to_string()]);
        let cone_iso = impacted_tests(&plan, &["pkg/isolated.py".into()]);
        assert!(cone_iso.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_81_abi_and_unique() {
        let py_a = "def calc(\n    x: int,\n    y: str,\n) -> float:\n    return 1.0\n";
        let py_b = "def calc(\n    x: int,\n    z: str,\n) -> float:\n    return 1.0\n";
        assert_ne!(python_abi(py_a), python_abi(py_b));

        let go_a = "package p\nfunc (s *Svc) Handle(id string) error { return nil }\n";
        let go_b = "package p\nfunc (s *Svc) HandleV2(id string) error { return nil }\n";
        assert_ne!(hash_abi("svc.go", go_a), hash_abi("svc.go", go_b));

        let mut dup = BTreeMap::new();
        dup.insert("a.py".into(), "x = 1\nx = 1\n".into());
        let hunks = parse_patches("<<<REFIT_PATCH path=\"a.py\">>>\n<<<SEARCH>>>\nx = 1\n<<<REPLACE>>>\nx = 2\n<<<END>>>");
        assert!(apply_patches(&dup, &hunks).is_err(), "ambiguous exact match must fail safely");
    }

    #[test]
    fn probe_82_patch_ops_ids() {
        let mut base = BTreeMap::new();
        base.insert("src/old.ts".into(), "export const a = 1;\n".into());
        base.insert("src/dead.ts".into(), "export const d = 0;\n".into());
        let patch = concat!(
            "<<<REFIT_PATCH path=\"src/shared.ts\" op=\"create\">>>\n<<<SEARCH>>>\n<<<REPLACE>>>\nexport const shared = 42;\n<<<END>>>\n",
            "<<<REFIT_PATCH path=\"src/dead.ts\" op=\"delete\">>>\n<<<SEARCH>>>\n<<<REPLACE>>>\n<<<END>>>\n",
            "<<<REFIT_PATCH path=\"src/old.ts\" op=\"rename\" new_path=\"src/core/new.ts\">>>\n<<<SEARCH>>>\na = 1\n<<<REPLACE>>>\na = 2\n<<<END>>>"
        );
        let hunks = parse_patches(patch);
        let res = apply_patches(&base, &hunks).unwrap();
        assert!(res.created.contains(&"src/shared.ts".to_string()));
        assert!(res.deleted.contains(&"src/dead.ts".to_string()));
        assert!(res.updated.contains_key("src/core/new.ts"));
        assert!(res.updated["src/core/new.ts"].contains("a = 2"));

        // Monorepo colliding stems (`app/auth/route.ts` vs `app/billing/route.ts`) must get distinct unit IDs!
        let dir = std::env::temp_dir().join(format!("refit_mono82_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("app/auth")).unwrap();
        fs::create_dir_all(dir.join("app/billing")).unwrap();
        fs::write(dir.join("app/auth/route.ts"), "export function GET() { return 1; }\n").unwrap();
        fs::write(dir.join("app/billing/route.ts"), "export function GET() { return 2; }\n").unwrap();
        let plan = dag_dir(&dir, Some(Path::new("/nonexistent/stem")), &HygieneScorer::new()).unwrap();
        let ids: HashSet<&str> = plan.units.iter().map(|u| u.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "colliding stems must disambiguate: {ids:?}");
        assert!(ids.contains("app/auth/route"));
        assert!(ids.contains("app/billing/route"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_96_patch_security() {
        let mut base = BTreeMap::new();
        base.insert("src/a.rs".into(), "pub fn keep(a: i32, b: i32) -> i32 { a + b }\n".into());
        // Traversal & metadata targets must be rejected.
        for evil in [
            "<<<REFIT_PATCH path=\"../../pwn.rs\" op=\"create\">>>\n<<<REPLACE>>>\npwn\n<<<END>>>",
            "<<<REFIT_PATCH path=\".git/hooks/pre-commit\" op=\"create\">>>\n<<<REPLACE>>>\npwn\n<<<END>>>",
            "<<<REFIT_PATCH op=\"rename\" new_path=\"../escape.rs\" path=\"src/a.rs\">>>\n<<<END>>>",
        ] {
            let hunks = parse_patches(evil);
            assert!(apply_patches(&base, &hunks).is_err(), "must reject {evil}");
        }
        // `new_path=` before `path=` must extract `path` cleanly, and concise `op="delete"` works.
        let rename_first = "<<<REFIT_PATCH op=\"rename\" new_path=\"src/b.rs\" path=\"src/a.rs\">>>\n<<<END>>>";
        let h_ren = parse_patches(rename_first);
        assert_eq!(h_ren[0].path, "src/a.rs");
        assert_eq!(h_ren[0].new_path.as_deref(), Some("src/b.rs"));
        let res = apply_patches(&base, &h_ren).unwrap();
        assert!(res.updated.contains_key("src/b.rs"));
        assert!(res.deleted.contains(&"src/a.rs".to_string()));

        // Formatter-wrapped Rust & TS destructured signatures keep identical ABI hashes.
        let rs_single = "pub fn compute(a: i32, b: i32) -> i32 { a + b }\n";
        let rs_wrapped = "pub fn compute(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a + b\n}\n";
        assert_eq!(hash_abi("x.rs", rs_single), hash_abi("x.rs", rs_wrapped));

        let ts_destr = "export function Card({ title }: Props) { return title; }\n";
        let ts_destr_changed = "export function Card({ label }: Props) { return label; }\n";
        assert_ne!(hash_abi("c.tsx", ts_destr), hash_abi("c.tsx", ts_destr_changed));
    }
}
