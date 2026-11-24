use crate::platform::walk_files;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[inline]
pub fn ext_str(p: &Path) -> &str {
    p.extension().and_then(|s| s.to_str()).unwrap_or("")
}

pub fn ext_of(p: impl AsRef<Path>) -> String {
    ext_str(p.as_ref()).to_ascii_lowercase()
}

#[inline]
pub fn stem_of(p: &Path) -> &str {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("")
}

pub fn dir_of(p: impl AsRef<Path>) -> String {
    let Some(d) = p.as_ref().parent() else { return String::new() };
    d.to_string_lossy().replace('\\', "/")
}

pub fn rel_norm(root: &Path, abs: &Path) -> Option<String> {
    let rel = abs.strip_prefix(root).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    Some(rel.to_string_lossy().replace('\\', "/"))
}

pub fn work_dir(repo_root: &Path, dot_name: &str, tmp_name: &str) -> PathBuf {
    let resolved = dunce::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    let parent = resolved.parent().filter(|p| !p.as_os_str().is_empty());
    match parent {
        Some(p) => p.join(dot_name),
        None => std::env::temp_dir().join(tmp_name),
    }
}

/// Count logical text lines from raw bytes without UTF-8 decoding.
#[inline]
pub fn count_lines(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    let nl = bytes.iter().filter(|&&b| b == b'\n').count();
    nl + usize::from(bytes.last() != Some(&b'\n'))
}

/// Compute quantile `q` in `[0.0, 1.0]` by sorting `vals` in place.
pub fn quantile(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = if (q - 0.5).abs() < 1e-9 {
        vals.len() / 2
    } else {
        ((vals.len() as f64) * q).floor() as usize
    };
    vals[idx.min(vals.len() - 1)]
}

#[inline]
pub fn is_js(ext: &str) -> bool {
    const JS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];
    JS.iter().any(|&e| ext.eq_ignore_ascii_case(e))
}

pub fn ext_lang(ext: &str) -> Option<&'static str> {
    if ext.eq_ignore_ascii_case("py") {
        return Some("python");
    }
    if ext.eq_ignore_ascii_case("rs") {
        return Some("rust");
    }
    if ext.eq_ignore_ascii_case("go") {
        return Some("go");
    }
    if ext.eq_ignore_ascii_case("ts") || ext.eq_ignore_ascii_case("tsx") {
        return Some("typescript");
    }
    is_js(ext).then_some("javascript")
}

#[inline]
pub fn supported_ext(ext: &str) -> bool {
    ext_lang(ext).is_some()
}

#[inline]
pub fn same_family(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b) || (is_js(a) && is_js(b))
}

pub fn test_path(rel: &str) -> bool {
    rel.starts_with("tests/")
        || rel.starts_with("test_")
        || rel.contains("/test_")
        || rel.ends_with("_test.go")
        || rel.contains(".test.")
        || rel.contains(".spec.")
}

#[inline]
pub fn source_rel(rel: &str) -> bool {
    !rel.ends_with("__init__.py") && !test_path(rel)
}

pub fn entrypoint(rel: &str) -> bool {
    const ENTRIES: &[&str] = &[
        "main.rs", "lib.rs", "main.py", "main.go",
        "index.ts", "index.js", "page.tsx", "route.ts",
    ];
    ENTRIES.iter().any(|e| rel.ends_with(e))
}

pub struct RepoScan {
    pub sources: Vec<(String, String, Option<PathBuf>)>,
    pub tests: Vec<(String, String, Option<PathBuf>)>,
    pub all: Vec<(String, String, Option<PathBuf>)>,
}

/// Partition repository source and test files in a single pass.
pub fn scan_files(root: &Path, preloaded: Option<&BTreeMap<String, Vec<u8>>>) -> RepoScan {
    let mut out = RepoScan { sources: Vec::new(), tests: Vec::new(), all: Vec::new() };

    let mut push = |rel: String, ext: String, abs: Option<PathBuf>| {
        out.all.push((rel.clone(), ext.clone(), abs.clone()));
        if rel.ends_with("__init__.py") {
            return;
        }
        if test_path(&rel) {
            out.tests.push((rel, ext, abs));
        } else {
            out.sources.push((rel, ext, abs));
        }
    };

    if let Some(map) = preloaded {
        for rel in map.keys() {
            let ext = ext_of(rel);
            if supported_ext(&ext) {
                push(rel.clone(), ext, None);
            }
        }
    } else {
        for path in walk_files(root, None) {
            let ext = ext_of(&path);
            if let Some(rel) = supported_ext(&ext).then(|| rel_norm(root, &path)).flatten() {
                push(rel, ext, Some(path));
            }
        }
    }

    out.sources.sort_by(|a, b| a.0.cmp(&b.0));
    out.tests.sort_by(|a, b| a.0.cmp(&b.0));
    out.all.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Borrow bytes from `preloaded` without cloning, or read from disk once.
pub fn read_bytes<'a>(
    root: &Path,
    rel: &str,
    preloaded: Option<&'a BTreeMap<String, Vec<u8>>>,
) -> Option<Cow<'a, [u8]>> {
    if let Some(bytes) = preloaded.and_then(|m| m.get(rel)) {
        return Some(Cow::Borrowed(bytes.as_slice()));
    }
    std::fs::read(root.join(rel)).ok().map(Cow::Owned)
}

/// Borrow UTF-8 text from `preloaded` with zero copies when valid UTF-8.
pub fn read_text<'a>(
    root: &Path,
    rel: &str,
    preloaded: Option<&'a BTreeMap<String, Vec<u8>>>,
) -> Option<Cow<'a, str>> {
    if let Some(bytes) = preloaded.and_then(|m| m.get(rel)) {
        return Some(String::from_utf8_lossy(bytes));
    }
    std::fs::read_to_string(root.join(rel)).ok().map(Cow::Owned)
}

#[derive(Debug, Clone)]
pub struct ManifestDep {
    pub ecosystem: &'static str,
    pub package: String,
    pub is_local: bool,
}

/// Parse dependencies from `package.json`, `Cargo.toml`, `pyproject.toml`, or `go.mod`.
pub fn parse_deps(file_name: &str, txt: &str) -> Vec<ManifestDep> {
    let mut out = Vec::new();
    match file_name {
        "package.json" => npm_deps(txt, &mut out),
        "Cargo.toml" => cargo_deps(txt, &mut out),
        "pyproject.toml" => py_deps(txt, &mut out),
        "go.mod" => go_deps(txt, &mut out),
        _ => {}
    }
    out
}

fn push_dep(out: &mut Vec<ManifestDep>, eco: &'static str, pkg: &str, is_local: bool) {
    out.push(ManifestDep { ecosystem: eco, package: pkg.to_string(), is_local });
}

fn npm_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(txt) else { return };
    let Some(deps) = v.get("dependencies").and_then(|d| d.as_object()) else { return };
    for pkg in deps.keys() {
        push_dep(out, "npm", pkg, false);
    }
}

fn cargo_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let mut in_deps = false;
    for line in txt.lines().map(str::trim) {
        if line.starts_with('[') {
            in_deps = line == "[dependencies]";
            continue;
        }
        let Some((k, val)) = in_deps.then(|| line.split_once('=')).flatten() else {
            continue;
        };
        let pkg = k.trim();
        if !pkg.is_empty() && !pkg.starts_with('#') {
            let local = val.contains("path =") || val.contains("path=");
            push_dep(out, "cargo", pkg, local);
        }
    }
}

fn py_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let (mut in_arr, mut in_poetry) = (false, false);
    for line in txt.lines().map(str::trim) {
        if line.starts_with('[') {
            in_arr = false;
            in_poetry = line == "[tool.poetry.dependencies]";
            continue;
        }
        if in_poetry {
            if let Some((k, _)) = line.split_once('=') {
                push_py(k, out);
            }
            continue;
        }
        if line.starts_with("dependencies") && line.contains('[') {
            let rest = &line[line.find('[').unwrap() + 1..];
            let (items, closed) = rest.find(']').map_or((rest, false), |c| (&rest[..c], true));
            for item in items.split(',') {
                push_py(item, out);
            }
            in_arr = !closed;
        } else if in_arr {
            if line.starts_with(']') {
                in_arr = false;
            } else {
                push_py(line, out);
            }
        }
    }
}

fn push_py(raw: &str, out: &mut Vec<ManifestDep>) {
    let cleaned = raw.trim().trim_matches(|c: char| matches!(c, '"' | '\'' | ','));
    let pkg = cleaned.split(['=', '>', '<', '~', '!', '[', ' ']).next().unwrap_or("");
    if pkg.is_empty() || pkg.starts_with('#') || pkg.eq_ignore_ascii_case("python") {
        return;
    }
    push_dep(out, "python", pkg, false);
}

fn go_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let mut in_req = false;
    for line in txt.lines().map(str::trim) {
        if line.starts_with("require (") {
            in_req = true;
            continue;
        }
        if in_req && line.starts_with(')') {
            in_req = false;
            continue;
        }
        let body = if in_req { Some(line) } else { line.strip_prefix("require ") };
        let Some(b) = body.filter(|s| !s.contains("// indirect") && !s.starts_with("//")) else {
            continue;
        };
        if let Some(m) = b.split_whitespace().next().filter(|s| !s.is_empty()) {
            push_dep(out, "go", m, false);
        }
    }
}
