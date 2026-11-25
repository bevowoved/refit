use crate::platform::{walk_files, walk_tree};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[inline] pub fn ext_str(p: &Path) -> &str { p.extension().and_then(|s| s.to_str()).unwrap_or("") }
pub fn ext_of(p: impl AsRef<Path>) -> String { ext_str(p.as_ref()).to_ascii_lowercase() }
#[inline] pub fn stem_of(p: &Path) -> &str { p.file_stem().and_then(|s| s.to_str()).unwrap_or("") }

pub fn dir_of(p: impl AsRef<Path>) -> String {
    match p.as_ref().parent() {
        Some(d) => d.to_string_lossy().replace('\\', "/"),
        None => String::new(),
    }
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
    resolved.parent().filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| std::env::temp_dir().join(tmp_name), |p| p.join(dot_name))
}

/// Count logical text lines from raw bytes without UTF-8 decoding.
#[inline]
pub fn count_lines(bytes: &[u8]) -> usize {
    if bytes.is_empty() { return 0; }
    memchr::memchr_iter(b'\n', bytes).count() + usize::from(bytes.last() != Some(&b'\n'))
}

/// Compute quantile `q` in `[0.0, 1.0]` by sorting `vals` in place.
pub fn quantile(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = if (q - 0.5).abs() < 1e-9 { vals.len() / 2 } else { ((vals.len() as f64) * q).floor() as usize };
    vals[idx.min(vals.len() - 1)]
}

#[inline]
pub fn is_js(ext: &str) -> bool {
    ["ts", "tsx", "js", "jsx", "mjs", "cjs"].iter().any(|&e| ext.eq_ignore_ascii_case(e))
}

pub fn ext_lang(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "py" => Some("python"), "rs" => Some("rust"), "go" => Some("go"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
        _ => None,
    }
}

#[inline] pub fn supported_ext(ext: &str) -> bool { ext_lang(ext).is_some() }
#[inline] pub fn same_family(a: &str, b: &str) -> bool { a.eq_ignore_ascii_case(b) || (is_js(a) && is_js(b)) }

pub fn test_path(rel: &str) -> bool {
    rel.starts_with("tests/") || rel.starts_with("test_") || rel.contains("/test_")
        || rel.ends_with("_test.go") || rel.contains(".test.") || rel.contains(".spec.")
}

#[inline] pub fn source_rel(rel: &str) -> bool { !rel.ends_with("__init__.py") && !test_path(rel) }

pub fn entrypoint(rel: &str) -> bool {
    ["main.rs", "lib.rs", "main.py", "main.go", "index.ts", "index.js", "page.tsx", "route.ts"]
        .iter().any(|e| rel.ends_with(e))
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
            if supported_ext(&ext) { push(rel.clone(), ext, None); }
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

const SNAP_DOC_FILENAMES: &[&str] = &[
    "readme.md", "claude.md", "agents.md", ".cursorrules", "contributing.md", "architecture.md",
];

fn is_snap_infra_file(rel: &str, fname: &str, ext_lo: &str) -> bool {
    fname.starts_with("Dockerfile")
        || fname.ends_with(".dockerfile")
        || ext_lo == "tf"
        || ext_lo == "tfvars"
        || (rel.starts_with(".github/workflows/") && matches!(ext_lo, "yml" | "yaml"))
        || ext_lo == "sql"
        || (fname.starts_with("docker-compose") && matches!(ext_lo, "yml" | "yaml"))
}

/// Single-walk byte-budgeted repository snapshot shared across `stem`, `hygiene`,
/// `similarity`, `drift`, and `infra` to eliminate redundant tree walks and disk re-reads.
pub struct RepoSnapshot {
    pub all_rels: Vec<String>,
    pub source_rels: Vec<String>,
    pub doc_rels: Vec<(String, PathBuf)>,
    pub infra_rels: Vec<(String, PathBuf)>,
    pub manifest_rels: Vec<(String, String, PathBuf)>,
    pub ts_configs: Vec<PathBuf>,
    pub preloaded: BTreeMap<String, Vec<u8>>,
    pub total_preloaded_bytes: usize,
}

const MAX_SINGLE_PRELOAD: u64 = 1024 * 1024;

impl RepoSnapshot {
    pub fn capture(repo_root: &Path, include_hidden_configs: bool) -> Self {
        const MIB: usize = 1024 * 1024;
        let b = crate::platform::budget();
        let max_preload = if b.is_constrained() {
            (b.mem_gate_bytes / 4).min(4 * MIB)
        } else {
            32 * MIB
        };

        let (mut all_rels, mut source_rels, mut doc_rels, mut seen_docs) = (Vec::new(), Vec::new(), Vec::new(), BTreeSet::new());
        let (mut infra_rels, mut manifest_rels, mut ts_configs) = (Vec::new(), Vec::new(), Vec::new());
        let (mut preloaded, mut total_preloaded_bytes) = (BTreeMap::new(), 0usize);

        for abs in walk_tree(repo_root, None, include_hidden_configs) {
            let Some(rel) = rel_norm(repo_root, &abs) else { continue };
            let (fname, ext) = (abs.file_name().and_then(|s| s.to_str()).unwrap_or(""), ext_str(&abs));
            all_rels.push(rel.clone());
            if matches!(fname, "tsconfig.json" | "jsconfig.json") { ts_configs.push(abs.clone()); }
            if matches!(fname, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
                manifest_rels.push((rel.clone(), fname.to_string(), abs.clone()));
            }
            if supported_ext(ext) {
                if source_rel(&rel) {
                    source_rels.push(rel.clone());
                }
                try_preload_source(&abs, &rel, max_preload, &mut total_preloaded_bytes, &mut preloaded);
            } else {
                let lower = rel.to_ascii_lowercase();
                let fname_lo = lower.rsplit('/').next().unwrap_or(&lower);
                let is_doc = SNAP_DOC_FILENAMES.contains(&fname_lo)
                    || (lower.starts_with("docs/") && lower.ends_with(".md"));
                if is_doc && seen_docs.insert(rel.clone()) {
                    doc_rels.push((rel.clone(), abs.clone()));
                }
                if is_snap_infra_file(&rel, fname, &ext.to_ascii_lowercase()) {
                    infra_rels.push((rel, abs));
                }
            }
        }
        let cursor_rules = repo_root.join(".cursorrules");
        if cursor_rules.is_file() && seen_docs.insert(".cursorrules".into()) {
            doc_rels.push((".cursorrules".into(), cursor_rules));
        }
        source_rels.sort();
        doc_rels.sort_by(|a, b| a.0.cmp(&b.0));
        Self { all_rels, source_rels, doc_rels, infra_rels, manifest_rels, ts_configs, preloaded, total_preloaded_bytes }
    }

    pub fn to_repo_scan(&self) -> RepoScan {
        let mut out = RepoScan { sources: Vec::new(), tests: Vec::new(), all: Vec::new() };
        for rel in &self.all_rels {
            let ext = ext_of(rel);
            if !supported_ext(&ext) {
                continue;
            }
            out.all.push((rel.clone(), ext.clone(), None));
            if rel.ends_with("__init__.py") {
                continue;
            }
            if test_path(rel) {
                out.tests.push((rel.clone(), ext, None));
            } else {
                out.sources.push((rel.clone(), ext, None));
            }
        }
        out.sources.sort_by(|a, b| a.0.cmp(&b.0));
        out.tests.sort_by(|a, b| a.0.cmp(&b.0));
        out.all.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

fn try_preload_source(
    abs: &Path,
    rel: &str,
    max_preload: usize,
    total_bytes: &mut usize,
    preloaded: &mut BTreeMap<String, Vec<u8>>,
) {
    if *total_bytes >= max_preload {
        return;
    }
    let len = std::fs::metadata(abs).map(|m| m.len()).unwrap_or(u64::MAX);
    if len > MAX_SINGLE_PRELOAD || total_bytes.saturating_add(len as usize) > max_preload {
        return;
    }
    if let Ok(bytes) = std::fs::read(abs) {
        *total_bytes += bytes.len();
        preloaded.insert(rel.to_string(), bytes);
    }
}

/// Borrow bytes from `preloaded` without cloning, or read from disk once.
pub fn read_bytes<'a>(root: &Path, rel: &str, preloaded: Option<&'a BTreeMap<String, Vec<u8>>>) -> Option<Cow<'a, [u8]>> {
    preloaded.and_then(|m| m.get(rel)).map(|b| Cow::Borrowed(b.as_slice()))
        .or_else(|| std::fs::read(root.join(rel)).ok().map(Cow::Owned))
}

/// Borrow UTF-8 text from `preloaded` with zero copies when valid UTF-8.
pub fn read_text<'a>(root: &Path, rel: &str, preloaded: Option<&'a BTreeMap<String, Vec<u8>>>) -> Option<Cow<'a, str>> {
    preloaded.and_then(|m| m.get(rel)).map(|b| String::from_utf8_lossy(b))
        .or_else(|| std::fs::read_to_string(root.join(rel)).ok().map(Cow::Owned))
}

#[derive(Debug, Clone)]
pub struct ManifestDep { pub ecosystem: &'static str, pub package: String, pub is_local: bool }

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
    if let Some(deps) = v.get("dependencies").and_then(|d| d.as_object()) {
        for pkg in deps.keys() { push_dep(out, "npm", pkg, false); }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CargoDepKind { Normal, Dev, Build, Target, Workspace }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CargoDepDetail {
    pub key: String, pub package: String, pub kind: CargoDepKind,
    pub is_local: bool, pub workspace_inherited: bool, pub target_cfg: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct CargoManifest {
    package: Option<toml::Value>,
    #[serde(default)] dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "dev-dependencies")] dev_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "build-dependencies")] build_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default)] workspace: Option<CargoWorkspaceSection>,
    #[serde(default)] target: BTreeMap<String, CargoTargetSection>,
}

#[derive(serde::Deserialize, Default)]
struct CargoWorkspaceSection { #[serde(default)] dependencies: BTreeMap<String, CargoDepSpec> }

#[derive(serde::Deserialize, Default)]
struct CargoTargetSection {
    #[serde(default)] dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "dev-dependencies")] dev_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "build-dependencies")] build_dependencies: BTreeMap<String, CargoDepSpec>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CargoDepSpec { Detailed(CargoDepTable), Simple(String) }

#[derive(serde::Deserialize, Default)]
struct CargoDepTable { package: Option<String>, path: Option<String>, #[serde(default)] workspace: bool }

fn push_cargo_table(
    tbl: &BTreeMap<String, CargoDepSpec>, kind: CargoDepKind, target_cfg: Option<&str>,
    ws_locals: &BTreeSet<String>, out: &mut Vec<CargoDepDetail>,
) {
    for (key, spec) in tbl {
        let (package, is_local, workspace_inherited) = match spec {
            CargoDepSpec::Simple(ver) => (key.clone(), ver.trim().is_empty(), false),
            CargoDepSpec::Detailed(d) => (
                d.package.clone().unwrap_or_else(|| key.clone()),
                d.path.is_some() || (d.workspace && ws_locals.contains(key)),
                d.workspace,
            ),
        };
        out.push(CargoDepDetail {
            key: key.clone(), package, kind, is_local, workspace_inherited,
            target_cfg: target_cfg.map(str::to_string),
        });
    }
}

fn parse_cargo_manifest_details(manifest: &CargoManifest) -> Vec<CargoDepDetail> {
    let mut ws_locals = BTreeSet::new();
    if let Some(ws) = manifest.workspace.as_ref() {
        for (k, spec) in &ws.dependencies {
            if let CargoDepSpec::Detailed(d) = spec && d.path.is_some() { ws_locals.insert(k.clone()); }
        }
    }
    let mut details = Vec::new();
    push_cargo_table(&manifest.dependencies, CargoDepKind::Normal, None, &ws_locals, &mut details);
    push_cargo_table(&manifest.dev_dependencies, CargoDepKind::Dev, None, &ws_locals, &mut details);
    push_cargo_table(&manifest.build_dependencies, CargoDepKind::Build, None, &ws_locals, &mut details);
    if let Some(ws) = manifest.workspace.as_ref() {
        push_cargo_table(&ws.dependencies, CargoDepKind::Workspace, None, &ws_locals, &mut details);
    }
    for (cfg, sec) in &manifest.target {
        push_cargo_table(&sec.dependencies, CargoDepKind::Target, Some(cfg), &ws_locals, &mut details);
        push_cargo_table(&sec.dev_dependencies, CargoDepKind::Dev, Some(cfg), &ws_locals, &mut details);
        push_cargo_table(&sec.build_dependencies, CargoDepKind::Build, Some(cfg), &ws_locals, &mut details);
    }
    details
}

/// Parse all Cargo dependency tables including workspace, renamed aliases, dev/build, and target-gated.
#[cfg(test)]
pub fn parse_cargo_deps_detailed(txt: &str) -> Vec<CargoDepDetail> {
    toml::from_str::<CargoManifest>(txt).map_or_else(|_| Vec::new(), |m| parse_cargo_manifest_details(&m))
}

fn cargo_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let Ok(manifest) = toml::from_str::<CargoManifest>(txt) else { return };
    let (is_virt, mut seen) = (manifest.package.is_none(), BTreeSet::new());
    for dep in parse_cargo_manifest_details(&manifest) {
        let include = match dep.kind {
            CargoDepKind::Normal => true,
            CargoDepKind::Target => dep.target_cfg.as_deref().is_some_and(|c| !c.is_empty()),
            CargoDepKind::Workspace => is_virt,
            CargoDepKind::Dev | CargoDepKind::Build => false,
        };
        if !include { continue; }
        let name = if dep.workspace_inherited && dep.package.is_empty() { dep.key.as_str() } else { dep.package.as_str() };
        if !name.is_empty() && seen.insert(name.to_string()) { push_dep(out, "cargo", name, dep.is_local); }
    }
}

fn py_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let Ok(doc) = toml::from_str::<toml::Value>(txt) else { return };
    if let Some(arr) = doc.get("project").and_then(|p| p.get("dependencies")).and_then(toml::Value::as_array) {
        for item in arr.iter().filter_map(toml::Value::as_str) {
            push_py(item, out);
        }
    }
    let poetry = doc.get("tool").and_then(|t| t.get("poetry")).and_then(|p| p.get("dependencies"));
    if let Some(tbl) = poetry.and_then(toml::Value::as_table) {
        for name in tbl.keys() {
            push_py(name, out);
        }
    }
}

fn push_py(raw: &str, out: &mut Vec<ManifestDep>) {
    let cleaned = raw.trim().trim_matches(|c: char| matches!(c, '"' | '\'' | ','));
    let pkg = cleaned.split(['=', '>', '<', '~', '!', '[', ' ']).next().unwrap_or("");
    if !pkg.is_empty() && !pkg.starts_with('#') && !pkg.eq_ignore_ascii_case("python") {
        push_dep(out, "python", pkg, false);
    }
}

fn go_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    for line in crate::platform::parse_go_block_lines(txt, "require") {
        if line.contains("// indirect") {
            continue;
        }
        if let Some(m) = line.split_whitespace().next().filter(|s| !s.is_empty()) {
            push_dep(out, "go", m, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_deps_workspace_aliases_and_dev() {
        let member_toml = r#"
[package]
name = "my-crate"
version = "0.1.0"
edition = "2024"

[dependencies]
serde = "1.0"
tokio.workspace = true
blake3 = { workspace = true }
local_core = { path = "../core" }
http_old = { package = "http", version = "0.2" }

[dev-dependencies]
criterion = "0.5"

[build-dependencies]
cc = "1.2"

[target.'cfg(unix)'.dependencies]
nix = { version = "0.31", features = ["process"] }
"#;

        let deps = parse_deps("Cargo.toml", member_toml);
        let pkgs: Vec<(&str, bool)> = deps.iter().map(|d| (d.package.as_str(), d.is_local)).collect();
        assert!(pkgs.contains(&("serde", false)));
        assert!(pkgs.contains(&("tokio", false)));
        assert!(pkgs.contains(&("blake3", false)));
        assert!(pkgs.contains(&("local_core", true)));
        assert!(pkgs.contains(&("http", false)));
        assert!(pkgs.contains(&("nix", false)));
        assert!(!pkgs.iter().any(|(p, _)| p.contains(".workspace")));
        assert!(!pkgs.iter().any(|(p, _)| *p == "criterion" || *p == "cc"));

        let detailed = parse_cargo_deps_detailed(member_toml);
        let tokio_dep = detailed.iter().find(|d| d.key == "tokio").unwrap();
        assert_eq!(tokio_dep.kind, CargoDepKind::Normal);
        assert!(tokio_dep.workspace_inherited);
        let alias_dep = detailed.iter().find(|d| d.key == "http_old").unwrap();
        assert_eq!(alias_dep.package, "http");
        let dev_dep = detailed.iter().find(|d| d.key == "criterion").unwrap();
        assert_eq!(dev_dep.kind, CargoDepKind::Dev);
        let target_dep = detailed.iter().find(|d| d.key == "nix").unwrap();
        assert_eq!(target_dep.kind, CargoDepKind::Target);
        assert_eq!(target_dep.target_cfg.as_deref(), Some("cfg(unix)"));

        let virtual_ws = r#"
[workspace]
members = ["crates/*"]

[workspace.dependencies]
anyhow = "1.0"
internal_util = { path = "crates/util" }
"#;
        let ws_deps = parse_deps("Cargo.toml", virtual_ws);
        assert!(ws_deps.iter().any(|d| d.package == "anyhow" && !d.is_local));
        assert!(ws_deps.iter().any(|d| d.package == "internal_util" && d.is_local));
    }
}
