use crate::platform::{walk_files, walk_tree};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
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
    let nl = memchr::memchr_iter(b'\n', bytes).count();
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

impl RepoSnapshot {
    pub fn capture(repo_root: &Path, include_hidden_configs: bool) -> Self {
        const MIB: usize = 1024 * 1024;
        const MAX_SINGLE_PRELOAD: u64 = 1024 * 1024;
        let b = crate::platform::budget();
        let max_preload = if b.is_constrained() {
            (b.mem_gate_bytes / 4).min(4 * MIB)
        } else {
            32 * MIB
        };

        let mut all_rels = Vec::new();
        let mut source_rels = Vec::new();
        let mut doc_rels = Vec::new();
        let mut seen_docs = BTreeSet::new();
        let mut infra_rels = Vec::new();
        let mut manifest_rels = Vec::new();
        let mut ts_configs = Vec::new();
        let mut preloaded = BTreeMap::new();
        let mut total_preloaded_bytes = 0usize;

        for abs in walk_tree(repo_root, None, include_hidden_configs) {
            let Some(rel) = rel_norm(repo_root, &abs) else { continue };
            let fname = abs.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let ext = ext_str(&abs);
            all_rels.push(rel.clone());

            if matches!(fname, "tsconfig.json" | "jsconfig.json") {
                ts_configs.push(abs.clone());
            }
            if matches!(fname, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
                manifest_rels.push((rel.clone(), fname.to_string(), abs.clone()));
            }

            if supported_ext(ext) {
                if source_rel(&rel) {
                    source_rels.push(rel.clone());
                }
                if total_preloaded_bytes < max_preload {
                    let len = std::fs::metadata(&abs).map(|m| m.len()).unwrap_or(u64::MAX);
                    if len <= MAX_SINGLE_PRELOAD
                        && total_preloaded_bytes.saturating_add(len as usize) <= max_preload
                        && let Ok(bytes) = std::fs::read(&abs)
                    {
                        total_preloaded_bytes += bytes.len();
                        preloaded.insert(rel.clone(), bytes);
                    }
                }
            } else {
                let lower = rel.to_ascii_lowercase();
                let fname_lo = lower.rsplit('/').next().unwrap_or(&lower);
                let is_doc = SNAP_DOC_FILENAMES.contains(&fname_lo)
                    || (lower.starts_with("docs/") && lower.ends_with(".md"));
                if is_doc && seen_docs.insert(rel.clone()) {
                    doc_rels.push((rel.clone(), abs.clone()));
                }
                let ext_lo = ext.to_ascii_lowercase();
                if is_snap_infra_file(&rel, fname, &ext_lo) {
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
        Self {
            all_rels,
            source_rels,
            doc_rels,
            infra_rels,
            manifest_rels,
            ts_configs,
            preloaded,
            total_preloaded_bytes,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CargoDepKind {
    Normal,
    Dev,
    Build,
    Target,
    Workspace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CargoDepDetail {
    pub key: String,
    pub package: String,
    pub kind: CargoDepKind,
    pub is_local: bool,
    pub workspace_inherited: bool,
    pub target_cfg: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct CargoManifest {
    package: Option<toml::Value>,
    #[serde(default)]
    dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "dev-dependencies")]
    dev_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "build-dependencies")]
    build_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default)]
    workspace: Option<CargoWorkspaceSection>,
    #[serde(default)]
    target: BTreeMap<String, CargoTargetSection>,
}

#[derive(serde::Deserialize, Default)]
struct CargoWorkspaceSection {
    #[serde(default)]
    dependencies: BTreeMap<String, CargoDepSpec>,
}

#[derive(serde::Deserialize, Default)]
struct CargoTargetSection {
    #[serde(default)]
    dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "dev-dependencies")]
    dev_dependencies: BTreeMap<String, CargoDepSpec>,
    #[serde(default, rename = "build-dependencies")]
    build_dependencies: BTreeMap<String, CargoDepSpec>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CargoDepSpec {
    Detailed(CargoDepTable),
    Simple(String),
}

#[derive(serde::Deserialize, Default)]
struct CargoDepTable {
    package: Option<String>,
    path: Option<String>,
    #[serde(default)]
    workspace: bool,
}

fn push_cargo_table(
    tbl: &BTreeMap<String, CargoDepSpec>,
    kind: CargoDepKind,
    target_cfg: Option<&str>,
    ws_locals: &std::collections::BTreeSet<String>,
    out: &mut Vec<CargoDepDetail>,
) {
    for (key, spec) in tbl {
        let (package, is_local, workspace_inherited) = match spec {
            CargoDepSpec::Simple(ver) => (key.clone(), ver.trim().is_empty(), false),
            CargoDepSpec::Detailed(d) => {
                let pkg = d.package.clone().unwrap_or_else(|| key.clone());
                let inherited_local = d.workspace && ws_locals.contains(key);
                (pkg, d.path.is_some() || inherited_local, d.workspace)
            }
        };
        out.push(CargoDepDetail {
            key: key.clone(),
            package,
            kind,
            is_local,
            workspace_inherited,
            target_cfg: target_cfg.map(str::to_string),
        });
    }
}

fn parse_cargo_manifest_details(manifest: &CargoManifest) -> Vec<CargoDepDetail> {
    let mut ws_locals = std::collections::BTreeSet::new();
    if let Some(ws) = manifest.workspace.as_ref() {
        for (k, spec) in &ws.dependencies {
            if let CargoDepSpec::Detailed(d) = spec
                && d.path.is_some()
            {
                ws_locals.insert(k.clone());
            }
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

/// Parse all Cargo dependency tables including `[workspace.dependencies]`, `.workspace = true`,
/// renamed `alias = { package = "..." }`, `[dev-dependencies]`, `[build-dependencies]`,
/// and target-gated `[target.'cfg(...)'.dependencies]`.
#[cfg(test)]
pub fn parse_cargo_deps_detailed(txt: &str) -> Vec<CargoDepDetail> {
    let Ok(manifest) = toml::from_str::<CargoManifest>(txt) else {
        return Vec::new();
    };
    parse_cargo_manifest_details(&manifest)
}

fn cargo_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let Ok(manifest) = toml::from_str::<CargoManifest>(txt) else {
        return;
    };
    let is_virtual_workspace = manifest.package.is_none();
    let details = parse_cargo_manifest_details(&manifest);
    let mut seen = std::collections::BTreeSet::new();
    for dep in details {
        let include = match dep.kind {
            CargoDepKind::Normal => true,
            CargoDepKind::Target => dep.target_cfg.as_deref().is_some_and(|c| !c.is_empty()),
            CargoDepKind::Workspace => is_virtual_workspace,
            CargoDepKind::Dev | CargoDepKind::Build => false,
        };
        if !include {
            continue;
        }
        let emit_name = if dep.workspace_inherited && dep.package.is_empty() {
            dep.key.as_str()
        } else {
            dep.package.as_str()
        };
        if !emit_name.is_empty() && seen.insert(emit_name.to_string()) {
            push_dep(out, "cargo", emit_name, dep.is_local);
        }
    }
}

#[derive(serde::Deserialize, Default)]
struct PyProjectManifest {
    project: Option<PyProjectSection>,
    tool: Option<PyToolSection>,
}

#[derive(serde::Deserialize, Default)]
struct PyProjectSection {
    #[serde(default)]
    dependencies: Vec<String>,
}

#[derive(serde::Deserialize, Default)]
struct PyToolSection {
    poetry: Option<PyPoetrySection>,
}

#[derive(serde::Deserialize, Default)]
struct PyPoetrySection {
    #[serde(default)]
    dependencies: BTreeMap<String, toml::Value>,
}

fn py_deps(txt: &str, out: &mut Vec<ManifestDep>) {
    let Ok(doc) = toml::from_str::<PyProjectManifest>(txt) else { return };
    if let Some(proj) = doc.project {
        for item in &proj.dependencies {
            push_py(item, out);
        }
    }
    if let Some(poetry) = doc.tool.and_then(|t| t.poetry) {
        for name in poetry.dependencies.keys() {
            push_py(name, out);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cargo_deps_toml_workspace_aliases_dev_and_target() {
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
