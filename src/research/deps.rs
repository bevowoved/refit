//! Lockfile extractors, crates.io sparse-index client, SemVer hops, and advisories.

use crate::net::{header_str, read_body_capped};
use crate::platform::{ensure_cachedir_tag, now_ms, parse_go_block_lines, read_zstd, write_zstd};
use crate::source::parse_deps;
use crate::types::DepAdvisory;
use futures_util::stream::{self, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use semver::Version;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_CRATES_INDEX_URL: &str = "https://index.crates.io";

const MAX_LOCKFILE_BYTES: u64 = 8 * 1024 * 1024;
const LOCKFILE_BUF_BYTES: usize = 64 * 1024;
const CRATES_INDEX_TTL_MS: u64 = 86_400_000;
const INDEX_CONCURRENCY: usize = 6;
const MAX_CRATES_INDEX_CRATES: usize = 12;
const MAX_INDEX_BODY_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ecosystem { Cargo, Npm, PyPi, Go }

impl Ecosystem {
    pub fn as_str(self) -> &'static str {
        match self { Self::Cargo => "cargo", Self::Npm => "npm", Self::PyPi => "pypi", Self::Go => "go" }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockedDep {
    pub ecosystem: Ecosystem, pub name: String, pub version: String,
    pub direct: bool, pub marker: Option<String>, pub integrity_verified: bool,
}

impl LockedDep {
    fn new(
        ecosystem: Ecosystem, name: String, version: String, direct: bool,
        marker: Option<String>, integrity_verified: bool,
    ) -> Self {
        Self { ecosystem, name, version, direct, marker, integrity_verified }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseVerdict {
    ReplaceWithStd { since_minor: u32, replacement: &'static str },
    ReplaceUnmaintained { advisory: &'static str, replacement: &'static str },
    UpgradeWithChangelog,
    KeepUpToDate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseNoteRef { pub version: String, pub breaking: bool, pub security: bool }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HopPlan {
    pub ecosystem: Ecosystem, pub name: String, pub installed: String, pub latest: String,
    pub hops: Vec<String>, pub breaking_count: usize, pub security_count: usize,
    pub verdict: ReuseVerdict, pub keep_guard: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedCrateVersions { etag: Option<String>, fetched_ms: u64, versions: Vec<String> }
#[derive(Debug, Default, Serialize, Deserialize)]
struct CratesIndexCache { entries: HashMap<String, CachedCrateVersions> }
#[derive(Deserialize)] struct SparseIndexLine { vers: String, #[serde(default)] yanked: bool }
#[derive(Deserialize, Default)] struct CargoLockDoc { #[serde(default)] package: Vec<CargoLockPkg> }
#[derive(Deserialize)] struct CargoLockPkg { name: String, version: String, source: Option<String> }
#[derive(Deserialize, Default)] struct UvLockDoc { #[serde(default)] package: Vec<UvLockPkg> }

#[derive(Deserialize)] #[serde(untagged)]
enum UvSource {
    Table { #[serde(rename = "virtual")] virt: Option<IgnoredAny>, editable: Option<IgnoredAny> },
    Other(IgnoredAny),
}
impl UvSource {
    fn is_virtual(&self) -> bool {
        matches!(self, Self::Table { virt: Some(_), .. } | Self::Table { editable: Some(_), .. })
    }
}

#[derive(Deserialize)]
struct UvLockPkg {
    name: String, #[serde(default)] version: String,
    source: Option<UvSource>, #[serde(default)] dependencies: Vec<UvLockDep>,
}
#[derive(Deserialize)] struct UvLockDep { name: String, marker: Option<String> }

#[derive(Deserialize)] #[serde(untagged)]
enum TomlStrOrOther { Str(String), Other(IgnoredAny) }
impl TomlStrOrOther {
    fn as_str(&self) -> Option<&str> { match self { Self::Str(s) => Some(s), Self::Other(_) => None } }
}
#[derive(Deserialize, Default)]
struct CargoPkgMeta { #[serde(rename = "rust-version")] rust_version: Option<TomlStrOrOther> }
#[derive(Deserialize, Default)] struct CargoWsMeta { package: Option<CargoPkgMeta> }
#[derive(Deserialize, Default)]
struct CargoManifestMsrv { package: Option<CargoPkgMeta>, workspace: Option<CargoWsMeta> }

/// Compute RFC 2789 crates.io sparse-index path (`1/a`, `2/ab`, `3/a/abc`, `ab/cd/abcd...`).
pub fn crates_index_rel_path(crate_name: &str) -> String {
    let norm = crate_name.trim().to_ascii_lowercase();
    if !norm.is_ascii() || norm.is_empty() { return String::new(); }
    match norm.len() {
        1 => format!("1/{norm}"),
        2 => format!("2/{norm}"),
        3 => format!("3/{}/{norm}", &norm[..1]),
        _ => format!("{}/{}/{norm}", &norm[..2], &norm[2..4]),
    }
}

/// Parse a flexible ecosystem version string into `semver::Version`.
pub fn parse_semver_lenient(raw: &str) -> Option<Version> {
    let trimmed = raw.trim().trim_start_matches(['v', 'V']);
    if let Ok(ver) = Version::parse(trimmed) { return Some(ver); }
    let (core, pre) = trimmed.split_once('-').map_or((trimmed, None), |(c, p)| (c, Some(p)));
    let parts: Vec<&str> = core.split('.').collect();
    let norm = match parts.as_slice() {
        [a] => format!("{a}.0.0"),
        [a, b] => format!("{a}.{b}.0"),
        [a, b, c, ..] => format!("{a}.{b}.{c}"),
        [] => return None,
    };
    let cand = pre.filter(|s| !s.is_empty()).map_or(norm.clone(), |s| format!("{norm}-{s}"));
    Version::parse(&cand).ok()
}

const STD_REPLACEMENTS: &[(&str, u32, &str)] = &[
    ("lazy_static", 80, "std::sync::LazyLock"),
    ("once_cell", 70, "std::sync::OnceLock"),
    ("fs2", 89, "std::fs::File::lock"),
    ("fs4", 89, "std::fs::File::lock"),
    ("atty", 70, "std::io::IsTerminal"),
    ("is-terminal", 70, "std::io::IsTerminal"),
    ("scopeguard", 100, "std::mem::DropGuard"),
    ("byteorder", 32, "primitive from_be_bytes/to_le_bytes"),
    ("matches", 42, "std::matches!"),
    ("try_from", 34, "std::convert::TryFrom"),
    ("num_cpus", 59, "std::thread::available_parallelism (respect cgroup v2 cpu.max)"),
];

const RUSTSEC_UNMAINTAINED: &[(&str, &str, &str)] = &[
    ("serde_yaml", "RUSTSEC-2024-0320", "serde_yaml_ng / yaml-rust2"),
    ("fxhash", "RUSTSEC-2025-0001", "rustc-hash"),
    ("instant", "RUSTSEC-2024-0384", "web-time"),
    ("dotenv", "RUSTSEC-2021-0141", "dotenvy"),
    ("yaml-rust", "RUSTSEC-2024-0320", "yaml-rust2"),
    ("structopt", "RUSTSEC-2022-0001", "clap"),
    ("ansi_term", "RUSTSEC-2021-0139", "anstyle"),
    ("failure", "RUSTSEC-2020-0036", "anyhow / thiserror"),
    ("mach", "RUSTSEC-2020-0168", "mach2"),
    ("net2", "RUSTSEC-2020-0016", "socket2"),
];

const KEEP_GUARDS: &[(&str, &str)] = &[
    ("mimalloc", "keeps RSS bounded with immediate purge vs glibc ptmalloc arenas"),
    ("memchr", "SIMD-vectorized byte search outperforms scalar loops"),
    ("similar", "patience/Myers diff outperforms naive line diffing"),
    ("petgraph", "standard graph algorithms with low overhead"),
    ("blake3", "SIMD cryptographic hashing beats sha2/handwritten digests"),
    ("zstd", "bounded windowLog streaming compression beats raw/gzip"),
    ("governor", "lock-free GCRA rate pacing avoids burst drift"),
    ("parking_lot", "compact non-poisoning locks outperform legacy mutexes"),
    ("dunce", "safe Windows verbatim prefix stripping preserving UNC/reserved names"),
    ("aho-corasick", "single-pass multi-pattern automaton avoids O(N*M) passes"),
    ("tree-sitter", "incremental error-tolerant parsing"),
];

#[inline]
fn is_semver_breaking(prev: &Version, next: &Version) -> bool {
    next.major > prev.major || (next.major == 0 && prev.major == 0 && next.minor > prev.minor)
}

/// Official `rust-lang/std-replacement-data` subset gated by project MSRV minor (`<= 1.100`).
pub fn std_replacement(crate_name: &str, project_msrv_minor: u32) -> Option<(u32, &'static str)> {
    let &(_, req_minor, repl) = STD_REPLACEMENTS.iter().find(|&&(k, _, _)| k == crate_name)?;
    (project_msrv_minor >= req_minor).then_some((req_minor, repl))
}

/// `RustSec` unmaintained / unsound advisory drop-in recommendations.
pub fn rustsec_unmaintained(crate_name: &str) -> Option<(&'static str, &'static str)> {
    RUSTSEC_UNMAINTAINED.iter().find(|&&(k, _, _)| k == crate_name).map(|&(_, adv, repl)| (adv, repl))
}

/// Return keep-crate rationale that blocks naive removal advice.
pub fn keep_guard(crate_name: &str) -> Option<&'static str> {
    KEEP_GUARDS.iter().find(|&&(k, _)| k == crate_name).map(|&(_, why)| why)
}

/// Classify a dependency and compute its SemVer hop plan over intermediate releases.
pub fn plan_dep_hops(
    eco: Ecosystem, name: &str, installed: &str, releases: &[ReleaseNoteRef], project_msrv_minor: u32,
) -> HopPlan {
    let installed_ver = parse_semver_lenient(installed);
    let mut newer: Vec<(Version, &ReleaseNoteRef)> = releases.iter().filter_map(|rel| {
        let (inst, ver) = (installed_ver.as_ref()?, parse_semver_lenient(&rel.version)?);
        (ver > *inst).then_some((ver, rel))
    }).collect();
    newer.sort_by(|a, b| a.0.cmp(&b.0));

    let (mut hops, mut breaking_count, mut security_count) = (Vec::with_capacity(newer.len()), 0, 0);
    let mut prev_ver = installed_ver;
    for (ver, rel) in &newer {
        hops.push(rel.version.clone());
        if rel.breaking || prev_ver.as_ref().is_some_and(|p| is_semver_breaking(p, ver)) {
            breaking_count += 1;
        }
        if rel.security { security_count += 1; }
        prev_ver = Some(ver.clone());
    }
    let latest = hops.last().cloned().unwrap_or_else(|| installed.to_string());
    let keep_guard = (eco == Ecosystem::Cargo).then(|| keep_guard(name)).flatten();
    let cargo_verdict = (eco == Ecosystem::Cargo).then(|| {
        std_replacement(name, project_msrv_minor)
            .map(|(since_minor, replacement)| ReuseVerdict::ReplaceWithStd { since_minor, replacement })
            .or_else(|| {
                rustsec_unmaintained(name).map(|(advisory, replacement)| {
                    ReuseVerdict::ReplaceUnmaintained { advisory, replacement }
                })
            })
    }).flatten();
    let fallback = if hops.is_empty() {
        ReuseVerdict::KeepUpToDate
    } else {
        ReuseVerdict::UpgradeWithChangelog
    };
    HopPlan {
        ecosystem: eco, name: name.to_string(), installed: installed.to_string(), latest,
        hops, breaking_count, security_count, verdict: cargo_verdict.unwrap_or(fallback), keep_guard,
    }
}

fn read_bounded_utf8(reader: impl Read) -> Option<String> {
    let mut buf = Vec::new();
    reader.take(MAX_LOCKFILE_BYTES + 1).read_to_end(&mut buf).ok()?;
    if (buf.len() as u64) > MAX_LOCKFILE_BYTES {
        tracing::debug!(bytes = buf.len(), "lockfile exceeds 8 MiB bound; skipping TOML parse");
        return None;
    }
    String::from_utf8(buf).ok()
}

fn parse_toml_lock<T: DeserializeOwned>(reader: impl Read) -> Option<T> {
    toml::from_str(&read_bounded_utf8(reader)?).ok()
}

/// Parse `Cargo.lock` v3/v4 via `toml::from_str` bounded by `MAX_LOCKFILE_BYTES`.
pub fn extract_cargo_lock(reader: impl Read, direct_names: &FxHashSet<String>) -> Vec<LockedDep> {
    let Some(doc) = parse_toml_lock::<CargoLockDoc>(reader) else { return Vec::new() };
    doc.package.into_iter()
        .filter(|p| p.source.is_some() && !p.name.is_empty() && !p.version.is_empty())
        .map(|p| {
            let direct = direct_names.contains(p.name.as_str());
            LockedDep::new(Ecosystem::Cargo, p.name, p.version, direct, None, true)
        })
        .collect()
}

/// Parse `uv.lock` via `toml::from_str` bounded by `MAX_LOCKFILE_BYTES` with marker awareness.
pub fn extract_uv_lock(reader: impl Read, direct_names: &FxHashSet<String>) -> Vec<LockedDep> {
    let Some(doc) = parse_toml_lock::<UvLockDoc>(reader) else { return Vec::new() };
    let (mut markers, mut pkgs) = (FxHashMap::default(), Vec::new());
    for pkg in doc.package {
        for dep in pkg.dependencies {
            if let Some(m) = dep.marker { markers.entry(dep.name).or_insert(m); }
        }
        if !pkg.source.as_ref().is_some_and(UvSource::is_virtual)
            && !pkg.name.is_empty() && !pkg.version.is_empty()
        {
            pkgs.push((pkg.name, pkg.version));
        }
    }
    pkgs.into_iter().map(|(name, ver)| {
        let (direct, marker) = (direct_names.contains(name.as_str()), markers.get(&name).cloned());
        LockedDep::new(Ecosystem::PyPi, name, ver, direct, marker, true)
    }).collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PnpmSec { Other, Importers, Packages, Snapshots }

fn parse_pnpm_section(trimmed: &str) -> PnpmSec {
    match trimmed {
        "importers:" => PnpmSec::Importers,
        "packages:" => PnpmSec::Packages,
        "snapshots:" => PnpmSec::Snapshots,
        _ => PnpmSec::Other,
    }
}

fn record_pnpm_pkg(
    trimmed: &str, seen: &mut FxHashSet<(String, String)>, raw_pkgs: &mut Vec<(String, String)>,
) {
    if let Some((name, ver)) = trimmed.strip_suffix(':').and_then(split_pnpm_pkg_key)
        && seen.insert((name.clone(), ver.clone()))
    {
        raw_pkgs.push((name, ver));
    }
}

/// Stream-parse `pnpm-lock.yaml` v9 (`packages:` only; skips `snapshots:` and peer suffixes).
pub fn extract_pnpm_lock_v9(mut reader: impl BufRead) -> Vec<LockedDep> {
    let (mut sec, mut in_deps, mut line) = (PnpmSec::Other, false, String::new());
    let (mut direct, mut raw_pkgs, mut seen) = (FxHashSet::default(), Vec::new(), FxHashSet::default());
    while reader.read_line(&mut line).is_ok_and(|n| n > 0) {
        let raw = line.trim_end_matches(['\r', '\n']);
        let (indent, trimmed) = (raw.bytes().take_while(|&b| b == b' ').count(), raw.trim());
        if trimmed.is_empty() || trimmed.starts_with('#') {
            line.clear();
            continue;
        }
        match (sec, indent) {
            (_, 0) => {
                in_deps = false;
                sec = parse_pnpm_section(trimmed);
            }
            (PnpmSec::Importers, 4) => {
                in_deps = matches!(trimmed, "dependencies:" | "optionalDependencies:");
            }
            (PnpmSec::Importers, 6) if in_deps => {
                let pkg = trimmed.strip_suffix(':').map_or("", |k| k.trim().trim_matches(['\'', '"']));
                if !pkg.is_empty() { direct.insert(pkg.to_string()); }
            }
            (PnpmSec::Packages, 2) => record_pnpm_pkg(trimmed, &mut seen, &mut raw_pkgs),
            _ => {}
        }
        line.clear();
    }
    raw_pkgs.into_iter().map(|(name, ver)| {
        let is_dir = direct.contains(&name);
        LockedDep::new(Ecosystem::Npm, name, ver, is_dir, None, true)
    }).collect()
}

fn split_pnpm_pkg_key(raw_key: &str) -> Option<(String, String)> {
    let unq = raw_key.trim().trim_matches(['\'', '"']);
    let head = unq.split('(').next().unwrap_or(unq);
    let base = head.trim_start_matches('/');
    let at = base.rfind('@').filter(|&p| p > 0)?;
    let (name, ver) = (&base[..at], &base[at + 1..]);
    (!name.is_empty() && !ver.is_empty()).then(|| (name.to_string(), ver.to_string()))
}

fn collect_go_sum_hashes(mut sum_r: impl BufRead) -> FxHashSet<(String, String)> {
    let (mut hashes, mut line) = (FxHashSet::default(), String::new());
    while sum_r.read_line(&mut line).is_ok_and(|n| n > 0) {
        let mut fields = line.split_whitespace();
        if let (Some(mod_path), Some(raw_ver), Some(hash)) = (fields.next(), fields.next(), fields.next())
            && hash.starts_with("h1:")
        {
            hashes.insert((mod_path.to_string(), raw_ver.trim_suffix("/go.mod").to_string()));
        }
        line.clear();
    }
    hashes
}

/// Stream-parse `go.mod` as primary source and `go.sum` strictly for integrity verification.
pub fn extract_go_mod_and_sum(go_mod: impl BufRead, go_sum: Option<impl BufRead>) -> Vec<LockedDep> {
    let sum_hashes = go_sum.map(collect_go_sum_hashes).unwrap_or_default();
    let Some(mod_txt) = read_bounded_utf8(go_mod) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in parse_go_block_lines(&mod_txt, "require") {
        let direct = !entry.contains("// indirect");
        let code = entry.split_once("//").map_or(entry, |(h, _)| h).trim();
        let mut parts = code.split_whitespace();
        let (Some(module), Some(version)) = (parts.next(), parts.next()) else { continue };
        let pair = (module.to_string(), version.to_string());
        let verified = sum_hashes.is_empty() || sum_hashes.contains(&pair);
        out.push(LockedDep::new(Ecosystem::Go, pair.0, pair.1, direct, None, verified));
    }
    out
}

/// Read Rust MSRV minor version from `[package]` or `[workspace.package]`.
pub fn project_msrv_minor(repo_root: &Path) -> u32 {
    let Ok(txt) = fs::read_to_string(repo_root.join("Cargo.toml")) else { return 100 };
    let Ok(doc) = toml::from_str::<CargoManifestMsrv>(&txt) else { return 100 };
    fn pkg_ver(pkg: Option<&CargoPkgMeta>) -> Option<&str> {
        pkg.and_then(|p| p.rust_version.as_ref()).and_then(TomlStrOrOther::as_str)
    }
    let ver = pkg_ver(doc.package.as_ref())
        .or_else(|| pkg_ver(doc.workspace.as_ref().and_then(|w| w.package.as_ref())));
    ver.and_then(parse_semver_lenient).map_or(100, |s| s.minor as u32)
}

fn direct_manifest_deps(repo_root: &Path, file: &str, skip_local: bool) -> FxHashSet<String> {
    fs::read_to_string(repo_root.join(file)).ok().map(|txt| {
        parse_deps(file, &txt).into_iter()
            .filter(|d| !skip_local || !d.is_local).map(|d| d.package).collect()
    }).unwrap_or_default()
}

fn open_buf(repo_root: &Path, file: &str) -> Option<BufReader<File>> {
    File::open(repo_root.join(file)).ok().map(|f| BufReader::with_capacity(LOCKFILE_BUF_BYTES, f))
}

/// Extract pinned dependencies across Cargo, pnpm v9, uv, and Go lockfiles in `repo_root`.
pub fn extract_repo_lockfiles(repo_root: &Path) -> Vec<LockedDep> {
    let mut deps = Vec::new();
    if let Ok(f) = File::open(repo_root.join("Cargo.lock")) {
        deps.extend(extract_cargo_lock(f, &direct_manifest_deps(repo_root, "Cargo.toml", true)));
    }
    if let Some(r) = open_buf(repo_root, "pnpm-lock.yaml") {
        deps.extend(extract_pnpm_lock_v9(r));
    }
    if let Ok(f) = File::open(repo_root.join("uv.lock")) {
        deps.extend(extract_uv_lock(f, &direct_manifest_deps(repo_root, "pyproject.toml", false)));
    }
    if let Some(mod_r) = open_buf(repo_root, "go.mod") {
        deps.extend(extract_go_mod_and_sum(mod_r, open_buf(repo_root, "go.sum")));
    }
    deps
}

/// Return up to 12 deduplicated direct external Cargo crate names from parsed lockfiles.
pub fn direct_cargo_crates_from(locked: &[LockedDep]) -> Vec<String> {
    let mut seen = FxHashSet::default();
    locked.iter()
        .filter(|d| d.ecosystem == Ecosystem::Cargo && d.direct && seen.insert(d.name.as_str()))
        .take(MAX_CRATES_INDEX_CRATES).map(|d| d.name.clone()).collect()
}

fn parse_sparse_index_versions(body: &str) -> Vec<String> {
    let mut buckets: BTreeMap<(u64, u64), Version> = BTreeMap::new();
    for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let Ok(row) = serde_json::from_str::<SparseIndexLine>(line) else { continue };
        let Ok(ver) = Version::parse(&row.vers) else { continue };
        if row.yanked || !ver.pre.is_empty() { continue; }
        let bucket = (ver.major, if ver.major == 0 { ver.minor } else { 0 });
        buckets.entry(bucket).and_modify(|cur| { if ver > *cur { *cur = ver.clone(); } }).or_insert(ver);
    }
    buckets.into_values().map(|v| v.to_string()).collect()
}

fn versions_to_release_refs(versions: &[String]) -> Vec<ReleaseNoteRef> {
    let mut parsed: Vec<(Version, &str)> = versions
        .iter()
        .filter_map(|raw| {
            let ver = Version::parse(raw).ok()?;
            Some((ver, raw.as_str()))
        })
        .collect();
    parsed.sort_by(|a, b| a.0.cmp(&b.0));
    parsed.dedup_by(|a, b| a.0 == b.0);
    let (mut out, mut prev) = (Vec::with_capacity(parsed.len()), None::<&Version>);
    for (ver, raw) in &parsed {
        let breaking = prev.is_some_and(|p| is_semver_breaking(p, ver));
        out.push(ReleaseNoteRef { version: (*raw).to_string(), breaking, security: false });
        prev = Some(ver);
    }
    out
}

enum IndexFetchOutcome {
    NotModified(Option<String>),
    Fresh { etag: Option<String>, versions: Vec<String> },
    Failed,
}

async fn fetch_crate(
    client: &reqwest::Client, base_url: &str, name: &str, prev_etag: Option<&str>,
) -> IndexFetchOutcome {
    let rel = crates_index_rel_path(name);
    if rel.is_empty() { return IndexFetchOutcome::Failed; }
    let url = format!("{}/{rel}", base_url.trim_end_matches('/'));
    let mut req = client.get(url).timeout(Duration::from_secs(5));
    if let Some(tag) = prev_etag { req = req.header(reqwest::header::IF_NONE_MATCH, tag); }
    let Ok(resp) = req.send().await else { return IndexFetchOutcome::Failed };
    let etag = header_str(resp.headers(), "etag").map(str::to_string);
    match resp.status().as_u16() {
        304 => IndexFetchOutcome::NotModified(etag),
        200 => {
            let body = read_body_capped(resp, MAX_INDEX_BODY_BYTES).await;
            IndexFetchOutcome::Fresh { etag, versions: parse_sparse_index_versions(&body) }
        }
        _ => IndexFetchOutcome::Failed,
    }
}

/// Fetch non-yanked crate versions with persistent ETag `304` caching.
pub async fn fetch_releases(
    client: &reqwest::Client, base_url: &str, cache_path: &Path, crate_names: &[&str],
) -> FxHashMap<String, Vec<ReleaseNoteRef>> {
    let mut cache = read_zstd::<CratesIndexCache>(cache_path).unwrap_or_default();
    let (now, mut seen) = (now_ms(), FxHashSet::default());
    let targets: Vec<String> = crate_names.iter().map(|s| s.trim().to_ascii_lowercase())
        .filter(|k| !k.is_empty() && seen.insert(k.clone())).take(MAX_CRATES_INDEX_CRATES).collect();
    let to_fetch: Vec<(String, Option<String>)> = targets.iter()
        .filter(|n| {
            !cache.entries.get(*n)
                .is_some_and(|e| now.saturating_sub(e.fetched_ms) < CRATES_INDEX_TTL_MS)
        })
        .map(|n| (n.clone(), cache.entries.get(n).and_then(|e| e.etag.clone()))).collect();

    let mut dirty = false;
    if !to_fetch.is_empty() {
        let fetched: Vec<(String, IndexFetchOutcome)> = stream::iter(to_fetch)
            .map(|(n, etag)| async move {
                let res = fetch_crate(client, base_url, &n, etag.as_deref()).await;
                (n, res)
            })
            .buffer_unordered(INDEX_CONCURRENCY).collect().await;
        for (name, outcome) in fetched {
            dirty |= apply_index_outcome(&mut cache, name, outcome, now);
        }
    }
    if dirty {
        if let Some(parent) = cache_path.parent() { ensure_cachedir_tag(parent); }
        let _ = write_zstd(cache_path, &cache);
    }
    targets.into_iter()
        .filter_map(|n| cache.entries.get(&n).map(|e| (n, versions_to_release_refs(&e.versions)))).collect()
}

fn apply_index_outcome(
    cache: &mut CratesIndexCache, name: String, outcome: IndexFetchOutcome, now: u64,
) -> bool {
    match outcome {
        IndexFetchOutcome::NotModified(new_etag) => {
            let Some(entry) = cache.entries.get_mut(&name) else { return false };
            entry.fetched_ms = now;
            if new_etag.is_some() { entry.etag = new_etag; }
            true
        }
        IndexFetchOutcome::Fresh { etag, versions } => {
            cache.entries.insert(name, CachedCrateVersions { etag, fetched_ms: now, versions });
            true
        }
        IndexFetchOutcome::Failed => false,
    }
}

fn format_changelog_query(dep: &LockedDep, plan: &HopPlan) -> String {
    let marker = dep.marker.as_deref().map_or(String::new(), |m| format!(" [{m}]"));
    let sum = if dep.integrity_verified { "" } else { " unverified-sum" };
    let hop = if plan.hops.is_empty() { String::new() } else {
        format!(" latest={} hops={} brk={} sec={}",
            plan.latest, plan.hops.len(), plan.breaking_count, plan.security_count)
    };
    format!("{} {} {}{marker}{sum}{hop} changelog breaking changes security advisories",
        plan.ecosystem.as_str(), plan.name, plan.installed)
}

fn manifest_queries(repo_root: &Path) -> Vec<String> {
    const SPECS: &[(&str, &str)] = &[
        ("package.json", "npm package {p} deprecation security canonical replacement"),
        ("Cargo.toml", "rust crate {p} latest version security advisories"),
        ("pyproject.toml", "python package {p} deprecation security canonical replacement"),
        ("go.mod", "go module {p} deprecation security advisories"),
    ];
    let (mut queries, mut seen) = (Vec::new(), BTreeSet::new());
    for &(file, tpl) in SPECS {
        let Ok(txt) = fs::read_to_string(repo_root.join(file)) else { continue };
        let external = parse_deps(file, &txt).into_iter().filter(|d| !d.is_local);
        for dep in external.take(4) {
            let q = tpl.replace("{p}", &dep.package);
            if seen.insert(q.clone()) {
                queries.push(q);
                if queries.len() >= 6 { return queries; }
            }
        }
    }
    queries
}

/// Evaluate lockfiles and releases into advisories and search queries.
pub fn audit_lockfiles(
    repo_root: &Path, releases_by_crate: &FxHashMap<String, Vec<ReleaseNoteRef>>,
) -> (Vec<DepAdvisory>, Vec<String>) {
    audit_lockfiles_from(repo_root, &extract_repo_lockfiles(repo_root), releases_by_crate)
}

pub fn audit_lockfiles_from(
    repo_root: &Path, locked: &[LockedDep],
    releases_by_crate: &FxHashMap<String, Vec<ReleaseNoteRef>>,
) -> (Vec<DepAdvisory>, Vec<String>) {
    let msrv_minor = project_msrv_minor(repo_root);
    let (mut advisories, mut seen_adv) = (Vec::new(), FxHashSet::default());
    let (mut queries, mut seen_q) = (Vec::new(), BTreeSet::new());
    let mut push_q = |q: String| {
        if queries.len() < 8 && seen_q.insert(q.clone()) { queries.push(q); }
    };

    for dep in locked.iter().filter(|d| d.direct || rustsec_unmaintained(&d.name).is_some()) {
        let rels = releases_by_crate.get(&dep.name.to_ascii_lowercase()).map_or(&[][..], Vec::as_slice);
        let plan = plan_dep_hops(dep.ecosystem, &dep.name, &dep.version, rels, msrv_minor);
        let adv = match plan.verdict {
            ReuseVerdict::ReplaceWithStd { since_minor, replacement } => Some((
                "std_replacement",
                format!("Replace with {replacement} (stabilized in Rust 1.{since_minor})"), None,
            )),
            ReuseVerdict::ReplaceUnmaintained { advisory, replacement } => Some((
                "rustsec_unmaintained",
                format!("{advisory}: replace unmaintained crate with {replacement}"),
                Some(format!("{} {} {} {advisory} unmaintained canonical replacement {replacement} migration",
                    plan.ecosystem.as_str(), plan.name, plan.installed)),
            )),
            ReuseVerdict::UpgradeWithChangelog => {
                let rec = if plan.breaking_count > 0 {
                    format!("Upgrade across {} breaking hop(s) to {}", plan.breaking_count, plan.latest)
                } else { format!("Upgrade to {}", plan.latest) };
                Some(("major_upgrade_hops", rec, Some(format_changelog_query(dep, &plan))))
            }
            ReuseVerdict::KeepUpToDate => { push_q(format_changelog_query(dep, &plan)); None }
        };
        if let Some((kind, recommendation, opt_q)) = adv {
            if seen_adv.insert((plan.ecosystem, plan.name.clone(), plan.installed.clone(), kind)) {
                advisories.push(DepAdvisory {
                    ecosystem: plan.ecosystem.as_str().to_string(), name: plan.name.clone(),
                    installed: plan.installed.clone(),
                    latest: (plan.latest != plan.installed).then(|| plan.latest.clone()),
                    kind: kind.to_string(), recommendation, hops: plan.hops.clone(),
                    breaking_hops: plan.breaking_count, keep_guard: plan.keep_guard.map(String::from),
                });
            }
            if let Some(q) = opt_q { push_q(q); }
        }
    }
    if queries.is_empty() && locked.is_empty() { queries = manifest_queries(repo_root); }
    (advisories, queries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hygiene::HygieneScorer;
    use crate::testutil::{TempRepo, TestAxum, assert_contains_all};
    use crate::untangle::untangle_repo;
    use axum::Router;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::get;
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn direct_set(name: &str) -> FxHashSet<String> { [name.into()].into_iter().collect() }

    #[test]
    fn lockfiles_and_size_limit() {
        let paths = ["a", "ab", "abc", "Reqwest", "", "caf\u{00E9}"].map(crates_index_rel_path);
        assert_eq!(paths, ["1/a", "2/ab", "3/a/abc", "re/qw/reqwest", "", ""]);

        let cargo = concat!(
            "version = 4\n[[package]]\nname = \"app\"\nversion = \"0.1.0\"\n",
            "[[package]]\nname = \"serde\"\nversion = \"1.0.229\"\n",
            "source = \"registry+https://github.com/rust-lang/crates.io-index\"\n",
        );
        let direct = direct_set("serde");
        let cdeps = extract_cargo_lock(Cursor::new(cargo), &direct);
        assert_eq!((cdeps.len(), cdeps[0].direct, cdeps[0].name.as_str()), (1, true, "serde"));

        let mut at_cap = cargo.as_bytes().to_vec();
        at_cap.resize(MAX_LOCKFILE_BYTES as usize, b' ');
        assert_eq!(extract_cargo_lock(Cursor::new(&at_cap), &direct).len(), 1);
        at_cap.push(b' ');
        assert!(extract_cargo_lock(Cursor::new(&at_cap), &direct).is_empty());

        let pnpm = "importers:\n  .:\n    dependencies:\n      react:\n        specifier: ^19\n\
            packages:\n  react@19.1.0(peer@1.0.0):\n    resolution: {}\nsnapshots:\n  g@0.1: {}\n";
        let pdeps = extract_pnpm_lock_v9(Cursor::new(pnpm));
        assert_eq!(pdeps.len(), 1);
        assert!(pdeps[0].direct);
        assert_eq!((pdeps[0].name.as_str(), pdeps[0].version.as_str()), ("react", "19.1.0"));

        let uv = "[[package]]\nname = \"d\"\nversion = \"0.1\"\nsource = { virtual = \".\" }\n\
            dependencies = [{ name = \"colorama\", marker = \"sys_platform == 'win32'\" }]\n\
            [[package]]\nname = \"colorama\"\nversion = \"0.4.6\"\n";
        let udeps = extract_uv_lock(Cursor::new(uv), &direct_set("colorama"));
        assert_eq!((udeps.len(), udeps[0].marker.as_deref()), (1, Some("sys_platform == 'win32'")));

        let gm = "module d\nrequire (\n\tgithub.com/gin-gonic/gin v1.10.0\n\
            \tgolang.org/x/sys v0.25.0 // indirect\n)\n";
        let gdeps = extract_go_mod_and_sum(
            Cursor::new(gm), Some(Cursor::new("github.com/gin-gonic/gin v1.10.0 h1:abc=\n")),
        );
        assert_eq!(gdeps.len(), 2);
        assert!(gdeps[0].direct && gdeps[0].integrity_verified);
        assert!(!gdeps[1].direct && !gdeps[1].integrity_verified);

        let ws = TempRepo::with_files("ws_msrv", &[(
            "Cargo.toml", "[package]\nname = \"a\"\nrust-version.workspace = true\n\
                [workspace.package]\nrust-version = \"1.85\"\n",
        )]);
        assert_eq!(project_msrv_minor(ws.path()), 85);
        assert!(std_replacement("lazy_static", 85).is_some() && std_replacement("fs2", 85).is_none());
    }

    async fn serve_etag(
        hdrs: HeaderMap, h1: Arc<AtomicUsize>, h304: Arc<AtomicUsize>,
    ) -> axum::response::Response {
        h1.fetch_add(1, Ordering::Relaxed);
        if header_str(&hdrs, "if-none-match") == Some("\"v1\"") {
            h304.fetch_add(1, Ordering::Relaxed);
            return (StatusCode::NOT_MODIFIED, [("etag", "\"v1\"")], "").into_response();
        }
        let body = "{\"vers\":\"1.0.0\",\"yanked\":false}\n\
            {\"vers\":\"2.0.0\",\"yanked\":false}\n{\"vers\":\"3.1.0\",\"yanked\":false}\n";
        (StatusCode::OK, [("etag", "\"v1\"")], body).into_response()
    }

    fn build_etag_router(h1: Arc<AtomicUsize>, h304: Arc<AtomicUsize>) -> Router {
        Router::new().route("/re/qw/reqwest", get(move |hdrs: HeaderMap| {
            serve_etag(hdrs, h1.clone(), h304.clone())
        }))
    }

    #[tokio::test]
    async fn crates_io_etag_304_and_hops() {
        let (hits, hits_304) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let app = build_etag_router(hits.clone(), hits_304.clone());
        let srv = TestAxum::spawn(app).await;
        let repo = TempRepo::with_files("deps_etag", &[
            ("Cargo.toml", "[package]\nname = \"d\"\nversion = \"0.1.0\"\nrust-version = \"1.100\"\n\
                [dependencies]\nlazy_static = \"1.5\"\nserde_yaml = \"0.9\"\nreqwest = \"1.0\"\n"),
            ("Cargo.lock", "version = 4\n[[package]]\nname = \"d\"\nversion = \"0.1.0\"\n\
                [[package]]\nname = \"lazy_static\"\nversion = \"1.5.0\"\nsource = \"registry+x\"\n\
                [[package]]\nname = \"serde_yaml\"\nversion = \"0.9.34\"\nsource = \"registry+x\"\n\
                [[package]]\nname = \"reqwest\"\nversion = \"1.0.0\"\nsource = \"registry+x\"\n"),
            ("src/lib.rs", "pub fn demo() -> usize { 1 }\n"),
        ]);

        let rel_0x = [
            ReleaseNoteRef { version: "0.2.0".into(), breaking: false, security: false },
            ReleaseNoteRef { version: "0.2.1".into(), breaking: false, security: false },
        ];
        let plan_0x = plan_dep_hops(Ecosystem::Cargo, "blake3", "0.1.0", &rel_0x, 100);
        assert_eq!((plan_0x.breaking_count, plan_0x.keep_guard.is_some()), (1, true));

        let (cache, client) = (repo.join(".refit/cache/crates_index.zst"), reqwest::Client::new());
        let rels1 = fetch_releases(&client, &srv.base_url, &cache, &["reqwest"]).await;
        assert_eq!((hits.load(Ordering::Relaxed), hits_304.load(Ordering::Relaxed)), (1, 0));

        let (advisories, _) = audit_lockfiles(repo.path(), &rels1);
        let find = |n: &str| advisories.iter().find(|a| a.name == n).unwrap();
        assert_eq!(find("lazy_static").kind, "std_replacement");
        assert_contains_all(&find("lazy_static").recommendation, &["std::sync::LazyLock"]);
        assert_eq!(find("serde_yaml").kind, "rustsec_unmaintained");
        assert_contains_all(&find("serde_yaml").recommendation, &["RUSTSEC-2024-0320"]);
        let rq = find("reqwest");
        assert_eq!((rq.kind.as_str(), &rq.hops, rq.breaking_hops),
            ("major_upgrade_hops", &vec!["2.0.0".into(), "3.1.0".into()], 2));

        let _ = fetch_releases(&client, &srv.base_url, &cache, &["reqwest"]).await;
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        let mut stale: CratesIndexCache = read_zstd(&cache).unwrap();
        stale.entries.get_mut("reqwest").unwrap().fetched_ms = 1;
        write_zstd(&cache, &stale).unwrap();
        let rels2 = fetch_releases(&client, &srv.base_url, &cache, &["reqwest"]).await;
        assert_eq!((hits.load(Ordering::Relaxed), hits_304.load(Ordering::Relaxed), rels2["reqwest"].len()),
            (2, 1, 3));

        let rep = untangle_repo(repo.path(), &HygieneScorer::open_persistent(repo.path())).unwrap();
        assert!(rep.dep_advisories.is_empty() && hits.load(Ordering::Relaxed) == 2);
        assert!(!serde_json::to_string(&rep).unwrap().contains("dep_advisories"));
    }
}
