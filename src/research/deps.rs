//! Lockfile extractors (`toml::from_str` + streaming pnpm/Go), RFC 2789 `index.crates.io`
//! ETag `304` sparse-index client, SemVer breaking-hop planner, `std-replacement-data`
//! (`MSRV <= 1.100`), `RustSec` unmaintained advisories, and counterexample keep-crate guards.

use crate::platform::{ensure_cachedir_tag, now_ms, read_zstd, write_zstd};
use crate::source::parse_deps;
use crate::types::DepAdvisory;
use futures_util::stream::{self, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use semver::Version;
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
const MAX_CRATES_INDEX_CONCURRENCY: usize = 6;
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
    pub ecosystem: Ecosystem,
    pub name: String,
    pub version: String,
    pub direct: bool,
    pub marker: Option<String>,
    pub integrity_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseVerdict {
    ReplaceWithStd { since_minor: u32, replacement: &'static str },
    ReplaceUnmaintained { advisory: &'static str, replacement: &'static str },
    UpgradeWithChangelog,
    KeepUpToDate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseNoteRef {
    pub version: String,
    pub breaking: bool,
    pub security: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HopPlan {
    pub ecosystem: Ecosystem,
    pub name: String,
    pub installed: String,
    pub latest: String,
    pub hops: Vec<String>,
    pub breaking_count: usize,
    pub security_count: usize,
    pub verdict: ReuseVerdict,
    pub keep_guard: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedCrateVersions { etag: Option<String>, fetched_ms: u64, versions: Vec<String> }

#[derive(Debug, Default, Serialize, Deserialize)]
struct CratesIndexCache { entries: HashMap<String, CachedCrateVersions> }

#[derive(Deserialize)]
struct SparseIndexLine { vers: String, #[serde(default)] yanked: bool }

#[derive(Deserialize, Default)]
struct CargoLockDoc { #[serde(default)] package: Vec<CargoLockPkg> }

#[derive(Deserialize)]
struct CargoLockPkg { name: String, version: String, source: Option<String> }

#[derive(Deserialize, Default)]
struct UvLockDoc { #[serde(default)] package: Vec<UvLockPkg> }

#[derive(Deserialize)]
struct UvLockPkg {
    name: String,
    #[serde(default)]
    version: String,
    source: Option<toml::Value>,
    #[serde(default)]
    dependencies: Vec<UvLockDep>,
}

#[derive(Deserialize)]
struct UvLockDep { name: String, marker: Option<String> }

/// Compute RFC 2789 crates.io sparse-index path (`1/a`, `2/ab`, `3/a/abc`, `ab/cd/abcd...`).
pub fn crates_index_rel_path(crate_name: &str) -> String {
    let n = crate_name.trim().to_ascii_lowercase();
    if !n.is_ascii() {
        return String::new();
    }
    match n.len() {
        0 => String::new(),
        1 => format!("1/{n}"),
        2 => format!("2/{n}"),
        3 => format!("3/{}/{n}", &n[..1]),
        _ => format!("{}/{}/{n}", &n[..2], &n[2..4]),
    }
}

/// Parse a flexible ecosystem version string into `semver::Version`.
pub fn parse_semver_lenient(raw: &str) -> Option<Version> {
    let trimmed = raw.trim().trim_start_matches(['v', 'V']);
    if let Ok(v) = Version::parse(trimmed) {
        return Some(v);
    }
    let (core, pre) = trimmed.split_once('-').map_or((trimmed, None), |(c, p)| (c, Some(p)));
    let p: Vec<&str> = core.split('.').collect();
    let norm = match p.len() {
        1 => format!("{}.0.0", p[0]),
        2 => format!("{}.{}.0", p[0], p[1]),
        _ => format!("{}.{}.{}", p[0], p[1], p[2]),
    };
    let cand = pre.filter(|s| !s.is_empty()).map_or(norm.clone(), |s| format!("{norm}-{s}"));
    Version::parse(&cand).ok()
}

/// Official `rust-lang/std-replacement-data` subset gated by project MSRV minor (`<= 1.100`).
pub fn std_replacement(crate_name: &str, project_msrv_minor: u32) -> Option<(u32, &'static str)> {
    let (req_minor, replacement) = match crate_name {
        "lazy_static" => (80, "std::sync::LazyLock"),
        "once_cell" => (70, "std::sync::OnceLock"),
        "fs2" | "fs4" => (89, "std::fs::File::lock"),
        "atty" | "is-terminal" => (70, "std::io::IsTerminal"),
        "scopeguard" => (100, "std::mem::DropGuard"),
        "byteorder" => (32, "primitive from_be_bytes/to_le_bytes"),
        "matches" => (42, "std::matches!"),
        "try_from" => (34, "std::convert::TryFrom"),
        "num_cpus" => (59, "std::thread::available_parallelism (respect cgroup v2 cpu.max)"),
        _ => return None,
    };
    (project_msrv_minor >= req_minor).then_some((req_minor, replacement))
}

/// `RustSec` unmaintained / unsound advisory drop-in recommendations.
pub fn rustsec_unmaintained(crate_name: &str) -> Option<(&'static str, &'static str)> {
    match crate_name {
        "serde_yaml" => Some(("RUSTSEC-2024-0320", "serde_yaml_ng / yaml-rust2")),
        "fxhash" => Some(("RUSTSEC-2025-0001", "rustc-hash")),
        "instant" => Some(("RUSTSEC-2024-0384", "web-time")),
        "dotenv" => Some(("RUSTSEC-2021-0141", "dotenvy")),
        "yaml-rust" => Some(("RUSTSEC-2024-0320", "yaml-rust2")),
        "structopt" => Some(("RUSTSEC-2022-0001", "clap")),
        "ansi_term" => Some(("RUSTSEC-2021-0139", "anstyle")),
        "failure" => Some(("RUSTSEC-2020-0036", "anyhow / thiserror")),
        "mach" => Some(("RUSTSEC-2020-0168", "mach2")),
        "net2" => Some(("RUSTSEC-2020-0016", "socket2")),
        _ => None,
    }
}

/// Counterexample keep-crate guard blocking naive "strip crate" false positives.
pub fn counterexample_keep_guard(crate_name: &str) -> Option<&'static str> {
    match crate_name {
        "mimalloc" => Some("keeps RSS bounded with immediate purge vs glibc ptmalloc arenas"),
        "memchr" => Some("SIMD-vectorized byte search outperforms scalar loops"),
        "similar" => Some("patience/Myers diff outperforms naive line diffing"),
        "petgraph" => Some("battle-tested graph algorithms with low overhead"),
        "blake3" => Some("SIMD cryptographic hashing beats sha2/handwritten digests"),
        "zstd" => Some("bounded windowLog streaming compression beats raw/gzip"),
        "governor" => Some("lock-free GCRA rate pacing avoids burst drift"),
        "parking_lot" => Some("compact non-poisoning locks outperform legacy mutexes"),
        "dunce" => Some("safe Windows verbatim prefix stripping preserving UNC/reserved names"),
        "aho-corasick" => Some("single-pass multi-pattern automaton avoids O(N*M) passes"),
        "tree-sitter" => Some("incremental error-tolerant parsing"),
        _ => None,
    }
}

/// Classify a dependency and compute its SemVer hop plan over intermediate releases.
pub fn plan_dep_hops(
    eco: Ecosystem,
    name: &str,
    installed: &str,
    releases: &[ReleaseNoteRef],
    project_msrv_minor: u32,
) -> HopPlan {
    let installed_ver = parse_semver_lenient(installed);
    let mut newer: Vec<(Version, &ReleaseNoteRef)> = releases
        .iter()
        .filter_map(|rel| {
            let inst = installed_ver.as_ref()?;
            let v = parse_semver_lenient(&rel.version)?;
            (v > *inst).then_some((v, rel))
        })
        .collect();
    newer.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hops = Vec::with_capacity(newer.len());
    let (mut breaking_count, mut security_count) = (0usize, 0usize);
    let mut prev_ver = installed_ver.clone();
    for (v, rel) in &newer {
        hops.push(rel.version.clone());
        let semver_break = prev_ver.as_ref().is_some_and(|p| {
            v.major > p.major || (v.major == 0 && p.major == 0 && v.minor > p.minor)
        });
        if rel.breaking || semver_break {
            breaking_count += 1;
        }
        if rel.security {
            security_count += 1;
        }
        prev_ver = Some(v.clone());
    }

    let latest = hops.last().cloned().unwrap_or_else(|| installed.to_string());
    let keep_guard = (eco == Ecosystem::Cargo).then(|| counterexample_keep_guard(name)).flatten();
    let verdict = if eco == Ecosystem::Cargo {
        if let Some((since_minor, replacement)) = std_replacement(name, project_msrv_minor) {
            ReuseVerdict::ReplaceWithStd { since_minor, replacement }
        } else if let Some((advisory, replacement)) = rustsec_unmaintained(name) {
            ReuseVerdict::ReplaceUnmaintained { advisory, replacement }
        } else if !hops.is_empty() {
            ReuseVerdict::UpgradeWithChangelog
        } else {
            ReuseVerdict::KeepUpToDate
        }
    } else if !hops.is_empty() {
        ReuseVerdict::UpgradeWithChangelog
    } else {
        ReuseVerdict::KeepUpToDate
    };

    HopPlan {
        ecosystem: eco,
        name: name.to_string(),
        installed: installed.to_string(),
        latest,
        hops,
        breaking_count,
        security_count,
        verdict,
        keep_guard,
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

/// Parse `Cargo.lock` v3/v4 via `toml::from_str` bounded by `MAX_LOCKFILE_BYTES`.
pub fn extract_cargo_lock(reader: impl Read, direct_names: &FxHashSet<String>) -> Vec<LockedDep> {
    let Some(txt) = read_bounded_utf8(reader) else { return Vec::new() };
    let Ok(doc) = toml::from_str::<CargoLockDoc>(&txt) else { return Vec::new() };
    doc.package
        .into_iter()
        .filter(|p| p.source.is_some() && !p.name.is_empty() && !p.version.is_empty())
        .map(|p| LockedDep {
            ecosystem: Ecosystem::Cargo,
            direct: direct_names.contains(p.name.as_str()),
            name: p.name,
            version: p.version,
            marker: None,
            integrity_verified: true,
        })
        .collect()
}

/// Parse `uv.lock` via `toml::from_str` bounded by `MAX_LOCKFILE_BYTES` with marker awareness.
pub fn extract_uv_lock(reader: impl Read, direct_names: &FxHashSet<String>) -> Vec<LockedDep> {
    let Some(txt) = read_bounded_utf8(reader) else { return Vec::new() };
    let Ok(doc) = toml::from_str::<UvLockDoc>(&txt) else { return Vec::new() };
    let mut markers: FxHashMap<String, String> = FxHashMap::default();
    let mut pkgs = Vec::new();

    for pkg in doc.package {
        for dep in pkg.dependencies {
            if let Some(m) = dep.marker {
                markers.entry(dep.name).or_insert(m);
            }
        }
        let is_virtual = pkg.source.as_ref().and_then(toml::Value::as_table).is_some_and(|t| {
            t.contains_key("virtual") || t.contains_key("editable")
        });
        if !is_virtual && !pkg.name.is_empty() && !pkg.version.is_empty() {
            pkgs.push((pkg.name, pkg.version));
        }
    }
    pkgs.into_iter()
        .map(|(name, version)| LockedDep {
            ecosystem: Ecosystem::PyPi,
            direct: direct_names.contains(name.as_str()),
            marker: markers.get(&name).cloned(),
            name,
            version,
            integrity_verified: true,
        })
        .collect()
}

/// Stream-parse `pnpm-lock.yaml` v9 (`packages:` only; skips `snapshots:` and peer suffixes).
pub fn extract_pnpm_lock_v9(mut reader: impl BufRead) -> Vec<LockedDep> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sec { Other, Importers, Packages, Snapshots }

    let (mut sec, mut in_deps) = (Sec::Other, false);
    let mut direct: FxHashSet<String> = FxHashSet::default();
    let mut raw_pkgs: Vec<(String, String)> = Vec::new();
    let mut seen: FxHashSet<(String, String)> = FxHashSet::default();
    let mut line = String::new();

    while reader.read_line(&mut line).is_ok_and(|n| n > 0) {
        let raw = line.trim_end_matches(['\r', '\n']);
        let indent = raw.bytes().take_while(|&b| b == b' ').count();
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            if indent == 0 {
                in_deps = false;
                sec = match trimmed {
                    "importers:" => Sec::Importers,
                    "packages:" => Sec::Packages,
                    "snapshots:" => Sec::Snapshots,
                    _ => Sec::Other,
                };
            } else if sec == Sec::Importers && indent == 4 {
                in_deps = trimmed == "dependencies:" || trimmed == "optionalDependencies:";
            } else if sec == Sec::Importers && indent == 6 && in_deps
                && let Some(k) = trimmed.strip_suffix(':')
            {
                let pkg = k.trim().trim_matches(['\'', '"']);
                if !pkg.is_empty() {
                    direct.insert(pkg.to_string());
                }
            } else if sec == Sec::Packages && indent == 2
                && let Some(k) = trimmed.strip_suffix(':')
                && let Some((name, ver)) = split_pnpm_pkg_key(k)
                && seen.insert((name.clone(), ver.clone()))
            {
                raw_pkgs.push((name, ver));
            }
        }
        line.clear();
    }

    raw_pkgs
        .into_iter()
        .map(|(name, version)| LockedDep {
            ecosystem: Ecosystem::Npm,
            direct: direct.contains(&name),
            name,
            version,
            marker: None,
            integrity_verified: true,
        })
        .collect()
}

fn split_pnpm_pkg_key(raw_key: &str) -> Option<(String, String)> {
    let unq = raw_key.trim().trim_matches(['\'', '"']);
    let base = unq.split('(').next().unwrap_or(unq).trim_start_matches('/');
    let at = base.rfind('@').filter(|&p| p > 0)?;
    let (name, ver) = (&base[..at], &base[at + 1..]);
    (!name.is_empty() && !ver.is_empty()).then(|| (name.to_string(), ver.to_string()))
}

/// Stream-parse `go.mod` as primary source and `go.sum` strictly for integrity verification.
pub fn extract_go_mod_and_sum(mut go_mod: impl BufRead, go_sum: Option<impl BufRead>) -> Vec<LockedDep> {
    let mut sum_hashes: FxHashSet<(String, String)> = FxHashSet::default();
    if let Some(mut sum_r) = go_sum {
        let mut line = String::new();
        while sum_r.read_line(&mut line).is_ok_and(|n| n > 0) {
            let mut toks = line.split_whitespace();
            if let (Some(m), Some(v), Some(h)) = (toks.next(), toks.next(), toks.next())
                && h.starts_with("h1:")
            {
                sum_hashes.insert((m.to_string(), v.strip_suffix("/go.mod").unwrap_or(v).to_string()));
            }
            line.clear();
        }
    }

    let (mut out, mut in_req, mut line) = (Vec::new(), false, String::new());
    while go_mod.read_line(&mut line).is_ok_and(|n| n > 0) {
        let t = line.trim();
        if t.starts_with("require (") {
            in_req = true;
        } else if in_req && t.starts_with(')') {
            in_req = false;
        } else if let Some(b) = (if in_req { Some(t) } else { t.strip_prefix("require ") })
            .filter(|s| !s.is_empty() && !s.starts_with("//"))
        {
            let direct = !b.contains("// indirect");
            let mut toks = b.split("//").next().unwrap_or(b).split_whitespace();
            if let (Some(m), Some(v)) = (toks.next(), toks.next()) {
                let verified = sum_hashes.is_empty() || sum_hashes.contains(&(m.to_string(), v.to_string()));
                out.push(LockedDep {
                    ecosystem: Ecosystem::Go,
                    name: m.to_string(),
                    version: v.to_string(),
                    direct,
                    marker: None,
                    integrity_verified: verified,
                });
            }
        }
        line.clear();
    }
    out
}

/// Read project Rust MSRV minor version from `Cargo.toml` (`rust-version = "1.100"` -> `100`).
pub fn project_msrv_minor(repo_root: &Path) -> u32 {
    let Ok(txt) = fs::read_to_string(repo_root.join("Cargo.toml")) else { return 100 };
    toml::from_str::<toml::Value>(&txt)
        .ok()
        .and_then(|v| v.get("package")?.get("rust-version")?.as_str().and_then(parse_semver_lenient))
        .map_or(100, |v| v.minor as u32)
}

/// Extract pinned dependencies across Cargo, pnpm v9, uv, and Go lockfiles in `repo_root`.
pub fn extract_repo_lockfiles(repo_root: &Path) -> Vec<LockedDep> {
    let mut deps = Vec::new();
    if let Ok(f) = File::open(repo_root.join("Cargo.lock")) {
        let direct: FxHashSet<String> = fs::read_to_string(repo_root.join("Cargo.toml"))
            .map(|t| parse_deps("Cargo.toml", &t).into_iter().filter(|d| !d.is_local).map(|d| d.package).collect())
            .unwrap_or_default();
        deps.extend(extract_cargo_lock(f, &direct));
    }
    if let Ok(f) = File::open(repo_root.join("pnpm-lock.yaml")) {
        deps.extend(extract_pnpm_lock_v9(BufReader::with_capacity(LOCKFILE_BUF_BYTES, f)));
    }
    if let Ok(f) = File::open(repo_root.join("uv.lock")) {
        let direct: FxHashSet<String> = fs::read_to_string(repo_root.join("pyproject.toml"))
            .map(|t| parse_deps("pyproject.toml", &t).into_iter().map(|d| d.package).collect())
            .unwrap_or_default();
        deps.extend(extract_uv_lock(f, &direct));
    }
    if let Ok(mod_f) = File::open(repo_root.join("go.mod")) {
        let sum_r = File::open(repo_root.join("go.sum")).ok().map(|f| BufReader::with_capacity(LOCKFILE_BUF_BYTES, f));
        deps.extend(extract_go_mod_and_sum(BufReader::with_capacity(LOCKFILE_BUF_BYTES, mod_f), sum_r));
    }
    deps
}

/// Return up to 12 deduplicated direct external Cargo crate names for sparse-index lookup.
pub fn direct_cargo_crates(repo_root: &Path) -> Vec<String> {
    let mut seen = FxHashSet::default();
    extract_repo_lockfiles(repo_root)
        .into_iter()
        .filter(|d| d.ecosystem == Ecosystem::Cargo && d.direct && seen.insert(d.name.clone()))
        .take(MAX_CRATES_INDEX_CRATES)
        .map(|d| d.name)
        .collect()
}

fn parse_sparse_index_versions(body: &str) -> Vec<String> {
    let mut buckets: BTreeMap<(u64, u64), Version> = BTreeMap::new();
    for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let Ok(row) = serde_json::from_str::<SparseIndexLine>(line) else { continue };
        let Ok(v) = Version::parse(&row.vers) else { continue };
        if row.yanked || !v.pre.is_empty() {
            continue;
        }
        let bucket = (v.major, if v.major == 0 { v.minor } else { 0 });
        buckets
            .entry(bucket)
            .and_modify(|cur| {
                if v > *cur {
                    *cur = v.clone();
                }
            })
            .or_insert(v);
    }
    buckets.into_values().map(|v| v.to_string()).collect()
}

fn versions_to_release_refs(versions: &[String]) -> Vec<ReleaseNoteRef> {
    let mut parsed: Vec<(Version, &str)> =
        versions.iter().filter_map(|s| Version::parse(s).ok().map(|v| (v, s.as_str()))).collect();
    parsed.sort_by(|a, b| a.0.cmp(&b.0));
    parsed.dedup_by(|a, b| a.0 == b.0);
    let mut out = Vec::with_capacity(parsed.len());
    let mut prev: Option<&Version> = None;
    for (v, raw) in &parsed {
        let breaking = prev.is_some_and(|p| v.major > p.major || (v.major == 0 && p.major == 0 && v.minor > p.minor));
        out.push(ReleaseNoteRef { version: (*raw).to_string(), breaking, security: false });
        prev = Some(v);
    }
    out
}

enum IndexFetchOutcome {
    NotModified(Option<String>),
    Fresh { etag: Option<String>, versions: Vec<String> },
    Failed,
}

async fn fetch_single_crate_index(
    client: &reqwest::Client,
    base_url: &str,
    name: &str,
    prev_etag: Option<&str>,
) -> IndexFetchOutcome {
    let rel = crates_index_rel_path(name);
    if rel.is_empty() {
        return IndexFetchOutcome::Failed;
    }
    let mut req = client.get(format!("{}/{rel}", base_url.trim_end_matches('/'))).timeout(Duration::from_secs(5));
    if let Some(tag) = prev_etag {
        req = req.header(reqwest::header::IF_NONE_MATCH, tag);
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(err) => {
            tracing::debug!(crate_name = name, error = %err, "crates.io sparse index fetch failed");
            return IndexFetchOutcome::Failed;
        }
    };
    let etag = resp.headers().get(reqwest::header::ETAG).and_then(|v| v.to_str().ok()).map(str::to_string);
    match resp.status().as_u16() {
        304 => IndexFetchOutcome::NotModified(etag),
        200 => {
            let body = super::read_body_capped(resp, MAX_INDEX_BODY_BYTES).await;
            IndexFetchOutcome::Fresh { etag, versions: parse_sparse_index_versions(&body) }
        }
        code => {
            tracing::debug!(crate_name = name, status = code, "crates.io sparse index non-200");
            IndexFetchOutcome::Failed
        }
    }
}

/// Fetch non-yanked crate versions with persistent ETag `304 Not Modified` cache
/// at `.refit/cache/crates_index.zst` (24h TTL, <= 6 concurrent requests, 5s timeout, top 12 crates).
pub async fn fetch_crates_io_releases(
    client: &reqwest::Client,
    base_url: &str,
    cache_path: &Path,
    crate_names: &[&str],
) -> FxHashMap<String, Vec<ReleaseNoteRef>> {
    let mut cache = read_zstd::<CratesIndexCache>(cache_path).unwrap_or_default();
    let now = now_ms();
    let mut seen = FxHashSet::default();
    let targets: Vec<String> = crate_names
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|k| !k.is_empty() && seen.insert(k.clone()))
        .take(MAX_CRATES_INDEX_CRATES)
        .collect();

    let to_fetch: Vec<(String, Option<String>)> = targets
        .iter()
        .filter(|name| {
            !cache.entries.get(*name).is_some_and(|e| now.saturating_sub(e.fetched_ms) < CRATES_INDEX_TTL_MS)
        })
        .map(|name| (name.clone(), cache.entries.get(name).and_then(|e| e.etag.clone())))
        .collect();

    let mut dirty = false;
    if !to_fetch.is_empty() {
        let fetched: Vec<(String, IndexFetchOutcome)> = stream::iter(to_fetch)
            .map(|(name, etag)| async move {
                let res = fetch_single_crate_index(client, base_url, &name, etag.as_deref()).await;
                (name, res)
            })
            .buffer_unordered(MAX_CRATES_INDEX_CONCURRENCY)
            .collect()
            .await;

        for (name, outcome) in fetched {
            match outcome {
                IndexFetchOutcome::NotModified(new_etag) => {
                    if let Some(entry) = cache.entries.get_mut(&name) {
                        entry.fetched_ms = now;
                        if new_etag.is_some() {
                            entry.etag = new_etag;
                        }
                        dirty = true;
                    }
                }
                IndexFetchOutcome::Fresh { etag, versions } => {
                    cache.entries.insert(name, CachedCrateVersions { etag, fetched_ms: now, versions });
                    dirty = true;
                }
                IndexFetchOutcome::Failed => {}
            }
        }
    }

    if dirty {
        if let Some(parent) = cache_path.parent() {
            ensure_cachedir_tag(parent);
        }
        let _ = write_zstd(cache_path, &cache);
    }

    targets
        .into_iter()
        .filter_map(|name| cache.entries.get(&name).map(|e| (name, versions_to_release_refs(&e.versions))))
        .collect()
}

fn format_changelog_query(dep: &LockedDep, plan: &HopPlan) -> String {
    let marker = dep.marker.as_deref().map(|m| format!(" [{m}]")).unwrap_or_default();
    let sum = if dep.integrity_verified { "" } else { " unverified-sum" };
    let hop = if plan.hops.is_empty() {
        String::new()
    } else {
        format!(" latest={} hops={} brk={} sec={}", plan.latest, plan.hops.len(), plan.breaking_count, plan.security_count)
    };
    format!(
        "{} {} {}{marker}{sum}{hop} changelog breaking changes security advisories 2026",
        plan.ecosystem.as_str(),
        plan.name,
        plan.installed
    )
}

fn plan_manifest_fallback_queries(repo_root: &Path) -> Vec<String> {
    const SPECS: &[(&str, &str)] = &[
        ("package.json", "npm package {p} deprecation security canonical replacement 2026"),
        ("Cargo.toml", "rust crate {p} latest version security advisories 2026"),
        ("pyproject.toml", "python package {p} deprecation security canonical replacement 2026"),
        ("go.mod", "go module {p} deprecation security advisories 2026"),
    ];
    let (mut queries, mut seen) = (Vec::new(), BTreeSet::new());
    for &(file, tpl) in SPECS {
        let Ok(txt) = fs::read_to_string(repo_root.join(file)) else { continue };
        for dep in parse_deps(file, &txt).into_iter().filter(|d| !d.is_local).take(4) {
            let query = tpl.replace("{p}", &dep.package);
            if seen.insert(query.clone()) {
                queries.push(query);
                if queries.len() >= 6 {
                    return queries;
                }
            }
        }
    }
    queries
}

/// Evaluate lockfiles + sparse-index releases into BOTH structured `DepAdvisory` findings
/// and prioritized changelog search queries (up to 8).
pub fn evaluate_repo_deps(
    repo_root: &Path,
    releases_by_crate: &FxHashMap<String, Vec<ReleaseNoteRef>>,
) -> (Vec<DepAdvisory>, Vec<String>) {
    let msrv_minor = project_msrv_minor(repo_root);
    let locked = extract_repo_lockfiles(repo_root);
    let (mut advisories, mut seen_adv) = (Vec::new(), FxHashSet::default());
    let (mut queries, mut seen_q) = (Vec::new(), BTreeSet::new());

    for dep in locked.iter().filter(|d| d.direct || rustsec_unmaintained(&d.name).is_some()) {
        let rels = releases_by_crate.get(&dep.name.to_ascii_lowercase()).map(Vec::as_slice).unwrap_or(&[]);
        let plan = plan_dep_hops(dep.ecosystem, &dep.name, &dep.version, rels, msrv_minor);

        let adv = match plan.verdict {
            ReuseVerdict::ReplaceWithStd { since_minor, replacement } => Some((
                "std_replacement",
                format!("Replace with {replacement} (stabilized in Rust 1.{since_minor})"),
                None,
            )),
            ReuseVerdict::ReplaceUnmaintained { advisory, replacement } => Some((
                "rustsec_unmaintained",
                format!("{advisory}: replace unmaintained crate with {replacement}"),
                Some(format!(
                    "{} {} {} {advisory} unmaintained canonical replacement {replacement} migration 2026",
                    plan.ecosystem.as_str(),
                    plan.name,
                    plan.installed
                )),
            )),
            ReuseVerdict::UpgradeWithChangelog => Some((
                "major_upgrade_hops",
                if plan.breaking_count > 0 {
                    format!("Upgrade across {} breaking hop(s) to {}", plan.breaking_count, plan.latest)
                } else {
                    format!("Upgrade to {}", plan.latest)
                },
                Some(format_changelog_query(dep, &plan)),
            )),
            ReuseVerdict::KeepUpToDate => {
                if queries.len() < 8 {
                    let q = format_changelog_query(dep, &plan);
                    if seen_q.insert(q.clone()) {
                        queries.push(q);
                    }
                }
                None
            }
        };

        if let Some((kind, recommendation, opt_q)) = adv {
            if seen_adv.insert((plan.ecosystem, plan.name.clone(), plan.installed.clone(), kind)) {
                advisories.push(DepAdvisory {
                    ecosystem: plan.ecosystem.as_str().to_string(),
                    name: plan.name.clone(),
                    installed: plan.installed.clone(),
                    latest: (plan.latest != plan.installed).then(|| plan.latest.clone()),
                    kind: kind.to_string(),
                    recommendation,
                    hops: plan.hops.clone(),
                    breaking_hops: plan.breaking_count,
                    keep_guard: plan.keep_guard.map(String::from),
                });
            }
            if let Some(q) = opt_q && queries.len() < 8 && seen_q.insert(q.clone()) {
                queries.push(q);
            }
        }
    }

    if queries.is_empty() && locked.is_empty() {
        queries = plan_manifest_fallback_queries(repo_root);
    }
    (advisories, queries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hygiene::HygieneScorer;
    use crate::testutil::TempRepo;
    use crate::untangle::untangle_repo;
    use axum::Router;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::get;
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    #[test]
    fn lockfile_extractors_and_rfc2789_paths() {
        assert_eq!(
            [crates_index_rel_path("a"), crates_index_rel_path("ab"), crates_index_rel_path("abc"), crates_index_rel_path("Reqwest")],
            ["1/a", "2/ab", "3/a/abc", "re/qw/reqwest"]
        );

        let cargo = "version = 4\n[[package]]\nname = \"app\"\nversion = \"0.1.0\"\n[[package]]\nname = \"serde\"\nversion = \"1.0.229\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n";
        let cdeps = extract_cargo_lock(Cursor::new(cargo), &["serde".into()].into_iter().collect());
        assert!(cdeps.len() == 1 && cdeps[0].direct && cdeps[0].name == "serde");

        let pnpm = "importers:\n  .:\n    dependencies:\n      react:\n        specifier: ^19\npackages:\n  react@19.1.0(peer@1.0.0):\n    resolution: {}\nsnapshots:\n  ghost@0.0.1: {}\n";
        let pdeps = extract_pnpm_lock_v9(Cursor::new(pnpm));
        assert!(pdeps.len() == 1 && pdeps[0].direct && pdeps[0].name == "react" && pdeps[0].version == "19.1.0");

        let uv = "[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\nsource = { virtual = \".\" }\ndependencies = [{ name = \"colorama\", marker = \"sys_platform == 'win32'\" }]\n[[package]]\nname = \"colorama\"\nversion = \"0.4.6\"\n";
        let udeps = extract_uv_lock(Cursor::new(uv), &["colorama".into()].into_iter().collect());
        assert!(udeps.len() == 1 && udeps[0].marker.as_deref() == Some("sys_platform == 'win32'"));

        let go_mod = "module demo\nrequire (\n\tgithub.com/gin-gonic/gin v1.10.0\n\tgolang.org/x/sys v0.25.0 // indirect\n)\n";
        let gdeps = extract_go_mod_and_sum(Cursor::new(go_mod), Some(Cursor::new("github.com/gin-gonic/gin v1.10.0 h1:abc=\n")));
        assert!(gdeps.len() == 2 && gdeps[0].direct && gdeps[0].integrity_verified && !gdeps[1].direct && !gdeps[1].integrity_verified);
    }

    #[tokio::test]
    async fn crates_io_etag_304_and_std_replacement_end_to_end() {
        let (hits, hits_304) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (h1, h304) = (hits.clone(), hits_304.clone());
        let app = Router::new().route(
            "/re/qw/reqwest",
            get(move |headers: HeaderMap| {
                let (h1, h304) = (h1.clone(), h304.clone());
                async move {
                    h1.fetch_add(1, Ordering::Relaxed);
                    if headers.get(axum::http::header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) == Some("\"v1\"") {
                        h304.fetch_add(1, Ordering::Relaxed);
                        return (StatusCode::NOT_MODIFIED, [("etag", "\"v1\"")], "").into_response();
                    }
                    let body = "{\"vers\":\"1.0.0\",\"yanked\":false}\n{\"vers\":\"2.0.0\",\"yanked\":false}\n{\"vers\":\"3.1.0\",\"yanked\":false}\n";
                    (StatusCode::OK, [("etag", "\"v1\"")], body).into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let srv = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let repo = TempRepo::with_files(
            "deps_etag_304",
            &[
                ("Cargo.toml", "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nrust-version = \"1.100\"\n[dependencies]\nlazy_static = \"1.5\"\nserde_yaml = \"0.9\"\nreqwest = \"1.0\"\n"),
                ("Cargo.lock", "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n[[package]]\nname = \"lazy_static\"\nversion = \"1.5.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n[[package]]\nname = \"serde_yaml\"\nversion = \"0.9.34\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n[[package]]\nname = \"reqwest\"\nversion = \"1.0.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n"),
                ("src/lib.rs", "pub fn demo() -> usize { 1 }\n"),
            ],
        );

        let (cache_path, client) = (repo.join(".refit/cache/crates_index.zst"), reqwest::Client::new());
        let rels1 = fetch_crates_io_releases(&client, &base_url, &cache_path, &["reqwest"]).await;
        assert_eq!((hits.load(Ordering::Relaxed), hits_304.load(Ordering::Relaxed)), (1, 0));

        let (advisories, _) = evaluate_repo_deps(repo.path(), &rels1);
        let find = |n: &str| advisories.iter().find(|a| a.name == n).unwrap();
        assert!(find("lazy_static").kind == "std_replacement" && find("lazy_static").recommendation.contains("std::sync::LazyLock"));
        assert!(find("serde_yaml").kind == "rustsec_unmaintained" && find("serde_yaml").recommendation.contains("RUSTSEC-2024-0320"));
        assert_eq!((find("reqwest").kind.as_str(), &find("reqwest").hops, find("reqwest").breaking_hops), ("major_upgrade_hops", &vec!["2.0.0".into(), "3.1.0".into()], 2));

        // Fresh TTL hit (0 HTTP calls), then stale cache -> If-None-Match 304 revalidation.
        let _ = fetch_crates_io_releases(&client, &base_url, &cache_path, &["reqwest"]).await;
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        let mut stale: CratesIndexCache = read_zstd(&cache_path).unwrap();
        stale.entries.get_mut("reqwest").unwrap().fetched_ms = 1;
        write_zstd(&cache_path, &stale).unwrap();
        let rels2 = fetch_crates_io_releases(&client, &base_url, &cache_path, &["reqwest"]).await;
        assert_eq!((hits.load(Ordering::Relaxed), hits_304.load(Ordering::Relaxed), rels2["reqwest"].len()), (2, 1, 3));

        // Offline untangle_repo makes 0 HTTP calls and omits empty dep_advisories from JSON.
        let report = untangle_repo(repo.path(), &HygieneScorer::open_persistent(repo.path())).unwrap();
        assert!(report.dep_advisories.is_empty() && hits.load(Ordering::Relaxed) == 2);
        assert!(!serde_json::to_string(&report).unwrap().contains("dep_advisories"));
        srv.abort();
    }
}
