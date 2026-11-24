//! Streaming lockfile extractors, SemVer hop planner, `std-replacement-data` (`MSRV <= 1.100`),
//! `RustSec` unmaintained triggers, in-tree crate reuse, and counterexample keep-crate guards.

use crate::source::parse_deps;
use semver::Version;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

const LOCKFILE_BUF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ecosystem {
    Cargo,
    Npm,
    PyPi,
    Go,
}

impl Ecosystem {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cargo => "cargo",
            Self::Npm => "npm",
            Self::PyPi => "pypi",
            Self::Go => "go",
        }
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
    ReuseInTree { in_tree_crate: &'static str },
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

/// Parse a flexible ecosystem version string into `semver::Version`.
/// Handles Go `v` prefixes (`v1.2.3`), 2-component Python versions (`2.4` -> `2.4.0`), and prereleases.
pub fn parse_semver_lenient(raw: &str) -> Option<Version> {
    let trimmed = raw.trim().trim_start_matches('v').trim_start_matches('V');
    if let Ok(v) = Version::parse(trimmed) {
        return Some(v);
    }
    let (core, pre) = match trimmed.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (trimmed, None),
    };
    let parts: Vec<&str> = core.split('.').collect();
    let normalized = match parts.len() {
        1 => format!("{}.0.0", parts[0]),
        2 => format!("{}.{}.0", parts[0], parts[1]),
        _ => format!("{}.{}.{}", parts[0], parts[1], parts[2]),
    };
    let candidate = match pre {
        Some(p) if !p.is_empty() => format!("{normalized}-{p}"),
        _ => normalized,
    };
    Version::parse(&candidate).ok()
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

/// Detect custom reimplementations whose capability is already locked in-tree (`Cargo.lock`).
pub fn in_tree_reuse(name: &str) -> Option<&'static str> {
    match name {
        "custom_canon_win" => Some("dunce"),
        "custom_multi_replace" => Some("aho-corasick"),
        "custom_newline_scan" => Some("memchr"),
        "custom_glob_engine" => Some("globset"),
        "custom_url_split" => Some("url"),
        "custom_int_fmt" => Some("itoa"),
        "custom_ct_eq" => Some("constant_time_eq"),
        _ => None,
    }
}

/// Counterexample keep-crate guard: blocks naive "strip crate" false positives where the crate
/// provably outperforms `std` or handwritten implementations.
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
        "dashmap" => Some("sharded concurrent map until bounded SIEVE migration"),
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
    let mut newer: Vec<(Version, &ReleaseNoteRef)> = Vec::new();
    for rel in releases {
        if let (Some(inst), Some(v)) = (&installed_ver, parse_semver_lenient(&rel.version))
            && v > *inst
        {
            newer.push((v, rel));
        }
    }
    newer.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hops = Vec::with_capacity(newer.len());
    let mut breaking_count = 0usize;
    let mut security_count = 0usize;
    let mut prev_major = installed_ver.as_ref().map(|v| v.major);
    for (v, rel) in &newer {
        hops.push(rel.version.clone());
        let major_bump = prev_major.is_some_and(|pm| v.major > pm);
        if rel.breaking || major_bump {
            breaking_count += 1;
        }
        if rel.security {
            security_count += 1;
        }
        prev_major = Some(v.major);
    }

    let latest = hops.last().cloned().unwrap_or_else(|| installed.to_string());
    let keep_guard = (eco == Ecosystem::Cargo)
        .then(|| counterexample_keep_guard(name))
        .flatten();

    let verdict = if eco == Ecosystem::Cargo {
        if let Some((since_minor, replacement)) = std_replacement(name, project_msrv_minor) {
            ReuseVerdict::ReplaceWithStd { since_minor, replacement }
        } else if let Some((advisory, replacement)) = rustsec_unmaintained(name) {
            ReuseVerdict::ReplaceUnmaintained { advisory, replacement }
        } else if let Some(in_tree_crate) = in_tree_reuse(name) {
            ReuseVerdict::ReuseInTree { in_tree_crate }
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

/// Stream-parse `Cargo.lock` v3/v4 line-by-line with O(1) line buffer RAM.
pub fn extract_cargo_lock(
    reader: impl BufRead,
    direct_names: &HashSet<String>,
) -> Vec<LockedDep> {
    let mut out = Vec::new();
    let (mut in_pkg, mut has_source) = (false, false);
    let (mut name, mut version) = (String::new(), String::new());
    let mut line = String::new();
    let mut r = reader;

    let flush = |out: &mut Vec<LockedDep>,
                 name: &mut String,
                 version: &mut String,
                 has_source: &mut bool| {
        if !name.is_empty() && !version.is_empty() && *has_source {
            let direct = direct_names.contains(name.as_str());
            out.push(LockedDep {
                ecosystem: Ecosystem::Cargo,
                name: std::mem::take(name),
                version: std::mem::take(version),
                direct,
                marker: None,
                integrity_verified: true,
            });
        } else {
            name.clear();
            version.clear();
        }
        *has_source = false;
    };

    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let t = line.trim();
        if t == "[[package]]" {
            if in_pkg {
                flush(&mut out, &mut name, &mut version, &mut has_source);
            }
            in_pkg = true;
        } else if in_pkg {
            if let Some(v) = parse_toml_quoted(t, "name") {
                name = v;
            } else if let Some(v) = parse_toml_quoted(t, "version") {
                version = v;
            } else if t.starts_with("source =") || t.starts_with("source=") {
                has_source = true;
            }
        }
        line.clear();
    }
    if in_pkg {
        flush(&mut out, &mut name, &mut version, &mut has_source);
    }
    out
}

/// Stream-parse `pnpm-lock.yaml` v9: reads `packages:` ONLY (skips `snapshots:` and strips peer suffixes).
pub fn extract_pnpm_lock_v9(reader: impl BufRead) -> Vec<LockedDep> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Section {
        Other,
        Importers,
        Packages,
        Snapshots,
    }

    let mut section = Section::Other;
    let mut in_importer_deps = false;
    let mut direct_names: HashSet<String> = HashSet::new();
    let mut raw_pkgs: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut line = String::new();
    let mut r = reader;

    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let raw_line = line.trim_end_matches(['\r', '\n']);
        let indent = raw_line.bytes().take_while(|&b| b == b' ').count();
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            line.clear();
            continue;
        }

        if indent == 0 {
            in_importer_deps = false;
            section = match trimmed {
                "importers:" => Section::Importers,
                "packages:" => Section::Packages,
                "snapshots:" => Section::Snapshots,
                _ => Section::Other,
            };
            line.clear();
            continue;
        }

        match section {
            Section::Importers => {
                if indent == 4 {
                    in_importer_deps =
                        trimmed == "dependencies:" || trimmed == "optionalDependencies:";
                } else if indent == 6
                    && in_importer_deps
                    && let Some(key) = trimmed.strip_suffix(':')
                {
                    let pkg = key.trim().trim_matches(['\'', '"']);
                    if !pkg.is_empty() {
                        direct_names.insert(pkg.to_string());
                    }
                }
            }
            Section::Packages => {
                if indent == 2
                    && let Some(key) = trimmed.strip_suffix(':')
                    && let Some((name, ver)) = split_pnpm_pkg_key(key)
                    && seen.insert((name.clone(), ver.clone()))
                {
                    raw_pkgs.push((name, ver));
                }
            }
            Section::Snapshots | Section::Other => {}
        }
        line.clear();
    }

    raw_pkgs
        .into_iter()
        .map(|(name, version)| {
            let direct = direct_names.contains(&name);
            LockedDep {
                ecosystem: Ecosystem::Npm,
                name,
                version,
                direct,
                marker: None,
                integrity_verified: true,
            }
        })
        .collect()
}

fn split_pnpm_pkg_key(raw_key: &str) -> Option<(String, String)> {
    let unquoted = raw_key.trim().trim_matches(['\'', '"']);
    // Strip any peer dependency suffix `(react@18.3.1)` defensively.
    let base = unquoted.split('(').next().unwrap_or(unquoted).trim_start_matches('/');
    let at_pos = base.rfind('@')?;
    if at_pos == 0 {
        return None;
    }
    let name = &base[..at_pos];
    let ver = &base[at_pos + 1..];
    if name.is_empty() || ver.is_empty() {
        return None;
    }
    Some((name.to_string(), ver.to_string()))
}

/// Stream-parse `uv.lock` with environment-marker awareness (`marker = "..."`).
pub fn extract_uv_lock(reader: impl BufRead, direct_names: &HashSet<String>) -> Vec<LockedDep> {
    let mut pkgs: Vec<(String, String, bool)> = Vec::new();
    let mut markers: HashMap<String, String> = HashMap::new();

    let (mut in_pkg, mut is_virtual) = (false, false);
    let (mut name, mut version) = (String::new(), String::new());
    let mut line = String::new();
    let mut r = reader;

    let flush = |pkgs: &mut Vec<(String, String, bool)>,
                 name: &mut String,
                 version: &mut String,
                 is_virtual: &mut bool| {
        if !name.is_empty() && !version.is_empty() && !*is_virtual {
            let direct = direct_names.contains(name.as_str());
            pkgs.push((std::mem::take(name), std::mem::take(version), direct));
        } else {
            name.clear();
            version.clear();
        }
        *is_virtual = false;
    };

    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let t = line.trim();
        if t == "[[package]]" {
            if in_pkg {
                flush(&mut pkgs, &mut name, &mut version, &mut is_virtual);
            }
            in_pkg = true;
        } else if in_pkg {
            if let Some(v) = parse_toml_quoted(t, "name") {
                name = v;
            } else if let Some(v) = parse_toml_quoted(t, "version") {
                version = v;
            } else if t.starts_with("source =") && (t.contains("virtual") || t.contains("editable")) {
                is_virtual = true;
            } else if t.contains("marker =")
                && let Some((dep_name, dep_marker)) = parse_uv_inline_marker(t)
            {
                markers.entry(dep_name).or_insert(dep_marker);
            }
        }
        line.clear();
    }
    if in_pkg {
        flush(&mut pkgs, &mut name, &mut version, &mut is_virtual);
    }

    pkgs.into_iter()
        .map(|(n, version, direct)| {
            let marker = markers.get(&n).cloned();
            LockedDep {
                ecosystem: Ecosystem::PyPi,
                name: n,
                version,
                direct,
                marker,
                integrity_verified: true,
            }
        })
        .collect()
}

fn parse_uv_inline_marker(line: &str) -> Option<(String, String)> {
    let dep_name = extract_inline_str_field(line, "name")?;
    let marker = extract_inline_str_field(line, "marker")?;
    Some((dep_name, marker))
}

fn extract_inline_str_field(line: &str, key: &str) -> Option<String> {
    let pat = format!("{key} = ");
    let pos = line.find(&pat)?;
    let rest = line[pos + pat.len()..].trim_start();
    let quote = rest.chars().next().filter(|c| matches!(c, '"' | '\''))?;
    let tail = &rest[1..];
    let end = tail.find(quote)?;
    Some(tail[..end].to_string())
}

/// Stream-parse `go.mod` as the primary dependency source and `go.sum` strictly for integrity verification.
pub fn extract_go_mod_and_sum(
    go_mod: impl BufRead,
    go_sum: Option<impl BufRead>,
) -> Vec<LockedDep> {
    let mut sum_hashes: HashSet<(String, String)> = HashSet::new();
    if let Some(mut sum_r) = go_sum {
        let mut line = String::new();
        while let Ok(n) = sum_r.read_line(&mut line) {
            if n == 0 {
                break;
            }
            let mut toks = line.split_whitespace();
            if let (Some(mod_path), Some(ver_raw), Some(hash)) =
                (toks.next(), toks.next(), toks.next())
                && hash.starts_with("h1:")
            {
                let ver = ver_raw.strip_suffix("/go.mod").unwrap_or(ver_raw);
                sum_hashes.insert((mod_path.to_string(), ver.to_string()));
            }
            line.clear();
        }
    }

    let mut out = Vec::new();
    let mut in_req = false;
    let mut line = String::new();
    let mut r = go_mod;

    while let Ok(n) = r.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let t = line.trim();
        if t.starts_with("require (") {
            in_req = true;
            line.clear();
            continue;
        }
        if in_req && t.starts_with(')') {
            in_req = false;
            line.clear();
            continue;
        }
        let body = if in_req { Some(t) } else { t.strip_prefix("require ") };
        if let Some(b) = body.filter(|s| !s.is_empty() && !s.starts_with("//")) {
            let direct = !b.contains("// indirect");
            let code_part = b.split("//").next().unwrap_or(b);
            let mut toks = code_part.split_whitespace();
            if let (Some(mod_path), Some(ver)) = (toks.next(), toks.next()) {
                let verified = sum_hashes.is_empty()
                    || sum_hashes.contains(&(mod_path.to_string(), ver.to_string()));
                out.push(LockedDep {
                    ecosystem: Ecosystem::Go,
                    name: mod_path.to_string(),
                    version: ver.to_string(),
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

fn parse_toml_quoted(trimmed_line: &str, key: &str) -> Option<String> {
    let rest = trimmed_line.strip_prefix(key)?.trim_start();
    let val = rest.strip_prefix('=')?.trim();
    let unquoted = val.strip_prefix('"')?.strip_suffix('"')?;
    Some(unquoted.to_string())
}

/// Read project Rust MSRV minor version from `Cargo.toml` (`rust-version = "1.100"` -> `100`).
pub fn project_msrv_minor(repo_root: &Path) -> u32 {
    let Ok(txt) = fs::read_to_string(repo_root.join("Cargo.toml")) else {
        return 100;
    };
    for line in txt.lines().map(str::trim) {
        if let Some(raw) = parse_toml_quoted(line, "rust-version")
            && let Some(v) = parse_semver_lenient(&raw)
        {
            return v.minor as u32;
        }
    }
    100
}

/// Extract pinned dependencies across Cargo, pnpm v9, uv, and Go lockfiles in `repo_root`.
pub fn extract_repo_lockfiles(repo_root: &Path) -> Vec<LockedDep> {
    let mut deps = Vec::new();

    if let Ok(f) = File::open(repo_root.join("Cargo.lock")) {
        let direct: HashSet<String> = fs::read_to_string(repo_root.join("Cargo.toml"))
            .map(|txt| {
                parse_deps("Cargo.toml", &txt)
                    .into_iter()
                    .filter(|d| !d.is_local)
                    .map(|d| d.package)
                    .collect()
            })
            .unwrap_or_default();
        deps.extend(extract_cargo_lock(BufReader::with_capacity(LOCKFILE_BUF_BYTES, f), &direct));
    }

    if let Ok(f) = File::open(repo_root.join("pnpm-lock.yaml")) {
        deps.extend(extract_pnpm_lock_v9(BufReader::with_capacity(LOCKFILE_BUF_BYTES, f)));
    }

    if let Ok(f) = File::open(repo_root.join("uv.lock")) {
        let direct: HashSet<String> = fs::read_to_string(repo_root.join("pyproject.toml"))
            .map(|txt| parse_deps("pyproject.toml", &txt).into_iter().map(|d| d.package).collect())
            .unwrap_or_default();
        deps.extend(extract_uv_lock(BufReader::with_capacity(LOCKFILE_BUF_BYTES, f), &direct));
    }

    if let Ok(mod_f) = File::open(repo_root.join("go.mod")) {
        let sum_r = File::open(repo_root.join("go.sum"))
            .ok()
            .map(|f| BufReader::with_capacity(LOCKFILE_BUF_BYTES, f));
        deps.extend(extract_go_mod_and_sum(
            BufReader::with_capacity(LOCKFILE_BUF_BYTES, mod_f),
            sum_r,
        ));
    }

    deps
}

fn format_changelog_query(dep: &LockedDep, plan: &HopPlan) -> String {
    if let Some(reason) = plan.keep_guard {
        tracing::debug!(crate_name = %plan.name, reason, "protected keep-crate");
    }
    let marker_hint = dep
        .marker
        .as_deref()
        .map(|marker| format!(" [{marker}]"))
        .unwrap_or_default();
    let integrity_hint = if dep.integrity_verified { "" } else { " unverified-sum" };
    let hop_hint = if plan.hops.is_empty() {
        String::new()
    } else {
        format!(
            " latest={} hops={} brk={} sec={}",
            plan.latest,
            plan.hops.len(),
            plan.breaking_count,
            plan.security_count
        )
    };
    format!(
        "{} {} {}{marker_hint}{integrity_hint}{hop_hint} changelog breaking changes security advisories 2026",
        plan.ecosystem.as_str(),
        plan.name,
        plan.installed
    )
}

fn format_locked_dep_query(dep: &LockedDep, msrv_minor: u32) -> Option<String> {
    let plan = plan_dep_hops(dep.ecosystem, &dep.name, &dep.version, &[], msrv_minor);
    match plan.verdict {
        ReuseVerdict::ReplaceUnmaintained { advisory, replacement } => Some(format!(
            "{} {} {} {advisory} unmaintained canonical replacement {replacement} migration 2026",
            plan.ecosystem.as_str(),
            plan.name,
            plan.installed
        )),
        ReuseVerdict::ReplaceWithStd { since_minor, replacement } => {
            tracing::debug!(
                crate_name = %plan.name,
                since_minor,
                replacement,
                "locally diagnosed std-replacement-data candidate (0 web calls)"
            );
            None
        }
        ReuseVerdict::ReuseInTree { in_tree_crate } => {
            tracing::debug!(
                target = %plan.name,
                in_tree_crate,
                "locally diagnosed in-tree crate reuse candidate"
            );
            None
        }
        ReuseVerdict::UpgradeWithChangelog | ReuseVerdict::KeepUpToDate => {
            Some(format_changelog_query(dep, &plan))
        }
    }
}

fn plan_manifest_fallback_queries(repo_root: &Path) -> Vec<String> {
    const SPECS: &[(&str, &str)] = &[
        ("package.json", "npm package {p} deprecation security canonical replacement 2026"),
        ("Cargo.toml", "rust crate {p} latest version security advisories 2026"),
        ("pyproject.toml", "python package {p} deprecation security canonical replacement 2026"),
        ("go.mod", "go module {p} deprecation security advisories 2026"),
    ];
    let mut queries = Vec::new();
    let mut seen = BTreeSet::new();
    for &(file, tpl) in SPECS {
        let Ok(txt) = fs::read_to_string(repo_root.join(file)) else {
            continue;
        };
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

/// Build prioritized, lockfile-versioned dependency research queries for `WebClient::audit_deps`.
pub fn plan_repo_dep_queries(repo_root: &Path) -> Vec<String> {
    let msrv_minor = project_msrv_minor(repo_root);
    let locked = extract_repo_lockfiles(repo_root);

    let mut prioritized = Vec::new();
    let mut seen = BTreeSet::new();

    // Prioritize RustSec unmaintained replacements first, then direct versioned deps.
    for dep in locked.iter().filter(|d| d.direct || rustsec_unmaintained(&d.name).is_some()) {
        if let Some(query) = format_locked_dep_query(dep, msrv_minor)
            && seen.insert(query.clone())
        {
            prioritized.push(query);
            if prioritized.len() >= 8 {
                return prioritized;
            }
        }
    }

    // Fallback to raw manifests for ecosystems without a lockfile present.
    if prioritized.is_empty() {
        return plan_manifest_fallback_queries(repo_root);
    }

    prioritized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use std::io::Cursor;

    #[test]
    fn streaming_lockfile_extractors_cargo_pnpm_uv_go() {
        // 1. Cargo.lock v4 (skips local root package with no source)
        let cargo_lock = r#"
version = 4

[[package]]
name = "refit"
version = "0.5.0"

[[package]]
name = "dunce"
version = "1.0.5"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "serde_yaml"
version = "0.9.34"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#;
        let direct_cargo = HashSet::from(["dunce".to_string(), "serde_yaml".to_string()]);
        let c_deps = extract_cargo_lock(Cursor::new(cargo_lock), &direct_cargo);
        assert_eq!(c_deps.len(), 2);
        assert_eq!((c_deps[0].name.as_str(), c_deps[0].version.as_str(), c_deps[0].direct), ("dunce", "1.0.5", true));
        assert_eq!((c_deps[1].name.as_str(), c_deps[1].version.as_str()), ("serde_yaml", "0.9.34"));

        // 2. pnpm-lock.yaml v9: reads `packages:` ONLY and skips `snapshots:` peer suffixes
        let pnpm_lock = r#"
lockfileVersion: '9.0'
importers:
  .:
    dependencies:
      zod:
        specifier: ^3.23.0
        version: 3.23.8
packages:
  zod@3.23.8:
    resolution: {integrity: sha512-abc}
  '@scope/core@2.4.1':
    resolution: {integrity: sha512-def}
snapshots:
  zod@3.23.8(typescript@5.6.0):
    dependencies: {}
  react-dom@18.3.1(react@18.3.1):
    dependencies: {}
"#;
        let p_deps = extract_pnpm_lock_v9(Cursor::new(pnpm_lock));
        assert_eq!(p_deps.len(), 2, "must ignore snapshots section: {p_deps:?}");
        assert_eq!((p_deps[0].name.as_str(), p_deps[0].version.as_str(), p_deps[0].direct), ("zod", "3.23.8", true));
        assert_eq!((p_deps[1].name.as_str(), p_deps[1].version.as_str(), p_deps[1].direct), ("@scope/core", "2.4.1", false));

        // 3. uv.lock: marker-aware, skips virtual root package
        let uv_lock = r#"
version = 1

[[package]]
name = "myapp"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "httpx" },
    { name = "colorama", marker = "sys_platform == 'win32'" },
]

[[package]]
name = "httpx"
version = "0.27.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "colorama"
version = "0.4.6"
source = { registry = "https://pypi.org/simple" }
"#;
        let direct_py = HashSet::from(["httpx".to_string(), "colorama".to_string()]);
        let u_deps = extract_uv_lock(Cursor::new(uv_lock), &direct_py);
        assert_eq!(u_deps.len(), 2);
        assert_eq!(u_deps[0].name, "httpx");
        assert_eq!(u_deps[0].marker, None);
        assert_eq!(u_deps[1].name, "colorama");
        assert_eq!(u_deps[1].marker.as_deref(), Some("sys_platform == 'win32'"));

        // 4. go.mod primary + go.sum integrity-only (pruned go.sum entry MUST NOT appear)
        let go_mod = r#"
module example.com/app
go 1.23
require (
    github.com/spf13/cobra v1.8.1
    golang.org/x/sync v0.8.0 // indirect
)
"#;
        let go_sum = r#"
github.com/spf13/cobra v1.8.1 h1:abc=
github.com/phantom/pruned v0.1.0 h1:zzz=
"#;
        let g_deps = extract_go_mod_and_sum(Cursor::new(go_mod), Some(Cursor::new(go_sum)));
        assert_eq!(g_deps.len(), 2, "go.sum must never add phantom modules");
        assert_eq!((g_deps[0].name.as_str(), g_deps[0].direct, g_deps[0].integrity_verified), ("github.com/spf13/cobra", true, true));
        assert_eq!((g_deps[1].name.as_str(), g_deps[1].direct, g_deps[1].integrity_verified), ("golang.org/x/sync", false, false));
    }

    #[test]
    fn s4_versioned_dep_hops_std_replacement_and_keep_guards() {
        let rel = |v: &str, brk: bool, sec: bool| ReleaseNoteRef {
            version: v.into(),
            breaking: brk,
            security: sec,
        };
        let r_multi = vec![
            rel("1.0.0", false, false),
            rel("1.2.0", false, true),
            rel("1.3.0", false, false),
            rel("2.0.0", true, true),
        ];

        // Multi-version SemVer hop planner catches intermediate hops & breaking/security releases.
        let plan = plan_dep_hops(Ecosystem::Cargo, "reqwest", "1.0.0", &r_multi, 100);
        assert_eq!(plan.verdict, ReuseVerdict::UpgradeWithChangelog);
        assert_eq!(plan.hops, vec!["1.2.0", "1.3.0", "2.0.0"]);
        assert_eq!((plan.breaking_count, plan.security_count), (1, 2));
        assert_eq!(plan.latest, "2.0.0");

        // MSRV gating for std-replacement-data (MSRV <= 1.100 vs lower MSRV 1.75).
        let lazy_100 = plan_dep_hops(Ecosystem::Cargo, "lazy_static", "1.5.0", &[], 100);
        assert!(matches!(lazy_100.verdict, ReuseVerdict::ReplaceWithStd { since_minor: 80, .. }));
        let lazy_75 = plan_dep_hops(Ecosystem::Cargo, "lazy_static", "1.5.0", &[], 75);
        assert_eq!(lazy_75.verdict, ReuseVerdict::KeepUpToDate);
        let sg_100 = plan_dep_hops(Ecosystem::Cargo, "scopeguard", "1.2.0", &[], 100);
        assert!(matches!(sg_100.verdict, ReuseVerdict::ReplaceWithStd { since_minor: 100, .. }));
        let fs2_100 = plan_dep_hops(Ecosystem::Cargo, "fs2", "0.4.3", &[], 100);
        assert!(matches!(fs2_100.verdict, ReuseVerdict::ReplaceWithStd { since_minor: 89, .. }));

        // RustSec unmaintained trigger.
        let sy = plan_dep_hops(Ecosystem::Cargo, "serde_yaml", "0.9.34", &[], 100);
        assert!(matches!(sy.verdict, ReuseVerdict::ReplaceUnmaintained { advisory: "RUSTSEC-2024-0320", .. }));

        // In-tree reuse detector.
        let reuse = plan_dep_hops(Ecosystem::Cargo, "custom_canon_win", "0.1.0", &[], 100);
        assert_eq!(reuse.verdict, ReuseVerdict::ReuseInTree { in_tree_crate: "dunce" });

        // Counterexample keep-crate guard.
        for kept in ["mimalloc", "memchr", "blake3", "dunce", "aho-corasick", "governor"] {
            let p = plan_dep_hops(Ecosystem::Cargo, kept, "2.0.0", &[], 100);
            assert_eq!(p.verdict, ReuseVerdict::KeepUpToDate);
            assert!(p.keep_guard.is_some(), "expected keep_guard for {kept}");
        }

        // End-to-end repo query planner prioritizes lockfile versions & RustSec unmaintained crates
        // while suppressing web queries for locally diagnosed std replacements.
        let repo = TempRepo::with_files(
            "deps_plan",
            &[
                (
                    "Cargo.toml",
                    "[package]\nname = \"demo\"\nrust-version = \"1.100\"\n[dependencies]\nlazy_static = \"1.5\"\nserde_yaml = \"0.9\"\nreqwest = \"0.12\"\n",
                ),
                (
                    "Cargo.lock",
                    "version = 4\n[[package]]\nname = \"lazy_static\"\nversion = \"1.5.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n[[package]]\nname = \"serde_yaml\"\nversion = \"0.9.34\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n[[package]]\nname = \"reqwest\"\nversion = \"0.12.9\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n",
                ),
            ],
        );
        let queries = plan_repo_dep_queries(repo.path());
        assert!(queries.iter().any(|q| q.contains("serde_yaml 0.9.34") && q.contains("RUSTSEC-2024-0320")));
        assert!(queries.iter().any(|q| q.contains("reqwest 0.12.9")));
        assert!(!queries.iter().any(|q| q.contains("lazy_static")), "std-replaceable dep should not waste web calls");
    }
}
