use crate::lex::{ident_byte, left_bound};
use crate::platform::{find_secret, walk_tree};
use crate::source::{ext_of, ext_str, parse_deps, rel_norm, supported_ext};
use crate::types::{InfraFinding, UnusedDep};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use fixedbitset::FixedBitSet;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Eco {
    Npm = 0,
    Cargo = 1,
    Python = 2,
    Go = 3,
}

impl Eco {
    const ALL: [Self; 4] = [Self::Npm, Self::Cargo, Self::Python, Self::Go];

    fn from_ext(ext: &str) -> Self {
        if ext.eq_ignore_ascii_case("rs") {
            Self::Cargo
        } else if ext.eq_ignore_ascii_case("py") {
            Self::Python
        } else if ext.eq_ignore_ascii_case("go") {
            Self::Go
        } else {
            Self::Npm
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "npm" => Self::Npm,
            "cargo" => Self::Cargo,
            "python" => Self::Python,
            _ => Self::Go,
        }
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    fn word_bound(self) -> bool {
        self != Self::Go
    }

    #[inline]
    fn case_insensitive(self) -> bool {
        self == Self::Python
    }
}

struct ParsedDep {
    manifest: String,
    ecosystem: &'static str,
    package: String,
    eco: Eco,
    slot: usize,
}

#[derive(Default)]
struct EcoBuilder {
    pkg_slots: HashMap<String, usize>,
    pat_ids: HashMap<String, usize>,
    patterns: Vec<String>,
    pat_slots: Vec<Vec<usize>>,
}

impl EcoBuilder {
    fn add_dep(&mut self, eco: Eco, package: &str) -> usize {
        let key = if eco.case_insensitive() {
            package.to_ascii_lowercase()
        } else {
            package.to_string()
        };
        let next_slot = self.pkg_slots.len();
        let slot = *self.pkg_slots.entry(key.clone()).or_insert(next_slot);
        if key.is_empty() {
            return slot;
        }
        self.add_pattern(&key, slot);
        if matches!(eco, Eco::Cargo | Eco::Python) && key.contains('-') {
            self.add_pattern(&key.replace('-', "_"), slot);
        }
        slot
    }

    fn add_pattern(&mut self, pat: &str, slot: usize) {
        let pat_id = if let Some(&existing) = self.pat_ids.get(pat) {
            existing
        } else {
            let new_id = self.patterns.len();
            self.patterns.push(pat.to_string());
            self.pat_ids.insert(pat.to_string(), new_id);
            self.pat_slots.push(Vec::new());
            new_id
        };
        let slots = &mut self.pat_slots[pat_id];
        if !slots.contains(&slot) {
            slots.push(slot);
        }
    }

    fn build(self, eco: Eco) -> Option<EcoMatcher> {
        let dep_count = self.pkg_slots.len();
        if dep_count == 0 || self.patterns.is_empty() {
            return None;
        }
        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(eco.case_insensitive())
            .build(&self.patterns)
            .ok()?;
        Some(EcoMatcher {
            ac,
            pat_slots: self.pat_slots,
            dep_count,
            word_bound: eco.word_bound(),
            hits: FixedBitSet::with_capacity(dep_count),
        })
    }
}

struct EcoMatcher {
    ac: AhoCorasick,
    pat_slots: Vec<Vec<usize>>,
    dep_count: usize,
    word_bound: bool,
    hits: FixedBitSet,
}

impl EcoMatcher {
    #[inline]
    fn is_complete(&self) -> bool {
        self.hits.count_ones(..) == self.dep_count
    }

    fn scan_bytes(&mut self, bytes: &[u8]) {
        if self.is_complete() || bytes.is_empty() {
            return;
        }
        for mat in self.ac.find_overlapping_iter(bytes) {
            if self.word_bound && !is_word_span(bytes, mat.start(), mat.end()) {
                continue;
            }
            let mut added = false;
            for &slot in &self.pat_slots[mat.pattern().as_usize()] {
                added |= !self.hits.put(slot);
            }
            if added && self.is_complete() {
                return;
            }
        }
    }
}

#[inline]
fn is_word_span(bytes: &[u8], start: usize, end: usize) -> bool {
    left_bound(bytes, start) && (end == bytes.len() || !ident_byte(bytes[end]))
}

#[derive(Clone, Copy)]
enum Kind {
    Tf,
    Docker,
    Gha,
    Sql,
}

struct Rule {
    kind: Kind,
    severity: &'static str,
    id: &'static str,
    msg: &'static str,
    check: fn(&str, &str, &str) -> bool,
}

const RULES: &[Rule] = &[
    Rule {
        kind: Kind::Tf,
        severity: "critical",
        id: "tf_wildcard_iam",
        msg: "Overly permissive wildcard IAM policy (`*` / `*:*`)",
        check: |t, tl, _| {
            t.contains("*:*")
                || ((tl.contains("action") || tl.contains("resources"))
                    && (t.contains("\"*\"") || t.contains("\\\"*\\\"")))
        },
    },
    Rule {
        kind: Kind::Tf,
        severity: "high",
        id: "tf_open_cidr",
        msg: "Security group/route open to 0.0.0.0/0 or ::/0",
        check: |t, _, _| t.contains("0.0.0.0/0") || t.contains("::/0"),
    },
    Rule {
        kind: Kind::Docker,
        severity: "medium",
        id: "docker_latest_tag",
        msg: "Unpinned `:latest` base image causes non-reproducible builds",
        check: |t, _, up| up.starts_with("FROM ") && t.contains(":latest"),
    },
    Rule {
        kind: Kind::Docker,
        severity: "high",
        id: "docker_add_remote_url",
        msg: "Dockerfile `ADD` fetches remote URL without integrity checksum",
        check: |t, _, up| {
            up.starts_with("ADD ") && (t.contains("http://") || t.contains("https://"))
        },
    },
    Rule {
        kind: Kind::Gha,
        severity: "high",
        id: "gha_write_all_perms",
        msg: "Workflow grants overly broad `write-all` permissions",
        check: |t, _, _| t.contains("write-all"),
    },
    Rule {
        kind: Kind::Gha,
        severity: "medium",
        id: "gha_unpinned_action",
        msg: "Action pinned to mutable branch (`@main`/`@master`) instead of tag/SHA",
        check: |t, _, _| {
            (t.starts_with("uses:") || t.starts_with("- uses:"))
                && (t.ends_with("@main") || t.ends_with("@master"))
        },
    },
    Rule {
        kind: Kind::Gha,
        severity: "high",
        id: "gha_script_injection",
        msg: "Untrusted `${{ github.event.* }}` interpolation risks script injection",
        check: |t, tl, _| t.contains("${{") && tl.contains("github.event."),
    },
    Rule {
        kind: Kind::Sql,
        severity: "high",
        id: "sql_destructive_migration",
        msg: "Unguarded destructive DROP/TRUNCATE statement in SQL migration",
        check: |_, _, up| {
            (up.starts_with("DROP TABLE ") && !up.contains("IF EXISTS"))
                || up.starts_with("TRUNCATE TABLE ")
                || up.starts_with("TRUNCATE ")
        },
    },
    Rule {
        kind: Kind::Sql,
        severity: "critical",
        id: "sql_hardcoded_password",
        msg: "Hardcoded database password literal in SQL script",
        check: |_, _, up| {
            up.contains("IDENTIFIED BY '") || up.contains("WITH PASSWORD '")
        },
    },
];

fn push_finding(
    out: &mut Vec<InfraFinding>,
    file: &str,
    line: usize,
    severity: &str,
    rule_id: &str,
    message: &str,
) {
    out.push(InfraFinding {
        file: file.to_string(),
        line,
        severity: severity.into(),
        rule_id: rule_id.into(),
        message: message.into(),
    });
}

/// Scan Terraform, Dockerfiles, GitHub Actions, SQL, and unused manifest dependencies.
pub fn scan_parts(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    let mut infra_count = 0usize;
    let mut findings = Vec::new();
    let mut manifests = Vec::new();
    let mut source_paths: Vec<(Eco, PathBuf)> = Vec::new();

    for abs in walk_tree(repo_root, None, true) {
        let Some(rel) = rel_norm(repo_root, &abs) else { continue };
        let fname = abs.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ext = ext_str(&abs);

        if supported_ext(ext) {
            if preloaded.is_none() {
                source_paths.push((Eco::from_ext(ext), abs));
            }
            continue;
        }

        if matches!(fname, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
            if let Ok(txt) = fs::read_to_string(&abs) {
                manifests.push((rel, fname.to_string(), txt));
            }
            continue;
        }

        let ext_lo = ext_of(&abs);
        let Some(kind) = infra_kind(&rel, fname, &ext_lo) else { continue };
        infra_count += 1;
        let Ok(txt) = fs::read_to_string(&abs) else { continue };
        scan_file(&rel, kind, &txt, &mut findings);
    }

    let unused = find_unused_deps(&manifests, preloaded, &source_paths);
    (infra_count, unused, findings)
}

fn infra_kind(rel: &str, fname: &str, ext: &str) -> Option<Option<Kind>> {
    if fname.starts_with("Dockerfile") || fname.ends_with(".dockerfile") {
        return Some(Some(Kind::Docker));
    }
    if ext == "tf" || ext == "tfvars" {
        return Some(Some(Kind::Tf));
    }
    if rel.starts_with(".github/workflows/") && matches!(ext, "yml" | "yaml") {
        return Some(Some(Kind::Gha));
    }
    if ext == "sql" {
        return Some(Some(Kind::Sql));
    }
    if fname.starts_with("docker-compose") && matches!(ext, "yml" | "yaml") {
        return Some(None);
    }
    None
}

fn scan_file(rel: &str, kind: Option<Kind>, txt: &str, out: &mut Vec<InfraFinding>) {
    for (idx, line) in txt.lines().enumerate() {
        let ln = idx + 1;
        if let Some(sk) = find_secret(line) {
            push_finding(
                out,
                rel,
                ln,
                "critical",
                &format!("infra_secret_{sk}"),
                "Hardcoded credential/secret committed in infrastructure config",
            );
        }

        let Some(k) = kind else { continue };
        let t = line.trim();
        let tl = t.to_ascii_lowercase();
        let up = t.to_ascii_uppercase();
        for rule in RULES.iter().filter(|r| r.kind as u8 == k as u8) {
            if (rule.check)(t, &tl, &up) {
                push_finding(out, rel, ln, rule.severity, rule.id, rule.msg);
            }
        }
    }

    match kind {
        Some(Kind::Tf) => check_tf(rel, txt, out),
        Some(Kind::Docker) => check_docker(rel, txt, out),
        _ => {}
    }
}

fn check_tf(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    if txt.contains("required_providers") {
        return;
    }

    let (mut in_prov, mut prov_ln, mut has_ver) = (false, 0usize, false);
    for (idx, line) in txt.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("provider ") && t.contains('{') {
            (in_prov, prov_ln, has_ver) = (true, idx + 1, false);
        } else if in_prov {
            has_ver |= t.contains("version");
            if t.starts_with('}') {
                if !has_ver {
                    push_finding(
                        out,
                        rel,
                        prov_ln,
                        "medium",
                        "tf_unversioned_provider",
                        "Terraform provider lacks pinned version constraint",
                    );
                }
                in_prov = false;
            }
        }
    }
}

fn check_docker(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    let mut non_root = false;
    for (idx, line) in txt.lines().enumerate() {
        let t = line.trim();
        let up = t.to_ascii_uppercase();
        if up.starts_with("USER ") {
            let user = t
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .split(':')
                .next()
                .unwrap_or("");
            non_root |= !user.is_empty()
                && !user.eq_ignore_ascii_case("root")
                && user != "0";
        }
        if has_baked(t, &up) {
            push_finding(
                out,
                rel,
                idx + 1,
                "high",
                "docker_baked_secret",
                "Sensitive ENV/ARG value baked into Docker image layer",
            );
        }
    }

    if !non_root && !txt.trim().is_empty() {
        push_finding(
            out,
            rel,
            1,
            "high",
            "docker_runs_as_root",
            "Dockerfile missing non-root `USER` instruction",
        );
    }
}

fn has_baked(t: &str, up: &str) -> bool {
    let is_env = up.starts_with("ENV ") || up.starts_with("ARG ");
    let sensitive = up.contains("SECRET")
        || up.contains("PASSWORD")
        || up.contains("API_KEY");
    if !(is_env && sensitive) {
        return false;
    }

    let rest = t[4..].trim();
    let val = rest
        .split_once('=')
        .or_else(|| rest.split_once(char::is_whitespace))
        .map(|(_, v)| v.trim().trim_matches(['"', '\'']))
        .unwrap_or("");
    !val.is_empty() && !val.starts_with('$')
}

fn collect_manifest_deps(
    manifests: &[(String, String, String)],
) -> (Vec<ParsedDep>, [Option<EcoMatcher>; 4]) {
    let mut builders: [EcoBuilder; 4] = Default::default();
    let mut parsed = Vec::new();
    for (manifest, fname, txt) in manifests {
        for dep in parse_deps(fname, txt) {
            let eco = Eco::from_name(dep.ecosystem);
            let slot = builders[eco.idx()].add_dep(eco, &dep.package);
            parsed.push(ParsedDep {
                manifest: manifest.clone(),
                ecosystem: dep.ecosystem,
                package: dep.package,
                eco,
                slot,
            });
        }
    }
    let [b0, b1, b2, b3] = builders;
    let matchers = [
        b0.build(Eco::ALL[0]),
        b1.build(Eco::ALL[1]),
        b2.build(Eco::ALL[2]),
        b3.build(Eco::ALL[3]),
    ];
    (parsed, matchers)
}

fn all_matchers_complete(matchers: &[Option<EcoMatcher>; 4]) -> bool {
    matchers.iter().all(|m| m.as_ref().is_none_or(EcoMatcher::is_complete))
}

fn scan_sources_for_deps(
    matchers: &mut [Option<EcoMatcher>; 4],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    source_paths: &[(Eco, PathBuf)],
) {
    if let Some(map) = preloaded {
        for (rel, bytes) in map {
            let eco = Eco::from_ext(&ext_of(rel));
            if let Some(m) = matchers[eco.idx()].as_mut() {
                m.scan_bytes(bytes);
            }
            if all_matchers_complete(matchers) {
                return;
            }
        }
        return;
    }

    for (eco, abs) in source_paths {
        let Some(m) = matchers[eco.idx()].as_mut() else { continue };
        if m.is_complete() {
            continue;
        }
        if let Ok(bytes) = fs::read(abs) {
            m.scan_bytes(&bytes);
        }
        if all_matchers_complete(matchers) {
            return;
        }
    }
}

fn find_unused_deps(
    manifests: &[(String, String, String)],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    source_paths: &[(Eco, PathBuf)],
) -> Vec<UnusedDep> {
    if manifests.is_empty() {
        return Vec::new();
    }
    let (parsed, mut matchers) = collect_manifest_deps(manifests);
    scan_sources_for_deps(&mut matchers, preloaded, source_paths);

    let mut unused = Vec::new();
    for dep in parsed {
        let used = matchers[dep.eco.idx()]
            .as_ref()
            .is_some_and(|m| m.hits.contains(dep.slot));
        if !used {
            unused.push(UnusedDep {
                manifest: dep.manifest,
                ecosystem: dep.ecosystem.into(),
                package: dep.package,
            });
        }
    }
    unused
}
