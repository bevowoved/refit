use crate::graph::dag_from;
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::{write_atomic, find_secret, walk_tree};
use crate::stem::index_repo;
use crate::types::{
    RoleCluster, CloneCluster, GodFileScore, InfraFinding, SustainFiles, TriageEntry,
    AuditReport, UnusedDep,
};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

/// Audit clones, dead symbols, large files, duplicate roles, unused dependencies, and infra configs.
pub fn untangle_repo(
    repo_root: &Path,
    hygiene: &HygieneScorer,
) -> Result<AuditReport> {
    let mut preloaded: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut source_rels: Vec<String> = Vec::new();
    for path in walk_tree(repo_root, None, false) {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !crate::graph::supported_ext(&ext) {
            continue;
        }
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        if let Ok(bytes) = fs::read(&path) {
            if !rel_s.starts_with("tests/")
                && !rel_s.starts_with("test_")
                && !rel_s.contains("/test_")
                && !rel_s.contains(".test.")
                && !rel_s.contains(".spec.")
                && !rel_s.ends_with("_test.go")
                && !rel_s.ends_with("__init__.py")
            {
                source_rels.push(rel_s.clone());
            }
            preloaded.insert(rel_s, bytes);
        }
    }
    source_rels.sort();

    let (discovery, snap) = index_repo(repo_root, Some(&preloaded));
    let plan = dag_from(repo_root, discovery, hygiene, Some(&preloaded))?;
    let (hygiene_median, hygiene_p10, per_file_h) =
        hygiene.score_files(repo_root, &source_rels, Some(&preloaded));

    // 1) Cross-file structural clone detection (normalized 4-line windows across >= 2 distinct files).
    let clone_clusters = find_clones(&source_rels, &preloaded);

    // 2) God-file structural risk ranking: (1 - hygiene) * ln(LOC) * (1 + fan_in + fan_out).
    let mut god_files: Vec<GodFileScore> = Vec::new();
    let mut orphan_files: Vec<String> = Vec::new();
    for rel in &source_rels {
        let loc = preloaded
            .get(rel)
            .map(|b| String::from_utf8_lossy(b).lines().count())
            .unwrap_or(0);
        let h_score = per_file_h.get(rel).map(|h| h.score).unwrap_or(0.90);
        let fi = snap.fan_in.get(rel).copied().unwrap_or(0);
        let fo = snap.fan_out.get(rel).copied().unwrap_or(0);
        if fi == 0
            && fo == 0
            && !rel.ends_with("main.rs")
            && !rel.ends_with("lib.rs")
            && !rel.ends_with("main.py")
            && !rel.ends_with("main.go")
            && !rel.ends_with("index.ts")
        {
            orphan_files.push(rel.clone());
        }
        let risk = (1.0 - h_score).max(0.02) * (loc.max(2) as f64).ln() * (1.0 + (fi + fo) as f64);
        let recommendation = if loc >= 500 && h_score < 0.75 {
            "split_god_file"
        } else if h_score < 0.85 {
            "refactor_in_place"
        } else {
            "keep"
        };
        if recommendation != "keep" || risk >= 6.0 {
            god_files.push(GodFileScore {
                file: rel.clone(),
                loc,
                hygiene_score: h_score,
                fan_in: fi,
                fan_out: fo,
                risk_score: (risk * 100.0).round() / 100.0,
                recommendation: recommendation.to_string(),
            });
        }
    }
    god_files.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    god_files.truncate(25);

    // 3) Conflicting / duplicated capability clusters (e.g., 3 separate auth or http implementations).
    let capability_clusters = dup_roles(&source_rels);

    // 4) IaC / Docker / GitHub Actions / SQL / Manifest unused-dependency scanner.
    let (total_infra_files, unused_deps, infra_findings) = scan_infra(repo_root);

    // 5) Synthesize prioritized untangle.work triage plan (Keep / Split / Consolidate / Harden / Delete).
    let mut triage_plan: Vec<TriageEntry> = Vec::new();
    for inf in infra_findings.iter().filter(|f| f.severity == "critical" || f.severity == "high").take(15) {
        triage_plan.push(TriageEntry {
            file: inf.file.clone(),
            action: "harden_infra".into(),
            reason: format!("[{}] L{}: {}", inf.rule_id, inf.line, inf.message),
            priority: if inf.severity == "critical" { 1 } else { 2 },
        });
    }
    for cap in &capability_clusters {
        triage_plan.push(TriageEntry {
            file: cap.files.join(", "),
            action: "consolidate".into(),
            reason: cap.recommendation.clone(),
            priority: 2,
        });
    }
    for gf in god_files.iter().filter(|g| g.recommendation == "split_god_file").take(10) {
        triage_plan.push(TriageEntry {
            file: gf.file.clone(),
            action: "split".into(),
            reason: format!(
                "{} LOC, hygiene {:.3}, fan_in={}, fan_out={} (risk {:.1})",
                gf.loc, gf.hygiene_score, gf.fan_in, gf.fan_out, gf.risk_score
            ),
            priority: 2,
        });
    }
    for ud in unused_deps.iter().take(15) {
        triage_plan.push(TriageEntry {
            file: ud.manifest.clone(),
            action: "remove_unused_dep".into(),
            reason: format!("Unused {} dependency `{}`", ud.ecosystem, ud.package),
            priority: 3,
        });
    }
    for orph in orphan_files.iter().take(10) {
        triage_plan.push(TriageEntry {
            file: orph.clone(),
            action: "review_delete".into(),
            reason: "Zero inbound and zero outbound graph edges (candidate dead file)".into(),
            priority: 4,
        });
    }
    triage_plan.sort_by_key(|t| (t.priority, t.file.clone()));

    Ok(AuditReport {
        repo_root: repo_root.display().to_string(),
        total_source_files: source_rels.len(),
        total_infra_files,
        hygiene_median,
        hygiene_p10,
        scc_count: plan.scc_count,
        dead_symbols: snap.dead_symbols.into_iter().take(80).collect(),
        orphan_files,
        clone_clusters,
        god_files,
        capability_clusters,
        unused_deps,
        infra_findings,
        triage_plan,
    })
}

fn find_clones(
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
) -> Vec<CloneCluster> {
    let mut windows: HashMap<String, (String, Vec<(String, usize)>)> = HashMap::new();
    for rel in source_rels {
        let Some(bytes) = preloaded.get(rel) else {
            continue;
        };
        let text = String::from_utf8_lossy(bytes);
        let norm_lines: Vec<(usize, String)> = text
            .lines()
            .enumerate()
            .filter_map(|(idx, l)| {
                let t = l.trim();
                if t.len() < 12
                    || t.starts_with("//")
                    || t.starts_with('#')
                    || t.starts_with("import ")
                    || t.starts_with("from ")
                    || t.starts_with("use ")
                {
                    None
                } else {
                    Some((idx + 1, t.split_whitespace().collect::<Vec<_>>().join(" ")))
                }
            })
            .collect();
        if norm_lines.len() < 4 {
            continue;
        }
        for w in norm_lines.windows(4) {
            let joined = format!("{}\n{}\n{}\n{}", w[0].1, w[1].1, w[2].1, w[3].1);
            let fp = MvccStore::hash_bytes(joined.as_bytes())[..12].to_string();
            let entry = windows
                .entry(fp)
                .or_insert_with(|| (w[0].1.clone(), Vec::new()));
            if !entry.1.iter().any(|(f, _)| f == rel) {
                entry.1.push((rel.clone(), w[0].0));
            }
        }
    }

    let mut clusters: Vec<CloneCluster> = windows
        .into_iter()
        .filter(|(_, (_, occ))| occ.len() >= 2)
        .map(|(fp, (sample, mut occ))| {
            occ.sort();
            CloneCluster {
                fingerprint: fp,
                line_count: 4,
                sample_line: sample.chars().take(100).collect(),
                occurrences: occ,
            }
        })
        .collect();
    clusters.sort_by(|a, b| {
        b.occurrences
            .len()
            .cmp(&a.occurrences.len())
            .then_with(|| a.occurrences.cmp(&b.occurrences))
            .then_with(|| a.fingerprint.cmp(&b.fingerprint))
    });
    // Suppress overlapping shifted 4-line windows on the same file set within 4 lines of an accepted clone.
    let mut deduped: Vec<CloneCluster> = Vec::new();
    for c in clusters {
        let overlaps = deduped.iter().any(|prev| {
            prev.occurrences.len() == c.occurrences.len()
                && prev
                    .occurrences
                    .iter()
                    .zip(c.occurrences.iter())
                    .all(|((pf, pl), (cf, cl))| pf == cf && pl.abs_diff(*cl) <= 4)
        });
        if !overlaps {
            deduped.push(c);
            if deduped.len() >= 25 {
                break;
            }
        }
    }
    deduped
}

fn split_ident(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if !ch.is_ascii_alphanumeric() {
            if !cur.is_empty() {
                tokens.push(cur.to_ascii_lowercase());
                cur.clear();
            }
        } else if ch.is_ascii_uppercase() && !cur.is_empty() && cur.chars().last().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
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

fn dup_roles(source_rels: &[String]) -> Vec<RoleCluster> {
    const DOMAINS: &[(&str, &[&str])] = &[
        ("auth", &["auth", "jwt", "oauth", "session", "login"]),
        ("http_client", &["http", "fetch", "client", "request"]),
        ("database", &["db", "database", "prisma", "sql", "store", "repo"]),
        ("config", &["config", "settings", "env"]),
        ("logging", &["log", "logger", "telemetry"]),
    ];
    let mut out = Vec::new();
    for &(cap, keywords) in DOMAINS {
        let mut matched: Vec<String> = source_rels
            .iter()
            .filter(|rel| {
                let p = Path::new(rel);
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let mut toks = split_ident(stem);
                if matches!(stem, "index" | "mod" | "lib" | "main") {
                    if let Some(parent) = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()) {
                        toks.extend(split_ident(parent));
                    }
                }
                keywords.iter().any(|k| toks.iter().any(|t| t == k))
            })
            .cloned()
            .collect();
        matched.sort();
        if matched.len() >= 2 {
            out.push(RoleCluster {
                capability: cap.to_string(),
                recommendation: format!(
                    "Detected {} separate `{cap}` modules; consolidate into one canonical implementation",
                    matched.len()
                ),
                files: matched,
            });
        }
    }
    out
}

#[derive(Default)]
struct LangCorpus {
    js: String,
    rs: String,
    py: String,
    go: String,
}

/// Scan Terraform, Dockerfiles, GitHub Actions, SQL scripts, and dependency manifests.
pub fn scan_infra(
    repo_root: &Path,
) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    let all_files = walk_tree(repo_root, None, true);
    let mut infra_count = 0usize;
    let mut findings: Vec<InfraFinding> = Vec::new();
    let mut corpora = LangCorpus::default();
    let mut manifests: Vec<(String, String)> = Vec::new();

    for abs in &all_files {
        let Ok(rel_p) = abs.strip_prefix(repo_root) else {
            continue;
        };
        let rel = rel_p.to_string_lossy().replace('\\', "/");
        let file_name = abs
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let ext = abs
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if crate::graph::supported_ext(&ext) {
            if let Ok(txt) = fs::read_to_string(abs) {
                match ext.as_str() {
                    "rs" => {
                        corpora.rs.push_str(&txt);
                        corpora.rs.push('\n');
                    }
                    "py" => {
                        corpora.py.push_str(&txt);
                        corpora.py.push('\n');
                    }
                    "go" => {
                        corpora.go.push_str(&txt);
                        corpora.go.push('\n');
                    }
                    _ => {
                        corpora.js.push_str(&txt);
                        corpora.js.push('\n');
                    }
                }
            }
            continue;
        }

        if matches!(file_name, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
            if let Ok(txt) = fs::read_to_string(abs) {
                manifests.push((rel.clone(), txt));
            }
            continue;
        }

        let is_dockerfile = file_name.starts_with("Dockerfile") || file_name.ends_with(".dockerfile");
        let is_tf = ext == "tf" || ext == "tfvars";
        let is_gha = rel.starts_with(".github/workflows/") && (ext == "yml" || ext == "yaml");
        let is_sql = ext == "sql";
        let is_compose = file_name.starts_with("docker-compose") && (ext == "yml" || ext == "yaml");

        if !(is_dockerfile || is_tf || is_gha || is_sql || is_compose) {
            continue;
        }
        infra_count += 1;
        let Ok(txt) = fs::read_to_string(abs) else {
            continue;
        };

        // Universal secret scan across ALL infra/config files.
        for (idx, line) in txt.lines().enumerate() {
            if let Some(sk) = find_secret(line) {
                findings.push(InfraFinding {
                    file: rel.clone(),
                    line: idx + 1,
                    severity: "critical".into(),
                    rule_id: format!("infra_secret_{sk}"),
                    message: "Hardcoded credential/secret committed in infrastructure config".into(),
                });
            }
        }

        if is_tf {
            let mut in_provider = false;
            let mut provider_line = 0usize;
            let mut provider_has_version = false;
            for (idx, line) in txt.lines().enumerate() {
                let t = line.trim();
                let tl = t.to_ascii_lowercase();
                if t.contains("*:*")
                    || ((tl.contains("action") || tl.contains("resources"))
                        && (t.contains("\"*\"") || t.contains("\\\"*\\\"")))
                {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "critical".into(),
                        rule_id: "tf_wildcard_iam".into(),
                        message: "Overly permissive wildcard IAM policy (`*` / `*:*`)".into(),
                    });
                }
                if t.contains("0.0.0.0/0") || t.contains("::/0") {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "high".into(),
                        rule_id: "tf_open_cidr".into(),
                        message: "Security group/route open to 0.0.0.0/0 or ::/0".into(),
                    });
                }
                if t.starts_with("provider ") && t.contains('{') {
                    in_provider = true;
                    provider_line = idx + 1;
                    provider_has_version = false;
                } else if in_provider {
                    if t.contains("version") {
                        provider_has_version = true;
                    }
                    if t.starts_with('}') {
                        if !provider_has_version && !txt.contains("required_providers") {
                            findings.push(InfraFinding {
                                file: rel.clone(),
                                line: provider_line,
                                severity: "medium".into(),
                                rule_id: "tf_unversioned_provider".into(),
                                message: "Terraform provider lacks pinned version constraint".into(),
                            });
                        }
                        in_provider = false;
                    }
                }
            }
        }

        if is_dockerfile {
            let mut has_non_root_user = false;
            for (idx, line) in txt.lines().enumerate() {
                let t = line.trim();
                let upper = t.to_ascii_uppercase();
                if upper.starts_with("USER ") {
                    let user_tok = t
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .split(':')
                        .next()
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if !user_tok.is_empty() && user_tok != "root" && user_tok != "0" {
                        has_non_root_user = true;
                    }
                }
                if upper.starts_with("FROM ") && t.contains(":latest") {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "medium".into(),
                        rule_id: "docker_latest_tag".into(),
                        message: "Unpinned `:latest` base image causes non-reproducible builds".into(),
                    });
                }
                if upper.starts_with("ADD ") && (t.contains("http://") || t.contains("https://")) {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "high".into(),
                        rule_id: "docker_add_remote_url".into(),
                        message: "Dockerfile `ADD` fetches remote URL without integrity checksum".into(),
                    });
                }
                if (upper.starts_with("ENV ") || upper.starts_with("ARG "))
                    && (upper.contains("SECRET") || upper.contains("PASSWORD") || upper.contains("API_KEY"))
                {
                    let rest = t[4..].trim();
                    let val_opt = if let Some((_, v)) = rest.split_once('=') {
                        Some(v.trim())
                    } else {
                        rest.split_once(char::is_whitespace).map(|(_, v)| v.trim())
                    };
                    if let Some(val) = val_opt {
                        let clean = val.trim_matches(['"', '\'']);
                        if !clean.is_empty() && !clean.starts_with('$') {
                            findings.push(InfraFinding {
                                file: rel.clone(),
                                line: idx + 1,
                                severity: "high".into(),
                                rule_id: "docker_baked_secret".into(),
                                message: "Sensitive ENV/ARG value baked into Docker image layer".into(),
                            });
                        }
                    }
                }
            }
            if !has_non_root_user && !txt.trim().is_empty() {
                findings.push(InfraFinding {
                    file: rel.clone(),
                    line: 1,
                    severity: "high".into(),
                    rule_id: "docker_runs_as_root".into(),
                    message: "Dockerfile missing non-root `USER` instruction".into(),
                });
            }
        }

        if is_gha {
            for (idx, line) in txt.lines().enumerate() {
                let t = line.trim();
                let tl = t.to_ascii_lowercase();
                if t.contains("write-all") {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "high".into(),
                        rule_id: "gha_write_all_perms".into(),
                        message: "GitHub Actions workflow grants overly broad `write-all` permissions".into(),
                    });
                }
                if (t.starts_with("uses:") || t.starts_with("- uses:"))
                    && (t.ends_with("@main") || t.ends_with("@master"))
                {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "medium".into(),
                        rule_id: "gha_unpinned_action".into(),
                        message: "GitHub Action pinned to mutable branch (`@main`/`@master`) instead of tag/SHA".into(),
                    });
                }
                if t.contains("${{") && tl.contains("github.event.") {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "high".into(),
                        rule_id: "gha_script_injection".into(),
                        message: "Untrusted `${{ github.event.* }}` interpolation risks GitHub Actions script injection".into(),
                    });
                }
            }
        }

        if is_sql {
            for (idx, line) in txt.lines().enumerate() {
                let upper = line.trim().to_ascii_uppercase();
                if (upper.starts_with("DROP TABLE ") && !upper.contains("IF EXISTS"))
                    || upper.starts_with("TRUNCATE TABLE ")
                    || upper.starts_with("TRUNCATE ")
                {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "high".into(),
                        rule_id: "sql_destructive_migration".into(),
                        message: "Unguarded destructive DROP/TRUNCATE statement in SQL migration".into(),
                    });
                }
                if upper.contains("IDENTIFIED BY '") || upper.contains("WITH PASSWORD '") {
                    findings.push(InfraFinding {
                        file: rel.clone(),
                        line: idx + 1,
                        severity: "critical".into(),
                        rule_id: "sql_hardcoded_password".into(),
                        message: "Hardcoded database password literal in SQL script".into(),
                    });
                }
            }
        }
    }

    let unused_deps = dead_deps(&manifests, &corpora);
    (infra_count, unused_deps, findings)
}

fn has_word(corpus: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = corpus.as_bytes();
    for (pos, _) in corpus.match_indices(needle) {
        let left_ok = pos == 0 || !(bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_');
        let end = pos + needle.len();
        let right_ok = end == bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

fn dead_deps(manifests: &[(String, String)], corpora: &LangCorpus) -> Vec<UnusedDep> {
    let mut unused = Vec::new();
    let py_lower = corpora.py.to_ascii_lowercase();
    for (manifest, txt) in manifests {
        let fname = Path::new(manifest).file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fname == "package.json" {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(txt) {
                if let Some(deps) = v.get("dependencies").and_then(|d| d.as_object()) {
                    for pkg in deps.keys() {
                        if !has_word(&corpora.js, pkg) {
                            unused.push(UnusedDep {
                                manifest: manifest.clone(),
                                ecosystem: "npm".into(),
                                package: pkg.clone(),
                            });
                        }
                    }
                }
            }
        } else if fname == "Cargo.toml" {
            let mut in_deps = false;
            for line in txt.lines() {
                let t = line.trim();
                if t.starts_with('[') {
                    in_deps = t == "[dependencies]";
                    continue;
                }
                if in_deps {
                    if let Some((k, _)) = t.split_once('=') {
                        let pkg = k.trim();
                        if !pkg.is_empty() && !pkg.starts_with('#') {
                            let ident = pkg.replace('-', "_");
                            if !has_word(&corpora.rs, &ident) && !has_word(&corpora.rs, pkg) {
                                unused.push(UnusedDep {
                                    manifest: manifest.clone(),
                                    ecosystem: "cargo".into(),
                                    package: pkg.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        } else if fname == "pyproject.toml" {
            let mut in_deps_array = false;
            let mut in_poetry_deps = false;
            for line in txt.lines() {
                let t = line.trim();
                if t.starts_with('[') {
                    in_deps_array = false;
                    in_poetry_deps = t == "[tool.poetry.dependencies]";
                    continue;
                }
                if in_poetry_deps {
                    if let Some((k, _)) = t.split_once('=') {
                        let pkg = k.trim();
                        if !pkg.is_empty() && !pkg.starts_with('#') && !pkg.eq_ignore_ascii_case("python") {
                            let py_ident = pkg.replace('-', "_").to_ascii_lowercase();
                            if !has_word(&py_lower, &py_ident) && !has_word(&py_lower, &pkg.to_ascii_lowercase()) {
                                unused.push(UnusedDep {
                                    manifest: manifest.clone(),
                                    ecosystem: "python".into(),
                                    package: pkg.to_string(),
                                });
                            }
                        }
                    }
                    continue;
                }
                if t.starts_with("dependencies") && t.contains('[') {
                    // Handle single-line `dependencies = ["requests", "httpx"]` as well as multi-line arrays.
                    if let Some(open) = t.find('[') {
                        let rest = &t[open + 1..];
                        let (items_part, closed) = if let Some(close) = rest.find(']') {
                            (&rest[..close], true)
                        } else {
                            (rest, false)
                        };
                        for raw_item in items_part.split(',') {
                            let pkg = raw_item
                                .trim()
                                .trim_matches(|c: char| c == '"' || c == '\'')
                                .split(['=', '>', '<', '~', '!', '[', ' '])
                                .next()
                                .unwrap_or("");
                            if !pkg.is_empty() {
                                let py_ident = pkg.replace('-', "_").to_ascii_lowercase();
                                if !has_word(&py_lower, &py_ident) && !has_word(&py_lower, &pkg.to_ascii_lowercase()) {
                                    unused.push(UnusedDep {
                                        manifest: manifest.clone(),
                                        ecosystem: "python".into(),
                                        package: pkg.to_string(),
                                    });
                                }
                            }
                        }
                        in_deps_array = !closed;
                    }
                    continue;
                }
                if in_deps_array {
                    if t.starts_with(']') {
                        in_deps_array = false;
                        continue;
                    }
                    let pkg = t
                        .trim_matches(|c: char| c == '"' || c == '\'' || c == ',')
                        .split(['=', '>', '<', '~', '!', '[', ' '])
                        .next()
                        .unwrap_or("");
                    if !pkg.is_empty() {
                        let py_ident = pkg.replace('-', "_").to_ascii_lowercase();
                        if !has_word(&py_lower, &py_ident) && !has_word(&py_lower, &pkg.to_ascii_lowercase()) {
                            unused.push(UnusedDep {
                                manifest: manifest.clone(),
                                ecosystem: "python".into(),
                                package: pkg.to_string(),
                            });
                        }
                    }
                }
            }
        } else if fname == "go.mod" {
            let mut in_require = false;
            for line in txt.lines() {
                let t = line.trim();
                if t.starts_with("require (") {
                    in_require = true;
                    continue;
                }
                if in_require {
                    if t.starts_with(')') {
                        in_require = false;
                        continue;
                    }
                    if t.contains("// indirect") || t.starts_with("//") {
                        continue;
                    }
                    if let Some(mod_path) = t.split_whitespace().next() {
                        if !mod_path.is_empty() && !corpora.go.contains(mod_path) {
                            unused.push(UnusedDep {
                                manifest: manifest.clone(),
                                ecosystem: "go".into(),
                                package: mod_path.to_string(),
                            });
                        }
                    }
                } else if let Some(rest) = t.strip_prefix("require ") {
                    if !rest.contains("// indirect") {
                        if let Some(mod_path) = rest.split_whitespace().next() {
                            if !mod_path.is_empty() && !corpora.go.contains(mod_path) {
                                unused.push(UnusedDep {
                                    manifest: manifest.clone(),
                                    ecosystem: "go".into(),
                                    package: mod_path.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    unused
}

pub fn render_audit(report: &AuditReport) -> String {
    let mut md = String::new();
    md.push_str("# refit Phase 0 Untangle Audit Report\n\n");
    md.push_str(&format!(
        "- **Repository:** `{}`\n- **Source files:** {} | **Infra/Config files:** {}\n- **Hygiene (codehygiene v0.7.0):** median `{:.3}`, p10 `{:.3}`\n- **SCC Cycles:** {} | **Dead Exported Symbols:** {} | **Cross-File Clones:** {} | **Unused Deps:** {} | **Infra Findings:** {}\n\n",
        report.repo_root,
        report.total_source_files,
        report.total_infra_files,
        report.hygiene_median,
        report.hygiene_p10,
        report.scc_count,
        report.dead_symbols.len(),
        report.clone_clusters.len(),
        report.unused_deps.len(),
        report.infra_findings.len()
    ));

    if !report.triage_plan.is_empty() {
        md.push_str("## Prioritized Triage Plan (Keep / Split / Consolidate / Harden / Delete)\n\n");
        md.push_str("| Priority | Action | Target | Reason |\n| :--- | :--- | :--- | :--- |\n");
        for t in report.triage_plan.iter().take(25) {
            md.push_str(&format!("| P{} | `{}` | `{}` | {} |\n", t.priority, t.action, t.file, t.reason));
        }
        md.push('\n');
    }
    if !report.god_files.is_empty() {
        md.push_str("## Top Structural Risk / God Files\n\n");
        for g in report.god_files.iter().take(10) {
            md.push_str(&format!(
                "- `{}` — {} LOC, hygiene `{:.3}`, fan-in `{}`, fan-out `{}`, risk `{:.2}` -> **{}**\n",
                g.file, g.loc, g.hygiene_score, g.fan_in, g.fan_out, g.risk_score, g.recommendation
            ));
        }
        md.push('\n');
    }
    md
}

/// Generate `AGENTS.md`, `.refit/invariants.yml`, `docs/ADR-001-untangle.md`, and the CI guard workflow.
pub fn write_sustain(
    repo_root: &Path,
    report: &AuditReport,
    dry_run: bool,
) -> Result<SustainFiles> {
    let mut dirs: BTreeSet<String> = BTreeSet::new();
    for g in &report.god_files {
        if let Some(top) = g.file.split('/').next() {
            dirs.insert(top.to_string());
        }
    }
    let layers_detected = dirs.len().max(1);

    let agents_md = format!(
        "# AGENTS.md — Repository Sustainability Guardrails\n\n\
         Generated by `refit v0.5.0 sustain` to prevent AI-sprawl re-tangling.\n\n\
         ## Architectural Rules\n\
         1. Keep `codehygiene v0.7.0` maintainability score `>= 0.85` on all modified files.\n\
         2. Never introduce circular imports (`scc_count` baseline: `{}`).\n\
         3. Consolidate shared capabilities instead of duplicating helpers (see `.refit/invariants.yml`).\n\
         4. Never commit hardcoded credentials, `*:*` IAM wildcards, or root-user Dockerfiles.\n\
         5. Run `refit scan . --fast` and `refit audit .` before opening a PR.\n",
        report.scc_count
    );

    let invariants_yml = "\
rules:\n\
- id: no_ui_to_db_direct_import\n\
  paths: [src/ui/, app/components/, ui/]\n\
  forbid_import: \"db\"\n\
  message: \"UI components must call service/API boundaries, not DB modules directly\"\n\
- id: no_wildcard_iam\n\
  languages: [tf, yml, yaml]\n\
  forbid: \"*:*\"\n\
  message: \"Wildcard IAM permissions forbidden under SOC 2 / ISO 27001 policy\"\n";

    let adr_md = format!(
        "# ADR-001: Architectural Untangling & Sustainability Baseline\n\n\
         - **Status:** Accepted\n\
         - **Baseline:** {} source files, {} infra files, median hygiene `{:.3}`, {} SCCs, {} clone clusters, {} dead exported symbols.\n\
         - **Decision:** Enforce layered module boundaries via `.refit/invariants.yml`, automated CI scans via `.github/workflows/refit-guard.yml`, and agent constraints via `AGENTS.md`.\n",
        report.total_source_files,
        report.total_infra_files,
        report.hygiene_median,
        report.scc_count,
        report.clone_clusters.len(),
        report.dead_symbols.len()
    );

    let ci_yml = "\
name: refit-sustain-guard\n\
on:\n\
  pull_request:\n\
permissions:\n\
  contents: read\n\
jobs:\n\
  untangle-guard:\n\
    runs-on: ubuntu-latest\n\
    steps:\n\
      - uses: actions/checkout@v4\n\
      - name: Run refit structural & hygiene audit\n\
        run: refit audit . --markdown\n";

    let artifacts: Vec<(&str, String)> = vec![
        ("AGENTS.md", agents_md),
        (".refit/invariants.yml", invariants_yml.to_string()),
        ("docs/ADR-001-untangle.md", adr_md),
        (".github/workflows/refit-guard.yml", ci_yml.to_string()),
    ];

    let mut written = Vec::new();
    for (rel, body) in artifacts {
        if !dry_run {
            write_atomic(&repo_root.join(rel), body.as_bytes())?;
        }
        written.push(rel.to_string());
    }

    Ok(SustainFiles {
        files_written: written,
        invariant_rules_count: 2,
        layers_detected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;

    #[test]
    fn probe_83_audit_sustain() {
        let dir = std::env::temp_dir().join(format!("refit_unt83_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("infra")).unwrap();
        fs::create_dir_all(dir.join(".github/workflows")).unwrap();

        let shared_block = "    const normalized = raw.trim().toLowerCase();\n    const parts = normalized.split(':');\n    const checksum = parts.length * 31;\n    const output = checksum + normalized.length;\n";
        fs::write(
            dir.join("src/auth_v1.ts"),
            format!("export function authOne(raw: string) {{\n{shared_block}    return output;\n}}\n"),
        )
        .unwrap();
        fs::write(
            dir.join("src/auth_v2.ts"),
            format!("export function authTwo(raw: string) {{\n{shared_block}    return output;\n}}\n"),
        )
        .unwrap();
        fs::write(
            dir.join("package.json"),
            r#"{"dependencies":{"unused-ghost-pkg":"^1.0.0"}}"#,
        )
        .unwrap();
        fs::write(
            dir.join("infra/main.tf"),
            "provider \"aws\" {}\nresource \"aws_iam_policy\" \"bad\" {\n  policy = \"{\\\"Action\\\": \\\"*:*\\\"}\"\n}\n",
        )
        .unwrap();
        fs::write(dir.join("Dockerfile"), "FROM node:latest\nENV API_KEY=hardcoded\n").unwrap();
        fs::write(
            dir.join(".github/workflows/ci.yml"),
            "permissions: write-all\nsteps:\n  - uses: actions/checkout@main\n",
        )
        .unwrap();

        let scorer = HygieneScorer::new();
        let report = untangle_repo(&dir, &scorer).unwrap();
        assert!(!report.clone_clusters.is_empty(), "expected cross-file clone cluster");
        assert!(!report.capability_clusters.is_empty(), "expected duplicate auth capability cluster");
        assert!(report.unused_deps.iter().any(|u| u.package == "unused-ghost-pkg"));
        assert!(report.infra_findings.iter().any(|f| f.rule_id == "tf_wildcard_iam"));
        assert!(report.infra_findings.iter().any(|f| f.rule_id == "docker_runs_as_root"));
        assert!(report.infra_findings.iter().any(|f| f.rule_id == "docker_latest_tag"));
        assert!(report.infra_findings.iter().any(|f| f.rule_id == "gha_write_all_perms"));
        assert!(report.infra_findings.iter().any(|f| f.rule_id == "gha_unpinned_action"));
        assert!(!report.triage_plan.is_empty());

        let md = render_audit(&report);
        assert!(md.contains("Untangle Audit Report"));

        let sustain = write_sustain(&dir, &report, false).unwrap();
        assert_eq!(sustain.files_written.len(), 4);
        assert!(dir.join("AGENTS.md").exists());
        assert!(dir.join(".refit/invariants.yml").exists());
        assert!(dir.join("docs/ADR-001-untangle.md").exists());
        assert!(dir.join(".github/workflows/refit-guard.yml").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_100_infra_edges() {
        let dir = std::env::temp_dir().join(format!("refit_unt100_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join(".github/workflows")).unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/app.py"), "import os\n").unwrap();
        fs::write(dir.join("src/main.go"), "package main\nfunc main() {}\n").unwrap();
        fs::write(dir.join("pyproject.toml"), "[project]\ndependencies = [\"ghost-py-pkg>=1.0\"]\n[tool.ruff]\nline-length = 88\n").unwrap();
        fs::write(dir.join("go.mod"), "module example.com/app\n\nrequire (\n\tgithub.com/unused/mod v1.0.0\n)\n").unwrap();
        fs::write(dir.join("Dockerfile"), "FROM alpine:3.20\nUSER rootless\n").unwrap();
        fs::write(
            dir.join(".github/workflows/pr.yml"),
            "steps:\n  - run: echo ${{ github.event.issue.title }}\n",
        )
        .unwrap();

        let (_, unused, findings) = scan_infra(&dir);
        assert!(unused.iter().any(|u| u.ecosystem == "python" && u.package == "ghost-py-pkg"), "unused={unused:?}");
        assert!(!unused.iter().any(|u| u.package.contains("line-length")), "toml bleed: {unused:?}");
        assert!(unused.iter().any(|u| u.ecosystem == "go" && u.package == "github.com/unused/mod"), "unused={unused:?}");
        assert!(!findings.iter().any(|f| f.rule_id == "docker_runs_as_root"), "USER rootless false-flagged: {findings:?}");
        assert!(findings.iter().any(|f| f.rule_id == "gha_script_injection"), "missing gha_script_injection: {findings:?}");
        let _ = fs::remove_dir_all(&dir);
    }
}
