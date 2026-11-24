use crate::lex::has_word;
use crate::platform::{find_secret, walk_tree};
use crate::source::{ext_of, parse_deps, rel_norm, supported_ext};
use crate::types::{InfraFinding, UnusedDep};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Default)]
struct LangCorpus {
    js: String,
    rs: String,
    py: String,
    go: String,
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
    let mut corpora = LangCorpus::default();
    let mut manifests = Vec::new();

    if let Some(map) = preloaded {
        for (rel, bytes) in map {
            push_corpus(&mut corpora, &ext_of(rel), &String::from_utf8_lossy(bytes));
        }
    }

    for abs in walk_tree(repo_root, None, true) {
        let Some(rel) = rel_norm(repo_root, &abs) else { continue };
        let fname = abs.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ext = ext_of(&abs);

        if supported_ext(&ext) {
            if preloaded.is_none()
                && let Ok(txt) = fs::read_to_string(&abs)
            {
                push_corpus(&mut corpora, &ext, &txt);
            }
            continue;
        }

        if matches!(fname, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
            if let Ok(txt) = fs::read_to_string(&abs) {
                manifests.push((rel, fname.to_string(), txt));
            }
            continue;
        }

        let Some(kind) = infra_kind(&rel, fname, &ext) else { continue };
        infra_count += 1;
        let Ok(txt) = fs::read_to_string(&abs) else { continue };
        scan_file(&rel, kind, &txt, &mut findings);
    }

    let unused = dead_deps(&manifests, &corpora);
    (infra_count, unused, findings)
}

fn push_corpus(corpora: &mut LangCorpus, ext: &str, txt: &str) {
    let dst = match ext {
        "rs" => &mut corpora.rs,
        "py" => &mut corpora.py,
        "go" => &mut corpora.go,
        _ => &mut corpora.js,
    };
    dst.push_str(txt);
    dst.push('\n');
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

fn dead_deps(manifests: &[(String, String, String)], corpora: &LangCorpus) -> Vec<UnusedDep> {
    let py_lower = corpora.py.to_ascii_lowercase();
    let mut unused = Vec::new();

    for (manifest, fname, txt) in manifests {
        for dep in parse_deps(fname, txt) {
            let used = match dep.ecosystem {
                "npm" => has_word(&corpora.js, &dep.package),
                "cargo" => {
                    let ident = dep.package.replace('-', "_");
                    has_word(&corpora.rs, &ident) || has_word(&corpora.rs, &dep.package)
                }
                "python" => {
                    let lo = dep.package.to_ascii_lowercase();
                    has_word(&py_lower, &lo.replace('-', "_")) || has_word(&py_lower, &lo)
                }
                _ => corpora.go.contains(&dep.package),
            };
            if !used {
                unused.push(UnusedDep {
                    manifest: manifest.clone(),
                    ecosystem: dep.ecosystem.into(),
                    package: dep.package,
                });
            }
        }
    }
    unused
}
