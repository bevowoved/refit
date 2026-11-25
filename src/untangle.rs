mod clones;
mod drift;
mod infra;
mod simiflow;

use crate::graph::dag_from;
use crate::hygiene::HygieneScorer;
use crate::platform::{walk_tree, write_atomic};
use crate::source::{count_lines, entrypoint, ext_of, rel_norm, source_rel, supported_ext};
use crate::stem::{csr, genome, index_repo};
use crate::types::{
    AuditReport, GodFileScore, InfraFinding, RoleCluster, SustainFiles, TriageEntry, UnusedDep,
};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

#[cfg(test)]
pub fn scan_infra(repo_root: &Path) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    infra::scan_parts(repo_root, None)
}

/// Audit clones, dead symbols, large files, duplicate roles, unused deps, and infra configs.
pub fn untangle_repo(repo_root: &Path, hygiene: &HygieneScorer) -> Result<AuditReport> {
    let (source_rels, preloaded) = load_sources(repo_root);
    let (discovery, snap) = index_repo(repo_root, Some(&preloaded));
    // Pre-warm hygiene cache via scheduled batch scoring BEFORE `dag_from` so cold audit
    // never falls back to sequential per-unit scoring inside `build_unit`.
    let (hygiene_median, hygiene_p10, per_file_h) =
        hygiene.score_files(repo_root, &source_rels, Some(&preloaded));
    let plan = dag_from(repo_root, discovery, hygiene, Some(&preloaded))?;

    let clone_clusters = clones::find_clones(&source_rels, &preloaded);
    let (god_files, orphan_files) = rank_files(&source_rels, &preloaded, &per_file_h, &snap);
    let capability_clusters = simiflow::find_capability_clusters(&source_rels, &preloaded);
    let (total_infra_files, unused_deps, infra_findings) =
        infra::scan_parts(repo_root, Some(&preloaded));
    let genome_summary = genome::analyze_git_cochange(repo_root, &source_rels);
    let genome_twins = genome::format_genome_twins(&genome_summary, 25);
    let deep_module_findings =
        csr::deep_module_findings(&source_rels, &snap.runtime_edges, &snap.metrics_by_file);
    let known_symbols = drift::collect_known_symbols(&snap, &preloaded);
    let doc_drift_findings =
        drift::scan_doc_drift(repo_root, &source_rels, &preloaded, &known_symbols);

    let triage_plan = build_triage(
        &infra_findings,
        &capability_clusters,
        &god_files,
        &unused_deps,
        &orphan_files,
    );

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
        genome_twins,
        deep_module_findings,
        doc_drift_findings,
    })
}

fn load_sources(repo_root: &Path) -> (Vec<String>, BTreeMap<String, Vec<u8>>) {
    let mut preloaded = BTreeMap::new();
    let mut source_rels = Vec::new();

    for path in walk_tree(repo_root, None, false) {
        if !supported_ext(&ext_of(&path)) {
            continue;
        }
        let Some(rel) = rel_norm(repo_root, &path) else { continue };
        let Ok(bytes) = fs::read(&path) else { continue };

        if source_rel(&rel) {
            source_rels.push(rel.clone());
        }
        preloaded.insert(rel, bytes);
    }

    source_rels.sort();
    (source_rels, preloaded)
}

fn rank_files(
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
    per_file_h: &HashMap<String, crate::types::HygieneResult>,
    snap: &crate::stem::SymbolGraph,
) -> (Vec<GodFileScore>, Vec<String>) {
    let mut god_files = Vec::new();
    let mut orphans = Vec::new();

    for rel in source_rels {
        let loc = preloaded.get(rel).map(|b| count_lines(b)).unwrap_or(0);
        let h_score = per_file_h.get(rel).map(|h| h.score).unwrap_or(0.90);
        let fi = snap.fan_in.get(rel).copied().unwrap_or(0);
        let fo = snap.fan_out.get(rel).copied().unwrap_or(0);

        if fi == 0 && fo == 0 && !entrypoint(rel) {
            orphans.push(rel.clone());
        }

        let risk = (1.0 - h_score).max(0.02)
            * (loc.max(2) as f64).ln()
            * (1.0 + (fi + fo) as f64);
        let rec = if loc >= 500 && h_score < 0.75 {
            "split_god_file"
        } else if h_score < 0.85 {
            "refactor_in_place"
        } else {
            "keep"
        };

        if rec != "keep" || risk >= 6.0 {
            god_files.push(GodFileScore {
                file: rel.clone(),
                loc,
                hygiene_score: h_score,
                fan_in: fi,
                fan_out: fo,
                risk_score: (risk * 100.0).round() / 100.0,
                recommendation: rec.into(),
            });
        }
    }

    god_files.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    god_files.truncate(25);
    (god_files, orphans)
}

fn build_triage(
    infra: &[InfraFinding],
    caps: &[RoleCluster],
    gods: &[GodFileScore],
    unused: &[UnusedDep],
    orphans: &[String],
) -> Vec<TriageEntry> {
    let mut plan = Vec::new();
    let mut push = |file: String, action: &str, reason: String, priority: u32| {
        plan.push(TriageEntry {
            file,
            action: action.into(),
            reason,
            priority,
        });
    };

    for f in infra.iter().filter(|f| matches!(f.severity.as_str(), "critical" | "high")).take(15) {
        let prio = if f.severity == "critical" { 1 } else { 2 };
        push(f.file.clone(), "harden_infra", format!("[{}] L{}: {}", f.rule_id, f.line, f.message), prio);
    }
    for cap in caps {
        push(cap.files.join(", "), "consolidate", cap.recommendation.clone(), 2);
    }
    for g in gods.iter().filter(|g| g.recommendation == "split_god_file").take(10) {
        let msg = format!(
            "{} LOC, hygiene {:.3}, fan_in={}, fan_out={} (risk {:.1})",
            g.loc, g.hygiene_score, g.fan_in, g.fan_out, g.risk_score
        );
        push(g.file.clone(), "split", msg, 2);
    }
    for u in unused.iter().take(15) {
        push(u.manifest.clone(), "remove_unused_dep", format!("Unused {} dep `{}`", u.ecosystem, u.package), 3);
    }
    for o in orphans.iter().take(10) {
        push(o.clone(), "review_delete", "Zero graph edges (candidate dead file)".into(), 4);
    }

    plan.sort_by_key(|t| (t.priority, t.file.clone()));
    plan
}

pub fn render_audit(report: &AuditReport) -> String {
    let mut md = format!(
        concat!(
            "# refit Phase 0 Untangle Audit Report\n\n",
            "- **Repository:** `{}`\n",
            "- **Source files:** {} | **Infra/Config files:** {}\n",
            "- **Hygiene (codehygiene v0.7.0):** median `{:.3}`, p10 `{:.3}`\n",
            "- **SCC Cycles:** {} | **Dead Exported Symbols:** {} | ",
            "**Cross-File Clones:** {} | **Unused Deps:** {} | **Infra Findings:** {}\n\n"
        ),
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
    );

    if !report.triage_plan.is_empty() {
        md.push_str("## Prioritized Triage Plan\n\n");
        md.push_str("| Priority | Action | Target | Reason |\n| :--- | :--- | :--- | :--- |\n");
        for t in report.triage_plan.iter().take(25) {
            md.push_str(&format!(
                "| P{} | `{}` | `{}` | {} |\n",
                t.priority, t.action, t.file, t.reason
            ));
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

    if !report.deep_module_findings.is_empty() {
        md.push_str("## Deep-Module Architectural Findings\n\n");
        for f in report.deep_module_findings.iter().take(15) {
            md.push_str(&format!("- {f}\n"));
        }
        md.push('\n');
    }

    if !report.genome_twins.is_empty() {
        md.push_str("## Top Git Co-Change Twins (Clean-Room NPMI)\n\n");
        for tw in report.genome_twins.iter().take(15) {
            md.push_str(&format!("- {tw}\n"));
        }
        md.push('\n');
    }

    if !report.doc_drift_findings.is_empty() {
        md.push_str("## Documentation & Comment Drift (ReCite / DOCER)\n\n");
        for df in report.doc_drift_findings.iter().take(20) {
            md.push_str(&format!(
                "- `{}` L{} [{}]: `{}` — {}\n",
                df.file, df.line, df.kind, df.reference, df.message
            ));
        }
        md.push('\n');
    }
    md
}

pub fn write_sustain(
    repo_root: &Path,
    report: &AuditReport,
    dry_run: bool,
) -> Result<SustainFiles> {
    let dirs: BTreeSet<&str> = report
        .god_files
        .iter()
        .filter_map(|g| g.file.split('/').next())
        .collect();
    let layers = dirs.len().max(1);

    let agents = format!(
        concat!(
            "# AGENTS.md — Repository Sustainability Guardrails\n\n",
            "Generated by `refit v0.5.0 sustain` to prevent AI-sprawl re-tangling.\n\n",
            "## Architectural Rules\n",
            "1. Keep `codehygiene v0.7.0` maintainability score `>= 0.85` on modified files.\n",
            "2. Never introduce circular imports (`scc_count` baseline: `{}`).\n",
            "3. Consolidate shared capabilities instead of duplicating helpers.\n",
            "4. Never commit credentials, `*:*` IAM wildcards, or root Dockerfiles.\n",
            "5. Run `refit scan . --fast` and `refit audit .` before opening a PR.\n\n",
            "## Code-Precedence Guardrail\n",
            "- When docs (`README.md`, `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, or comments) ",
            "and code disagree, trust live code & tests first and update stale markdown immediately.\n",
            "- Keep all backticked file paths and `ident()` symbol references verified against live code.\n"
        ),
        report.scc_count
    );

    let invariants = concat!(
        "rules:\n",
        "- id: no_ui_to_db_direct_import\n",
        "  paths: [src/ui/, app/components/, ui/]\n",
        "  forbid_import: \"db\"\n",
        "  message: \"UI components must call service/API boundaries, not DB modules\"\n",
        "- id: no_wildcard_iam\n",
        "  languages: [tf, yml, yaml]\n",
        "  forbid: \"*:*\"\n",
        "  message: \"Wildcard IAM permissions forbidden under SOC 2 / ISO 27001\"\n"
    );

    let adr = format!(
        concat!(
            "# ADR-001: Architectural Untangling & Sustainability Baseline\n\n",
            "- **Status:** Accepted\n",
            "- **Baseline:** {} source files, {} infra files, median hygiene `{:.3}`, ",
            "{} SCCs, {} clone clusters, {} dead exported symbols.\n",
            "- **Decision:** Enforce layered boundaries via `.refit/invariants.yml` ",
            "and CI scans via `.github/workflows/refit-guard.yml`.\n"
        ),
        report.total_source_files,
        report.total_infra_files,
        report.hygiene_median,
        report.scc_count,
        report.clone_clusters.len(),
        report.dead_symbols.len()
    );

    let ci = concat!(
        "name: refit-sustain-guard\n",
        "on:\n  pull_request:\n",
        "permissions:\n  contents: read\n",
        "jobs:\n  untangle-guard:\n    runs-on: ubuntu-latest\n    steps:\n",
        "      - uses: actions/checkout@v4\n",
        "      - name: Run refit structural & hygiene audit\n",
        "        run: refit audit . --markdown\n"
    );

    let items = [
        ("AGENTS.md", agents),
        (".refit/invariants.yml", invariants.into()),
        ("docs/ADR-001-untangle.md", adr),
        (".github/workflows/refit-guard.yml", ci.into()),
    ];

    let mut written = Vec::with_capacity(items.len());
    for (rel, body) in items {
        if !dry_run {
            write_atomic(&repo_root.join(rel), body.as_bytes())?;
        }
        written.push(rel.to_string());
    }

    Ok(SustainFiles {
        files_written: written,
        invariant_rules_count: 2,
        layers_detected: layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn test_untangle_audit_infra_and_sustain() {
        let block = concat!(
            "    const normalized = raw.trim().toLowerCase();\n",
            "    const parts = normalized.split(':');\n",
            "    const checksum = parts.length * 31;\n",
            "    const output = checksum + normalized.length;\n",
        );
        let a1 = format!("export function authOne(raw: string) {{\n{block}    return output;\n}}\n");
        let a2 = format!("export function authTwo(raw: string) {{\n{block}    return output;\n}}\n");
        let tf = "provider \"aws\" {}\nresource \"aws_iam_policy\" \"bad\" {\n  policy = \"{\\\"Action\\\": \\\"*:*\\\"}\"\n}\n";
        let py = "[project]\ndependencies = [\"ghost-py-pkg>=1.0\"]\n[tool.ruff]\nline-length = 88\n";
        let go = "module example.com/app\n\nrequire (\n\tgithub.com/unused/mod v1.0.0\n)\n";
        let ci = "permissions: write-all\nsteps:\n  - uses: actions/checkout@main\n  - run: echo ${{ github.event.issue.title }}\n";

        let dir = TempRepo::with_files(
            "unt_all",
            &[
                ("src/auth_v1.ts", &a1),
                ("src/auth_v2.ts", &a2),
                ("src/app.py", "import os\n"),
                ("src/main.go", "package main\nfunc main() {}\n"),
                ("package.json", r#"{"dependencies":{"unused-ghost-pkg":"^1.0.0"}}"#),
                ("pyproject.toml", py),
                ("go.mod", go),
                ("infra/main.tf", tf),
                ("Dockerfile", "FROM node:latest\nENV API_KEY=hardcoded\n"),
                (".github/workflows/ci.yml", ci),
            ],
        );

        let rep = untangle_repo(dir.path(), &HygieneScorer::new()).unwrap();
        assert!(!rep.clone_clusters.is_empty() && !rep.capability_clusters.is_empty());
        for (eco, pkg) in [("npm", "unused-ghost-pkg"), ("python", "ghost-py-pkg"), ("go", "github.com/unused/mod")] {
            assert!(rep.unused_deps.iter().any(|u| u.ecosystem == eco && u.package == pkg), "missing {eco}:{pkg}");
        }
        assert!(!rep.unused_deps.iter().any(|u| u.package.contains("line-length")));

        let rules: BTreeSet<&str> = rep.infra_findings.iter().map(|f| f.rule_id.as_str()).collect();
        for id in [
            "tf_wildcard_iam", "docker_runs_as_root", "docker_latest_tag",
            "gha_write_all_perms", "gha_unpinned_action", "gha_script_injection",
        ] {
            assert!(rules.contains(id), "missing {id}");
        }
        assert!(!rep.triage_plan.is_empty() && render_audit(&rep).contains("Untangle Audit Report"));

        let sus = write_sustain(dir.path(), &rep, false).unwrap();
        assert_eq!(sus.files_written.len(), 4);
        for rel in ["AGENTS.md", ".refit/invariants.yml", "docs/ADR-001-untangle.md", ".github/workflows/refit-guard.yml"] {
            assert!(dir.has(rel), "missing {rel}");
        }

        dir.write("Dockerfile", "FROM alpine:3.20\nUSER rootless\n");
        let (_, _, f2) = scan_infra(dir.path());
        assert!(!f2.iter().any(|f| f.rule_id == "docker_runs_as_root"));
    }
}
