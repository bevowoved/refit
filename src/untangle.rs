mod infra;
mod similarity;
#[cfg(any(feature = "ste100", test))]
pub mod ste100;

use crate::hygiene::HygieneScorer;
use crate::lex::{
    LangKind, LexCursor, LexKind, extract_comment_body, find_ascii_ci, iter_backtick_spans,
};
use crate::platform::write_atomic;
use crate::source::{RepoSnapshot, count_lines, entrypoint, ext_of, read_bytes, test_path};
use crate::stem::{SymbolGraph, analysis, index_repo_snap};
use crate::types::{
    AuditReport, DocDriftFinding, GodFileScore, InfraFinding, RoleCluster, SustainFiles,
    TriageEntry, UnusedDep,
};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditScope {
    Full,
    Sustain,
}

#[cfg(test)]
pub fn scan_infra(repo_root: &Path) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    infra::scan_parts(repo_root, None)
}

/// Audit clones, dead symbols, large files, duplicate roles, unused deps, and infra configs.
pub fn untangle_repo(repo_root: &Path, hygiene: &HygieneScorer) -> Result<AuditReport> {
    untangle_repo_with_scope(repo_root, hygiene, AuditScope::Full)
}

/// Scoped repository audit allowing `sustain` to skip genome, simiflow, deep-module, and doc-drift.
pub fn untangle_repo_with_scope(
    repo_root: &Path,
    hygiene: &HygieneScorer,
    scope: AuditScope,
) -> Result<AuditReport> {
    let repo_snap = RepoSnapshot::capture(repo_root, true);
    let source_rels = &repo_snap.source_rels;
    let preloaded = &repo_snap.preloaded;
    tracing::debug!(
        sources = source_rels.len(),
        preloaded_bytes = repo_snap.total_preloaded_bytes,
        "captured single-walk repository snapshot"
    );

    let (_discovery, sym_graph) = index_repo_snap(repo_root, &repo_snap);
    let (hygiene_median, hygiene_p10, per_file_h) =
        hygiene.score_files(repo_root, source_rels, Some(preloaded));

    let (csr, _) = analysis::CsrGraph::from_named_edges(source_rels, &sym_graph.runtime_edges);
    let sccs = csr.pearce_scc();
    let scc_count = sccs.iter().filter(|c| c.len() >= 2).count();

    let clone_clusters = similarity::find_clones(source_rels, preloaded);
    let (god_files, orphan_files) = rank_files(repo_root, source_rels, preloaded, &per_file_h, &sym_graph);
    let (total_infra_files, unused_deps, infra_findings) = infra::scan_parts_snap(repo_root, &repo_snap);

    let (capability_clusters, doc_drift_findings, genome_twins, deep_module_findings) =
        if scope == AuditScope::Full {
            let caps = similarity::find_capability_clusters(source_rels, preloaded);
            let drift = if cfg!(any(feature = "doc-drift", test)) {
                scan_doc_drift(repo_root, &repo_snap, &sym_graph)
            } else {
                Vec::new()
            };
            let genome_summary = analysis::analyze_git_cochange(repo_root, source_rels);
            let twins = analysis::format_genome_twins(&genome_summary, 25);
            let arch = analysis::ArchGraphRef {
                all_edges: &sym_graph.edges,
                runtime_edges: &sym_graph.runtime_edges,
                metrics: &sym_graph.metrics_by_file,
                defs_by_file: Some(&sym_graph.defs_by_file),
                refs_by_file: Some(&sym_graph.refs_by_file),
            };
            let ego_map = analysis::build_ego_contexts(
                source_rels,
                &sccs,
                &arch,
                &genome_summary.file_commits,
                &clone_clusters,
            );
            let deep = analysis::deep_module_findings_with_csr(source_rels, &sccs, &arch, &ego_map);
            (caps, drift, twins, deep)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

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
        scc_count,
        dead_symbols: sym_graph.dead_symbols.into_iter().take(80).collect(),
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
        dep_advisories: Vec::new(),
    })
}

fn rank_files(
    repo_root: &Path,
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
    per_file_h: &HashMap<String, crate::types::HygieneResult>,
    snap: &crate::stem::SymbolGraph,
) -> (Vec<GodFileScore>, Vec<String>) {
    let mut god_files = Vec::new();
    let mut orphans = Vec::new();

    for rel in source_rels {
        let loc = preloaded
            .get(rel)
            .map(|b| count_lines(b))
            .or_else(|| read_bytes(repo_root, rel, None).map(|b| count_lines(&b)))
            .unwrap_or(0);
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

    if !report.dep_advisories.is_empty() {
        md.push_str("## Dependency Advisories & SemVer Hops\n\n");
        for adv in report.dep_advisories.iter().take(25) {
            let latest = adv.latest.as_deref().unwrap_or(&adv.installed);
            let hops = if adv.hops.is_empty() {
                String::new()
            } else {
                format!(" (hops: {}, breaking: {})", adv.hops.join(" -> "), adv.breaking_hops)
            };
            let guard = adv
                .keep_guard
                .as_deref()
                .map(|g| format!(" [keep-guard: {g}]"))
                .unwrap_or_default();
            md.push_str(&format!(
                "- [{}] `{}/{}` `{}` -> `{}`{} — {}{}\n",
                adv.kind, adv.ecosystem, adv.name, adv.installed, latest, hops, adv.recommendation, guard
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

/// Compile and lint an audit report using ASD-STE100 Issue 9 rules + optional user `--ste-dict`.
#[cfg(any(feature = "ste100", test))]
pub fn render_audit_ste100(
    report: &AuditReport,
    dict_path: Option<&Path>,
) -> Result<(String, ste100::DocumentLintReport)> {
    let dict = dict_path.map(ste100::SteDictionary::load_from_file).transpose()?;
    let compiled = ste100::compile_audit_markdown(report);
    let lint = ste100::lint_markdown(&compiled, dict.as_ref());
    Ok((compiled, lint))
}

/// Generate sustainability guardrails normalized and linted with ASD-STE100 Issue 9 rules.
#[cfg(any(feature = "ste100", test))]
pub fn write_sustain_ste100(
    repo_root: &Path,
    report: &AuditReport,
    dry_run: bool,
    dict_path: Option<&Path>,
) -> Result<(SustainFiles, ste100::DocumentLintReport)> {
    let dict = dict_path.map(ste100::SteDictionary::load_from_file).transpose()?;
    let sustain = write_sustain(repo_root, report, dry_run)?;
    let agents_doc = format!(
        concat!(
            "# AGENTS.md\n\n",
            "1. Keep `codehygiene v0.7.0` score above `0.85` on changed files.\n",
            "2. Keep circular imports at or below `{}`.\n",
            "3. Consolidate shared helpers instead of duplicate code.\n",
            "4. Remove hardcoded credentials, `*:*` wildcards, and root containers.\n",
            "5. Run `refit scan . --fast` and `refit audit .` before a PR.\n"
        ),
        report.scc_count
    );
    let normalized = ste100::normalize_clarity_verbs(&agents_doc);
    if !dry_run {
        write_atomic(&repo_root.join("AGENTS.md"), normalized.as_bytes())?;
    }
    let lint = ste100::lint_markdown(&normalized, dict.as_ref());
    Ok((sustain, lint))
}

// ---------------------------------------------------------------------------
// Folded ReCite / DOCER documentation & comment drift scanner
// ---------------------------------------------------------------------------

const KNOWN_PATH_ROOTS: &[&str] = &[
    "src", "lib", "app", "pkg", "packages", "crates", "cmd", "internal", "tests", "docs",
    "scripts", "infra", ".github", "config", "migrations", "ui", "skills",
];
const KNOWN_FILE_EXTS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "md", "toml", "json", "yml",
    "yaml", "sql", "sh", "html", "css", "tf", "dockerfile",
];
const RUNTIME_PREFIXES: &[&str] = &[
    ".refit/", "target/", "dist/", "build/", "node_modules/", ".venv/", "venv/", "telemetry/",
    "coverage/", "out/", "tmp/",
];
const PLACEHOLDER_SEGMENTS: &[&str] = &[
    "example", "examples", "foo", "bar", "baz", "sample", "dummy", "placeholder", "your_",
    "my_", "path/to/", "some/", "dir/", "mock_",
];
const IGNORED_SYMBOLS: &[&str] = &[
    "main", "new", "default", "clone", "copy", "drop", "len", "is_empty", "push", "pop",
    "get", "set", "insert", "remove", "contains", "iter", "into_iter", "collect", "unwrap",
    "expect", "format", "println", "eprintln", "print", "write", "writeln", "read", "open",
    "close", "flush", "parse", "to_string", "as_str", "as_bytes", "into", "from", "lock",
    "join", "spawn", "sleep", "exit", "abort", "assert", "assert_eq", "range", "super",
    "init", "fetch", "require", "resolve", "reject", "filter", "map", "reduce", "foreach",
    "split", "trim", "slice", "replace", "starts_with", "ends_with", "sort", "dedup",
    "clear", "extend", "find", "any", "all", "count", "take", "skip", "zip", "chain",
    "example", "sample", "handler", "callback", "func", "ident", "method", "my_func",
    "some_fn", "foo_bar", "do_work", "helper", "todo", "unimplemented", "panic", "error",
    "typeof", "sizeof", "alignof", "nameof", "print_r", "printf", "sprintf", "console",
];
const BUILTIN_PKG_SCRIPTS: &[&str] = &[
    "install", "add", "remove", "update", "up", "init", "test", "start", "stop", "restart",
    "publish", "pack", "login", "logout", "audit", "outdated", "link", "unlink", "exec",
    "dlx", "create", "why", "list", "ls", "prune", "dedupe", "ci", "help", "version", "run",
];

struct DriftCtx<'a> {
    repo_root: &'a Path,
    repo_files: &'a BTreeSet<String>,
    repo_dirs: &'a BTreeSet<String>,
    known_symbols: &'a BTreeSet<String>,
    npm_scripts: Option<&'a BTreeSet<String>>,
}

/// Zero-LLM ReCite / DOCER scanner reusing `RepoSnapshot` and `LexCursor` (`!in_test`).
fn scan_doc_drift(
    repo_root: &Path,
    repo_snap: &RepoSnapshot,
    snap: &SymbolGraph,
) -> Vec<DocDriftFinding> {
    let known_symbols: BTreeSet<String> = snap
        .defs_by_file
        .values()
        .chain(snap.refs_by_file.values())
        .flat_map(|syms| syms.iter().cloned())
        .collect();
    let mut repo_files = BTreeSet::new();
    let mut repo_dirs = BTreeSet::new();
    for rel in &repo_snap.all_rels {
        repo_files.insert(rel.clone());
        if let Some((top, _)) = rel.split_once('/') {
            repo_dirs.insert(top.to_string());
        }
    }
    let npm_scripts = load_package_scripts(repo_root);
    let ctx = DriftCtx {
        repo_root,
        repo_files: &repo_files,
        repo_dirs: &repo_dirs,
        known_symbols: &known_symbols,
        npm_scripts: npm_scripts.as_ref(),
    };

    let mut findings = Vec::new();
    for (rel, abs) in &repo_snap.doc_rels {
        if let Ok(content) = fs::read_to_string(abs) {
            scan_markdown_doc(&ctx, rel, &content, &mut findings);
        }
    }
    for rel in &repo_snap.source_rels {
        if test_path(rel) {
            continue;
        }
        let Some(bytes) = read_bytes(repo_root, rel, Some(&repo_snap.preloaded)) else { continue };
        let content = String::from_utf8_lossy(&bytes);
        scan_source_comments(&ctx, rel, &content, &mut findings);
    }

    findings.sort_by(|a, b| {
        a.file
            .cmp(&b.file)
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.reference.cmp(&b.reference))
    });
    findings.dedup_by(|a, b| a.file == b.file && a.line == b.line && a.reference == b.reference);
    findings
}

fn load_package_scripts(repo_root: &Path) -> Option<BTreeSet<String>> {
    let raw = fs::read_to_string(repo_root.join("package.json")).ok()?;
    let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(val.get("scripts")?.as_object()?.keys().cloned().collect())
}

fn scan_markdown_doc(ctx: &DriftCtx<'_>, rel: &str, content: &str, out: &mut Vec<DocDriftFinding>) {
    let mut in_fence = false;
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || trimmed.is_empty() || contains_ignore_marker(trimmed) {
            continue;
        }
        for span in iter_backtick_spans(line) {
            check_doc_span(ctx, rel, idx + 1, span, out);
        }
    }
}

fn contains_ignore_marker(text: &str) -> bool {
    find_ascii_ci(text.as_bytes(), b"refit: ignore").is_some()
}

fn check_doc_span(ctx: &DriftCtx<'_>, doc_file: &str, ln: usize, span: &str, out: &mut Vec<DocDriftFinding>) {
    if let Some(script) = extract_cli_script_ref(span) {
        if let Some(scripts) = ctx.npm_scripts
            && !scripts.contains(script)
            && !is_placeholder_text(script)
        {
            out.push(DocDriftFinding {
                file: doc_file.to_string(),
                line: ln,
                kind: "doc_stale_cli".into(),
                reference: span.to_string(),
                message: format!("Documented script `{script}` missing from package.json scripts"),
            });
        }
        return;
    }
    if let Some(clean_path) = parse_repo_path_candidate(span, ctx.repo_dirs) {
        if !ctx.repo_files.contains(clean_path) && !ctx.repo_root.join(clean_path).exists() {
            out.push(DocDriftFinding {
                file: doc_file.to_string(),
                line: ln,
                kind: "doc_stale_path".into(),
                reference: clean_path.to_string(),
                message: format!("Referenced file path `{clean_path}` does not exist in repository"),
            });
        }
        return;
    }
    if !ctx.known_symbols.is_empty()
        && let Some(sym) = parse_backtick_call_symbol(span)
        && !is_ignored_symbol(sym)
        && !ctx.known_symbols.contains(sym)
    {
        out.push(DocDriftFinding {
            file: doc_file.to_string(),
            line: ln,
            kind: "doc_stale_symbol".into(),
            reference: format!("{sym}()"),
            message: format!("Referenced symbol `{sym}()` not found in live codebase"),
        });
    }
}

fn extract_cli_script_ref(span: &str) -> Option<&str> {
    let rest = span
        .strip_prefix("npm run ")
        .or_else(|| span.strip_prefix("pnpm run "))
        .or_else(|| span.strip_prefix("yarn run "))?;
    let name = rest.split_whitespace().next()?;
    let ok = !name.is_empty()
        && !name.starts_with('-')
        && !has_placeholder_chars(name)
        && !BUILTIN_PKG_SCRIPTS.contains(&name);
    ok.then_some(name)
}

fn has_placeholder_chars(text: &str) -> bool {
    text.bytes().any(|b| matches!(b, b'*' | b'?' | b'<' | b'>' | b'{' | b'}' | b'[' | b']' | b'$'))
        || text.contains("...")
}

fn is_placeholder_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    PLACEHOLDER_SEGMENTS.iter().any(|seg| lower.contains(seg))
}

fn parse_repo_path_candidate<'a>(span: &'a str, repo_dirs: &BTreeSet<String>) -> Option<&'a str> {
    if span.contains(char::is_whitespace)
        || span.contains("://")
        || span.starts_with(['/', '~', '@'])
        || has_placeholder_chars(span)
    {
        return None;
    }
    let no_frag = span.split('#').next().unwrap_or(span);
    let stripped = match no_frag.rsplit_once(':') {
        Some((h, t)) if !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit() || b == b'-') => h,
        _ => no_frag,
    };
    let clean = stripped.trim_start_matches("./");
    if !clean.contains('/') || clean.contains("..") || is_placeholder_text(clean)
        || RUNTIME_PREFIXES.iter().any(|p| clean.starts_with(p))
    {
        return None;
    }
    let (top, _) = clean.split_once('/')?;
    if !KNOWN_PATH_ROOTS.contains(&top) && !repo_dirs.contains(top) {
        return None;
    }
    let (_, ext) = clean.rsplit_once('.')?;
    KNOWN_FILE_EXTS.contains(&ext.to_ascii_lowercase().as_str()).then_some(clean)
}

fn parse_backtick_call_symbol(span: &str) -> Option<&str> {
    if span.contains(char::is_whitespace) || has_placeholder_chars(span) {
        return None;
    }
    let head = span.strip_suffix("()")?;
    let sym = head.rsplit("::").next().unwrap_or(head);
    if sym.contains(['.', '/']) || sym.len() < 4 || is_placeholder_text(sym) {
        return None;
    }
    let valid = sym.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && sym.as_bytes()[0].is_ascii_alphabetic();
    valid.then_some(sym)
}

fn is_ignored_symbol(sym: &str) -> bool {
    IGNORED_SYMBOLS.contains(&sym.to_ascii_lowercase().as_str())
}

fn check_comment_line(
    ctx: &DriftCtx<'_>,
    rel: &str,
    ln: usize,
    comment: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    if comment.is_empty() || contains_ignore_marker(comment) {
        return;
    }
    for span in iter_backtick_spans(comment) {
        if let Some(clean_path) = parse_repo_path_candidate(span, ctx.repo_dirs)
            && !ctx.repo_files.contains(clean_path)
            && !ctx.repo_root.join(clean_path).exists()
        {
            out.push(DocDriftFinding {
                file: rel.to_string(),
                line: ln,
                kind: "doc_stale_path".into(),
                reference: clean_path.to_string(),
                message: format!("Comment cites missing file `{clean_path}`"),
            });
        }
    }
    if !ctx.known_symbols.is_empty() {
        scan_comment_call_refs(ctx, rel, ln, comment, out);
    }
}

fn scan_source_comments(ctx: &DriftCtx<'_>, rel: &str, content: &str, out: &mut Vec<DocDriftFinding>) {
    let ext = ext_of(rel);
    if ext == "rs" {
        let mut cursor = LexCursor::new(LangKind::Rust, content.as_bytes());
        while let Some(tok) = cursor.next_token() {
            if !matches!(tok.kind, LexKind::LineComment | LexKind::BlockComment) || tok.flags.in_test() {
                continue;
            }
            for (offset, raw_line) in tok.as_str().lines().enumerate() {
                if let Some(comment) = extract_comment_body("rs", raw_line) {
                    check_comment_line(ctx, rel, (tok.line as usize) + offset, comment, out);
                }
            }
        }
        return;
    }

    for (idx, line) in content.lines().enumerate() {
        if let Some(comment) = extract_comment_body(&ext, line) {
            check_comment_line(ctx, rel, idx + 1, comment, out);
        }
    }
}

fn scan_comment_call_refs(
    ctx: &DriftCtx<'_>,
    rel: &str,
    ln: usize,
    comment: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    let bytes = comment.as_bytes();
    let mut pos = 0usize;
    while pos + 2 < bytes.len() {
        if !(bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_') {
            pos += 1;
            continue;
        }
        let start = pos;
        pos += 1;
        while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
            pos += 1;
        }
        let is_call = pos + 1 < bytes.len() && bytes[pos] == b'(' && bytes[pos + 1] == b')';
        let preceded_by_dot = start > 0 && bytes[start - 1] == b'.';
        if !is_call || preceded_by_dot {
            continue;
        }
        let sym = &comment[start..pos];
        if sym.len() >= 5
            && !is_ignored_symbol(sym)
            && !is_placeholder_text(sym)
            && !ctx.known_symbols.contains(sym)
        {
            out.push(DocDriftFinding {
                file: rel.to_string(),
                line: ln,
                kind: "doc_stale_symbol".into(),
                reference: format!("{sym}()"),
                message: format!("Stale comment reference `{sym}()` not found in live codebase"),
            });
        }
    }
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

        let (ste_md, ste_lint) = render_audit_ste100(&rep, None).unwrap();
        assert!(ste_md.contains("STE100 Issue 9 Audit Report"));
        assert!(ste_lint.compliant, "compiled STE100 audit had violations: {:?}", ste_lint.issues);

        let (sus_ste, sus_lint) = write_sustain_ste100(dir.path(), &rep, true, None).unwrap();
        assert_eq!(sus_ste.files_written.len(), 4);
        assert!(sus_lint.compliant, "compiled STE100 sustain had violations: {:?}", sus_lint.issues);
    }

    #[test]
    fn drift_skips_cfg_test_comments_catches_prod() {
        let auth_rs = concat!(
            "// Delegates to live_verify_token() safely\n",
            "// Legacy note: previously called legacy_parse_wal() and `src/ghost_prod.rs`\n",
            "pub fn live_verify_token(raw: &str) -> bool { !raw.is_empty() }\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    // Comment inside cfg(test) citing fake_test_only_fn() and `src/ghost_test.rs`\n",
            "    #[test]\n",
            "    fn ok() { assert!(super::live_verify_token(\"x\")); }\n",
            "}\n",
        );
        let readme = concat!(
            "# Demo\n",
            "Live `src/auth.rs` calls `live_verify_token()`.\n",
            "Dead `src/deleted_legacy.rs`, `old_broken_helper()`, `npm run ghost-task`.\n",
            "Ignore `src/example/foo.rs`, `to_string()`, and live `npm run build`.\n",
        );
        let dir = TempRepo::with_files(
            "doc_drift_cfg",
            &[
                ("src/auth.rs", auth_rs),
                ("tests/smoke_test.rs", "// Mentions test_only_ghost_fn() in test path\n"),
                ("README.md", readme),
                ("package.json", r#"{"scripts":{"build":"vite build"}}"#),
            ],
        );

        let rep = untangle_repo(dir.path(), &HygieneScorer::new()).unwrap();
        let has = |file: &str, kind: &str, r: &str| {
            rep.doc_drift_findings.iter().any(|f| f.file == file && f.kind == kind && f.reference == r)
        };
        assert!(has("src/auth.rs", "doc_stale_symbol", "legacy_parse_wal()"), "{:?}", rep.doc_drift_findings);
        assert!(has("src/auth.rs", "doc_stale_path", "src/ghost_prod.rs"), "{:?}", rep.doc_drift_findings);
        assert!(has("README.md", "doc_stale_path", "src/deleted_legacy.rs"), "{:?}", rep.doc_drift_findings);
        assert!(has("README.md", "doc_stale_symbol", "old_broken_helper()"), "{:?}", rep.doc_drift_findings);
        assert!(has("README.md", "doc_stale_cli", "npm run ghost-task"), "{:?}", rep.doc_drift_findings);

        // Zero false positives on #[cfg(test)] block, test-only path, placeholders, or live symbols.
        for bad in ["fake_test_only_fn", "ghost_test.rs", "test_only_ghost_fn", "live_verify_token", "example", "to_string", "npm run build"] {
            assert!(
                !rep.doc_drift_findings.iter().any(|f| f.reference.contains(bad)),
                "false positive `{bad}` in {:?}",
                rep.doc_drift_findings
            );
        }
    }
}
