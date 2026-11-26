mod infra;
mod similarity;
#[cfg(any(feature = "ste100", test))]
pub mod ste100;
pub use infra::{
    SecretScanReport, escape_gh_prop, install_secret_hooks, push_sarif_result, render_secrets_gh,
    render_secrets_sarif, run_agent_hook, scan_repo_secrets,
};

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
use anyhow::{Context, Result};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

static REPO_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:\./)?([A-Za-z0-9._-]+/[^ \t\r\n*?<>{}\[\]$#:]+?\.([A-Za-z]+))",
        r"(?::[0-9-]+)?(?:#[^ \t\r\n*?<>{}\[\]$]*)?$",
    ))
    .expect("valid repo path regex")
});
static BT_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[A-Za-z0-9_]+::)*([A-Za-z][A-Za-z0-9_]{3,})\(\)$").expect("valid bt call regex")
});
static COMMENT_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([A-Za-z_][A-Za-z0-9_]{4,})\(\)").expect("valid comment call regex")
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditScope { Full, Sustain }

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
    repo_root: &Path, hygiene: &HygieneScorer, scope: AuditScope,
) -> Result<AuditReport> {
    let repo_snap = RepoSnapshot::capture(repo_root, true);
    let (source_rels, preloaded) = (&repo_snap.source_rels, &repo_snap.preloaded);
    tracing::debug!(sources = source_rels.len(), bytes = repo_snap.total_preloaded_bytes, "snapshot");

    let (_disc, sym_graph) = index_repo_snap(repo_root, &repo_snap);
    let (hygiene_median, hygiene_p10, per_file_h) =
        hygiene.score_files(repo_root, source_rels, Some(preloaded));
    let (csr, _) = analysis::CsrGraph::from_named_edges(source_rels, &sym_graph.runtime_edges);
    let sccs = csr.pearce_scc();
    let scc_count = sccs.iter().filter(|c| c.len() >= 2).count();
    let clone_clusters = similarity::find_clones(source_rels, preloaded);
    let (god_files, orphan_files) =
        rank_files(repo_root, source_rels, preloaded, &per_file_h, &sym_graph);
    let (total_infra_files, unused_deps, infra_findings) =
        infra::scan_parts_snap(repo_root, &repo_snap);

    let (capability_clusters, doc_drift_findings, genome_twins, deep_module_findings) =
        if scope == AuditScope::Full {
            let caps = similarity::find_capability_clusters(source_rels, preloaded);
            let drift = if cfg!(any(feature = "doc-drift", test)) {
                scan_doc_drift(repo_root, &repo_snap, &sym_graph)
            } else { Vec::new() };
            let genome = analysis::analyze_git_cochange(repo_root, source_rels);
            let twins = analysis::format_genome_twins(&genome, 25);
            let arch = analysis::ArchGraphRef {
                all_edges: &sym_graph.edges, runtime_edges: &sym_graph.runtime_edges,
                metrics: &sym_graph.metrics_by_file,
                defs_by_file: Some(&sym_graph.defs_by_file),
                refs_by_file: Some(&sym_graph.refs_by_file),
                sources: Some(preloaded),
            };
            let (_ego, deep) = analysis::analyze_ego_and_deep_modules(
                source_rels, &sccs, &arch, &genome.file_commits, &clone_clusters,
            );
            (caps, drift, twins, deep)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

    let triage_plan =
        build_triage(&infra_findings, &capability_clusters, &god_files, &unused_deps, &orphan_files);
    Ok(AuditReport {
        repo_root: repo_root.display().to_string(), total_source_files: source_rels.len(),
        total_infra_files, hygiene_median, hygiene_p10, scc_count,
        dead_symbols: sym_graph.dead_symbols.into_iter().take(80).collect(),
        orphan_files, clone_clusters, god_files, capability_clusters, unused_deps, infra_findings,
        triage_plan, genome_twins, deep_module_findings, doc_drift_findings,
        dep_advisories: Vec::new(),
    })
}

fn rank_files(
    repo_root: &Path, source_rels: &[String], preloaded: &BTreeMap<String, Vec<u8>>,
    per_file_h: &HashMap<String, crate::types::HygieneResult>, snap: &SymbolGraph,
) -> (Vec<GodFileScore>, Vec<String>) {
    let (mut god_files, mut orphans) = (Vec::new(), Vec::new());
    for rel in source_rels {
        let loc = preloaded.get(rel).map(|b| count_lines(b))
            .or_else(|| read_bytes(repo_root, rel, None).map(|b| count_lines(&b))).unwrap_or(0);
        let h_score = per_file_h.get(rel).map_or(0.90, |h| h.score);
        let fi = snap.fan_in.get(rel).copied().unwrap_or(0);
        let fo = snap.fan_out.get(rel).copied().unwrap_or(0);
        if fi == 0 && fo == 0 && !entrypoint(rel) { orphans.push(rel.clone()); }
        let risk = (1.0 - h_score).max(0.02) * (loc.max(2) as f64).ln() * (1.0 + (fi + fo) as f64);
        let rec = if loc >= 500 && h_score < 0.75 { "split_god_file" }
            else if h_score < 0.85 { "refactor_in_place" } else { "keep" };
        if rec != "keep" || risk >= 6.0 {
            god_files.push(GodFileScore {
                file: rel.clone(), loc, hygiene_score: h_score, fan_in: fi, fan_out: fo,
                risk_score: (risk * 100.0).round() / 100.0, recommendation: rec.into(),
            });
        }
    }
    god_files.sort_by(|a, b| b.risk_score.total_cmp(&a.risk_score));
    god_files.truncate(25);
    (god_files, orphans)
}

fn build_triage(
    infra: &[InfraFinding], caps: &[RoleCluster], gods: &[GodFileScore],
    unused: &[UnusedDep], orphans: &[String],
) -> Vec<TriageEntry> {
    let mut plan = Vec::new();
    let mut push = |file: String, action: &str, reason: String, priority: u32| {
        plan.push(TriageEntry { file, action: action.into(), reason, priority });
    };
    for finding in infra.iter().filter(|f| matches!(f.severity.as_str(), "critical" | "high")).take(15) {
        let prio = if finding.severity == "critical" { 1 } else { 2 };
        push(finding.file.clone(), "harden_infra",
            format!("[{}] L{}: {}", finding.rule_id, finding.line, finding.message), prio);
    }
    for cap in caps { push(cap.files.join(", "), "consolidate", cap.recommendation.clone(), 2); }
    for god in gods.iter().filter(|g| g.recommendation == "split_god_file").take(10) {
        let msg = format!("{} LOC, hygiene {:.3}, fan_in={}, fan_out={} (risk {:.1})",
            god.loc, god.hygiene_score, god.fan_in, god.fan_out, god.risk_score);
        push(god.file.clone(), "split", msg, 2);
    }
    for dep in unused.iter().take(15) {
        push(dep.manifest.clone(), "remove_unused_dep",
            format!("Unused {} dep `{}`", dep.ecosystem, dep.package), 3);
    }
    for orp in orphans.iter().take(10) {
        push(orp.clone(), "review_delete", "Zero graph edges (candidate dead file)".into(), 4);
    }
    plan.sort_by_key(|t| (t.priority, t.file.clone()));
    plan
}

fn push_md_section<T>(
    md: &mut String, header: &str, items: &[T], limit: usize, mut fmt_item: impl FnMut(&mut String, &T),
) {
    if items.is_empty() { return; }
    md.push_str(header);
    for item in items.iter().take(limit) { fmt_item(md, item); }
    md.push('\n');
}

pub fn render_audit(report: &AuditReport) -> String {
    let mut md = format!(
        concat!(
            "# refit Audit Report\n\n",
            "- **Repository:** `{}`\n- **Source files:** {} | **Infra/Config files:** {}\n",
            "- **Hygiene:** median `{:.3}`, p10 `{:.3}`\n",
            "- **SCC Cycles:** {} | **Dead Exported Symbols:** {} | ",
            "**Cross-File Clones:** {} | **Unused Deps:** {} | **Infra Findings:** {}\n\n"
        ),
        report.repo_root, report.total_source_files, report.total_infra_files,
        report.hygiene_median, report.hygiene_p10, report.scc_count, report.dead_symbols.len(),
        report.clone_clusters.len(), report.unused_deps.len(), report.infra_findings.len()
    );

    let triage_hdr = "## Prioritized Triage Plan\n\n\
        | Priority | Action | Target | Reason |\n| :--- | :--- | :--- | :--- |\n";
    push_md_section(&mut md, triage_hdr, &report.triage_plan, 25, |s, t| {
        let _ = writeln!(s, "| P{} | `{}` | `{}` | {} |", t.priority, t.action, t.file, t.reason);
    });
    push_md_section(&mut md, "## Top Structural Risk / God Files\n\n", &report.god_files, 10, |s, g| {
        let _ = writeln!(s, "- `{}` — {} LOC, hygiene `{:.3}`, fan-in `{}`, fan-out `{}`, risk `{:.2}` -> **{}**",
            g.file, g.loc, g.hygiene_score, g.fan_in, g.fan_out, g.risk_score, g.recommendation);
    });
    for (hdr, items) in [
        ("## Deep-Module Architectural Findings\n\n", &report.deep_module_findings),
        ("## Top Git Co-Change Twins\n\n", &report.genome_twins),
    ] {
        push_md_section(&mut md, hdr, items, 15, |s, item| { let _ = writeln!(s, "- {item}"); });
    }
    push_md_section(
        &mut md, "## Documentation & Comment Drift\n\n", &report.doc_drift_findings, 20, |s, df| {
            let _ = writeln!(s, "- `{}` L{} [{}]: `{}` — {}",
                df.file, df.line, df.kind, df.reference, df.message);
        },
    );
    push_md_section(
        &mut md, "## Dependency Advisories & SemVer Hops\n\n", &report.dep_advisories, 25, |s, a| {
            let latest = a.latest.as_deref().unwrap_or(&a.installed);
            let hops = if a.hops.is_empty() { String::new() } else {
                format!(" (hops: {}, breaking: {})", a.hops.join(" -> "), a.breaking_hops)
            };
            let guard = a.keep_guard.as_deref().map_or(String::new(), |g| format!(" [keep-guard: {g}]"));
            let _ = writeln!(s, "- [{}] `{}/{}` `{}` -> `{}`{} — {}{}",
                a.kind, a.ecosystem, a.name, a.installed, latest, hops, a.recommendation, guard);
        },
    );
    md
}

fn write_sustain_docs(
    repo_root: &Path, report: &AuditReport, agents: String, dry_run: bool,
) -> Result<SustainFiles> {
    let layers = report.god_files.iter().filter_map(|g| g.file.split('/').next())
        .collect::<BTreeSet<_>>().len().max(1);
    let invariants = concat!(
        "rules:\n",
        "- id: no_ui_to_db_direct_import\n  paths: [src/ui/, app/components/, ui/]\n",
        "  forbid_import: \"db\"\n",
        "  message: \"UI components must call service/API boundaries, not DB modules\"\n",
        "- id: no_wildcard_iam\n  languages: [tf, yml, yaml]\n  forbid: \"*:*\"\n",
        "  message: \"Wildcard IAM permissions forbidden under SOC 2 / ISO 27001\"\n"
    );
    let adr = format!(
        concat!(
            "# ADR-001: Architectural Untangling & Sustainability Baseline\n\n- **Status:** Accepted\n",
            "- **Baseline:** {} source files, {} infra files, median hygiene `{:.3}`, ",
            "{} SCCs, {} clone clusters, {} dead exported symbols.\n",
            "- **Decision:** Enforce layered boundaries via `.refit/invariants.yml` ",
            "and CI scans via `.github/workflows/refit-guard.yml`.\n"
        ),
        report.total_source_files, report.total_infra_files, report.hygiene_median,
        report.scc_count, report.clone_clusters.len(), report.dead_symbols.len()
    );
    let ci = concat!(
        "name: refit-sustain-guard\non:\n  pull_request:\npermissions:\n  contents: read\n",
        "jobs:\n  untangle-guard:\n    runs-on: ubuntu-latest\n    steps:\n",
        "      - uses: actions/checkout@v4\n      - name: Run refit structural & hygiene audit\n",
        "        run: refit audit . --markdown\n"
    );

    let items = [
        ("AGENTS.md", agents), (".refit/invariants.yml", invariants.into()),
        ("docs/ADR-001-untangle.md", adr), (".github/workflows/refit-guard.yml", ci.into()),
    ];
    let mut written = Vec::with_capacity(items.len());
    for (rel, body) in items {
        if !dry_run { write_atomic(&repo_root.join(rel), body.as_bytes())?; }
        written.push(rel.to_string());
    }
    Ok(SustainFiles { files_written: written, invariant_rules_count: 2, layers_detected: layers })
}

pub fn write_sustain(repo_root: &Path, report: &AuditReport, dry_run: bool) -> Result<SustainFiles> {
    let agents = format!(
        concat!(
            "# AGENTS.md — Repository Guardrails\n\nGenerated by `refit sustain`.\n\n",
            "## Architectural Rules\n",
            "1. Keep maintainability score `>= 0.85` on modified files.\n",
            "2. Never introduce circular imports (`scc_count` baseline: `{}`).\n",
            "3. Consolidate shared capabilities instead of duplicating helpers.\n",
            "4. Never commit credentials, `*:*` IAM wildcards, or root Dockerfiles.\n",
            "5. Run `refit scan . --fast` and `refit audit .` before opening a PR.\n\n",
            "## Code-Precedence Guardrail\n",
            "- When docs (`README.md`, `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, or comments) ",
            "and code disagree, trust live code & tests first and update stale markdown immediately.\n",
            "- Keep all backticked file paths and `ident()` symbol references verified against live code.\n",
            "- Verify user-facing web/mobile changes with `refit check . --kane <spec>` (Gate 4 via Kane CLI).\n"
        ),
        report.scc_count
    );
    write_sustain_docs(repo_root, report, agents, dry_run)
}

fn load_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() { return Ok(Map::new()); }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if raw.trim().is_empty() { return Ok(Map::new()); }
    let val: Value = serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    val.as_object().cloned().with_context(|| format!("expected JSON object in {}", path.display()))
}

fn ensure_object<'a>(map: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let entry = map.entry(key.to_string()).or_insert_with(|| Value::Object(Map::new()));
    if !entry.is_object() { *entry = Value::Object(Map::new()); }
    entry.as_object_mut().expect("object")
}

fn ensure_array<'a>(map: &'a mut Map<String, Value>, key: &str) -> &'a mut Vec<Value> {
    let entry = map.entry(key.to_string()).or_insert_with(|| Value::Array(Vec::new()));
    if !entry.is_array() { *entry = Value::Array(Vec::new()); }
    entry.as_array_mut().expect("array")
}

fn has_refit_hook_command(entry: &Value) -> bool {
    if entry.get("command").and_then(Value::as_str) == Some("refit hook") { return true; }
    entry.get("hooks").and_then(Value::as_array).is_some_and(|arr| {
        arr.iter().any(|h| h.get("command").and_then(Value::as_str) == Some("refit hook"))
    })
}

fn merge_nested_agent_hook(path: &Path, event: &str, matcher: &str, dry_run: bool) -> Result<()> {
    let mut root = load_json_object(path)?;
    let hooks = ensure_object(&mut root, "hooks");
    let list = ensure_array(hooks, event);
    if !list.iter().any(has_refit_hook_command) {
        list.push(json!({
            "matcher": matcher,
            "hooks": [{ "type": "command", "command": "refit hook" }]
        }));
    }
    if !dry_run {
        let body = format!("{}\n", serde_json::to_string_pretty(&Value::Object(root))?);
        write_atomic(path, body.as_bytes())?;
    }
    Ok(())
}

fn merge_cursor_hooks(path: &Path, dry_run: bool) -> Result<()> {
    let mut root = load_json_object(path)?;
    root.entry("version".to_string()).or_insert_with(|| json!(1));
    let hooks = ensure_object(&mut root, "hooks");
    for event in ["beforeShellExecution", "beforeReadFile", "afterFileEdit"] {
        let list = ensure_array(hooks, event);
        if !list.iter().any(has_refit_hook_command) {
            list.push(json!({ "command": "refit hook" }));
        }
    }
    if !dry_run {
        let body = format!("{}\n", serde_json::to_string_pretty(&Value::Object(root))?);
        write_atomic(path, body.as_bytes())?;
    }
    Ok(())
}

/// One-shot repository onboarding: sustain guardrails, chained Git hooks, and AI agent hooks.
pub fn init_repo_integrations(
    repo_root: &Path, report: &AuditReport, dry_run: bool,
) -> Result<Vec<String>> {
    let mut written = write_sustain(repo_root, report, dry_run)?.files_written;
    if dry_run {
        written.extend(infra::planned_secret_hook_paths(repo_root));
    } else {
        written.extend(install_secret_hooks(repo_root)?);
    }
    merge_nested_agent_hook(
        &repo_root.join(".claude/settings.json"), "PreToolUse", "Bash|Write|Edit|MultiEdit|Read", dry_run,
    )?;
    merge_nested_agent_hook(
        &repo_root.join(".claude/settings.json"), "PostToolUse", "Write|Edit|MultiEdit", dry_run,
    )?;
    merge_cursor_hooks(&repo_root.join(".cursor/hooks.json"), dry_run)?;
    merge_nested_agent_hook(
        &repo_root.join(".gemini/settings.json"), "BeforeTool", "*", dry_run,
    )?;
    merge_nested_agent_hook(
        &repo_root.join(".gemini/settings.json"), "AfterTool", "write_file|replace|edit_file", dry_run,
    )?;
    written.extend([
        ".claude/settings.json".into(),
        ".cursor/hooks.json".into(),
        ".gemini/settings.json".into(),
    ]);
    if repo_root.join(".kiro").is_dir() {
        let kiro_rel = ".kiro/hooks/refit-kane-verify.kiro.hook";
        if !dry_run {
            let kiro_body = serde_json::to_string_pretty(&json!({
                "enabled": true, "name": "refit + Kane CLI Verify", "version": "1",
                "when": { "type": "fileEdited", "patterns": ["docs/**/*.html", "src/**/*.tsx", "src/**/*.ts", "README.md"] },
                "then": { "type": "askAgent", "prompt": "Run `refit check . --verify-docs --kane <spec>` to verify static safety and live browser proof before pushing." }
            }))?;
            write_atomic(&repo_root.join(kiro_rel), format!("{kiro_body}\n").as_bytes())?;
        }
        written.push(kiro_rel.into());
    }
    Ok(written)
}

/// Compile and lint an audit report with simplified prose rules and optional wordlist.
#[cfg(any(feature = "ste100", test))]
pub fn render_audit_prose(
    report: &AuditReport, dict_path: Option<&Path>,
) -> Result<(String, ste100::ProseReport)> {
    let dict = dict_path.map(ste100::WordList::load_from_file).transpose()?;
    let compiled = ste100::audit_md(report);
    let lint = ste100::lint_md(&compiled, dict.as_ref());
    Ok((compiled, lint))
}

/// Generate sustain guardrails normalized and linted with simplified prose rules.
#[cfg(any(feature = "ste100", test))]
pub fn write_sustain_prose(
    repo_root: &Path, report: &AuditReport, dry_run: bool, dict_path: Option<&Path>,
) -> Result<(SustainFiles, ste100::ProseReport)> {
    let dict = dict_path.map(ste100::WordList::load_from_file).transpose()?;
    let agents_doc = format!(
        concat!(
            "# AGENTS.md\n\n",
            "1. Keep `codehygiene` score above `0.85` on changed files.\n",
            "2. Keep circular imports at or below `{}`.\n",
            "3. Consolidate shared helpers instead of duplicate code.\n",
            "4. Remove hardcoded credentials, `*:*` wildcards, and root containers.\n",
            "5. Run `refit scan . --fast` and `refit audit .` before a PR.\n"
        ),
        report.scc_count
    );
    let normalized = ste100::simplify_verbs(&agents_doc);
    let lint = ste100::lint_md(&normalized, dict.as_ref());
    Ok((write_sustain_docs(repo_root, report, normalized, dry_run)?, lint))
}

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
    repo_root: &'a Path, repo_files: &'a BTreeSet<String>, repo_dirs: &'a BTreeSet<String>,
    known_symbols: &'a BTreeSet<String>, npm_scripts: Option<&'a BTreeSet<String>>,
}

impl DriftCtx<'_> {
    fn is_missing_path(&self, clean_path: &str) -> bool {
        !self.repo_files.contains(clean_path) && !self.repo_root.join(clean_path).exists()
    }

    fn check_path(
        &self, out: &mut Vec<DocDriftFinding>, file: &str, ln: usize, span: &str, msg: &str,
    ) -> bool {
        let Some(clean) = parse_repo_path_candidate(span, self.repo_dirs) else { return false };
        if self.is_missing_path(clean) {
            push_drift(out, file, ln, "doc_stale_path", clean.into(), msg.replace("{p}", clean));
        }
        true
    }

    fn check_sym(&self, out: &mut Vec<DocDriftFinding>, file: &str, ln: usize, sym: &str, msg: &str) {
        if !self.known_symbols.is_empty() && !is_ignored_symbol(sym)
            && !is_placeholder_text(sym) && !self.known_symbols.contains(sym)
        {
            push_drift(out, file, ln, "doc_stale_symbol", format!("{sym}()"), msg.replace("{s}", sym));
        }
    }
}

fn push_drift(
    out: &mut Vec<DocDriftFinding>, file: &str, line: usize, kind: &str, reference: String, message: String,
) {
    out.push(DocDriftFinding { file: file.to_string(), line, kind: kind.into(), reference, message });
}

/// Fast standalone static doc-drift scan (used by `refit check --verify-docs`).
pub fn scan_doc_drift_quick(repo_root: &Path, warm_graph: Option<&SymbolGraph>) -> Vec<DocDriftFinding> {
    let repo_snap = RepoSnapshot::capture(repo_root, false);
    let owned;
    let snap = match warm_graph {
        Some(g) => g,
        None => { owned = index_repo_snap(repo_root, &repo_snap).1; &owned }
    };
    scan_doc_drift(repo_root, &repo_snap, snap)
}

/// Scan docs and comments for stale paths and symbol references.
fn scan_doc_drift(repo_root: &Path, repo_snap: &RepoSnapshot, snap: &SymbolGraph) -> Vec<DocDriftFinding> {
    let known_symbols: BTreeSet<String> = snap
        .defs_by_file
        .values()
        .chain(snap.refs_by_file.values())
        .flat_map(|syms| syms.iter().cloned())
        .collect();
    let (mut repo_files, mut repo_dirs) = (BTreeSet::new(), BTreeSet::new());
    for rel in &repo_snap.all_rels {
        repo_files.insert(rel.clone());
        if let Some((top, _)) = rel.split_once('/') { repo_dirs.insert(top.to_string()); }
    }
    let npm_scripts = load_package_scripts(repo_root);
    let ctx = DriftCtx {
        repo_root, repo_files: &repo_files, repo_dirs: &repo_dirs,
        known_symbols: &known_symbols, npm_scripts: npm_scripts.as_ref(),
    };

    let mut findings = Vec::new();
    for (rel, abs) in &repo_snap.doc_rels {
        if let Ok(content) = fs::read_to_string(abs) {
            scan_markdown_doc(&ctx, rel, &content, &mut findings);
        }
    }
    for rel in repo_snap.source_rels.iter().filter(|r| !test_path(r)) {
        if let Some(bytes) = read_bytes(repo_root, rel, Some(&repo_snap.preloaded)) {
            scan_source_comments(&ctx, rel, &String::from_utf8_lossy(&bytes), &mut findings);
        }
    }
    findings.sort_by(|a, b| {
        (&a.file, a.line, &a.kind, &a.reference).cmp(&(&b.file, b.line, &b.kind, &b.reference))
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
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") { in_fence = !in_fence; continue; }
        if in_fence || trimmed.is_empty() || contains_ignore_marker(trimmed) { continue; }
        for span in iter_backtick_spans(line) { check_doc_span(ctx, rel, idx + 1, span, out); }
    }
}

fn contains_ignore_marker(text: &str) -> bool { find_ascii_ci(text.as_bytes(), b"refit: ignore").is_some() }

fn check_doc_span(ctx: &DriftCtx<'_>, doc: &str, ln: usize, span: &str, out: &mut Vec<DocDriftFinding>) {
    if let Some(script) = extract_cli_script_ref(span) {
        if let Some(scripts) = ctx.npm_scripts && !scripts.contains(script) && !is_placeholder_text(script) {
            push_drift(out, doc, ln, "doc_stale_cli", span.into(),
                format!("Documented script `{script}` missing from package.json scripts"));
        }
        return;
    }
    if ctx.check_path(out, doc, ln, span, "Referenced file path `{p}` does not exist in repository") {
        return;
    }
    if let Some(sym) = BT_CALL_RE.captures(span).and_then(|c| c.get(1)).map(|m| m.as_str()) {
        ctx.check_sym(out, doc, ln, sym, "Referenced symbol `{s}()` not found in live codebase");
    }
}

fn extract_cli_script_ref(span: &str) -> Option<&str> {
    let rest = ["npm run ", "pnpm run ", "yarn run "].iter().find_map(|p| span.strip_prefix(p))?;
    let name = rest.split_whitespace().next()?;
    let ok = !name.is_empty() && !name.starts_with('-')
        && !has_placeholder_chars(name) && !BUILTIN_PKG_SCRIPTS.contains(&name);
    ok.then_some(name)
}

fn has_placeholder_chars(text: &str) -> bool {
    text.bytes().any(|b| b"*?<>{}[]$".contains(&b)) || text.contains("...")
}

fn is_placeholder_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    PLACEHOLDER_SEGMENTS.iter().any(|seg| lower.contains(seg))
}

fn parse_repo_path_candidate<'a>(span: &'a str, repo_dirs: &BTreeSet<String>) -> Option<&'a str> {
    let cap = REPO_PATH_RE.captures(span)?;
    let (clean, ext) = (cap.get(1)?.as_str(), cap.get(2)?.as_str());
    if clean.contains("..") || is_placeholder_text(clean)
        || RUNTIME_PREFIXES.iter().any(|p| clean.starts_with(p))
    { return None; }
    let (top, _) = clean.split_once('/')?;
    if !KNOWN_PATH_ROOTS.contains(&top) && !repo_dirs.contains(top) { return None; }
    KNOWN_FILE_EXTS.contains(&ext.to_ascii_lowercase().as_str()).then_some(clean)
}

fn is_ignored_symbol(sym: &str) -> bool { IGNORED_SYMBOLS.contains(&sym.to_ascii_lowercase().as_str()) }

fn check_comment_line(
    ctx: &DriftCtx<'_>, rel: &str, ln: usize, comment: &str, out: &mut Vec<DocDriftFinding>,
) {
    if comment.is_empty() || contains_ignore_marker(comment) { return; }
    for span in iter_backtick_spans(comment) {
        ctx.check_path(out, rel, ln, span, "Comment cites missing file `{p}`");
    }
    let bytes = comment.as_bytes();
    for cap in COMMENT_CALL_RE.captures_iter(comment) {
        let Some(mat) = cap.get(1) else { continue };
        let prev_bad = mat.start() > 0 && {
            let b = bytes[mat.start() - 1];
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.')
        };
        if !prev_bad {
            ctx.check_sym(out, rel, ln, mat.as_str(),
                "Stale comment reference `{s}()` not found in live codebase");
        }
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
            for (off, raw_line) in tok.as_str().lines().enumerate() {
                if let Some(cmt) = extract_comment_body("rs", raw_line) {
                    check_comment_line(ctx, rel, (tok.line as usize) + off, cmt, out);
                }
            }
        }
        return;
    }
    for (idx, line) in content.lines().enumerate() {
        if let Some(cmt) = extract_comment_body(&ext, line) {
            check_comment_line(ctx, rel, idx + 1, cmt, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn audit_infra_and_sustain() {
        let block = concat!(
            "    const normalized = raw.trim().toLowerCase();\n",
            "    const parts = normalized.split(':');\n",
            "    const checksum = parts.length * 31;\n",
            "    const output = checksum + normalized.length;\n",
        );
        let dir = TempRepo::with_files("unt_all", &[
            ("src/auth_v1.ts", &format!("export function authOne(raw: string) {{\n{block}    return output;\n}}\n")),
            ("src/auth_v2.ts", &format!("export function authTwo(raw: string) {{\n{block}    return output;\n}}\n")),
            ("src/app.py", "import os\n"),
            ("src/main.go", "package main\nfunc main() {}\n"),
            ("package.json", r#"{"dependencies":{"unused-ghost-pkg":"^1.0.0"}}"#),
            ("pyproject.toml", "[project]\ndependencies = [\"ghost-py-pkg>=1.0\"]\n[tool.ruff]\nline-length = 88\n"),
            ("go.mod", "module example.com/app\n\nrequire (\n\tgithub.com/unused/mod v1.0.0\n)\n"),
            ("infra/main.tf", "provider \"aws\" {}\nresource \"aws_iam_policy\" \"bad\" {\n  policy = \"{\\\"Action\\\": \\\"*:*\\\"}\"\n}\n"),
            ("Dockerfile", "FROM node:latest\nENV API_KEY=hardcoded\n"),
            (".github/workflows/ci.yml", "permissions: write-all\nsteps:\n  - uses: actions/checkout@main\n  - run: echo ${{ github.event.issue.title }}\n"),
        ]);

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
        assert!(!rep.triage_plan.is_empty() && render_audit(&rep).contains("refit Audit Report"));

        let sus = write_sustain(dir.path(), &rep, false).unwrap();
        assert_eq!(sus.files_written.len(), 4);
        for rel in ["AGENTS.md", ".refit/invariants.yml", "docs/ADR-001-untangle.md", ".github/workflows/refit-guard.yml"] {
            assert!(dir.has(rel), "missing {rel}");
        }

        dir.write("Dockerfile", "FROM alpine:3.20\nUSER rootless\n");
        let (_, _, f2) = scan_infra(dir.path());
        assert!(!f2.iter().any(|f| f.rule_id == "docker_runs_as_root"));

        let (ste_md, ste_lint) = render_audit_prose(&rep, None).unwrap();
        assert!(ste_md.contains("# refit Audit Report") && ste_lint.compliant);
        let (sus_ste, sus_lint) = write_sustain_prose(dir.path(), &rep, true, None).unwrap();
        assert!(sus_ste.files_written.len() == 4 && sus_lint.compliant);
    }

    #[test]
    fn drift_skips_test_catches_prod() {
        let auth_rs = concat!(
            "// Delegates to live_verify_token() safely\n",
            "// Legacy note: previously called legacy_parse_wal() and `src/ghost_prod.rs`\n",
            "pub fn live_verify_token(raw: &str) -> bool { !raw.is_empty() }\n",
            "#[cfg(test)]\nmod tests {\n",
            "    // Comment inside cfg(test) citing fake_test_only_fn() and `src/ghost_test.rs`\n",
            "    #[test]\n    fn ok() { assert!(super::live_verify_token(\"x\")); }\n}\n",
        );
        let readme = concat!(
            "# Demo\nLive `src/auth.rs` calls `live_verify_token()`.\n",
            "Dead `src/deleted_legacy.rs`, `old_broken_helper()`, `npm run ghost-task`.\n",
            "Ignore `src/example/foo.rs`, `to_string()`, and live `npm run build`.\n",
        );
        let dir = TempRepo::with_files("doc_drift_cfg", &[
            ("src/auth.rs", auth_rs),
            ("tests/smoke_test.rs", "// Mentions test_only_ghost_fn() in test path\n"),
            ("README.md", readme),
            ("package.json", r#"{"scripts":{"build":"vite build"}}"#),
        ]);

        let rep = untangle_repo(dir.path(), &HygieneScorer::new()).unwrap();
        for (file, kind, r) in [
            ("src/auth.rs", "doc_stale_symbol", "legacy_parse_wal()"),
            ("src/auth.rs", "doc_stale_path", "src/ghost_prod.rs"),
            ("README.md", "doc_stale_path", "src/deleted_legacy.rs"),
            ("README.md", "doc_stale_symbol", "old_broken_helper()"),
            ("README.md", "doc_stale_cli", "npm run ghost-task"),
        ] {
            assert!(
                rep.doc_drift_findings.iter().any(|f| f.file == file && f.kind == kind && f.reference == r),
                "missing ({file}, {kind}, {r}) in {:?}", rep.doc_drift_findings
            );
        }
        for bad in [
            "fake_test_only_fn", "ghost_test.rs", "test_only_ghost_fn",
            "live_verify_token", "example", "to_string", "npm run build",
        ] {
            assert!(
                !rep.doc_drift_findings.iter().any(|f| f.reference.contains(bad)),
                "false positive `{bad}` in {:?}", rep.doc_drift_findings
            );
        }
    }

    #[test]
    fn init_repo_integrations_merges_idempotently() {
        let dir = TempRepo::git("init_int", "app", &[
            ("src/lib.rs", "pub fn ok() -> usize { 1 }\n"),
            (".claude/settings.json", r#"{"customKey":true,"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"echo hi"}]}]}}"#),
            (".cursor/hooks.json", r#"{"version":1,"hooks":{"beforeShellExecution":[{"command":"echo cur"}]}}"#),
            (".gemini/settings.json", r#"{"theme":"dark"}"#),
        ]);
        let rep = untangle_repo(dir.path(), &HygieneScorer::new()).unwrap();

        let dry = init_repo_integrations(dir.path(), &rep, true).unwrap();
        assert!(dry.iter().any(|p| p == ".claude/settings.json"));
        assert!(!dir.has("AGENTS.md"));

        let first = init_repo_integrations(dir.path(), &rep, false).unwrap();
        assert!(first.len() >= 9);
        let _second = init_repo_integrations(dir.path(), &rep, false).unwrap();

        let claude: Value = serde_json::from_str(&fs::read_to_string(dir.join(".claude/settings.json")).unwrap()).unwrap();
        assert_eq!(claude["customKey"], true);
        let pre = claude["hooks"]["PreToolUse"].as_array().unwrap();
        assert_eq!(pre.len(), 2);

        let cursor: Value = serde_json::from_str(&fs::read_to_string(dir.join(".cursor/hooks.json")).unwrap()).unwrap();
        assert_eq!(cursor["hooks"]["beforeShellExecution"].as_array().unwrap().len(), 2);
        assert_eq!(cursor["hooks"]["beforeReadFile"].as_array().unwrap().len(), 1);
        assert_eq!(cursor["hooks"]["afterFileEdit"].as_array().unwrap().len(), 1);

        let gemini: Value = serde_json::from_str(&fs::read_to_string(dir.join(".gemini/settings.json")).unwrap()).unwrap();
        assert_eq!(gemini["theme"], "dark");
        assert_eq!(gemini["hooks"]["BeforeTool"].as_array().unwrap().len(), 1);

        let sec_rep = scan_repo_secrets(dir.path(), false, None, true).unwrap();
        assert_eq!(render_secrets_sarif(&sec_rep)["version"], "2.1.0");
        assert!(render_secrets_gh(&sec_rep).is_empty());
        assert_eq!(run_agent_hook("").0, 0);
    }
}
