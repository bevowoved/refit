mod gates;
mod graph;
mod hygiene;
mod learn;
mod lex;
mod mcp;
mod mvcc;
mod net;
mod platform;
mod proxy;
mod research;
mod runner;
mod source;
mod state;
mod stem;
mod telemetry;
#[cfg(test)]
mod testutil;
mod types;
mod untangle;
mod worktree;

#[cfg(all(feature = "alloc-mimalloc", feature = "alloc-jemalloc"))]
compile_error!("features `alloc-mimalloc` and `alloc-jemalloc` are mutually exclusive");

#[cfg(feature = "alloc-mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc")))]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static MALLOC_CONF: &[u8] = b"dirty_decay_ms:50,muzzy_decay_ms:50,narenas:2\0";

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use runner::DEFAULT_UPSTREAM;
use serde::Serialize;
use serde_json::json;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
    Md,
    Gh,
    Sarif,
}

#[derive(Parser)]
#[command(
    name = "refit",
    version = env!("CARGO_PKG_VERSION"),
    about = "Cross-platform code audit, verification, and parallel refactor engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Probe the Cerebras endpoint through the proxy and print latency metrics.
    Probe {
        #[arg(long, default_value = DEFAULT_UPSTREAM)] upstream: String,
        #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")] model: String,
        #[arg(long)] api_key_file: Option<PathBuf>,
        #[arg(long)] openclaw_url: Option<String>,
        #[arg(long)] openclaw_token: Option<String>,
    },
    /// Scan the repository DAG, symbol graph, and hygiene scores.
    Scan {
        repo: PathBuf,
        #[arg(long)] no_cache: bool,
        #[arg(long)] fast: bool,
        #[arg(long)] approx_long: bool,
    },
    /// Audit clones, dead symbols, large files, unused dependencies, and infra configs.
    Audit {
        repo: PathBuf,
        #[arg(long)] markdown: bool,
        #[arg(long)] research: bool,
        #[arg(long)] ste100: bool,
        #[arg(long)] ste_dict: Option<PathBuf>,
        #[arg(long)] approx_long: bool,
    },
    /// One-command repo setup: AGENTS.md, invariants, worktree-safe Git hooks, and agent hooks.
    Init {
        repo: PathBuf,
        #[arg(long)] dry_run: bool,
    },
    /// Generate AGENTS.md, .refit/invariants.yml, ADR-001, and the CI guard workflow.
    Sustain {
        repo: PathBuf,
        #[arg(long)] dry_run: bool,
        #[arg(long)] init: bool,
        #[arg(long)] ste100: bool,
        #[arg(long)] ste_dict: Option<PathBuf>,
    },
    /// Prune unreferenced CAS blobs, expired stage directories, and bounded disk caches.
    Gc {
        repo: PathBuf,
        #[arg(long, default_value_t = 32)] keep_revs: usize,
        #[arg(long, default_value_t = 600)] cache_max_mb: u64,
    },
    /// Inspect persisted learned failure rules and run SymbolGraph-backed rewind impact queries.
    Learn {
        repo: PathBuf,
        #[arg(long)] rewind: Option<String>,
        #[arg(long, value_enum, default_value_t = learn::RewindTrigger::NewSmellPattern)]
        trigger: learn::RewindTrigger,
    },
    /// Run parallel DAG refactoring with formatters, Gates 0 to 3.5, and MVCC WAL.
    Run {
        #[command(flatten)] opts: runner::RunOptions,
        #[arg(long)] resume: bool,
    },
    /// Resume an interrupted run with the same flags as `refit run`.
    Resume { #[command(flatten)] opts: runner::RunOptions },
    /// Run alternating OS CPU and RSS benchmarks.
    Bench {
        #[arg(long)] before: PathBuf,
        #[arg(long)] after: PathBuf,
        #[arg(long)] script: Option<PathBuf>,
        #[arg(long)] cmd: Option<String>,
        #[arg(long, default_value_t = 5)] pairs: usize,
    },
    /// Start the HTTP/2 Cerebras proxy and live HUD server.
    Proxy {
        #[arg(long, default_value = "127.0.0.1:18790")] listen: String,
        #[arg(long, default_value = DEFAULT_UPSTREAM)] upstream: String,
        #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")] model: String,
        #[arg(long, default_value = "docs/index.html")] ui_index: PathBuf,
        #[arg(long, default_value = "telemetry/proxy.jsonl")] telemetry_out: PathBuf,
        #[arg(long)] repo: Option<PathBuf>,
        #[arg(long)] auth_token: Option<String>,
        #[arg(long)] api_key_file: Option<PathBuf>,
        #[arg(long)] cache_retention: Option<String>,
        #[arg(long)] keep_warm: bool,
    },
    /// Start the concurrent JSON-RPC 2.0 MCP server (stdio, or stateless HTTP via --http).
    Mcp { repo: PathBuf, #[arg(long)] http: Option<String> },
    /// Unified pre-flight check: tiered secrets, invisible Unicode, optional OSV.dev SCA, doc-drift & Gate 4 Kane CLI.
    Check {
        repo: PathBuf,
        #[arg(long)] staged: bool,
        #[arg(long)] diff: Option<String>,
        #[arg(long)] deps: bool,
        #[arg(long)] strict: bool,
        #[arg(long)] no_cache: bool,
        #[arg(long)] kane: Option<String>,
        #[arg(long)] url: Option<String>,
        #[arg(long)] policy: Option<String>,
        #[arg(long)] target: Option<String>,
        #[arg(long)] app: Option<String>,
        #[arg(long)] remote: bool,
        #[arg(long)] headed: bool,
        #[arg(long)] bug_detection: bool,
        #[arg(long)] no_adaptive_heal: bool,
        #[arg(long, default_value_t = 180)] timeout: u64,
        #[arg(long)] verify_docs: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)] format: OutputFormat,
    },
    /// Native <5ms stdin-JSON agent hook for Claude Code, Cursor, and Gemini CLI.
    Hook,
    /// Scan repo, staged index, or git diff for secrets (or install pre-commit/pre-push hooks).
    Secrets {
        repo: PathBuf,
        #[arg(long)] staged: bool,
        #[arg(long)] diff: Option<String>,
        #[arg(long)] install_hooks: bool,
        #[arg(long)] no_cache: bool,
        #[arg(long)] strict: bool,
        #[arg(long)] sarif: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)] format: OutputFormat,
    },
    /// Run batch web research searches.
    Research {
        #[arg(long)] lang: Option<String>,
        #[arg(long)] repo: Option<PathBuf>,
        #[arg(long, default_value_t = 5)] max_results: usize,
        #[arg(required = true)] queries: Vec<String>,
    },
}

fn print_json<T: Serialize>(val: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(val)?);
    Ok(())
}

fn read_key(file_opt: Option<&PathBuf>) -> Result<String> {
    if let Some(env_k) = platform::env_trimmed("CEREBRAS_API_KEY") {
        return Ok(env_k);
    }

    if let Some(p) = file_opt {
        let raw = std::fs::read_to_string(p)
            .with_context(|| format!("read api key file {}", p.display()))?;
        if !raw.trim().is_empty() {
            return Ok(raw.trim().to_string());
        }
    }
    anyhow::bail!("CEREBRAS_API_KEY env var is not set and --api-key-file was not provided");
}

fn cmd_run(mut opts: runner::RunOptions, resume: bool) -> Result<()> {
    init_tracing();
    opts.api_key = read_key(opts.api_key_file.as_ref())?;
    opts.resume = resume;
    print_json(&build_rt(true)?.block_on(runner::execute_run(opts))?)
}

fn cmd_scan(repo: &Path, no_cache: bool, fast: bool, approx_long: bool) -> Result<()> {
    let mut scorer = hygiene::HygieneScorer::open_persistent(repo);
    scorer.set_approx_long(approx_long);
    let started = Instant::now();
    let (bundle, hit) = gates::scan_cached(repo, !no_cache, fast, &scorer)?;
    let style = hygiene::repo_style(repo);
    print_json(&json!({
        "version": env!("CARGO_PKG_VERSION"), "allocator": platform::allocator_name(),
        "pure_rust_hygiene": "codehygiene_exact", "scan_cache_hit": hit, "fast_mode": fast,
        "repo_fingerprint": bundle.repo_fingerprint, "scan_ms": started.elapsed().as_millis() as u64,
        "detected_languages": bundle.detected_languages, "polyglot_toolchains": bundle.polyglot_toolchains,
        "style_config": { "max_line_width": style.max_line_width, "indent_width": style.indent_width },
        "host_os": bundle.plan.host_os, "stem_used": bundle.plan.stem_used,
        "stem_index_ms": bundle.plan.stem_index_ms, "warm_hygiene_us": scorer.warm_us(),
        "total_files": bundle.plan.total_files, "total_loc": bundle.plan.total_loc,
        "units": bundle.plan.units.len(), "scc_count": bundle.plan.scc_count,
        "wave_count": bundle.plan.wave_count, "tests_total": bundle.audit.tests_total,
        "tests_failed": bundle.audit.tests_failed, "ruff_errors": bundle.audit.ruff_errors,
        "ty_errors": bundle.audit.ty_errors, "hygiene_median": bundle.audit.hygiene_median,
        "hygiene_p10": bundle.audit.hygiene_p10,
        "engine_rss_mb": (platform::self_metrics().rss_bytes as f64) / (1024.0 * 1024.0)
    }))
}

#[cfg(not(feature = "ste100"))]
fn require_ste100() -> Result<()> {
    anyhow::bail!("STE100 linting (`--ste100` / `--ste-dict`) requires `--features ste100`")
}

fn append_dep_research(md: &mut String, snippets: &str) {
    if !snippets.is_empty() {
        md.push_str("## Dependency Research\n\n");
        md.push_str(snippets);
    }
}

fn emit_audit_output(
    report: &types::AuditReport, dep_res: &[research::SearchResult],
    markdown: bool, ste100: bool, ste_dict: Option<&Path>,
) -> Result<()> {
    if ste100 || ste_dict.is_some() {
        return emit_audit_prose(report, dep_res, markdown, ste_dict);
    }
    if markdown {
        let mut md = untangle::render_audit(report);
        if !dep_res.is_empty() {
            append_dep_research(&mut md, &research::WebClient::format_snippets(dep_res));
        }
        println!("{md}");
        return Ok(());
    }
    print_json(&json!({ "audit": report, "dependency_research": dep_res }))
}

#[cfg(feature = "ste100")]
fn emit_audit_prose(
    report: &types::AuditReport,
    dep_res: &[research::SearchResult],
    markdown: bool,
    ste_dict: Option<&Path>,
) -> Result<()> {
    let (mut compiled, lint) = untangle::render_audit_prose(report, ste_dict)?;
    if markdown {
        if !dep_res.is_empty() {
            append_dep_research(&mut compiled, &research::WebClient::format_prose(dep_res));
        }
        println!("{compiled}");
        return Ok(());
    }
    print_json(&json!({
        "audit": report,
        "dependency_research": dep_res,
        "ste100": lint
    }))
}

#[cfg(not(feature = "ste100"))]
fn emit_audit_prose(
    _: &types::AuditReport, _: &[research::SearchResult], _: bool, _: Option<&Path>,
) -> Result<()> {
    require_ste100()
}

async fn cmd_probe(
    upstream: String, model: String, api_key_file: Option<PathBuf>,
    openclaw_url: Option<String>, openclaw_token: Option<String>,
) -> Result<()> {
    let (key, addr) = (read_key(api_key_file.as_ref())?, "127.0.0.1:18799".to_string());
    let tel = telemetry::TelemetryHub::new(std::env::temp_dir().join("refit-probe.jsonl"))?;
    let cfg = proxy::ProxyConfig::new(addr.clone(), upstream, key, model.clone());
    let (pstate, handle) = proxy::start_proxy(cfg, tel).await?;
    let client = platform::new_http(concat!("refit-probe/", env!("CARGO_PKG_VERSION")), 120, 4)?;
    let url = openclaw_url.as_ref().map_or_else(
        || format!("http://{addr}/v1/chat/completions"),
        |oc| format!("{}/chat/completions", oc.trim_end_matches('/')),
    );
    let body_req = json!({
        "model": model, "stream": false, "max_completion_tokens": 32,
        "reasoning_effort": "low", "reasoning_format": "parsed", "temperature": 1.0,
        "messages": [{"role": "system", "content": "Reply OK."}, {"role": "user", "content": "OK"}]
    });
    let mut req = client
        .post(&url)
        .header("x-refit-module", "probe")
        .header("x-refit-attempt", "1")
        .json(&body_req);
    if let Some(tok) = openclaw_token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await?;
    let code = resp.status().as_u16();
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    print_json(&json!({
        "http_status": code, "via_openclaw": openclaw_url.is_some(),
        "content": body["choices"][0]["message"]["content"],
        "metrics": pstate.take_metrics("probe"), "proxy_snapshot": pstate.snapshot()
    }))?;
    handle.abort();
    Ok(())
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn build_rt(multi: bool) -> Result<tokio::runtime::Runtime> {
    let b = platform::budget();
    let mut builder = if multi && !b.is_constrained() {
        let mut mt = tokio::runtime::Builder::new_multi_thread();
        mt.worker_threads(b.effective_vcpus.clamp(1, 2));
        mt
    } else {
        tokio::runtime::Builder::new_current_thread()
    };
    builder.enable_all().build().context("build tokio runtime")
}

fn cmd_audit_sync(
    repo: &Path, markdown: bool, do_research: bool, ste100: bool, ste_dict: Option<&Path>, approx_long: bool,
) -> Result<()> {
    let mut scorer = hygiene::HygieneScorer::open_persistent(repo);
    scorer.set_approx_long(approx_long);
    let mut report = untangle::untangle_repo(repo, &scorer)?;
    let dep_res = if do_research {
        let (advisories, res) = build_rt(false)?
            .block_on(research::WebClient::for_repo(repo, None).audit_deps_full(repo));
        report.dep_advisories = advisories;
        res
    } else {
        Vec::new()
    };
    platform::collect_os_memory();
    emit_audit_output(&report, &dep_res, markdown, ste100, ste_dict)
}

fn cmd_sustain(
    repo: &Path, dry_run: bool, init: bool, ste100: bool, ste_dict: Option<&Path>,
) -> Result<()> {
    let scorer = hygiene::HygieneScorer::open_persistent(repo);
    let rep = untangle::untangle_repo_with_scope(repo, &scorer, untangle::AuditScope::Sustain)?;
    if init {
        let written = untangle::init_repo_integrations(repo, &rep, dry_run)?;
        platform::collect_os_memory();
        return print_json(&json!({ "dry_run": dry_run, "files_written": written }));
    }
    if ste100 || ste_dict.is_some() {
        return emit_sustain_prose(repo, &rep, dry_run, ste_dict);
    }
    let out = untangle::write_sustain(repo, &rep, dry_run)?;
    platform::collect_os_memory();
    print_json(&out)
}

#[cfg(feature = "ste100")]
fn emit_sustain_prose(
    repo: &Path, rep: &types::AuditReport, dry_run: bool, ste_dict: Option<&Path>,
) -> Result<()> {
    let (out, lint) = untangle::write_sustain_prose(repo, rep, dry_run, ste_dict)?;
    platform::collect_os_memory();
    print_json(&json!({ "sustain": out, "ste100": lint }))
}

#[cfg(not(feature = "ste100"))]
fn emit_sustain_prose(_: &Path, _: &types::AuditReport, _: bool, _: Option<&Path>) -> Result<()> {
    require_ste100()
}

fn cmd_mcp(repo: PathBuf, http: Option<String>) -> Result<()> {
    init_tracing();
    match http {
        Some(addr) => build_rt(false)?.block_on(mcp::serve_http(repo, addr)),
        None => build_rt(false)?.block_on(mcp::serve_mcp(repo)),
    }
}

struct CheckOpts<'a> {
    staged: bool,
    diff: Option<&'a str>,
    deps: bool,
    strict: bool,
    no_cache: bool,
    verify_docs: bool,
    kane: Option<gates::KaneRequest>,
    format: OutputFormat,
}

fn render_check_human(
    rep: &untangle::SecretScanReport, osv: &[research::OsvFinding],
    doc_drift: &[types::DocDriftFinding], kane: Option<&gates::KaneReceipt>,
) -> String {
    let mut out = String::new();
    for hit in &rep.findings {
        let lvl = if hit.tier == "high" { "error" } else { "warning" };
        out.push_str(&format!(
            "{lvl}[refit:{}]: {}:{}\n  help: remove credential/hazard or suppress false positives with `refit:ignore`\n",
            hit.kind, hit.file, hit.line.max(1)
        ));
    }
    for v in osv {
        out.push_str(&format!(
            "error[refit:osv:{}]: {} ({} {}@{}) {}\n",
            v.advisory_id, v.lockfile, v.ecosystem, v.package, v.version, v.summary
        ));
    }
    for d in doc_drift {
        out.push_str(&format!(
            "warning[refit:{}]: {}:{} — {} ({})\n",
            d.kind, d.file, d.line.max(1), d.message, d.reference
        ));
    }
    if let Some(k) = kane {
        let lvl = if k.ok { "ok" } else { "error" };
        out.push_str(&format!(
            "{lvl}[refit:kane:{}]: {} ({:.2}s, replay={}, author={}, credits={})\n",
            k.verdict, k.summary, k.duration_s, k.replay_steps, k.author_steps,
            k.credits_consumed.map_or_else(|| "0".into(), |c| format!("{c:.2}"))
        ));
        for f in &k.failed_steps {
            let tag = if f.is_agent_info { "AGENT_INFO" } else { "step" };
            out.push_str(&format!("  {tag} #{}: {}\n", f.step, f.remark));
        }
        for lk in &k.evidence_secret_leaks {
            out.push_str(&format!("  error[refit:kane/evidence_secret_leak]: {lk}\n"));
        }
        if let Some(cmd) = &k.evidence_serve_cmd {
            out.push_str(&format!("  evidence: {cmd}\n"));
        }
        if !k.suspected_spans.is_empty() {
            let spans: Vec<String> = k.suspected_spans.iter()
                .map(|s| format!("{}:{} ({})", s.file, s.line, s.symbol)).collect();
            out.push_str(&format!("  rewind: {}\n", spans.join(", ")));
        }
    }
    out.push_str(&format!(
        "refit check ({}): {} files ({} cached), {} blocking, {} warnings, {} osv, {} doc_drift in {}ms\n",
        rep.mode, rep.files_checked, rep.cache_hits, rep.blocking_count, rep.warning_count,
        osv.len(), doc_drift.len(), rep.duration_ms
    ));
    out
}

fn render_check_md(
    rep: &untangle::SecretScanReport, osv: &[research::OsvFinding],
    doc_drift: &[types::DocDriftFinding], kane: Option<&gates::KaneReceipt>,
) -> String {
    let mut out = format!(
        "## refit check ({})\n\n- Files: {} ({} cached) | Blocking: {} | Warnings: {} | OSV: {} | DocDrift: {} | {}ms\n",
        rep.mode, rep.files_checked, rep.cache_hits, rep.blocking_count, rep.warning_count,
        osv.len(), doc_drift.len(), rep.duration_ms
    );
    if !rep.findings.is_empty() {
        out.push_str("\n| Tier | Rule | Location |\n|---|---|---|\n");
        for h in &rep.findings {
            out.push_str(&format!("| {} | `{}` | `{}:{}` |\n", h.tier, h.kind, h.file, h.line.max(1)));
        }
    }
    if !osv.is_empty() {
        out.push_str("\n| Advisory | Package | Lockfile |\n|---|---|---|\n");
        for v in osv {
            out.push_str(&format!("| `{}` | `{}@{}` | `{}` |\n", v.advisory_id, v.package, v.version, v.lockfile));
        }
    }
    if !doc_drift.is_empty() {
        out.push_str("\n### Doc Drift\n\n| Kind | Location | Reference |\n|---|---|---|\n");
        for d in doc_drift {
            out.push_str(&format!("| `{}` | `{}:{}` | `{}` |\n", d.kind, d.file, d.line.max(1), d.reference));
        }
    }
    if let Some(k) = kane {
        out.push_str("\n### Gate 4 — Kane CLI\n\n| Surface | Verdict | Duration | Credits | Replay/Author | Evidence |\n|---|---|---:|---:|---|---|\n");
        out.push_str(&format!(
            "| `{:?}` | `{}` | {:.2}s | {} | {}/{} | `{}` |\n",
            k.surface, k.verdict, k.duration_s,
            k.credits_consumed.map_or_else(|| "0".into(), |c| format!("{c:.2}")),
            k.replay_steps, k.author_steps,
            k.evidence_serve_cmd.as_deref().unwrap_or("-")
        ));
    }
    out
}

fn append_extra_sarif(
    sarif: &mut serde_json::Value, osv: &[research::OsvFinding],
    doc_drift: &[types::DocDriftFinding], kane: Option<&gates::KaneReceipt>,
) {
    for v in osv {
        let msg = format!("{} {}@{}: {}", v.ecosystem, v.package, v.version, v.summary);
        untangle::push_sarif_result(sarif, &v.advisory_id, "error", &v.lockfile, 1, &msg);
    }
    for d in doc_drift {
        let msg = format!("{} (`{}`)", d.message, d.reference);
        untangle::push_sarif_result(sarif, &d.kind, "warning", &d.file, d.line, &msg);
    }
    if let Some(k) = kane {
        let (uri, ln) = k.suspected_spans.first()
            .map_or(("README.md", 1usize), |s| (s.file.as_str(), s.line));
        if !k.ok {
            let rule_id = match k.verdict.as_str() {
                "bug_740" => "kane/product_bug_740",
                "policy_failed" => "kane/policy_fail",
                "policy_errored" => "kane/policy_errored",
                _ => "kane/step_failure",
            };
            let level = gates::KANE_SARIF_RULES.iter()
                .find(|(id, _, _)| *id == rule_id).map_or("error", |(_, l, _)| *l);
            untangle::push_sarif_result(sarif, rule_id, level, uri, ln, &k.summary);
        }
        for lk in &k.evidence_secret_leaks {
            untangle::push_sarif_result(sarif, "kane/evidence_secret_leak", "error", uri, 1, lk);
        }
        for uv in &k.unresolved_vars {
            untangle::push_sarif_result(sarif, "kane/unresolved_vars", "warning", uri, 1, &format!("unresolved Kane variable `{uv}`"));
        }
        for adv in &k.advisories {
            untangle::push_sarif_result(sarif, "kane/schema_version_drift", "note", uri, 1, adv);
        }
    }
}

fn emit_check_output(
    rep: &untangle::SecretScanReport, osv: &[research::OsvFinding],
    doc_drift: &[types::DocDriftFinding], kane: Option<&gates::KaneReceipt>,
    format: OutputFormat, include_osv: bool,
) -> Result<()> {
    match format {
        OutputFormat::Sarif => {
            let mut sarif = untangle::render_secrets_sarif(rep);
            append_extra_sarif(&mut sarif, osv, doc_drift, kane);
            print_json(&sarif)
        }
        OutputFormat::Gh => {
            print!("{}", untangle::render_secrets_gh(rep));
            for v in osv {
                let f = untangle::escape_gh_prop(&v.lockfile);
                println!(
                    "::error file={f},line=1::[refit:osv:{}] {}@{} {}",
                    v.advisory_id, v.package, v.version, v.summary
                );
            }
            for d in doc_drift {
                let f = untangle::escape_gh_prop(&d.file);
                println!(
                    "::warning file={f},line={}::[refit:{}] {} ({})",
                    d.line.max(1), d.kind, d.message, d.reference
                );
            }
            if let Some(k) = kane {
                let (uri, ln) = k.suspected_spans.first()
                    .map_or(("README.md", 1usize), |s| (s.file.as_str(), s.line));
                let f = untangle::escape_gh_prop(uri);
                if !k.ok {
                    println!("::error file={f},line={ln}::[refit:kane:{}] {}", k.verdict, k.summary);
                }
                for lk in &k.evidence_secret_leaks {
                    println!("::error file={f},line=1::[refit:kane/evidence_secret_leak] {lk}");
                }
            }
            Ok(())
        }
        OutputFormat::Human => { print!("{}", render_check_human(rep, osv, doc_drift, kane)); Ok(()) }
        OutputFormat::Md => { print!("{}", render_check_md(rep, osv, doc_drift, kane)); Ok(()) }
        OutputFormat::Json => {
            if include_osv {
                let kane_fail = usize::from(kane.is_some_and(|k| !k.ok));
                let blocking = rep.blocking_count + osv.len() + kane_fail;
                let mut body = json!({
                    "ok": blocking == 0, "blocking_count": blocking, "warning_count": rep.warning_count,
                    "secrets": rep, "osv_findings": osv
                });
                if let Some(obj) = body.as_object_mut() {
                    if !doc_drift.is_empty() { obj.insert("doc_drift".into(), json!(doc_drift)); }
                    if let Some(k) = kane { obj.insert("kane".into(), json!(k)); }
                }
                print_json(&body)
            } else {
                print_json(rep)
            }
        }
    }
}

fn short_circuit_receipt(repo: &Path, req: &gates::KaneRequest, verdict: &str, exit_code: i32, summary: String) -> gates::KaneReceipt {
    gates::KaneReceipt {
        ok: false, verdict: verdict.into(), surface: req.surface.unwrap_or(gates::KaneSurface::Run),
        exit_code, cli_version: None, duration_s: 0.0, credits_consumed: Some(0.0),
        replay_steps: 0, author_steps: 0, adaptive_healed: false, confirmed_bug_740: false,
        summary, test_url: None, evidence_path: None, evidence_serve_cmd: None,
        failed_steps: Vec::new(), evidence_excerpts: Vec::new(), evidence_secret_leaks: Vec::new(),
        unresolved_vars: Vec::new(), policy_verdict: None, final_state: None,
        suspected_spans: Vec::new(), rewind_affected_files: Vec::new(), advisories: Vec::new(),
        git_head: platform::read_git_head_fast(repo), recorded_ms: platform::now_ms(),
    }
}

fn enrich_kane_rewind(repo: &Path, receipt: &mut gates::KaneReceipt) {
    if receipt.ok { return; }
    let mut diag = receipt.summary.clone();
    for f in &receipt.failed_steps { diag.push(' '); diag.push_str(&f.remark); }
    for e in &receipt.evidence_excerpts { diag.push(' '); diag.push_str(e); }
    let (spans, affected) = learn::rewind_kane_tokens(repo, None, &diag);
    receipt.suspected_spans = spans.into_iter()
        .map(|(file, line, symbol)| gates::CodeSpanHint { file, line, symbol }).collect();
    receipt.rewind_affected_files = affected;
    if let Ok(raw) = serde_json::to_vec_pretty(receipt) {
        let _ = platform::write_atomic(&repo.join(".refit/kane_last.json"), &raw);
    }
}

fn cmd_check(repo: &Path, opts: &CheckOpts<'_>) -> Result<()> {
    let rep = untangle::scan_repo_secrets(repo, opts.staged, opts.diff, opts.no_cache)?;
    let osv = if opts.deps { build_rt(false)?.block_on(research::scan_lockfiles_osv(repo))? } else { Vec::new() };
    let doc_drift = if opts.verify_docs { untangle::scan_doc_drift_quick(repo, None) } else { Vec::new() };
    let static_blocked = rep.blocking_count + osv.len()
        + if opts.strict { rep.warning_count + doc_drift.len() } else { 0 };
    let kane = if let Some(req) = &opts.kane {
        if static_blocked > 0 {
            Some(short_circuit_receipt(
                repo, req, "short_circuited", 1,
                "skipped Kane CLI because static pre-flight blocked (0 credits spent)".into(),
            ))
        } else if let Some(u) = req.url.as_deref().filter(|s| !s.is_empty())
            && let Err(msg) = build_rt(false)?.block_on(gates::probe_url_reachable(u))
        {
            Some(short_circuit_receipt(repo, req, "did_not_start", 2, format!("URL preflight failed (0 credits spent): {msg}")))
        } else {
            let mut rec = gates::run_kane(repo, req);
            enrich_kane_rewind(repo, &mut rec);
            Some(rec)
        }
    } else {
        None
    };
    emit_check_output(&rep, &osv, &doc_drift, kane.as_ref(), opts.format, true)?;
    let kane_blocked = usize::from(kane.as_ref().is_some_and(|k| !k.ok));
    let blocked = static_blocked + kane_blocked;
    if blocked > 0 { anyhow::bail!("blocked: {blocked} finding(s) detected"); }
    Ok(())
}

fn cmd_hook() -> Result<()> {
    let mut stdin_buf = String::new();
    std::io::stdin().read_to_string(&mut stdin_buf)?;
    let (code, out, err) = untangle::run_agent_hook(&stdin_buf);
    if !out.is_empty() { println!("{out}"); }
    if !err.is_empty() { eprintln!("{err}"); }
    if code != 0 { std::process::exit(code); }
    Ok(())
}

fn cmd_secrets(
    repo: &Path, staged: bool, diff: Option<&str>, install_hooks: bool, no_cache: bool,
    strict: bool, format: OutputFormat,
) -> Result<()> {
    if install_hooks {
        return print_json(&json!({ "installed_hooks": untangle::install_secret_hooks(repo)? }));
    }
    let rep = untangle::scan_repo_secrets(repo, staged, diff, no_cache)?;
    emit_check_output(&rep, &[], &[], None, format, false)?;
    let blocked = rep.blocking_count + if strict { rep.warning_count } else { 0 };
    if blocked > 0 { anyhow::bail!("blocked: {blocked} secret(s) detected"); }
    Ok(())
}

fn main() -> Result<()> {
    platform::init_allocator(platform::budget());
    match Cli::parse().command {
        Commands::Scan { repo, no_cache, fast, approx_long } => {
            let res = cmd_scan(&repo, no_cache, fast, approx_long);
            platform::collect_os_memory();
            res
        }
        Commands::Audit { repo, markdown, research, ste100, ste_dict, approx_long } => {
            cmd_audit_sync(&repo, markdown, research, ste100, ste_dict.as_deref(), approx_long)
        }
        Commands::Init { repo, dry_run } => cmd_sustain(&repo, dry_run, true, false, None),
        Commands::Sustain { repo, dry_run, init, ste100, ste_dict } => {
            cmd_sustain(&repo, dry_run, init, ste100, ste_dict.as_deref())
        }
        Commands::Gc { repo, keep_revs, cache_max_mb } => {
            let pruned_cas_blobs = mvcc::MvccStore::open(&repo, "gc")?.gc_cas(keep_revs)?;
            let swept_orphan_stages = gates::sweep_stages(&repo, 0);
            let cache_gc = gates::prune_repo_caches(&repo, cache_max_mb.saturating_mul(1024 * 1024))?;
            print_json(&json!({
                "pruned_cas_blobs": pruned_cas_blobs, "swept_orphan_stages": swept_orphan_stages,
                "active_refit_worktrees": worktree::list_worktrees(&repo),
                "cache_max_mb": cache_max_mb, "cache_gc": cache_gc
            }))
        }
        Commands::Learn { repo, rewind, trigger } => {
            print_json(&learn::query_rewind(&repo, rewind.as_deref(), trigger)?)
        }
        Commands::Probe { upstream, model, api_key_file, openclaw_url, openclaw_token } => {
            init_tracing();
            build_rt(false)?.block_on(cmd_probe(upstream, model, api_key_file, openclaw_url, openclaw_token))
        }
        Commands::Run { opts, resume } => cmd_run(opts, resume),
        Commands::Resume { opts } => cmd_run(opts, true),
        Commands::Bench { before, after, script, cmd, pairs } => {
            let res = if let Some(c) = cmd {
                runner::bench_command(&c, &before, &after, pairs)?
            } else if let Some(s) = script {
                runner::bench_python(&gates::read_gates(&after).python_bin, &s, &before, &after, pairs)?
            } else {
                anyhow::bail!("Provide either --script <path> or --cmd <command-template>");
            };
            print_json(&res)
        }
        Commands::Proxy {
            listen, upstream, model, ui_index, telemetry_out, repo,
            auth_token, api_key_file, cache_retention, keep_warm,
        } => {
            init_tracing();
            let key = read_key(api_key_file.as_ref())?;
            build_rt(true)?.block_on(async move {
                let tel = telemetry::TelemetryHub::new(telemetry_out)?;
                let mut cfg = proxy::ProxyConfig::new(listen.clone(), upstream, key, model);
                cfg.ui_index_path = ui_index;
                cfg.cache_retention = proxy::CacheMode::parse_mode(cache_retention.as_deref());
                cfg.keep_warm = keep_warm;
                cfg.auth_token = auth_token;
                let (state, handle) = proxy::start_proxy(cfg, tel).await?;
                if let Some(r) = repo { state.bind_repo(r); }
                tracing::info!("refit proxy + HUD listening on http://{listen}");
                handle.await?;
                Ok(())
            })
        }
        Commands::Mcp { repo, http } => cmd_mcp(repo, http),
        Commands::Check {
            repo, staged, diff, deps, strict, no_cache, kane, url, policy, target, app,
            remote, headed, bug_detection, no_adaptive_heal, timeout, verify_docs, format,
        } => {
            let kane_req = kane.map(|spec| gates::KaneRequest {
                spec, surface: None, url, policy, target, app, tunnel_name: None,
                remote, headless: !headed, bug_detection, no_adaptive_heal, timeout_secs: timeout,
            });
            cmd_check(&repo, &CheckOpts {
                staged, diff: diff.as_deref(), deps, strict, no_cache, verify_docs, kane: kane_req, format,
            })
        }
        Commands::Hook => cmd_hook(),
        Commands::Secrets { repo, staged, diff, install_hooks, no_cache, strict, sarif, format } => {
            let fmt = if sarif { OutputFormat::Sarif } else { format };
            cmd_secrets(&repo, staged, diff.as_deref(), install_hooks, no_cache, strict, fmt)
        }
        Commands::Research { lang, repo, max_results, queries } => {
            init_tracing();
            build_rt(false)?.block_on(async move {
                let client = repo.as_deref().map_or_else(
                    || research::WebClient::from_env(None),
                    |r| research::WebClient::for_repo(r, None),
                );
                let expanded = lang.map_or(queries.clone(), |l| {
                    queries.into_iter().map(|q| format!("best native {l} library for {q} benchmark")).collect()
                });
                let results = client.research_batch(expanded, max_results).await;
                print_json(&json!({
                    "online": client.is_online(), "rate_limit_rps": 10,
                    "cache_entries": client.cache_len(), "results": results
                }))
            })
        }
    }
}
