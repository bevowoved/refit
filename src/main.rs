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
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_UPSTREAM: &str = "https://api.cerebras.ai/v1";

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

#[derive(Args, Debug, Clone)]
struct RunCliArgs {
    repo: PathBuf,
    #[arg(long, default_value = DEFAULT_UPSTREAM)] upstream: String,
    #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")] model: String,
    #[arg(long, default_value = "Cerebras WSE-3T")] provider_label: String,
    #[arg(long)] openclaw_url: Option<String>,
    #[arg(long)] openclaw_token: Option<String>,
    #[arg(long, default_value = "openclaw:main")] openclaw_model: String,
    #[arg(long, default_value = "127.0.0.1:18790")] proxy_addr: String,
    #[arg(long, default_value = "ui/index.html")] ui_index: PathBuf,
    #[arg(long, default_value = "skills/refit/SKILL.md")] skill_path: PathBuf,
    #[arg(long, default_value = "telemetry/run.jsonl")] telemetry_out: PathBuf,
    #[arg(long)] bench_script: Option<PathBuf>,
    #[arg(long)] bench_cmd: Option<String>,
    #[arg(long)] baseline_copy: Option<PathBuf>,
    #[arg(long)] api_key_file: Option<PathBuf>,
    #[arg(long, default_value_t = 12)] initial_k: usize,
    #[arg(long, default_value_t = 20)] max_k: usize,
    #[arg(long, default_value_t = 2)] max_retries: u32,
    #[arg(long, default_value_t = 0)] hold_secs: u64,
    #[arg(long)] max_units: Option<usize>,
    #[arg(long)] worktree: bool,
    #[arg(long)] include_dirty: bool,
    #[arg(long)] cache_retention: Option<String>,
    #[arg(long)] keep_warm: bool,
    #[arg(long)] strip_tools: bool,
    #[arg(long)] strict_tools: bool,
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
    /// Generate AGENTS.md, .refit/invariants.yml, ADR-001, and the CI guard workflow.
    Sustain {
        repo: PathBuf,
        #[arg(long)] dry_run: bool,
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
        #[arg(long, default_value = "smell")] trigger: String,
    },
    /// Run parallel DAG refactoring with formatters, Gates 0 to 3.5, and MVCC WAL.
    Run {
        #[command(flatten)] args: RunCliArgs,
        #[arg(long)] resume: bool,
    },
    /// Resume an interrupted run with the same flags as `refit run`.
    Resume { #[command(flatten)] args: RunCliArgs },
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
        #[arg(long, default_value = "ui/index.html")] ui_index: PathBuf,
        #[arg(long, default_value = "telemetry/proxy.jsonl")] telemetry_out: PathBuf,
        #[arg(long)] repo: Option<PathBuf>,
        #[arg(long)] auth_token: Option<String>,
        #[arg(long)] api_key_file: Option<PathBuf>,
        #[arg(long)] cache_retention: Option<String>,
        #[arg(long)] keep_warm: bool,
    },
    /// Start the concurrent JSON-RPC 2.0 MCP server (stdio, or stateless HTTP via --http).
    Mcp {
        repo: PathBuf,
        #[arg(long)] http: Option<String>,
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

fn run_opts(a: RunCliArgs, api_key: String, resume: bool) -> runner::RunOptions {
    let retention = proxy::CacheMode::parse_mode(a.cache_retention.as_deref());
    let kw = a.keep_warm || platform::env_truthy("REFIT_KEEP_WARM");
    runner::RunOptions {
        repo_root: a.repo, api_key, upstream_base_url: a.upstream, openclaw_base_url: a.openclaw_url,
        openclaw_token: a.openclaw_token, openclaw_model: a.openclaw_model, model: a.model,
        provider_label: a.provider_label, proxy_addr: a.proxy_addr, ui_index_path: a.ui_index,
        skill_path: a.skill_path, telemetry_path: a.telemetry_out, bench_script: a.bench_script,
        bench_cmd: a.bench_cmd, baseline_copy: a.baseline_copy, resume, initial_k: a.initial_k,
        max_k: a.max_k, max_retries: a.max_retries, hold_server_secs: a.hold_secs,
        max_units: a.max_units, worktree: a.worktree, include_dirty_worktree: a.include_dirty,
        cache_retention: retention, keep_warm: kw, preserve_tools: !a.strip_tools,
        strict_polyglot_tools: a.strict_tools,
    }
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

fn cmd_sustain(repo: &Path, dry_run: bool, ste100: bool, ste_dict: Option<&Path>) -> Result<()> {
    let scorer = hygiene::HygieneScorer::open_persistent(repo);
    let rep = untangle::untangle_repo_with_scope(repo, &scorer, untangle::AuditScope::Sustain)?;
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
        Commands::Sustain { repo, dry_run, ste100, ste_dict } => {
            cmd_sustain(&repo, dry_run, ste100, ste_dict.as_deref())
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
            let trig = learn::RewindTrigger::parse(&trigger);
            print_json(&learn::query_rewind(&repo, rewind.as_deref(), trig)?)
        }
        Commands::Probe { upstream, model, api_key_file, openclaw_url, openclaw_token } => {
            init_tracing();
            build_rt(false)?.block_on(cmd_probe(upstream, model, api_key_file, openclaw_url, openclaw_token))
        }
        Commands::Run { args, resume } => {
            init_tracing();
            let key = read_key(args.api_key_file.as_ref())?;
            print_json(&build_rt(true)?.block_on(runner::execute_run(run_opts(args, key, resume)))?)
        }
        Commands::Resume { args } => {
            init_tracing();
            let key = read_key(args.api_key_file.as_ref())?;
            print_json(&build_rt(true)?.block_on(runner::execute_run(run_opts(args, key, true)))?)
        }
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
