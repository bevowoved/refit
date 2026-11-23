mod gates;
mod graph;
mod hygiene;
mod learn;
mod mcp;
mod mvcc;
mod platform;
mod proxy;
mod research;
mod runner;
mod state;
mod stem;
mod telemetry;
mod types;
mod untangle;
mod worktree;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "refit",
    version = "0.5.0",
    about = "Pure-Rust cross-platform polyglot untangling & refactoring engine for Kimi K3.1 Pro on Cerebras"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Shared CLI arguments for `refit run` and `refit resume`.
#[derive(Args, Debug, Clone)]
struct RunCliArgs {
    repo: PathBuf,
    #[arg(long, default_value = "https://api.cerebras.ai/v1")]
    upstream: String,
    #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")]
    model: String,
    #[arg(long, default_value = "Cerebras WSE-3T")]
    provider_label: String,
    #[arg(long)]
    openclaw_url: Option<String>,
    #[arg(long)]
    openclaw_token: Option<String>,
    #[arg(long, default_value = "openclaw:main")]
    openclaw_model: String,
    #[arg(long, default_value = "127.0.0.1:18790")]
    proxy_addr: String,
    #[arg(long, default_value = "ui/index.html")]
    ui_index: PathBuf,
    #[arg(long, default_value = "skills/refit/SKILL.md")]
    skill_path: PathBuf,
    #[arg(long, default_value = "telemetry/run.jsonl")]
    telemetry_out: PathBuf,
    #[arg(long)]
    bench_script: Option<PathBuf>,
    /// Benchmark command template with `{repo}` placeholder.
    #[arg(long)]
    bench_cmd: Option<String>,
    #[arg(long)]
    baseline_copy: Option<PathBuf>,
    #[arg(long)]
    api_key_file: Option<PathBuf>,
    #[arg(long, default_value_t = 12)]
    initial_k: usize,
    #[arg(long, default_value_t = 20)]
    max_k: usize,
    #[arg(long, default_value_t = 2)]
    max_retries: u32,
    #[arg(long, default_value_t = 0)]
    hold_secs: u64,
    #[arg(long)]
    max_units: Option<usize>,
    /// Run waves in an isolated git worktree (`refit/<run_id>`).
    #[arg(long)]
    worktree: bool,
    /// Copy uncommitted changes into the isolated worktree.
    #[arg(long)]
    include_dirty: bool,
    /// Prompt cache mode: short, long, or none.
    #[arg(long)]
    cache_retention: Option<String>,
    /// Send keep-warm pings during idle gaps.
    #[arg(long)]
    keep_warm: bool,
    /// Strip tool schemas on proxy requests.
    #[arg(long)]
    strip_tools: bool,
    /// Fail at Gate 0 if a required language tool is missing.
    #[arg(long)]
    strict_tools: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Probe the Cerebras endpoint through the proxy and print latency metrics.
    Probe {
        #[arg(long, default_value = "https://api.cerebras.ai/v1")]
        upstream: String,
        #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")]
        model: String,
        #[arg(long)]
        api_key_file: Option<PathBuf>,
        #[arg(long)]
        openclaw_url: Option<String>,
        #[arg(long)]
        openclaw_token: Option<String>,
    },
    /// Scan the repository DAG, symbol graph, and hygiene scores.
    Scan {
        repo: PathBuf,
        #[arg(long)]
        no_cache: bool,
        #[arg(long)]
        fast: bool,
    },
    /// Audit clones, dead symbols, large files, unused dependencies, and infra configs.
    Audit {
        repo: PathBuf,
        /// Print the audit report as Markdown.
        #[arg(long)]
        markdown: bool,
        /// Audit manifest dependencies through KeenAble when KEENABLE_API_KEY is set.
        #[arg(long)]
        research: bool,
    },
    /// Generate AGENTS.md, .refit/invariants.yml, ADR-001, and the CI guard workflow.
    Sustain {
        repo: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// Prune unreferenced CAS blobs and expired stage directories.
    Gc {
        repo: PathBuf,
        #[arg(long, default_value_t = 32)]
        keep_revs: usize,
    },
    /// Run parallel DAG refactoring with formatters, Gates 0 to 3.5, and MVCC WAL.
    Run {
        #[command(flatten)]
        args: RunCliArgs,
        #[arg(long)]
        resume: bool,
    },
    /// Resume an interrupted run with the same flags as `refit run`.
    Resume {
        #[command(flatten)]
        args: RunCliArgs,
    },
    /// Run alternating OS CPU and RSS benchmarks.
    Bench {
        #[arg(long)]
        before: PathBuf,
        #[arg(long)]
        after: PathBuf,
        #[arg(long)]
        script: Option<PathBuf>,
        #[arg(long)]
        cmd: Option<String>,
        #[arg(long, default_value_t = 5)]
        pairs: usize,
    },
    /// Start the HTTP/2 Cerebras proxy and live HUD server.
    Proxy {
        #[arg(long, default_value = "127.0.0.1:18790")]
        listen: String,
        #[arg(long, default_value = "https://api.cerebras.ai/v1")]
        upstream: String,
        #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")]
        model: String,
        #[arg(long, default_value = "ui/index.html")]
        ui_index: PathBuf,
        #[arg(long, default_value = "telemetry/proxy.jsonl")]
        telemetry_out: PathBuf,
        #[arg(long)]
        repo: Option<PathBuf>,
        #[arg(long)]
        auth_token: Option<String>,
        #[arg(long)]
        api_key_file: Option<PathBuf>,
        #[arg(long)]
        cache_retention: Option<String>,
        #[arg(long)]
        keep_warm: bool,
    },
    /// Start the concurrent JSON-RPC 2.0 MCP stdio server.
    Mcp {
        repo: PathBuf,
    },
    /// Run batch KeenAble web searches.
    Research {
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        repo: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        max_results: usize,
        #[arg(required = true)]
        queries: Vec<String>,
    },
}

fn read_key(file_opt: Option<&PathBuf>) -> Result<String> {
    if let Ok(env_k) = std::env::var("CEREBRAS_API_KEY") {
        let t = env_k.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    if let Some(p) = file_opt {
        let raw = std::fs::read_to_string(p)
            .with_context(|| format!("read api key file {}", p.display()))?;
        let t = raw.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }
    anyhow::bail!("CEREBRAS_API_KEY env var is not set and --api-key-file was not provided");
}

fn run_opts(a: RunCliArgs, api_key: String, resume: bool) -> runner::RunOptions {
    let retention = proxy::CacheMode::parse_mode(a.cache_retention.as_deref());
    let kw = a.keep_warm
        || std::env::var("REFIT_KEEP_WARM")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
    runner::RunOptions {
        repo_root: a.repo,
        api_key,
        upstream_base_url: a.upstream,
        openclaw_base_url: a.openclaw_url,
        openclaw_token: a.openclaw_token,
        openclaw_model: a.openclaw_model,
        model: a.model,
        provider_label: a.provider_label,
        proxy_addr: a.proxy_addr,
        ui_index_path: a.ui_index,
        skill_path: a.skill_path,
        telemetry_path: a.telemetry_out,
        bench_script: a.bench_script,
        bench_cmd: a.bench_cmd,
        baseline_copy: a.baseline_copy,
        resume,
        initial_k: a.initial_k,
        max_k: a.max_k,
        max_retries: a.max_retries,
        hold_server_secs: a.hold_secs,
        max_units: a.max_units,
        worktree: a.worktree,
        include_dirty_worktree: a.include_dirty,
        cache_retention: retention,
        keep_warm: kw,
        preserve_tools: !a.strip_tools,
        strict_polyglot_tools: a.strict_tools,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Scan {
            repo,
            no_cache,
            fast,
        } => {
            let scorer = hygiene::HygieneScorer::open_persistent(&repo);
            let started = Instant::now();
            let (bundle, scan_cache_hit) =
                gates::scan_cached(&repo, !no_cache, fast, &scorer)?;
            let scan_ms = started.elapsed().as_millis() as u64;
            let style = hygiene::repo_style(&repo);
            let out = serde_json::json!({
                "version": "0.5.0",
                "allocator": "mimalloc",
                "pure_rust_hygiene": "codehygiene_v0.7.0_exact",
                "scan_cache_hit": scan_cache_hit,
                "fast_mode": fast,
                "repo_fingerprint": bundle.repo_fingerprint,
                "scan_ms": scan_ms,
                "detected_languages": bundle.detected_languages,
                "polyglot_toolchains": bundle.polyglot_toolchains,
                "style_config": {
                    "max_line_width": style.max_line_width,
                    "indent_width": style.indent_width
                },
                "host_os": bundle.plan.host_os,
                "stem_used": bundle.plan.stem_used,
                "stem_index_ms": bundle.plan.stem_index_ms,
                "warm_hygiene_us": scorer.warm_us(),
                "total_files": bundle.plan.total_files,
                "total_loc": bundle.plan.total_loc,
                "units": bundle.plan.units.len(),
                "scc_count": bundle.plan.scc_count,
                "wave_count": bundle.plan.wave_count,
                "tests_total": bundle.audit.tests_total,
                "tests_failed": bundle.audit.tests_failed,
                "ruff_errors": bundle.audit.ruff_errors,
                "ty_errors": bundle.audit.ty_errors,
                "hygiene_median": bundle.audit.hygiene_median,
                "hygiene_p10": bundle.audit.hygiene_p10,
                "engine_rss_mb": (platform::self_metrics().rss_bytes as f64) / (1024.0 * 1024.0)
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::Audit {
            repo,
            markdown,
            research,
        } => {
            let scorer = hygiene::HygieneScorer::open_persistent(&repo);
            let report = untangle::untangle_repo(&repo, &scorer)?;
            let dep_research = if research {
                let rc = research::WebClient::for_repo(&repo, None);
                rc.audit_deps(&repo).await
            } else {
                Vec::new()
            };
            if markdown {
                let mut md = untangle::render_audit(&report);
                if !dep_research.is_empty() {
                    md.push_str("## Dependency Research (KeenAble)\n\n");
                    md.push_str(&research::WebClient::format_snippets(&dep_research));
                }
                println!("{md}");
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "audit": report,
                        "dependency_research": dep_research
                    }))?
                );
            }
        }
        Commands::Sustain { repo, dry_run } => {
            let scorer = hygiene::HygieneScorer::open_persistent(&repo);
            let report = untangle::untangle_repo(&repo, &scorer)?;
            let out = untangle::write_sustain(&repo, &report, dry_run)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::Gc { repo, keep_revs } => {
            let store = mvcc::MvccStore::open(&repo, "gc")?;
            let pruned_cas = store.gc_cas(keep_revs)?;
            let swept_stages = gates::sweep_stages(&repo, 0);
            let worktrees = worktree::list_worktrees(&repo);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "pruned_cas_blobs": pruned_cas,
                    "swept_orphan_stages": swept_stages,
                    "active_refit_worktrees": worktrees
                }))?
            );
        }
        Commands::Probe {
            upstream,
            model,
            api_key_file,
            openclaw_url,
            openclaw_token,
        } => {
            let api_key = read_key(api_key_file.as_ref())?;
            let tel = telemetry::TelemetryHub::new(std::env::temp_dir().join("refit-probe.jsonl"))?;
            let proxy_addr = "127.0.0.1:18799".to_string();
            let (pstate, handle) = proxy::start_proxy(
                proxy::ProxyConfig {
                    listen_addr: proxy_addr.clone(),
                    upstream_base_url: upstream,
                    api_key,
                    model_override: model.clone(),
                    ui_index_path: PathBuf::from("ui/index.html"),
                    static_prefix_override: None,
                    cache_retention: proxy::CacheMode::Short,
                    keep_warm: false,
                    auth_token: None,
                    lang_prefixes: BTreeMap::new(),
                    preserve_tools: true,
                },
                tel,
            )
            .await?;

            let client = platform::new_http("refit-probe/0.5.0", 120, 4)?;
            let target_url = if let Some(oc) = &openclaw_url {
                format!("{}/chat/completions", oc.trim_end_matches('/'))
            } else {
                format!("http://{proxy_addr}/v1/chat/completions")
            };

            let mut req = client
                .post(&target_url)
                .header("x-refit-module", "probe")
                .header("x-refit-attempt", "1")
                .json(&serde_json::json!({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Reply OK."},
                        {"role": "user", "content": "OK"}
                    ],
                    "stream": false,
                    "max_completion_tokens": 32,
                    "reasoning_effort": "low",
                    "reasoning_format": "parsed",
                    "temperature": 1.0
                }));
            if let Some(tok) = openclaw_token {
                req = req.bearer_auth(tok);
            }

            let resp = req.send().await?;
            let code = resp.status().as_u16();
            let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            let m = pstate.take_metrics("probe");
            let snap = pstate.snapshot();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "http_status": code,
                    "via_openclaw": openclaw_url.is_some(),
                    "content": body["choices"][0]["message"]["content"],
                    "metrics": m,
                    "proxy_snapshot": snap
                }))?
            );
            handle.abort();
        }
        Commands::Run { args, resume } => {
            let api_key = read_key(args.api_key_file.as_ref())?;
            let summary = runner::execute_run(run_opts(args, api_key, resume)).await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Resume { args } => {
            let api_key = read_key(args.api_key_file.as_ref())?;
            let summary = runner::execute_run(run_opts(args, api_key, true)).await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Bench {
            before,
            after,
            script,
            cmd,
            pairs,
        } => {
            let res = if let Some(c) = cmd {
                runner::bench_command(&c, &before, &after, pairs)?
            } else if let Some(s) = script {
                let cfg = gates::read_gates(&after);
                runner::bench_python(&cfg.python_bin, &s, &before, &after, pairs)?
            } else {
                anyhow::bail!("Provide either --script <path> or --cmd <command-template>");
            };
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Commands::Proxy {
            listen,
            upstream,
            model,
            ui_index,
            telemetry_out,
            repo,
            auth_token,
            api_key_file,
            cache_retention,
            keep_warm,
        } => {
            let api_key = read_key(api_key_file.as_ref())?;
            let tel = telemetry::TelemetryHub::new(telemetry_out)?;
            let (state, handle) = proxy::start_proxy(
                proxy::ProxyConfig {
                    listen_addr: listen.clone(),
                    upstream_base_url: upstream,
                    api_key,
                    model_override: model,
                    ui_index_path: ui_index,
                    static_prefix_override: None,
                    cache_retention: proxy::CacheMode::parse_mode(cache_retention.as_deref()),
                    keep_warm,
                    auth_token,
                    lang_prefixes: BTreeMap::new(),
                    preserve_tools: true,
                },
                tel,
            )
            .await?;
            if let Some(r) = repo {
                state.bind_repo(r);
            }
            tracing::info!("refit proxy + HUD listening on http://{listen}");
            handle.await?;
        }
        Commands::Mcp { repo } => {
            mcp::serve_mcp(repo).await?;
        }
        Commands::Research {
            lang,
            repo,
            max_results,
            queries,
        } => {
            let client = match repo {
                Some(ref r) => research::WebClient::for_repo(r, None),
                None => research::WebClient::from_env(None),
            };
            let expanded: Vec<String> = if let Some(l) = lang {
                queries
                    .into_iter()
                    .map(|q| {
                        format!(
                            "best latest battle-tested native {l} library for {q} benchmark counterexamples vs handwritten 2026"
                        )
                    })
                    .collect()
            } else {
                queries
            };
            let results = client.research_batch(expanded, max_results).await;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "online": client.is_online(),
                    "rate_limit_rps": 10,
                    "cache_entries": client.cache_len(),
                    "results": results
                }))?
            );
        }
    }

    Ok(())
}
