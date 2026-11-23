mod gates;
mod graph;
mod hygiene;
mod learn;
mod mcp;
mod mvcc;
mod platform;
mod proxy;
mod runner;
mod stem;
mod telemetry;
mod types;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "refit",
    version = "0.3.0",
    about = "Cross-platform Rust + OpenClaw refactoring engine for Kimi K3.1 Pro on Cerebras"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Probe Cerebras endpoint (through :18790 proxy) and print live 50K TPM headers, queue_ms, and decode_tps
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
    /// Scan repository via sandboxed stem v0.4.0 + petgraph DAG + baseline Gates 1-3 audit
    Scan {
        repo: PathBuf,
    },
    /// Run full parallel DAG refactoring across repo with Gates 0-3 + MVCC WAL
    Run {
        repo: PathBuf,
        #[arg(long)]
        resume: bool,
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
    },
    /// Shorthand for `refit run --resume`
    Resume {
        repo: PathBuf,
        #[arg(long, default_value = "https://api.cerebras.ai/v1")]
        upstream: String,
        #[arg(long, default_value = "moonshotai-kimi-k3.1-pro-trial")]
        model: String,
        #[arg(long)]
        openclaw_url: Option<String>,
        #[arg(long)]
        openclaw_token: Option<String>,
        #[arg(long, default_value = "127.0.0.1:18790")]
        proxy_addr: String,
        #[arg(long, default_value = "ui/index.html")]
        ui_index: PathBuf,
        #[arg(long, default_value = "skills/refit/SKILL.md")]
        skill_path: PathBuf,
        #[arg(long, default_value = "telemetry/run.jsonl")]
        telemetry_out: PathBuf,
        #[arg(long)]
        api_key_file: Option<PathBuf>,
        #[arg(long)]
        max_units: Option<usize>,
    },
    /// Run 5 alternating before/after OS CPU and peak RSS benchmarks
    Bench {
        #[arg(long)]
        before: PathBuf,
        #[arg(long)]
        after: PathBuf,
        #[arg(long)]
        script: PathBuf,
        #[arg(long, default_value_t = 5)]
        pairs: usize,
    },
    /// Run standalone :18790 HTTP/2 Cerebras proxy + live 30 Hz HUD server
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
        api_key_file: Option<PathBuf>,
    },
    /// Run JSON-RPC 2.0 MCP stdio server for OpenClaw / MCP clients
    Mcp {
        repo: PathBuf,
    },
}

fn load_api_key(file_opt: Option<&PathBuf>) -> Result<String> {
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
        Commands::Scan { repo } => {
            let worker = hygiene::HygieneScorer::resolve_default_worker(&repo);
            let scorer = hygiene::HygieneScorer::new(worker);
            let plan = graph::build_dag_plan(&repo, None, &scorer)?;
            let gate_cfg = gates::detect_gate_config(&repo);
            let rels: Vec<String> = plan.units.iter().flat_map(|u| u.files.clone()).collect();
            let audit = gates::run_full_repo_audit(&gate_cfg, &rels, &scorer)?;
            let out = serde_json::json!({
                "host_os": plan.host_os,
                "stem_used": plan.stem_used,
                "stem_index_ms": plan.stem_index_ms,
                "total_files": plan.total_files,
                "total_loc": plan.total_loc,
                "units": plan.units.len(),
                "scc_count": plan.scc_count,
                "wave_count": plan.wave_count,
                "tests_total": audit.tests_total,
                "tests_failed": audit.tests_failed,
                "ruff_errors": audit.ruff_errors,
                "ty_errors": audit.ty_errors,
                "hygiene_median": audit.hygiene_median,
                "hygiene_p10": audit.hygiene_p10,
                "engine_rss_mb": (platform::self_metrics().rss_bytes as f64) / (1024.0 * 1024.0)
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::Probe {
            upstream,
            model,
            api_key_file,
            openclaw_url,
            openclaw_token,
        } => {
            let api_key = load_api_key(api_key_file.as_ref())?;
            let tel = telemetry::TelemetryHub::new(std::env::temp_dir().join("refit-probe.jsonl"))?;
            let proxy_addr = "127.0.0.1:18799".to_string();
            let (pstate, handle) = proxy::start_proxy_server(
                proxy::ProxyConfig {
                    listen_addr: proxy_addr.clone(),
                    upstream_base_url: upstream,
                    api_key,
                    model_override: model.clone(),
                    ui_index_path: PathBuf::from("ui/index.html"),
                    static_prefix_override: None,
                },
                tel,
            )
            .await?;

            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?;

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
            let m = pstate.take_module_metrics("probe");
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
        Commands::Run {
            repo,
            resume,
            upstream,
            model,
            provider_label,
            openclaw_url,
            openclaw_token,
            openclaw_model,
            proxy_addr,
            ui_index,
            skill_path,
            telemetry_out,
            bench_script,
            baseline_copy,
            api_key_file,
            initial_k,
            max_k,
            max_retries,
            hold_secs,
            max_units,
        } => {
            let api_key = load_api_key(api_key_file.as_ref())?;
            let summary = runner::execute_run(runner::RunOptions {
                repo_root: repo,
                api_key,
                upstream_base_url: upstream,
                openclaw_base_url: openclaw_url,
                openclaw_token,
                openclaw_model,
                model,
                provider_label,
                proxy_addr,
                ui_index_path: ui_index,
                skill_path,
                telemetry_path: telemetry_out,
                bench_script,
                baseline_copy,
                resume,
                initial_k,
                max_k,
                max_retries,
                hold_server_secs: hold_secs,
                max_units,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Resume {
            repo,
            upstream,
            model,
            openclaw_url,
            openclaw_token,
            proxy_addr,
            ui_index,
            skill_path,
            telemetry_out,
            api_key_file,
            max_units,
        } => {
            let api_key = load_api_key(api_key_file.as_ref())?;
            let summary = runner::execute_run(runner::RunOptions {
                repo_root: repo,
                api_key,
                upstream_base_url: upstream,
                openclaw_base_url: openclaw_url,
                openclaw_token,
                openclaw_model: "openclaw:main".into(),
                model,
                provider_label: "Cerebras WSE-3T".into(),
                proxy_addr,
                ui_index_path: ui_index,
                skill_path,
                telemetry_path: telemetry_out,
                bench_script: None,
                baseline_copy: None,
                resume: true,
                initial_k: 12,
                max_k: 20,
                max_retries: 2,
                hold_server_secs: 0,
                max_units,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Commands::Bench {
            before,
            after,
            script,
            pairs,
        } => {
            let cfg = gates::detect_gate_config(&after);
            let res = runner::run_alternating_os_bench(&cfg.python_bin, &script, &before, &after, pairs)?;
            println!("{}", serde_json::to_string_pretty(&res)?);
        }
        Commands::Proxy {
            listen,
            upstream,
            model,
            ui_index,
            telemetry_out,
            api_key_file,
        } => {
            let api_key = load_api_key(api_key_file.as_ref())?;
            let tel = telemetry::TelemetryHub::new(telemetry_out)?;
            let (_state, handle) = proxy::start_proxy_server(
                proxy::ProxyConfig {
                    listen_addr: listen.clone(),
                    upstream_base_url: upstream,
                    api_key,
                    model_override: model,
                    ui_index_path: ui_index,
                    static_prefix_override: None,
                },
                tel,
            )
            .await?;
            tracing::info!("refit proxy + HUD listening on http://{listen}");
            handle.await?;
        }
        Commands::Mcp { repo } => {
            mcp::run_mcp_stdio(repo).await?;
        }
    }

    Ok(())
}
