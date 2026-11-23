use crate::gates::{GateConfig, detect_gate_config, run_full_repo_audit, verify_candidate};
use crate::graph::{
    apply_asrf_hunks, build_dag_plan, parse_asrf_hunks, slice_ast_packet, units_interfere,
};
use crate::hygiene::HygieneScorer;
use crate::learn::LearnStore;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::platform::{host_os_tag, now_ms, run_sandboxed_cmd, self_metrics};
use crate::proxy::{ProxyConfig, ProxyState, start_proxy_server};
use crate::telemetry::TelemetryHub;
use crate::types::{BenchComparison, ModuleUnit, RunSummary};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub repo_root: PathBuf,
    pub api_key: String,
    pub upstream_base_url: String,
    pub openclaw_base_url: Option<String>,
    pub openclaw_token: Option<String>,
    pub openclaw_model: String,
    pub model: String,
    pub provider_label: String,
    pub proxy_addr: String,
    pub ui_index_path: PathBuf,
    pub skill_path: PathBuf,
    pub telemetry_path: PathBuf,
    pub bench_script: Option<PathBuf>,
    pub baseline_copy: Option<PathBuf>,
    pub resume: bool,
    pub initial_k: usize,
    pub max_k: usize,
    pub max_retries: u32,
    pub hold_server_secs: u64,
    pub max_units: Option<usize>,
}

struct WorkerOutcome {
    unit: ModuleUnit,
    committed: bool,
    skipped_resumed: bool,
    cas_conflicts: u64,
    rollbacks_g0: u64,
    rollbacks_g1: u64,
    rollbacks_g2: u64,
    rollbacks_g3: u64,
    retries: u64,
    saw_429: bool,
    queue_ms_samples: Vec<f64>,
    decode_tps_samples: Vec<f64>,
}

pub async fn execute_run(opts: RunOptions) -> Result<RunSummary> {
    let wall_start = Instant::now();
    let run_id = format!("refit-{}", now_ms() % 10_000_000);

    let static_system_prompt = load_static_system_prompt(&opts.skill_path);

    let telemetry = TelemetryHub::new(opts.telemetry_path.clone())?;
    let proxy_cfg = ProxyConfig {
        listen_addr: opts.proxy_addr.clone(),
        upstream_base_url: opts.upstream_base_url.clone(),
        api_key: opts.api_key.clone(),
        model_override: opts.model.clone(),
        ui_index_path: opts.ui_index_path.clone(),
        static_prefix_override: Some(static_system_prompt.clone()),
    };
    let (proxy_state, proxy_handle) = start_proxy_server(proxy_cfg, telemetry.clone()).await?;

    let worker_script = HygieneScorer::resolve_default_worker(&opts.repo_root);
    let hygiene = Arc::new(HygieneScorer::new(worker_script));
    let gate_cfg = Arc::new(detect_gate_config(&opts.repo_root));
    let mvcc = Arc::new(MvccStore::open(&opts.repo_root, &run_id)?);
    let learn = Arc::new(LearnStore::open(&opts.repo_root)?);

    // Phase 0: Optional WAL crash recovery before building DAG
    let mut already_committed: HashMap<String, BTreeMap<String, String>> = HashMap::new();
    let mut rolled_back_on_resume = 0usize;
    if opts.resume {
        let rep = mvcc.recover_from_wal()?;
        rolled_back_on_resume = rep.rolled_back_txs;
        already_committed = rep.committed_modules;
        tracing::info!(
            "WAL resume: rolled_back_txs={} committed_modules={} head_rev={}",
            rep.rolled_back_txs,
            already_committed.len(),
            rep.head_rev
        );
    }

    let plan = build_dag_plan(&opts.repo_root, None, &hygiene)?;
    let all_rel_files: Vec<String> = plan.units.iter().flat_map(|u| u.files.clone()).collect();
    let baseline_audit = run_full_repo_audit(&gate_cfg, &all_rel_files, &hygiene)?;

    // Decide chat endpoint (prefer OpenClaw Gateway if reachable, else local :18790 proxy)
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(210))
        .build()?;
    let (chat_url, chat_bearer, chat_model, via_openclaw) =
        resolve_chat_endpoint(&client, &opts).await;

    telemetry.emit(
        "run_start",
        None,
        None,
        None,
        Some(0),
        Some(json!({
            "run_id": run_id,
            "host_os": plan.host_os,
            "model": opts.model,
            "provider_label": opts.provider_label,
            "via_openclaw": via_openclaw,
            "resume": opts.resume,
            "rolled_back_on_resume": rolled_back_on_resume,
            "stem_used": plan.stem_used,
            "stem_index_ms": plan.stem_index_ms,
            "total_files": plan.total_files,
            "total_loc": plan.total_loc,
            "scc_count": plan.scc_count,
            "wave_count": plan.wave_count,
            "uncached_tpm_cap": proxy_state.snapshot().uncached_tpm_cap,
            "baseline": {
                "tests_total": baseline_audit.tests_total,
                "tests_failed": baseline_audit.tests_failed,
                "ruff_errors": baseline_audit.ruff_errors,
                "ty_errors": baseline_audit.ty_errors,
                "hygiene_median": baseline_audit.hygiene_median,
                "hygiene_p10": baseline_audit.hygiene_p10
            }
        })),
        None,
    );

    telemetry.emit(
        "dag_ready",
        None,
        None,
        None,
        Some(0),
        Some(serde_json::to_value(&plan)?),
        None,
    );

    // Filter units that still need refactoring (skipping WAL-verified committed modules on resume!)
    let mut pending: Vec<ModuleUnit> = Vec::new();
    let mut completed_ids: HashSet<String> = HashSet::new();
    let mut skipped_resumed_units = 0usize;

    let mut target_units = plan.units.clone();
    if let Some(lim) = opts.max_units {
        target_units.truncate(lim);
    }

    for u in target_units {
        if opts.resume {
            if let Some(saved_hashes) = already_committed.get(&u.id) {
                let all_match = u.files.iter().all(|rel| {
                    saved_hashes
                        .get(rel)
                        .and_then(|expected| {
                            MvccStore::hash_file(&opts.repo_root.join(rel))
                                .ok()
                                .map(|actual| &actual == expected)
                        })
                        .unwrap_or(false)
                });
                if all_match {
                    skipped_resumed_units += 1;
                    completed_ids.insert(u.id.clone());
                    telemetry.emit(
                        "mvcc_commit",
                        Some(u.id.clone()),
                        Some(u.layer),
                        Some(u.wave),
                        None,
                        Some(json!({"skipped_resumed": true, "head_rev": mvcc.head_rev()})),
                        None,
                    );
                    continue;
                }
            }
        }
        pending.push(u);
    }

    // Also pre-mark units outside max_units as completed so dependencies resolve cleanly
    for u in &plan.units {
        if !pending.iter().any(|p| p.id == u.id) {
            completed_ids.insert(u.id.clone());
        }
    }

    // Phase 1: AIMD Sliding-Window Kahn Parallel Dispatch
    let mut current_k = opts.initial_k.clamp(2, opts.max_k.max(2));
    let mut in_flight: HashMap<String, ModuleUnit> = HashMap::new();
    let mut join_set: JoinSet<WorkerOutcome> = JoinSet::new();

    let mut committed_units = skipped_resumed_units;
    let mut cas_conflicts = 0u64;
    let mut rollbacks_g0 = 0u64;
    let mut rollbacks_g1 = 0u64;
    let mut rollbacks_g2 = 0u64;
    let mut rollbacks_g3 = 0u64;
    let mut retries_total = 0u64;
    let mut all_queue_ms: Vec<f64> = Vec::new();
    let mut all_decode_tps: Vec<f64> = Vec::new();

    let diag_map = Arc::new(baseline_audit.per_file_diagnostics);
    let sys_prompt_arc = Arc::new(static_system_prompt);

    loop {
        // Fill available AIMD slots with disjoint ready DAG units
        while in_flight.len() < current_k {
            let next_pos = pending.iter().position(|cand| {
                let deps_done = cand.deps.iter().all(|d| completed_ids.contains(d));
                let disjoint = !in_flight.values().any(|active| units_interfere(cand, active));
                deps_done && disjoint
            });

            let Some(idx) = next_pos else { break };
            let unit = pending.remove(idx);
            in_flight.insert(unit.id.clone(), unit.clone());

            let run_id_c = run_id.clone();
            let repo_c = opts.repo_root.clone();
            let client_c = client.clone();
            let url_c = chat_url.clone();
            let bearer_c = chat_bearer.clone();
            let model_c = chat_model.clone();
            let sys_c = sys_prompt_arc.clone();
            let diag_c = diag_map.clone();
            let hygiene_c = hygiene.clone();
            let gate_c = gate_cfg.clone();
            let mvcc_c = mvcc.clone();
            let learn_c = learn.clone();
            let proxy_c = proxy_state.clone();
            let tel_c = telemetry.clone();
            let max_retries = opts.max_retries;
            let k_snap = current_k;

            join_set.spawn(async move {
                process_single_unit(
                    run_id_c,
                    repo_c,
                    unit,
                    client_c,
                    url_c,
                    bearer_c,
                    model_c,
                    sys_c,
                    diag_c,
                    hygiene_c,
                    gate_c,
                    mvcc_c,
                    learn_c,
                    proxy_c,
                    tel_c,
                    max_retries,
                    k_snap,
                )
                .await
            });
        }

        if in_flight.is_empty() {
            if pending.is_empty() {
                break;
            }
            // Break any remaining dependency wait defensively by relaxing deps for stuck units
            let forced = pending.remove(0);
            completed_ids.insert(forced.id.clone());
            continue;
        }

        if let Some(res) = join_set.join_next().await {
            if let Ok(out) = res {
                in_flight.remove(&out.unit.id);
                completed_ids.insert(out.unit.id.clone());
                if out.committed && !out.skipped_resumed {
                    committed_units += 1;
                }
                cas_conflicts += out.cas_conflicts;
                rollbacks_g0 += out.rollbacks_g0;
                rollbacks_g1 += out.rollbacks_g1;
                rollbacks_g2 += out.rollbacks_g2;
                rollbacks_g3 += out.rollbacks_g3;
                retries_total += out.retries;
                all_queue_ms.extend(out.queue_ms_samples);
                all_decode_tps.extend(out.decode_tps_samples);

                // AIMD concurrency adjustment
                if out.saw_429 {
                    current_k = (current_k / 2).max(2);
                } else if out.committed && current_k < opts.max_k {
                    current_k += 1;
                }
            }
        }
    }

    // Phase 2: Full repo verification audit
    let final_audit = run_full_repo_audit(&gate_cfg, &all_rel_files, &hygiene)?;

    // Optional 5-run alternating OS CPU/RSS benchmark
    let bench = match (&opts.bench_script, &opts.baseline_copy) {
        (Some(script), Some(base_dir)) if script.exists() && base_dir.exists() => {
            match run_alternating_os_bench(&gate_cfg.python_bin, script, base_dir, &opts.repo_root, 5) {
                Ok(b) => {
                    telemetry.emit(
                        "bench_result",
                        None,
                        None,
                        None,
                        None,
                        Some(serde_json::to_value(&b)?),
                        None,
                    );
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("Alternating bench skipped due to error: {e}");
                    None
                }
            }
        }
        _ => None,
    };

    let p_snap = proxy_state.snapshot();
    let (proxy_q, proxy_tps) = proxy_state.timing_samples();
    if all_queue_ms.is_empty() {
        all_queue_ms = proxy_q;
    }
    if all_decode_tps.is_empty() {
        all_decode_tps = proxy_tps;
    }
    let self_m = self_metrics();
    let total_wall_s = wall_start.elapsed().as_secs_f64();
    let avg_queue_s = if all_queue_ms.is_empty() {
        0.0
    } else {
        (all_queue_ms.iter().sum::<f64>() / all_queue_ms.len() as f64) / 1000.0
    };
    let median_decode_tps = median_f64(&mut all_decode_tps);
    let prefix_cache_hit_pct = if p_snap.total_prompt_tokens > 0 {
        (p_snap.total_cached_tokens as f64 / p_snap.total_prompt_tokens as f64) * 100.0
    } else {
        0.0
    };

    let summary = RunSummary {
        run_id,
        host_os: host_os_tag(),
        model: opts.model.clone(),
        provider_label: opts.provider_label.clone(),
        total_files: plan.total_files,
        total_loc: plan.total_loc,
        total_units: plan.units.len(),
        committed_units,
        skipped_resumed_units,
        total_wall_s,
        avg_queue_s,
        median_decode_tps,
        hygiene_before_median: baseline_audit.hygiene_median,
        hygiene_before_p10: baseline_audit.hygiene_p10,
        hygiene_after_median: final_audit.hygiene_median,
        hygiene_after_p10: final_audit.hygiene_p10,
        tests_total: final_audit.tests_total.max(baseline_audit.tests_total),
        tests_before_fail: baseline_audit.tests_failed,
        tests_after_fail: final_audit.tests_failed,
        gate3_before_ruff: baseline_audit.ruff_errors,
        gate3_before_ty: baseline_audit.ty_errors,
        gate3_after_ruff: final_audit.ruff_errors,
        gate3_after_ty: final_audit.ty_errors,
        cas_conflicts,
        rollbacks_g0,
        rollbacks_g1,
        rollbacks_g2,
        rollbacks_g3,
        retries_total,
        http_429_count: p_snap.http_429_count,
        circuit_pauses: p_snap.circuit_pauses,
        prefix_drift_events: p_snap.prefix_drift_events,
        prefix_cache_hit_pct,
        uncached_tpm_cap: p_snap.uncached_tpm_cap,
        total_uncached_tokens: p_snap.total_uncached_tokens,
        total_cached_tokens: p_snap.total_cached_tokens,
        engine_rss_mb: (self_m.rss_bytes as f64) / (1024.0 * 1024.0),
        bench,
    };

    telemetry.emit(
        "run_end",
        None,
        None,
        None,
        Some(p_snap.uncached_tpm_rolling),
        Some(json!({
            "via_openclaw": via_openclaw,
            "learn_rules": learn.stats().0,
            "learn_helpful": learn.stats().1
        })),
        Some(summary.clone()),
    );

    if opts.hold_server_secs > 0 {
        tokio::time::sleep(Duration::from_secs(opts.hold_server_secs)).await;
    }

    proxy_handle.abort();
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
async fn process_single_unit(
    run_id: String,
    repo_root: PathBuf,
    mut unit: ModuleUnit,
    client: reqwest::Client,
    chat_url: String,
    chat_bearer: String,
    chat_model: String,
    static_sys: Arc<String>,
    baseline_diags: Arc<HashMap<String, Vec<String>>>,
    hygiene: Arc<HygieneScorer>,
    gate_cfg: Arc<GateConfig>,
    mvcc: Arc<MvccStore>,
    learn: Arc<LearnStore>,
    proxy: Arc<ProxyState>,
    telemetry: TelemetryHub,
    max_retries: u32,
    current_k: usize,
) -> WorkerOutcome {
    let mut out = WorkerOutcome {
        unit: unit.clone(),
        committed: false,
        skipped_resumed: false,
        cas_conflicts: 0,
        rollbacks_g0: 0,
        rollbacks_g1: 0,
        rollbacks_g2: 0,
        rollbacks_g3: 0,
        retries: 0,
        saw_429: false,
        queue_ms_samples: Vec::new(),
        decode_tps_samples: Vec::new(),
    };

    let dir_prefix = Path::new(&unit.files[0])
        .parent()
        .map(|p| p.display().to_string().replace('\\', "/"))
        .unwrap_or_else(|| "*".into());

    let mut cur_diags: Vec<String> = unit
        .files
        .iter()
        .flat_map(|f| baseline_diags.get(f).cloned().unwrap_or_default())
        .collect();

    let mut prev_assistant: Option<(String, String)> = None; // (reasoning, content)
    let mut prev_user_msg: Option<String> = None;

    for attempt in 1..=(max_retries + 1) {
        if attempt > 1 {
            out.retries += 1;
        }

        // Refresh disk base hashes cleanly before each attempt
        let mut base_texts = BTreeMap::new();
        let mut expected_hashes = BTreeMap::new();
        for rel in &unit.files {
            if let Ok(txt) = fs::read_to_string(repo_root.join(rel)) {
                expected_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
                base_texts.insert(rel.clone(), txt);
            }
        }
        unit.base_blake3 = expected_hashes.clone();

        let learn_suffix = learn.prompt_suffix(&dir_prefix);
        let slice_budget = if unit.files.len() > 1 { 24_000 } else { 10_000 };
        let user_prompt = match slice_ast_packet(&repo_root, &unit, &cur_diags, &learn_suffix, slice_budget) {
            Ok(p) => p,
            Err(_) => break,
        };

        telemetry.emit(
            "wave_dispatch",
            Some(unit.id.clone()),
            Some(unit.layer),
            Some(unit.wave),
            Some(proxy.snapshot().uncached_tpm_rolling),
            Some(json!({ "attempt": attempt, "k": current_k })),
            None,
        );

        let mut messages = vec![json!({ "role": "system", "content": static_sys.as_str() })];
        if attempt == 2 {
            if let (Some(prev_u), Some((prev_r, prev_c))) = (&prev_user_msg, &prev_assistant) {
                messages.push(json!({ "role": "user", "content": prev_u }));
                let mut asst = json!({ "role": "assistant", "content": prev_c });
                if !prev_r.is_empty() {
                    asst["reasoning"] = json!(prev_r);
                }
                messages.push(asst);
                let retry_feedback = format!(
                    "Your previous patch failed verification and was ROLLED BACK (the file on disk is UNCHANGED). Every <<<SEARCH>>> block MUST match the ORIGINAL file shown in the first user message verbatim. Fix EVERY diagnostic below and emit ONLY complete <<<REFIT_PATCH>>> blocks:\n- {}\n{}",
                    cur_diags.join("\n- "),
                    learn_suffix
                );
                messages.push(json!({ "role": "user", "content": retry_feedback }));
            } else {
                messages.push(json!({ "role": "user", "content": user_prompt }));
                prev_user_msg = Some(user_prompt);
            }
        } else {
            messages.push(json!({ "role": "user", "content": user_prompt }));
            prev_user_msg = Some(user_prompt);
        }

        let mut req_body = json!({
            "model": chat_model,
            "messages": messages,
            "stream": true,
            "max_completion_tokens": if unit.files.len() > 1 { 5500 } else { 3200 },
            "temperature": 1.0,
            "reasoning_effort": "low",
            "reasoning_format": "parsed"
        });
        if attempt >= 2 {
            let short_run: String = run_id.chars().take(10).collect();
            req_body["prompt_cache_key"] = json!(format!("refit:{short_run}:m:{}", unit.id));
        }

        let resp_res = client
            .post(&chat_url)
            .bearer_auth(&chat_bearer)
            .header("x-refit-module", &unit.id)
            .header("x-refit-attempt", attempt.to_string())
            .json(&req_body)
            .send()
            .await;

        let (status_u16, assistant_reasoning, assistant_content) = match resp_res {
            Ok(resp) => {
                let code = resp.status().as_u16();
                if code == 429 {
                    out.saw_429 = true;
                }
                if code != 200 {
                    let body = resp.text().await.unwrap_or_default();
                    cur_diags = vec![format!("HTTP {code}: {}", body.chars().take(160).collect::<String>())];
                    tokio::time::sleep(Duration::from_secs(3 * attempt as u64)).await;
                    continue;
                }
                read_sse_assistant_text(resp).await
            }
            Err(err) => {
                cur_diags = vec![format!("network error: {err}")];
                tokio::time::sleep(Duration::from_secs(3)).await;
                continue;
            }
        };

        if let Some(m) = proxy.take_module_metrics(&unit.id) {
            if m.queue_ms > 0.0 {
                out.queue_ms_samples.push(m.queue_ms);
            }
            if m.decode_tps > 0.0 {
                out.decode_tps_samples.push(m.decode_tps);
            }
            if m.http_status == 429 {
                out.saw_429 = true;
            }
        }
        let _ = status_u16;

        prev_assistant = Some((assistant_reasoning, assistant_content.clone()));

        // Parse & apply raw unescaped ASRF hunks
        let hunks = parse_asrf_hunks(&assistant_content);
        let candidate = match apply_asrf_hunks(&base_texts, &hunks) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("asrf_apply: {e}");
                out.rollbacks_g0 += 1;
                let _ = mvcc.record_rollback(&unit.id, "asrf_apply", &msg);
                learn.record_failure("asrf_apply", &dir_prefix, &msg);
                telemetry.emit(
                    "mvcc_rollback",
                    Some(unit.id.clone()),
                    Some(unit.layer),
                    Some(unit.wave),
                    None,
                    Some(json!({"gate": "asrf_apply", "attempt": attempt, "reason": msg})),
                    None,
                );
                cur_diags = vec![msg];
                continue;
            }
        };

        // Run local 4-stage verification inside blocking pool
        let g_cfg_b = gate_cfg.clone();
        let unit_b = unit.clone();
        let hyg_b = hygiene.clone();
        let verdict = match tokio::task::spawn_blocking(move || {
            verify_candidate(&g_cfg_b, &unit_b, &candidate, &hyg_b)
        })
        .await
        {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                cur_diags = vec![format!("gate runner error: {e}")];
                continue;
            }
            Err(e) => {
                cur_diags = vec![format!("blocking join error: {e}")];
                continue;
            }
        };

        telemetry.emit(
            "gate_result",
            Some(unit.id.clone()),
            Some(unit.layer),
            Some(unit.wave),
            None,
            Some(serde_json::to_value(&verdict).unwrap_or(Value::Null)),
            None,
        );

        if !verdict.passed {
            let fg = verdict.failed_gate.clone().unwrap_or_else(|| "gate".into());
            match fg.as_str() {
                "gate0_syntax" | "gate0_abi" => out.rollbacks_g0 += 1,
                "gate1_tests" => out.rollbacks_g1 += 1,
                "gate2_hygiene" => out.rollbacks_g2 += 1,
                _ => out.rollbacks_g3 += 1,
            }
            let summary_diag = verdict.diagnostics.join(" | ");
            let _ = mvcc.record_rollback(&unit.id, &fg, &summary_diag);
            learn.record_failure(&fg, &dir_prefix, &summary_diag);
            telemetry.emit(
                "mvcc_rollback",
                Some(unit.id.clone()),
                Some(unit.layer),
                Some(unit.wave),
                None,
                Some(json!({"gate": fg, "attempt": attempt, "diagnostics": verdict.diagnostics})),
                None,
            );
            cur_diags = verdict.diagnostics;
            continue;
        }

        // All 4 gates passed -> Atomic MVCC CAS + WAL Commit!
        match mvcc.commit_module(&unit.id, &expected_hashes, &verdict.final_contents) {
            Ok(CommitOutcome::Committed { head_rev, tx_id, .. }) => {
                out.committed = true;
                learn.record_success(&dir_prefix);
                telemetry.emit(
                    "mvcc_commit",
                    Some(unit.id.clone()),
                    Some(unit.layer),
                    Some(unit.wave),
                    Some(proxy.snapshot().uncached_tpm_rolling),
                    Some(json!({
                        "head_rev": head_rev,
                        "tx_id": tx_id,
                        "attempt": attempt,
                        "hygiene_before": verdict.gate2_score_before,
                        "hygiene_after": verdict.gate2_score_after,
                        "ruff_errors": verdict.gate3_ruff_errors,
                        "ty_errors": verdict.gate3_ty_errors,
                        "tests_passed": verdict.gate1_tests_passed
                    })),
                    None,
                );
                break;
            }
            Ok(CommitOutcome::CasConflict { path, .. }) => {
                out.cas_conflicts += 1;
                cur_diags = vec![format!("MVCC CAS conflict on `{path}`; rebasing against latest disk bytes")];
                continue;
            }
            Err(e) => {
                cur_diags = vec![format!("MVCC commit IO error: {e}")];
                continue;
            }
        }
    }

    out
}

async fn read_sse_assistant_text(resp: reqwest::Response) -> (u16, String, String) {
    let status = resp.status().as_u16();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !ct.contains("text/event-stream") {
        let text = resp.text().await.unwrap_or_default();
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let content = v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let reasoning = v["choices"][0]["message"]["reasoning"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return (status, reasoning, content);
        }
        return (status, String::new(), text);
    }

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();

    while let Some(Ok(chunk)) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(d) = v["choices"][0]["delta"]["content"].as_str() {
                        content.push_str(d);
                    }
                    if let Some(r) = v["choices"][0]["delta"]["reasoning"].as_str() {
                        reasoning.push_str(r);
                    }
                }
            }
        }
    }

    (status, reasoning, content)
}

async fn resolve_chat_endpoint(
    client: &reqwest::Client,
    opts: &RunOptions,
) -> (String, String, String, bool) {
    if let Some(oc_base) = &opts.openclaw_base_url {
        let probe_url = format!("{}/models", oc_base.trim_end_matches('/'));
        let mut req = client.get(&probe_url).timeout(Duration::from_secs(2));
        if let Some(tok) = &opts.openclaw_token {
            req = req.bearer_auth(tok);
        }
        if let Ok(resp) = req.send().await {
            if resp.status().as_u16() < 500 {
                let chat_url = format!("{}/chat/completions", oc_base.trim_end_matches('/'));
                let bearer = opts
                    .openclaw_token
                    .clone()
                    .unwrap_or_else(|| "refit-local-gateway".into());
                return (chat_url, bearer, opts.openclaw_model.clone(), true);
            }
        }
    }
    let local_url = format!("http://{}/v1/chat/completions", opts.proxy_addr);
    (local_url, "refit-local-proxy".into(), opts.model.clone(), false)
}

fn load_static_system_prompt(skill_path: &Path) -> String {
    if let Ok(raw) = fs::read_to_string(skill_path) {
        if raw.len() >= 200 {
            return raw;
        }
    }
    "You are refit, a surgical Python refactoring assistant. Emit ONLY raw <<<REFIT_PATCH path=\"...\">>> <<<SEARCH>>> ... <<<REPLACE>>> ... <<<END>>> blocks preserving public function and parameter names while achieving 0 pytest failures, 0 ruff/ty warnings, and codehygiene >= 0.85.".into()
}

/// Runs 5 alternating before/after benchmark pairs (A, B, A, B, ...) inside the OS process sandbox
/// and asserts exact deterministic checksum equality (benchmark-checklist Q5 & Q7).
pub fn run_alternating_os_bench(
    python_bin: &str,
    bench_script: &Path,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchComparison> {
    let mut before_cpu = Vec::with_capacity(pairs);
    let mut after_cpu = Vec::with_capacity(pairs);
    let mut before_rss = Vec::with_capacity(pairs);
    let mut after_rss = Vec::with_capacity(pairs);
    let mut expected_checksum: Option<String> = None;

    for _pair in 0..pairs {
        for (is_after, repo) in [(false, before_repo), (true, after_repo)] {
            let args = vec![
                bench_script.display().to_string(),
                "--repo".to_string(),
                repo.display().to_string(),
                "--iterations".to_string(),
                "120".to_string(),
            ];
            let res = run_sandboxed_cmd(python_bin, &args, repo, &[], false, 30_000)
                .with_context(|| format!("run bench on {}", repo.display()))?;
            if res.exit_code != 0 {
                anyhow::bail!("bench script failed on {}: {}", repo.display(), res.stderr);
            }
            if let Some(last_line) = res.stdout.lines().rev().find(|l| l.trim().starts_with('{')) {
                if let Ok(v) = serde_json::from_str::<Value>(last_line) {
                    if let Some(chk) = v.get("checksum").map(|c| c.to_string()) {
                        if let Some(prev) = &expected_checksum {
                            if prev != &chk {
                                anyhow::bail!("Benchmark checksum mismatch (Q7): before={prev} after={chk}");
                            }
                        } else {
                            expected_checksum = Some(chk);
                        }
                    }
                }
            }
            let cpu = (res.cpu_ms.max(res.wall_ms)) as f64;
            let rss_mb = (res.peak_rss_bytes as f64) / (1024.0 * 1024.0);
            if is_after {
                after_cpu.push(cpu);
                after_rss.push(rss_mb);
            } else {
                before_cpu.push(cpu);
                before_rss.push(rss_mb);
            }
        }
    }

    let b_cpu = median_f64(&mut before_cpu).max(1.0);
    let a_cpu = median_f64(&mut after_cpu).max(1.0);
    let b_rss = median_f64(&mut before_rss).max(1.0);
    let a_rss = median_f64(&mut after_rss).max(1.0);

    let cpu_ratio = a_cpu / b_cpu;
    let rss_ratio = a_rss / b_rss;

    Ok(BenchComparison {
        pairs,
        before_cpu_ms_median: b_cpu,
        after_cpu_ms_median: a_cpu,
        cpu_ratio,
        before_rss_mb_median: b_rss,
        after_rss_mb_median: a_rss,
        rss_ratio,
        passed_anti_regression: cpu_ratio <= 1.05 && rss_ratio <= 1.05,
    })
}

fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}
