use crate::gates::{
    GateConfig, read_gates, scan_repo, audit_repo,
    stage_hardlinks, run_gates,
};
use crate::graph::{
    apply_patches, deps_context, critical_depths,
    parse_patches, build_packet, units_interfere,
};
use crate::hygiene::HygieneScorer;
use crate::learn::LearnStore;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::platform::{
    stream_http, os_tag, sandbox_env, now_ms, backoff_ms,
    run_cmd, self_metrics,
};
use crate::proxy::{CacheMode, ProxyConfig, ProxyState, start_proxy};
use crate::research::WebClient;
use crate::state::RunState;
use crate::telemetry::TelemetryHub;
use crate::types::{BenchReport, ModuleUnit, RunSummary};
use crate::worktree::{open_worktree, diffstat};
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
    pub bench_cmd: Option<String>,
    pub baseline_copy: Option<PathBuf>,
    pub resume: bool,
    pub initial_k: usize,
    pub max_k: usize,
    pub max_retries: u32,
    pub hold_server_secs: u64,
    pub max_units: Option<usize>,
    pub worktree: bool,
    pub include_dirty_worktree: bool,
    pub cache_retention: CacheMode,
    pub keep_warm: bool,
    pub preserve_tools: bool,
    pub strict_polyglot_tools: bool,
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
    formatted_files: u64,
    saw_429: bool,
    queue_ms_samples: Vec<f64>,
    decode_tps_samples: Vec<f64>,
}

pub async fn execute_run(opts: RunOptions) -> Result<RunSummary> {
    let wall_start = Instant::now();
    let run_id = format!("refit-{}-{}", now_ms(), std::process::id());

    let isolated_wt = if opts.worktree {
        Some(open_worktree(
            &opts.repo_root,
            &run_id,
            opts.include_dirty_worktree,
        )?)
    } else {
        None
    };
    let active_repo = isolated_wt
        .as_ref()
        .map(|w| w.worktree_path.clone())
        .unwrap_or_else(|| opts.repo_root.clone());

    let static_system_prompt = read_prompt(&opts.skill_path);
    let lang_prefixes = prefix_map(&static_system_prompt);
    // Only enforce proxy auth_token when explicitly configured via REFIT_PROXY_TOKEN so unpatched OpenClaw & local HUD work cleanly out-of-the-box.
    let explicit_proxy_token = std::env::var("REFIT_PROXY_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let proxy_client_token = explicit_proxy_token
        .clone()
        .unwrap_or_else(|| "refit-local-proxy".to_string());

    let telemetry = TelemetryHub::open_hub(opts.telemetry_path.clone(), opts.resume)?;
    let proxy_cfg = ProxyConfig {
        listen_addr: opts.proxy_addr.clone(),
        upstream_base_url: opts.upstream_base_url.clone(),
        api_key: opts.api_key.clone(),
        model_override: opts.model.clone(),
        ui_index_path: opts.ui_index_path.clone(),
        static_prefix_override: Some(static_system_prompt.clone()),
        cache_retention: opts.cache_retention,
        keep_warm: opts.keep_warm,
        auth_token: explicit_proxy_token,
        lang_prefixes,
        preserve_tools: opts.preserve_tools,
    };
    let (proxy_state, proxy_handle) = start_proxy(proxy_cfg, telemetry.clone()).await?;
    proxy_state.bind_repo(active_repo.clone());

    let hygiene = Arc::new(HygieneScorer::open_persistent(&active_repo));
    let mut gate_cfg_raw = read_gates(&active_repo);
    gate_cfg_raw.strict_polyglot_tools = opts.strict_polyglot_tools;
    let gate_cfg = Arc::new(gate_cfg_raw);
    let mvcc = Arc::new(MvccStore::open(&active_repo, &run_id)?);
    let learn = Arc::new(LearnStore::open(&active_repo)?);

    let mut already_committed: HashMap<String, BTreeMap<String, String>> = HashMap::new();
    let mut rolled_back_on_resume = 0usize;
    if opts.resume {
        let rep = mvcc.recover_wal()?;
        rolled_back_on_resume = rep.rolled_back_txs;
        already_committed = rep.committed_modules;
        tracing::info!(
            "WAL resume: rolled_back_txs={} committed_modules={} head_rev={}",
            rep.rolled_back_txs,
            already_committed.len(),
            rep.head_rev
        );
    }

    let (scan_bundle, _scan_cache_hit) = scan_repo(&active_repo, true, &hygiene)?;
    let detected_languages = scan_bundle.detected_languages;
    let plan = scan_bundle.plan;
    let baseline_audit = scan_bundle.audit;
    let shared_state = RunState::new(plan.total_loc);
    telemetry.bind_state(shared_state.clone());
    proxy_state.attach_state(shared_state.clone());

    // Wire repo-scoped WebClient into Phase 0 manifest audit AND retry diagnosis.
    let research = WebClient::for_repo(&active_repo, Some(shared_state.clone()));
    let phase0_research_block = if research.is_online() {
        let dep_results = research.audit_deps(&active_repo).await;
        WebClient::format_snippets(&dep_results)
    } else {
        String::new()
    };
    let phase0_research_arc = Arc::new(phase0_research_block);
    let all_rel_files: Vec<String> = plan.units.iter().flat_map(|u| u.files.clone()).collect();

    let client = stream_http("refit-runner/0.5.0", 15, 600, 32)?;
    let (chat_url, chat_bearer, chat_model, via_openclaw) =
        chat_endpoint(&client, &opts, &proxy_client_token).await;

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
            "research_online": research.is_online(),
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

    let mut pending: Vec<ModuleUnit> = Vec::new();
    let mut completed_ids: HashSet<String> = HashSet::new();
    let mut skipped_resumed_units = 0usize;

    // Filter clean unchanged & resumed units FIRST, then apply --max-units.
    for u in plan.units.clone() {
        let unit_has_diags = u
            .files
            .iter()
            .any(|f| !baseline_audit.per_file_diagnostics.get(f).map(|v| v.is_empty()).unwrap_or(true));
        if !unit_has_diags && u.baseline_hygiene >= gate_cfg.min_hygiene_target {
            skipped_resumed_units += 1;
            completed_ids.insert(u.id.clone());
            continue;
        }
        if opts.resume {
            if let Some(saved_hashes) = already_committed.get(&u.id) {
                let all_match = u.files.iter().all(|rel| {
                    saved_hashes
                        .get(rel)
                        .and_then(|expected| {
                            MvccStore::hash_file(&active_repo.join(rel))
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

    // Sort by (wave, -critical_path_depth, id) FIRST, then apply --max-units so high-priority leaves execute first.
    let cp_depths = critical_depths(&plan.units);
    pending.sort_by(|a, b| {
        a.wave
            .cmp(&b.wave)
            .then_with(|| {
                cp_depths
                    .get(&b.id)
                    .copied()
                    .unwrap_or(0)
                    .cmp(&cp_depths.get(&a.id).copied().unwrap_or(0))
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    if let Some(lim) = opts.max_units {
        if pending.len() > lim {
            for dropped in pending.drain(lim..) {
                completed_ids.insert(dropped.id);
            }
        }
    }

    for u in &plan.units {
        if !pending.iter().any(|p| p.id == u.id) {
            completed_ids.insert(u.id.clone());
        }
    }

    let mut current_k = opts.initial_k.clamp(2, opts.max_k.max(2));
    let mut in_flight: HashMap<String, ModuleUnit> = HashMap::new();
    let mut task_to_unit: HashMap<tokio::task::Id, ModuleUnit> = HashMap::new();
    let mut join_set: JoinSet<WorkerOutcome> = JoinSet::new();

    let mut committed_units = skipped_resumed_units;
    let mut cas_conflicts = 0u64;
    let mut rollbacks_g0 = 0u64;
    let mut rollbacks_g1 = 0u64;
    let mut rollbacks_g2 = 0u64;
    let mut rollbacks_g3 = 0u64;
    let mut retries_total = 0u64;
    let mut formatted_files_total = 0u64;
    let mut all_queue_ms: Vec<f64> = Vec::new();
    let mut all_decode_tps: Vec<f64> = Vec::new();

    let diag_map = Arc::new(baseline_audit.per_file_diagnostics.clone());
    let sys_prompt_arc = Arc::new(static_system_prompt);
    let all_units_arc = Arc::new(plan.units.clone());

    loop {
        shared_state.await_resume().await;
        if shared_state.abort_asked() {
            pending.clear();
        }
        if let Some(k_ov) = shared_state.k_override() {
            current_k = k_ov.clamp(1, opts.max_k.max(2));
        }

        // Explicitly unblock dependents when a unit is live-steered to skip.
        let mut skipped_now = Vec::new();
        pending.retain(|u| {
            if shared_state.is_skipped(&u.id) {
                skipped_now.push(u.id.clone());
                false
            } else {
                true
            }
        });
        for sid in skipped_now {
            completed_ids.insert(sid.clone());
            telemetry.emit(
                "unit_skipped",
                Some(sid),
                None,
                None,
                None,
                Some(json!({"reason": "live_steer_skip"})),
                None,
            );
        }

        while in_flight.len() < current_k {
            // Preserve unready prioritized units in queue until deps/interference clear.
            let steered_pos = shared_state
                .pop_ready(|pid| {
                    pending.iter().any(|cand| {
                        cand.id == pid
                            && cand.deps.iter().all(|d| completed_ids.contains(d))
                            && !in_flight.values().any(|active| units_interfere(cand, active))
                    })
                })
                .and_then(|pid| pending.iter().position(|cand| cand.id == pid));

            let next_pos = steered_pos.or_else(|| {
                pending.iter().position(|cand| {
                    let deps_done = cand.deps.iter().all(|d| completed_ids.contains(d));
                    let disjoint = !in_flight.values().any(|active| units_interfere(cand, active));
                    deps_done && disjoint
                })
            });

            let Some(idx) = next_pos else { break };
            let unit = pending.remove(idx);
            let deps_abi = Arc::new(deps_context(&unit, &all_units_arc, &active_repo));
            in_flight.insert(unit.id.clone(), unit.clone());
            shared_state.step_inflight(1);

            let run_id_c = run_id.clone();
            let repo_c = active_repo.clone();
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
            let state_c = shared_state.clone();
            let research_c = research.clone();
            let p0_res_c = phase0_research_arc.clone();
            let tel_c = telemetry.clone();
            let max_retries = opts.max_retries;
            let k_snap = current_k;

            let unit_for_task = unit.clone();
            let abort_h = join_set.spawn(async move {
                run_unit(
                    run_id_c,
                    repo_c,
                    unit,
                    client_c,
                    url_c,
                    bearer_c,
                    model_c,
                    sys_c,
                    deps_abi,
                    diag_c,
                    hygiene_c,
                    gate_c,
                    mvcc_c,
                    learn_c,
                    proxy_c,
                    state_c,
                    research_c,
                    p0_res_c,
                    tel_c,
                    max_retries,
                    k_snap,
                )
                .await
            });
            task_to_unit.insert(abort_h.id(), unit_for_task);
        }

        if in_flight.is_empty() {
            if pending.is_empty() {
                break;
            }
            // Never silently drop unexecuted units as completed.
            for stuck in pending.drain(..) {
                let unmet: Vec<String> = stuck
                    .deps
                    .iter()
                    .filter(|d| !completed_ids.contains(*d))
                    .cloned()
                    .collect();
                let reason = format!("unsatisfied_dependencies: {}", unmet.join(", "));
                shared_state.mark_blocked(&stuck.id, &reason);
                telemetry.emit(
                    "unit_blocked",
                    Some(stuck.id.clone()),
                    Some(stuck.layer),
                    Some(stuck.wave),
                    None,
                    Some(json!({"reason": reason, "unmet_deps": unmet})),
                    None,
                );
            }
            break;
        }

        match join_set.join_next().await {
            Some(Ok(out)) => {
                in_flight.remove(&out.unit.id);
                task_to_unit.retain(|_, u| u.id != out.unit.id);
                shared_state.step_inflight(-1);
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
                formatted_files_total += out.formatted_files;
                all_queue_ms.extend(out.queue_ms_samples);
                all_decode_tps.extend(out.decode_tps_samples);

                if out.saw_429 {
                    current_k = (current_k / 2).max(2);
                } else if out.committed && current_k < opts.max_k {
                    current_k += 1;
                }
            }
            Some(Err(join_err)) => {
                // Recover cleanly from worker panics so `in_flight` never leaks into an infinite busy-loop.
                if let Some(panicked) = task_to_unit.remove(&join_err.id()) {
                    in_flight.remove(&panicked.id);
                    shared_state.step_inflight(-1);
                    let reason = format!("worker_task_panicked: {join_err}");
                    shared_state.mark_blocked(&panicked.id, &reason);
                    completed_ids.insert(panicked.id.clone());
                    telemetry.emit(
                        "unit_blocked",
                        Some(panicked.id.clone()),
                        Some(panicked.layer),
                        Some(panicked.wave),
                        None,
                        Some(json!({"reason": reason})),
                        None,
                    );
                }
            }
            None => {
                // Safety valve: if join_set is drained, clear any residual tracking and exit cleanly.
                for (_, orphaned) in in_flight.drain() {
                    shared_state.step_inflight(-1);
                    shared_state.mark_blocked(&orphaned.id, "worker_join_set_drained");
                }
                break;
            }
        }
    }

    // Re-walk source files post-run so newly created/renamed split files are included in final audit & loc_after.
    let final_rel_files: Vec<String> = crate::platform::walk_files(&active_repo, None)
        .into_iter()
        .filter_map(|p| {
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            if !crate::graph::supported_ext(&ext) {
                return None;
            }
            let rel = p.strip_prefix(&active_repo).ok()?.to_string_lossy().replace('\\', "/");
            if rel.starts_with("tests/") || rel.ends_with("__init__.py") {
                return None;
            }
            Some(rel)
        })
        .collect();
    let audit_rel_files = if final_rel_files.is_empty() {
        all_rel_files
    } else {
        final_rel_files
    };

    let final_audit = if committed_units == skipped_resumed_units {
        baseline_audit.clone()
    } else {
        audit_repo(&gate_cfg, &audit_rel_files, &hygiene)?
    };
    hygiene.flush_cache();

    let bench = match (&opts.bench_script, &opts.bench_cmd, &opts.baseline_copy) {
        (_, Some(cmd_tpl), Some(base_dir)) if base_dir.exists() => {
            match bench_command(cmd_tpl, base_dir, &active_repo, 5) {
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
                    tracing::warn!("Alternating generic bench skipped due to error: {e}");
                    None
                }
            }
        }
        (Some(script), None, Some(base_dir)) if script.exists() && base_dir.exists() => {
            match bench_python(&gate_cfg.python_bin, script, base_dir, &active_repo, 5) {
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

    let loc_after: usize = audit_rel_files
        .iter()
        .map(|rel| {
            fs::read_to_string(active_repo.join(rel))
                .map(|s| s.lines().count())
                .unwrap_or(0)
        })
        .sum();
    let st_digest = shared_state.digest();
    let (_, tel_raw_bytes, tel_zstd_bytes) = telemetry
        .flush_zstd()
        .unwrap_or_else(|_| (PathBuf::new(), telemetry.bytes_written(), 0));
    let wt_meta = isolated_wt.as_ref().map(|w| {
        json!({
            "core_repo": w.core_repo.display().to_string(),
            "branch": w.branch,
            "carried_dirty_changes": w.carried_dirty_changes,
            "diff_stat": diffstat(w)
        })
    });

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
        host_os: os_tag(),
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
        loc_before: plan.total_loc,
        loc_after,
        telemetry_raw_bytes: tel_raw_bytes,
        telemetry_zstd_bytes: tel_zstd_bytes,
        stage_hardlinked_files: stage_hardlinks(),
        research_queries: st_digest.research_queries,
        research_cache_hits: st_digest.research_cache_hits,
        worktree_branch: isolated_wt.as_ref().map(|w| w.branch.clone()),
        worktree_path: isolated_wt
            .as_ref()
            .map(|w| w.worktree_path.display().to_string()),
        ready_for_manual_merge: isolated_wt.is_some(),
        detected_languages,
        blocked_units: shared_state.blocked_count(),
        formatted_files_total,
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
            "learn_helpful": learn.stats().1,
            "tracked_units": st_digest.tracked_units,
            "blocked_units": shared_state.blocked_count(),
            "telemetry_dedup_skipped": telemetry.dedup_skipped(),
            "worktree": wt_meta
        })),
        Some(summary.clone()),
    );

    if opts.hold_server_secs > 0 {
        tokio::time::sleep(Duration::from_secs(opts.hold_server_secs)).await;
    }

    proxy_handle.abort();
    Ok(summary)
}

fn guess_lang(unit: &ModuleUnit) -> &'static str {
    let ext = unit
        .files
        .first()
        .and_then(|f| Path::new(f).extension())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "py" => "py",
        "rs" => "rs",
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => "ts",
        "go" => "go",
        _ => "polyglot",
    }
}

fn pick_effort(unit: &ModuleUnit, attempt: u32, diags: &[String]) -> &'static str {
    let has_logic_or_test_fail = diags.iter().any(|d| {
        let l = d.to_ascii_lowercase();
        l.contains("gate 1") || l.contains("pytest") || l.contains("failed") || l.contains("assert")
    });
    if unit.files.len() > 1 || has_logic_or_test_fail || attempt >= 2 {
        "medium"
    } else {
        "low"
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_unit(
    run_id: String,
    repo_root: PathBuf,
    mut unit: ModuleUnit,
    client: reqwest::Client,
    chat_url: String,
    chat_bearer: String,
    chat_model: String,
    static_sys: Arc<String>,
    deps_abi: Arc<String>,
    baseline_diags: Arc<HashMap<String, Vec<String>>>,
    hygiene: Arc<HygieneScorer>,
    gate_cfg: Arc<GateConfig>,
    mvcc: Arc<MvccStore>,
    learn: Arc<LearnStore>,
    proxy: Arc<ProxyState>,
    shared_state: Arc<RunState>,
    research: Arc<WebClient>,
    phase0_research: Arc<String>,
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
        formatted_files: 0,
        saw_429: false,
        queue_ms_samples: Vec::new(),
        decode_tps_samples: Vec::new(),
    };

    let unit_lang = guess_lang(&unit);
    let dir_prefix = Path::new(&unit.files[0])
        .parent()
        .map(|p| p.display().to_string().replace('\\', "/"))
        .unwrap_or_else(|| "*".into());

    let mut cur_diags: Vec<String> = unit
        .files
        .iter()
        .flat_map(|f| baseline_diags.get(f).cloned().unwrap_or_default())
        .collect();

    let mut initial_user_msg: Option<String> = None;
    let mut prev_assistant: Option<(String, String)> = None;
    let mut prior_failures: Vec<String> = Vec::new();
    let mut retry_research_note = String::new();
    let mut applied_learn_keys: Vec<String> = Vec::new();

    for attempt in 1..=(max_retries + 1) {
        if attempt > 1 {
            out.retries += 1;
        }
        shared_state.set_phase(&unit.id, "dispatching", attempt, unit.loc);

        let mut base_texts = BTreeMap::new();
        let mut expected_hashes = BTreeMap::new();
        for rel in &unit.files {
            if let Ok(txt) = fs::read_to_string(repo_root.join(rel)) {
                expected_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
                base_texts.insert(rel.clone(), txt);
            }
        }
        unit.base_blake3 = expected_hashes.clone();

        let (learn_suffix, learn_keys) = learn.rule_suffix(&dir_prefix);
        for lk in learn_keys {
            if !applied_learn_keys.contains(&lk) {
                applied_learn_keys.push(lk);
            }
        }
        let slice_budget = if unit.files.len() > 1 { 24_000 } else { 12_000 };

        // Keep messages[0..1] byte-identical across ALL retry attempts (2, 3, 4...) for KV-cache hits.
        let mut messages = vec![json!({ "role": "system", "content": static_sys.as_str() })];
        if attempt == 1 || initial_user_msg.is_none() {
            let unit_packet =
                match build_packet(&repo_root, &unit, &cur_diags, &learn_suffix, slice_budget) {
                    Ok(p) => p,
                    Err(_) => break,
                };
            let mut prompt = String::new();
            if !deps_abi.is_empty() {
                prompt.push_str(&deps_abi);
                prompt.push('\n');
            }
            if !phase0_research.is_empty() {
                prompt.push_str(&phase0_research);
                prompt.push('\n');
            }
            prompt.push_str(&unit_packet);
            messages.push(json!({ "role": "user", "content": prompt }));
            initial_user_msg = Some(prompt);
        } else {
            messages.push(json!({ "role": "user", "content": initial_user_msg.as_ref().unwrap() }));
            if let Some((prev_r, prev_c)) = &prev_assistant {
                // Bound echoed failed patch so retries don't create an attention error echo chamber.
                let compact_c: String = prev_c.chars().take(2400).collect();
                let mut asst = json!({ "role": "assistant", "content": compact_c });
                if !prev_r.is_empty() {
                    let compact_r: String = prev_r.chars().take(600).collect();
                    asst["reasoning"] = json!(compact_r);
                }
                messages.push(asst);
            }
            let history_block = if prior_failures.is_empty() {
                String::new()
            } else {
                format!(
                    "Negative constraints from earlier failed attempts (DO NOT repeat):\n- {}\n",
                    prior_failures.join("\n- ")
                )
            };
            let retry_feedback = format!(
                "[attempt={attempt}] Previous patch failed verification and was ROLLED BACK (disk UNCHANGED). Every <<<SEARCH>>> block MUST uniquely match the ORIGINAL file verbatim.\n{history_block}Fix EVERY diagnostic below:\n- {}\n{}{}",
                cur_diags.join("\n- "),
                learn_suffix,
                retry_research_note
            );
            messages.push(json!({ "role": "user", "content": retry_feedback }));
        }

        telemetry.emit(
            "wave_dispatch",
            Some(unit.id.clone()),
            Some(unit.layer),
            Some(unit.wave),
            Some(proxy.snapshot().uncached_tpm_rolling),
            Some(json!({
                "attempt": attempt,
                "k": current_k,
                "lang": unit_lang
            })),
            None,
        );

        let effort = pick_effort(&unit, attempt, &cur_diags);
        let mut req_body = json!({
            "model": chat_model,
            "messages": messages,
            "stream": true,
            "max_completion_tokens": if unit.files.len() > 1 { 5500 } else { 3200 },
            "temperature": 1.0,
            "reasoning_effort": effort,
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
            .header("x-refit-lang", unit_lang)
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
                    let err_line = format!("HTTP {code}: {}", body.chars().take(160).collect::<String>());
                    cur_diags = vec![err_line];
                    let wait_ms = backoff_ms(attempt, 800, 15_000, None, now_ms());
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
                match collect_sse(resp, Duration::from_secs(60)).await {
                    Ok(triple) => triple,
                    Err(e) => {
                        cur_diags = vec![format!("stream stalled or interrupted: {e}")];
                        let wait_ms = backoff_ms(attempt, 500, 10_000, None, now_ms());
                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                        continue;
                    }
                }
            }
            Err(err) => {
                cur_diags = vec![format!("network error: {err}")];
                let wait_ms = backoff_ms(attempt, 500, 10_000, None, now_ms());
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                continue;
            }
        };

        if let Some(m) = proxy.take_metrics(&unit.id) {
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

        let hunks = parse_patches(&assistant_content);
        let applied = match apply_patches(&base_texts, &hunks) {
            Ok(a) => a,
            Err(e) => {
                let msg = format!("asrf_apply: {e}");
                out.rollbacks_g0 += 1;
                let _ = mvcc.record_rollback(&unit.id, "asrf_apply", &msg);
                learn.record_failure("asrf_apply", &dir_prefix, &msg);
                prior_failures.push(format!("Attempt {attempt} failed at asrf_apply: {}", msg.chars().take(120).collect::<String>()));
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

        shared_state.set_phase(&unit.id, "verifying", attempt, unit.loc);
        let hl_before = stage_hardlinks();
        let g_cfg_b = gate_cfg.clone();
        let unit_b = unit.clone();
        let hyg_b = hygiene.clone();
        let cand_updated = applied.updated;
        let cand_deleted = applied.deleted;
        let verdict = match tokio::task::spawn_blocking(move || {
            run_gates(&g_cfg_b, &unit_b, &cand_updated, &cand_deleted, &hyg_b)
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
        shared_state.add_hardlinks(stage_hardlinks().saturating_sub(hl_before));
        out.formatted_files += verdict.formatted_files as u64;

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
                "gate0_syntax" | "gate0_abi" | "gate0_rust_check" | "gate0_ts_check" | "gate0_go_vet" => {
                    out.rollbacks_g0 += 1;
                }
                "gate1_tests" => out.rollbacks_g1 += 1,
                "gate2_hygiene" => out.rollbacks_g2 += 1,
                _ => out.rollbacks_g3 += 1,
            }
            let summary_diag = verdict.diagnostics.join(" | ");
            let _ = mvcc.record_rollback(&unit.id, &fg, &summary_diag);
            learn.record_failure(&fg, &dir_prefix, &summary_diag);
            prior_failures.push(format!(
                "Attempt {attempt} failed at {fg}: {}",
                summary_diag.chars().take(140).collect::<String>()
            ));
            // Bounded KeenAble web error-diagnosis on first Gate 1/3/0 failure when online.
            if attempt == 1
                && research.is_online()
                && retry_research_note.is_empty()
                && matches!(fg.as_str(), "gate1_tests" | "gate3_ty" | "gate0_rust_check" | "gate0_ts_check")
            {
                let topic: String = summary_diag.chars().take(120).collect();
                let diag_query = format!("{unit_lang} fix error {topic}");
                if let Ok(res) = research.search_web(&diag_query, 3).await {
                    retry_research_note = format!(
                        "\n{}",
                        WebClient::format_snippets(&[res])
                    );
                }
            }
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

        match mvcc.commit_tx(
            &unit.id,
            &expected_hashes,
            &verdict.final_contents,
            &verdict.deleted_files,
        ) {
            Ok(CommitOutcome::Committed {
                head_rev,
                tx_id,
                new_hashes,
            }) => {
                out.committed = true;
                let after_loc: usize = verdict
                    .final_contents
                    .values()
                    .map(|b| b.iter().filter(|&&c| c == b'\n').count())
                    .sum();
                let first_hash = new_hashes.values().next().cloned().unwrap_or_default();
                shared_state.record_commit(
                    &unit.id,
                    attempt,
                    unit.loc,
                    after_loc,
                    verdict.gate2_score_after,
                    first_hash,
                );
                learn.credit_keys(&dir_prefix, &applied_learn_keys);
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
                        "tests_passed": verdict.gate1_tests_passed,
                        "formatted_files": verdict.formatted_files
                    })),
                    None,
                );
                break;
            }
            Ok(CommitOutcome::CasConflict {
                path,
                expected,
                actual,
            }) => {
                out.cas_conflicts += 1;
                cur_diags = vec![format!(
                    "MVCC CAS conflict on `{path}` (expected {expected}, disk {actual}); rebasing against latest disk bytes"
                )];
                initial_user_msg = None;
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

/// Read an SSE stream into raw bytes and decode complete lines to preserve split UTF-8 characters.
async fn collect_sse(
    resp: reqwest::Response,
    chunk_idle_timeout: Duration,
) -> Result<(u16, String, String)> {
    let status = resp.status().as_u16();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !ct.contains("text/event-stream") {
        let text = resp.text().await.context("read JSON response body")?;
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            let content = v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let reasoning = v["choices"][0]["message"]["reasoning"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return Ok((status, reasoning, content));
        }
        return Ok((status, String::new(), text));
    }

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut byte_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut stream = resp.bytes_stream();

    loop {
        let next_chunk = tokio::time::timeout(chunk_idle_timeout, stream.next())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "SSE chunk idle timeout after {}ms",
                    chunk_idle_timeout.as_millis()
                )
            })?;
        let Some(item) = next_chunk else {
            break;
        };
        let chunk = item.context("SSE transport stream error mid-response")?;
        byte_buf.extend_from_slice(&chunk);
        while let Some(nl) = byte_buf.iter().position(|&b| b == b'\n') {
            let raw_line: Vec<u8> = byte_buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&raw_line).trim().to_string();
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
    if !byte_buf.is_empty() {
        let line = String::from_utf8_lossy(&byte_buf).trim().to_string();
        if let Some(data) = line.strip_prefix("data: ") {
            if data != "[DONE]" {
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

    Ok((status, reasoning, content))
}

async fn chat_endpoint(
    client: &reqwest::Client,
    opts: &RunOptions,
    proxy_token: &str,
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
    (local_url, proxy_token.to_string(), opts.model.clone(), false)
}

fn read_prompt(skill_path: &Path) -> String {
    if let Ok(raw) = fs::read_to_string(skill_path) {
        if raw.len() >= 200 {
            return raw;
        }
    }
    "You are refit, a surgical polyglot untangling engineer. Emit raw <<<REFIT_PATCH path=\"...\">>> <<<SEARCH>>> ... <<<REPLACE>>> ... <<<END>>> blocks preserving public exported signatures while passing all local verification gates and exact codehygiene v0.7.0 >= 0.85.".into()
}

fn prefix_map(base_skill: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        "py".into(),
        format!(
            "{base_skill}\n\n[Language Profile: Python]\n- Tooling: ruff format + ruff check + ty --error-on-warning + pytest.\n- Use explicit PEP 484 annotations, Optional narrowing, and `None` defaults instead of mutable []/{{}}."
        ),
    );
    m.insert(
        "rs".into(),
        format!(
            "{base_skill}\n\n[Language Profile: Rust]\n- Tooling: rustfmt + cargo check + cargo test.\n- Prefer early `?` propagation and small helpers over deep match/if nesting; avoid `.unwrap()` in production code."
        ),
    );
    m.insert(
        "ts".into(),
        format!(
            "{base_skill}\n\n[Language Profile: TypeScript/JavaScript]\n- Tooling: biome + tsc --noEmit + node --test.\n- Prefer strict types, narrow discriminated unions, and clean ESM imports."
        ),
    );
    m.insert(
        "go".into(),
        format!(
            "{base_skill}\n\n[Language Profile: Go]\n- Tooling: gofmt + go vet + go test ./...\n- Always check returned errors explicitly (`if err != nil`); never discard errors with `_ = err`."
        ),
    );
    m
}

pub fn bench_python(
    python_bin: &str,
    bench_script: &Path,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchReport> {
    let mut before_cpu = Vec::with_capacity(pairs);
    let mut after_cpu = Vec::with_capacity(pairs);
    let mut before_wall = Vec::with_capacity(pairs);
    let mut after_wall = Vec::with_capacity(pairs);
    let mut before_rss = Vec::with_capacity(pairs);
    let mut after_rss = Vec::with_capacity(pairs);
    let mut expected_checksum: Option<String> = None;
    let env = sandbox_env(&[]);

    for _pair in 0..pairs {
        for (is_after, repo) in [(false, before_repo), (true, after_repo)] {
            let args = vec![
                bench_script.display().to_string(),
                "--repo".to_string(),
                repo.display().to_string(),
                "--iterations".to_string(),
                "120".to_string(),
            ];
            let res = run_cmd(python_bin, &args, repo, &env, true, 30_000)
                .with_context(|| format!("run bench on {}", repo.display()))?;
            if res.exit_code != 0 {
                anyhow::bail!("bench script failed on {}: {}", repo.display(), res.stderr);
            }
            verify_digest(&res.stdout, &mut expected_checksum)?;
            // Report real CPU separately from wall-clock I/O wait.
            let cpu = if res.cpu_ms > 0 { res.cpu_ms as f64 } else { res.wall_ms as f64 };
            let wall = res.wall_ms as f64;
            let rss_mb = (res.peak_rss_bytes as f64) / (1024.0 * 1024.0);
            if is_after {
                after_cpu.push(cpu);
                after_wall.push(wall);
                after_rss.push(rss_mb);
            } else {
                before_cpu.push(cpu);
                before_wall.push(wall);
                before_rss.push(rss_mb);
            }
        }
    }

    finish_bench(
        pairs,
        &mut before_cpu,
        &mut after_cpu,
        &mut before_wall,
        &mut after_wall,
        &mut before_rss,
        &mut after_rss,
    )
}

/// Run alternating OS benchmarks using a command template with `{repo}` substitution.
pub fn bench_command(
    cmd_template: &str,
    before_repo: &Path,
    after_repo: &Path,
    pairs: usize,
) -> Result<BenchReport> {
    let mut before_cpu = Vec::with_capacity(pairs);
    let mut after_cpu = Vec::with_capacity(pairs);
    let mut before_wall = Vec::with_capacity(pairs);
    let mut after_wall = Vec::with_capacity(pairs);
    let mut before_rss = Vec::with_capacity(pairs);
    let mut after_rss = Vec::with_capacity(pairs);
    let mut expected_checksum: Option<String> = None;
    let env = sandbox_env(&[]);

    for _pair in 0..pairs {
        for (is_after, repo) in [(false, before_repo), (true, after_repo)] {
            let expanded = cmd_template.replace("{repo}", &repo.display().to_string());
            let (prog, args) = if cfg!(windows) {
                ("cmd", vec!["/C".to_string(), expanded])
            } else {
                ("sh", vec!["-c".to_string(), expanded])
            };
            let res = run_cmd(prog, &args, repo, &env, true, 30_000)
                .with_context(|| format!("run generic bench on {}", repo.display()))?;
            if res.exit_code != 0 {
                anyhow::bail!("bench cmd failed on {}: {}", repo.display(), res.stderr);
            }
            verify_digest(&res.stdout, &mut expected_checksum)?;
            let cpu = if res.cpu_ms > 0 { res.cpu_ms as f64 } else { res.wall_ms as f64 };
            let wall = res.wall_ms as f64;
            let rss_mb = (res.peak_rss_bytes as f64) / (1024.0 * 1024.0);
            if is_after {
                after_cpu.push(cpu);
                after_wall.push(wall);
                after_rss.push(rss_mb);
            } else {
                before_cpu.push(cpu);
                before_wall.push(wall);
                before_rss.push(rss_mb);
            }
        }
    }

    finish_bench(
        pairs,
        &mut before_cpu,
        &mut after_cpu,
        &mut before_wall,
        &mut after_wall,
        &mut before_rss,
        &mut after_rss,
    )
}

fn verify_digest(stdout: &str, expected: &mut Option<String>) -> Result<()> {
    if let Some(last_line) = stdout.lines().rev().find(|l| l.trim().starts_with('{')) {
        if let Ok(v) = serde_json::from_str::<Value>(last_line) {
            if let Some(chk) = v.get("checksum").map(|c| c.to_string()) {
                if let Some(prev) = expected {
                    if prev != &chk {
                        anyhow::bail!("Benchmark checksum mismatch (Q7): before={prev} after={chk}");
                    }
                } else {
                    *expected = Some(chk);
                }
            }
        }
    }
    Ok(())
}

fn finish_bench(
    pairs: usize,
    before_cpu: &mut [f64],
    after_cpu: &mut [f64],
    before_wall: &mut [f64],
    after_wall: &mut [f64],
    before_rss: &mut [f64],
    after_rss: &mut [f64],
) -> Result<BenchReport> {
    let b_cpu = median_f64(before_cpu).max(1.0);
    let a_cpu = median_f64(after_cpu).max(1.0);
    let b_wall = median_f64(before_wall).max(1.0);
    let a_wall = median_f64(after_wall).max(1.0);
    let b_rss = median_f64(before_rss).max(1.0);
    let a_rss = median_f64(after_rss).max(1.0);

    let cpu_ratio = a_cpu / b_cpu;
    let rss_ratio = a_rss / b_rss;

    Ok(BenchReport {
        pairs,
        before_cpu_ms_median: b_cpu,
        after_cpu_ms_median: a_cpu,
        cpu_ratio,
        before_wall_ms_median: b_wall,
        after_wall_ms_median: a_wall,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn probe_75_sse_error() {
        let srv = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = srv.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = srv.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf).await;
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
            let frame = format!("{:x}\r\n{}\r\n", chunk.len(), chunk);
            let _ = sock.write_all(hdr.as_bytes()).await;
            let _ = sock.write_all(frame.as_bytes()).await;
            let _ = sock.flush().await;
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/stream"))
            .send()
            .await
            .unwrap();
        let err = collect_sse(resp, Duration::from_millis(120)).await;
        assert!(err.is_err(), "mid-stream stall must return Err rather than silent truncated EOF");
    }

    #[tokio::test]
    async fn probe_94_sse_utf8() {
        let srv = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = srv.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = srv.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf).await;
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
            let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"café 🚀 日本語\"}}]}\n\n".as_bytes();
            // Intentionally split right inside the 4-byte UTF-8 emoji sequence!
            let emoji_pos = payload.windows(4).position(|w| w == "🚀".as_bytes()).unwrap();
            let part1 = &payload[..emoji_pos + 2];
            let part2 = &payload[emoji_pos + 2..];
            let _ = sock.write_all(hdr.as_bytes()).await;
            let _ = sock.write_all(format!("{:x}\r\n", part1.len()).as_bytes()).await;
            let _ = sock.write_all(part1).await;
            let _ = sock.write_all(b"\r\n").await;
            let _ = sock.flush().await;
            tokio::time::sleep(Duration::from_millis(15)).await;
            let _ = sock.write_all(format!("{:x}\r\n", part2.len()).as_bytes()).await;
            let _ = sock.write_all(part2).await;
            let _ = sock.write_all(b"\r\n0\r\n\r\n").await;
            let _ = sock.flush().await;
        });

        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/stream"))
            .send()
            .await
            .unwrap();
        let (_, _, text) = collect_sse(resp, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(text, "café 🚀 日本語");
        assert!(!text.contains('\u{FFFD}'));
    }

    #[test]
    fn probe_95_effort_routing() {
        let single = ModuleUnit {
            id: "a".into(),
            files: vec!["a.py".into()],
            layer: 0,
            wave: 0,
            deps: vec![],
            dependents: vec![],
            test_files: vec![],
            loc: 20,
            base_blake3: BTreeMap::new(),
            abi_hash: BTreeMap::new(),
            baseline_hygiene: 0.7,
            allow_abi_change: false,
        };
        let mut scc = single.clone();
        scc.files.push("b.py".into());
        assert_eq!(pick_effort(&single, 1, &["ruff F401".into()]), "low");
        assert_eq!(pick_effort(&single, 1, &["Gate 1 tests failed".into()]), "medium");
        assert_eq!(pick_effort(&single, 2, &["ruff F401".into()]), "medium");
        assert_eq!(pick_effort(&scc, 1, &["ruff F401".into()]), "medium");
    }
}
