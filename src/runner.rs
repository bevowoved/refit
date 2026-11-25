mod bench;
mod prompts;
mod worker;

pub use bench::{bench_command, bench_python};

use crate::gates::{
    AuditStats, GateConfig, ScanCache, audit_repo, read_gates, scan_repo, stage_hardlinks,
};
use crate::graph::{critical_depths, deps_context, units_interfere};
use crate::hygiene::HygieneScorer;
use crate::learn::{Durability, InvertedRewindIndex, LearnStore};
use crate::mvcc::MvccStore;
use crate::platform::{now_ms, os_tag, self_metrics, stream_http, walk_files};
use crate::proxy::{CacheMode, ProxyConfig, start_proxy};
use crate::research::WebClient;
use crate::source::{count_lines, ext_str, quantile, rel_norm, source_rel, supported_ext};
use crate::state::RunState;
use crate::stem::load_symbols;
use crate::telemetry::{EventSpec, TelemetryHub};
use crate::types::{BenchReport, ModuleUnit, RunSummary};
use crate::worktree::{Worktree, diffstat, open_worktree};
use anyhow::Result;
use prompts::{prefix_map, read_prompt};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use worker::{WorkerCtx, WorkerOutcome, run_unit};

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

#[derive(Default)]
struct RunTotals {
    committed: usize,
    skipped: usize,
    cas_conflicts: u64,
    rollbacks_g0: u64,
    rollbacks_g1: u64,
    rollbacks_g2: u64,
    rollbacks_g3: u64,
    retries: u64,
    formatted: u64,
    queue_ms: Vec<f64>,
    decode_tps: Vec<f64>,
}

struct SchedState {
    pending: Vec<ModuleUnit>,
    completed: HashSet<String>,
    rewound_once: HashSet<String>,
    in_flight: HashMap<String, ModuleUnit>,
    task_map: HashMap<tokio::task::Id, ModuleUnit>,
    join_set: JoinSet<WorkerOutcome>,
    tot: RunTotals,
    k: usize,
}

pub async fn execute_run(opts: RunOptions) -> Result<RunSummary> {
    let (wall_start, run_id) = (Instant::now(), format!("refit-{}-{}", now_ms(), std::process::id()));
    let isolated_wt = opts.worktree
        .then(|| open_worktree(&opts.repo_root, &run_id, opts.include_dirty_worktree)).transpose()?;
    let active_repo = isolated_wt.as_ref().map_or_else(|| opts.repo_root.clone(), |w| w.worktree_path.clone());
    let sys_prompt = read_prompt(&opts.skill_path);
    let explicit_tok = std::env::var("REFIT_PROXY_TOKEN").ok().filter(|s| !s.trim().is_empty());
    let client_tok = explicit_tok.clone().unwrap_or_else(|| "refit-local-proxy".into());

    let telemetry = TelemetryHub::open_hub(opts.telemetry_path.clone(), opts.resume)?;
    let proxy_cfg = ProxyConfig {
        listen_addr: opts.proxy_addr.clone(), upstream_base_url: opts.upstream_base_url.clone(),
        api_key: opts.api_key.clone(), model_override: opts.model.clone(),
        ui_index_path: opts.ui_index_path.clone(), static_prefix_override: Some(sys_prompt.clone()),
        cache_retention: opts.cache_retention, keep_warm: opts.keep_warm, auth_token: explicit_tok,
        lang_prefixes: prefix_map(&sys_prompt), preserve_tools: opts.preserve_tools,
    };
    let (proxy_state, proxy_handle) = start_proxy(proxy_cfg, telemetry.clone()).await?;
    proxy_state.bind_repo(active_repo.clone());

    let hygiene = Arc::new(HygieneScorer::open_persistent(&active_repo));
    let mut gate_raw = read_gates(&active_repo);
    gate_raw.strict_polyglot_tools = opts.strict_polyglot_tools;
    let gate_cfg = Arc::new(gate_raw);
    let mvcc = Arc::new(MvccStore::open(&active_repo, &run_id)?);
    let learn = Arc::new(LearnStore::open(&active_repo)?);

    let (committed_map, rolled_back) = if opts.resume {
        let rep = mvcc.recover_wal()?;
        tracing::info!("WAL resume: rb={} ok={} rev={}", rep.rolled_back_txs, rep.committed_modules.len(), rep.head_rev);
        (rep.committed_modules, rep.rolled_back_txs)
    } else {
        (HashMap::new(), 0)
    };

    let (bundle, _) = scan_repo(&active_repo, true, &hygiene)?;
    let shared_state = RunState::new(bundle.plan.total_loc);
    telemetry.bind_state(shared_state.clone());
    proxy_state.attach_state(shared_state.clone());

    let research = WebClient::for_repo(&active_repo, Some(shared_state.clone()));
    let phase0_res = if research.is_online() {
        WebClient::format_snippets(&research.audit_deps(&active_repo).await)
    } else { String::new() };

    let client = stream_http("refit-runner/0.5.0", 15, 600, crate::platform::budget().http_pool_max)?;
    let (chat_url, chat_bearer, chat_model, via_oc) = chat_endpoint(&client, &opts, &client_tok).await;
    let (rewind_index, file_to_unit) = if cfg!(any(feature = "live-rewind", test)) {
        build_rewind_index(&active_repo, &bundle, &gate_cfg)
    } else {
        (InvertedRewindIndex::default(), HashMap::new())
    };

    let ctx = Arc::new(WorkerCtx {
        run_id, repo_root: active_repo.clone(), client, chat_url, chat_bearer, chat_model,
        static_sys: sys_prompt, baseline_diags: bundle.audit.per_file_diagnostics.clone(),
        hygiene: hygiene.clone(), gate_cfg: gate_cfg.clone(), mvcc, learn,
        rewind_index: Arc::new(rewind_index), file_to_unit: Arc::new(file_to_unit),
        proxy: proxy_state, shared_state, research, phase0_research: phase0_res,
        telemetry: telemetry.clone(), max_retries: opts.max_retries,
    });
    emit_start(&ctx, &opts, &bundle, via_oc, rolled_back)?;
    let (pending, completed, skipped) = filter_pending(&ctx, &bundle, &opts, &committed_map);
    let totals = run_sched(ctx.clone(), &bundle.plan.units, pending, completed, skipped, &opts).await;
    let fallback: Vec<String> = bundle.plan.units.iter().flat_map(|u| u.files.clone()).collect();
    let audit_rel = final_files(&active_repo, fallback);
    let final_audit = if totals.committed == totals.skipped {
        bundle.audit.clone()
    } else {
        audit_repo(&gate_cfg, &audit_rel, &hygiene)?
    };
    hygiene.flush_cache();

    let bench = run_bench(&opts, &gate_cfg, &active_repo, &telemetry);
    let summary = build_summary(&ctx, SummaryInput {
        opts: &opts, wall_start, bundle: &bundle, final_audit: &final_audit,
        audit_rel: &audit_rel, tot: totals, bench, wt: &isolated_wt, via_oc,
    });
    if opts.hold_server_secs > 0 { tokio::time::sleep(Duration::from_secs(opts.hold_server_secs)).await; }
    proxy_handle.abort();
    Ok(summary)
}

fn emit_start(
    ctx: &WorkerCtx,
    opts: &RunOptions,
    bundle: &ScanCache,
    via_oc: bool,
    rolled_back: usize,
) -> Result<()> {
    let (plan, audit) = (&bundle.plan, &bundle.audit);
    let start_data = json!({
        "run_id": ctx.run_id, "host_os": plan.host_os, "model": opts.model,
        "provider_label": opts.provider_label, "via_openclaw": via_oc,
        "resume": opts.resume, "rolled_back_on_resume": rolled_back,
        "stem_used": plan.stem_used, "stem_index_ms": plan.stem_index_ms,
        "rewind_indexed_files": ctx.rewind_index.file_count(),
        "total_files": plan.total_files, "total_loc": plan.total_loc,
        "scc_count": plan.scc_count, "wave_count": plan.wave_count,
        "research_online": ctx.research.is_online(),
        "uncached_tpm_cap": ctx.proxy.snapshot().uncached_tpm_cap,
        "baseline": {
            "tests_total": audit.tests_total, "tests_failed": audit.tests_failed,
            "ruff_errors": audit.ruff_errors, "ty_errors": audit.ty_errors,
            "hygiene_median": audit.hygiene_median, "hygiene_p10": audit.hygiene_p10
        }
    });
    ctx.telemetry.emit(
        "run_start",
        EventSpec { uncached_tpm: Some(0), data: Some(start_data), ..EventSpec::default() },
    );
    ctx.telemetry.emit(
        "dag_ready",
        EventSpec {
            uncached_tpm: Some(0),
            data: Some(serde_json::to_value(plan)?),
            ..EventSpec::default()
        },
    );
    Ok(())
}

fn filter_pending(
    ctx: &WorkerCtx, bundle: &ScanCache, opts: &RunOptions,
    committed: &HashMap<String, BTreeMap<String, String>>,
) -> (Vec<ModuleUnit>, HashSet<String>, usize) {
    let (plan, audit) = (&bundle.plan, &bundle.audit);
    let (mut pending, mut done, mut skipped) = (Vec::new(), HashSet::new(), 0usize);
    for u in &plan.units {
        let has_diags = u.files.iter().any(|f| audit.per_file_diagnostics.get(f).is_some_and(|v| !v.is_empty()));
        if !has_diags && u.baseline_hygiene >= ctx.gate_cfg.min_hygiene_target {
            skipped += 1;
            done.insert(u.id.clone());
            continue;
        }
        if opts.resume && resumed_ok(&ctx.repo_root, u, committed.get(&u.id)) {
            skipped += 1;
            done.insert(u.id.clone());
            ctx.telemetry.emit_unit("mvcc_commit", u, json!({"skipped_resumed": true, "head_rev": ctx.mvcc.head_rev()}));
            continue;
        }
        pending.push(u.clone());
    }
    let cp = critical_depths(&plan.units);
    pending.sort_by(|a, b| {
        let (da, db) = (cp.get(&a.id).copied().unwrap_or(0), cp.get(&b.id).copied().unwrap_or(0));
        a.wave.cmp(&b.wave).then_with(|| db.cmp(&da)).then_with(|| a.id.cmp(&b.id))
    });
    if let Some(lim) = opts.max_units && pending.len() > lim {
        for dropped in pending.drain(lim..) { done.insert(dropped.id); }
    }
    for u in &plan.units {
        if !pending.iter().any(|p| p.id == u.id) { done.insert(u.id.clone()); }
    }
    (pending, done, skipped)
}

fn resumed_ok(repo: &Path, u: &ModuleUnit, saved: Option<&BTreeMap<String, String>>) -> bool {
    let Some(hashes) = saved else { return false };
    u.files.iter().all(|rel| {
        hashes.get(rel).is_some_and(|exp| MvccStore::hash_file(&repo.join(rel)).ok().is_some_and(|a| &a == exp))
    })
}

async fn run_sched(
    ctx: Arc<WorkerCtx>, all_units: &[ModuleUnit], pending: Vec<ModuleUnit>,
    completed: HashSet<String>, skipped: usize, opts: &RunOptions,
) -> RunTotals {
    let mut s = SchedState {
        pending, completed, rewound_once: HashSet::new(), in_flight: HashMap::new(),
        task_map: HashMap::new(), join_set: JoinSet::new(),
        tot: RunTotals { committed: skipped, skipped, ..Default::default() },
        k: opts.initial_k.clamp(2, opts.max_k.max(2)),
    };
    loop {
        ctx.shared_state.await_resume().await;
        if ctx.shared_state.abort_asked() { s.pending.clear(); }
        if let Some(k_ov) = ctx.shared_state.k_override() { s.k = k_ov.clamp(1, opts.max_k.max(2)); }
        drain_skipped(&ctx, &mut s);
        apply_rewind_events(&ctx, all_units, &mut s);
        spawn_ready(&ctx, all_units, &mut s);
        if s.in_flight.is_empty() { mark_blocked(&ctx, &mut s); break; }
        match s.join_set.join_next().await {
            Some(Ok(out)) => apply_outcome(out, &ctx.shared_state, &mut s, opts.max_k),
            Some(Err(err)) => handle_panic(&ctx, err, &mut s),
            None => {
                for (_, orph) in s.in_flight.drain() {
                    ctx.shared_state.step_inflight(-1);
                    ctx.shared_state.mark_blocked(&orph.id, "worker_join_set_drained");
                }
                break;
            }
        }
    }
    s.tot
}

fn build_rewind_index(
    repo_root: &Path, bundle: &ScanCache, gate_cfg: &GateConfig,
) -> (InvertedRewindIndex, HashMap<String, String>) {
    let total_files: usize = bundle.plan.units.iter().map(|u| u.files.len()).sum();
    let mut file_to_unit = HashMap::with_capacity(total_files);
    let mut unit_files: HashMap<&str, &[String]> = HashMap::with_capacity(bundle.plan.units.len());
    let mut dur_by_file = rustc_hash::FxHashMap::with_capacity_and_hasher(total_files, Default::default());

    for unit in &bundle.plan.units {
        unit_files.insert(unit.id.as_str(), unit.files.as_slice());
        let has_diags = unit.files.iter()
            .any(|rel| bundle.audit.per_file_diagnostics.get(rel).is_some_and(|v| !v.is_empty()));
        let dur = if !has_diags && unit.baseline_hygiene >= gate_cfg.min_hygiene_target {
            Durability::High
        } else {
            Durability::Low
        };
        for rel in &unit.files {
            file_to_unit.insert(rel.clone(), unit.id.clone());
            dur_by_file.insert(rel.as_str(), dur);
        }
    }
    let mut index = InvertedRewindIndex::from_symbol_graph(&load_symbols(repo_root, None), &dur_by_file);
    for unit in &bundle.plan.units {
        let Some(src_file) = unit.files.first() else { continue };
        for dep_id in &unit.deps {
            if let Some(dep_files) = unit_files.get(dep_id.as_str()) {
                for dep_file in *dep_files { index.add_dep_by_path(src_file, dep_file); }
            }
        }
    }
    (index, file_to_unit)
}

fn apply_rewind_events(ctx: &WorkerCtx, all_units: &[ModuleUnit], s: &mut SchedState) {
    for (_, flagged_units) in ctx.shared_state.drain_rewind_events() {
        for uid in flagged_units {
            if s.rewound_once.contains(&uid) || ctx.shared_state.is_skipped(&uid)
                || s.in_flight.contains_key(&uid) || s.pending.iter().any(|u| u.id == uid)
            {
                continue;
            }
            if let Some(unit) = all_units.iter().find(|u| u.id == uid) {
                s.completed.remove(&uid);
                s.rewound_once.insert(uid);
                s.pending.push(unit.clone());
            }
        }
    }
}

fn drain_skipped(ctx: &WorkerCtx, s: &mut SchedState) {
    let mut skipped_now = Vec::new();
    s.pending.retain(|u| {
        if ctx.shared_state.is_skipped(&u.id) { skipped_now.push(u.id.clone()); false } else { true }
    });
    for sid in skipped_now {
        s.completed.insert(sid.clone());
        let data = json!({"reason": "live_steer_skip"});
        ctx.telemetry.emit("unit_skipped", EventSpec { module_id: Some(sid), data: Some(data), ..EventSpec::default() });
    }
}

fn spawn_ready(ctx: &Arc<WorkerCtx>, all_units: &[ModuleUnit], s: &mut SchedState) {
    while s.in_flight.len() < s.k {
        let ready = |c: &ModuleUnit| {
            c.deps.iter().all(|d| s.completed.contains(d)) && !s.in_flight.values().any(|a| units_interfere(c, a))
        };
        let steered = ctx.shared_state
            .pop_ready(|pid| s.pending.iter().any(|c| c.id == pid && ready(c)))
            .and_then(|pid| s.pending.iter().position(|c| c.id == pid));
        let Some(idx) = steered.or_else(|| s.pending.iter().position(ready)) else { break };
        let unit = s.pending.remove(idx);
        let deps_abi = deps_context(&unit, all_units, &ctx.repo_root);
        s.in_flight.insert(unit.id.clone(), unit.clone());
        ctx.shared_state.step_inflight(1);
        let (ctx_c, u_map, k_cur) = (ctx.clone(), unit.clone(), s.k);
        let h = s.join_set.spawn(async move { run_unit(ctx_c, unit, deps_abi, k_cur).await });
        s.task_map.insert(h.id(), u_map);
    }
}

fn mark_blocked(ctx: &WorkerCtx, s: &mut SchedState) {
    for stuck in s.pending.drain(..) {
        let unmet: Vec<String> = stuck.deps.iter().filter(|d| !s.completed.contains(*d)).cloned().collect();
        let reason = format!("unsatisfied_dependencies: {}", unmet.join(", "));
        ctx.shared_state.mark_blocked(&stuck.id, &reason);
        ctx.telemetry.emit_unit("unit_blocked", &stuck, json!({"reason": reason, "unmet_deps": unmet}));
    }
}

fn apply_outcome(out: WorkerOutcome, state: &RunState, s: &mut SchedState, max_k: usize) {
    s.in_flight.remove(&out.unit.id);
    s.task_map.retain(|_, u| u.id != out.unit.id);
    state.step_inflight(-1);
    s.completed.insert(out.unit.id);
    if out.committed && !out.skipped_resumed { s.tot.committed += 1; }
    s.tot.cas_conflicts += out.cas_conflicts;
    s.tot.rollbacks_g0 += out.rollbacks_g0;
    s.tot.rollbacks_g1 += out.rollbacks_g1;
    s.tot.rollbacks_g2 += out.rollbacks_g2;
    s.tot.rollbacks_g3 += out.rollbacks_g3;
    s.tot.retries += out.retries;
    s.tot.formatted += out.formatted_files;
    s.tot.queue_ms.extend(out.queue_ms_samples);
    s.tot.decode_tps.extend(out.decode_tps_samples);
    if out.saw_429 {
        s.k = (s.k / 2).max(2);
    } else if out.committed && s.k < max_k {
        s.k += 1;
    }
}

fn handle_panic(ctx: &WorkerCtx, err: tokio::task::JoinError, s: &mut SchedState) {
    let Some(panicked) = s.task_map.remove(&err.id()) else { return };
    s.in_flight.remove(&panicked.id);
    ctx.shared_state.step_inflight(-1);
    let reason = format!("worker_task_panicked: {err}");
    ctx.shared_state.mark_blocked(&panicked.id, &reason);
    s.completed.insert(panicked.id.clone());
    ctx.telemetry.emit_unit("unit_blocked", &panicked, json!({"reason": reason}));
}

fn final_files(repo: &Path, fallback: Vec<String>) -> Vec<String> {
    let found: Vec<String> = walk_files(repo, None).into_iter().filter_map(|p| {
        if !supported_ext(ext_str(&p)) { return None; }
        let rel = rel_norm(repo, &p)?;
        source_rel(&rel).then_some(rel)
    }).collect();
    if found.is_empty() { fallback } else { found }
}

fn run_bench(opts: &RunOptions, gate_cfg: &GateConfig, repo: &Path, telemetry: &TelemetryHub) -> Option<BenchReport> {
    let base = opts.baseline_copy.as_deref().filter(|p| p.exists())?;
    let res = if let Some(cmd) = opts.bench_cmd.as_deref() {
        bench_command(cmd, base, repo, 5)
    } else {
        bench_python(&gate_cfg.python_bin, opts.bench_script.as_deref().filter(|p| p.exists())?, base, repo, 5)
    };
    match res {
        Ok(b) => {
            if let Ok(val) = serde_json::to_value(&b) { telemetry.emit_data("bench_result", val); }
            Some(b)
        }
        Err(e) => { tracing::warn!("Alternating bench skipped due to error: {e}"); None }
    }
}

struct SummaryInput<'a> {
    opts: &'a RunOptions, wall_start: Instant, bundle: &'a ScanCache, final_audit: &'a AuditStats,
    audit_rel: &'a [String], tot: RunTotals, bench: Option<BenchReport>, wt: &'a Option<Worktree>, via_oc: bool,
}

fn build_summary(ctx: &WorkerCtx, mut inp: SummaryInput<'_>) -> RunSummary {
    let (plan, base_audit) = (&inp.bundle.plan, &inp.bundle.audit);
    let loc_after: usize = inp.audit_rel.iter()
        .map(|rel| fs::read(ctx.repo_root.join(rel)).map_or(0, |b| count_lines(&b))).sum();
    let digest = ctx.shared_state.digest();
    let (_, raw_bytes, zst_bytes) = ctx.telemetry.flush_zstd()
        .unwrap_or_else(|_| (PathBuf::new(), ctx.telemetry.bytes_written(), 0));
    let (p_snap, (pq, ptps)) = (ctx.proxy.snapshot(), ctx.proxy.timing_samples());
    if inp.tot.queue_ms.is_empty() { inp.tot.queue_ms = pq; }
    if inp.tot.decode_tps.is_empty() { inp.tot.decode_tps = ptps; }
    let avg_queue_s = if inp.tot.queue_ms.is_empty() { 0.0 } else {
        (inp.tot.queue_ms.iter().sum::<f64>() / inp.tot.queue_ms.len() as f64) / 1000.0
    };
    let cache_pct = if p_snap.total_prompt_tokens > 0 {
        (p_snap.total_cached_tokens as f64 / p_snap.total_prompt_tokens as f64) * 100.0
    } else { 0.0 };

    let summary = RunSummary {
        run_id: ctx.run_id.clone(), host_os: os_tag(), model: inp.opts.model.clone(),
        provider_label: inp.opts.provider_label.clone(), total_files: plan.total_files,
        total_loc: plan.total_loc, total_units: plan.units.len(), committed_units: inp.tot.committed,
        skipped_resumed_units: inp.tot.skipped, total_wall_s: inp.wall_start.elapsed().as_secs_f64(),
        avg_queue_s, median_decode_tps: quantile(&mut inp.tot.decode_tps, 0.5),
        hygiene_before_median: base_audit.hygiene_median, hygiene_before_p10: base_audit.hygiene_p10,
        hygiene_after_median: inp.final_audit.hygiene_median, hygiene_after_p10: inp.final_audit.hygiene_p10,
        tests_total: inp.final_audit.tests_total.max(base_audit.tests_total),
        tests_before_fail: base_audit.tests_failed, tests_after_fail: inp.final_audit.tests_failed,
        gate3_before_ruff: base_audit.ruff_errors, gate3_before_ty: base_audit.ty_errors,
        gate3_after_ruff: inp.final_audit.ruff_errors, gate3_after_ty: inp.final_audit.ty_errors,
        cas_conflicts: inp.tot.cas_conflicts, rollbacks_g0: inp.tot.rollbacks_g0,
        rollbacks_g1: inp.tot.rollbacks_g1, rollbacks_g2: inp.tot.rollbacks_g2,
        rollbacks_g3: inp.tot.rollbacks_g3, retries_total: inp.tot.retries,
        http_429_count: p_snap.http_429_count, circuit_pauses: p_snap.circuit_pauses,
        prefix_drift_events: p_snap.prefix_drift_events, prefix_cache_hit_pct: cache_pct,
        uncached_tpm_cap: p_snap.uncached_tpm_cap, total_uncached_tokens: p_snap.total_uncached_tokens,
        total_cached_tokens: p_snap.total_cached_tokens,
        engine_rss_mb: (self_metrics().rss_bytes as f64) / (1024.0 * 1024.0),
        bench: inp.bench, loc_before: plan.total_loc, loc_after,
        telemetry_raw_bytes: raw_bytes, telemetry_zstd_bytes: zst_bytes,
        stage_hardlinked_files: stage_hardlinks(), research_queries: digest.research_queries,
        research_cache_hits: digest.research_cache_hits,
        worktree_branch: inp.wt.as_ref().map(|w| w.branch.clone()),
        worktree_path: inp.wt.as_ref().map(|w| w.worktree_path.display().to_string()),
        ready_for_manual_merge: inp.wt.is_some(), detected_languages: inp.bundle.detected_languages.clone(),
        blocked_units: ctx.shared_state.blocked_count(), formatted_files_total: inp.tot.formatted,
    };
    let wt_meta = inp.wt.as_ref().map(|w| json!({
        "core_repo": w.core_repo.display().to_string(), "branch": w.branch,
        "carried_dirty_changes": w.carried_dirty_changes, "diff_stat": diffstat(w)
    }));
    let (rules, helpful) = ctx.learn.stats();
    let end_data = json!({
        "via_openclaw": inp.via_oc, "learn_rules": rules, "learn_helpful": helpful,
        "tracked_units": digest.tracked_units, "blocked_units": ctx.shared_state.blocked_count(),
        "rewind_triggers": ctx.shared_state.rewind_count(),
        "telemetry_dedup_skipped": ctx.telemetry.dedup_skipped(), "worktree": wt_meta
    });
    ctx.telemetry.emit("run_end", EventSpec {
        uncached_tpm: Some(p_snap.uncached_tpm_rolling), data: Some(end_data),
        summary: Some(summary.clone()), ..EventSpec::default()
    });
    summary
}

async fn chat_endpoint(client: &reqwest::Client, opts: &RunOptions, proxy_token: &str) -> (String, String, String, bool) {
    if let Some(oc_base) = &opts.openclaw_base_url {
        let base = oc_base.trim_end_matches('/');
        let mut req = client.get(format!("{base}/models")).timeout(Duration::from_secs(2));
        if let Some(tok) = &opts.openclaw_token { req = req.bearer_auth(tok); }
        if req.send().await.is_ok_and(|r| r.status().as_u16() < 500) {
            let bearer = opts.openclaw_token.clone().unwrap_or_else(|| "refit-local-gateway".into());
            return (format!("{base}/chat/completions"), bearer, opts.openclaw_model.clone(), true);
        }
    }
    (format!("http://{}/v1/chat/completions", opts.proxy_addr), proxy_token.to_string(), opts.model.clone(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{mock_sse, sse_delta, unit_stub};
    use prompts::pick_effort;
    use worker::collect_sse;

    async fn get_sse(addr: &str) -> reqwest::Response {
        let req = reqwest::Client::new().get(format!("http://{addr}/stream"));
        req.send().await.unwrap()
    }

    #[tokio::test]
    async fn test_sse_stream_timeout_and_utf8() {
        let stalled = mock_sse(vec![sse_delta("partial")], 0, 5).await;
        assert!(collect_sse(get_sse(&stalled).await, Duration::from_millis(120)).await.is_err());

        let payload = sse_delta("café 🚀 日本語");
        let pos = payload.windows(4).position(|w| w == "🚀".as_bytes()).unwrap();
        let chunks = vec![payload[..pos + 2].to_vec(), payload[pos + 2..].to_vec()];
        let addr = mock_sse(chunks, 15, 0).await;
        let (_, _, text) = collect_sse(get_sse(&addr).await, Duration::from_secs(2)).await.unwrap();
        assert_eq!(text, "café 🚀 日本語");
        assert!(!text.contains('\u{FFFD}'));
    }

    #[test]
    fn test_effort_routing() {
        let single = unit_stub("a", &["a.py"]);
        let scc = unit_stub("a", &["a.py", "b.py"]);
        for (unit, attempt, diag, want) in [
            (&single, 1, "ruff F401", "low"),
            (&single, 1, "Gate 1 tests failed", "medium"),
            (&single, 2, "ruff F401", "medium"),
            (&scc, 1, "ruff F401", "medium"),
        ] {
            assert_eq!(pick_effort(unit, attempt, &[diag.into()]), want);
        }
    }
}
