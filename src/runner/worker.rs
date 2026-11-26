use super::prompts::{cold_verifier_hint, guess_lang, pick_effort};
use crate::gates::{GateConfig, net_new_diagnostics, run_gates, stage_hardlinks};
use crate::graph::{apply_patches, build_packet, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::learn::{
    InvertedRewindIndex, LearnStore, MIN_KERNEL_COST_US, RewindEvent, should_use_index,
};
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::net::drain_sse;
use crate::platform::{MemGate, backoff_ms, char_prefix, now_ms};
use crate::proxy::ProxyState;
use crate::research::WebClient;
use crate::source::{count_lines, dir_of};
use crate::state::RunState;
use crate::telemetry::{EventSpec, TelemetryHub};
use crate::types::{GateVerdict, ModuleUnit};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(Serialize)]
struct ChatMsg {
    role: &'static str, content: String,
    #[serde(skip_serializing_if = "Option::is_none")] reasoning: Option<String>,
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str, messages: Vec<ChatMsg>, stream: bool, max_completion_tokens: usize,
    temperature: f32, reasoning_effort: &'static str, reasoning_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")] prompt_cache_key: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChatResp { #[serde(default)] choices: Vec<ChatChoice> }

impl ChatResp {
    fn first_choice(self) -> Option<ChatChoice> { self.choices.into_iter().next() }
}

#[derive(Deserialize, Default)]
struct ChatChoice { #[serde(default)] message: ChatPart, #[serde(default)] delta: ChatPart }

#[derive(Deserialize, Default)]
struct ChatPart { #[serde(default)] content: Option<String>, #[serde(default)] reasoning: Option<String> }

#[derive(Serialize)]
struct DispatchData { attempt: u32, k: usize, lang: &'static str, verifier_hint: &'static str }

#[derive(Serialize)]
struct RollbackData<'a> {
    gate: &'a str, attempt: u32,
    #[serde(skip_serializing_if = "Option::is_none")] reason: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")] diagnostics: Option<&'a [String]>,
}

#[derive(Serialize)]
struct RewindData<'a> {
    trigger: &'static str, rule_key: &'a str, token: &'a str,
    affected_files: usize, flagged_units: &'a [String],
}

#[derive(Serialize)]
struct CommitData {
    head_rev: u64, tx_id: String, attempt: u32, hygiene_before: f64, hygiene_after: f64,
    ruff_errors: usize, ty_errors: usize, tests_passed: usize, formatted_files: usize,
}

fn to_val<T: Serialize>(v: &T) -> Value { serde_json::to_value(v).unwrap_or(Value::Null) }

pub(super) struct WorkerCtx {
    pub(super) run_id: String, pub(super) repo_root: PathBuf, pub(super) client: reqwest::Client,
    pub(super) chat_url: String, pub(super) chat_bearer: String, pub(super) chat_model: String,
    pub(super) static_sys: String, pub(super) baseline_diags: HashMap<String, Vec<String>>,
    pub(super) hygiene: Arc<HygieneScorer>, pub(super) gate_cfg: Arc<GateConfig>,
    pub(super) mvcc: Arc<MvccStore>, pub(super) learn: Arc<LearnStore>,
    pub(super) rewind_index: Arc<InvertedRewindIndex>,
    pub(super) file_to_unit: Arc<HashMap<String, String>>, pub(super) proxy: Arc<ProxyState>,
    pub(super) shared_state: Arc<RunState>, pub(super) research: Arc<WebClient>,
    pub(super) phase0_research: String, pub(super) telemetry: TelemetryHub,
    pub(super) max_retries: u32,
}

#[derive(Default)]
pub(super) struct WorkerOutcome {
    pub(super) unit_id: String, pub(super) committed: bool, pub(super) cas_conflicts: u64,
    pub(super) rollbacks_g0: u64, pub(super) rollbacks_g1: u64, pub(super) rollbacks_g2: u64,
    pub(super) rollbacks_g3: u64, pub(super) retries: u64, pub(super) formatted_files: u64,
    pub(super) saw_429: bool, pub(super) queue_ms_samples: Vec<f64>,
    pub(super) decode_tps_samples: Vec<f64>,
}

struct UnitRun {
    unit: ModuleUnit, lang: &'static str, dir: String, k: usize, out: WorkerOutcome,
    base_diags: Vec<String>, diags: Vec<String>, initial_user: Option<String>,
    prev_asst: Option<(String, String)>, prior_fails: Vec<String>,
    retry_note: String, learn_keys: Vec<String>,
}

impl UnitRun {
    fn new(ctx: &WorkerCtx, unit: ModuleUnit, k: usize) -> Self {
        let raw_dir = dir_of(&unit.files[0]);
        let dir = if raw_dir.is_empty() { "*".into() } else { raw_dir };
        let base_diags: Vec<String> = unit.files.iter()
            .flat_map(|f| ctx.baseline_diags.get(f).cloned().unwrap_or_default())
            .collect();
        let lang = guess_lang(&unit);
        let out = WorkerOutcome { unit_id: unit.id.clone(), ..WorkerOutcome::default() };
        Self {
            unit, lang, dir, k, out, diags: base_diags.clone(), base_diags,
            initial_user: None, prev_asst: None, prior_fails: Vec::new(),
            retry_note: String::new(), learn_keys: Vec::new(),
        }
    }
}

pub(super) async fn run_unit(
    ctx: Arc<WorkerCtx>, unit: ModuleUnit, deps_abi: String, current_k: usize,
) -> WorkerOutcome {
    let mut r = UnitRun::new(&ctx, unit, current_k);
    for attempt in 1..=(ctx.max_retries + 1) {
        if attempt > 1 { r.out.retries += 1; }
        ctx.shared_state.set_phase(&r.unit.id, "dispatching", attempt, r.unit.loc);

        let (base_texts, exp_hashes) = read_bases(&ctx.repo_root, &r.unit.files);
        r.unit.base_blake3 = exp_hashes.clone();
        let (learn_suf, keys) = ctx.learn.rule_suffix(&r.dir);
        for lk in keys {
            if !r.learn_keys.contains(&lk) { r.learn_keys.push(lk); }
        }

        let Some(msgs) = build_msgs(&ctx, &mut r, &deps_abi, &learn_suf, attempt) else { break };
        let (reason, content) = match call_llm(&ctx, &mut r, attempt, msgs).await {
            Ok(pair) => pair,
            Err(err_diag) => { r.diags = vec![err_diag]; continue; }
        };
        r.prev_asst = Some((reason, content.clone()));

        let applied = match apply_patches(&base_texts, &parse_patches(&content)) {
            Ok(a) => a,
            Err(e) => { apply_err(&ctx, &mut r, attempt, &e.to_string()); continue; }
        };

        let verdict = match verify_unit(&ctx, &r.unit, attempt, applied.updated, applied.deleted).await {
            Ok(v) => v,
            Err(e) => { r.diags = vec![e]; continue; }
        };
        r.out.formatted_files += verdict.formatted_files as u64;

        if !verdict.passed {
            gate_fail(&ctx, &mut r, attempt, verdict).await;
            continue;
        }
        if commit_unit(&ctx, &mut r, attempt, &exp_hashes, verdict) { break; }
    }
    r.out
}

fn read_bases(root: &Path, files: &[String]) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let (mut texts, mut hashes) = (BTreeMap::new(), BTreeMap::new());
    for rel in files {
        if let Ok(txt) = fs::read_to_string(root.join(rel)) {
            hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
            texts.insert(rel.clone(), txt);
        }
    }
    (texts, hashes)
}

fn chat_msg(role: &'static str, content: impl Into<String>, reasoning: Option<String>) -> ChatMsg {
    ChatMsg { role, content: content.into(), reasoning }
}

fn build_msgs(
    ctx: &WorkerCtx, r: &mut UnitRun, deps_abi: &str, learn_suf: &str, attempt: u32,
) -> Option<Vec<ChatMsg>> {
    let mut msgs = vec![chat_msg("system", ctx.static_sys.clone(), None)];
    let budget = if r.unit.files.len() > 1 { 24_000 } else { 12_000 };

    if attempt == 1 || r.initial_user.is_none() {
        let pkt = build_packet(&ctx.repo_root, &r.unit, &r.diags, learn_suf, budget).ok()?;
        let mut prompt = String::with_capacity(deps_abi.len() + pkt.len() + 64);
        for part in [deps_abi, ctx.phase0_research.as_str()] {
            if !part.is_empty() {
                prompt.push_str(part);
                prompt.push('\n');
            }
        }
        prompt.push_str(&pkt);
        msgs.push(chat_msg("user", prompt.clone(), None));
        r.initial_user = Some(prompt);
        return Some(msgs);
    }

    msgs.push(chat_msg("user", r.initial_user.as_deref().unwrap_or(""), None));
    if let Some((prev_r, prev_c)) = &r.prev_asst {
        let reasoning = (!prev_r.is_empty()).then(|| char_prefix(prev_r, 600).to_string());
        msgs.push(chat_msg("assistant", char_prefix(prev_c, 2400), reasoning));
    }

    let hist = if r.prior_fails.is_empty() {
        String::new()
    } else {
        format!(
            "Negative constraints from earlier failed attempts (DO NOT repeat):\n- {}\n",
            r.prior_fails.join("\n- ")
        )
    };
    let feedback = format!(
        "[attempt={attempt}] Previous patch failed verification and was ROLLED BACK \
         (disk UNCHANGED). Every <<<SEARCH>>> block MUST uniquely match the ORIGINAL \
         file verbatim.\n{hist}Fix EVERY diagnostic below:\n- {}\n{}{}",
        r.diags.join("\n- "), learn_suf, r.retry_note
    );
    msgs.push(chat_msg("user", feedback, None));
    Some(msgs)
}

async fn call_llm(
    ctx: &WorkerCtx, r: &mut UnitRun, attempt: u32, msgs: Vec<ChatMsg>,
) -> std::result::Result<(String, String), String> {
    let dispatch = to_val(&DispatchData {
        attempt, k: r.k, lang: r.lang, verifier_hint: cold_verifier_hint(r.lang),
    });
    ctx.telemetry.emit("wave_dispatch", EventSpec {
        uncached_tpm: Some(ctx.proxy.snapshot().uncached_tpm_rolling),
        data: Some(dispatch), ..EventSpec::unit(&r.unit)
    });

    let body = ChatReq {
        model: &ctx.chat_model, messages: msgs, stream: true,
        max_completion_tokens: if r.unit.files.len() > 1 { 5500 } else { 3200 },
        temperature: 1.0, reasoning_effort: pick_effort(&r.unit, attempt, &r.diags),
        reasoning_format: "parsed",
        prompt_cache_key: (attempt >= 2)
            .then(|| format!("refit:{}:m:{}", char_prefix(&ctx.run_id, 10), r.unit.id)),
    };

    let req = ctx.client.post(&ctx.chat_url).bearer_auth(&ctx.chat_bearer)
        .header("x-refit-module", &r.unit.id)
        .header("x-refit-attempt", attempt.to_string())
        .header("x-refit-lang", r.lang)
        .json(&body);
    let resp = match req.send().await {
        Ok(resp) => resp,
        Err(err) => {
            sleep_backoff(attempt, 500, 10_000).await;
            return Err(format!("network error: {err}"));
        }
    };

    let code = resp.status().as_u16();
    r.out.saw_429 |= code == 429;
    if code != 200 {
        let txt = resp.text().await.unwrap_or_default();
        sleep_backoff(attempt, 800, 15_000).await;
        return Err(format!("HTTP {code}: {}", char_prefix(&txt, 160)));
    }

    let (_, reason, content) = match collect_sse(resp, Duration::from_secs(60)).await {
        Ok(triple) => triple,
        Err(e) => {
            sleep_backoff(attempt, 500, 10_000).await;
            return Err(format!("stream stalled or interrupted: {e}"));
        }
    };

    if let Some(m) = ctx.proxy.take_metrics(&r.unit.id) {
        if m.queue_ms > 0.0 {
            r.out.queue_ms_samples.push(m.queue_ms);
        }
        if m.decode_tps > 0.0 {
            r.out.decode_tps_samples.push(m.decode_tps);
        }
        r.out.saw_429 |= m.http_status == 429;
    }
    Ok((reason, content))
}

async fn sleep_backoff(attempt: u32, base_ms: u64, cap_ms: u64) {
    let wait = backoff_ms(attempt, base_ms, cap_ms, None, now_ms());
    tokio::time::sleep(Duration::from_millis(wait)).await;
}

fn apply_err(ctx: &WorkerCtx, r: &mut UnitRun, attempt: u32, err: &str) {
    let msg = format!("asrf_apply: {err}");
    r.out.rollbacks_g0 += 1;
    let _ = ctx.mvcc.record_rollback(&r.unit.id, "asrf_apply", &msg);
    ctx.learn.record_failure("asrf_apply", &r.dir, &msg);
    r.prior_fails.push(format!("Attempt {attempt} failed at asrf_apply: {}", char_prefix(&msg, 120)));
    let data = to_val(&RollbackData {
        gate: "asrf_apply", attempt, reason: Some(&msg), diagnostics: None,
    });
    ctx.telemetry.emit_unit("mvcc_rollback", &r.unit, data);
    r.diags = vec![msg];
}

async fn verify_unit(
    ctx: &WorkerCtx, unit: &ModuleUnit, attempt: u32,
    updated: BTreeMap<String, String>, deleted: Vec<String>,
) -> std::result::Result<GateVerdict, String> {
    ctx.shared_state.set_phase(&unit.id, "verifying", attempt, unit.loc);
    let hl_before = stage_hardlinks();
    let (cfg, hyg, unit_b) = (ctx.gate_cfg.clone(), ctx.hygiene.clone(), unit.clone());

    let verdict = tokio::task::spawn_blocking(move || {
        let gate = MemGate::global();
        let reserve = (4 * 1024 * 1024).min(gate.capacity());
        let _mem = gate.reserve_or_wait(reserve, Duration::from_secs(10)).ok();
        run_gates(&cfg, &unit_b, &updated, &deleted, &hyg)
    })
    .await
    .map_err(|e| format!("blocking join error: {e}"))?
    .map_err(|e| format!("gate runner error: {e}"))?;

    ctx.shared_state.add_hardlinks(stage_hardlinks().saturating_sub(hl_before));
    let val = serde_json::to_value(&verdict).unwrap_or(Value::Null);
    ctx.telemetry.emit_unit("gate_result", unit, val);
    Ok(verdict)
}

async fn gate_fail(ctx: &WorkerCtx, r: &mut UnitRun, attempt: u32, verdict: GateVerdict) {
    let fg = verdict.failed_gate.unwrap_or_else(|| "gate".into());
    match fg.as_str() {
        "gate0_syntax" | "gate0_abi" | "gate0_rust_check" | "gate0_ts_check" | "gate0_go_vet" => {
            r.out.rollbacks_g0 += 1;
        }
        "gate1_tests" => r.out.rollbacks_g1 += 1,
        "gate2_hygiene" => r.out.rollbacks_g2 += 1,
        _ => r.out.rollbacks_g3 += 1,
    }

    let summary = verdict.diagnostics.join(" | ");
    let _ = ctx.mvcc.record_rollback(&r.unit.id, &fg, &summary);
    let net_new = net_new_diagnostics(&r.base_diags, &verdict.diagnostics);
    let learn_diag = if net_new.is_empty() { summary.clone() } else { net_new.join(" | ") };
    if let Some(event) = ctx.learn.record_failure(&fg, &r.dir, &learn_diag) {
        dispatch_rewind(ctx, &r.unit, event);
    }
    r.prior_fails.push(format!("Attempt {attempt} failed at {fg}: {}", char_prefix(&summary, 140)));

    let want_web = attempt == 1
        && ctx.research.is_online()
        && r.retry_note.is_empty()
        && matches!(fg.as_str(), "gate1_tests" | "gate3_ty" | "gate0_rust_check" | "gate0_ts_check");
    if want_web {
        let query = format!("{} fix error {}", r.lang, char_prefix(&summary, 120));
        if let Ok(res) = ctx.research.search_web(&query, 3).await {
            r.retry_note = format!("\n{}", WebClient::format_snippets(&[res]));
        }
    }

    let data = to_val(&RollbackData {
        gate: &fg, attempt, reason: None, diagnostics: Some(&verdict.diagnostics),
    });
    ctx.telemetry.emit_unit("mvcc_rollback", &r.unit, data);
    r.diags = verdict.diagnostics;
}

fn dispatch_rewind(ctx: &WorkerCtx, unit: &ModuleUnit, event: RewindEvent) {
    if ctx.rewind_index.file_count() == 0 || !should_use_index(MIN_KERNEL_COST_US) {
        return;
    }
    let affected = ctx.rewind_index.query_event_paths(&event, MIN_KERNEL_COST_US);
    let mut flagged: Vec<String> = affected
        .iter()
        .filter_map(|rel| ctx.file_to_unit.get(rel))
        .filter(|uid| uid.as_str() != unit.id.as_str())
        .cloned()
        .collect();
    flagged.sort();
    flagged.dedup();
    ctx.shared_state.record_rewind(event.clone(), &flagged);
    let data = to_val(&RewindData {
        trigger: event.trigger().as_str(), rule_key: &event.rule_key, token: &event.token,
        affected_files: affected.len(), flagged_units: &flagged,
    });
    ctx.telemetry.emit_unit("live_rewind", unit, data);
}

fn commit_unit(
    ctx: &WorkerCtx, r: &mut UnitRun, attempt: u32,
    exp_hashes: &BTreeMap<String, String>, verdict: GateVerdict,
) -> bool {
    let res = ctx.mvcc.commit_tx(
        &r.unit.id, exp_hashes, &verdict.final_contents, &verdict.deleted_files,
    );
    match res {
        Ok(CommitOutcome::Committed { head_rev, tx_id, new_hashes }) => {
            r.out.committed = true;
            let after_loc: usize = verdict.final_contents.values().map(|b| count_lines(b)).sum();
            let first = new_hashes.values().next().map_or_else(String::new, String::clone);
            ctx.shared_state.record_commit(
                &r.unit.id, attempt, r.unit.loc, after_loc, verdict.gate2_score_after, first,
            );
            ctx.learn.credit_keys(&r.dir, &r.learn_keys);
            let commit_data = to_val(&CommitData {
                head_rev, tx_id, attempt,
                hygiene_before: verdict.gate2_score_before,
                hygiene_after: verdict.gate2_score_after,
                ruff_errors: verdict.gate3_ruff_errors, ty_errors: verdict.gate3_ty_errors,
                tests_passed: verdict.gate1_tests_passed, formatted_files: verdict.formatted_files,
            });
            ctx.telemetry.emit("mvcc_commit", EventSpec {
                uncached_tpm: Some(ctx.proxy.snapshot().uncached_tpm_rolling),
                data: Some(commit_data), ..EventSpec::unit(&r.unit)
            });
            true
        }
        Ok(CommitOutcome::CasConflict { path, expected, actual }) => {
            r.out.cas_conflicts += 1;
            let msg = format!("MVCC CAS conflict on `{path}` (expected {expected}, disk {actual}); rebasing");
            r.diags = vec![msg];
            r.initial_user = None;
            false
        }
        Err(e) => {
            r.diags = vec![format!("MVCC commit IO error: {e}")];
            false
        }
    }
}

/// Read an SSE stream into raw bytes and decode complete lines preserving UTF-8.
pub(super) async fn collect_sse(
    resp: reqwest::Response, chunk_idle_timeout: Duration,
) -> Result<(u16, String, String)> {
    let status = resp.status().as_u16();
    let is_sse = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));
    if !is_sse {
        let text = resp.text().await.context("read JSON response body")?;
        let Ok(parsed) = serde_json::from_str::<ChatResp>(&text) else {
            return Ok((status, String::new(), text));
        };
        let msg = parsed.first_choice().map(|c| c.message).unwrap_or_default();
        return Ok((status, msg.reasoning.unwrap_or_default(), msg.content.unwrap_or_default()));
    }

    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut buf = Vec::with_capacity(4096);
    let mut stream = resp.bytes_stream();
    let mut on_val = |v: Value| {
        let Ok(chunk) = serde_json::from_value::<ChatResp>(v) else { return };
        let Some(choice) = chunk.first_choice() else { return };
        if let Some(d) = choice.delta.content { content.push_str(&d); }
        if let Some(r) = choice.delta.reasoning { reasoning.push_str(&r); }
    };

    loop {
        let next = tokio::time::timeout(chunk_idle_timeout, stream.next())
            .await
            .map_err(|_| anyhow::anyhow!("SSE idle timeout {}ms", chunk_idle_timeout.as_millis()))?;
        let Some(item) = next else { break };
        buf.extend_from_slice(&item.context("SSE transport stream error mid-response")?);
        drain_sse(&mut buf, false, &mut on_val);
    }

    drain_sse(&mut buf, true, &mut on_val);
    Ok((status, reasoning, content))
}
