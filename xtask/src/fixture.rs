//! Deterministic 10,000-LOC 64-module Python target fixture generator (`cargo xtask gen-fixture`).

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const LAYERS: [&str; 4] = ["core", "engine", "services", "api"];
const BUG_A: [usize; 5] = [1, 13, 27, 39, 51];
const BUG_B: [usize; 5] = [5, 19, 31, 43, 55];
const BUG_C: [usize; 5] = [9, 23, 35, 47, 59];

const DOMAINS: [&str; 16] = [
    "batch ledger engine and core aggregation",
    "token bucket limiter normalization",
    "payload sanitizer and string cleanup",
    "decimal ledger entry normalizer",
    "exponential backoff window planner",
    "sliding window counter evaluator",
    "checksum normalizer and guard",
    "fee schedule pricer helpers",
    "batch chunker and offset math",
    "rate-limit accumulator engine",
    "utf8 field cleaner utilities",
    "variance estimator calculations",
    "currency rounding helpers",
    "bucket refill planner logic",
    "tag canonicalizer pipeline",
    "weighted moving average core",
];

fn scc_peer(idx: usize) -> Option<(&'static str, usize, f64)> {
    match idx {
        16 => Some(("engine", 17, 1.02)),
        17 => Some(("engine", 16, 1.03)),
        32 => Some(("services", 33, 1.02)),
        33 => Some(("services", 32, 1.03)),
        _ => None,
    }
}

fn round_dp(v: f64, dp: i32) -> f64 {
    let scale = 10f64.powi(dp);
    (v * scale).round() / scale
}

fn tier_weights(idx: usize) -> [f64; 8] {
    let shift = (idx % 4) as f64 * 0.01;
    std::array::from_fn(|k| round_dp(1.05 + k as f64 * 0.05 + shift, 2))
}

fn chan_factor(ch: &str) -> Option<f64> {
    match ch {
        "api" => Some(1.02),
        "batch" => Some(0.98),
        "cli" => Some(1.00),
        "web" => Some(1.04),
        _ => None,
    }
}

fn expected_eval(idx: usize, amt: f64, wt: f64, tier: &str, active: bool, code: &str, ch: &str) -> f64 {
    let (rc, c) = (code.trim().to_ascii_lowercase(), ch.trim().to_ascii_lowercase());
    let Some(cf) = chan_factor(&c) else { return 0.0 };
    if !active || rc.is_empty() || wt <= 0.0 || amt < 10.0 { return 0.0; }
    let res = (amt * wt * cf).clamp(0.0, 100.0);
    let clean = tier.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let tw = tier_weights(idx);
    let mult = clean.strip_prefix('t').and_then(|n| n.parse::<usize>().ok())
        .and_then(|k| tw.get(k).copied()).unwrap_or(1.0);
    round_dp((res * mult * 1.05).clamp(0.0, 100.0), 4)
}

fn render_imports(idx: usize) -> (Vec<String>, Vec<String>) {
    let (layer, local) = (idx / 16, idx % 16);
    let mut lines = vec!["import math".into(), "import os".into(), "from typing import TypedDict".into()];
    let mut fns = Vec::new();
    if layer == 0 { return (lines, fns); }
    lines.push(String::new());
    if let Some((pl, pi, _)) = scc_peer(idx) {
        lines.push(format!("import pkg.{pl}.mod_{pi:02}"));
    }
    match layer {
        1 => {
            let mut deps = [local, (local + 14) % 16];
            deps.sort_unstable();
            for dep in deps {
                let fn_name = format!("compute_weighted_metric_{dep:02}");
                fns.push(fn_name.clone());
                if idx == 18 && dep == 0 {
                    lines.push("from pkg.core.mod_00 import compute_weighted_metric_00, process_batch_00".into());
                } else {
                    lines.push(format!("from pkg.core.mod_{dep:02} import {fn_name}"));
                }
            }
        }
        2 => {
            let (c_fn, e_fn) = (format!("compute_weighted_metric_{local:02}"), format!("compute_weighted_metric_{:02}", 16 + local));
            fns.push(c_fn.clone());
            fns.push(e_fn.clone());
            lines.push(format!("from pkg.core.mod_{local:02} import {c_fn}"));
            if idx == 34 {
                lines.push("from pkg.engine.mod_18 import aggregate_ledger_18, compute_weighted_metric_18".into());
            } else {
                lines.push(format!("from pkg.engine.mod_{:02} import {e_fn}", 16 + local));
            }
        }
        _ => {
            let mut deps = [32 + local, 32 + ((local + 2) % 16)];
            deps.sort_unstable();
            for dep in deps {
                let fn_name = format!("compute_weighted_metric_{dep:02}");
                fns.push(fn_name.clone());
                if idx == 50 && dep == 34 {
                    lines.push("from pkg.services.mod_34 import (".into());
                    lines.push("    compute_weighted_metric_34,".into());
                    lines.push("    run_service_pipeline_34,".into());
                    lines.push(")".into());
                } else {
                    lines.push(format!("from pkg.services.mod_{dep:02} import {fn_name}"));
                }
            }
        }
    }
    (lines, fns)
}

fn hotspot_extra(idx: usize) -> &'static str {
    match idx {
        0 => concat!(
            "\n\ndef process_batch_00(\n    rows: list[tuple[int, float, float]],\n) -> dict[str, float]:\n",
            "    \"\"\"Normalize and aggregate numeric ledger rows deterministically.\"\"\"\n",
            "    if not rows:\n        return {\"count\": 0.0, \"weighted_sum\": 0.0, \"spread\": 0.0, \"checksum\": 0.0}\n",
            "\n    weighted_sum = 0.0\n    checksum = 0.0\n    bucket_acc: list[float] = [0.0] * 8\n",
            "    for idx, amount, weight in rows:\n        clamped = clamp_score_00(amount, 0.0, MAX_SCORE_00)\n",
            "        factor = 1.0 + (idx & 7) * 0.05\n        scaled = clamped * weight * factor\n",
            "        weighted_sum += scaled\n        bucket_acc[idx & 7] += scaled\n",
            "        checksum += ((idx + 1) * round(scaled, 4)) % 9973.0\n",
            "\n    spread = max(bucket_acc) - min(bucket_acc)\n",
            "    return {\n        \"count\": float(len(rows)),\n        \"weighted_sum\": round(weighted_sum, 4),\n",
            "        \"spread\": round(spread, 4),\n        \"checksum\": round(checksum, 4),\n    }",
        ),
        18 => concat!(
            "\n\ndef aggregate_ledger_18(\n    entries: list[tuple[int, float, float, str]],\n) -> dict[str, float]:\n",
            "    \"\"\"Roll up ledger entries by channel using core batch processing.\"\"\"\n",
            "    if not entries:\n        return {\"rows\": 0.0, \"net_total\": 0.0, \"channel_spread\": 0.0, \"checksum\": 0.0}\n",
            "\n    core_rows: list[tuple[int, float, float]] = []\n    channel_totals: dict[str, float] = {}\n",
            "    for idx, amount, weight, channel in entries:\n        ch_mult = CHANNEL_FACTORS_18.get(channel, 1.0)\n",
            "        adj_weight = round(weight * ch_mult, 6)\n        core_rows.append((idx, amount, adj_weight))\n",
            "        metric = compute_weighted_metric_02(amount, adj_weight)\n",
            "        channel_totals[channel] = channel_totals.get(channel, 0.0) + metric\n",
            "\n    batch_stats = process_batch_00(core_rows)\n    totals_vals = list(channel_totals.values())\n",
            "    ch_spread = max(totals_vals) - min(totals_vals) if totals_vals else 0.0\n",
            "    net_total = round(batch_stats[\"weighted_sum\"] + sum(totals_vals), 4)\n",
            "    combined_check = round((batch_stats[\"checksum\"] + ch_spread * 13.0) % 1000003.0, 4)\n",
            "    return {\n        \"rows\": batch_stats[\"count\"],\n        \"net_total\": net_total,\n",
            "        \"channel_spread\": round(ch_spread, 4),\n        \"checksum\": combined_check,\n    }",
        ),
        34 => concat!(
            "\n\ndef run_service_pipeline_34(\n    entries: list[tuple[int, float, float, str]],\n) -> dict[str, float]:\n",
            "    \"\"\"Execute Layer 2 ledger service pipeline over engine aggregation.\"\"\"\n",
            "    if not entries:\n        return {\"rows\": 0.0, \"service_total\": 0.0, \"checksum\": 0.0}\n",
            "\n    agg = aggregate_ledger_18(entries)\n    service_factor = compute_weighted_metric_02(1.0, 1.04)\n",
            "    service_total = round(agg[\"net_total\"] * service_factor, 4)\n",
            "    raw_check = agg[\"checksum\"] * 7.0 + agg[\"channel_spread\"]\n",
            "    return {\"rows\": agg[\"rows\"], \"service_total\": service_total, \"checksum\": round(raw_check % 1000003.0, 4)}",
        ),
        50 => concat!(
            "\n\ndef handle_api_request_50(\n    entries: list[tuple[int, float, float, str]],\n) -> dict[str, float]:\n",
            "    \"\"\"Execute Layer 3 API handler returning aggregated ledger response.\"\"\"\n",
            "    if not entries:\n        return {\"status\": 200.0, \"rows\": 0.0, \"api_total\": 0.0, \"checksum\": 0.0}\n",
            "\n    svc = run_service_pipeline_34(entries)\n    api_scale = compute_weighted_metric_36(1.0, 1.0)\n",
            "    api_total = round(svc[\"service_total\"] * api_scale, 4)\n",
            "    raw_check = svc[\"checksum\"] + api_total * 0.01\n",
            "    return {\"status\": 200.0, \"rows\": svc[\"rows\"], \"api_total\": api_total, \"checksum\": round(raw_check % 1000003.0, 4)}",
        ),
        _ => "",
    }
}

fn render_module(idx: usize) -> String {
    let (layer, layer_name, local) = (idx / 16, LAYERS[idx / 16], idx % 16);
    let tag = format!("{idx:02}");
    let tw = tier_weights(idx);
    let heavy = idx % 5 == 2;
    let (imports, used_fns) = render_imports(idx);
    let mut s = String::with_capacity(4096);
    let _ = writeln!(s, "\"\"\"Layer {layer} ({layer_name}) module mod_{tag}: {}.\"\"\"\n", DOMAINS[local]);
    for imp in imports { let _ = writeln!(s, "{imp}"); }
    let clamp_ret = if BUG_B.contains(&idx) { "    return lower if value > upper else value" } else { "    return upper if value > upper else value" };
    let _ = write!(
        s,
        "\n\nMODULE_ID_{tag}: str = \"{layer_name}.mod_{tag}\"\nMIN_ELIGIBLE_{tag}: float = 10.0\nMAX_SCORE_{tag}: float = 100.0\nDEFAULT_BONUS_{tag}: float = 0.25\n\n\
TIER_WEIGHTS_{tag}: dict[str, float] = {{\n    \"t0\": {:.2}, \"t1\": {:.2},\n    \"t2\": {:.2}, \"t3\": {:.2},\n    \"t4\": {:.2}, \"t5\": {:.2},\n    \"t6\": {:.2}, \"t7\": {:.2},\n}}\n\n\
CHANNEL_FACTORS_{tag}: dict[str, float] = {{\n    \"api\": 1.02, \"batch\": 0.98,\n    \"cli\": 1.00, \"web\": 1.04,\n}}\n\n\n\
class Record{tag}(TypedDict):\n    amount: float\n    weight: float\n    tier: str\n    active: bool\n    code: str\n    channel: str\n\n\n\
class BatchSummary{tag}(TypedDict):\n    count: int\n    total: float\n    average: float\n    peak: float\n\n\n\
def normalize_token_{tag}(raw: str) -> str:\n    \"\"\"Trim and lowercase an input identifier token.\"\"\"\n    return raw.strip().lower()\n\n\n\
def clamp_score_{tag}(\n    value: float, lower: float = 0.0, upper: float = 100.0\n) -> float:\n    \"\"\"Clamp value inclusively into [lower, upper].\"\"\"\n    if value < lower:\n        return lower\n{clamp_ret}\n\n\n\
def compute_weighted_metric_{tag}(value: float, factor: float) -> float:\n    \"\"\"Scale a positive metric value by a positive factor.\"\"\"\n    if value <= 0.0 or factor <= 0.0:\n        return 0.0\n    return round(value * factor, 4)\n\n\n",
        tw[0], tw[1], tw[2], tw[3], tw[4], tw[5], tw[6], tw[7],
    );
    if layer == 0 {
        let _ = writeln!(s, "def compose_layer_signal_{tag}(amount: float, weight: float) -> float:\n    \"\"\"Compute Layer 0 base signal using local weighted metric.\"\"\"\n    clamped = clamp_score_{tag}(amount, 0.0, MAX_SCORE_{tag})\n    return compute_weighted_metric_{tag}(clamped, weight)");
    } else {
        let _ = writeln!(s, "def compose_layer_signal_{tag}(amount: float, weight: float) -> float:\n    \"\"\"Combine lower-layer signals for Layer {layer} module {tag}.\"\"\"\n    left = {}(amount, weight)\n    right = {}(amount, weight * 0.5)\n    return round(left + right, 4)", used_fns[0], used_fns[1]);
    }
    let _ = writeln!(s, "\n\ndef _helper_{tag}(items: list[float] | None = []) -> list[float]:\n    \"\"\"Return a shallow copy of buffered numeric items.\"\"\"\n    return list(items)\n\n\ndef adjust_optional_{tag}(base: float, bonus: float | None = None) -> float:\n    \"\"\"Add an optional bonus to base, treating None as zero.\"\"\"\n    return round(base + bonus, 4)");
    if let Some((pl, pi, sc)) = scc_peer(idx) {
        let _ = writeln!(s, "\n\ndef peer_scale_{tag}(value: float) -> float:\n    \"\"\"Return local peer scaling factor without recursing into sibling.\"\"\"\n    if value <= 0.0:\n        return 0.0\n    return round(value * {sc}, 4)\n\n\ndef cross_reconcile_{tag}(value: float) -> float:\n    \"\"\"Reconcile value via cyclic SCC peer pkg.{pl}.mod_{pi:02}.\"\"\"\n    peer_val = pkg.{pl}.mod_{pi:02}.peer_scale_{pi:02}(value)\n    return round(peer_val + peer_scale_{tag}(value), 4)");
    }
    let extra = hotspot_extra(idx);
    if !extra.is_empty() { let _ = writeln!(s, "{extra}"); }
    let cmp_op = if BUG_A.contains(&idx) { ">" } else { ">=" };
    let _ = writeln!(s, "\n\ndef evaluate_record_{tag}(record: Record{tag}) -> float:\n    \"\"\"Evaluate a ledger record; apply tier multiplier when amount >= 10.0.\"\"\"\n    # Lower and strip raw_code from record\n    raw_code = record[\"code\"].strip().lower()\n    channel = record[\"channel\"].strip().lower()\n    result = 0.0\n    active_count = 0");
    if heavy {
        let _ = writeln!(s, "    # Normalize channel_key from channel\n    channel_key = channel.strip().lower().replace(\"-\", \"_\").replace(\" \", \"_\"); is_enabled = True if record[\"active\"] else (False if len(raw_code) == 0 else False)\n    if is_enabled:\n        if len(raw_code) > 0:\n            if record[\"weight\"] > 0.0:\n                if channel_key in CHANNEL_FACTORS_{tag}:\n                    if record[\"amount\"] {cmp_op} MIN_ELIGIBLE_{tag}:\n                        channel_factor = CHANNEL_FACTORS_{tag}[channel_key]\n                        raw_product = record[\"amount\"] * record[\"weight\"]\n                        base_scaled = raw_product * channel_factor\n                        result = clamp_score_{tag}(base_scaled, 0.0, MAX_SCORE_{tag})\n                        # Increment active_count by one\n                        active_count += 1");
    } else {
        let _ = writeln!(s, "    if record[\"active\"]:\n        if len(raw_code) > 0:\n            if record[\"weight\"] > 0.0:\n                if channel in CHANNEL_FACTORS_{tag}:\n                    if record[\"amount\"] {cmp_op} MIN_ELIGIBLE_{tag}:\n                        channel_factor = CHANNEL_FACTORS_{tag}[channel]\n                        base_scaled = record[\"amount\"] * record[\"weight\"] * channel_factor\n                        result = clamp_score_{tag}(base_scaled, 0.0, MAX_SCORE_{tag})\n                        # Increment active_count by one\n                        active_count += 1");
    }
    let _ = writeln!(s, "    if active_count == 0:\n        return 0.0\n    clean_tag = record[\"tier\"].strip().lower().replace(\"-\", \"_\").replace(\" \", \"_\"); bonus_mult = 1.05 if active_count > 0 else (1.02 if result > 50.0 else 1.0)");
    for (k, w) in tw.iter().enumerate() {
        let kw = if k == 0 { "if" } else { "elif" };
        let _ = writeln!(s, "    {kw} clean_tag == \"t{k}\":\n        tier_mult = {w:.2}");
    }
    let _ = writeln!(s, "    else:\n        tier_mult = 1.0\n    final_score = clamp_score_{tag}(result * tier_mult * bonus_mult, 0.0, MAX_SCORE_{tag}); return round(final_score, 4) if final_score >= 0.0 else 0.0\n\n\ndef summarize_batch_{tag}(\n    primary: list[Record{tag}],\n    secondary: list[Record{tag}],\n) -> BatchSummary{tag}:\n    \"\"\"Summarize primary and secondary batches; average is 0.0 when empty.\"\"\"\n    validated: list[Record{tag}] = []");
    let block = if heavy {
        format!("        norm_code = normalize_token_{tag}(item[\"code\"])\n        norm_chan = normalize_token_{tag}(item[\"channel\"])\n        norm_tier = normalize_token_{tag}(item[\"tier\"])\n        checked_amt = clamp_score_{tag}(item[\"amount\"], 0.0, MAX_SCORE_{tag})\n        scaled_wt = adjust_optional_{tag}(item[\"weight\"], DEFAULT_BONUS_{tag})\n        check_val = compute_weighted_metric_{tag}(checked_amt, scaled_wt)\n        if norm_code and norm_chan and norm_tier and check_val >= 0.0:\n            validated.append(item)\n")
    } else {
        format!("        norm_code = normalize_token_{tag}(item[\"code\"])\n        norm_chan = normalize_token_{tag}(item[\"channel\"])\n        checked_amt = clamp_score_{tag}(item[\"amount\"], 0.0, MAX_SCORE_{tag})\n        scaled_wt = adjust_optional_{tag}(item[\"weight\"], DEFAULT_BONUS_{tag})\n        if norm_code and norm_chan and checked_amt >= 0.0 and scaled_wt > 0.0:\n            validated.append(item)\n")
    };
    let avg = if BUG_C.contains(&idx) { "    average = round(total_score / count, 4)" } else { "    average = round(total_score / count, 4) if count > 0 else 0.0" };
    let _ = writeln!(s, "    for item in primary:\n{block}\n    for item in secondary:\n{block}\n    total_score = 0.0\n    peak_score = 0.0\n    for item in validated:\n        scored_val = evaluate_record_{tag}(item)\n        total_score += compute_weighted_metric_{tag}(scored_val, 1.0)\n        if scored_val > peak_score:\n            peak_score = scored_val\n    count = len(validated)\n{avg}\n    return {{\n        \"count\": count,\n        \"total\": round(total_score, 4),\n        \"average\": average,\n        \"peak\": round(peak_score, 4),\n    }}");
    s
}

fn render_test(idx: usize) -> String {
    let (layer, layer_name) = (idx / 16, LAYERS[idx / 16]);
    let tag = format!("{idx:02}");
    let mut fns = vec![
        format!("adjust_optional_{tag}"), format!("clamp_score_{tag}"),
        format!("compose_layer_signal_{tag}"), format!("evaluate_record_{tag}"),
        format!("summarize_batch_{tag}"),
    ];
    if scc_peer(idx).is_some() { fns.push(format!("cross_reconcile_{tag}")); }
    match idx {
        0 => fns.push("process_batch_00".into()),
        18 => fns.push("aggregate_ledger_18".into()),
        34 => fns.push("run_service_pipeline_34".into()),
        50 => fns.push("handle_api_request_50".into()),
        _ => {}
    }
    fns.sort();
    let std_score = expected_eval(idx, 20.0, 1.0, "t2", true, " REC-01 ", "web");
    let bnd_score = expected_eval(idx, 10.0, 1.0, "t0", true, "BND-01", "cli");
    let exp_sig = if layer == 0 { 10.0 } else { 15.0 };
    let batch = [
        (20.0, 1.0, "t1", true, "P-01", "web"),
        (30.0, 1.0, "t3", true, "P-02", "api"),
        (25.0, 1.0, "t2", true, "S-01", "cli"),
    ];
    let (mut total, mut peak) = (0.0f64, 0.0f64);
    for (a, w, t, act, c, ch) in batch {
        let sc = expected_eval(idx, a, w, t, act, c, ch);
        if sc > 0.0 { total += round_dp(sc, 4); }
        peak = peak.max(sc);
    }
    let (exp_total, exp_avg, exp_peak) = (round_dp(total, 4), round_dp(total / 3.0, 4), round_dp(peak, 4));
    let mut s = format!("\"\"\"Unit tests for pkg.{layer_name}.mod_{tag}.\"\"\"\n\nfrom pkg.{layer_name}.mod_{tag} import (\n    Record{tag},\n");
    for f in fns { let _ = writeln!(s, "    {f},"); }
    let _ = write!(
        s,
        ")\n\n\ndef _make_record_{tag}(\n    amount: float,\n    weight: float = 1.0,\n    tier: str = \"t2\",\n    active: bool = True,\n    code: str = \" REC-01 \",\n    channel: str = \"web\",\n) -> Record{tag}:\n    return {{\"amount\": amount, \"weight\": weight, \"tier\": tier, \"active\": active, \"code\": code, \"channel\": channel}}\n\n\n\
def test_evaluate_and_clamp_{tag}() -> None:\n    assert adjust_optional_{tag}(10.0, 2.5) == 12.5\n    assert compose_layer_signal_{tag}(10.0, 1.0) == {exp_sig}\n    assert clamp_score_{tag}(150.0, 0.0, 100.0) == 100.0\n    assert clamp_score_{tag}(-5.0, 0.0, 100.0) == 0.0\n\n\
    rec = _make_record_{tag}(20.0)\n    assert abs(evaluate_record_{tag}(rec) - {std_score}) < 1e-4\n    inactive = _make_record_{tag}(20.0, active=False)\n    assert evaluate_record_{tag}(inactive) == 0.0\n\n\
    boundary = _make_record_{tag}(10.0, tier=\"t0\", code=\"BND-01\", channel=\"cli\")\n    assert abs(evaluate_record_{tag}(boundary) - {bnd_score}) < 1e-4\n\n\n\
def test_summarize_batch_{tag}() -> None:\n    primary = [_make_record_{tag}(20.0, tier=\"t1\", code=\"P-01\", channel=\"web\"), _make_record_{tag}(30.0, tier=\"t3\", code=\"P-02\", channel=\"api\")]\n    secondary = [_make_record_{tag}(25.0, tier=\"t2\", code=\"S-01\", channel=\"cli\")]\n    summary = summarize_batch_{tag}(primary, secondary)\n    assert summary[\"count\"] == 3\n    assert abs(summary[\"total\"] - {exp_total}) < 1e-3\n    assert abs(summary[\"average\"] - {exp_avg}) < 1e-3\n    assert abs(summary[\"peak\"] - {exp_peak}) < 1e-3\n\n\
    empty = summarize_batch_{tag}([], [])\n    assert empty[\"count\"] == 0\n    assert empty[\"average\"] == 0.0\n",
    );
    if scc_peer(idx).is_some() { let _ = writeln!(s, "    assert cross_reconcile_{tag}(10.0) == 20.5"); }
    match idx {
        0 => { let _ = writeln!(s, "    batch_res = process_batch_00([(0, 20.0, 1.0), (1, 30.0, 1.5)])\n    assert batch_res[\"count\"] == 2.0\n    assert batch_res[\"weighted_sum\"] > 0.0"); }
        18 => { let _ = writeln!(s, "    eng_res = aggregate_ledger_18([(0, 20.0, 1.0, \"web\"), (1, 30.0, 1.0, \"api\")])\n    assert eng_res[\"rows\"] == 2.0\n    assert eng_res[\"net_total\"] > 0.0"); }
        34 => { let _ = writeln!(s, "    svc_res = run_service_pipeline_34([(0, 20.0, 1.0, \"web\")])\n    assert svc_res[\"rows\"] == 1.0\n    assert svc_res[\"service_total\"] > 0.0"); }
        50 => { let _ = writeln!(s, "    api_res = handle_api_request_50([(0, 20.0, 1.0, \"web\")])\n    assert api_res[\"status\"] == 200.0\n    assert api_res[\"rows\"] == 1.0"); }
        _ => {}
    }
    s
}

pub fn generate(output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir).with_context(|| format!("create {}", output_dir.display()))?;
    for sub in ["pkg", "tests"] {
        let p = output_dir.join(sub);
        if p.exists() { fs::remove_dir_all(&p)?; }
    }
    let (pkg_dir, tests_dir) = (output_dir.join("pkg"), output_dir.join("tests"));
    fs::create_dir_all(&pkg_dir)?;
    fs::create_dir_all(&tests_dir)?;
    fs::write(pkg_dir.join("__init__.py"), "\"\"\"Refit 64-module modular Python target fixture package.\"\"\"\n")?;
    for layer in LAYERS {
        let d = pkg_dir.join(layer);
        fs::create_dir_all(&d)?;
        fs::write(d.join("__init__.py"), format!("\"\"\"Refit target fixture {layer} layer.\"\"\"\n"))?;
    }
    fs::write(
        output_dir.join("pyproject.toml"),
        "[project]\nname = \"refit-target-fixture\"\nversion = \"0.1.0\"\nrequires-python = \">=3.11\"\n\n\
[tool.ruff]\nline-length = 88\n\n[tool.ruff.lint]\nselect = [\"E\", \"F\", \"W\", \"B\", \"I\"]\n\n\
[tool.ruff.lint.isort]\nknown-first-party = [\"pkg\"]\n\n[tool.pytest.ini_options]\npythonpath = [\".\"]\ntestpaths = [\"tests\"]\n",
    )?;

    let mut total_loc = 0usize;
    let mut per_layer: BTreeMap<&str, usize> = LAYERS.into_iter().map(|l| (l, 0)).collect();
    for idx in 0..64 {
        let layer = LAYERS[idx / 16];
        let src = render_module(idx);
        let loc = src.lines().count();
        total_loc += loc;
        *per_layer.get_mut(layer).unwrap() += loc;
        fs::write(pkg_dir.join(layer).join(format!("mod_{idx:02}.py")), src)?;
        fs::write(tests_dir.join(format!("test_mod_{idx:02}.py")), render_test(idx))?;
    }
    if !(9400..=10800).contains(&total_loc) {
        bail!("generated fixture LOC {total_loc} outside expected [9400, 10800] bounds");
    }
    let summary = json!({
        "ok": true,
        "output_dir": output_dir.display().to_string(),
        "source_modules": 64,
        "test_files": 64,
        "total_tests": 128,
        "seeded_failing_tests": BUG_A.len() + BUG_B.len() + BUG_C.len(),
        "scc_pairs": ["pkg.engine.mod_16<->mod_17", "pkg.services.mod_32<->mod_33"],
        "total_source_loc": total_loc,
        "avg_loc_per_module": round_dp(total_loc as f64 / 64.0, 2),
        "per_layer_loc": per_layer,
    });
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
