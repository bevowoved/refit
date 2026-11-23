"""Deterministic 10,000-LOC 64-module Python fixture generator for refit."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

LAYER_NAMES: tuple[str, str, str, str] = ("core", "engine", "services", "api")

BUG_KIND_A: frozenset[int] = frozenset({1, 13, 27, 39, 51})
BUG_KIND_B: frozenset[int] = frozenset({5, 19, 31, 43, 55})
BUG_KIND_C: frozenset[int] = frozenset({9, 23, 35, 47, 59})
BUGGY_MODULES: frozenset[int] = BUG_KIND_A | BUG_KIND_B | BUG_KIND_C

HEAVY_DEBT_MODULES: frozenset[int] = frozenset(i for i in range(64) if i % 5 == 2)

SCC_PEERS: dict[int, tuple[str, int, float, float]] = {
    16: ("engine", 17, 1.02, 1.03),
    17: ("engine", 16, 1.03, 1.02),
    32: ("services", 33, 1.02, 1.03),
    33: ("services", 32, 1.03, 1.02),
}

DOMAIN_DESCRIPTIONS: tuple[str, ...] = (
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
)

CHANNEL_FACTORS: dict[str, float] = {
    "api": 1.02,
    "batch": 0.98,
    "cli": 1.00,
    "web": 1.04,
}


def _tier_weights(idx: int) -> dict[str, float]:
    shift = (idx % 4) * 0.01
    return {f"t{k}": round(1.05 + k * 0.05 + shift, 2) for k in range(8)}


def _expected_clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if value < lower:
        return lower
    return upper if value > upper else value


def _expected_eval(
    idx: int,
    amount: float,
    weight: float,
    tier: str,
    active: bool,
    code: str,
    channel: str,
) -> float:
    raw_code = code.strip().lower()
    chan = channel.strip().lower()
    if not active or not raw_code or weight <= 0.0:
        return 0.0
    if chan not in CHANNEL_FACTORS or amount < 10.0:
        return 0.0
    base_scaled = amount * weight * CHANNEL_FACTORS[chan]
    result = _expected_clamp(base_scaled, 0.0, 100.0)
    clean_tag = tier.strip().lower().replace("-", "_").replace(" ", "_")
    tier_mult = _tier_weights(idx).get(clean_tag, 1.0)
    final_score = _expected_clamp(result * tier_mult * 1.05, 0.0, 100.0)
    return round(final_score, 4)


def _expected_summary(
    idx: int,
    records: list[tuple[float, float, str, bool, str, str]],
) -> tuple[int, float, float, float]:
    total = 0.0
    peak = 0.0
    valid = 0
    for amount, weight, tier, active, code, channel in records:
        if not code.strip() or not channel.strip():
            continue
        score = _expected_eval(idx, amount, weight, tier, active, code, channel)
        total += round(score * 1.0, 4) if score > 0.0 else 0.0
        if score > peak:
            peak = score
        valid += 1
    avg = round(total / valid, 4) if valid > 0 else 0.0
    return valid, round(total, 4), avg, round(peak, 4)


def _layer_for(idx: int) -> tuple[int, str, int]:
    layer = idx // 16
    return layer, LAYER_NAMES[layer], idx % 16


def _render_imports(idx: int) -> tuple[list[str], list[tuple[str, str]]]:
    layer, _layer_name, local = _layer_for(idx)
    lines: list[str] = [
        "import math",
        "import os",
        "from typing import TypedDict",
    ]
    used_metrics: list[tuple[str, str]] = []

    if layer == 0:
        return lines, used_metrics

    lines.append("")
    if idx in SCC_PEERS:
        peer_layer, peer_idx, _, _ = SCC_PEERS[idx]
        lines.append(f"import pkg.{peer_layer}.mod_{peer_idx:02d}")

    if layer == 1:
        dep_a = local
        dep_b = (local + 14) % 16
        for dep in sorted((dep_a, dep_b)):
            fn_name = f"compute_weighted_metric_{dep:02d}"
            used_metrics.append((f"pkg.core.mod_{dep:02d}", fn_name))
            if idx == 18 and dep == 0:
                lines.append(
                    "from pkg.core.mod_00 import "
                    "compute_weighted_metric_00, process_batch_00"
                )
            else:
                lines.append(f"from pkg.core.mod_{dep:02d} import {fn_name}")
    elif layer == 2:
        core_dep = local
        eng_dep = 16 + local
        core_fn = f"compute_weighted_metric_{core_dep:02d}"
        eng_fn = f"compute_weighted_metric_{eng_dep:02d}"
        used_metrics.append((f"pkg.core.mod_{core_dep:02d}", core_fn))
        used_metrics.append((f"pkg.engine.mod_{eng_dep:02d}", eng_fn))
        lines.append(f"from pkg.core.mod_{core_dep:02d} import {core_fn}")
        if idx == 34:
            lines.append(
                "from pkg.engine.mod_18 import "
                "aggregate_ledger_18, compute_weighted_metric_18"
            )
        else:
            lines.append(f"from pkg.engine.mod_{eng_dep:02d} import {eng_fn}")
    else:
        svc_a = 32 + local
        svc_b = 32 + ((local + 2) % 16)
        for dep in sorted((svc_a, svc_b)):
            fn_name = f"compute_weighted_metric_{dep:02d}"
            used_metrics.append((f"pkg.services.mod_{dep:02d}", fn_name))
            if idx == 50 and dep == 34:
                lines.extend(
                    [
                        "from pkg.services.mod_34 import (",
                        "    compute_weighted_metric_34,",
                        "    run_service_pipeline_34,",
                        ")",
                    ]
                )
            else:
                lines.append(f"from pkg.services.mod_{dep:02d} import {fn_name}")

    return lines, used_metrics


def _render_hotspot_extras(idx: int) -> list[str]:
    if idx == 0:
        return [
            "",
            "",
            "def process_batch_00(",
            "    rows: list[tuple[int, float, float]],",
            ") -> dict[str, float]:",
            '    """Normalize and aggregate numeric ledger rows deterministically."""',
            "    if not rows:",
            "        return {",
            '            "count": 0.0,',
            '            "weighted_sum": 0.0,',
            '            "spread": 0.0,',
            '            "checksum": 0.0,',
            "        }",
            "",
            "    weighted_sum = 0.0",
            "    checksum = 0.0",
            "    bucket_acc: list[float] = [0.0] * 8",
            "    for idx, amount, weight in rows:",
            "        clamped = clamp_score_00(amount, 0.0, MAX_SCORE_00)",
            "        factor = 1.0 + (idx & 7) * 0.05",
            "        scaled = clamped * weight * factor",
            "        weighted_sum += scaled",
            "        bucket_acc[idx & 7] += scaled",
            "        checksum += ((idx + 1) * round(scaled, 4)) % 9973.0",
            "",
            "    spread = max(bucket_acc) - min(bucket_acc)",
            "    return {",
            '        "count": float(len(rows)),',
            '        "weighted_sum": round(weighted_sum, 4),',
            '        "spread": round(spread, 4),',
            '        "checksum": round(checksum, 4),',
            "    }",
        ]
    if idx == 18:
        return [
            "",
            "",
            "def aggregate_ledger_18(",
            "    entries: list[tuple[int, float, float, str]],",
            ") -> dict[str, float]:",
            '    """Roll up ledger entries by channel using core batch processing."""',
            "    if not entries:",
            "        return {",
            '            "rows": 0.0,',
            '            "net_total": 0.0,',
            '            "channel_spread": 0.0,',
            '            "checksum": 0.0,',
            "        }",
            "",
            "    core_rows: list[tuple[int, float, float]] = []",
            "    channel_totals: dict[str, float] = {}",
            "    for idx, amount, weight, channel in entries:",
            "        ch_mult = CHANNEL_FACTORS_18.get(channel, 1.0)",
            "        adj_weight = round(weight * ch_mult, 6)",
            "        core_rows.append((idx, amount, adj_weight))",
            "        metric = compute_weighted_metric_02(amount, adj_weight)",
            "        channel_totals[channel] = channel_totals.get(channel, 0.0) + metric",
            "",
            "    batch_stats = process_batch_00(core_rows)",
            "    totals_vals = list(channel_totals.values())",
            "    ch_spread = max(totals_vals) - min(totals_vals) if totals_vals else 0.0",
            '    net_total = round(batch_stats["weighted_sum"] + sum(totals_vals), 4)',
            "    combined_check = round(",
            '        (batch_stats["checksum"] + ch_spread * 13.0) % 1000003.0,',
            "        4,",
            "    )",
            "    return {",
            '        "rows": batch_stats["count"],',
            '        "net_total": net_total,',
            '        "channel_spread": round(ch_spread, 4),',
            '        "checksum": combined_check,',
            "    }",
        ]
    if idx == 34:
        return [
            "",
            "",
            "def run_service_pipeline_34(",
            "    entries: list[tuple[int, float, float, str]],",
            ") -> dict[str, float]:",
            '    """Execute Layer 2 ledger service pipeline over engine aggregation."""',
            "    if not entries:",
            '        return {"rows": 0.0, "service_total": 0.0, "checksum": 0.0}',
            "",
            "    agg = aggregate_ledger_18(entries)",
            "    service_factor = compute_weighted_metric_02(1.0, 1.04)",
            '    service_total = round(agg["net_total"] * service_factor, 4)',
            "    raw_check = agg[\"checksum\"] * 7.0 + agg[\"channel_spread\"]",
            "    return {",
            '        "rows": agg["rows"],',
            '        "service_total": service_total,',
            '        "checksum": round(raw_check % 1000003.0, 4),',
            "    }",
        ]
    if idx == 50:
        return [
            "",
            "",
            "def handle_api_request_50(",
            "    entries: list[tuple[int, float, float, str]],",
            ") -> dict[str, float]:",
            '    """Execute Layer 3 API handler returning aggregated ledger response."""',
            "    if not entries:",
            "        return {",
            '            "status": 200.0,',
            '            "rows": 0.0,',
            '            "api_total": 0.0,',
            '            "checksum": 0.0,',
            "        }",
            "",
            "    svc = run_service_pipeline_34(entries)",
            "    api_scale = compute_weighted_metric_36(1.0, 1.0)",
            '    api_total = round(svc["service_total"] * api_scale, 4)',
            '    raw_check = svc["checksum"] + api_total * 0.01',
            "    return {",
            '        "status": 200.0,',
            '        "rows": svc["rows"],',
            '        "api_total": api_total,',
            '        "checksum": round(raw_check % 1000003.0, 4),',
            "    }",
        ]
    return []


def _render_scc_helpers(idx: int) -> list[str]:
    if idx not in SCC_PEERS:
        return []
    peer_layer, peer_idx, own_scale, _ = SCC_PEERS[idx]
    tag = f"{idx:02d}"
    ptag = f"{peer_idx:02d}"
    return [
        "",
        "",
        f"def peer_scale_{tag}(value: float) -> float:",
        '    """Return local peer scaling factor without recursing into sibling."""',
        "    if value <= 0.0:",
        "        return 0.0",
        f"    return round(value * {own_scale}, 4)",
        "",
        "",
        f"def cross_reconcile_{tag}(value: float) -> float:",
        f'    """Reconcile value via cyclic SCC peer pkg.{peer_layer}.mod_{ptag}."""',
        f"    peer_val = pkg.{peer_layer}.mod_{ptag}.peer_scale_{ptag}(value)",
        f"    return round(peer_val + peer_scale_{tag}(value), 4)",
    ]


def _render_module(idx: int) -> str:
    layer, layer_name, local = _layer_for(idx)
    tag = f"{idx:02d}"
    domain = DOMAIN_DESCRIPTIONS[local]
    weights = _tier_weights(idx)
    heavy = idx in HEAVY_DEBT_MODULES
    import_lines, used_metrics = _render_imports(idx)

    lines: list[str] = [
        f'"""Layer {layer} ({layer_name}) module mod_{tag}: {domain}."""',
        "",
        *import_lines,
        "",
        "",
        f'MODULE_ID_{tag}: str = "{layer_name}.mod_{tag}"',
        f"MIN_ELIGIBLE_{tag}: float = 10.0",
        f"MAX_SCORE_{tag}: float = 100.0",
        f"DEFAULT_BONUS_{tag}: float = 0.25",
        "",
        f"TIER_WEIGHTS_{tag}: dict[str, float] = {{",
        f'    "t0": {weights["t0"]:.2f}, "t1": {weights["t1"]:.2f},',
        f'    "t2": {weights["t2"]:.2f}, "t3": {weights["t3"]:.2f},',
        f'    "t4": {weights["t4"]:.2f}, "t5": {weights["t5"]:.2f},',
        f'    "t6": {weights["t6"]:.2f}, "t7": {weights["t7"]:.2f},',
        "}",
        "",
        f"CHANNEL_FACTORS_{tag}: dict[str, float] = {{",
        '    "api": 1.02, "batch": 0.98,',
        '    "cli": 1.00, "web": 1.04,',
        "}",
        "",
        "",
        f"class Record{tag}(TypedDict):",
        "    amount: float",
        "    weight: float",
        "    tier: str",
        "    active: bool",
        "    code: str",
        "    channel: str",
        "",
        "",
        f"class BatchSummary{tag}(TypedDict):",
        "    count: int",
        "    total: float",
        "    average: float",
        "    peak: float",
        "",
        "",
        f"def normalize_token_{tag}(raw: str) -> str:",
        '    """Trim and lowercase an input identifier token."""',
        "    return raw.strip().lower()",
        "",
        "",
        f"def clamp_score_{tag}(",
        "    value: float, lower: float = 0.0, upper: float = 100.0",
        ") -> float:",
        '    """Clamp value inclusively into [lower, upper]."""',
        "    if value < lower:",
        "        return lower",
        (
            "    return lower if value > upper else value"
            if idx in BUG_KIND_B
            else "    return upper if value > upper else value"
        ),
        "",
        "",
        f"def compute_weighted_metric_{tag}(value: float, factor: float) -> float:",
        '    """Scale a positive metric value by a positive factor."""',
        "    if value <= 0.0 or factor <= 0.0:",
        "        return 0.0",
        "    return round(value * factor, 4)",
        "",
        "",
    ]

    if layer == 0:
        lines.extend(
            [
                f"def compose_layer_signal_{tag}(amount: float, weight: float) -> float:",
                '    """Compute Layer 0 base signal using local weighted metric."""',
                f"    clamped = clamp_score_{tag}(amount, 0.0, MAX_SCORE_{tag})",
                f"    return compute_weighted_metric_{tag}(clamped, weight)",
            ]
        )
    else:
        fn_a = used_metrics[0][1]
        fn_b = used_metrics[1][1]
        lines.extend(
            [
                f"def compose_layer_signal_{tag}(amount: float, weight: float) -> float:",
                f'    """Combine lower-layer signals for Layer {layer} module {tag}."""',
                f"    left = {fn_a}(amount, weight)",
                f"    right = {fn_b}(amount, weight * 0.5)",
                "    return round(left + right, 4)",
            ]
        )

    lines.extend(
        [
            "",
            "",
            f"def _helper_{tag}(items: list[float] | None = []) -> list[float]:",
            '    """Return a shallow copy of buffered numeric items."""',
            "    return list(items)",
            "",
            "",
            f"def adjust_optional_{tag}(base: float, bonus: float | None = None) -> float:",
            '    """Add an optional bonus to base, treating None as zero."""',
            "    return round(base + bonus, 4)",
        ]
    )

    lines.extend(_render_scc_helpers(idx))
    lines.extend(_render_hotspot_extras(idx))

    cmp_op = ">" if idx in BUG_KIND_A else ">="
    lines.extend(
        [
            "",
            "",
            f"def evaluate_record_{tag}(record: Record{tag}) -> float:",
            '    """Evaluate a ledger record; apply tier multiplier when amount >= 10.0."""',
            "    # Lower and strip raw_code from record",
            '    raw_code = record["code"].strip().lower()',
            '    channel = record["channel"].strip().lower()',
            "    result = 0.0",
            "    active_count = 0",
        ]
    )

    if heavy:
        lines.extend(
            [
                "    # Normalize channel_key from channel",
                (
                    '    channel_key = channel.strip().lower().replace("-", "_").replace(" ", "_"); '
                    'is_enabled = True if record["active"] else (False if len(raw_code) == 0 else False)'
                ),
                "    if is_enabled:",
                "        if len(raw_code) > 0:",
                '            if record["weight"] > 0.0:',
                f"                if channel_key in CHANNEL_FACTORS_{tag}:",
                f'                    if record["amount"] {cmp_op} MIN_ELIGIBLE_{tag}:',
                f"                        channel_factor = CHANNEL_FACTORS_{tag}[channel_key]",
                '                        raw_product = record["amount"] * record["weight"]',
                "                        base_scaled = raw_product * channel_factor",
                f"                        result = clamp_score_{tag}(base_scaled, 0.0, MAX_SCORE_{tag})",
                "                        # Increment active_count by one",
                "                        active_count += 1",
            ]
        )
    else:
        lines.extend(
            [
                '    if record["active"]:',
                "        if len(raw_code) > 0:",
                '            if record["weight"] > 0.0:',
                f"                if channel in CHANNEL_FACTORS_{tag}:",
                f'                    if record["amount"] {cmp_op} MIN_ELIGIBLE_{tag}:',
                f"                        channel_factor = CHANNEL_FACTORS_{tag}[channel]",
                '                        base_scaled = record["amount"] * record["weight"] * channel_factor',
                f"                        result = clamp_score_{tag}(base_scaled, 0.0, MAX_SCORE_{tag})",
                "                        # Increment active_count by one",
                "                        active_count += 1",
            ]
        )

    lines.extend(
        [
            "    if active_count == 0:",
            "        return 0.0",
            (
                '    clean_tag = record["tier"].strip().lower().replace("-", "_").replace(" ", "_"); '
                "bonus_mult = 1.05 if active_count > 0 else (1.02 if result > 50.0 else 1.0)"
            ),
            f'    if clean_tag == "t0":',
            f'        tier_mult = {weights["t0"]:.2f}',
            f'    elif clean_tag == "t1":',
            f'        tier_mult = {weights["t1"]:.2f}',
            f'    elif clean_tag == "t2":',
            f'        tier_mult = {weights["t2"]:.2f}',
            f'    elif clean_tag == "t3":',
            f'        tier_mult = {weights["t3"]:.2f}',
            f'    elif clean_tag == "t4":',
            f'        tier_mult = {weights["t4"]:.2f}',
            f'    elif clean_tag == "t5":',
            f'        tier_mult = {weights["t5"]:.2f}',
            f'    elif clean_tag == "t6":',
            f'        tier_mult = {weights["t6"]:.2f}',
            f'    elif clean_tag == "t7":',
            f'        tier_mult = {weights["t7"]:.2f}',
            "    else:",
            "        tier_mult = 1.0",
            (
                f"    final_score = clamp_score_{tag}(result * tier_mult * bonus_mult, 0.0, MAX_SCORE_{tag}); "
                "return round(final_score, 4) if final_score >= 0.0 else 0.0"
            ),
            "",
            "",
            f"def summarize_batch_{tag}(",
            f"    primary: list[Record{tag}],",
            f"    secondary: list[Record{tag}],",
            f") -> BatchSummary{tag}:",
            '    """Summarize primary and secondary batches; average is 0.0 when empty."""',
            f"    validated: list[Record{tag}] = []",
        ]
    )

    if heavy:
        block = [
            f'        norm_code = normalize_token_{tag}(item["code"])',
            f'        norm_chan = normalize_token_{tag}(item["channel"])',
            f'        norm_tier = normalize_token_{tag}(item["tier"])',
            f'        checked_amt = clamp_score_{tag}(item["amount"], 0.0, MAX_SCORE_{tag})',
            f'        scaled_wt = adjust_optional_{tag}(item["weight"], DEFAULT_BONUS_{tag})',
            f"        check_val = compute_weighted_metric_{tag}(checked_amt, scaled_wt)",
            "        if norm_code and norm_chan and norm_tier and check_val >= 0.0:",
            "            validated.append(item)",
        ]
    else:
        block = [
            f'        norm_code = normalize_token_{tag}(item["code"])',
            f'        norm_chan = normalize_token_{tag}(item["channel"])',
            f'        checked_amt = clamp_score_{tag}(item["amount"], 0.0, MAX_SCORE_{tag})',
            f'        scaled_wt = adjust_optional_{tag}(item["weight"], DEFAULT_BONUS_{tag})',
            "        if norm_code and norm_chan and checked_amt >= 0.0 and scaled_wt > 0.0:",
            "            validated.append(item)",
        ]

    avg_line = (
        "    average = round(total_score / count, 4)"
        if idx in BUG_KIND_C
        else "    average = round(total_score / count, 4) if count > 0 else 0.0"
    )

    lines.extend(
        [
            "    for item in primary:",
            *block,
            "",
            "    for item in secondary:",
            *block,
            "",
            "    total_score = 0.0",
            "    peak_score = 0.0",
            "    for item in validated:",
            f"        scored_val = evaluate_record_{tag}(item)",
            f"        total_score += compute_weighted_metric_{tag}(scored_val, 1.0)",
            "        if scored_val > peak_score:",
            "            peak_score = scored_val",
            "    count = len(validated)",
            avg_line,
            "    return {",
            '        "count": count,',
            '        "total": round(total_score, 4),',
            '        "average": average,',
            '        "peak": round(peak_score, 4),',
            "    }",
            "",
        ]
    )

    return "\n".join(lines)


def _render_test(idx: int) -> str:
    layer, layer_name, _ = _layer_for(idx)
    tag = f"{idx:02d}"

    extra_imports: list[str] = []
    if idx in SCC_PEERS:
        extra_imports.append(f"cross_reconcile_{tag}")
    if idx == 0:
        extra_imports.append("process_batch_00")
    elif idx == 18:
        extra_imports.append("aggregate_ledger_18")
    elif idx == 34:
        extra_imports.append("run_service_pipeline_34")
    elif idx == 50:
        extra_imports.append("handle_api_request_50")

    fn_imports = sorted(
        [
            f"adjust_optional_{tag}",
            f"clamp_score_{tag}",
            f"compose_layer_signal_{tag}",
            f"evaluate_record_{tag}",
            f"summarize_batch_{tag}",
            *extra_imports,
        ]
    )

    std_score = _expected_eval(idx, 20.0, 1.0, "t2", True, " REC-01 ", "web")
    boundary_score = _expected_eval(idx, 10.0, 1.0, "t0", True, "BND-01", "cli")
    expected_signal = 10.0 if layer == 0 else 15.0

    batch_records = [
        (20.0, 1.0, "t1", True, "P-01", "web"),
        (30.0, 1.0, "t3", True, "P-02", "api"),
        (25.0, 1.0, "t2", True, "S-01", "cli"),
    ]
    exp_count, exp_total, exp_avg, exp_peak = _expected_summary(idx, batch_records)

    lines: list[str] = [
        f'"""Unit tests for pkg.{layer_name}.mod_{tag}."""',
        "",
        f"from pkg.{layer_name}.mod_{tag} import (",
        f"    Record{tag},",
        *[f"    {name}," for name in fn_imports],
        ")",
        "",
        "",
        f"def _make_record_{tag}(",
        "    amount: float,",
        "    weight: float = 1.0,",
        '    tier: str = "t2",',
        "    active: bool = True,",
        '    code: str = " REC-01 ",',
        '    channel: str = "web",',
        f") -> Record{tag}:",
        "    return {",
        '        "amount": amount,',
        '        "weight": weight,',
        '        "tier": tier,',
        '        "active": active,',
        '        "code": code,',
        '        "channel": channel,',
        "    }",
        "",
        "",
        f"def test_evaluate_and_clamp_{tag}() -> None:",
        f"    assert adjust_optional_{tag}(10.0, 2.5) == 12.5",
        f"    assert compose_layer_signal_{tag}(10.0, 1.0) == {expected_signal}",
        f"    assert clamp_score_{tag}(150.0, 0.0, 100.0) == 100.0",
        f"    assert clamp_score_{tag}(-5.0, 0.0, 100.0) == 0.0",
        "",
        f"    rec = _make_record_{tag}(20.0)",
        f"    assert abs(evaluate_record_{tag}(rec) - {std_score!r}) < 1e-4",
        f"    inactive = _make_record_{tag}(20.0, active=False)",
        f"    assert evaluate_record_{tag}(inactive) == 0.0",
        "",
        f'    boundary = _make_record_{tag}(10.0, tier="t0", code="BND-01", channel="cli")',
        f"    assert abs(evaluate_record_{tag}(boundary) - {boundary_score!r}) < 1e-4",
        "",
        "",
        f"def test_summarize_batch_{tag}() -> None:",
        "    primary = [",
        f'        _make_record_{tag}(20.0, tier="t1", code="P-01", channel="web"),',
        f'        _make_record_{tag}(30.0, tier="t3", code="P-02", channel="api"),',
        "    ]",
        "    secondary = [",
        f'        _make_record_{tag}(25.0, tier="t2", code="S-01", channel="cli"),',
        "    ]",
        f"    summary = summarize_batch_{tag}(primary, secondary)",
        f'    assert summary["count"] == {exp_count}',
        f'    assert abs(summary["total"] - {exp_total!r}) < 1e-3',
        f'    assert abs(summary["average"] - {exp_avg!r}) < 1e-3',
        f'    assert abs(summary["peak"] - {exp_peak!r}) < 1e-3',
        "",
        f"    empty = summarize_batch_{tag}([], [])",
        '    assert empty["count"] == 0',
        '    assert empty["average"] == 0.0',
    ]

    if idx in SCC_PEERS:
        lines.append(f"    assert cross_reconcile_{tag}(10.0) == 20.5")
    if idx == 0:
        lines.extend(
            [
                "    batch_res = process_batch_00([(0, 20.0, 1.0), (1, 30.0, 1.5)])",
                '    assert batch_res["count"] == 2.0',
                '    assert batch_res["weighted_sum"] > 0.0',
            ]
        )
    elif idx == 18:
        lines.extend(
            [
                "    eng_res = aggregate_ledger_18(",
                '        [(0, 20.0, 1.0, "web"), (1, 30.0, 1.0, "api")]',
                "    )",
                '    assert eng_res["rows"] == 2.0',
                '    assert eng_res["net_total"] > 0.0',
            ]
        )
    elif idx == 34:
        lines.extend(
            [
                '    svc_res = run_service_pipeline_34([(0, 20.0, 1.0, "web")])',
                '    assert svc_res["rows"] == 1.0',
                '    assert svc_res["service_total"] > 0.0',
            ]
        )
    elif idx == 50:
        lines.extend(
            [
                '    api_res = handle_api_request_50([(0, 20.0, 1.0, "web")])',
                '    assert api_res["status"] == 200.0',
                '    assert api_res["rows"] == 1.0',
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _render_pyproject() -> str:
    return "\n".join(
        [
            "[project]",
            'name = "refit-target-fixture"',
            'version = "0.1.0"',
            'requires-python = ">=3.11"',
            "",
            "[tool.ruff]",
            "line-length = 88",
            "",
            "[tool.ruff.lint]",
            'select = ["E", "F", "W", "B", "I"]',
            "",
            "[tool.ruff.lint.isort]",
            'known-first-party = ["pkg"]',
            "",
            "[tool.pytest.ini_options]",
            'pythonpath = ["."]',
            'testpaths = ["tests"]',
            "",
        ]
    )


def generate_repo(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("pkg", "tests"):
        sub_path = output_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)

    pkg_dir = output_dir / "pkg"
    tests_dir = output_dir / "tests"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        '"""Refit 64-module modular Python target fixture package."""\n',
        encoding="utf-8",
        newline="\n",
    )
    for layer_name in LAYER_NAMES:
        layer_dir = pkg_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "__init__.py").write_text(
            f'"""Refit target fixture {layer_name} layer."""\n',
            encoding="utf-8",
            newline="\n",
        )

    (output_dir / "pyproject.toml").write_text(
        _render_pyproject(),
        encoding="utf-8",
        newline="\n",
    )

    total_source_loc = 0
    per_layer_loc: dict[str, int] = {name: 0 for name in LAYER_NAMES}
    source_count = 0
    test_count = 0

    for idx in range(64):
        _, layer_name, _ = _layer_for(idx)
        tag = f"{idx:02d}"

        src_code = _render_module(idx)
        ast.parse(src_code, filename=f"pkg/{layer_name}/mod_{tag}.py")
        src_path = pkg_dir / layer_name / f"mod_{tag}.py"
        src_path.write_text(src_code, encoding="utf-8", newline="\n")
        mod_loc = len(src_code.splitlines())
        total_source_loc += mod_loc
        per_layer_loc[layer_name] += mod_loc
        source_count += 1

        test_code = _render_test(idx)
        ast.parse(test_code, filename=f"tests/test_mod_{tag}.py")
        test_path = tests_dir / f"test_mod_{tag}.py"
        test_path.write_text(test_code, encoding="utf-8", newline="\n")
        test_count += 1

    if source_count != 64 or test_count != 64:
        raise RuntimeError(
            f"Expected 64 source and 64 test files, got {source_count}/{test_count}"
        )
    if not (9600 <= total_source_loc <= 10600):
        raise RuntimeError(
            f"Total LOC {total_source_loc} outside required [9600, 10600] bounds"
        )

    return {
        "ok": True,
        "output_dir": str(output_dir),
        "source_modules": source_count,
        "test_files": test_count,
        "total_tests": test_count * 2,
        "seeded_failing_tests": len(BUGGY_MODULES),
        "scc_pairs": ["pkg.engine.mod_16<->mod_17", "pkg.services.mod_32<->mod_33"],
        "total_source_loc": total_source_loc,
        "avg_loc_per_module": round(total_source_loc / source_count, 2),
        "per_layer_loc": per_layer_loc,
    }


def main() -> int:
    default_out = Path(__file__).resolve().parent / "target_repo"
    parser = argparse.ArgumentParser(
        description="Generate deterministic 10,000-LOC 64-module Python target fixture."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(default_out),
        help=f"Target repository directory (default: {default_out})",
    )
    args = parser.parse_args()
    summary = generate_repo(Path(args.output_dir))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
