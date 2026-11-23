"""Deterministic CPU & RSS benchmark workload for refit before/after verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

BATCH_SIZE = 512
CORE_SLICE_SIZE = 128
CHANNELS: tuple[str, str, str, str] = ("api", "batch", "cli", "web")


def _build_iteration_entries(
    iteration: int,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[int, float, float, str]]:
    offset = (iteration * 31) % 97
    entries: list[tuple[int, float, float, str]] = []
    for row in range(batch_size):
        idx = row + offset
        amount = 10.0 + float((idx * 17 + iteration * 7) % 85) + ((row % 5) * 0.25)
        weight = 0.75 + float((idx * 11) % 10) * 0.1
        channel = CHANNELS[(row + iteration) & 3]
        entries.append((idx, amount, weight, channel))
    return entries


def _load_target_modules(repo_path: Path) -> tuple[Any, Any, Any, Any]:
    resolved = str(repo_path.resolve())
    if not sys.path or sys.path[0] != resolved:
        sys.path.insert(0, resolved)

    for key in list(sys.modules):
        if key == "pkg" or key.startswith("pkg."):
            del sys.modules[key]
    importlib.invalidate_caches()

    mod_00 = importlib.import_module("pkg.core.mod_00")
    mod_18 = importlib.import_module("pkg.engine.mod_18")
    mod_34 = importlib.import_module("pkg.services.mod_34")
    mod_50 = importlib.import_module("pkg.api.mod_50")
    return mod_00, mod_18, mod_34, mod_50


def run_benchmark(
    repo_path: Path,
    iterations: int,
    expected_checksum: str | None = None,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError(f"--iterations must be >= 1, got {iterations}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    mod_00, mod_18, mod_34, mod_50 = _load_target_modules(repo_path)

    hasher = hashlib.sha256()
    total_rows = 0
    grand_total = 0.0
    status_ok = True

    t0 = time.perf_counter()
    for it in range(iterations):
        entries = _build_iteration_entries(it, BATCH_SIZE)
        api_res: dict[str, float] = mod_50.handle_api_request_50(entries)
        core_slice = [
            (idx, amount, weight)
            for idx, amount, weight, _ in entries[:CORE_SLICE_SIZE]
        ]
        core_res: dict[str, float] = mod_00.process_batch_00(core_slice)

        if api_res["status"] != 200.0 or int(api_res["rows"]) != BATCH_SIZE:
            status_ok = False
        if int(core_res["count"]) != CORE_SLICE_SIZE:
            status_ok = False

        total_rows += int(api_res["rows"])
        grand_total += api_res["api_total"]

        digest_line = (
            f"{it}:{int(api_res['rows'])}:{api_res['api_total']:.4f}:"
            f"{api_res['checksum']:.4f}:{core_res['checksum']:.4f}\n"
        )
        hasher.update(digest_line.encode("ascii"))

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    checksum = hasher.hexdigest()[:16]

    probe = _build_iteration_entries(0, 64)
    probe_api_1 = mod_50.handle_api_request_50(probe)
    probe_api_2 = mod_50.handle_api_request_50(probe)
    probe_svc = mod_34.run_service_pipeline_34(probe)
    probe_eng = mod_18.aggregate_ledger_18(probe)

    deterministic_ok = (
        probe_api_1 == probe_api_2
        and int(probe_svc["rows"]) == 64
        and int(probe_eng["rows"]) == 64
        and probe_api_1["api_total"] > 0.0
        and grand_total > 0.0
        and total_rows == iterations * BATCH_SIZE
    )
    checksum_ok = expected_checksum is None or checksum == expected_checksum
    ok = bool(status_ok and deterministic_ok and checksum_ok)

    return {
        "ok": ok,
        "checksum": checksum,
        "wall_ms": wall_ms,
        "rows": total_rows,
    }


def main() -> int:
    default_repo = (
        Path(__file__).resolve().parent.parent / "fixture" / "target_repo"
    )
    parser = argparse.ArgumentParser(
        description="Run deterministic 4-layer CPU/RSS benchmark on target fixture."
    )
    parser.add_argument(
        "--repo",
        default=str(default_repo),
        help=f"Target repository path (default: {default_repo})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Number of deterministic benchmark iterations (default: 150)",
    )
    parser.add_argument(
        "--expected-checksum",
        default=None,
        help="Optional 16-hex checksum to assert parity against.",
    )
    args = parser.parse_args()

    result = run_benchmark(
        repo_path=Path(args.repo),
        iterations=args.iterations,
        expected_checksum=args.expected_checksum,
    )
    print(json.dumps(result, separators=(", ", ": ")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
