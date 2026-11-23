# `refit`

Cross-platform Rust and OpenClaw refactoring engine for **Cerebras WSE-3T** (`moonshotai-kimi-k3.1-pro-trial` and `qwen-3.8-27b`).

Built to survive the live **50K uncached TPM / 500K total TPM** hackathon rate limit and **30s+ server queues** without stalling, corrupting files, or duplicating LLM calls.

## Verified end-to-end results (64 modules, 10,881 LOC)

Every gate below was verified live on **standard unpatched OpenClaw `2026.11.6`** (`:18789`) routed through `refit`'s `:18790` HTTP/2 proxy to Cerebras WSE-3T (`qwen-3.8-27b` after the `moonshotai-kimi-k3.1-pro-trial` hackathon endpoint retired with HTTP 404 at `02:00 CET`; live `moonshotai-kimi-k3.1-pro-trial` probes at `23:22 CET` and `01:58 CET` verified `50,000` uncached TPM, `1,000` RPM, `891` to `938 tok/s` decode, and `32.44 s` peak queue).

| Verification gate | Baseline (Phase 0) | Refactored (Phase 2) | Evidence |
| :--- | :---: | :---: | :--- |
| Modules and SCC cycles committed | `0 / 62` units (`64` files) | **`62 / 62` units (`100%`)** | All `64` files including `2` Tarjan SCC pairs committed atomically |
| Gate 1 (`pytest`, 64 untouched test files) | `15` failed / `128` total | **`0` failed (`128 passed`)** | `0` of `64` test files modified (`pytest -q`: `0.55 s`) |
| Gate 3 (`ruff` + `ty --error-on-warning`) | `589` ruff / `320` ty | **`0` ruff / `0` ty** | Independent `ruff check pkg` and `ty check --error-on-warning pkg` exit `0` |
| Gate 2 (`codehygiene v0.7.0` layout score) | `0.7512` med / `0.7211` p10 | **`0.8838` med / `0.8772` p10** | Warm worker (`weights.npz`, 8,400 B) + `DashMap` BLAKE3 cache |
| 5-run alternating OS CPU / RSS benchmark | `282.0 ms` / `15.26 MB` | **`278.0 ms` (`0.986x`) / `15.29 MB` (`1.002x`)** | Deterministic SHA-256 checksum `3db944c2245e37fa`, `<= 1.05x` pass |
| Live WAL crash-resume (`refit run --resume`) | `52` / `59` / `61` committed | **`0` torn files, `0` dup calls** | BLAKE3 disk verification skipped all `52`, `59`, and `61` committed units |
| Cerebras proxy telemetry (`:18790`) | `50K` uncached / `132K` synced | **`2,651.4 tok/s` median decode** | `0` HTTP 429s, `0` prefix drift, `219,136` cached tokens (`27.93%` hit) |
| Rust engine footprint & OS portability | Windows + Linux WSL2 | **`7.56 MB` scan / `21.85 MB` run** | `cargo build --release` + `cargo test` + WSL2 `Ubuntu-24.04` `cargo check` |

Note on run composition: the complete 62/62-unit refactoring completed across one initial parallel run (`52/62` committed) followed by three live `refit run --resume` passes (`59/62`, `61/62`, `62/62`, proving `0` duplicate LLM calls on all committed units). Raw per-pass logs (`run-pass1.jsonl`, `resume-proof.jsonl`, `resume-final.jsonl`, `resume-final2.jsonl`) and the merged interactive replay (`telemetry/run.jsonl`) are preserved in `telemetry/`.

## Architecture

1. **Single cross-platform Rust binary (`7.6 MB` to `21.9 MB` RSS):** OS-native process sandboxing (Windows Job Objects, Linux `setpgid`/`killpg` and `/proc/self/statm`, macOS `rusage`) and atomic file replacement (`MoveFileExW` / `rename`).
2. **Airtight `proxysoul/stem v0.4.0` discovery:** Runs `stem mcp` SQL inside `env_clear()` and an isolated `0700` `.refit/stem-home/` directory (zero access to host `~/.stem/.env` or API keys) feeding `petgraph 0.8.3` Tarjan SCC cycle condensation and Kahn disjoint wave scheduling.
3. **Zero-collision MVCC CAS + ACID WAL:** Every edit stages in an isolated copy, verifies BLAKE3 expected hashes before commit, writes `PREPARED` -> atomic rename -> `COMMITTED` in `.refit/wal.jsonl`, and deterministically recovers incomplete transactions on `refit resume` with **0 duplicate LLM calls** on committed modules.
4. **Four local verification gates (Gates 0 to 3):**
   - **Gate 0 (`< 1 ms`):** Python AST syntax check and canonical exported `abi_hash` pin.
   - **Gate 1:** Sandboxed impact-cone `pytest` unit tests.
   - **Gate 2 (`< 5 ms` warm):** Embedded `brianbell-x/codehygiene v0.7.0` geometric layout scorer (`weights.npz`, 8,400 B) over a persistent warm worker with `DashMap` BLAKE3 memoization.
   - **Gate 3:** `ruff check --fix` + `ruff check` (`0` diagnostics) + Astral `ty check --error-on-warning` (`0` errors, `0` warnings).
5. **Cerebras HTTP/2 proxy (`127.0.0.1:18790`):** Enforces exact Kimi K3.1 Pro wire parameters (`reasoning_effort: "low"`, `reasoning_format: "parsed"`, `temperature: 1.0`) and Qwen 3.8 wire parameters (`disable_reasoning: true`), compacts OpenClaw's 21K-token default tool boilerplate into the byte-identical static `SKILL.md` prefix (`0` prefix drift), runs a live header-calibrated 3-bucket pacer with an automatic 5-minute Outage Circuit Breaker, uses raw unescaped `<<<REFIT_PATCH>>>` blocks for high-speed decoding, and serves a 30 Hz SSE HUD plus static `?replay=` view.

## Quick start

```bash
# Build cross-platform Rust binary
cargo build --release

# Generate 10,437-LOC, 64-module benchmark repo
python fixture/generate_fixture.py fixture/target_repo

# Scan DAG + Gates 1-3 baseline
./target/release/refit scan fixture/target_repo

# Run live through standard unpatched OpenClaw 2026.11.6
export CEREBRAS_API_KEY="your-key"
OPENCLAW_CONFIG_PATH=./openclaw/openclaw.refit.json openclaw gateway run --port 18789 --bind loopback --auth token --token refit-local-gw --raw-stream
./target/release/refit run fixture/target_repo --openclaw-url http://127.0.0.1:18789/v1 --openclaw-token refit-local-gw
```

## Benchmark protocol (`benchmark-checklist` Q1 to Q7)

- **Q1 (Workload):** 64 modular Python files (`10,437` LOC baseline, `10,881` LOC refactored) across 4 dependency layers with 2 cyclic Tarjan SCC pairs, 128 unit tests (`15` seeded failures), `589` `ruff` diagnostics, and `320` `ty` diagnostics.
- **Q2 (Environment):** Windows 11 x64 (`rustc 1.100.0`, Python `3.15`, `pytest 9.2.0`, `ruff 0.17.6`, `ty 0.0.108`, OpenClaw `2026.11.6`); Linux compatibility verified via `Ubuntu-24.04` WSL2 `cargo check`.
- **Q3 (Model & Endpoint Disclosure):** Live `moonshotai-kimi-k3.1-pro-trial` probes at `23:22 CET` and `01:58 CET` confirmed `50,000` uncached TPM, `1,000` RPM, `891` to `938 tok/s` decode, and `32.44 s` peak queue before the trial model retired at `02:00 CET` (`HTTP 404`). The full 62/62-unit run and 3 live WAL resumes ran on Cerebras WSE-3T `qwen-3.8-27b` via standard unpatched OpenClaw `2026.11.6` (`2,651.4 tok/s` median decode, `0.37 s` avg queue, `0` HTTP 429s).
- **Q4 (Local Gates vs Correctness):** Gate 1 (`pytest`, `0/64` test files altered) and Gate 3 (`ruff` + `ty --error-on-warning`) verify functional and static correctness. Gate 2 (`codehygiene v0.7.0`) scores geometric layout maintainability only.
- **Q5 & Q7 (Anti-Regression Benchmark):** `refit bench` executed 5 alternating before/after OS-sandboxed pairs (`61,440` ledger rows per run), verifying exact deterministic SHA-256 output checksum equality (`3db944c2245e37fa`), `0.986x` median CPU time (`282.0 ms -> 278.0 ms`), and `1.002x` median peak RSS (`15.26 MB -> 15.29 MB`).
- **Q6 (Caching Disclosure):** Cerebras prompt caching (`cached_tokens`) excludes cached prefix tokens from the 50K uncached TPM cap and speeds up prefill (`219,136` cached tokens across the run). AST micro-slicing and BLAKE3 resume skip reduce billable input tokens.

## Attribution

- Code graph discovery integrates [`proxysoul/stem` `v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key sandbox.
- Gate 2 geometric layout scoring uses [`brianbell-x/codehygiene` `v0.7.0`](https://github.com/brianbell-x/codehygiene) (`engine.js` + 8,400 B `weights.npz`).
- All public X and `r/cerebras` demos are recorded on standard unpatched OpenClaw `2026.11.6`.
