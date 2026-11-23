# `refit`

Cross-platform Rust + OpenClaw refactoring engine for **Kimi K3.1 Pro on Cerebras**.

Built to survive the live **50K uncached TPM / 500K total TPM** hackathon rate limit and **30s+ server queues** without stalling, corrupting files, or wasting tokens.

## Architecture

1. **Single cross-platform Rust binary (`17.6 MB` RSS):** OS-native process sandboxing (`Windows Job Objects`, Linux `setpgid`/`killpg` + `/proc/self/statm`, macOS `rusage`) and atomic file replacement (`MoveFileExW` / `rename`).
2. **Airtight `proxysoul/stem v0.4.0` discovery:** Runs inside `env_clear()` + isolated `0700` `.refit/stem-home/` (zero access to host `~/.stem/.env` or API keys) feeding `petgraph 0.8.3` Tarjan SCC cycle condensation and Kahn disjoint wave scheduling.
3. **Zero-collision MVCC CAS + ACID WAL:** Every edit stages in an isolated copy, verifies BLAKE3 expected hashes before commit, writes `PREPARED` -> atomic rename -> `COMMITTED` in `.refit/wal.jsonl`, and deterministically recovers torn transactions on `refit resume` with **0 duplicate LLM calls** on committed modules.
4. **Four local verification gates (Gates 0 to 3):**
   - **Gate 0 (`< 1 ms`):** Python AST syntax check + canonical exported `abi_hash` pin.
   - **Gate 1:** Sandboxed impact-cone `pytest` unit tests.
   - **Gate 2 (`< 5 ms` warm):** Embedded `brianbell-x/codehygiene v0.7.0` geometric layout scorer (`weights.npz`, 8,400 B) over persistent warm worker with `DashMap` BLAKE3 memoization.
   - **Gate 3 (`15 to 90 ms`):** `ruff check --fix` + `ruff check` (`0` diagnostics) + Astral `ty check --error-on-warning` (`0` errors, `0` warnings).
5. **50K uncached TPM Cerebras HTTP/2 proxy (`127.0.0.1:18790`):** Enforces exact Kimi K3.1 Pro wire parameters (`reasoning_effort: "low"`, `reasoning_format: "parsed"`, `temperature: 1.0`, `gzip`), BLAKE3 static prefix-drift guard, live header-calibrated 3-bucket pacer (`44K` uncached / `440K` total / `500` RPM), 5-minute Outage Circuit Breaker, raw unescaped `<<<REFIT_PATCH>>>` blocks to preserve speculative decoding, and 30 Hz SSE HUD + lossless `telemetry/run.jsonl` replay.

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
OPENCLAW_CONFIG_PATH=./openclaw/openclaw.refit.json openclaw gateway --port 18789 --bind loopback --auth token --token refit-local-gw --raw-stream
./target/release/refit run fixture/target_repo --openclaw-url http://127.0.0.1:18789/v1 --openclaw-token refit-local-gw
```

## Attribution and honest disclosure

- Code graph discovery integrates [`proxysoul/stem` `v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key sandbox.
- Gate 2 geometric layout scoring uses [`brianbell-x/codehygiene` `v0.7.0`](https://github.com/brianbell-x/codehygiene) (`engine.js` + 8,400 B `weights.npz`); functional and type correctness are verified separately by Gate 1 (`pytest`) and Gate 3 (`ruff` + `ty`).
- Cerebras prompt caching (`cached_tokens`) excludes cached prefix blocks from the 50K uncached TPM cap and speeds up prefill, while AST micro-slicing and BLAKE3 skip reduce billable input tokens.
- All public X and `r/cerebras` demos are recorded on standard unpatched OpenClaw `2026.11.6`.
