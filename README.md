# `refit`

Cross-platform Rust and OpenClaw refactoring engine built for **Kimi K3.1 Pro on Cerebras** (`moonshotai-kimi-k3.1-pro-trial`).

Built to survive the live **50K uncached TPM / 500K total TPM** hackathon rate limit and **32.4s live server queues** without stalling, corrupting files, or duplicating LLM calls.

## Measured Kimi K3.1 Pro and local engine benchmarks

| Component | Measured evidence | Details |
| :--- | :--- | :--- |
| Live Cerebras Kimi K3.1 Pro probes (`moonshotai-kimi-k3.1-pro-trial`) | **`891` to `938 tok/s`** wafer decode (`18.8x` vs ~50 tok/s GPU) | Probe 1 (`23:22 CET`): `HTTP 200`, `50,000` uncached TPM, `1,000` RPM, `32.44 s` queue, `41 ms` prefill, `27 ms` decode (`891 tok/s`). Probe 2 (`01:58 CET`): `HTTP 200`, `6.8 ms` queue, `11.6 ms` prefill, `17.1 ms` decode (`938 tok/s`). |
| Standard unpatched OpenClaw harness | **OpenClaw `2026.11.6`** (`openclaw config validate` passed) | Routes `cerebras/moonshotai-kimi-k3.1-pro-trial` (`thinkingDefault: "low"`) through `:18790` HTTP/2 proxy. |
| Zero-key `proxysoul/stem v0.4.0` discovery | **`stem_used: true`** in **`245 ms`** | Runs `stem mcp` SQL inside `env_clear()` + isolated `0700` `.refit/stem-home/` (`0` host API keys exposed). |
| Modular Python target codebase | **`64` files, `10,437` LOC, `62` units (`2` Tarjan SCC cycles)** | `4` dependency layers, `128` unit tests (`15` seeded failures), `589` `ruff` + `320` `ty` baseline diagnostics. |
| Local verification gates (Gates 0 to 3) | **Gate 0 `< 1 ms`, Gate 1 `pytest`, Gate 2 `codehygiene`, Gate 3 `ruff` + `ty`** | Isolated per-transaction staging verifies syntax/ABI pin, unit tests, warm `codehygiene v0.7.0` (`8,400 B` weights), and `ruff` + `ty --error-on-warning`. |
| MVCC CAS + ACID WAL crash recovery | **`0` torn files, `0` duplicate LLM calls** | Verified by `cargo test` (`mvcc_cas_and_crash_recovery_roundtrip`) and BLAKE3 disk hash validation on `refit resume`. |
| 5-run alternating OS CPU / RSS bench | **`0.986x` CPU (`282 ms -> 278 ms`), `1.002x` RSS** | `61,440` ledger rows per run, exact deterministic SHA-256 checksum `3db944c2245e37fa`, `<= 1.05x` anti-regression guard. |
| Cross-platform Rust binary footprint | **`7.56 MB` scan / `21.85 MB` run RSS** | Verified with `cargo build --release`, `cargo test`, and WSL2 `Ubuntu-24.04` Linux `cargo check` (`0` errors). |

Note on interactive `?replay=` data (per `HACKATHON.md`): `telemetry/kimi-probes.json` stores the live measured `moonshotai-kimi-k3.1-pro-trial` probes (`50,000` uncached TPM, `1,000` RPM, `891` to `938 tok/s`). Because `moonshotai-kimi-k3.1-pro-trial` returned `HTTP 404 model_not_found` on this preview key after `02:00 CET`, the 64-module wave timeline in `telemetry/run.jsonl` and the demo video uses probe-calibrated mock wave data and states so explicitly in its footnote. Re-running `refit run` with any active Kimi K3.1 Pro key replaces `telemetry/run.jsonl` with a live multi-wave trace in one command.

## Architecture

1. **Single cross-platform Rust binary (`7.6 MB` to `21.9 MB` RSS):** OS-native process sandboxing (Windows Job Objects, Linux `setpgid`/`killpg` and `/proc/self/statm`, macOS `rusage`) and atomic file replacement (`MoveFileExW` / `rename`).
2. **Airtight `proxysoul/stem v0.4.0` discovery:** Runs `stem mcp` SQL inside `env_clear()` and isolated `0700` `.refit/stem-home/` feeding `petgraph 0.8.3` Tarjan SCC cycle condensation and Kahn disjoint wave scheduling.
3. **Zero-collision MVCC CAS + ACID WAL:** Every edit stages in an isolated copy, verifies BLAKE3 expected hashes before commit, writes `PREPARED` -> atomic rename -> `COMMITTED` in `.refit/wal.jsonl`, and deterministically recovers incomplete transactions on `refit resume`.
4. **Four local verification gates (Gates 0 to 3):**
   - **Gate 0 (`< 1 ms`):** Python AST syntax check and canonical exported `abi_hash` pin.
   - **Gate 1:** Sandboxed impact-cone `pytest` unit tests.
   - **Gate 2 (`< 5 ms` warm):** Embedded `brianbell-x/codehygiene v0.7.0` geometric layout scorer (`weights.npz`, 8,400 B) over a persistent warm worker with `DashMap` BLAKE3 memoization.
   - **Gate 3:** `ruff check --fix` + `ruff check` (`0` diagnostics) + Astral `ty check --error-on-warning` (`0` errors, `0` warnings).
5. **50K uncached TPM Kimi K3.1 Pro HTTP/2 proxy (`127.0.0.1:18790`):** Enforces exact Kimi K3.1 Pro wire parameters (`model: "moonshotai-kimi-k3.1-pro-trial"`, `reasoning_effort: "low"`, `reasoning_format: "parsed"`, `temperature: 1.0`), compacts OpenClaw's 21K-token default workspace boilerplate into the byte-identical static `SKILL.md` prefix (`0` prefix drift), paces under `44,000 / 50,000` uncached TPM with an automatic 5-minute Outage Circuit Breaker, uses raw unescaped `<<<REFIT_PATCH>>>` blocks, and serves the light-editorial 30 Hz SSE HUD + static `?replay=`.

## Quick start

```bash
# Build cross-platform Rust binary
cargo build --release

# Generate 10,437-LOC, 64-module benchmark repo
python fixture/generate_fixture.py fixture/target_repo

# Scan DAG + Gates 1-3 baseline
./target/release/refit scan fixture/target_repo

# Run live on Kimi K3.1 Pro through standard unpatched OpenClaw 2026.11.6
export CEREBRAS_API_KEY="your-key"
OPENCLAW_CONFIG_PATH=./openclaw/openclaw.refit.json openclaw gateway run --port 18789 --bind loopback --auth token --token refit-local-gw --raw-stream
./target/release/refit run fixture/target_repo --model moonshotai-kimi-k3.1-pro-trial --openclaw-url http://127.0.0.1:18789/v1 --openclaw-token refit-local-gw
```

## Attribution and honest disclosure

- Code graph discovery integrates [`proxysoul/stem` `v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key sandbox.
- Gate 2 geometric layout scoring uses [`brianbell-x/codehygiene` `v0.7.0`](https://github.com/brianbell-x/codehygiene) (`engine.js` + 8,400 B `weights.npz`); functional and type correctness are verified separately by Gate 1 (`pytest`) and Gate 3 (`ruff` + `ty`).
- Cerebras prompt caching (`cached_tokens`) excludes cached prefix tokens from the 50K uncached TPM cap and speeds up prefill, while AST micro-slicing and BLAKE3 skip reduce billable input tokens.
- All public X and `r/cerebras` demos are built for standard unpatched OpenClaw `2026.11.6` (no speed patch used).
