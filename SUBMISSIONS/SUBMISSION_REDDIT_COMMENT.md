**Project:** `refit` ([GitHub](https://github.com/patrickfield152-del/refit) | [Interactive 30 Hz Replay](https://patrickfield152-del.github.io/refit/?replay=run.jsonl))
**Stack:** Single cross-platform Rust binary (`17.6 MB` RSS, Windows Job Objects + Unix `setpgid`/`/proc/self/statm`), standard unpatched **OpenClaw `2026.11.6`**, **`moonshotai-kimi-k3.1-pro-trial`** on Cerebras, sandboxed `proxysoul/stem v0.4.0`, `brianbell-x/codehygiene v0.7.0`, `ruff 0.17.6`, `ty 0.0.108`, `pytest 9.2.0`.

### Why we built `refit` for tonight's `50K` uncached TPM cap and `32s+` queues

When hackathon rate limits dropped from `100K` to **`50,000` uncached TPM (`500K` total TPM, `1,000` RPM)** and live `queue_time` hit **`32.4 s`** (with **`891 tok/s`** wafer decode on `moonshotai-kimi-k3.1-pro-trial`), naive full-file sequential agents stalled hard:
- `6,200` uncached tokens/call = only `8` calls/min before 429s, or `38.5 min` sequential across 70 calls.
- `refit` uses **90% AST micro-slicing** (`~900` uncached tokens/call) + a byte-identical static prefix (`cached_tokens` excluded from the 50K uncached cap) + **Kahn sliding-window parallel waves** (`petgraph 0.8.3` Tarjan SCC + disjoint 1-hop cohorts) + a **live header-calibrated 3-bucket HTTP/2 proxy (`:18790`)** with an automatic **5-minute Outage Circuit Breaker** and **ACID `wal.jsonl` + BLAKE3 CAS crash-resume** (`0` torn files, `0` duplicate LLM calls).

### Verified Architecture and Benchmark Protocol (`benchmark-checklist` Q1 to Q7)

| Check | Measured evidence |
| :--- | :--- |
| Target codebase | `10,437` LOC across `64` modular Python files (`4` layers, `2` Tarjan SCC cycles, `128` tests with `15` seeded failures, `589` `ruff` + `320` `ty` baseline diagnostics) |
| Live Cerebras probe | `HTTP 200` on `moonshotai-kimi-k3.1-pro-trial`: `x-ratelimit-limit-tokens-minute: 50000`, `x-ratelimit-limit-requests-minute: 1000`, `queue_time: 32.44 s`, `prompt_time: 41 ms`, `completion_time: 27 ms` (`891 tok/s` decode) |
| 4 local verification gates | Gate 0 (`ast.parse` + exported `abi_hash`), Gate 1 (sandboxed `pytest`), Gate 2 (warm `codehygiene v0.7.0` `8,400 B` weights with `DashMap` BLAKE3 cache), Gate 3 (`ruff --fix` + `ty --error-on-warning` to `0/0`) |
| Concurrency & crash safety | Isolated per-tx staging + single-writer BLAKE3 CAS + `PREPARED` -> atomic rename -> `COMMITTED` WAL (`cargo test` unit test proves deterministic restore after simulated `SIGKILL`) |
| Rust engine footprint | **`17.58 MB` RSS** (`cargo build --release`), zero API keys exposed to `stem v0.4.0` (`env_clear()` + `0700` `.refit/stem-home/`) |
| 5-run alternating OS bench | `refit bench` runs 5 alternating before/after pairs (`61,440` ledger rows, verified deterministic SHA-256 checksum `3db944c2245e37fa`, `<= 1.05x` anti-regression guard) |

### Attribution and honest disclosures

- Integrates [`proxysoul/stem v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key sandbox and [`brianbell-x/codehygiene v0.7.0`](https://github.com/brianbell-x/codehygiene) strictly for Gate 2 geometric layout scoring (functional & type correctness are verified separately by `pytest`, `ruff`, and `ty`).
- Recorded on **standard unpatched OpenClaw `2026.11.6`** (no speed patch used in public X or Reddit footage).
- Cerebras prompt caching removes `cached_tokens` from the `50K` uncached TPM cap and speeds up prefill; AST slicing + BLAKE3 skip generate billable input token savings.
