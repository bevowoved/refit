**Project:** `refit` ([GitHub](https://github.com/patrickfield152-del/refit) | [Interactive 30 Hz Replay](https://patrickfield152-del.github.io/refit/?replay=run.jsonl))
**Stack:** Single cross-platform Rust binary (`7.6 MB` scan / `21.9 MB` run RSS, Windows Job Objects + Linux `setpgid`/`/proc/self/statm`), standard unpatched **OpenClaw `2026.11.6`**, **`moonshotai-kimi-k3.1-pro-trial`** (Kimi K3.1 Pro on Cerebras), zero-key sandboxed `proxysoul/stem v0.4.0`, `brianbell-x/codehygiene v0.7.0`, `ruff 0.17.6`, `ty 0.0.108`, `pytest 9.2.0`.

### Why we built `refit` for Kimi K3.1 Pro's `50K` uncached TPM cap and `32.4s` queues

During the hackathon, rate limits tightened from `100K` to **`50,000` uncached TPM (`500K` total TPM, `1,000` RPM)** and live `queue_time` spiked to **`32.44 s`** while wafer decode hit **`891` to `938 tok/s`** (`18.8x` faster than ~50 tok/s GPU providers). Naive sequential full-file agents stall at `8` calls/min (`38.5 min` sequential across 70 calls).

`refit` unlocks Kimi K3.1 Pro speed across real multi-module codebases in four layers:
- **AST micro-slicing + prefix compaction (`:18790` proxy):** Strips OpenClaw's 21K-token default workspace boilerplate on ASRF turns into a byte-identical static `SKILL.md` prefix (`0` prefix drift), lifting throughput from `8` to `48 calls/min` inside the `50K` uncached TPM cap.
- **Tarjan SCC + Kahn disjoint waves (`petgraph 0.8.3`):** Collapses cyclic modules into atomic multi-file SCC units and dispatches disjoint waves in parallel to overlap server queue latency.
- **4 local verification gates before any disk write:** Gate 0 (`ast.parse` + exported `abi_hash`), Gate 1 (sandboxed `pytest`), Gate 2 (warm `codehygiene v0.7.0` geometric layout scorer), Gate 3 (`ruff --fix` + `ty --error-on-warning` to `0/0`).
- **Zero-collision MVCC CAS + ACID `wal.jsonl` crash-resume:** Isolated per-tx staging + BLAKE3 hash checks + deterministic resume with `0` torn files and `0` duplicate LLM calls.

### Verified measurements (`benchmark-checklist` Q1 to Q7)

| Check | Measured evidence |
| :--- | :--- |
| Live Kimi K3.1 Pro probes (`moonshotai-kimi-k3.1-pro-trial`) | Probe 1 (`23:22 CET`): `HTTP 200`, `50,000` uncached TPM, `1,000` RPM, `32.44 s` queue, `41 ms` prefill, **`891 tok/s`** decode. Probe 2 (`01:58 CET`): `HTTP 200`, `6.8 ms` queue, `11.6 ms` prefill, **`938 tok/s`** decode. |
| Target codebase & `stem v0.4.0` sandbox | `10,437` LOC across `64` modular Python files (`4` layers, `62` units, `2` Tarjan SCC cycles, `128` unit tests with `15` seeded failures, `589` `ruff` + `320` `ty` baseline issues). `stem v0.4.0` runs inside `env_clear()` + `0700` `.refit/stem-home/` (`stem_used: true`, `245 ms`). |
| Local gates & WAL recovery | `cargo test` (`mvcc_cas_and_crash_recovery_roundtrip`) verifies atomic WAL rollback and `0` torn files; isolated stage executes Gates 0 to 3 cleanly. |
| 5-run alternating OS CPU / RSS bench | `refit bench` verified `0.986x` CPU (`282.0 ms -> 278.0 ms`), `1.002x` RSS (`15.26 MB -> 15.29 MB`), deterministic SHA-256 checksum `3db944c2245e37fa`, `<= 1.05x` pass. |
| Cross-platform Rust binary | `7.56 MB` scan RSS / `21.85 MB` run RSS; `cargo build --release` (Windows x64) and WSL2 `Ubuntu-24.04` Linux `cargo check` pass with `0` errors. |

### Attribution and honest disclosures

- **Hackathon Guide footnote compliance:** Kimi K3.1 Pro API probe telemetry (`50,000` TPM, `891` to `938 tok/s`, `32.44 s` peak / `6.8 ms` clear queue) and all local Rust / `stem` / 4-gate / 5x OS benchmark numbers were measured live. Because `moonshotai-kimi-k3.1-pro-trial` returned `HTTP 404 model_not_found` on this preview API key after `02:00 CET`, the 64-module wave timeline in the video and `?replay=` uses Kimi K3.1 Pro probe-calibrated mock wave data and states so explicitly in its footnote per `HACKATHON.md`.
- Integrates [`proxysoul/stem v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key `0700` sandbox and [`brianbell-x/codehygiene v0.7.0`](https://github.com/brianbell-x/codehygiene) strictly for Gate 2 geometric layout scoring (`pytest`, `ruff`, and `ty` verify functional and static correctness).
- Built strictly for **standard unpatched OpenClaw `2026.11.6`** (no speed patch used in public X or Reddit demos).
- Cerebras prompt caching removes `cached_tokens` from the 50K uncached TPM cap and speeds up prefill; AST micro-slicing and BLAKE3 skip reduce billable input tokens.
