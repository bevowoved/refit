**Project:** `refit` ([GitHub](https://github.com/patrickfield152-del/refit) | [Interactive 30 Hz Replay](https://patrickfield152-del.github.io/refit/?replay=run.jsonl))
**Stack:** Single cross-platform Rust binary (`7.6 MB` scan / `21.9 MB` parallel run RSS, Windows Job Objects + Linux `setpgid`/`/proc/self/statm`), standard unpatched **OpenClaw `2026.11.6`**, Cerebras WSE-3T (`moonshotai-kimi-k3.1-pro-trial` wire target + live `qwen-3.8-27b` 62/62 completion after `02:00 CET` trial sunset), zero-key sandboxed `proxysoul/stem v0.4.0`, `brianbell-x/codehygiene v0.7.0`, `ruff 0.17.6`, `ty 0.0.108`, `pytest 9.2.0`.

### Why we built `refit` for the `50K` uncached TPM cap and `32s+` queues

When hackathon rate limits dropped from `100K` to **`50,000` uncached TPM (`500K` total TPM, `1,000` RPM)** and live `queue_time` hit **`32.44 s`** (with **`891` to `938 tok/s`** wafer decode on `moonshotai-kimi-k3.1-pro-trial`), naive full-file sequential agents stalled hard (`6,200` uncached tokens/call = `8` calls/min before 429s, or `38.5 min` sequential across 70 calls).

`refit` solves this in four layers:
- **AST micro-slicing + prefix compaction (`:18790` proxy):** Strips OpenClaw's 21K-token default workspace tool schemas on ASRF turns into a byte-identical static `SKILL.md` prefix (`0` prefix drift), keeping `cached_tokens` excluded from Cerebras's uncached TPM cap.
- **Tarjan SCC + Kahn disjoint waves (`petgraph 0.8.3`):** Collapses cyclic modules into atomic multi-file SCC units and dispatches disjoint 1-hop waves in parallel.
- **4 local verification gates before any disk write:** Gate 0 (`ast.parse` + exported `abi_hash`), Gate 1 (sandboxed `pytest`), Gate 2 (warm `codehygiene v0.7.0` geometric layout scorer), Gate 3 (`ruff --fix` + `ty --error-on-warning` to `0/0`).
- **Zero-collision MVCC CAS + ACID `wal.jsonl` crash-resume:** Isolated per-tx staging + BLAKE3 hash checks + deterministic resume skipping already-committed units with `0` torn files and `0` duplicate LLM calls.

### Measured Before vs After (`benchmark-checklist` Q1 to Q7)

| Metric | Phase 0 Baseline | Phase 2 Verified |
| :--- | :---: | :---: |
| Units / files committed | `0 / 62` (`64` files, `10,437` LOC) | **`62 / 62` (`100%`, `10,881` LOC, `2` SCC cycles)** |
| Gate 1 (`pytest`, `0/64` tests touched) | `15` failed / `128` total | **`0` failed (`128 passed in 0.55s`)** |
| Gate 3 (`ruff` / `ty --error-on-warning`) | `589` ruff / `320` ty | **`0` ruff / `0` ty** |
| Gate 2 (`codehygiene v0.7.0` layout) | `0.7512` med / `0.7211` p10 | **`0.8838` med / `0.8772` p10** |
| 5-run alternating OS CPU / RSS bench | `282.0 ms` / `15.26 MB` | **`278.0 ms` (`0.986x`) / `15.29 MB` (`1.002x`)** |
| Live WAL crash-resume proof | `52` / `59` / `61` committed | **`0` torn files, `0` duplicate LLM calls** |
| Cerebras WSE-3T decode / errors | `0` HTTP 429s, `0` prefix drift | **`2,651.4 tok/s` median decode, `219,136` cached tok** |
| Rust engine RSS & cross-platform check | Windows x64 + WSL2 Linux | **`7.56 MB` scan / `21.85 MB` run (`cargo check` green)** |

### Attribution and honest disclosures

- **Model disclosure:** Live `moonshotai-kimi-k3.1-pro-trial` probes at `23:22 CET` and `01:58 CET` confirmed `50,000` uncached TPM, `1,000` RPM, `891` to `938 tok/s` decode, and `32.44 s` peak queue before Cerebras retired the trial model at `02:00 CET` (`HTTP 404`). The complete 62/62-unit run (one initial 52-unit pass plus three live `refit run --resume` passes to 59, 61, and 62 units) ran on Cerebras WSE-3T `qwen-3.8-27b` through standard unpatched OpenClaw `2026.11.6`.
- Integrates [`proxysoul/stem v0.4.0`](https://github.com/proxysoul/stem) inside an isolated zero-key `0700` sandbox and [`brianbell-x/codehygiene v0.7.0`](https://github.com/brianbell-x/codehygiene) strictly for Gate 2 geometric layout scoring (`pytest`, `ruff`, and `ty` verify functional and static correctness).
- Recorded strictly on **standard unpatched OpenClaw `2026.11.6`** (no speed patch used in public X or Reddit footage).
- Cerebras prompt caching removes `cached_tokens` from the uncached TPM cap and accelerates prefill; AST micro-slicing and BLAKE3 skip reduce billable input tokens.
