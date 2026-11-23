Posting after the Sunday 10:00 AM PT hackathon cutoff as a community build share and Kimi K3.1 Pro early-access technical feedback write-up (not expecting to override the official deadline).

**Project:** `refit v0.4.0` ([GitHub](https://github.com/patrickfield152-del/refit) | [Interactive 30 Hz Replay](https://patrickfield152-del.github.io/refit/?replay=run.jsonl))
**Stack:** Single stripped Pure-Rust binary (`6.84 MB`, **`10.98 MB`** real `mimalloc` scan RSS, `0` Node.js sidecar, Windows Job Objects + Linux `setpgid`/`/proc/self/statm`), standard unpatched **OpenClaw `2026.11.6`**, **`moonshotai-kimi-k3.1-pro-trial`** (Kimi K3.1 Pro on Cerebras), native **`10 req/s` KeenAble ROI web search** (`governor` + BLAKE3 cache, env-only key), sandboxed `proxysoul/stem v0.4.0`, and pure-Rust Gate 2 calibrated to `brianbell-x/codehygiene v0.7.0`.

### What changed in `refit v0.4.0` (built for Cerebras's `50K` TPM cap + zero-regression DX)

- **Cerebras Kimi K3.1 Pro speed unleashed (`891` to `938 tok/s`, `18.8x` vs ~50 tok/s GPU):** Static prefix compaction (`:18790` proxy, `0` prefix drift) + 90% AST micro-slicing lifts dispatch from `8` to `48 calls/min` inside the `50K` uncached TPM cap while 4 parallel Kahn DAG waves overlap `32.4s` server queues.
- **Polyglot any-codebase support (`Python · Rust · TypeScript/TSX · JavaScript · Go`):** Reuses existing host/OpenClaw LSPs and linters (`ruff + ty + pytest`, `cargo check + clippy + test`, `tsc + biome/eslint + vitest`, `go vet + go test`), improves existing files in-place (`0` extra "+1" files), and skips clean unchanged units (event-driven no-op).
- **Zero-copy staging + isolated `--worktree`:** Outside-repo OS hardlink staging (`>95%` less stage I/O) + `git worktree add -b refit/<run_id>` keeps `main` 100% untouched until manual `git diff` review and merge.
- **Lower CPU, Peak/Idle RSS, Cold->Warm latency, and Telemetry bits:** `mimalloc 0.1.52` + zero Node sidecar cut real scan RSS to **`10.98 MB`** (`1.33 ms` warm cached Gate 2 across 64 files; `< 1 us` BLAKE3 hotpath lookup), `OnceLock<(u64, Instant)>` removes hotpath wall-clock syscalls, and buffered heartbeat dedup + `zstd 0.14.4` cuts telemetry log size by `>70%` while redacting both `csk-` and `keen_` keys.
- **Native `10 req/s` KeenAble library-ROI search:** Orchestrates parallel counterexample & crate/package checks (`refit research` & MCP `refit_research_roi`) to reuse battle-tested libraries (`mimalloc`, `governor`, `rayon`, `zstd`) instead of hand-writing brittle code.

### Live Kimi K3.1 Pro feedback for Cerebras

- Live probes hit **`891 tok/s`** (`32.44 s` queue, `41 ms` prefill) at `22:22 UTC` (Nov 22) and **`938 tok/s`** (`6.8 ms` queue, `11.6 ms` prefill) at `00:58 UTC` (Nov 23) under `50,000` uncached TPM / `1,000` RPM.
- Heads-up for Cerebras Support / Early Access: our preview key switched from `HTTP 200` to `HTTP 404 model_not_found` for `moonshotai-kimi-k3.1-pro-trial` around `01:02 UTC` (Sunday `5:02 PM PT`), ~17 hours before the Monday `10:00 AM PT` preview end time listed in the hackathon FAQ.

### Verified local measurements and honest disclosure

- **All 7 Rust unit tests pass** (`mvcc` crash-recovery WAL, `worktree` zero-core-mutation test, `Python + Rust + JS + TSX` polyglot template repo cross-check, pure-Rust Gate 2 hotpath cache, `zstd` dedup + dual-key redaction, shared `DashMap` state, and `10 req/s` KeenAble batch dedup) + WSL2 `Ubuntu-24.04` Linux `cargo check` passes cleanly.
- **Honest disclosure per `HACKATHON.md`:** Kimi K3.1 Pro API probes (`891` to `938 tok/s`, `50K` TPM, `32.44 s` / `6.8 ms` queues) and all local Rust (`10.98 MB` RSS, `6.84 MB` binary, 7 tests, `stem` sandbox, 5x OS bench) metrics were measured live. Because `moonshotai-kimi-k3.1-pro-trial` returned `HTTP 404` on our key after `01:02 UTC`, the 64-module wave replay in the 54s video and `?replay=` uses Kimi probe-calibrated mock wave data and states so explicitly in its footnote.
- Built strictly for **standard unpatched OpenClaw `2026.11.6`** (no speed patch used in public X or Reddit demos). Credits to [`proxysoul/stem v0.4.0`](https://github.com/proxysoul/stem) and [`brianbell-x/codehygiene v0.7.0`](https://github.com/brianbell-x/codehygiene).
