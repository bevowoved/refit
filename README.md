# refit

[![CI](https://github.com/patrickfield152-del/refit/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickfield152-del/refit/actions/workflows/ci.yml)
[![Release](https://github.com/patrickfield152-del/refit/actions/workflows/release.yml/badge.svg)](https://github.com/patrickfield152-del/refit/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license--credits)
[![Rust 1.100+](https://img.shields.io/badge/rust-1.100%2B-orange.svg)](rust-toolchain.toml)

**The two-speed closed-loop verification spine for coding agents:** `<7 ms` Rust static pre-flight shield (AVX2 secret guard, invisible Unicode Tag/BiDi prompt-injection guard, static OSV.dev lockfile SCA, README doc-drift detector, embedded neural hygiene, atomic MVCC WAL) **+** **Gate 4 TestMu AI Kane CLI** live desktop/mobile browser proof, `.evidence` secret auditing, and `SymbolGraph` rewind in **one static Rust binary** (`0` active Cargo duplicates).

- **Live HUD & Gate 4 Proof:** [patrickfield152-del.github.io/refit/?demo=1&at=100&tab=kane](https://patrickfield152-del.github.io/refit/?demo=1&at=100&tab=kane) (also `?replay=run.jsonl`)
- **Agent Skill:** [skills/refit/SKILL.md](skills/refit/SKILL.md) (open `agentskills.io` / `skills.sh` standard + **SEP-2640 Skills over MCP** `skill://refit/SKILL.md`)

---

## Agent & Skill Setup (60 Seconds)

### 1. One-Command Onboarding (`refit init .`)

```bash
refit init .   # alias of `refit sustain . --init`
```

Configures worktree-safe chained Git hooks, Claude Code / Cursor / Gemini CLI hooks, `AGENTS.md`, `.refit/invariants.yml`, and the CI guard in one command. Or install the standalone Agent Skill:

```bash
npx skills add patrickfield152-del/refit --skill refit
```

### 2. Connect the Converged MCP Server (`stdio` or Stateless HTTP)

```json
{
  "mcpServers": {
    "refit": {
      "command": "refit",
      "args": ["mcp", "."]
    }
  }
}
```

Includes annotated tools (`readOnlyHint` / `destructiveHint`), polymorphic `refit_tx`, **SEP-2640 Skills over MCP** (`skill://refit/SKILL.md`), Unicode-sanitized `refit_read`, and keyless Keenable `refit_search`. Also runs stateless HTTP (`refit mcp . --http 127.0.0.1:18791`) or OpenClaw ([openclaw/openclaw.refit.json](openclaw/openclaw.refit.json)).

### 3. Unified Pre-Flight Gate, Gate 4 Kane CLI & `<5 ms` Agent Loop Hook

```bash
# Unified pre-flight: secrets + Unicode guard + OSV.dev SCA + README doc-drift + Gate 4 Kane CLI
refit check . [--staged | --diff origin/main] [--deps] [--verify-docs] \
  [--kane tests/kane/hud_desktop_replay_test.md] [--policy tests/kane/refit-policy.yaml] \
  [--format human|json|md|gh|sarif]

# Real-time <5 ms stdin-JSON hook for Claude Code (PreToolUse/PostToolUse), Cursor, Gemini CLI, Kiro
refit hook

# Direct secret scanning & hooks
refit secrets . --staged            # <10 ms staged blob scan via git cat-file --batch
refit secrets . --diff origin/main  # <10 ms unpushed commit scan
refit secrets .                     # parallel zero-recompute repo scan (BLAKE3 cache)
```

### 4. Gate 4: Native `refit × Kane CLI` (`@testmuai/kane-cli@0.10.9`) Integration

1. **Static-First Short-Circuit (`0` Credits Wasted):** Every `refit check --kane` and MCP `refit_check(kane=...)` runs `<10 ms` static secrets, invisible Unicode, optional OSV.dev SCA, `--verify-docs` static README drift, and async URL reachability pre-flight *before* launching a browser.
2. **Upgrade-Proof Wire & Capability Negotiation:** Probes `kane-cli --version` via `semver`, handles polymorphic `result_code` (`100` integer vs `"740"` confirmed product bug), `0.10.8` red `AGENT_INFO` steps, `0.10.0+` declarative `--policy` verdicts ([tests/kane/refit-policy.yaml](tests/kane/refit-policy.yaml)), `events.ndjson` disk-tee recovery, and `0`-credit cached `_test.md` replays.
3. **`.evidence` Secret Redaction Auditor (`kane-cli 0.10.2+`):** Because `kane-cli >= 0.10.2` seals `.evidence` packs without pre-seal secret redaction, `refit` inspects extracted `failure.yaml` / `result.yaml` / `*-console.ndjson` / `*.har` entries via its zip-bomb-capped `rawzip` reader, flags unredacted secrets (`kane/evidence_secret_leak` in SARIF/GH), and redacts excerpts before feeding back to coding agents.
4. **12 Converged MCP Tools (Antigravity Tool-Cap Ready):** Kane runs converge into existing `refit_check` (12 tools total vs 92 for Playwright Test MCP), automatically inheriting MCP `2026-07-28` durable `task: true` + `tasks/cancel` and warm `SymbolGraph` rewind.
5. **Opt-In `<2 ms` Push Gate & Dirty-UI Tracking:** Set `REFIT_KANE_GATE=1` (or create `.refit/kane.json`) so `refit hook` marks UI/doc edits dirty on `PostToolUse`/`AfterTool`/`afterFileEdit` and blocks `git push` until `.refit/kane_last.json` is green for the current `.git/HEAD` (zero `git` subprocesses).
6. **Honest Limits:** `refit` runs statically in `<7 ms` (`0` credits) but cannot see rendered DOM or CSS; Kane CLI authors once and replays cached `_test.md` steps at `~0` LLM credits. Because `kane-cli 0.10.2+` seals `.evidence` packs unredacted, `refit` scans them before commit, and `.testmuai/evidence/` stays gitignored. Local Android emulators / iOS simulators require macOS Apple Silicon, so Windows/Linux hosts route mobile specs ([tests/kane/hud_mobile_viewport_test.md](tests/kane/hud_mobile_viewport_test.md), [tests/kane/mobile_cloud_example.md](tests/kane/mobile_cloud_example.md)) to TestMu AI Cloud (`--target android --remote`).

### 5. Composing `refit` With the TestMu AI Ecosystem & Google Antigravity (`18 / 100` Tool Cap)

Google Antigravity caps active MCP tools globally at `~100`, and Playwright 1.64's Test MCP server burns `92` slots alone. Pairing `refit` (`12` local tools, keyless) + **TestMu AI Hosted MCP** (`6` cloud suites via `"serverUrl"`) + **`kane-cli 0.10.9`** (`0` MCP slots via CLI/skill) uses `<20%` of Antigravity's tool cap, and composes cleanly alongside `@e2e-dev/testmu` (code-first TS real-device runs: `refit check . --deps && npx e2e run --target testmu-web`) and `@testmuai/rook` (OWASP Agentic Top 10 red-teaming):

```json
{
  "mcpServers": {
    "refit": { "command": "refit", "args": ["mcp", "."] },
    "testmu-ai": { "serverUrl": "https://mcp.lambdatest.com/mcp" }
  }
}
```

### 6. `<20s` Judge Fast-Path (Zero Account / Zero API Key Required)

```bash
# 1) Static pre-flight + README doc-drift verification (<30 ms, 0 credits, REFIT_OFFLINE=1 safe)
refit check . --verify-docs --format human
refit audit . --markdown

# 2) Open the live interactive Gate 4 HUD (desktop or 390x844 mobile)
#    https://patrickfield152-del.github.io/refit/?demo=1&at=100&tab=kane

# 3) Live Kane CLI Gate 4 replay (with TestMu account, replays cached _test.md at ~0 LLM credits)
refit check . --verify-docs --kane tests/kane/hud_desktop_replay_test.md --policy tests/kane/refit-policy.yaml --format human
```

---

## Architecture & Pipeline

```text
  Repo / Git Index / Agent Hook
        │
        ├─► [Pre-Flight]    Tiered CRC32-verified secret guard (2-slab ≤64 AVX2 Fat-Teddy SIMD,
        │                   130+ rules + Tier-4 entropy, BLAKE3 cache) + invisible Unicode Tag/BiDi
        │                   prompt-injection guard + hygiene/ABI check + static OSV.dev lockfile SCA
        ├─► [Agent Loop]    <5 ms `refit hook` (Claude Code / Cursor / Gemini CLI / Kiro) + converged `refit mcp` (12 tools)
        ├─► [Gate 4 Live]   TestMu AI Kane CLI `0.10.9` (`--agent` NDJSON, `--policy`, `.evidence` secret audit)
        ├─► [Symbol Graph]  Polyglot lexer (Py/Rs/TS/JS/Go) + zstd SST + Pearce SCC + Kahn waves
        ├─► [Neural Gate]   Embedded ternary codehygiene (`weights.npz`, 0.000000 parity vs JS)
        └─► [Atomic MVCC]   BLAKE3 CAS + CRC32C WAL + ephemeral isolated stages (clear_env=true)
```

### Secret Engine Benchmark (Measured: 10K LOC & 10K Files / 400K LOC / 39.5 MiB)

| Variant | Startup | 10K LOC (Cold) | 10K Files / 400K LOC (Cold) | 10K Files (Warm Cache) |
| :--- | ---: | ---: | ---: | ---: |
| V1 — Flat `AhoCorasick` (>64 pats, scalar NFA) | `0.30 ms` | `1.22 ms` | `12.79 ms` | — |
| V3 — `RegexSet` (121 compiled regexes) | `2.51 ms` | `1.42 ms` | `11.03 ms` | — |
| **V2 (Shipped) — 2× ≤64 AVX2 Fat-Teddy + Zero-Regex Table + BLAKE3 Cache** | **`0.51 ms`** | **`0.54 ms`** | **`8.85 ms`** (~4.5 GB/s) | **`0.24 ms`** |

Supports inline false-positive comments across all ecosystems: `gitleaks:allow`, `pragma: allowlist secret`, `trufflehog:ignore`, and `refit:ignore`.

---

## Verified `v0.6.0` Footprint & Benchmarks

| Metric | Measured Value | Verification |
| :--- | :--- | :--- |
| Binary size | **`12.71 MiB`** (Win x64 `--release --features full`) · **`10.81 MiB`** (v0.6.0 `--profile dist`) | Single binary w/ embedded `weights.npz` |
| Active Cargo duplicates | **`0`** across all 4 targets | `cargo tree --locked -e normal --duplicates` |
| Neural parity vs JS | **`0.000000`** max delta | `S4` exact SIMD + scalar parity |
| Real-repo cold scan | **`79 ms`** (`gin`) · **`346 ms`** (`hono`) · **`1.04 s`** (`uv`) | `refit scan --no-cache` |
| 1-vCPU Peak RSS (`jem_full`) | **`17.57 MiB`** (`mcp`) · **`27.34 MiB`** (`scan`) · **`29.71 MiB`** (`audit`) | Warm `56.95 ms`, sustain `27.67 ms` |
| Quality & Verification Gates | **`114/114`** Win (`108` unit + `6` E2E) · `S3/S10/S14` PASS · **`8/8`** `T14` mutants killed | `47/48` `.rs` files, `0` clippy warnings, `0` clones, `0` `doc_drift` |

---

## Install & Supply-Chain Verification

```bash
# 1) Prebuilt binary via cargo-binstall (Linux, macOS Apple Silicon, Windows)
cargo binstall refit

# 2) Verify SLSA Level 3 build provenance + SHA-256 checksums for GitHub Release archives
gh attestation verify refit-<target>.<ext> --repo patrickfield152-del/refit
sha256sum -c --ignore-missing SHA256SUMS.txt
tar -xzf refit-x86_64-unknown-linux-gnu.tar.gz   # or refit-aarch64-apple-darwin.tar.gz

# 3) Or build from source (Rust 1.100+, parallel ThinLTO; release binaries use --profile dist)
cargo install --path . --locked --features full
```

## CLI & Verification Quick Start

```bash
# Verify (S1-S14) and mutation-test (T14) via pure-Rust xtask
cargo clippy --locked --workspace --all-targets --features full -- -D warnings
cargo test --locked --features full
cargo xtask verify
cargo xtask mutate

# Everyday agent & developer commands
refit init .                              # one-command hooks, AGENTS.md, .refit/invariants.yml, CI guard
refit check . --staged --deps             # unified pre-flight: secrets, Unicode guard, hygiene/ABI, OSV.dev SCA
refit hook                                # <5 ms stdin-JSON hook (Claude Code / Cursor / Gemini CLI / Kiro)
refit mcp .                               # converged MCP server (stdio; add --http <addr> for stateless HTTP)
refit secrets . --staged                  # millisecond pre-commit / pre-push secret guard
refit scan . --fast                       # DAG, symbol graph, and neural hygiene
refit audit . --markdown                  # clones, dead symbols, unused deps, infra security
refit sustain . --dry-run                 # preview AGENTS.md, .refit/invariants.yml, ADR-001, CI guard
refit learn . --rewind my_symbol          # SymbolGraph 1-hop rewind impact query
refit gc .                                # prune CAS blobs, stages, and bounded caches
refit run . --worktree --openclaw-url http://127.0.0.1:18789/v1
```

## Feature Flags

| Feature | Purpose |
| :--- | :--- |
| `alloc-mimalloc` (default) | Uses `mimalloc` global allocator |
| `alloc-jemalloc` | Uses `tikv-jemallocator` on non-MSVC targets |
| `wal-crc32c` | Hardware CRC32C checksums on MVCC WAL frames |
| `sst-storage` | Zstd front-coded symbol SST index with Bloom filter |
| `live-rewind` | SymbolGraph-backed rewind impact queries on retry |
| `doc-drift` | Documentation drift audit checks |
| `ste100` | ASD-STE100 plain-language prose compilation and linting |
| `build-cache` | Copy-on-write reflink stage acceleration |
| `full` | Enables `wal-crc32c`, `sst-storage`, `live-rewind`, `doc-drift`, `ste100`, `build-cache` |

## License & Credits

Dual-licensed under **`MIT OR Apache-2.0`** (standard Rust ecosystem license; compatible with permissive rule ecosystems).

- [`proxysoul/stem`](https://github.com/proxysoul/stem) — symbol graph inspiration
- [`brianbell-x/codehygiene`](https://github.com/brianbell-x/codehygiene) — ternary neural code hygiene weights (`weights.npz`)
- [`mongodb/kingfisher`](https://github.com/mongodb/kingfisher) (Apache-2.0) & [`gitleaks/gitleaks`](https://github.com/gitleaks/gitleaks) (MIT) — permissive secret pattern formats & prefilter design references
