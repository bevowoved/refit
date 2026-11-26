# refit

[![CI](https://github.com/bevowoved/refit/actions/workflows/ci.yml/badge.svg)](https://github.com/bevowoved/refit/actions/workflows/ci.yml)
[![Release](https://github.com/bevowoved/refit/actions/workflows/release.yml/badge.svg)](https://github.com/bevowoved/refit/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license--converged-ecosystem-credits)
[![Rust 1.100+](https://img.shields.io/badge/rust-1.100%2B-orange.svg)](rust-toolchain.toml)

**The converged neural code-hygiene, architectural untangle, and closed-loop verification machine for AI coding agents** — unifying embedded ternary neural `codehygiene` (`0.000000` parity), polyglot `stem` SymbolGraph + 1-hop rewind, clean-room `Genome` / `SimiFlow` clone & cycle untangling, deslop, web-wired MCP guidance (keyless Keenable search + OSV.dev SCA + SEP-2640 skills), event-driven `<5 ms` agent hooks (including secret & invisible Unicode protection), and atomic MVCC WAL transactions in **one zero-duplicate Rust binary** wired natively into **TestMu AI Kane CLI** for real browser & mobile proof.

> **Why this matches TestMu AI's Loop Engineering thesis:** TestMu AI's 2026 quality research found that **81% of teams have merged AI-written code they didn't fully understand**, and that **no agent harness can reliably audit itself**. `refit × Kane CLI` implements the exact two-tier verification loop that solves it: an **independent code-hygiene & structural gate** (`refit`) at every edit and merge, feeding directly into an **independent runtime behavioral gate** (**TestMu AI Kane CLI**) across real Chrome and mobile devices.

- **Live Interactive HUD & Gate 4 Proof:** [bevowoved.github.io/refit/?demo=1&at=100&tab=kane](https://bevowoved.github.io/refit/?demo=1&at=100&tab=kane) *(also `?replay=run.jsonl`)*
- **Agent Skill (`skills.sh` & SEP-2640 MCP):** [skills/refit/SKILL.md](skills/refit/SKILL.md) (`npx skills add bevowoved/refit --skill refit` + `skill://refit/SKILL.md`)

---

## Multi-Repo Convergence: Best of Multiple Ecosystems in One Zero-Duplicate Engine

Rather than shipping a single-purpose wrapper, `refit` distills and converges the sharpest ideas across specialized open-source ecosystems into one coherent, zero-duplicate Rust architecture wired natively to **Kane CLI**:

| Converged Layer | What `refit` Delivers Locally | Inspired By / Composed With |
| :--- | :--- | :--- |
| **Neural Code Hygiene & Deslop** | Embedded ternary neural `codehygiene` (`weights.npz`, width-gated SIMD, `0.000000` parity vs JS), anti-gaming strict rules, complexity & deslop scoring (`median=0.845`, `p10=0.821`) | [`brianbell-x/codehygiene`](https://github.com/brianbell-x/codehygiene) & [`bevowoved/codehygiene`](https://github.com/bevowoved/codehygiene) |
| **Polyglot `stem` SymbolGraph & Rewind** | 5-language lexer (Py/Rs/TS/JS/Go), Zstd front-coded SST + Bloom index, Pearce CSR SCC cycle detection, Kahn topological waves, and 1-hop `InvertedRewindIndex` failure causality | [`proxysoul/stem`](https://github.com/proxysoul/stem) |
| **Architectural Untangle & `Genome`** | Cross-file Type-2 clone detection (`SimiFlow`), co-change `Genome` twins, dead exported symbol pruning, unused dependency audits, static README doc-drift (`--verify-docs`), and ASD-STE100 plain-language prose compilation | Clean-room `untangle-codebase`, `Genome`, `SimiFlow`, `ASD-STE100` |
| **Web-Wired MCP Guidance & Skills** | **12 converged annotated MCP tools** (`stdio` + stateless HTTP), keyless Keenable web research (`refit_search`), static OSV.dev `/v1/querybatch` lockfile SCA, **SEP-2640** `skill://refit/SKILL.md`, and `agent-plugins.org` manifests (`plugin.json` + `mcp.json`) | [Keenable](https://api.keenable.ai/mcp-public), [OSV.dev](https://osv.dev/), [skills.sh](https://skills.sh/), [agent-plugins.org](https://agent-plugins.org/), [OpenClaw](openclaw/openclaw.refit.json) |
| **Event-Driven Agent Hooks & Security Shield** | `<5 ms` stdin-JSON `refit hook` (`PreToolUse`/`PostToolUse`/`BeforeTool`/`AfterTool`/`afterFileEdit`/`fileEdited`) for Claude Code, Cursor, Gemini CLI & Kiro — tracking dirty UI/docs, enforcing Gate 4 push readiness, blocking invisible Unicode Tag/BiDi injections, and catching leaked credentials (`0.54 ms` / 10K LOC 2-slab AVX2 Fat-Teddy) | [`mongodb/kingfisher`](https://github.com/mongodb/kingfisher) & [`gitleaks/gitleaks`](https://github.com/gitleaks/gitleaks) permissive rule ecosystems |
| **Atomic MVCC Refactoring Transactions** | `refit_tx` isolated stages (`clear_env=true`) backed by BLAKE3 content-addressed storage and hardware CRC32C WAL — committing multi-file repairs only when Gates 0–3.5 pass | Pure-Rust MVCC WAL (`src/mvcc.rs`) |
| **Gate 4 Live Browser & Mobile Proof** | Native capability-negotiated **TestMu AI Kane CLI** (`@testmuai/kane-cli`) spine: static-first short-circuit (`0` credits on red), polymorphic `result_code` (`100` / `"740"`), red `AGENT_INFO` parsing, declarative `--policy` verdicts, `.evidence` ZIP secret auditing, and `SymbolGraph` rewind | [`LambdaTest/kane-cli`](https://github.com/LambdaTest/kane-cli) + TestMu AI Hosted MCP + `@e2e-dev/testmu` + `@testmuai/rook` |

---

## Agent & Skill Setup (60 Seconds)

### 1. One-Command Onboarding (`refit init .`) & `skills.sh`

```bash
refit init .   # wires Git hooks, Claude Code / Cursor / Gemini CLI / Kiro hooks, AGENTS.md, .refit/invariants.yml, CI
```

Install the standalone Agent Skill via **`skills.sh`** (`agentskills.io` open standard):

```bash
npx skills add bevowoved/refit --skill refit
```

### 2. Connect the Converged 12-Tool MCP Server (`stdio` or Stateless HTTP)

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

Exposes **12 annotated tools** (`readOnlyHint` / `destructiveHint` / `openWorldHint`), polymorphic `refit_tx`, **SEP-2640 Skills over MCP** (`skill://refit/SKILL.md`), Unicode-sanitized `refit_read`, and keyless web-wired `refit_search` (Keenable). Also runs stateless HTTP (`refit mcp . --http 127.0.0.1:18791`) or OpenClaw ([openclaw/openclaw.refit.json](openclaw/openclaw.refit.json)).

### 3. Neural Hygiene, Untangle, Pre-Flight & Gate 4 Kane CLI

```bash
# Architectural untangle + neural codehygiene + clones + dead symbols + doc-drift
refit scan . --fast
refit audit . --markdown

# Unified pre-flight + README truth + Gate 4 Kane CLI live browser/mobile proof
refit check . [--staged | --diff origin/main] [--deps] [--verify-docs] \
  [--kane tests/kane/hud_desktop_replay_test.md] [--policy tests/kane/refit-policy.yaml] \
  [--format human|json|md|gh|sarif]

# Event-driven <5 ms agent hook (Claude Code, Cursor, Gemini CLI, Kiro)
refit hook

# 1-hop SymbolGraph impact rewind & learned repair rules
refit learn . --rewind my_symbol
```

### 4. Gate 4: Native `refit × TestMu AI Kane CLI` Integration

1. **Static-First Short-Circuit (`0` Credits Wasted):** Every `refit check --kane` and MCP `refit_check(kane=...)` runs `<10 ms` static hygiene/secret/Unicode checks, optional OSV.dev SCA, `--verify-docs` README drift, and async URL reachability pre-flight *before* launching a browser.
2. **Upgrade-Proof Wire & Capability Negotiation:** Probes `kane-cli --version` via `semver`, handles polymorphic `result_code` (`100` integer vs `"740"` confirmed product bug), red `AGENT_INFO` steps, declarative `--policy` verdicts ([tests/kane/refit-policy.yaml](tests/kane/refit-policy.yaml)), `events.ndjson` disk-tee recovery, and `0`-credit cached `_test.md` replays.
3. **Sealed `.evidence` Secret Auditor:** Because newer Kane CLI builds seal `.evidence` ZIP packs unredacted for forensic fidelity, `refit` inspects extracted `failure.yaml` / `result.yaml` / `*-console.ndjson` / `*.har` entries via its zip-bomb-capped `rawzip` reader, flags leaks (`kane/evidence_secret_leak` in SARIF/GH), and redacts excerpts before returning to agents.
4. **12 Converged MCP Tools (Antigravity `18 / 100` Tool-Cap Ready):** Kane runs converge into existing `refit_check` — keeping **12 total local MCP tools** (vs `92` for Playwright Test MCP under Google Antigravity's `~100` global cap) while inheriting MCP `2026-07-28` durable `task: true` + `tasks/cancel` and warm `SymbolGraph` rewind.
5. **Opt-In `<2 ms` Push Gate & Dirty-UI Tracking:** Set `REFIT_KANE_GATE=1` (or create `.refit/kane.json`) so `refit hook` marks UI/doc edits dirty on `PostToolUse`/`AfterTool`/`afterFileEdit`/`fileEdited` and blocks `git push` until `.refit/kane_last.json` is green for the current `.git/HEAD` (zero `git` subprocesses).
6. **Honest Limits:** `refit` verifies structural code hygiene, symbol graphs, and security statically in milliseconds (`0` credits) but cannot see rendered DOM or CSS; Kane CLI authors once and replays cached `_test.md` steps at `~0` LLM credits. Because sealed `.evidence` packs are unredacted, `refit` audits them before commit and `.testmuai/evidence/` stays gitignored. Local Android emulators / iOS simulators require macOS Apple Silicon, so Windows/Linux hosts route mobile specs ([tests/kane/hud_mobile_viewport_test.md](tests/kane/hud_mobile_viewport_test.md), [tests/kane/mobile_cloud_example.md](tests/kane/mobile_cloud_example.md)) to TestMu AI Cloud (`--target android --remote`).

### 5. Composing With the Full TestMu AI Ecosystem (`18 / 100` Antigravity Cap)

Pairing `refit` (`12` local tools, keyless) + **TestMu AI Hosted MCP** (`6` cloud suites via `"serverUrl"`) + **`kane-cli`** (`0` MCP slots via CLI/skill) uses `<20%` of Antigravity's `~100` global tool cap, and composes cleanly alongside `@e2e-dev/testmu` (`refit check . --deps && npx e2e run --target testmu-web`) and `@testmuai/rook` (OWASP Agentic Top 10 red-teaming):

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
# 1) Neural hygiene, untangle audit & README doc-drift (zero API keys, REFIT_OFFLINE=1 safe)
refit audit . --markdown
refit check . --verify-docs --format human

# 2) Open the live interactive Gate 4 HUD (desktop or 390x844 mobile)
#    https://bevowoved.github.io/refit/?demo=1&at=100&tab=kane

# 3) Live Kane CLI Gate 4 replay (with TestMu account, replays cached _test.md at ~0 LLM credits)
refit check . --verify-docs --kane tests/kane/hud_desktop_replay_test.md --policy tests/kane/refit-policy.yaml --format human
```

---

## Architecture & Two-Tier Loop Engineering Pipeline

```text
  Coding Agent / Git Index / Event-Driven Hook (<5 ms)
        │
        ├─► [Neural Gate]    Embedded ternary codehygiene (`weights.npz`, 0.000000 parity vs JS) + deslop
        ├─► [Stem & Untangle] Polyglot lexer (Py/Rs/TS/JS/Go) + zstd SST + Pearce SCC + Kahn waves
        │                    + Type-2 clones (`SimiFlow`) + `Genome` twins + dead symbols + doc-drift
        ├─► [Web-Wired MCP]  12 converged annotated tools (`refit mcp`) + keyless Keenable `refit_search`
        │                    + OSV.dev lockfile SCA + SEP-2640 `skill://refit/SKILL.md`
        ├─► [Event & Shield] <5 ms `refit hook` (Claude Code / Cursor / Gemini CLI / Kiro)
        │                    + AVX2 Fat-Teddy secret guard (0.54 ms / 10K LOC) + Unicode Tag/BiDi guard
        ├─► [Atomic MVCC]    BLAKE3 CAS + CRC32C WAL + ephemeral isolated stages (`clear_env=true`)
        └─► [Gate 4 Kane]    TestMu AI Kane CLI independent browser & mobile proof + `.evidence` audit
                             + 1-hop `SymbolGraph` rewind back to the coding agent
```

## Measured Footprint & Benchmarks

| Metric | Measured Value | Verification |
| :--- | :--- | :--- |
| Binary size | **`12.71 MiB`** (Win x64 `--release --features full`) · **`10.81 MiB`** (`--profile dist`) | Single static Rust binary w/ embedded `weights.npz` |
| Source footprint & crates | **`47 / 48` `.rs` files** · **`0` active Cargo duplicates** across all targets | `S3` self-audit & `cargo tree --locked -e normal --duplicates` |
| Neural hygiene parity | **`0.000000`** max delta vs reference JS (`median 0.845`, `p10 0.821`) | `S4` exact SIMD + scalar parity |
| Real-repo cold scan | **`79 ms`** (`gin`) · **`346 ms`** (`hono`) · **`1.04 s`** (`uv`) | `refit scan --no-cache` |
| Secret pre-flight layer | **`0.54 ms`** (10K LOC cold) · **`8.85 ms`** (400K LOC / 10K files) · **`0.24 ms`** (warm) | 2-slab ≤64 AVX2 Fat-Teddy + BLAKE3 cache |
| 1-vCPU Peak RSS (`jem_full`) | **`17.57 MiB`** (`mcp`) · **`27.34 MiB`** (`scan`) · **`29.71 MiB`** (`audit`) | Warm `56.95 ms`, sustain `27.67 ms` |
| Quality & Mutation Gates | **`114/114`** tests (`108` unit + `6` E2E) · `S3/S10/S14` PASS · **`8/8`** `T14` mutants killed | `0` clippy warnings, `0` `#[allow(`, `0` SCCs, `0` clones, `0` `doc_drift` |

Supports inline false-positive comments across ecosystems: `gitleaks:allow`, `pragma: allowlist secret`, `trufflehog:ignore`, and `refit:ignore`.

---

## Install & Supply-Chain Verification

```bash
# 1) Prebuilt binary via cargo-binstall (Linux, macOS Apple Silicon, Windows)
cargo binstall refit

# 2) Verify SLSA Level 3 build provenance + SHA-256 checksums for GitHub Release archives
gh attestation verify refit-<target>.<ext> --repo bevowoved/refit
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
refit scan . --fast                       # neural codehygiene, Stem symbol graph, and DAG waves
refit audit . --markdown                  # cycles, clones, Genome twins, dead symbols, unused deps, doc-drift
refit check . --staged --deps --verify-docs # unified pre-flight: hygiene/security, OSV SCA, doc-drift, Gate 4 Kane
refit hook                                # <5 ms event-driven stdin-JSON hook (Claude Code / Cursor / Gemini CLI / Kiro)
refit mcp .                               # 12-tool converged MCP server (stdio; add --http <addr> for stateless HTTP)
refit learn . --rewind my_symbol          # SymbolGraph 1-hop rewind impact query
refit sustain . --dry-run                 # preview AGENTS.md, .refit/invariants.yml, ADR-001, CI guard
refit secrets . --staged                  # direct sub-ms secret scan via git cat-file --batch
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

## License & Converged Ecosystem Credits

Dual-licensed under **`MIT OR Apache-2.0`**.

- [`proxysoul/stem`](https://github.com/proxysoul/stem) — polyglot symbol graph & SST index inspiration
- [`brianbell-x/codehygiene`](https://github.com/brianbell-x/codehygiene) & [`bevowoved/codehygiene`](https://github.com/bevowoved/codehygiene) — ternary neural code hygiene weights (`weights.npz`)
- Clean-room architectural untangling (`untangle-codebase`, `Genome` co-change twins, `SimiFlow` Type-2 clones, `ASD-STE100` prose linting)
- [`LambdaTest/kane-cli`](https://github.com/LambdaTest/kane-cli) — TestMu AI Kane CLI Gate 4 live browser & mobile verification spine
- [`mongodb/kingfisher`](https://github.com/mongodb/kingfisher) (Apache-2.0) & [`gitleaks/gitleaks`](https://github.com/gitleaks/gitleaks) (MIT) — permissive secret pattern formats & prefilter design references
