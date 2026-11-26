# refit

[![CI](https://github.com/patrickfield152-del/refit/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickfield152-del/refit/actions/workflows/ci.yml)
[![Release](https://github.com/patrickfield152-del/refit/actions/workflows/release.yml/badge.svg)](https://github.com/patrickfield152-del/refit/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license--credits)
[![Rust 1.100+](https://img.shields.io/badge/rust-1.100%2B-orange.svg)](rust-toolchain.toml)

**The local pre-flight engine for coding agents:** sub-millisecond secret push-guard, embedded ternary neural code hygiene, polyglot symbol graph, and atomic MVCC refactoring in **one static Rust binary** (zero Node/Bun runtime, zero active dependency duplicates).

- **Interactive Replay:** [patrickfield152-del.github.io/refit/?replay=run.jsonl](https://patrickfield152-del.github.io/refit/?replay=run.jsonl)
- **Agent Skill:** [skills/refit/SKILL.md](skills/refit/SKILL.md) (open `agentskills.io` / `skills.sh` standard)

---

## Agent & Skill Setup (60 Seconds)

### 1. Install the Agent Skill (`Claude Code`, `Cursor`, `Antigravity`, `Codex`)

```bash
npx skills add patrickfield152-del/refit --skill refit
```

### 2. Connect the 17-Tool MCP Server (`stdio` or Stateless HTTP)

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

Or run stateless HTTP (`refit mcp . --http 127.0.0.1:18791`) or OpenClaw ([openclaw/openclaw.refit.json](openclaw/openclaw.refit.json)).

### 3. Block Secret Leaks Locally in Milliseconds (`pre-commit` + `pre-push`)

```bash
refit secrets . --install-hooks     # installs .git/hooks/{pre-commit,pre-push}
refit secrets . --staged            # <10 ms staged blob scan via git cat-file --batch
refit secrets . --diff origin/main  # <10 ms unpushed commit scan
refit secrets .                     # parallel zero-recompute repo scan (BLAKE3 cache)
```

---

## Architecture & Pipeline

```text
  Repo / Git Index
        │
        ├─► [Secret Guard]  2-slab ≤64 AVX2 Fat-Teddy SIMD (130+ explicit rules + Tier-4 entropy)
        │                   + BLAKE3 zero-recompute cache → 0.54 ms (10K LOC) / 0.24 ms warm (10K files)
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
| Binary size | **`10.81 MiB`** (Win x64) · **`13.82 MiB`** (Linux x64 `jemalloc`) | Single binary w/ embedded `weights.npz` |
| Active Cargo duplicates | **`0`** across all 4 targets | `cargo tree --locked -e normal --duplicates` |
| Neural parity vs JS | **`0.000000`** max delta | `S4` exact SIMD + scalar parity |
| Real-repo cold scan | **`79 ms`** (`gin`) · **`346 ms`** (`hono`) · **`1.04 s`** (`uv`) | `refit scan --no-cache` |
| 1-vCPU Peak RSS (`jem_full`) | **`17.57 MiB`** (`mcp`) · **`27.34 MiB`** (`scan`) · **`29.71 MiB`** (`audit`) | Warm `56.95 ms`, sustain `27.67 ms` |
| Quality & Mutation Gates | **`101/101`** Win · **`102/102`** WSL2 · **`8/8`** `T14` mutants killed | `0` clippy warnings, `0` squiggles, `0` infra findings |

---

## Install

```bash
# 1) Prebuilt binary via cargo-binstall (Linux, macOS Apple Silicon, Windows)
cargo binstall refit

# 2) Or download archives + SHA256SUMS.txt from GitHub Releases
sha256sum -c --ignore-missing SHA256SUMS.txt
tar -xzf refit-x86_64-unknown-linux-gnu.tar.gz   # or refit-aarch64-apple-darwin.tar.gz

# 3) Or build from source (Rust 1.100+, parallel ThinLTO)
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
refit secrets . --staged                  # millisecond pre-commit / pre-push secret guard
refit scan . --fast                       # DAG, symbol graph, and neural hygiene
refit audit . --markdown                  # clones, dead symbols, unused deps, infra security
refit sustain . --dry-run                 # generate AGENTS.md, invariants.yml, ADR-001, CI guard
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
