# refit

`refit` is a pure-Rust polyglot code audit, verification, and parallel refactoring engine. It scans Python, Rust, TypeScript, JavaScript, Go, and infra files with zero Node or Bun runtime.

- **Single Binary:** Builds on Linux, macOS, and Windows. Embeds the `codehygiene` ternary neural model (`weights.npz`) with zero external JS sidecar.
- **Interactive Replay:** [patrickfield152-del.github.io/refit/?replay=run.jsonl](https://patrickfield152-del.github.io/refit/?replay=run.jsonl)

## Core Capabilities

1. **Repository Scan and Audit (`refit scan`, `refit audit`):** Detect cross-file clones, dead exported symbols, orphan files, large high-risk files, duplicate modules, unused dependencies (`package.json`, `Cargo.toml`, `pyproject.toml`, `go.mod`, lockfiles), and infra issues (Terraform, Dockerfiles, GitHub Actions, SQL).
2. **Exact Pure-Rust Hygiene (`src/hygiene.rs`):** Run the embedded ternary neural model (`weights.npz`) with `0.000000` max parity delta against `vendor/codehygiene/worker.mjs`. Supports optional `--approx-long` fast path for long files while keeping exact caches isolated.
3. **Polyglot Symbol Graph and DAG (`src/stem.rs`, `src/graph.rs`):** Index Python, Rust, TypeScript/TSX, JavaScript, and Go with `.gitignore` filtering, test-code stripping, and zstd front-coded SST storage for large repos (`>= 500` files). Collapse cyclic SCCs and dispatch disjoint Kahn waves.
4. **Atomic Multi-File Transactions (`src/graph/patch.rs`, `src/mvcc.rs`):** Apply `edit`, `create`, `delete`, and `rename` patches inside atomic MVCC WAL transactions with BLAKE3 CAS checks, OS file locks, CRC32C checksums, and crash recovery.
5. **Sandboxed Formatters and Gates (`src/gates.rs`, `src/gates/stage.rs`):** Run `ruff format`, `rustfmt`, `gofmt`, and `biome format` before Gates 0 to 3.5 inside ephemeral stages (`clear_env=true`, `CACHEDIR.TAG`, bounded LRU GC).
6. **Hardened Proxy, Web Search, and Concurrent MCP (`src/proxy.rs`, `src/research.rs`, `src/mcp.rs`):** Pace rolling uncached TPM with exact refunds, route web research across configurable providers (`REFIT_SEARCH_PROVIDERS`), and serve 17 concurrent MCP tools over stdio or stateless HTTP (`--http`).
7. **Governance Artifacts and STE100 Prose (`refit sustain`, `refit learn`):** Generate `AGENTS.md`, `.refit/invariants.yml`, `docs/ADR-001-untangle.md`, and `.github/workflows/refit-guard.yml`, with optional `--ste100` prose linting and SymbolGraph-backed `--rewind` impact queries.

## Install

### Prebuilt Release Binaries (Linux, macOS Apple Silicon, Windows)

Download the archive for your platform plus `SHA256SUMS.txt` from [GitHub Releases](https://github.com/patrickfield152-del/refit/releases):

```bash
# Linux / macOS (verify checksum, then extract)
sha256sum -c --ignore-missing SHA256SUMS.txt
tar -xzf refit-x86_64-unknown-linux-gnu.tar.gz   # or refit-aarch64-apple-darwin.tar.gz
```

```powershell
# Windows (PowerShell)
Get-FileHash refit-x86_64-pc-windows-msvc.zip -Algorithm SHA256
Expand-Archive refit-x86_64-pc-windows-msvc.zip -DestinationPath .\bin
```

### Build from Source (Rust `1.100.0`)

```bash
cargo install --path . --locked --features full
```

## Verified `v0.6.0` Benchmarks & Footprint

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| Release binary size | **`10.81 MiB`** (Win x64) · **`13.82 MiB`** (Linux x64 `jemalloc`) | Single static binary w/ embedded `weights.npz` |
| Active Cargo dep duplicates | **`0`** across all 4 targets | `cargo tree --locked -e normal --duplicates` |
| Neural parity vs JS reference | **`0.000000`** max delta | Pure-Rust SIMD + scalar parity verified |
| Scan latency (real repos) | **`79 ms`** (`gin`) · **`346 ms`** (`hono`) · **`1.04 s`** (`uv`) | Cold `refit scan --no-cache` |
| 1-vCPU Peak RSS (`jem_full`) | **`17.57 MiB`** (`mcp`) · **`27.34 MiB`** (`scan`) · **`29.71 MiB`** (`audit`) | Warm `56.95 ms`, sustain `27.67 ms` |
| Quality & Mutation Gates | **`101/101`** Win · **`102/102`** WSL2 · **`8/8`** `T14` mutants killed | `0` clippy warnings, `0` squiggles (`51/51` files) |

## MCP Client Setup

`refit mcp` exposes 17 concurrent tools over stdio (default) or stateless HTTP (`--http`).

### Claude Code / Cursor / Antigravity / Codex (`stdio`)

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

### Stateless HTTP Server

```bash
refit mcp /path/to/repo --http 127.0.0.1:18791
```

### OpenClaw

Use [openclaw/openclaw.refit.json](openclaw/openclaw.refit.json) and [skills/refit/SKILL.md](skills/refit/SKILL.md).

## Quick Start

```bash
# Build, verify (S1-S14), and mutation-test (T14) via pure-Rust xtask
cargo clippy --locked --workspace --all-targets --features full -- -D warnings
cargo test --locked --features full
cargo build --release --locked --features full
cargo xtask verify
cargo xtask mutate

# Scan, audit, or sustain a repository
./target/release/refit scan ./fixture/target_repo --fast
./target/release/refit audit ./your_repo --markdown
./target/release/refit sustain ./your_repo --dry-run
./target/release/refit learn ./your_repo --rewind my_symbol
./target/release/refit gc ./your_repo

# Run parallel refactoring in an isolated git worktree
export CEREBRAS_API_KEY="csk-..."
export REFIT_SEARCH_PROVIDERS="tavily,exa,brave"
./target/release/refit run ./your_repo --worktree --openclaw-url http://127.0.0.1:18789/v1
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

## Credits

- [`proxysoul/stem`](https://github.com/proxysoul/stem)
- [`brianbell-x/codehygiene`](https://github.com/brianbell-x/codehygiene)
