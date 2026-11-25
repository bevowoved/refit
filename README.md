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

## Quick Start

```bash
# Build and test with warnings denied
cargo clippy --all-targets --features full -- -D warnings
cargo test --features full
cargo build --release --features full

# Scan, audit, or sustain a repository
./target/release/refit scan ./fixture/target_repo --fast
./target/release/refit audit ./your_repo --markdown
./target/release/refit sustain ./your_repo --dry-run
./target/release/refit learn ./your_repo --rewind my_symbol
./target/release/refit gc ./your_repo

# Run concurrent MCP server (stdio or HTTP)
./target/release/refit mcp ./your_repo
./target/release/refit mcp ./your_repo --http 127.0.0.1:18791

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
