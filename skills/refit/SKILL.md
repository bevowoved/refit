---
name: refit
description: >
  Converged local neural code hygiene, architectural untangle (cycles, clones, Genome twins,
  dead symbols, doc-drift, deslop), polyglot Stem symbol graph, web-wired MCP guidance
  (Keenable search + OSV.dev SCA), event-driven agent hooks (including secret & Unicode
  prompt-injection protection), atomic MVCC refactoring, and Gate 4 TestMu AI Kane CLI
  live browser/mobile verification across Python, Rust, TypeScript, JavaScript, Go, and infra.
license: MIT OR Apache-2.0
compatibility: Single static Rust binary (Linux, macOS, Windows); zero Node/Bun runtime required
metadata:
  author: bevowoved
  version: "1.0.0"
allowed-tools: Read, Glob, Grep, Bash(refit:*)
---

# refit Polyglot Agent Contract

Use `refit` as your converged code-hygiene, architectural untangle, web-guided MCP, and closed-loop verification spine. Every edit passes neural `codehygiene` scoring, isolated staging (`clear_env=true`), formatters, Gates 0–3.5, atomic MVCC commit, and Gate 4 **TestMu AI Kane CLI** independent browser/mobile verification.

## 1. Converged CLI & MCP Workflow

1. **Initialize & wire event-driven hooks once.** Run `refit init .` to bootstrap `.refit/invariants.yml`, `AGENTS.md`, CI, and `<5 ms` `refit hook` triggers for Claude Code, Cursor, Gemini CLI, and Kiro (dirty-UI tracking, secret/Unicode leak protection, and push gating).
2. **Audit, untangle, and deslop continuously.** Run `refit scan . --fast` (neural `codehygiene` + Stem DAG) and `refit audit . --markdown` (Pearce SCC cycles, Type-2 clones, co-change `Genome` twins, dead symbols, unused deps, README doc-drift, and optional ASD-STE100 plain-language prose).
3. **Pre-flight & Gate 4 Kane CLI verification.** Run `refit check [--staged|--diff|--deps|--verify-docs|--kane <spec>] [--format sarif|gh|md|human|json]` (or MCP tool `refit_check`). `--deps` performs static OSV.dev `/v1/querybatch` lockfile SCA; `--verify-docs` catches stale README paths/symbols; `--kane` short-circuits on red static checks (`0` credits spent), runs TestMu AI Kane CLI (`--agent` NDJSON, `_test.md` cached replays, `--policy`, `--bug-detection`), audits sealed `.evidence` ZIPs for unredacted secrets, and rewinds failures to exact `file:line` spans via `SymbolGraph`.
4. **Inspect & research via web-wired MCP.** Call read-only MCP tools (`refit_read`, `refit_symbols`, `refit_scan`, `refit_audit`, `refit_rewind`, `refit_search`)—all 12 converged tools expose MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), `refit_read` strips invisible Unicode Tag/BiDi characters, and `refit_search` provides keyless Keenable web research (`https://api.keenable.ai/mcp-public`; `REFIT_OFFLINE=1` for air-gapped mode).
5. **Apply atomically.** Use polymorphic `refit_tx` (`op: begin|stage|create|delete|move|commit|abort`) or `refit_apply` to verify multi-file patches through Gates 0–3.5 before MVCC WAL commit.
6. **MCP server & SEP-2640 Skills.** Launch `refit mcp .` (stdio or stateless HTTP) exposing 12 annotated tools (converged `refit_check` inherits MCP `2026-07-28` durable `task: true` + `tasks/cancel` for Kane runs) and `skill://refit/SKILL.md` via `skills/list`, `skills/get`, `resources/list`, and `resources/read`.

## 2. Patch Format

Write raw `<<<REFIT_PATCH ...>>>` blocks directly (never wrapped in JSON):

```text
<<<REFIT_PATCH path="src/mod.ext">>>
<<<SEARCH>>>
exact unique lines copied verbatim from the source file
<<<REPLACE>>>
refactored replacement lines
<<<END>>>
```

Structural operations for multi-file transactions:
- **Create:** `<<<REFIT_PATCH path="src/shared/auth.ts" op="create">>>` with full content in `<<<REPLACE>>>`.
- **Delete:** `<<<REFIT_PATCH path="src/dead.py" op="delete">>>` followed by `<<<END>>>`.
- **Rename:** `<<<REFIT_PATCH path="src/old.rs" op="rename" new_path="src/core/new.rs">>>` (optional `<<<SEARCH>>>`/`<<<REPLACE>>>`).

Constraints:
1. Keep every `<<<SEARCH>>>` block unique with 2–3 context lines when needed.
2. Keep exported public signatures unchanged unless `allow_abi_change` is enabled.
3. Never commit hardcoded secrets, `*:*` IAM wildcards, or root-user Dockerfiles.

## 3. Verification Gates

- **Pre-pass:** Deterministic formatters (`ruff format`, `rustfmt`, `biome format`, `gofmt`) in stage.
- **Gate 0 (Syntax & ABI):** Syntax check, typechecker (`ast.parse`, `cargo check`, `tsc --noEmit`, `node --check`, `go vet`), and exported ABI pin.
- **Gate 1 (Tests):** Impacted `pytest`, `cargo test`, `vitest`/`node --test`, or `go test` with `clear_env=true`.
- **Gate 2 (Hygiene):** Embedded pure-Rust `codehygiene` (`weights.npz`); target `>= 0.85`.
- **Gate 3 (Lint):** `ruff` + `ty --error-on-warning` (Python), `cargo clippy` (Rust), `biome` (TS/JS), `go vet` (Go).
- **Gate 3.5 (Security & Rules):** Secret guard + `.refit/invariants.yml` import boundaries.
- **Gate 4 (Live Browser/Mobile Proof via Kane CLI):** `refit check . --verify-docs --kane <spec>` + `<2 ms` opt-in `refit hook` dirty-UI push gate (`.refit/kane.json` or `REFIT_KANE_GATE=1`).

## 4. Retry Rules

- Read the exact failure summary on retry turns; never repeat a rejected patch.
- Keep edits minimal and avoid touching clean lines outside diagnostic anchors.
