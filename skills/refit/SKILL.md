---
name: refit
description: >
  Local pre-flight code audit, sub-millisecond secret guard, polyglot symbol graph,
  and atomic MVCC refactoring engine for coding agents. Use when auditing repos,
  blocking secret leaks pre-push, untangling cycles/clones/dead symbols, or applying
  verified multi-file edits across Python, Rust, TypeScript, JavaScript, Go, and infra.
license: MIT OR Apache-2.0
compatibility: Single static Rust binary (Linux, macOS, Windows); zero Node/Bun runtime required
metadata:
  author: patrickfield152-del
  version: "0.6.0"
allowed-tools: Read, Glob, Grep, Bash(refit:*)
---

# refit Polyglot Agent Contract

Use `refit` as your deterministic local verification and refactoring engine. Every change passes isolated staging (`clear_env=true`), formatters, Gates 0–3.5, and atomic MVCC commit.

## 1. Workflow Rules

1. **Inspect before editing.** Call read-only tools (`refit_symbols`, `refit_read`, `refit_search`, `refit_audit`, `refit_rewind`, `refit_dep_audit`) to verify callers, dead symbols, and docs. Never guess unseen code.
2. **Guard secrets before commit/push.** Run `refit secrets . --staged` (or install native hooks via `refit secrets . --install-hooks`) before committing or pushing.
3. **Untangle structure.** Edit in place or batch `create`/`delete`/`rename` patches in one atomic MVCC transaction.
4. **Keep tests strong.** Never weaken or drop assertions; rely on stage formatters (`ruff format`, `rustfmt`, `biome format`, `gofmt`).

## 2. Patch Format

Write raw `<<<REFIT_PATCH ...>>>` blocks directly. Do not wrap blocks in JSON.

```text
<<<REFIT_PATCH path="src/mod.ext">>>
<<<SEARCH>>>
exact unique lines copied verbatim from the source file
<<<REPLACE>>>
refactored replacement lines
<<<END>>>
```

Use structural operations for multi-file changes:
- **Create file:** `<<<REFIT_PATCH path="src/shared/auth.ts" op="create">>>` with full content in `<<<REPLACE>>>`.
- **Delete file:** `<<<REFIT_PATCH path="src/dead.py" op="delete">>>` followed by `<<<END>>>`.
- **Rename file:** `<<<REFIT_PATCH path="src/old.rs" op="rename" new_path="src/core/new.rs">>>` (add `<<<SEARCH>>>`/`<<<REPLACE>>>` to edit during the move).

Patch constraints:
1. Make every `<<<SEARCH>>>` block unique. Include 2 to 3 context lines when lines repeat.
2. Keep exported public signatures unchanged unless `allow_abi_change` is enabled.
3. Do not commit hardcoded secrets, `*:*` IAM wildcards, or root-user Dockerfiles.

## 3. Verification Gates

- **Pre-pass:** Run deterministic formatters (`ruff format`, `rustfmt`, `biome format`, `gofmt`) in stage.
- **Gate 0 (Syntax and ABI):** Run syntax check, typechecker (`ast.parse`, `cargo check`, `tsc --noEmit`, `node --check`, `go vet`), and exported ABI pin.
- **Gate 1 (Tests):** Run impacted `pytest`, `cargo test`, `vitest`/`node --test`, or `go test` with `clear_env=true`.
- **Gate 2 (Hygiene):** Run embedded pure-Rust `codehygiene` (`weights.npz`). Target `>= 0.85` and fix `deep_nesting`, `dense_line`, `cramped_layout`, `duplicate_block`, `repetitive_structure`, and `comment_echo`.
- **Gate 3 (Lint):** Run `ruff` and `ty --error-on-warning` (Python), `cargo clippy` (Rust), `biome` (TS/JS), and `go vet` (Go).
- **Gate 3.5 (Security and Rules):** Block hardcoded credentials and enforce `.refit/invariants.yml` import boundaries.

## 4. Reasoning and Retry Rules

- Use `low` reasoning effort on routine cleanups. Use `medium` or `high` effort on test failures and cyclic SCCs.
- Read the exact failure summary on retry turns. Do not repeat a rejected patch.
- Keep edits minimal. Do not rewrite clean lines outside diagnostic anchors.
