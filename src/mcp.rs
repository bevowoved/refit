use crate::gates::{
    read_gates, scan_cached, run_gates,
};
use crate::graph::{
    apply_patches, hash_abi, parse_patches, impacted_tests,
};
use crate::hygiene::HygieneScorer;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::platform::{write_atomic, check_rel, now_ms};
use crate::research::WebClient;
use crate::stem::query_symbols;
use crate::types::ModuleUnit;
use crate::untangle::{untangle_repo, render_audit, write_sustain};
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

static MCP_TX_SEQ: AtomicU64 = AtomicU64::new(1);
static TX_HANDLE_LOCKS: OnceLock<DashMap<String, Arc<Mutex<()>>>> = OnceLock::new();

fn lock_tx(handle: &str) -> Arc<Mutex<()>> {
    let map = TX_HANDLE_LOCKS.get_or_init(DashMap::new);
    map.entry(handle.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StagedTxFile {
    base_blake3: String,
    original_content: String,
    staged_content: String,
    #[serde(default)]
    created: bool,
    #[serde(default)]
    deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TxMeta {
    tx_handle: String,
    created_ms: u64,
    ttl_ms: u64,
    staged: BTreeMap<String, StagedTxFile>,
}

fn txs_dir(repo_root: &Path) -> PathBuf {
    repo_root
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.join(".refit-tx"))
        .unwrap_or_else(|| std::env::temp_dir().join("refit-tx"))
}

fn check_txid(handle: &str) -> bool {
    !handle.is_empty()
        && handle.len() <= 64
        && handle
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn rel_safe(rel: &str) -> bool {
    check_rel(rel)
}

/// Verify that the canonical target or its nearest existing parent stays inside `repo_root`.
fn check_target(repo_root: &Path, rel: &str) -> bool {
    if !rel_safe(rel) {
        return false;
    }
    let Ok(canon_root) = fs::canonicalize(repo_root) else {
        return true;
    };
    let candidate = repo_root.join(rel);
    if candidate.exists() {
        return fs::canonicalize(&candidate)
            .map(|c| c.starts_with(&canon_root))
            .unwrap_or(false);
    }
    let mut anc = candidate.parent();
    while let Some(dir) = anc {
        if dir.exists() {
            return fs::canonicalize(dir)
                .map(|c| c.starts_with(&canon_root))
                .unwrap_or(false);
        }
        anc = dir.parent();
    }
    false
}

fn sweep_txs(tx_root: &Path, now: u64) -> usize {
    let Ok(rd) = fs::read_dir(tx_root) else {
        return 0;
    };
    let corrupt_cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(1800))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut swept = 0usize;
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let meta_path = dir.join("meta.json");
        let expired = match fs::read_to_string(&meta_path)
            .ok()
            .and_then(|s| serde_json::from_str::<TxMeta>(&s).ok())
        {
            Some(m) => now >= m.created_ms.saturating_add(m.ttl_ms),
            None => fs::metadata(&dir)
                .and_then(|m| m.modified())
                .map(|t| t <= corrupt_cutoff)
                .unwrap_or(true),
        };
        if expired && fs::remove_dir_all(&dir).is_ok() {
            swept += 1;
        }
    }
    swept
}

/// Run the concurrent JSON-RPC 2.0 MCP server over stdio with a serialized writer task.
pub async fn serve_mcp(repo_root: PathBuf) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<String>(64);
    let writer_handle = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(mut line) = out_rx.recv().await {
            line.push('\n');
            if stdout.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if stdout.flush().await.is_err() {
                break;
            }
        }
    });

    let hygiene = Arc::new(HygieneScorer::open_persistent(&repo_root));
    let gate_cfg = Arc::new(read_gates(&repo_root));
    let research = WebClient::for_repo(&repo_root, None);
    let mut tasks = tokio::task::JoinSet::new();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = req.get("id").cloned();
        let method = req
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();

        match method.as_str() {
            "initialize" => {
                let resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "refit", "version": "0.5.0" }
                    }
                });
                let _ = out_tx.send(serde_json::to_string(&resp)?).await;
            }
            "notifications/initialized" => {}
            "ping" => {
                let resp = json!({ "jsonrpc": "2.0", "id": id, "result": {} });
                let _ = out_tx.send(serde_json::to_string(&resp)?).await;
            }
            "tools/list" => {
                let resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "tools": tool_catalog() }
                });
                let _ = out_tx.send(serde_json::to_string(&resp)?).await;
            }
            "tools/call" => {
                let name = req["params"]["name"].as_str().unwrap_or("").to_string();
                let args = req["params"]["arguments"].clone();
                let repo_c = repo_root.clone();
                let hyg_c = hygiene.clone();
                let cfg_c = gate_cfg.clone();
                let res_c = research.clone();
                let tx_c = out_tx.clone();
                tasks.spawn(async move {
                    let payload = if name == "refit_search" {
                        let lang = args["language"].as_str().unwrap_or("Rust");
                        let cap = args["capability"].as_str().unwrap_or("rate limiter");
                        match res_c.check_library(lang, cap).await {
                            Ok(r) => serde_json::to_value(r).unwrap_or_else(|_| json!({"ok": true})),
                            Err(e) => json!({"error": e.to_string()}),
                        }
                    } else {
                        tokio::task::spawn_blocking(move || {
                            dispatch_tool(&name, &args, &repo_c, &hyg_c, &cfg_c, &res_c)
                        })
                        .await
                        .unwrap_or_else(|e| json!({"error": format!("mcp worker join error: {e}")}))
                    };
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".into())
                            }]
                        }
                    });
                    if let Ok(encoded) = serde_json::to_string(&resp) {
                        let _ = tx_c.send(encoded).await;
                    }
                });
            }
            _ => {
                if let Some(req_id) = id {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": { "code": -32601, "message": format!("Method not found: {method}") }
                    });
                    let _ = out_tx.send(serde_json::to_string(&resp)?).await;
                }
            }
        }
    }

    while tasks.join_next().await.is_some() {}
    drop(out_tx);
    let _ = writer_handle.await;
    Ok(())
}

fn tool_catalog() -> Value {
    json!([
        {
            "name": "refit_scan",
            "description": "Scan the repository DAG and codehygiene scores. Uses fast cached mode by default.",
            "inputSchema": {
                "type": "object",
                "properties": { "fast": { "type": "boolean" } }
            }
        },
        {
            "name": "refit_audit",
            "description": "Audit the repository for clones, dead symbols, large files, unused dependencies, and infra issues.",
            "inputSchema": {
                "type": "object",
                "properties": { "markdown": { "type": "boolean" } }
            }
        },
        {
            "name": "refit_sustain",
            "description": "Generate AGENTS.md, .refit/invariants.yml, ADR-001, and the CI guard workflow.",
            "inputSchema": {
                "type": "object",
                "properties": { "dry_run": { "type": "boolean" } }
            }
        },
        {
            "name": "refit_symbols",
            "description": "Query symbol definitions, callers, file dependencies, fan-in, fan-out, and dead symbols.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": { "type": "string" },
                    "file": { "type": "string" }
                }
            }
        },
        {
            "name": "refit_read",
            "description": "Read a bounded line range from a repository file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "start_line": { "type": "integer" },
                    "end_line": { "type": "integer" }
                },
                "required": ["path"]
            }
        },
        {
            "name": "refit_apply",
            "description": "Apply ASRF patches, run formatters and Gates 0 to 3.5, and commit via MVCC WAL.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module_id": { "type": "string" },
                    "patch_text": { "type": "string" },
                    "dry_run": { "type": "boolean" },
                    "allow_abi_change": { "type": "boolean" }
                },
                "required": ["module_id", "patch_text"]
            }
        },
        {
            "name": "refit_tx_begin",
            "description": "Start a multi-file shadow transaction outside the repository and return a tx_handle.",
            "inputSchema": {
                "type": "object",
                "properties": { "ttl_secs": { "type": "integer" } }
            }
        },
        {
            "name": "refit_tx_stage",
            "description": "Stage ASRF patches (edit, create, delete, rename) inside an active transaction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tx_handle": { "type": "string" },
                    "patch_text": { "type": "string" }
                },
                "required": ["tx_handle", "patch_text"]
            }
        },
        {
            "name": "refit_tx_create",
            "description": "Stage a new file inside an active transaction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tx_handle": { "type": "string" },
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["tx_handle", "path", "content"]
            }
        },
        {
            "name": "refit_tx_delete",
            "description": "Stage file deletion inside an active transaction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tx_handle": { "type": "string" },
                    "path": { "type": "string" }
                },
                "required": ["tx_handle", "path"]
            }
        },
        {
            "name": "refit_tx_move",
            "description": "Stage a file move or rename inside an active transaction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tx_handle": { "type": "string" },
                    "from_path": { "type": "string" },
                    "to_path": { "type": "string" }
                },
                "required": ["tx_handle", "from_path", "to_path"]
            }
        },
        {
            "name": "refit_tx_commit",
            "description": "Verify all staged files with formatters and Gates 0 to 3.5, then commit via MVCC WAL.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tx_handle": { "type": "string" },
                    "dry_run": { "type": "boolean" },
                    "allow_abi_change": { "type": "boolean" }
                },
                "required": ["tx_handle"]
            }
        },
        {
            "name": "refit_tx_abort",
            "description": "Abort an active shadow transaction and remove its staged files.",
            "inputSchema": {
                "type": "object",
                "properties": { "tx_handle": { "type": "string" } },
                "required": ["tx_handle"]
            }
        },
        {
            "name": "refit_resume",
            "description": "Recover unfinished WAL transactions from CAS and report committed modules.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "refit_verify",
            "description": "Run a full repository audit with BLAKE3 content caching.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "refit_search",
            "description": "Run native 10 req/s GCRA-paced KeenAble web search with repo-scoped BLAKE3 cache.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "language": { "type": "string" },
                    "capability": { "type": "string" }
                },
                "required": ["language", "capability"]
            }
        }
    ])
}

fn read_tx(repo_root: &Path, handle: &str) -> std::result::Result<(PathBuf, TxMeta), Value> {
    if !check_txid(handle) {
        return Err(json!({"ok": false, "error": "invalid tx_handle"}));
    }
    let meta_path = txs_dir(repo_root).join(handle).join("meta.json");
    let Some(meta) = fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<TxMeta>(&s).ok())
    else {
        return Err(json!({"ok": false, "error": format!("unknown tx_handle `{handle}`")}));
    };
    Ok((meta_path, meta))
}

fn write_tx(meta_path: &Path, meta: &TxMeta) -> std::result::Result<(), Value> {
    let encoded = serde_json::to_string_pretty(meta)
        .map_err(|e| json!({"ok": false, "error": format!("encode tx meta: {e}")}))?;
    write_atomic(meta_path, encoded.as_bytes())
        .map_err(|e| json!({"ok": false, "error": format!("persist tx meta: {e}")}))
}

fn dispatch_tool(
    name: &str,
    args: &Value,
    repo_root: &Path,
    hygiene: &HygieneScorer,
    gate_cfg: &crate::gates::GateConfig,
    _research: &Arc<WebClient>,
) -> Value {
    match name {
        "refit_scan" => {
            let fast = args["fast"].as_bool().unwrap_or(true);
            match scan_cached(repo_root, true, fast, hygiene) {
                Ok((bundle, cache_hit)) => {
                    let mut v = serde_json::to_value(&bundle.plan).unwrap_or_else(|_| json!({"ok": true}));
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("scan_cache_hit".into(), json!(cache_hit));
                        obj.insert("fast_mode".into(), json!(fast));
                        obj.insert("detected_languages".into(), json!(bundle.detected_languages));
                        obj.insert("polyglot_toolchains".into(), json!(bundle.polyglot_toolchains));
                    }
                    v
                }
                Err(e) => json!({"error": e.to_string()}),
            }
        }
        "refit_audit" => match untangle_repo(repo_root, hygiene) {
            Ok(rep) => {
                if args["markdown"].as_bool().unwrap_or(false) {
                    json!({"markdown": render_audit(&rep), "summary": {
                        "clones": rep.clone_clusters.len(),
                        "dead_symbols": rep.dead_symbols.len(),
                        "god_files": rep.god_files.len(),
                        "infra_findings": rep.infra_findings.len(),
                        "unused_deps": rep.unused_deps.len()
                    }})
                } else {
                    serde_json::to_value(&rep).unwrap_or_else(|_| json!({"ok": true}))
                }
            }
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_sustain" => {
            let dry_run = args["dry_run"].as_bool().unwrap_or(false);
            match untangle_repo(repo_root, hygiene)
                .and_then(|rep| write_sustain(repo_root, &rep, dry_run))
            {
                Ok(out) => serde_json::to_value(&out).unwrap_or_else(|_| json!({"ok": true})),
                Err(e) => json!({"error": e.to_string()}),
            }
        }
        "refit_symbols" => {
            let sym = args["symbol"].as_str();
            let file = args["file"].as_str();
            query_symbols(repo_root, sym, file)
        }
        "refit_read" => {
            let rel = args["path"].as_str().unwrap_or("");
            if !check_target(repo_root, rel) {
                return json!({"error": format!("unsafe path `{rel}`")});
            }
            let Ok(content) = fs::read_to_string(repo_root.join(rel)) else {
                return json!({"error": format!("file not found `{rel}`")});
            };
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let start = args["start_line"].as_u64().unwrap_or(1).max(1) as usize;
            let end = args["end_line"]
                .as_u64()
                .map(|e| e as usize)
                .unwrap_or_else(|| start.saturating_add(199).min(total))
                .min(total)
                .min(start.saturating_add(400));
            let slice = if start <= end && start >= 1 {
                lines[start - 1..end]
                    .iter()
                    .enumerate()
                    .map(|(idx, l)| format!("{}: {}", start + idx, l))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                String::new()
            };
            json!({
                "path": rel,
                "start_line": start,
                "end_line": end,
                "total_lines": total,
                "content": slice
            })
        }
        "refit_resume" => match MvccStore::open(repo_root, "mcp-resume")
            .and_then(|s| s.recover_wal())
        {
            Ok(rep) => json!({
                "rolled_back_txs": rep.rolled_back_txs,
                "committed_modules": rep.committed_modules.len(),
                "head_rev": rep.head_rev
            }),
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_verify" => match scan_cached(repo_root, true, false, hygiene) {
            Ok((bundle, cache_hit)) => json!({
                "scan_cache_hit": cache_hit,
                "tests_total": bundle.audit.tests_total,
                "tests_failed": bundle.audit.tests_failed,
                "ruff_errors": bundle.audit.ruff_errors,
                "ty_errors": bundle.audit.ty_errors,
                "hygiene_median": bundle.audit.hygiene_median,
                "hygiene_p10": bundle.audit.hygiene_p10
            }),
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_tx_begin" => {
            let now = now_ms();
            let ttl_secs = args["ttl_secs"].as_u64().unwrap_or(1800).clamp(1, 86_400);
            let ttl_ms = ttl_secs * 1000;
            let root = txs_dir(repo_root);
            let swept = sweep_txs(&root, now);
            let handle = format!("tx-{}-{}", now, MCP_TX_SEQ.fetch_add(1, Ordering::Relaxed));
            let tx_dir = root.join(&handle);
            if let Err(e) = fs::create_dir_all(&tx_dir) {
                return json!({"error": format!("create tx dir: {e}")});
            }
            let meta = TxMeta {
                tx_handle: handle.clone(),
                created_ms: now,
                ttl_ms,
                staged: BTreeMap::new(),
            };
            if let Err(e) = write_tx(&tx_dir.join("meta.json"), &meta) {
                return e;
            }
            json!({
                "tx_handle": handle,
                "created_ms": now,
                "ttl_ms": ttl_ms,
                "swept_orphans": swept
            })
        }
        "refit_tx_create" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            let path = args["path"].as_str().unwrap_or("").replace('\\', "/");
            let content = args["content"].as_str().unwrap_or("").to_string();
            if !check_target(repo_root, &path) {
                return json!({"ok": false, "error": format!("unsafe path `{path}`")});
            }
            let lock = lock_tx(handle);
            let _g = lock.lock();
            let (meta_path, mut meta) = match read_tx(repo_root, handle) {
                Ok(v) => v,
                Err(e) => return e,
            };
            let deleted_in_tx = meta.staged.get(&path).map(|s| s.deleted).unwrap_or(false);
            if repo_root.join(&path).exists() && !deleted_in_tx {
                return json!({"ok": false, "error": format!("file already exists on disk: `{path}`")});
            }
            if let Some(prev) = meta.staged.get_mut(&path) {
                prev.staged_content = content;
                prev.deleted = false;
            } else {
                meta.staged.insert(
                    path.clone(),
                    StagedTxFile {
                        base_blake3: String::new(),
                        original_content: String::new(),
                        staged_content: content,
                        created: true,
                        deleted: false,
                    },
                );
            }
            if let Err(e) = write_tx(&meta_path, &meta) {
                return e;
            }
            json!({"ok": true, "tx_handle": handle, "created": path, "staged_files_count": meta.staged.len()})
        }
        "refit_tx_delete" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            let path = args["path"].as_str().unwrap_or("").replace('\\', "/");
            if !check_target(repo_root, &path) {
                return json!({"ok": false, "error": format!("unsafe path `{path}`")});
            }
            let lock = lock_tx(handle);
            let _g = lock.lock();
            let (meta_path, mut meta) = match read_tx(repo_root, handle) {
                Ok(v) => v,
                Err(e) => return e,
            };
            if let Some(prev) = meta.staged.get(&path) {
                if prev.created {
                    meta.staged.remove(&path);
                    if let Err(e) = write_tx(&meta_path, &meta) {
                        return e;
                    }
                    return json!({"ok": true, "tx_handle": handle, "deleted": path, "staged_files_count": meta.staged.len()});
                }
            }
            let (base_hash, orig) = if let Some(prev) = meta.staged.get(&path) {
                (prev.base_blake3.clone(), prev.original_content.clone())
            } else {
                let Ok(disk) = fs::read_to_string(repo_root.join(&path)) else {
                    return json!({"ok": false, "error": format!("file not found on disk: `{path}`")});
                };
                (MvccStore::hash_bytes(disk.as_bytes()), disk)
            };
            meta.staged.insert(
                path.clone(),
                StagedTxFile {
                    base_blake3: base_hash,
                    original_content: orig,
                    staged_content: String::new(),
                    created: false,
                    deleted: true,
                },
            );
            if let Err(e) = write_tx(&meta_path, &meta) {
                return e;
            }
            json!({"ok": true, "tx_handle": handle, "deleted": path, "staged_files_count": meta.staged.len()})
        }
        "refit_tx_move" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            let from_p = args["from_path"].as_str().unwrap_or("").replace('\\', "/");
            let to_p = args["to_path"].as_str().unwrap_or("").replace('\\', "/");
            if !check_target(repo_root, &from_p) || !check_target(repo_root, &to_p) {
                return json!({"ok": false, "error": "unsafe from_path or to_path"});
            }
            if from_p == to_p {
                return json!({"ok": false, "error": "from_path and to_path are identical"});
            }
            let lock = lock_tx(handle);
            let _g = lock.lock();
            let (meta_path, mut meta) = match read_tx(repo_root, handle) {
                Ok(v) => v,
                Err(e) => return e,
            };
            let to_deleted_in_tx = meta.staged.get(&to_p).map(|s| s.deleted).unwrap_or(false);
            if repo_root.join(&to_p).exists() && !to_deleted_in_tx {
                return json!({"ok": false, "error": format!("destination already exists on disk: `{to_p}`")});
            }
            let (content, from_was_created, from_orig, from_base_hash) = if let Some(prev) = meta.staged.get(&from_p) {
                if prev.deleted {
                    return json!({"ok": false, "error": format!("source already deleted in tx: `{from_p}`")});
                }
                (
                    prev.staged_content.clone(),
                    prev.created,
                    prev.original_content.clone(),
                    prev.base_blake3.clone(),
                )
            } else {
                let Ok(txt) = fs::read_to_string(repo_root.join(&from_p)) else {
                    return json!({"ok": false, "error": format!("source file not found: `{from_p}`")});
                };
                let h = MvccStore::hash_bytes(txt.as_bytes());
                (txt.clone(), false, txt, h)
            };
            if from_was_created {
                meta.staged.remove(&from_p);
            } else {
                meta.staged.insert(
                    from_p.clone(),
                    StagedTxFile {
                        base_blake3: from_base_hash,
                        original_content: from_orig,
                        staged_content: String::new(),
                        created: false,
                        deleted: true,
                    },
                );
            }
            if let Some(prev_to) = meta.staged.get_mut(&to_p) {
                prev_to.staged_content = content;
                prev_to.deleted = false;
            } else {
                meta.staged.insert(
                    to_p.clone(),
                    StagedTxFile {
                        base_blake3: String::new(),
                        original_content: String::new(),
                        staged_content: content,
                        created: true,
                        deleted: false,
                    },
                );
            }
            if let Err(e) = write_tx(&meta_path, &meta) {
                return e;
            }
            json!({"ok": true, "tx_handle": handle, "moved": [from_p, to_p], "staged_files_count": meta.staged.len()})
        }
        "refit_tx_stage" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            let patch_text = args["patch_text"].as_str().unwrap_or("");
            let lock = lock_tx(handle);
            let _g = lock.lock();
            let (meta_path, mut meta) = match read_tx(repo_root, handle) {
                Ok(v) => v,
                Err(e) => return e,
            };
            let hunks = parse_patches(patch_text);
            if hunks.is_empty() {
                return json!({"ok": false, "error": "no <<<REFIT_PATCH>>> hunks found"});
            }
            for h in &hunks {
                if !check_target(repo_root, &h.path) {
                    return json!({"ok": false, "error": format!("unsafe path outside repo: `{}`", h.path)});
                }
                if let Some(ref np) = h.new_path {
                    if !check_target(repo_root, np) {
                        return json!({"ok": false, "error": format!("unsafe new_path outside repo: `{np}`")});
                    }
                }
            }
            let mut working: BTreeMap<String, String> = BTreeMap::new();
            for h in &hunks {
                if working.contains_key(&h.path) {
                    continue;
                }
                if let Some(prev) = meta.staged.get(&h.path) {
                    if !prev.deleted {
                        working.insert(h.path.clone(), prev.staged_content.clone());
                    }
                } else if h.op != crate::types::PatchOp::Create {
                    let abs = repo_root.join(&h.path);
                    let Ok(disk_txt) = fs::read_to_string(&abs) else {
                        return json!({"ok": false, "error": format!("target file not found: `{}`", h.path)});
                    };
                    working.insert(h.path.clone(), disk_txt);
                }
            }
            let applied = match apply_patches(&working, &hunks) {
                Ok(m) => m,
                Err(e) => return json!({"ok": false, "failed_gate": "asrf_parse", "error": e.to_string()}),
            };
            for del_rel in &applied.deleted {
                if let Some(prev) = meta.staged.get(del_rel) {
                    if prev.created {
                        meta.staged.remove(del_rel);
                        continue;
                    }
                    let preserved_hash = prev.base_blake3.clone();
                    let preserved_orig = prev.original_content.clone();
                    meta.staged.insert(
                        del_rel.clone(),
                        StagedTxFile {
                            base_blake3: preserved_hash,
                            original_content: preserved_orig,
                            staged_content: String::new(),
                            created: false,
                            deleted: true,
                        },
                    );
                } else {
                    let orig = working.get(del_rel).cloned().unwrap_or_default();
                    let hash = MvccStore::hash_bytes(orig.as_bytes());
                    meta.staged.insert(
                        del_rel.clone(),
                        StagedTxFile {
                            base_blake3: hash,
                            original_content: orig,
                            staged_content: String::new(),
                            created: false,
                            deleted: true,
                        },
                    );
                }
            }
            for (rel, next_txt) in applied.updated {
                let is_created = applied.created.contains(&rel);
                if let Some(entry) = meta.staged.get_mut(&rel) {
                    entry.staged_content = next_txt;
                    entry.deleted = false;
                } else if is_created {
                    meta.staged.insert(
                        rel,
                        StagedTxFile {
                            base_blake3: String::new(),
                            original_content: String::new(),
                            staged_content: next_txt,
                            created: true,
                            deleted: false,
                        },
                    );
                } else {
                    let orig = working.get(&rel).cloned().unwrap_or_default();
                    let hash = MvccStore::hash_bytes(orig.as_bytes());
                    meta.staged.insert(
                        rel,
                        StagedTxFile {
                            base_blake3: hash,
                            original_content: orig,
                            staged_content: next_txt,
                            created: false,
                            deleted: false,
                        },
                    );
                }
            }
            if let Err(e) = write_tx(&meta_path, &meta) {
                return e;
            }
            let files: Vec<String> = meta.staged.keys().cloned().collect();
            json!({
                "ok": true,
                "tx_handle": handle,
                "staged_files_count": files.len(),
                "files": files
            })
        }
        "refit_tx_commit" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            let dry_run = args["dry_run"].as_bool().unwrap_or(false);
            let allow_abi_change = args["allow_abi_change"].as_bool().unwrap_or(false);
            if !check_txid(handle) {
                return json!({"passed": false, "error": "invalid tx_handle"});
            }
            let lock = lock_tx(handle);
            let _g = lock.lock();
            let tx_dir = txs_dir(repo_root).join(handle);
            let meta_path = tx_dir.join("meta.json");
            let meta: TxMeta = match fs::read_to_string(&meta_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
            {
                Some(m) => m,
                None => return json!({"passed": false, "error": format!("unknown tx_handle `{handle}`")}),
            };
            if meta.staged.is_empty() {
                return json!({"passed": false, "error": "no staged files in transaction"});
            }

            let mut files = Vec::with_capacity(meta.staged.len());
            let mut base_blake3 = BTreeMap::new();
            let mut abi_hash = BTreeMap::new();
            let mut candidate_map = BTreeMap::new();
            let mut deleted_files = Vec::new();
            let mut loc = 0usize;
            let mut score_sum = 0.0f64;
            let mut scored_n = 0usize;

            for (rel, st) in &meta.staged {
                files.push(rel.clone());
                if st.created {
                    // Enforce CAS-absent check on newly created files so concurrent creators never silently clobber.
                    base_blake3.insert(rel.clone(), String::new());
                } else if !st.base_blake3.is_empty() {
                    base_blake3.insert(rel.clone(), st.base_blake3.clone());
                }
                if st.deleted {
                    deleted_files.push(rel.clone());
                } else {
                    if !st.created {
                        abi_hash.insert(rel.clone(), hash_abi(rel, &st.original_content));
                        loc += st.original_content.lines().count();
                        score_sum += hygiene.score_content(rel, &st.original_content).score;
                        scored_n += 1;
                    }
                    candidate_map.insert(rel.clone(), st.staged_content.clone());
                }
            }
            let affected_tests = scan_cached(repo_root, true, true, hygiene)
                .map(|(b, _)| impacted_tests(&b.plan, &files))
                .unwrap_or_default();
            let unit = ModuleUnit {
                id: handle.to_string(),
                files,
                layer: 0,
                wave: 0,
                deps: Vec::new(),
                dependents: Vec::new(),
                test_files: affected_tests,
                loc,
                base_blake3: base_blake3.clone(),
                abi_hash,
                baseline_hygiene: if scored_n > 0 { score_sum / (scored_n as f64) } else { 0.85 },
                allow_abi_change,
            };

            let verdict = match run_gates(
                gate_cfg,
                &unit,
                &candidate_map,
                &deleted_files,
                hygiene,
            ) {
                Ok(v) => v,
                Err(e) => return json!({"passed": false, "error": e.to_string()}),
            };
            if !verdict.passed {
                return serde_json::to_value(&verdict).unwrap_or_else(|_| json!({"passed": false}));
            }
            if dry_run {
                return json!({
                    "passed": true,
                    "dry_run": true,
                    "committed": false,
                    "tx_handle": handle,
                    "staged_files_count": meta.staged.len(),
                    "deleted_files_count": deleted_files.len(),
                    "gate0_syntax_ok": verdict.gate0_syntax_ok,
                    "gate0_abi_ok": verdict.gate0_abi_ok,
                    "gate2_score_before": verdict.gate2_score_before,
                    "gate2_score_after": verdict.gate2_score_after,
                    "diagnostics": verdict.diagnostics
                });
            }
            match MvccStore::open(repo_root, "mcp-tx").and_then(|s| {
                s.commit_tx(
                    handle,
                    &base_blake3,
                    &verdict.final_contents,
                    &verdict.deleted_files,
                )
            }) {
                Ok(CommitOutcome::Committed {
                    head_rev,
                    tx_id,
                    new_hashes,
                }) => {
                    let _ = fs::remove_dir_all(&tx_dir);
                    json!({
                        "passed": true,
                        "committed": true,
                        "tx_handle": handle,
                        "head_rev": head_rev,
                        "tx_id": tx_id,
                        "files_committed": new_hashes.len(),
                        "files_deleted": verdict.deleted_files.len(),
                        "gate2_score_after": verdict.gate2_score_after
                    })
                }
                Ok(CommitOutcome::CasConflict {
                    path,
                    expected,
                    actual,
                }) => json!({
                    "passed": false,
                    "failed_gate": "mvcc_cas_conflict",
                    "path": path,
                    "expected": expected,
                    "actual": actual
                }),
                Err(e) => json!({"passed": false, "error": e.to_string()}),
            }
        }
        "refit_tx_abort" => {
            let handle = args["tx_handle"].as_str().unwrap_or("");
            if !check_txid(handle) {
                return json!({"rolled_back": false, "error": "invalid tx_handle"});
            }
            let tx_dir = txs_dir(repo_root).join(handle);
            let existed = tx_dir.exists();
            let _ = fs::remove_dir_all(&tx_dir);
            json!({
                "rolled_back": existed,
                "tx_handle": handle
            })
        }
        "refit_apply" => {
            let module_id = args["module_id"].as_str().unwrap_or("");
            let patch_text = args["patch_text"].as_str().unwrap_or("");
            let dry_run = args["dry_run"].as_bool().unwrap_or(false);
            let allow_abi_override = args["allow_abi_change"].as_bool();
            let (bundle, _) = match scan_cached(repo_root, true, true, hygiene) {
                Ok(b) => b,
                Err(e) => return json!({"error": e.to_string()}),
            };
            let Some(mut unit) = bundle.plan.units.into_iter().find(|u| u.id == module_id) else {
                return json!({"error": format!("unknown module_id `{module_id}`")});
            };
            if let Some(a) = allow_abi_override {
                unit.allow_abi_change = a;
            }
            let mut base_texts = BTreeMap::new();
            let mut live_base_hashes = BTreeMap::new();
            for rel in &unit.files {
                let txt = fs::read_to_string(repo_root.join(rel)).unwrap_or_default();
                live_base_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
                base_texts.insert(rel.clone(), txt);
            }
            let hunks = parse_patches(patch_text);
            for h in &hunks {
                if !check_target(repo_root, &h.path) {
                    return json!({"passed": false, "failed_gate": "asrf_parse", "error": format!("unsafe path `{}`", h.path)});
                }
                if let Some(ref np) = h.new_path {
                    if !check_target(repo_root, np) {
                        return json!({"passed": false, "failed_gate": "asrf_parse", "error": format!("unsafe new_path `{np}`")});
                    }
                }
            }
            let applied = match apply_patches(&base_texts, &hunks) {
                Ok(c) => c,
                Err(e) => return json!({"passed": false, "failed_gate": "asrf_parse", "error": e.to_string()}),
            };
            for created_rel in &applied.created {
                live_base_hashes.entry(created_rel.clone()).or_default();
            }
            let verdict = match run_gates(
                gate_cfg,
                &unit,
                &applied.updated,
                &applied.deleted,
                hygiene,
            ) {
                Ok(v) => v,
                Err(e) => return json!({"passed": false, "error": e.to_string()}),
            };
            if !verdict.passed {
                return serde_json::to_value(&verdict).unwrap_or_else(|_| json!({"passed": false}));
            }
            if dry_run {
                return json!({
                    "passed": true,
                    "dry_run": true,
                    "committed": false,
                    "gate0_syntax_ok": verdict.gate0_syntax_ok,
                    "gate0_abi_ok": verdict.gate0_abi_ok,
                    "gate2_score_before": verdict.gate2_score_before,
                    "gate2_score_after": verdict.gate2_score_after,
                    "gate3_ruff_errors": verdict.gate3_ruff_errors,
                    "gate3_ty_errors": verdict.gate3_ty_errors,
                    "diagnostics": verdict.diagnostics
                });
            }
            match MvccStore::open(repo_root, "mcp-apply").and_then(|s| {
                s.commit_tx(
                    &unit.id,
                    &live_base_hashes,
                    &verdict.final_contents,
                    &verdict.deleted_files,
                )
            }) {
                Ok(CommitOutcome::Committed {
                    head_rev,
                    tx_id,
                    new_hashes,
                }) => json!({
                    "passed": true,
                    "committed": true,
                    "head_rev": head_rev,
                    "tx_id": tx_id,
                    "files_committed": new_hashes.len(),
                    "gate2_score_after": verdict.gate2_score_after,
                    "gate3_ruff_errors": verdict.gate3_ruff_errors,
                    "gate3_ty_errors": verdict.gate3_ty_errors
                }),
                Ok(CommitOutcome::CasConflict {
                    path,
                    expected,
                    actual,
                }) => json!({
                    "passed": false,
                    "failed_gate": "mvcc_cas_conflict",
                    "path": path,
                    "expected": expected,
                    "actual": actual
                }),
                Err(e) => json!({"passed": false, "error": e.to_string()}),
            }
        }
        other => json!({"error": format!("unknown tool `{other}`")}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_55_mcp_dry_run() {
        let dir = std::env::temp_dir().join(format!("refit_mcp55_{}_{}", std::process::id(), now_ms()));
        fs::create_dir_all(dir.join("src")).unwrap();
        let orig = "export function add(a: number, b: number): number {\n  return a + b;\n}\n";
        fs::write(dir.join("src/calc.ts"), orig).unwrap();

        let hygiene = HygieneScorer::new();
        let gate_cfg = read_gates(&dir);
        let research = WebClient::for_repo(&dir, None);

        let s1 = dispatch_tool("refit_scan", &json!({}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(s1["scan_cache_hit"], json!(false));
        let s2 = dispatch_tool("refit_scan", &json!({}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(s2["scan_cache_hit"], json!(true));

        let patch = "<<<REFIT_PATCH path=\"src/calc.ts\">>>\n<<<SEARCH>>>\n  return a + b;\n<<<REPLACE>>>\n  const total = a + b;\n  return total;\n<<<END>>>";
        let dry = dispatch_tool(
            "refit_apply",
            &json!({"module_id": "calc", "patch_text": patch, "dry_run": true}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        assert_eq!(dry["passed"], json!(true));
        assert_eq!(dry["dry_run"], json!(true));
        assert_eq!(dry["committed"], json!(false));
        assert_eq!(fs::read_to_string(dir.join("src/calc.ts")).unwrap(), orig);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_60_mcp_tx_commit() {
        let parent = std::env::temp_dir().join(format!("refit_mcp60_{}_{}", std::process::id(), now_ms()));
        let dir = parent.join("repo");
        fs::create_dir_all(dir.join("src")).unwrap();
        let a0 = "export function alpha(x: number): number {\n  return x + 1;\n}\n";
        let b0 = "export function beta(y: number): number {\n  return y + 2;\n}\n";
        fs::write(dir.join("src/a.ts"), a0).unwrap();
        fs::write(dir.join("src/b.ts"), b0).unwrap();

        let hygiene = HygieneScorer::new();
        let gate_cfg = read_gates(&dir);
        let research = WebClient::for_repo(&dir, None);

        let began = dispatch_tool("refit_tx_begin", &json!({}), &dir, &hygiene, &gate_cfg, &research);
        let h = began["tx_handle"].as_str().unwrap().to_string();

        let p_a = "<<<REFIT_PATCH path=\"src/a.ts\">>>\n<<<SEARCH>>>\n  return x + 1;\n<<<REPLACE>>>\n  const next = x + 10;\n  return next;\n<<<END>>>";
        let p_b = "<<<REFIT_PATCH path=\"src/b.ts\">>>\n<<<SEARCH>>>\n  return y + 2;\n<<<REPLACE>>>\n  const next = y + 20;\n  return next;\n<<<END>>>";
        assert_eq!(
            dispatch_tool("refit_tx_stage", &json!({"tx_handle": h, "patch_text": p_a}), &dir, &hygiene, &gate_cfg, &research)["staged_files_count"],
            json!(1)
        );
        assert_eq!(
            dispatch_tool("refit_tx_stage", &json!({"tx_handle": h, "patch_text": p_b}), &dir, &hygiene, &gate_cfg, &research)["staged_files_count"],
            json!(2)
        );

        let dry = dispatch_tool("refit_tx_commit", &json!({"tx_handle": h, "dry_run": true}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(dry["passed"], json!(true));
        assert_eq!(dry["committed"], json!(false));
        assert_eq!(fs::read_to_string(dir.join("src/a.ts")).unwrap(), a0);

        let committed = dispatch_tool("refit_tx_commit", &json!({"tx_handle": h}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(committed["committed"], json!(true));
        assert_eq!(committed["files_committed"], json!(2));
        assert!(fs::read_to_string(dir.join("src/a.ts")).unwrap().contains("x + 10"));
        assert!(fs::read_to_string(dir.join("src/b.ts")).unwrap().contains("y + 20"));

        let began2 = dispatch_tool("refit_tx_begin", &json!({}), &dir, &hygiene, &gate_cfg, &research);
        let h2 = began2["tx_handle"].as_str().unwrap().to_string();
        let bad_b = "<<<REFIT_PATCH path=\"src/b.ts\">>>\n<<<SEARCH>>>\n  return next;\n}\n<<<REPLACE>>>\n  return next;\n<<<END>>>";
        dispatch_tool("refit_tx_stage", &json!({"tx_handle": h2, "patch_text": bad_b}), &dir, &hygiene, &gate_cfg, &research);
        let failed = dispatch_tool("refit_tx_commit", &json!({"tx_handle": h2}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(failed["passed"], json!(false));
        assert_eq!(failed["failed_gate"], json!("gate0_syntax"));
        let rb = dispatch_tool("refit_tx_abort", &json!({"tx_handle": h2}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(rb["rolled_back"], json!(true));
        assert!(fs::read_to_string(dir.join("src/b.ts")).unwrap().contains("return next;\n}"));
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn probe_61_mcp_tx_gc() {
        let parent = std::env::temp_dir().join(format!("refit_mcp61_{}_{}", std::process::id(), now_ms()));
        let dir = parent.join("repo");
        fs::create_dir_all(&dir).unwrap();

        let tx_root = txs_dir(&dir);
        let stale = tx_root.join("tx-stale-orphan");
        let fresh = tx_root.join("tx-fresh-active");
        fs::create_dir_all(&stale).unwrap();
        fs::create_dir_all(&fresh).unwrap();
        let now = now_ms();
        fs::write(
            stale.join("meta.json"),
            serde_json::to_string(&TxMeta {
                tx_handle: "tx-stale-orphan".into(),
                created_ms: now.saturating_sub(120_000),
                ttl_ms: 1_000,
                staged: BTreeMap::new(),
            })
            .unwrap(),
        )
        .unwrap();
        fs::write(
            fresh.join("meta.json"),
            serde_json::to_string(&TxMeta {
                tx_handle: "tx-fresh-active".into(),
                created_ms: now,
                ttl_ms: 600_000,
                staged: BTreeMap::new(),
            })
            .unwrap(),
        )
        .unwrap();

        let res = dispatch_tool(
            "refit_tx_begin",
            &json!({}),
            &dir,
            &HygieneScorer::new(),
            &read_gates(&dir),
            &WebClient::for_repo(&dir, None),
        );
        assert_eq!(res["swept_orphans"], json!(1));
        assert!(!stale.exists());
        assert!(fresh.exists());
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn probe_74_mcp_tx_cas() {
        let parent = std::env::temp_dir().join(format!("refit_mcp74_{}_{}", std::process::id(), now_ms()));
        let dir = parent.join("repo");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/item.ts"), "export function item(x: number): number {\n  return x + 1;\n}\n").unwrap();

        let hygiene = HygieneScorer::new();
        let gate_cfg = read_gates(&dir);
        let research = WebClient::for_repo(&dir, None);
        let _ = dispatch_tool("refit_scan", &json!({}), &dir, &hygiene, &gate_cfg, &research);
        fs::write(dir.join("src/item.ts"), "export function item(x: number): number {\n  return x + 5;\n}\n").unwrap();

        let patch = "<<<REFIT_PATCH path=\"src/item.ts\">>>\n<<<SEARCH>>>\n  return x + 5;\n<<<REPLACE>>>\n  const v = x + 6;\n  return v;\n<<<END>>>";
        let res = dispatch_tool(
            "refit_apply",
            &json!({"module_id": "item", "patch_text": patch}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        assert_eq!(res["committed"], json!(true));
        assert!(fs::read_to_string(dir.join("src/item.ts")).unwrap().contains("x + 6"));
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn probe_87_mcp_tx_ops() {
        let parent = std::env::temp_dir().join(format!("refit_mcp87_{}_{}", std::process::id(), now_ms()));
        let dir = parent.join("repo");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/legacy.ts"), "export function oldFn(x: number): number {\n  return x;\n}\n").unwrap();

        let hygiene = HygieneScorer::new();
        let gate_cfg = read_gates(&dir);
        let research = WebClient::for_repo(&dir, None);

        let sl = dispatch_tool(
            "refit_read",
            &json!({"path": "src/legacy.ts", "start_line": 1, "end_line": 2}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        assert!(sl["content"].as_str().unwrap().contains("1: export function oldFn"));

        let h1 = dispatch_tool("refit_tx_begin", &json!({}), &dir, &hygiene, &gate_cfg, &research)["tx_handle"]
            .as_str()
            .unwrap()
            .to_string();
        dispatch_tool(
            "refit_tx_move",
            &json!({"tx_handle": h1, "from_path": "src/legacy.ts", "to_path": "src/core/modern.ts"}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        let c1 = dispatch_tool(
            "refit_tx_commit",
            &json!({"tx_handle": h1, "allow_abi_change": true}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        assert_eq!(c1["committed"], json!(true));
        assert_eq!(c1["head_rev"], json!(1));
        assert!(!dir.join("src/legacy.ts").exists());
        assert!(dir.join("src/core/modern.ts").exists());

        let h2 = dispatch_tool("refit_tx_begin", &json!({}), &dir, &hygiene, &gate_cfg, &research)["tx_handle"]
            .as_str()
            .unwrap()
            .to_string();
        dispatch_tool(
            "refit_tx_create",
            &json!({"tx_handle": h2, "path": "src/core/helper.ts", "content": "export function help(n: number): number {\n  return n + 1;\n}\n"}),
            &dir,
            &hygiene,
            &gate_cfg,
            &research,
        );
        let c2 = dispatch_tool("refit_tx_commit", &json!({"tx_handle": h2}), &dir, &hygiene, &gate_cfg, &research);
        assert_eq!(c2["committed"], json!(true));
        assert_eq!(c2["head_rev"], json!(2));
        let _ = fs::remove_dir_all(&parent);
    }
}
