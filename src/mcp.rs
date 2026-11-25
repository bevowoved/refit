//! Protocol-agnostic MCP engine supporting stdio and stateless Streamable HTTP.

mod tx;

use crate::gates::{GateConfig, read_gates, scan_cached};
use crate::graph::{apply_patches, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::net::{json_str, rpc_err, rpc_ok, rpc_text};
use crate::platform::{
    EpochRegistry, MemGate, ensure_cachedir_tag, file_mtime_size, now_ms, write_atomic,
};
use crate::research::WebClient;
use crate::stem::{SymbolGraph, load_symbols, query_symbol_graph, query_symbols};
use crate::untangle::{render_audit, untangle_repo, write_sustain};
use anyhow::{Context, Result};
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tx::{
    CommitReq, ToolCtx, check_hunks, check_target, check_txid, tx_abort, tx_begin, tx_commit,
    tx_create, tx_delete, tx_move, tx_stage, verify_commit,
};

const PROTO_STATELESS: &str = "2026-07-28";
const PROTO_BASE: &str = "2024-11-05";
const ERR_PARSE: i64 = -32700;
const ERR_INVALID_REQUEST: i64 = -32600;
const ERR_METHOD_NOT_FOUND: i64 = -32601;
const ERR_INVALID_PARAMS: i64 = -32602;
const ERR_HEADER_MISMATCH: i64 = -32020;
const ERR_OVERLOADED: i64 = -32001;

const CATALOG_TTL_MS: u64 = 3_600_000;
const CATALOG_CACHE_SCOPE: &str = "server";
const MRTR_TOKEN_PREFIX: &str = "mrtr1.";
const MRTR_DEFAULT_TTL_MS: u64 = 300_000;
const HEAVY_TOOL_RESERVE_BYTES: usize = 2 * 1024 * 1024;
const LIGHT_TOOL_RESERVE_BYTES: usize = 256 * 1024;
const EPOCH_MEMO_TTL: Duration = Duration::from_secs(5);

static TASK_SEQ: AtomicU64 = AtomicU64::new(1);
static NONCE_SEQ: AtomicU64 = AtomicU64::new(1);

// Tool catalog

type ToolRow = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static [&'static str],
);

/// Tool definitions sorted strictly A->Z by `name` for prompt-cache stability.
const TOOLS: &[ToolRow] = &[
    (
        "refit_apply", "Apply ASRF patches, run Gates 0 to 3.5, and commit via MVCC WAL.",
        &[("allow_abi_change", "boolean"), ("dry_run", "boolean"), ("module_id", "string"), ("patch_text", "string")],
        &["module_id", "patch_text"],
    ),
    (
        "refit_audit", "Audit clones, dead symbols, large files, unused deps, and infra configs.",
        &[("markdown", "boolean"), ("research", "boolean")], &[],
    ),
    (
        "refit_read", "Read a bounded line range from a repository file.",
        &[("end_line", "integer"), ("path", "string"), ("start_line", "integer")], &["path"],
    ),
    (
        "refit_resume", "Recover unfinished WAL transactions from CAS and report committed modules.",
        &[], &[],
    ),
    (
        "refit_rewind", "Query SymbolGraph-backed rewind impact set and learned rules for a symbol token.",
        &[("token", "string"), ("trigger", "string")], &["token"],
    ),
    (
        "refit_scan", "Scan repository DAG and codehygiene scores (fast cached by default).",
        &[("fast", "boolean")], &[],
    ),
    (
        "refit_search", "Run 10 req/s GCRA-paced web search with repo-scoped BLAKE3 cache.",
        &[("capability", "string"), ("language", "string")], &["capability", "language"],
    ),
    (
        "refit_sustain", "Generate AGENTS.md, .refit/invariants.yml, ADR-001, and CI guard.",
        &[("dry_run", "boolean")], &[],
    ),
    (
        "refit_symbols", "Query symbol definitions, callers, dependencies, fan-in/out, and dead symbols.",
        &[("file", "string"), ("symbol", "string")], &[],
    ),
    (
        "refit_tx_abort", "Abort an active shadow transaction and remove its staged files.",
        &[("tx_handle", "string")], &["tx_handle"],
    ),
    (
        "refit_tx_begin", "Start a multi-file shadow transaction outside the repository.",
        &[("ttl_secs", "integer")], &[],
    ),
    (
        "refit_tx_commit", "Verify all staged files with Gates 0 to 3.5, then commit via MVCC WAL.",
        &[("allow_abi_change", "boolean"), ("dry_run", "boolean"), ("tx_handle", "string")],
        &["tx_handle"],
    ),
    (
        "refit_tx_create", "Stage a new file inside an active transaction.",
        &[("content", "string"), ("path", "string"), ("tx_handle", "string")],
        &["content", "path", "tx_handle"],
    ),
    (
        "refit_tx_delete", "Stage file deletion inside an active transaction.",
        &[("path", "string"), ("tx_handle", "string")], &["path", "tx_handle"],
    ),
    (
        "refit_tx_move", "Stage a file move or rename inside an active transaction.",
        &[("from_path", "string"), ("to_path", "string"), ("tx_handle", "string")],
        &["from_path", "to_path", "tx_handle"],
    ),
    (
        "refit_tx_stage", "Stage ASRF patches (edit, create, delete, rename) in an active transaction.",
        &[("patch_text", "string"), ("tx_handle", "string")], &["patch_text", "tx_handle"],
    ),
    (
        "refit_verify", "Run a full repository audit with BLAKE3 content caching.",
        &[], &[],
    ),
];

struct FrozenCatalog { value: Value, bytes: Box<[u8]>, digest: String }

static CATALOG: OnceLock<FrozenCatalog> = OnceLock::new();

fn frozen_catalog() -> &'static FrozenCatalog {
    CATALOG.get_or_init(|| {
        let mut rows: Vec<&ToolRow> = TOOLS.iter().collect();
        rows.sort_by_key(|row| row.0);
        let value = Value::Array(rows.into_iter().map(encode_tool).collect());
        let raw = serde_json::to_vec(&value).unwrap_or_else(|_| b"[]".to_vec());
        let digest = format!("blake3:{}", blake3::hash(&raw).to_hex());
        FrozenCatalog { value, bytes: raw.into_boxed_slice(), digest }
    })
}

fn tool_catalog() -> &'static Value { &frozen_catalog().value }
fn catalog_bytes() -> &'static [u8] { &frozen_catalog().bytes }
fn catalog_digest() -> &'static str { &frozen_catalog().digest }

fn catalog_meta() -> Value {
    json!({
        "cacheScope": CATALOG_CACHE_SCOPE,
        "catalogDigest": catalog_digest(),
        "ttlMs": CATALOG_TTL_MS,
    })
}

fn encode_base_tools(id: Option<Value>) -> String {
    let id_json = serde_json::to_string(&id).unwrap_or_else(|_| "null".into());
    let tools_str = std::str::from_utf8(catalog_bytes()).unwrap_or("[]");
    format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{"tools":{tools_str}}}}}"#)
}

fn encode_tool(&(name, desc, prop_list, req): &ToolRow) -> Value {
    let mut props = Map::with_capacity(prop_list.len());
    for &(key, ty) in prop_list { props.insert(key.to_string(), json!({ "type": ty })); }
    let mut schema = Map::new();
    schema.insert("properties".into(), Value::Object(props));
    if !req.is_empty() {
        let mut req_sorted: Vec<&str> = req.to_vec();
        req_sorted.sort_unstable();
        schema.insert("required".into(), json!(req_sorted));
    }
    schema.insert("type".into(), json!("object"));
    json!({ "description": desc, "inputSchema": Value::Object(schema), "name": name })
}

// Base64URL and resume tokens

const B64U: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() * 4).div_ceil(3));
    for c in data.chunks(3) {
        let b1 = if c.len() > 1 { c[1] as u32 } else { 0 };
        let b2 = if c.len() > 2 { c[2] as u32 } else { 0 };
        let n = ((c[0] as u32) << 16) | (b1 << 8) | b2;
        out.push(B64U[((n >> 18) & 63) as usize] as char);
        out.push(B64U[((n >> 12) & 63) as usize] as char);
        if c.len() > 1 { out.push(B64U[((n >> 6) & 63) as usize] as char); }
        if c.len() > 2 { out.push(B64U[(n & 63) as usize] as char); }
    }
    out
}

fn b64u_val(b: u8) -> Option<u32> {
    match b {
        b'A'..=b'Z' => Some((b - b'A') as u32),
        b'a'..=b'z' => Some((b - b'a' + 26) as u32),
        b'0'..=b'9' => Some((b - b'0' + 52) as u32),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

fn b64url_decode(raw: &str) -> Option<Vec<u8>> {
    let bytes = raw.as_bytes();
    if bytes.len() % 4 == 1 { return None; }
    let mut out = Vec::with_capacity((bytes.len() * 3) / 4);
    for c in bytes.chunks(4) {
        let (v0, v1) = (b64u_val(c[0])?, b64u_val(c[1])?);
        match c.len() {
            2 if v1 & 0x0F == 0 => out.push(((v0 << 2) | (v1 >> 4)) as u8),
            3 => {
                let v2 = b64u_val(c[2])?;
                if v2 & 0x03 != 0 { return None; }
                let n = (v0 << 18) | (v1 << 12) | (v2 << 6);
                out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8]);
            }
            4 => {
                let (v2, v3) = (b64u_val(c[2])?, b64u_val(c[3])?);
                let n = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
                out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8]);
            }
            _ => return None,
        }
    }
    Some(out)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ContinuationPayload {
    tool: String, arguments: Value, prompt: String,
    issued_ms: u64, expires_ms: u64, nonce: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MrtrError { Malformed, InvalidSignature, Expired }

impl std::fmt::Display for MrtrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Malformed => "malformed continuationToken",
            Self::InvalidSignature => "invalid continuationToken signature",
            Self::Expired => "continuationToken expired",
        })
    }
}

#[derive(Clone)]
struct MrtrSigner { key: [u8; 32] }

impl MrtrSigner {
    fn from_seed(seed: &[u8]) -> Self {
        Self { key: blake3::derive_key("refit mcp token key", seed) }
    }

    fn for_repo(repo_root: &Path) -> Self {
        if let Ok(secret) = std::env::var("REFIT_MCP_TOKEN_SECRET")
            && !secret.trim().is_empty()
        {
            return Self::from_seed(secret.trim().as_bytes());
        }
        let key_path = repo_root.join(".refit").join(".mrtr.key");
        if let Ok(bytes) = fs::read(&key_path) && bytes.len() >= 32 {
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes[..32]);
            return Self { key };
        }
        let key = crate::platform::os_random_bytes::<32>().unwrap_or_else(|_| {
            let fb = format!("ephemeral-{}-{}", std::process::id(), now_ms());
            blake3::derive_key("refit mcp token fallback", fb.as_bytes())
        });
        if let Some(parent) = key_path.parent() { let _ = fs::create_dir_all(parent); }
        let _ = write_atomic(&key_path, &key);
        #[cfg(unix)]
        if key_path.exists() {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600));
        }
        Self { key }
    }

    fn sign(&self, payload: &ContinuationPayload) -> Result<String> {
        let raw = serde_json::to_vec(payload).context("encode resume payload")?;
        let mac = blake3::keyed_hash(&self.key, &raw);
        Ok(format!("{MRTR_TOKEN_PREFIX}{}.{}", b64url_encode(&raw), b64url_encode(mac.as_bytes())))
    }

    fn verify(&self, token: &str, now: u64) -> std::result::Result<ContinuationPayload, MrtrError> {
        let rest = token.strip_prefix(MRTR_TOKEN_PREFIX).ok_or(MrtrError::Malformed)?;
        let (payload_b64, sig_b64) = rest.split_once('.').ok_or(MrtrError::Malformed)?;
        let payload_bytes = b64url_decode(payload_b64).ok_or(MrtrError::Malformed)?;
        let sig_bytes = b64url_decode(sig_b64).ok_or(MrtrError::InvalidSignature)?;
        if sig_bytes.len() != 32 { return Err(MrtrError::InvalidSignature); }
        let expected = blake3::keyed_hash(&self.key, &payload_bytes);
        if !constant_time_eq::constant_time_eq(expected.as_bytes(), &sig_bytes) {
            return Err(MrtrError::InvalidSignature);
        }
        let payload: ContinuationPayload =
            serde_json::from_slice(&payload_bytes).map_err(|_| MrtrError::Malformed)?;
        if now > payload.expires_ms { return Err(MrtrError::Expired); }
        Ok(payload)
    }
}

// Task store

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TaskState { Working, InputRequired, Completed, Cancelled, Failed }

impl TaskState {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskRecord {
    #[serde(rename = "taskId")] task_id: String,
    tool: String,
    status: TaskState,
    #[serde(rename = "createdMs")] created_ms: u64,
    #[serde(rename = "updatedMs")] updated_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")] progress: Option<f64>,
    #[serde(default, rename = "statusMessage", skip_serializing_if = "Option::is_none")]
    status_message: Option<String>,
    #[serde(default, rename = "continuationToken", skip_serializing_if = "Option::is_none")]
    continuation_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")] error: Option<String>,
}

struct TaskStore {
    dir: PathBuf,
    state_lock: Mutex<()>,
    cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl TaskStore {
    fn for_repo(repo_root: &Path) -> Self {
        Self {
            dir: repo_root.join(".refit").join("tasks"),
            state_lock: Mutex::new(()),
            cancel_flags: Mutex::new(HashMap::new()),
        }
    }

    fn record_path(&self, task_id: &str) -> Option<PathBuf> {
        check_txid(task_id).then(|| self.dir.join(format!("{task_id}.json")))
    }

    fn create(&self, tool: &str, msg: Option<String>) -> Result<(TaskRecord, Arc<AtomicBool>)> {
        let ts = now_ms();
        let task_id = format!("task-{ts}-{}", TASK_SEQ.fetch_add(1, Ordering::Relaxed));
        let rec = TaskRecord {
            task_id: task_id.clone(), tool: tool.to_string(), status: TaskState::Working,
            created_ms: ts, updated_ms: ts, progress: Some(0.0), status_message: msg,
            continuation_token: None, result: None, error: None,
        };
        self.save(&rec)?;
        let flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags.lock().insert(task_id, flag.clone());
        Ok((rec, flag))
    }

    fn save(&self, rec: &TaskRecord) -> Result<()> {
        let path = self.record_path(&rec.task_id)
            .ok_or_else(|| anyhow::anyhow!("invalid taskId `{}`", rec.task_id))?;
        ensure_cachedir_tag(&self.dir);
        let raw = serde_json::to_vec_pretty(rec).context("encode task record")?;
        write_atomic(&path, &raw)
    }

    fn load(&self, task_id: &str) -> Option<TaskRecord> {
        let raw = fs::read_to_string(self.record_path(task_id)?).ok()?;
        serde_json::from_str(&raw).ok()
    }

    fn clear_flag(&self, task_id: &str) { self.cancel_flags.lock().remove(task_id); }

    fn mutate_task(
        &self, task_id: &str, f: impl FnOnce(&mut TaskRecord) -> std::result::Result<bool, String>,
    ) -> std::result::Result<TaskRecord, String> {
        let _guard = self.state_lock.lock();
        let mut rec = self.load(task_id).ok_or_else(|| format!("unknown taskId `{task_id}`"))?;
        if f(&mut rec)? {
            rec.updated_ms = now_ms();
            self.save(&rec).map_err(|e| e.to_string())?;
        }
        Ok(rec)
    }

    fn cancel(&self, task_id: &str) -> std::result::Result<TaskRecord, String> {
        self.mutate_task(task_id, |rec| {
            if let Some(flag) = self.cancel_flags.lock().remove(task_id) {
                flag.store(true, Ordering::Release);
            }
            if rec.status.is_terminal() { return Ok(false); }
            rec.status = TaskState::Cancelled;
            rec.status_message = Some("cancelled cooperatively via tasks/cancel".into());
            Ok(true)
        })
    }

    fn update(
        &self, task_id: &str, next_status: Option<TaskState>, progress: Option<f64>,
        message: Option<String>, continuation: Option<String>,
    ) -> std::result::Result<TaskRecord, String> {
        self.mutate_task(task_id, |rec| {
            if rec.status.is_terminal() {
                return Err(format!("task `{task_id}` is already in terminal state {:?}", rec.status));
            }
            if let Some(status) = next_status {
                rec.status = status;
                if status.is_terminal() { self.clear_flag(task_id); }
            }
            if let Some(p) = progress { rec.progress = Some(p.clamp(0.0, 1.0)); }
            if let Some(m) = message { rec.status_message = Some(m); }
            if let Some(t) = continuation { rec.continuation_token = Some(t); }
            Ok(true)
        })
    }

    fn finish(&self, task_id: &str, outcome: std::result::Result<Value, String>) {
        let _guard = self.state_lock.lock();
        self.clear_flag(task_id);
        let Some(mut rec) = self.load(task_id) else { return };
        if rec.status.is_terminal() { return; }
        rec.updated_ms = now_ms();
        match outcome {
            Ok(val) => {
                rec.status = TaskState::Completed;
                rec.progress = Some(1.0);
                rec.result = Some(val);
            }
            Err(err) => {
                rec.status = TaskState::Failed;
                rec.error = Some(err);
            }
        }
        let _ = self.save(&rec);
    }
}

// Context and epoch memoization

struct EpochMemo<T> { epoch: u64, stat_sig: u64, stored_at: Instant, value: T }

fn quick_repo_stat_sig(repo_root: &Path) -> u64 {
    let mut h = FxHasher::default();
    for p in [
        repo_root.to_path_buf(), repo_root.join(".git/HEAD"),
        repo_root.join(".git/index"), repo_root.join(".refit/wal.jsonl"),
    ] {
        file_mtime_size(&p).hash(&mut h);
    }
    h.finish()
}

struct McpCtx {
    repo_root: PathBuf,
    hygiene: Arc<HygieneScorer>,
    gate_cfg: GateConfig,
    research: Arc<WebClient>,
    blocking_sem: Arc<tokio::sync::Semaphore>,
    epoch: Arc<EpochRegistry>,
    tasks: Arc<TaskStore>,
    signer: MrtrSigner,
    scan_memo: Mutex<Option<(bool, EpochMemo<Value>)>>,
    symbols_memo: Mutex<Option<EpochMemo<Arc<SymbolGraph>>>>,
    #[cfg(test)]
    test_delay_ms: AtomicU64,
}

impl McpCtx {
    fn new(repo_root: PathBuf) -> Arc<Self> { Self::with_epoch(repo_root, EpochRegistry::global_arc()) }

    fn with_epoch(repo_root: PathBuf, epoch: Arc<EpochRegistry>) -> Arc<Self> {
        let permits = crate::platform::budget().mcp_blocking_permits;
        Arc::new(Self {
            hygiene: Arc::new(HygieneScorer::open_persistent(&repo_root)),
            gate_cfg: read_gates(&repo_root),
            research: WebClient::for_repo(&repo_root, None),
            tasks: Arc::new(TaskStore::for_repo(&repo_root)),
            signer: MrtrSigner::for_repo(&repo_root),
            repo_root,
            blocking_sem: Arc::new(tokio::sync::Semaphore::new(permits)),
            epoch,
            scan_memo: Mutex::new(None),
            symbols_memo: Mutex::new(None),
            #[cfg(test)]
            test_delay_ms: AtomicU64::new(0),
        })
    }

    fn snapshot_epoch(&self, tool: &str, args: &Value) -> (u64, u64) {
        let key = args["path"].as_str().unwrap_or(tool);
        (self.epoch.global_epoch(), self.epoch.slot_epoch(key))
    }
}

// JSON-RPC core

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proto { Base, Stateless }

fn detect_protocol(req: &Value) -> Proto {
    let top_v = req.get("_meta").and_then(|m| m.get("protocolVersion")).and_then(Value::as_str);
    let par_v = req.get("params").and_then(|p| p.get("_meta"))
        .and_then(|m| m.get("protocolVersion")).and_then(Value::as_str);
    let init_v = req.get("params").and_then(|p| p.get("protocolVersion")).and_then(Value::as_str);
    let method = req["method"].as_str().unwrap_or("");
    let is_stateless = top_v == Some(PROTO_STATELESS) || par_v == Some(PROTO_STATELESS)
        || init_v == Some(PROTO_STATELESS)
        || method == "server/discover" || method.starts_with("tasks/");
    if is_stateless { Proto::Stateless } else { Proto::Base }
}

fn batch_err() -> Value {
    let msg = "Invalid Request: JSON-RPC batch arrays are forbidden";
    rpc_err(Value::Null, ERR_INVALID_REQUEST, msg)
}

fn discover_result() -> Value {
    json!({
        "protocolVersion": PROTO_STATELESS,
        "supportedProtocolVersions": [PROTO_STATELESS, PROTO_BASE],
        "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
        "capabilities": {
            "tools": { "listChanged": false },
            "extensions": {
                "io.modelcontextprotocol/tasks": {
                    "methods": ["tasks/get", "tasks/update", "tasks/cancel"]
                },
                "io.modelcontextprotocol/mrtr": { "stateless": true }
            }
        },
        "tools": tool_catalog(),
        "_meta": catalog_meta(),
    })
}

fn eq_param_key(a: &str, b: &str) -> bool {
    let ia = a.bytes().filter(|&c| c != b'_' && c != b'-').map(|c| c.to_ascii_lowercase());
    let ib = b.bytes().filter(|&c| c != b'_' && c != b'-').map(|c| c.to_ascii_lowercase());
    ia.eq(ib)
}

fn check_param_hdrs(headers: &HeaderMap, body: &Value) -> std::result::Result<(), String> {
    let hdr = |k: &str| headers.get(k)?.to_str().ok().map(str::trim);
    let body_method = body.get("method").and_then(Value::as_str)
        .ok_or_else(|| "missing JSON-RPC `method` field".to_string())?;
    let hdr_method = hdr("mcp-method")
        .ok_or_else(|| "missing required `Mcp-Method` header".to_string())?;
    if hdr_method != body_method {
        return Err(format!("Mcp-Method `{hdr_method}` does not match body method `{body_method}`"));
    }
    let body_name = body.get("params").and_then(|p| p.get("name")).and_then(Value::as_str);
    let hdr_name = hdr("mcp-name");
    if body_method == "tools/call" {
        let got = hdr_name.ok_or_else(|| "missing required `Mcp-Name` header for tools/call".to_string())?;
        if got != body_name.unwrap_or("") {
            return Err(format!("Mcp-Name `{got}` does not match body tool `{}`", body_name.unwrap_or("")));
        }
    } else if let Some(got) = hdr_name && body_name != Some(got) {
        return Err(format!("unexpected `Mcp-Name` header `{got}` for `{body_method}`"));
    }

    let args_obj = body.get("params").and_then(|p| p.get("arguments")).and_then(Value::as_object);
    let params_obj = body.get("params").and_then(Value::as_object);
    for (name, raw_val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        let Some(suffix) = lower.strip_prefix("mcp-param-") else { continue };
        let hdr_val = raw_val.to_str().map(str::trim)
            .map_err(|_| format!("non-UTF8 value in header `{lower}`"))?;
        let actual = [args_obj, params_obj]
            .into_iter()
            .flatten()
            .find_map(|m| {
                m.iter()
                    .find(|(k, _)| k.as_str() == suffix || eq_param_key(k.as_str(), suffix))
                    .map(|(_, v)| v)
            })
            .ok_or_else(|| format!("header `{lower}` has no matching body parameter"))?;
        let actual_str = match actual {
            Value::String(t) => t.clone(),
            Value::Bool(f) => f.to_string(),
            Value::Number(n) => n.to_string(),
            _ => return Err(format!("parameter `{suffix}` is not scalar for `{lower}`")),
        };
        if actual_str != hdr_val {
            return Err(format!("header `{lower}` value `{hdr_val}` mismatches body `{actual_str}`"));
        }
    }
    Ok(())
}

fn estimate_tool_bytes(name: &str) -> usize {
    match name {
        "refit_read" | "refit_symbols" | "refit_search" | "refit_tx_abort" => LIGHT_TOOL_RESERVE_BYTES,
        _ => HEAVY_TOOL_RESERVE_BYTES,
    }
}

async fn handle_rpc(ctx: &Arc<McpCtx>, req: &Value, mode: Proto) -> (StatusCode, Option<Value>) {
    if req.is_array() { return (StatusCode::BAD_REQUEST, Some(batch_err())); }
    let id = req.get("id").cloned();
    let method = req["method"].as_str().unwrap_or("");
    match method {
        "server/discover" => (StatusCode::OK, Some(rpc_ok(id, discover_result()))),
        "ping" => (StatusCode::OK, Some(rpc_ok(id, json!({})))),
        "notifications/initialized" => (StatusCode::ACCEPTED, None),
        "initialize" => {
            let res = match mode {
                Proto::Base => json!({
                    "protocolVersion": PROTO_BASE,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
                }),
                Proto::Stateless => json!({
                    "protocolVersion": PROTO_STATELESS,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
                    "_meta": catalog_meta(),
                }),
            };
            (StatusCode::OK, Some(rpc_ok(id, res)))
        }
        "tools/list" => {
            let res = match mode {
                Proto::Base => json!({ "tools": tool_catalog() }),
                Proto::Stateless => json!({ "tools": tool_catalog(), "_meta": catalog_meta() }),
            };
            (StatusCode::OK, Some(rpc_ok(id, res)))
        }
        "tools/call" => handle_tool_call(ctx, id, &req["params"], mode).await,
        "tasks/get" => (StatusCode::OK, Some(handle_tasks_get(ctx, id, &req["params"]))),
        "tasks/update" => (StatusCode::OK, Some(handle_tasks_update(ctx, id, &req["params"]))),
        "tasks/cancel" => (StatusCode::OK, Some(handle_tasks_cancel(ctx, id, &req["params"]))),
        "tasks/list" | "tasks/result" => {
            let msg = format!("Method not found: `{method}` is not supported");
            (StatusCode::OK, Some(rpc_err(id.unwrap_or(Value::Null), ERR_METHOD_NOT_FOUND, msg)))
        }
        other => match id {
            Some(req_id) => {
                let err = rpc_err(req_id, ERR_METHOD_NOT_FOUND, format!("Method not found: {other}"));
                (StatusCode::OK, Some(err))
            }
            None => (StatusCode::ACCEPTED, None),
        },
    }
}

async fn handle_tool_call(
    ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value, mode: Proto,
) -> (StatusCode, Option<Value>) {
    if mode == Proto::Stateless {
        let raw_tok = params.get("continuationToken").or_else(|| params.get("requestState"))
            .and_then(Value::as_str).filter(|s| !s.is_empty());
        if let Some(tok) = raw_tok { return handle_mrtr_resume(ctx, id, params, tok).await; }
    }
    let (name, args) = (json_str(params, "name"), params["arguments"].clone());
    if mode == Proto::Stateless {
        let needs_confirm = args["confirmed"].as_bool() != Some(true)
            && (args["require_confirmation"].as_bool() == Some(true)
                || (name == "refit_apply" && args["allow_abi_change"].as_bool() == Some(true))
                || name == "refit_tx_delete");
        if needs_confirm { return resume_challenge(ctx, id, &name, args); }
        let wants_task = params["task"].as_bool() == Some(true)
            || params["mode"].as_str() == Some("task")
            || args["async_task"].as_bool() == Some(true);
        if wants_task { return spawn_durable_task(ctx, id, name, args); }
    }
    run_tool(ctx, id, name, args, mode).await
}

fn resume_challenge(
    ctx: &Arc<McpCtx>, id: Option<Value>, name: &str, args: Value,
) -> (StatusCode, Option<Value>) {
    let issued = now_ms();
    let payload = ContinuationPayload {
        tool: name.to_string(), arguments: args,
        prompt: format!("Confirm execution of `{name}` with destructive or ABI-changing flags."),
        issued_ms: issued, expires_ms: issued.saturating_add(MRTR_DEFAULT_TTL_MS),
        nonce: NONCE_SEQ.fetch_add(1, Ordering::Relaxed),
    };
    let token = match ctx.signer.sign(&payload) {
        Ok(tok) => tok,
        Err(err) => {
            let e = rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string());
            return (StatusCode::INTERNAL_SERVER_ERROR, Some(e));
        }
    };
    let res = json!({
        "resultType": "input_required", "continuationToken": token, "requestState": token,
        "inputRequest": {
            "prompt": payload.prompt,
            "fields": [{ "name": "confirm", "type": "boolean", "required": true }]
        },
        "content": [{ "type": "text", "text": payload.prompt }],
    });
    (StatusCode::OK, Some(rpc_ok(id, res)))
}

async fn handle_mrtr_resume(
    ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value, token: &str,
) -> (StatusCode, Option<Value>) {
    let payload = match ctx.signer.verify(token, now_ms()) {
        Ok(p) => p,
        Err(err) => {
            let e = rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, err.to_string());
            return (StatusCode::BAD_REQUEST, Some(e));
        }
    };
    if let Some(req_tool) = params.get("name").and_then(Value::as_str) && req_tool != payload.tool {
        let msg = format!(
            "HeaderMismatchError: params.name `{req_tool}` mismatches token tool `{}`",
            payload.tool
        );
        return (
            StatusCode::BAD_REQUEST,
            Some(rpc_err(id.unwrap_or(Value::Null), ERR_HEADER_MISMATCH, msg)),
        );
    }
    let resp = &params["inputResponses"];
    let confirmed = resp.get("confirm").and_then(Value::as_bool).or_else(|| {
        resp.as_array()?.iter()
            .find(|it| it["name"].as_str() == Some("confirm"))
            .and_then(|it| it["value"].as_bool())
    });
    let Some(confirmed) = confirmed else {
        let msg = "missing boolean `confirm` inside `inputResponses`";
        return (
            StatusCode::BAD_REQUEST,
            Some(rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg)),
        );
    };
    if !confirmed {
        let declined = json!({ "cancelled": true, "reason": "user declined confirmation" });
        return (StatusCode::OK, Some(wrap_tool_reply(id, &declined, Proto::Stateless, 0, 0)));
    }
    let mut args = payload.arguments;
    if let Some(obj) = args.as_object_mut() { obj.insert("confirmed".into(), json!(true)); }
    run_tool(ctx, id, payload.tool, args, Proto::Stateless).await
}

async fn execute_tool_async(
    ctx: &Arc<McpCtx>, name: String, args: Value,
    cancel: Option<&AtomicBool>, wait_reserve: Option<usize>,
) -> std::result::Result<Value, String> {
    #[cfg(test)]
    {
        let delay = ctx.test_delay_ms.load(Ordering::Acquire);
        if delay > 0 { tokio::time::sleep(Duration::from_millis(delay)).await; }
    }
    if cancel.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err("cancelled before execution".into());
    }
    if name == "refit_search" {
        let lang = args["language"].as_str().unwrap_or("Rust");
        let cap = args["capability"].as_str().unwrap_or("rate limiter");
        return Ok(ok_json(ctx.research.check_library(lang, cap).await));
    }
    if name == "refit_audit" && args["research"].as_bool().unwrap_or(false) {
        return Ok(run_audit_async(ctx, &args).await);
    }
    let _sem = ctx.blocking_sem.acquire().await.ok();
    if cancel.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err("cancelled before execution".into());
    }
    let cc = ctx.clone();
    tokio::task::spawn_blocking(move || {
        let _wait = wait_reserve
            .and_then(|b| MemGate::global().reserve_or_wait(b, Duration::from_secs(5)).ok());
        dispatch_tool_sync(&name, &args, &cc)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))
}

async fn run_tool(
    ctx: &Arc<McpCtx>, id: Option<Value>, name: String, args: Value, mode: Proto,
) -> (StatusCode, Option<Value>) {
    let gate = MemGate::global();
    let reserve = estimate_tool_bytes(&name).min(gate.capacity());
    let pre_permit = gate.try_admit(reserve).ok();
    if mode == Proto::Stateless && pre_permit.is_none() {
        let e = rpc_err(id.unwrap_or(Value::Null), ERR_OVERLOADED, "server busy: MemGate limit");
        return (StatusCode::SERVICE_UNAVAILABLE, Some(e));
    }
    let wait_reserve = (mode == Proto::Base && pre_permit.is_none()).then_some(reserve);
    let (tool_ep, args_ep) = (name.clone(), args.clone());
    let payload = {
        let _held = pre_permit;
        execute_tool_async(ctx, name, args, None, wait_reserve)
            .await
            .unwrap_or_else(|e| json!({"error": e}))
    };
    let (epoch, slot_epoch) = ctx.snapshot_epoch(&tool_ep, &args_ep);
    (StatusCode::OK, Some(wrap_tool_reply(id, &payload, mode, epoch, slot_epoch)))
}

fn wrap_tool_reply(
    id: Option<Value>, payload: &Value, mode: Proto, epoch: u64, slot_epoch: u64,
) -> Value {
    if mode == Proto::Base { return rpc_text(id, payload); }
    let text = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".into());
    rpc_ok(id, json!({
        "resultType": "complete",
        "content": [{ "type": "text", "text": text }],
        "_meta": { "epoch": epoch, "slotEpoch": slot_epoch, "catalogDigest": catalog_digest() },
    }))
}

fn spawn_durable_task(
    ctx: &Arc<McpCtx>, id: Option<Value>, name: String, args: Value,
) -> (StatusCode, Option<Value>) {
    let (epoch, _) = ctx.snapshot_epoch(&name, &args);
    let gate = MemGate::global();
    let Ok(permit) = gate.try_admit(estimate_tool_bytes(&name).min(gate.capacity())) else {
        let e = rpc_err(id.unwrap_or(Value::Null), ERR_OVERLOADED, "server busy: MemGate denied");
        return (StatusCode::SERVICE_UNAVAILABLE, Some(e));
    };
    let (record, cancel_flag) = match ctx.tasks.create(&name, Some("queued for execution".into())) {
        Ok(pair) => pair,
        Err(err) => {
            drop(permit);
            let e = rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string());
            return (StatusCode::INTERNAL_SERVER_ERROR, Some(e));
        }
    };
    let (tid, cc) = (record.task_id.clone(), ctx.clone());
    tokio::spawn(async move {
        let _held = permit;
        if cancel_flag.load(Ordering::Acquire) { cc.tasks.clear_flag(&tid); return; }
        let outcome = execute_tool_async(&cc, name, args, Some(&cancel_flag), None).await;
        if cancel_flag.load(Ordering::Acquire) { cc.tasks.clear_flag(&tid); return; }
        cc.tasks.finish(&tid, outcome);
    });
    (StatusCode::OK, Some(rpc_ok(id, json!({
        "resultType": "task", "taskId": record.task_id, "status": record.status,
        "task": record, "_meta": { "epoch": epoch },
    }))))
}

fn task_id_param(params: &Value) -> &str {
    params["taskId"]
        .as_str()
        .or_else(|| params["task_id"].as_str())
        .unwrap_or("")
}

fn task_reply(id: Option<Value>, res: std::result::Result<TaskRecord, String>) -> Value {
    match res {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
}

fn handle_tasks_get(ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value) -> Value {
    let tid = task_id_param(params);
    task_reply(id, ctx.tasks.load(tid).ok_or_else(|| format!("unknown taskId `{tid}`")))
}

fn handle_tasks_update(ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value) -> Value {
    let tid = task_id_param(params);
    let next_status = match params.get("status") {
        None => None,
        Some(v) => match serde_json::from_value::<TaskState>(v.clone()) {
            Ok(st) => Some(st),
            Err(e) => return rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, format!("invalid task status: {e}")),
        },
    };
    let progress = params.get("progress").and_then(Value::as_f64);
    let msg = params.get("statusMessage").or_else(|| params.get("status_message"))
        .and_then(Value::as_str).map(String::from);
    let cont = params.get("continuationToken").and_then(Value::as_str).map(String::from);
    task_reply(id, ctx.tasks.update(tid, next_status, progress, msg, cont))
}

fn handle_tasks_cancel(ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value) -> Value {
    task_reply(id, ctx.tasks.cancel(task_id_param(params)))
}

// Transports

/// Run the concurrent JSON-RPC 2.0 MCP server over stdio.
pub async fn serve_mcp(repo_root: PathBuf) -> Result<()> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let (out_tx, mut out_rx) = mpsc::channel::<String>(64);
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(mut line) = out_rx.recv().await {
            line.push('\n');
            if stdout.write_all(line.as_bytes()).await.is_err() || stdout.flush().await.is_err() {
                break;
            }
        }
    });

    let ctx = McpCtx::new(repo_root);
    let mut tasks = tokio::task::JoinSet::new();
    let mut last_scan: Option<oneshot::Receiver<()>> = None;
    while let Some(line) = lines.next_line().await? {
        let Ok(req) = serde_json::from_str::<Value>(line.trim()) else { continue };
        dispatch_stdio(&req, &ctx, &out_tx, &mut tasks, &mut last_scan).await?;
    }
    while tasks.join_next().await.is_some() {}
    drop(out_tx);
    drop(ctx);
    let _ = writer.await;
    Ok(())
}

async fn dispatch_stdio(
    req: &Value, ctx: &Arc<McpCtx>, out_tx: &mpsc::Sender<String>,
    tasks: &mut tokio::task::JoinSet<()>, last_scan: &mut Option<oneshot::Receiver<()>>,
) -> Result<()> {
    if req.is_array() {
        let _ = out_tx.send(serde_json::to_string(&batch_err())?).await;
        return Ok(());
    }
    let mode = detect_protocol(req);
    let method = req["method"].as_str().unwrap_or("");
    if method == "tools/list" && mode == Proto::Base {
        let _ = out_tx.send(encode_base_tools(req.get("id").cloned())).await;
        return Ok(());
    }
    if method == "tools/call" {
        let is_scan = req["params"]["name"].as_str() == Some("refit_scan");
        let wait_prev = if is_scan { last_scan.take() } else { None };
        let done_tx = if is_scan {
            let (tx, rx) = oneshot::channel();
            *last_scan = Some(rx);
            Some(tx)
        } else { None };
        let (cc, tx, owned) = (ctx.clone(), out_tx.clone(), req.clone());
        tasks.spawn(async move {
            if let Some(prev) = wait_prev { let _ = prev.await; }
            let (_, resp_opt) = handle_rpc(&cc, &owned, mode).await;
            if let Some(done) = done_tx { let _ = done.send(()); }
            if let Some(resp) = resp_opt && let Ok(enc) = serde_json::to_string(&resp) {
                let _ = tx.send(enc).await;
            }
        });
        return Ok(());
    }
    if let (_, Some(resp)) = handle_rpc(ctx, req, mode).await {
        let _ = out_tx.send(serde_json::to_string(&resp)?).await;
    }
    Ok(())
}

fn json_response(status: StatusCode, body: Value, retry_after: bool) -> Response {
    let raw = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = (status, raw).into_response();
    resp.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    if retry_after { resp.headers_mut().insert("retry-after", HeaderValue::from_static("1")); }
    resp
}

async fn handle_health() -> Response {
    let body = json!({
        "ok": true, "protocolVersion": PROTO_STATELESS,
        "stateless": true, "catalogDigest": catalog_digest(),
    });
    json_response(StatusCode::OK, body, false)
}

async fn handle_http_mcp(
    State(ctx): State<Arc<McpCtx>>, headers: HeaderMap, body: Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            let e = rpc_err(Value::Null, ERR_PARSE, format!("Parse error: {err}"));
            return json_response(StatusCode::BAD_REQUEST, e, false);
        }
    };
    if parsed.is_array() {
        return json_response(StatusCode::BAD_REQUEST, batch_err(), false);
    }
    if let Err(detail) = check_param_hdrs(&headers, &parsed) {
        let err_body = json!({
            "jsonrpc": "2.0", "id": parsed.get("id").cloned().unwrap_or(Value::Null),
            "error": {
                "code": ERR_HEADER_MISMATCH,
                "message": format!("HeaderMismatchError: {detail}"),
                "data": { "error": "HeaderMismatchError" }
            }
        });
        return json_response(StatusCode::BAD_REQUEST, err_body, false);
    }
    let (status, resp_opt) = handle_rpc(&ctx, &parsed, Proto::Stateless).await;
    match resp_opt {
        Some(val) => json_response(status, val, status == StatusCode::SERVICE_UNAVAILABLE),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

fn build_http_router(ctx: Arc<McpCtx>) -> Router {
    Router::new()
        .route("/mcp", post(handle_http_mcp))
        .route("/health", get(handle_health))
        .with_state(ctx)
}

async fn bind_http_server(
    ctx: Arc<McpCtx>, listen_addr: &str,
) -> Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let app = build_http_router(ctx);
    let listener = TcpListener::bind(listen_addr).await
        .with_context(|| format!("bind MCP HTTP server on {listen_addr}"))?;
    let bound = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            tracing::error!("MCP HTTP server stopped: {err}");
        }
    });
    Ok((bound, handle))
}

/// Serve stateless MCP over Streamable HTTP (`POST /mcp`, `GET /health`).
pub async fn serve_http(repo_root: PathBuf, listen_addr: String) -> Result<()> {
    let (bound, handle) = bind_http_server(McpCtx::new(repo_root), &listen_addr).await?;
    tracing::info!("refit MCP stateless HTTP listening on http://{bound}/mcp");
    handle.await.context("MCP HTTP server join")?;
    Ok(())
}

// Tool handlers

fn ok_json<T: serde::Serialize>(res: Result<T>) -> Value {
    match res {
        Ok(val) => serde_json::to_value(val).unwrap_or_else(|_| json!({"ok": true})),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn dispatch_tool_sync(name: &str, args: &Value, ctx: &McpCtx) -> Value {
    let repo_root = ctx.repo_root.as_path();
    let tctx = ToolCtx { repo_root, hygiene: &ctx.hygiene, gate_cfg: &ctx.gate_cfg };
    match name {
        "refit_scan" => run_scan(ctx, args),
        "refit_audit" => run_audit(repo_root, args, &ctx.hygiene),
        "refit_sustain" => run_sustain(repo_root, args, &ctx.hygiene),
        "refit_symbols" => run_symbols(ctx, args),
        "refit_read" => run_read(ctx, args),
        "refit_resume" => run_resume(repo_root),
        "refit_rewind" => run_rewind(ctx, args),
        "refit_verify" => run_verify(repo_root, &ctx.hygiene),
        "refit_tx_begin" => tx_begin(repo_root, args),
        "refit_tx_create" => tx_create(repo_root, args),
        "refit_tx_delete" => tx_delete(repo_root, args),
        "refit_tx_move" => tx_move(repo_root, args),
        "refit_tx_stage" => tx_stage(repo_root, args),
        "refit_tx_commit" => tx_commit(&tctx, args),
        "refit_tx_abort" => tx_abort(repo_root, args),
        "refit_apply" => run_apply(&tctx, args),
        other => json!({"error": format!("unknown tool `{other}`")}),
    }
}

fn attach_epoch_meta(val: &mut Value, hit: bool, epoch: u64) {
    if let Some(obj) = val.as_object_mut() {
        obj.insert("epoch_cache_hit".into(), json!(hit));
        obj.insert("epoch".into(), json!(epoch));
    }
}

fn run_scan(ctx: &McpCtx, args: &Value) -> Value {
    let fast = args["fast"].as_bool().unwrap_or(true);
    let (ep, sig) = (ctx.epoch.global_epoch(), quick_repo_stat_sig(&ctx.repo_root));
    if let Some((cached_fast, ref memo)) = *ctx.scan_memo.lock()
        && cached_fast == fast && memo.epoch == ep
        && memo.stat_sig == sig && memo.stored_at.elapsed() < EPOCH_MEMO_TTL
    {
        let mut v = memo.value.clone();
        if let Some(obj) = v.as_object_mut() { obj.insert("scan_cache_hit".into(), json!(true)); }
        attach_epoch_meta(&mut v, true, ep);
        return v;
    }

    match scan_cached(&ctx.repo_root, true, fast, &ctx.hygiene) {
        Ok((bundle, hit)) => {
            let ep_after = ctx.epoch.global_epoch();
            let mut v = serde_json::to_value(&bundle.plan).unwrap_or_else(|_| json!({"ok": true}));
            if let Some(obj) = v.as_object_mut() {
                obj.insert("scan_cache_hit".into(), json!(hit));
                obj.insert("fast_mode".into(), json!(fast));
                obj.insert("detected_languages".into(), json!(bundle.detected_languages));
                obj.insert("polyglot_toolchains".into(), json!(bundle.polyglot_toolchains));
            }
            attach_epoch_meta(&mut v, false, ep);
            if ep_after == ep {
                let sig_after = quick_repo_stat_sig(&ctx.repo_root);
                *ctx.scan_memo.lock() = Some((fast, EpochMemo {
                    epoch: ep, stat_sig: sig_after, stored_at: Instant::now(), value: v.clone(),
                }));
            }
            v
        }
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn peek_symbols_memo(ctx: &McpCtx, ep: u64, sig: u64) -> Option<Arc<SymbolGraph>> {
    ctx.symbols_memo.lock().as_ref()
        .filter(|m| m.epoch == ep && m.stat_sig == sig && m.stored_at.elapsed() < EPOCH_MEMO_TTL)
        .map(|m| m.value.clone())
}

fn cached_symbols(ctx: &McpCtx) -> (Arc<SymbolGraph>, u64, bool) {
    let (ep, sig) = (ctx.epoch.global_epoch(), quick_repo_stat_sig(&ctx.repo_root));
    if let Some(snap) = peek_symbols_memo(ctx, ep, sig) { return (snap, ep, true); }
    let snap = Arc::new(load_symbols(&ctx.repo_root, None));
    if ctx.epoch.global_epoch() == ep {
        let sig_now = quick_repo_stat_sig(&ctx.repo_root);
        *ctx.symbols_memo.lock() = Some(EpochMemo {
            epoch: ep, stat_sig: sig_now, stored_at: Instant::now(), value: snap.clone(),
        });
    }
    (snap, ep, false)
}

fn run_symbols(ctx: &McpCtx, args: &Value) -> Value {
    let (sym, file) = (args["symbol"].as_str(), args["file"].as_str());
    let (ep, sig) = (ctx.epoch.global_epoch(), quick_repo_stat_sig(&ctx.repo_root));
    let (mut v, hit, epoch) = if let Some(snap) = peek_symbols_memo(ctx, ep, sig) {
        (query_symbol_graph(&snap, sym, file), true, ep)
    } else if sym.is_none() && file.is_some() && ctx.repo_root.join(".refit/cache/sym.sst").is_file() {
        (query_symbols(&ctx.repo_root, None, file), false, ep)
    } else {
        let (snap, ep_loaded, hit_loaded) = cached_symbols(ctx);
        (query_symbol_graph(&snap, sym, file), hit_loaded, ep_loaded)
    };
    attach_epoch_meta(&mut v, hit, epoch);
    v
}

fn run_rewind(ctx: &McpCtx, args: &Value) -> Value {
    let token = args["token"].as_str();
    let trigger = crate::learn::RewindTrigger::parse(args["trigger"].as_str().unwrap_or("smell"));
    let (snap, ep, hit) = cached_symbols(ctx);
    let mut val = ok_json(crate::learn::query_rewind_graph(
        &ctx.repo_root, Some(&snap), token, trigger,
    ));
    attach_epoch_meta(&mut val, hit, ep);
    val
}

fn format_audit_value(rep: crate::types::AuditReport, args: &Value) -> Value {
    if args["markdown"].as_bool().unwrap_or(false) {
        json!({
            "markdown": render_audit(&rep),
            "summary": {
                "clones": rep.clone_clusters.len(), "dead_symbols": rep.dead_symbols.len(),
                "dep_advisories": rep.dep_advisories.len(), "god_files": rep.god_files.len(),
                "infra_findings": rep.infra_findings.len(), "unused_deps": rep.unused_deps.len()
            }
        })
    } else {
        ok_json(Ok(rep))
    }
}

async fn run_audit_async(ctx: &McpCtx, args: &Value) -> Value {
    let (repo_root, hygiene) = (ctx.repo_root.clone(), ctx.hygiene.clone());
    let _sem_permit = ctx.blocking_sem.acquire().await.ok();
    let base = tokio::task::spawn_blocking(move || untangle_repo(&repo_root, &hygiene))
        .await
        .map_err(|e| anyhow::anyhow!("mcp audit join error: {e}"))
        .and_then(|r| r);
    let mut rep = match base {
        Ok(r) => r,
        Err(e) => return json!({"error": e.to_string()}),
    };
    let (advisories, _) = ctx.research.audit_deps_full(&ctx.repo_root).await;
    rep.dep_advisories = advisories;
    format_audit_value(rep, args)
}

fn run_audit(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    match untangle_repo(repo_root, hygiene) {
        Ok(mut rep) => {
            if args["research"].as_bool().unwrap_or(false) {
                let (advisories, _) = crate::research::deps::audit_lockfiles(
                    repo_root, &rustc_hash::FxHashMap::default(),
                );
                rep.dep_advisories = advisories;
            }
            format_audit_value(rep, args)
        }
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_sustain(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    let dry = args["dry_run"].as_bool().unwrap_or(false);
    ok_json(untangle_repo(repo_root, hygiene).and_then(|r| write_sustain(repo_root, &r, dry)))
}

fn run_read(ctx: &McpCtx, args: &Value) -> Value {
    use std::io::BufRead;

    let repo_root = ctx.repo_root.as_path();
    let rel = args["path"].as_str().unwrap_or("");
    if !check_target(repo_root, rel) { return json!({"error": format!("unsafe path `{rel}`")}); }
    let (epoch, file_epoch) = (ctx.epoch.global_epoch(), ctx.epoch.slot_epoch(rel));
    let Ok(file) = fs::File::open(repo_root.join(rel)) else {
        return json!({"error": format!("file not found `{rel}`")});
    };

    let raw_start = args["start_line"].as_u64().unwrap_or(1);
    let start = raw_start.max(1) as usize;
    let req_end = args["end_line"].as_u64().map(|e| e as usize)
        .unwrap_or_else(|| start.saturating_add(199)).min(start.saturating_add(400));

    let (mut reader, mut buf, mut total, mut kept) =
        (std::io::BufReader::new(file), String::new(), 0usize, Vec::new());
    loop {
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                total += 1;
                if total >= start && total <= req_end {
                    let trimmed = buf.trim_end_matches(['\n', '\r']);
                    kept.push(format!("{total}: {trimmed}"));
                }
            }
        }
    }

    let end = req_end.min(total);
    let slice = if start <= end { kept.join("\n") } else { String::new() };
    json!({
        "path": rel, "start_line": start, "end_line": end,
        "total_lines": total, "content": slice, "epoch": epoch, "file_epoch": file_epoch
    })
}

fn run_resume(repo_root: &Path) -> Value {
    match MvccStore::open(repo_root, "mcp-resume").and_then(|s| s.recover_wal()) {
        Ok(rep) => json!({
            "rolled_back_txs": rep.rolled_back_txs,
            "committed_modules": rep.committed_modules.len(), "head_rev": rep.head_rev
        }),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_verify(repo_root: &Path, hygiene: &HygieneScorer) -> Value {
    match scan_cached(repo_root, true, false, hygiene) {
        Ok((b, hit)) => json!({
            "scan_cache_hit": hit, "tests_total": b.audit.tests_total,
            "tests_failed": b.audit.tests_failed, "ruff_errors": b.audit.ruff_errors,
            "ty_errors": b.audit.ty_errors, "hygiene_median": b.audit.hygiene_median,
            "hygiene_p10": b.audit.hygiene_p10
        }),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_apply(ctx: &ToolCtx<'_>, args: &Value) -> Value {
    let module_id = args["module_id"].as_str().unwrap_or("");
    let patch_text = args["patch_text"].as_str().unwrap_or("");
    let dry_run = args["dry_run"].as_bool().unwrap_or(false);

    let (bundle, _) = match scan_cached(ctx.repo_root, true, true, ctx.hygiene) {
        Ok(b) => b,
        Err(e) => return json!({"error": e.to_string()}),
    };
    let Some(mut unit) = bundle.plan.units.into_iter().find(|u| u.id == module_id) else {
        return json!({"error": format!("unknown module_id `{module_id}`")});
    };
    if let Some(a) = args["allow_abi_change"].as_bool() { unit.allow_abi_change = a; }

    let (mut base_texts, mut base_hashes) = (BTreeMap::new(), BTreeMap::new());
    for rel in &unit.files {
        let txt = fs::read_to_string(ctx.repo_root.join(rel)).unwrap_or_default();
        base_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
        base_texts.insert(rel.clone(), txt);
    }

    let hunks = parse_patches(patch_text);
    if let Err(v) = check_hunks(ctx.repo_root, &hunks, "passed", Some("asrf_parse")) { return v; }
    let applied = match apply_patches(&base_texts, &hunks) {
        Ok(c) => c,
        Err(e) => return json!({"passed": false, "failed_gate": "asrf_parse", "error": e.to_string()}),
    };
    for rel in &applied.created { base_hashes.entry(rel.clone()).or_default(); }

    let req = CommitReq {
        ctx, unit: &unit, cands: &applied.updated, deleted: &applied.deleted,
        base: &base_hashes, dry_run, tag: "mcp-apply", extra_dry: Map::new(), extra_ok: Map::new(),
    };
    verify_commit(req, || {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TempRepo, asrf_edit};
    use tx::{TxMeta, txs_dir};

    struct Harness { repo: TempRepo, ctx: Arc<McpCtx> }

    impl Harness {
        fn new(tag: &str, files: &[(&str, &str)]) -> Self {
            let repo = TempRepo::nested_with(tag, "repo", files);
            let ctx = McpCtx::with_epoch(repo.path().to_path_buf(), Arc::new(EpochRegistry::new()));
            Self { repo, ctx }
        }
        fn call(&self, tool: &str, args: Value) -> Value { dispatch_tool_sync(tool, &args, &self.ctx) }
        async fn rpc(&self, req: Value) -> (StatusCode, Option<Value>) {
            handle_rpc(&self.ctx, &req, Proto::Stateless).await
        }
        fn begin_tx(&self) -> String {
            let res = self.call("refit_tx_begin", json!({}));
            let handle = res["tx_handle"].as_str().unwrap();
            handle.to_string()
        }
        fn set_delay_ms(&self, ms: u64) { self.ctx.test_delay_ms.store(ms, Ordering::Release); }
        fn stage(&self, id: &str, patch: &str) -> Value {
            self.call("refit_tx_stage", json!({"tx_handle": id, "patch_text": patch}))
        }
        fn commit(&self, id: &str, dry: bool, abi: bool) -> Value {
            self.call("refit_tx_commit", json!({"tx_handle": id, "dry_run": dry, "allow_abi_change": abi}))
        }
        fn apply(&self, mid: &str, patch: &str, dry: bool) -> Value {
            self.call("refit_apply", json!({"module_id": mid, "patch_text": patch, "dry_run": dry}))
        }
        fn check_dry(&self, res: &Value, rel: &str, orig: &str) {
            assert_eq!((&res["passed"], &res["committed"]), (&json!(true), &json!(false)));
            assert_eq!(self.repo.read(rel), orig);
        }
    }

    fn rpc_req(id: i64, method: &str, params: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
    }

    fn rpc_txt(v: &Value) -> &str {
        v["result"]["content"][0]["text"].as_str().unwrap()
    }

    fn obj_keys(v: &Value) -> Vec<&str> {
        let obj = v.as_object().unwrap();
        obj.keys().map(String::as_str).collect()
    }

    async fn post_mcp(
        client: &reqwest::Client, url: &str, hdrs: &[(&str, &str)], payload: &Value,
    ) -> reqwest::Response {
        let mut req = client.post(url);
        for &(k, v) in hdrs { req = req.header(k, v); }
        req.json(payload).send().await.unwrap()
    }

    async fn post_json(
        client: &reqwest::Client, url: &str, hdrs: &[(&str, &str)], payload: &Value,
    ) -> (u16, Value) {
        let resp = post_mcp(client, url, hdrs, payload).await;
        (resp.status().as_u16(), resp.json().await.unwrap())
    }

    #[test]
    fn catalog_sorted_and_hashed() {
        let arr = tool_catalog().as_array().unwrap();
        assert_eq!(arr.len(), 17);
        let mut names = Vec::with_capacity(arr.len());
        for tool in arr {
            names.push(tool["name"].as_str().unwrap());
        }
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);

        let bytes = catalog_bytes();
        let want = format!("blake3:{}", blake3::hash(bytes).to_hex());
        let encoded = serde_json::to_vec(tool_catalog()).unwrap();
        assert_eq!((catalog_digest(), bytes), (want.as_str(), encoded.as_slice()));
        for tool in arr {
            assert_eq!(obj_keys(tool), vec!["description", "inputSchema", "name"]);
            let pkeys = obj_keys(&tool["inputSchema"]["properties"]);
            let mut ps = pkeys.clone();
            ps.sort_unstable();
            assert_eq!(pkeys, ps);
        }
        let meta = catalog_meta();
        assert_eq!(
            (&meta["ttlMs"], &meta["cacheScope"], &meta["catalogDigest"]),
            (&json!(3_600_000), &json!("server"), &json!(want))
        );
        let base: Value = serde_json::from_str(&encode_base_tools(Some(json!(7)))).unwrap();
        let no_meta = base["result"].get("_meta").is_none();
        assert_eq!((&base["id"], &base["result"]["tools"], no_meta), (&json!(7), tool_catalog(), true));
    }

    #[test]
    fn b64url_strict_decode() {
        for raw in [b"".as_slice(), b"f", b"fo", b"foo", b"\xfb\xff\xfe\x00\x01"] {
            let enc = b64url_encode(raw);
            assert!(!enc.contains('=') && b64url_decode(&enc).as_deref() == Some(raw));
        }
        assert_eq!(b64url_encode(b"\xfb\xff\xfe"), "-__-");
        for bad in ["Zg==", "a", "ab+c", "ab/c", "AB", "AAB"] {
            assert!(b64url_decode(bad).is_none(), "bad `{bad}`");
        }
    }

    #[test]
    fn mcp_scan_apply_and_tx() {
        let orig = "export function add(a: number, b: number): number {\n  return a + b;\n}\n";
        let b0 = "export function beta(y: number): number {\n  return y + 2;\n}\n";
        let h = Harness::new("mcp_tx", &[("src/calc.ts", orig), ("src/b.ts", b0)]);

        assert_eq!(h.call("refit_scan", json!({}))["scan_cache_hit"], json!(false));
        assert_eq!(h.call("refit_scan", json!({}))["scan_cache_hit"], json!(true));
        let dry_p = asrf_edit("src/calc.ts", "  return a + b;", "  const s = a + b;\n  return s;");
        h.check_dry(&h.apply("calc", &dry_p, true), "src/calc.ts", orig);

        h.repo.write("src/calc.ts", orig.replace("a + b;", "a + b + 5;"));
        let live_p = asrf_edit("src/calc.ts", "  return a + b + 5;", "  const v = a + b + 6;\n  return v;");
        assert_eq!(h.apply("calc", &live_p, false)["committed"], json!(true));
        assert!(h.repo.read("src/calc.ts").contains("a + b + 6"));

        let rw = h.call("refit_rewind", json!({"token": "add", "trigger": "advisory"}));
        assert_eq!(
            (&rw["epoch_cache_hit"], &rw["rewind"]["affected_files"]),
            (&json!(false), &json!(["src/calc.ts"]))
        );
        let rw2 = h.call("refit_rewind", json!({"token": "add", "trigger": "advisory"}));
        assert_eq!(rw2["epoch_cache_hit"], json!(true));

        let root = txs_dir(h.repo.path());
        let (stale, fresh, now) = (root.join("tx-stale-1"), root.join("tx-fresh-1"), now_ms());
        for (dir, id, created, ttl) in [
            (&stale, "tx-stale-1", now.saturating_sub(120_000), 1_000),
            (&fresh, "tx-fresh-1", now, 600_000),
        ] {
            fs::create_dir_all(dir).unwrap();
            let meta = TxMeta { tx_handle: id.into(), created_ms: created, ttl_ms: ttl, staged: BTreeMap::new() };
            fs::write(dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        }
        assert_eq!(h.call("refit_tx_begin", json!({}))["swept_orphans"], json!(1));
        assert!(!stale.exists() && fresh.exists());

        let rd = h.call("refit_read", json!({"path": "src/b.ts", "start_line": 1, "end_line": 2}));
        let rd_txt = rd["content"].as_str().unwrap();
        assert!(rd_txt.contains("1: export function beta"));

        let id = h.begin_tx();
        let pb = asrf_edit("src/b.ts", "  return y + 2;", "  const next = y + 20;\n  return next;");
        assert_eq!(h.stage(&id, &pb)["staged_files_count"], json!(1));
        h.check_dry(&h.commit(&id, true, false), "src/b.ts", b0);
        let ok = h.commit(&id, false, false);
        assert_eq!((&ok["committed"], &ok["files_committed"]), (&json!(true), &json!(1)));

        let id2 = h.begin_tx();
        h.stage(&id2, &asrf_edit("src/b.ts", "  return next;\n}", "  return next;"));
        let fail = h.commit(&id2, false, false);
        assert_eq!((&fail["passed"], &fail["failed_gate"]), (&json!(false), &json!("gate0_syntax")));
        assert_eq!(h.call("refit_tx_abort", json!({"tx_handle": id2}))["rolled_back"], json!(true));

        let mv = h.begin_tx();
        h.call("refit_tx_move", json!({"tx_handle": mv, "from_path": "src/b.ts", "to_path": "src/core/b.ts"}));
        assert_eq!(h.commit(&mv, false, true)["committed"], json!(true));
        assert!(!h.repo.has("src/b.ts") && h.repo.has("src/core/b.ts"));

        let cr = h.begin_tx();
        let body = "export function help(n: number): number {\n  return n + 1;\n}\n";
        h.call("refit_tx_create", json!({"tx_handle": cr, "path": "src/core/h.ts", "content": body}));
        assert_eq!(h.commit(&cr, false, false)["committed"], json!(true));
    }

    #[tokio::test]
    async fn stdio_and_http_framing() {
        let h = Harness::new("mcp_stdio", &[("src/ok.ts", "export function ok(): number { return 1; }\n")]);
        let (out_tx, mut out_rx) = mpsc::channel::<String>(32);
        let (mut tasks, mut last_scan) = (tokio::task::JoinSet::new(), None);
        macro_rules! roundtrip {
            ($req:expr) => {{
                dispatch_stdio(&$req, &h.ctx, &out_tx, &mut tasks, &mut last_scan).await.unwrap();
                while tasks.join_next().await.is_some() {}
                serde_json::from_str::<Value>(&out_rx.recv().await.unwrap()).unwrap()
            }};
        }

        let init = roundtrip!(rpc_req(1, "initialize", json!({"protocolVersion": PROTO_BASE})));
        assert_eq!(init["result"]["protocolVersion"], json!(PROTO_BASE));

        let notif = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        dispatch_stdio(&notif, &h.ctx, &out_tx, &mut tasks, &mut last_scan).await.unwrap();
        assert!(out_rx.try_recv().is_err());

        let list = roundtrip!(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        assert!(list["result"].get("_meta").is_none());
        assert_eq!(list["result"]["tools"][0]["name"], json!("refit_apply"));

        let call = roundtrip!(rpc_req(3, "tools/call", json!({"name": "refit_read", "arguments": {"path": "src/ok.ts"}})));
        assert!(call["result"].get("resultType").is_none());
        assert!(rpc_txt(&call).contains("export function ok"));

        let disc = roundtrip!(json!({"jsonrpc": "2.0", "id": 4, "method": "server/discover", "_meta": {"protocolVersion": PROTO_STATELESS}}));
        assert_eq!(
            (&disc["result"]["protocolVersion"], &disc["result"]["_meta"]["ttlMs"]),
            (&json!(PROTO_STATELESS), &json!(3_600_000))
        );
        let st = roundtrip!(json!({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "_meta": {"protocolVersion": PROTO_STATELESS},
            "params": {"name": "refit_read", "arguments": {"path": "src/ok.ts"}}
        }));
        assert_eq!(st["result"]["resultType"], json!("complete"));

        let batch = roundtrip!(json!([{"jsonrpc": "2.0", "id": 5, "method": "ping"}]));
        assert_eq!(batch["error"]["code"], json!(ERR_INVALID_REQUEST));
    }

    #[tokio::test]
    async fn http_discover_and_headers() {
        let h = Harness::new("mcp_http", &[("src/demo.rs", "pub fn demo() -> i32 { 42 }\n")]);
        let (addr, handle) = bind_http_server(h.ctx.clone(), "127.0.0.1:0").await.unwrap();
        let (url, client) = (format!("http://{addr}/mcp"), reqwest::Client::new());

        let disc_req = json!({
            "jsonrpc": "2.0", "id": 1, "method": "server/discover",
            "_meta": {"protocolVersion": PROTO_STATELESS},
        });
        let resp = post_mcp(&client, &url, &[("Mcp-Method", "server/discover")], &disc_req).await;
        assert_eq!(resp.status().as_u16(), 200);
        assert!(resp.headers().get("mcp-session-id").is_none());
        let body: Value = resp.json().await.unwrap();
        let tools = body["result"]["tools"].as_array().unwrap();
        assert_eq!(
            (&body["result"]["protocolVersion"], &body["result"]["_meta"]["catalogDigest"], tools.len()),
            (&json!(PROTO_STATELESS), &json!(catalog_digest()), 17)
        );

        let list_req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
        let (_, lr) = post_json(&client, &url, &[("Mcp-Method", "tools/list")], &list_req).await;
        assert_eq!(
            (&lr["result"]["_meta"]["ttlMs"], &lr["result"]["_meta"]["cacheScope"]),
            (&json!(3_600_000), &json!("server"))
        );

        assert_eq!(detect_protocol(&rpc_req(0, "initialize", json!({"protocolVersion": PROTO_STATELESS}))), Proto::Stateless);
        let call_req = rpc_req(3, "tools/call", json!({"name": "refit_read", "arguments": {"path": "src/demo.rs", "start_line": 1, "end_line": 3}}));
        let call_hdrs = [
            ("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"),
            ("Mcp-Param-Path", "src/demo.rs"), ("Mcp-Param-Start-Line", "1"), ("Mcp-Param-EndLine", "3"),
        ];
        let (_, cv) = post_json(&client, &url, &call_hdrs, &call_req).await;
        assert_eq!(cv["result"]["resultType"], json!("complete"));
        assert!(rpc_txt(&cv).contains("pub fn demo"));

        for (hdrs, payload) in [
            (vec![], &disc_req),
            (vec![("Mcp-Method", "tools/list")], &disc_req),
            (vec![("Mcp-Method", "tools/call")], &call_req),
            (vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_scan")], &call_req),
            (vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"), ("Mcp-Param-Path", "src/evil.rs")], &call_req),
            (vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"), ("Mcp-Param-Ghost", "1")], &call_req),
        ] {
            let (st, v) = post_json(&client, &url, &hdrs, payload).await;
            let msg = v["error"]["message"].as_str().unwrap();
            assert_eq!((st, &v["error"]["code"]), (400, &json!(ERR_HEADER_MISMATCH)));
            assert!(msg.contains("HeaderMismatchError"));
        }

        let (bst, bv) = post_json(&client, &url, &[("Mcp-Method", "server/discover")], &json!([disc_req])).await;
        assert_eq!((bst, &bv["error"]["code"]), (400, &json!(ERR_INVALID_REQUEST)));
        for omitted in ["tasks/list", "tasks/result"] {
            let body = json!({"jsonrpc": "2.0", "id": 9, "method": omitted});
            let (_, r) = post_json(&client, &url, &[("Mcp-Method", omitted)], &body).await;
            assert_eq!(r["error"]["code"], json!(ERR_METHOD_NOT_FOUND));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn resume_token_sign_and_expire() {
        let h = Harness::new("mcp_tok", &[("src/ok.rs", "pub fn ok() -> i32 { 1 }\n")]);
        let req = rpc_req(10, "tools/call", json!({"name": "refit_read", "arguments": {"path": "src/ok.rs", "require_confirmation": true}}));
        let (st, resp) = h.rpc(req).await;
        let body = resp.unwrap();
        assert_eq!((st, &body["result"]["resultType"]), (StatusCode::OK, &json!("input_required")));
        let raw_tok = body["result"]["continuationToken"].as_str().unwrap();
        let token = raw_tok.to_string();
        assert!(token.starts_with(MRTR_TOKEN_PREFIX));

        let mut tampered = token.clone();
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'A' { 'E' } else { 'A' });
        assert_eq!(h.ctx.signer.verify(&tampered, now_ms()), Err(MrtrError::InvalidSignature));

        let exp = ContinuationPayload {
            tool: "refit_read".into(), arguments: json!({"path": "src/ok.rs"}),
            prompt: "confirm".into(), issued_ms: 100, expires_ms: 200, nonce: 1,
        };
        assert_eq!(h.ctx.signer.verify(&h.ctx.signer.sign(&exp).unwrap(), 201), Err(MrtrError::Expired));

        let decl = rpc_req(12, "tools/call", json!({"name": "refit_read", "continuationToken": token, "inputResponses": {"confirm": false}}));
        let decl_res = h.rpc(decl).await.1.unwrap();
        assert!(rpc_txt(&decl_res).contains("\"cancelled\": true"));

        let resume = rpc_req(11, "tools/call", json!({"name": "refit_read", "continuationToken": token, "inputResponses": {"confirm": true}}));
        let (st2, resp2) = h.rpc(resume).await;
        let b2 = resp2.unwrap();
        assert_eq!((st2, &b2["result"]["resultType"]), (StatusCode::OK, &json!("complete")));
        assert!(rpc_txt(&b2).contains("pub fn ok"));
    }

    #[tokio::test]
    async fn resume_token_rejects_forgery() {
        let h = Harness::new("mcp_sec", &[("src/lib.rs", "pub fn x() {}\n")]);
        assert!(h.repo.path().join(".refit/.mrtr.key").is_file());
        let seed = format!("refit-local-mrtr:{}", h.repo.path().display());
        let forged = MrtrSigner::from_seed(seed.as_bytes());
        let now = now_ms();
        let payload = ContinuationPayload {
            tool: "refit_tx_delete".into(), arguments: json!({"path": "src/lib.rs"}),
            prompt: "confirm".into(), issued_ms: now, expires_ms: now + 60_000, nonce: 99,
        };
        let forged_tok = forged.sign(&payload).unwrap();
        assert_eq!(h.ctx.signer.verify(&forged_tok, now_ms()), Err(MrtrError::InvalidSignature));

        let real = h.ctx.signer.sign(&payload).unwrap();
        let req = rpc_req(42, "tools/call", json!({"name": "refit_read", "continuationToken": real, "inputResponses": {"confirm": true}}));
        let (st, resp) = h.rpc(req).await;
        assert_eq!((st, &resp.unwrap()["error"]["code"]), (StatusCode::BAD_REQUEST, &json!(ERR_HEADER_MISMATCH)));
    }

    #[tokio::test]
    async fn task_store_states_and_cancel() {
        let h = Harness::new("mcp_tasks", &[("src/lib.rs", "pub fn hello() -> i32 { 7 }\n")]);
        let task_id_of = |v: &Value| -> String {
            let raw = v["result"]["taskId"].as_str().unwrap();
            raw.to_owned()
        };
        let start = rpc_req(
            20, "tools/call",
            json!({"name": "refit_read", "task": true, "arguments": {"path": "src/lib.rs"}}),
        );
        let (st, resp) = h.rpc(start).await;
        let val = resp.unwrap();
        assert_eq!((st, &val["result"]["resultType"]), (StatusCode::OK, &json!("task")));
        let tid = task_id_of(&val);
        assert!(h.repo.join(&format!(".refit/tasks/{tid}.json")).exists());

        let mut done = false;
        for _ in 0..50 {
            let poll = rpc_req(21, "tasks/get", json!({"taskId": tid}));
            let g = h.rpc(poll).await.1.unwrap();
            if g["result"]["status"] == json!("completed") { done = true; break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(done);

        h.set_delay_ms(250);
        let slow = rpc_req(
            22, "tools/call",
            json!({"name": "refit_read", "mode": "task", "arguments": {"path": "src/lib.rs"}}),
        );
        let sid = task_id_of(&h.rpc(slow).await.1.unwrap());
        let cancel = rpc_req(23, "tasks/cancel", json!({"taskId": sid}));
        let cr = h.rpc(cancel).await.1.unwrap();
        h.set_delay_ms(0);
        assert_eq!(cr["result"]["status"], json!("cancelled"));
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(h.ctx.tasks.load(&sid).unwrap().status, TaskState::Cancelled);

        let (rec, _) = h.ctx.tasks.create("refit_audit", Some("running".into())).unwrap();
        assert_eq!(rec.status, TaskState::Working);
        let upd = rpc_req(
            24, "tasks/update",
            json!({"taskId": rec.task_id, "status": "input_required", "progress": 0.5}),
        );
        assert_eq!(h.rpc(upd).await.1.unwrap()["result"]["status"], json!("input_required"));
        let bad = rpc_req(25, "tasks/update", json!({"taskId": rec.task_id, "status": "bogus"}));
        assert_eq!(h.rpc(bad).await.1.unwrap()["error"]["code"], json!(ERR_INVALID_PARAMS));
        h.ctx.tasks.finish(&rec.task_id, Err("failed".into()));
        assert_eq!(h.ctx.tasks.load(&rec.task_id).unwrap().status, TaskState::Failed);
    }

    #[test]
    fn epoch_memo_and_invalidation() {
        let h = Harness::new("mcp_ep", &[("src/mod_a.rs", "pub fn initial_sym() -> i32 { 1 }\n")]);
        assert_eq!(h.call("refit_scan", json!({}))["epoch_cache_hit"], json!(false));
        let y1 = h.call("refit_symbols", json!({"symbol": "initial_sym"}));
        assert_eq!((&y1["epoch_cache_hit"], &y1["definers"]), (&json!(false), &json!(["src/mod_a.rs"])));

        let s2 = h.call("refit_scan", json!({}));
        let y2 = h.call("refit_symbols", json!({"symbol": "initial_sym"}));
        assert_eq!(
            (&s2["epoch_cache_hit"], &s2["scan_cache_hit"], &y2["epoch_cache_hit"]),
            (&json!(true), &json!(true), &json!(true))
        );
        let rd = h.call("refit_read", json!({"path": "src/mod_a.rs"}));
        assert_eq!((&rd["epoch"], &rd["file_epoch"]), (&json!(1), &json!(1)));

        let store = MvccStore::open_with_epoch(h.repo.path(), "memo", h.ctx.epoch.clone()).unwrap();
        let exp = BTreeMap::from([("src/mod_a.rs".into(), h.repo.hash("src/mod_a.rs"))]);
        let next = b"pub fn initial_sym() -> i32 { 1 }\npub fn fresh_sym() -> i32 { 42 }\n".to_vec();
        let out = store.commit_tx("mod_a", &exp, &BTreeMap::from([("src/mod_a.rs".into(), next)]), &[]).unwrap();
        assert!(matches!(out, crate::mvcc::CommitOutcome::Committed { head_rev: 1, .. }));

        let s3 = h.call("refit_scan", json!({}));
        let y3 = h.call("refit_symbols", json!({"symbol": "fresh_sym"}));
        assert_eq!((&s3["epoch_cache_hit"], &s3["epoch"]), (&json!(false), &json!(2)));
        assert_eq!((&y3["epoch_cache_hit"], &y3["definers"]), (&json!(false), &json!(["src/mod_a.rs"])));
    }

    #[test]
    fn memgate_blocks_over_cap() {
        let gate = MemGate::new(1024);
        let e1 = gate.try_admit(2048).unwrap_err().to_string();
        assert!(e1.contains("exceeds capacity"));
        let permit = gate.try_admit(768).unwrap();
        let e2 = gate.try_admit(512).unwrap_err().to_string();
        assert!(e2.contains("busy"));
        drop(permit);
        let p2 = gate.try_admit(512).unwrap();
        assert_eq!(gate.used(), 512);
        drop(p2);
        assert_eq!(gate.used(), 0);
    }
}
