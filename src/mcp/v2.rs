//! Stateless MCP `2026-07-28` (`mcp-v2`) core, Streamable HTTP (`POST /mcp`),
//! `SEP-2243` routing-header validation, `SEP-2322` stateless keyed-BLAKE3 MRTR,
//! and `SEP-2663` durable atomic task lifecycle with cooperative cancellation.

use super::catalog::{catalog_digest, sep2549_meta, tool_catalog};
use crate::net::{json_str, rpc_err, rpc_ok};
use crate::platform::{EpochRegistry, MemGate, ensure_cachedir_tag, now_ms, write_atomic};
use anyhow::{Context, Result};
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::net::TcpListener;

pub const PROTOCOL_V2: &str = "2026-07-28";
pub const PROTOCOL_LEGACY: &str = "2024-11-05";
pub const ERR_PARSE: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_HEADER_MISMATCH: i64 = -32020;
pub const ERR_OVERLOADED: i64 = -32001;

const MRTR_TOKEN_PREFIX: &str = "mrtr1.";
const MRTR_DEFAULT_TTL_MS: u64 = 300_000;
const HEAVY_TOOL_RESERVE_BYTES: usize = 2 * 1024 * 1024;
const LIGHT_TOOL_RESERVE_BYTES: usize = 256 * 1024;

static TASK_SEQ: AtomicU64 = AtomicU64::new(1);
static NONCE_SEQ: AtomicU64 = AtomicU64::new(1);

/// Five-state task lifecycle defined by `SEP-2663`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Working,
    InputRequired,
    Completed,
    Cancelled,
    Failed,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

/// Durable task state persisted atomically to `.refit/tasks/<task_id>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    #[serde(rename = "taskId")]
    pub task_id: String,
    pub tool: String,
    pub status: TaskState,
    #[serde(rename = "createdMs")]
    pub created_ms: u64,
    #[serde(rename = "updatedMs")]
    pub updated_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(default, rename = "statusMessage", skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    #[serde(default, rename = "continuationToken", skip_serializing_if = "Option::is_none")]
    pub continuation_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Stateless MRTR continuation payload signed with keyed BLAKE3 (`SEP-2322`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContinuationPayload {
    pub tool: String,
    pub arguments: Value,
    pub prompt: String,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub nonce: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MrtrError {
    Malformed,
    InvalidSignature,
    Expired,
}

impl std::fmt::Display for MrtrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed => write!(f, "malformed continuationToken"),
            Self::InvalidSignature => write!(f, "invalid continuationToken signature"),
            Self::Expired => write!(f, "continuationToken expired"),
        }
    }
}

/// Stateless keyed-BLAKE3 token signer and verifier for `SEP-2322` MRTR flows.
#[derive(Clone)]
pub struct MrtrSigner {
    key: [u8; 32],
}

impl MrtrSigner {
    pub fn from_seed(seed: &[u8]) -> Self {
        Self { key: blake3::derive_key("refit mcp-v2 sep-2322 mrtr key v1", seed) }
    }

    pub fn for_repo(repo_root: &Path) -> Self {
        if let Ok(secret) = std::env::var("REFIT_MCP_TOKEN_SECRET")
            && !secret.trim().is_empty()
        {
            return Self::from_seed(secret.trim().as_bytes());
        }
        let key = load_or_create_mrtr_key(repo_root);
        Self { key }
    }

    pub fn sign(&self, payload: &ContinuationPayload) -> Result<String> {
        let raw = serde_json::to_vec(payload).context("encode MRTR payload")?;
        let mac = blake3::keyed_hash(&self.key, &raw);
        let payload_hex = hex_encode(&raw);
        Ok(format!("{MRTR_TOKEN_PREFIX}{payload_hex}.{}", mac.to_hex()))
    }

    pub fn verify(&self, token: &str, current_ms: u64) -> std::result::Result<ContinuationPayload, MrtrError> {
        let rest = token.strip_prefix(MRTR_TOKEN_PREFIX).ok_or(MrtrError::Malformed)?;
        let (payload_hex, sig_hex) = rest.split_once('.').ok_or(MrtrError::Malformed)?;
        let payload_bytes = hex_decode(payload_hex).ok_or(MrtrError::Malformed)?;
        let sig_bytes = hex_decode(sig_hex).ok_or(MrtrError::Malformed)?;
        if sig_bytes.len() != 32 {
            return Err(MrtrError::InvalidSignature);
        }

        let expected = blake3::keyed_hash(&self.key, &payload_bytes);
        if !constant_time_eq::constant_time_eq(expected.as_bytes(), &sig_bytes) {
            return Err(MrtrError::InvalidSignature);
        }

        let payload: ContinuationPayload =
            serde_json::from_slice(&payload_bytes).map_err(|_| MrtrError::Malformed)?;
        if current_ms > payload.expires_ms {
            return Err(MrtrError::Expired);
        }
        Ok(payload)
    }
}

fn load_or_create_mrtr_key(repo_root: &Path) -> [u8; 32] {
    let key_path = repo_root.join(".refit").join(".mrtr.key");
    if let Ok(bytes) = fs::read(&key_path)
        && bytes.len() >= 32
    {
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes[..32]);
        return key;
    }
    let key = crate::platform::os_random_bytes::<32>().unwrap_or_else(|_| {
        let fallback = format!("ephemeral-{}-{}", std::process::id(), now_ms());
        blake3::derive_key("refit mcp-v2 mrtr ephemeral fallback v1", fallback.as_bytes())
    });
    let _ = persist_private_key(&key_path, &key);
    key
}

fn persist_private_key(key_path: &Path, key: &[u8; 32]) -> Result<()> {
    if let Some(parent) = key_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(key_path, key)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(key_path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn hex_decode(hex: &str) -> Option<Vec<u8>> {
    let bytes = hex.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let (pairs, _) = bytes.as_chunks::<2>();
    for &[a, b] in pairs {
        let hi = hex_nibble(a)?;
        let lo = hex_nibble(b)?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

/// Durable atomic `.refit/tasks/<id>.json` store with cooperative cancellation tokens.
pub struct TaskStore {
    dir: PathBuf,
    state_lock: Mutex<()>,
    cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl TaskStore {
    pub fn for_repo(repo_root: &Path) -> Self {
        Self {
            dir: repo_root.join(".refit").join("tasks"),
            state_lock: Mutex::new(()),
            cancel_flags: Mutex::new(HashMap::new()),
        }
    }

    pub fn check_task_id(task_id: &str) -> bool {
        !task_id.is_empty()
            && task_id.len() <= 64
            && task_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    }

    fn record_path(&self, task_id: &str) -> Option<PathBuf> {
        Self::check_task_id(task_id).then(|| self.dir.join(format!("{task_id}.json")))
    }

    pub fn create(&self, tool: &str, initial_msg: Option<String>) -> Result<(TaskRecord, Arc<AtomicBool>)> {
        let ts = now_ms();
        let seq = TASK_SEQ.fetch_add(1, Ordering::Relaxed);
        let task_id = format!("task-{ts}-{seq}");
        let rec = TaskRecord {
            task_id: task_id.clone(),
            tool: tool.to_string(),
            status: TaskState::Working,
            created_ms: ts,
            updated_ms: ts,
            progress: Some(0.0),
            status_message: initial_msg,
            continuation_token: None,
            result: None,
            error: None,
        };
        self.save(&rec)?;
        let flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags.lock().insert(task_id, flag.clone());
        Ok((rec, flag))
    }

    pub fn save(&self, rec: &TaskRecord) -> Result<()> {
        let path = self
            .record_path(&rec.task_id)
            .ok_or_else(|| anyhow::anyhow!("invalid taskId `{}`", rec.task_id))?;
        ensure_cachedir_tag(&self.dir);
        let encoded = serde_json::to_vec_pretty(rec).context("encode task record")?;
        write_atomic(&path, &encoded)
    }

    pub fn load(&self, task_id: &str) -> Option<TaskRecord> {
        let path = self.record_path(task_id)?;
        let raw = fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    pub fn clear_flag(&self, task_id: &str) {
        self.cancel_flags.lock().remove(task_id);
    }

    pub fn cancel(&self, task_id: &str) -> std::result::Result<TaskRecord, String> {
        let _guard = self.state_lock.lock();
        let mut rec = self.load(task_id).ok_or_else(|| format!("unknown taskId `{task_id}`"))?;
        if let Some(flag) = self.cancel_flags.lock().remove(task_id) {
            flag.store(true, Ordering::Release);
        }
        if !rec.status.is_terminal() {
            rec.status = TaskState::Cancelled;
            rec.updated_ms = now_ms();
            rec.status_message = Some("cancelled cooperatively via tasks/cancel".into());
            self.save(&rec).map_err(|e| e.to_string())?;
        }
        Ok(rec)
    }

    pub fn update(
        &self,
        task_id: &str,
        next_status: Option<TaskState>,
        progress: Option<f64>,
        message: Option<String>,
        continuation: Option<String>,
    ) -> std::result::Result<TaskRecord, String> {
        let _guard = self.state_lock.lock();
        let mut rec = self.load(task_id).ok_or_else(|| format!("unknown taskId `{task_id}`"))?;
        if rec.status.is_terminal() {
            return Err(format!("task `{task_id}` is already in terminal state {:?}", rec.status));
        }
        if let Some(status) = next_status {
            rec.status = status;
            if status.is_terminal() {
                self.clear_flag(task_id);
            }
        }
        if let Some(prog) = progress {
            rec.progress = Some(prog.clamp(0.0, 1.0));
        }
        if let Some(msg) = message {
            rec.status_message = Some(msg);
        }
        if let Some(tok) = continuation {
            rec.continuation_token = Some(tok);
        }
        rec.updated_ms = now_ms();
        self.save(&rec).map_err(|e| e.to_string())?;
        Ok(rec)
    }

    pub fn finish(&self, task_id: &str, outcome: std::result::Result<Value, String>) {
        let _guard = self.state_lock.lock();
        self.clear_flag(task_id);
        let Some(mut rec) = self.load(task_id) else { return };
        if rec.status.is_terminal() {
            return;
        }
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

pub type SearchFuture<'a> = Pin<Box<dyn Future<Output = Value> + Send + 'a>>;

/// Dependency-inverted backend boundary keeping `mcp/v2.rs` strictly acyclic (zero `super::*` ctx imports).
pub trait McpBackend: Send + Sync + 'static {
    fn repo_root(&self) -> &Path;
    fn dispatch_sync(&self, name: &str, args: &Value) -> Value;
    fn dispatch_search(&self, language: String, capability: String) -> SearchFuture<'_>;
    fn blocking_sem(&self) -> &Arc<tokio::sync::Semaphore>;
}

/// Shared stateless MCP `2026-07-28` server state (zero in-memory sessions!).
pub struct McpV2State {
    pub backend: Arc<dyn McpBackend>,
    pub tasks: Arc<TaskStore>,
    pub signer: MrtrSigner,
}

impl McpV2State {
    pub fn from_backend(backend: Arc<dyn McpBackend>) -> Arc<Self> {
        let root = backend.repo_root();
        let tasks = Arc::new(TaskStore::for_repo(root));
        let signer = MrtrSigner::for_repo(root);
        Arc::new(Self { backend, tasks, signer })
    }
}

/// Return `true` when a JSON-RPC request targets stateless MCP `2026-07-28`.
pub fn is_v2_request(req: &Value) -> bool {
    let top_meta = req
        .get("_meta")
        .and_then(|m| m.get("protocolVersion"))
        .and_then(Value::as_str);
    let param_meta = req
        .get("params")
        .and_then(|p| p.get("_meta"))
        .and_then(|m| m.get("protocolVersion"))
        .and_then(Value::as_str);
    let method = req["method"].as_str().unwrap_or("");
    top_meta == Some(PROTOCOL_V2)
        || param_meta == Some(PROTOCOL_V2)
        || method == "server/discover"
        || method.starts_with("tasks/")
}

/// Build the JSON-RPC `-32600` error for forbidden JSON-RPC batch arrays.
pub fn batch_rejection_error() -> Value {
    rpc_err(
        Value::Null,
        ERR_INVALID_REQUEST,
        "Invalid Request: JSON-RPC batch arrays are forbidden in MCP 2026-07-28",
    )
}

/// Build the mandatory stateless `server/discover` result payload with `SEP-2549` `_meta`.
pub fn discover_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_V2,
        "supportedProtocolVersions": [PROTOCOL_V2, PROTOCOL_LEGACY],
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
        "_meta": sep2549_meta(),
    })
}

/// Build the `SEP-2549` `tools/list` result payload for MCP `2026-07-28`.
pub fn tools_list_v2_result() -> Value {
    json!({
        "tools": tool_catalog(),
        "_meta": sep2549_meta(),
    })
}

/// Validate `SEP-2243` routing headers (`Mcp-Method`, `Mcp-Name`, `Mcp-Param-*`) against JSON-RPC body.
pub fn validate_sep2243_headers(headers: &HeaderMap, body: &Value) -> std::result::Result<(), String> {
    let body_method = body
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing JSON-RPC `method` field".to_string())?;

    let hdr_method = header_str(headers, "mcp-method")
        .ok_or_else(|| "missing required `Mcp-Method` header".to_string())?;
    if hdr_method != body_method {
        return Err(format!("Mcp-Method `{hdr_method}` does not match body method `{body_method}`"));
    }

    let body_name = body.get("params").and_then(|p| p.get("name")).and_then(Value::as_str);
    let hdr_name = header_str(headers, "mcp-name");
    if body_method == "tools/call" {
        let got = hdr_name.ok_or_else(|| "missing required `Mcp-Name` header for tools/call".to_string())?;
        let want = body_name.unwrap_or("");
        if got != want {
            return Err(format!("Mcp-Name `{got}` does not match body tool `{want}`"));
        }
    } else if let Some(got) = hdr_name
        && body_name != Some(got)
    {
        return Err(format!("unexpected `Mcp-Name` header `{got}` for `{body_method}`"));
    }

    validate_param_headers(headers, body)
}

fn header_str<'a>(headers: &'a HeaderMap, key: &str) -> Option<&'a str> {
    headers.get(key).and_then(|v| v.to_str().ok()).map(str::trim)
}

fn validate_param_headers(headers: &HeaderMap, body: &Value) -> std::result::Result<(), String> {
    let args_obj = body.get("params").and_then(|p| p.get("arguments")).and_then(Value::as_object);
    let params_obj = body.get("params").and_then(Value::as_object);

    for (name, raw_val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        let Some(suffix) = lower.strip_prefix("mcp-param-") else { continue };
        let hdr_val = raw_val
            .to_str()
            .map(str::trim)
            .map_err(|_| format!("non-UTF8 value in header `{lower}`"))?;
        let matched = find_param_value(args_obj, suffix).or_else(|| find_param_value(params_obj, suffix));
        let Some(actual) = matched else {
            return Err(format!("header `{lower}` has no matching body parameter"));
        };
        let actual_str = value_as_header_string(actual)
            .ok_or_else(|| format!("parameter `{suffix}` is not scalar for `{lower}`"))?;
        if actual_str != hdr_val {
            return Err(format!("header `{lower}` value `{hdr_val}` mismatches body `{actual_str}`"));
        }
    }
    Ok(())
}

fn find_param_value<'a>(
    map: Option<&'a serde_json::Map<String, Value>>,
    kebab_suffix: &str,
) -> Option<&'a Value> {
    let obj = map?;
    let snake = kebab_suffix.replace('-', "_");
    for (key, val) in obj {
        let key_snake = key.to_ascii_lowercase().replace('-', "_");
        if key == kebab_suffix || key_snake == snake || camel_to_snake(key) == snake {
            return Some(val);
        }
    }
    None
}

fn camel_to_snake(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for (idx, ch) in key.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if idx > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn value_as_header_string(val: &Value) -> Option<String> {
    match val {
        Value::String(text) => Some(text.clone()),
        Value::Bool(flag) => Some(if *flag { "true".into() } else { "false".into() }),
        Value::Number(num) => Some(num.to_string()),
        _ => None,
    }
}

/// Snapshot wait-free `EpochRegistry` state synchronously and drop all borrows BEFORE any `.await`.
fn snapshot_epoch_sync(tool: &str, args: &Value) -> (u64, Option<(u64, u64)>) {
    let reg = EpochRegistry::global();
    let global = reg.global_epoch();
    let key = args["path"].as_str().unwrap_or(tool);
    let slot = reg.try_read(key);
    (global, slot)
}

fn estimate_tool_bytes(name: &str) -> usize {
    match name {
        "refit_read" | "refit_symbols" | "refit_search" | "refit_tx_abort" => LIGHT_TOOL_RESERVE_BYTES,
        _ => HEAVY_TOOL_RESERVE_BYTES,
    }
}

/// Dispatch one stateless MCP `2026-07-28` request (used by both stdio and HTTP `POST /mcp`).
pub async fn handle_v2_request(state: &Arc<McpV2State>, req: &Value) -> (StatusCode, Option<Value>) {
    if req.is_array() {
        return (StatusCode::BAD_REQUEST, Some(batch_rejection_error()));
    }
    let id = req.get("id").cloned();
    let method = req["method"].as_str().unwrap_or("");

    match method {
        "server/discover" => (StatusCode::OK, Some(rpc_ok(id, discover_result()))),
        "tools/list" => (StatusCode::OK, Some(rpc_ok(id, tools_list_v2_result()))),
        "ping" => (StatusCode::OK, Some(rpc_ok(id, json!({})))),
        "notifications/initialized" => (StatusCode::ACCEPTED, None),
        "initialize" => {
            let res = json!({
                "protocolVersion": PROTOCOL_V2,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
                "_meta": sep2549_meta(),
            });
            (StatusCode::OK, Some(rpc_ok(id, res)))
        }
        "tools/call" => handle_v2_tool_call(state, id, req).await,
        "tasks/get" => (StatusCode::OK, Some(handle_tasks_get(state, id, &req["params"]))),
        "tasks/update" => (StatusCode::OK, Some(handle_tasks_update(state, id, &req["params"]))),
        "tasks/cancel" => (StatusCode::OK, Some(handle_tasks_cancel(state, id, &req["params"]))),
        "tasks/list" | "tasks/result" => {
            let err_id = id.unwrap_or(Value::Null);
            let msg = format!("Method not found: `{method}` is explicitly omitted per SEP-2663");
            (StatusCode::OK, Some(rpc_err(err_id, ERR_METHOD_NOT_FOUND, msg)))
        }
        other => {
            let Some(req_id) = id else {
                return (StatusCode::ACCEPTED, None);
            };
            let err = rpc_err(req_id, ERR_METHOD_NOT_FOUND, format!("Method not found: {other}"));
            (StatusCode::OK, Some(err))
        }
    }
}

async fn handle_v2_tool_call(
    state: &Arc<McpV2State>,
    id: Option<Value>,
    req: &Value,
) -> (StatusCode, Option<Value>) {
    let params = &req["params"];
    if let Some(token) = continuation_token_from(params) {
        return handle_mrtr_resume(state, id, params, token).await;
    }

    let name = json_str(params, "name");
    let args = params["arguments"].clone();
    if needs_mrtr_confirmation(&name, &args) {
        return build_mrtr_challenge(state, id, &name, args);
    }

    if wants_async_task(params, &args) {
        return spawn_durable_task(state, id, name, args);
    }

    run_complete_tool_call(state, id, name, args).await
}

fn continuation_token_from(params: &Value) -> Option<&str> {
    params
        .get("continuationToken")
        .or_else(|| params.get("requestState"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn needs_mrtr_confirmation(name: &str, args: &Value) -> bool {
    if args["confirmed"].as_bool() == Some(true) {
        return false;
    }
    if args["require_confirmation"].as_bool() == Some(true) {
        return true;
    }
    (name == "refit_apply" && args["allow_abi_change"].as_bool() == Some(true))
        || name == "refit_tx_delete"
}

fn wants_async_task(params: &Value, args: &Value) -> bool {
    params["task"].as_bool() == Some(true)
        || params["mode"].as_str() == Some("task")
        || args["async_task"].as_bool() == Some(true)
}

fn build_mrtr_challenge(
    state: &Arc<McpV2State>,
    id: Option<Value>,
    name: &str,
    args: Value,
) -> (StatusCode, Option<Value>) {
    let issued = now_ms();
    let payload = ContinuationPayload {
        tool: name.to_string(),
        arguments: args,
        prompt: format!("Confirm execution of `{name}` with destructive or ABI-changing flags."),
        issued_ms: issued,
        expires_ms: issued.saturating_add(MRTR_DEFAULT_TTL_MS),
        nonce: NONCE_SEQ.fetch_add(1, Ordering::Relaxed),
    };
    let token = match state.signer.sign(&payload) {
        Ok(tok) => tok,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Some(rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string())),
            );
        }
    };
    let result = json!({
        "resultType": "input_required",
        "continuationToken": token,
        "requestState": token,
        "inputRequest": {
            "prompt": payload.prompt,
            "fields": [{ "name": "confirm", "type": "boolean", "required": true }]
        },
        "content": [{ "type": "text", "text": payload.prompt }],
    });
    (StatusCode::OK, Some(rpc_ok(id, result)))
}

async fn handle_mrtr_resume(
    state: &Arc<McpV2State>,
    id: Option<Value>,
    params: &Value,
    token: &str,
) -> (StatusCode, Option<Value>) {
    let payload = match state.signer.verify(token, now_ms()) {
        Ok(p) => p,
        Err(err) => {
            let err_id = id.unwrap_or(Value::Null);
            return (StatusCode::BAD_REQUEST, Some(rpc_err(err_id, ERR_INVALID_PARAMS, err.to_string())));
        }
    };

    if let Some(req_tool) = params.get("name").and_then(Value::as_str)
        && req_tool != payload.tool
    {
        let err_id = id.unwrap_or(Value::Null);
        let msg = format!(
            "HeaderMismatchError: params.name `{req_tool}` does not match continuation token tool `{}`",
            payload.tool
        );
        return (StatusCode::BAD_REQUEST, Some(rpc_err(err_id, ERR_HEADER_MISMATCH, msg)));
    }

    let Some(confirmed) = parse_confirm_response(&params["inputResponses"]) else {
        let err_id = id.unwrap_or(Value::Null);
        let msg = "missing boolean `confirm` inside `inputResponses`";
        return (StatusCode::BAD_REQUEST, Some(rpc_err(err_id, ERR_INVALID_PARAMS, msg)));
    };
    if !confirmed {
        let declined = json!({ "cancelled": true, "reason": "user declined MRTR confirmation" });
        return (StatusCode::OK, Some(wrap_complete_result(id, &declined, 0)));
    }

    let mut args = payload.arguments;
    if let Some(obj) = args.as_object_mut() {
        obj.insert("confirmed".into(), json!(true));
    }
    run_complete_tool_call(state, id, payload.tool, args).await
}

fn parse_confirm_response(val: &Value) -> Option<bool> {
    if let Some(flag) = val.get("confirm").and_then(Value::as_bool) {
        return Some(flag);
    }
    let arr = val.as_array()?;
    for item in arr {
        if item["name"].as_str() == Some("confirm") {
            return item["value"].as_bool();
        }
    }
    None
}

async fn run_complete_tool_call(
    state: &Arc<McpV2State>,
    id: Option<Value>,
    name: String,
    args: Value,
) -> (StatusCode, Option<Value>) {
    // W3 <-> W2 invariant: snapshot EpochRegistry synchronously and drop before any `.await`.
    let (epoch, _slot) = snapshot_epoch_sync(&name, &args);

    // W3 <-> W4 invariant: check MemGate::try_admit BEFORE spawning blocking/async work.
    let gate = MemGate::global();
    let reserve = estimate_tool_bytes(&name).min(gate.capacity());
    let Ok(mem_permit) = gate.try_admit(reserve) else {
        let err_id = id.unwrap_or(Value::Null);
        let err = rpc_err(err_id, ERR_OVERLOADED, "server busy: MemGate admission limit reached");
        return (StatusCode::SERVICE_UNAVAILABLE, Some(err));
    };

    let payload = if name == "refit_search" {
        let lang = args["language"].as_str().unwrap_or("Rust").to_string();
        let cap = args["capability"].as_str().unwrap_or("rate limiter").to_string();
        let res = state.backend.dispatch_search(lang, cap).await;
        drop(mem_permit);
        res
    } else {
        let _sem = state.backend.blocking_sem().acquire().await.ok();
        let backend = state.backend.clone();
        tokio::task::spawn_blocking(move || {
            let _held = mem_permit;
            backend.dispatch_sync(&name, &args)
        })
        .await
        .unwrap_or_else(|e| json!({"error": format!("mcp worker join error: {e}")}))
    };

    (StatusCode::OK, Some(wrap_complete_result(id, &payload, epoch)))
}

fn wrap_complete_result(id: Option<Value>, payload: &Value, epoch: u64) -> Value {
    let text = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".into());
    rpc_ok(
        id,
        json!({
            "resultType": "complete",
            "content": [{ "type": "text", "text": text }],
            "_meta": { "epoch": epoch, "catalogDigest": catalog_digest() },
        }),
    )
}

fn spawn_durable_task(
    state: &Arc<McpV2State>,
    id: Option<Value>,
    name: String,
    args: Value,
) -> (StatusCode, Option<Value>) {
    // W3 <-> W2: snapshot epoch synchronously before spawn.
    let (_epoch, _slot) = snapshot_epoch_sync(&name, &args);

    // W3 <-> W4: MUST check MemGate::try_admit BEFORE calling `tokio::spawn`!
    let gate = MemGate::global();
    let reserve = estimate_tool_bytes(&name).min(gate.capacity());
    let Ok(mem_permit) = gate.try_admit(reserve) else {
        let err_id = id.unwrap_or(Value::Null);
        let err = rpc_err(err_id, ERR_OVERLOADED, "server busy: MemGate denied task admission");
        return (StatusCode::SERVICE_UNAVAILABLE, Some(err));
    };

    let (record, cancel_flag) = match state.tasks.create(&name, Some("queued for execution".into())) {
        Ok(pair) => pair,
        Err(err) => {
            drop(mem_permit);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Some(rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string())),
            );
        }
    };

    let task_id = record.task_id.clone();
    let tasks_store = state.tasks.clone();
    let backend = state.backend.clone();

    tokio::spawn(async move {
        let _held_permit = mem_permit;
        if cancel_flag.load(Ordering::Acquire) {
            tasks_store.clear_flag(&task_id);
            return;
        }
        let outcome = execute_task_tool(&name, args, &backend, &cancel_flag).await;
        if cancel_flag.load(Ordering::Acquire) {
            tasks_store.clear_flag(&task_id);
            return;
        }
        tasks_store.finish(&task_id, outcome);
    });

    let res = json!({
        "resultType": "task",
        "taskId": record.task_id,
        "status": record.status,
        "task": record,
    });
    (StatusCode::OK, Some(rpc_ok(id, res)))
}

async fn execute_task_tool(
    name: &str,
    args: Value,
    backend: &Arc<dyn McpBackend>,
    cancel_flag: &AtomicBool,
) -> std::result::Result<Value, String> {
    if name == "refit_search" {
        let lang = args["language"].as_str().unwrap_or("Rust").to_string();
        let cap = args["capability"].as_str().unwrap_or("rate limiter").to_string();
        return Ok(backend.dispatch_search(lang, cap).await);
    }
    let _sem = backend.blocking_sem().acquire().await.ok();
    if cancel_flag.load(Ordering::Acquire) {
        return Err("cancelled before execution".into());
    }
    let b = backend.clone();
    let tool_name = name.to_string();
    tokio::task::spawn_blocking(move || b.dispatch_sync(&tool_name, &args))
        .await
        .map_err(|e| format!("task join error: {e}"))
}

fn task_id_param(params: &Value) -> &str {
    params["taskId"]
        .as_str()
        .or_else(|| params["task_id"].as_str())
        .unwrap_or("")
}

fn handle_tasks_get(state: &Arc<McpV2State>, id: Option<Value>, params: &Value) -> Value {
    let tid = task_id_param(params);
    match state.tasks.load(tid) {
        Some(rec) => rpc_ok(id, json!(rec)),
        None => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, format!("unknown taskId `{tid}`")),
    }
}

fn handle_tasks_update(state: &Arc<McpV2State>, id: Option<Value>, params: &Value) -> Value {
    let tid = task_id_param(params);
    let next_status: Option<TaskState> = match params.get("status") {
        None => None,
        Some(v) => match serde_json::from_value(v.clone()) {
            Ok(st) => Some(st),
            Err(err) => {
                return rpc_err(
                    id.unwrap_or(Value::Null),
                    ERR_INVALID_PARAMS,
                    format!("invalid task status: {err}"),
                );
            }
        },
    };
    let progress = params.get("progress").and_then(Value::as_f64);
    let message = params
        .get("statusMessage")
        .or_else(|| params.get("status_message"))
        .and_then(Value::as_str)
        .map(String::from);
    let continuation = params
        .get("continuationToken")
        .and_then(Value::as_str)
        .map(String::from);

    match state.tasks.update(tid, next_status, progress, message, continuation) {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
}

fn handle_tasks_cancel(state: &Arc<McpV2State>, id: Option<Value>, params: &Value) -> Value {
    let tid = task_id_param(params);
    match state.tasks.cancel(tid) {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
}

fn json_response(status: StatusCode, body: Value, retry_after: bool) -> Response {
    let raw = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = (status, raw).into_response();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    if retry_after {
        resp.headers_mut().insert("retry-after", HeaderValue::from_static("1"));
    }
    resp
}

async fn handle_health() -> Response {
    json_response(
        StatusCode::OK,
        json!({
            "ok": true,
            "protocolVersion": PROTOCOL_V2,
            "stateless": true,
            "catalogDigest": catalog_digest(),
        }),
        false,
    )
}

async fn handle_http_mcp(
    State(state): State<Arc<McpV2State>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                rpc_err(Value::Null, ERR_PARSE, format!("Parse error: {err}")),
                false,
            );
        }
    };

    if parsed.is_array() {
        return json_response(StatusCode::BAD_REQUEST, batch_rejection_error(), false);
    }

    if let Err(detail) = validate_sep2243_headers(&headers, &parsed) {
        let req_id = parsed.get("id").cloned().unwrap_or(Value::Null);
        let err_body = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": ERR_HEADER_MISMATCH,
                "message": format!("HeaderMismatchError: {detail}"),
                "data": { "error": "HeaderMismatchError" }
            }
        });
        return json_response(StatusCode::BAD_REQUEST, err_body, false);
    }

    let (status, resp_opt) = handle_v2_request(&state, &parsed).await;
    match resp_opt {
        Some(val) => json_response(status, val, status == StatusCode::SERVICE_UNAVAILABLE),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// Build the stateless Streamable HTTP (`POST /mcp`, `session_store: None`) Axum router.
pub fn build_http_router(state: Arc<McpV2State>) -> Router {
    Router::new()
        .route("/mcp", post(handle_http_mcp))
        .route("/health", get(handle_health))
        .with_state(state)
}

/// Bind and start the stateless Streamable HTTP MCP `2026-07-28` server (default loopback).
pub async fn bind_http_server(
    backend: Arc<dyn McpBackend>,
    listen_addr: &str,
) -> Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let state = McpV2State::from_backend(backend);
    let app = build_http_router(state);

    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("bind MCP HTTP server on {listen_addr}"))?;
    let bound = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            tracing::error!("mcp-v2 HTTP server stopped: {err}");
        }
    });
    Ok((bound, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    struct MockBackend {
        root: PathBuf,
        delay_ms: AtomicU64,
        sem: Arc<tokio::sync::Semaphore>,
    }

    impl MockBackend {
        fn new(root: PathBuf) -> Arc<Self> {
            Arc::new(Self {
                root,
                delay_ms: AtomicU64::new(0),
                sem: Arc::new(tokio::sync::Semaphore::new(4)),
            })
        }
    }

    impl McpBackend for MockBackend {
        fn repo_root(&self) -> &Path {
            &self.root
        }

        fn dispatch_sync(&self, name: &str, args: &Value) -> Value {
            let delay = self.delay_ms.load(Ordering::Acquire);
            if delay > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
            if name == "refit_read" {
                let rel = args["path"].as_str().unwrap_or("");
                let content = fs::read_to_string(self.root.join(rel)).unwrap_or_default();
                return json!({ "path": rel, "content": content });
            }
            json!({ "ok": true, "tool": name })
        }

        fn dispatch_search(&self, language: String, capability: String) -> SearchFuture<'_> {
            Box::pin(async move { json!({ "language": language, "capability": capability }) })
        }

        fn blocking_sem(&self) -> &Arc<tokio::sync::Semaphore> {
            &self.sem
        }
    }

    fn test_state(tag: &str, files: &[(&str, &str)]) -> (TempRepo, Arc<MockBackend>, Arc<McpV2State>) {
        let repo = TempRepo::nested_with(tag, "repo", files);
        let backend = MockBackend::new(repo.path().to_path_buf());
        let state = McpV2State::from_backend(backend.clone());
        (repo, backend, state)
    }

    #[tokio::test]
    async fn stateless_discover_sep2549_batch_rejection_and_http_sep2243() {
        let repo = TempRepo::nested_with(
            "mcp_v2_http",
            "repo",
            &[("src/demo.rs", "pub fn demo() -> i32 {\n    42\n}\n")],
        );
        let backend = MockBackend::new(repo.path().to_path_buf());
        let (addr, handle) = bind_http_server(backend, "127.0.0.1:0").await.unwrap();
        let url = format!("http://{addr}/mcp");
        let client = reqwest::Client::new();

        // 1. Stateless zero-handshake `server/discover` with valid SEP-2243 headers
        let disc_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "_meta": { "protocolVersion": "2026-07-28" }
        });
        let resp = client
            .post(&url)
            .header("Mcp-Method", "server/discover")
            .json(&disc_req)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        assert!(resp.headers().get("mcp-session-id").is_none());
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["result"]["protocolVersion"], json!("2026-07-28"));
        assert_eq!(body["result"]["_meta"]["ttlMs"], json!(3_600_000));
        assert_eq!(body["result"]["_meta"]["cacheScope"], json!("server"));
        assert_eq!(body["result"]["_meta"]["catalogDigest"], json!(catalog_digest()));
        assert_eq!(body["result"]["tools"].as_array().unwrap().len(), 16);

        // 2. `tools/list` returns SEP-2549 metadata
        let list_resp: Value = client
            .post(&url)
            .header("Mcp-Method", "tools/list")
            .json(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list_resp["result"]["_meta"]["ttlMs"], json!(3_600_000));
        assert_eq!(list_resp["result"]["_meta"]["cacheScope"], json!("server"));

        // 3. Valid `tools/call` with `Mcp-Method` + `Mcp-Name` + `Mcp-Param-Path`
        let call_req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "refit_read",
                "arguments": { "path": "src/demo.rs", "start_line": 1, "end_line": 3 }
            }
        });
        let call_resp = client
            .post(&url)
            .header("Mcp-Method", "tools/call")
            .header("Mcp-Name", "refit_read")
            .header("Mcp-Param-Path", "src/demo.rs")
            .header("Mcp-Param-Start-Line", "1")
            .json(&call_req)
            .send()
            .await
            .unwrap();
        assert_eq!(call_resp.status().as_u16(), 200);
        let call_val: Value = call_resp.json().await.unwrap();
        assert_eq!(call_val["result"]["resultType"], json!("complete"));
        assert!(call_val["result"]["content"][0]["text"].as_str().unwrap().contains("pub fn demo"));

        // 4. SEP-2243 mismatches -> HTTP 400 + JSON-RPC -32020 (`HeaderMismatchError`)
        type HeaderCase<'a> = (&'a str, Vec<(&'a str, &'a str)>, Value);
        let bad_cases: Vec<HeaderCase<'_>> = vec![
            ("missing Mcp-Method", vec![], disc_req.clone()),
            ("wrong Mcp-Method", vec![("Mcp-Method", "tools/list")], disc_req.clone()),
            ("missing Mcp-Name", vec![("Mcp-Method", "tools/call")], call_req.clone()),
            (
                "wrong Mcp-Name",
                vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_scan")],
                call_req.clone(),
            ),
            (
                "wrong Mcp-Param-Path",
                vec![
                    ("Mcp-Method", "tools/call"),
                    ("Mcp-Name", "refit_read"),
                    ("Mcp-Param-Path", "src/evil.rs"),
                ],
                call_req.clone(),
            ),
            (
                "unknown Mcp-Param-Ghost",
                vec![
                    ("Mcp-Method", "tools/call"),
                    ("Mcp-Name", "refit_read"),
                    ("Mcp-Param-Ghost", "1"),
                ],
                call_req.clone(),
            ),
        ];
        for (label, hdrs, payload) in bad_cases {
            let mut builder = client.post(&url);
            for (k, v) in hdrs {
                builder = builder.header(k, v);
            }
            let r = builder.json(&payload).send().await.unwrap();
            assert_eq!(r.status().as_u16(), 400, "case `{label}` must return 400");
            let v: Value = r.json().await.unwrap();
            assert_eq!(v["error"]["code"], json!(-32020), "case `{label}`: {v}");
            assert!(
                v["error"]["message"].as_str().unwrap().contains("HeaderMismatchError"),
                "case `{label}`: {v}"
            );
        }

        // 5. JSON-RPC batch array explicitly rejected with HTTP 400 + -32600
        let batch_resp = client
            .post(&url)
            .header("Mcp-Method", "server/discover")
            .json(&json!([disc_req]))
            .send()
            .await
            .unwrap();
        assert_eq!(batch_resp.status().as_u16(), 400);
        let batch_body: Value = batch_resp.json().await.unwrap();
        assert_eq!(batch_body["error"]["code"], json!(-32600));

        // 6. Explicit omission of `tasks/list` and `tasks/result` -> -32601
        for omitted in ["tasks/list", "tasks/result"] {
            let r: Value = client
                .post(&url)
                .header("Mcp-Method", omitted)
                .json(&json!({"jsonrpc": "2.0", "id": 9, "method": omitted}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(r["error"]["code"], json!(-32601), "{omitted}: {r}");
        }

        handle.abort();
    }

    #[tokio::test]
    async fn sep2322_mrtr_signed_token_tamper_expiry_and_stateless_resume() {
        let (_repo, _backend, state) = test_state(
            "mcp_v2_mrtr",
            &[("src/ok.rs", "pub fn ok() -> i32 { 1 }\n")],
        );

        // Trigger input_required via `require_confirmation: true`
        let req = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "_meta": { "protocolVersion": "2026-07-28" },
            "params": {
                "name": "refit_read",
                "arguments": { "path": "src/ok.rs", "require_confirmation": true }
            }
        });
        let (st, resp) = handle_v2_request(&state, &req).await;
        assert_eq!(st, StatusCode::OK);
        let body = resp.unwrap();
        assert_eq!(body["result"]["resultType"], json!("input_required"));
        let token = body["result"]["continuationToken"].as_str().unwrap().to_string();
        assert!(token.starts_with(MRTR_TOKEN_PREFIX));

        // Tampered token rejected
        let mut tampered = token.clone();
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'a' { 'b' } else { 'a' });
        assert_eq!(state.signer.verify(&tampered, now_ms()), Err(MrtrError::InvalidSignature));

        // Expired token rejected
        let expired_payload = ContinuationPayload {
            tool: "refit_read".into(),
            arguments: json!({"path": "src/ok.rs"}),
            prompt: "confirm".into(),
            issued_ms: 100,
            expires_ms: 200,
            nonce: 1,
        };
        let exp_tok = state.signer.sign(&expired_payload).unwrap();
        assert_eq!(state.signer.verify(&exp_tok, 201), Err(MrtrError::Expired));

        // Resume statelessly with confirmation -> completes!
        let resume_req = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "_meta": { "protocolVersion": "2026-07-28" },
            "params": {
                "name": "refit_read",
                "continuationToken": token,
                "inputResponses": { "confirm": true }
            }
        });
        let (st2, resp2) = handle_v2_request(&state, &resume_req).await;
        assert_eq!(st2, StatusCode::OK);
        let b2 = resp2.unwrap();
        assert_eq!(b2["result"]["resultType"], json!("complete"));
        assert!(b2["result"]["content"][0]["text"].as_str().unwrap().contains("pub fn ok"));
    }

    #[tokio::test]
    async fn sep2663_tasks_atomic_persistence_five_states_and_cooperative_cancel() {
        let (repo, backend, state) = test_state(
            "mcp_v2_tasks",
            &[("src/lib.rs", "pub fn hello() -> i32 { 7 }\n")],
        );

        // 1. Spawn a durable task and wait for completion
        let start_req = json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "refit_read",
                "task": true,
                "arguments": { "path": "src/lib.rs" }
            }
        });
        let (st, resp) = handle_v2_request(&state, &start_req).await;
        assert_eq!(st, StatusCode::OK);
        let val = resp.unwrap();
        assert_eq!(val["result"]["resultType"], json!("task"));
        let task_id = val["result"]["taskId"].as_str().unwrap().to_string();
        assert!(repo.join(&format!(".refit/tasks/{task_id}.json")).exists());

        let mut completed = false;
        for _ in 0..50 {
            let (_, get_resp) = handle_v2_request(
                &state,
                &json!({"jsonrpc": "2.0", "id": 21, "method": "tasks/get", "params": { "taskId": task_id }}),
            )
            .await;
            let status = get_resp.unwrap()["result"]["status"].as_str().unwrap().to_string();
            if status == "completed" {
                completed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(completed, "durable task should reach `completed`");

        // 2. Cooperative cancellation via `tasks/cancel` using `MockBackend` delay
        backend.delay_ms.store(250, Ordering::Release);
        let slow_req = json!({
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "refit_read",
                "mode": "task",
                "arguments": { "path": "src/lib.rs" }
            }
        });
        let (_, slow_resp) = handle_v2_request(&state, &slow_req).await;
        let slow_id = slow_resp.unwrap()["result"]["taskId"].as_str().unwrap().to_string();
        let (_, cancel_resp) = handle_v2_request(
            &state,
            &json!({"jsonrpc": "2.0", "id": 23, "method": "tasks/cancel", "params": { "taskId": slow_id }}),
        )
        .await;
        assert_eq!(cancel_resp.unwrap()["result"]["status"], json!("cancelled"));
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        backend.delay_ms.store(0, Ordering::Release);
        let persisted = state.tasks.load(&slow_id).unwrap();
        assert_eq!(persisted.status, TaskState::Cancelled);

        // 3. Explicit 5-state lifecycle check via `tasks/update` (working -> input_required -> failed)
        let (rec, _flag) = state.tasks.create("refit_audit", Some("running".into())).unwrap();
        assert_eq!(rec.status, TaskState::Working);
        let (_, upd1) = handle_v2_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": 24,
                "method": "tasks/update",
                "params": {
                    "taskId": rec.task_id,
                    "status": "input_required",
                    "progress": 0.5,
                    "statusMessage": "awaiting confirmation"
                }
            }),
        )
        .await;
        assert_eq!(upd1.unwrap()["result"]["status"], json!("input_required"));
        let (_, bad_upd) = handle_v2_request(
            &state,
            &json!({
                "jsonrpc": "2.0",
                "id": 25,
                "method": "tasks/update",
                "params": { "taskId": rec.task_id, "status": "bogus_state" }
            }),
        )
        .await;
        assert_eq!(bad_upd.unwrap()["error"]["code"], json!(ERR_INVALID_PARAMS));
        state.tasks.finish(&rec.task_id, Err("simulated failure".into()));
        assert_eq!(state.tasks.load(&rec.task_id).unwrap().status, TaskState::Failed);
    }

    #[tokio::test]
    async fn mrtr_rejects_forged_repo_path_key_and_tool_confusion() {
        let (dir, _backend, state) = test_state("mcp_v2_sec", &[("src/lib.rs", "pub fn x() {}\n")]);
        assert!(dir.path().join(".refit/.mrtr.key").is_file());

        // Forged key derived from public repo_root must fail verification.
        let forged = MrtrSigner::from_seed(format!("refit-local-mrtr:{}", dir.path().display()).as_bytes());
        let now = now_ms();
        let payload = ContinuationPayload {
            tool: "refit_tx_delete".into(),
            arguments: json!({"path": "src/lib.rs"}),
            prompt: "confirm".into(),
            issued_ms: now,
            expires_ms: now + 60_000,
            nonce: 99,
        };
        let forged_tok = forged.sign(&payload).unwrap();
        assert!(state.signer.verify(&forged_tok, now_ms()).is_err());

        // Tool confusion: token for `refit_tx_delete` resumed with `params.name = refit_read`.
        let real_tok = state.signer.sign(&payload).unwrap();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "refit_read",
                "continuationToken": real_tok,
                "inputResponses": { "confirm": true }
            }
        });
        let (status, resp) = handle_v2_request(&state, &req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(resp.unwrap()["error"]["code"], json!(ERR_HEADER_MISMATCH));
    }

    #[test]
    fn memgate_try_admit_blocks_over_capacity() {
        let local_gate = MemGate::new(1024);
        let permit = local_gate.try_admit(768).unwrap();
        assert!(local_gate.try_admit(512).is_err());
        drop(permit);
        assert!(local_gate.try_admit(512).is_ok());
    }
}
