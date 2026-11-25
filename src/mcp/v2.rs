//! Stateless MCP `2026-07-28` (`mcp-v2`) leaf engine: Streamable HTTP (`POST /mcp`),
//! `SEP-2243` routing headers, `SEP-2322` Base64URL keyed-BLAKE3 MRTR, and `SEP-2663` durable tasks.

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
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use heck::ToSnakeCase;
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
        f.write_str(match self {
            Self::Malformed => "malformed continuationToken",
            Self::InvalidSignature => "invalid continuationToken signature",
            Self::Expired => "continuationToken expired",
        })
    }
}

/// Stateless keyed-BLAKE3 token signer and verifier for `SEP-2322` MRTR flows (Base64URL).
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
        let key_path = repo_root.join(".refit").join(".mrtr.key");
        if let Ok(bytes) = fs::read(&key_path)
            && bytes.len() >= 32
        {
            let mut key = [0u8; 32];
            key.copy_from_slice(&bytes[..32]);
            return Self { key };
        }
        let key = crate::platform::os_random_bytes::<32>().unwrap_or_else(|_| {
            let fb = format!("ephemeral-{}-{}", std::process::id(), now_ms());
            blake3::derive_key("refit mcp-v2 mrtr ephemeral fallback v1", fb.as_bytes())
        });
        if let Some(parent) = key_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = write_atomic(&key_path, &key);
        #[cfg(unix)]
        if key_path.exists() {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600));
        }
        Self { key }
    }

    pub fn sign(&self, payload: &ContinuationPayload) -> Result<String> {
        let raw = serde_json::to_vec(payload).context("encode MRTR payload")?;
        let mac = blake3::keyed_hash(&self.key, &raw);
        Ok(format!(
            "{MRTR_TOKEN_PREFIX}{}.{}",
            URL_SAFE_NO_PAD.encode(&raw),
            URL_SAFE_NO_PAD.encode(mac.as_bytes())
        ))
    }

    pub fn verify(&self, token: &str, current_ms: u64) -> std::result::Result<ContinuationPayload, MrtrError> {
        let rest = token.strip_prefix(MRTR_TOKEN_PREFIX).ok_or(MrtrError::Malformed)?;
        let (payload_b64, sig_b64) = rest.split_once('.').ok_or(MrtrError::Malformed)?;
        let payload_bytes = URL_SAFE_NO_PAD.decode(payload_b64).map_err(|_| MrtrError::Malformed)?;
        let sig_bytes = URL_SAFE_NO_PAD.decode(sig_b64).map_err(|_| MrtrError::InvalidSignature)?;
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

    fn record_path(&self, task_id: &str) -> Option<PathBuf> {
        let valid = !task_id.is_empty()
            && task_id.len() <= 64
            && task_id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'));
        valid.then(|| self.dir.join(format!("{task_id}.json")))
    }

    pub fn create(&self, tool: &str, initial_msg: Option<String>) -> Result<(TaskRecord, Arc<AtomicBool>)> {
        let ts = now_ms();
        let task_id = format!("task-{ts}-{}", TASK_SEQ.fetch_add(1, Ordering::Relaxed));
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
        let path = self.record_path(&rec.task_id).ok_or_else(|| anyhow::anyhow!("invalid taskId `{}`", rec.task_id))?;
        ensure_cachedir_tag(&self.dir);
        write_atomic(&path, &serde_json::to_vec_pretty(rec).context("encode task record")?)
    }

    pub fn load(&self, task_id: &str) -> Option<TaskRecord> {
        let raw = fs::read_to_string(self.record_path(task_id)?).ok()?;
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
        if let Some(prog) = progress { rec.progress = Some(prog.clamp(0.0, 1.0)); }
        if let Some(msg) = message { rec.status_message = Some(msg); }
        if let Some(tok) = continuation { rec.continuation_token = Some(tok); }
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
            Ok(val) => { rec.status = TaskState::Completed; rec.progress = Some(1.0); rec.result = Some(val); }
            Err(err) => { rec.status = TaskState::Failed; rec.error = Some(err); }
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
    fn dispatch_audit_research(&self, _args: Value) -> Option<SearchFuture<'_>> {
        None
    }
    fn blocking_sem(&self) -> &Arc<tokio::sync::Semaphore>;
}

pub struct McpV2State {
    pub backend: Arc<dyn McpBackend>,
    pub tasks: Arc<TaskStore>,
    pub signer: MrtrSigner,
}

impl McpV2State {
    pub fn from_backend(backend: Arc<dyn McpBackend>) -> Arc<Self> {
        let root = backend.repo_root();
        Arc::new(Self {
            tasks: Arc::new(TaskStore::for_repo(root)),
            signer: MrtrSigner::for_repo(root),
            backend,
        })
    }
}

pub fn is_v2_request(req: &Value) -> bool {
    let top_v = req.get("_meta").and_then(|m| m.get("protocolVersion")).and_then(Value::as_str);
    let par_v = req.get("params").and_then(|p| p.get("_meta")).and_then(|m| m.get("protocolVersion")).and_then(Value::as_str);
    let init_v = req.get("params").and_then(|p| p.get("protocolVersion")).and_then(Value::as_str);
    let method = req["method"].as_str().unwrap_or("");
    top_v == Some(PROTOCOL_V2)
        || par_v == Some(PROTOCOL_V2)
        || init_v == Some(PROTOCOL_V2)
        || method == "server/discover"
        || method.starts_with("tasks/")
}

pub fn batch_rejection_error() -> Value {
    rpc_err(Value::Null, ERR_INVALID_REQUEST, "Invalid Request: JSON-RPC batch arrays are forbidden in MCP 2026-07-28")
}

pub fn discover_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_V2,
        "supportedProtocolVersions": [PROTOCOL_V2, PROTOCOL_LEGACY],
        "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
        "capabilities": {
            "tools": { "listChanged": false },
            "extensions": {
                "io.modelcontextprotocol/tasks": { "methods": ["tasks/get", "tasks/update", "tasks/cancel"] },
                "io.modelcontextprotocol/mrtr": { "stateless": true }
            }
        },
        "tools": tool_catalog(),
        "_meta": sep2549_meta(),
    })
}

pub fn validate_sep2243_headers(headers: &HeaderMap, body: &Value) -> std::result::Result<(), String> {
    let hdr = |k: &str| headers.get(k).and_then(|v| v.to_str().ok()).map(str::trim);
    let body_method = body.get("method").and_then(Value::as_str).ok_or_else(|| "missing JSON-RPC `method` field".to_string())?;
    let hdr_method = hdr("mcp-method").ok_or_else(|| "missing required `Mcp-Method` header".to_string())?;
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
    } else if let Some(got) = hdr_name
        && body_name != Some(got)
    {
        return Err(format!("unexpected `Mcp-Name` header `{got}` for `{body_method}`"));
    }

    let args_obj = body.get("params").and_then(|p| p.get("arguments")).and_then(Value::as_object);
    let params_obj = body.get("params").and_then(Value::as_object);
    for (name, raw_val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        let Some(suffix) = lower.strip_prefix("mcp-param-") else { continue };
        let hdr_val = raw_val.to_str().map(str::trim).map_err(|_| format!("non-UTF8 value in header `{lower}`"))?;
        let snake = suffix.to_snake_case();
        let actual = [args_obj, params_obj]
            .into_iter()
            .flatten()
            .find_map(|m| {
                m.iter()
                    .find(|(k, _)| {
                        k.as_str() == suffix
                            || k.to_snake_case() == snake
                            || eq_param_key(k.as_str(), suffix)
                    })
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

fn eq_param_key(a: &str, b: &str) -> bool {
    let ia = a.bytes().filter(|&c| c != b'_' && c != b'-').map(|c| c.to_ascii_lowercase());
    let ib = b.bytes().filter(|&c| c != b'_' && c != b'-').map(|c| c.to_ascii_lowercase());
    ia.eq(ib)
}

fn estimate_tool_bytes(name: &str) -> usize {
    match name {
        "refit_read" | "refit_symbols" | "refit_search" | "refit_tx_abort" => LIGHT_TOOL_RESERVE_BYTES,
        _ => HEAVY_TOOL_RESERVE_BYTES,
    }
}

pub async fn handle_v2_request(state: &Arc<McpV2State>, req: &Value) -> (StatusCode, Option<Value>) {
    if req.is_array() {
        return (StatusCode::BAD_REQUEST, Some(batch_rejection_error()));
    }
    let id = req.get("id").cloned();
    let method = req["method"].as_str().unwrap_or("");
    match method {
        "server/discover" => (StatusCode::OK, Some(rpc_ok(id, discover_result()))),
        "tools/list" => (StatusCode::OK, Some(rpc_ok(id, json!({ "tools": tool_catalog(), "_meta": sep2549_meta() })))),
        "ping" => (StatusCode::OK, Some(rpc_ok(id, json!({})))),
        "notifications/initialized" => (StatusCode::ACCEPTED, None),
        "initialize" => (
            StatusCode::OK,
            Some(rpc_ok(id, json!({
                "protocolVersion": PROTOCOL_V2,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
                "_meta": sep2549_meta(),
            }))),
        ),
        "tools/call" => handle_v2_tool_call(state, id, &req["params"]).await,
        "tasks/get" => (StatusCode::OK, Some(handle_tasks_get(state, id, &req["params"]))),
        "tasks/update" => (StatusCode::OK, Some(handle_tasks_update(state, id, &req["params"]))),
        "tasks/cancel" => (StatusCode::OK, Some(handle_tasks_cancel(state, id, &req["params"]))),
        "tasks/list" | "tasks/result" => (
            StatusCode::OK,
            Some(rpc_err(id.unwrap_or(Value::Null), ERR_METHOD_NOT_FOUND, format!("Method not found: `{method}` is explicitly omitted per SEP-2663"))),
        ),
        other => match id {
            Some(req_id) => (StatusCode::OK, Some(rpc_err(req_id, ERR_METHOD_NOT_FOUND, format!("Method not found: {other}")))),
            None => (StatusCode::ACCEPTED, None),
        },
    }
}

async fn handle_v2_tool_call(state: &Arc<McpV2State>, id: Option<Value>, params: &Value) -> (StatusCode, Option<Value>) {
    if let Some(tok) = params.get("continuationToken").or_else(|| params.get("requestState")).and_then(Value::as_str).filter(|s| !s.is_empty()) {
        return handle_mrtr_resume(state, id, params, tok).await;
    }
    let name = json_str(params, "name");
    let args = params["arguments"].clone();
    let needs_confirm = args["confirmed"].as_bool() != Some(true)
        && (args["require_confirmation"].as_bool() == Some(true)
            || (name == "refit_apply" && args["allow_abi_change"].as_bool() == Some(true))
            || name == "refit_tx_delete");
    if needs_confirm {
        return build_mrtr_challenge(state, id, &name, args);
    }
    let wants_task = params["task"].as_bool() == Some(true)
        || params["mode"].as_str() == Some("task")
        || args["async_task"].as_bool() == Some(true);
    if wants_task {
        return spawn_durable_task(state, id, name, args);
    }
    run_complete_tool_call(state, id, name, args).await
}

fn build_mrtr_challenge(state: &Arc<McpV2State>, id: Option<Value>, name: &str, args: Value) -> (StatusCode, Option<Value>) {
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
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, Some(rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string()))),
    };
    let res = json!({
        "resultType": "input_required",
        "continuationToken": token,
        "requestState": token,
        "inputRequest": {
            "prompt": payload.prompt,
            "fields": [{ "name": "confirm", "type": "boolean", "required": true }]
        },
        "content": [{ "type": "text", "text": payload.prompt }],
    });
    (StatusCode::OK, Some(rpc_ok(id, res)))
}

async fn handle_mrtr_resume(state: &Arc<McpV2State>, id: Option<Value>, params: &Value, token: &str) -> (StatusCode, Option<Value>) {
    let payload = match state.signer.verify(token, now_ms()) {
        Ok(p) => p,
        Err(err) => return (StatusCode::BAD_REQUEST, Some(rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, err.to_string()))),
    };
    if let Some(req_tool) = params.get("name").and_then(Value::as_str)
        && req_tool != payload.tool
    {
        let msg = format!("HeaderMismatchError: params.name `{req_tool}` does not match continuation token tool `{}`", payload.tool);
        return (StatusCode::BAD_REQUEST, Some(rpc_err(id.unwrap_or(Value::Null), ERR_HEADER_MISMATCH, msg)));
    }
    let resp = &params["inputResponses"];
    let confirmed = resp.get("confirm").and_then(Value::as_bool).or_else(|| {
        resp.as_array()?.iter().find(|it| it["name"].as_str() == Some("confirm")).and_then(|it| it["value"].as_bool())
    });
    let Some(confirmed) = confirmed else {
        return (StatusCode::BAD_REQUEST, Some(rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, "missing boolean `confirm` inside `inputResponses`")));
    };
    if !confirmed {
        let declined = json!({ "cancelled": true, "reason": "user declined MRTR confirmation" });
        return (StatusCode::OK, Some(wrap_complete_result(id, &declined, 0, 0)));
    }
    let mut args = payload.arguments;
    if let Some(obj) = args.as_object_mut() {
        obj.insert("confirmed".into(), json!(true));
    }
    run_complete_tool_call(state, id, payload.tool, args).await
}

fn snapshot_epoch_sync(tool: &str, args: &Value) -> (u64, u64) {
    let reg = EpochRegistry::global();
    (reg.global_epoch(), reg.slot_epoch(args["path"].as_str().unwrap_or(tool)))
}

async fn run_complete_tool_call(state: &Arc<McpV2State>, id: Option<Value>, name: String, args: Value) -> (StatusCode, Option<Value>) {
    let gate = MemGate::global();
    let Ok(mem_permit) = gate.try_admit(estimate_tool_bytes(&name).min(gate.capacity())) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Some(rpc_err(id.unwrap_or(Value::Null), ERR_OVERLOADED, "server busy: MemGate admission limit reached")));
    };
    let tool_for_ep = name.clone();
    let args_for_ep = args.clone();
    let payload = if name == "refit_search" {
        let res = state.backend.dispatch_search(
            args["language"].as_str().unwrap_or("Rust").to_string(),
            args["capability"].as_str().unwrap_or("rate limiter").to_string(),
        ).await;
        drop(mem_permit);
        res
    } else if name == "refit_audit"
        && args["research"].as_bool().unwrap_or(false)
        && let Some(fut) = state.backend.dispatch_audit_research(args.clone())
    {
        let res = fut.await;
        drop(mem_permit);
        res
    } else {
        let _sem = state.backend.blocking_sem().acquire().await.ok();
        let backend = state.backend.clone();
        tokio::task::spawn_blocking(move || { let _held = mem_permit; backend.dispatch_sync(&name, &args) })
            .await
            .unwrap_or_else(|e| json!({"error": format!("mcp worker join error: {e}")}))
    };
    let (epoch, slot_epoch) = snapshot_epoch_sync(&tool_for_ep, &args_for_ep);
    (StatusCode::OK, Some(wrap_complete_result(id, &payload, epoch, slot_epoch)))
}

fn wrap_complete_result(id: Option<Value>, payload: &Value, epoch: u64, slot_epoch: u64) -> Value {
    let text = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".into());
    rpc_ok(id, json!({
        "resultType": "complete",
        "content": [{ "type": "text", "text": text }],
        "_meta": { "epoch": epoch, "slotEpoch": slot_epoch, "catalogDigest": catalog_digest() },
    }))
}

fn spawn_durable_task(state: &Arc<McpV2State>, id: Option<Value>, name: String, args: Value) -> (StatusCode, Option<Value>) {
    let (epoch, _) = snapshot_epoch_sync(&name, &args);
    let gate = MemGate::global();
    let Ok(mem_permit) = gate.try_admit(estimate_tool_bytes(&name).min(gate.capacity())) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Some(rpc_err(id.unwrap_or(Value::Null), ERR_OVERLOADED, "server busy: MemGate denied task admission")));
    };
    let (record, cancel_flag) = match state.tasks.create(&name, Some("queued for execution".into())) {
        Ok(pair) => pair,
        Err(err) => {
            drop(mem_permit);
            return (StatusCode::INTERNAL_SERVER_ERROR, Some(rpc_err(id.unwrap_or(Value::Null), -32603, err.to_string())));
        }
    };
    let (task_id, tasks_store, backend) = (record.task_id.clone(), state.tasks.clone(), state.backend.clone());
    tokio::spawn(async move {
        let _held = mem_permit;
        if cancel_flag.load(Ordering::Acquire) {
            tasks_store.clear_flag(&task_id);
            return;
        }
        let outcome = if name == "refit_search" {
            Ok(backend.dispatch_search(
                args["language"].as_str().unwrap_or("Rust").to_string(),
                args["capability"].as_str().unwrap_or("rate limiter").to_string(),
            ).await)
        } else if name == "refit_audit"
            && args["research"].as_bool().unwrap_or(false)
            && let Some(fut) = backend.dispatch_audit_research(args.clone())
        {
            Ok(fut.await)
        } else {
            let _sem = backend.blocking_sem().acquire().await.ok();
            if cancel_flag.load(Ordering::Acquire) {
                Err("cancelled before execution".into())
            } else {
                let b = backend.clone();
                tokio::task::spawn_blocking(move || b.dispatch_sync(&name, &args))
                    .await
                    .map_err(|e| format!("task join error: {e}"))
            }
        };
        if cancel_flag.load(Ordering::Acquire) {
            tasks_store.clear_flag(&task_id);
            return;
        }
        tasks_store.finish(&task_id, outcome);
    });
    (StatusCode::OK, Some(rpc_ok(id, json!({
        "resultType": "task", "taskId": record.task_id, "status": record.status,
        "task": record, "_meta": { "epoch": epoch },
    }))))
}

fn task_id_param(params: &Value) -> &str {
    params["taskId"].as_str().or_else(|| params["task_id"].as_str()).unwrap_or("")
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
    let next_status = match params.get("status") {
        None => None,
        Some(v) => match serde_json::from_value::<TaskState>(v.clone()) {
            Ok(st) => Some(st),
            Err(err) => return rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, format!("invalid task status: {err}")),
        },
    };
    let progress = params.get("progress").and_then(Value::as_f64);
    let message = params.get("statusMessage").or_else(|| params.get("status_message")).and_then(Value::as_str).map(String::from);
    let cont = params.get("continuationToken").and_then(Value::as_str).map(String::from);
    match state.tasks.update(tid, next_status, progress, message, cont) {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
}

fn handle_tasks_cancel(state: &Arc<McpV2State>, id: Option<Value>, params: &Value) -> Value {
    match state.tasks.cancel(task_id_param(params)) {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
}

fn json_response(status: StatusCode, body: Value, retry_after: bool) -> Response {
    let raw = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = (status, raw).into_response();
    resp.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    if retry_after {
        resp.headers_mut().insert("retry-after", HeaderValue::from_static("1"));
    }
    resp
}

async fn handle_health() -> Response {
    json_response(
        StatusCode::OK,
        json!({ "ok": true, "protocolVersion": PROTOCOL_V2, "stateless": true, "catalogDigest": catalog_digest() }),
        false,
    )
}

async fn handle_http_mcp(State(state): State<Arc<McpV2State>>, headers: HeaderMap, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => return json_response(StatusCode::BAD_REQUEST, rpc_err(Value::Null, ERR_PARSE, format!("Parse error: {err}")), false),
    };
    if parsed.is_array() {
        return json_response(StatusCode::BAD_REQUEST, batch_rejection_error(), false);
    }
    if let Err(detail) = validate_sep2243_headers(&headers, &parsed) {
        let req_id = parsed.get("id").cloned().unwrap_or(Value::Null);
        let err_body = json!({
            "jsonrpc": "2.0", "id": req_id,
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

pub fn build_http_router(state: Arc<McpV2State>) -> Router {
    Router::new()
        .route("/mcp", post(handle_http_mcp))
        .route("/health", get(handle_health))
        .with_state(state)
}

pub async fn bind_http_server(
    backend: Arc<dyn McpBackend>,
    listen_addr: &str,
) -> Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let app = build_http_router(McpV2State::from_backend(backend));
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
            Arc::new(Self { root, delay_ms: AtomicU64::new(0), sem: Arc::new(tokio::sync::Semaphore::new(4)) })
        }
    }

    impl McpBackend for MockBackend {
        fn repo_root(&self) -> &Path { &self.root }
        fn dispatch_sync(&self, name: &str, args: &Value) -> Value {
            let d = self.delay_ms.load(Ordering::Acquire);
            if d > 0 { std::thread::sleep(std::time::Duration::from_millis(d)); }
            if name == "refit_read" {
                let rel = args["path"].as_str().unwrap_or("");
                return json!({ "path": rel, "content": fs::read_to_string(self.root.join(rel)).unwrap_or_default() });
            }
            json!({ "ok": true, "tool": name })
        }
        fn dispatch_search(&self, language: String, capability: String) -> SearchFuture<'_> {
            Box::pin(async move { json!({ "language": language, "capability": capability }) })
        }
        fn blocking_sem(&self) -> &Arc<tokio::sync::Semaphore> { &self.sem }
    }

    fn test_state(tag: &str, files: &[(&str, &str)]) -> (TempRepo, Arc<MockBackend>, Arc<McpV2State>) {
        let repo = TempRepo::nested_with(tag, "repo", files);
        let backend = MockBackend::new(repo.path().to_path_buf());
        let state = McpV2State::from_backend(backend.clone());
        (repo, backend, state)
    }

    #[tokio::test]
    async fn stateless_discover_sep2549_batch_rejection_and_http_sep2243() {
        let repo = TempRepo::nested_with("mcp_v2_http", "repo", &[("src/demo.rs", "pub fn demo() -> i32 { 42 }\n")]);
        let (addr, handle) = bind_http_server(MockBackend::new(repo.path().to_path_buf()), "127.0.0.1:0").await.unwrap();
        let (url, client) = (format!("http://{addr}/mcp"), reqwest::Client::new());

        let disc_req = json!({"jsonrpc": "2.0", "id": 1, "method": "server/discover", "_meta": {"protocolVersion": "2026-07-28"}});
        let resp = client.post(&url).header("Mcp-Method", "server/discover").json(&disc_req).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        assert!(resp.headers().get("mcp-session-id").is_none());
        let body: Value = resp.json().await.unwrap();
        assert_eq!((&body["result"]["protocolVersion"], &body["result"]["_meta"]["ttlMs"], &body["result"]["_meta"]["cacheScope"]), (&json!("2026-07-28"), &json!(3_600_000), &json!("server")));
        assert_eq!((&body["result"]["_meta"]["catalogDigest"], body["result"]["tools"].as_array().unwrap().len()), (&json!(catalog_digest()), 17));

        let list_resp: Value = client.post(&url).header("Mcp-Method", "tools/list").json(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).send().await.unwrap().json().await.unwrap();
        assert_eq!((&list_resp["result"]["_meta"]["ttlMs"], &list_resp["result"]["_meta"]["cacheScope"]), (&json!(3_600_000), &json!("server")));

        assert!(is_v2_request(&json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2026-07-28"}})));
        let call_req = json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "refit_read", "arguments": {"path": "src/demo.rs", "start_line": 1, "end_line": 3}}});
        let call_val: Value = client.post(&url)
            .header("Mcp-Method", "tools/call").header("Mcp-Name", "refit_read")
            .header("Mcp-Param-Path", "src/demo.rs").header("Mcp-Param-Start-Line", "1")
            .header("Mcp-Param-EndLine", "3")
            .json(&call_req).send().await.unwrap().json().await.unwrap();
        assert_eq!(call_val["result"]["resultType"], json!("complete"));
        assert!(call_val["result"]["content"][0]["text"].as_str().unwrap().contains("pub fn demo"));

        type HeaderCase<'a> = (&'a str, Vec<(&'a str, &'a str)>, Value);
        let bad_cases: Vec<HeaderCase<'_>> = vec![
            ("missing Mcp-Method", vec![], disc_req.clone()),
            ("wrong Mcp-Method", vec![("Mcp-Method", "tools/list")], disc_req.clone()),
            ("missing Mcp-Name", vec![("Mcp-Method", "tools/call")], call_req.clone()),
            ("wrong Mcp-Name", vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_scan")], call_req.clone()),
            ("wrong Mcp-Param-Path", vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"), ("Mcp-Param-Path", "src/evil.rs")], call_req.clone()),
            ("unknown Mcp-Param-Ghost", vec![("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"), ("Mcp-Param-Ghost", "1")], call_req.clone()),
        ];
        for (label, hdrs, payload) in bad_cases {
            let mut b = client.post(&url);
            for (k, v) in hdrs { b = b.header(k, v); }
            let r = b.json(&payload).send().await.unwrap();
            assert_eq!(r.status().as_u16(), 400, "case `{label}`");
            let v: Value = r.json().await.unwrap();
            assert_eq!(v["error"]["code"], json!(-32020), "case `{label}`: {v}");
            assert!(v["error"]["message"].as_str().unwrap().contains("HeaderMismatchError"));
        }

        let batch = client.post(&url).header("Mcp-Method", "server/discover").json(&json!([disc_req])).send().await.unwrap();
        assert_eq!(batch.status().as_u16(), 400);
        assert_eq!(batch.json::<Value>().await.unwrap()["error"]["code"], json!(-32600));

        for omitted in ["tasks/list", "tasks/result"] {
            let r: Value = client.post(&url).header("Mcp-Method", omitted).json(&json!({"jsonrpc": "2.0", "id": 9, "method": omitted})).send().await.unwrap().json().await.unwrap();
            assert_eq!(r["error"]["code"], json!(-32601), "{omitted}: {r}");
        }
        handle.abort();
    }

    #[tokio::test]
    async fn sep2322_mrtr_signed_token_tamper_expiry_and_stateless_resume() {
        let (_repo, _backend, state) = test_state("mcp_v2_mrtr", &[("src/ok.rs", "pub fn ok() -> i32 { 1 }\n")]);
        let req = json!({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "_meta": {"protocolVersion": "2026-07-28"}, "params": {"name": "refit_read", "arguments": {"path": "src/ok.rs", "require_confirmation": true}}});
        let (st, resp) = handle_v2_request(&state, &req).await;
        assert_eq!(st, StatusCode::OK);
        let body = resp.unwrap();
        assert_eq!(body["result"]["resultType"], json!("input_required"));
        let token = body["result"]["continuationToken"].as_str().unwrap().to_string();
        assert!(token.starts_with(MRTR_TOKEN_PREFIX));

        let mut tampered = token.clone();
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'A' { 'E' } else { 'A' });
        assert_eq!(state.signer.verify(&tampered, now_ms()), Err(MrtrError::InvalidSignature));

        let exp = ContinuationPayload { tool: "refit_read".into(), arguments: json!({"path": "src/ok.rs"}), prompt: "confirm".into(), issued_ms: 100, expires_ms: 200, nonce: 1 };
        assert_eq!(state.signer.verify(&state.signer.sign(&exp).unwrap(), 201), Err(MrtrError::Expired));

        let resume = json!({"jsonrpc": "2.0", "id": 11, "method": "tools/call", "_meta": {"protocolVersion": "2026-07-28"}, "params": {"name": "refit_read", "continuationToken": token, "inputResponses": {"confirm": true}}});
        let (st2, resp2) = handle_v2_request(&state, &resume).await;
        let b2 = resp2.unwrap();
        assert_eq!((st2, &b2["result"]["resultType"]), (StatusCode::OK, &json!("complete")));
        assert!(b2["result"]["content"][0]["text"].as_str().unwrap().contains("pub fn ok"));
    }

    #[tokio::test]
    async fn sep2663_tasks_atomic_persistence_five_states_and_cooperative_cancel() {
        let (repo, backend, state) = test_state("mcp_v2_tasks", &[("src/lib.rs", "pub fn hello() -> i32 { 7 }\n")]);
        let start = json!({"jsonrpc": "2.0", "id": 20, "method": "tools/call", "params": {"name": "refit_read", "task": true, "arguments": {"path": "src/lib.rs"}}});
        let (st, resp) = handle_v2_request(&state, &start).await;
        let val = resp.unwrap();
        assert_eq!((st, &val["result"]["resultType"]), (StatusCode::OK, &json!("task")));
        let task_id = val["result"]["taskId"].as_str().unwrap().to_string();
        assert!(repo.join(&format!(".refit/tasks/{task_id}.json")).exists());

        let mut completed = false;
        for _ in 0..50 {
            let (_, g) = handle_v2_request(&state, &json!({"jsonrpc": "2.0", "id": 21, "method": "tasks/get", "params": {"taskId": task_id}})).await;
            if g.unwrap()["result"]["status"] == json!("completed") { completed = true; break; }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(completed, "durable task should reach `completed`");

        backend.delay_ms.store(250, Ordering::Release);
        let slow = json!({"jsonrpc": "2.0", "id": 22, "method": "tools/call", "params": {"name": "refit_read", "mode": "task", "arguments": {"path": "src/lib.rs"}}});
        let slow_id = handle_v2_request(&state, &slow).await.1.unwrap()["result"]["taskId"].as_str().unwrap().to_string();
        let (_, cancel_resp) = handle_v2_request(&state, &json!({"jsonrpc": "2.0", "id": 23, "method": "tasks/cancel", "params": {"taskId": slow_id}})).await;
        assert_eq!(cancel_resp.unwrap()["result"]["status"], json!("cancelled"));
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        backend.delay_ms.store(0, Ordering::Release);
        assert_eq!(state.tasks.load(&slow_id).unwrap().status, TaskState::Cancelled);

        let (rec, _flag) = state.tasks.create("refit_audit", Some("running".into())).unwrap();
        assert_eq!(rec.status, TaskState::Working);
        let (_, u1) = handle_v2_request(&state, &json!({"jsonrpc": "2.0", "id": 24, "method": "tasks/update", "params": {"taskId": rec.task_id, "status": "input_required", "progress": 0.5, "statusMessage": "awaiting confirmation"}})).await;
        assert_eq!(u1.unwrap()["result"]["status"], json!("input_required"));
        let (_, bad) = handle_v2_request(&state, &json!({"jsonrpc": "2.0", "id": 25, "method": "tasks/update", "params": {"taskId": rec.task_id, "status": "bogus_state"}})).await;
        assert_eq!(bad.unwrap()["error"]["code"], json!(ERR_INVALID_PARAMS));
        state.tasks.finish(&rec.task_id, Err("simulated failure".into()));
        assert_eq!(state.tasks.load(&rec.task_id).unwrap().status, TaskState::Failed);
    }

    #[tokio::test]
    async fn mrtr_rejects_forged_repo_path_key_and_tool_confusion() {
        let (dir, _backend, state) = test_state("mcp_v2_sec", &[("src/lib.rs", "pub fn x() {}\n")]);
        assert!(dir.path().join(".refit/.mrtr.key").is_file());
        let forged = MrtrSigner::from_seed(format!("refit-local-mrtr:{}", dir.path().display()).as_bytes());
        let now = now_ms();
        let payload = ContinuationPayload { tool: "refit_tx_delete".into(), arguments: json!({"path": "src/lib.rs"}), prompt: "confirm".into(), issued_ms: now, expires_ms: now + 60_000, nonce: 99 };
        assert!(state.signer.verify(&forged.sign(&payload).unwrap(), now_ms()).is_err());

        let real_tok = state.signer.sign(&payload).unwrap();
        let req = json!({"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {"name": "refit_read", "continuationToken": real_tok, "inputResponses": {"confirm": true}}});
        let (status, resp) = handle_v2_request(&state, &req).await;
        assert_eq!((status, &resp.unwrap()["error"]["code"]), (StatusCode::BAD_REQUEST, &json!(ERR_HEADER_MISMATCH)));
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
