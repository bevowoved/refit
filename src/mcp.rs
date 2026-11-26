//! Protocol-agnostic MCP engine supporting stdio and stateless Streamable HTTP.

mod tx;

use crate::gates::{GateConfig, read_gates, scan_cached};
use crate::graph::{apply_patches, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::net::{rpc_err, rpc_ok, rpc_text};
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
    CommitReq, ToolCtx, check_hunks, check_target, check_txid, err_val, fail_val, parse_args,
    tx_abort, tx_begin, tx_commit, tx_mutate, verify_commit,
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

const SKILL_MD: &str = include_str!("../skills/refit/SKILL.md");
const SKILL_URI: &str = "skill://refit";
const SKILL_MD_URI: &str = "skill://refit/SKILL.md";
const MCP_INSTRUCTIONS: &str = concat!(
    "Run refit_check before commit/push for <10ms secret/OSV/doc-drift checks and optional Gate 4 Kane CLI live verification. ",
    "Inspect code with refit_read and refit_symbols before editing. ",
    "Use refit_tx (op: begin|stage|create|delete|move|commit|abort) or refit_apply ",
    "for verified MVCC edits. Read skill://refit/SKILL.md for the full workflow."
);

// Tool catalog

type ToolAnnotations = (bool, bool, bool, bool);

type ToolRow = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static [&'static str],
    ToolAnnotations,
);

/// Tool definitions sorted strictly A->Z by `name` for prompt-cache stability.
/// Annotations tuple: `(read_only, destructive, idempotent, open_world)`.
const TOOLS: &[ToolRow] = &[
    ("refit_apply", "Apply ASRF patches, run Gates 0 to 3.5, and commit via MVCC WAL.",
        &[("allow_abi_change", "boolean"), ("dry_run", "boolean"), ("module_id", "string"), ("patch_text", "string")],
        &["module_id", "patch_text"],
        (false, true, false, false)),
    ("refit_audit", "Audit clones, dead symbols, large files, unused deps, and infra configs.",
        &[("markdown", "boolean"), ("research", "boolean")], &[],
        (true, false, true, false)),
    ("refit_check", "Run <10ms static secret/OSV/doc-drift check and optional Gate 4 Kane CLI live browser/mobile verification.",
        &[
            ("app", "string"), ("bug_detection", "boolean"), ("deps", "boolean"),
            ("diff", "string"), ("headless", "boolean"), ("kane", "string"),
            ("kane_op", "string"), ("no_adaptive_heal", "boolean"), ("no_cache", "boolean"),
            ("policy", "string"), ("remote", "boolean"), ("staged", "boolean"),
            ("target", "string"), ("timeout_secs", "integer"), ("url", "string"),
            ("verify_docs", "boolean"),
        ], &[],
        (true, false, false, true)),
    ("refit_read", "Read a bounded line range from a repository file.",
        &[("end_line", "integer"), ("path", "string"), ("start_line", "integer")], &["path"],
        (true, false, true, false)),
    ("refit_resume", "Recover unfinished WAL transactions from CAS and report committed modules.", &[], &[],
        (false, false, true, false)),
    ("refit_rewind", "Query SymbolGraph-backed rewind impact set and learned rules for a symbol token.",
        &[("token", "string"), ("trigger", "string")], &["token"],
        (true, false, true, false)),
    ("refit_scan", "Scan repository DAG and codehygiene scores (fast cached by default).",
        &[("fast", "boolean")], &[],
        (true, false, true, false)),
    ("refit_search", "Run 10 req/s GCRA-paced web search with repo-scoped BLAKE3 cache.",
        &[("capability", "string"), ("language", "string")], &["capability", "language"],
        (true, false, true, true)),
    ("refit_sustain", "Generate AGENTS.md, .refit/invariants.yml, ADR-001, and CI guard.",
        &[("dry_run", "boolean")], &[],
        (false, false, true, false)),
    ("refit_symbols", "Query symbol definitions, callers, dependencies, fan-in/out, and dead symbols.",
        &[("file", "string"), ("symbol", "string")], &[],
        (true, false, true, false)),
    ("refit_tx", "Manage multi-file shadow transactions (op: begin|stage|create|delete|move|commit|abort).",
        &[
            ("allow_abi_change", "boolean"), ("content", "string"), ("dry_run", "boolean"),
            ("from_path", "string"), ("op", "string"), ("patch_text", "string"),
            ("path", "string"), ("to_path", "string"), ("ttl_secs", "integer"), ("tx_handle", "string"),
        ],
        &["op"],
        (false, true, false, false)),
    ("refit_verify", "Run a full repository audit with BLAKE3 content caching.", &[], &[],
        (true, false, true, false)),
];

struct FrozenCatalog { value: Value, bytes: Box<[u8]>, digest: String }

static CATALOG: OnceLock<FrozenCatalog> = OnceLock::new();
static SKILL_DIGEST: OnceLock<String> = OnceLock::new();

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
    json!({ "cacheScope": CATALOG_CACHE_SCOPE, "catalogDigest": catalog_digest(), "ttlMs": CATALOG_TTL_MS })
}

fn encode_base_tools(id: Option<Value>) -> String {
    let id_json = serde_json::to_string(&id).unwrap_or_else(|_| "null".into());
    let tools_str = std::str::from_utf8(catalog_bytes()).unwrap_or("[]");
    format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{"tools":{tools_str}}}}}"#)
}

fn encode_tool(&(name, desc, prop_list, req, (ro, dest, idem, ow)): &ToolRow) -> Value {
    let props: Map<String, Value> = prop_list.iter().map(|&(k, ty)| (k.into(), json!({ "type": ty }))).collect();
    let mut schema = Map::from_iter([("properties".into(), Value::Object(props))]);
    if !req.is_empty() {
        let mut req_sorted = req.to_vec();
        req_sorted.sort_unstable();
        schema.insert("required".into(), json!(req_sorted));
    }
    schema.insert("type".into(), "object".into());
    json!({
        "annotations": {
            "destructiveHint": dest, "idempotentHint": idem, "openWorldHint": ow, "readOnlyHint": ro,
        },
        "description": desc, "inputSchema": Value::Object(schema), "name": name,
    })
}

// Pure-Rust SHA-256 (FIPS 180-4) and SEP-2640 Skills helpers

const SHA256_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_compress(st: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (slot, chunk) in w[..16].iter_mut().zip(block.as_chunks::<4>().0) {
        *slot = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *st;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
    }
    for (slot, delta) in st.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(delta);
    }
}

fn sha256_hex(data: &[u8]) -> String {
    use std::fmt::Write;
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut buf = data.to_vec();
    buf.push(0x80);
    while (buf.len() % 64) != 56 { buf.push(0); }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    let mut state = SHA256_INIT;
    for chunk in buf.as_chunks::<64>().0 { sha256_compress(&mut state, chunk); }
    let mut hex = String::with_capacity(64);
    for word in state { let _ = write!(&mut hex, "{word:08x}"); }
    hex
}

fn skill_digest() -> &'static str {
    SKILL_DIGEST.get_or_init(|| format!("sha256:{}", sha256_hex(SKILL_MD.as_bytes())))
}

fn skill_description() -> String {
    let mut in_fm = false;
    let mut in_desc = false;
    let mut parts = Vec::new();
    for line in SKILL_MD.lines() {
        let trim = line.trim();
        if trim == "---" {
            if in_fm { break; }
            in_fm = true;
            continue;
        }
        if !in_fm { continue; }
        if in_desc {
            if line.starts_with(' ') || line.starts_with('\t') {
                if !trim.is_empty() { parts.push(trim); }
                continue;
            }
            break;
        }
        if let Some(rest) = trim.strip_prefix("description:") {
            let val = rest.trim();
            if val == ">" || val == "|" || val.is_empty() { in_desc = true; } else { return val.into(); }
        }
    }
    if parts.is_empty() {
        "Local pre-flight code audit, secret guard, symbol graph, and atomic MVCC refactoring engine.".into()
    } else {
        parts.join(" ")
    }
}

fn refit_skill_obj() -> Value {
    json!({
        "uri": SKILL_URI,
        "frontmatter": { "name": "refit", "description": skill_description() },
        "resources": [{ "uri": SKILL_MD_URI, "size": SKILL_MD.len(), "digest": skill_digest() }],
    })
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

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
enum MrtrError {
    #[error("malformed continuationToken")]
    Malformed,
    #[error("invalid continuationToken signature")]
    InvalidSignature,
    #[error("continuationToken expired")]
    Expired,
}

#[derive(Clone)]
struct MrtrSigner { key: [u8; 32] }

impl MrtrSigner {
    fn from_seed(seed: &[u8]) -> Self {
        Self { key: blake3::derive_key("refit mcp token key", seed) }
    }

    fn for_repo(repo_root: &Path) -> Self {
        if let Ok(secret) = std::env::var("REFIT_MCP_TOKEN_SECRET") && !secret.trim().is_empty() {
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
        let sig_arr: [u8; 32] = sig_bytes.try_into().map_err(|_| MrtrError::InvalidSignature)?;
        if blake3::keyed_hash(&self.key, &payload_bytes) != blake3::Hash::from_bytes(sig_arr) {
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
    fn is_terminal(self) -> bool { matches!(self, Self::Completed | Self::Cancelled | Self::Failed) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskRecord {
    task_id: String, tool: String, status: TaskState, created_ms: u64, updated_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")] progress: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] status_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] continuation_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")] error: Option<String>,
}

struct TaskStore {
    dir: PathBuf, state_lock: Mutex<()>, cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl TaskStore {
    fn for_repo(repo_root: &Path) -> Self {
        Self {
            dir: repo_root.join(".refit").join("tasks"),
            state_lock: Mutex::new(()), cancel_flags: Mutex::new(HashMap::new()),
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
        write_atomic(&path, &serde_json::to_vec_pretty(rec).context("encode task record")?)
    }

    fn load(&self, task_id: &str) -> Option<TaskRecord> {
        serde_json::from_str(&fs::read_to_string(self.record_path(task_id)?).ok()?).ok()
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
            if let Some(flag) = self.cancel_flags.lock().remove(task_id) { flag.store(true, Ordering::Release); }
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
        self.clear_flag(task_id);
        let _ = self.mutate_task(task_id, |rec| {
            if rec.status.is_terminal() { return Ok(false); }
            match outcome {
                Ok(val) => {
                    rec.status = TaskState::Completed;
                    rec.progress = Some(1.0);
                    rec.result = Some(val);
                }
                Err(err) => { rec.status = TaskState::Failed; rec.error = Some(err); }
            }
            Ok(true)
        });
    }
}

// Typed RPC & tool argument structs

#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct ProtoMeta { protocol_version: Option<String> }
#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct ProtoParams { protocol_version: Option<String>, #[serde(rename = "_meta")] meta: ProtoMeta }
#[derive(Deserialize, Default)]
#[serde(default)]
struct RpcReq { id: Option<Value>, method: String, params: Value, #[serde(rename = "_meta")] meta: ProtoMeta }
#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct CallParams {
    name: Option<String>, arguments: Value, task: Option<bool>, mode: Option<String>,
    #[serde(alias = "requestState")] continuation_token: Option<String>,
    input_responses: Option<Value>,
}
#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct TaskParams {
    #[serde(alias = "task_id")] task_id: String, status: Option<Value>, progress: Option<f64>,
    #[serde(alias = "status_message")] status_message: Option<String>,
    continuation_token: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputResponses { Map { confirm: Option<bool> }, List(Vec<InputFieldResp>) }
#[derive(Deserialize)]
struct InputFieldResp { name: String, value: Option<bool> }

#[derive(Deserialize, Default)]
#[serde(default)]
struct ToolArgs {
    path: String, fast: Option<bool>, markdown: bool, research: bool, dry_run: bool,
    confirmed: bool, require_confirmation: bool, allow_abi_change: Option<bool>, async_task: bool,
    symbol: Option<String>, file: Option<String>, token: Option<String>, trigger: Option<String>,
    language: Option<String>, capability: Option<String>, start_line: Option<u64>,
    end_line: Option<u64>, module_id: String, patch_text: String,
    op: Option<String>, deps: bool, staged: bool, diff: Option<String>, no_cache: bool,
    kane: Option<String>, kane_op: Option<String>, url: Option<String>, policy: Option<String>,
    target: Option<String>, app: Option<String>, remote: bool, headless: Option<bool>,
    bug_detection: bool, no_adaptive_heal: bool, timeout_secs: Option<u64>, verify_docs: bool,
}

#[derive(Serialize)]
struct ReadResp<'a> {
    path: &'a str, start_line: usize, end_line: usize, total_lines: usize,
    content: String, epoch: u64, file_epoch: u64,
    invisible_unicode_stripped: usize, taint: &'static str,
}
#[derive(Serialize)]
struct VerifyResp {
    scan_cache_hit: bool, tests_total: usize, tests_failed: usize,
    ruff_errors: usize, ty_errors: usize, hygiene_median: f64, hygiene_p10: f64,
}

// Context and epoch memoization

struct EpochMemo<T> { epoch: u64, stat_sig: u64, stored_at: Instant, value: T }

impl<T> EpochMemo<T> {
    fn new(epoch: u64, stat_sig: u64, value: T) -> Self {
        Self { epoch, stat_sig, stored_at: Instant::now(), value }
    }
    fn fresh(&self, epoch: u64, stat_sig: u64) -> bool {
        self.epoch == epoch && self.stat_sig == stat_sig && self.stored_at.elapsed() < EPOCH_MEMO_TTL
    }
}

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
            epoch, scan_memo: Mutex::new(None), symbols_memo: Mutex::new(None),
            #[cfg(test)] test_delay_ms: AtomicU64::new(0),
        })
    }

    fn snapshot_epoch(&self, tool: &str, args: &Value) -> (u64, u64) {
        let a: ToolArgs = parse_args(args);
        let key = if a.path.is_empty() { tool } else { a.path.as_str() };
        (self.epoch.global_epoch(), self.epoch.slot_epoch(key))
    }
}

// JSON-RPC core

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proto { Base, Stateless }

fn detect_protocol(req: &Value) -> Proto {
    let env: RpcReq = parse_args(req);
    let pp: ProtoParams = parse_args(&env.params);
    let is_stateless = [
        env.meta.protocol_version.as_deref(),
        pp.meta.protocol_version.as_deref(),
        pp.protocol_version.as_deref(),
    ]
    .contains(&Some(PROTO_STATELESS))
        || env.method == "server/discover"
        || env.method.starts_with("tasks/");
    if is_stateless { Proto::Stateless } else { Proto::Base }
}

fn rpc_fail(
    status: StatusCode, id: Option<Value>, code: i64, msg: impl Into<String>,
) -> (StatusCode, Option<Value>) {
    (status, Some(rpc_err(id.unwrap_or(Value::Null), code, msg)))
}

fn batch_err() -> Value {
    rpc_err(Value::Null, ERR_INVALID_REQUEST, "Invalid Request: JSON-RPC batch arrays are forbidden")
}

fn discover_result() -> Value {
    json!({
        "protocolVersion": PROTO_STATELESS, "supportedProtocolVersions": [PROTO_STATELESS, PROTO_BASE],
        "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
        "instructions": MCP_INSTRUCTIONS,
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": {},
            "extensions": {
                "io.modelcontextprotocol/tasks": { "methods": ["tasks/get", "tasks/update", "tasks/cancel"] },
                "io.modelcontextprotocol/mrtr": { "stateless": true },
                "io.modelcontextprotocol/skills": { "directoryRead": false }
            }
        },
        "tools": tool_catalog(), "_meta": catalog_meta(),
    })
}

fn norm_key_bytes(s: &str) -> impl Iterator<Item = u8> + '_ {
    s.bytes().filter(|&c| c != b'_' && c != b'-').map(|c| c.to_ascii_lowercase())
}

fn eq_param_key(a: &str, b: &str) -> bool { norm_key_bytes(a).eq(norm_key_bytes(b)) }

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
        let hdr_val = raw_val.to_str().map(str::trim).map_err(|_| format!("non-UTF8 value in header `{lower}`"))?;
        let actual = [args_obj, params_obj].into_iter().flatten()
            .find_map(|m| m.iter().find(|(k, _)| k.as_str() == suffix || eq_param_key(k, suffix)).map(|(_, v)| v))
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

fn estimate_tool_bytes(name: &str, args: &Value) -> usize {
    let light_tx = name == "refit_tx"
        && matches!(args.get("op").and_then(Value::as_str), Some("abort" | "begin"));
    if light_tx
        || matches!(name, "refit_read" | "refit_symbols" | "refit_search" | "refit_check" | "refit_tx_abort")
    {
        LIGHT_TOOL_RESERVE_BYTES
    } else {
        HEAVY_TOOL_RESERVE_BYTES
    }
}

fn with_stateless_meta(mut res: Value, mode: Proto) -> Value {
    if mode == Proto::Stateless && let Some(m) = res.as_object_mut() {
        m.insert("_meta".into(), catalog_meta());
    }
    res
}

fn handle_skills_or_resources(
    method: &str, id: Option<Value>, params: &Value, mode: Proto,
) -> (StatusCode, Option<Value>) {
    let uri = params.get("uri").and_then(Value::as_str).unwrap_or("");
    match method {
        "skills/list" => (
            StatusCode::OK,
            Some(rpc_ok(id, with_stateless_meta(json!({ "skills": [refit_skill_obj()] }), mode))),
        ),
        "skills/get" if uri == SKILL_URI || uri == SKILL_MD_URI => (
            StatusCode::OK,
            Some(rpc_ok(id, with_stateless_meta(refit_skill_obj(), mode))),
        ),
        "skills/get" => rpc_fail(StatusCode::OK, id, ERR_INVALID_PARAMS, format!("unknown skill uri `{uri}`")),
        "resources/list" => (
            StatusCode::OK,
            Some(rpc_ok(id, with_stateless_meta(json!({
                "resources": [{
                    "uri": SKILL_MD_URI, "name": "refit/SKILL.md",
                    "mimeType": "text/markdown", "size": SKILL_MD.len(),
                }]
            }), mode))),
        ),
        "resources/read" if uri == SKILL_MD_URI => (
            StatusCode::OK,
            Some(rpc_ok(id, with_stateless_meta(json!({
                "contents": [{ "uri": SKILL_MD_URI, "mimeType": "text/markdown", "text": SKILL_MD }]
            }), mode))),
        ),
        _ => rpc_fail(StatusCode::OK, id, ERR_INVALID_PARAMS, format!("unknown resource uri `{uri}`")),
    }
}

async fn handle_rpc(ctx: &Arc<McpCtx>, req: &Value, mode: Proto) -> (StatusCode, Option<Value>) {
    if req.is_array() { return (StatusCode::BAD_REQUEST, Some(batch_err())); }
    let env: RpcReq = parse_args(req);
    let (id, params) = (env.id, &env.params);
    match env.method.as_str() {
        "server/discover" => (StatusCode::OK, Some(rpc_ok(id, discover_result()))),
        "ping" => (StatusCode::OK, Some(rpc_ok(id, json!({})))),
        "notifications/initialized" => (StatusCode::ACCEPTED, None),
        "initialize" => {
            let stateless = mode == Proto::Stateless;
            let tools_cap = if stateless { json!({ "listChanged": false }) } else { json!({}) };
            let res = json!({
                "protocolVersion": if stateless { PROTO_STATELESS } else { PROTO_BASE },
                "instructions": MCP_INSTRUCTIONS,
                "capabilities": {
                    "tools": tools_cap,
                    "resources": {},
                    "extensions": { "io.modelcontextprotocol/skills": { "directoryRead": false } }
                },
                "serverInfo": { "name": "refit", "version": env!("CARGO_PKG_VERSION") },
            });
            (StatusCode::OK, Some(rpc_ok(id, with_stateless_meta(res, mode))))
        }
        "tools/list" => (
            StatusCode::OK,
            Some(rpc_ok(id, with_stateless_meta(json!({ "tools": tool_catalog() }), mode))),
        ),
        "tools/call" => handle_tool_call(ctx, id, params, mode).await,
        "skills/list" | "skills/get" | "resources/list" | "resources/read" => {
            handle_skills_or_resources(&env.method, id, params, mode)
        }
        "tasks/get" | "tasks/update" | "tasks/cancel" => {
            (StatusCode::OK, Some(handle_tasks(ctx, id, &env.method, params)))
        }
        "tasks/list" | "tasks/result" => rpc_fail(
            StatusCode::OK, id, ERR_METHOD_NOT_FOUND,
            format!("Method not found: `{}` is not supported", env.method),
        ),
        other => match id {
            Some(req_id) => rpc_fail(
                StatusCode::OK, Some(req_id), ERR_METHOD_NOT_FOUND, format!("Method not found: {other}"),
            ),
            None => (StatusCode::ACCEPTED, None),
        },
    }
}

async fn handle_tool_call(
    ctx: &Arc<McpCtx>, id: Option<Value>, params: &Value, mode: Proto,
) -> (StatusCode, Option<Value>) {
    let p: CallParams = parse_args(params);
    if mode == Proto::Stateless
        && let Some(tok) = p.continuation_token.as_deref().filter(|s| !s.is_empty())
    {
        return handle_mrtr_resume(ctx, id, &p, tok).await;
    }
    let name = p.name.unwrap_or_default();
    let args = p.arguments;
    if mode == Proto::Stateless {
        let f: ToolArgs = parse_args(&args);
        let op = f.op.as_deref();
        let destructive_tx = name == "refit_tx_delete"
            || (name == "refit_tx" && (op == Some("delete") || (op == Some("commit") && f.allow_abi_change == Some(true))));
        let needs_confirm = !f.confirmed
            && (f.require_confirmation || (name == "refit_apply" && f.allow_abi_change == Some(true)) || destructive_tx);
        if needs_confirm { return resume_challenge(ctx, id, &name, args); }
        if p.task == Some(true) || p.mode.as_deref() == Some("task") || f.async_task {
            return spawn_durable_task(ctx, id, name, args);
        }
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
        Err(err) => return rpc_fail(StatusCode::INTERNAL_SERVER_ERROR, id, -32603, err.to_string()),
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
    ctx: &Arc<McpCtx>, id: Option<Value>, params: &CallParams, token: &str,
) -> (StatusCode, Option<Value>) {
    let payload = match ctx.signer.verify(token, now_ms()) {
        Ok(p) => p,
        Err(err) => return rpc_fail(StatusCode::BAD_REQUEST, id, ERR_INVALID_PARAMS, err.to_string()),
    };
    if let Some(req_tool) = params.name.as_deref() && req_tool != payload.tool {
        let msg = format!("HeaderMismatchError: params.name `{req_tool}` mismatches token tool `{}`", payload.tool);
        return rpc_fail(StatusCode::BAD_REQUEST, id, ERR_HEADER_MISMATCH, msg);
    }
    let confirmed = params.input_responses.as_ref()
        .and_then(|v| serde_json::from_value::<InputResponses>(v.clone()).ok())
        .and_then(|r| match r {
            InputResponses::Map { confirm } => confirm,
            InputResponses::List(items) => items.into_iter().find(|it| it.name == "confirm").and_then(|it| it.value),
        });
    let Some(confirmed) = confirmed else {
        return rpc_fail(StatusCode::BAD_REQUEST, id, ERR_INVALID_PARAMS, "missing boolean `confirm` inside `inputResponses`");
    };
    if !confirmed {
        let declined = json!({ "cancelled": true, "reason": "user declined confirmation" });
        return (StatusCode::OK, Some(wrap_tool_reply(id, &declined, Proto::Stateless, 0, 0)));
    }
    let mut args = payload.arguments;
    if let Some(obj) = args.as_object_mut() { obj.insert("confirmed".into(), Value::Bool(true)); }
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
    if cancel.is_some_and(|f| f.load(Ordering::Acquire)) { return Err("cancelled before execution".into()); }
    if name == "refit_search" {
        let a: ToolArgs = parse_args(&args);
        let lang = a.language.as_deref().unwrap_or("Rust");
        let cap = a.capability.as_deref().unwrap_or("rate limiter");
        return Ok(ok_json(ctx.research.check_library(lang, cap).await));
    }
    if name == "refit_audit" && parse_args::<ToolArgs>(&args).research {
        return Ok(run_audit_async(ctx, &args).await);
    }
    if name == "refit_check" {
        let a: ToolArgs = parse_args(&args);
        if a.deps || a.kane.is_some() || a.kane_op.is_some() || a.verify_docs {
            return Ok(run_check_async(ctx, &args).await);
        }
    }
    let _sem = ctx.blocking_sem.acquire().await.ok();
    if cancel.is_some_and(|f| f.load(Ordering::Acquire)) { return Err("cancelled before execution".into()); }
    let cc = ctx.clone();
    tokio::task::spawn_blocking(move || {
        let _wait = wait_reserve
            .and_then(|b| MemGate::global().reserve_or_wait(b, Duration::from_secs(5)).ok());
        dispatch_tool_sync(&name, &args, &cc)
    })
    .await
    .map_err(|e| format!("task join error: {e}"))
}

fn try_admit_tool(name: &str, args: &Value) -> (usize, Option<impl Drop + Send + 'static>) {
    let gate = MemGate::global();
    let reserve = estimate_tool_bytes(name, args).min(gate.capacity());
    (reserve, gate.try_admit(reserve).ok())
}

async fn run_tool(
    ctx: &Arc<McpCtx>, id: Option<Value>, name: String, args: Value, mode: Proto,
) -> (StatusCode, Option<Value>) {
    let (reserve, pre_permit) = try_admit_tool(&name, &args);
    if mode == Proto::Stateless && pre_permit.is_none() {
        return rpc_fail(StatusCode::SERVICE_UNAVAILABLE, id, ERR_OVERLOADED, "server busy: MemGate limit");
    }
    let wait_reserve = (mode == Proto::Base && pre_permit.is_none()).then_some(reserve);
    let (tool_ep, args_ep) = (name.clone(), args.clone());
    let payload = {
        let _held = pre_permit;
        execute_tool_async(ctx, name, args, None, wait_reserve).await.unwrap_or_else(err_val)
    };
    let (epoch, slot_epoch) = ctx.snapshot_epoch(&tool_ep, &args_ep);
    (StatusCode::OK, Some(wrap_tool_reply(id, &payload, mode, epoch, slot_epoch)))
}

fn wrap_tool_reply(id: Option<Value>, payload: &Value, mode: Proto, epoch: u64, slot_epoch: u64) -> Value {
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
    let (_, Some(permit)) = try_admit_tool(&name, &args) else {
        return rpc_fail(StatusCode::SERVICE_UNAVAILABLE, id, ERR_OVERLOADED, "server busy: MemGate denied");
    };
    let (record, cancel_flag) = match ctx.tasks.create(&name, Some("queued for execution".into())) {
        Ok(pair) => pair,
        Err(err) => {
            drop(permit);
            return rpc_fail(StatusCode::INTERNAL_SERVER_ERROR, id, -32603, err.to_string());
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

fn handle_tasks(ctx: &Arc<McpCtx>, id: Option<Value>, method: &str, params: &Value) -> Value {
    let p: TaskParams = parse_args(params);
    let res = match method {
        "tasks/get" => ctx.tasks.load(&p.task_id).ok_or_else(|| format!("unknown taskId `{}`", p.task_id)),
        "tasks/cancel" => ctx.tasks.cancel(&p.task_id),
        _ => match p.status.map(serde_json::from_value::<TaskState>).transpose() {
            Ok(st) => ctx.tasks.update(&p.task_id, st, p.progress, p.status_message, p.continuation_token),
            Err(e) => Err(format!("invalid task status: {e}")),
        },
    };
    match res {
        Ok(rec) => rpc_ok(id, json!(rec)),
        Err(msg) => rpc_err(id.unwrap_or(Value::Null), ERR_INVALID_PARAMS, msg),
    }
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
            if stdout.write_all(line.as_bytes()).await.is_err() || stdout.flush().await.is_err() { break; }
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
    let env: RpcReq = parse_args(req);
    if env.method == "tools/list" && mode == Proto::Base {
        let _ = out_tx.send(encode_base_tools(env.id)).await;
        return Ok(());
    }
    if env.method == "tools/call" {
        let is_scan = parse_args::<CallParams>(&env.params).name.as_deref() == Some("refit_scan");
        let wait_prev = if is_scan { last_scan.take() } else { None };
        let done_tx = is_scan.then(|| {
            let (tx, rx) = oneshot::channel();
            *last_scan = Some(rx);
            tx
        });
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
    json_response(StatusCode::OK, json!({
        "ok": true, "protocolVersion": PROTO_STATELESS, "stateless": true, "catalogDigest": catalog_digest(),
    }), false)
}

async fn handle_http_mcp(State(ctx): State<Arc<McpCtx>>, headers: HeaderMap, body: Bytes) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => return json_response(StatusCode::BAD_REQUEST, rpc_err(Value::Null, ERR_PARSE, format!("Parse error: {err}")), false),
    };
    if parsed.is_array() { return json_response(StatusCode::BAD_REQUEST, batch_err(), false); }
    if let Err(detail) = check_param_hdrs(&headers, &parsed) {
        let err_body = json!({
            "jsonrpc": "2.0", "id": parsed.get("id").cloned().unwrap_or(Value::Null),
            "error": { "code": ERR_HEADER_MISMATCH, "message": format!("HeaderMismatchError: {detail}"), "data": { "error": "HeaderMismatchError" } }
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

async fn bind_http_server(ctx: Arc<McpCtx>, listen_addr: &str) -> Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(listen_addr).await
        .with_context(|| format!("bind MCP HTTP server on {listen_addr}"))?;
    let bound = listener.local_addr()?;
    let app = build_http_router(ctx);
    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await { tracing::error!("MCP HTTP server stopped: {err}"); }
    });
    Ok((bound, handle))
}

/// Serve stateless MCP over Streamable HTTP (`POST /mcp`, `GET /health`).
pub async fn serve_http(repo_root: PathBuf, listen_addr: String) -> Result<()> {
    let (bound, handle) = bind_http_server(McpCtx::new(repo_root), &listen_addr).await?;
    tracing::info!("refit MCP stateless HTTP listening on http://{bound}/mcp");
    handle.await.context("MCP HTTP server join")
}

// Tool handlers

fn ok_json<T: Serialize>(res: Result<T>) -> Value {
    match res {
        Ok(val) => serde_json::to_value(val).unwrap_or_else(|_| json!({ "ok": true })),
        Err(e) => err_val(e.to_string()),
    }
}

fn dispatch_tx(tctx: &ToolCtx<'_>, args: &Value) -> Value {
    let op = args.get("op").and_then(Value::as_str).unwrap_or("");
    match op {
        "begin" => tx_begin(tctx.repo_root, args),
        "stage" => tx_mutate(tctx.repo_root, "stage", args),
        "create" => tx_mutate(tctx.repo_root, "create", args),
        "delete" => tx_mutate(tctx.repo_root, "delete", args),
        "move" => tx_mutate(tctx.repo_root, "move", args),
        "commit" => tx_commit(tctx, args),
        "abort" => tx_abort(tctx.repo_root, args),
        other => err_val(format!(
            "invalid refit_tx op `{other}` (expected begin|stage|create|delete|move|commit|abort)"
        )),
    }
}

fn resolve_kane_spec(a: &ToolArgs) -> Option<String> {
    a.kane.clone().or_else(|| match a.kane_op.as_deref() {
        Some("cover") => Some("cover".into()),
        Some("bugs") => Some("bugs".into()),
        Some("validate") if !a.path.is_empty() => Some(format!("validate:{}", a.path)),
        _ => None,
    })
}

async fn run_kane_for_mcp(
    ctx: &McpCtx, a: &ToolArgs, spec: String, static_ok: bool, warm_snap: Option<Arc<SymbolGraph>>,
) -> Option<Value> {
    if !static_ok {
        return Some(json!({
            "ok": false, "verdict": "short_circuited", "credits_consumed": 0.0,
            "summary": "skipped Kane CLI because static pre-flight blocked (0 credits spent)"
        }));
    }
    if let Some(u) = a.url.as_deref().filter(|s| !s.is_empty())
        && let Err(msg) = crate::gates::probe_url_reachable(u).await
    {
        return Some(json!({
            "ok": false, "verdict": "did_not_start", "exit_code": 2,
            "credits_consumed": 0.0, "summary": msg
        }));
    }
    let req = crate::gates::KaneRequest {
        spec, surface: None, url: a.url.clone(), policy: a.policy.clone(),
        target: a.target.clone(), app: a.app.clone(), tunnel_name: None,
        remote: a.remote, headless: a.headless.unwrap_or(true),
        bug_detection: a.bug_detection, no_adaptive_heal: a.no_adaptive_heal,
        timeout_secs: a.timeout_secs.unwrap_or(180),
    };
    let r = ctx.repo_root.clone();
    let rec = tokio::task::spawn_blocking(move || {
        let mut rec = crate::gates::run_kane(&r, &req);
        if !rec.ok {
            let mut diag = rec.summary.clone();
            for f in &rec.failed_steps { diag.push(' '); diag.push_str(&f.remark); }
            let (spans, aff) = crate::learn::rewind_kane_tokens(&r, warm_snap.as_deref(), &diag);
            rec.suspected_spans = spans.into_iter()
                .map(|(file, line, symbol)| crate::gates::CodeSpanHint { file, line, symbol }).collect();
            rec.rewind_affected_files = aff;
        }
        rec
    }).await;
    rec.ok().and_then(|v| serde_json::to_value(v).ok())
}

async fn run_check_async(ctx: &McpCtx, args: &Value) -> Value {
    let a: ToolArgs = parse_args(args);
    if a.kane_op.as_deref() == Some("status") {
        return crate::gates::read_kane_status(&ctx.repo_root);
    }
    let (repo_root, owned_args) = (ctx.repo_root.clone(), args.clone());
    let _sem_permit = ctx.blocking_sem.acquire().await.ok();
    let mut base = tokio::task::spawn_blocking(move || run_check(&repo_root, &owned_args))
        .await.unwrap_or_else(|e| err_val(format!("mcp check join error: {e}")));
    let osv = if a.deps {
        crate::research::scan_lockfiles_osv(&ctx.repo_root).await.unwrap_or_default()
    } else {
        Vec::new()
    };
    let need_graph = a.verify_docs || a.kane.is_some() || a.kane_op.is_some();
    let warm_snap = need_graph.then(|| cached_symbols(ctx).0);
    let doc_drift = if a.verify_docs {
        let (r, g) = (ctx.repo_root.clone(), warm_snap.clone());
        tokio::task::spawn_blocking(move || crate::untangle::scan_doc_drift_quick(&r, g.as_deref()))
            .await.unwrap_or_default()
    } else {
        Vec::new()
    };
    let static_ok = base.get("ok").and_then(Value::as_bool).unwrap_or(false) && osv.is_empty();
    let kane_val = match resolve_kane_spec(&a) {
        Some(spec) => run_kane_for_mcp(ctx, &a, spec, static_ok, warm_snap).await,
        None => None,
    };
    if let Some(obj) = base.as_object_mut() {
        let kane_ok = kane_val.as_ref().and_then(|v| v.get("ok")).and_then(Value::as_bool).unwrap_or(true);
        obj.insert("ok".into(), (static_ok && kane_ok).into());
        if a.deps {
            obj.insert("osv_count".into(), osv.len().into());
            obj.insert("osv_findings".into(), json!(osv));
        }
        if a.verify_docs {
            obj.insert("doc_drift_count".into(), doc_drift.len().into());
            obj.insert("doc_drift".into(), json!(doc_drift));
        }
        if let Some(kv) = kane_val { obj.insert("kane".into(), kv); }
    }
    base
}

fn run_check(repo_root: &Path, args: &Value) -> Value {
    let a: ToolArgs = parse_args(args);
    match crate::untangle::scan_repo_secrets(repo_root, a.staged, a.diff.as_deref(), a.no_cache) {
        Ok(rep) => json!({
            "ok": rep.findings.is_empty(),
            "mode": rep.mode,
            "files_checked": rep.files_checked,
            "cache_hits": rep.cache_hits,
            "duration_ms": rep.duration_ms,
            "rule_count": rep.rule_count,
            "findings_count": rep.findings.len(),
            "findings": rep.findings,
        }),
        Err(e) => err_val(e.to_string()),
    }
}

fn dispatch_tool_sync(name: &str, args: &Value, ctx: &McpCtx) -> Value {
    let repo_root = ctx.repo_root.as_path();
    let tctx = ToolCtx { repo_root, hygiene: &ctx.hygiene, gate_cfg: &ctx.gate_cfg };
    match name {
        "refit_scan" => run_scan(ctx, args),
        "refit_audit" => run_audit(repo_root, args, &ctx.hygiene),
        "refit_check" => run_check(repo_root, args),
        "refit_sustain" => run_sustain(repo_root, args, &ctx.hygiene),
        "refit_symbols" => run_symbols(ctx, args),
        "refit_read" => run_read(ctx, args),
        "refit_resume" => run_resume(repo_root),
        "refit_rewind" => run_rewind(ctx, args),
        "refit_verify" => run_verify(repo_root, &ctx.hygiene),
        "refit_tx" => dispatch_tx(&tctx, args),
        "refit_tx_begin" => tx_begin(repo_root, args),
        "refit_tx_create" => tx_mutate(repo_root, "create", args),
        "refit_tx_delete" => tx_mutate(repo_root, "delete", args),
        "refit_tx_move" => tx_mutate(repo_root, "move", args),
        "refit_tx_stage" => tx_mutate(repo_root, "stage", args),
        "refit_tx_commit" => tx_commit(&tctx, args),
        "refit_tx_abort" => tx_abort(repo_root, args),
        "refit_apply" => run_apply(&tctx, args),
        other => err_val(format!("unknown tool `{other}`")),
    }
}

fn attach_epoch_meta(val: &mut Value, hit: bool, epoch: u64) {
    if let Some(obj) = val.as_object_mut() {
        obj.insert("epoch_cache_hit".into(), hit.into());
        obj.insert("epoch".into(), epoch.into());
    }
}

fn run_scan(ctx: &McpCtx, args: &Value) -> Value {
    let fast = parse_args::<ToolArgs>(args).fast.unwrap_or(true);
    let (ep, sig) = (ctx.epoch.global_epoch(), quick_repo_stat_sig(&ctx.repo_root));
    if let Some((cached_fast, ref memo)) = *ctx.scan_memo.lock()
        && cached_fast == fast && memo.fresh(ep, sig)
    {
        let mut v = memo.value.clone();
        if let Some(obj) = v.as_object_mut() { obj.insert("scan_cache_hit".into(), true.into()); }
        attach_epoch_meta(&mut v, true, ep);
        return v;
    }
    let (bundle, hit) = match scan_cached(&ctx.repo_root, true, fast, &ctx.hygiene) {
        Ok(pair) => pair,
        Err(e) => return err_val(e.to_string()),
    };
    let mut v = serde_json::to_value(&bundle.plan).unwrap_or_else(|_| json!({ "ok": true }));
    if let Some(obj) = v.as_object_mut() {
        obj.insert("scan_cache_hit".into(), hit.into());
        obj.insert("fast_mode".into(), fast.into());
        obj.insert("detected_languages".into(), json!(bundle.detected_languages));
        obj.insert("polyglot_toolchains".into(), json!(bundle.polyglot_toolchains));
    }
    attach_epoch_meta(&mut v, false, ep);
    if ctx.epoch.global_epoch() == ep {
        let sig_after = quick_repo_stat_sig(&ctx.repo_root);
        *ctx.scan_memo.lock() = Some((fast, EpochMemo::new(ep, sig_after, v.clone())));
    }
    v
}

fn peek_symbols_memo(ctx: &McpCtx, ep: u64, sig: u64) -> Option<Arc<SymbolGraph>> {
    ctx.symbols_memo.lock()
        .as_ref()
        .filter(|m| m.fresh(ep, sig))
        .map(|m| m.value.clone())
}

fn cached_symbols(ctx: &McpCtx) -> (Arc<SymbolGraph>, u64, bool) {
    let (ep, sig) = (ctx.epoch.global_epoch(), quick_repo_stat_sig(&ctx.repo_root));
    if let Some(snap) = peek_symbols_memo(ctx, ep, sig) { return (snap, ep, true); }
    let snap = Arc::new(load_symbols(&ctx.repo_root, None));
    if ctx.epoch.global_epoch() == ep {
        let sig_now = quick_repo_stat_sig(&ctx.repo_root);
        *ctx.symbols_memo.lock() = Some(EpochMemo::new(ep, sig_now, snap.clone()));
    }
    (snap, ep, false)
}

fn run_symbols(ctx: &McpCtx, args: &Value) -> Value {
    let a: ToolArgs = parse_args(args);
    let (sym, file) = (a.symbol.as_deref(), a.file.as_deref());
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
    let a: ToolArgs = parse_args(args);
    let trigger = crate::learn::RewindTrigger::parse(a.trigger.as_deref().unwrap_or("smell"));
    let (snap, ep, hit) = cached_symbols(ctx);
    let mut val = ok_json(crate::learn::query_rewind_graph(
        &ctx.repo_root, Some(&snap), a.token.as_deref(), trigger,
    ));
    attach_epoch_meta(&mut val, hit, ep);
    val
}

fn format_audit_value(rep: crate::types::AuditReport, markdown: bool) -> Value {
    if !markdown { return ok_json(Ok(rep)); }
    json!({
        "markdown": render_audit(&rep),
        "summary": {
            "clones": rep.clone_clusters.len(), "dead_symbols": rep.dead_symbols.len(),
            "dep_advisories": rep.dep_advisories.len(), "god_files": rep.god_files.len(),
            "infra_findings": rep.infra_findings.len(), "unused_deps": rep.unused_deps.len()
        }
    })
}

async fn run_audit_async(ctx: &McpCtx, args: &Value) -> Value {
    let a: ToolArgs = parse_args(args);
    let (repo_root, hygiene) = (ctx.repo_root.clone(), ctx.hygiene.clone());
    let _sem_permit = ctx.blocking_sem.acquire().await.ok();
    let base = tokio::task::spawn_blocking(move || untangle_repo(&repo_root, &hygiene))
        .await.map_err(|e| anyhow::anyhow!("mcp audit join error: {e}")).and_then(|r| r);
    let mut rep = match base { Ok(r) => r, Err(e) => return err_val(e.to_string()) };
    rep.dep_advisories = ctx.research.audit_deps_full(&ctx.repo_root).await.0;
    format_audit_value(rep, a.markdown)
}

fn run_audit(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    let a: ToolArgs = parse_args(args);
    let mut rep = match untangle_repo(repo_root, hygiene) { Ok(r) => r, Err(e) => return err_val(e.to_string()) };
    if a.research {
        rep.dep_advisories = crate::research::deps::audit_lockfiles(repo_root, &rustc_hash::FxHashMap::default()).0;
    }
    format_audit_value(rep, a.markdown)
}

fn run_sustain(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    let a: ToolArgs = parse_args(args);
    ok_json(untangle_repo(repo_root, hygiene).and_then(|r| write_sustain(repo_root, &r, a.dry_run)))
}

fn run_read(ctx: &McpCtx, args: &Value) -> Value {
    use std::io::BufRead;
    let a: ToolArgs = parse_args(args);
    let repo_root = ctx.repo_root.as_path();
    if !check_target(repo_root, &a.path) { return err_val(format!("unsafe path `{}`", a.path)); }
    let (epoch, file_epoch) = (ctx.epoch.global_epoch(), ctx.epoch.slot_epoch(&a.path));
    let Ok(file) = fs::File::open(repo_root.join(&a.path)) else {
        return err_val(format!("file not found `{}`", a.path));
    };
    let start = a.start_line.unwrap_or(1).max(1) as usize;
    let default_end = start.saturating_add(199);
    let req_end = a.end_line.map_or(default_end, |e| e as usize).min(start.saturating_add(400));
    let (mut reader, mut buf, mut total, mut kept) = (std::io::BufReader::new(file), String::new(), 0usize, Vec::new());
    while reader.read_line(&mut buf).is_ok_and(|n| n > 0) {
        total += 1;
        if total >= start && total <= req_end { kept.push(format!("{total}: {}", buf.trim_end_matches(['\n', '\r']))); }
        buf.clear();
    }
    let end = req_end.min(total);
    let raw_content = if start <= end { kept.join("\n") } else { String::new() };
    let (content, invisible_unicode_stripped) = crate::platform::sanitize_invisible_unicode(&raw_content);
    ok_json(Ok(ReadResp {
        path: &a.path, start_line: start, end_line: end, total_lines: total,
        content, epoch, file_epoch, invisible_unicode_stripped, taint: "untrusted_repo_file",
    }))
}

fn run_resume(repo_root: &Path) -> Value {
    ok_json(MvccStore::open(repo_root, "mcp-resume").and_then(|s| s.recover_wal()).map(|rep| json!({
        "rolled_back_txs": rep.rolled_back_txs, "committed_modules": rep.committed_modules.len(), "head_rev": rep.head_rev
    })))
}

fn run_verify(repo_root: &Path, hygiene: &HygieneScorer) -> Value {
    ok_json(scan_cached(repo_root, true, false, hygiene).map(|(b, hit)| VerifyResp {
        scan_cache_hit: hit, tests_total: b.audit.tests_total, tests_failed: b.audit.tests_failed,
        ruff_errors: b.audit.ruff_errors, ty_errors: b.audit.ty_errors,
        hygiene_median: b.audit.hygiene_median, hygiene_p10: b.audit.hygiene_p10,
    }))
}

fn run_apply(ctx: &ToolCtx<'_>, args: &Value) -> Value {
    let a: ToolArgs = parse_args(args);
    let (bundle, _) = match scan_cached(ctx.repo_root, true, true, ctx.hygiene) { Ok(b) => b, Err(e) => return err_val(e.to_string()) };
    let Some(mut unit) = bundle.plan.units.into_iter().find(|u| u.id == a.module_id) else {
        return err_val(format!("unknown module_id `{}`", a.module_id));
    };
    if let Some(allow) = a.allow_abi_change { unit.allow_abi_change = allow; }
    let (mut base_texts, mut base_hashes) = (BTreeMap::new(), BTreeMap::new());
    for rel in &unit.files {
        let txt = fs::read_to_string(ctx.repo_root.join(rel)).unwrap_or_default();
        base_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
        base_texts.insert(rel.clone(), txt);
    }
    let hunks = parse_patches(&a.patch_text);
    if let Err(v) = check_hunks(ctx.repo_root, &hunks, "passed", Some("asrf_parse")) { return v; }
    let applied = match apply_patches(&base_texts, &hunks) {
        Ok(c) => c,
        Err(e) => return fail_val("passed", e.to_string(), Some("asrf_parse")),
    };
    for rel in &applied.created { base_hashes.entry(rel.clone()).or_default(); }
    let req = CommitReq {
        ctx, unit: &unit, cands: &applied.updated, deleted: &applied.deleted,
        base: &base_hashes, dry_run: a.dry_run, tag: "mcp-apply", extra_dry: Map::new(), extra_ok: Map::new(),
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
        async fn rpc(&self, req: Value) -> (StatusCode, Value) {
            let (st, v) = handle_rpc(&self.ctx, &req, Proto::Stateless).await;
            (st, v.unwrap())
        }
        fn begin_tx(&self) -> String {
            let resp = self.call("refit_tx_begin", json!({}));
            let handle = resp["tx_handle"].as_str().unwrap();
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
    fn rpc_bare(id: i64, method: &str) -> Value { json!({"jsonrpc": "2.0", "id": id, "method": method}) }
    fn rpc_st(id: i64, method: &str, params: Option<Value>) -> Value {
        let mut v = json!({"jsonrpc": "2.0", "id": id, "method": method, "_meta": {"protocolVersion": PROTO_STATELESS}});
        if let Some(p) = params { v["params"] = p; }
        v
    }
    fn tool_rpc(id: i64, name: &str, arguments: Value) -> Value {
        rpc_req(id, "tools/call", json!({"name": name, "arguments": arguments}))
    }
    fn resume_rpc(id: i64, name: &str, tok: &str, confirm: bool) -> Value {
        rpc_req(id, "tools/call", json!({"name": name, "continuationToken": tok, "inputResponses": {"confirm": confirm}}))
    }
    fn mk_payload(tool: &str, path: &str, issued: u64, expires: u64, nonce: u64) -> ContinuationPayload {
        ContinuationPayload {
            tool: tool.into(), arguments: json!({"path": path}), prompt: "confirm".into(),
            issued_ms: issued, expires_ms: expires, nonce,
        }
    }
    fn rpc_txt(v: &Value) -> &str { v["result"]["content"][0]["text"].as_str().unwrap() }
    fn obj_keys(v: &Value) -> Vec<&str> {
        v.as_object().unwrap()
            .keys()
            .map(String::as_str)
            .collect()
    }

    async fn post_mcp(c: &reqwest::Client, url: &str, hdrs: &[(&str, &str)], body: &Value) -> reqwest::Response {
        let mut req = c.post(url);
        for &(k, v) in hdrs { req = req.header(k, v); }
        req.json(body).send().await.unwrap()
    }
    async fn post_json(c: &reqwest::Client, url: &str, hdrs: &[(&str, &str)], body: &Value) -> (u16, Value) {
        let resp = post_mcp(c, url, hdrs, body).await;
        (resp.status().as_u16(), resp.json().await.unwrap())
    }

    #[test]
    fn catalog_sorted_and_hashed() {
        let arr = tool_catalog().as_array().unwrap();
        let names: Vec<&str> = arr.iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!((arr.len(), &names), (12, &sorted));

        let (bytes, enc) = (catalog_bytes(), serde_json::to_vec(tool_catalog()).unwrap());
        let want = format!("blake3:{}", blake3::hash(bytes).to_hex());
        assert_eq!((catalog_digest(), bytes), (want.as_str(), enc.as_slice()));
        for tool in arr {
            assert_eq!(obj_keys(tool), vec!["annotations", "description", "inputSchema", "name"]);
            let ann = &tool["annotations"];
            assert!(ann["readOnlyHint"].is_boolean() && ann["destructiveHint"].is_boolean());
            assert!(ann["idempotentHint"].is_boolean() && ann["openWorldHint"].is_boolean());
            let pkeys = obj_keys(&tool["inputSchema"]["properties"]);
            let mut ps = pkeys.clone();
            ps.sort_unstable();
            assert_eq!(pkeys, ps);
        }
        let by_name = |n: &str| arr.iter().find(|t| t["name"] == n).unwrap();
        assert_eq!(by_name("refit_read")["annotations"]["readOnlyHint"], json!(true));
        assert_eq!(by_name("refit_search")["annotations"]["openWorldHint"], json!(true));
        assert_eq!(by_name("refit_tx")["annotations"]["destructiveHint"], json!(true));
        assert_eq!(by_name("refit_check")["annotations"]["readOnlyHint"], json!(true));

        let meta = catalog_meta();
        assert_eq!((&meta["ttlMs"], &meta["cacheScope"], &meta["catalogDigest"]), (&json!(3_600_000), &json!("server"), &json!(want)));
        let base: Value = serde_json::from_str(&encode_base_tools(Some(json!(7)))).unwrap();
        assert_eq!((&base["id"], &base["result"]["tools"], base["result"].get("_meta").is_none()), (&json!(7), tool_catalog(), true));
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

        for hit in [false, true] { assert_eq!(h.call("refit_scan", json!({}))["scan_cache_hit"], json!(hit)); }
        let dry_p = asrf_edit("src/calc.ts", "  return a + b;", "  const s = a + b;\n  return s;");
        h.check_dry(&h.apply("calc", &dry_p, true), "src/calc.ts", orig);

        h.repo.write("src/calc.ts", orig.replace("a + b;", "a + b + 5;"));
        let live_p = asrf_edit("src/calc.ts", "  return a + b + 5;", "  const v = a + b + 6;\n  return v;");
        assert_eq!(h.apply("calc", &live_p, false)["committed"], json!(true));
        assert!(h.repo.read("src/calc.ts").contains("a + b + 6"));

        let rw_a = json!({"token": "add", "trigger": "advisory"});
        let (rw, rw2) = (h.call("refit_rewind", rw_a.clone()), h.call("refit_rewind", rw_a));
        assert_eq!((&rw["epoch_cache_hit"], &rw["rewind"]["affected_files"], &rw2["epoch_cache_hit"]),
            (&json!(false), &json!(["src/calc.ts"]), &json!(true)));

        let (root, now) = (txs_dir(h.repo.path()), now_ms());
        let (stale, fresh) = (root.join("tx-stale-1"), root.join("tx-fresh-1"));
        for (dir, id, created, ttl) in [
            (&stale, "tx-stale-1", now.saturating_sub(120_000), 1_000),
            (&fresh, "tx-fresh-1", now, 600_000),
        ] {
            fs::create_dir_all(dir).unwrap();
            let m = TxMeta { tx_handle: id.into(), created_ms: created, ttl_ms: ttl, staged: BTreeMap::new() };
            fs::write(dir.join("meta.json"), serde_json::to_string(&m).unwrap()).unwrap();
        }
        assert_eq!(h.call("refit_tx_begin", json!({}))["swept_orphans"], json!(1));
        assert!(!stale.exists() && fresh.exists());
        let rd = h.call("refit_read", json!({"path": "src/b.ts", "start_line": 1, "end_line": 2}));
        let body_txt = rd["content"].as_str().unwrap();
        assert!(body_txt.contains("1: export function beta"));

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
        assert!(h.commit(&mv, false, true)["committed"] == json!(true) && !h.repo.has("src/b.ts") && h.repo.has("src/core/b.ts"));

        let cr = h.begin_tx();
        let body = "export function help(n: number): number {\n  return n + 1;\n}\n";
        h.call("refit_tx_create", json!({"tx_handle": cr, "path": "src/core/h.ts", "content": body}));
        assert_eq!(h.commit(&cr, false, false)["committed"], json!(true));
    }

    #[tokio::test]
    async fn stdio_and_http_framing() {
        let h = Harness::new("mcp_stdio", &[("src/ok.ts", "export function ok(): number { return 1; }\n")]);
        let (tx, mut rx) = mpsc::channel::<String>(32);
        let (mut tasks, mut scan) = (tokio::task::JoinSet::new(), None);
        macro_rules! rt {
            ($req:expr) => {{
                dispatch_stdio(&$req, &h.ctx, &tx, &mut tasks, &mut scan).await.unwrap();
                while tasks.join_next().await.is_some() {}
                serde_json::from_str::<Value>(&rx.recv().await.unwrap()).unwrap()
            }};
        }

        let init = rt!(rpc_req(1, "initialize", json!({"protocolVersion": PROTO_BASE})));
        assert_eq!(init["result"]["protocolVersion"], json!(PROTO_BASE));

        let notif = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        dispatch_stdio(&notif, &h.ctx, &tx, &mut tasks, &mut scan).await.unwrap();
        assert!(rx.try_recv().is_err());

        let list = rt!(rpc_bare(2, "tools/list"));
        assert!(list["result"].get("_meta").is_none() && list["result"]["tools"][0]["name"] == json!("refit_apply"));

        let call = rt!(tool_rpc(3, "refit_read", json!({"path": "src/ok.ts"})));
        assert!(call["result"].get("resultType").is_none());
        assert!(rpc_txt(&call).contains("export function ok"));

        let disc = rt!(rpc_st(4, "server/discover", None));
        assert_eq!((&disc["result"]["protocolVersion"], &disc["result"]["_meta"]["ttlMs"]), (&json!(PROTO_STATELESS), &json!(3_600_000)));
        let st = rt!(rpc_st(6, "tools/call", Some(json!({"name": "refit_read", "arguments": {"path": "src/ok.ts"}}))));
        assert_eq!(st["result"]["resultType"], json!("complete"));

        let batch = rt!(json!([rpc_bare(5, "ping")]));
        assert_eq!(batch["error"]["code"], json!(ERR_INVALID_REQUEST));
    }

    #[tokio::test]
    async fn http_discover_and_headers() {
        let h = Harness::new("mcp_http", &[("src/demo.rs", "pub fn demo() -> i32 { 42 }\n")]);
        let (addr, handle) = bind_http_server(h.ctx.clone(), "127.0.0.1:0").await.unwrap();
        let (url, client) = (format!("http://{addr}/mcp"), reqwest::Client::new());

        let disc_req = rpc_st(1, "server/discover", None);
        let resp = post_mcp(&client, &url, &[("Mcp-Method", "server/discover")], &disc_req).await;
        assert_eq!(resp.status().as_u16(), 200);
        assert!(resp.headers().get("mcp-session-id").is_none());
        let body: Value = resp.json().await.unwrap();
        let tools_arr = body["result"]["tools"].as_array().unwrap();
        let n_tools = tools_arr.len();
        assert_eq!((&body["result"]["protocolVersion"], &body["result"]["_meta"]["catalogDigest"], n_tools),
            (&json!(PROTO_STATELESS), &json!(catalog_digest()), 12));

        let (_, lr) = post_json(&client, &url, &[("Mcp-Method", "tools/list")], &rpc_bare(2, "tools/list")).await;
        assert_eq!((&lr["result"]["_meta"]["ttlMs"], &lr["result"]["_meta"]["cacheScope"]), (&json!(3_600_000), &json!("server")));

        assert_eq!(detect_protocol(&rpc_req(0, "initialize", json!({"protocolVersion": PROTO_STATELESS}))), Proto::Stateless);
        let call_req = tool_rpc(3, "refit_read", json!({"path": "src/demo.rs", "start_line": 1, "end_line": 3}));
        let call_hdrs = [
            ("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"),
            ("Mcp-Param-Path", "src/demo.rs"), ("Mcp-Param-Start-Line", "1"), ("Mcp-Param-EndLine", "3"),
        ];
        let (_, cv) = post_json(&client, &url, &call_hdrs, &call_req).await;
        assert!(cv["result"]["resultType"] == json!("complete") && rpc_txt(&cv).contains("pub fn demo"));

        let (mh, nr) = (("Mcp-Method", "tools/call"), ("Mcp-Name", "refit_read"));
        let bad_cases: &[(&[(&str, &str)], &Value)] = &[
            (&[], &disc_req), (&[("Mcp-Method", "tools/list")], &disc_req), (&[mh], &call_req),
            (&[mh, ("Mcp-Name", "refit_scan")], &call_req),
            (&[mh, nr, ("Mcp-Param-Path", "src/evil.rs")], &call_req), (&[mh, nr, ("Mcp-Param-Ghost", "1")], &call_req),
        ];
        for &(hdrs, payload) in bad_cases {
            let (st, v) = post_json(&client, &url, hdrs, payload).await;
            assert_eq!((st, &v["error"]["code"]), (400, &json!(ERR_HEADER_MISMATCH)));
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(msg.contains("HeaderMismatchError"));
        }

        let (bst, bv) = post_json(&client, &url, &[("Mcp-Method", "server/discover")], &json!([disc_req])).await;
        assert_eq!((bst, &bv["error"]["code"]), (400, &json!(ERR_INVALID_REQUEST)));
        for omitted in ["tasks/list", "tasks/result"] {
            let (_, r) = post_json(&client, &url, &[("Mcp-Method", omitted)], &rpc_bare(9, omitted)).await;
            assert_eq!(r["error"]["code"], json!(ERR_METHOD_NOT_FOUND));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn resume_token_sign_and_expire() {
        let h = Harness::new("mcp_tok", &[("src/ok.rs", "pub fn ok() -> i32 { 1 }\n")]);
        let (st, body) = h.rpc(tool_rpc(10, "refit_read", json!({"path": "src/ok.rs", "require_confirmation": true}))).await;
        assert_eq!((st, &body["result"]["resultType"]), (StatusCode::OK, &json!("input_required")));
        let tok_val = body["result"]["continuationToken"].as_str().unwrap();
        let token = tok_val.to_string();
        assert!(token.starts_with(MRTR_TOKEN_PREFIX));

        let mut tampered = token.clone();
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'A' { 'E' } else { 'A' });
        assert_eq!(h.ctx.signer.verify(&tampered, now_ms()), Err(MrtrError::InvalidSignature));

        let exp = mk_payload("refit_read", "src/ok.rs", 100, 200, 1);
        let signed_exp = h.ctx.signer.sign(&exp).unwrap();
        assert_eq!(h.ctx.signer.verify(&signed_exp, 201), Err(MrtrError::Expired));
        assert!(rpc_txt(&h.rpc(resume_rpc(12, "refit_read", &token, false)).await.1).contains("\"cancelled\": true"));

        let (st2, b2) = h.rpc(resume_rpc(11, "refit_read", &token, true)).await;
        assert_eq!((st2, &b2["result"]["resultType"]), (StatusCode::OK, &json!("complete")));
        assert!(rpc_txt(&b2).contains("pub fn ok"));
    }

    #[tokio::test]
    async fn resume_token_rejects_forgery() {
        let h = Harness::new("mcp_sec", &[("src/lib.rs", "pub fn x() {}\n")]);
        assert!(h.repo.path().join(".refit/.mrtr.key").is_file());
        let forged = MrtrSigner::from_seed(format!("refit-local-mrtr:{}", h.repo.path().display()).as_bytes());
        let now = now_ms();
        let payload = mk_payload("refit_tx_delete", "src/lib.rs", now, now + 60_000, 99);
        assert_eq!(h.ctx.signer.verify(&forged.sign(&payload).unwrap(), now_ms()), Err(MrtrError::InvalidSignature));

        let real = h.ctx.signer.sign(&payload).unwrap();
        let (st, resp) = h.rpc(resume_rpc(42, "refit_read", &real, true)).await;
        assert_eq!((st, &resp["error"]["code"]), (StatusCode::BAD_REQUEST, &json!(ERR_HEADER_MISMATCH)));
    }

    #[tokio::test]
    async fn task_store_states_and_cancel() {
        let h = Harness::new("mcp_tasks", &[("src/lib.rs", "pub fn hello() -> i32 { 7 }\n")]);
        let tid_of = |v: &Value| {
            let id = v["result"]["taskId"].as_str().unwrap();
            id.to_owned()
        };
        let (st, val) = h.rpc(rpc_req(20, "tools/call", json!({"name": "refit_read", "task": true, "arguments": {"path": "src/lib.rs"}}))).await;
        assert_eq!((st, &val["result"]["resultType"]), (StatusCode::OK, &json!("task")));
        let tid = tid_of(&val);
        assert!(h.repo.join(&format!(".refit/tasks/{tid}.json")).exists());

        let mut done = false;
        for _ in 0..50 {
            if h.rpc(rpc_req(21, "tasks/get", json!({"taskId": tid}))).await.1["result"]["status"] == json!("completed") {
                done = true; break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(done);

        h.set_delay_ms(250);
        let sid = tid_of(&h.rpc(rpc_req(22, "tools/call", json!({"name": "refit_read", "mode": "task", "arguments": {"path": "src/lib.rs"}}))).await.1);
        let cr = h.rpc(rpc_req(23, "tasks/cancel", json!({"taskId": sid}))).await.1;
        h.set_delay_ms(0);
        assert_eq!(cr["result"]["status"], json!("cancelled"));
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(h.ctx.tasks.load(&sid).unwrap().status, TaskState::Cancelled);

        let (rec, _) = h.ctx.tasks.create("refit_audit", Some("running".into())).unwrap();
        assert_eq!(rec.status, TaskState::Working);
        let upd = rpc_req(24, "tasks/update", json!({"taskId": rec.task_id, "status": "input_required", "progress": 0.5}));
        assert_eq!(h.rpc(upd).await.1["result"]["status"], json!("input_required"));
        let bad = rpc_req(25, "tasks/update", json!({"taskId": rec.task_id, "status": "bogus"}));
        assert_eq!(h.rpc(bad).await.1["error"]["code"], json!(ERR_INVALID_PARAMS));
        h.ctx.tasks.finish(&rec.task_id, Err("failed".into()));
        assert_eq!(h.ctx.tasks.load(&rec.task_id).unwrap().status, TaskState::Failed);
    }

    #[test]
    fn epoch_memo_and_invalidation() {
        let h = Harness::new("mcp_ep", &[("src/mod_a.rs", "pub fn initial_sym() -> i32 { 1 }\n")]);
        assert_eq!(h.call("refit_scan", json!({}))["epoch_cache_hit"], json!(false));
        let y1 = h.call("refit_symbols", json!({"symbol": "initial_sym"}));
        assert_eq!((&y1["epoch_cache_hit"], &y1["definers"]), (&json!(false), &json!(["src/mod_a.rs"])));

        let (s2, y2) = (h.call("refit_scan", json!({})), h.call("refit_symbols", json!({"symbol": "initial_sym"})));
        assert_eq!((&s2["epoch_cache_hit"], &s2["scan_cache_hit"], &y2["epoch_cache_hit"]), (&json!(true), &json!(true), &json!(true)));
        let rd = h.call("refit_read", json!({"path": "src/mod_a.rs"}));
        assert_eq!((&rd["epoch"], &rd["file_epoch"]), (&json!(1), &json!(1)));

        let store = MvccStore::open_with_epoch(h.repo.path(), "memo", h.ctx.epoch.clone()).unwrap();
        let exp = BTreeMap::from([("src/mod_a.rs".into(), h.repo.hash("src/mod_a.rs"))]);
        let next = b"pub fn initial_sym() -> i32 { 1 }\npub fn fresh_sym() -> i32 { 42 }\n".to_vec();
        let out = store.commit_tx("mod_a", &exp, &BTreeMap::from([("src/mod_a.rs".into(), next)]), &[]).unwrap();
        std::assert_matches!(out, crate::mvcc::CommitOutcome::Committed { head_rev: 1, .. });

        let (s3, y3) = (h.call("refit_scan", json!({})), h.call("refit_symbols", json!({"symbol": "fresh_sym"})));
        assert_eq!((&s3["epoch_cache_hit"], &s3["epoch"]), (&json!(false), &json!(2)));
        assert_eq!((&y3["epoch_cache_hit"], &y3["definers"]), (&json!(false), &json!(["src/mod_a.rs"])));
    }

    #[test]
    fn memgate_blocks_over_cap() {
        let gate = MemGate::new(1024);
        let over_err = gate.try_admit(2048).unwrap_err().to_string();
        assert!(over_err.contains("exceeds capacity"));
        let permit = gate.try_admit(768).unwrap();
        let busy_err = gate.try_admit(512).unwrap_err().to_string();
        assert!(busy_err.contains("busy"));
        drop(permit);
        let p2 = gate.try_admit(512).unwrap();
        assert_eq!(gate.used(), 512);
        drop(p2);
        assert_eq!(gate.used(), 0);
    }

    #[test]
    fn polymorphic_tx_and_check() {
        let orig = "export function mul(a: number, b: number): number {\n  return a * b;\n}\n";
        let h = Harness::new("mcp_poly_tx", &[("src/m.ts", orig)]);

        let chk = h.call("refit_check", json!({}));
        assert_eq!((&chk["ok"], &chk["findings_count"]), (&json!(true), &json!(0)));

        let id = h.call("refit_tx", json!({"op": "begin"}))["tx_handle"].as_str().unwrap().to_string();
        let patch = asrf_edit("src/m.ts", "  return a * b;", "  const out = a * b;\n  return out;");
        assert_eq!(h.call("refit_tx", json!({"op": "stage", "tx_handle": id, "patch_text": patch}))["staged_files_count"], json!(1));
        let done = h.call("refit_tx", json!({"op": "commit", "tx_handle": id, "dry_run": false}));
        assert_eq!((&done["committed"], &done["files_committed"]), (&json!(true), &json!(1)));
        assert!(h.repo.read("src/m.ts").contains("const out = a * b;"));

        let bad = h.call("refit_tx", json!({"op": "unknown_op"}));
        assert!(bad["error"].as_str().unwrap().contains("invalid refit_tx op"));
    }

    #[tokio::test]
    async fn skills_resources_and_instructions() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        let h = Harness::new("mcp_skills", &[("src/a.rs", "pub fn a() {}\n")]);

        let (_, init) = h.rpc(rpc_req(1, "initialize", json!({"protocolVersion": PROTO_STATELESS}))).await;
        assert!(init["result"]["instructions"].as_str().unwrap().contains("refit_check"));
        assert_eq!(
            init["result"]["capabilities"]["extensions"]["io.modelcontextprotocol/skills"]["directoryRead"],
            json!(false),
        );
        assert!(init["result"]["capabilities"]["resources"].is_object());

        let (_, slist) = h.rpc(rpc_bare(2, "skills/list")).await;
        let skill = &slist["result"]["skills"][0];
        let want_digest = format!("sha256:{}", sha256_hex(SKILL_MD.as_bytes()));
        assert_eq!((&skill["uri"], &skill["frontmatter"]["name"]), (&json!(SKILL_URI), &json!("refit")));
        assert_eq!(
            (&skill["resources"][0]["uri"], &skill["resources"][0]["size"], &skill["resources"][0]["digest"]),
            (&json!(SKILL_MD_URI), &json!(SKILL_MD.len()), &json!(want_digest)),
        );

        let (_, sget) = h.rpc(rpc_req(3, "skills/get", json!({"uri": SKILL_URI}))).await;
        assert_eq!(sget["result"]["uri"], json!(SKILL_URI));
        let (_, sget2) = h.rpc(rpc_req(4, "skills/get", json!({"uri": SKILL_MD_URI}))).await;
        assert_eq!(sget2["result"]["uri"], json!(SKILL_URI));
        let (_, sbad) = h.rpc(rpc_req(5, "skills/get", json!({"uri": "skill://nope"}))).await;
        assert_eq!(sbad["error"]["code"], json!(ERR_INVALID_PARAMS));

        let (_, rlist) = h.rpc(rpc_bare(6, "resources/list")).await;
        assert_eq!(rlist["result"]["resources"][0]["uri"], json!(SKILL_MD_URI));

        let (_, rread) = h.rpc(rpc_req(7, "resources/read", json!({"uri": SKILL_MD_URI}))).await;
        assert_eq!(rread["result"]["contents"][0]["text"], json!(SKILL_MD));
        let (_, rbad) = h.rpc(rpc_req(8, "resources/read", json!({"uri": "skill://missing"}))).await;
        assert_eq!(rbad["error"]["code"], json!(ERR_INVALID_PARAMS));
    }

    #[test]
    fn read_strips_invisible_unicode_and_marks_taint() {
        let poisoned = "let ok = 1;\u{E0041}\u{202E} // hidden\n";
        let h = Harness::new("mcp_taint", &[("src/p.rs", poisoned)]);
        let rd = h.call("refit_read", json!({"path": "src/p.rs"}));
        let content = rd["content"].as_str().unwrap();
        assert!(!content.contains('\u{E0041}') && !content.contains('\u{202E}'));
        assert!(content.contains("let ok = 1; // hidden"));
        assert!(rd["invisible_unicode_stripped"].as_u64().unwrap() >= 2);
        assert_eq!(rd["taint"], json!("untrusted_repo_file"));
    }
}
