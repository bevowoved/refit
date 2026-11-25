use crate::gates::{GateConfig, run_gates, scan_cached};
use crate::graph::{apply_patches, hash_abi, impacted_tests, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::net::{json_path, json_str};
use crate::platform::{SieveCache, check_target_in_repo, ensure_cachedir_tag, now_ms, write_atomic};
use crate::source::{count_lines, work_dir};
use crate::types::{ModuleUnit, PatchOp};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

static MCP_TX_SEQ: AtomicU64 = AtomicU64::new(1);
static TX_LOCKS: OnceLock<SieveCache<String, Arc<Mutex<()>>>> = OnceLock::new();

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StagedFile {
    pub base_blake3: String, pub original_content: String, pub staged_content: String,
    #[serde(default)] pub created: bool, #[serde(default)] pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxMeta {
    pub tx_handle: String, pub created_ms: u64, pub ttl_ms: u64,
    pub staged: BTreeMap<String, StagedFile>,
}

impl StagedFile {
    fn created(content: String) -> Self { Self { staged_content: content, created: true, ..Default::default() } }
    fn deleted(base_blake3: String, original_content: String) -> Self {
        Self { base_blake3, original_content, deleted: true, ..Default::default() }
    }
    fn edited(base_blake3: String, original_content: String, staged_content: String) -> Self {
        Self { base_blake3, original_content, staged_content, ..Default::default() }
    }
}

pub struct ToolCtx<'a> { pub repo_root: &'a Path, pub hygiene: &'a HygieneScorer, pub gate_cfg: &'a GateConfig }

pub struct CommitReq<'a> {
    pub ctx: &'a ToolCtx<'a>, pub unit: &'a ModuleUnit, pub cands: &'a BTreeMap<String, String>,
    pub deleted: &'a [String], pub base: &'a BTreeMap<String, String>, pub dry_run: bool,
    pub tag: &'a str, pub extra_dry: Map<String, Value>, pub extra_ok: Map<String, Value>,
}

pub fn txs_dir(repo_root: &Path) -> PathBuf { work_dir(repo_root, ".refit-tx", "refit-tx") }

fn lock_tx(handle: &str) -> Arc<Mutex<()>> {
    let cache = TX_LOCKS.get_or_init(|| SieveCache::new(4_096, 1024 * 1024));
    cache.get_or_insert_with(handle.to_string(), handle.len().saturating_add(64), || Arc::new(Mutex::new(())))
}

fn evict_tx_lock(handle: &str) {
    if let Some(cache) = TX_LOCKS.get() { cache.remove(&handle.to_string()); }
}

pub fn check_txid(handle: &str) -> bool {
    !handle.is_empty() && handle.len() <= 64
        && handle.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

pub fn check_target(repo_root: &Path, rel: &str) -> bool { check_target_in_repo(repo_root, rel) }

fn sweep_txs(tx_root: &Path, now: u64) -> usize {
    let Ok(rd) = fs::read_dir(tx_root) else { return 0 };
    let cutoff = SystemTime::now().checked_sub(Duration::from_secs(1800)).unwrap_or(SystemTime::UNIX_EPOCH);
    let mut swept = 0usize;
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let expired = fs::read_to_string(dir.join("meta.json")).ok()
            .and_then(|s| serde_json::from_str::<TxMeta>(&s).ok())
            .map(|m| now >= m.created_ms.saturating_add(m.ttl_ms))
            .unwrap_or_else(|| fs::metadata(&dir).and_then(|m| m.modified()).map_or(true, |t| t <= cutoff));
        if expired && fs::remove_dir_all(&dir).is_ok() {
            if let Some(name) = entry.file_name().to_str() { evict_tx_lock(name); }
            swept += 1;
        }
    }
    swept
}

fn write_tx(meta_path: &Path, meta: &TxMeta) -> std::result::Result<(), String> {
    let raw = serde_json::to_string_pretty(meta).map_err(|e| format!("encode tx: {e}"))?;
    write_atomic(meta_path, raw.as_bytes()).map_err(|e| format!("persist tx: {e}"))
}

fn upsert_created(meta: &mut TxMeta, path: &str, content: String) {
    if let Some(prev) = meta.staged.get_mut(path) {
        (prev.staged_content, prev.deleted) = (content, false);
    } else {
        meta.staged.insert(path.to_string(), StagedFile::created(content));
    }
}

fn with_tx(
    repo_root: &Path, handle: &str, err_key: &str, save: bool,
    f: impl FnOnce(&Path, &mut TxMeta) -> std::result::Result<Value, Value>,
) -> Value {
    if !check_txid(handle) { return json!({ err_key: false, "error": "invalid tx_handle" }); }
    let lock = lock_tx(handle);
    let _g = lock.lock();
    let tx_dir = txs_dir(repo_root).join(handle);
    let meta_path = tx_dir.join("meta.json");
    let raw_meta = fs::read_to_string(&meta_path).ok();
    let Some(mut meta) = raw_meta.and_then(|s| serde_json::from_str::<TxMeta>(&s).ok()) else {
        return json!({ err_key: false, "error": format!("unknown tx_handle `{handle}`") });
    };
    match f(&tx_dir, &mut meta) {
        Ok(resp) => {
            if save && let Err(e) = write_tx(&meta_path, &meta) {
                return json!({ err_key: false, "error": e });
            }
            resp
        }
        Err(err_val) => err_val,
    }
}

pub fn tx_begin(repo_root: &Path, args: &Value) -> Value {
    let now = now_ms();
    let ttl_secs = args["ttl_secs"].as_u64().unwrap_or(1800).clamp(1, 86_400);
    let ttl_ms = ttl_secs * 1000;
    let root = txs_dir(repo_root);
    let swept = sweep_txs(&root, now);
    let handle = format!("tx-{now}-{}", MCP_TX_SEQ.fetch_add(1, Ordering::Relaxed));
    let tx_dir = root.join(&handle);
    if let Err(e) = fs::create_dir_all(&tx_dir) { return json!({"error": format!("create tx dir: {e}")}); }
    ensure_cachedir_tag(&root);
    let meta = TxMeta { tx_handle: handle.clone(), created_ms: now, ttl_ms, staged: BTreeMap::new() };
    if let Err(e) = write_tx(&tx_dir.join("meta.json"), &meta) { return json!({"ok": false, "error": e}); }
    json!({ "tx_handle": handle, "created_ms": now, "ttl_ms": ttl_ms, "swept_orphans": swept })
}

pub fn tx_create(repo_root: &Path, args: &Value) -> Value {
    let (handle, path, content) = (args["tx_handle"].as_str().unwrap_or(""), json_path(args, "path"), json_str(args, "content"));
    if !check_target(repo_root, &path) { return json!({"ok": false, "error": format!("unsafe path `{path}`")}); }
    with_tx(repo_root, handle, "ok", true, |_, meta| {
        if repo_root.join(&path).exists() && !meta.staged.get(&path).is_some_and(|s| s.deleted) {
            return Err(json!({"ok": false, "error": format!("already exists: `{path}`")}));
        }
        upsert_created(meta, &path, content);
        Ok(json!({ "ok": true, "tx_handle": handle, "created": path, "staged_files_count": meta.staged.len() }))
    })
}

pub fn tx_delete(repo_root: &Path, args: &Value) -> Value {
    let (handle, path) = (args["tx_handle"].as_str().unwrap_or(""), json_path(args, "path"));
    if !check_target(repo_root, &path) { return json!({"ok": false, "error": format!("unsafe path `{path}`")}); }
    with_tx(repo_root, handle, "ok", true, |_, meta| {
        if meta.staged.get(&path).is_some_and(|s| s.created) {
            meta.staged.remove(&path);
        } else {
            let (hash, orig) = existing_base(repo_root, meta, &path)?;
            meta.staged.insert(path.clone(), StagedFile::deleted(hash, orig));
        }
        Ok(json!({ "ok": true, "tx_handle": handle, "deleted": path, "staged_files_count": meta.staged.len() }))
    })
}

pub fn tx_move(repo_root: &Path, args: &Value) -> Value {
    let (handle, from, to) = (args["tx_handle"].as_str().unwrap_or(""), json_path(args, "from_path"), json_path(args, "to_path"));
    if !check_target(repo_root, &from) || !check_target(repo_root, &to) {
        return json!({"ok": false, "error": "unsafe from_path or to_path"});
    }
    if from == to { return json!({"ok": false, "error": "from_path and to_path are identical"}); }
    with_tx(repo_root, handle, "ok", true, |_, meta| {
        if repo_root.join(&to).exists() && !meta.staged.get(&to).is_some_and(|s| s.deleted) {
            return Err(json!({"ok": false, "error": format!("destination exists: `{to}`")}));
        }
        let (content, created, orig, hash) = read_source(repo_root, meta, &from)?;
        if created {
            meta.staged.remove(&from);
        } else {
            meta.staged.insert(from.clone(), StagedFile::deleted(hash, orig));
        }
        upsert_created(meta, &to, content);
        Ok(json!({ "ok": true, "tx_handle": handle, "moved": [from, to], "staged_files_count": meta.staged.len() }))
    })
}

pub fn tx_stage(repo_root: &Path, args: &Value) -> Value {
    let handle = args["tx_handle"].as_str().unwrap_or("");
    let patch_text = args["patch_text"].as_str().unwrap_or("");
    with_tx(repo_root, handle, "ok", true, |_, meta| {
        let hunks = parse_patches(patch_text);
        if hunks.is_empty() { return Err(json!({"ok": false, "error": "no <<<REFIT_PATCH>>> hunks found"})); }
        check_hunks(repo_root, &hunks, "ok", None)?;
        let working = load_working(repo_root, meta, &hunks)?;
        let applied = apply_patches(&working, &hunks)
            .map_err(|e| json!({"ok": false, "failed_gate": "asrf_parse", "error": e.to_string()}))?;
        record_applied(meta, &working, applied);
        let files: Vec<String> = meta.staged.keys().cloned().collect();
        Ok(json!({ "ok": true, "tx_handle": handle, "staged_files_count": files.len(), "files": files }))
    })
}

pub fn tx_commit(ctx: &ToolCtx<'_>, args: &Value) -> Value {
    let handle = args["tx_handle"].as_str().unwrap_or("");
    let dry_run = args["dry_run"].as_bool().unwrap_or(false);
    let allow_abi = args["allow_abi_change"].as_bool().unwrap_or(false);
    let mut committed = false;
    let res = with_tx(ctx.repo_root, handle, "passed", false, |tx_dir, meta| {
        if meta.staged.is_empty() {
            return Err(json!({"passed": false, "error": "no staged files in transaction"}));
        }
        let (unit, cands, deleted, base) = build_unit(ctx.repo_root, handle, allow_abi, meta, ctx.hygiene);
        let extra_dry = Map::from_iter([
            ("tx_handle".into(), json!(handle)), ("staged_files_count".into(), json!(meta.staged.len())),
            ("deleted_files_count".into(), json!(deleted.len())),
        ]);
        let extra_ok = Map::from_iter([("tx_handle".into(), json!(handle)), ("files_deleted".into(), json!(deleted.len()))]);
        let req = CommitReq { ctx, unit: &unit, cands: &cands, deleted: &deleted, base: &base, dry_run, tag: "mcp-tx", extra_dry, extra_ok };
        let (dir_c, committed_ref) = (tx_dir.to_path_buf(), &mut committed);
        Ok(verify_commit(req, move || { let _ = fs::remove_dir_all(&dir_c); *committed_ref = true; }))
    });
    if committed { evict_tx_lock(handle); }
    res
}

pub fn tx_abort(repo_root: &Path, args: &Value) -> Value {
    let handle = args["tx_handle"].as_str().unwrap_or("");
    if !check_txid(handle) { return json!({"rolled_back": false, "error": "invalid tx_handle"}); }
    let tx_lock = lock_tx(handle);
    let guard = tx_lock.lock();
    let tx_dir = txs_dir(repo_root).join(handle);
    let existed = tx_dir.exists();
    let _ = fs::remove_dir_all(&tx_dir);
    drop(guard);
    evict_tx_lock(handle);
    json!({"rolled_back": existed, "tx_handle": handle})
}

pub fn check_hunks(
    repo_root: &Path, hunks: &[crate::types::AsrfHunk], err_key: &str, gate: Option<&str>,
) -> std::result::Result<(), Value> {
    for h in hunks {
        let bad = if !check_target(repo_root, &h.path) {
            Some(format!("unsafe path `{}`", h.path))
        } else {
            h.new_path.as_deref().filter(|np| !check_target(repo_root, np)).map(|np| format!("unsafe new_path `{np}`"))
        };
        if let Some(msg) = bad {
            let mut v = json!({ err_key: false, "error": msg });
            if let Some(g) = gate { v["failed_gate"] = json!(g); }
            return Err(v);
        }
    }
    Ok(())
}

pub fn verify_commit(mut req: CommitReq<'_>, on_commit: impl FnOnce()) -> Value {
    let verdict = match run_gates(req.ctx.gate_cfg, req.unit, req.cands, req.deleted, req.ctx.hygiene) {
        Ok(v) => v,
        Err(e) => return json!({"passed": false, "error": e.to_string()}),
    };
    if !verdict.passed {
        return serde_json::to_value(&verdict).unwrap_or_else(|_| json!({"passed": false}));
    }
    if req.dry_run {
        let mut out = json!({
            "passed": true, "dry_run": true, "committed": false,
            "gate0_syntax_ok": verdict.gate0_syntax_ok, "gate0_abi_ok": verdict.gate0_abi_ok,
            "gate2_score_before": verdict.gate2_score_before, "gate2_score_after": verdict.gate2_score_after,
            "gate3_ruff_errors": verdict.gate3_ruff_errors, "gate3_ty_errors": verdict.gate3_ty_errors,
            "diagnostics": verdict.diagnostics
        });
        if let Some(m) = out.as_object_mut() { m.append(&mut req.extra_dry); }
        return out;
    }
    let res = MvccStore::open(req.ctx.repo_root, req.tag)
        .and_then(|s| s.commit_tx(&req.unit.id, req.base, &verdict.final_contents, &verdict.deleted_files));
    match res {
        Ok(CommitOutcome::Committed { head_rev, tx_id, new_hashes }) => {
            on_commit();
            let mut out = json!({
                "passed": true, "committed": true, "head_rev": head_rev, "tx_id": tx_id,
                "files_committed": new_hashes.len(), "gate2_score_after": verdict.gate2_score_after,
                "gate3_ruff_errors": verdict.gate3_ruff_errors, "gate3_ty_errors": verdict.gate3_ty_errors
            });
            if let Some(m) = out.as_object_mut() { m.append(&mut req.extra_ok); }
            out
        }
        Ok(CommitOutcome::CasConflict { path, expected, actual }) => json!({
            "passed": false, "failed_gate": "mvcc_cas_conflict", "path": path, "expected": expected, "actual": actual
        }),
        Err(e) => json!({"passed": false, "error": e.to_string()}),
    }
}

fn existing_base(repo_root: &Path, meta: &TxMeta, path: &str) -> std::result::Result<(String, String), Value> {
    if let Some(prev) = meta.staged.get(path) {
        return Ok((prev.base_blake3.clone(), prev.original_content.clone()));
    }
    let Ok(disk) = fs::read_to_string(repo_root.join(path)) else {
        return Err(json!({"ok": false, "error": format!("file not found: `{path}`")}));
    };
    Ok((MvccStore::hash_bytes(disk.as_bytes()), disk))
}

fn read_source(repo_root: &Path, meta: &TxMeta, from: &str) -> std::result::Result<(String, bool, String, String), Value> {
    if let Some(prev) = meta.staged.get(from) {
        if prev.deleted { return Err(json!({"ok": false, "error": format!("already deleted: `{from}`")})); }
        return Ok((prev.staged_content.clone(), prev.created, prev.original_content.clone(), prev.base_blake3.clone()));
    }
    let Ok(txt) = fs::read_to_string(repo_root.join(from)) else {
        return Err(json!({"ok": false, "error": format!("source not found: `{from}`")}));
    };
    let h = MvccStore::hash_bytes(txt.as_bytes());
    Ok((txt.clone(), false, txt, h))
}

fn load_working(repo_root: &Path, meta: &TxMeta, hunks: &[crate::types::AsrfHunk]) -> std::result::Result<BTreeMap<String, String>, Value> {
    let mut working = BTreeMap::new();
    for h in hunks {
        if working.contains_key(&h.path) { continue; }
        if let Some(prev) = meta.staged.get(&h.path) {
            if !prev.deleted {
                working.insert(h.path.clone(), prev.staged_content.clone());
            }
        } else if h.op != PatchOp::Create {
            let Ok(disk) = fs::read_to_string(repo_root.join(&h.path)) else {
                return Err(json!({"ok": false, "error": format!("not found: `{}`", h.path)}));
            };
            working.insert(h.path.clone(), disk);
        }
    }
    Ok(working)
}

fn hash_working(working: &BTreeMap<String, String>, key: &str) -> (String, String) {
    let orig = working.get(key).cloned().unwrap_or_default();
    (MvccStore::hash_bytes(orig.as_bytes()), orig)
}

fn record_applied(meta: &mut TxMeta, working: &BTreeMap<String, String>, applied: crate::graph::AppliedPatch) {
    for del in &applied.deleted {
        if meta.staged.get(del).is_some_and(|s| s.created) {
            meta.staged.remove(del);
            continue;
        }
        let (h, orig) = match meta.staged.get(del) {
            Some(prev) => (prev.base_blake3.clone(), prev.original_content.clone()),
            None => hash_working(working, del),
        };
        meta.staged.insert(del.clone(), StagedFile::deleted(h, orig));
    }
    for (rel, next) in applied.updated {
        if meta.staged.contains_key(&rel) || applied.created.contains(&rel) {
            upsert_created(meta, &rel, next);
        } else {
            let (h, o) = hash_working(working, &rel);
            meta.staged.insert(rel, StagedFile::edited(h, o, next));
        }
    }
}

fn build_unit(
    repo_root: &Path, handle: &str, allow_abi: bool, meta: &TxMeta, hygiene: &HygieneScorer,
) -> (ModuleUnit, BTreeMap<String, String>, Vec<String>, BTreeMap<String, String>) {
    let (mut files, mut base, mut abi, mut cands) = (Vec::with_capacity(meta.staged.len()), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    let (mut deleted, mut loc, mut sum, mut n) = (Vec::new(), 0usize, 0.0f64, 0usize);
    for (rel, st) in &meta.staged {
        files.push(rel.clone());
        if st.created { base.insert(rel.clone(), String::new()); }
        else if !st.base_blake3.is_empty() { base.insert(rel.clone(), st.base_blake3.clone()); }
        if st.deleted { deleted.push(rel.clone()); continue; }
        if !st.created {
            abi.insert(rel.clone(), hash_abi(rel, &st.original_content));
            loc += count_lines(st.original_content.as_bytes());
            sum += hygiene.score_content(rel, &st.original_content).score;
            n += 1;
        }
        cands.insert(rel.clone(), st.staged_content.clone());
    }
    let test_files = scan_cached(repo_root, true, true, hygiene)
        .map(|(b, _)| impacted_tests(&b.plan, &files)).unwrap_or_default();
    let unit = ModuleUnit {
        id: handle.to_string(), files, layer: 0, wave: 0, deps: Vec::new(), dependents: Vec::new(),
        test_files, loc, base_blake3: base.clone(), abi_hash: abi,
        baseline_hygiene: if n > 0 { sum / (n as f64) } else { 0.85 }, allow_abi_change: allow_abi,
    };
    (unit, cands, deleted, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn check_target_is_fail_closed() {
        let missing_root = std::env::temp_dir().join("refit_nonexistent_root_fail_closed_99999");
        let _ = fs::remove_dir_all(&missing_root);
        assert!(
            !check_target(&missing_root, "src/main.rs"),
            "check_target MUST fail closed when repo_root cannot be canonicalized"
        );

        let repo = TempRepo::with_files("tx_fail_closed", &[("src/ok.rs", "fn main() {}\n")]);
        assert!(check_target(repo.path(), "src/ok.rs"));
        assert!(check_target(repo.path(), "src/new_sub/file.rs"));
        assert!(!check_target(repo.path(), "../escape.rs"));
        assert!(!check_target(repo.path(), ".git/config"));
        assert!(!check_target(repo.path(), ".gi\u{0307}t/HEAD"));
        assert!(!check_target(repo.path(), "src/CON.rs"));
        assert!(!check_target(repo.path(), "src/a.rs:ads"));
        assert!(!check_target(repo.path(), "src/bad. "));
    }
}
