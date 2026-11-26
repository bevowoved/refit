use crate::gates::{GateConfig, run_gates, scan_cached};
use crate::graph::{apply_patches, hash_abi, impacted_tests, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::platform::{SieveCache, check_target_in_repo, ensure_cachedir_tag, now_ms, write_atomic};
use crate::source::{count_lines, work_dir};
use crate::types::{ModuleUnit, PatchOp};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
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
    fn created(staged_content: String) -> Self { Self { staged_content, created: true, ..Default::default() } }
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

#[derive(Deserialize, Default)]
#[serde(default)]
struct TxArgs {
    tx_handle: String, ttl_secs: Option<u64>, path: String, content: String,
    from_path: String, to_path: String, patch_text: String, dry_run: bool, allow_abi_change: bool,
}

pub(super) fn parse_args<T: DeserializeOwned + Default>(args: &Value) -> T {
    serde_json::from_value(args.clone()).unwrap_or_default()
}
pub(super) fn err_val(msg: impl Into<String>) -> Value { json!({ "error": msg.into() }) }

pub(super) fn fail_val(flag: &str, msg: impl Into<String>, gate: Option<&str>) -> Value {
    let mut m = Map::from_iter([(flag.into(), Value::Bool(false)), ("error".into(), msg.into().into())]);
    if let Some(g) = gate { m.insert("failed_gate".into(), g.into()); }
    Value::Object(m)
}

fn tx_ok(handle: &str, count: usize, key: &str, val: Value) -> Value {
    Value::Object(Map::from_iter([
        ("ok".into(), true.into()), ("tx_handle".into(), handle.into()),
        ("staged_files_count".into(), count.into()), (key.into(), val),
    ]))
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

fn ensure_absent(repo_root: &Path, meta: &TxMeta, path: &str, msg: &str) -> std::result::Result<(), Value> {
    if repo_root.join(path).exists() && !meta.staged.get(path).is_some_and(|s| s.deleted) {
        return Err(fail_val("ok", format!("{msg}: `{path}`"), None));
    }
    Ok(())
}

fn with_tx(
    repo_root: &Path, handle: &str, err_key: &str, save: bool,
    f: impl FnOnce(&Path, &mut TxMeta) -> std::result::Result<Value, Value>,
) -> Value {
    if !check_txid(handle) { return fail_val(err_key, "invalid tx_handle", None); }
    let lock = lock_tx(handle);
    let _g = lock.lock();
    let tx_dir = txs_dir(repo_root).join(handle);
    let meta_path = tx_dir.join("meta.json");
    let raw_meta = fs::read_to_string(&meta_path).ok();
    let Some(mut meta) = raw_meta.and_then(|s| serde_json::from_str::<TxMeta>(&s).ok()) else {
        return fail_val(err_key, format!("unknown tx_handle `{handle}`"), None);
    };
    match f(&tx_dir, &mut meta) {
        Ok(resp) => {
            if save && let Err(e) = write_tx(&meta_path, &meta) { return fail_val(err_key, e, None); }
            resp
        }
        Err(err_v) => err_v,
    }
}

pub fn tx_begin(repo_root: &Path, args: &Value) -> Value {
    let a: TxArgs = parse_args(args);
    let (now, ttl_ms) = (now_ms(), a.ttl_secs.unwrap_or(1800).clamp(1, 86_400) * 1000);
    let root = txs_dir(repo_root);
    let swept = sweep_txs(&root, now);
    let handle = format!("tx-{now}-{}", MCP_TX_SEQ.fetch_add(1, Ordering::Relaxed));
    let tx_dir = root.join(&handle);
    if let Err(e) = fs::create_dir_all(&tx_dir) { return err_val(format!("create tx dir: {e}")); }
    ensure_cachedir_tag(&root);
    let meta = TxMeta { tx_handle: handle.clone(), created_ms: now, ttl_ms, staged: BTreeMap::new() };
    if let Err(e) = write_tx(&tx_dir.join("meta.json"), &meta) { return fail_val("ok", e, None); }
    json!({ "tx_handle": handle, "created_ms": now, "ttl_ms": ttl_ms, "swept_orphans": swept })
}

fn tx_delete(repo_root: &Path, meta: &mut TxMeta, handle: &str, path: String) -> std::result::Result<Value, Value> {
    if meta.staged.get(&path).is_some_and(|s| s.created) {
        meta.staged.remove(&path);
    } else {
        let (h, orig) = existing_base(repo_root, meta, &path)?;
        meta.staged.insert(path.clone(), StagedFile::deleted(h, orig));
    }
    Ok(tx_ok(handle, meta.staged.len(), "deleted", path.into()))
}

fn tx_move(repo_root: &Path, meta: &mut TxMeta, handle: &str, from: String, to: String) -> std::result::Result<Value, Value> {
    ensure_absent(repo_root, meta, &to, "destination exists")?;
    let (content, created, orig, hash) = read_source(repo_root, meta, &from)?;
    if created { meta.staged.remove(&from); }
    else { meta.staged.insert(from.clone(), StagedFile::deleted(hash, orig)); }
    upsert_created(meta, &to, content);
    Ok(tx_ok(handle, meta.staged.len(), "moved", json!([from, to])))
}

fn tx_stage(repo_root: &Path, meta: &mut TxMeta, handle: &str, patch: &str) -> std::result::Result<Value, Value> {
    let hunks = parse_patches(patch);
    if hunks.is_empty() { return Err(fail_val("ok", "no <<<REFIT_PATCH>>> hunks found", None)); }
    check_hunks(repo_root, &hunks, "ok", None)?;
    let working = load_working(repo_root, meta, &hunks)?;
    let applied = apply_patches(&working, &hunks).map_err(|e| fail_val("ok", e.to_string(), Some("asrf_parse")))?;
    record_applied(meta, &working, applied);
    let files: Vec<String> = meta.staged.keys().cloned().collect();
    Ok(tx_ok(handle, files.len(), "files", json!(files)))
}

fn apply_tx_op(
    repo_root: &Path, op: &str, a: TxArgs, path: String, from: String, to: String, meta: &mut TxMeta,
) -> std::result::Result<Value, Value> {
    match op {
        "create" => {
            ensure_absent(repo_root, meta, &path, "already exists")?;
            upsert_created(meta, &path, a.content);
            Ok(tx_ok(&a.tx_handle, meta.staged.len(), "created", path.into()))
        }
        "delete" => tx_delete(repo_root, meta, &a.tx_handle, path),
        "move" => tx_move(repo_root, meta, &a.tx_handle, from, to),
        _ => tx_stage(repo_root, meta, &a.tx_handle, &a.patch_text),
    }
}

pub fn tx_mutate(repo_root: &Path, op: &str, args: &Value) -> Value {
    let a: TxArgs = parse_args(args);
    let (path, from, to) = (a.path.replace('\\', "/"), a.from_path.replace('\\', "/"), a.to_path.replace('\\', "/"));
    match op {
        "create" | "delete" if !check_target(repo_root, &path) => return fail_val("ok", format!("unsafe path `{path}`"), None),
        "move" if !check_target(repo_root, &from) || !check_target(repo_root, &to) => return fail_val("ok", "unsafe from_path or to_path", None),
        "move" if from == to => return fail_val("ok", "from_path and to_path are identical", None),
        _ => {}
    }
    let handle = a.tx_handle.clone();
    with_tx(repo_root, &handle, "ok", true, |_, meta| apply_tx_op(repo_root, op, a, path, from, to, meta))
}

pub fn tx_commit(ctx: &ToolCtx<'_>, args: &Value) -> Value {
    let a: TxArgs = parse_args(args);
    let mut committed = false;
    let res = with_tx(ctx.repo_root, &a.tx_handle, "passed", false, |tx_dir, meta| {
        if meta.staged.is_empty() { return Err(fail_val("passed", "no staged files in transaction", None)); }
        let (unit, cands, deleted, base) = build_unit(ctx.repo_root, &a.tx_handle, a.allow_abi_change, meta, ctx.hygiene);
        let extra_dry = Map::from_iter([
            ("tx_handle".into(), a.tx_handle.clone().into()),
            ("staged_files_count".into(), meta.staged.len().into()),
            ("deleted_files_count".into(), deleted.len().into()),
        ]);
        let extra_ok = Map::from_iter([("tx_handle".into(), a.tx_handle.clone().into()), ("files_deleted".into(), deleted.len().into())]);
        let req = CommitReq { ctx, unit: &unit, cands: &cands, deleted: &deleted, base: &base, dry_run: a.dry_run, tag: "mcp-tx", extra_dry, extra_ok };
        let (dir_c, committed_ref) = (tx_dir.to_path_buf(), &mut committed);
        Ok(verify_commit(req, move || { let _ = fs::remove_dir_all(&dir_c); *committed_ref = true; }))
    });
    if committed { evict_tx_lock(&a.tx_handle); }
    res
}

pub fn tx_abort(repo_root: &Path, args: &Value) -> Value {
    let a: TxArgs = parse_args(args);
    if !check_txid(&a.tx_handle) { return fail_val("rolled_back", "invalid tx_handle", None); }
    let tx_lock = lock_tx(&a.tx_handle);
    let guard = tx_lock.lock();
    let tx_dir = txs_dir(repo_root).join(&a.tx_handle);
    let existed = tx_dir.exists();
    let _ = fs::remove_dir_all(&tx_dir);
    drop(guard);
    evict_tx_lock(&a.tx_handle);
    json!({ "rolled_back": existed, "tx_handle": a.tx_handle })
}

pub fn check_hunks(
    repo_root: &Path, hunks: &[crate::types::AsrfHunk], err_key: &str, gate: Option<&str>,
) -> std::result::Result<(), Value> {
    for h in hunks {
        let bad = if !check_target(repo_root, &h.path) { Some(format!("unsafe path `{}`", h.path)) }
        else { h.new_path.as_deref().filter(|np| !check_target(repo_root, np)).map(|np| format!("unsafe new_path `{np}`")) };
        if let Some(msg) = bad { return Err(fail_val(err_key, msg, gate)); }
    }
    Ok(())
}

fn with_extra(mut val: Value, extra: &mut Map<String, Value>) -> Value {
    if let Some(m) = val.as_object_mut() { m.append(extra); }
    val
}

pub fn verify_commit(mut req: CommitReq<'_>, on_commit: impl FnOnce()) -> Value {
    let verdict = match run_gates(req.ctx.gate_cfg, req.unit, req.cands, req.deleted, req.ctx.hygiene) {
        Ok(v) => v,
        Err(e) => return fail_val("passed", e.to_string(), None),
    };
    if !verdict.passed {
        return serde_json::to_value(&verdict).unwrap_or_else(|_| json!({ "passed": false }));
    }
    if req.dry_run {
        return with_extra(json!({
            "passed": true, "dry_run": true, "committed": false,
            "gate0_syntax_ok": verdict.gate0_syntax_ok, "gate0_abi_ok": verdict.gate0_abi_ok,
            "gate2_score_before": verdict.gate2_score_before, "gate2_score_after": verdict.gate2_score_after,
            "gate3_ruff_errors": verdict.gate3_ruff_errors, "gate3_ty_errors": verdict.gate3_ty_errors,
            "diagnostics": verdict.diagnostics,
        }), &mut req.extra_dry);
    }
    let res = MvccStore::open(req.ctx.repo_root, req.tag)
        .and_then(|s| s.commit_tx(&req.unit.id, req.base, &verdict.final_contents, &verdict.deleted_files));
    match res {
        Ok(CommitOutcome::Committed { head_rev, tx_id, new_hashes }) => {
            on_commit();
            with_extra(json!({
                "passed": true, "committed": true, "head_rev": head_rev, "tx_id": tx_id,
                "files_committed": new_hashes.len(), "gate2_score_after": verdict.gate2_score_after,
                "gate3_ruff_errors": verdict.gate3_ruff_errors, "gate3_ty_errors": verdict.gate3_ty_errors,
            }), &mut req.extra_ok)
        }
        Ok(CommitOutcome::CasConflict { path, expected, actual }) => json!({
            "passed": false, "failed_gate": "mvcc_cas_conflict",
            "path": path, "expected": expected, "actual": actual
        }),
        Err(e) => fail_val("passed", e.to_string(), None),
    }
}

fn read_disk(repo_root: &Path, path: &str, label: &str) -> std::result::Result<String, Value> {
    fs::read_to_string(repo_root.join(path)).map_err(|_| fail_val("ok", format!("{label}: `{path}`"), None))
}

fn existing_base(repo_root: &Path, meta: &TxMeta, path: &str) -> std::result::Result<(String, String), Value> {
    if let Some(prev) = meta.staged.get(path) {
        return Ok((prev.base_blake3.clone(), prev.original_content.clone()));
    }
    let disk = read_disk(repo_root, path, "file not found")?;
    Ok((MvccStore::hash_bytes(disk.as_bytes()), disk))
}

fn read_source(repo_root: &Path, meta: &TxMeta, from: &str) -> std::result::Result<(String, bool, String, String), Value> {
    if let Some(prev) = meta.staged.get(from) {
        if prev.deleted { return Err(fail_val("ok", format!("already deleted: `{from}`"), None)); }
        return Ok((prev.staged_content.clone(), prev.created, prev.original_content.clone(), prev.base_blake3.clone()));
    }
    let txt = read_disk(repo_root, from, "source not found")?;
    let h = MvccStore::hash_bytes(txt.as_bytes());
    Ok((txt.clone(), false, txt, h))
}

fn load_working(
    repo_root: &Path, meta: &TxMeta, hunks: &[crate::types::AsrfHunk],
) -> std::result::Result<BTreeMap<String, String>, Value> {
    let mut working = BTreeMap::new();
    for h in hunks {
        if working.contains_key(&h.path) { continue; }
        if let Some(prev) = meta.staged.get(&h.path) {
            if !prev.deleted { working.insert(h.path.clone(), prev.staged_content.clone()); }
            continue;
        }
        if h.op != PatchOp::Create {
            let disk = read_disk(repo_root, &h.path, "not found")?;
            working.insert(h.path.clone(), disk);
        }
    }
    Ok(working)
}

fn hash_working(working: &BTreeMap<String, String>, key: &str) -> (String, String) {
    let orig = working.get(key).cloned().unwrap_or_default();
    (MvccStore::hash_bytes(orig.as_bytes()), orig)
}

fn record_deleted(meta: &mut TxMeta, working: &BTreeMap<String, String>, del: &str) {
    if meta.staged.get(del).is_some_and(|s| s.created) {
        meta.staged.remove(del);
        return;
    }
    let (h, orig) = match meta.staged.get(del) {
        Some(p) => (p.base_blake3.clone(), p.original_content.clone()),
        None => hash_working(working, del),
    };
    meta.staged.insert(del.to_string(), StagedFile::deleted(h, orig));
}

fn record_applied(meta: &mut TxMeta, working: &BTreeMap<String, String>, applied: crate::graph::AppliedPatch) {
    for del in &applied.deleted { record_deleted(meta, working, del); }
    for (rel, next) in applied.updated {
        if meta.staged.contains_key(&rel) || applied.created.contains(&rel) {
            upsert_created(meta, &rel, next);
            continue;
        }
        let (h, o) = hash_working(working, &rel);
        meta.staged.insert(rel, StagedFile::edited(h, o, next));
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
        .map(|(b, _)| impacted_tests(&b.plan, &files))
        .unwrap_or_default();
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
        assert!(!check_target(&missing_root, "src/main.rs"), "check_target MUST fail closed when repo_root cannot be canonicalized");
        let repo = TempRepo::with_files("tx_fail_closed", &[("src/ok.rs", "fn main() {}\n")]);
        assert!(check_target(repo.path(), "src/ok.rs"));
        assert!(check_target(repo.path(), "src/new_sub/file.rs"));
        for bad in ["../escape.rs", ".git/config", ".gi\u{0307}t/HEAD", "src/CON.rs", "src/a.rs:ads", "src/bad. "] {
            assert!(!check_target(repo.path(), bad), "must reject `{bad}`");
        }
    }
}
