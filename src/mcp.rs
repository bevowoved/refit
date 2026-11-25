mod catalog;
mod tx;
#[cfg(any(feature = "mcp-v2", test))]
mod v2;

use crate::gates::{GateConfig, read_gates, scan_cached};
use crate::graph::{apply_patches, parse_patches};
use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::net::{json_str, rpc_err, rpc_ok, rpc_text};
use crate::platform::MemGate;
use crate::research::WebClient;
use crate::stem::query_symbols;
use crate::untangle::{render_audit, untangle_repo, write_sustain};
use anyhow::Result;
use catalog::{encode_legacy_tools_list_rpc, sep2549_meta, tool_catalog};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tx::{
    CommitReq, ToolCtx, check_hunks, check_target, tx_abort, tx_begin, tx_commit, tx_create,
    tx_delete, tx_move, tx_stage, verify_commit,
};

struct McpCtx {
    repo_root: PathBuf,
    hygiene: HygieneScorer,
    gate_cfg: GateConfig,
    research: Arc<WebClient>,
    out_tx: tokio::sync::mpsc::Sender<String>,
    blocking_sem: Arc<tokio::sync::Semaphore>,
}

fn build_mcp_ctx(repo_root: PathBuf, out_tx: tokio::sync::mpsc::Sender<String>) -> Arc<McpCtx> {
    let permits = crate::platform::budget().mcp_blocking_permits;
    Arc::new(McpCtx {
        hygiene: HygieneScorer::open_persistent(&repo_root),
        gate_cfg: read_gates(&repo_root),
        research: WebClient::for_repo(&repo_root, None),
        repo_root,
        out_tx,
        blocking_sem: Arc::new(tokio::sync::Semaphore::new(permits)),
    })
}

/// Serve stateless MCP `2026-07-28` Streamable HTTP (`POST /mcp`) when `mcp-v2` is enabled.
#[cfg(feature = "mcp-v2")]
pub async fn serve_http(repo_root: PathBuf, listen_addr: String) -> Result<()> {
    v2::serve_http(repo_root, listen_addr).await
}

/// Run the concurrent JSON-RPC 2.0 MCP server over stdio with a serialized writer task.
/// Preserves legacy `2024-11-05` verbatim while supporting stateless `2026-07-28` framing.
pub async fn serve_mcp(repo_root: PathBuf) -> Result<()> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<String>(64);

    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(mut line) = out_rx.recv().await {
            line.push('\n');
            if stdout.write_all(line.as_bytes()).await.is_err() || stdout.flush().await.is_err() {
                break;
            }
        }
    });

    let ctx = build_mcp_ctx(repo_root, out_tx);
    #[cfg(any(feature = "mcp-v2", test))]
    let v2_state = v2::McpV2State::from_ctx(ctx.clone());
    let mut tasks = tokio::task::JoinSet::new();

    while let Some(line) = lines.next_line().await? {
        let Ok(req) = serde_json::from_str::<Value>(line.trim()) else { continue };
        #[cfg(any(feature = "mcp-v2", test))]
        dispatch_stdio_value(&req, &ctx, &v2_state, &mut tasks).await?;
        #[cfg(not(any(feature = "mcp-v2", test)))]
        dispatch_stdio_legacy(&req, &ctx, &mut tasks).await?;
    }

    while tasks.join_next().await.is_some() {}
    #[cfg(any(feature = "mcp-v2", test))]
    drop(v2_state);
    drop(ctx);
    let _ = writer.await;
    Ok(())
}

#[cfg(any(feature = "mcp-v2", test))]
async fn dispatch_stdio_value(
    req: &Value,
    ctx: &Arc<McpCtx>,
    v2_state: &Arc<v2::McpV2State>,
    tasks: &mut tokio::task::JoinSet<()>,
) -> Result<()> {
    if req.is_array() {
        let encoded = serde_json::to_string(&v2::batch_rejection_error())?;
        let _ = ctx.out_tx.send(encoded).await;
        return Ok(());
    }
    if v2::is_v2_request(req) {
        let (_status, resp_opt) = v2::handle_v2_request(v2_state, req).await;
        if let Some(resp) = resp_opt {
            let _ = ctx.out_tx.send(serde_json::to_string(&resp)?).await;
        }
        return Ok(());
    }
    dispatch_stdio_legacy(req, ctx, tasks).await
}

async fn dispatch_stdio_legacy(
    req: &Value,
    ctx: &Arc<McpCtx>,
    tasks: &mut tokio::task::JoinSet<()>,
) -> Result<()> {
    if req.is_array() {
        let err = rpc_err(
            Value::Null,
            -32600,
            "Invalid Request: JSON-RPC batch arrays are not supported",
        );
        let _ = ctx.out_tx.send(serde_json::to_string(&err)?).await;
        return Ok(());
    }
    let id = req.get("id").cloned();
    let method = req["method"].as_str().unwrap_or("");
    let wants_v2_meta = req
        .get("_meta")
        .and_then(|m| m.get("protocolVersion"))
        .and_then(Value::as_str)
        == Some("2026-07-28");

    match method {
        "initialize" => {
            let res = json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "refit", "version": "0.5.0" }
            });
            let _ = ctx.out_tx.send(serde_json::to_string(&rpc_ok(id, res))?).await;
        }
        "notifications/initialized" => {}
        "ping" => {
            let _ = ctx.out_tx.send(serde_json::to_string(&rpc_ok(id, json!({})))?).await;
        }
        "server/discover" => {
            let res = json!({
                "protocolVersion": "2026-07-28",
                "serverInfo": { "name": "refit", "version": "0.5.0" },
                "capabilities": { "tools": { "listChanged": false } },
                "tools": tool_catalog(),
                "_meta": sep2549_meta(),
            });
            let _ = ctx.out_tx.send(serde_json::to_string(&rpc_ok(id, res))?).await;
        }
        "tools/list" => {
            let encoded = if wants_v2_meta {
                let res = json!({ "tools": tool_catalog(), "_meta": sep2549_meta() });
                serde_json::to_string(&rpc_ok(id, res))?
            } else {
                encode_legacy_tools_list_rpc(id)
            };
            let _ = ctx.out_tx.send(encoded).await;
        }
        "tools/call" => spawn_call(tasks, id, req, ctx),
        other => {
            if let Some(req_id) = id {
                let err = rpc_err(req_id, -32601, format!("Method not found: {other}"));
                let _ = ctx.out_tx.send(serde_json::to_string(&err)?).await;
            }
        }
    }
    Ok(())
}

fn spawn_call(tasks: &mut tokio::task::JoinSet<()>, id: Option<Value>, req: &Value, ctx: &Arc<McpCtx>) {
    let name = json_str(&req["params"], "name");
    let args = req["params"]["arguments"].clone();
    let ctx_clone = ctx.clone();

    let gate = MemGate::global();
    let reserve = (2 * 1024 * 1024).min(gate.capacity());
    let pre_permit = gate.try_admit(reserve).ok();

    tasks.spawn(async move {
        let payload = if name == "refit_search" {
            let lang = args["language"].as_str().unwrap_or("Rust");
            let cap = args["capability"].as_str().unwrap_or("rate limiter");
            let out = ok_json(ctx_clone.research.check_library(lang, cap).await);
            drop(pre_permit);
            out
        } else {
            let _sem_permit = ctx_clone.blocking_sem.acquire().await.ok();
            let cc = ctx_clone.clone();
            tokio::task::spawn_blocking(move || {
                let _held = pre_permit.or_else(|| {
                    gate.reserve_or_wait(reserve, std::time::Duration::from_secs(5)).ok()
                });
                dispatch_tool(&name, &args, &cc.repo_root, &cc.hygiene, &cc.gate_cfg)
            })
            .await
            .unwrap_or_else(|e| json!({"error": format!("mcp worker join error: {e}")}))
        };

        if let Ok(encoded) = serde_json::to_string(&rpc_text(id, &payload)) {
            let _ = ctx_clone.out_tx.send(encoded).await;
        }
    });
}

fn ok_json<T: serde::Serialize>(res: Result<T>) -> Value {
    match res {
        Ok(val) => serde_json::to_value(val).unwrap_or_else(|_| json!({"ok": true})),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn dispatch_tool(
    name: &str,
    args: &Value,
    repo_root: &Path,
    hygiene: &HygieneScorer,
    gate_cfg: &GateConfig,
) -> Value {
    let tctx = ToolCtx { repo_root, hygiene, gate_cfg };
    match name {
        "refit_scan" => run_scan(repo_root, args, hygiene),
        "refit_audit" => run_audit(repo_root, args, hygiene),
        "refit_sustain" => run_sustain(repo_root, args, hygiene),
        "refit_symbols" => query_symbols(repo_root, args["symbol"].as_str(), args["file"].as_str()),
        "refit_read" => run_read(repo_root, args),
        "refit_resume" => run_resume(repo_root),
        "refit_verify" => run_verify(repo_root, hygiene),
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

fn run_scan(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    let fast = args["fast"].as_bool().unwrap_or(true);
    match scan_cached(repo_root, true, fast, hygiene) {
        Ok((bundle, hit)) => {
            let mut v = serde_json::to_value(&bundle.plan).unwrap_or_else(|_| json!({"ok": true}));
            if let Some(obj) = v.as_object_mut() {
                obj.insert("scan_cache_hit".into(), json!(hit));
                obj.insert("fast_mode".into(), json!(fast));
                obj.insert("detected_languages".into(), json!(bundle.detected_languages));
                obj.insert("polyglot_toolchains".into(), json!(bundle.polyglot_toolchains));
            }
            v
        }
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_audit(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    match untangle_repo(repo_root, hygiene) {
        Ok(rep) if args["markdown"].as_bool().unwrap_or(false) => json!({
            "markdown": render_audit(&rep),
            "summary": {
                "clones": rep.clone_clusters.len(),
                "dead_symbols": rep.dead_symbols.len(),
                "god_files": rep.god_files.len(),
                "infra_findings": rep.infra_findings.len(),
                "unused_deps": rep.unused_deps.len()
            }
        }),
        Ok(rep) => ok_json(Ok(rep)),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_sustain(repo_root: &Path, args: &Value, hygiene: &HygieneScorer) -> Value {
    let dry = args["dry_run"].as_bool().unwrap_or(false);
    ok_json(untangle_repo(repo_root, hygiene).and_then(|r| write_sustain(repo_root, &r, dry)))
}

fn run_read(repo_root: &Path, args: &Value) -> Value {
    use std::io::BufRead;

    let rel = args["path"].as_str().unwrap_or("");
    if !check_target(repo_root, rel) {
        return json!({"error": format!("unsafe path `{rel}`")});
    }
    let Ok(file) = fs::File::open(repo_root.join(rel)) else {
        return json!({"error": format!("file not found `{rel}`")});
    };
    let _epoch = crate::platform::EpochRegistry::global().global_epoch();
    let _file_epoch = crate::platform::EpochRegistry::global().try_read(rel);

    let start = args["start_line"].as_u64().unwrap_or(1).max(1) as usize;
    let req_end = args["end_line"]
        .as_u64()
        .map(|e| e as usize)
        .unwrap_or_else(|| start.saturating_add(199))
        .min(start.saturating_add(400));

    let mut reader = std::io::BufReader::new(file);
    let mut buf = String::new();
    let mut total = 0usize;
    let mut kept = Vec::new();
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
        "total_lines": total, "content": slice
    })
}

fn run_resume(repo_root: &Path) -> Value {
    match MvccStore::open(repo_root, "mcp-resume").and_then(|s| s.recover_wal()) {
        Ok(rep) => json!({
            "rolled_back_txs": rep.rolled_back_txs,
            "committed_modules": rep.committed_modules.len(),
            "head_rev": rep.head_rev
        }),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn run_verify(repo_root: &Path, hygiene: &HygieneScorer) -> Value {
    match scan_cached(repo_root, true, false, hygiene) {
        Ok((b, hit)) => json!({
            "scan_cache_hit": hit,
            "tests_total": b.audit.tests_total,
            "tests_failed": b.audit.tests_failed,
            "ruff_errors": b.audit.ruff_errors,
            "ty_errors": b.audit.ty_errors,
            "hygiene_median": b.audit.hygiene_median,
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
    if let Some(a) = args["allow_abi_change"].as_bool() {
        unit.allow_abi_change = a;
    }

    let (mut base_texts, mut base_hashes) = (BTreeMap::new(), BTreeMap::new());
    for rel in &unit.files {
        let txt = fs::read_to_string(ctx.repo_root.join(rel)).unwrap_or_default();
        base_hashes.insert(rel.clone(), MvccStore::hash_bytes(txt.as_bytes()));
        base_texts.insert(rel.clone(), txt);
    }

    let hunks = parse_patches(patch_text);
    if let Err(v) = check_hunks(ctx.repo_root, &hunks, "passed", Some("asrf_parse")) {
        return v;
    }
    let applied = match apply_patches(&base_texts, &hunks) {
        Ok(c) => c,
        Err(e) => return json!({"passed": false, "failed_gate": "asrf_parse", "error": e.to_string()}),
    };
    for rel in &applied.created {
        base_hashes.entry(rel.clone()).or_default();
    }

    let req = CommitReq {
        ctx, unit: &unit, cands: &applied.updated, deleted: &applied.deleted,
        base: &base_hashes, dry_run, tag: "mcp-apply",
        extra_dry: Map::new(), extra_ok: Map::new(),
    };
    verify_commit(req, || {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;
    use crate::testutil::{TempRepo, asrf_edit};
    use tx::{TxMeta, txs_dir};

    struct Harness {
        repo: TempRepo,
        hygiene: HygieneScorer,
        gates: GateConfig,
    }

    impl Harness {
        fn new(tag: &str, files: &[(&str, &str)]) -> Self {
            let repo = TempRepo::nested_with(tag, "repo", files);
            let gates = read_gates(repo.path());
            Self { repo, hygiene: HygieneScorer::new(), gates }
        }

        fn call(&self, tool: &str, args: Value) -> Value {
            dispatch_tool(tool, &args, self.repo.path(), &self.hygiene, &self.gates)
        }

        fn begin_tx(&self) -> String {
            let res = self.call("refit_tx_begin", json!({}));
            let raw = res["tx_handle"].as_str().unwrap();
            raw.to_string()
        }

        fn stage(&self, id: &str, patch: &str) -> Value {
            self.call("refit_tx_stage", json!({"tx_handle": id, "patch_text": patch}))
        }

        fn commit(&self, id: &str, dry_run: bool, allow_abi: bool) -> Value {
            self.call("refit_tx_commit", json!({"tx_handle": id, "dry_run": dry_run, "allow_abi_change": allow_abi}))
        }

        fn apply(&self, mid: &str, patch: &str, dry_run: bool) -> Value {
            self.call("refit_apply", json!({"module_id": mid, "patch_text": patch, "dry_run": dry_run}))
        }

        fn check_dry(&self, res: &Value, rel: &str, orig: &str) {
            assert_eq!((&res["passed"], &res["committed"]), (&json!(true), &json!(false)));
            assert_eq!(self.repo.read(rel), orig);
        }
    }

    #[test]
    fn test_mcp_scan_and_apply() {
        let orig = "export function add(a: number, b: number): number {\n  return a + b;\n}\n";
        let h = Harness::new("mcp_app", &[("src/calc.ts", orig)]);

        assert_eq!(h.call("refit_scan", json!({}))["scan_cache_hit"], json!(false));
        assert_eq!(h.call("refit_scan", json!({}))["scan_cache_hit"], json!(true));

        let dry_p = asrf_edit("src/calc.ts", "  return a + b;", "  const total = a + b;\n  return total;");
        let dry = h.apply("calc", &dry_p, true);
        assert_eq!(dry["dry_run"], json!(true));
        h.check_dry(&dry, "src/calc.ts", orig);

        h.repo.write("src/calc.ts", orig.replace("a + b;", "a + b + 5;"));
        let live_p = asrf_edit("src/calc.ts", "  return a + b + 5;", "  const v = a + b + 6;\n  return v;");
        assert_eq!(h.apply("calc", &live_p, false)["committed"], json!(true));
        assert!(h.repo.read("src/calc.ts").contains("a + b + 6"));
    }

    #[test]
    fn test_mcp_tx_orphan_gc() {
        let h = Harness::new("mcp_gc", &[]);
        let root = txs_dir(h.repo.path());
        let (stale, fresh) = (root.join("tx-stale-orphan"), root.join("tx-fresh-active"));
        fs::create_dir_all(&stale).unwrap();
        fs::create_dir_all(&fresh).unwrap();

        let now = now_ms();
        for (dir, id, created, ttl) in [
            (&stale, "tx-stale-orphan", now.saturating_sub(120_000), 1_000),
            (&fresh, "tx-fresh-active", now, 600_000),
        ] {
            let meta = TxMeta { tx_handle: id.into(), created_ms: created, ttl_ms: ttl, staged: BTreeMap::new() };
            fs::write(dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        }
        assert_eq!(h.call("refit_tx_begin", json!({}))["swept_orphans"], json!(1));
        assert!(!stale.exists() && fresh.exists());
    }

    #[test]
    fn test_mcp_tx_lifecycle_and_ops() {
        let a0 = "export function alpha(x: number): number {\n  return x + 1;\n}\n";
        let b0 = "export function beta(y: number): number {\n  return y + 2;\n}\n";
        let h = Harness::new("mcp_tx", &[("src/a.ts", a0), ("src/b.ts", b0)]);

        let rd = h.call("refit_read", json!({"path": "src/a.ts", "start_line": 1, "end_line": 2}));
        let rd_txt = rd["content"].as_str().unwrap();
        assert!(rd_txt.contains("1: export function alpha"));

        let id = h.begin_tx();
        let pa = asrf_edit("src/a.ts", "  return x + 1;", "  const next = x + 10;\n  return next;");
        let pb = asrf_edit("src/b.ts", "  return y + 2;", "  const next = y + 20;\n  return next;");
        assert_eq!(h.stage(&id, &pa)["staged_files_count"], json!(1));
        assert_eq!(h.stage(&id, &pb)["staged_files_count"], json!(2));

        let dry = h.commit(&id, true, false);
        h.check_dry(&dry, "src/a.ts", a0);

        let ok = h.commit(&id, false, false);
        assert_eq!((&ok["committed"], &ok["files_committed"], &ok["head_rev"]), (&json!(true), &json!(2), &json!(1)));
        assert!(h.repo.read("src/a.ts").contains("x + 10"));

        let id2 = h.begin_tx();
        h.stage(&id2, &asrf_edit("src/b.ts", "  return next;\n}", "  return next;"));
        let fail = h.commit(&id2, false, false);
        assert_eq!((&fail["passed"], &fail["failed_gate"]), (&json!(false), &json!("gate0_syntax")));
        assert_eq!(h.call("refit_tx_abort", json!({"tx_handle": id2}))["rolled_back"], json!(true));

        let mv_id = h.begin_tx();
        h.call("refit_tx_move", json!({"tx_handle": mv_id, "from_path": "src/a.ts", "to_path": "src/core/modern.ts"}));
        let cm = h.commit(&mv_id, false, true);
        assert_eq!((&cm["committed"], &cm["head_rev"]), (&json!(true), &json!(2)));
        assert!(!h.repo.has("src/a.ts") && h.repo.has("src/core/modern.ts"));

        let cr_id = h.begin_tx();
        let body = "export function help(n: number): number {\n  return n + 1;\n}\n";
        h.call("refit_tx_create", json!({"tx_handle": cr_id, "path": "src/core/helper.ts", "content": body}));
        let cc = h.commit(&cr_id, false, false);
        assert_eq!((&cc["committed"], &cc["head_rev"]), (&json!(true), &json!(3)));
    }

    #[tokio::test]
    async fn test_legacy_2024_11_05_stdio_compat_and_v2_framing() {
        let repo = TempRepo::nested_with(
            "mcp_stdio_compat",
            "repo",
            &[("src/ok.ts", "export function ok(): number { return 1; }\n")],
        );
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<String>(32);
        let ctx = build_mcp_ctx(repo.path().to_path_buf(), out_tx);
        let v2_state = v2::McpV2State::from_ctx(ctx.clone());
        let mut tasks = tokio::task::JoinSet::new();

        // 1. Legacy 2024-11-05 handshake + notifications/initialized + tools/list + tools/call
        let init_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2024-11-05" }
        });
        dispatch_stdio_value(&init_req, &ctx, &v2_state, &mut tasks).await.unwrap();
        let init_res: Value = serde_json::from_str(&out_rx.recv().await.unwrap()).unwrap();
        assert_eq!(init_res["result"]["protocolVersion"], json!("2024-11-05"));

        let notif = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        dispatch_stdio_value(&notif, &ctx, &v2_state, &mut tasks).await.unwrap();
        assert!(out_rx.try_recv().is_err());

        let list_req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
        dispatch_stdio_value(&list_req, &ctx, &v2_state, &mut tasks).await.unwrap();
        let list_res: Value = serde_json::from_str(&out_rx.recv().await.unwrap()).unwrap();
        assert!(list_res["result"].get("_meta").is_none());
        assert_eq!(list_res["result"]["tools"][0]["name"], json!("refit_apply"));

        let call_req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "refit_read", "arguments": { "path": "src/ok.ts" } }
        });
        dispatch_stdio_value(&call_req, &ctx, &v2_state, &mut tasks).await.unwrap();
        while tasks.join_next().await.is_some() {}
        let call_res: Value = serde_json::from_str(&out_rx.recv().await.unwrap()).unwrap();
        assert!(call_res["result"].get("resultType").is_none());
        assert!(call_res["result"]["content"][0]["text"].as_str().unwrap().contains("export function ok"));

        // 2. Stateless 2026-07-28 `server/discover` and batch `-32600` rejection on stdio
        let disc_req = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "server/discover",
            "_meta": { "protocolVersion": "2026-07-28" }
        });
        dispatch_stdio_value(&disc_req, &ctx, &v2_state, &mut tasks).await.unwrap();
        let disc_res: Value = serde_json::from_str(&out_rx.recv().await.unwrap()).unwrap();
        assert_eq!(disc_res["result"]["protocolVersion"], json!("2026-07-28"));
        assert_eq!(disc_res["result"]["_meta"]["ttlMs"], json!(3_600_000));

        let batch_req = json!([{"jsonrpc": "2.0", "id": 5, "method": "ping"}]);
        dispatch_stdio_value(&batch_req, &ctx, &v2_state, &mut tasks).await.unwrap();
        let batch_res: Value = serde_json::from_str(&out_rx.recv().await.unwrap()).unwrap();
        assert_eq!(batch_res["error"]["code"], json!(-32600));
    }
}
