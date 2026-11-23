use crate::gates::{detect_gate_config, run_full_repo_audit, verify_candidate};
use crate::graph::{apply_asrf_hunks, build_dag_plan, parse_asrf_hunks};
use crate::hygiene::HygieneScorer;
use crate::mvcc::{CommitOutcome, MvccStore};
use anyhow::Result;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Standards-compliant MCP 2024-11-05 JSON-RPC 2.0 stdio server exposing 4 typed refit tools.
pub async fn run_mcp_stdio(repo_root: PathBuf) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut lines = stdin.lines();

    let worker = HygieneScorer::resolve_default_worker(&repo_root);
    let hygiene = HygieneScorer::new(worker);
    let gate_cfg = detect_gate_config(&repo_root);

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
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "refit", "version": "0.3.0" }
                }
            })),
            "notifications/initialized" => None,
            "ping" => Some(json!({ "jsonrpc": "2.0", "id": id, "result": {} })),
            "tools/list" => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "refit_scan_repo",
                            "description": "Scan repo via sandboxed stem v0.4.0 + petgraph Tarjan SCC/Kahn DAG and return baseline Gates 1-3 summary.",
                            "inputSchema": { "type": "object", "properties": {} }
                        },
                        {
                            "name": "refit_apply_and_gate",
                            "description": "Apply raw unescaped <<<REFIT_PATCH>>> text for a module, verify Gates 0 to 3 inside an isolated stage, and atomically commit via MVCC CAS WAL.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "module_id": { "type": "string" },
                                    "patch_text": { "type": "string" }
                                },
                                "required": ["module_id", "patch_text"]
                            }
                        },
                        {
                            "name": "refit_checkpoint_resume",
                            "description": "Recover any PREPARED WAL transaction from BLAKE3 CAS and report committed modules ready to skip.",
                            "inputSchema": { "type": "object", "properties": {} }
                        },
                        {
                            "name": "refit_verify_full",
                            "description": "Run full repo pytest, ruff, ty --error-on-warning, and codehygiene audit.",
                            "inputSchema": { "type": "object", "properties": {} }
                        }
                    ]
                }
            })),
            "tools/call" => {
                let name = req["params"]["name"].as_str().unwrap_or("");
                let args = &req["params"]["arguments"];
                let payload = handle_tool_call(name, args, &repo_root, &hygiene, &gate_cfg);
                Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".into())
                        }]
                    }
                }))
            }
            _ => id.map(|req_id| {
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": { "code": -32601, "message": format!("Method not found: {method}") }
                })
            }),
        };

        if let Some(resp) = response {
            let mut out = serde_json::to_string(&resp)?;
            out.push('\n');
            stdout.write_all(out.as_bytes()).await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

fn handle_tool_call(
    name: &str,
    args: &Value,
    repo_root: &PathBuf,
    hygiene: &HygieneScorer,
    gate_cfg: &crate::gates::GateConfig,
) -> Value {
    match name {
        "refit_scan_repo" => match build_dag_plan(repo_root, None, hygiene) {
            Ok(plan) => serde_json::to_value(plan).unwrap_or_else(|_| json!({"ok": true})),
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_checkpoint_resume" => match MvccStore::open(repo_root, "mcp-resume")
            .and_then(|s| s.recover_from_wal())
        {
            Ok(rep) => json!({
                "rolled_back_txs": rep.rolled_back_txs,
                "committed_modules": rep.committed_modules.len(),
                "head_rev": rep.head_rev
            }),
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_verify_full" => match build_dag_plan(repo_root, None, hygiene) {
            Ok(plan) => {
                let rels: Vec<String> = plan.units.iter().flat_map(|u| u.files.clone()).collect();
                match run_full_repo_audit(gate_cfg, &rels, hygiene) {
                    Ok(a) => json!({
                        "tests_total": a.tests_total,
                        "tests_failed": a.tests_failed,
                        "ruff_errors": a.ruff_errors,
                        "ty_errors": a.ty_errors,
                        "hygiene_median": a.hygiene_median,
                        "hygiene_p10": a.hygiene_p10
                    }),
                    Err(e) => json!({"error": e.to_string()}),
                }
            }
            Err(e) => json!({"error": e.to_string()}),
        },
        "refit_apply_and_gate" => {
            let module_id = args["module_id"].as_str().unwrap_or("");
            let patch_text = args["patch_text"].as_str().unwrap_or("");
            let plan = match build_dag_plan(repo_root, None, hygiene) {
                Ok(p) => p,
                Err(e) => return json!({"error": e.to_string()}),
            };
            let Some(unit) = plan.units.into_iter().find(|u| u.id == module_id) else {
                return json!({"error": format!("unknown module_id `{module_id}`")});
            };
            let mut base_texts = BTreeMap::new();
            for rel in &unit.files {
                let txt = std::fs::read_to_string(repo_root.join(rel)).unwrap_or_default();
                base_texts.insert(rel.clone(), txt);
            }
            let hunks = parse_asrf_hunks(patch_text);
            let candidate = match apply_asrf_hunks(&base_texts, &hunks) {
                Ok(c) => c,
                Err(e) => return json!({"passed": false, "failed_gate": "asrf_parse", "error": e.to_string()}),
            };
            let verdict = match verify_candidate(gate_cfg, &unit, &candidate, hygiene) {
                Ok(v) => v,
                Err(e) => return json!({"passed": false, "error": e.to_string()}),
            };
            if !verdict.passed {
                return serde_json::to_value(&verdict).unwrap_or_else(|_| json!({"passed": false}));
            }
            match MvccStore::open(repo_root, "mcp-apply")
                .and_then(|s| s.commit_module(&unit.id, &unit.base_blake3, &verdict.final_contents))
            {
                Ok(CommitOutcome::Committed { head_rev, tx_id, .. }) => json!({
                    "passed": true,
                    "committed": true,
                    "head_rev": head_rev,
                    "tx_id": tx_id,
                    "gate2_score_after": verdict.gate2_score_after,
                    "gate3_ruff_errors": verdict.gate3_ruff_errors,
                    "gate3_ty_errors": verdict.gate3_ty_errors
                }),
                Ok(CommitOutcome::CasConflict { path, .. }) => json!({
                    "passed": false,
                    "failed_gate": "mvcc_cas_conflict",
                    "path": path
                }),
                Err(e) => json!({"passed": false, "error": e.to_string()}),
            }
        }
        other => json!({"error": format!("unknown tool `{other}`")}),
    }
}
