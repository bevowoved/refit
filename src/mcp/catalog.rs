use serde_json::{Value, json};
use std::sync::OnceLock;

type ToolRow = (&'static str, &'static str, &'static [(&'static str, &'static str)], &'static [&'static str]);

const TOOLS: &[ToolRow] = &[
    ("refit_scan", "Scan repository DAG and codehygiene scores (fast cached by default).", &[("fast", "boolean")], &[]),
    ("refit_audit", "Audit clones, dead symbols, large files, unused deps, and infra configs.", &[("markdown", "boolean")], &[]),
    ("refit_sustain", "Generate AGENTS.md, .refit/invariants.yml, ADR-001, and CI guard.", &[("dry_run", "boolean")], &[]),
    ("refit_symbols", "Query symbol definitions, callers, dependencies, fan-in/out, and dead symbols.", &[("symbol", "string"), ("file", "string")], &[]),
    ("refit_read", "Read a bounded line range from a repository file.", &[("path", "string"), ("start_line", "integer"), ("end_line", "integer")], &["path"]),
    ("refit_apply", "Apply ASRF patches, run Gates 0 to 3.5, and commit via MVCC WAL.", &[("module_id", "string"), ("patch_text", "string"), ("dry_run", "boolean"), ("allow_abi_change", "boolean")], &["module_id", "patch_text"]),
    ("refit_tx_begin", "Start a multi-file shadow transaction outside the repository.", &[("ttl_secs", "integer")], &[]),
    ("refit_tx_stage", "Stage ASRF patches (edit, create, delete, rename) in an active transaction.", &[("tx_handle", "string"), ("patch_text", "string")], &["tx_handle", "patch_text"]),
    ("refit_tx_create", "Stage a new file inside an active transaction.", &[("tx_handle", "string"), ("path", "string"), ("content", "string")], &["tx_handle", "path", "content"]),
    ("refit_tx_delete", "Stage file deletion inside an active transaction.", &[("tx_handle", "string"), ("path", "string")], &["tx_handle", "path"]),
    ("refit_tx_move", "Stage a file move or rename inside an active transaction.", &[("tx_handle", "string"), ("from_path", "string"), ("to_path", "string")], &["tx_handle", "from_path", "to_path"]),
    ("refit_tx_commit", "Verify all staged files with Gates 0 to 3.5, then commit via MVCC WAL.", &[("tx_handle", "string"), ("dry_run", "boolean"), ("allow_abi_change", "boolean")], &["tx_handle"]),
    ("refit_tx_abort", "Abort an active shadow transaction and remove its staged files.", &[("tx_handle", "string")], &["tx_handle"]),
    ("refit_resume", "Recover unfinished WAL transactions from CAS and report committed modules.", &[], &[]),
    ("refit_verify", "Run a full repository audit with BLAKE3 content caching.", &[], &[]),
    ("refit_search", "Run 10 req/s GCRA-paced KeenAble web search with repo-scoped BLAKE3 cache.", &[("language", "string"), ("capability", "string")], &["language", "capability"]),
];

static CATALOG: OnceLock<Value> = OnceLock::new();

/// Build the MCP tool catalog once per process and borrow it thereafter.
pub fn tool_catalog() -> &'static Value {
    CATALOG.get_or_init(|| Value::Array(TOOLS.iter().map(encode_tool).collect()))
}

fn encode_tool(&(name, desc, prop_list, req): &ToolRow) -> Value {
    let mut props = serde_json::Map::new();
    for &(k, ty) in prop_list {
        props.insert(k.into(), json!({ "type": ty }));
    }

    let mut schema = json!({ "type": "object", "properties": props });
    if !req.is_empty() {
        schema["required"] = json!(req);
    }
    json!({ "name": name, "description": desc, "inputSchema": schema })
}
