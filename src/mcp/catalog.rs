use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub const CATALOG_TTL_MS: u64 = 3_600_000;
pub const CATALOG_CACHE_SCOPE: &str = "server";

type ToolRow = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static [&'static str],
);

/// Tool definitions sorted strictly lexicographically A->Z by `name` for prompt-cache stability.
const TOOLS: &[ToolRow] = &[
    (
        "refit_apply",
        "Apply ASRF patches, run Gates 0 to 3.5, and commit via MVCC WAL.",
        &[
            ("allow_abi_change", "boolean"),
            ("dry_run", "boolean"),
            ("module_id", "string"),
            ("patch_text", "string"),
        ],
        &["module_id", "patch_text"],
    ),
    (
        "refit_audit",
        "Audit clones, dead symbols, large files, unused deps, and infra configs.",
        &[("markdown", "boolean")],
        &[],
    ),
    (
        "refit_read",
        "Read a bounded line range from a repository file.",
        &[("end_line", "integer"), ("path", "string"), ("start_line", "integer")],
        &["path"],
    ),
    (
        "refit_resume",
        "Recover unfinished WAL transactions from CAS and report committed modules.",
        &[],
        &[],
    ),
    (
        "refit_scan",
        "Scan repository DAG and codehygiene scores (fast cached by default).",
        &[("fast", "boolean")],
        &[],
    ),
    (
        "refit_search",
        "Run 10 req/s GCRA-paced KeenAble web search with repo-scoped BLAKE3 cache.",
        &[("capability", "string"), ("language", "string")],
        &["capability", "language"],
    ),
    (
        "refit_sustain",
        "Generate AGENTS.md, .refit/invariants.yml, ADR-001, and CI guard.",
        &[("dry_run", "boolean")],
        &[],
    ),
    (
        "refit_symbols",
        "Query symbol definitions, callers, dependencies, fan-in/out, and dead symbols.",
        &[("file", "string"), ("symbol", "string")],
        &[],
    ),
    (
        "refit_tx_abort",
        "Abort an active shadow transaction and remove its staged files.",
        &[("tx_handle", "string")],
        &["tx_handle"],
    ),
    (
        "refit_tx_begin",
        "Start a multi-file shadow transaction outside the repository.",
        &[("ttl_secs", "integer")],
        &[],
    ),
    (
        "refit_tx_commit",
        "Verify all staged files with Gates 0 to 3.5, then commit via MVCC WAL.",
        &[
            ("allow_abi_change", "boolean"),
            ("dry_run", "boolean"),
            ("tx_handle", "string"),
        ],
        &["tx_handle"],
    ),
    (
        "refit_tx_create",
        "Stage a new file inside an active transaction.",
        &[("content", "string"), ("path", "string"), ("tx_handle", "string")],
        &["content", "path", "tx_handle"],
    ),
    (
        "refit_tx_delete",
        "Stage file deletion inside an active transaction.",
        &[("path", "string"), ("tx_handle", "string")],
        &["path", "tx_handle"],
    ),
    (
        "refit_tx_move",
        "Stage a file move or rename inside an active transaction.",
        &[("from_path", "string"), ("to_path", "string"), ("tx_handle", "string")],
        &["from_path", "to_path", "tx_handle"],
    ),
    (
        "refit_tx_stage",
        "Stage ASRF patches (edit, create, delete, rename) in an active transaction.",
        &[("patch_text", "string"), ("tx_handle", "string")],
        &["patch_text", "tx_handle"],
    ),
    (
        "refit_verify",
        "Run a full repository audit with BLAKE3 content caching.",
        &[],
        &[],
    ),
];

struct FrozenCatalog {
    value: Value,
    bytes: Box<[u8]>,
    digest: String,
}

static CATALOG: OnceLock<FrozenCatalog> = OnceLock::new();

fn frozen_catalog() -> &'static FrozenCatalog {
    CATALOG.get_or_init(|| {
        let mut rows: Vec<&ToolRow> = TOOLS.iter().collect();
        rows.sort_by_key(|row| row.0);

        let canonical = Value::Array(rows.into_iter().map(encode_tool).collect());
        let raw = serde_json::to_vec(&canonical).unwrap_or_else(|_| b"[]".to_vec());
        let digest = format!("blake3:{}", blake3::hash(&raw).to_hex());
        FrozenCatalog { value: canonical, bytes: raw.into_boxed_slice(), digest }
    })
}

/// Recursively sort all JSON object keys using `BTreeMap` for byte-exact determinism.
pub fn canonicalize_value(val: Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (key, item) in map {
                sorted.insert(key, canonicalize_value(item));
            }
            let mut out = Map::with_capacity(sorted.len());
            for (key, item) in sorted {
                out.insert(key, item);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize_value).collect()),
        other => other,
    }
}

/// Build the canonical A->Z MCP tool catalog once per process and borrow it thereafter.
pub fn tool_catalog() -> &'static Value {
    &frozen_catalog().value
}

/// Return the frozen immutable deterministic UTF-8 JSON bytes of the sorted tool catalog.
pub fn catalog_bytes() -> &'static [u8] {
    &frozen_catalog().bytes
}

/// Return the `blake3:<hex>` digest over the frozen deterministic catalog bytes.
pub fn catalog_digest() -> &'static str {
    &frozen_catalog().digest
}

/// Build `SEP-2549` cache metadata for MCP `2026-07-28` (`mcp-v2`) catalog responses.
pub fn sep2549_meta() -> Value {
    canonicalize_value(json!({
        "cacheScope": CATALOG_CACHE_SCOPE,
        "catalogDigest": catalog_digest(),
        "ttlMs": CATALOG_TTL_MS,
    }))
}

/// Format a legacy `2024-11-05` `tools/list` JSON-RPC frame directly from frozen catalog bytes.
pub fn encode_legacy_tools_list_rpc(id: Option<Value>) -> String {
    let id_json = serde_json::to_string(&id).unwrap_or_else(|_| "null".into());
    let tools_str = std::str::from_utf8(catalog_bytes()).unwrap_or("[]");
    format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{"tools":{tools_str}}}}}"#)
}

fn encode_tool(&(name, desc, prop_list, req): &ToolRow) -> Value {
    let mut props = BTreeMap::new();
    for &(key, ty) in prop_list {
        props.insert(key.to_string(), json!({ "type": ty }));
    }

    let mut props_map = Map::with_capacity(props.len());
    for (key, val) in props {
        props_map.insert(key, val);
    }

    let mut schema = Map::new();
    schema.insert("properties".into(), Value::Object(props_map));
    if !req.is_empty() {
        let mut req_sorted: Vec<&str> = req.to_vec();
        req_sorted.sort_unstable();
        schema.insert("required".into(), json!(req_sorted));
    }
    schema.insert("type".into(), json!("object"));

    canonicalize_value(json!({
        "description": desc,
        "inputSchema": Value::Object(schema),
        "name": name,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_sorted_canonical_and_digest_frozen() {
        let arr = tool_catalog().as_array().expect("catalog array");
        assert_eq!(arr.len(), 16);

        let names: Vec<&str> = arr.iter().map(|t| t["name"].as_str().unwrap()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "TOOLS must be sorted A->Z by name");

        let bytes = catalog_bytes();
        let expected_digest = format!("blake3:{}", blake3::hash(bytes).to_hex());
        assert_eq!(catalog_digest(), expected_digest);
        assert_eq!(bytes, serde_json::to_vec(tool_catalog()).unwrap().as_slice());

        // Verify recursively sorted keys on every tool object and schema
        for tool in arr {
            let keys: Vec<&str> = tool.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(keys, vec!["description", "inputSchema", "name"]);
            let props = tool["inputSchema"]["properties"].as_object().unwrap();
            let prop_keys: Vec<&str> = props.keys().map(String::as_str).collect();
            let mut prop_sorted = prop_keys.clone();
            prop_sorted.sort_unstable();
            assert_eq!(prop_keys, prop_sorted);
        }

        let meta = sep2549_meta();
        assert_eq!(meta["ttlMs"], json!(3_600_000));
        assert_eq!(meta["cacheScope"], json!("server"));
        assert_eq!(meta["catalogDigest"], json!(expected_digest));

        let legacy = encode_legacy_tools_list_rpc(Some(json!(7)));
        let parsed: Value = serde_json::from_str(&legacy).unwrap();
        assert_eq!(parsed["id"], json!(7));
        assert_eq!(parsed["result"]["tools"], *tool_catalog());
        assert!(parsed["result"].get("_meta").is_none());
    }
}
