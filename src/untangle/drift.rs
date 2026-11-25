use crate::platform::walk_tree;
use crate::source::{ext_of, rel_norm};
use crate::stem::SymbolGraph;
use crate::types::DocDriftFinding;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const DOC_FILENAMES: &[&str] = &[
    "readme.md",
    "claude.md",
    "agents.md",
    ".cursorrules",
    "contributing.md",
    "architecture.md",
];

const KNOWN_PATH_ROOTS: &[&str] = &[
    "src", "lib", "app", "pkg", "packages", "crates", "cmd", "internal", "tests", "docs",
    "scripts", "infra", ".github", "config", "migrations", "ui", "skills",
];

const KNOWN_FILE_EXTS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "md", "toml", "json", "yml",
    "yaml", "sql", "sh", "html", "css", "tf", "dockerfile",
];

const RUNTIME_PREFIXES: &[&str] = &[
    ".refit/", "target/", "dist/", "build/", "node_modules/", ".venv/", "venv/", "telemetry/",
    "coverage/", "out/", "tmp/",
];

const PLACEHOLDER_SEGMENTS: &[&str] = &[
    "example", "examples", "foo", "bar", "baz", "sample", "dummy", "placeholder", "your_",
    "my_", "path/to/", "some/", "dir/", "mock_",
];

const IGNORED_SYMBOLS: &[&str] = &[
    "main", "new", "default", "clone", "copy", "drop", "len", "is_empty", "push", "pop",
    "get", "set", "insert", "remove", "contains", "iter", "into_iter", "collect", "unwrap",
    "expect", "format", "println", "eprintln", "print", "write", "writeln", "read", "open",
    "close", "flush", "parse", "to_string", "as_str", "as_bytes", "into", "from", "lock",
    "join", "spawn", "sleep", "exit", "abort", "assert", "assert_eq", "range", "super",
    "init", "fetch", "require", "resolve", "reject", "filter", "map", "reduce", "foreach",
    "split", "trim", "slice", "replace", "starts_with", "ends_with", "sort", "dedup",
    "clear", "extend", "find", "any", "all", "count", "take", "skip", "zip", "chain",
    "example", "sample", "handler", "callback", "func", "ident", "method", "my_func",
    "some_fn", "foo_bar", "do_work", "helper", "todo", "unimplemented", "panic", "error",
    "typeof", "sizeof", "alignof", "nameof", "print_r", "printf", "sprintf", "console",
];

const BUILTIN_PKG_SCRIPTS: &[&str] = &[
    "install", "add", "remove", "update", "up", "init", "test", "start", "stop", "restart",
    "publish", "pack", "login", "logout", "audit", "outdated", "link", "unlink", "exec",
    "dlx", "create", "why", "list", "ls", "prune", "dedupe", "ci", "help", "version", "run",
];

/// Build the live symbol table from `SymbolGraph` plus local function/type declarations.
pub fn collect_known_symbols(
    snap: &SymbolGraph,
    preloaded: &BTreeMap<String, Vec<u8>>,
) -> BTreeSet<String> {
    let mut known = BTreeSet::new();
    for syms in snap.defs_by_file.values().chain(snap.refs_by_file.values()) {
        for sym in syms {
            known.insert(sym.clone());
        }
    }
    for bytes in preloaded.values() {
        let text = String::from_utf8_lossy(bytes);
        collect_declared_names(&text, &mut known);
    }
    known
}

fn collect_declared_names(text: &str, known: &mut BTreeSet<String>) {
    for raw in text.lines() {
        let trimmed = raw.trim();
        for kw in ["fn ", "def ", "function ", "func ", "struct ", "enum ", "class ", "trait "] {
            if let Some(idx) = trimmed.find(kw) {
                let after = &trimmed[idx + kw.len()..];
                if let Some(ident) = leading_ident(after)
                    && ident.len() >= 3
                {
                    known.insert(ident.to_string());
                }
            }
        }
    }
}

fn leading_ident(slice: &str) -> Option<&str> {
    let bytes = slice.as_bytes();
    let first = *bytes.first()?;
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    let mut end = 1usize;
    while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
        end += 1;
    }
    Some(&slice[..end])
}

/// Zero-LLM ReCite + DocPrism LCEF + DOCER scanner for stale doc paths, symbols, and scripts.
pub fn scan_doc_drift(
    repo_root: &Path,
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
    known_symbols: &BTreeSet<String>,
) -> Vec<DocDriftFinding> {
    let (repo_files, repo_dirs, doc_files) = collect_repo_inventory(repo_root, source_rels, preloaded);
    let npm_scripts = load_package_scripts(repo_root);
    let ctx = DriftCtx {
        repo_root,
        repo_files: &repo_files,
        repo_dirs: &repo_dirs,
        known_symbols,
        npm_scripts: npm_scripts.as_ref(),
    };

    let mut findings = Vec::new();
    for (rel, abs) in &doc_files {
        if let Ok(content) = fs::read_to_string(abs) {
            scan_markdown_doc(&ctx, rel, &content, &mut findings);
        }
    }
    for rel in source_rels {
        let Some(bytes) = preloaded.get(rel) else { continue };
        let content = String::from_utf8_lossy(bytes);
        scan_source_comments(&ctx, rel, &content, &mut findings);
    }

    findings.sort_by(|lhs, rhs| {
        lhs.file
            .cmp(&rhs.file)
            .then_with(|| lhs.line.cmp(&rhs.line))
            .then_with(|| lhs.kind.cmp(&rhs.kind))
            .then_with(|| lhs.reference.cmp(&rhs.reference))
    });
    findings.dedup_by(|lhs, rhs| {
        lhs.file == rhs.file && lhs.line == rhs.line && lhs.reference == rhs.reference
    });
    findings
}

struct DriftCtx<'a> {
    repo_root: &'a Path,
    repo_files: &'a BTreeSet<String>,
    repo_dirs: &'a BTreeSet<String>,
    known_symbols: &'a BTreeSet<String>,
    npm_scripts: Option<&'a BTreeSet<String>>,
}

fn collect_repo_inventory(
    repo_root: &Path,
    source_rels: &[String],
    preloaded: &BTreeMap<String, Vec<u8>>,
) -> (BTreeSet<String>, BTreeSet<String>, Vec<(String, PathBuf)>) {
    let mut repo_files = BTreeSet::new();
    let mut repo_dirs = BTreeSet::new();
    let mut doc_files = Vec::new();
    let mut seen_docs = BTreeSet::new();

    for rel in source_rels.iter().chain(preloaded.keys()) {
        repo_files.insert(rel.clone());
        if let Some((top, _)) = rel.split_once('/') {
            repo_dirs.insert(top.to_string());
        }
    }

    for abs in walk_tree(repo_root, None, true) {
        let Some(rel) = rel_norm(repo_root, &abs) else { continue };
        repo_files.insert(rel.clone());
        if let Some((top, _)) = rel.split_once('/') {
            repo_dirs.insert(top.to_string());
        }
        if is_doc_target(&rel) && seen_docs.insert(rel.clone()) {
            doc_files.push((rel, abs));
        }
    }

    let cursor_rules = repo_root.join(".cursorrules");
    if cursor_rules.is_file() && seen_docs.insert(".cursorrules".into()) {
        doc_files.push((".cursorrules".into(), cursor_rules));
    }

    doc_files.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    (repo_files, repo_dirs, doc_files)
}

fn is_doc_target(rel: &str) -> bool {
    let lower = rel.to_ascii_lowercase();
    let fname = lower.rsplit('/').next().unwrap_or(&lower);
    DOC_FILENAMES.contains(&fname) || (lower.starts_with("docs/") && lower.ends_with(".md"))
}

fn load_package_scripts(repo_root: &Path) -> Option<BTreeSet<String>> {
    let raw = fs::read_to_string(repo_root.join("package.json")).ok()?;
    let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let obj = val.get("scripts")?.as_object()?;
    Some(obj.keys().cloned().collect())
}

fn scan_markdown_doc(ctx: &DriftCtx<'_>, rel: &str, content: &str, out: &mut Vec<DocDriftFinding>) {
    let mut in_fence = false;
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || trimmed.is_empty() || contains_ignore_marker(trimmed) {
            continue;
        }
        let ln = idx + 1;
        for span in extract_backtick_spans(line) {
            check_doc_span(ctx, rel, ln, span, out);
        }
    }
}

fn contains_ignore_marker(text: &str) -> bool {
    let bytes = text.as_bytes();
    let needle = b"refit: ignore";
    bytes.windows(needle.len()).any(|win| win.eq_ignore_ascii_case(needle))
}

fn extract_backtick_spans(line: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        let span = after[..close].trim();
        if !span.is_empty() {
            spans.push(span);
        }
        rest = &after[close + 1..];
    }
    spans
}

fn check_doc_span(
    ctx: &DriftCtx<'_>,
    doc_file: &str,
    ln: usize,
    span: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    if let Some(script_name) = extract_cli_script_ref(span) {
        check_cli_script(ctx, doc_file, ln, span, script_name, out);
        return;
    }
    if let Some(clean_path) = parse_repo_path_candidate(span, ctx.repo_dirs) {
        if !ctx.repo_files.contains(clean_path) && !ctx.repo_root.join(clean_path).exists() {
            out.push(DocDriftFinding {
                file: doc_file.to_string(),
                line: ln,
                kind: "doc_stale_path".into(),
                reference: clean_path.to_string(),
                message: format!("Referenced file path `{clean_path}` does not exist in repository"),
            });
        }
        return;
    }
    if !ctx.known_symbols.is_empty()
        && let Some(sym) = parse_backtick_call_symbol(span)
        && !is_ignored_symbol(sym)
        && !ctx.known_symbols.contains(sym)
    {
        out.push(DocDriftFinding {
            file: doc_file.to_string(),
            line: ln,
            kind: "doc_stale_symbol".into(),
            reference: format!("{sym}()"),
            message: format!("Referenced symbol `{sym}()` not found in live codebase"),
        });
    }
}

fn extract_cli_script_ref(span: &str) -> Option<&str> {
    let rest = span
        .strip_prefix("npm run ")
        .or_else(|| span.strip_prefix("pnpm run "))
        .or_else(|| span.strip_prefix("yarn run "))?;
    let name = rest.split_whitespace().next()?;
    let valid = !name.is_empty()
        && !name.starts_with('-')
        && !has_placeholder_chars(name)
        && !BUILTIN_PKG_SCRIPTS.contains(&name);
    valid.then_some(name)
}

fn check_cli_script(
    ctx: &DriftCtx<'_>,
    doc_file: &str,
    ln: usize,
    span: &str,
    script: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    let Some(scripts) = ctx.npm_scripts else { return };
    if !scripts.contains(script) && !is_placeholder_text(script) {
        out.push(DocDriftFinding {
            file: doc_file.to_string(),
            line: ln,
            kind: "doc_stale_cli".into(),
            reference: span.to_string(),
            message: format!("Documented script `{script}` missing from package.json scripts"),
        });
    }
}

fn has_placeholder_chars(text: &str) -> bool {
    text.bytes()
        .any(|b| matches!(b, b'*' | b'?' | b'<' | b'>' | b'{' | b'}' | b'[' | b']' | b'$'))
        || text.contains("...")
}

fn is_placeholder_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    PLACEHOLDER_SEGMENTS.iter().any(|seg| lower.contains(seg))
}

fn strip_path_line_suffix(raw: &str) -> &str {
    let no_frag = raw.split('#').next().unwrap_or(raw);
    if let Some((head, tail)) = no_frag.rsplit_once(':')
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit() || b == b'-')
    {
        return head;
    }
    no_frag
}

fn parse_repo_path_candidate<'a>(
    span: &'a str,
    repo_dirs: &BTreeSet<String>,
) -> Option<&'a str> {
    if span.contains(char::is_whitespace)
        || span.contains("://")
        || span.starts_with('/')
        || span.starts_with('~')
        || span.starts_with('@')
        || has_placeholder_chars(span)
    {
        return None;
    }
    let clean = strip_path_line_suffix(span).trim_start_matches("./");
    if !clean.contains('/') || clean.contains("..") || is_placeholder_text(clean) {
        return None;
    }
    if RUNTIME_PREFIXES.iter().any(|pref| clean.starts_with(pref)) {
        return None;
    }
    let (top, _) = clean.split_once('/')?;
    if !KNOWN_PATH_ROOTS.contains(&top) && !repo_dirs.contains(top) {
        return None;
    }
    let (_, ext) = clean.rsplit_once('.')?;
    let ext_lo = ext.to_ascii_lowercase();
    KNOWN_FILE_EXTS.contains(&ext_lo.as_str()).then_some(clean)
}

fn parse_backtick_call_symbol(span: &str) -> Option<&str> {
    if span.contains(char::is_whitespace) || has_placeholder_chars(span) {
        return None;
    }
    let head = span.strip_suffix("()")?;
    let sym = head.rsplit("::").next().unwrap_or(head);
    if sym.contains('.') || sym.contains('/') || sym.len() < 4 || is_placeholder_text(sym) {
        return None;
    }
    let valid = sym
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && sym.as_bytes()[0].is_ascii_alphabetic();
    valid.then_some(sym)
}

fn is_ignored_symbol(sym: &str) -> bool {
    let lower = sym.to_ascii_lowercase();
    IGNORED_SYMBOLS.contains(&lower.as_str())
}

fn extract_comment_body<'a>(ext: &str, line: &'a str) -> Option<&'a str> {
    let trimmed = line.trim();
    if ext == "py" {
        return trimmed.strip_prefix('#').map(str::trim);
    }
    if let Some(rest) = trimmed.strip_prefix("///").or_else(|| trimmed.strip_prefix("//!")) {
        return Some(rest.trim());
    }
    if let Some(rest) = trimmed.strip_prefix("//") {
        return Some(rest.trim());
    }
    if let Some(rest) = trimmed.strip_prefix("/**").or_else(|| trimmed.strip_prefix("/*")) {
        return Some(rest.trim_end_matches("*/").trim());
    }
    if let Some(rest) = trimmed.strip_prefix("* ") {
        return Some(rest.trim());
    }
    None
}

fn scan_source_comments(
    ctx: &DriftCtx<'_>,
    rel: &str,
    content: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    let ext = ext_of(rel);
    for (idx, line) in content.lines().enumerate() {
        let Some(comment) = extract_comment_body(&ext, line) else { continue };
        if comment.is_empty() || contains_ignore_marker(comment) {
            continue;
        }
        let ln = idx + 1;
        for span in extract_backtick_spans(comment) {
            if let Some(clean_path) = parse_repo_path_candidate(span, ctx.repo_dirs)
                && !ctx.repo_files.contains(clean_path)
                && !ctx.repo_root.join(clean_path).exists()
            {
                out.push(DocDriftFinding {
                    file: rel.to_string(),
                    line: ln,
                    kind: "doc_stale_path".into(),
                    reference: clean_path.to_string(),
                    message: format!("Comment cites missing file `{clean_path}`"),
                });
            }
        }
        if !ctx.known_symbols.is_empty() {
            scan_comment_call_refs(ctx, rel, ln, comment, out);
        }
    }
}

fn scan_comment_call_refs(
    ctx: &DriftCtx<'_>,
    rel: &str,
    ln: usize,
    comment: &str,
    out: &mut Vec<DocDriftFinding>,
) {
    let bytes = comment.as_bytes();
    let mut pos = 0usize;
    while pos + 2 < bytes.len() {
        if !(bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_') {
            pos += 1;
            continue;
        }
        let start = pos;
        pos += 1;
        while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
            pos += 1;
        }
        let is_call = pos + 1 < bytes.len() && bytes[pos] == b'(' && bytes[pos + 1] == b')';
        let preceded_by_dot = start > 0 && bytes[start - 1] == b'.';
        if !is_call || preceded_by_dot {
            continue;
        }
        let sym = &comment[start..pos];
        if sym.len() >= 5
            && !is_ignored_symbol(sym)
            && !is_placeholder_text(sym)
            && !ctx.known_symbols.contains(sym)
        {
            out.push(DocDriftFinding {
                file: rel.to_string(),
                line: ln,
                kind: "doc_stale_symbol".into(),
                reference: format!("{sym}()"),
                message: format!("Stale comment reference `{sym}()` not found in live codebase"),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stem::index_repo;
    use crate::testutil::TempRepo;

    #[test]
    fn doc_drift_recite_docer_catches_stale_paths_and_symbols() {
        let live_rs = concat!(
            "// Delegates to live_verify_token() safely\n",
            "// Legacy note: previously called legacy_parse_wal() before flush\n",
            "pub fn live_verify_token(raw: &str) -> bool {\n",
            "    !raw.is_empty()\n",
            "}\n",
        );
        let readme = concat!(
            "# Demo Repo\n\n",
            "Entry point lives in `src/auth.rs` and calls `live_verify_token()`.\n",
            "Old path `src/deleted_legacy.rs` and `old_broken_helper()` were removed.\n",
            "Ignore placeholder `src/example/foo.rs`, URL `https://example.com/a/b.rs`, ",
            "and stdlib `to_string()`.\n",
            "Scripts: `npm run build` is live, `npm run ghost-task` is dead.\n",
        );
        let claude_md = "See `src/auth.rs` and `src/renamed_engine.rs` (`removed_dispatcher()`).\n";
        let agents_md = "Check `src/auth.rs` via `live_verify_token()`. Dead: `src/old_router.ts`.\n";

        let dir = TempRepo::with_files(
            "doc_drift",
            &[
                ("src/auth.rs", live_rs),
                ("README.md", readme),
                ("CLAUDE.md", claude_md),
                ("AGENTS.md", agents_md),
                ("package.json", r#"{"scripts":{"build":"vite build"}}"#),
            ],
        );

        let preloaded = BTreeMap::from([("src/auth.rs".to_string(), live_rs.as_bytes().to_vec())]);
        let source_rels = vec!["src/auth.rs".to_string()];
        let (_, snap) = index_repo(dir.path(), Some(&preloaded));
        let known = collect_known_symbols(&snap, &preloaded);
        let findings = scan_doc_drift(dir.path(), &source_rels, &preloaded, &known);

        let has = |file: &str, kind: &str, reference: &str| {
            findings
                .iter()
                .any(|f| f.file == file && f.kind == kind && f.reference == reference)
        };

        assert!(has("README.md", "doc_stale_path", "src/deleted_legacy.rs"), "{findings:?}");
        assert!(has("README.md", "doc_stale_symbol", "old_broken_helper()"), "{findings:?}");
        assert!(has("README.md", "doc_stale_cli", "npm run ghost-task"), "{findings:?}");
        assert!(has("CLAUDE.md", "doc_stale_path", "src/renamed_engine.rs"), "{findings:?}");
        assert!(has("CLAUDE.md", "doc_stale_symbol", "removed_dispatcher()"), "{findings:?}");
        assert!(has("AGENTS.md", "doc_stale_path", "src/old_router.ts"), "{findings:?}");
        assert!(has("src/auth.rs", "doc_stale_symbol", "legacy_parse_wal()"), "{findings:?}");

        assert!(!findings.iter().any(|f| f.reference.contains("src/auth.rs")), "{findings:?}");
        assert!(!findings.iter().any(|f| f.reference.contains("live_verify_token")), "{findings:?}");
        assert!(!findings.iter().any(|f| f.reference.contains("example")), "{findings:?}");
        assert!(!findings.iter().any(|f| f.reference.contains("to_string")), "{findings:?}");
        assert!(!findings.iter().any(|f| f.reference.contains("npm run build")), "{findings:?}");
    }
}
