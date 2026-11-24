use super::csr::{EdgeKind, ModuleMetrics};
use super::strip::lex_sentinel;
use super::witness::{
    BOUNDARY_CTOR_INVENTORY, BOUNDARY_DYNAMIC_REF, BOUNDARY_FFI_NO_MANGLE,
    BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_MACRO_DERIVE, BOUNDARY_PY_ALL, DEF_VIS_INTERNAL,
};
use crate::lex::{
    RUST_PUB, TS_EXPORT, char_end, first_ident, go_export, ident_byte, match_prefix, quoted_lit,
    quoted_path,
};
use crate::source::{count_lines, is_js};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

const PY_DEFS: &[&str] = &["async def ", "def ", "class "];

const NOISE_WORDS: &[&str] = &[
    "new", "get", "set", "run", "main", "init", "default", "Error", "Result",
    "Option", "String", "Vec", "len", "test", "self", "this", "true", "false",
    "None", "Some", "Ok", "Err", "from", "into", "clone", "fmt", "pub", "return",
    "const", "let", "var", "function", "export", "import", "class", "def",
    "async", "await", "func", "type", "struct", "enum", "trait", "impl",
    "package", "crate", "super", "mod", "use", "for", "while", "match", "else",
    "elif", "pass", "break", "continue",
];

const NEXT_ROUTE_EXPORTS: &[&str] = &[
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS",
    "generateStaticParams", "generateMetadata", "dynamic", "revalidate", "runtime",
];

const GO_IFACE_METHODS: &[&str] = &[
    "ServeHTTP", "MarshalJSON", "UnmarshalJSON", "MarshalText", "UnmarshalText",
    "Error", "String", "Read", "Write", "Close", "Len", "Less", "Swap",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedImport {
    pub target: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DefMeta {
    pub intra_non_def_hits: u16,
    pub boundary_flags: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSymbols {
    pub blake3: String,
    pub defs: Vec<String>,
    #[serde(default)]
    pub def_meta: Vec<DefMeta>,
    pub refs_tokens: Vec<String>,
    #[serde(default)]
    pub test_refs_tokens: Vec<String>,
    pub imports: Vec<String>,
    #[serde(default)]
    pub typed_imports: Vec<TypedImport>,
    #[serde(default)]
    pub glob_reexports: Vec<String>,
    #[serde(default)]
    pub attr_refs: Vec<String>,
    #[serde(default)]
    pub metrics: ModuleMetrics,
}

#[inline]
fn is_noise(name: &str) -> bool {
    name.len() < 3 || NOISE_WORDS.contains(&name)
}

pub fn parse_symbols(ext: &str, content: &str, blake3: String) -> FileSymbols {
    let lexed = lex_sentinel(ext, content);
    let mut defs_map: FxHashMap<String, u16> = FxHashMap::default();
    let mut typed_imports: FxHashSet<TypedImport> = FxHashSet::default();
    let mut glob_reexports: FxHashSet<String> = FxHashSet::default();
    let mut refs: FxHashSet<String> = FxHashSet::default();
    let mut intra_hits: FxHashMap<String, u16> = FxHashMap::default();
    let mut attr_refs: FxHashSet<String> = FxHashSet::default();
    let mut inventory_refs: FxHashSet<String> = FxHashSet::default();
    let mut py_all_syms: FxHashSet<String> = FxHashSet::default();
    let mut pub_fields: FxHashSet<String> = FxHashSet::default();
    let mut field_reads: FxHashSet<String> = FxHashSet::default();

    let raw_lines: Vec<&str> = content.lines().collect();
    let go_linknames = if ext == "go" {
        extract_go_linknames(&raw_lines)
    } else {
        FxHashSet::default()
    };
    let has_py_dynamic = ext == "py"
        && (content.contains("getattr(")
            || content.contains("setattr(")
            || content.contains("globals()[")
            || content.contains("locals()[")
            || content.contains("importlib."));

    let mut imp_acc = ImportAcc::default();
    let mut py_all_buf = String::new();
    let mut in_inventory = false;
    let mut pending_flags = 0u16;
    let mut in_rust_struct = false;
    let mut struct_depth = 0i32;

    // Enclosing function scope to exclude recursive self-calls from `intra_non_def_hits`.
    let mut active_fn: Option<(String, i32)> = None;
    let mut brace_depth = 0i32;
    let mut active_fn_lines = 0usize;
    let mut active_fn_calls = 0usize;
    let mut total_fns = 0usize;
    let mut passthrough_fns = 0usize;

    let mut line_start = 0usize;
    let mut line_idx = 0usize;
    let bytes = content.as_bytes();
    while line_start <= bytes.len() {
        let mut line_end = line_start;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let clean_cow = lexed.clean_slice(content, line_start, line_end);
        let line = clean_cow.trim_end_matches('\r');
        let trimmed = line.trim();
        let raw_prev = line_idx.checked_sub(1).and_then(|idx| raw_lines.get(idx)).copied().unwrap_or("");

        if !trimmed.is_empty() {
            // Collect leading attributes/decorators on preceding lines.
            if is_leading_boundary_attr(ext, trimmed, &mut pending_flags, &mut attr_refs) {
                if ext == "rs" && trimmed.contains("inventory::submit!") {
                    in_inventory = !trimmed.contains('}');
                    collect_token_set(ext, trimmed, &mut inventory_refs);
                }
            } else {
                if ext == "go" && raw_prev.trim().starts_with("//export ") {
                    pending_flags |= BOUNDARY_FFI_NO_MANGLE;
                }
                if ext == "rs" && in_inventory {
                    collect_token_set(ext, trimmed, &mut inventory_refs);
                    if trimmed.contains('}') {
                        in_inventory = false;
                    }
                }
                if ext == "rs" && trimmed.contains("inventory::submit!") {
                    collect_token_set(ext, trimmed, &mut inventory_refs);
                    if !trimmed.contains('}') {
                        in_inventory = true;
                    }
                }
                if ext == "py" {
                    collect_py_all(trimmed, &mut py_all_buf, &mut py_all_syms);
                }

                // Multi-line imports & edge kind classification.
                imp_acc.feed(ext, trimmed, &mut typed_imports, &mut glob_reexports);

                // Track Python active_fn exit on dedent to column 0.
                let indent = line.bytes().take_while(|&b| matches!(b, b' ' | b'\t')).count();
                if ext == "py" && indent == 0 && !trimmed.starts_with('@') {
                    if active_fn.take().is_some() {
                        if active_fn_lines <= 3 && active_fn_calls <= 2 {
                            passthrough_fns += 1;
                        }
                        active_fn_lines = 0;
                        active_fn_calls = 0;
                    }
                }

                // Extract public/exported definitions and attach boundary flags.
                let top = indent == 0;
                let mut defined_here: Option<&str> = None;
                if let Some((sym, def_flags, is_fn)) = extract_def_with_meta(ext, trimmed, top)
                    && !sym.starts_with('_')
                    && !is_noise(sym)
                {
                    let mut flags = pending_flags | def_flags;
                    if ext == "go" && (go_linknames.contains(sym) || GO_IFACE_METHODS.contains(&sym)) {
                        flags |= if go_linknames.contains(sym) {
                            BOUNDARY_FFI_NO_MANGLE
                        } else {
                            BOUNDARY_FRAMEWORK_ROUTE
                        };
                    }
                    if is_js(ext) && NEXT_ROUTE_EXPORTS.contains(&sym) {
                        flags |= BOUNDARY_FRAMEWORK_ROUTE;
                    }
                    if has_py_dynamic {
                        flags |= BOUNDARY_DYNAMIC_REF;
                    }
                    *defs_map.entry(sym.to_string()).or_insert(0) |= flags;
                    defined_here = Some(sym);
                    if is_fn {
                        total_fns += 1;
                        active_fn = Some((sym.to_string(), brace_depth));
                        active_fn_lines = 0;
                        active_fn_calls = 0;
                    }
                }
                if !trimmed.starts_with("#[") && !trimmed.starts_with('@') {
                    pending_flags = 0;
                }

                // Track public struct fields for `field-leak` deep-module rule.
                if ext == "rs" {
                    if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub(crate) struct ") {
                        if trimmed.contains('{') && !trimmed.contains('}') {
                            in_rust_struct = true;
                            struct_depth = brace_depth + count_braces_delta(trimmed);
                        }
                    } else if in_rust_struct {
                        if let Some(rest) = trimmed
                            .strip_prefix("pub ")
                            .or_else(|| trimmed.strip_prefix("pub(crate) "))
                            && !rest.starts_with("fn ")
                            && !rest.starts_with("async ")
                            && let Some(field) = first_ident(rest)
                            && field.len() >= 3
                            && !is_noise(field)
                        {
                            pub_fields.insert(field.to_string());
                        }
                    }
                } else if is_js(ext) && let Some(rest) = trimmed.strip_prefix("public ") {
                    if !rest.contains('(')
                        && let Some(field) = first_ident(rest)
                        && field.len() >= 3
                        && !is_noise(field)
                    {
                        pub_fields.insert(field.to_string());
                    }
                }

                // Collect `.field` accesses for `field-leak`.
                collect_field_reads(trimmed, &mut field_reads);

                // Collect identifier references while excluding the definition header token
                // AND excluding direct recursive self-calls inside the function's own body.
                let active_name = active_fn.as_ref().map(|(n, _)| n.as_str());
                collect_refs_tracking(
                    ext,
                    trimmed,
                    defined_here,
                    active_name,
                    &mut refs,
                    &mut intra_hits,
                );

                // Update brace depth and passthrough wrapper heuristic.
                if active_fn.is_some() {
                    active_fn_lines += 1;
                    active_fn_calls += trimmed.bytes().filter(|&b| b == b'(').count();
                }
                if ext != "py" {
                    let delta = count_braces_delta(trimmed);
                    brace_depth += delta;
                    if in_rust_struct && brace_depth < struct_depth {
                        in_rust_struct = false;
                    }
                    if let Some((_, start_depth)) = active_fn.as_ref() {
                        let single_line_closed = trimmed.contains('{') && brace_depth <= *start_depth;
                        let decl_only = !trimmed.contains('{') && trimmed.ends_with(';');
                        let multi_line_closed = delta < 0 && brace_depth <= *start_depth;
                        if single_line_closed || decl_only || multi_line_closed {
                            if active_fn_lines <= 3 && active_fn_calls <= 2 && !decl_only {
                                passthrough_fns += 1;
                            }
                            active_fn = None;
                            active_fn_lines = 0;
                            active_fn_calls = 0;
                        }
                    }
                }
            }
        }

        if line_end >= bytes.len() {
            break;
        }
        line_start = line_end + 1;
        line_idx += 1;
    }

    // Flush Python trailing fn passthrough stats.
    if active_fn.is_some() && active_fn_lines <= 3 && active_fn_calls <= 2 {
        passthrough_fns += 1;
    }

    // Collect references & test-only imports from inline `#[cfg(test)]` spans!
    let mut test_refs: FxHashSet<String> = FxHashSet::default();
    let mut rs_test_use_buf = String::new();
    for &(a, b) in &lexed.test_spans {
        if a < b && b <= content.len() {
            let seg = &content[a..b];
            for tl in seg.lines().map(str::trim).filter(|s| !s.is_empty()) {
                collect_token_set(ext, tl, &mut test_refs);
                if ext == "rs" {
                    let mut dummy_globs = FxHashSet::default();
                    let mut test_imps = FxHashSet::default();
                    rs_imports_multiline(tl, &mut rs_test_use_buf, &mut test_imps, &mut dummy_globs);
                    for mut ti in test_imps {
                        ti.kind = EdgeKind::TestImport;
                        typed_imports.insert(ti);
                    }
                }
            }
        }
    }

    // Propagate __all__, serde/macro attr refs, and inventory::submit! refs onto def flags.
    for sym in py_all_syms {
        if let Some(flags) = defs_map.get_mut(&sym) {
            *flags |= BOUNDARY_PY_ALL;
        }
    }
    let mut attr_refs_v: Vec<String> = Vec::with_capacity(attr_refs.len());
    for sym in attr_refs {
        if let Some(flags) = defs_map.get_mut(&sym) {
            *flags |= BOUNDARY_DYNAMIC_REF;
        }
        attr_refs_v.push(sym);
    }
    attr_refs_v.sort();
    for sym in inventory_refs {
        if let Some(flags) = defs_map.get_mut(&sym) {
            *flags |= BOUNDARY_CTOR_INVENTORY;
        }
    }

    let mut defs_pairs: Vec<(String, DefMeta)> = defs_map
        .into_iter()
        .map(|(name, boundary_flags)| {
            let hits = intra_hits.get(&name).copied().unwrap_or(0);
            (
                name,
                DefMeta {
                    intra_non_def_hits: hits,
                    boundary_flags,
                },
            )
        })
        .collect();
    defs_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let mut defs_v = Vec::with_capacity(defs_pairs.len());
    let mut def_meta = Vec::with_capacity(defs_pairs.len());
    for (name, meta) in defs_pairs {
        defs_v.push(name);
        def_meta.push(meta);
    }

    let mut refs_v: Vec<String> = refs.into_iter().collect();
    let mut test_refs_v: Vec<String> = test_refs.into_iter().collect();
    let mut typed_v: Vec<TypedImport> = typed_imports.into_iter().collect();
    let mut glob_v: Vec<String> = glob_reexports.into_iter().collect();
    refs_v.sort();
    test_refs_v.sort();
    typed_v.sort();
    glob_v.sort();

    let mut imps_set: FxHashSet<String> = FxHashSet::default();
    for ti in &typed_v {
        if ti.kind != EdgeKind::TestImport {
            imps_set.insert(ti.target.clone());
        }
    }
    let mut imps_v: Vec<String> = imps_set.into_iter().collect();
    imps_v.sort();

    let mut pf_v: Vec<String> = pub_fields.into_iter().collect();
    let mut fr_v: Vec<String> = field_reads.into_iter().collect();
    pf_v.sort();
    fr_v.sort();

    let metrics = ModuleMetrics {
        loc: count_lines(bytes),
        exported_symbols: defs_v.len(),
        total_fns,
        passthrough_fns,
        pub_fields: pf_v,
        field_reads: fr_v,
    };

    FileSymbols {
        blake3,
        defs: defs_v,
        def_meta,
        refs_tokens: refs_v,
        test_refs_tokens: test_refs_v,
        imports: imps_v,
        typed_imports: typed_v,
        glob_reexports: glob_v,
        attr_refs: attr_refs_v,
        metrics,
    }
}

fn is_leading_boundary_attr(
    ext: &str,
    trimmed: &str,
    pending_flags: &mut u16,
    attr_refs: &mut FxHashSet<String>,
) -> bool {
    if ext == "rs" && trimmed.starts_with("#[") {
        if trimmed.contains("no_mangle") || trimmed.contains("export_name") {
            *pending_flags |= BOUNDARY_FFI_NO_MANGLE;
        }
        if trimmed.contains("ctor") || trimmed.contains("dtor") || trimmed.contains("distributed_slice") {
            *pending_flags |= BOUNDARY_CTOR_INVENTORY;
        }
        if trimmed.contains("proc_macro")
            || trimmed.contains("wasm_bindgen")
            || trimmed.contains("pyfunction")
            || trimmed.contains("pymethods")
            || trimmed.contains("pyclass")
            || trimmed.contains("napi")
        {
            *pending_flags |= BOUNDARY_MACRO_DERIVE;
        }
        if trimmed.contains("serde(")
            && let Some(sym) = quoted_path(trimmed)
        {
            let tail = sym.rsplit("::").next().unwrap_or(sym);
            if !tail.is_empty() && tail.bytes().all(ident_byte) {
                attr_refs.insert(tail.to_string());
            }
        }
        return true;
    }
    if ext == "py" && trimmed.starts_with('@') {
        if trimmed.starts_with("@app.")
            || trimmed.starts_with("@router.")
            || trimmed.starts_with("@pytest.")
            || trimmed.starts_with("@celery.")
            || trimmed.starts_with("@click.")
            || trimmed.starts_with("@bp.")
        {
            *pending_flags |= BOUNDARY_FRAMEWORK_ROUTE;
        }
        return true;
    }
    false
}

fn extract_go_linknames(lines: &[&str]) -> FxHashSet<String> {
    let mut out = FxHashSet::default();
    for &raw in lines {
        let t = raw.trim();
        if let Some(rest) = t.strip_prefix("//go:linkname ")
            && let Some(local) = rest.split_whitespace().next()
        {
            out.insert(local.to_string());
        }
        if let Some(rest) = t.strip_prefix("//export ")
            && let Some(name) = rest.split_whitespace().next()
        {
            out.insert(name.to_string());
        }
    }
    out
}

fn collect_py_all(trimmed: &str, buf: &mut String, out: &mut FxHashSet<String>) {
    if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(trimmed);
        if trimmed.contains(']') || trimmed.contains(')') {
            extract_quoted_idents(buf, out);
            buf.clear();
        }
        return;
    }
    if trimmed.starts_with("__all__") && trimmed.contains('=') {
        if trimmed.contains(']') || trimmed.contains(')') {
            extract_quoted_idents(trimmed, out);
        } else {
            buf.push_str(trimmed);
        }
    }
}

fn extract_quoted_idents(s: &str, out: &mut FxHashSet<String>) {
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        if matches!(b[i], b'"' | b'\'') {
            let q = b[i];
            let start = i + 1;
            let mut end = start;
            while end < b.len() && b[end] != q {
                end += 1;
            }
            if end > start && end < b.len() {
                let slice = &s[start..end];
                if slice.bytes().all(ident_byte) {
                    out.insert(slice.to_string());
                }
            }
            i = (end + 1).min(b.len());
        } else {
            i += 1;
        }
    }
}

fn count_braces_delta(line: &str) -> i32 {
    let b = line.as_bytes();
    let mut i = 0usize;
    let mut delta = 0i32;
    while i < b.len() {
        let ch = b[i];
        if matches!(ch, b'"' | b'\'' | b'`') {
            i = quoted_lit(b, i, ch != b'`', true).0;
            continue;
        }
        if ch == b'{' {
            delta += 1;
        } else if ch == b'}' {
            delta -= 1;
        }
        i += 1;
    }
    delta
}

fn collect_field_reads(line: &str, out: &mut FxHashSet<String>) {
    let b = line.as_bytes();
    let mut i = 0usize;
    while i + 2 < b.len() {
        if b[i] == b'.' && (b[i + 1].is_ascii_alphabetic() || b[i + 1] == b'_') {
            let start = i + 1;
            let mut end = start;
            while end < b.len() && ident_byte(b[end]) {
                end += 1;
            }
            // Exclude method calls `.foo(`
            if end < b.len() && b[end] == b'(' {
                i = end;
                continue;
            }
            let name = &line[start..end];
            if name.len() >= 3 && !is_noise(name) {
                out.insert(name.to_string());
            }
            i = end;
            continue;
        }
        i += 1;
    }
}

#[derive(Default)]
struct ImportAcc {
    in_go: bool,
    rs_buf: String,
    ts_buf: String,
    py_buf: String,
}

impl ImportAcc {
    fn feed(
        &mut self,
        ext: &str,
        t: &str,
        out: &mut FxHashSet<TypedImport>,
        globs: &mut FxHashSet<String>,
    ) {
        if ext == "py" {
            py_imports_multiline(t, &mut self.py_buf, out);
        } else if ext == "rs" {
            rs_imports_multiline(t, &mut self.rs_buf, out, globs);
        } else if is_js(ext) {
            js_imports_multiline(t, &mut self.ts_buf, out, globs);
        } else if ext == "go" {
            go_imports(t, &mut self.in_go, out);
        }
    }
}

fn py_imports_multiline(t: &str, buf: &mut String, out: &mut FxHashSet<TypedImport>) {
    let stmt = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(t);
        if !t.contains(')') {
            return;
        }
        std::mem::take(buf)
    } else if t.starts_with("from ") && t.contains('(') && !t.contains(')') {
        buf.push_str(t);
        return;
    } else {
        t.to_string()
    };

    if let Some(rest) = stmt.strip_prefix("from ") {
        if let Some((mod_part, after)) = rest.split_once(" import ") {
            let m = mod_part.trim();
            if !m.is_empty() && m.bytes().all(|b| b == b'.') {
                for item in after.trim_matches(['(', ')']).split(',') {
                    if let Some(sym) = first_ident(item) {
                        out.insert(TypedImport {
                            target: format!("{m}{sym}"),
                            kind: EdgeKind::Runtime,
                        });
                    }
                }
            } else if !m.is_empty() {
                out.insert(TypedImport {
                    target: m.to_string(),
                    kind: EdgeKind::Runtime,
                });
                // Also record `m.item` when importing submodules (`from pkg import submod`).
                for item in after.trim_matches(['(', ')']).split(',') {
                    if let Some(sym) = first_ident(item) {
                        out.insert(TypedImport {
                            target: format!("{m}.{sym}"),
                            kind: EdgeKind::Runtime,
                        });
                    }
                }
            }
        } else if let Some(m) = rest.split_whitespace().next() {
            out.insert(TypedImport {
                target: m.to_string(),
                kind: EdgeKind::Runtime,
            });
        }
        return;
    }

    if let Some(rest) = stmt.strip_prefix("import ") {
        for item in rest.split(',') {
            let head = item.split_whitespace().next().unwrap_or("");
            if !head.is_empty() {
                out.insert(TypedImport {
                    target: head.to_string(),
                    kind: EdgeKind::Runtime,
                });
            }
        }
    }
}

fn rs_imports_multiline(
    t: &str,
    buf: &mut String,
    out: &mut FxHashSet<TypedImport>,
    globs: &mut FxHashSet<String>,
) {
    // Handle `mod foo;` / `pub mod foo;` / `pub(crate) mod foo;`
    if buf.is_empty()
        && t.ends_with(';')
        && let Some(mod_rest) = t
            .strip_prefix("mod ")
            .or_else(|| t.strip_prefix("pub mod "))
            .or_else(|| t.strip_prefix("pub(crate) mod "))
            .or_else(|| t.strip_prefix("pub(super) mod "))
        && let Some(id) = first_ident(mod_rest)
    {
        out.insert(TypedImport {
            target: id.to_string(),
            kind: EdgeKind::ModDecl,
        });
        return;
    }

    let is_use_start = t.starts_with("use ")
        || t.starts_with("pub use ")
        || t.starts_with("pub(crate) use ")
        || t.starts_with("pub(super) use ");
    let stmt = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(t);
        if !t.contains(';') {
            return;
        }
        std::mem::take(buf)
    } else if is_use_start && !t.contains(';') {
        buf.push_str(t);
        return;
    } else {
        t.to_string()
    };

    let is_pub_reexport = stmt.starts_with("pub use ");
    let mut scan = stmt.as_str();
    while let Some(pos) = scan.find("crate::") {
        let rest = &scan[pos + 7..];
        let mut expanded = Vec::new();
        expand_rs_use_tree(rest, "", &mut expanded, is_pub_reexport, globs);
        for target in expanded {
            out.insert(TypedImport {
                target,
                kind: EdgeKind::Runtime,
            });
        }
        scan = rest;
    }

    for (kw, prefix) in [("use super::", "super::"), ("pub use super::", "super::"), ("use self::", ""), ("pub use self::", "")] {
        if let Some(rest) = stmt.strip_prefix(kw) {
            let mut expanded = Vec::new();
            expand_rs_use_tree(rest, prefix, &mut expanded, is_pub_reexport, globs);
            for target in expanded {
                out.insert(TypedImport {
                    target,
                    kind: EdgeKind::Runtime,
                });
            }
        }
    }
}

fn expand_rs_use_tree(
    rest: &str,
    prefix: &str,
    out: &mut Vec<String>,
    is_pub_reexport: bool,
    globs: &mut FxHashSet<String>,
) {
    let trimmed = rest.trim().trim_end_matches(';').trim();
    if let Some(inner) = trimmed.strip_prefix('{') {
        let body = matching_brace_slice(inner);
        for part in split_top_commas(body) {
            expand_rs_use_tree(part, prefix, out, is_pub_reexport, globs);
        }
        return;
    }
    let before_as = trimmed.split(" as ").next().unwrap_or(trimmed).trim();
    if let Some(brace_pos) = before_as.find("::{") {
        let head = &before_as[..brace_pos];
        let mod_path = if prefix.is_empty() {
            head.replace("::", "/")
        } else {
            format!("{prefix}{}", head.replace("::", "/"))
        };
        if !mod_path.is_empty() {
            out.push(mod_path.clone());
            if let Some(first) = head.split("::").next()
                && !first.is_empty()
                && first != head
            {
                let top = if prefix.is_empty() {
                    first.to_string()
                } else {
                    format!("{prefix}{first}")
                };
                out.push(top);
            }
        }
        let inner = matching_brace_slice(&before_as[brace_pos + 3..]);
        for part in split_top_commas(inner) {
            let p = part.trim();
            if p == "*" && is_pub_reexport {
                globs.insert(mod_path.clone());
            } else if p != "self" && p != "*" {
                let sub_prefix = format!("{mod_path}/");
                expand_rs_use_tree(p, &sub_prefix, out, is_pub_reexport, globs);
            }
        }
        return;
    }
    let segs: Vec<&str> = before_as.split("::").map(str::trim).filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return;
    }
    if segs.last() == Some(&"*") && is_pub_reexport && segs.len() >= 2 {
        let mod_part = segs[..segs.len() - 1].join("/");
        let full = if prefix.is_empty() { mod_part } else { format!("{prefix}{mod_part}") };
        globs.insert(full.clone());
        out.push(full);
        return;
    }
    if let Some(first) = first_ident(segs[0]) {
        let first_t = if prefix.is_empty() {
            first.to_string()
        } else {
            format!("{prefix}{first}")
        };
        out.push(first_t);
    }
    if segs.len() >= 2 {
        let mut acc = String::from(prefix);
        for (idx, seg) in segs.iter().take(segs.len() - 1).enumerate() {
            if let Some(id) = first_ident(seg) {
                if idx > 0 {
                    acc.push('/');
                }
                acc.push_str(id);
            }
        }
        if !acc.is_empty() {
            out.push(acc);
        }
    }
}

fn matching_brace_slice(after_open: &str) -> &str {
    let mut depth = 1usize;
    for (idx, b) in after_open.bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &after_open[..idx];
                }
            }
            _ => {}
        }
    }
    after_open
}

fn split_top_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, b) in s.bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                let part = s[start..idx].trim();
                if !part.is_empty() {
                    out.push(part);
                }
                start = idx + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn js_imports_multiline(
    t: &str,
    buf: &mut String,
    out: &mut FxHashSet<TypedImport>,
    globs: &mut FxHashSet<String>,
) {
    // Dynamic `import('...')` expressions anywhere on the line.
    let mut dscan = t;
    while let Some(pos) = dscan.find("import(") {
        let after = &dscan[pos + 7..];
        if let Some(p) = quoted_path(after) {
            out.insert(TypedImport {
                target: p.to_string(),
                kind: EdgeKind::Dynamic,
            });
        }
        dscan = after;
    }

    // `require('...')` anywhere on the line.
    let mut rscan = t;
    while let Some(pos) = rscan.find("require(") {
        let after = &rscan[pos + 8..];
        if let Some(p) = quoted_path(after) {
            out.insert(TypedImport {
                target: p.to_string(),
                kind: EdgeKind::Runtime,
            });
        }
        rscan = after;
    }

    let stmt = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(t);
        if !t.contains(" from ") && !t.ends_with(';') {
            if buf.len() > 2048 {
                buf.clear();
            }
            return;
        }
        std::mem::take(buf)
    } else if (t.starts_with("import ") || t.starts_with("export "))
        && t.contains('{')
        && !t.contains(" from ")
        && !t.ends_with(';')
    {
        buf.push_str(t);
        return;
    } else {
        t.to_string()
    };

    let st = stmt.trim();
    let is_type_only = st.starts_with("import type ")
        || st.starts_with("export type ")
        || (st.starts_with("import {") && is_all_inline_type_imports(st));
    let kind = if is_type_only {
        EdgeKind::TypeOnly
    } else {
        EdgeKind::Runtime
    };

    if let Some(pos) = st.find(" from ") {
        if let Some(p) = quoted_path(&st[pos + 6..]) {
            out.insert(TypedImport {
                target: p.to_string(),
                kind,
            });
            if st.starts_with("export *") {
                globs.insert(p.to_string());
            }
        }
    } else if st.starts_with("import ")
        && !st.starts_with("import(")
        && let Some(p) = quoted_path(st)
    {
        out.insert(TypedImport {
            target: p.to_string(),
            kind,
        });
    }
}

fn is_all_inline_type_imports(st: &str) -> bool {
    let Some(open) = st.find('{') else { return false };
    let Some(close) = st[open + 1..].find('}') else { return false };
    let inner = &st[open + 1..open + 1 + close];
    let items: Vec<&str> = inner.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    !items.is_empty() && items.iter().all(|it| it.starts_with("type "))
}

fn go_imports(t: &str, in_block: &mut bool, out: &mut FxHashSet<TypedImport>) {
    if t.starts_with("import (") {
        *in_block = true;
        return;
    }
    if *in_block && t.starts_with(')') {
        *in_block = false;
        return;
    }
    if (*in_block || t.starts_with("import "))
        && let Some(full) = quoted_path(t)
        && !full.is_empty()
    {
        out.insert(TypedImport {
            target: full.to_string(),
            kind: EdgeKind::Runtime,
        });
    }
}

fn extract_def_with_meta<'a>(ext: &str, t: &'a str, top: bool) -> Option<(&'a str, u16, bool)> {
    if ext == "py" && top {
        let is_fn = t.starts_with("def ") || t.starts_with("async def ");
        return match_prefix(t, PY_DEFS)
            .and_then(first_ident)
            .map(|id| (id, 0, is_fn));
    }
    if ext == "rs" && top {
        let internal = if t.starts_with("pub(crate) ") || t.starts_with("pub(super) ") {
            DEF_VIS_INTERNAL
        } else {
            0
        };
        let ffi = if t.contains("extern \"C\"") || t.contains("extern \"system\"") {
            BOUNDARY_FFI_NO_MANGLE
        } else {
            0
        };
        let is_fn = t.contains(" fn ");
        if let Some(rest) = t
            .strip_prefix("pub extern \"C\" fn ")
            .or_else(|| t.strip_prefix("pub unsafe extern \"C\" fn "))
            .or_else(|| t.strip_prefix("pub unsafe fn "))
            .or_else(|| t.strip_prefix("pub(crate) unsafe fn "))
        {
            return first_ident(rest).map(|id| (id, internal | ffi, true));
        }
        return match_prefix(t, RUST_PUB)
            .and_then(first_ident)
            .map(|id| (id, internal | ffi, is_fn));
    }
    if is_js(ext) {
        let is_fn = t.contains("function ") || t.contains("=>");
        return match_prefix(t, TS_EXPORT)
            .and_then(first_ident)
            .map(|id| (id, 0, is_fn));
    }
    if ext == "go" {
        let is_fn = t.starts_with("func ");
        return go_export(t).map(|id| (id, 0, is_fn));
    }
    None
}

fn collect_token_set(ext: &str, t: &str, out: &mut FxHashSet<String>) {
    let b = t.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let ch = b[i];
        if ext == "rs" && ch == b'\'' {
            i = char_end(b, i).unwrap_or(i + 1);
            continue;
        }
        if ch == b'"' || (ext != "rs" && ch == b'\'') || ch == b'`' {
            i = quoted_lit(b, i, ch != b'`', true).0;
            continue;
        }
        if ident_byte(ch) {
            let start = i;
            while i < b.len() && ident_byte(b[i]) {
                i += 1;
            }
            let tok = &t[start..i];
            if !b[start].is_ascii_digit() && !is_noise(tok) && !out.contains(tok) {
                out.insert(tok.to_string());
            }
            continue;
        }
        i += 1;
    }
}

/// Extract identifier tokens while tracking intra-file non-def hits and excluding
/// direct recursive self-calls inside a function's own body.
fn collect_refs_tracking(
    ext: &str,
    t: &str,
    defined_here: Option<&str>,
    active_fn: Option<&str>,
    refs: &mut FxHashSet<String>,
    intra_hits: &mut FxHashMap<String, u16>,
) {
    let b = t.as_bytes();
    let mut i = 0usize;
    let mut skipped_decl_token = false;

    while i < b.len() {
        let ch = b[i];
        if ext == "rs" && ch == b'\'' {
            i = char_end(b, i).unwrap_or(i + 1);
            continue;
        }
        if ch == b'"' || (ext != "rs" && ch == b'\'') || ch == b'`' {
            i = quoted_lit(b, i, ch != b'`', true).0;
            continue;
        }

        if ident_byte(ch) {
            let start = i;
            while i < b.len() && ident_byte(b[i]) {
                i += 1;
            }
            let tok = &t[start..i];
            if !b[start].is_ascii_digit() && !is_noise(tok) {
                if !skipped_decl_token && defined_here == Some(tok) {
                    skipped_decl_token = true;
                    continue;
                }
                if active_fn == Some(tok) {
                    // Direct recursive self-call inside `tok`'s own body: not an external/caller witness.
                    continue;
                }
                if !refs.contains(tok) {
                    refs.insert(tok.to_string());
                }
                let entry = intra_hits.entry(tok.to_string()).or_insert(0);
                *entry = entry.saturating_add(1);
            }
            continue;
        }
        i += 1;
    }
}
