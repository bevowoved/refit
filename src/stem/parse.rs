use super::analysis::{
    BOUNDARY_CTOR_INVENTORY, BOUNDARY_DYNAMIC_REF, BOUNDARY_FFI_NO_MANGLE,
    BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_MACRO_DERIVE, BOUNDARY_PY_ALL, DEF_VIS_INTERNAL, EdgeKind,
    ModuleMetrics,
};
use crate::lex::{
    LangKind, LexCursor, LexKind, RUST_PUB, TS_EXPORT, blank_noise_sentinel, first_ident,
    go_export, ident_byte, match_prefix, quoted_path, scan_code,
};
use crate::platform::parse_go_block_lines;
use crate::source::{count_lines, is_js};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::ops::ControlFlow;

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

const RS_MACRO_ATTRS: &[&str] = &[
    "proc_macro", "wasm_bindgen", "pyfunction", "pymethods", "pyclass", "napi",
];

const PY_ROUTE_DECORATORS: &[&str] = &[
    "@app.", "@router.", "@pytest.", "@celery.", "@click.", "@bp.",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedImport { pub target: String, pub kind: EdgeKind }

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DefMeta { pub intra_non_def_hits: u16, pub boundary_flags: u16 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSymbols {
    pub blake3: String, pub defs: Vec<String>,
    #[serde(default)] pub def_meta: Vec<DefMeta>,
    pub refs_tokens: Vec<String>,
    #[serde(default)] pub test_refs_tokens: Vec<String>,
    pub imports: Vec<String>,
    #[serde(default)] pub typed_imports: Vec<TypedImport>,
    #[serde(default)] pub glob_reexports: Vec<String>,
    #[serde(default)] pub attr_refs: Vec<String>,
    #[serde(default)] pub metrics: ModuleMetrics,
}

#[inline] fn is_noise(name: &str) -> bool { name.len() < 3 || NOISE_WORDS.contains(&name) }

#[inline]
fn insert_non_noise(set: &mut FxHashSet<String>, ident: Option<&str>) {
    if let Some(name) = ident.filter(|s| !is_noise(s)) { set.insert(name.to_string()); }
}

#[derive(Default)]
struct ParseState {
    defs_map: FxHashMap<String, u16>, typed_imports: FxHashSet<TypedImport>,
    glob_reexports: FxHashSet<String>, refs: FxHashSet<String>,
    intra_hits: FxHashMap<String, u16>, attr_refs: FxHashSet<String>,
    inventory_refs: FxHashSet<String>, py_all_syms: FxHashSet<String>,
    pub_fields: FxHashSet<String>, field_reads: FxHashSet<String>,
    rs_buf: String, ts_buf: String, py_buf: String, py_all_buf: String,
    in_inventory: bool, pending_flags: u16, in_rust_struct: bool, struct_depth: i32,
    active_fn: Option<(String, i32)>, brace_depth: i32,
    active_fn_lines: usize, active_fn_calls: usize, total_fns: usize, passthrough_fns: usize,
}

struct SymbolParser<'a> {
    ext: &'a str, lang: LangKind, go_linknames: FxHashSet<String>, has_py_dynamic: bool, st: ParseState,
}

fn process_parser_line(p: &mut SymbolParser<'_>, line: &str, raw_prev: &str) {
    let trimmed = line.trim();
    if trimmed.is_empty() { return; }
    if is_leading_boundary_attr(p.ext, trimmed, &mut p.st.pending_flags, &mut p.st.attr_refs) {
        if p.ext == "rs" && trimmed.contains("inventory::submit!") { p.track_inventory(trimmed); }
        return;
    }
    if p.ext == "go" && raw_prev.trim().starts_with("//export ") { p.st.pending_flags |= BOUNDARY_FFI_NO_MANGLE; }
    let rs_inv = p.ext == "rs" && (p.st.in_inventory || trimmed.contains("inventory::submit!"));
    if rs_inv { p.track_inventory(trimmed); }
    if p.ext == "py" { collect_py_all(trimmed, &mut p.st.py_all_buf, &mut p.st.py_all_syms); }

    p.feed_imports(trimmed);
    let indent = leading_indent(line);
    if p.ext == "py" && indent == 0 && !trimmed.starts_with('@') { p.finish_active_fn(true); }
    let defined_here = p.record_definition(trimmed, indent == 0);
    if !trimmed.starts_with("#[") && !trimmed.starts_with('@') { p.st.pending_flags = 0; }

    let (brace_delta, call_opens) = p.scan_line_tokens(trimmed, defined_here);
    p.track_public_fields(trimmed, brace_delta);
    p.update_fn_scope(trimmed, brace_delta, call_opens);
}

impl<'a> SymbolParser<'a> {
    fn new(ext: &'a str, content: &str) -> Self {
        let go_linknames = if ext == "go" { extract_go_linknames(content) } else { FxHashSet::default() };
        let has_py_dynamic = ext == "py"
            && ["getattr(", "setattr(", "globals()[", "locals()[", "importlib."]
                .iter().any(|needle| content.contains(needle));
        Self { ext, lang: LangKind::from_ext(ext), go_linknames, has_py_dynamic, st: ParseState::default() }
    }

    fn track_inventory(&mut self, trimmed: &str) {
        collect_token_set(self.lang, trimmed, &mut self.st.inventory_refs);
        self.st.in_inventory = !trimmed.contains('}');
    }

    fn process_line(&mut self, line: &str, raw_prev: &str) {
        process_parser_line(self, line, raw_prev);
    }

    fn feed_imports(&mut self, trimmed: &str) {
        match self.ext {
            "py" => py_imports_multiline(trimmed, &mut self.st.py_buf, &mut self.st.typed_imports),
            "rs" => rs_imports_multiline(
                trimmed, &mut self.st.rs_buf, &mut self.st.typed_imports, &mut self.st.glob_reexports,
            ),
            e if is_js(e) => js_imports_multiline(
                trimmed, &mut self.st.ts_buf, &mut self.st.typed_imports, &mut self.st.glob_reexports,
            ),
            _ => {}
        }
    }

    fn finish_active_fn(&mut self, count_passthrough: bool) { finish_fn_in(&mut self.st, count_passthrough); }

    /// Single-pass `LexCursor<'a>` scan over a cleaned line collecting brace delta,
    /// call parens, `.field` reads, and non-def/non-self-call identifier references.
    fn scan_line_tokens(&mut self, trimmed: &str, defined_here: Option<&str>) -> (i32, usize) {
        scan_tokens_into(&mut self.st, self.lang, trimmed, defined_here)
    }

    fn extra_def_flags(&self, sym: &str) -> u16 {
        match self.ext {
            "go" if self.go_linknames.contains(sym) => BOUNDARY_FFI_NO_MANGLE,
            "go" if GO_IFACE_METHODS.contains(&sym) => BOUNDARY_FRAMEWORK_ROUTE,
            e if is_js(e) && NEXT_ROUTE_EXPORTS.contains(&sym) => BOUNDARY_FRAMEWORK_ROUTE,
            "py" if self.has_py_dynamic => BOUNDARY_DYNAMIC_REF,
            _ => 0,
        }
    }

    fn record_definition<'b>(&mut self, trimmed: &'b str, top: bool) -> Option<&'b str> {
        let (sym, def_flags, is_fn) = extract_def_with_meta(self.ext, trimmed, top)?;
        if sym.starts_with('_') || is_noise(sym) { return None; }
        let flags = self.st.pending_flags | def_flags | self.extra_def_flags(sym);
        *self.st.defs_map.entry(sym.to_string()).or_insert(0) |= flags;
        if is_fn {
            self.st.total_fns += 1;
            self.st.active_fn = Some((sym.to_string(), self.st.brace_depth));
            (self.st.active_fn_lines, self.st.active_fn_calls) = (0, 0);
        }
        Some(sym)
    }

    fn track_public_fields(&mut self, trimmed: &str, brace_delta: i32) {
        if self.ext == "rs" {
            track_rs_pub_field(&mut self.st, trimmed, brace_delta);
        } else if is_js(self.ext) && let Some(rest) = trimmed.strip_prefix("public ") && !rest.contains('(') {
            insert_non_noise(&mut self.st.pub_fields, first_ident(rest));
        }
    }

    fn update_fn_scope(&mut self, trimmed: &str, brace_delta: i32, call_opens: usize) {
        update_scope_state(&mut self.st, self.ext, trimmed, brace_delta, call_opens);
    }

    fn collect_test_span_refs(&mut self, content: &str, test_spans: &[(usize, usize)]) -> FxHashSet<String> {
        let (mut test_refs, mut rs_use_buf) = (FxHashSet::default(), String::new());
        for &(start, end) in test_spans {
            if start >= end || end > content.len() { continue; }
            scan_test_span(self.ext, self.lang, &content[start..end], &mut rs_use_buf, &mut test_refs, &mut self.st.typed_imports);
        }
        test_refs
    }

    fn finish(mut self, blake3: String, content: &str, test_spans: &[(usize, usize)]) -> FileSymbols {
        self.finish_active_fn(true);
        let test_refs = self.collect_test_span_refs(content, test_spans);

        let attr_refs_v = into_sorted(std::mem::take(&mut self.st.attr_refs));
        apply_boundary_flags(&mut self.st.defs_map, &self.st.py_all_syms, BOUNDARY_PY_ALL);
        apply_boundary_flags(&mut self.st.defs_map, &attr_refs_v, BOUNDARY_DYNAMIC_REF);
        apply_boundary_flags(&mut self.st.defs_map, &self.st.inventory_refs, BOUNDARY_CTOR_INVENTORY);

        let intra_hits = self.st.intra_hits;
        let mut defs_pairs: Vec<(String, DefMeta)> = self.st.defs_map.into_iter().map(|(name, flags)| {
            let hits = intra_hits.get(&name).copied().unwrap_or(0);
            (name, DefMeta { intra_non_def_hits: hits, boundary_flags: flags })
        }).collect();
        defs_pairs.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let (defs_v, def_meta): (Vec<String>, Vec<DefMeta>) = defs_pairs.into_iter().unzip();

        let typed_v = into_sorted(self.st.typed_imports);
        let mut imps_v = into_sorted(
            typed_v.iter().filter(|i| i.kind != EdgeKind::TestImport).map(|i| i.target.clone()),
        );
        imps_v.dedup();

        FileSymbols {
            blake3,
            metrics: ModuleMetrics {
                loc: count_lines(content.as_bytes()), exported_symbols: defs_v.len(),
                total_fns: self.st.total_fns, passthrough_fns: self.st.passthrough_fns,
                pub_fields: into_sorted(self.st.pub_fields), field_reads: into_sorted(self.st.field_reads),
            },
            defs: defs_v, def_meta, refs_tokens: into_sorted(self.st.refs),
            test_refs_tokens: into_sorted(test_refs), imports: imps_v, typed_imports: typed_v,
            glob_reexports: into_sorted(self.st.glob_reexports), attr_refs: attr_refs_v,
        }
    }
}

fn finish_fn_in(st: &mut ParseState, count_passthrough: bool) {
    if st.active_fn.take().is_none() { return; }
    if count_passthrough && st.active_fn_lines <= 3 && st.active_fn_calls <= 2 {
        st.passthrough_fns += 1;
    }
    (st.active_fn_lines, st.active_fn_calls) = (0, 0);
}

fn record_ident_in(st: &mut ParseState, ident: &str) {
    if st.active_fn.as_ref().is_some_and(|(name, _)| name == ident) { return; }
    if !st.refs.contains(ident) { st.refs.insert(ident.to_string()); }
    let hits = st.intra_hits.entry(ident.to_string()).or_insert(0);
    *hits = hits.saturating_add(1);
}

fn scan_tokens_into(
    st: &mut ParseState, lang: LangKind, trimmed: &str, defined_here: Option<&str>,
) -> (i32, usize) {
    let mut cursor = LexCursor::new(lang, trimmed.as_bytes());
    let (mut brace_delta, mut call_opens) = (0i32, 0usize);
    let (mut skipped_decl, mut prev_was_dot, mut pending_field) = (false, false, None);

    while let Some(tok) = cursor.next_token() {
        if tok.kind == LexKind::LParen { pending_field = None; }
        else { insert_non_noise(&mut st.field_reads, pending_field.take()); }

        match tok.kind {
            LexKind::LBrace => brace_delta += 1,
            LexKind::RBrace => brace_delta -= 1,
            LexKind::LParen => call_opens += 1,
            LexKind::Dot => { prev_was_dot = true; continue; }
            LexKind::Ident => {
                let ident = tok.as_str();
                if prev_was_dot { pending_field = Some(ident); }
                if !is_noise(ident) {
                    if !skipped_decl && defined_here == Some(ident) { skipped_decl = true; }
                    else { record_ident_in(st, ident); }
                }
            }
            _ => {}
        }
        prev_was_dot = false;
    }
    insert_non_noise(&mut st.field_reads, pending_field);
    (brace_delta, call_opens)
}

fn update_scope_state(st: &mut ParseState, ext: &str, trimmed: &str, brace_delta: i32, call_opens: usize) {
    if st.active_fn.is_some() {
        st.active_fn_lines += 1;
        st.active_fn_calls += call_opens;
    }
    if ext == "py" { return; }

    st.brace_depth += brace_delta;
    if st.in_rust_struct && st.brace_depth < st.struct_depth { st.in_rust_struct = false; }
    let Some((_, start_depth)) = st.active_fn.as_ref() else { return };
    let has_brace = trimmed.contains('{');
    let decl_only = !has_brace && trimmed.ends_with(';');
    if decl_only || ((has_brace || brace_delta < 0) && st.brace_depth <= *start_depth) {
        finish_fn_in(st, !decl_only);
    }
}

fn track_rs_pub_field(st: &mut ParseState, trimmed: &str, brace_delta: i32) {
    if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub(crate) struct ") {
        if trimmed.contains('{') && !trimmed.contains('}') {
            (st.in_rust_struct, st.struct_depth) = (true, st.brace_depth + brace_delta);
        }
        return;
    }
    if !st.in_rust_struct { return; }
    if let Some(rest) = match_prefix(trimmed, &["pub ", "pub(crate) "])
        && !rest.starts_with("fn ") && !rest.starts_with("async ")
    {
        insert_non_noise(&mut st.pub_fields, first_ident(rest));
    }
}

fn scan_test_span(
    ext: &str, lang: LangKind, slice: &str, rs_use_buf: &mut String,
    test_refs: &mut FxHashSet<String>, typed_imports: &mut FxHashSet<TypedImport>,
) {
    for test_line in slice.lines().map(str::trim).filter(|s| !s.is_empty()) {
        collect_token_set(lang, test_line, test_refs);
        if ext != "rs" { continue; }
        let (mut dummy_globs, mut test_imps) = (FxHashSet::default(), FxHashSet::default());
        rs_imports_multiline(test_line, rs_use_buf, &mut test_imps, &mut dummy_globs);
        for mut imp in test_imps {
            imp.kind = EdgeKind::TestImport;
            typed_imports.insert(imp);
        }
    }
}

fn apply_boundary_flags<'a>(
    defs_map: &mut FxHashMap<String, u16>, syms: impl IntoIterator<Item = &'a String>, flag: u16,
) {
    for sym in syms {
        if let Some(flags) = defs_map.get_mut(sym.as_str()) { *flags |= flag; }
    }
}

fn into_sorted<T: Ord>(iter: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut out: Vec<T> = iter.into_iter().collect();
    out.sort();
    out
}

fn accumulate_stmt<'a>(
    line: &'a str, buf: &mut String, starts_multi: bool, done: bool, max_len: usize,
) -> Option<Cow<'a, str>> {
    if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(line);
        if !done {
            if max_len > 0 && buf.len() > max_len { buf.clear(); }
            return None;
        }
        return Some(Cow::Owned(std::mem::take(buf)));
    }
    if starts_multi && !done {
        buf.push_str(line);
        return None;
    }
    Some(Cow::Borrowed(line))
}

pub fn parse_symbols(ext: &str, content: &str, blake3: String) -> FileSymbols {
    let lexed = blank_noise_sentinel(ext, content);
    let mut parser = SymbolParser::new(ext, content);

    if ext == "go" {
        let clean_go = lexed.clean_slice(content, 0, content.len());
        for line in parse_go_block_lines(&clean_go, "import") {
            if let Some(full) = quoted_path(line).filter(|s| !s.is_empty()) {
                push_rt_import(&mut parser.st.typed_imports, full.to_string());
            }
        }
    }

    let bytes = content.as_bytes();
    let (mut line_start, mut raw_prev) = (0usize, "");
    while line_start <= bytes.len() {
        let line_end = memchr::memchr(b'\n', &bytes[line_start..])
            .map_or(bytes.len(), |rel| line_start + rel);
        let clean_cow = lexed.clean_slice(content, line_start, line_end);
        parser.process_line(clean_cow.trim_end_matches('\r'), raw_prev);
        raw_prev = content[line_start..line_end].trim_end_matches('\r');
        if line_end >= bytes.len() { break; }
        line_start = line_end + 1;
    }

    parser.finish(blake3, content, &lexed.test_spans)
}

fn collect_token_set(lang: LangKind, line: &str, out: &mut FxHashSet<String>) {
    let mut cursor = LexCursor::new(lang, line.as_bytes());
    while let Some(tok) = cursor.next_token() {
        if tok.kind == LexKind::Ident {
            let s = tok.as_str();
            if !is_noise(s) && !out.contains(s) { out.insert(s.to_string()); }
        }
    }
}

fn parse_rs_attr(trimmed: &str, pending_flags: &mut u16, attr_refs: &mut FxHashSet<String>) {
    const RS_ATTR_RULES: &[(&[&str], u16)] = &[
        (&["no_mangle", "export_name"], BOUNDARY_FFI_NO_MANGLE),
        (&["ctor", "dtor", "distributed_slice"], BOUNDARY_CTOR_INVENTORY),
        (RS_MACRO_ATTRS, BOUNDARY_MACRO_DERIVE),
    ];
    for &(needles, flag) in RS_ATTR_RULES {
        if needles.iter().any(|k| trimmed.contains(k)) { *pending_flags |= flag; }
    }
    if trimmed.contains("serde(") && let Some(sym) = quoted_path(trimmed) {
        let tail = sym.rsplit("::").next().unwrap_or(sym);
        if !tail.is_empty() && tail.bytes().all(ident_byte) { attr_refs.insert(tail.to_string()); }
    }
}

fn is_leading_boundary_attr(
    ext: &str, trimmed: &str, pending_flags: &mut u16, attr_refs: &mut FxHashSet<String>,
) -> bool {
    if ext == "rs" && trimmed.starts_with("#[") {
        parse_rs_attr(trimmed, pending_flags, attr_refs);
        return true;
    }
    if ext == "py" && trimmed.starts_with('@') {
        if PY_ROUTE_DECORATORS.iter().any(|p| trimmed.starts_with(p)) {
            *pending_flags |= BOUNDARY_FRAMEWORK_ROUTE;
        }
        return true;
    }
    false
}

fn extract_go_linknames(content: &str) -> FxHashSet<String> {
    let mut out = FxHashSet::default();
    for raw in content.lines() {
        if let Some(rest) = match_prefix(raw.trim(), &["//go:linkname ", "//export "])
            && let Some(name) = rest.split_whitespace().next()
        {
            out.insert(name.to_string());
        }
    }
    out
}

fn collect_py_all(trimmed: &str, buf: &mut String, out: &mut FxHashSet<String>) {
    if buf.is_empty() && !(trimmed.starts_with("__all__") && trimmed.contains('=')) { return; }
    let done = trimmed.contains(']') || trimmed.contains(')');
    let Some(target) = accumulate_stmt(trimmed, buf, true, done, 0) else { return };
    let mut cursor = LexCursor::new(LangKind::Python, target.as_bytes());
    while let Some(tok) = cursor.next_token() {
        if tok.kind == LexKind::StrLit && tok.slice.len() >= 3 {
            let s = tok.as_str();
            let inner = &s[1..s.len().saturating_sub(1)];
            if !inner.is_empty() && inner.bytes().all(ident_byte) { out.insert(inner.to_string()); }
        }
    }
}

fn py_imports_multiline(line: &str, buf: &mut String, out: &mut FxHashSet<TypedImport>) {
    let starts = line.starts_with("from ") && line.contains('(');
    let Some(stmt) = accumulate_stmt(line, buf, starts, line.contains(')'), 0) else { return };
    if let Some(rest) = stmt.strip_prefix("from ") {
        parse_py_from_import(rest, out);
    } else if let Some(rest) = stmt.strip_prefix("import ") {
        for item in rest.split(',') {
            if let Some(head) = item.split_whitespace().next().filter(|s| !s.is_empty()) {
                push_rt_import(out, head.to_string());
            }
        }
    }
}

fn parse_py_from_import(rest: &str, out: &mut FxHashSet<TypedImport>) {
    let Some((mod_part, after)) = rest.split_once(" import ") else {
        if let Some(module) = rest.split_whitespace().next() { push_rt_import(out, module.to_string()); }
        return;
    };
    let module = mod_part.trim();
    if module.is_empty() { return; }
    let dots_only = module.bytes().all(|b| b == b'.');
    if !dots_only { push_rt_import(out, module.to_string()); }
    let sep = if dots_only { "" } else { "." };
    for item in after.trim_matches(['(', ')']).split(',') {
        if let Some(sym) = first_ident(item) { push_rt_import(out, format!("{module}{sep}{sym}")); }
    }
}

fn rs_imports_multiline(
    t: &str, buf: &mut String, out: &mut FxHashSet<TypedImport>, globs: &mut FxHashSet<String>,
) {
    const MOD_PREFIXES: &[&str] = &["mod ", "pub mod ", "pub(crate) mod ", "pub(super) mod "];
    const USE_PREFIXES: &[&str] = &["use ", "pub use ", "pub(crate) use ", "pub(super) use "];

    if buf.is_empty()
        && t.ends_with(';')
        && let Some(mod_rest) = match_prefix(t, MOD_PREFIXES)
        && let Some(id) = first_ident(mod_rest)
    {
        out.insert(TypedImport { target: id.to_string(), kind: EdgeKind::ModDecl });
        return;
    }

    let starts = match_prefix(t, USE_PREFIXES).is_some();
    let Some(stmt) = accumulate_stmt(t, buf, starts, t.contains(';'), 0) else { return };
    let is_pub_reexport = stmt.starts_with("pub use ");
    let mut scan = stmt.as_ref();
    while let Some(pos) = scan.find("crate::") {
        let rest = &scan[pos + 7..];
        expand_rs_use_tree(rest, "", out, is_pub_reexport, globs);
        scan = rest;
    }

    for (kw, prefix) in [
        ("use super::", "super::"),
        ("pub use super::", "super::"),
        ("use self::", ""),
        ("pub use self::", ""),
    ] {
        if let Some(rest) = stmt.strip_prefix(kw) {
            expand_rs_use_tree(rest, prefix, out, is_pub_reexport, globs);
        }
    }
}

fn join_use_prefix(prefix: &str, path: &str) -> String {
    if prefix.is_empty() { path.to_string() } else { format!("{prefix}{path}") }
}

fn push_rt_import(out: &mut FxHashSet<TypedImport>, target: String) {
    out.insert(TypedImport { target, kind: EdgeKind::Runtime });
}

fn expand_rs_brace_tree(
    head: &str, body: &str, prefix: &str, out: &mut FxHashSet<TypedImport>,
    is_pub_reexport: bool, globs: &mut FxHashSet<String>,
) {
    let mod_path = join_use_prefix(prefix, &head.replace("::", "/"));
    if !mod_path.is_empty() {
        push_rt_import(out, mod_path.clone());
        if let Some(first) = head.split("::").next().filter(|f| !f.is_empty() && *f != head) {
            push_rt_import(out, join_use_prefix(prefix, first));
        }
    }

    for part in split_top_commas(matching_brace_slice(body)) {
        let tp = part.trim();
        if tp == "*" && is_pub_reexport { globs.insert(mod_path.clone()); }
        else if tp != "self" && tp != "*" {
            expand_rs_use_tree(tp, &format!("{mod_path}/"), out, is_pub_reexport, globs);
        }
    }
}

fn emit_rs_use_segs(
    before_as: &str, prefix: &str, out: &mut FxHashSet<TypedImport>,
    is_pub_reexport: bool, globs: &mut FxHashSet<String>,
) {
    let segs: Vec<&str> = before_as.split("::")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if segs.is_empty() { return; }

    if segs.last() == Some(&"*") && is_pub_reexport && segs.len() >= 2 {
        let full = join_use_prefix(prefix, &segs[..segs.len() - 1].join("/"));
        globs.insert(full.clone());
        push_rt_import(out, full);
        return;
    }

    if let Some(first) = first_ident(segs[0]) { push_rt_import(out, join_use_prefix(prefix, first)); }
    if segs.len() < 2 { return; }
    let parts: Vec<&str> = segs[..segs.len() - 1].iter()
        .filter_map(|s| first_ident(s))
        .collect();
    if !parts.is_empty() || !prefix.is_empty() {
        push_rt_import(out, join_use_prefix(prefix, &parts.join("/")));
    }
}

fn expand_rs_use_tree(
    rest: &str, prefix: &str, out: &mut FxHashSet<TypedImport>,
    is_pub_reexport: bool, globs: &mut FxHashSet<String>,
) {
    let trimmed = rest.trim().trim_end_matches(';').trim();
    if let Some(inner) = trimmed.strip_prefix('{') {
        for part in split_top_commas(matching_brace_slice(inner)) {
            expand_rs_use_tree(part, prefix, out, is_pub_reexport, globs);
        }
        return;
    }
    let head_as = trimmed.split(" as ").next().unwrap_or(trimmed);
    let before_as = head_as.trim();
    if let Some(brace_pos) = before_as.find("::{") {
        expand_rs_brace_tree(&before_as[..brace_pos], &before_as[brace_pos + 3..], prefix, out, is_pub_reexport, globs);
        return;
    }
    emit_rs_use_segs(before_as, prefix, out, is_pub_reexport, globs);
}

fn matching_brace_slice(after_open: &str) -> &str {
    let mut depth = 1usize;
    for (idx, b) in after_open.bytes().enumerate() {
        if b == b'{' { depth += 1; continue; }
        if b != b'}' { continue; }
        depth -= 1;
        if depth == 0 { return &after_open[..idx]; }
    }
    after_open
}

pub(crate) fn leading_indent(line: &str) -> usize {
    line.bytes().take_while(|&b| matches!(b, b' ' | b'\t')).count()
}

pub(crate) fn strip_py_comment(line: &str) -> &str {
    let mut cut = line.len();
    scan_code(line.as_bytes(), b"\"'", |i, b| {
        if b == b'#' { cut = i; return ControlFlow::Break(()); }
        ControlFlow::Continue(())
    });
    line[..cut].trim_end()
}

pub(crate) fn bracket_delta(s: &str) -> i32 {
    let mut d = 0i32;
    scan_code(s.as_bytes(), b"\"'", |_, b| {
        match b {
            b'(' | b'[' | b'{' => d += 1,
            b')' | b']' | b'}' => d -= 1,
            _ => {}
        }
        ControlFlow::Continue(())
    });
    d
}

pub(crate) fn py_def_colon(sig: &str) -> Option<usize> {
    let (mut depth, mut closed, mut found) = (0i32, false, None);
    scan_code(sig.as_bytes(), b"\"'", |i, b| {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => { depth -= 1; closed |= b == b')' && depth == 0; }
            b':' if depth == 0 && closed => { found = Some(i); return ControlFlow::Break(()); }
            _ => {}
        }
        ControlFlow::Continue(())
    });
    found
}

pub(crate) fn norm_sig(sig: &str) -> String {
    let mut collapsed = String::with_capacity(sig.len());
    for (idx, word) in sig.split_whitespace().enumerate() {
        if idx > 0 { collapsed.push(' '); }
        collapsed.push_str(word);
    }

    let bytes = collapsed.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        let next = bytes.get(i + 1).copied();
        let drop_sp = b == b' '
            && (matches!(out.last(), Some(b'(' | b'<' | b'[')) || matches!(next, Some(b')' | b'>' | b']' | b',')));
        if drop_sp || (b == b',' && matches!(next, Some(b')' | b'>'))) { continue; }
        if matches!(b, b')' | b'>') && out.last() == Some(&b',') { out.pop(); }
        out.push(b);
    }
    String::from_utf8(out).unwrap_or(collapsed)
}

fn step_sig_delim(c: u8, d: &mut [i32; 3]) -> bool {
    match c {
        b'(' => d[0] += 1,
        b')' => d[0] = (d[0] - 1).max(0),
        b'[' => d[1] += 1,
        b']' => d[1] = (d[1] - 1).max(0),
        b'<' => d[2] += 1,
        b'>' => d[2] = (d[2] - 1).max(0),
        _ => return false,
    }
    true
}

fn step_sig_byte(
    b: &[u8], i: usize, allow_semi: bool, stop_arrow: bool,
    d: &mut [i32; 3], skip_to: &mut usize, found: &mut Option<usize>,
) -> ControlFlow<()> {
    if i < *skip_to { return ControlFlow::Continue(()); }
    let (c, top) = (b[i], *d == [0, 0, 0]);
    let two = b.get(i..i + 2);
    if two == Some(b"->") || (two == Some(b"=>") && !(stop_arrow && top)) {
        *skip_to = i + 2;
        return ControlFlow::Continue(());
    }
    if two == Some(b"=>") { *found = Some(i); return ControlFlow::Break(()); }
    if step_sig_delim(c, d) || !top { return ControlFlow::Continue(()); }
    if c == b'{' && b.get(i + 1) == Some(&b'}') {
        *skip_to = i + 2;
        return ControlFlow::Continue(());
    }
    if c == b'{' || (allow_semi && c == b';') {
        *found = Some(i);
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}

pub(crate) fn sig_end(sig: &str, allow_semi: bool, stop_arrow: bool) -> Option<usize> {
    let b = sig.as_bytes();
    let (mut d, mut skip_to, mut found) = ([0i32; 3], 0usize, None);
    scan_code(b, b"\"'`", |i, _| step_sig_byte(b, i, allow_semi, stop_arrow, &mut d, &mut skip_to, &mut found));
    found
}

pub(crate) fn join_sig_lines(
    lines: &[&str], i: &mut usize, strip_hash: bool, stop: impl Fn(&str) -> bool,
) -> String {
    let first = lines[*i].trim();
    let mut sig = if strip_hash { strip_py_comment(first) } else { first }.to_string();
    while !stop(&sig) && *i + 1 < lines.len() {
        *i += 1;
        sig.push(' ');
        let next = lines[*i].trim();
        sig.push_str(if strip_hash { strip_py_comment(next) } else { next });
    }
    sig
}

pub(crate) fn split_top_commas(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0usize, 0usize);
    for (idx, b) in s.bytes().enumerate() {
        match b {
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                let part = s[start..idx].trim();
                if !part.is_empty() { out.push(part); }
                start = idx + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() { out.push(tail); }
    out
}

fn js_imports_multiline(
    line: &str, buf: &mut String, out: &mut FxHashSet<TypedImport>, globs: &mut FxHashSet<String>,
) {
    for (needle, kind) in [("import(", EdgeKind::Dynamic), ("require(", EdgeKind::Runtime)] {
        let mut scan = line;
        while let Some(pos) = scan.find(needle) {
            let after = &scan[pos + needle.len()..];
            if let Some(path) = quoted_path(after) {
                out.insert(TypedImport { target: path.to_string(), kind });
            }
            scan = after;
        }
    }

    let starts = (line.starts_with("import ") || line.starts_with("export ")) && line.contains('{');
    let done = line.contains(" from ") || line.ends_with(';');
    let Some(stmt) = accumulate_stmt(line, buf, starts, done, 2048) else { return };

    let st = stmt.trim();
    let is_type_only = st.starts_with("import type ")
        || st.starts_with("export type ")
        || (st.starts_with("import {") && is_all_inline_type_imports(st));
    let kind = if is_type_only { EdgeKind::TypeOnly } else { EdgeKind::Runtime };

    if let Some(pos) = st.find(" from ") {
        if let Some(path) = quoted_path(&st[pos + 6..]) {
            out.insert(TypedImport { target: path.to_string(), kind });
            if st.starts_with("export *") { globs.insert(path.to_string()); }
        }
    } else if st.starts_with("import ") && !st.starts_with("import(")
        && let Some(path) = quoted_path(st)
    {
        out.insert(TypedImport { target: path.to_string(), kind });
    }
}

fn is_all_inline_type_imports(st: &str) -> bool {
    let Some(open) = st.find('{') else { return false };
    let Some(close) = st[open + 1..].find('}') else { return false };
    let mut it = st[open + 1..open + 1 + close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .peekable();
    it.peek().is_some() && it.all(|item| item.starts_with("type "))
}

fn extract_rs_def(t: &str) -> Option<(&str, u16, bool)> {
    const RS_EXTRA_FNS: &[&str] = &[
        "pub extern \"C\" fn ", "pub unsafe extern \"C\" fn ", "pub unsafe fn ", "pub(crate) unsafe fn ",
    ];
    let internal = if t.starts_with("pub(crate) ") || t.starts_with("pub(super) ") { DEF_VIS_INTERNAL } else { 0 };
    let ffi = if t.contains("extern \"C\"") || t.contains("extern \"system\"") { BOUNDARY_FFI_NO_MANGLE } else { 0 };
    if let Some(rest) = match_prefix(t, RS_EXTRA_FNS) {
        return first_ident(rest).map(|id| (id, internal | ffi, true));
    }
    match_prefix(t, RUST_PUB)
        .and_then(first_ident)
        .map(|id| (id, internal | ffi, t.contains(" fn ")))
}

fn extract_def_with_meta<'a>(ext: &str, t: &'a str, top: bool) -> Option<(&'a str, u16, bool)> {
    if ext == "py" && top {
        return match_prefix(t, PY_DEFS).and_then(first_ident)
            .map(|id| (id, 0, t.starts_with("def ") || t.starts_with("async def ")));
    }
    if ext == "rs" && top { return extract_rs_def(t); }
    if is_js(ext) {
        return match_prefix(t, TS_EXPORT).and_then(first_ident)
            .map(|id| (id, 0, t.contains("function ") || t.contains("=>")));
    }
    if ext == "go" {
        return go_export(t).map(|id| (id, 0, t.starts_with("func ")));
    }
    None
}
