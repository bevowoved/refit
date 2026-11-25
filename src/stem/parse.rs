use super::analysis::{
    BOUNDARY_CTOR_INVENTORY, BOUNDARY_DYNAMIC_REF, BOUNDARY_FFI_NO_MANGLE,
    BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_MACRO_DERIVE, BOUNDARY_PY_ALL, DEF_VIS_INTERNAL, EdgeKind,
    ModuleMetrics,
};
use crate::lex::{
    LangKind, LexCursor, LexKind, RUST_PUB, TS_EXPORT, blank_noise_sentinel, first_ident,
    go_export, ident_byte, match_prefix, quoted_path,
};
use crate::platform::parse_go_block_lines;
use crate::source::{count_lines, is_js};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

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

#[inline]
fn insert_non_noise(set: &mut FxHashSet<String>, ident: Option<&str>) {
    if let Some(name) = ident.filter(|s| !is_noise(s)) {
        set.insert(name.to_string());
    }
}

#[derive(Default)]
struct ParseState {
    defs_map: FxHashMap<String, u16>,
    typed_imports: FxHashSet<TypedImport>,
    glob_reexports: FxHashSet<String>,
    refs: FxHashSet<String>,
    intra_hits: FxHashMap<String, u16>,
    attr_refs: FxHashSet<String>,
    inventory_refs: FxHashSet<String>,
    py_all_syms: FxHashSet<String>,
    pub_fields: FxHashSet<String>,
    field_reads: FxHashSet<String>,
    rs_buf: String,
    ts_buf: String,
    py_buf: String,
    py_all_buf: String,
    in_inventory: bool,
    pending_flags: u16,
    in_rust_struct: bool,
    struct_depth: i32,
    active_fn: Option<(String, i32)>,
    brace_depth: i32,
    active_fn_lines: usize,
    active_fn_calls: usize,
    total_fns: usize,
    passthrough_fns: usize,
}

struct SymbolParser<'a> {
    ext: &'a str,
    lang: LangKind,
    go_linknames: FxHashSet<String>,
    has_py_dynamic: bool,
    st: ParseState,
}

impl<'a> SymbolParser<'a> {
    fn new(ext: &'a str, content: &str) -> Self {
        let go_linknames = if ext == "go" { extract_go_linknames(content) } else { FxHashSet::default() };
        let has_py_dynamic = ext == "py"
            && ["getattr(", "setattr(", "globals()[", "locals()[", "importlib."]
                .iter()
                .any(|needle| content.contains(needle));
        Self { ext, lang: LangKind::from_ext(ext), go_linknames, has_py_dynamic, st: ParseState::default() }
    }

    fn process_line(&mut self, line: &str, raw_prev: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() { return; }
        if is_leading_boundary_attr(self.ext, trimmed, &mut self.st.pending_flags, &mut self.st.attr_refs) {
            if self.ext == "rs" && trimmed.contains("inventory::submit!") {
                self.st.in_inventory = !trimmed.contains('}');
                collect_token_set(self.lang, trimmed, &mut self.st.inventory_refs);
            }
            return;
        }
        if self.ext == "go" && raw_prev.trim().starts_with("//export ") {
            self.st.pending_flags |= BOUNDARY_FFI_NO_MANGLE;
        }
        if self.ext == "rs" && (self.st.in_inventory || trimmed.contains("inventory::submit!")) {
            collect_token_set(self.lang, trimmed, &mut self.st.inventory_refs);
            self.st.in_inventory = !trimmed.contains('}');
        }
        if self.ext == "py" {
            collect_py_all(trimmed, &mut self.st.py_all_buf, &mut self.st.py_all_syms);
        }

        self.feed_imports(trimmed);
        let indent = line.bytes().take_while(|&b| matches!(b, b' ' | b'\t')).count();
        if self.ext == "py" && indent == 0 && !trimmed.starts_with('@') {
            self.finish_active_fn(true);
        }
        let defined_here = self.record_definition(trimmed, indent == 0);
        if !trimmed.starts_with("#[") && !trimmed.starts_with('@') {
            self.st.pending_flags = 0;
        }

        let (brace_delta, call_opens) = self.scan_line_tokens(trimmed, defined_here);
        self.track_public_fields(trimmed, brace_delta);
        self.update_fn_scope(trimmed, brace_delta, call_opens);
    }

    fn feed_imports(&mut self, trimmed: &str) {
        if self.ext == "py" {
            py_imports_multiline(trimmed, &mut self.st.py_buf, &mut self.st.typed_imports);
        } else if self.ext == "rs" {
            rs_imports_multiline(
                trimmed, &mut self.st.rs_buf, &mut self.st.typed_imports, &mut self.st.glob_reexports,
            );
        } else if is_js(self.ext) {
            js_imports_multiline(
                trimmed, &mut self.st.ts_buf, &mut self.st.typed_imports, &mut self.st.glob_reexports,
            );
        }
    }

    fn finish_active_fn(&mut self, count_passthrough: bool) {
        if self.st.active_fn.take().is_some() {
            if count_passthrough && self.st.active_fn_lines <= 3 && self.st.active_fn_calls <= 2 {
                self.st.passthrough_fns += 1;
            }
            self.st.active_fn_lines = 0;
            self.st.active_fn_calls = 0;
        }
    }

    fn record_ident_ref(&mut self, ident: &str) {
        if self.st.active_fn.as_ref().is_some_and(|(name, _)| name == ident) { return; }
        if !self.st.refs.contains(ident) { self.st.refs.insert(ident.to_string()); }
        let hits = self.st.intra_hits.entry(ident.to_string()).or_insert(0);
        *hits = hits.saturating_add(1);
    }

    fn handle_scanned_ident<'b>(
        &mut self, ident: &'b str, prev_was_dot: bool, defined_here: Option<&str>,
        skipped_decl: &mut bool, pending_field: &mut Option<&'b str>,
    ) {
        if prev_was_dot { *pending_field = Some(ident); }
        if is_noise(ident) { return; }
        if !*skipped_decl && defined_here == Some(ident) {
            *skipped_decl = true;
            return;
        }
        self.record_ident_ref(ident);
    }

    /// Single-pass `LexCursor<'a>` scan over a cleaned line collecting brace delta,
    /// call parens, `.field` reads, and non-def/non-self-call identifier references.
    fn scan_line_tokens(&mut self, trimmed: &str, defined_here: Option<&str>) -> (i32, usize) {
        let mut cursor = LexCursor::new(self.lang, trimmed.as_bytes());
        let (mut brace_delta, mut call_opens) = (0i32, 0usize);
        let (mut skipped_decl, mut prev_was_dot) = (false, false);
        let mut pending_field: Option<&str> = None;

        while let Some(tok) = cursor.next_token() {
            if tok.kind != LexKind::LParen {
                insert_non_noise(&mut self.st.field_reads, pending_field.take());
            } else {
                pending_field = None;
            }
            match tok.kind {
                LexKind::LBrace => brace_delta += 1,
                LexKind::RBrace => brace_delta -= 1,
                LexKind::LParen => call_opens += 1,
                LexKind::Dot => {
                    prev_was_dot = true;
                    continue;
                }
                LexKind::Ident => self.handle_scanned_ident(
                    tok.as_str(), prev_was_dot, defined_here, &mut skipped_decl, &mut pending_field,
                ),
                _ => {}
            }
            prev_was_dot = false;
        }
        insert_non_noise(&mut self.st.field_reads, pending_field);
        (brace_delta, call_opens)
    }

    fn record_definition<'b>(&mut self, trimmed: &'b str, top: bool) -> Option<&'b str> {
        let (sym, def_flags, is_fn) = extract_def_with_meta(self.ext, trimmed, top)?;
        if sym.starts_with('_') || is_noise(sym) { return None; }
        let mut flags = self.st.pending_flags | def_flags;
        if self.ext == "go" {
            if self.go_linknames.contains(sym) {
                flags |= BOUNDARY_FFI_NO_MANGLE;
            } else if GO_IFACE_METHODS.contains(&sym) {
                flags |= BOUNDARY_FRAMEWORK_ROUTE;
            }
        }
        if is_js(self.ext) && NEXT_ROUTE_EXPORTS.contains(&sym) {
            flags |= BOUNDARY_FRAMEWORK_ROUTE;
        }
        if self.has_py_dynamic { flags |= BOUNDARY_DYNAMIC_REF; }
        *self.st.defs_map.entry(sym.to_string()).or_insert(0) |= flags;
        if is_fn {
            self.st.total_fns += 1;
            self.st.active_fn = Some((sym.to_string(), self.st.brace_depth));
            self.st.active_fn_lines = 0;
            self.st.active_fn_calls = 0;
        }
        Some(sym)
    }

    fn track_public_fields(&mut self, trimmed: &str, brace_delta: i32) {
        if self.ext == "rs" {
            if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub(crate) struct ") {
                if trimmed.contains('{') && !trimmed.contains('}') {
                    self.st.in_rust_struct = true;
                    self.st.struct_depth = self.st.brace_depth + brace_delta;
                }
                return;
            }
            if !self.st.in_rust_struct {
                return;
            }
            if let Some(rest) = match_prefix(trimmed, &["pub ", "pub(crate) "])
                && !rest.starts_with("fn ")
                && !rest.starts_with("async ")
            {
                insert_non_noise(&mut self.st.pub_fields, first_ident(rest));
            }
        } else if is_js(self.ext)
            && let Some(rest) = trimmed.strip_prefix("public ")
            && !rest.contains('(')
        {
            insert_non_noise(&mut self.st.pub_fields, first_ident(rest));
        }
    }

    fn update_fn_scope(&mut self, trimmed: &str, brace_delta: i32, call_opens: usize) {
        if self.st.active_fn.is_some() {
            self.st.active_fn_lines += 1;
            self.st.active_fn_calls += call_opens;
        }
        if self.ext == "py" {
            return;
        }
        self.st.brace_depth += brace_delta;
        if self.st.in_rust_struct && self.st.brace_depth < self.st.struct_depth {
            self.st.in_rust_struct = false;
        }
        let Some((_, start_depth)) = self.st.active_fn.as_ref() else {
            return;
        };
        let decl_only = !trimmed.contains('{') && trimmed.ends_with(';');
        let closed = (trimmed.contains('{') || brace_delta < 0) && self.st.brace_depth <= *start_depth;
        if closed || decl_only {
            self.finish_active_fn(!decl_only);
        }
    }

    fn collect_test_span_refs(
        &mut self, content: &str, test_spans: &[(usize, usize)],
    ) -> FxHashSet<String> {
        let (mut test_refs, mut rs_test_use_buf) = (FxHashSet::default(), String::new());
        for &(start, end) in test_spans {
            if start >= end || end > content.len() {
                continue;
            }
            let lines = content[start..end].lines().map(str::trim);
            for test_line in lines.filter(|s| !s.is_empty()) {
                collect_token_set(self.lang, test_line, &mut test_refs);
                if self.ext == "rs" {
                    let (mut dummy_globs, mut test_imps) = (FxHashSet::default(), FxHashSet::default());
                    rs_imports_multiline(test_line, &mut rs_test_use_buf, &mut test_imps, &mut dummy_globs);
                    for mut imp in test_imps {
                        imp.kind = EdgeKind::TestImport;
                        self.st.typed_imports.insert(imp);
                    }
                }
            }
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
                loc: count_lines(content.as_bytes()),
                exported_symbols: defs_v.len(),
                total_fns: self.st.total_fns,
                passthrough_fns: self.st.passthrough_fns,
                pub_fields: into_sorted(self.st.pub_fields),
                field_reads: into_sorted(self.st.field_reads),
            },
            defs: defs_v,
            def_meta,
            refs_tokens: into_sorted(self.st.refs),
            test_refs_tokens: into_sorted(test_refs),
            imports: imps_v,
            typed_imports: typed_v,
            glob_reexports: into_sorted(self.st.glob_reexports),
            attr_refs: attr_refs_v,
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

fn is_leading_boundary_attr(
    ext: &str, trimmed: &str, pending_flags: &mut u16, attr_refs: &mut FxHashSet<String>,
) -> bool {
    if ext == "rs" && trimmed.starts_with("#[") {
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
            if !tail.is_empty() && tail.bytes().all(ident_byte) {
                attr_refs.insert(tail.to_string());
            }
        }
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
        let Some((mod_part, after)) = rest.split_once(" import ") else {
            if let Some(module) = rest.split_whitespace().next() {
                push_rt_import(out, module.to_string());
            }
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
    } else if let Some(rest) = stmt.strip_prefix("import ") {
        for item in rest.split(',') {
            if let Some(head) = item.split_whitespace().next().filter(|s| !s.is_empty()) {
                push_rt_import(out, head.to_string());
            }
        }
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
    let before_as = trimmed
        .split(" as ")
        .next()
        .unwrap_or(trimmed)
        .trim();
    if let Some(brace_pos) = before_as.find("::{") {
        let head = &before_as[..brace_pos];
        let mod_path = join_use_prefix(prefix, &head.replace("::", "/"));
        if !mod_path.is_empty() {
            push_rt_import(out, mod_path.clone());
            if let Some(first) = head.split("::").next().filter(|f| !f.is_empty() && *f != head) {
                push_rt_import(out, join_use_prefix(prefix, first));
            }
        }
        for part in split_top_commas(matching_brace_slice(&before_as[brace_pos + 3..])) {
            let tp = part.trim();
            if tp == "*" && is_pub_reexport {
                globs.insert(mod_path.clone());
            } else if tp != "self" && tp != "*" {
                expand_rs_use_tree(tp, &format!("{mod_path}/"), out, is_pub_reexport, globs);
            }
        }
        return;
    }

    let segs: Vec<&str> = before_as
        .split("::")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if segs.is_empty() {
        return;
    }
    if segs.last() == Some(&"*") && is_pub_reexport && segs.len() >= 2 {
        let full = join_use_prefix(prefix, &segs[..segs.len() - 1].join("/"));
        globs.insert(full.clone());
        push_rt_import(out, full);
        return;
    }
    if let Some(first) = first_ident(segs[0]) {
        push_rt_import(out, join_use_prefix(prefix, first));
    }
    if segs.len() >= 2 {
        let parts: Vec<&str> = segs[..segs.len() - 1]
            .iter()
            .filter_map(|s| first_ident(s))
            .collect();
        if !parts.is_empty() || !prefix.is_empty() {
            push_rt_import(out, join_use_prefix(prefix, &parts.join("/")));
        }
    }
}

fn matching_brace_slice(after_open: &str) -> &str {
    let mut depth = 1usize;
    for (idx, b) in after_open.bytes().enumerate() {
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 { return &after_open[..idx]; }
        }
    }
    after_open
}

fn split_top_commas(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0usize, 0usize);
    for (idx, b) in s.bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
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
    let items: Vec<&str> = st[open + 1..open + 1 + close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    !items.is_empty() && items.iter().all(|it| it.starts_with("type "))
}

fn extract_def_with_meta<'a>(ext: &str, t: &'a str, top: bool) -> Option<(&'a str, u16, bool)> {
    const RS_EXTRA_FNS: &[&str] = &[
        "pub extern \"C\" fn ", "pub unsafe extern \"C\" fn ",
        "pub unsafe fn ", "pub(crate) unsafe fn ",
    ];

    if ext == "py" && top {
        let is_fn = t.starts_with("def ") || t.starts_with("async def ");
        return match_prefix(t, PY_DEFS).and_then(first_ident).map(|id| (id, 0, is_fn));
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
        if let Some(rest) = match_prefix(t, RS_EXTRA_FNS) {
            return first_ident(rest).map(|id| (id, internal | ffi, true));
        }
        return match_prefix(t, RUST_PUB)
            .and_then(first_ident)
            .map(|id| (id, internal | ffi, t.contains(" fn ")));
    }
    if is_js(ext) {
        let is_fn = t.contains("function ") || t.contains("=>");
        return match_prefix(t, TS_EXPORT).and_then(first_ident).map(|id| (id, 0, is_fn));
    }
    if ext == "go" {
        return go_export(t).map(|id| (id, 0, t.starts_with("func ")));
    }
    None
}
