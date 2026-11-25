use super::analysis::{
    BOUNDARY_CTOR_INVENTORY, BOUNDARY_DYNAMIC_REF, BOUNDARY_FFI_NO_MANGLE,
    BOUNDARY_FRAMEWORK_ROUTE, BOUNDARY_MACRO_DERIVE, BOUNDARY_PY_ALL, DEF_VIS_INTERNAL, EdgeKind,
    ModuleMetrics,
};
use crate::lex::{
    LangKind, LexCursor, LexKind, RUST_PUB, TS_EXPORT, blank_noise_sentinel, first_ident,
    go_export, ident_byte, match_prefix, quoted_path,
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

#[derive(Default)]
struct ImportAcc {
    in_go: bool,
    rs_buf: String,
    ts_buf: String,
    py_buf: String,
}

struct SymbolParser<'a> {
    ext: &'a str,
    lang: LangKind,
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
    go_linknames: FxHashSet<String>,
    has_py_dynamic: bool,
    imp_acc: ImportAcc,
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

impl<'a> SymbolParser<'a> {
    fn new(ext: &'a str, content: &str, raw_lines: &[&str]) -> Self {
        let go_linknames = if ext == "go" {
            extract_go_linknames(raw_lines)
        } else {
            FxHashSet::default()
        };
        let has_py_dynamic = ext == "py"
            && ["getattr(", "setattr(", "globals()[", "locals()[", "importlib."]
                .iter()
                .any(|needle| content.contains(needle));
        Self {
            ext,
            lang: LangKind::from_ext(ext),
            defs_map: FxHashMap::default(),
            typed_imports: FxHashSet::default(),
            glob_reexports: FxHashSet::default(),
            refs: FxHashSet::default(),
            intra_hits: FxHashMap::default(),
            attr_refs: FxHashSet::default(),
            inventory_refs: FxHashSet::default(),
            py_all_syms: FxHashSet::default(),
            pub_fields: FxHashSet::default(),
            field_reads: FxHashSet::default(),
            go_linknames,
            has_py_dynamic,
            imp_acc: ImportAcc::default(),
            py_all_buf: String::new(),
            in_inventory: false,
            pending_flags: 0,
            in_rust_struct: false,
            struct_depth: 0,
            active_fn: None,
            brace_depth: 0,
            active_fn_lines: 0,
            active_fn_calls: 0,
            total_fns: 0,
            passthrough_fns: 0,
        }
    }

    fn process_line(&mut self, line: &str, raw_prev: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        if is_leading_boundary_attr(self.ext, trimmed, &mut self.pending_flags, &mut self.attr_refs) {
            if self.ext == "rs" && trimmed.contains("inventory::submit!") {
                self.in_inventory = !trimmed.contains('}');
                collect_token_set(self.lang, trimmed, &mut self.inventory_refs);
            }
            return;
        }

        if self.ext == "go" && raw_prev.trim().starts_with("//export ") {
            self.pending_flags |= BOUNDARY_FFI_NO_MANGLE;
        }
        if self.ext == "rs" && (self.in_inventory || trimmed.contains("inventory::submit!")) {
            collect_token_set(self.lang, trimmed, &mut self.inventory_refs);
            self.in_inventory = !trimmed.contains('}');
        }
        if self.ext == "py" {
            collect_py_all(trimmed, &mut self.py_all_buf, &mut self.py_all_syms);
        }

        self.imp_acc
            .feed(self.ext, trimmed, &mut self.typed_imports, &mut self.glob_reexports);

        let indent = line.bytes().take_while(|&byte| matches!(byte, b' ' | b'\t')).count();
        if self.ext == "py"
            && indent == 0
            && !trimmed.starts_with('@')
            && self.active_fn.take().is_some()
        {
            if self.active_fn_lines <= 3 && self.active_fn_calls <= 2 {
                self.passthrough_fns += 1;
            }
            self.active_fn_lines = 0;
            self.active_fn_calls = 0;
        }

        let defined_here = self.record_definition(trimmed, indent == 0);
        if !trimmed.starts_with("#[") && !trimmed.starts_with('@') {
            self.pending_flags = 0;
        }

        let (brace_delta, call_opens) = self.scan_line_tokens(trimmed, defined_here);
        self.track_public_fields(trimmed, brace_delta);
        self.update_fn_scope(trimmed, brace_delta, call_opens);
    }

    /// Single-pass `LexCursor<'a>` scan over a cleaned line collecting brace delta,
    /// call parens, `.field` reads, and non-def/non-self-call identifier references.
    fn scan_line_tokens(&mut self, trimmed: &str, defined_here: Option<&str>) -> (i32, usize) {
        let mut cursor = LexCursor::new(self.lang, trimmed.as_bytes());
        let mut brace_delta = 0i32;
        let mut call_opens = 0usize;
        let mut skipped_decl = false;
        let mut prev_was_dot = false;
        let mut pending_field: Option<&str> = None;
        let active_name = self.active_fn.as_ref().map(|(name, _)| name.as_str());

        while let Some(tok) = cursor.next_token() {
            if let Some(field) = pending_field.take()
                && tok.kind != LexKind::LParen
                && field.len() >= 3
                && !is_noise(field)
            {
                self.field_reads.insert(field.to_string());
            }
            match tok.kind {
                LexKind::LBrace => brace_delta += 1,
                LexKind::RBrace => brace_delta -= 1,
                LexKind::LParen => call_opens += 1,
                LexKind::Dot => {
                    prev_was_dot = true;
                    continue;
                }
                LexKind::Ident => {
                    let ident = tok.as_str();
                    if prev_was_dot {
                        pending_field = Some(ident);
                    }
                    if !is_noise(ident) {
                        if !skipped_decl && defined_here == Some(ident) {
                            skipped_decl = true;
                        } else if active_name != Some(ident) {
                            if !self.refs.contains(ident) {
                                self.refs.insert(ident.to_string());
                            }
                            let hits = self.intra_hits.entry(ident.to_string()).or_insert(0);
                            *hits = hits.saturating_add(1);
                        }
                    }
                }
                _ => {}
            }
            prev_was_dot = false;
        }
        if let Some(field) = pending_field
            && field.len() >= 3
            && !is_noise(field)
        {
            self.field_reads.insert(field.to_string());
        }
        (brace_delta, call_opens)
    }

    fn record_definition<'b>(&mut self, trimmed: &'b str, top: bool) -> Option<&'b str> {
        let (sym, def_flags, is_fn) = extract_def_with_meta(self.ext, trimmed, top)?;
        if sym.starts_with('_') || is_noise(sym) {
            return None;
        }
        let mut flags = self.pending_flags | def_flags;
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
        if self.has_py_dynamic {
            flags |= BOUNDARY_DYNAMIC_REF;
        }
        *self.defs_map.entry(sym.to_string()).or_insert(0) |= flags;
        if is_fn {
            self.total_fns += 1;
            self.active_fn = Some((sym.to_string(), self.brace_depth));
            self.active_fn_lines = 0;
            self.active_fn_calls = 0;
        }
        Some(sym)
    }

    fn track_public_fields(&mut self, trimmed: &str, brace_delta: i32) {
        if self.ext == "rs" {
            if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub(crate) struct ") {
                if trimmed.contains('{') && !trimmed.contains('}') {
                    self.in_rust_struct = true;
                    self.struct_depth = self.brace_depth + brace_delta;
                }
                return;
            }
            if !self.in_rust_struct {
                return;
            }
            let Some(rest) = trimmed
                .strip_prefix("pub ")
                .or_else(|| trimmed.strip_prefix("pub(crate) "))
            else {
                return;
            };
            if !rest.starts_with("fn ")
                && !rest.starts_with("async ")
                && let Some(field) = first_ident(rest)
                && field.len() >= 3
                && !is_noise(field)
            {
                self.pub_fields.insert(field.to_string());
            }
        } else if is_js(self.ext)
            && let Some(rest) = trimmed.strip_prefix("public ")
            && !rest.contains('(')
            && let Some(field) = first_ident(rest)
            && field.len() >= 3
            && !is_noise(field)
        {
            self.pub_fields.insert(field.to_string());
        }
    }

    fn update_fn_scope(&mut self, trimmed: &str, brace_delta: i32, call_opens: usize) {
        if self.active_fn.is_some() {
            self.active_fn_lines += 1;
            self.active_fn_calls += call_opens;
        }
        if self.ext == "py" {
            return;
        }
        self.brace_depth += brace_delta;
        if self.in_rust_struct && self.brace_depth < self.struct_depth {
            self.in_rust_struct = false;
        }
        let Some((_, start_depth)) = self.active_fn.as_ref() else {
            return;
        };
        let single_line_closed = trimmed.contains('{') && self.brace_depth <= *start_depth;
        let decl_only = !trimmed.contains('{') && trimmed.ends_with(';');
        let multi_line_closed = brace_delta < 0 && self.brace_depth <= *start_depth;
        if single_line_closed || decl_only || multi_line_closed {
            if self.active_fn_lines <= 3 && self.active_fn_calls <= 2 && !decl_only {
                self.passthrough_fns += 1;
            }
            self.active_fn = None;
            self.active_fn_lines = 0;
            self.active_fn_calls = 0;
        }
    }

    fn finish(mut self, blake3: String, content: &str, test_spans: &[(usize, usize)]) -> FileSymbols {
        if self.active_fn.is_some() && self.active_fn_lines <= 3 && self.active_fn_calls <= 2 {
            self.passthrough_fns += 1;
        }

        let mut test_refs: FxHashSet<String> = FxHashSet::default();
        let mut rs_test_use_buf = String::new();
        for &(start, end) in test_spans {
            if start >= end || end > content.len() {
                continue;
            }
            for test_line in content[start..end].lines().map(str::trim).filter(|s| !s.is_empty()) {
                collect_token_set(self.lang, test_line, &mut test_refs);
                if self.ext == "rs" {
                    let (mut dummy_globs, mut test_imps) = (FxHashSet::default(), FxHashSet::default());
                    rs_imports_multiline(test_line, &mut rs_test_use_buf, &mut test_imps, &mut dummy_globs);
                    for mut imp in test_imps {
                        imp.kind = EdgeKind::TestImport;
                        self.typed_imports.insert(imp);
                    }
                }
            }
        }

        for sym in std::mem::take(&mut self.py_all_syms) {
            if let Some(flags) = self.defs_map.get_mut(&sym) {
                *flags |= BOUNDARY_PY_ALL;
            }
        }
        let mut attr_refs_v: Vec<String> = Vec::with_capacity(self.attr_refs.len());
        for sym in std::mem::take(&mut self.attr_refs) {
            if let Some(flags) = self.defs_map.get_mut(&sym) {
                *flags |= BOUNDARY_DYNAMIC_REF;
            }
            attr_refs_v.push(sym);
        }
        attr_refs_v.sort();
        for sym in std::mem::take(&mut self.inventory_refs) {
            if let Some(flags) = self.defs_map.get_mut(&sym) {
                *flags |= BOUNDARY_CTOR_INVENTORY;
            }
        }

        let mut defs_pairs: Vec<(String, DefMeta)> = self
            .defs_map
            .into_iter()
            .map(|(name, boundary_flags)| {
                let hits = self.intra_hits.get(&name).copied().unwrap_or(0);
                (name, DefMeta { intra_non_def_hits: hits, boundary_flags })
            })
            .collect();
        defs_pairs.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let (mut defs_v, mut def_meta) = (Vec::with_capacity(defs_pairs.len()), Vec::with_capacity(defs_pairs.len()));
        for (name, meta) in defs_pairs {
            defs_v.push(name);
            def_meta.push(meta);
        }

        let mut refs_v: Vec<String> = self.refs.into_iter().collect();
        let mut test_refs_v: Vec<String> = test_refs.into_iter().collect();
        let mut typed_v: Vec<TypedImport> = self.typed_imports.into_iter().collect();
        let mut glob_v: Vec<String> = self.glob_reexports.into_iter().collect();
        refs_v.sort();
        test_refs_v.sort();
        typed_v.sort();
        glob_v.sort();

        let mut imps_v: Vec<String> = typed_v
            .iter()
            .filter(|imp| imp.kind != EdgeKind::TestImport)
            .map(|imp| imp.target.clone())
            .collect();
        imps_v.sort();
        imps_v.dedup();

        let mut pf_v: Vec<String> = self.pub_fields.into_iter().collect();
        let mut fr_v: Vec<String> = self.field_reads.into_iter().collect();
        pf_v.sort();
        fr_v.sort();

        FileSymbols {
            blake3,
            metrics: ModuleMetrics {
                loc: count_lines(content.as_bytes()),
                exported_symbols: defs_v.len(),
                total_fns: self.total_fns,
                passthrough_fns: self.passthrough_fns,
                pub_fields: pf_v,
                field_reads: fr_v,
            },
            defs: defs_v,
            def_meta,
            refs_tokens: refs_v,
            test_refs_tokens: test_refs_v,
            imports: imps_v,
            typed_imports: typed_v,
            glob_reexports: glob_v,
            attr_refs: attr_refs_v,
        }
    }
}

pub fn parse_symbols(ext: &str, content: &str, blake3: String) -> FileSymbols {
    let lexed = blank_noise_sentinel(ext, content);
    let raw_lines: Vec<&str> = content.lines().collect();
    let mut parser = SymbolParser::new(ext, content, &raw_lines);

    let bytes = content.as_bytes();
    let mut line_start = 0usize;
    let mut line_idx = 0usize;
    while line_start <= bytes.len() {
        let line_end = match memchr::memchr(b'\n', &bytes[line_start..]) {
            Some(rel) => line_start + rel,
            None => bytes.len(),
        };
        let clean_cow = lexed.clean_slice(content, line_start, line_end);
        let line = clean_cow.trim_end_matches('\r');
        let raw_prev = line_idx
            .checked_sub(1)
            .and_then(|idx| raw_lines.get(idx))
            .copied()
            .unwrap_or("");
        parser.process_line(line, raw_prev);
        if line_end >= bytes.len() {
            break;
        }
        line_start = line_end + 1;
        line_idx += 1;
    }

    parser.finish(blake3, content, &lexed.test_spans)
}

fn collect_token_set(lang: LangKind, line: &str, out: &mut FxHashSet<String>) {
    let mut cursor = LexCursor::new(lang, line.as_bytes());
    while let Some(tok) = cursor.next_token() {
        if tok.kind == LexKind::Ident {
            let s = tok.as_str();
            if !is_noise(s) && !out.contains(s) {
                out.insert(s.to_string());
            }
        }
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
        if RS_MACRO_ATTRS.iter().any(|k| trimmed.contains(k)) {
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
        if PY_ROUTE_DECORATORS.iter().any(|p| trimmed.starts_with(p)) {
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
        for prefix in ["//go:linkname ", "//export "] {
            if let Some(rest) = t.strip_prefix(prefix)
                && let Some(name) = rest.split_whitespace().next()
            {
                out.insert(name.to_string());
            }
        }
    }
    out
}

fn collect_py_all(trimmed: &str, buf: &mut String, out: &mut FxHashSet<String>) {
    let target = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(trimmed);
        if !trimmed.contains(']') && !trimmed.contains(')') {
            return;
        }
        std::mem::take(buf)
    } else if trimmed.starts_with("__all__") && trimmed.contains('=') {
        if !trimmed.contains(']') && !trimmed.contains(')') {
            buf.push_str(trimmed);
            return;
        }
        trimmed.to_string()
    } else {
        return;
    };

    let mut cursor = LexCursor::new(LangKind::Python, target.as_bytes());
    while let Some(tok) = cursor.next_token() {
        if tok.kind == LexKind::StrLit && tok.slice.len() >= 3 {
            let inner = &tok.as_str()[1..tok.as_str().len().saturating_sub(1)];
            if !inner.is_empty() && inner.bytes().all(ident_byte) {
                out.insert(inner.to_string());
            }
        }
    }
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

fn py_imports_multiline(line: &str, buf: &mut String, out: &mut FxHashSet<TypedImport>) {
    let stmt = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(line);
        if !line.contains(')') {
            return;
        }
        std::mem::take(buf)
    } else if line.starts_with("from ") && line.contains('(') && !line.contains(')') {
        buf.push_str(line);
        return;
    } else {
        line.to_string()
    };

    if let Some(rest) = stmt.strip_prefix("from ") {
        let Some((mod_part, after)) = rest.split_once(" import ") else {
            if let Some(module) = rest.split_whitespace().next() {
                out.insert(TypedImport { target: module.to_string(), kind: EdgeKind::Runtime });
            }
            return;
        };
        let module = mod_part.trim();
        if module.is_empty() {
            return;
        }
        let dots_only = module.bytes().all(|b| b == b'.');
        if !dots_only {
            out.insert(TypedImport { target: module.to_string(), kind: EdgeKind::Runtime });
        }
        let sep = if dots_only { "" } else { "." };
        for item in after.trim_matches(['(', ')']).split(',') {
            if let Some(sym) = first_ident(item) {
                out.insert(TypedImport { target: format!("{module}{sep}{sym}"), kind: EdgeKind::Runtime });
            }
        }
    } else if let Some(rest) = stmt.strip_prefix("import ") {
        for item in rest.split(',') {
            let head = item.split_whitespace().next().unwrap_or("");
            if !head.is_empty() {
                out.insert(TypedImport { target: head.to_string(), kind: EdgeKind::Runtime });
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

    let is_use_start = match_prefix(t, USE_PREFIXES).is_some();
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
            out.insert(TypedImport { target, kind: EdgeKind::Runtime });
        }
        scan = rest;
    }

    for (kw, prefix) in [
        ("use super::", "super::"),
        ("pub use super::", "super::"),
        ("use self::", ""),
        ("pub use self::", ""),
    ] {
        if let Some(rest) = stmt.strip_prefix(kw) {
            let mut expanded = Vec::new();
            expand_rs_use_tree(rest, prefix, &mut expanded, is_pub_reexport, globs);
            for target in expanded {
                out.insert(TypedImport { target, kind: EdgeKind::Runtime });
            }
        }
    }
}

fn join_use_prefix(prefix: &str, path: &str) -> String {
    if prefix.is_empty() { path.to_string() } else { format!("{prefix}{path}") }
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
        for part in split_top_commas(matching_brace_slice(inner)) {
            expand_rs_use_tree(part, prefix, out, is_pub_reexport, globs);
        }
        return;
    }
    let before_as = trimmed.split(" as ").next().unwrap_or(trimmed).trim();
    if let Some(brace_pos) = before_as.find("::{") {
        let head = &before_as[..brace_pos];
        let mod_path = join_use_prefix(prefix, &head.replace("::", "/"));
        if !mod_path.is_empty() {
            out.push(mod_path.clone());
            if let Some(first) = head.split("::").next()
                && !first.is_empty()
                && first != head
            {
                out.push(join_use_prefix(prefix, first));
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

    let segs: Vec<&str> = before_as.split("::").map(str::trim).filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return;
    }
    if segs.last() == Some(&"*") && is_pub_reexport && segs.len() >= 2 {
        let full = join_use_prefix(prefix, &segs[..segs.len() - 1].join("/"));
        globs.insert(full.clone());
        out.push(full);
        return;
    }
    if let Some(first) = first_ident(segs[0]) {
        out.push(join_use_prefix(prefix, first));
    }
    if segs.len() >= 2 {
        let mut acc = String::from(prefix);
        for (idx, seg) in segs.iter().take(segs.len() - 1).enumerate() {
            if let Some(ident) = first_ident(seg) {
                if idx > 0 {
                    acc.push('/');
                }
                acc.push_str(ident);
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
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return &after_open[..idx];
            }
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
    line: &str,
    buf: &mut String,
    out: &mut FxHashSet<TypedImport>,
    globs: &mut FxHashSet<String>,
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

    let stmt = if !buf.is_empty() {
        buf.push(' ');
        buf.push_str(line);
        if !line.contains(" from ") && !line.ends_with(';') {
            if buf.len() > 2048 {
                buf.clear();
            }
            return;
        }
        std::mem::take(buf)
    } else if (line.starts_with("import ") || line.starts_with("export "))
        && line.contains('{')
        && !line.contains(" from ")
        && !line.ends_with(';')
    {
        buf.push_str(line);
        return;
    } else {
        line.to_string()
    };

    let st = stmt.trim();
    let is_type_only = st.starts_with("import type ")
        || st.starts_with("export type ")
        || (st.starts_with("import {") && is_all_inline_type_imports(st));
    let kind = if is_type_only { EdgeKind::TypeOnly } else { EdgeKind::Runtime };

    if let Some(pos) = st.find(" from ") {
        if let Some(path) = quoted_path(&st[pos + 6..]) {
            out.insert(TypedImport { target: path.to_string(), kind });
            if st.starts_with("export *") {
                globs.insert(path.to_string());
            }
        }
    } else if st.starts_with("import ")
        && !st.starts_with("import(")
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

fn go_imports(t: &str, in_block: &mut bool, out: &mut FxHashSet<TypedImport>) {
    if t.starts_with("import (") {
        *in_block = true;
    } else if *in_block && t.starts_with(')') {
        *in_block = false;
    } else if (*in_block || t.starts_with("import "))
        && let Some(full) = quoted_path(t)
        && !full.is_empty()
    {
        out.insert(TypedImport { target: full.to_string(), kind: EdgeKind::Runtime });
    }
}

fn extract_def_with_meta<'a>(ext: &str, t: &'a str, top: bool) -> Option<(&'a str, u16, bool)> {
    const RS_EXTRA_FNS: &[&str] = &[
        "pub extern \"C\" fn ",
        "pub unsafe extern \"C\" fn ",
        "pub unsafe fn ",
        "pub(crate) unsafe fn ",
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
