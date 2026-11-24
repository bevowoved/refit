use super::strip::strip_noise;
use crate::lex::{
    RUST_PUB, TS_EXPORT, char_end, first_ident, go_export, ident_byte, match_prefix, quoted_lit,
    quoted_path,
};
use crate::source::is_js;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSymbols {
    pub blake3: String,
    pub defs: Vec<String>,
    pub refs_tokens: Vec<String>,
    pub imports: Vec<String>,
}

#[inline]
fn is_noise(name: &str) -> bool {
    name.len() < 3 || NOISE_WORDS.contains(&name)
}

pub fn parse_symbols(ext: &str, content: &str, blake3: String) -> FileSymbols {
    let cleaned = strip_noise(ext, content);
    let mut defs = HashSet::new();
    let mut imports = HashSet::new();
    let mut refs = HashSet::new();
    let mut in_go_imp = false;

    for line in cleaned.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        extract_imports(ext, trimmed, &mut in_go_imp, &mut imports);
        let top = !line.starts_with([' ', '\t']);
        if let Some(sym) = extract_def(ext, trimmed, top)
            && !sym.starts_with('_')
            && !is_noise(sym)
        {
            defs.insert(sym.to_string());
        }
        collect_refs(ext, trimmed, &mut refs);
    }

    for d in &defs {
        refs.remove(d);
    }

    let mut defs_v: Vec<String> = defs.into_iter().collect();
    let mut refs_v: Vec<String> = refs.into_iter().collect();
    let mut imps_v: Vec<String> = imports.into_iter().collect();
    defs_v.sort();
    refs_v.sort();
    imps_v.sort();

    FileSymbols {
        blake3,
        defs: defs_v,
        refs_tokens: refs_v,
        imports: imps_v,
    }
}

fn extract_imports(ext: &str, t: &str, in_go: &mut bool, out: &mut HashSet<String>) {
    if ext == "py" {
        py_imports(t, out);
    } else if ext == "rs" {
        rs_imports(t, out);
    } else if is_js(ext) {
        js_imports(t, out);
    } else if ext == "go" {
        go_imports(t, in_go, out);
    }
}

fn py_imports(t: &str, out: &mut HashSet<String>) {
    if let Some(rest) = t.strip_prefix("from ") {
        if let Some((mod_part, after)) = rest.split_once(" import ") {
            let m = mod_part.trim();
            if !m.is_empty() && m.bytes().all(|b| b == b'.') {
                for item in after.trim_matches(['(', ')']).split(',') {
                    if let Some(sym) = first_ident(item) {
                        out.insert(format!("{m}{sym}"));
                    }
                }
            } else if !m.is_empty() {
                out.insert(m.to_string());
            }
        } else if let Some(m) = rest.split_whitespace().next() {
            out.insert(m.to_string());
        }
        return;
    }

    if let Some(rest) = t.strip_prefix("import ") {
        for item in rest.split(',') {
            let head = item.split_whitespace().next().unwrap_or("");
            if !head.is_empty() {
                out.insert(head.to_string());
            }
        }
    }
}

fn push_braced(rest: &str, prefix: &str, out: &mut HashSet<String>) {
    if let Some(inner) = rest.strip_prefix('{') {
        if let Some(close) = inner.find('}') {
            for item in inner[..close].split(',') {
                if let Some(id) = first_ident(item) {
                    out.insert(format!("{prefix}{id}"));
                }
            }
        }
    } else if let Some(id) = first_ident(rest) {
        out.insert(format!("{prefix}{id}"));
    }
}

fn rs_imports(t: &str, out: &mut HashSet<String>) {
    let mut scan = t;
    while let Some(pos) = scan.find("crate::") {
        let rest = &scan[pos + 7..];
        push_braced(rest, "", out);
        scan = rest;
    }

    if let Some(rest) = t.strip_prefix("use super::") {
        push_braced(rest, "super::", out);
        return;
    }

    let mod_rest = t.strip_prefix("use self::").or_else(|| {
        t.ends_with(';')
            .then(|| t.strip_prefix("mod ").or_else(|| t.strip_prefix("pub mod ")))
            .flatten()
    });
    if let Some(id) = mod_rest.and_then(first_ident) {
        out.insert(id.to_string());
    }
}

fn js_imports(t: &str, out: &mut HashSet<String>) {
    if let Some(pos) = t.find(" from ") {
        if let Some(p) = quoted_path(&t[pos + 6..]) {
            out.insert(p.to_string());
        }
    } else if t.starts_with("import ")
        && let Some(p) = quoted_path(t)
    {
        out.insert(p.to_string());
    }

    if let Some(pos) = t.find("require(")
        && let Some(p) = quoted_path(&t[pos + 8..])
    {
        out.insert(p.to_string());
    }
}

fn go_imports(t: &str, in_block: &mut bool, out: &mut HashSet<String>) {
    if t.starts_with("import (") {
        *in_block = true;
        return;
    }
    if *in_block && t.starts_with(')') {
        *in_block = false;
        return;
    }
    if (*in_block || t.starts_with("import "))
        && let Some(last) = quoted_path(t).and_then(|p| p.rsplit('/').next())
    {
        out.insert(last.to_string());
    }
}

fn extract_def<'a>(ext: &str, t: &'a str, top: bool) -> Option<&'a str> {
    if ext == "py" && top {
        return match_prefix(t, PY_DEFS).and_then(first_ident);
    }
    if ext == "rs" && top {
        return match_prefix(t, RUST_PUB).and_then(first_ident);
    }
    if is_js(ext) {
        return match_prefix(t, TS_EXPORT).and_then(first_ident);
    }
    if ext == "go" {
        return go_export(t);
    }
    None
}

/// Extract unique identifier tokens via borrowed slices (zero temporary String allocations).
fn collect_refs(ext: &str, t: &str, refs: &mut HashSet<String>) {
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
            if !b[start].is_ascii_digit() && !is_noise(tok) && !refs.contains(tok) {
                refs.insert(tok.to_string());
            }
            continue;
        }
        i += 1;
    }
}
