use std::borrow::Cow;
use std::ops::ControlFlow;
use std::path::Path;

pub const RUST_PUB: &[&str] = &[
    "pub async fn ", "pub fn ", "pub(crate) async fn ", "pub(crate) fn ",
    "pub struct ", "pub(crate) struct ", "pub enum ", "pub(crate) enum ",
    "pub trait ", "pub(crate) trait ", "pub type ", "pub(crate) type ",
    "pub const ", "pub(crate) const ",
];

pub const TS_EXPORT: &[&str] = &[
    "export default async function ", "export default function ",
    "export default class ", "export async function ", "export function ",
    "export class ", "export interface ", "export type ", "export const ",
];

/// Sentinel byte (`0xFF`, never valid in UTF-8) used to blank comments and `#[cfg(test)]`
/// items in-place while preserving exact 1:1 byte offsets and `\n` line numbers.
pub const TEST_BLANK_SENTINEL: u8 = 0xFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangKind {
    Rust,
    Js,
    Jsx,
    Go,
    Python,
    Other,
}

impl LangKind {
    pub fn from_ext(ext: &str) -> Self {
        if ext.eq_ignore_ascii_case("rs") {
            Self::Rust
        } else if ext.eq_ignore_ascii_case("go") {
            Self::Go
        } else if ext.eq_ignore_ascii_case("py") {
            Self::Python
        } else if ext.eq_ignore_ascii_case("tsx") || ext.eq_ignore_ascii_case("jsx") {
            Self::Jsx
        } else if matches!(
            ext.to_ascii_lowercase().as_str(),
            "ts" | "js" | "mjs" | "cjs" | "mts" | "cts"
        ) {
            Self::Js
        } else {
            Self::Other
        }
    }

    #[inline]
    pub fn is_js_like(self) -> bool {
        matches!(self, Self::Js | Self::Jsx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexMode {
    Standard,
    JsxText,
    JsxExpr { brace_depth: usize },
    Template { brace_depth: usize },
    RegexLiteral,
    PyFString { quote: u8, triple: bool, brace_depth: usize },
    RustCfgTest {
        bracket_depth: usize,
        paren_depth: usize,
        brace_depth: usize,
        in_braced_item: bool,
    },
}

/// 1:1 byte-aligned lexing view where comment bytes and Rust `#[cfg(test)]` items are
/// overwritten with `TEST_BLANK_SENTINEL` (`0xFF`) while preserving every `\n` verbatim.
#[derive(Debug, Clone)]
pub struct SentinelLex {
    pub bytes: Vec<u8>,
    pub test_spans: Vec<(usize, usize)>,
}

impl SentinelLex {
    /// Materialize a clean UTF-8 view of `source[start..end]` with `0xFF` sentinel bytes
    /// removed (borrowing zero-copy whenever no sentinel bytes occur in the slice).
    pub fn clean_slice<'a>(&self, source: &'a str, start: usize, end: usize) -> Cow<'a, str> {
        let slice = &self.bytes[start..end];
        if !slice.contains(&TEST_BLANK_SENTINEL) {
            return Cow::Borrowed(&source[start..end]);
        }
        let mut out = String::with_capacity(end - start);
        let mut p = start;
        let mut prev_was_ident = false;
        while p < end {
            if self.bytes[p] == TEST_BLANK_SENTINEL {
                let mut q = p + 1;
                while q < end && self.bytes[q] == TEST_BLANK_SENTINEL {
                    q += 1;
                }
                let next_is_ident = q < end && ident_byte(self.bytes[q]);
                if prev_was_ident && next_is_ident {
                    out.push(' ');
                }
                p = q;
                continue;
            }
            let run_start = p;
            while p < end && self.bytes[p] != TEST_BLANK_SENTINEL {
                p += 1;
            }
            out.push_str(&source[run_start..p]);
            prev_was_ident = p > run_start && ident_byte(self.bytes[p - 1]);
        }
        Cow::Owned(out)
    }

    pub fn to_clean_string(&self, source: &str) -> String {
        self.clean_slice(source, 0, self.bytes.len()).into_owned()
    }
}

#[inline]
pub fn ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
pub fn left_bound(bytes: &[u8], pos: usize) -> bool {
    pos == 0 || !ident_byte(bytes[pos - 1])
}

pub fn has_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    haystack.match_indices(needle).any(|(pos, _)| {
        let end = pos + needle.len();
        left_bound(bytes, pos) && (end == bytes.len() || !ident_byte(bytes[end]))
    })
}

/// Compare two lines after whitespace normalization without allocating.
#[inline]
pub fn eq_ws(a: &str, b: &str) -> bool {
    a.split_whitespace().eq(b.split_whitespace())
}

/// Return the leading ASCII identifier slice without allocating.
pub fn first_ident(s: &str) -> Option<&str> {
    let trimmed = s.trim_start();
    let end = trimmed.bytes().take_while(|&b| ident_byte(b)).count();
    if end == 0 || trimmed.as_bytes()[0].is_ascii_digit() {
        return None;
    }
    Some(&trimmed[..end])
}

/// Extract the first single- or double-quoted slice without allocating.
pub fn quoted_path(s: &str) -> Option<&str> {
    let q = s.find(['"', '\''])?;
    let quote = s.as_bytes()[q] as char;
    let rest = &s[q + 1..];
    Some(&rest[..rest.find(quote)?])
}

/// Return the exported Go identifier from a trimmed `func` or `type` line.
pub fn go_export(trimmed: &str) -> Option<&str> {
    let rest = if let Some(r) = trimmed.strip_prefix("func ") {
        if r.starts_with('(') {
            r.find(')').map(|p| &r[p + 1..])?
        } else {
            r
        }
    } else {
        trimmed.strip_prefix("type ")?
    };
    first_ident(rest).filter(|id| id.as_bytes()[0].is_ascii_uppercase())
}

pub fn match_prefix<'a>(s: &'a str, table: &[&str]) -> Option<&'a str> {
    table.iter().find_map(|&p| s.strip_prefix(p))
}

#[inline]
pub fn to_eol(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i] != b'\n' {
        i += 1;
    }
    i
}

/// Scan `/* ... */` starting at `i`. Returns `(end, newlines, closed)`.
pub fn block_comment(bytes: &[u8], mut i: usize, nested: bool) -> (usize, usize, bool) {
    i += 2;
    let (mut depth, mut nl) = (1usize, 0usize);

    while i < bytes.len() && depth > 0 {
        if bytes[i] == b'\n' {
            nl += 1;
            i += 1;
        } else if nested && bytes.get(i..i + 2) == Some(b"/*") {
            depth += 1;
            i += 2;
        } else if bytes.get(i..i + 2) == Some(b"*/") {
            depth -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    (i, nl, depth == 0)
}

/// End offset if `bytes[i]` starts a Rust char literal instead of a lifetime.
pub fn char_end(bytes: &[u8], i: usize) -> Option<usize> {
    if i + 2 >= bytes.len() || bytes[i] != b'\'' {
        return None;
    }
    if bytes[i + 1] == b'\\' {
        let mut p = (i + 3).min(bytes.len());
        while p < bytes.len() && p < i + 14 && !matches!(bytes[p], b'\'' | b'\n') {
            p += 1;
        }
        return (p < bytes.len() && bytes[p] == b'\'').then_some(p + 1);
    }

    let ch = std::str::from_utf8(&bytes[i + 1..]).ok()?.chars().next()?;
    let next = i + 1 + ch.len_utf8();
    (next < bytes.len() && bytes[next] == b'\'' && ch != '\n').then_some(next + 1)
}

pub enum RawStr {
    Closed { end: usize, newlines: usize },
    Unterminated,
}

/// Scan a Rust raw string (`r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`).
pub fn raw_str(bytes: &[u8], i: usize) -> Option<RawStr> {
    let b = bytes[i];
    let start_r = if b == b'r' {
        i
    } else if matches!(b, b'b' | b'c') && bytes.get(i + 1) == Some(&b'r') {
        i + 1
    } else {
        return None;
    };
    if !left_bound(bytes, i) {
        return None;
    }

    let mut p = start_r + 1;
    while p < bytes.len() && bytes[p] == b'#' {
        p += 1;
    }
    let hashes = p - (start_r + 1);
    if bytes.get(p) != Some(&b'"') {
        return None;
    }

    p += 1;
    let mut newlines = 0usize;
    while p < bytes.len() {
        if bytes[p] == b'\n' {
            newlines += 1;
        } else if bytes[p] == b'"' && has_hashes(bytes, p + 1, hashes) {
            return Some(RawStr::Closed { end: p + 1 + hashes, newlines });
        }
        p += 1;
    }
    Some(RawStr::Unterminated)
}

#[inline]
fn has_hashes(bytes: &[u8], start: usize, count: usize) -> bool {
    let Some(slice) = bytes.get(start..start + count) else { return false };
    slice.iter().all(|&c| c == b'#')
}

/// Scan a quoted literal starting at `bytes[i]`. Returns `(end, newlines, closed)`.
pub fn quoted_lit(
    bytes: &[u8],
    mut i: usize,
    escapes: bool,
    multiline: bool,
) -> (usize, usize, bool) {
    let q = bytes[i];
    i += 1;
    let mut nl = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
        if escapes && c == b'\\' {
            i = (i + 2).min(bytes.len());
            continue;
        }
        if c == b'\n' {
            nl += 1;
            if !multiline {
                return (i, nl, false);
            }
        }
        i += 1;
        if c == q {
            return (i, nl, true);
        }
    }
    (i, nl, false)
}

/// Visit every byte outside quoted string literals without allocating.
pub fn scan_code(
    bytes: &[u8],
    quotes: &[u8],
    mut on_byte: impl FnMut(usize, u8) -> ControlFlow<()>,
) {
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if quotes.contains(&b) {
            i = quoted_lit(bytes, i, true, true).0;
            continue;
        }
        if on_byte(i, b).is_break() {
            break;
        }
        i += 1;
    }
}

/// Strip `//`, `/* */`, and trailing commas from JSONC; borrows when unchanged.
pub fn strip_jsonc(raw: &str) -> Cow<'_, str> {
    if !raw.contains("//") && !raw.contains("/*") && !raw.contains(',') {
        return Cow::Borrowed(raw);
    }

    let b = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0usize;

    while i < b.len() {
        if b[i] == b'"' {
            let end = quoted_lit(b, i, true, true).0;
            out.push_str(&raw[i..end]);
            i = end;
        } else if b.get(i..i + 2) == Some(b"//") {
            i = to_eol(b, i + 2);
        } else if b.get(i..i + 2) == Some(b"/*") {
            i = block_comment(b, i, false).0;
        } else if b[i] == b',' && matches!(next_json(b, i + 1), Some(b'}' | b']')) {
            i += 1;
        } else {
            let Some(ch) = raw[i..].chars().next() else { break };
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    Cow::Owned(out)
}

fn next_json(b: &[u8], mut k: usize) -> Option<u8> {
    while k < b.len() {
        if b[k].is_ascii_whitespace() {
            k += 1;
        } else if b.get(k..k + 2) == Some(b"//") {
            k = to_eol(b, k + 2);
        } else if b.get(k..k + 2) == Some(b"/*") {
            k = block_comment(b, k, false).0;
        } else {
            return Some(b[k]);
        }
    }
    None
}

// ============================================================================
// Unified Pushdown Stateful Byte Lexer (Standard / JsxText / JsxExpr /
// Template / RegexLiteral / PyFString / RustCfgTest)
// ============================================================================

#[inline]
fn blank_range(out: &mut [u8], start: usize, end: usize) {
    let stop = end.min(out.len());
    for b in &mut out[start..stop] {
        if *b != b'\n' {
            *b = TEST_BLANK_SENTINEL;
        }
    }
}

fn allow_regex(prev_sig: u8, prev_word: &[u8]) -> bool {
    let kw = matches!(
        prev_word,
        b"return" | b"case" | b"throw" | b"in" | b"of" | b"typeof"
            | b"void" | b"delete" | b"yield" | b"await" | b"new"
    );
    let punct = matches!(
        prev_sig,
        b'(' | b'[' | b'{' | b',' | b';' | b':' | b'=' | b'!' | b'&' | b'|'
            | b'?' | b'+' | b'-' | b'~'
    );
    kw || punct
}

fn scan_regex_literal(bytes: &[u8], i: usize) -> Option<usize> {
    let (mut p, mut in_class) = (i + 1, false);
    while p < bytes.len() && bytes[p] != b'\n' {
        if bytes[p] == b'\\' {
            p = (p + 2).min(bytes.len());
            continue;
        }
        if bytes[p] == b'[' {
            in_class = true;
        } else if bytes[p] == b']' {
            in_class = false;
        } else if bytes[p] == b'/' && !in_class {
            p += 1;
            while p < bytes.len() && bytes[p].is_ascii_alphabetic() {
                p += 1;
            }
            return Some(p);
        }
        p += 1;
    }
    None
}

fn lookahead_jsx_text(bytes: &[u8], start: usize) -> bool {
    let (mut look, mut saw_alpha) = (start, false);
    while look < bytes.len() && !matches!(bytes[look], b'<' | b'{') {
        let c = bytes[look];
        if c.is_ascii_alphabetic() {
            saw_alpha = true;
        } else if matches!(c, b';' | b')' | b']' | b'}' | b'?' | b'=' | b'"' | b'`') {
            return false;
        }
        look += 1;
    }
    saw_alpha && look < bytes.len()
}

/// Match `#[cfg(test)]` or `#[cfg(all(test, ...))]` (allowing internal ASCII whitespace).
/// Returns byte end right after the closing `]`.
fn match_rust_cfg_test_attr(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i..i + 2) != Some(b"#[") {
        return None;
    }
    let mut p = i + 2;
    let mut depth = 1usize;
    while p < bytes.len() && depth > 0 {
        match bytes[p] {
            b'[' => depth += 1,
            b']' => depth -= 1,
            b'"' => {
                p = quoted_lit(bytes, p, true, true).0;
                continue;
            }
            _ => {}
        }
        p += 1;
    }
    if depth != 0 {
        return None;
    }
    let inner = std::str::from_utf8(&bytes[i + 2..p - 1]).ok()?;
    let compact: String = inner.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let is_cfg_test = compact == "cfg(test)"
        || compact.starts_with("cfg(all(test,")
        || compact.ends_with(",test))");
    is_cfg_test.then_some(p)
}

fn detect_py_fstring(bytes: &[u8], i: usize) -> Option<(usize, u8, bool)> {
    if !left_bound(bytes, i) {
        return None;
    }
    let mut p = i;
    let mut has_f = false;
    while p < bytes.len() && p - i < 2 {
        match bytes[p] {
            b'f' | b'F' => {
                has_f = true;
                p += 1;
            }
            b'r' | b'R' => p += 1,
            _ => break,
        }
    }
    if !has_f || p >= bytes.len() || !matches!(bytes[p], b'"' | b'\'') {
        return None;
    }
    let q = bytes[p];
    let triple = bytes.get(p..p + 3) == Some(&[q, q, q]);
    let after_open = p + if triple { 3 } else { 1 };
    Some((after_open, q, triple))
}

fn skip_py_triple(bytes: &[u8], mut i: usize, q: u8) -> usize {
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if bytes.get(i..i + 3) == Some(&[q, q, q]) {
            return i + 3;
        }
        i += 1;
    }
    i
}

/// Run the unified pushdown byte lexer in sentinel-blanking mode.
/// Preserves exact 1:1 byte length and `\n` offsets while capturing `#[cfg(test)]` spans.
pub fn blank_noise_sentinel(ext: &str, content: &str) -> SentinelLex {
    let lang = LangKind::from_ext(ext);
    let bytes = content.as_bytes();
    let mut out = bytes.to_vec();
    let mut test_spans = Vec::new();
    let mut modes = vec![LexMode::Standard];
    let mut brace_depth = 0usize;
    let mut prev_sig = b';';
    let mut prev_word: &[u8] = b"";
    let mut i = 0usize;

    if lang == LangKind::Python {
        lex_python_sentinel(bytes, &mut out, &mut modes);
        return SentinelLex { bytes: out, test_spans };
    }

    let is_rs = lang == LangKind::Rust;
    let is_go = lang == LangKind::Go;
    let is_js = lang.is_js_like();
    let is_jsx = lang == LangKind::Jsx;
    let mut active_test_seg: Option<usize> = None;

    while i < bytes.len() {
        let top = *modes.last().unwrap_or(&LexMode::Standard);
        match top {
            LexMode::RustCfgTest {
                mut bracket_depth,
                mut paren_depth,
                brace_depth: mut cfg_braces,
                mut in_braced_item,
            } => {
                if bytes.get(i..i + 2) == Some(b"//") {
                    if let Some(s) = active_test_seg.take()
                        && s < i
                    {
                        test_spans.push((s, i));
                    }
                    let end = to_eol(bytes, i + 2);
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                if bytes.get(i..i + 2) == Some(b"/*") {
                    if let Some(s) = active_test_seg.take()
                        && s < i
                    {
                        test_spans.push((s, i));
                    }
                    let (end, _, _) = block_comment(bytes, i, true);
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                if active_test_seg.is_none() {
                    active_test_seg = Some(i);
                }
                if let Some(res) = raw_str(bytes, i) {
                    let end = match res {
                        RawStr::Closed { end, .. } => end,
                        RawStr::Unterminated => bytes.len(),
                    };
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                let b = bytes[i];
                if b == b'\'' {
                    let end = char_end(bytes, i).unwrap_or(i + 1);
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                if b == b'"' {
                    let end = quoted_lit(bytes, i, true, true).0;
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                blank_range(&mut out, i, i + 1);
                let mut finished = false;
                match b {
                    b'[' => bracket_depth += 1,
                    b']' => bracket_depth = bracket_depth.saturating_sub(1),
                    b'(' => paren_depth += 1,
                    b')' => paren_depth = paren_depth.saturating_sub(1),
                    b'{' => {
                        if bracket_depth == 0 && paren_depth == 0 {
                            in_braced_item = true;
                        }
                        cfg_braces += 1;
                    }
                    b'}' => {
                        cfg_braces = cfg_braces.saturating_sub(1);
                        if in_braced_item && cfg_braces == 0 && bracket_depth == 0 && paren_depth == 0
                        {
                            finished = true;
                        }
                    }
                    b';' if !in_braced_item && bracket_depth == 0 && paren_depth == 0 && cfg_braces == 0 => {
                        finished = true;
                    }
                    _ => {}
                }
                i += 1;
                if finished {
                    if let Some(s) = active_test_seg.take()
                        && s < i
                    {
                        test_spans.push((s, i));
                    }
                    modes.pop();
                } else if let Some(slot) = modes.last_mut() {
                    *slot = LexMode::RustCfgTest {
                        bracket_depth,
                        paren_depth,
                        brace_depth: cfg_braces,
                        in_braced_item,
                    };
                }
            }
            LexMode::RegexLiteral => {
                if let Some(end) = scan_regex_literal(bytes, i) {
                    i = end;
                    prev_sig = b'0';
                    prev_word = b"";
                } else {
                    i += 1;
                }
                modes.pop();
            }
            LexMode::JsxText => {
                let b = bytes[i];
                if b == b'<' {
                    modes.pop();
                    prev_sig = b'<';
                    prev_word = b"";
                    i += 1;
                } else if b == b'{' {
                    let depth = brace_depth;
                    brace_depth += 1;
                    modes.push(LexMode::JsxExpr { brace_depth: depth });
                    prev_sig = b'{';
                    prev_word = b"";
                    i += 1;
                } else {
                    i += 1;
                }
            }
            LexMode::Template { brace_depth: tpl_depth } if brace_depth == tpl_depth && bytes[i] == b'}' => {
                modes.pop();
                i += 1;
                i = scan_template_sentinel(bytes, i, brace_depth, &mut modes);
                prev_sig = b'`';
                prev_word = b"";
            }
            LexMode::Standard | LexMode::JsxExpr { .. } | LexMode::Template { .. } => {
                let b = bytes[i];
                if b == b'\n' || b.is_ascii_whitespace() {
                    i += 1;
                    continue;
                }
                if bytes.get(i..i + 2) == Some(b"//") {
                    let end = to_eol(bytes, i + 2);
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                if bytes.get(i..i + 2) == Some(b"/*") {
                    let (end, _, _) = block_comment(bytes, i, is_rs);
                    blank_range(&mut out, i, end);
                    i = end;
                    continue;
                }
                if is_rs
                    && b == b'#'
                    && let Some(attr_end) = match_rust_cfg_test_attr(bytes, i)
                {
                    blank_range(&mut out, i, attr_end);
                    modes.push(LexMode::RustCfgTest {
                        bracket_depth: 0,
                        paren_depth: 0,
                        brace_depth: 0,
                        in_braced_item: false,
                    });
                    active_test_seg = None;
                    i = attr_end;
                    continue;
                }
                if is_rs
                    && let Some(res) = raw_str(bytes, i)
                {
                    let end = match res {
                        RawStr::Closed { end, .. } => end,
                        RawStr::Unterminated => bytes.len(),
                    };
                    i = end;
                    prev_sig = b'"';
                    prev_word = b"";
                    continue;
                }
                if is_rs && b == b'\'' {
                    i = char_end(bytes, i).unwrap_or(i + 1);
                    prev_sig = b'\'';
                    prev_word = b"";
                    continue;
                }
                if is_js
                    && b == b'/'
                    && allow_regex(prev_sig, prev_word)
                    && scan_regex_literal(bytes, i).is_some()
                {
                    modes.push(LexMode::RegexLiteral);
                    continue;
                }
                if is_js && b == b'`' {
                    i = scan_template_sentinel(bytes, i + 1, brace_depth, &mut modes);
                    prev_sig = b'`';
                    prev_word = b"";
                    continue;
                }
                if b == b'"' || (!is_rs && b == b'\'') || (is_go && b == b'`') {
                    let (end, _, _) = quoted_lit(bytes, i, b != b'`', true);
                    i = end;
                    prev_sig = b;
                    prev_word = b"";
                    continue;
                }
                if is_jsx && b == b'>' && prev_sig != b'=' && lookahead_jsx_text(bytes, i + 1) {
                    modes.push(LexMode::JsxText);
                    prev_sig = b'>';
                    prev_word = b"";
                    i += 1;
                    continue;
                }
                if is_js && (b.is_ascii_alphabetic() || b == b'_') {
                    let start = i;
                    while i < bytes.len() && ident_byte(bytes[i]) {
                        i += 1;
                    }
                    prev_word = &bytes[start..i];
                    prev_sig = bytes[i - 1];
                    continue;
                }

                prev_word = b"";
                if b == b'{' {
                    brace_depth += 1;
                } else if b == b'}' {
                    brace_depth = brace_depth.saturating_sub(1);
                    if let LexMode::JsxExpr { brace_depth: target } = top
                        && brace_depth == target
                    {
                        modes.pop();
                    }
                }
                prev_sig = b;
                i += 1;
            }
            LexMode::PyFString { .. } => {
                modes.pop();
            }
        }
    }

    if let Some(s) = active_test_seg.take()
        && s < bytes.len()
    {
        test_spans.push((s, bytes.len()));
    }

    SentinelLex { bytes: out, test_spans }
}

fn scan_template_sentinel(
    bytes: &[u8],
    mut i: usize,
    brace_depth: usize,
    modes: &mut Vec<LexMode>,
) -> usize {
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            i = (i + 2).min(bytes.len());
            continue;
        }
        if bytes.get(i..i + 2) == Some(b"${") {
            modes.push(LexMode::Template { brace_depth });
            return i + 2;
        }
        if c == b'`' {
            return i + 1;
        }
        i += 1;
    }
    i
}

fn lex_python_sentinel(bytes: &[u8], out: &mut [u8], modes: &mut Vec<LexMode>) {
    let mut i = 0usize;
    while i < bytes.len() {
        let top = *modes.last().unwrap_or(&LexMode::Standard);
        match top {
            LexMode::PyFString { quote, triple, mut brace_depth } => {
                let b = bytes[i];
                if brace_depth == 0 {
                    if b == b'\\' {
                        i = (i + 2).min(bytes.len());
                        continue;
                    }
                    if bytes.get(i..i + 2) == Some(b"{{") || bytes.get(i..i + 2) == Some(b"}}") {
                        i += 2;
                        continue;
                    }
                    if b == b'{' {
                        brace_depth = 1;
                        if let Some(slot) = modes.last_mut() {
                            *slot = LexMode::PyFString { quote, triple, brace_depth };
                        }
                        i += 1;
                        continue;
                    }
                    if triple && bytes.get(i..i + 3) == Some(&[quote, quote, quote]) {
                        modes.pop();
                        i += 3;
                        continue;
                    }
                    if !triple && b == quote {
                        modes.pop();
                        i += 1;
                        continue;
                    }
                    i += 1;
                } else {
                    if b == b'#' {
                        let end = to_eol(bytes, i + 1);
                        blank_range(out, i, end);
                        i = end;
                        continue;
                    }
                    if matches!(b, b'"' | b'\'') {
                        i = quoted_lit(bytes, i, true, false).0;
                        continue;
                    }
                    if b == b'{' {
                        brace_depth += 1;
                    } else if b == b'}' {
                        brace_depth = brace_depth.saturating_sub(1);
                    }
                    if let Some(slot) = modes.last_mut() {
                        *slot = LexMode::PyFString { quote, triple, brace_depth };
                    }
                    i += 1;
                }
            }
            _ => {
                let b = bytes[i];
                if b == b'#' {
                    let end = to_eol(bytes, i + 1);
                    blank_range(out, i, end);
                    i = end;
                    continue;
                }
                if let Some((after_open, quote, triple)) = detect_py_fstring(bytes, i) {
                    modes.push(LexMode::PyFString { quote, triple, brace_depth: 0 });
                    i = after_open;
                    continue;
                }
                if matches!(b, b'"' | b'\'') && bytes.get(i..i + 3) == Some(&[b, b, b]) {
                    let end = skip_py_triple(bytes, i + 3, b);
                    blank_range(out, i, end);
                    i = end;
                    continue;
                }
                if matches!(b, b'"' | b'\'') {
                    i = quoted_lit(bytes, i, true, false).0;
                    continue;
                }
                i += 1;
            }
        }
    }
}

/// Unified pushdown stateful brace & literal balance checker across Rust, TS, JS, JSX, and Go.
pub fn check_balanced_syntax(rel_path: &str, source: &str) -> Result<(), String> {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let lang = LangKind::from_ext(ext);
    if !matches!(lang, LangKind::Rust | LangKind::Go | LangKind::Js | LangKind::Jsx) {
        return Ok(());
    }

    let is_rs = lang == LangKind::Rust;
    let is_go = lang == LangKind::Go;
    let is_js = lang.is_js_like();
    let is_jsx = lang == LangKind::Jsx;
    let bytes = source.as_bytes();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut prev_sig = b';';
    let mut prev_word: &[u8] = b"";
    let mut delims: Vec<(u8, usize)> = Vec::new();
    let mut modes = vec![LexMode::Standard];

    while i < bytes.len() {
        let top = *modes.last().unwrap_or(&LexMode::Standard);
        match top {
            LexMode::RegexLiteral => {
                if let Some(end) = scan_regex_literal(bytes, i) {
                    i = end;
                    prev_sig = b'0';
                    prev_word = b"";
                } else {
                    i += 1;
                }
                modes.pop();
            }
            LexMode::JsxText => {
                let b = bytes[i];
                if b == b'\n' {
                    line += 1;
                    i += 1;
                } else if b == b'<' {
                    modes.pop();
                    prev_sig = b'<';
                    prev_word = b"";
                    i += 1;
                } else if b == b'{' {
                    let depth = delims.len();
                    delims.push((b'{', line));
                    modes.push(LexMode::JsxExpr { brace_depth: depth });
                    prev_sig = b'{';
                    prev_word = b"";
                    i += 1;
                } else {
                    i += 1;
                }
            }
            LexMode::Template { brace_depth } if bytes[i] == b'}' && delims.len() == brace_depth => {
                modes.pop();
                i += 1;
                scan_template_checked(bytes, &mut i, &mut line, delims.len(), &mut modes)?;
                prev_sig = b'`';
                prev_word = b"";
            }
            LexMode::Standard | LexMode::JsxExpr { .. } | LexMode::Template { .. } => {
                let b = bytes[i];
                if b == b'\n' || b.is_ascii_whitespace() {
                    line += usize::from(b == b'\n');
                    i += 1;
                    continue;
                }
                if bytes.get(i..i + 2) == Some(b"//") {
                    i = to_eol(bytes, i + 2);
                    continue;
                }
                if bytes.get(i..i + 2) == Some(b"/*") {
                    let start = line;
                    let (end, nl, closed) = block_comment(bytes, i, is_rs);
                    if !closed {
                        return Err(format!("unterminated block comment at L{start}"));
                    }
                    line += nl;
                    i = end;
                    continue;
                }
                if is_rs
                    && let Some(res) = raw_str(bytes, i)
                {
                    match res {
                        RawStr::Closed { end, newlines } => {
                            line += newlines;
                            i = end;
                            prev_sig = b'"';
                            prev_word = b"";
                            continue;
                        }
                        RawStr::Unterminated => {
                            return Err(format!("unterminated Rust raw string at L{line}"));
                        }
                    }
                }
                if is_js
                    && b == b'/'
                    && allow_regex(prev_sig, prev_word)
                    && scan_regex_literal(bytes, i).is_some()
                {
                    modes.push(LexMode::RegexLiteral);
                    continue;
                }
                if is_js && b == b'`' {
                    i += 1;
                    scan_template_checked(bytes, &mut i, &mut line, delims.len(), &mut modes)?;
                    prev_sig = b'`';
                    prev_word = b"";
                    continue;
                }
                if is_jsx && b == b'>' && prev_sig != b'=' {
                    i += 1;
                    prev_sig = b'>';
                    prev_word = b"";
                    if lookahead_jsx_text(bytes, i) {
                        modes.push(LexMode::JsxText);
                    }
                    continue;
                }
                if b == b'"' || (is_go && b == b'`') || b == b'\'' {
                    if b == b'\'' && is_rs {
                        i = char_end(bytes, i).unwrap_or(i + 1);
                        prev_sig = b'\'';
                        prev_word = b"";
                        continue;
                    }
                    let start = line;
                    let (end, nl, closed) = quoted_lit(bytes, i, b != b'`', b == b'`' || is_rs);
                    if !closed {
                        return Err(format!("unterminated string literal at L{start}"));
                    }
                    line += nl;
                    i = end;
                    prev_sig = b;
                    prev_word = b"";
                    continue;
                }
                if is_js && (b.is_ascii_alphabetic() || b == b'_') {
                    let start = i;
                    while i < bytes.len() && ident_byte(bytes[i]) {
                        i += 1;
                    }
                    prev_word = &bytes[start..i];
                    prev_sig = bytes[i - 1];
                    continue;
                }

                prev_word = b"";
                if matches!(b, b'{' | b'(' | b'[') {
                    delims.push((b, line));
                } else if matches!(b, b'}' | b')' | b']') {
                    pop_delim(b, line, &mut delims)?;
                    if let LexMode::JsxExpr { brace_depth } = top
                        && b == b'}'
                        && delims.len() == brace_depth
                    {
                        modes.pop();
                    }
                }
                prev_sig = b;
                i += 1;
            }
            LexMode::PyFString { .. } | LexMode::RustCfgTest { .. } => {
                modes.pop();
            }
        }
    }

    if modes.iter().any(|m| matches!(m, LexMode::Template { .. })) {
        return Err("unterminated JS/TS template literal interpolation".into());
    }
    if let Some((open, open_line)) = delims.pop() {
        return Err(format!("unclosed `{}` opened at L{open_line}", open as char));
    }
    Ok(())
}

fn scan_template_checked(
    bytes: &[u8],
    i: &mut usize,
    line: &mut usize,
    brace_depth: usize,
    modes: &mut Vec<LexMode>,
) -> Result<(), String> {
    let start = *line;
    while *i < bytes.len() {
        let c = bytes[*i];
        if c == b'\\' {
            *i = (*i + 2).min(bytes.len());
            continue;
        }
        *line += usize::from(c == b'\n');
        if bytes.get(*i..*i + 2) == Some(b"${") {
            *i += 2;
            modes.push(LexMode::Template { brace_depth });
            return Ok(());
        }
        if c == b'`' {
            *i += 1;
            return Ok(());
        }
        *i += 1;
    }
    Err(format!("unterminated template literal at L{start}"))
}

fn pop_delim(b: u8, line: usize, stack: &mut Vec<(u8, usize)>) -> Result<(), String> {
    let expected = match b {
        b'}' => b'{',
        b')' => b'(',
        _ => b'[',
    };
    match stack.pop() {
        Some((open, _)) if open == expected => Ok(()),
        Some((open, open_line)) => Err(format!(
            "mismatched `{}` at L{line} (expected close for `{}` from L{open_line})",
            b as char, open as char
        )),
        None => Err(format!("unexpected closing `{}` at L{line}", b as char)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentinel_preserves_offsets_and_rust_array_semicolon_in_cfg_test() {
        let src = concat!(
            "pub fn live_fn() -> usize { 1 }\n",
            "#[cfg(test)]\n",
            "fn helper(buf: [u8; 16]) -> usize {\n",
            "    let arr: [u8; 16] = [0; 16];\n",
            "    live_fn() + buf.len() + arr.len()\n",
            "}\n",
            "pub fn after_test() -> usize { 2 }\n",
        );
        let lexed = blank_noise_sentinel("rs", src);
        assert_eq!(lexed.bytes.len(), src.len());
        assert_eq!(
            lexed.bytes.iter().filter(|&&b| b == b'\n').count(),
            src.as_bytes().iter().filter(|&&b| b == b'\n').count()
        );
        let cleaned = lexed.to_clean_string(src);
        assert!(cleaned.contains("pub fn live_fn"));
        assert!(cleaned.contains("pub fn after_test"));
        assert!(!cleaned.contains("arr"), "cfg(test) body leaked past [u8; 16]: {cleaned}");
        assert!(!lexed.test_spans.is_empty());
        let test_text: String = lexed
            .test_spans
            .iter()
            .map(|&(a, b)| &src[a..b])
            .collect();
        assert!(test_text.contains("live_fn"));
    }

    #[test]
    fn test_py_fstring_and_jsx_apostrophe_sentinel() {
        let py = "def greet(user_name):\n    # comment ignore_me\n    return f\"hi {user_name}\"\n";
        let lexed = blank_noise_sentinel("py", py);
        assert_eq!(lexed.bytes.len(), py.len());
        let cleaned = lexed.to_clean_string(py);
        assert!(cleaned.contains("user_name"));
        assert!(!cleaned.contains("ignore_me"));

        let tsx = "export const V = () => <div>Don't stop {runAction()}</div>;\n";
        assert!(check_balanced_syntax("src/V.tsx", tsx).is_ok());
        let lexed_tsx = blank_noise_sentinel("tsx", tsx);
        assert!(lexed_tsx.to_clean_string(tsx).contains("runAction"));
    }
}
