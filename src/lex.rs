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

struct SentinelLexer<'a> {
    bytes: &'a [u8],
    out: Vec<u8>,
    test_spans: Vec<(usize, usize)>,
    modes: Vec<LexMode>,
    brace_depth: usize,
    prev_sig: u8,
    prev_word: &'a [u8],
    pos: usize,
    is_rs: bool,
    is_go: bool,
    is_js: bool,
    is_jsx: bool,
    active_test_seg: Option<usize>,
}

impl<'a> SentinelLexer<'a> {
    fn new(lang: LangKind, bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            out: bytes.to_vec(),
            test_spans: Vec::new(),
            modes: vec![LexMode::Standard],
            brace_depth: 0,
            prev_sig: b';',
            prev_word: b"",
            pos: 0,
            is_rs: lang == LangKind::Rust,
            is_go: lang == LangKind::Go,
            is_js: lang.is_js_like(),
            is_jsx: lang == LangKind::Jsx,
            active_test_seg: None,
        }
    }

    fn flush_test_seg(&mut self, end: usize) {
        if let Some(start) = self.active_test_seg.take()
            && start < end
        {
            self.test_spans.push((start, end));
        }
    }

    fn try_blank_rust_literal(&mut self) -> bool {
        if let Some(res) = raw_str(self.bytes, self.pos) {
            let end = match res {
                RawStr::Closed { end, .. } => end,
                RawStr::Unterminated => self.bytes.len(),
            };
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return true;
        }
        let byte = self.bytes[self.pos];
        if byte == b'\'' {
            let end = char_end(self.bytes, self.pos).unwrap_or(self.pos + 1);
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return true;
        }
        if byte == b'"' {
            let end = quoted_lit(self.bytes, self.pos, true, true).0;
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return true;
        }
        false
    }

    fn step_rust_cfg_test(
        &mut self,
        mut bracket_depth: usize,
        mut paren_depth: usize,
        mut cfg_braces: usize,
        mut in_braced_item: bool,
    ) {
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"//") {
            self.flush_test_seg(self.pos);
            let end = to_eol(self.bytes, self.pos + 2);
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return;
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"/*") {
            self.flush_test_seg(self.pos);
            let (end, _, _) = block_comment(self.bytes, self.pos, true);
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return;
        }
        if self.active_test_seg.is_none() {
            self.active_test_seg = Some(self.pos);
        }
        if self.try_blank_rust_literal() {
            return;
        }
        let byte = self.bytes[self.pos];
        blank_range(&mut self.out, self.pos, self.pos + 1);
        let finished = advance_cfg_test_delim(
            byte,
            &mut bracket_depth,
            &mut paren_depth,
            &mut cfg_braces,
            &mut in_braced_item,
        );
        self.pos += 1;
        if finished {
            self.flush_test_seg(self.pos);
            self.modes.pop();
        } else if let Some(slot) = self.modes.last_mut() {
            *slot = LexMode::RustCfgTest {
                bracket_depth,
                paren_depth,
                brace_depth: cfg_braces,
                in_braced_item,
            };
        }
    }

    fn step_regex_literal(&mut self) {
        if let Some(end) = scan_regex_literal(self.bytes, self.pos) {
            self.pos = end;
            self.prev_sig = b'0';
            self.prev_word = b"";
        } else {
            self.pos += 1;
        }
        self.modes.pop();
    }

    fn step_jsx_text(&mut self) {
        let byte = self.bytes[self.pos];
        if byte == b'<' {
            self.modes.pop();
            self.prev_sig = b'<';
            self.prev_word = b"";
        } else if byte == b'{' {
            let depth = self.brace_depth;
            self.brace_depth += 1;
            self.modes.push(LexMode::JsxExpr { brace_depth: depth });
            self.prev_sig = b'{';
            self.prev_word = b"";
        }
        self.pos += 1;
    }

    fn try_step_trivia_or_rust(&mut self, byte: u8) -> bool {
        if byte == b'\n' || byte.is_ascii_whitespace() {
            self.pos += 1;
            return true;
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"//") {
            let end = to_eol(self.bytes, self.pos + 2);
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return true;
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"/*") {
            let (end, _, _) = block_comment(self.bytes, self.pos, self.is_rs);
            blank_range(&mut self.out, self.pos, end);
            self.pos = end;
            return true;
        }
        if self.is_rs
            && byte == b'#'
            && let Some(attr_end) = match_rust_cfg_test_attr(self.bytes, self.pos)
        {
            blank_range(&mut self.out, self.pos, attr_end);
            self.modes.push(LexMode::RustCfgTest {
                bracket_depth: 0,
                paren_depth: 0,
                brace_depth: 0,
                in_braced_item: false,
            });
            self.active_test_seg = None;
            self.pos = attr_end;
            return true;
        }
        if self.is_rs
            && let Some(res) = raw_str(self.bytes, self.pos)
        {
            self.pos = match res {
                RawStr::Closed { end, .. } => end,
                RawStr::Unterminated => self.bytes.len(),
            };
            self.prev_sig = b'"';
            self.prev_word = b"";
            return true;
        }
        if self.is_rs && byte == b'\'' {
            self.pos = char_end(self.bytes, self.pos).unwrap_or(self.pos + 1);
            self.prev_sig = b'\'';
            self.prev_word = b"";
            return true;
        }
        false
    }

    fn try_step_js_or_quote(&mut self, byte: u8) -> bool {
        if self.is_js
            && byte == b'/'
            && allow_regex(self.prev_sig, self.prev_word)
            && scan_regex_literal(self.bytes, self.pos).is_some()
        {
            self.modes.push(LexMode::RegexLiteral);
            return true;
        }
        if self.is_js && byte == b'`' {
            self.pos = scan_template_sentinel(self.bytes, self.pos + 1, self.brace_depth, &mut self.modes);
            self.prev_sig = b'`';
            self.prev_word = b"";
            return true;
        }
        if byte == b'"' || (!self.is_rs && byte == b'\'') || (self.is_go && byte == b'`') {
            let (end, _, _) = quoted_lit(self.bytes, self.pos, byte != b'`', true);
            self.pos = end;
            self.prev_sig = byte;
            self.prev_word = b"";
            return true;
        }
        if self.is_jsx
            && byte == b'>'
            && self.prev_sig != b'='
            && lookahead_jsx_text(self.bytes, self.pos + 1)
        {
            self.modes.push(LexMode::JsxText);
            self.prev_sig = b'>';
            self.prev_word = b"";
            self.pos += 1;
            return true;
        }
        if self.is_js && (byte.is_ascii_alphabetic() || byte == b'_') {
            let start = self.pos;
            while self.pos < self.bytes.len() && ident_byte(self.bytes[self.pos]) {
                self.pos += 1;
            }
            self.prev_word = &self.bytes[start..self.pos];
            self.prev_sig = self.bytes[self.pos - 1];
            return true;
        }
        false
    }

    fn step_standard(&mut self, top: LexMode) {
        let byte = self.bytes[self.pos];
        if self.try_step_trivia_or_rust(byte) || self.try_step_js_or_quote(byte) {
            return;
        }
        self.prev_word = b"";
        if byte == b'{' {
            self.brace_depth += 1;
        } else if byte == b'}' {
            self.brace_depth = self.brace_depth.saturating_sub(1);
            if let LexMode::JsxExpr { brace_depth: target } = top
                && self.brace_depth == target
            {
                self.modes.pop();
            }
        }
        self.prev_sig = byte;
        self.pos += 1;
    }

    fn run(mut self) -> SentinelLex {
        while self.pos < self.bytes.len() {
            let top = *self.modes.last().unwrap_or(&LexMode::Standard);
            match top {
                LexMode::RustCfgTest {
                    bracket_depth,
                    paren_depth,
                    brace_depth: cfg_braces,
                    in_braced_item,
                } => self.step_rust_cfg_test(bracket_depth, paren_depth, cfg_braces, in_braced_item),
                LexMode::RegexLiteral => self.step_regex_literal(),
                LexMode::JsxText => self.step_jsx_text(),
                LexMode::Template { brace_depth: tpl_depth }
                    if self.brace_depth == tpl_depth && self.bytes[self.pos] == b'}' =>
                {
                    self.modes.pop();
                    self.pos = scan_template_sentinel(
                        self.bytes,
                        self.pos + 1,
                        self.brace_depth,
                        &mut self.modes,
                    );
                    self.prev_sig = b'`';
                    self.prev_word = b"";
                }
                LexMode::Standard | LexMode::JsxExpr { .. } | LexMode::Template { .. } => {
                    self.step_standard(top);
                }
                LexMode::PyFString { .. } => {
                    self.modes.pop();
                }
            }
        }
        self.flush_test_seg(self.bytes.len());
        SentinelLex { bytes: self.out, test_spans: self.test_spans }
    }
}

fn advance_cfg_test_delim(
    byte: u8,
    bracket_depth: &mut usize,
    paren_depth: &mut usize,
    cfg_braces: &mut usize,
    in_braced_item: &mut bool,
) -> bool {
    match byte {
        b'[' => *bracket_depth += 1,
        b']' => *bracket_depth = bracket_depth.saturating_sub(1),
        b'(' => *paren_depth += 1,
        b')' => *paren_depth = paren_depth.saturating_sub(1),
        b'{' => {
            if *bracket_depth == 0 && *paren_depth == 0 {
                *in_braced_item = true;
            }
            *cfg_braces += 1;
        }
        b'}' => {
            *cfg_braces = cfg_braces.saturating_sub(1);
            return *in_braced_item && *cfg_braces == 0 && *bracket_depth == 0 && *paren_depth == 0;
        }
        b';' if !*in_braced_item && *bracket_depth == 0 && *paren_depth == 0 && *cfg_braces == 0 => {
            return true;
        }
        _ => {}
    }
    false
}

/// Run the unified pushdown byte lexer in sentinel-blanking mode.
/// Preserves exact 1:1 byte length and `\n` offsets while capturing `#[cfg(test)]` spans.
pub fn blank_noise_sentinel(ext: &str, content: &str) -> SentinelLex {
    let lang = LangKind::from_ext(ext);
    let bytes = content.as_bytes();
    if lang == LangKind::Python {
        let mut out = bytes.to_vec();
        let mut modes = vec![LexMode::Standard];
        lex_python_sentinel(bytes, &mut out, &mut modes);
        return SentinelLex { bytes: out, test_spans: Vec::new() };
    }
    SentinelLexer::new(lang, bytes).run()
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

fn step_py_fstring_text(
    bytes: &[u8],
    pos: usize,
    quote: u8,
    triple: bool,
    modes: &mut Vec<LexMode>,
) -> usize {
    let byte = bytes[pos];
    if byte == b'\\' {
        return (pos + 2).min(bytes.len());
    }
    if bytes.get(pos..pos + 2) == Some(b"{{") || bytes.get(pos..pos + 2) == Some(b"}}") {
        return pos + 2;
    }
    if byte == b'{' {
        if let Some(slot) = modes.last_mut() {
            *slot = LexMode::PyFString { quote, triple, brace_depth: 1 };
        }
        return pos + 1;
    }
    if triple && bytes.get(pos..pos + 3) == Some(&[quote, quote, quote]) {
        modes.pop();
        return pos + 3;
    }
    if !triple && byte == quote {
        modes.pop();
        return pos + 1;
    }
    pos + 1
}

fn step_py_fstring_expr(
    bytes: &[u8],
    out: &mut [u8],
    pos: usize,
    quote: u8,
    triple: bool,
    mut brace_depth: usize,
    modes: &mut Vec<LexMode>,
) -> usize {
    let byte = bytes[pos];
    if byte == b'#' {
        let end = to_eol(bytes, pos + 1);
        blank_range(out, pos, end);
        return end;
    }
    if matches!(byte, b'"' | b'\'') {
        return quoted_lit(bytes, pos, true, false).0;
    }
    if byte == b'{' {
        brace_depth += 1;
    } else if byte == b'}' {
        brace_depth = brace_depth.saturating_sub(1);
    }
    if let Some(slot) = modes.last_mut() {
        *slot = LexMode::PyFString { quote, triple, brace_depth };
    }
    pos + 1
}

fn step_py_standard(bytes: &[u8], out: &mut [u8], pos: usize, modes: &mut Vec<LexMode>) -> usize {
    let byte = bytes[pos];
    if byte == b'#' {
        let end = to_eol(bytes, pos + 1);
        blank_range(out, pos, end);
        return end;
    }
    if let Some((after_open, quote, triple)) = detect_py_fstring(bytes, pos) {
        modes.push(LexMode::PyFString { quote, triple, brace_depth: 0 });
        return after_open;
    }
    if matches!(byte, b'"' | b'\'') && bytes.get(pos..pos + 3) == Some(&[byte, byte, byte]) {
        let end = skip_py_triple(bytes, pos + 3, byte);
        blank_range(out, pos, end);
        return end;
    }
    if matches!(byte, b'"' | b'\'') {
        return quoted_lit(bytes, pos, true, false).0;
    }
    pos + 1
}

fn lex_python_sentinel(bytes: &[u8], out: &mut [u8], modes: &mut Vec<LexMode>) {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let top = *modes.last().unwrap_or(&LexMode::Standard);
        pos = match top {
            LexMode::PyFString { quote, triple, brace_depth: 0 } => {
                step_py_fstring_text(bytes, pos, quote, triple, modes)
            }
            LexMode::PyFString { quote, triple, brace_depth } => {
                step_py_fstring_expr(bytes, out, pos, quote, triple, brace_depth, modes)
            }
            _ => step_py_standard(bytes, out, pos, modes),
        };
    }
}

struct BalanceChecker<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    prev_sig: u8,
    prev_word: &'a [u8],
    delims: Vec<(u8, usize)>,
    modes: Vec<LexMode>,
    is_rs: bool,
    is_go: bool,
    is_js: bool,
    is_jsx: bool,
}

impl<'a> BalanceChecker<'a> {
    fn new(lang: LangKind, bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            line: 1,
            prev_sig: b';',
            prev_word: b"",
            delims: Vec::new(),
            modes: vec![LexMode::Standard],
            is_rs: lang == LangKind::Rust,
            is_go: lang == LangKind::Go,
            is_js: lang.is_js_like(),
            is_jsx: lang == LangKind::Jsx,
        }
    }

    fn step_regex_literal(&mut self) {
        if let Some(end) = scan_regex_literal(self.bytes, self.pos) {
            self.pos = end;
            self.prev_sig = b'0';
            self.prev_word = b"";
        } else {
            self.pos += 1;
        }
        self.modes.pop();
    }

    fn step_jsx_text(&mut self) {
        let byte = self.bytes[self.pos];
        if byte == b'\n' {
            self.line += 1;
        } else if byte == b'<' {
            self.modes.pop();
            self.prev_sig = b'<';
            self.prev_word = b"";
        } else if byte == b'{' {
            let depth = self.delims.len();
            self.delims.push((b'{', self.line));
            self.modes.push(LexMode::JsxExpr { brace_depth: depth });
            self.prev_sig = b'{';
            self.prev_word = b"";
        }
        self.pos += 1;
    }

    fn try_step_trivia_or_rust(&mut self, byte: u8) -> Result<bool, String> {
        if byte == b'\n' || byte.is_ascii_whitespace() {
            self.line += usize::from(byte == b'\n');
            self.pos += 1;
            return Ok(true);
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"//") {
            self.pos = to_eol(self.bytes, self.pos + 2);
            return Ok(true);
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"/*") {
            let start = self.line;
            let (end, nl, closed) = block_comment(self.bytes, self.pos, self.is_rs);
            if !closed {
                return Err(format!("unterminated block comment at L{start}"));
            }
            self.line += nl;
            self.pos = end;
            return Ok(true);
        }
        if self.is_rs
            && let Some(res) = raw_str(self.bytes, self.pos)
        {
            return match res {
                RawStr::Closed { end, newlines } => {
                    self.line += newlines;
                    self.pos = end;
                    self.prev_sig = b'"';
                    self.prev_word = b"";
                    Ok(true)
                }
                RawStr::Unterminated => {
                    Err(format!("unterminated Rust raw string at L{}", self.line))
                }
            };
        }
        Ok(false)
    }

    fn scan_checked_quote(&mut self, byte: u8) -> Result<(), String> {
        if byte == b'\'' && self.is_rs {
            self.pos = char_end(self.bytes, self.pos).unwrap_or(self.pos + 1);
            self.prev_sig = b'\'';
            self.prev_word = b"";
            return Ok(());
        }
        let start = self.line;
        let (end, nl, closed) =
            quoted_lit(self.bytes, self.pos, byte != b'`', byte == b'`' || self.is_rs);
        if !closed {
            return Err(format!("unterminated string literal at L{start}"));
        }
        self.line += nl;
        self.pos = end;
        self.prev_sig = byte;
        self.prev_word = b"";
        Ok(())
    }

    fn try_step_js_or_quote(&mut self, byte: u8) -> Result<bool, String> {
        if self.is_js
            && byte == b'/'
            && allow_regex(self.prev_sig, self.prev_word)
            && scan_regex_literal(self.bytes, self.pos).is_some()
        {
            self.modes.push(LexMode::RegexLiteral);
            return Ok(true);
        }
        if self.is_js && byte == b'`' {
            self.pos += 1;
            scan_template_checked(
                self.bytes,
                &mut self.pos,
                &mut self.line,
                self.delims.len(),
                &mut self.modes,
            )?;
            self.prev_sig = b'`';
            self.prev_word = b"";
            return Ok(true);
        }
        if self.is_jsx && byte == b'>' && self.prev_sig != b'=' {
            self.pos += 1;
            self.prev_sig = b'>';
            self.prev_word = b"";
            if lookahead_jsx_text(self.bytes, self.pos) {
                self.modes.push(LexMode::JsxText);
            }
            return Ok(true);
        }
        if byte == b'"' || (self.is_go && byte == b'`') || byte == b'\'' {
            self.scan_checked_quote(byte)?;
            return Ok(true);
        }
        if self.is_js && (byte.is_ascii_alphabetic() || byte == b'_') {
            let start = self.pos;
            while self.pos < self.bytes.len() && ident_byte(self.bytes[self.pos]) {
                self.pos += 1;
            }
            self.prev_word = &self.bytes[start..self.pos];
            self.prev_sig = self.bytes[self.pos - 1];
            return Ok(true);
        }
        Ok(false)
    }

    fn step_standard(&mut self, top: LexMode) -> Result<(), String> {
        let byte = self.bytes[self.pos];
        if self.try_step_trivia_or_rust(byte)? || self.try_step_js_or_quote(byte)? {
            return Ok(());
        }
        self.prev_word = b"";
        if matches!(byte, b'{' | b'(' | b'[') {
            self.delims.push((byte, self.line));
        } else if matches!(byte, b'}' | b')' | b']') {
            pop_delim(byte, self.line, &mut self.delims)?;
            if let LexMode::JsxExpr { brace_depth } = top
                && byte == b'}'
                && self.delims.len() == brace_depth
            {
                self.modes.pop();
            }
        }
        self.prev_sig = byte;
        self.pos += 1;
        Ok(())
    }

    fn run(mut self) -> Result<(), String> {
        while self.pos < self.bytes.len() {
            let top = *self.modes.last().unwrap_or(&LexMode::Standard);
            match top {
                LexMode::RegexLiteral => self.step_regex_literal(),
                LexMode::JsxText => self.step_jsx_text(),
                LexMode::Template { brace_depth }
                    if self.bytes[self.pos] == b'}' && self.delims.len() == brace_depth =>
                {
                    self.modes.pop();
                    self.pos += 1;
                    scan_template_checked(
                        self.bytes,
                        &mut self.pos,
                        &mut self.line,
                        self.delims.len(),
                        &mut self.modes,
                    )?;
                    self.prev_sig = b'`';
                    self.prev_word = b"";
                }
                LexMode::Standard | LexMode::JsxExpr { .. } | LexMode::Template { .. } => {
                    self.step_standard(top)?;
                }
                LexMode::PyFString { .. } | LexMode::RustCfgTest { .. } => {
                    self.modes.pop();
                }
            }
        }
        if self.modes.iter().any(|mode| matches!(mode, LexMode::Template { .. })) {
            return Err("unterminated JS/TS template literal interpolation".into());
        }
        if let Some((open, open_line)) = self.delims.pop() {
            return Err(format!("unclosed `{}` opened at L{open_line}", open as char));
        }
        Ok(())
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
    BalanceChecker::new(lang, source.as_bytes()).run()
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
