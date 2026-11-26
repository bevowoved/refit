use bstr::ByteSlice;
use smallvec::{SmallVec, smallvec};
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
pub enum LangKind { Rust, Js, Jsx, Go, Python, Other }

impl LangKind {
    pub fn from_ext(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Self::Rust,
            "go" => Self::Go,
            "py" => Self::Python,
            "tsx" | "jsx" => Self::Jsx,
            "ts" | "js" | "mjs" | "cjs" | "mts" | "cts" => Self::Js,
            _ => Self::Other,
        }
    }
}

/// Static per-language scanner rules shared across all crate consumers.
#[derive(Debug, Clone, Copy)]
pub struct LangSpec {
    pub nested_block_comments: bool,
    pub hash_line_comments: bool,
    pub slash_comments: bool,
    pub backtick_raw_str: bool,
    pub js_templates_regex: bool,
    pub jsx_text: bool,
    pub rust_raw_and_cfg: bool,
}

impl LangSpec {
    pub const fn for_lang(kind: LangKind) -> Self {
        let (is_rs, is_go, is_py) = (
            matches!(kind, LangKind::Rust), matches!(kind, LangKind::Go), matches!(kind, LangKind::Python),
        );
        Self {
            nested_block_comments: is_rs, hash_line_comments: is_py, slash_comments: !is_py,
            backtick_raw_str: is_go, js_templates_regex: matches!(kind, LangKind::Js | LangKind::Jsx),
            jsx_text: matches!(kind, LangKind::Jsx), rust_raw_and_cfg: is_rs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexMode {
    Standard,
    JsxText,
    JsxExpr { brace_depth: usize },
    Template { brace_depth: usize },
    PyFString { quote: u8, triple: bool, brace_depth: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexKind {
    Ident, Number, StrLit, CharLit, LineComment, BlockComment,
    LBrace, RBrace, LParen, RParen, LBracket, RBracket,
    Semi, Comma, Colon, Dot, Hash, Bang, Eq, Lt, Gt, Pipe,
    FatArrow, AndAnd, OrOr, Arrow, DoubleColon, Punct, Newline,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct TokFlags: u8 {
        const IN_TEST = 1 << 0;
        const IN_OS_CFG = 1 << 1;
        const IN_ATTR = 1 << 2;
        const DOC_COMMENT = 1 << 3;
        const SAFETY_COMMENT = 1 << 4;
        const MULTILINE = 1 << 5;
    }
}

impl TokFlags {
    #[inline] pub const fn with(self, flag: Self) -> Self { self.union(flag) }
    #[inline] pub const fn has(self, flag: Self) -> bool { self.contains(flag) }
    #[inline] pub const fn in_test(self) -> bool { self.has(Self::IN_TEST) }
    #[inline] pub const fn in_attr(self) -> bool { self.has(Self::IN_ATTR) }
    #[inline] pub const fn in_os_cfg(self) -> bool { self.has(Self::IN_OS_CFG) }
    #[inline] pub const fn is_doc_comment(self) -> bool { self.has(Self::DOC_COMMENT) }
    #[inline] pub const fn is_safety_comment(self) -> bool { self.has(Self::SAFETY_COMMENT) }
}

#[derive(Debug, Clone, Copy)]
pub struct LexTok<'a> {
    pub kind: LexKind,
    pub slice: &'a [u8],
    pub start: usize,
    pub end: usize,
    pub line: u32,
    pub col: u32,
    pub end_line: u32,
    pub end_col: u32,
    pub flags: TokFlags,
}

impl<'a> LexTok<'a> {
    #[inline] pub fn as_str(&self) -> &'a str { std::str::from_utf8(self.slice).unwrap_or("") }
}

/// 1:1 byte-aligned lexing view where comment bytes and Rust `#[cfg(test)]` items are
/// overwritten with `TEST_BLANK_SENTINEL` (`0xFF`) while preserving every `\n` verbatim.
#[derive(Debug, Clone)]
pub struct SentinelLex { pub bytes: Vec<u8>, pub test_spans: Vec<(usize, usize)> }

impl SentinelLex {
    /// Materialize a clean UTF-8 view of `source[start..end]` with `0xFF` sentinel bytes
    /// removed (borrowing zero-copy whenever no sentinel bytes occur in the slice).
    pub fn clean_slice<'a>(&self, source: &'a str, start: usize, end: usize) -> Cow<'a, str> {
        clean_sentinel_slice(&self.bytes, source, start, end)
    }

    pub fn to_clean_string(&self, source: &str) -> String {
        self.clean_slice(source, 0, self.bytes.len()).into_owned()
    }
}

fn clean_sentinel_slice<'a>(bytes: &[u8], source: &'a str, start: usize, end: usize) -> Cow<'a, str> {
    let window = &bytes[start..end];
    if !window.contains(&TEST_BLANK_SENTINEL) {
        return Cow::Borrowed(&source[start..end]);
    }
    let mut out = String::with_capacity(window.len());
    let mut seg_start: Option<usize> = None;
    for (idx, &byte) in window.iter().enumerate() {
        if byte == TEST_BLANK_SENTINEL && let Some(from) = seg_start.take() {
            out.push_str(&source[start + from..start + idx]);
        } else if byte != TEST_BLANK_SENTINEL && seg_start.is_none() {
            if ident_byte(byte) && out.as_bytes().last().is_some_and(|&prev| ident_byte(prev)) {
                out.push(' ');
            }
            seg_start = Some(idx);
        }
    }
    if let Some(from) = seg_start {
        out.push_str(&source[start + from..end]);
    }
    Cow::Owned(out)
}

#[inline] pub fn ident_byte(byte: u8) -> bool { byte.is_ascii_alphanumeric() || byte == b'_' }
#[inline] pub fn left_bound(bytes: &[u8], pos: usize) -> bool { pos == 0 || !ident_byte(bytes[pos - 1]) }
#[inline] pub fn eq_ws(lhs: &str, rhs: &str) -> bool { lhs.split_whitespace().eq(rhs.split_whitespace()) }

pub fn has_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() { return false; }
    let bytes = haystack.as_bytes();
    haystack.match_indices(needle).any(|(pos, _)| {
        let end = pos + needle.len();
        left_bound(bytes, pos) && (end == bytes.len() || !ident_byte(bytes[end]))
    })
}

/// Return the leading ASCII identifier slice without allocating.
pub fn first_ident(text: &str) -> Option<&str> {
    let trimmed = text.trim_start();
    let end = trimmed.bytes().take_while(|&byte| ident_byte(byte)).count();
    (end > 0 && !trimmed.as_bytes()[0].is_ascii_digit()).then(|| &trimmed[..end])
}

/// Extract the first single- or double-quoted slice without allocating.
pub fn quoted_path(text: &str) -> Option<&str> {
    let idx = text.find(['"', '\''])?;
    let rest = &text[idx + 1..];
    Some(&rest[..rest.find(text.as_bytes()[idx] as char)?])
}

/// Return the exported Go identifier from a trimmed `func` or `type` line.
pub fn go_export(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("func ")
        .and_then(|rem| {
            rem.strip_prefix('(')
                .map_or(Some(rem), |rcv| rcv.split_once(')').map(|(_, tail)| tail))
        })
        .or_else(|| trimmed.strip_prefix("type "))?;
    first_ident(rest).filter(|id| id.as_bytes()[0].is_ascii_uppercase())
}

pub fn match_prefix<'a>(text: &'a str, table: &[&str]) -> Option<&'a str> {
    table.iter().find_map(|&pfx| text.strip_prefix(pfx))
}

#[inline]
pub fn to_eol(bytes: &[u8], pos: usize) -> usize {
    bytes.get(pos..).map_or(pos, |tail| tail.find_byte(b'\n').map_or(bytes.len(), |rel| pos + rel))
}

/// Zero-copy ASCII case-insensitive substring search.
pub fn find_ascii_ci(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() { return None; }
    haystack.windows(needle.len()).position(|win| win.eq_ignore_ascii_case(needle))
}

/// Zero-copy iterator yielding byte `(open, close)` indices of paired backtick spans.
pub fn iter_backtick_ranges(line: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut offset = 0usize;
    std::iter::from_fn(move || {
        let open = offset + line.get(offset..)?.find('`')?;
        let close = open + 1 + line[open + 1..].find('`')?;
        offset = close + 1;
        Some((open, close))
    })
}

/// Zero-copy iterator yielding trimmed non-empty inline backtick spans.
pub fn iter_backtick_spans(line: &str) -> impl Iterator<Item = &str> {
    iter_backtick_ranges(line).filter_map(|(open, close)| {
        let span = line[open + 1..close].trim();
        (!span.is_empty()).then_some(span)
    })
}

/// Zero-copy ASCII identifier slice iterator (`[A-Za-z_][A-Za-z0-9_]*`).
pub fn iter_idents(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        let rel = bytes[pos..].iter().position(|&byte| byte.is_ascii_alphabetic() || byte == b'_')?;
        let start = pos + rel;
        let mut end = start + 1;
        while end < bytes.len() && ident_byte(bytes[end]) {
            end += 1;
        }
        pos = end;
        Some(&bytes[start..pos])
    })
}

/// Extract trimmed comment text from a single source line across Python/Rust/JS/TS/Go.
pub fn extract_comment_body<'a>(ext: &str, line: &'a str) -> Option<&'a str> {
    let trimmed = line.trim();
    if ext == "py" { return trimmed.strip_prefix('#').map(str::trim); }
    match_prefix(trimmed, &["///", "//!", "//", "* "])
        .or_else(|| match_prefix(trimmed, &["/**", "/*"]).map(|rest| rest.trim_end_matches("*/")))
        .map(str::trim)
}

/// Scan `/* ... */` starting at `pos`. Returns `(end, newlines, closed)`.
pub fn block_comment(bytes: &[u8], mut pos: usize, nested: bool) -> (usize, usize, bool) {
    pos += 2;
    let (mut depth, mut newlines) = (1usize, 0usize);
    while pos < bytes.len() && depth > 0 {
        if bytes[pos] == b'\n' { newlines += 1; pos += 1; }
        else if nested && bytes.get(pos..pos + 2) == Some(b"/*") { depth += 1; pos += 2; }
        else if bytes.get(pos..pos + 2) == Some(b"*/") { depth -= 1; pos += 2; }
        else { pos += 1; }
    }
    (pos, newlines, depth == 0)
}

/// End offset if `bytes[pos]` starts a Rust char literal instead of a lifetime.
pub fn char_end(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 2 >= bytes.len() || bytes[pos] != b'\'' { return None; }
    if bytes[pos + 1] == b'\\' {
        let mut idx = (pos + 3).min(bytes.len());
        while idx < bytes.len() && idx < pos + 14 && !matches!(bytes[idx], b'\'' | b'\n') { idx += 1; }
        return (idx < bytes.len() && bytes[idx] == b'\'').then_some(idx + 1);
    }
    let ch = std::str::from_utf8(&bytes[pos + 1..]).ok()?.chars().next()?;
    let next = pos + 1 + ch.len_utf8();
    (next < bytes.len() && bytes[next] == b'\'' && ch != '\n' && ch != '\'').then_some(next + 1)
}

pub enum RawStr { Closed { end: usize }, Unterminated }

/// Scan a Rust raw string (`r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`).
pub fn raw_str(bytes: &[u8], pos: usize) -> Option<RawStr> {
    let start_r = match bytes[pos] {
        b'r' => pos,
        b'b' | b'c' if bytes.get(pos + 1) == Some(&b'r') => pos + 1,
        _ => return None,
    };
    if !left_bound(bytes, pos) { return None; }
    let mut idx = start_r + 1;
    while bytes.get(idx) == Some(&b'#') {
        idx += 1;
    }
    let hashes = idx - (start_r + 1);
    if bytes.get(idx) != Some(&b'"') { return None; }
    idx += 1;
    while idx < bytes.len() {
        if bytes[idx] == b'"'
            && bytes.get(idx + 1..idx + 1 + hashes).is_some_and(|sl| sl.iter().all(|&ch| ch == b'#'))
        {
            return Some(RawStr::Closed { end: idx + 1 + hashes });
        }
        idx += 1;
    }
    Some(RawStr::Unterminated)
}

/// Scan a quoted literal starting at `bytes[pos]`. Returns `(end, newlines, closed)`.
pub fn quoted_lit(bytes: &[u8], mut pos: usize, escapes: bool, multiline: bool) -> (usize, usize, bool) {
    let quote = bytes[pos];
    pos += 1;
    let mut newlines = 0usize;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if escapes && byte == b'\\' {
            if bytes.get(pos + 1) == Some(&b'\n') { newlines += 1; }
            pos = (pos + 2).min(bytes.len());
            continue;
        }
        if byte == b'\n' {
            newlines += 1;
            if !multiline { return (pos, newlines, false); }
        }
        pos += 1;
        if byte == quote { return (pos, newlines, true); }
    }
    (pos, newlines, false)
}

/// Visit every byte outside quoted string literals without allocating.
pub fn scan_code(bytes: &[u8], quotes: &[u8], mut on_byte: impl FnMut(usize, u8) -> ControlFlow<()>) {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let byte = bytes[pos];
        if quotes.contains(&byte) {
            pos = quoted_lit(bytes, pos, true, true).0;
            continue;
        }
        if on_byte(pos, byte).is_break() { break; }
        pos += 1;
    }
}

fn skip_c_comment(bytes: &[u8], pos: usize) -> Option<usize> {
    match bytes.get(pos..pos + 2) {
        Some(b"//") => Some(to_eol(bytes, pos + 2)),
        Some(b"/*") => Some(block_comment(bytes, pos, false).0),
        _ => None,
    }
}

/// Strip `//`, `/* */`, and trailing commas from JSONC; borrows when unchanged.
pub fn strip_jsonc(raw: &str) -> Cow<'_, str> {
    if !raw.contains("//") && !raw.contains("/*") && !raw.contains(',') {
        return Cow::Borrowed(raw);
    }
    let bytes = raw.as_bytes();
    let (mut out, mut pos) = (String::with_capacity(raw.len()), 0usize);

    while pos < bytes.len() {
        if bytes[pos] == b'"' {
            let end = quoted_lit(bytes, pos, true, true).0;
            out.push_str(&raw[pos..end]);
            pos = end;
        } else if let Some(next) = skip_c_comment(bytes, pos) {
            pos = next;
        } else if bytes[pos] == b',' && matches!(next_json(bytes, pos + 1), Some(b'}' | b']')) {
            pos += 1;
        } else {
            let Some(ch) = raw[pos..].chars().next() else { break };
            out.push(ch);
            pos += ch.len_utf8();
        }
    }
    Cow::Owned(out)
}

fn next_json(bytes: &[u8], mut pos: usize) -> Option<u8> {
    while pos < bytes.len() {
        if bytes[pos].is_ascii_whitespace() { pos += 1; }
        else if let Some(next) = skip_c_comment(bytes, pos) { pos = next; }
        else { return Some(bytes[pos]); }
    }
    None
}

// ============================================================================
// Unified Zero-Copy Pushdown LexCursor<'a>
// ============================================================================

#[inline]
fn utf8_char_len(first: u8) -> usize {
    match first { ..0x80 => 1, 0x80..0xE0 => 2, 0xE0..0xF0 => 3, _ => 4 }
}

#[inline]
fn next_unicode_char(bytes: &[u8], pos: usize) -> Option<(char, usize)> {
    let first = *bytes.get(pos)?;
    if first < 0x80 { return Some((first as char, 1)); }
    let len = utf8_char_len(first);
    let ch = std::str::from_utf8(bytes.get(pos..pos + len)?).ok()?.chars().next()?;
    Some((ch, ch.len_utf8()))
}

#[inline]
fn is_unicode_ident_start(bytes: &[u8], pos: usize) -> bool {
    let Some(&byte) = bytes.get(pos) else { return false };
    if byte < 0x80 { return byte == b'_' || byte.is_ascii_alphabetic(); }
    next_unicode_char(bytes, pos).is_some_and(|(ch, _)| ch.is_alphabetic())
}

#[inline]
fn scan_unicode_ident_end(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let byte = bytes[pos];
        if byte < 0x80 {
            if ident_byte(byte) { pos += 1; } else { break; }
        } else if let Some((ch, len)) = next_unicode_char(bytes, pos) && ch.is_alphanumeric() {
            pos += len;
        } else { break; }
    }
    pos
}

#[inline]
fn blank_range(out: &mut [u8], start: usize, end: usize) {
    let stop = end.min(out.len());
    for byte in &mut out[start..stop] {
        if *byte != b'\n' { *byte = TEST_BLANK_SENTINEL; }
    }
}

fn allow_regex(prev_sig: u8, prev_word: &[u8]) -> bool {
    matches!(
        prev_word,
        b"return" | b"case" | b"throw" | b"in" | b"of" | b"typeof"
            | b"void" | b"delete" | b"yield" | b"await" | b"new"
    ) || matches!(
        prev_sig,
        b'(' | b'[' | b'{' | b',' | b';' | b':' | b'=' | b'!' | b'&' | b'|' | b'?' | b'+' | b'-' | b'~'
    )
}

fn scan_regex_literal(bytes: &[u8], start: usize) -> Option<usize> {
    let (mut pos, mut in_class) = (start + 1, false);
    while pos < bytes.len() && bytes[pos] != b'\n' {
        match bytes[pos] {
            b'\\' => pos = (pos + 2).min(bytes.len()),
            b'[' => { in_class = true; pos += 1; }
            b']' => { in_class = false; pos += 1; }
            b'/' if !in_class => {
                pos += 1;
                while pos < bytes.len() && bytes[pos].is_ascii_alphabetic() { pos += 1; }
                return Some(pos);
            }
            _ => pos += 1,
        }
    }
    None
}

fn lookahead_jsx_text(bytes: &[u8], start: usize) -> bool {
    let (mut look, mut saw_alpha) = (start, false);
    while look < bytes.len() && !matches!(bytes[look], b'<' | b'{') {
        let ch = bytes[look];
        if ch.is_ascii_alphabetic() { saw_alpha = true; }
        else if matches!(ch, b';' | b')' | b']' | b'}' | b'?' | b'=' | b'"' | b'`') { return false; }
        look += 1;
    }
    saw_alpha && look < bytes.len()
}

fn find_matching_bracket_bytes(bytes: &[u8], open_idx: usize) -> Option<usize> {
    let (mut depth, mut pos) = (0usize, open_idx);
    while pos < bytes.len() {
        match bytes[pos] {
            b'"' => pos = quoted_lit(bytes, pos, true, true).0,
            b'/' if bytes.get(pos + 1) == Some(&b'/') => pos = to_eol(bytes, pos + 2),
            b'[' => { depth += 1; pos += 1; }
            b']' if depth <= 1 => return Some(pos),
            b']' => { depth -= 1; pos += 1; }
            _ => pos += 1,
        }
    }
    None
}

/// Classify a Rust `#[cfg(...)]` attribute body into `(is_test, is_os)`.
fn classify_rust_cfg_attr(inner: &[u8]) -> (bool, bool) {
    let (mut buf, mut len) = ([0u8; 64], 0usize);
    for &byte in inner {
        if byte.is_ascii_whitespace() { continue; }
        if len >= buf.len() { return (false, false); }
        buf[len] = byte;
        len += 1;
    }
    let compact = &buf[..len];
    let is_test = compact == b"cfg(test)"
        || (compact.starts_with(b"cfg(all(")
            && (compact.starts_with(b"cfg(all(test,")
                || compact.ends_with(b",test))")
                || compact.find(b",test,").is_some()));
    let is_os = matches!(
        compact,
        b"cfg(windows)" | b"cfg(unix)" | b"cfg(any(windows,unix))" | b"cfg(any(unix,windows))"
    );
    (is_test, is_os)
}

fn detect_py_fstring(bytes: &[u8], start: usize) -> Option<(usize, u8, bool)> {
    if !left_bound(bytes, start) { return None; }
    let (mut pos, mut has_f) = (start, false);
    while pos < bytes.len() && pos - start < 2 {
        match bytes[pos] {
            b'f' | b'F' => { has_f = true; pos += 1; }
            b'r' | b'R' => pos += 1,
            _ => break,
        }
    }
    if !has_f || pos >= bytes.len() || !matches!(bytes[pos], b'"' | b'\'') { return None; }
    let quote = bytes[pos];
    let triple = bytes.get(pos..pos + 3) == Some(&[quote, quote, quote]);
    Some((pos + if triple { 3 } else { 1 }, quote, triple))
}

fn skip_py_triple(bytes: &[u8], mut pos: usize, quote: u8) -> usize {
    while pos < bytes.len() {
        if bytes[pos] == b'\\' && pos + 1 < bytes.len() { pos += 2; continue; }
        if bytes.get(pos..pos + 3) == Some(&[quote, quote, quote]) { return pos + 3; }
        pos += 1;
    }
    pos
}

#[derive(Clone, Copy, Default)]
struct AttrScope { armed: bool, active: bool, brace_depth: usize, paren_depth: usize, bracket_depth: usize }

impl AttrScope {
    #[inline] fn is_in_scope(&self) -> bool { self.armed || self.active }

    fn arm(&mut self) {
        if !self.active { *self = Self { armed: true, ..Self::default() }; }
    }

    fn step_before(&mut self, kind: LexKind) {
        if self.armed && !self.active && self.paren_depth == 0
            && self.bracket_depth == 0 && kind == LexKind::RBrace
        { self.armed = false; }
    }

    fn step_after(&mut self, kind: LexKind) {
        step_attr_after(self, kind);
    }
}

fn step_attr_after(scope: &mut AttrScope, kind: LexKind) {
    if scope.active {
        match kind {
            LexKind::LBrace => scope.brace_depth += 1,
            LexKind::RBrace => {
                scope.brace_depth = scope.brace_depth.saturating_sub(1);
                if scope.brace_depth == 0 {
                    scope.active = false;
                    scope.armed = false;
                }
            }
            _ => {}
        }
        return;
    }
    if !scope.armed { return; }
    let top = scope.paren_depth == 0 && scope.bracket_depth == 0;
    match kind {
        LexKind::LParen => scope.paren_depth += 1,
        LexKind::RParen => scope.paren_depth = scope.paren_depth.saturating_sub(1),
        LexKind::LBracket => scope.bracket_depth += 1,
        LexKind::RBracket => scope.bracket_depth = scope.bracket_depth.saturating_sub(1),
        LexKind::LBrace if top => { scope.active = true; scope.brace_depth = 1; }
        LexKind::Semi | LexKind::Comma if top => scope.armed = false,
        _ => {}
    }
}

fn classify_punct(bytes: &[u8], pos: usize) -> (LexKind, usize) {
    let (byte, next, third) = (bytes[pos], bytes.get(pos + 1).copied(), bytes.get(pos + 2).copied());
    match (byte, next) {
        (b'=', Some(b'>')) => (LexKind::FatArrow, 2), (b'&', Some(b'&')) => (LexKind::AndAnd, 2),
        (b'|', Some(b'|')) => (LexKind::OrOr, 2), (b'-', Some(b'>')) => (LexKind::Arrow, 2),
        (b':', Some(b':')) => (LexKind::DoubleColon, 2),
        (b'=' | b'!' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/', Some(b'=')) => (LexKind::Punct, 2),
        (b'.', Some(b'.')) if third == Some(b'=') => (LexKind::Punct, 3),
        (b'.', Some(b'.')) => (LexKind::Punct, 2),
        (b'{', _) => (LexKind::LBrace, 1), (b'}', _) => (LexKind::RBrace, 1),
        (b'(', _) => (LexKind::LParen, 1), (b')', _) => (LexKind::RParen, 1),
        (b'[', _) => (LexKind::LBracket, 1), (b']', _) => (LexKind::RBracket, 1),
        (b';', _) => (LexKind::Semi, 1), (b',', _) => (LexKind::Comma, 1),
        (b':', _) => (LexKind::Colon, 1), (b'.', _) => (LexKind::Dot, 1),
        (b'#', _) => (LexKind::Hash, 1), (b'!', _) => (LexKind::Bang, 1),
        (b'=', _) => (LexKind::Eq, 1), (b'<', _) => (LexKind::Lt, 1),
        (b'>', _) => (LexKind::Gt, 1), (b'|', _) => (LexKind::Pipe, 1),
        _ => (LexKind::Punct, utf8_char_len(byte).min(bytes.len() - pos)),
    }
}

/// Unified zero-copy pushdown byte lexer with stack-only `SmallVec<[LexMode; 8]>`.
pub struct LexCursor<'a> {
    bytes: &'a [u8], pos: usize, line: u32, col: u32, spec: LangSpec,
    modes: SmallVec<[LexMode; 8]>, brace_depth: usize, prev_sig: u8, prev_word: &'a [u8],
    cfg_test: AttrScope, os_cfg: AttrScope, attr_end: Option<usize>,
}

impl<'a> LexCursor<'a> {
    pub fn new(lang: LangKind, src: &'a [u8]) -> Self {
        Self {
            bytes: src, pos: 0, line: 1, col: 0, spec: LangSpec::for_lang(lang),
            modes: smallvec![LexMode::Standard], brace_depth: 0, prev_sig: b';', prev_word: b"",
            cfg_test: AttrScope::default(), os_cfg: AttrScope::default(), attr_end: None,
        }
    }

    #[inline]
    pub fn has_unclosed_template(&self) -> bool {
        self.modes.iter().any(|mode| matches!(mode, LexMode::Template { .. }))
    }

    #[inline]
    fn base_flags(&self) -> TokFlags {
        let mut flags = TokFlags::empty();
        flags.set(TokFlags::IN_TEST, self.cfg_test.is_in_scope());
        flags.set(TokFlags::IN_OS_CFG, self.os_cfg.is_in_scope());
        flags.set(TokFlags::IN_ATTR, self.attr_end.is_some());
        flags
    }

    fn make_tok(
        &mut self, kind: LexKind, start: usize, end: usize, start_line: u32, mut flags: TokFlags,
    ) -> LexTok<'a> {
        let start_col = self.col;
        advance_cursor_pos(self, end);
        if self.line > start_line { flags = flags.with(TokFlags::MULTILINE); }
        LexTok {
            kind, slice: &self.bytes[start..end], start, end,
            line: start_line, col: start_col, end_line: self.line, end_col: self.col, flags,
        }
    }

    fn emit_lit(
        &mut self, kind: LexKind, start: usize, end: usize, start_line: u32, sig: u8, extra_flag: TokFlags,
    ) -> LexTok<'a> {
        self.prev_sig = sig;
        self.prev_word = b"";
        let flags = self.base_flags().with(extra_flag);
        self.make_tok(kind, start, end, start_line, flags)
    }

    fn emit_plain(
        &mut self, kind: LexKind, start: usize, end: usize, start_line: u32, sig: u8,
    ) -> LexTok<'a> {
        self.emit_lit(kind, start, end, start_line, sig, TokFlags::empty())
    }

    pub fn next_token(&mut self) -> Option<LexTok<'a>> { step_token(self, false).ok().flatten() }
    pub fn next_checked(&mut self) -> Result<Option<LexTok<'a>>, String> { step_token(self, true) }
}

/// Advance cursor byte offset, 1-based line, and column through `to`.
fn advance_cursor_pos(cur: &mut LexCursor<'_>, to: usize) {
    let mut idx = cur.pos;
    while idx < to {
        let byte = cur.bytes[idx];
        let step = match byte {
            b'\n' => { cur.line += 1; cur.col = 0; 1 }
            b'\t' => { cur.col += 4 - (cur.col % 4); 1 }
            b'\r' => 1,
            0..0x80 => { cur.col += 1; 1 }
            _ => { cur.col += 1; utf8_char_len(byte) }
        };
        idx = (idx + step).min(to);
    }
    cur.pos = to;
}

fn skip_ws_same_line(cur: &mut LexCursor<'_>) {
    while cur.pos < cur.bytes.len() {
        let byte = cur.bytes[cur.pos];
        if byte == b' ' { cur.col += 1; }
        else if byte == b'\t' { cur.col += 4 - (cur.col % 4); }
        else if byte != b'\r' { break; }
        cur.pos += 1;
    }
}

/// Scan a JS/TS template literal segment until `${` or closing backtick.
fn scan_template_segment<'a>(
    cur: &mut LexCursor<'a>, start: usize, scan_from: usize, start_line: u32, checked: bool,
) -> Result<LexTok<'a>, String> {
    let mut idx = scan_from;
    while idx < cur.bytes.len() {
        let ch = cur.bytes[idx];
        if ch == b'\\' {
            idx = (idx + 2).min(cur.bytes.len());
            continue;
        }
        if cur.bytes.get(idx..idx + 2) == Some(b"${") {
            cur.modes.push(LexMode::Template { brace_depth: cur.brace_depth });
            return Ok(cur.emit_plain(LexKind::StrLit, start, idx + 2, start_line, b'{'));
        }
        if ch == b'`' {
            return Ok(cur.emit_plain(LexKind::StrLit, start, idx + 1, start_line, b'`'));
        }
        idx += 1;
    }
    if checked {
        return Err(format!("unterminated template literal at L{start_line}"));
    }
    Ok(cur.emit_plain(LexKind::StrLit, start, idx, start_line, b'`'))
}

/// Scan JSX text runs until the next `<`, `{`, or newline.
fn step_jsx_text<'a>(cur: &mut LexCursor<'a>) -> LexTok<'a> {
    let (start, start_line, byte) = (cur.pos, cur.line, cur.bytes[cur.pos]);
    if byte == b'<' {
        cur.modes.pop();
        return cur.emit_plain(LexKind::Lt, start, start + 1, start_line, b'<');
    }
    if byte == b'{' {
        let depth = cur.brace_depth;
        cur.brace_depth += 1;
        cur.modes.push(LexMode::JsxExpr { brace_depth: depth });
        return cur.emit_plain(LexKind::LBrace, start, start + 1, start_line, b'{');
    }
    let flags = cur.base_flags();
    if byte == b'\n' { return cur.make_tok(LexKind::Newline, start, start + 1, start_line, flags); }
    let mut end = start + 1;
    while end < cur.bytes.len() && !matches!(cur.bytes[end], b'<' | b'{' | b'\n') { end += 1; }
    cur.make_tok(LexKind::StrLit, start, end, start_line, flags)
}

fn scan_py_fstring_end(
    bytes: &[u8], mut idx: usize, quote: u8, triple: bool,
) -> (usize, bool, bool) {
    let close3 = [quote, quote, quote];
    while idx < bytes.len() {
        let byte = bytes[idx];
        match byte {
            b'\\' => idx = (idx + 2).min(bytes.len()),
            b'{' if bytes.get(idx + 1) == Some(&b'{') => idx += 2,
            b'}' if bytes.get(idx + 1) == Some(&b'}') => idx += 2,
            b'{' => return (idx + 1, true, false),
            _ if triple && bytes.get(idx..idx + 3) == Some(&close3) => return (idx + 3, false, true),
            _ if !triple && byte == quote => return (idx + 1, false, true),
            _ => idx += 1,
        }
    }
    (idx, false, false)
}

fn step_py_fstring_text<'a>(cur: &mut LexCursor<'a>, quote: u8, triple: bool) -> LexTok<'a> {
    let (start, start_line) = (cur.pos, cur.line);
    let (end, opened_expr, closed) = scan_py_fstring_end(cur.bytes, start, quote, triple);
    if opened_expr && let Some(slot) = cur.modes.last_mut() {
        *slot = LexMode::PyFString { quote, triple, brace_depth: 1 };
    } else if !opened_expr && closed {
        cur.modes.pop();
    }
    let flags = cur.base_flags();
    cur.make_tok(LexKind::StrLit, start, end, start_line, flags)
}

/// Arm Rust `#[cfg(test)]` and OS `#[cfg(...)]` scope trackers on outer attributes.
fn peek_rust_attr(cur: &mut LexCursor<'_>) {
    if !cur.spec.rust_raw_and_cfg || cur.bytes[cur.pos] != b'#' || cur.attr_end.is_some() {
        return;
    }
    let bracket = if cur.bytes.get(cur.pos + 1) == Some(&b'[') {
        Some((cur.pos + 1, false))
    } else if cur.bytes.get(cur.pos + 1..cur.pos + 3) == Some(b"![") {
        Some((cur.pos + 2, true))
    } else {
        None
    };
    let Some((open, is_inner)) = bracket else { return };
    let Some(close) = find_matching_bracket_bytes(cur.bytes, open) else { return };
    if !is_inner {
        let (is_test, is_os) = classify_rust_cfg_attr(&cur.bytes[open + 1..close]);
        if is_test { cur.cfg_test.arm(); }
        else if is_os { cur.os_cfg.arm(); }
    }
    cur.attr_end = Some(close);
}

fn emit_comment<'a>(
    cur: &mut LexCursor<'a>, kind: LexKind, start: usize, end: usize, start_line: u32, is_doc: bool,
) -> LexTok<'a> {
    let mut flags = cur.base_flags();
    if is_doc {
        flags = flags.with(TokFlags::DOC_COMMENT);
    }
    if find_ascii_ci(&cur.bytes[start..end], b"safety:").is_some() {
        flags = flags.with(TokFlags::SAFETY_COMMENT);
    }
    cur.make_tok(kind, start, end, start_line, flags)
}

/// Lex a line or block comment when the active language syntax permits it.
fn try_step_comment<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool,
) -> Result<Option<LexTok<'a>>, String> {
    let two = cur.bytes.get(start..start + 2);
    if cur.spec.slash_comments && two == Some(b"//") {
        let end = to_eol(cur.bytes, start + 2);
        let sl = &cur.bytes[start..end];
        let doc = (sl.starts_with(b"///") && !sl.starts_with(b"////"))
            || sl.starts_with(b"//!");
        return Ok(Some(emit_comment(cur, LexKind::LineComment, start, end, start_line, doc)));
    }

    if cur.spec.slash_comments && two == Some(b"/*") {
        let (end, _, closed) = block_comment(cur.bytes, start, cur.spec.nested_block_comments);
        if checked && !closed {
            return Err(format!("unterminated block comment at L{start_line}"));
        }
        let sl = &cur.bytes[start..end];
        let doc = cur.spec.rust_raw_and_cfg
            && matches!(sl.get(2), Some(b'*' | b'!'))
            && sl.get(3) != Some(&b'/');
        return Ok(Some(emit_comment(cur, LexKind::BlockComment, start, end, start_line, doc)));
    }

    if cur.spec.hash_line_comments && cur.bytes[start] == b'#' {
        let (end, flags) = (to_eol(cur.bytes, start + 1), cur.base_flags());
        return Ok(Some(cur.make_tok(LexKind::LineComment, start, end, start_line, flags)));
    }
    Ok(None)
}

fn step_rust_raw_str<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool, res: RawStr,
) -> Result<LexTok<'a>, String> {
    let end = match res {
        RawStr::Closed { end } => end,
        RawStr::Unterminated if checked => {
            return Err(format!("unterminated Rust raw string at L{start_line}"));
        }
        RawStr::Unterminated => cur.bytes.len(),
    };
    Ok(cur.emit_plain(LexKind::StrLit, start, end, start_line, b'"'))
}

/// Lex Rust raw strings, byte/C string literals, and char literals.
fn try_step_rust_lit<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool,
) -> Result<Option<LexTok<'a>>, String> {
    if let Some(res) = raw_str(cur.bytes, start) {
        return step_rust_raw_str(cur, start, start_line, checked, res).map(Some);
    }

    let byte = cur.bytes[start];
    if matches!(byte, b'b' | b'c')
        && cur.bytes.get(start + 1) == Some(&b'"')
        && left_bound(cur.bytes, start)
    {
        let (end, _, closed) = quoted_lit(cur.bytes, start + 1, true, true);
        if checked && !closed {
            return Err(format!("unterminated string literal at L{start_line}"));
        }
        return Ok(Some(cur.emit_plain(LexKind::StrLit, start, end, start_line, b'"')));
    }

    if byte == b'b'
        && left_bound(cur.bytes, start)
        && let Some(end) = char_end(cur.bytes, start + 1)
    {
        return Ok(Some(cur.emit_plain(LexKind::CharLit, start, end, start_line, b'\'')));
    }

    if byte == b'\'' {
        let (kind, end) = char_end(cur.bytes, start)
            .map_or((LexKind::Punct, start + 1), |end| (LexKind::CharLit, end));
        return Ok(Some(cur.emit_plain(kind, start, end, start_line, b'\'')));
    }
    Ok(None)
}

fn finish_quoted_lit<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, quote: u8, escape: bool,
    multiline: bool, checked: bool,
) -> Result<LexTok<'a>, String> {
    let (end, _, closed) = quoted_lit(cur.bytes, start, escape, multiline);
    if checked && !closed { return Err(format!("unterminated string literal at L{start_line}")); }
    Ok(cur.emit_plain(LexKind::StrLit, start, end, start_line, quote))
}

fn try_step_py_lit<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32,
) -> Option<LexTok<'a>> {
    if let Some((after_open, quote, triple)) = detect_py_fstring(cur.bytes, start) {
        cur.modes.push(LexMode::PyFString { quote, triple, brace_depth: 0 });
        return Some(cur.emit_plain(LexKind::StrLit, start, after_open, start_line, quote));
    }
    let byte = cur.bytes[start];
    if matches!(byte, b'"' | b'\'') && cur.bytes.get(start..start + 3) == Some(&[byte, byte, byte]) {
        let end = skip_py_triple(cur.bytes, start + 3, byte);
        return Some(cur.emit_lit(LexKind::StrLit, start, end, start_line, byte, TokFlags::DOC_COMMENT));
    }
    None
}

fn try_step_js_lit<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool,
) -> Result<Option<LexTok<'a>>, String> {
    let byte = cur.bytes[start];
    if byte == b'/'
        && allow_regex(cur.prev_sig, cur.prev_word)
        && let Some(end) = scan_regex_literal(cur.bytes, start)
    {
        return Ok(Some(cur.emit_plain(LexKind::StrLit, start, end, start_line, b'0')));
    }
    if byte == b'`' {
        return scan_template_segment(cur, start, start + 1, start_line, checked).map(Some);
    }
    Ok(None)
}

fn try_step_script_lit<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool,
) -> Result<Option<LexTok<'a>>, String> {
    if cur.spec.hash_line_comments && let Some(tok) = try_step_py_lit(cur, start, start_line) {
        return Ok(Some(tok));
    }
    if cur.spec.js_templates_regex && let Some(tok) = try_step_js_lit(cur, start, start_line, checked)? {
        return Ok(Some(tok));
    }
    if cur.spec.backtick_raw_str && cur.bytes[start] == b'`' {
        return finish_quoted_lit(cur, start, start_line, b'`', false, true, checked).map(Some);
    }
    Ok(None)
}

/// Lex any language-specific or quoted string/char literal at `start`.
fn try_step_literal<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32, checked: bool,
) -> Result<Option<LexTok<'a>>, String> {
    if cur.spec.rust_raw_and_cfg && let Some(tok) = try_step_rust_lit(cur, start, start_line, checked)? {
        return Ok(Some(tok));
    }
    if let Some(tok) = try_step_script_lit(cur, start, start_line, checked)? {
        return Ok(Some(tok));
    }
    let byte = cur.bytes[start];
    let is_quote = matches!(byte, b'"') || (!cur.spec.rust_raw_and_cfg && byte == b'\'');
    if !is_quote {
        return Ok(None);
    }
    let multiline = cur.spec.rust_raw_and_cfg || (!checked && !cur.spec.hash_line_comments);
    let tok = finish_quoted_lit(cur, start, start_line, byte, true, multiline, checked)?;
    Ok(Some(tok))
}

fn try_step_ident_or_num<'a>(
    cur: &mut LexCursor<'a>, start: usize, start_line: u32,
) -> Option<LexTok<'a>> {
    let raw_ident = cur.spec.rust_raw_and_cfg
        && cur.bytes.get(start..start + 2) == Some(b"r#")
        && is_unicode_ident_start(cur.bytes, start + 2);
    let ident_from = if raw_ident {
        Some(start + 2)
    } else if is_unicode_ident_start(cur.bytes, start) {
        Some(start)
    } else {
        None
    };
    if let Some(from) = ident_from {
        let end = scan_unicode_ident_end(cur.bytes, from);
        cur.prev_word = &cur.bytes[start..end];
        cur.prev_sig = cur.bytes[end - 1];
        let flags = cur.base_flags();
        return Some(cur.make_tok(LexKind::Ident, start, end, start_line, flags));
    }
    if !cur.bytes[start].is_ascii_digit() {
        return None;
    }
    let mut end = start + 1;
    while end < cur.bytes.len() && ident_byte(cur.bytes[end]) { end += 1; }
    Some(cur.emit_plain(LexKind::Number, start, end, start_line, cur.bytes[end - 1]))
}

fn update_brace_mode(cur: &mut LexCursor<'_>, top: LexMode, kind: LexKind) {
    if kind == LexKind::LBrace {
        cur.brace_depth += 1;
        if let Some(LexMode::PyFString { brace_depth, .. }) = cur.modes.last_mut() {
            *brace_depth += 1;
        }
        return;
    }
    if kind != LexKind::RBrace {
        return;
    }
    cur.brace_depth = cur.brace_depth.saturating_sub(1);
    if let LexMode::JsxExpr { brace_depth: target } = top && cur.brace_depth == target {
        cur.modes.pop();
    } else if let Some(LexMode::PyFString { brace_depth, .. }) = cur.modes.last_mut() {
        *brace_depth = brace_depth.saturating_sub(1);
    }
}

/// Lex an identifier, numeric literal, or punctuation token and update mode stacks.
fn step_code_or_punct<'a>(
    cur: &mut LexCursor<'a>, top: LexMode, start: usize, start_line: u32,
) -> LexTok<'a> {
    if let Some(tok) = try_step_ident_or_num(cur, start, start_line) {
        return tok;
    }
    let (kind, step) = classify_punct(cur.bytes, start);
    let in_attr = cur.attr_end.is_some();
    if !in_attr && cur.spec.rust_raw_and_cfg {
        cur.cfg_test.step_before(kind);
        cur.os_cfg.step_before(kind);
    }
    let flags = cur.base_flags();
    if cur.spec.jsx_text && kind == LexKind::Gt && cur.prev_sig != b'='
        && lookahead_jsx_text(cur.bytes, start + 1)
    {
        cur.modes.push(LexMode::JsxText);
    }
    update_brace_mode(cur, top, kind);
    if !in_attr && cur.spec.rust_raw_and_cfg {
        cur.cfg_test.step_after(kind);
        cur.os_cfg.step_after(kind);
    }
    cur.prev_word = b"";
    cur.prev_sig = cur.bytes[start + step - 1];
    cur.make_tok(kind, start, start + step, start_line, flags)
}

/// Advance `cur` by one token across all pushdown modes (JSX, f-strings, templates).
fn step_token<'a>(cur: &mut LexCursor<'a>, checked: bool) -> Result<Option<LexTok<'a>>, String> {
    if cur.pos >= cur.bytes.len() { return Ok(None); }
    let top = *cur.modes.last().unwrap_or(&LexMode::Standard);
    if matches!(top, LexMode::JsxText) { return Ok(Some(step_jsx_text(cur))); }
    if let LexMode::PyFString { quote, triple, brace_depth: 0 } = top {
        return Ok(Some(step_py_fstring_text(cur, quote, triple)));
    }
    skip_ws_same_line(cur);
    if cur.pos >= cur.bytes.len() { return Ok(None); }
    if cur.attr_end.is_some_and(|end| cur.pos > end) { cur.attr_end = None; }

    let (start, start_line, byte) = (cur.pos, cur.line, cur.bytes[cur.pos]);
    if byte == b'\n' {
        let flags = cur.base_flags();
        return Ok(Some(cur.make_tok(LexKind::Newline, start, start + 1, start_line, flags)));
    }
    if let LexMode::Template { brace_depth: tpl_depth } = top
        && cur.brace_depth == tpl_depth && byte == b'}'
    {
        cur.modes.pop();
        return scan_template_segment(cur, start, start + 1, start_line, checked).map(Some);
    }
    if let Some(tok) = try_step_comment(cur, start, start_line, checked)? { return Ok(Some(tok)); }
    peek_rust_attr(cur);
    if let Some(tok) = try_step_literal(cur, start, start_line, checked)? { return Ok(Some(tok)); }
    Ok(Some(step_code_or_punct(cur, top, start, start_line)))
}

/// Run the unified zero-copy `LexCursor<'a>` in sentinel-blanking mode.
/// Preserves exact 1:1 byte length and `\n` offsets while capturing `#[cfg(test)]` spans.
pub fn blank_noise_sentinel(ext: &str, content: &str) -> SentinelLex {
    let (lang, bytes) = (LangKind::from_ext(ext), content.as_bytes());
    let (mut out, mut test_spans, mut active_test) = (bytes.to_vec(), Vec::new(), None::<(usize, usize)>);
    let mut cursor = LexCursor::new(lang, bytes);
    let flush = |active: &mut Option<(usize, usize)>, spans: &mut Vec<(usize, usize)>| {
        if let Some((start, end)) = active.take() && start < end { spans.push((start, end)); }
    };

    while let Some(tok) = cursor.next_token() {
        let is_cmt = matches!(tok.kind, LexKind::LineComment | LexKind::BlockComment)
            || (lang == LangKind::Python && tok.kind == LexKind::StrLit && tok.flags.is_doc_comment());
        if is_cmt || (tok.flags.in_test() && tok.flags.in_attr()) {
            flush(&mut active_test, &mut test_spans);
            blank_range(&mut out, tok.start, tok.end);
        } else if tok.flags.in_test() && tok.kind != LexKind::Newline {
            let prev_end = match &mut active_test {
                Some((_, end)) => { let old = *end; *end = tok.end; old }
                None => { active_test = Some((tok.start, tok.end)); tok.start }
            };
            blank_range(&mut out, prev_end, tok.end);
        } else if !tok.flags.in_test() && tok.kind != LexKind::Newline {
            flush(&mut active_test, &mut test_spans);
        }
    }
    flush(&mut active_test, &mut test_spans);
    SentinelLex { bytes: out, test_spans }
}

/// Unified pushdown stateful brace & literal balance checker across Rust, TS, JS, JSX, and Go.
pub fn check_balanced_syntax(rel_path: &str, source: &str) -> Result<(), String> {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("");
    let lang = LangKind::from_ext(ext);
    if !matches!(lang, LangKind::Rust | LangKind::Go | LangKind::Js | LangKind::Jsx) { return Ok(()); }
    let (mut cursor, mut delims) = (LexCursor::new(lang, source.as_bytes()), Vec::new());
    while let Some(tok) = cursor.next_checked()? {
        match tok.kind {
            LexKind::LBrace => delims.push((b'{', tok.line)),
            LexKind::LParen => delims.push((b'(', tok.line)),
            LexKind::LBracket => delims.push((b'[', tok.line)),
            LexKind::RBrace => pop_delim(b'}', tok.line, &mut delims)?,
            LexKind::RParen => pop_delim(b')', tok.line, &mut delims)?,
            LexKind::RBracket => pop_delim(b']', tok.line, &mut delims)?,
            _ => {}
        }
    }
    if cursor.has_unclosed_template() {
        return Err("unterminated JS/TS template literal interpolation".into());
    }
    if let Some((open, open_line)) = delims.pop() {
        return Err(format!("unclosed `{}` opened at L{open_line}", open as char));
    }
    Ok(())
}

fn pop_delim(close: u8, line: u32, stack: &mut Vec<(u8, u32)>) -> Result<(), String> {
    let expected = match close { b'}' => b'{', b')' => b'(', _ => b'[' };
    match stack.pop() {
        Some((open, _)) if open == expected => Ok(()),
        Some((open, open_line)) => Err(format!(
            "mismatched `{}` at L{line} (expected close for `{}` from L{open_line})",
            close as char, open as char
        )),
        None => Err(format!("unexpected closing `{}` at L{line}", close as char)),
    }
}

#[cfg(test)]
use crate::testutil::TestRng;

#[cfg(test)]
fn assert_sentinel_ok(ext: &str, src: &str) -> SentinelLex {
    let lexed = blank_noise_sentinel(ext, src);
    let nl = |raw: &[u8]| raw.iter().filter(|&&byte| byte == b'\n').count();
    assert_eq!(lexed.bytes.len(), src.len(), "byte len drift on {ext}");
    assert_eq!(nl(&lexed.bytes), nl(src.as_bytes()), "newline drift on {ext}");
    lexed
}

#[cfg(test)]
#[test]
fn sentinel_offsets_and_cfg_test() {
    let src = concat!(
        "pub fn live_fn() -> usize { 1 }\n",
        "#[cfg(test)]\n",
        "fn helper(buf: [u8; 16]) -> usize {\n",
        "    let arr: [u8; 16] = [0; 16];\n",
        "    live_fn() + buf.len() + arr.len()\n",
        "}\n",
        "pub fn after_test() -> usize { 2 }\n",
    );
    let lexed = assert_sentinel_ok("rs", src);
    let cleaned = lexed.to_clean_string(src);
    assert!(cleaned.contains("pub fn live_fn") && cleaned.contains("pub fn after_test"));
    assert!(!cleaned.contains("arr"), "cfg(test) body leaked past [u8; 16]: {cleaned}");
    assert_eq!(lexed.test_spans.len(), 1);
    let (start, end) = lexed.test_spans[0];
    assert!(src[start..end].contains("live_fn") && src[start..end].contains("arr"));
}

#[cfg(test)]
#[test]
fn sentinel_fuzz_offset_invariants() {
    let mut rng = TestRng::new(0x1E45_F022);
    let pieces = [
        "pub fn f() { ", "let s = \"hi \\\" // x\"; ", "r#\"raw\"# ", "// cmt\n",
        "/* blk\n* cmt */ ", "# py cmt\n", "f\"v={x}\" ", "`tpl ${y}` ", "'a' ",
        "#[cfg(test)] mod t { fn g() {} }\n", "日本語🚀 ", "}\n",
    ];
    for ext in ["rs", "py", "ts", "tsx", "go"] {
        for _ in 0..60 {
            let doc: String = (0..10).map(|_| pieces[rng.next_usize(pieces.len())]).collect();
            let _ = assert_sentinel_ok(ext, &doc);
        }
    }

    let py = "def greet(user_name):\n    # comment ignore_me\n    return f\"hi {user_name}\"\n";
    let py_clean = blank_noise_sentinel("py", py).to_clean_string(py);
    assert!(py_clean.contains("user_name") && !py_clean.contains("ignore_me"));
    let tsx = "export const V = () => <div>Don't stop {runAction()}</div>;\n";
    assert_eq!(check_balanced_syntax("src/V.tsx", tsx), Ok(()));
    assert!(blank_noise_sentinel("tsx", tsx).to_clean_string(tsx).contains("runAction"));
}

#[cfg(test)]
#[test]
fn lex_cursor_flags() {
    let rs = concat!(
        "/// Doc comment\n// SAFETY: verified\n#[cfg(unix)]\n",
        "pub fn os_only<'a>(x: &'a str) -> char { 'z' }\n",
        "#[cfg(test)]\nmod tests { fn inner() {} }\n",
    );
    let mut cursor = LexCursor::new(LangKind::Rust, rs.as_bytes());
    let (mut saw_doc, mut saw_safety, mut saw_os) = (false, false, false);
    let (mut saw_test, mut saw_char) = (false, false);

    while let Some(tok) = cursor.next_token() {
        saw_doc |= tok.kind == LexKind::LineComment && tok.flags.is_doc_comment();
        saw_safety |= tok.kind == LexKind::LineComment
            && tok.flags.has(TokFlags::SAFETY_COMMENT);
        saw_os |= tok.kind == LexKind::Ident
            && tok.as_str() == "os_only"
            && tok.flags.has(TokFlags::IN_OS_CFG);
        saw_test |= tok.kind == LexKind::Ident
            && tok.as_str() == "inner"
            && tok.flags.in_test();
        saw_char |= tok.kind == LexKind::CharLit && tok.as_str() == "'z'";
    }
    assert!(saw_doc && saw_safety && saw_os && saw_test && saw_char);
}

#[cfg(test)]
#[test]
fn lex_helpers_and_cfg_any() {
    let bt: Vec<_> = iter_backtick_spans("see `src/lex.rs` and `foo()`").collect();
    assert_eq!(bt, vec!["src/lex.rs", "foo()"]);
    assert_eq!(find_ascii_ci(b"Hello ReFiT:Ignore", b"refit:ignore"), Some(6));

    let ids: Vec<_> = iter_idents(b"let foo_1 = bar2 + 99;").collect();
    assert_eq!(ids, vec![b"let".as_slice(), b"foo_1", b"bar2"]);
    for (ext, line, want) in [
        ("rs", "  /// hello world", "hello world"),
        ("py", "  # py note", "py note"),
    ] {
        assert_eq!(extract_comment_body(ext, line), Some(want));
    }

    let rs_any = concat!(
        "#[cfg(any(feature = \"ste100\", test))]\npub fn prod_feat() {}\n",
        "#[cfg(all(unix, test))]\nfn unix_test() {}\n",
    );
    let (mut prod_in_test, mut all_in_test) = (false, false);
    let mut cur = LexCursor::new(LangKind::Rust, rs_any.as_bytes());
    while let Some(tok) = cur.next_token() {
        if tok.kind == LexKind::Ident {
            if tok.as_str() == "prod_feat" {
                prod_in_test = tok.flags.in_test();
            }
            if tok.as_str() == "unix_test" {
                all_in_test = tok.flags.in_test();
            }
        }
    }
    assert!(!prod_in_test && all_in_test);
}
