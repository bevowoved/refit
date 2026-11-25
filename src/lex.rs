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
        if ext.eq_ignore_ascii_case("rs") { Self::Rust }
        else if ext.eq_ignore_ascii_case("go") { Self::Go }
        else if ext.eq_ignore_ascii_case("py") { Self::Python }
        else if ext.eq_ignore_ascii_case("tsx") || ext.eq_ignore_ascii_case("jsx") { Self::Jsx }
        else if matches!(ext.to_ascii_lowercase().as_str(), "ts" | "js" | "mjs" | "cjs" | "mts" | "cts") {
            Self::Js
        } else { Self::Other }
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokFlags(u8);

impl TokFlags {
    pub const IN_TEST: u8 = 1 << 0;
    pub const IN_OS_CFG: u8 = 1 << 1;
    pub const IN_ATTR: u8 = 1 << 2;
    pub const DOC_COMMENT: u8 = 1 << 3;
    pub const SAFETY_COMMENT: u8 = 1 << 4;
    pub const MULTILINE: u8 = 1 << 5;

    #[inline] pub const fn with(self, bit: u8) -> Self { Self(self.0 | bit) }
    #[inline] pub const fn has(self, bit: u8) -> bool { (self.0 & bit) != 0 }
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
pub fn to_eol(bytes: &[u8], i: usize) -> usize {
    let Some(tail) = bytes.get(i..) else { return i };
    match memchr::memchr(b'\n', tail) {
        Some(rel) => i + rel,
        None => bytes.len(),
    }
}

/// Zero-copy ASCII case-insensitive substring search.
pub fn find_ascii_ci(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|win| win.eq_ignore_ascii_case(needle))
}

/// Zero-copy iterator yielding byte `(open, close)` indices of paired backtick spans.
pub fn iter_backtick_ranges(line: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut offset = 0usize;
    std::iter::from_fn(move || {
        let rest = line.get(offset..)?;
        let open = rest.find('`')?;
        let abs_open = offset + open;
        let after = &line[abs_open + 1..];
        let close = after.find('`')?;
        let abs_close = abs_open + 1 + close;
        offset = abs_close + 1;
        Some((abs_open, abs_close))
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
        while pos < bytes.len() {
            let b = bytes[pos];
            if b.is_ascii_alphabetic() || b == b'_' {
                let start = pos;
                pos += 1;
                while pos < bytes.len() && ident_byte(bytes[pos]) {
                    pos += 1;
                }
                return Some(&bytes[start..pos]);
            }
            pos += 1;
        }
        None
    })
}

/// Extract trimmed comment text from a single source line across Python/Rust/JS/TS/Go.
pub fn extract_comment_body<'a>(ext: &str, line: &'a str) -> Option<&'a str> {
    let t = line.trim();
    if ext == "py" {
        return t.strip_prefix('#').map(str::trim);
    }
    t.strip_prefix("///")
        .or_else(|| t.strip_prefix("//!"))
        .or_else(|| t.strip_prefix("//"))
        .or_else(|| t.strip_prefix("/**").map(|r| r.trim_end_matches("*/")))
        .or_else(|| t.strip_prefix("/*").map(|r| r.trim_end_matches("*/")))
        .or_else(|| t.strip_prefix("* "))
        .map(str::trim)
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
    (next < bytes.len() && bytes[next] == b'\'' && ch != '\n' && ch != '\'').then_some(next + 1)
}

pub enum RawStr {
    Closed { end: usize },
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
    while p < bytes.len() {
        if bytes[p] == b'"' && has_hashes(bytes, p + 1, hashes) {
            return Some(RawStr::Closed { end: p + 1 + hashes });
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
            if bytes.get(i + 1) == Some(&b'\n') {
                nl += 1;
            }
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
// Unified Zero-Copy Pushdown LexCursor<'a>
// ============================================================================

#[inline]
fn utf8_char_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

#[inline]
fn next_unicode_char(bytes: &[u8], pos: usize) -> Option<(char, usize)> {
    let slice = bytes.get(pos..)?;
    if slice.is_empty() {
        return None;
    }
    if slice[0] < 0x80 {
        return Some((slice[0] as char, 1));
    }
    let ch = std::str::from_utf8(slice).ok()?.chars().next()?;
    Some((ch, ch.len_utf8()))
}

#[inline]
fn is_unicode_ident_start(bytes: &[u8], pos: usize) -> bool {
    let Some(&b) = bytes.get(pos) else { return false };
    if b < 0x80 {
        return b == b'_' || b.is_ascii_alphabetic();
    }
    next_unicode_char(bytes, pos).is_some_and(|(ch, _)| ch.is_alphabetic())
}

#[inline]
fn scan_unicode_ident_end(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 0x80 {
            if ident_byte(b) {
                pos += 1;
            } else {
                break;
            }
        } else if let Some((ch, len)) = next_unicode_char(bytes, pos)
            && ch.is_alphanumeric()
        {
            pos += len;
        } else {
            break;
        }
    }
    pos
}

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

fn find_matching_bracket_bytes(bytes: &[u8], open_idx: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut pos = open_idx;
    while pos < bytes.len() {
        let b = bytes[pos];
        if b == b'"' {
            pos = quoted_lit(bytes, pos, true, true).0;
            continue;
        }
        if bytes.get(pos..pos + 2) == Some(b"//") {
            pos = to_eol(bytes, pos + 2);
            continue;
        }
        if b == b'[' {
            depth += 1;
        } else if b == b']' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(pos);
            }
        }
        pos += 1;
    }
    None
}

fn classify_rust_cfg_attr(inner: &[u8]) -> (bool, bool) {
    let mut buf = [0u8; 64];
    let mut len = 0usize;
    for &b in inner {
        if !b.is_ascii_whitespace() {
            if len >= buf.len() {
                return (false, false);
            }
            buf[len] = b;
            len += 1;
        }
    }
    let compact = &buf[..len];
    let is_test = compact == b"cfg(test)"
        || (compact.starts_with(b"cfg(all(")
            && (compact.starts_with(b"cfg(all(test,")
                || compact.ends_with(b",test))")
                || compact.windows(6).any(|w| w == b",test,")));
    let is_os = compact == b"cfg(windows)"
        || compact == b"cfg(unix)"
        || compact == b"cfg(any(windows,unix))"
        || compact == b"cfg(any(unix,windows))";
    (is_test, is_os)
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
        if self.active {
            match kind {
                LexKind::LBrace => self.brace_depth += 1,
                LexKind::RBrace => {
                    self.brace_depth = self.brace_depth.saturating_sub(1);
                    if self.brace_depth == 0 { self.active = false; self.armed = false; }
                }
                _ => {}
            }
            return;
        }
        if !self.armed { return; }
        let top = self.paren_depth == 0 && self.bracket_depth == 0;
        match kind {
            LexKind::LParen => self.paren_depth += 1,
            LexKind::RParen => self.paren_depth = self.paren_depth.saturating_sub(1),
            LexKind::LBracket => self.bracket_depth += 1,
            LexKind::RBracket => self.bracket_depth = self.bracket_depth.saturating_sub(1),
            LexKind::LBrace if top => { self.active = true; self.brace_depth = 1; }
            LexKind::Semi | LexKind::Comma if top => self.armed = false,
            _ => {}
        }
    }
}

fn classify_punct(bytes: &[u8], pos: usize) -> (LexKind, usize) {
    let (b, next, third) = (bytes[pos], bytes.get(pos + 1).copied(), bytes.get(pos + 2).copied());
    match (b, next) {
        (b'=', Some(b'>')) => (LexKind::FatArrow, 2),
        (b'&', Some(b'&')) => (LexKind::AndAnd, 2),
        (b'|', Some(b'|')) => (LexKind::OrOr, 2),
        (b'-', Some(b'>')) => (LexKind::Arrow, 2),
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
        _ => (LexKind::Punct, utf8_char_len(b).min(bytes.len() - pos)),
    }
}

/// Unified zero-copy pushdown byte lexer with stack-only `SmallVec<[LexMode; 8]>`.
pub struct LexCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
    spec: LangSpec,
    modes: SmallVec<[LexMode; 8]>,
    brace_depth: usize,
    prev_sig: u8,
    prev_word: &'a [u8],
    cfg_test: AttrScope,
    os_cfg: AttrScope,
    attr_end: Option<usize>,
}

impl<'a> LexCursor<'a> {
    pub fn new(lang: LangKind, src: &'a [u8]) -> Self {
        Self {
            bytes: src,
            pos: 0,
            line: 1,
            col: 0,
            spec: LangSpec::for_lang(lang),
            modes: smallvec![LexMode::Standard],
            brace_depth: 0,
            prev_sig: b';',
            prev_word: b"",
            cfg_test: AttrScope::default(),
            os_cfg: AttrScope::default(),
            attr_end: None,
        }
    }

    #[inline]
    pub fn has_unclosed_template(&self) -> bool {
        self.modes.iter().any(|m| matches!(m, LexMode::Template { .. }))
    }

    #[inline]
    fn base_flags(&self) -> TokFlags {
        let mut f = TokFlags::default();
        if self.cfg_test.is_in_scope() {
            f = f.with(TokFlags::IN_TEST);
        }
        if self.os_cfg.is_in_scope() {
            f = f.with(TokFlags::IN_OS_CFG);
        }
        if self.attr_end.is_some() {
            f = f.with(TokFlags::IN_ATTR);
        }
        f
    }

    fn advance_pos(&mut self, to: usize) {
        let mut i = self.pos;
        while i < to {
            let b = self.bytes[i];
            if b == b'\n' {
                self.line += 1;
                self.col = 0;
                i += 1;
            } else if b == b'\t' {
                self.col += 4 - (self.col % 4);
                i += 1;
            } else if b == b'\r' {
                i += 1;
            } else if b < 0x80 {
                self.col += 1;
                i += 1;
            } else {
                let step = utf8_char_len(b);
                self.col += 1;
                i = (i + step).min(to);
            }
        }
        self.pos = to;
    }

    fn make_tok(
        &mut self,
        kind: LexKind,
        start: usize,
        end: usize,
        start_line: u32,
        mut flags: TokFlags,
    ) -> LexTok<'a> {
        let start_col = self.col;
        self.advance_pos(end);
        if self.line > start_line { flags = flags.with(TokFlags::MULTILINE); }
        LexTok {
            kind, slice: &self.bytes[start..end], start, end,
            line: start_line, col: start_col, end_line: self.line, end_col: self.col, flags,
        }
    }

    fn emit_lit(
        &mut self, kind: LexKind, start: usize, end: usize,
        start_line: u32, sig: u8, extra_flag: u8,
    ) -> LexTok<'a> {
        self.prev_sig = sig;
        self.prev_word = b"";
        let mut flags = self.base_flags();
        if extra_flag != 0 { flags = flags.with(extra_flag); }
        self.make_tok(kind, start, end, start_line, flags)
    }

    pub fn next_token(&mut self) -> Option<LexTok<'a>> { self.step_token(false).ok().flatten() }
    pub fn next_checked(&mut self) -> Result<Option<LexTok<'a>>, String> { self.step_token(true) }

    fn skip_ws_same_line(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' => { self.col += 1; self.pos += 1; }
                b'\t' => { self.col += 4 - (self.col % 4); self.pos += 1; }
                b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn scan_template_segment(
        &mut self, start: usize, scan_from: usize, start_line: u32, checked: bool,
    ) -> Result<LexTok<'a>, String> {
        let mut i = scan_from;
        while i < self.bytes.len() {
            let c = self.bytes[i];
            if c == b'\\' { i = (i + 2).min(self.bytes.len()); continue; }
            if self.bytes.get(i..i + 2) == Some(b"${") {
                self.modes.push(LexMode::Template { brace_depth: self.brace_depth });
                return Ok(self.emit_lit(LexKind::StrLit, start, i + 2, start_line, b'{', 0));
            }
            if c == b'`' {
                return Ok(self.emit_lit(LexKind::StrLit, start, i + 1, start_line, b'`', 0));
            }
            i += 1;
        }
        if checked { return Err(format!("unterminated template literal at L{start_line}")); }
        Ok(self.emit_lit(LexKind::StrLit, start, i, start_line, b'`', 0))
    }

    fn step_jsx_text(&mut self) -> LexTok<'a> {
        let (start, start_line, byte) = (self.pos, self.line, self.bytes[self.pos]);
        if byte == b'<' {
            self.modes.pop();
            return self.emit_lit(LexKind::Lt, start, start + 1, start_line, b'<', 0);
        }
        if byte == b'{' {
            let depth = self.brace_depth;
            self.brace_depth += 1;
            self.modes.push(LexMode::JsxExpr { brace_depth: depth });
            return self.emit_lit(LexKind::LBrace, start, start + 1, start_line, b'{', 0);
        }
        let flags = self.base_flags();
        if byte == b'\n' { return self.make_tok(LexKind::Newline, start, start + 1, start_line, flags); }
        let mut end = start + 1;
        while end < self.bytes.len() && !matches!(self.bytes[end], b'<' | b'{' | b'\n') { end += 1; }
        self.make_tok(LexKind::StrLit, start, end, start_line, flags)
    }

    fn step_py_fstring_text(&mut self, quote: u8, triple: bool) -> LexTok<'a> {
        let (start, start_line, mut i) = (self.pos, self.line, self.pos);
        while i < self.bytes.len() {
            let b = self.bytes[i];
            if b == b'\\' { i = (i + 2).min(self.bytes.len()); continue; }
            if matches!(self.bytes.get(i..i + 2), Some(b"{{" | b"}}")) { i += 2; continue; }
            if b == b'{' {
                if let Some(slot) = self.modes.last_mut() {
                    *slot = LexMode::PyFString { quote, triple, brace_depth: 1 };
                }
                i += 1;
                break;
            }
            if triple && self.bytes.get(i..i + 3) == Some(&[quote, quote, quote]) {
                self.modes.pop();
                i += 3;
                break;
            }
            if !triple && b == quote {
                self.modes.pop();
                i += 1;
                break;
            }
            i += 1;
        }
        let flags = self.base_flags();
        self.make_tok(LexKind::StrLit, start, i, start_line, flags)
    }

    fn peek_rust_attr(&mut self) {
        if !self.spec.rust_raw_and_cfg || self.bytes[self.pos] != b'#' || self.attr_end.is_some() { return; }
        let bracket = if self.bytes.get(self.pos + 1) == Some(&b'[') {
            Some((self.pos + 1, false))
        } else if self.bytes.get(self.pos + 1..self.pos + 3) == Some(b"![") {
            Some((self.pos + 2, true))
        } else { None };
        if let Some((open, is_inner)) = bracket
            && let Some(close) = find_matching_bracket_bytes(self.bytes, open)
        {
            if !is_inner {
                let (is_test, is_os) = classify_rust_cfg_attr(&self.bytes[open + 1..close]);
                if is_test { self.cfg_test.arm(); } else if is_os { self.os_cfg.arm(); }
            }
            self.attr_end = Some(close);
        }
    }

    fn try_step_comment(
        &mut self, start: usize, start_line: u32, checked: bool,
    ) -> Result<Option<LexTok<'a>>, String> {
        let two = self.bytes.get(start..start + 2);
        if self.spec.slash_comments && two == Some(b"//") {
            let end = to_eol(self.bytes, start + 2);
            let slice = &self.bytes[start..end];
            let mut flags = self.base_flags();
            if (slice.starts_with(b"///") && !slice.starts_with(b"////")) || slice.starts_with(b"//!") {
                flags = flags.with(TokFlags::DOC_COMMENT);
            }
            if find_ascii_ci(slice, b"safety:").is_some() { flags = flags.with(TokFlags::SAFETY_COMMENT); }
            return Ok(Some(self.make_tok(LexKind::LineComment, start, end, start_line, flags)));
        }
        if self.spec.slash_comments && two == Some(b"/*") {
            let (end, _, closed) = block_comment(self.bytes, start, self.spec.nested_block_comments);
            if checked && !closed { return Err(format!("unterminated block comment at L{start_line}")); }
            let slice = &self.bytes[start..end];
            let mut flags = self.base_flags();
            if self.spec.rust_raw_and_cfg
                && matches!(slice.get(2), Some(b'*' | b'!'))
                && slice.get(3) != Some(&b'/')
            {
                flags = flags.with(TokFlags::DOC_COMMENT);
            }
            if find_ascii_ci(slice, b"safety:").is_some() { flags = flags.with(TokFlags::SAFETY_COMMENT); }
            return Ok(Some(self.make_tok(LexKind::BlockComment, start, end, start_line, flags)));
        }
        if self.spec.hash_line_comments && self.bytes[start] == b'#' {
            let (end, flags) = (to_eol(self.bytes, start + 1), self.base_flags());
            return Ok(Some(self.make_tok(LexKind::LineComment, start, end, start_line, flags)));
        }
        Ok(None)
    }

    fn try_step_rust_lit(
        &mut self, start: usize, start_line: u32, checked: bool,
    ) -> Result<Option<LexTok<'a>>, String> {
        let byte = self.bytes[start];
        if let Some(res) = raw_str(self.bytes, start) {
            let end = match res {
                RawStr::Closed { end } => end,
                RawStr::Unterminated if checked => return Err(format!("unterminated Rust raw string at L{start_line}")),
                RawStr::Unterminated => self.bytes.len(),
            };
            return Ok(Some(self.emit_lit(LexKind::StrLit, start, end, start_line, b'"', 0)));
        }
        if matches!(byte, b'b' | b'c') && self.bytes.get(start + 1) == Some(&b'"') && left_bound(self.bytes, start) {
            let (end, _, closed) = quoted_lit(self.bytes, start + 1, true, true);
            if checked && !closed { return Err(format!("unterminated string literal at L{start_line}")); }
            return Ok(Some(self.emit_lit(LexKind::StrLit, start, end, start_line, b'"', 0)));
        }
        if byte == b'b' && left_bound(self.bytes, start)
            && let Some(end) = char_end(self.bytes, start + 1)
        {
            return Ok(Some(self.emit_lit(LexKind::CharLit, start, end, start_line, b'\'', 0)));
        }
        if byte == b'\'' {
            let (kind, end) = match char_end(self.bytes, start) {
                Some(end) => (LexKind::CharLit, end),
                None => (LexKind::Punct, start + 1),
            };
            return Ok(Some(self.emit_lit(kind, start, end, start_line, b'\'', 0)));
        }
        Ok(None)
    }

    fn finish_quoted_lit(
        &mut self, start: usize, start_line: u32, quote: u8, escape: bool, multiline: bool, checked: bool,
    ) -> Result<LexTok<'a>, String> {
        let (end, _, closed) = quoted_lit(self.bytes, start, escape, multiline);
        if checked && !closed {
            return Err(format!("unterminated string literal at L{start_line}"));
        }
        Ok(self.emit_lit(LexKind::StrLit, start, end, start_line, quote, 0))
    }

    fn try_step_script_lit(
        &mut self, start: usize, start_line: u32, checked: bool,
    ) -> Result<Option<LexTok<'a>>, String> {
        let byte = self.bytes[start];
        if self.spec.hash_line_comments {
            if let Some((after_open, quote, triple)) = detect_py_fstring(self.bytes, start) {
                self.modes.push(LexMode::PyFString { quote, triple, brace_depth: 0 });
                let tok = self.emit_lit(LexKind::StrLit, start, after_open, start_line, quote, 0);
                return Ok(Some(tok));
            }
            if matches!(byte, b'"' | b'\'') && self.bytes.get(start..start + 3) == Some(&[byte, byte, byte]) {
                let end = skip_py_triple(self.bytes, start + 3, byte);
                let tok = self.emit_lit(LexKind::StrLit, start, end, start_line, byte, TokFlags::DOC_COMMENT);
                return Ok(Some(tok));
            }
        }
        if self.spec.js_templates_regex {
            if byte == b'/'
                && allow_regex(self.prev_sig, self.prev_word)
                && let Some(end) = scan_regex_literal(self.bytes, start)
            {
                return Ok(Some(self.emit_lit(LexKind::StrLit, start, end, start_line, b'0', 0)));
            }
            if byte == b'`' {
                return self.scan_template_segment(start, start + 1, start_line, checked).map(Some);
            }
        }
        if self.spec.backtick_raw_str && byte == b'`' {
            return self.finish_quoted_lit(start, start_line, b'`', false, true, checked).map(Some);
        }
        Ok(None)
    }

    fn try_step_literal(
        &mut self, start: usize, start_line: u32, checked: bool,
    ) -> Result<Option<LexTok<'a>>, String> {
        if self.spec.rust_raw_and_cfg
            && let Some(tok) = self.try_step_rust_lit(start, start_line, checked)?
        {
            return Ok(Some(tok));
        }
        if let Some(tok) = self.try_step_script_lit(start, start_line, checked)? {
            return Ok(Some(tok));
        }
        let byte = self.bytes[start];
        if byte == b'"' || (!self.spec.rust_raw_and_cfg && byte == b'\'') {
            let multiline = self.spec.rust_raw_and_cfg || (!checked && !self.spec.hash_line_comments);
            return self.finish_quoted_lit(start, start_line, byte, true, multiline, checked).map(Some);
        }
        Ok(None)
    }

    fn step_code_or_punct(&mut self, top: LexMode, start: usize, start_line: u32) -> LexTok<'a> {
        let byte = self.bytes[start];
        let ident_from = if self.spec.rust_raw_and_cfg
            && self.bytes.get(start..start + 2) == Some(b"r#")
            && is_unicode_ident_start(self.bytes, start + 2)
        {
            Some(start + 2)
        } else if is_unicode_ident_start(self.bytes, start) {
            Some(start)
        } else { None };
        if let Some(from) = ident_from {
            let end = scan_unicode_ident_end(self.bytes, from);
            self.prev_word = &self.bytes[start..end];
            self.prev_sig = self.bytes[end - 1];
            let flags = self.base_flags();
            return self.make_tok(LexKind::Ident, start, end, start_line, flags);
        }
        if byte.is_ascii_digit() {
            let mut end = start + 1;
            while end < self.bytes.len() && ident_byte(self.bytes[end]) { end += 1; }
            return self.emit_lit(LexKind::Number, start, end, start_line, self.bytes[end - 1], 0);
        }

        let (kind, step) = classify_punct(self.bytes, start);
        let in_attr = self.attr_end.is_some();
        if !in_attr && self.spec.rust_raw_and_cfg {
            self.cfg_test.step_before(kind);
            self.os_cfg.step_before(kind);
        }
        let flags = self.base_flags();
        if self.spec.jsx_text && kind == LexKind::Gt && self.prev_sig != b'='
            && lookahead_jsx_text(self.bytes, start + 1)
        {
            self.modes.push(LexMode::JsxText);
        }
        match kind {
            LexKind::LBrace => {
                self.brace_depth += 1;
                if let Some(LexMode::PyFString { brace_depth, .. }) = self.modes.last_mut() {
                    *brace_depth += 1;
                }
            }
            LexKind::RBrace => {
                self.brace_depth = self.brace_depth.saturating_sub(1);
                if let LexMode::JsxExpr { brace_depth: target } = top && self.brace_depth == target {
                    self.modes.pop();
                } else if let Some(LexMode::PyFString { brace_depth, .. }) = self.modes.last_mut() {
                    *brace_depth = brace_depth.saturating_sub(1);
                }
            }
            _ => {}
        }
        if !in_attr && self.spec.rust_raw_and_cfg {
            self.cfg_test.step_after(kind);
            self.os_cfg.step_after(kind);
        }
        self.prev_word = b"";
        self.prev_sig = self.bytes[start + step - 1];
        self.make_tok(kind, start, start + step, start_line, flags)
    }

    fn step_token(&mut self, checked: bool) -> Result<Option<LexTok<'a>>, String> {
        if self.pos >= self.bytes.len() { return Ok(None); }
        let top = *self.modes.last().unwrap_or(&LexMode::Standard);
        if matches!(top, LexMode::JsxText) { return Ok(Some(self.step_jsx_text())); }
        if let LexMode::PyFString { quote, triple, brace_depth: 0 } = top {
            return Ok(Some(self.step_py_fstring_text(quote, triple)));
        }
        self.skip_ws_same_line();
        if self.pos >= self.bytes.len() { return Ok(None); }
        if self.attr_end.is_some_and(|end| self.pos > end) { self.attr_end = None; }

        let (start, start_line, byte) = (self.pos, self.line, self.bytes[self.pos]);
        if byte == b'\n' {
            let flags = self.base_flags();
            return Ok(Some(self.make_tok(LexKind::Newline, start, start + 1, start_line, flags)));
        }
        if let LexMode::Template { brace_depth: tpl_depth } = top
            && self.brace_depth == tpl_depth && byte == b'}'
        {
            self.modes.pop();
            return self.scan_template_segment(start, start + 1, start_line, checked).map(Some);
        }
        if let Some(tok) = self.try_step_comment(start, start_line, checked)? { return Ok(Some(tok)); }
        self.peek_rust_attr();
        if let Some(tok) = self.try_step_literal(start, start_line, checked)? { return Ok(Some(tok)); }
        Ok(Some(self.step_code_or_punct(top, start, start_line)))
    }
}

/// Run the unified zero-copy `LexCursor<'a>` in sentinel-blanking mode.
/// Preserves exact 1:1 byte length and `\n` offsets while capturing `#[cfg(test)]` spans.
pub fn blank_noise_sentinel(ext: &str, content: &str) -> SentinelLex {
    let (lang, bytes) = (LangKind::from_ext(ext), content.as_bytes());
    let (mut out, mut test_spans, mut active_test) = (bytes.to_vec(), Vec::new(), None::<(usize, usize)>);
    let mut cursor = LexCursor::new(lang, bytes);
    let flush = |active: &mut Option<(usize, usize)>, spans: &mut Vec<(usize, usize)>| {
        if let Some((s, e)) = active.take() && s < e { spans.push((s, e)); }
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
    if !matches!(lang, LangKind::Rust | LangKind::Go | LangKind::Js | LangKind::Jsx) {
        return Ok(());
    }
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

fn pop_delim(b: u8, line: u32, stack: &mut Vec<(u8, u32)>) -> Result<(), String> {
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
    use crate::testutil::TestRng;

    fn count_nl(bytes: &[u8]) -> usize {
        bytes.iter().filter(|&&b| b == b'\n').count()
    }

    fn assert_sentinel_ok(ext: &str, src: &str) -> SentinelLex {
        let lexed = blank_noise_sentinel(ext, src);
        assert_eq!(lexed.bytes.len(), src.len(), "byte len drift on {ext}");
        assert_eq!(count_nl(&lexed.bytes), count_nl(src.as_bytes()), "newline drift on {ext}");
        lexed
    }

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
        let (a, b) = lexed.test_spans[0];
        assert!(src[a..b].contains("live_fn") && src[a..b].contains("arr"));
    }

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
                let mut doc = String::new();
                for _ in 0..10 {
                    doc.push_str(pieces[rng.next_usize(pieces.len())]);
                }
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

    #[test]
    fn lex_cursor_flags_and_helpers() {
        let rs = concat!(
            "/// Doc comment\n",
            "// SAFETY: verified\n",
            "#[cfg(unix)]\n",
            "pub fn os_only<'a>(x: &'a str) -> char { 'z' }\n",
            "#[cfg(test)]\n",
            "mod tests { fn inner() {} }\n",
        );
        let mut cursor = LexCursor::new(LangKind::Rust, rs.as_bytes());
        let (mut saw_doc, mut saw_safety, mut saw_os, mut saw_test, mut saw_char) =
            (false, false, false, false, false);
        while let Some(tok) = cursor.next_token() {
            saw_doc |= tok.kind == LexKind::LineComment && tok.flags.is_doc_comment();
            saw_safety |= tok.kind == LexKind::LineComment && tok.flags.has(TokFlags::SAFETY_COMMENT);
            saw_os |= tok.kind == LexKind::Ident && tok.as_str() == "os_only" && tok.flags.has(TokFlags::IN_OS_CFG);
            saw_test |= tok.kind == LexKind::Ident && tok.as_str() == "inner" && tok.flags.in_test();
            saw_char |= tok.kind == LexKind::CharLit && tok.as_str() == "'z'";
        }
        assert!(saw_doc && saw_safety && saw_os && saw_test && saw_char);

        assert_eq!(
            iter_backtick_spans("see `src/lex.rs` and `foo()`").collect::<Vec<_>>(),
            vec!["src/lex.rs", "foo()"]
        );
        assert_eq!(find_ascii_ci(b"Hello ReFiT:Ignore", b"refit:ignore"), Some(6));
        assert_eq!(
            iter_idents(b"let foo_1 = bar2 + 99;").collect::<Vec<_>>(),
            vec![b"let".as_slice(), b"foo_1", b"bar2"]
        );
        assert_eq!(extract_comment_body("rs", "  /// hello world"), Some("hello world"));
        let rs_any = "#[cfg(any(feature = \"ste100\", test))]\npub fn prod_feat() {}\n#[cfg(all(unix, test))]\nfn unix_test() {}\n";
        let (mut prod_in_test, mut all_in_test) = (false, false);
        let mut c2 = LexCursor::new(LangKind::Rust, rs_any.as_bytes());
        while let Some(tok) = c2.next_token() {
            if tok.kind == LexKind::Ident && tok.as_str() == "prod_feat" { prod_in_test = tok.flags.in_test(); }
            if tok.kind == LexKind::Ident && tok.as_str() == "unix_test" { all_in_test = tok.flags.in_test(); }
        }
        assert!(!prod_in_test && all_in_test);
        assert_eq!(extract_comment_body("py", "  # py note"), Some("py note"));
    }
}
