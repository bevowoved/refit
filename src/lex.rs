use std::borrow::Cow;
use std::ops::ControlFlow;

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
