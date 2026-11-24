use crate::lex::{RawStr, block_comment, char_end, ident_byte, quoted_lit, raw_str, to_eol};
use crate::source::{ext_str, is_js};
use std::path::Path;

struct LexCtx<'a> {
    bytes: &'a [u8],
    is_rs: bool,
    is_go: bool,
    is_js: bool,
    is_jsx: bool,
    i: usize,
    line: usize,
    prev_sig: u8,
    prev_word: &'a [u8],
    stack: Vec<(u8, usize)>,
    tpl_depths: Vec<usize>,
}

/// Check balanced braces and strings across Rust, TypeScript, JavaScript, JSX, and Go.
pub fn check_braces(rel_path: &str, source: &str) -> Result<(), String> {
    let ext = ext_str(Path::new(rel_path));
    let is_rs = ext.eq_ignore_ascii_case("rs");
    let is_go = ext.eq_ignore_ascii_case("go");
    let is_js_like = is_js(ext);
    if !is_rs && !is_go && !is_js_like {
        return Ok(());
    }

    let mut cx = LexCtx {
        bytes: source.as_bytes(),
        is_rs,
        is_go,
        is_js: is_js_like,
        is_jsx: ext.eq_ignore_ascii_case("tsx") || ext.eq_ignore_ascii_case("jsx"),
        i: 0,
        line: 1,
        prev_sig: b';',
        prev_word: b"",
        stack: Vec::new(),
        tpl_depths: Vec::new(),
    };

    while cx.i < cx.bytes.len() {
        step_one(&mut cx)?;
    }
    if !cx.tpl_depths.is_empty() {
        return Err("unterminated JS/TS template literal interpolation".into());
    }
    if let Some((open, open_line)) = cx.stack.pop() {
        return Err(format!("unclosed `{}` opened at L{open_line}", open as char));
    }
    Ok(())
}

fn step_one(cx: &mut LexCtx<'_>) -> Result<(), String> {
    let b = cx.bytes[cx.i];
    if b == b'\n' || b.is_ascii_whitespace() {
        cx.line += usize::from(b == b'\n');
        cx.i += 1;
        return Ok(());
    }
    if cx.bytes.get(cx.i..cx.i + 2) == Some(b"//") {
        cx.i = to_eol(cx.bytes, cx.i + 2);
        return Ok(());
    }
    if cx.bytes.get(cx.i..cx.i + 2) == Some(b"/*") {
        return step_comment(cx);
    }
    if step_special(cx, b)? {
        return Ok(());
    }

    if b == b'"' || (cx.is_go && b == b'`') || b == b'\'' {
        return step_quote(cx, b);
    }
    if cx.is_js && (b.is_ascii_alphabetic() || b == b'_') {
        let start = cx.i;
        while cx.i < cx.bytes.len() && ident_byte(cx.bytes[cx.i]) {
            cx.i += 1;
        }
        cx.prev_word = &cx.bytes[start..cx.i];
        cx.prev_sig = cx.bytes[cx.i - 1];
        return Ok(());
    }

    cx.prev_word = b"";
    if matches!(b, b'{' | b'(' | b'[') {
        cx.stack.push((b, cx.line));
    } else if matches!(b, b'}' | b')' | b']') {
        pop_delim(b, cx.line, &mut cx.stack)?;
    }
    cx.prev_sig = b;
    cx.i += 1;
    Ok(())
}

fn step_comment(cx: &mut LexCtx<'_>) -> Result<(), String> {
    let start = cx.line;
    let (end, nl, closed) = block_comment(cx.bytes, cx.i, cx.is_rs);
    if !closed {
        return Err(format!("unterminated block comment at L{start}"));
    }
    cx.line += nl;
    cx.i = end;
    Ok(())
}

fn step_special(cx: &mut LexCtx<'_>, b: u8) -> Result<bool, String> {
    if cx.is_rs
        && let Some(res) = raw_str(cx.bytes, cx.i)
    {
        return match res {
            RawStr::Closed { end, newlines } => {
                cx.line += newlines;
                cx.i = end;
                cx.prev_sig = b'"';
                cx.prev_word = b"";
                Ok(true)
            }
            RawStr::Unterminated => Err(format!("unterminated Rust raw string at L{}", cx.line)),
        };
    }

    if cx.is_js
        && b == b'/'
        && allow_regex(cx.prev_sig, cx.prev_word)
        && let Some(end) = scan_regex(cx.bytes, cx.i)
    {
        cx.i = end;
        cx.prev_sig = b'0';
        cx.prev_word = b"";
        return Ok(true);
    }

    if cx.is_js && (b == b'`' || (b == b'}' && cx.tpl_depths.last() == Some(&cx.stack.len()))) {
        if b == b'}' {
            cx.tpl_depths.pop();
        }
        cx.i += 1;
        scan_template(cx)?;
        cx.prev_sig = b'`';
        cx.prev_word = b"";
        return Ok(true);
    }

    if cx.is_jsx && b == b'>' && cx.prev_sig != b'=' {
        cx.i += 1;
        cx.prev_sig = b'>';
        cx.prev_word = b"";
        if let Some((next_i, add_nl)) = scan_jsx(cx.bytes, cx.i) {
            cx.line += add_nl;
            cx.i = next_i;
        }
        return Ok(true);
    }
    Ok(false)
}

fn step_quote(cx: &mut LexCtx<'_>, b: u8) -> Result<(), String> {
    if b == b'\'' && cx.is_rs {
        cx.i = char_end(cx.bytes, cx.i).unwrap_or(cx.i + 1);
        cx.prev_sig = b'\'';
        cx.prev_word = b"";
        return Ok(());
    }

    let start = cx.line;
    let (end, nl, closed) = quoted_lit(cx.bytes, cx.i, b != b'`', b == b'`' || cx.is_rs);
    if !closed {
        return Err(format!("unterminated string literal at L{start}"));
    }
    cx.line += nl;
    cx.i = end;
    cx.prev_sig = b;
    cx.prev_word = b"";
    Ok(())
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

fn scan_regex(bytes: &[u8], i: usize) -> Option<usize> {
    let (mut p, mut in_class) = (i + 1, false);
    while p < bytes.len() && bytes[p] != b'\n' {
        if bytes[p] == b'\\' {
            p += 2;
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

fn scan_jsx(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let (mut look, mut lines, mut saw_alpha) = (start, 0usize, false);
    while look < bytes.len() && !matches!(bytes[look], b'<' | b'{') {
        let c = bytes[look];
        if c == b'\n' {
            lines += 1;
        } else if c.is_ascii_alphabetic() {
            saw_alpha = true;
        } else if matches!(c, b';' | b')' | b']' | b'}' | b'?' | b'=' | b'"' | b'`') {
            return None;
        }
        look += 1;
    }
    (saw_alpha && look < bytes.len()).then_some((look, lines))
}

fn scan_template(cx: &mut LexCtx<'_>) -> Result<(), String> {
    let start = cx.line;
    while cx.i < cx.bytes.len() {
        let c = cx.bytes[cx.i];
        if c == b'\\' {
            cx.i += 2;
            continue;
        }
        cx.line += usize::from(c == b'\n');
        if cx.bytes.get(cx.i..cx.i + 2) == Some(b"${") {
            cx.i += 2;
            cx.tpl_depths.push(cx.stack.len());
            return Ok(());
        }
        if c == b'`' {
            cx.i += 1;
            return Ok(());
        }
        cx.i += 1;
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
