use crate::lex::{RawStr, block_comment, char_end, quoted_lit, raw_str, to_eol};
use crate::source::is_js;

#[derive(Default)]
struct CfgSkip {
    pending: bool,
    depth: i32,
    in_brace: bool,
}

impl CfgSkip {
    #[inline]
    fn active(&self) -> bool {
        self.pending && (self.in_brace || self.depth > 0)
    }

    fn arm(&mut self) {
        self.pending = true;
        self.depth = 0;
        self.in_brace = false;
    }

    fn step(&mut self, b: u8, out: &mut String) {
        if b == b'{' {
            self.depth += 1;
            self.in_brace = true;
        } else if b == b'}' && self.in_brace {
            self.depth -= 1;
            if self.depth <= 0 {
                *self = Self::default();
            }
        } else if b == b';' && !self.in_brace {
            self.pending = false;
        } else if b == b'\n' {
            out.push('\n');
        }
    }
}

fn rs_lit(content: &str, bytes: &[u8], i: usize, pending: bool, out: &mut String) -> Option<usize> {
    if let Some(res) = raw_str(bytes, i) {
        let end = match res {
            RawStr::Closed { end, .. } => end,
            RawStr::Unterminated => bytes.len(),
        };
        if !pending {
            out.push_str(&content[i..end]);
        }
        return Some(end);
    }

    if bytes[i] == b'\'' {
        let end = char_end(bytes, i).unwrap_or(i + 1);
        if !pending {
            out.push_str(&content[i..end]);
        }
        return Some(end);
    }
    None
}

/// Strip comments and Rust `#[cfg(test)]` items in one pass while keeping strings intact.
pub fn strip_noise(ext: &str, content: &str) -> String {
    if ext == "py" {
        return strip_python(content);
    }

    let is_rs = ext == "rs";
    let has_tick = is_js(ext) || ext == "go";
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut cfg = CfgSkip::default();
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if bytes.get(i..i + 2) == Some(b"//") {
            i = to_eol(bytes, i + 2);
            continue;
        }

        if bytes.get(i..i + 2) == Some(b"/*") {
            let (end, nl, _) = block_comment(bytes, i, is_rs);
            if !cfg.active() {
                out.extend(std::iter::repeat_n('\n', nl));
            }
            i = end;
            continue;
        }

        if is_rs && !cfg.active() && b == b'#' && content[i..].starts_with("#[cfg(test)]") {
            cfg.arm();
            i += "#[cfg(test)]".len();
            continue;
        }

        if is_rs
            && let Some(end) = rs_lit(content, bytes, i, cfg.pending, &mut out)
        {
            i = end;
            continue;
        }

        if b == b'"' || (!is_rs && b == b'\'') || (has_tick && b == b'`') {
            let (end, _, _) = quoted_lit(bytes, i, b != b'`', true);
            if !cfg.pending {
                out.push_str(&content[i..end]);
            }
            i = end;
            continue;
        }

        if is_rs && cfg.pending {
            cfg.step(b, &mut out);
            i += 1;
            continue;
        }

        let Some(ch) = content[i..].chars().next() else { break };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn strip_python(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'#' {
            i = to_eol(bytes, i + 1);
            continue;
        }

        if matches!(b, b'"' | b'\'') && bytes.get(i..i + 3) == Some(&[b, b, b]) {
            i = skip_triple(bytes, i + 3, b, &mut out);
            continue;
        }

        if matches!(b, b'"' | b'\'') {
            let (end, _, _) = quoted_lit(bytes, i, true, false);
            out.push_str(&content[i..end]);
            i = end;
            continue;
        }

        let Some(ch) = content[i..].chars().next() else { break };
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn skip_triple(bytes: &[u8], mut i: usize, q: u8, out: &mut String) -> usize {
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if bytes[i] == b'\n' {
            out.push('\n');
        }
        if bytes.get(i..i + 3) == Some(&[q, q, q]) {
            return i + 3;
        }
        i += 1;
    }
    i
}
