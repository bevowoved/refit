use crate::lex::{RUST_PUB, TS_EXPORT, first_ident, go_export, match_prefix, scan_code};
use crate::mvcc::MvccStore;
use crate::platform::EpochRegistry;
use crate::source::{ext_of, is_js};
use crate::stem::strip_noise;
use std::ops::ControlFlow;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterfaceCutoff {
    pub pkg_id: String,
    pub interface_blake3: String,
    pub changed: bool,
    pub epoch: u64,
}

pub fn hash_abi(rel_path: &str, source: &str) -> String {
    let ext = ext_of(rel_path);
    match ext.as_str() {
        "py" => python_abi(source),
        "rs" => rust_abi(source),
        "go" => go_abi(source),
        e if is_js(e) => ts_abi(source),
        _ => MvccStore::hash_bytes(b""),
    }
}

/// Compute a deterministic Salsa-style package `interface_blake3` over sorted `(rel, hash_abi)` pairs.
pub fn interface_blake3(files: &[(&str, &str)]) -> String {
    let mut items: Vec<(&str, String)> = files
        .iter()
        .map(|&(rel, src)| (rel, hash_abi(rel, src)))
        .collect();
    items.sort_by(|lhs, rhs| lhs.0.cmp(rhs.0));

    let mut hasher = blake3::Hasher::new();
    for (rel, abi) in items {
        hasher.update(rel.as_bytes());
        hasher.update(&[0u8]);
        hasher.update(abi.as_bytes());
        hasher.update(b"\n");
    }
    hasher.finalize().to_hex().to_string()
}

/// Salsa early-cutoff helper wired to `hash_abi` and `EpochRegistry`.
/// Preserves the current epoch with zero invalidations when `interface_blake3` is unchanged.
pub fn check_interface_cutoff(
    pkg_id: &str,
    files: &[(&str, &str)],
    prev_interface_blake3: Option<&str>,
    registry: &EpochRegistry,
) -> InterfaceCutoff {
    let next_hash = interface_blake3(files);
    if prev_interface_blake3 == Some(next_hash.as_str()) {
        let epoch = registry
            .try_read(pkg_id)
            .map_or_else(|| registry.global_epoch(), |(ep, _)| ep);
        return InterfaceCutoff {
            pkg_id: pkg_id.to_string(),
            interface_blake3: next_hash,
            changed: false,
            epoch,
        };
    }
    let rev = u64::from_str_radix(&next_hash[..16], 16).unwrap_or(1);
    let epoch = registry.publish(pkg_id, rev);
    InterfaceCutoff {
        pkg_id: pkg_id.to_string(),
        interface_blake3: next_hash,
        changed: true,
        epoch,
    }
}

/// Compute the Python exported ABI hash across multi-line defs, classes, and `__all__`.
pub fn python_abi(source: &str) -> String {
    let mut sigs = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut cur_class: Option<String> = None;
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i];
        let indent = line.bytes().take_while(|&b| matches!(b, b' ' | b'\t')).count();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            i += 1;
            continue;
        }

        if indent == 0 {
            if let Some(rest) = t.strip_prefix("class ") {
                cur_class = first_ident(rest)
                    .filter(|c| !c.starts_with('_'))
                    .map(|c| {
                        sigs.push(format!("class {c}"));
                        c.to_string()
                    });
                i += 1;
                continue;
            }
            if !t.starts_with('@') {
                cur_class = None;
            }
            if t.starts_with("__all__") {
                let mut sig = t.to_string();
                while paren_delta(&sig) > 0 && i + 1 < lines.len() {
                    i += 1;
                    sig.push(' ');
                    sig.push_str(lines[i].trim());
                }
                sigs.push(norm_sig(&sig));
                i += 1;
                continue;
            }
        }

        if let Some(rest) = t.strip_prefix("async def ").or_else(|| t.strip_prefix("def ")) {
            let public = !rest.starts_with('_') || rest.starts_with("__init__");
            let in_scope = indent == 0 || (indent <= 4 && cur_class.is_some());
            if public && in_scope {
                push_pydef(&lines, &mut i, indent, cur_class.as_deref(), &mut sigs);
            }
        }
        i += 1;
    }

    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn push_pydef(
    lines: &[&str],
    i: &mut usize,
    indent: usize,
    cur_class: Option<&str>,
    sigs: &mut Vec<String>,
) {
    let mut sig = strip_hash(lines[*i].trim()).to_string();
    while (paren_delta(&sig) > 0 || def_colon(&sig).is_none()) && *i + 1 < lines.len() {
        *i += 1;
        sig.push(' ');
        sig.push_str(strip_hash(lines[*i].trim()));
    }
    if let Some(colon) = def_colon(&sig) {
        sig.truncate(colon);
    }

    let norm = norm_params(&sig);
    if indent > 0 {
        if let Some(cls) = cur_class {
            sigs.push(format!("{cls}.{norm}"));
        }
    } else {
        sigs.push(norm);
    }
}

fn strip_hash(line: &str) -> &str {
    let mut cut = line.len();
    scan_code(line.as_bytes(), b"\"'", |i, b| {
        if b == b'#' {
            cut = i;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    line[..cut].trim_end()
}

fn paren_delta(s: &str) -> i32 {
    let mut d = 0i32;
    scan_code(s.as_bytes(), b"\"'", |_, b| {
        match b {
            b'(' | b'[' | b'{' => d += 1,
            b')' | b']' | b'}' => d -= 1,
            _ => {}
        }
        ControlFlow::Continue(())
    });
    d
}

fn def_colon(sig: &str) -> Option<usize> {
    let (mut depth, mut closed, mut found) = (0i32, false, None);
    scan_code(sig.as_bytes(), b"\"'", |i, b| {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                closed |= b == b')' && depth == 0;
            }
            b':' if depth == 0 && closed => {
                found = Some(i);
                return ControlFlow::Break(());
            }
            _ => {}
        }
        ControlFlow::Continue(())
    });
    found
}

fn norm_sig(sig: &str) -> String {
    let mut collapsed = String::with_capacity(sig.len());
    for (idx, word) in sig.split_whitespace().enumerate() {
        if idx > 0 {
            collapsed.push(' ');
        }
        collapsed.push_str(word);
    }
    let bytes = collapsed.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b' ' {
            let prev = out.last().copied();
            let next = bytes.get(i + 1).copied();
            if matches!(prev, Some(b'(' | b'<' | b'['))
                || matches!(next, Some(b')' | b'>' | b']' | b','))
            {
                i += 1;
                continue;
            }
        }
        if b == b',' && matches!(bytes.get(i + 1), Some(b')' | b'>')) {
            i += 1;
            continue;
        }
        if matches!(b, b')' | b'>') && out.last() == Some(&b',') {
            out.pop();
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).unwrap_or(collapsed)
}

fn sig_end(sig: &str, allow_semi: bool, stop_arrow: bool) -> Option<usize> {
    let b = sig.as_bytes();
    let (mut paren, mut bracket, mut angle, mut skip_to, mut found) = (0i32, 0i32, 0i32, 0usize, None);

    scan_code(b, b"\"'`", |i, c| {
        if i < skip_to {
            return ControlFlow::Continue(());
        }
        if b.get(i..i + 2) == Some(b"->") {
            skip_to = i + 2;
            return ControlFlow::Continue(());
        }
        if b.get(i..i + 2) == Some(b"=>") {
            if stop_arrow && paren == 0 && bracket == 0 && angle == 0 {
                found = Some(i);
                return ControlFlow::Break(());
            }
            skip_to = i + 2;
            return ControlFlow::Continue(());
        }

        match c {
            b'(' => paren += 1,
            b')' => paren = (paren - 1).max(0),
            b'[' => bracket += 1,
            b']' => bracket = (bracket - 1).max(0),
            b'<' => angle += 1,
            b'>' => angle = (angle - 1).max(0),
            b'{' if paren == 0 && bracket == 0 && angle == 0 => {
                if b.get(i + 1) == Some(&b'}') {
                    skip_to = i + 2;
                } else {
                    found = Some(i);
                    return ControlFlow::Break(());
                }
            }
            b';' if allow_semi && paren == 0 && bracket == 0 && angle == 0 => {
                found = Some(i);
                return ControlFlow::Break(());
            }
            _ => {}
        }
        ControlFlow::Continue(())
    });
    found
}

fn join_sig(lines: &[&str], i: &mut usize, stop: impl Fn(&str) -> bool) -> String {
    let mut sig = lines[*i].trim().to_string();
    while !stop(&sig) && *i + 1 < lines.len() {
        *i += 1;
        sig.push(' ');
        sig.push_str(lines[*i].trim());
    }
    sig
}

fn finish_abi(mut sigs: Vec<String>) -> String {
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn push_ended(lines: &[&str], i: &mut usize, sigs: &mut Vec<String>, braces: bool, eq: bool) {
    let sig = join_sig(lines, i, |s| sig_end(s, braces, eq).is_some());
    let end = sig_end(&sig, braces, eq).unwrap_or(sig.len());
    sigs.push(norm_sig(&sig[..end]));
}

fn rust_abi(source: &str) -> String {
    let cleaned = strip_noise("rs", source);
    let lines: Vec<&str> = cleaned.lines().collect();
    let (mut sigs, mut i) = (Vec::new(), 0usize);

    while i < lines.len() {
        if match_prefix(lines[i].trim(), RUST_PUB).is_some() {
            push_ended(&lines, &mut i, &mut sigs, true, false);
        }
        i += 1;
    }
    finish_abi(sigs)
}

fn ts_abi(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let (mut sigs, mut i) = (Vec::new(), 0usize);

    while i < lines.len() {
        let t = lines[i].trim();
        if t.starts_with("export {") {
            let sig = join_sig(&lines, &mut i, |s| s.contains('}'));
            sigs.push(norm_sig(sig.split(';').next().unwrap_or(&sig)));
        } else if match_prefix(t, TS_EXPORT).is_some() || t.starts_with("export *") {
            push_ended(&lines, &mut i, &mut sigs, true, true);
        }
        i += 1;
    }
    finish_abi(sigs)
}

fn go_abi(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let (mut sigs, mut i) = (Vec::new(), 0usize);

    while i < lines.len() {
        if go_export(lines[i].trim()).is_some() {
            let sig = join_sig(&lines, &mut i, |s| {
                sig_end(s, false, false).is_some() || !s.contains('(') || paren_delta(s) <= 0
            });
            let end = sig_end(&sig, false, false).unwrap_or(sig.len());
            sigs.push(norm_sig(&sig[..end]));
        }
        i += 1;
    }
    finish_abi(sigs)
}

fn norm_params(sig: &str) -> String {
    if sig.starts_with("class ") {
        let head = sig.split('(').next().unwrap_or(sig);
        return head.trim().to_string();
    }
    let (Some(open), Some(close)) = (sig.find('('), sig.rfind(')')) else {
        return sig.trim().to_string();
    };

    let prefix = sig[..open].trim();
    let raw = &sig[open + 1..close];
    let mut names = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);

    for (idx, ch) in raw.char_indices() {
        match ch {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                push_param(&raw[start..idx], &mut names);
                start = idx + 1;
            }
            _ => {}
        }
    }
    push_param(&raw[start..], &mut names);
    format!("{prefix}({})", names.join(", "))
}

fn push_param<'a>(raw: &'a str, out: &mut Vec<&'a str>) {
    let name = raw
        .trim()
        .split('=')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .trim();
    if !name.is_empty() {
        out.push(name);
    }
}
