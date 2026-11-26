use crate::lex::{RUST_PUB, TS_EXPORT, blank_noise_sentinel, first_ident, go_export, match_prefix};
use crate::mvcc::MvccStore;
#[cfg(test)]
use crate::platform::EpochRegistry;
use crate::source::{ext_of, is_js};
use crate::stem::parse::{
    bracket_delta, join_sig_lines, leading_indent, norm_sig, py_def_colon, sig_end,
    split_top_commas,
};

#[cfg(test)]
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
        "rs" => {
            let cleaned = blank_noise_sentinel("rs", source).to_clean_string(source);
            brace_abi(&cleaned, false)
        }
        "go" => go_abi(source),
        e if is_js(e) => brace_abi(source, true),
        _ => MvccStore::hash_bytes(b""),
    }
}

#[cfg(test)]
pub fn hash_digests(digests: &[(&str, &str)]) -> String {
    let mut items = digests.to_vec();
    items.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = blake3::Hasher::new();
    for (rel, abi) in items {
        for part in [rel.as_bytes(), &[0u8], abi.as_bytes(), b"\n"] {
            h.update(part);
        }
    }
    h.finalize().to_hex().to_string()
}

#[cfg(test)]
pub fn check_interface_cutoff(
    pkg_id: &str,
    files: &[(&str, &str)],
    prev_interface_blake3: Option<&str>,
    registry: &EpochRegistry,
) -> InterfaceCutoff {
    let owned: Vec<(&str, String)> = files.iter().map(|&(r, s)| (r, hash_abi(r, s))).collect();
    let refs: Vec<(&str, &str)> = owned.iter().map(|(r, a)| (*r, a.as_str())).collect();
    let next_hash = hash_digests(&refs);
    let changed = prev_interface_blake3 != Some(next_hash.as_str());
    let epoch = if changed {
        registry.publish(pkg_id, u64::from_str_radix(&next_hash[..16], 16).unwrap_or(1))
    } else {
        registry.slot_epoch(pkg_id)
    };
    InterfaceCutoff { pkg_id: pkg_id.into(), interface_blake3: next_hash, changed, epoch }
}

fn finish_abi(mut sigs: Vec<String>) -> String {
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn python_top_item(t: &str, lines: &[&str], i: &mut usize, sigs: &mut Vec<String>, cur_class: &mut Option<String>) -> bool {
    if let Some(rest) = t.strip_prefix("class ") {
        *cur_class = first_ident(rest).filter(|c| !c.starts_with('_')).map(|c| {
            sigs.push(format!("class {c}"));
            c.to_string()
        });
        return true;
    }
    if !t.starts_with('@') { *cur_class = None; }
    if t.starts_with("__all__") {
        let joined = join_sig_lines(lines, i, false, |s| bracket_delta(s) <= 0);
        sigs.push(norm_sig(&joined));
        return true;
    }
    false
}

fn step_py_line(lines: &[&str], i: &mut usize, sigs: &mut Vec<String>, cur_class: &mut Option<String>) {
    let indent = leading_indent(lines[*i]);
    let t = lines[*i].trim();
    if t.is_empty() || t.starts_with('#') { return; }
    if indent == 0 && python_top_item(t, lines, i, sigs, cur_class) { return; }
    let Some(rest) = t.strip_prefix("async def ").or_else(|| t.strip_prefix("def ")) else { return };
    let is_pub = !rest.starts_with('_') || rest.starts_with("__init__");
    if is_pub && (indent == 0 || (indent <= 4 && cur_class.is_some())) {
        let owner = cur_class.as_deref().filter(|_| indent > 0);
        sigs.push(parse_py_def(lines, i, owner));
    }
}

/// Compute the Python exported ABI hash across multi-line defs, classes, and `__all__`.
pub fn python_abi(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let (mut sigs, mut cur_class, mut i) = (Vec::new(), None::<String>, 0usize);
    while i < lines.len() {
        step_py_line(&lines, &mut i, &mut sigs, &mut cur_class);
        i += 1;
    }
    finish_abi(sigs)
}

fn parse_py_def(lines: &[&str], i: &mut usize, owner: Option<&str>) -> String {
    let mut sig = join_sig_lines(lines, i, true, |s| {
        bracket_delta(s) <= 0 && py_def_colon(s).is_some()
    });
    if let Some(colon) = py_def_colon(&sig) { sig.truncate(colon); }
    let norm = norm_params(&sig);
    match owner {
        Some(cls) => format!("{cls}.{norm}"),
        None => norm,
    }
}

fn is_brace_export(t: &str, is_ts: bool) -> bool {
    if is_ts {
        match_prefix(t, TS_EXPORT).is_some() || t.starts_with("export *")
    } else {
        match_prefix(t, RUST_PUB).is_some()
    }
}

fn brace_abi(source: &str, is_ts: bool) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let (mut sigs, mut i) = (Vec::new(), 0usize);
    while i < lines.len() {
        let t = lines[i].trim();
        if is_ts && t.starts_with("export {") {
            let sig = join_sig_lines(&lines, &mut i, false, |s| s.contains('}'));
            sigs.push(norm_sig(sig.split(';').next().unwrap_or(&sig)));
        } else if is_brace_export(t, is_ts) {
            let sig = join_sig_lines(&lines, &mut i, false, |s| sig_end(s, true, is_ts).is_some());
            let end = sig_end(&sig, true, is_ts).unwrap_or(sig.len());
            sigs.push(norm_sig(&sig[..end]));
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
            let sig = join_sig_lines(&lines, &mut i, false, |s| {
                sig_end(s, false, false).is_some() || !s.contains('(') || bracket_delta(s) <= 0
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
    let names: Vec<&str> = split_top_commas(&sig[open + 1..close])
        .into_iter()
        .filter_map(|raw| {
            let head = raw.split('=').next()?.split(':').next()?.trim();
            (!head.is_empty()).then_some(head)
        })
        .collect();
    format!("{}({})", sig[..open].trim(), names.join(", "))
}
