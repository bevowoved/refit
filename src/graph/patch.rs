use crate::lex::eq_ws;
use crate::platform::check_rel;
use crate::types::{AsrfHunk, PatchOp};
use anyhow::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct AppliedPatch {
    pub updated: BTreeMap<String, String>,
    pub created: Vec<String>,
    pub deleted: Vec<String>,
    pub renamed: Vec<(String, String)>,
}

/// Parse ASRF hunks supporting `edit`, `create`, `delete`, and `rename`.
pub fn parse_patches(content: &str) -> Vec<AsrfHunk> {
    let mut hunks = Vec::new();
    let norm = content.replace("\r\n", "\n");
    let mut cursor = 0usize;

    while let Some(rel_start) = norm[cursor..].find("<<<REFIT_PATCH") {
        let start = cursor + rel_start;
        let Some(hdr_rel) = norm[start..].find(">>>") else { break };
        let header = &norm[start..start + hdr_rel];
        let after_hdr = start + hdr_rel + 3;
        let Some(end_rel) = norm[after_hdr..].find("<<<END>>>") else { break };

        let body = &norm[after_hdr..after_hdr + end_rel];
        cursor = after_hdr + end_rel + "<<<END>>>".len();

        if let Some(hunk) = parse_hunk(header, body) {
            hunks.push(hunk);
        }
    }
    hunks
}

fn parse_hunk(header: &str, body: &str) -> Option<AsrfHunk> {
    let path = extract_attr(header, "path=");
    let op = match extract_attr(header, "op=").to_ascii_lowercase().as_str() {
        "create" => PatchOp::Create,
        "delete" => PatchOp::Delete,
        "rename" | "move" => PatchOp::Rename,
        _ => PatchOp::Edit,
    };
    let np = extract_attr(header, "new_path=");
    let new_path = (!np.is_empty()).then(|| np.replace('\\', "/"));

    let (search, replace) = split_body(body, op)?;
    let valid = match op {
        PatchOp::Edit => !path.is_empty() && !search.is_empty(),
        PatchOp::Create | PatchOp::Delete => !path.is_empty(),
        PatchOp::Rename => !path.is_empty() && new_path.is_some(),
    };
    valid.then(|| AsrfHunk {
        path: path.replace('\\', "/"),
        op,
        new_path,
        search,
        replace,
    })
}

fn split_body(body: &str, op: PatchOp) -> Option<(String, String)> {
    const S_TAG: &str = "<<<SEARCH>>>";
    const R_TAG: &str = "<<<REPLACE>>>";

    if let (Some(sr), Some(rr)) = (body.find(S_TAG), body.find(R_TAG)) {
        if rr < sr + S_TAG.len() {
            return None;
        }
        return Some((
            trim_newline(&body[sr + S_TAG.len()..rr]),
            trim_newline(&body[rr + R_TAG.len()..]),
        ));
    }
    if let Some(rr) = body.find(R_TAG) {
        return Some((String::new(), trim_newline(&body[rr + R_TAG.len()..])));
    }
    matches!(op, PatchOp::Delete | PatchOp::Rename).then(|| (String::new(), String::new()))
}

fn extract_attr(header: &str, key: &str) -> String {
    for (pos, _) in header.match_indices(key) {
        if pos > 0 && !header.as_bytes()[pos - 1].is_ascii_whitespace() {
            continue;
        }
        let rest = header[pos + key.len()..].trim_start();
        if let Some(q) = rest.chars().next().filter(|c| matches!(c, '"' | '\''))
            && let Some(end) = rest[1..].find(q)
        {
            return rest[1..1 + end].to_string();
        }
        return rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(['"', '\''])
            .to_string();
    }
    String::new()
}

fn trim_newline(s: &str) -> String {
    s.strip_prefix('\n')
        .unwrap_or(s)
        .strip_suffix('\n')
        .unwrap_or(s)
        .to_string()
}

pub fn apply_patches(
    base_files: &BTreeMap<String, String>,
    hunks: &[AsrfHunk],
) -> Result<AppliedPatch> {
    if hunks.is_empty() {
        anyhow::bail!("No <<<REFIT_PATCH>>> blocks found in assistant output");
    }

    let mut patch = AppliedPatch {
        updated: base_files.clone(),
        ..AppliedPatch::default()
    };

    for (idx, h) in hunks.iter().enumerate() {
        if !check_rel(&h.path) {
            anyhow::bail!("Hunk #{idx} unsafe path outside repository: `{}`", h.path);
        }
        if let Some(ref np) = h.new_path
            && !check_rel(np)
        {
            anyhow::bail!("Hunk #{idx} unsafe new_path outside repository: `{np}`");
        }
        apply_one(&mut patch, h, idx)?;
    }
    Ok(patch)
}

fn apply_one(patch: &mut AppliedPatch, h: &AsrfHunk, idx: usize) -> Result<()> {
    match h.op {
        PatchOp::Create => {
            patch.updated.insert(h.path.clone(), h.replace.replace("\r\n", "\n"));
            push_unique(&mut patch.created, &h.path);
            patch.deleted.retain(|d| d != &h.path);
        }
        PatchOp::Delete => {
            let key = match_target(&patch.updated, &h.path, idx, false)?;
            patch.updated.remove(&key);
            patch.created.retain(|c| c != &key);
            push_unique(&mut patch.deleted, &key);
        }
        PatchOp::Rename => {
            let key = match_target(&patch.updated, &h.path, idx, false)?;
            let dest = h
                .new_path
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Hunk #{idx} rename missing new_path"))?;
            let mut body = patch.updated.remove(&key).unwrap_or_default().replace("\r\n", "\n");
            if !h.search.is_empty() {
                body = apply_hunk(&body, &h.search, &h.replace, idx, &key)?;
            }
            if key != dest {
                push_unique(&mut patch.deleted, &key);
                patch.deleted.retain(|d| d != &dest);
                push_unique(&mut patch.created, &dest);
                patch.renamed.push((key, dest.clone()));
            }
            patch.updated.insert(dest, body);
        }
        PatchOp::Edit => {
            let key = match_target(&patch.updated, &h.path, idx, true)?;
            let raw = patch.updated.get(&key).map_or("", String::as_str);
            let cur = raw.replace("\r\n", "\n");
            let next = apply_hunk(&cur, &h.search, &h.replace, idx, &key)?;
            patch.updated.insert(key, next);
        }
    }
    Ok(())
}

fn push_unique(list: &mut Vec<String>, item: &str) {
    if !list.iter().any(|x| x == item) {
        list.push(item.to_string());
    }
}

fn match_target(
    updated: &BTreeMap<String, String>,
    path: &str,
    idx: usize,
    allow_single: bool,
) -> Result<String> {
    if updated.contains_key(path) {
        return Ok(path.to_string());
    }
    if let Some(k) = updated
        .keys()
        .find(|k| k.ends_with(&format!("/{path}")) || path.ends_with(&format!("/{k}")))
    {
        return Ok(k.clone());
    }
    if allow_single && updated.len() == 1 {
        let only = updated.keys().next().unwrap();
        return Ok(only.clone());
    }
    anyhow::bail!("Hunk #{idx} targets unknown path `{path}`")
}

fn apply_hunk(cur: &str, search: &str, replace: &str, idx: usize, key: &str) -> Result<String> {
    let s_norm = search.replace("\r\n", "\n");
    let r_norm = replace.replace("\r\n", "\n");

    let mut exact = cur.match_indices(&s_norm).map(|(p, _)| p);
    if let Some(pos) = exact.next() {
        if exact.next().is_some() {
            anyhow::bail!("Hunk #{idx} ambiguous SEARCH block in `{key}`; add context lines");
        }
        let mut next = String::with_capacity(cur.len() + r_norm.len());
        next.push_str(&cur[..pos]);
        next.push_str(&r_norm);
        next.push_str(&cur[pos + s_norm.len()..]);
        return Ok(next);
    }

    if let Some(next) = replace_window(cur, &s_norm, &r_norm, |a, b| a.trim() == b.trim()) {
        return Ok(next);
    }
    if let Some(next) = replace_elided(cur, &s_norm, &r_norm) {
        return Ok(next);
    }
    if let Some(next) = replace_window(cur, &s_norm, &r_norm, eq_ws) {
        return Ok(next);
    }

    anyhow::bail!(
        "Hunk #{idx} SEARCH block did not match `{key}` (first line: `{}`)",
        s_norm.lines().next().unwrap_or("")
    )
}

fn splice_lines(doc: &str, chunks: &[&[&str]]) -> String {
    let cap: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(cap);
    for chunk in chunks {
        out.extend_from_slice(chunk);
    }

    let mut joined = out.join("\n");
    if doc.ends_with('\n') {
        joined.push('\n');
    }
    joined
}

/// Zero-allocation line-window search using a borrowed predicate (`trim` or `eq_ws`).
pub(crate) fn replace_window(
    doc: &str,
    search: &str,
    replace: &str,
    eq_line: fn(&str, &str) -> bool,
) -> Option<String> {
    let d_lines: Vec<&str> = doc.lines().collect();
    let s_lines: Vec<&str> = search.lines().collect();
    if s_lines.is_empty() || s_lines.len() > d_lines.len() {
        return None;
    }

    let mut hits = (0..=d_lines.len() - s_lines.len()).filter(|&i| {
        s_lines
            .iter()
            .enumerate()
            .all(|(j, sl)| eq_line(d_lines[i + j], sl))
    });
    let start = hits.next()?;
    if hits.next().is_some() {
        return None;
    }

    let r_lines: Vec<&str> = replace.lines().collect();
    Some(splice_lines(
        doc,
        &[&d_lines[..start], &r_lines, &d_lines[start + s_lines.len()..]],
    ))
}

fn is_elision(line: &str) -> bool {
    let t = line.trim().to_ascii_lowercase();
    t.contains("... existing code ...")
        || t.contains("... unchanged ...")
        || matches!(t.as_str(), "// ..." | "# ..." | "/* ... */")
}

fn find_block(doc: &[&str], needle: &[&str], from: usize) -> Option<usize> {
    if needle.is_empty() || from + needle.len() > doc.len() {
        return None;
    }
    let mut hits = (from..=doc.len() - needle.len()).filter(|&i| {
        needle
            .iter()
            .enumerate()
            .all(|(j, nl)| doc[i + j].trim() == nl.trim())
    });
    let first = hits.next()?;
    hits.next().is_none().then_some(first)
}

fn replace_elided(doc: &str, search: &str, replace: &str) -> Option<String> {
    let s_lines: Vec<&str> = search.lines().collect();
    let r_lines: Vec<&str> = replace.lines().collect();
    let s_mark = s_lines.iter().position(|l| is_elision(l))?;
    let r_mark = r_lines.iter().position(|l| is_elision(l))?;
    if s_lines[s_mark + 1..].iter().any(|l| is_elision(l))
        || r_lines[r_mark + 1..].iter().any(|l| is_elision(l))
    {
        return None;
    }

    let (s_head, s_tail) = (&s_lines[..s_mark], &s_lines[s_mark + 1..]);
    if s_head.is_empty() || s_tail.is_empty() {
        return None;
    }

    let d_lines: Vec<&str> = doc.lines().collect();
    let h_start = find_block(&d_lines, s_head, 0)?;
    let h_end = h_start + s_head.len();
    let t_start = find_block(&d_lines, s_tail, h_end)?;
    let t_end = t_start + s_tail.len();

    Some(splice_lines(
        doc,
        &[
            &d_lines[..h_start],
            &r_lines[..r_mark],
            &d_lines[h_end..t_start],
            &r_lines[r_mark + 1..],
            &d_lines[t_end..],
        ],
    ))
}
