use crate::lex::{first_ident, ident_byte};
use crate::platform::char_prefix;
use crate::types::ModuleUnit;
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

const DECL_STARTS: &[&str] = &[
    "def ", "async def ", "class ", "pub fn ", "pub async fn ",
    "fn ", "export ", "function ", "func ",
];

fn top_decl(line: &str) -> bool {
    !line.starts_with([' ', '\t']) && DECL_STARTS.iter().any(|p| line.starts_with(p))
}

fn parse_diags(diagnostics: &[String]) -> HashSet<usize> {
    let mut lines = HashSet::new();
    for d in diagnostics {
        parse_linekw(d, &mut lines);
        parse_lcolon(d, &mut lines);
    }
    lines
}

fn push_line(tok: &str, out: &mut HashSet<usize>) {
    if let Ok(n) = tok.parse::<usize>()
        && n > 0
        && n < 100_000
    {
        out.insert(n);
    }
}

fn parse_linekw(d: &str, out: &mut HashSet<usize>) {
    let lower = d.to_ascii_lowercase();
    let mut scan = lower.as_str();
    while let Some(pos) = scan.find("line ") {
        let after = &scan[pos + 5..];
        let digits: &str = &after[..after.bytes().take_while(u8::is_ascii_digit).count()];
        push_line(digits, out);
        scan = after;
    }
}

fn parse_lcolon(d: &str, out: &mut HashSet<usize>) {
    let b = d.as_bytes();
    for i in 0..b.len() {
        let is_l = b[i] == b'L' && (i == 0 || !b[i - 1].is_ascii_alphanumeric());
        let is_colon = b[i] == b':'
            && b.get(i + 1..i + 3) != Some(b"//")
            && !["http", "https", "127.0.0.1", "localhost"]
                .iter()
                .any(|p| d[..i].ends_with(p));
        if !(is_l || is_colon) {
            continue;
        }

        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        let ok_end = j == b.len() || !b[j].is_ascii_alphabetic();
        if j > i + 1 && ok_end {
            push_line(&d[i + 1..j], out);
        }
    }
}

fn failed_test(test_src: &str, diagnostics: &[String], max_chars: usize) -> String {
    let mut names = Vec::new();
    for d in diagnostics {
        for marker in ["::test_", "::probe_", "FAIL: Test", "test "] {
            let mut rest = d.as_str();
            while let Some(pos) = rest.find(marker) {
                let off = match marker {
                    "::test_" | "::probe_" => 2,
                    "FAIL: Test" => 6,
                    _ => marker.len(),
                };
                let slice = &rest[pos + off..];
                let tail = slice.rsplit("::").next().unwrap_or("");
                let end = tail.bytes().take_while(|&b| ident_byte(b)).count();
                let name = &tail[..end];
                if name.len() >= 4 && !names.iter().any(|n| n == name) {
                    names.push(name.to_string());
                }
                rest = &rest[pos + marker.len()..];
            }
        }
    }

    if names.is_empty() {
        return char_prefix(test_src, max_chars).to_string();
    }

    let src_lines: Vec<&str> = test_src.lines().collect();
    let mut out = String::new();
    for fname in &names {
        let needles = [
            format!("def {fname}"),
            format!("fn {fname}"),
            format!("func {fname}"),
            format!("'{fname}'"),
            format!("\"{fname}\""),
        ];
        if let Some(start) = src_lines
            .iter()
            .position(|l| needles.iter().any(|n| l.contains(n.as_str())))
        {
            let end = (start + 45).min(src_lines.len());
            out.push_str(&src_lines[start..end].join("\n"));
            out.push('\n');
            if out.chars().count() >= max_chars {
                break;
            }
        }
    }

    if out.is_empty() {
        char_prefix(test_src, max_chars).to_string()
    } else {
        char_prefix(&out, max_chars).to_string()
    }
}

pub fn build_packet(
    repo_root: &Path,
    unit: &ModuleUnit,
    diagnostics: &[String],
    learn_suffix: &str,
    max_chars: usize,
) -> Result<String> {
    let mut out = format!(
        "Refactor unit `{}` (wave {}, layer {}, baseline hygiene {:.2}).\n",
        unit.id, unit.wave, unit.layer, unit.baseline_hygiene
    );

    let unique: BTreeSet<&str> = diagnostics
        .iter()
        .map(|d| d.trim())
        .filter(|d| !d.is_empty())
        .collect();
    if !unique.is_empty() {
        out.push_str("Current Gate 1/2/3 diagnostics to fix completely:\n");
        for d in unique.iter().take(36) {
            out.push_str("- ");
            out.push_str(d);
            out.push('\n');
        }
    }

    if unit.files.len() > 1 {
        out.push_str(concat!(
            "IMPORTANT: This cyclic SCC unit has multiple files. ",
            "Emit <<<REFIT_PATCH>>> blocks to fix ALL defects in EVERY file below.\n"
        ));
    }

    let test_fail = unique.iter().any(|d| {
        d.to_lowercase().contains("pytest")
            || d.contains("FAILED")
            || d.contains("assert")
            || d.contains("Gate 1")
    });
    if test_fail {
        for tf in unit.test_files.iter().take(2) {
            if let Ok(tc) = fs::read_to_string(repo_root.join(tf)) {
                let snip = failed_test(&tc, diagnostics, 2400);
                out.push_str(&format!("\nRelevant failing test `{tf}`:\n```\n{snip}\n```\n"));
            }
        }
    }

    let diag_lines = parse_diags(diagnostics);
    let per_file = max_chars / unit.files.len().max(1);
    for rel in &unit.files {
        let content = fs::read_to_string(repo_root.join(rel))
            .with_context(|| format!("read {rel}"))?;
        let sliced = slice_anchored(&content, per_file, &diag_lines);
        out.push_str(&format!("\nFile `{rel}`:\n```\n{sliced}\n```\n"));
    }

    if !learn_suffix.is_empty() {
        out.push_str(learn_suffix);
    }
    out.push_str(concat!(
        "\nEmit ONLY raw <<<REFIT_PATCH path=\"...\">>> blocks directly in your reply. ",
        "Every <<<SEARCH>>> block must match uniquely. ",
        "Preserve exported signatures unless allow_abi_change is enabled.\n"
    ));
    Ok(out)
}

pub(crate) fn slice_anchored(
    content: &str,
    budget_chars: usize,
    diag_lines: &HashSet<usize>,
) -> String {
    if content.chars().count() <= budget_chars {
        return content.to_string();
    }

    let lines: Vec<&str> = content.lines().collect();
    let blocks = split_blocks(&lines, diag_lines);
    let mut out = String::new();
    let (mut chars, mut omitted) = (0usize, 0usize);

    for (start, end, hot) in blocks {
        let chunk = lines[start..end].join("\n");
        let chunk_c = chunk.chars().count();
        if hot || chars + chunk_c <= budget_chars {
            if omitted > 0 {
                let marker = format!("\n// ... ({omitted} clean lines omitted) ...\n");
                chars += marker.chars().count();
                out.push_str(&marker);
                omitted = 0;
            }
            out.push_str(&chunk);
            out.push('\n');
            chars += chunk_c + 1;
        } else {
            if let Some(first) = lines[start..end].iter().find(|l| top_decl(l)) {
                let stub = format!("// stub: {}\n", char_prefix(first.trim(), 120));
                chars += stub.chars().count();
                out.push_str(&stub);
            }
            omitted += end.saturating_sub(start);
        }
    }
    out
}

fn is_leading_meta(line: &str) -> bool {
    if line.starts_with([' ', '\t']) {
        return false;
    }
    let t = line.trim();
    t.starts_with("#[")
        || t.starts_with('@')
        || t.starts_with("///")
        || t.starts_with("/**")
        || t.starts_with("* ")
        || t.starts_with("*/")
        || t.starts_with("//")
}

fn attach_leading_meta(lines: &[&str], decl_idx: usize, min_idx: usize) -> usize {
    let mut start = decl_idx;
    while start > min_idx {
        let prev = lines[start - 1];
        if prev.trim().is_empty() || !is_leading_meta(prev) {
            break;
        }
        start -= 1;
    }
    start
}

fn split_blocks(lines: &[&str], diag_lines: &HashSet<usize>) -> Vec<(usize, usize, bool)> {
    let mut decl_starts = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if top_decl(line) {
            decl_starts.push(idx);
        }
    }
    if decl_starts.is_empty() {
        return vec![(0, lines.len(), true)];
    }

    let first_start = attach_leading_meta(lines, decl_starts[0], 0);
    let mut blocks = Vec::with_capacity(decl_starts.len() + 1);
    if first_start > 0 {
        blocks.push((0, first_start, true));
    }

    let mut block_starts = Vec::with_capacity(decl_starts.len());
    let mut min_bound = first_start;
    for (k, &d_idx) in decl_starts.iter().enumerate() {
        let s = if k == 0 {
            first_start
        } else {
            attach_leading_meta(lines, d_idx, min_bound)
        };
        block_starts.push(s);
        min_bound = d_idx + 1;
    }

    for k in 0..block_starts.len() {
        let start = block_starts[k];
        let end = block_starts.get(k + 1).copied().unwrap_or(lines.len());
        let hot = is_hot(&lines[start..end], start, end, diag_lines);
        blocks.push((start, end, hot));
    }
    blocks
}

fn is_hot(slice: &[&str], start: usize, end: usize, diag_lines: &HashSet<usize>) -> bool {
    if diag_lines
        .iter()
        .any(|&dl| dl > start.saturating_sub(25) && dl <= end + 25)
    {
        return true;
    }

    slice.iter().any(|l| {
        let indent = l.len().saturating_sub(l.trim_start().len());
        indent >= 12
            || l.chars().count() > 88
            || l.contains("=[]")
            || l.contains("={}")
            || l.contains("None")
            || l.trim().starts_with("elif ")
            || l.contains("TODO")
            || l.contains("unwrap()")
    })
}

pub fn deps_context(unit: &ModuleUnit, all_units: &[ModuleUnit], repo_root: &Path) -> String {
    if unit.deps.is_empty() {
        return String::new();
    }

    let by_id: HashMap<&str, &ModuleUnit> = all_units.iter().map(|u| (u.id.as_str(), u)).collect();
    let mut fan_in_by_file: HashMap<&str, usize> = HashMap::new();
    let mut dep_units = Vec::new();
    for id in &unit.deps {
        if let Some(&u) = by_id.get(id.as_str()) {
            for rel in &u.files {
                fan_in_by_file.insert(rel.as_str(), u.dependents.len());
            }
            dep_units.push(u.clone());
        }
    }
    abi_summary_enriched(
        "UPSTREAM DEPENDENCY ABI (read-only imports used by this unit):",
        &dep_units,
        repo_root,
        2400,
        Some(&fan_in_by_file),
    )
}

#[cfg(test)]
pub fn wave_context(wave_units: &[ModuleUnit], repo_root: &Path) -> String {
    abi_summary_enriched(
        "WAVE SHARED ABI PREFIX (read-only across wave siblings):",
        wave_units,
        repo_root,
        1200,
        None,
    )
}

fn abi_summary_enriched(
    header: &str,
    units: &[ModuleUnit],
    repo_root: &Path,
    max_chars: usize,
    fan_in_by_file: Option<&HashMap<&str, usize>>,
) -> String {
    let rels: Vec<&str> = units
        .iter()
        .flat_map(|u| u.files.iter().map(String::as_str))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let extract = |rel: &&str| {
        let src = fs::read_to_string(repo_root.join(rel)).ok()?;
        let sigs = file_heads(&src);
        if let Some(fi_map) = fan_in_by_file {
            let fi = fi_map.get(*rel).copied().unwrap_or(0);
            let is_new = crate::stem::genome::is_recent_48h(repo_root, rel);
            let summary = crate::stem::genome::synthetic_summary(rel, &src, &sigs, fi, is_new, None);
            return Some(format!("- {summary}"));
        }
        (!sigs.is_empty()).then(|| format!("- {rel}: {}", sigs.join(" | ")))
    };
    let lines: Vec<String> = if crate::platform::budget().is_constrained() {
        rels.iter().filter_map(extract).collect()
    } else {
        rels.par_iter().filter_map(extract).collect()
    };
    if lines.is_empty() {
        return String::new();
    }

    let mut out = format!("{header}\n");
    for line in lines {
        if out.chars().count() + line.chars().count() + 1 > max_chars {
            break;
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn file_heads(src: &str) -> Vec<String> {
    let mut sigs = Vec::new();
    for l in src.lines() {
        let t = l.trim();
        let wanted = top_decl(l)
            || t.starts_with("pub struct ")
            || t.starts_with("pub enum ")
            || t.starts_with("export interface ")
            || t.starts_with("export type ");
        if wanted && first_ident(t).is_some() {
            let before = t.split('{').next().unwrap_or(t);
            let head = before.trim();
            sigs.push(char_prefix(head, 96).to_string());
            if sigs.len() >= 6 {
                break;
            }
        }
    }
    sigs
}
