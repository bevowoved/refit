use crate::lex::first_ident;
use crate::platform::{char_prefix, par_map};
use crate::types::ModuleUnit;
use anyhow::{Context, Result};
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

static DIAG_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([Ll][Ii][Nn][Ee] |[:L])([0-9]+)").expect("valid diag regex"));
static FAIL_TEST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?:::(test_[A-Za-z0-9_]*|probe_[A-Za-z0-9_]*)|",
        r"FAIL: (Test[A-Za-z0-9_]*)|test (?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]{4,}))",
    ))
    .expect("valid fail test regex")
});

const DECL_STARTS: &[&str] = &[
    "def ", "async def ", "class ", "pub fn ", "pub async fn ",
    "fn ", "export ", "function ", "func ",
];
const EXTRA_HEAD_STARTS: &[&str] = &[
    "pub struct ", "pub enum ", "export interface ", "export type ",
];
const DOC_PREFIXES: &[&str] = &["///", "//!", "/**", "* ", "\"\"\"", "# "];
const RECENT_WINDOW_SECS: u64 = 48 * 3600;

fn top_decl(line: &str) -> bool {
    !line.starts_with([' ', '\t']) && DECL_STARTS.iter().any(|p| line.starts_with(p))
}

fn diag_line_ok(diag: &str, bytes: &[u8], start: usize, end: usize, pfx: &str) -> bool {
    let end_ok = end == bytes.len() || !bytes[end].is_ascii_alphabetic();
    match pfx {
        "L" => end_ok && (start == 0 || !bytes[start - 1].is_ascii_alphanumeric()),
        ":" => end_ok && !["http", "https", "127.0.0.1", "localhost"]
            .iter().any(|p| diag[..start].ends_with(p)),
        _ => true,
    }
}

fn parse_diags(diagnostics: &[String]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for diag in diagnostics {
        let bytes = diag.as_bytes();
        for cap in DIAG_LINE_RE.captures_iter(diag) {
            let (Some(mat), pfx, digits) = (cap.get(0), &cap[1], &cap[2]) else { continue };
            if diag_line_ok(diag, bytes, mat.start(), mat.end(), pfx)
                && let Ok(num) = digits.parse::<usize>()
                && (1..100_000).contains(&num)
            {
                out.insert(num);
            }
        }
    }
    out
}

fn find_failed_names(diagnostics: &[String]) -> Vec<&str> {
    let mut names = Vec::new();
    for diag in diagnostics {
        for cap in FAIL_TEST_RE.captures_iter(diag) {
            let mat = cap.get(1).or_else(|| cap.get(2)).or_else(|| cap.get(3));
            let name = mat.map_or("", |m| m.as_str());
            if name.len() >= 4 && !names.contains(&name) { names.push(name); }
        }
    }
    names
}

fn failed_test(test_src: &str, diagnostics: &[String], max_chars: usize) -> String {
    let names = find_failed_names(diagnostics);
    if names.is_empty() { return char_prefix(test_src, max_chars).to_string(); }

    let src_lines: Vec<&str> = test_src.lines().collect();
    let mut out = String::new();
    for fname in &names {
        let needles = [
            format!("def {fname}"), format!("fn {fname}"), format!("func {fname}"),
            format!("'{fname}'"), format!("\"{fname}\""),
        ];
        if let Some(start) = src_lines.iter().position(|l| needles.iter().any(|nd| l.contains(nd.as_str()))) {
            out.push_str(&src_lines[start..(start + 45).min(src_lines.len())].join("\n"));
            out.push('\n');
            if out.chars().count() >= max_chars { break; }
        }
    }
    char_prefix(if out.is_empty() { test_src } else { &out }, max_chars).to_string()
}

fn append_failing_tests(
    out: &mut String, repo_root: &Path, unit: &ModuleUnit, diagnostics: &[String],
) {
    for tf in unit.test_files.iter().take(2) {
        if let Ok(tc) = fs::read_to_string(repo_root.join(tf)) {
            let snip = failed_test(&tc, diagnostics, 2400);
            let _ = write!(out, "\nRelevant failing test `{tf}`:\n```\n{snip}\n```\n");
        }
    }
}

pub fn build_packet(
    repo_root: &Path, unit: &ModuleUnit, diagnostics: &[String], learn_suffix: &str, max_chars: usize,
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
        for diag in unique.iter().take(36) { let _ = writeln!(out, "- {diag}"); }
    }
    if unit.files.len() > 1 {
        out.push_str(concat!(
            "IMPORTANT: This cyclic SCC unit has multiple files. ",
            "Emit <<<REFIT_PATCH>>> blocks to fix ALL defects in EVERY file below.\n"
        ));
    }

    let test_fail = unique.iter().any(|d| {
        d.to_lowercase().contains("pytest")
            || ["FAILED", "assert", "Gate 1"].iter().any(|k| d.contains(k))
    });
    if test_fail { append_failing_tests(&mut out, repo_root, unit, diagnostics); }

    let diag_lines = parse_diags(diagnostics);
    let per_file = max_chars / unit.files.len().max(1);
    for rel in &unit.files {
        let content = fs::read_to_string(repo_root.join(rel)).with_context(|| format!("read {rel}"))?;
        let sliced = slice_anchored(&content, per_file, &diag_lines);
        let _ = write!(out, "\nFile `{rel}`:\n```\n{sliced}\n```\n");
    }

    if !learn_suffix.is_empty() { out.push_str(learn_suffix); }
    out.push_str(concat!(
        "\nEmit ONLY raw <<<REFIT_PATCH path=\"...\">>> blocks directly in your reply. ",
        "Every <<<SEARCH>>> block must match uniquely. ",
        "Preserve exported signatures unless allow_abi_change is enabled.\n"
    ));
    Ok(out)
}

fn push_decl_stub(out: &mut String, slice: &[&str]) -> usize {
    let Some(first) = slice.iter().find(|l| top_decl(l)) else { return 0 };
    let stub = format!("// stub: {}\n", char_prefix(first.trim(), 120));
    out.push_str(&stub);
    stub.chars().count()
}

fn push_omitted_marker(out: &mut String, omitted: usize) -> usize {
    let marker = format!("\n// ... ({omitted} clean lines omitted) ...\n");
    out.push_str(&marker);
    marker.chars().count()
}

pub(crate) fn slice_anchored(content: &str, budget_chars: usize, diag_lines: &HashSet<usize>) -> String {
    if content.chars().count() <= budget_chars { return content.to_string(); }

    let lines: Vec<&str> = content.lines().collect();
    let (mut out, mut chars, mut omitted) = (String::new(), 0usize, 0usize);
    for (start, end, hot) in split_blocks(&lines, diag_lines) {
        let chunk = lines[start..end].join("\n");
        let chunk_c = chunk.chars().count();
        if !hot && chars + chunk_c > budget_chars {
            chars += push_decl_stub(&mut out, &lines[start..end]);
            omitted += end.saturating_sub(start);
            continue;
        }
        if omitted > 0 {
            chars += push_omitted_marker(&mut out, omitted);
            omitted = 0;
        }
        out.push_str(&chunk);
        out.push('\n');
        chars += chunk_c + 1;
    }
    out
}

fn is_leading_meta(line: &str) -> bool {
    const META_PREFIXES: &[&str] = &["#[", "@", "///", "/**", "* ", "*/", "//"];
    let trimmed = line.trim();
    !line.starts_with([' ', '\t']) && META_PREFIXES.iter().any(|p| trimmed.starts_with(p))
}

fn attach_leading_meta(lines: &[&str], decl_idx: usize, min_idx: usize) -> usize {
    let mut start = decl_idx;
    while start > min_idx && !lines[start - 1].trim().is_empty() && is_leading_meta(lines[start - 1]) {
        start -= 1;
    }
    start
}

fn split_blocks(lines: &[&str], diag_lines: &HashSet<usize>) -> Vec<(usize, usize, bool)> {
    let decls: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| top_decl(l).then_some(i))
        .collect();
    if decls.is_empty() { return vec![(0, lines.len(), true)]; }

    let first_start = attach_leading_meta(lines, decls[0], 0);
    let mut blocks = Vec::with_capacity(decls.len() + 1);
    if first_start > 0 { blocks.push((0, first_start, true)); }

    let (mut starts, mut min_bound) = (Vec::with_capacity(decls.len()), first_start);
    for (k, &d_idx) in decls.iter().enumerate() {
        starts.push(if k == 0 { first_start } else { attach_leading_meta(lines, d_idx, min_bound) });
        min_bound = d_idx + 1;
    }
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(lines.len());
        blocks.push((start, end, is_hot(&lines[start..end], start, end, diag_lines)));
    }
    blocks
}

fn is_hot(slice: &[&str], start: usize, end: usize, diag_lines: &HashSet<usize>) -> bool {
    diag_lines.iter().any(|&dl| dl > start.saturating_sub(25) && dl <= end + 25)
        || slice.iter().any(|l| {
            l.len().saturating_sub(l.trim_start().len()) >= 12
                || l.chars().count() > 88
                || l.trim().starts_with("elif ")
                || ["=[]", "={}", "None", "TODO", "unwrap()"].iter().any(|p| l.contains(p))
        })
}

pub fn deps_context(unit: &ModuleUnit, all_units: &[ModuleUnit], repo_root: &Path) -> String {
    if unit.deps.is_empty() { return String::new(); }
    let by_id: HashMap<&str, &ModuleUnit> = all_units.iter().map(|u| (u.id.as_str(), u)).collect();
    let (mut fan_in, mut dep_units) = (HashMap::new(), Vec::new());
    for id in &unit.deps {
        if let Some(&dep) = by_id.get(id.as_str()) {
            for rel in &dep.files { fan_in.insert(rel.as_str(), dep.dependents.len()); }
            dep_units.push(dep.clone());
        }
    }
    abi_summary_enriched(
        "UPSTREAM DEPENDENCY ABI (read-only imports used by this unit):",
        &dep_units, repo_root, 2400, Some(&fan_in),
    )
}

#[cfg(test)]
pub fn wave_context(wave_units: &[ModuleUnit], repo_root: &Path) -> String {
    abi_summary_enriched(
        "WAVE SHARED ABI PREFIX (read-only across wave siblings):",
        wave_units, repo_root, 1200, None,
    )
}

fn abi_summary_enriched(
    header: &str, units: &[ModuleUnit], repo_root: &Path, max_chars: usize,
    fan_in_by_file: Option<&HashMap<&str, usize>>,
) -> String {
    let rels: Vec<&str> = units.iter()
        .flat_map(|u| u.files.iter().map(String::as_str))
        .collect::<BTreeSet<_>>().into_iter().collect();

    let mapped = par_map(&rels, |rel| {
        let src = fs::read_to_string(repo_root.join(rel)).ok()?;
        let sigs = file_heads(&src);
        if let Some(fi_map) = fan_in_by_file {
            let fi = fi_map.get(*rel).copied().unwrap_or(0);
            let sum = synthetic_summary(rel, &src, &sigs, fi, is_recent_48h(repo_root, rel), None);
            return Some(format!("- {sum}"));
        }
        (!sigs.is_empty()).then(|| format!("- {rel}: {}", sigs.join(" | ")))
    });
    let lines: Vec<String> = mapped.into_iter().flatten().collect();
    if lines.is_empty() { return String::new(); }

    let mut out = format!("{header}\n");
    for line in lines {
        if out.chars().count() + line.chars().count() + 1 > max_chars { break; }
        let _ = writeln!(out, "{line}");
    }
    out
}

fn file_heads(src: &str) -> Vec<String> {
    let mut sigs = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        let wanted = top_decl(line) || EXTRA_HEAD_STARTS.iter().any(|p| trimmed.starts_with(p));
        if wanted && first_ident(trimmed).is_some() {
            let before_brace = trimmed.split('{').next().unwrap_or(trimmed);
            let head = before_brace.trim();
            sigs.push(char_prefix(head, 96).to_string());
            if sigs.len() >= 6 { break; }
        }
    }
    sigs
}

fn is_recent_48h(repo_root: &Path, rel: &str) -> bool {
    let Ok(mtime) = fs::metadata(repo_root.join(rel)).and_then(|m| m.modified()) else { return false };
    std::time::SystemTime::now().duration_since(mtime).map_or(true, |age| age.as_secs() <= RECENT_WINDOW_SECS)
}

fn first_doc_sentence(src: &str) -> Option<String> {
    for line in src.lines().take(20) {
        let trimmed = line.trim();
        let Some(raw) = DOC_PREFIXES.iter().find_map(|p| trimmed.strip_prefix(p)) else { continue };
        let stripped = raw.trim().trim_end_matches("*/").trim_end_matches("\"\"\"");
        let clean = stripped.trim();
        if clean.len() >= 8 {
            let first = clean.split(". ").next().unwrap_or(clean);
            return Some(first.trim().to_string());
        }
    }
    None
}

fn synthetic_summary(
    rel: &str, src: &str, exports: &[String], fan_in: usize, is_new: bool, top_twin: Option<(&str, f64)>,
) -> String {
    let mut parts = Vec::with_capacity(4);
    if is_new { parts.push("[NEW]".to_string()); }
    if fan_in > 0 { parts.push(format!("(->{fan_in})")); }
    if !exports.is_empty() {
        let top: Vec<&str> = exports
            .iter()
            .take(5)
            .map(String::as_str)
            .collect();
        parts.push(format!("exports:[{}]", top.join(", ")));
    }
    if let Some(doc) = first_doc_sentence(src) { parts.push(char_prefix(&doc, 72).to_string()); }
    if let Some((twin, npmi)) = top_twin { parts.push(format!("twin:{twin}({npmi:.2})")); }
    if parts.is_empty() { format!("{rel} (->0)") } else { format!("{rel} {}", parts.join(" | ")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_summary_badges() {
        let s = synthetic_summary(
            "src/auth.rs", "/// Validates JWT bearer tokens.\npub fn verify() {}\n",
            &["verify".into()], 7, true, Some(("src/session.rs", 0.86)),
        );
        assert!(s.contains("[NEW]") && s.contains("(->7)") && s.contains("twin:src/session.rs(0.86)"));

        let diags = vec![
            "src/a.rs:42:7 error at Line 19 near L88 http://127.0.0.1:8080".into(),
            "FAILED tests/foo.py::test_alpha --- FAIL: TestBeta --- test mod_x::gamma_ok".into(),
        ];
        let lines = parse_diags(&diags);
        assert!(lines.contains(&42) && lines.contains(&7) && lines.contains(&19) && lines.contains(&88));
        assert!(!lines.contains(&8080));
        let snip = failed_test("def test_alpha():\n    assert False\n", &diags, 400);
        assert!(snip.contains("def test_alpha"));
    }
}
