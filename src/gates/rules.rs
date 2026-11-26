use crate::lex::has_word;
use crate::platform::{find_secret, parse_go_block_lines};
use crate::source::ext_of;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
struct GuardRule {
    id: String, languages: Vec<String>, paths: Vec<String>, path_matcher: Option<GlobSet>,
    forbid: Option<String>, require: Option<String>, forbid_import: Option<String>, message: String,
}

fn strip_val(rest: &str) -> String { rest.trim().trim_matches(['"', '\'']).to_string() }

fn parse_list(rest: &str) -> Vec<String> {
    rest.trim()
        .trim_matches(['[', ']'])
        .split(',')
        .map(strip_val)
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_rules(text: &str) -> Vec<GuardRule> {
    let (mut rules, mut cur): (Vec<GuardRule>, Option<GuardRule>) = (Vec::new(), None);
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line == "rules:" { continue; }
        if let Some(rest) = line.strip_prefix("- id:") {
            if let Some(prev) = cur.take() { rules.push(prev); }
            cur = Some(GuardRule {
                id: strip_val(rest), message: "architectural invariant violated".into(),
                ..Default::default()
            });
            continue;
        }
        if let Some(ref mut rule) = cur { apply_field(rule, line); }
    }
    if let Some(last) = cur { rules.push(last); }
    for rule in &mut rules { rule.path_matcher = compile_paths(&rule.paths); }
    rules
}

fn apply_field(rule: &mut GuardRule, line: &str) {
    let Some((key, rest)) = line.split_once(':') else { return };
    match key.trim() {
        "languages" => {
            rule.languages = parse_list(rest)
                .into_iter()
                .map(|l| l.to_ascii_lowercase())
                .collect();
        }
        "paths" => rule.paths = parse_list(rest),
        "forbid_import" => rule.forbid_import = Some(strip_val(rest)),
        "forbid" => rule.forbid = Some(strip_val(rest)),
        "require" => rule.require = Some(strip_val(rest)),
        "message" => rule.message = strip_val(rest),
        _ => {}
    }
}

fn is_comment(trimmed: &str) -> bool {
    trimmed.is_empty() || trimmed == "*" || ["//", "#", "/*", "* ", "--"].iter().any(|p| trimmed.starts_with(p))
}

fn is_import(trimmed: &str) -> bool {
    ["import ", "from ", "use ", "pub use ", "pub(crate) use "].iter().any(|p| trimmed.starts_with(p))
        || trimmed.contains("require(") || (trimmed.starts_with("export ") && trimmed.contains(" from "))
}

fn add_glob(builder: &mut GlobSetBuilder, pat: &str) {
    if let Ok(glob) = GlobBuilder::new(pat).literal_separator(true).build() { builder.add(glob); }
}

fn compile_paths(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() { return None; }
    let mut builder = GlobSetBuilder::new();
    for raw in patterns {
        let norm = raw.replace('\\', "/");
        if norm.ends_with('/') { add_glob(&mut builder, &format!("{norm}**")); }
        else if norm.contains('*') || norm.contains('?') { add_glob(&mut builder, &norm); }
        else { add_glob(&mut builder, &norm); add_glob(&mut builder, &format!("{norm}/**")); }
    }
    builder.build().ok()
}

fn match_globs(rel: &str, matcher: Option<&GlobSet>) -> bool {
    matcher.is_none_or(|set| set.is_match(rel.replace('\\', "/")))
}

fn scan_secrets(rel: &str, content: &str, out: &mut Vec<String>) {
    for (idx, line) in content.lines().enumerate() {
        let Some(kind) = find_secret(line) else { continue };
        out.push(format!(
            "Gate 3.5 invariant [builtin_no_secrets:{kind}] `{rel}` L{}: hardcoded secret literal forbidden",
            idx + 1
        ));
    }
}

pub fn check_rules(repo_root: &Path, candidates: &BTreeMap<String, String>) -> Vec<String> {
    let mut out = Vec::new();
    for (rel, content) in candidates {
        scan_secrets(rel, content, &mut out);
    }
    let Ok(yml) = fs::read_to_string(repo_root.join(".refit/invariants.yml")) else { return out };
    let rules = parse_rules(&yml);
    for (rel, content) in candidates {
        let ext = ext_of(rel);
        for rule in &rules { check_rule(rel, &ext, content, rule, &mut out); }
    }
    out
}

fn check_forbid(rel: &str, content: &str, pat: &str, rule: &GuardRule, out: &mut Vec<String>) {
    for (idx, line) in content.lines().enumerate() {
        let t = line.trim();
        if is_comment(t) || !t.contains(pat) { continue; }
        out.push(format!("Gate 3.5 invariant [{}] `{rel}` L{}: {}", rule.id, idx + 1, rule.message));
    }
}

fn check_rule(rel: &str, ext: &str, content: &str, rule: &GuardRule, out: &mut Vec<String>) {
    if (!rule.languages.is_empty() && !rule.languages.iter().any(|l| l == ext))
        || !match_globs(rel, rule.path_matcher.as_ref())
    {
        return;
    }
    if let Some(pat) = rule.forbid.as_deref().filter(|s| !s.is_empty()) {
        check_forbid(rel, content, pat, rule, out);
    }
    if let Some(imp) = rule.forbid_import.as_deref().filter(|s| !s.is_empty()) {
        check_imports(rel, ext, content, imp, rule, out);
    }
    if let Some(req) = rule.require.as_deref().filter(|s| !s.is_empty()) && !content.contains(req) {
        out.push(format!(
            "Gate 3.5 invariant [{}] `{rel}` L1: missing required pattern `{req}` ({})",
            rule.id, rule.message
        ));
    }
}

fn line_number_at(content: &str, slice: &str) -> usize {
    let off = content.substr_range(slice).map_or(0, |r| r.start);
    let newlines = content[..off]
        .bytes()
        .filter(|&b| b == b'\n')
        .count();
    newlines + 1
}

fn check_imports(rel: &str, ext: &str, content: &str, imp: &str, rule: &GuardRule, out: &mut Vec<String>) {
    let hits: Vec<usize> = if ext == "go" {
        parse_go_block_lines(content, "import").into_iter()
            .filter(|l| !is_comment(l) && has_word(l, imp)).map(|l| line_number_at(content, l)).collect()
    } else {
        content.lines().enumerate()
            .filter(|(_, l)| { let t = l.trim(); !is_comment(t) && is_import(t) && has_word(t, imp) })
            .map(|(i, _)| i + 1).collect()
    };
    for ln in hits {
        out.push(format!(
            "Gate 3.5 import-boundary [{}] `{rel}` L{ln}: forbidden import `{imp}` ({})",
            rule.id, rule.message
        ));
    }
}
