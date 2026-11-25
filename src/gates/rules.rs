use crate::lex::has_word;
use crate::platform::{find_secret, parse_go_block_lines};
use crate::source::ext_of;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
struct GuardRule {
    id: String,
    languages: Vec<String>,
    paths: Vec<String>,
    path_matcher: Option<GlobSet>,
    forbid: Option<String>,
    require: Option<String>,
    forbid_import: Option<String>,
    message: String,
}

fn parse_list(rest: &str) -> Vec<String> {
    rest.trim()
        .trim_matches(['[', ']'])
        .split(',')
        .map(|item| item.trim().trim_matches(['"', '\'']).to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

fn strip_val(rest: &str) -> String {
    rest.trim().trim_matches(['"', '\'']).to_string()
}

fn parse_rules(text: &str) -> Vec<GuardRule> {
    let mut rules = Vec::new();
    let mut cur: Option<GuardRule> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line == "rules:" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("- id:") {
            if let Some(prev) = cur.take() {
                rules.push(prev);
            }
            cur = Some(GuardRule {
                id: strip_val(rest),
                message: "architectural invariant violated".into(),
                ..GuardRule::default()
            });
            continue;
        }
        if let Some(ref mut rule) = cur {
            apply_field(rule, line);
        }
    }

    if let Some(last) = cur {
        rules.push(last);
    }
    for rule in &mut rules {
        rule.path_matcher = compile_paths(&rule.paths);
    }
    rules
}

fn apply_field(rule: &mut GuardRule, line: &str) {
    let Some((key, rest)) = line.split_once(':') else {
        return;
    };
    match key.trim() {
        "languages" => {
            rule.languages = parse_list(rest)
                .into_iter()
                .map(|lang| lang.to_ascii_lowercase())
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
    trimmed.is_empty()
        || trimmed == "*"
        || ["//", "#", "/*", "* ", "--"]
            .iter()
            .any(|p| trimmed.starts_with(p))
}

fn is_import(trimmed: &str) -> bool {
    ["import ", "from ", "use ", "pub use ", "pub(crate) use "]
        .iter()
        .any(|p| trimmed.starts_with(p))
        || trimmed.contains("require(")
        || (trimmed.starts_with("export ") && trimmed.contains(" from "))
}

fn add_glob(builder: &mut GlobSetBuilder, pat: &str) {
    if let Ok(glob) = GlobBuilder::new(pat).literal_separator(true).build() {
        builder.add(glob);
    }
}

fn compile_paths(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }
    let mut builder = GlobSetBuilder::new();
    for raw in patterns {
        let norm = raw.replace('\\', "/");
        if norm.ends_with('/') {
            add_glob(&mut builder, &format!("{norm}**"));
        } else if norm.contains('*') || norm.contains('?') {
            add_glob(&mut builder, &norm);
        } else {
            add_glob(&mut builder, &norm);
            add_glob(&mut builder, &format!("{norm}/**"));
        }
    }
    builder.build().ok()
}

fn match_globs(rel: &str, matcher: Option<&GlobSet>) -> bool {
    let Some(set) = matcher else { return true };
    let norm = rel.replace('\\', "/");
    set.is_match(norm)
}

pub fn check_rules(repo_root: &Path, candidates: &BTreeMap<String, String>) -> Vec<String> {
    let mut out = Vec::new();
    for (rel, content) in candidates {
        for (idx, line) in content.lines().enumerate() {
            if let Some(kind) = find_secret(line) {
                let ln = idx + 1;
                out.push(format!(
                    "Gate 3.5 invariant [builtin_no_secrets:{kind}] `{rel}` L{ln}: \
                     hardcoded secret literal forbidden"
                ));
            }
        }
    }

    let Ok(yml) = fs::read_to_string(repo_root.join(".refit/invariants.yml")) else {
        return out;
    };
    let rules = parse_rules(&yml);

    for (rel, content) in candidates {
        let ext = ext_of(rel);
        for rule in &rules {
            check_rule(rel, &ext, content, rule, &mut out);
        }
    }
    out
}

fn check_rule(rel: &str, ext: &str, content: &str, rule: &GuardRule, out: &mut Vec<String>) {
    if !rule.languages.is_empty() && !rule.languages.iter().any(|lang| lang == ext) {
        return;
    }
    if !match_globs(rel, rule.path_matcher.as_ref()) {
        return;
    }

    if let Some(pat) = rule.forbid.as_deref().filter(|s| !s.is_empty()) {
        for (idx, line) in content.lines().enumerate() {
            let t = line.trim();
            if !is_comment(t) && t.contains(pat) {
                let ln = idx + 1;
                out.push(format!(
                    "Gate 3.5 invariant [{}] `{rel}` L{ln}: {}",
                    rule.id, rule.message
                ));
            }
        }
    }

    if let Some(imp) = rule.forbid_import.as_deref().filter(|s| !s.is_empty()) {
        check_imports(rel, ext, content, imp, rule, out);
    }
    if let Some(req) = rule.require.as_deref().filter(|s| !s.is_empty())
        && !content.contains(req)
    {
        out.push(format!(
            "Gate 3.5 invariant [{}] `{rel}` L1: missing required pattern `{req}` ({})",
            rule.id, rule.message
        ));
    }
}

fn line_number_at(content: &str, slice: &str) -> usize {
    let off = slice.as_ptr() as usize - content.as_ptr() as usize;
    let prefix = &content[..off];
    prefix.bytes().filter(|&b| b == b'\n').count() + 1
}

fn check_imports(
    rel: &str,
    ext: &str,
    content: &str,
    imp: &str,
    rule: &GuardRule,
    out: &mut Vec<String>,
) {
    let mut hits = Vec::new();
    if ext == "go" {
        for line in parse_go_block_lines(content, "import") {
            if !is_comment(line) && has_word(line, imp) {
                hits.push(line_number_at(content, line));
            }
        }
    } else {
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if !is_comment(trimmed) && is_import(trimmed) && has_word(trimmed, imp) {
                hits.push(idx + 1);
            }
        }
    }
    for ln in hits {
        out.push(format!(
            "Gate 3.5 import-boundary [{}] `{rel}` L{ln}: forbidden import `{imp}` ({})",
            rule.id, rule.message
        ));
    }
}
