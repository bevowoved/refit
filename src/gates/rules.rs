use crate::lex::has_word;
use crate::platform::find_secret;
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
    if let Some(rest) = line.strip_prefix("languages:") {
        let items = parse_list(rest);
        rule.languages = items.into_iter().map(|lang| lang.to_ascii_lowercase()).collect();
    } else if let Some(rest) = line.strip_prefix("paths:") {
        rule.paths = parse_list(rest);
    } else if let Some(rest) = line.strip_prefix("forbid_import:") {
        rule.forbid_import = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("forbid:") {
        rule.forbid = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("require:") {
        rule.require = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("message:") {
        rule.message = strip_val(rest);
    }
}

fn is_comment(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with('#')
        || trimmed.starts_with("/*")
        || trimmed == "*"
        || trimmed.starts_with("* ")
        || trimmed.starts_with("--")
}

fn is_import(trimmed: &str, in_go: bool) -> bool {
    in_go
        || trimmed.starts_with("import ")
        || trimmed.starts_with("from ")
        || trimmed.starts_with("use ")
        || trimmed.starts_with("pub use ")
        || trimmed.starts_with("pub(crate) use ")
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
                out.push(format!("Gate 3.5 invariant [{}] `{rel}` L{ln}: {}", rule.id, rule.message));
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

fn check_imports(
    rel: &str,
    ext: &str,
    content: &str,
    imp: &str,
    rule: &GuardRule,
    out: &mut Vec<String>,
) {
    let mut in_go = false;
    for (idx, line) in content.lines().enumerate() {
        let t = line.trim();
        if ext == "go" && t.starts_with("import (") {
            in_go = true;
            continue;
        }
        if in_go && t.starts_with(')') {
            in_go = false;
            continue;
        }
        if !is_comment(t) && is_import(t, in_go) && has_word(t, imp) {
            let ln = idx + 1;
            out.push(format!(
                "Gate 3.5 import-boundary [{}] `{rel}` L{ln}: forbidden import `{imp}` ({})",
                rule.id, rule.message
            ));
        }
    }
}
