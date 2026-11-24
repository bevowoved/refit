use crate::lex::has_word;
use crate::platform::find_secret;
use crate::source::ext_of;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
struct GuardRule {
    id: String,
    languages: Vec<String>,
    paths: Vec<String>,
    forbid: Option<String>,
    require: Option<String>,
    forbid_import: Option<String>,
    message: String,
}

fn parse_list(rest: &str) -> Vec<String> {
    rest.trim()
        .trim_matches(['[', ']'])
        .split(',')
        .map(|s| s.trim().trim_matches(['"', '\'']).to_string())
        .filter(|s| !s.is_empty())
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
        if let Some(ref mut r) = cur {
            apply_field(r, line);
        }
    }

    if let Some(last) = cur {
        rules.push(last);
    }
    rules
}

fn apply_field(r: &mut GuardRule, line: &str) {
    if let Some(rest) = line.strip_prefix("languages:") {
        let items = parse_list(rest);
        r.languages = items.into_iter().map(|s| s.to_ascii_lowercase()).collect();
    } else if let Some(rest) = line.strip_prefix("paths:") {
        r.paths = parse_list(rest);
    } else if let Some(rest) = line.strip_prefix("forbid_import:") {
        r.forbid_import = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("forbid:") {
        r.forbid = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("require:") {
        r.require = Some(strip_val(rest));
    } else if let Some(rest) = line.strip_prefix("message:") {
        r.message = strip_val(rest);
    }
}

fn is_comment(t: &str) -> bool {
    t.is_empty()
        || t.starts_with("//")
        || t.starts_with('#')
        || t.starts_with("/*")
        || t == "*"
        || t.starts_with("* ")
        || t.starts_with("--")
}

fn is_import(t: &str, in_go: bool) -> bool {
    in_go
        || t.starts_with("import ")
        || t.starts_with("from ")
        || t.starts_with("use ")
        || t.starts_with("pub use ")
        || t.starts_with("pub(crate) use ")
        || t.contains("require(")
        || (t.starts_with("export ") && t.contains(" from "))
}

fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern.ends_with('/') {
        return text.starts_with(pattern);
    }
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return text == prefix || text.starts_with(&format!("{prefix}/"));
    }

    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star_p, mut star_t) = (0, 0, None, 0);

    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            let dbl = pi + 1 < p.len() && p[pi + 1] == '*';
            pi += if dbl { 2 } else { 1 };
            star_p = Some((pi, dbl));
            star_t = ti;
        } else if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if let Some((sp, dbl)) = star_p {
            if !dbl && t[star_t] == '/' {
                return false;
            }
            star_t += 1;
            ti = star_t;
            pi = sp;
        } else {
            return false;
        }
    }

    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn match_globs(rel: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let norm = rel.replace('\\', "/");
    patterns.iter().any(|p| {
        let pn = p.replace('\\', "/");
        if pn.contains('*') || pn.contains('?') {
            glob_match(&pn, &norm)
        } else if pn.ends_with('/') {
            norm.starts_with(&pn)
        } else {
            norm == pn || norm.starts_with(&format!("{pn}/"))
        }
    })
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
    if !rule.languages.is_empty() && !rule.languages.iter().any(|l| l == ext) {
        return;
    }
    if !match_globs(rel, &rule.paths) {
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
