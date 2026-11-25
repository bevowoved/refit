//! Structural prose linter and report compiler for simplified technical English.
//!
//! No dictionary ships with this crate; pass `--ste-dict <path>` at runtime to load one.

use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{Context, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

const MAX_PROC_WORDS: usize = 20;
const MAX_DESC_WORDS: usize = 25;
const MAX_NOUN_CLUSTER: usize = 3;

const LATIN_ABBREVS: &[&str] = &["e.g.", "i.e.", "etc.", "vs.", "cf."];
const CONTRACTIONS: &[&str] = &[
    "don't", "doesn't", "can't", "won't", "isn't", "aren't", "it's", "we're", "you're",
    "shouldn't", "couldn't", "haven't", "hasn't", "didn't", "wasn't", "weren't",
];
const VERB_SWAPS: &[(&str, &str)] = &[
    ("utilize", "use"), ("utilizes", "uses"), ("utilized", "used"),
    ("commence", "start"), ("commences", "starts"),
    ("terminate", "stop"), ("terminates", "stops"),
    ("perform", "do"), ("performs", "does"), ("performed", "did"),
    ("leverage", "use"), ("leverages", "uses"),
    ("facilitate", "help"), ("facilitates", "helps"),
    ("obtain", "get"), ("obtains", "gets"),
    ("instantiate", "create"), ("instantiates", "creates"),
    ("attempt", "try"), ("attempts", "tries"),
];

static VERB_AC: LazyLock<AhoCorasick> = LazyLock::new(|| {
    AhoCorasick::builder()
        .ascii_case_insensitive(true)
        .match_kind(MatchKind::LeftmostLongest)
        .build(VERB_SWAPS.iter().map(|(bad, _)| *bad))
        .expect("valid static verb swap patterns")
});

const PASSIVE_AUX: &[&str] = &["is", "are", "was", "were", "be", "been"];
const IRREG_PARTICIPLES: &[&str] = &[
    "written", "built", "sent", "drawn", "driven", "chosen", "given", "taken", "broken",
    "found", "held", "kept", "made", "paid", "read", "run", "set", "shown",
];
const BAD_OPENERS: &[&str] = &["you", "please", "it", "we", "developers", "users"];
const DETERMINERS: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "each", "every",
    "one", "my", "our", "its", "their", "your",
];
const NOUN_ING_WORDS: &[&str] = &["string", "thing", "spring", "ceiling", "morning", "warning"];
const NON_NOUN_WORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "into", "onto", "not",
    "are", "was", "were", "been", "has", "have", "had", "can", "will", "must", "may", "should",
    "would", "could", "when", "while", "then", "than", "only", "each", "every", "both", "all",
    "any", "some", "one", "two", "three", "four", "five", "zero", "less", "more", "much", "before",
    "after", "under", "over", "inside", "outside", "across", "without", "during", "near", "against",
    "between", "through", "until", "because", "since", "soon", "first", "now", "also", "still",
    "never", "always", "forever", "per", "via", "remove", "set", "start", "sort", "protect", "call",
    "write", "pass", "use", "detect", "replace", "cap", "keep", "push", "free", "flush", "normalize",
    "check", "drop", "run", "send", "read", "separate", "strip", "return", "provides", "removes",
    "returns", "refills", "finishes", "merges", "strips", "enables", "changes", "cross", "normalizes",
    "compares", "hashes", "groups", "improve", "uses", "cuts", "preserves", "resolves", "shares",
    "counts", "catches", "occur", "separates", "rejects", "fails", "catch", "handles", "passes",
    "requires", "binds", "runs", "starts", "locks", "corrupts", "configure", "update", "invoke",
    "rebuild", "split", "audit", "verify", "commit", "abort", "stage", "create", "delete", "move",
    "open", "close", "avoid", "review", "consolidate", "harden", "prune", "native", "exact", "raw",
    "fast", "tiny", "small", "large", "long", "short", "high", "low", "new", "old", "good", "bad",
    "clean", "final", "intermediate", "atomic", "safe", "unsafe", "single", "single-pass",
    "short-lived", "time-sortable", "idle", "transient", "anonymous", "sequential", "uninitialized",
    "silent", "trailing", "upstream", "duplicate", "direct", "opaque", "strict", "empty", "extra",
    "plain", "public", "private", "dirty", "cold", "warm", "hot", "hourly", "mandatory", "stateless",
    "deterministic", "bounded", "virtual", "external", "internal", "legacy", "local", "remote",
    "relative", "minimal", "actual", "other", "non-ascii", "headless", "active", "valid", "invalid",
    "dead", "unused", "circular", "deep", "shallow", "prioritized", "structural", "technical", "full",
    "total",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SentenceKind { Procedural, Descriptive }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintIssue { pub line: usize, pub rule: String, pub detail: String, pub sentence: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProseReport {
    pub total_sentences: usize,
    pub compliant_sentences: usize,
    pub violation_count: usize,
    pub compliant: bool,
    pub custom_dict_loaded: bool,
    pub custom_dict_entries: usize,
    pub issues: Vec<LintIssue>,
}

/// Optional user-supplied wordlist loaded via `--ste-dict <path>`.
#[derive(Debug, Clone, Default)]
pub struct WordList {
    approved: FxHashSet<String>,
    unapproved: FxHashMap<String, Option<String>>,
    technical_nouns: FxHashSet<String>,
}

impl WordList {
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read --ste-dict file {}", path.display()))?;
        Ok(Self::parse_str(&raw))
    }

    pub fn parse_str(raw: &str) -> Self {
        let mut dict = Self::default();
        for line in raw.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") { continue; }
            if let Some((left, right)) = line.split_once(':') {
                let word = left.trim().to_ascii_lowercase();
                if word.is_empty() { continue; }
                let (st, swap) = right.trim().split_once("->").map_or((right.trim(), None), |(s, r)| {
                    (s.trim(), Some(r.trim().to_string()).filter(|x| !x.is_empty()))
                });
                match st.to_ascii_lowercase().as_str() {
                    "technical" | "technical_noun" | "tn" => { dict.technical_nouns.insert(word); }
                    "unapproved" | "banned" | "forbid" | "forbidden" | "disallow" => {
                        dict.unapproved.insert(word, swap);
                    }
                    _ => { dict.approved.insert(word); }
                }
            } else if let Some(clean) = line.split_whitespace().next().map(str::to_ascii_lowercase)
                && !clean.is_empty()
            {
                dict.approved.insert(clean);
            }
        }
        dict
    }

    pub fn entry_count(&self) -> usize {
        self.approved.len() + self.unapproved.len() + self.technical_nouns.len()
    }

    fn is_tech_noun(&self, word: &str) -> bool { self.technical_nouns.contains(word) }
    fn is_approved(&self, word: &str) -> bool {
        self.approved.contains(word) || self.technical_nouns.contains(word)
    }
}

/// Classify whether a backtick span is a technical code token.
pub fn is_code_token(span: &str) -> bool {
    let trimmed = span.trim();
    if trimmed.is_empty() { return true; }
    let count = trimmed.split_whitespace().count();
    if count >= 5 { return false; }
    count == 1
        || trimmed.chars().any(|ch| {
            matches!(
                ch,
                ':' | '/' | '\\' | '_' | '-' | '=' | '(' | ')' | '[' | ']' | '.' | ';' | '$' | '%'
                    | '@' | '<' | '>' | '{' | '}' | '0'..='9'
            )
        })
}

/// Mask code spans as `CODETOK` and unwrap prose disguised inside backticks.
pub fn mask_code_spans(input: &str) -> (String, bool) {
    let (mut out, mut last, mut disguised) = (String::with_capacity(input.len()), 0usize, false);
    for (open, close) in crate::lex::iter_backtick_ranges(input) {
        out.push_str(&input[last..open]);
        let span = &input[open + 1..close];
        if is_code_token(span) {
            out.push_str(" CODETOK ");
        } else {
            disguised = true;
            let _ = write!(out, " {span} ");
        }
        last = close + 1;
    }
    let tail = &input[last..];
    if let Some((before, after)) = tail.split_once('`') {
        let _ = write!(out, "{before} {after}");
    } else {
        out.push_str(tail);
    }
    (out, disguised)
}

fn push_unique(errs: &mut Vec<&'static str>, rule: &'static str) {
    if !errs.contains(&rule) { errs.push(rule); }
}

/// Lint one sentence under structural prose rules and an optional wordlist.
pub fn lint_sentence(sentence: &str, kind: SentenceKind, dict: Option<&WordList>) -> Vec<&'static str> {
    let mut errs = Vec::new();
    let (masked, disguised_prose) = mask_code_spans(sentence);
    if disguised_prose { push_unique(&mut errs, "disguised_prose_in_code_span"); }
    if masked.contains(';') { push_unique(&mut errs, "semicolon_banned"); }

    let raw_words: Vec<&str> = masked.split_whitespace().collect();
    let limit = if kind == SentenceKind::Procedural { MAX_PROC_WORDS } else { MAX_DESC_WORDS };
    if raw_words.len() > limit { push_unique(&mut errs, "word_limit_exceeded"); }

    let lowered_clean = scan_lexical_rules(&raw_words, dict, &mut errs);
    check_voice(&lowered_clean, dict, &mut errs);
    check_noun_clusters(&raw_words, dict, &mut errs);
    if kind == SentenceKind::Procedural
        && lowered_clean.first().is_some_and(|first| BAD_OPENERS.contains(&first.as_str()))
    {
        push_unique(&mut errs, "non_imperative_procedural");
    }
    errs
}

fn scan_lexical_rules(
    raw_words: &[&str], dict: Option<&WordList>, errs: &mut Vec<&'static str>,
) -> Vec<String> {
    let mut cleaned = Vec::with_capacity(raw_words.len());
    for word in raw_words {
        let lower = word.to_ascii_lowercase();
        if LATIN_ABBREVS.contains(&lower.trim_matches([',', '(', ')', ':', '"'])) {
            push_unique(errs, "latin_abbrev");
        }
        let clean = lower.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'').to_string();
        if CONTRACTIONS.contains(&clean.as_str()) { push_unique(errs, "contraction"); }
        check_verb_and_dict(&clean, dict, errs);
        cleaned.push(clean);
    }
    cleaned
}

fn check_verb_and_dict(clean: &str, dict: Option<&WordList>, errs: &mut Vec<&'static str>) {
    if clean.is_empty() || clean == "codetok" { return; }
    if !dict.is_some_and(|d| d.is_approved(clean))
        && VERB_SWAPS.iter().any(|(b, _)| *b == clean)
    {
        push_unique(errs, "unapproved_verb");
    }
    let Some(user_dict) = dict else { return };
    if user_dict.unapproved.contains_key(clean) {
        push_unique(errs, "dict_unapproved_word");
    } else if !user_dict.approved.is_empty()
        && clean.len() >= 2
        && clean.bytes().all(|b| b.is_ascii_alphabetic())
        && !user_dict.is_approved(clean)
    {
        push_unique(errs, "dict_not_in_wordlist");
    }
}

fn check_voice(tokens: &[String], dict: Option<&WordList>, errs: &mut Vec<&'static str>) {
    for (aux_idx, word) in tokens.iter().enumerate() {
        if !PASSIVE_AUX.contains(&word.as_str()) { continue; }
        let end = (aux_idx + 3).min(tokens.len());
        let mut det_between = false;
        for (offset, tok) in tokens[(aux_idx + 1)..end].iter().enumerate() {
            let s = tok.as_str();
            if offset == 0 && DETERMINERS.contains(&s) {
                det_between = true;
                continue;
            }
            if det_between { continue; }
            if (s.len() > 4 && s.ends_with("ed")) || IRREG_PARTICIPLES.contains(&s) {
                push_unique(errs, "passive_voice");
                break;
            }
            if s.len() >= 5 && s.ends_with("ing")
                && !dict.is_some_and(|d| d.is_tech_noun(s)) && !NOUN_ING_WORDS.contains(&s)
            {
                push_unique(errs, "progressive_ing");
                break;
            }
        }
    }
}

fn check_noun_clusters(raw_words: &[&str], dict: Option<&WordList>, errs: &mut Vec<&'static str>) {
    let mut run = 0usize;
    for raw in raw_words {
        if raw.starts_with(['(', '[']) { run = 0; }
        if is_noun_word(raw, dict) {
            run += 1;
            if run > MAX_NOUN_CLUSTER {
                push_unique(errs, "noun_cluster_exceeded");
                return;
            }
        } else {
            run = 0;
        }
        if raw.ends_with([',', '.', ':', ';', '?', '!', ')']) { run = 0; }
    }
}

fn is_noun_word(raw: &str, dict: Option<&WordList>) -> bool {
    let trimmed = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
    if trimmed.len() <= 2
        || trimmed == "CODETOK"
        || trimmed.bytes().any(|b| b.is_ascii_digit())
        || trimmed.bytes().all(|b| b.is_ascii_uppercase() || b == b'-')
    { return false; }
    let lower = trimmed.to_ascii_lowercase();
    !lower.ends_with("ly")
        && !lower.ends_with("ed")
        && !NON_NOUN_WORDS.contains(&lower.as_str())
        && !dict.is_some_and(|d| d.is_tech_noun(&lower))
}

/// Replace verbose verbs outside backtick code spans.
pub fn simplify_verbs(text: &str) -> String {
    let (mut out, mut rest) = (String::with_capacity(text.len()), text);
    while let Some(open) = rest.find('`') {
        rewrite_plain_segment(&rest[..open], &mut out);
        out.push('`');
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else {
            out.push_str(after);
            return out;
        };
        out.push_str(&after[..=close]);
        rest = &after[close + 1..];
    }
    rewrite_plain_segment(rest, &mut out);
    out
}

fn rewrite_plain_segment(segment: &str, out: &mut String) {
    let (bytes, mut cursor) = (segment.as_bytes(), 0usize);
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    for m in VERB_AC.find_iter(segment) {
        let before_ok = m.start() == 0 || !is_word(bytes[m.start() - 1]);
        let after_ok = m.end() == bytes.len() || !is_word(bytes[m.end()]);
        if !before_ok || !after_ok { continue; }
        out.push_str(&segment[cursor..m.start()]);
        let swap = VERB_SWAPS[m.pattern().as_usize()].1;
        if bytes[m.start()].is_ascii_uppercase() {
            let mut chars = swap.chars();
            if let Some(first) = chars.next() {
                out.push(first.to_ascii_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push_str(swap);
        }
        cursor = m.end();
    }
    out.push_str(&segment[cursor..]);
}

/// Lint a Markdown document, skipping fenced code blocks and headings.
pub fn lint_md(markdown: &str, dict: Option<&WordList>) -> ProseReport {
    let (mut in_fence, mut in_untrusted) = (false, false);
    let (mut total_sentences, mut compliant_sentences, mut issues) = (0usize, 0usize, Vec::new());

    for (idx, raw_line) in markdown.lines().enumerate() {
        let t = raw_line.trim();
        if t.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if t.starts_with("<untrusted_web_evidence") {
            in_untrusted = true;
            continue;
        }
        if t.starts_with("</untrusted_web_evidence") {
            in_untrusted = false;
            continue;
        }
        if in_fence || in_untrusted || t.is_empty()
            || t.starts_with(['#', '|']) || t.starts_with("---") || t.starts_with("> ")
        {
            continue;
        }
        let (kind, body) = classify_md_line(t);
        for sentence in split_sentences(body) {
            total_sentences += 1;
            let rules = lint_sentence(&sentence, kind, dict);
            if rules.is_empty() {
                compliant_sentences += 1;
            } else {
                for rule in rules {
                    issues.push(LintIssue {
                        line: idx + 1, rule: rule.to_string(),
                        detail: format!("Prose rule `{rule}` violated"),
                        sentence: sentence.clone(),
                    });
                }
            }
        }
    }

    ProseReport {
        total_sentences, compliant_sentences, violation_count: issues.len(),
        compliant: issues.is_empty(), custom_dict_loaded: dict.is_some(),
        custom_dict_entries: dict.map_or(0, WordList::entry_count), issues,
    }
}

fn classify_md_line(trimmed: &str) -> (SentenceKind, &str) {
    let digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && let Some(rest) = trimmed[digits..].strip_prefix(". ") {
        return (SentenceKind::Procedural, rest.trim());
    }
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return (SentenceKind::Descriptive, rest.trim());
    }
    (SentenceKind::Descriptive, trimmed)
}

fn split_sentences(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut in_code) = (Vec::new(), String::new(), false);
    let chars: Vec<char> = line.chars().collect();
    for (idx, &ch) in chars.iter().enumerate() {
        if ch == '`' { in_code = !in_code; }
        cur.push(ch);
        if !in_code && matches!(ch, '.' | '!' | '?') && chars.get(idx + 1).is_none_or(|n| n.is_whitespace()) {
            let trimmed = cur.trim();
            if !trimmed.is_empty() { out.push(trimmed.to_string()); }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
    out
}

/// Compile a simplified technical English audit report from an `AuditReport`.
pub fn audit_md(report: &crate::types::AuditReport) -> String {
    let mut md = format!(
        concat!(
            "# refit Audit Report\n\n",
            "- Repository `{}` has {} source files and {} infra files.\n",
            "- Maintainability median is `{:.3}` with p10 at `{:.3}`.\n",
            "- Scan found {} cycles, {} dead symbols, {} clones, and {} infra findings.\n\n",
        ),
        report.repo_root, report.total_source_files, report.total_infra_files,
        report.hygiene_median, report.hygiene_p10, report.scc_count,
        report.dead_symbols.len(), report.clone_clusters.len(), report.infra_findings.len(),
    );
    if !report.triage_plan.is_empty() {
        md.push_str("## Action Plan\n\n");
        for (idx, item) in report.triage_plan.iter().take(25).enumerate() {
            let verb = match item.action.as_str() {
                "harden_infra" => "Harden", "consolidate" => "Consolidate", "split" => "Split",
                "remove_unused_dep" => "Remove unused dependency in",
                "review_delete" => "Review and delete", _ => "Inspect",
            };
            let _ = writeln!(md, "{}. {verb} `{}` for priority `P{}` (`{}`).", idx + 1, item.file, item.priority, item.action);
        }
        md.push('\n');
    }
    if !report.god_files.is_empty() {
        md.push_str("## Structural Risk Files\n\n");
        for god in report.god_files.iter().take(10) {
            let _ = writeln!(
                md, "- File `{}` has {} lines, score `{:.3}`, and action `{}`.",
                god.file, god.loc, god.hygiene_score, god.recommendation
            );
        }
        md.push('\n');
    }
    simplify_verbs(&md)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use SentenceKind::{Descriptive as D, Procedural as P};

    #[test]
    fn ste100_rules_and_swaps() {
        for &(kind, sent) in &[
            (P, "Remove `fs2` from `Cargo.toml` and call `std::fs::File::lock`."),
            (P, "Set `MIMALLOC_PURGE_DELAY=0` before you start the container."),
            (P, "Call `std::mem::take` with `--edition=2024` on `\\\\?\\C:\\foo\\bar.rs`."),
            (P, "Check `utilize_cache();` inside backticks without triggering linter errors."),
            (P, "Send `If-None-Match` with the saved ETag to `index.crates.io`."),
            (D, "Rust 1.89 provides native OS file locking in `std::fs::File::lock`."),
            (D, "Code spans like `e.g.; don't utilize` pass the linter cleanly."),
            (D, "The value is a string."),
            (P, "Read the file page cache."),
            (D, "`ArcSwap::load()` returns a read guard without atomic `Arc::clone` traffic."),
        ] {
            assert!(lint_sentence(sent, kind, None).is_empty(), "false positive on `{sent}`");
        }

        let long_proc = concat!(
            "Open the repository root directory and read every single manifest file and parse ",
            "every pinned version and sort all crate entries and send every query now.",
        );
        let disguised = concat!(
            "Check `this entire unapproved English prose sentence is stuffed inside backticks ",
            "to bypass word limits` now.",
        );
        for &(kind, sent, expected) in &[
            (P, "Utilize `std::fs::File::lock` to lock the file.", "unapproved_verb"),
            (P, "Commence the repository scan immediately.", "unapproved_verb"),
            (P, "Remove old caches; run `mbx gc` afterward.", "semicolon_banned"),
            (P, "Skip legacy tools, e.g. `fs2` and `atty`.", "latin_abbrev"),
            (D, "The cache doesn't keep expired entries on disk.", "contraction"),
            (P, "Please check `memory.current` before spawning Cargo.", "non_imperative_procedural"),
            (D, "The WAL frame was written to disk before crash.", "passive_voice"),
            (P, "Invoke `TaskContext::cancelled()` because `tasks/cancel` is running inside the daemon.", "progressive_ing"),
            (P, long_proc, "word_limit_exceeded"),
            (D, disguised, "disguised_prose_in_code_span"),
            (P, "Check repository dependency version lockfile cache.", "noun_cluster_exceeded"),
        ] {
            assert!(lint_sentence(sent, kind, None).contains(&expected), "missed `{expected}` on `{sent}`");
        }
        let rewritten = simplify_verbs("Utilize `utilize_fn()` and utilize_id and reutilize to obtain results.");
        assert_eq!(rewritten, "Use `utilize_fn()` and utilize_id and reutilize to get results.");
        assert_eq!(simplify_verbs(&rewritten), rewritten);
    }

    #[test]
    fn ste100_dict_and_markdown() {
        let dict_txt = "# User wordlist\ndeploy: unapproved -> install\nblazing: technical\ninstall\nremove\nfile\n";
        let tmp = TempRepo::with_files("ste_dict", &[("local_dict.txt", dict_txt)]);
        let dict = WordList::load_from_file(&tmp.join("local_dict.txt")).unwrap();
        assert_eq!(dict.entry_count(), 5);
        for (kind, sent, expected) in [
            (P, "Deploy `app.bin`.", Some("dict_unapproved_word")),
            (P, "Install `app.bin`.", None),
            (P, "Inspect file.", Some("dict_not_in_wordlist")),
        ] {
            let errs = lint_sentence(sent, kind, Some(&dict));
            match expected {
                Some(rule) => assert!(errs.contains(&rule), "missed {rule} on `{sent}`"),
                None => assert!(errs.is_empty(), "unexpected {errs:?} on `{sent}`"),
            }
        }
        assert!(!lint_sentence("Mode is blazing.", D, Some(&dict)).contains(&"progressive_ing"));

        let md = concat!(
            "# Report\n\n```rust\n// Utilize old code; it's fine.\n```\n\n",
            "1. Remove `fs2` from `Cargo.toml`.\n",
            "- Rust 1.89 provides native OS file locking in `std::fs::File::lock`.\n",
        );
        let rep = lint_md(md, None);
        assert!(rep.compliant && rep.total_sentences == 2 && rep.violation_count == 0);
    }
}
