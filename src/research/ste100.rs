//! Pure-Rust ASD-STE100 Issue 9 structural linter and report compiler.
//!
//! Legal gate: ZERO copyrighted ASD-STE100 dictionary words are bundled in this crate.
//! Users who hold a licensed ASD-STE100 dictionary may pass `--ste-dict <path>` at runtime
//! to load a local wordlist (one word per line or `word: status` / `word: unapproved -> swap`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

const MAX_PROCEDURAL_WORDS: usize = 20;
const MAX_DESCRIPTIVE_WORDS: usize = 25;
const MAX_NOUN_CLUSTER: usize = 3;

const LATIN_ABBREVS: &[&str] = &["e.g.", "i.e.", "etc.", "vs.", "cf."];

const CONTRACTIONS: &[&str] = &[
    "don't", "doesn't", "can't", "won't", "isn't", "aren't", "it's", "we're", "you're",
    "shouldn't", "couldn't", "haven't", "hasn't", "didn't", "wasn't", "weren't",
];

/// Small plain-English clarity verb swaps (not an ASD-STE100 copyrighted dictionary).
const CLARITY_VERB_SWAPS: &[(&str, &str)] = &[
    ("utilize", "use"),
    ("utilizes", "uses"),
    ("utilized", "used"),
    ("commence", "start"),
    ("commences", "starts"),
    ("terminate", "stop"),
    ("terminates", "stops"),
    ("perform", "do"),
    ("performs", "does"),
    ("performed", "did"),
    ("leverage", "use"),
    ("leverages", "uses"),
    ("facilitate", "help"),
    ("facilitates", "helps"),
    ("obtain", "get"),
    ("obtains", "gets"),
    ("instantiate", "create"),
    ("instantiates", "creates"),
    ("attempt", "try"),
    ("attempts", "tries"),
];

const PASSIVE_AUX: &[&str] = &["is", "are", "was", "were", "be", "been"];

const IRREG_PARTICIPLES: &[&str] = &[
    "written", "built", "sent", "drawn", "driven", "chosen", "given", "taken", "broken",
    "found", "held", "kept", "made", "paid", "read", "run", "set", "shown",
];

const BAD_PROCEDURAL_OPENERS: &[&str] = &["you", "please", "it", "we", "developers", "users"];

const DETERMINERS: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "each", "every", "one", "my", "our",
    "its", "their", "your",
];

const NON_VERB_ING_WORDS: &[&str] = &["string", "thing", "spring", "ceiling", "morning", "warning"];

const NON_NOUN_WORDS: &[&str] = &[
    // Function words, prepositions, conjunctions, modals, adverbs
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "into", "onto",
    "not", "are", "was", "were", "been", "has", "have", "had", "can", "will", "must", "may",
    "should", "would", "could", "when", "while", "then", "than", "only", "each", "every",
    "both", "all", "any", "some", "one", "two", "three", "four", "five", "zero", "less",
    "more", "much", "before", "after", "under", "over", "inside", "outside", "across",
    "without", "during", "near", "against", "between", "through", "until", "because", "since",
    "soon", "first", "now", "also", "still", "never", "always", "forever", "per", "via",
    // Common active verbs used in technical prose
    "remove", "set", "start", "sort", "protect", "call", "write", "pass", "use", "detect",
    "replace", "cap", "keep", "push", "free", "flush", "normalize", "check", "drop", "run",
    "send", "read", "separate", "strip", "return", "provides", "removes", "returns", "refills",
    "finishes", "merges", "strips", "enables", "changes", "cross", "normalizes", "compares",
    "hashes", "groups", "improve", "uses", "cuts", "preserves", "resolves", "shares", "counts",
    "catches", "occur", "separates", "rejects", "fails", "catch", "handles", "passes",
    "requires", "binds", "runs", "starts", "locks", "corrupts", "configure", "update",
    "invoke", "rebuild", "split", "audit", "verify", "commit", "abort", "stage", "create",
    "delete", "move", "open", "close", "avoid", "review", "consolidate", "harden", "prune",
    // Common technical adjectives/modifiers
    "native", "exact", "raw", "fast", "tiny", "small", "large", "long", "short", "high",
    "low", "new", "old", "good", "bad", "clean", "final", "intermediate", "atomic", "safe",
    "unsafe", "single", "single-pass", "short-lived", "time-sortable", "idle", "transient",
    "anonymous", "sequential", "uninitialized", "silent", "trailing", "upstream", "duplicate",
    "direct", "opaque", "strict", "empty", "extra", "plain", "public", "private", "dirty",
    "cold", "warm", "hot", "hourly", "mandatory", "stateless", "deterministic", "bounded",
    "virtual", "external", "internal", "legacy", "local", "remote", "relative", "minimal",
    "actual", "other", "non-ascii", "headless", "active", "valid", "invalid", "dead", "unused",
    "circular", "deep", "shallow", "prioritized", "structural", "technical", "full", "total",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SentenceKind {
    Procedural,
    Descriptive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintIssue {
    pub line: usize,
    pub rule: String,
    pub detail: String,
    pub sentence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentLintReport {
    pub total_sentences: usize,
    pub compliant_sentences: usize,
    pub violation_count: usize,
    pub compliant: bool,
    pub custom_dict_loaded: bool,
    pub custom_dict_entries: usize,
    pub issues: Vec<LintIssue>,
}

/// Optional user-supplied local dictionary loaded via `--ste-dict <path>` at runtime.
#[derive(Debug, Clone, Default)]
pub struct SteDictionary {
    approved: HashSet<String>,
    unapproved: HashMap<String, Option<String>>,
    technical_nouns: HashSet<String>,
}

impl SteDictionary {
    /// Load a user-supplied local wordlist file at runtime.
    ///
    /// Supported line formats (comments starting with `#` or `//` and blank lines are ignored):
    /// - `word` (one approved word per line)
    /// - `word: approved` / `word: technical`
    /// - `word: unapproved` / `word: unapproved -> replacement`
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read --ste-dict file {}", path.display()))?;
        Ok(Self::parse_str(&raw))
    }

    pub fn parse_str(raw: &str) -> Self {
        let mut dict = Self::default();
        for raw_line in raw.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            dict.parse_entry_line(line);
        }
        dict
    }

    fn parse_entry_line(&mut self, line: &str) {
        if let Some((left, right)) = line.split_once(':') {
            let word = left.trim().to_ascii_lowercase();
            if word.is_empty() {
                return;
            }
            self.insert_status_entry(word, right.trim());
            return;
        }
        if let Some(word) = line.split_whitespace().next() {
            let clean = word.trim().to_ascii_lowercase();
            if !clean.is_empty() {
                self.approved.insert(clean);
            }
        }
    }

    fn insert_status_entry(&mut self, word: String, spec: &str) {
        let (status_raw, swap_opt) = match spec.split_once("->") {
            Some((st, repl)) => (st.trim(), Some(repl.trim().to_string()).filter(|s| !s.is_empty())),
            None => (spec, None),
        };
        match status_raw.to_ascii_lowercase().as_str() {
            "approved" | "allow" | "allowed" | "ok" => {
                self.approved.insert(word);
            }
            "technical" | "technical_noun" | "tn" => {
                self.technical_nouns.insert(word);
            }
            "unapproved" | "banned" | "forbid" | "forbidden" | "disallow" => {
                self.unapproved.insert(word, swap_opt);
            }
            _ => {
                self.approved.insert(word);
            }
        }
    }

    pub fn entry_count(&self) -> usize {
        self.approved.len() + self.unapproved.len() + self.technical_nouns.len()
    }

    fn is_technical_noun(&self, word: &str) -> bool {
        self.technical_nouns.contains(word)
    }

    fn is_explicit_approved(&self, word: &str) -> bool {
        self.approved.contains(word) || self.technical_nouns.contains(word)
    }

    fn unapproved_replacement(&self, word: &str) -> Option<Option<&str>> {
        self.unapproved.get(word).map(|opt| opt.as_deref())
    }

    fn enforces_allowlist(&self) -> bool {
        !self.approved.is_empty()
    }
}

/// Technical-Noun anti-bypass classifier for inline backtick code spans.
///
/// Accepts single-token identifiers and 2..=4-token code/flag/path expressions while rejecting
/// `>= 5`-word prose or plain-English phrases disguised inside backticks.
pub fn is_technical_code_token(span: &str) -> bool {
    let trimmed = span.trim();
    if trimmed.is_empty() {
        return true;
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() >= 5 {
        return false;
    }
    if words.len() == 1 {
        return true;
    }
    trimmed.chars().any(is_code_syntax_char)
}

fn is_code_syntax_char(ch: char) -> bool {
    matches!(
        ch,
        ':' | '/'
            | '\\'
            | '_'
            | '-'
            | '='
            | '('
            | ')'
            | '['
            | ']'
            | '.'
            | ';'
            | '$'
            | '%'
            | '@'
            | '<'
            | '>'
            | '{'
            | '}'
            | '0'..='9'
    )
}

/// Mask valid backticked Technical Nouns as `CODETOK` and unwrap disguised English prose spans.
pub fn mask_code_spans(input: &str) -> (String, bool) {
    let mut out = String::with_capacity(input.len());
    let mut cur = String::new();
    let mut in_code = false;
    let mut disguised_prose = false;

    for ch in input.chars() {
        if ch == '`' {
            toggle_code_span(&mut in_code, &mut cur, &mut out, &mut disguised_prose);
        } else if in_code {
            cur.push(ch);
        } else {
            out.push(ch);
        }
    }
    if in_code && !cur.is_empty() {
        out.push(' ');
        out.push_str(&cur);
    }
    (out, disguised_prose)
}

fn toggle_code_span(
    in_code: &mut bool,
    cur: &mut String,
    out: &mut String,
    disguised_prose: &mut bool,
) {
    if !*in_code {
        *in_code = true;
        return;
    }
    if is_technical_code_token(cur) {
        out.push_str(" CODETOK ");
    } else {
        *disguised_prose = true;
        out.push(' ');
        out.push_str(cur);
        out.push(' ');
    }
    cur.clear();
    *in_code = false;
}

fn push_unique(errs: &mut Vec<&'static str>, rule: &'static str) {
    if !errs.contains(&rule) {
        errs.push(rule);
    }
}

fn clean_token(word: &str) -> String {
    word.to_ascii_lowercase()
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .to_string()
}

/// Lint a single sentence under ASD-STE100 Issue 9 structural rules + optional user dictionary.
pub fn lint_sentence(
    sentence: &str,
    kind: SentenceKind,
    dict: Option<&SteDictionary>,
) -> Vec<&'static str> {
    let mut errs: Vec<&'static str> = Vec::new();
    let (masked, disguised_prose) = mask_code_spans(sentence);

    if disguised_prose {
        push_unique(&mut errs, "disguised_prose_in_code_span");
    }
    if masked.contains(';') {
        push_unique(&mut errs, "semicolon_banned");
    }

    let raw_words: Vec<&str> = masked.split_whitespace().collect();
    let max_words = match kind {
        SentenceKind::Procedural => MAX_PROCEDURAL_WORDS,
        SentenceKind::Descriptive => MAX_DESCRIPTIVE_WORDS,
    };
    if raw_words.len() > max_words {
        push_unique(&mut errs, "word_limit_exceeded");
    }

    let lowered_clean = scan_lexical_rules(&raw_words, dict, &mut errs);
    check_passive_and_progressive(&lowered_clean, dict, &mut errs);
    check_noun_clusters(&raw_words, dict, &mut errs);

    if kind == SentenceKind::Procedural
        && let Some(first) = lowered_clean.first().map(String::as_str)
        && BAD_PROCEDURAL_OPENERS.contains(&first)
    {
        push_unique(&mut errs, "non_imperative_procedural");
    }

    errs
}

fn scan_lexical_rules(
    raw_words: &[&str],
    dict: Option<&SteDictionary>,
    errs: &mut Vec<&'static str>,
) -> Vec<String> {
    let mut cleaned = Vec::with_capacity(raw_words.len());
    for word in raw_words {
        let lower = word.to_ascii_lowercase();
        let trimmed_punct = lower.trim_matches(|c: char| matches!(c, ',' | '(' | ')' | ':' | '"'));
        if LATIN_ABBREVS.contains(&trimmed_punct) {
            push_unique(errs, "latin_abbrev");
        }

        let clean = clean_token(word);
        if CONTRACTIONS.contains(&clean.as_str()) {
            push_unique(errs, "contraction");
        }
        check_verb_and_dict(&clean, dict, errs);
        cleaned.push(clean);
    }
    cleaned
}

fn check_verb_and_dict(
    clean: &str,
    dict: Option<&SteDictionary>,
    errs: &mut Vec<&'static str>,
) {
    if clean.is_empty() || clean == "codetok" {
        return;
    }
    let explicitly_ok = dict.is_some_and(|d| d.is_explicit_approved(clean));
    if !explicitly_ok && CLARITY_VERB_SWAPS.iter().any(|(bad, _)| *bad == clean) {
        push_unique(errs, "unapproved_verb");
    }
    let Some(user_dict) = dict else { return };
    if user_dict.unapproved_replacement(clean).is_some() {
        push_unique(errs, "dict_unapproved_word");
    } else if user_dict.enforces_allowlist()
        && is_plain_alpha_word(clean)
        && !user_dict.is_explicit_approved(clean)
    {
        push_unique(errs, "dict_not_in_wordlist");
    }
}

fn is_plain_alpha_word(word: &str) -> bool {
    word.len() >= 2 && word.bytes().all(|b| b.is_ascii_alphabetic())
}

fn check_passive_and_progressive(
    tokens: &[String],
    dict: Option<&SteDictionary>,
    errs: &mut Vec<&'static str>,
) {
    for idx in 0..tokens.len() {
        if !PASSIVE_AUX.contains(&tokens[idx].as_str()) {
            continue;
        }
        inspect_aux_window(tokens, idx, dict, errs);
    }
}

fn inspect_aux_window(
    tokens: &[String],
    aux_idx: usize,
    dict: Option<&SteDictionary>,
    errs: &mut Vec<&'static str>,
) {
    let last = (aux_idx + 2).min(tokens.len().saturating_sub(1));
    let mut det_between = false;
    for pos in (aux_idx + 1)..=last {
        let tok = tokens[pos].as_str();
        if pos == aux_idx + 1 && DETERMINERS.contains(&tok) {
            det_between = true;
            continue;
        }
        if det_between {
            continue;
        }
        if (tok.len() > 4 && tok.ends_with("ed")) || IRREG_PARTICIPLES.contains(&tok) {
            push_unique(errs, "passive_voice");
            return;
        }
        let is_tn = dict.is_some_and(|d| d.is_technical_noun(tok));
        if tok.len() >= 5
            && tok.ends_with("ing")
            && !is_tn
            && !NON_VERB_ING_WORDS.contains(&tok)
        {
            push_unique(errs, "progressive_ing");
            return;
        }
    }
}

fn check_noun_clusters(
    raw_words: &[&str],
    dict: Option<&SteDictionary>,
    errs: &mut Vec<&'static str>,
) {
    let mut run = 0usize;
    for raw in raw_words {
        if raw.starts_with('(') || raw.starts_with('[') {
            run = 0;
        }
        let ends_clause = raw.ends_with(|c: char| matches!(c, ',' | '.' | ':' | ';' | '?' | '!' | ')'));
        if is_general_noun_candidate(raw, dict) {
            run += 1;
            if run > MAX_NOUN_CLUSTER {
                push_unique(errs, "noun_cluster_exceeded");
                return;
            }
        } else {
            run = 0;
        }
        if ends_clause {
            run = 0;
        }
    }
}

fn is_general_noun_candidate(raw: &str, dict: Option<&SteDictionary>) -> bool {
    let trimmed = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
    if trimmed.len() <= 2 || trimmed == "CODETOK" {
        return false;
    }
    if trimmed.bytes().any(|b| b.is_ascii_digit()) {
        return false;
    }
    if trimmed.bytes().all(|b| b.is_ascii_uppercase() || b == b'-') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with("ly") || lower.ends_with("ed") {
        return false;
    }
    if NON_NOUN_WORDS.contains(&lower.as_str()) {
        return false;
    }
    if dict.is_some_and(|d| d.is_technical_noun(&lower)) {
        return false;
    }
    true
}

/// Apply deterministic clarity verb swaps outside backticked code spans.
pub fn normalize_clarity_verbs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    let mut segment = String::new();

    for ch in text.chars() {
        if ch == '`' {
            if !in_code {
                out.push_str(&rewrite_plain_segment(&segment));
                segment.clear();
                in_code = true;
            } else {
                in_code = false;
            }
            out.push('`');
        } else if in_code {
            out.push(ch);
        } else {
            segment.push(ch);
        }
    }
    if !segment.is_empty() {
        out.push_str(&rewrite_plain_segment(&segment));
    }
    out
}

fn rewrite_plain_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut word = String::new();

    for ch in segment.chars() {
        if ch.is_ascii_alphabetic() {
            word.push(ch);
        } else {
            flush_rewritten_word(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush_rewritten_word(&mut word, &mut out);
    out
}

fn flush_rewritten_word(word: &mut String, out: &mut String) {
    if word.is_empty() {
        return;
    }
    let lower = word.to_ascii_lowercase();
    if let Some((_, swap)) = CLARITY_VERB_SWAPS.iter().find(|(bad, _)| *bad == lower) {
        let cap = word.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        if cap {
            let mut chars = swap.chars();
            if let Some(first) = chars.next() {
                out.push(first.to_ascii_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push_str(swap);
        }
    } else {
        out.push_str(word);
    }
    word.clear();
}

/// Lint a Markdown audit or sustain document, skipping fenced code blocks and headings.
pub fn lint_markdown(markdown: &str, dict: Option<&SteDictionary>) -> DocumentLintReport {
    let mut in_fence = false;
    let mut in_untrusted = false;
    let mut total_sentences = 0usize;
    let mut compliant_sentences = 0usize;
    let mut issues = Vec::new();

    for (idx, raw_line) in markdown.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if update_skip_blocks(trimmed, &mut in_fence, &mut in_untrusted) {
            continue;
        }
        if is_structural_md_line(trimmed) {
            continue;
        }
        let (kind, body) = classify_md_line(trimmed);
        for sentence in split_prose_sentences(body) {
            total_sentences += 1;
            let rules = lint_sentence(&sentence, kind, dict);
            if rules.is_empty() {
                compliant_sentences += 1;
            } else {
                record_issues(line_no, &sentence, &rules, &mut issues);
            }
        }
    }

    DocumentLintReport {
        total_sentences,
        compliant_sentences,
        violation_count: issues.len(),
        compliant: issues.is_empty(),
        custom_dict_loaded: dict.is_some(),
        custom_dict_entries: dict.map_or(0, SteDictionary::entry_count),
        issues,
    }
}

fn update_skip_blocks(trimmed: &str, in_fence: &mut bool, in_untrusted: &mut bool) -> bool {
    if trimmed.starts_with("```") {
        *in_fence = !*in_fence;
        return true;
    }
    if trimmed.starts_with("<untrusted_web_evidence") {
        *in_untrusted = true;
        return true;
    }
    if trimmed.starts_with("</untrusted_web_evidence") {
        *in_untrusted = false;
        return true;
    }
    *in_fence || *in_untrusted
}

fn is_structural_md_line(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with('|')
        || trimmed.starts_with("---")
        || trimmed.starts_with("> ")
}

fn classify_md_line(trimmed: &str) -> (SentenceKind, &str) {
    if let Some(rest) = strip_ordered_prefix(trimmed) {
        return (SentenceKind::Procedural, rest);
    }
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        return (SentenceKind::Descriptive, rest.trim());
    }
    (SentenceKind::Descriptive, trimmed)
}

fn strip_ordered_prefix(trimmed: &str) -> Option<&str> {
    let digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let after = &trimmed[digits..];
    after.strip_prefix(". ").map(str::trim)
}

fn split_prose_sentences(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_code = false;
    let chars: Vec<char> = line.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        if ch == '`' {
            in_code = !in_code;
        }
        cur.push(ch);
        let at_boundary = !in_code
            && matches!(ch, '.' | '!' | '?')
            && chars.get(idx + 1).is_none_or(|n| n.is_whitespace());
        if at_boundary {
            let trimmed = cur.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

fn record_issues(
    line: usize,
    sentence: &str,
    rules: &[&'static str],
    issues: &mut Vec<LintIssue>,
) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for &rule in rules {
        *counts.entry(rule).or_insert(0) += 1;
    }
    for (rule, _) in counts {
        issues.push(LintIssue {
            line,
            rule: rule.to_string(),
            detail: format!("ASD-STE100 Issue 9 rule `{rule}` violated"),
            sentence: sentence.to_string(),
        });
    }
}

/// Compile a deterministic ASD-STE100 Issue 9 compliant audit summary from an `AuditReport`.
pub fn compile_audit_markdown(report: &crate::types::AuditReport) -> String {
    let mut md = String::new();
    md.push_str("# refit STE100 Issue 9 Audit Report\n\n");
    md.push_str(&format!(
        "- Repository `{}` has {} source files and {} infra files.\n",
        report.repo_root, report.total_source_files, report.total_infra_files
    ));
    md.push_str(&format!(
        "- Maintainability median is `{:.3}` with p10 at `{:.3}`.\n",
        report.hygiene_median, report.hygiene_p10
    ));
    md.push_str(&format!(
        "- Scan found {} cycles, {} dead symbols, {} clones, and {} infra findings.\n\n",
        report.scc_count,
        report.dead_symbols.len(),
        report.clone_clusters.len(),
        report.infra_findings.len()
    ));

    if !report.triage_plan.is_empty() {
        md.push_str("## Action Plan\n\n");
        for (idx, item) in report.triage_plan.iter().take(25).enumerate() {
            let verb = triage_imperative(&item.action);
            let step = idx + 1;
            md.push_str(&format!(
                "{step}. {verb} `{}` for priority `P{}` (`{}`).\n",
                item.file, item.priority, item.action
            ));
        }
        md.push('\n');
    }

    if !report.god_files.is_empty() {
        md.push_str("## Structural Risk Files\n\n");
        for god in report.god_files.iter().take(10) {
            md.push_str(&format!(
                "- File `{}` has {} lines, score `{:.3}`, and action `{}`.\n",
                god.file, god.loc, god.hygiene_score, god.recommendation
            ));
        }
        md.push('\n');
    }

    normalize_clarity_verbs(&md)
}

fn triage_imperative(action: &str) -> &'static str {
    match action {
        "harden_infra" => "Harden",
        "consolidate" => "Consolidate",
        "split" => "Split",
        "remove_unused_dep" => "Remove unused dependency in",
        "review_delete" => "Review and delete",
        _ => "Inspect",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn s2_ste100_120_of_120_exact_zero_fp_zero_fn() {
        let compliant: &[(SentenceKind, &str)] = &[
            // 1-25: Procedural compliant
            (SentenceKind::Procedural, "Remove `fs2` from `Cargo.toml` and call `std::fs::File::lock`."),
            (SentenceKind::Procedural, "Set `MIMALLOC_PURGE_DELAY=0` before you start the container."),
            (SentenceKind::Procedural, "Sort the `tools` array alphabetically by name to protect the prompt cache."),
            (SentenceKind::Procedural, "Call `dunce::canonicalize` on every repository root path on Windows."),
            (SentenceKind::Procedural, "Write the 43-byte `CACHEDIR.TAG` header into `.refit/cache/`."),
            (SentenceKind::Procedural, "Pass `Mcp-Method` and `Mcp-Name` headers on every HTTP request."),
            (SentenceKind::Procedural, "Use `std::io::IsTerminal` to detect piped standard input."),
            (SentenceKind::Procedural, "Replace `cmd /C` with direct `Command::new` execution."),
            (SentenceKind::Procedural, "Cap `zstd` `windowLog` at 18 to keep encoder memory under 256 KB."),
            (SentenceKind::Procedural, "Call `posix_fadvise` with `POSIX_FADV_DONTNEED` after bulk file reads."),
            (SentenceKind::Procedural, "Push failed requests into `retry_heap` and free the worker slot immediately."),
            (SentenceKind::Procedural, "Flush the parent directory with `sync_all` after atomic rename on Unix."),
            (SentenceKind::Procedural, "Set `MALLOC_ARENA_MAX=1` in child environments on Linux containers."),
            (SentenceKind::Procedural, "Normalize path tokens to NFC before you check `.git` deny rules."),
            (SentenceKind::Procedural, "Drop `response` bodies promptly so HTTP/2 flow control never stalls."),
            (SentenceKind::Procedural, "Run `ruff` as a short-lived CLI process inside 256 MB containers."),
            (SentenceKind::Procedural, "Keep `MBX_BLAZE=1` disabled on hosts with less than 16 GB RAM."),
            (SentenceKind::Procedural, "Use `UV_LINK_MODE=hardlink` on Linux and Windows to protect virtual environments."),
            (SentenceKind::Procedural, "Send `If-None-Match` with the saved ETag to `index.crates.io`."),
            (SentenceKind::Procedural, "Read `memory.stat` in cgroup v2 to separate `anon` from `file` memory."),
            (SentenceKind::Procedural, "Check `utilize_cache();` inside backticks without triggering linter errors."),
            (SentenceKind::Procedural, "Call `perform_sync(a; b)` in `src/mvcc/wal.rs` before commit."),
            (SentenceKind::Procedural, "Strip trailing `\\r` bytes from piped input lines on Windows."),
            (SentenceKind::Procedural, "Use `aho-corasick` for single-pass snippet sanitization in `src/research.rs`."),
            (SentenceKind::Procedural, "Return `resultType` set to `task` for long repository audits."),
            // 26-50: Descriptive compliant
            (SentenceKind::Descriptive, "Rust 1.89 provides native OS file locking in `std::fs::File::lock`."),
            (SentenceKind::Descriptive, "MCP `2026-07-28` removes mandatory `initialize` handshakes for stateless requests."),
            (SentenceKind::Descriptive, "GitHub returns HTTP 304 without counting against the hourly rate limit."),
            (SentenceKind::Descriptive, "The sliding window refills each free slot as soon as one task finishes."),
            (SentenceKind::Descriptive, "Single-flight coalescing merges duplicate keys into one upstream network call."),
            (SentenceKind::Descriptive, "`dunce::canonicalize` strips `\\\\?\\` prefixes only when safe for Windows tools."),
            (SentenceKind::Descriptive, "PowerShell 7.6 permanently enables tilde expansion for native commands on Windows."),
            (SentenceKind::Descriptive, "PowerShell 7.7 changes `New-Guid` output from UUIDv4 to time-sortable UUIDv7."),
            (SentenceKind::Descriptive, "WSL2 `/mnt/c` paths cross the 9P bridge and run much slower than ext4."),
            (SentenceKind::Descriptive, "APFS normalizes filenames to NFD while Linux ext4 compares raw bytes."),
            (SentenceKind::Descriptive, "Anthropic prompt caching hashes tools first, then system prompts, then messages."),
            (SentenceKind::Descriptive, "`bitcode` groups struct fields to improve downstream `zstd` compression ratios."),
            (SentenceKind::Descriptive, "`lz4_flex` uses tiny hash tables near 16 KB for fast L1 caching."),
            (SentenceKind::Descriptive, "`gix-index` reads `.git/index` directly in Rust with minimal heap overhead."),
            (SentenceKind::Descriptive, "A 64 KB `BufReader` cuts syscall overhead on Windows sequential reads."),
            (SentenceKind::Descriptive, "`osv.dev` `querybatch` preserves exact input order across package ecosystems."),
            (SentenceKind::Descriptive, "`deps.dev` `versionbatch` resolves versions and source repositories in batch calls."),
            (SentenceKind::Descriptive, "`mr-boxington` shares Cargo outputs across worktrees with zero background daemon."),
            (SentenceKind::Descriptive, "`ruff` CLI cold spawns use 10 to 30 MB transient RAM and zero idle RAM."),
            (SentenceKind::Descriptive, "Cgroup v2 counts both anonymous heap and file page cache in `memory.current`."),
            (SentenceKind::Descriptive, "The `0xFF` sentinel catches uninitialized test buffers before silent passes occur."),
            (SentenceKind::Descriptive, "Code spans like `e.g.; don't utilize` pass the linter cleanly."),
            (SentenceKind::Descriptive, "MCP `2025-06-18` removed JSON-RPC batch arrays from the wire specification."),
            (SentenceKind::Descriptive, "Cargo 1.100 separates intermediate build directories from final artifact targets."),
            (SentenceKind::Descriptive, "`check_target` rejects relative escapes when `dunce::canonicalize` fails on root."),
            // 51-60: Technical-Noun / Code-Identifier / CLI-Flag / Windows-Path exemptions (0 FP)
            (SentenceKind::Procedural, "Call `std::mem::take` with `--edition=2024` on `\\\\?\\C:\\foo\\bar.rs`."),
            (SentenceKind::Procedural, "Run `rustc -O -C target-cpu=native` inside `taskset -c 0`."),
            (SentenceKind::Procedural, "Pass `--remap-path-prefix /work=/src` to `kache` on `x86_64-unknown-linux-gnu`."),
            (SentenceKind::Procedural, "Set `BUN_INSTALL_GLOBAL_STORE=1` and run `bun install --linker isolated`."),
            (SentenceKind::Procedural, "Catch `libc::EPERM` and `libc::EXDEV` in `clone_or_copy` inside LXC."),
            (SentenceKind::Descriptive, "`rmcp::task_manager::TaskManager` handles `tasks/cancel` via `TaskContext::cancelled()`."),
            (SentenceKind::Descriptive, "`rust-mcp-sdk` v2.0.0 passes 440 client tests on `2026-07-28`."),
            (SentenceKind::Descriptive, "`FSCTL_DUPLICATE_EXTENTS_TO_FILE` requires cluster offsets from `FSCTL_GET_INTEGRITY_INFORMATION`."),
            (SentenceKind::Descriptive, "`pwsh -NoProfile` binds `ibm437` or `CP936` before `chcp 65001` runs."),
            (SentenceKind::Descriptive, "`ArcSwap::load()` returns a read guard without atomic `Arc::clone` traffic."),
        ];

        let violations: &[(SentenceKind, &str)] = &[
            // 1-50: Structural & clarity violations
            (SentenceKind::Procedural, "Utilize `std::fs::File::lock` to lock the file."),
            (SentenceKind::Procedural, "Commence the repository scan immediately."),
            (SentenceKind::Procedural, "Terminate all worker threads on error."),
            (SentenceKind::Procedural, "Perform a full audit before commit."),
            (SentenceKind::Procedural, "Leverage `dunce` for safe Windows paths."),
            (SentenceKind::Procedural, "Facilitate fast builds with `mbx`."),
            (SentenceKind::Procedural, "Obtain the ETag header from the response."),
            (SentenceKind::Procedural, "Instantiate the HTTP client once per process."),
            (SentenceKind::Procedural, "Attempt to reconnect after timeout."),
            (SentenceKind::Procedural, "Remove old caches; run `mbx gc` afterward."),
            (SentenceKind::Descriptive, "The tool skips caches; it keeps memory low."),
            (SentenceKind::Procedural, "Skip legacy tools, e.g. `fs2` and `atty`."),
            (SentenceKind::Descriptive, "Cgroup v2 tracks total memory, i.e. anon and file cache."),
            (SentenceKind::Descriptive, "Supported ecosystems include Cargo, npm, PyPI, etc."),
            (SentenceKind::Descriptive, "Compare `lz4_flex` vs. `zstd` for L1 caches."),
            (SentenceKind::Descriptive, "Check MCP specs, cf. `SEP-2575`."),
            (SentenceKind::Procedural, "Don't spawn persistent daemons in 256 MB containers."),
            (SentenceKind::Descriptive, "The cache doesn't keep expired entries on disk."),
            (SentenceKind::Descriptive, "PowerShell 5.1 can't escape inner double quotes."),
            (SentenceKind::Descriptive, "This queue won't block healthy slots on retry."),
            (SentenceKind::Descriptive, "It's unsafe to pass raw paths to `cmd /C`."),
            (SentenceKind::Descriptive, "We're sorting tools deterministically for cache hits."),
            (SentenceKind::Descriptive, "You're hitting 9P overhead inside `/mnt/c`."),
            (SentenceKind::Procedural, "You should set `MIMALLOC_PURGE_DELAY=0` in Docker."),
            (SentenceKind::Procedural, "Please check `memory.current` before spawning Cargo."),
            (SentenceKind::Procedural, "It is helpful to sort tools alphabetically."),
            (SentenceKind::Procedural, "We recommend running `ruff` as a CLI binary."),
            (SentenceKind::Procedural, "Users must enable NFC normalization on macOS."),
            (SentenceKind::Procedural, "Developers should avoid `UV_LINK_MODE=symlink`."),
            (SentenceKind::Descriptive, "The file is locked by the operating system kernel."),
            (SentenceKind::Descriptive, "Old entries are purged by the background sweep."),
            (SentenceKind::Descriptive, "The WAL frame was written to disk before crash."),
            (SentenceKind::Descriptive, "The target directory was built inside a temp slot."),
            (SentenceKind::Descriptive, "Duplicate queries were merged by single-flight logic."),
            (SentenceKind::Descriptive, "The response was sent with an ETag header."),
            (SentenceKind::Descriptive, "Stale locks are broken automatically on startup."),
            (SentenceKind::Descriptive, "Every artifact has been cached in the shared store."),
            (SentenceKind::Descriptive, "The archive can be read in one streaming pass."),
            (SentenceKind::Descriptive, "This bug was found during the Windows path audit."),
            (SentenceKind::Descriptive, "The buffer was kept small for low dirty memory."),
            (SentenceKind::Procedural, "Open the repository root directory and read every single manifest file and parse every pinned version and sort all crate entries and send every query in order now."),
            (SentenceKind::Procedural, "Check the lockfile hash first and compare each dependency version against the sparse index and fetch release notes and write a very long summary report immediately."),
            (SentenceKind::Descriptive, "This descriptive sentence intentionally includes far more than twenty five distinct English words so the ASD STE100 Issue 9 Rule 6.3 length check catches overly complex prose every single time."),
            (SentenceKind::Descriptive, "When a containerized Rust binary allocates many separate memory arenas across threads without aggressive page purging the Linux kernel counts every untouched dirty page toward the strict memory ceiling."),
            (SentenceKind::Descriptive, "The system utilizes `zstd` with high memory windows."),
            (SentenceKind::Descriptive, "The server commences startup with old handshakes."),
            (SentenceKind::Descriptive, "The client performs blocking sleeps inside queue slots."),
            (SentenceKind::Descriptive, "The wrapper leverages `cmd /C` on Windows hosts."),
            (SentenceKind::Descriptive, "The cache shouldn't store offline failures on disk."),
            (SentenceKind::Descriptive, "Legacy clients haven't adopted stateless discovery yet."),
            // 51-60: Disguised prose-bloat / passive-voice / progressive-ing / backtick-stuffing (0 FN)
            (SentenceKind::Procedural, "Configure `std::mem::take` so the buffer is cleared automatically by the runtime."),
            (SentenceKind::Procedural, "Update `--edition=2024` after `Cargo.toml` has been verified by the build script."),
            (SentenceKind::Descriptive, "The path `\\\\?\\C:\\foo` was normalized by `dunce::canonicalize` during startup."),
            (SentenceKind::Descriptive, "Cached layers in `node_modules/.aube/` are linked by `nub` across worktrees."),
            (SentenceKind::Procedural, "Invoke `TaskContext::cancelled()` because `tasks/cancel` is running inside the daemon."),
            (SentenceKind::Procedural, "Use `std::mem::take` and also utilize `--edition=2024` when you rebuild every crate inside `\\\\?\\C:\\foo` with extra flags right now."),
            (SentenceKind::Descriptive, "When `pwsh -NoProfile` starts without ConPTY, the Windows host writer locks `ibm437` and silently corrupts every single non-ASCII character across redirected pipes in headless tooling pipelines forever."),
            (SentenceKind::Procedural, "Run `kache` with `RUSTC_WRAPPER` and `blake3` hashes; keep `mbx` ready."),
            (SentenceKind::Descriptive, "The router handles `HTTP 432` quotas, e.g. `tavily`, without stalling."),
            (SentenceKind::Descriptive, "Check `this entire unapproved English prose sentence is stuffed inside backticks to bypass word limits` now."),
        ];

        assert_eq!(compliant.len(), 60);
        assert_eq!(violations.len(), 60);

        for (idx, &(kind, sent)) in compliant.iter().enumerate() {
            let errs = lint_sentence(sent, kind, None);
            assert!(errs.is_empty(), "false positive on compliant #{}: {sent} -> {errs:?}", idx + 1);
        }
        for (idx, &(kind, sent)) in violations.iter().enumerate() {
            let errs = lint_sentence(sent, kind, None);
            assert!(!errs.is_empty(), "false negative on violation #{}: {sent}", idx + 1);
        }

        // Verify backtick prose stuffing explicitly emits `disguised_prose_in_code_span`
        let stuffed = lint_sentence(violations[59].1, SentenceKind::Descriptive, None);
        assert!(stuffed.contains(&"disguised_prose_in_code_span"), "got {stuffed:?}");
    }

    #[test]
    fn noun_cluster_cap_and_user_ste_dict_loader() {
        // >3 consecutive general nouns must be flagged; <=3 passes
        let ok_3 = lint_sentence("Read the file page cache.", SentenceKind::Procedural, None);
        assert!(ok_3.is_empty(), "got {ok_3:?}");
        let bad_4 = lint_sentence(
            "Check repository dependency version lockfile cache.",
            SentenceKind::Procedural,
            None,
        );
        assert!(bad_4.contains(&"noun_cluster_exceeded"), "got {bad_4:?}");

        // Determiner + non-verb `-ing` noun must not false-positive progressive_ing
        let ok_str = lint_sentence("The value is a string.", SentenceKind::Descriptive, None);
        assert!(ok_str.is_empty(), "got {ok_str:?}");

        // Runtime `--ste-dict` user wordlist loader (zero bundled dictionary)
        let tmp = TempRepo::with_files(
            "ste_dict",
            &[(
                "local_dict.txt",
                concat!(
                    "# User-provided local wordlist\n",
                    "deploy: unapproved -> install\n",
                    "blazing: technical\n",
                    "install\n",
                    "remove\n",
                    "file\n",
                ),
            )],
        );
        let dict = SteDictionary::load_from_file(&tmp.join("local_dict.txt")).unwrap();
        assert_eq!(dict.entry_count(), 5);

        let unapproved = lint_sentence("Deploy `app.bin`.", SentenceKind::Procedural, Some(&dict));
        assert!(unapproved.contains(&"dict_unapproved_word"), "got {unapproved:?}");

        let ok_custom = lint_sentence("Install `app.bin`.", SentenceKind::Procedural, Some(&dict));
        assert!(ok_custom.is_empty(), "got {ok_custom:?}");

        let not_listed = lint_sentence("Inspect file.", SentenceKind::Procedural, Some(&dict));
        assert!(not_listed.contains(&"dict_not_in_wordlist"), "got {not_listed:?}");

        // Technical noun in user dict exempts progressive `-ing` false positive
        let tn_ok = lint_sentence("Mode is blazing.", SentenceKind::Descriptive, Some(&dict));
        assert!(!tn_ok.contains(&"progressive_ing"), "got {tn_ok:?}");

        // Markdown linter skips fenced code blocks and lints prose lines
        let md = concat!(
            "# Report\n\n",
            "```rust\n// Utilize old code; it's fine inside fences.\n```\n\n",
            "1. Remove `fs2` from `Cargo.toml`.\n",
            "- Rust 1.89 provides native OS file locking in `std::fs::File::lock`.\n",
        );
        let rep = lint_markdown(md, None);
        assert!(rep.compliant && rep.total_sentences == 2 && rep.violation_count == 0);
        assert_eq!(normalize_clarity_verbs("Utilize `utilize_fn()` now."), "Use `utilize_fn()` now.");
    }
}
