use super::ops::sigmoid_f64;
use crate::types::HygieneResult;
use std::collections::{BTreeMap, BTreeSet};

const MAX_COLS: usize = 128;
const MAX_CF_DEPTH_VETO: usize = 6;
const MAX_COGNITIVE_VETO: usize = 35;
const WARN_CF_DEPTH: usize = 5;
const WARN_COGNITIVE: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Veto,
    Warning,
}

#[derive(Debug, Clone)]
pub struct StrictFinding {
    pub severity: Severity,
    pub rule: &'static str,
    pub line: usize,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct StrictMetrics {
    pub total_lines: usize,
    pub prod_lines: usize,
    pub blank_prod_lines: usize,
    pub nondoc_comment_lines: usize,
    pub indent_mode: &'static str,
    pub indent_quantum: usize,
    pub max_logical_depth: usize,
    pub max_cf_depth: usize,
    pub max_cognitive: usize,
    pub max_col: usize,
    pub max_tail_code_chars: usize,
    pub prod_bindings: usize,
    pub prod_golf_1char: usize,
    pub prod_unwraps: usize,
    pub unsafe_blocks: usize,
}

impl Default for StrictMetrics {
    fn default() -> Self {
        Self {
            total_lines: 0,
            prod_lines: 0,
            blank_prod_lines: 0,
            nondoc_comment_lines: 0,
            indent_mode: "spaces",
            indent_quantum: 4,
            max_logical_depth: 0,
            max_cf_depth: 0,
            max_cognitive: 0,
            max_col: 0,
            max_tail_code_chars: 0,
            prod_bindings: 0,
            prod_golf_1char: 0,
            prod_unwraps: 0,
            unsafe_blocks: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrictReport {
    pub metrics: StrictMetrics,
    pub findings: Vec<StrictFinding>,
    pub veto_cap: f64,
    pub strict_score: f64,
}

impl StrictReport {
    pub fn veto_rules(&self) -> Vec<&'static str> {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Veto)
            .map(|f| f.rule)
            .collect()
    }

    pub fn formatted_vetoes(&self) -> Vec<String> {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Veto)
            .map(|f| format!("{} (L{}): {}", f.rule, f.line, f.detail))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct AntiGamingAudit {
    pub path: String,
    pub passed: bool,
    pub veto_cap: f64,
    pub indent_mode: &'static str,
    pub indent_quantum: usize,
    pub before_vetoes: Vec<String>,
    pub after_vetoes: Vec<String>,
    pub introduced_vetoes: Vec<String>,
}

/// Apply CH-V4 dynamic indent & 5-axis non-compensatory anti-gaming caps on top of a `v0.7.0` result.
/// Preserves `base.score` untouched and populates `strict_score`, `veto_cap`, and `triggered_vetoes`.
pub fn score_strict(path: &str, content: &str, base: &HygieneResult) -> HygieneResult {
    let report = audit_source(path, content, base.score, is_test_only_path(path));
    let mut out = base.clone();
    out.strict_score = Some(report.strict_score);
    out.veto_cap = Some(report.veto_cap);
    out.triggered_vetoes = report.formatted_vetoes();
    out
}

/// Compare anti-gaming compliance before and after an edit.
pub fn audit_anti_gaming(path: &str, before: &str, after: &str) -> AntiGamingAudit {
    let is_test = is_test_only_path(path);
    let rep_before = audit_source(path, before, 0.8808, is_test);
    let rep_after = audit_source(path, after, 0.8808, is_test);
    let before_rules: BTreeSet<&str> = rep_before.veto_rules().into_iter().collect();
    let introduced_vetoes: Vec<String> = rep_after
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Veto && !before_rules.contains(f.rule))
        .map(|f| format!("{} (L{}): {}", f.rule, f.line, f.detail))
        .collect();
    let after_vetoes = rep_after.formatted_vetoes();
    AntiGamingAudit {
        path: path.to_string(),
        passed: after_vetoes.is_empty(),
        veto_cap: rep_after.veto_cap,
        indent_mode: rep_after.metrics.indent_mode,
        indent_quantum: rep_after.metrics.indent_quantum,
        before_vetoes: rep_before.formatted_vetoes(),
        after_vetoes,
        introduced_vetoes,
    }
}

#[cfg(test)]
pub fn audit_source_with_test_flag(
    path: &str,
    content: &str,
    base_score: f64,
    is_test_only: bool,
) -> StrictReport {
    audit_source(path, content, base_score, is_test_only)
}

pub fn is_test_only_path(rel: &str) -> bool {
    let norm = rel.replace('\\', "/");
    let fname = norm.rsplit('/').next().unwrap_or(&norm);
    norm.starts_with("tests/")
        || norm.contains("/tests/")
        || fname == "testutil.rs"
        || fname == "test_util.rs"
        || fname.ends_with("_test.rs")
        || fname.ends_with("_tests.rs")
        || fname.ends_with("_test.go")
}

/// Discover `#[cfg(test)] mod <name>;` module targets in a Rust source file.
#[cfg(test)]
pub fn find_cfg_test_mod_decls(rel_path: &str, src: &str) -> Vec<String> {
    let lex = lex_source(src, false, false);
    let parent = rel_path.rsplit_once('/').map_or("", |(dir, _)| dir);
    let mut out = Vec::new();
    for idx in 0..lex.tokens.len() {
        let tok = &lex.tokens[idx];
        let is_mod = tok.in_test && tok.kind == TokKind::Ident && tok.text == "mod";
        let next_ident = lex.tokens.get(idx + 1).filter(|n| n.kind == TokKind::Ident);
        let has_semi = lex.tokens.get(idx + 2).is_some_and(|n| n.kind == TokKind::Semi);
        if let (true, Some(name_tok), true) = (is_mod, next_ident, has_semi) {
            let name = &name_tok.text;
            if parent.is_empty() {
                out.push(format!("{name}.rs"));
                out.push(format!("{name}/mod.rs"));
            } else {
                out.push(format!("{parent}/{name}.rs"));
                out.push(format!("{parent}/{name}/mod.rs"));
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LangMode {
    Rust,
    Go,
    Other,
}

fn lang_mode(path: &str) -> LangMode {
    if path.ends_with(".go") {
        LangMode::Go
    } else if path.ends_with(".py")
        || path.ends_with(".js")
        || path.ends_with(".mjs")
        || path.ends_with(".cjs")
        || path.ends_with(".ts")
        || path.ends_with(".tsx")
    {
        LangMode::Other
    } else {
        LangMode::Rust
    }
}

fn audit_source(path: &str, src: &str, base_score: f64, file_is_test_only: bool) -> StrictReport {
    let mode = lang_mode(path);
    let is_go = mode == LangMode::Go;
    let mut metrics = StrictMetrics::default();
    let mut findings = Vec::new();

    check_bidi_and_invisibles(src, &mut findings);
    let lex = lex_source(src, file_is_test_only, is_go);
    check_uts39_homoglyphs(&lex.tokens, &lex.lines, &mut findings);

    if !file_is_test_only {
        if mode == LangMode::Rust {
            check_attrs_and_macros(&lex.tokens, &mut findings);
            check_unsafe_and_unwraps(&lex.tokens, &lex.lines, &mut metrics, &mut findings);
        } else if mode == LangMode::Other {
            check_polyglot_suppressions(&lex.lines, &mut findings);
        }
        check_line_tails_blanks_and_comments(&lex.lines, &mut metrics, &mut findings);
        if mode != LangMode::Other {
            analyze_functions_and_golf(&lex.tokens, is_go, &mut metrics, &mut findings);
        }
    } else {
        metrics.total_lines = lex.lines.len().saturating_sub(1);
    }

    let (veto_cap, strict_score) = compute_calibration(base_score, &findings);
    StrictReport { metrics, findings, veto_cap, strict_score }
}

#[inline]
fn logit_f64(prob: f64) -> f64 {
    let clamped = prob.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

fn compute_calibration(base_score: f64, findings: &[StrictFinding]) -> (f64, f64) {
    let mut veto_cap = 1.0f64;
    let mut delta_slop = 0.0f64;
    let mut has_veto = false;

    for finding in findings {
        if finding.severity != Severity::Veto {
            continue;
        }
        has_veto = true;
        match finding.rule {
            "bidi_trojan_source" | "uts39_homoglyph" => {
                veto_cap = veto_cap.min(0.15);
                delta_slop += 3.5;
            }
            "prod_placeholder_macro" => {
                veto_cap = veto_cap.min(0.40);
                delta_slop += 2.0;
            }
            "lint_suppression" | "serde_flatten" | "unsafe_undocumented" | "std_lock_unwrap" => {
                veto_cap = veto_cap.min(0.45);
                delta_slop += 1.8;
            }
            _ => {
                veto_cap = veto_cap.min(0.50);
                delta_slop += 1.6;
            }
        }
    }
    if !has_veto {
        return (1.0, base_score);
    }
    let cal = sigmoid_f64(logit_f64(base_score) - delta_slop).min(veto_cap);
    let score = (cal * 10_000.0).round() / 10_000.0;
    (veto_cap, score)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokKind {
    Ident,
    StrLit,
    NumOrChar,
    Punct,
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Semi,
    Comma,
    Colon,
    FatArrow,
    AndAnd,
    OrOr,
    Pipe,
    Hash,
    Bang,
    Dot,
    Eq,
    Lt,
    Gt,
}

struct Token {
    kind: TokKind,
    text: String,
    line: usize,
    line_indent: usize,
    in_test: bool,
    in_os_cfg: bool,
}

#[derive(Default)]
struct LineScan {
    raw_cols: usize,
    line_indent: usize,
    leading_tabs: usize,
    leading_spaces: usize,
    in_test: bool,
    in_multiline_str: bool,
    has_code: bool,
    has_nondoc_comment: bool,
    has_doc_comment: bool,
    has_safety_comment: bool,
    tail_ident_chars: usize,
    tail_ops: usize,
    tail_has_cf: bool,
    mid_line_semi: bool,
    total_semis: usize,
    has_pad_smuggle: bool,
    comment_text: String,
    code_idents: Vec<String>,
}

struct LexOutput {
    tokens: Vec<Token>,
    lines: Vec<LineScan>,
}

#[derive(Clone, Copy, Default)]
struct AttrScope {
    armed: bool,
    active: bool,
    brace_depth: usize,
    paren_depth: usize,
}

impl AttrScope {
    fn is_in_scope(&self) -> bool {
        self.armed || self.active
    }

    fn arm(&mut self) {
        if !self.active {
            self.armed = true;
            self.brace_depth = 0;
            self.paren_depth = 0;
        }
    }

    fn step_before(&mut self, kind: TokKind) {
        if self.armed && !self.active && self.paren_depth == 0 && kind == TokKind::RBrace {
            self.armed = false;
        }
    }

    fn step_after(&mut self, kind: TokKind) {
        if self.active {
            match kind {
                TokKind::LBrace => self.brace_depth += 1,
                TokKind::RBrace => {
                    self.brace_depth = self.brace_depth.saturating_sub(1);
                    if self.brace_depth == 0 {
                        self.active = false;
                        self.armed = false;
                    }
                }
                _ => {}
            }
            return;
        }
        if !self.armed {
            return;
        }
        match kind {
            TokKind::LParen => self.paren_depth += 1,
            TokKind::RParen => self.paren_depth = self.paren_depth.saturating_sub(1),
            TokKind::LBrace if self.paren_depth == 0 => {
                self.active = true;
                self.brace_depth = 1;
            }
            TokKind::Semi | TokKind::Comma if self.paren_depth == 0 => {
                self.armed = false;
            }
            _ => {}
        }
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_ident_cont(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn is_cf_keyword(text: &str) -> bool {
    matches!(
        text,
        "if" | "else"
            | "match"
            | "switch"
            | "for"
            | "while"
            | "loop"
            | "return"
            | "unsafe"
            | "break"
            | "continue"
    )
}

fn is_tail_op(kind: TokKind, text: &str) -> bool {
    match kind {
        TokKind::FatArrow
        | TokKind::AndAnd
        | TokKind::OrOr
        | TokKind::Pipe
        | TokKind::Bang
        | TokKind::Dot
        | TokKind::Eq
        | TokKind::Lt
        | TokKind::Gt
        | TokKind::Semi
        | TokKind::LBrace
        | TokKind::RBrace => true,
        TokKind::Punct => matches!(
            text,
            "+" | "-" | "*" | "/" | "%" | "?" | "->" | "==" | "!=" | "<=" | ">=" | "+=" | "-="
                | "*=" | "/=" | ".." | "..="
        ),
        _ => false,
    }
}

fn init_line_scans(src: &str, file_is_test_only: bool) -> Vec<LineScan> {
    let mut lines = vec![LineScan::default()];
    for raw_line in src.lines() {
        let mut col = 0usize;
        let mut indent = 0usize;
        let mut tabs = 0usize;
        let mut spaces = 0usize;
        let mut seen_non_ws = false;
        for ch in raw_line.chars() {
            if ch == '\t' {
                let step = 4 - (col % 4);
                col += step;
                if !seen_non_ws {
                    indent += step;
                    tabs += 1;
                }
            } else {
                col += 1;
                if !seen_non_ws && ch == ' ' {
                    indent += 1;
                    spaces += 1;
                } else if !seen_non_ws {
                    seen_non_ws = true;
                }
            }
        }
        lines.push(LineScan {
            raw_cols: col,
            line_indent: indent,
            leading_tabs: tabs,
            leading_spaces: spaces,
            in_test: file_is_test_only,
            ..LineScan::default()
        });
    }
    if lines.len() == 1 {
        lines.push(LineScan { in_test: file_is_test_only, ..LineScan::default() });
    }
    lines
}

fn char_lit_end(chars: &[char], idx: usize) -> Option<usize> {
    if idx + 2 >= chars.len() || chars[idx] != '\'' {
        return None;
    }
    if chars[idx + 1] == '\\' {
        let mut pos = (idx + 3).min(chars.len());
        while pos < chars.len() && pos < idx + 16 && chars[pos] != '\'' && chars[pos] != '\n' {
            pos += 1;
        }
        return (chars.get(pos) == Some(&'\'')).then_some(pos + 1);
    }
    if chars[idx + 1] != '\n' && chars[idx + 1] != '\'' && chars[idx + 2] == '\'' {
        return Some(idx + 3);
    }
    None
}

fn raw_str_start(chars: &[char], idx: usize) -> Option<(usize, usize)> {
    let start_r = if chars[idx] == 'r' {
        idx
    } else if (chars[idx] == 'b' || chars[idx] == 'c') && chars.get(idx + 1) == Some(&'r') {
        idx + 1
    } else {
        return None;
    };
    if idx > 0 && is_ident_cont(chars[idx - 1]) {
        return None;
    }
    let mut pos = start_r + 1;
    while pos < chars.len() && chars[pos] == '#' {
        pos += 1;
    }
    let hashes = pos - (start_r + 1);
    (chars.get(pos) == Some(&'"')).then_some((pos, hashes))
}

fn find_matching_bracket(chars: &[char], open_idx: usize) -> usize {
    let mut depth = 0usize;
    let mut pos = open_idx;
    while pos < chars.len() {
        let ch = chars[pos];
        if ch == '"' {
            pos = skip_quoted_in_attr(chars, pos + 1);
            continue;
        }
        if ch == '/' && chars.get(pos + 1) == Some(&'/') {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }
        if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return pos;
            }
        }
        pos += 1;
    }
    chars.len().saturating_sub(1)
}

fn skip_quoted_in_attr(chars: &[char], mut pos: usize) -> usize {
    while pos < chars.len() {
        if chars[pos] == '\\' {
            pos = (pos + 2).min(chars.len());
            continue;
        }
        pos += 1;
        if chars[pos - 1] == '"' {
            break;
        }
    }
    pos
}

fn normalized_attr(chars: &[char], open_idx: usize, close_idx: usize) -> String {
    chars[open_idx + 1..close_idx.min(chars.len())]
        .iter()
        .copied()
        .filter(|c| !c.is_whitespace())
        .collect()
}

struct Lexer {
    chars: Vec<char>,
    lines: Vec<LineScan>,
    tokens: Vec<Token>,
    file_is_test_only: bool,
    is_go: bool,
    cfg_test: AttrScope,
    os_cfg: AttrScope,
    attr_end: Option<usize>,
    pos: usize,
    line: usize,
    col: usize,
    seen_code_on_line: bool,
    space_run: usize,
    block_cmt_pad: bool,
    saw_semi_on_line: bool,
    bracket_depth_on_line: usize,
}

impl Lexer {
    fn new(src: &str, file_is_test_only: bool, is_go: bool) -> Self {
        Self {
            chars: src.chars().collect(),
            lines: init_line_scans(src, file_is_test_only),
            tokens: Vec::new(),
            file_is_test_only,
            is_go,
            cfg_test: AttrScope::default(),
            os_cfg: AttrScope::default(),
            attr_end: None,
            pos: 0,
            line: 1,
            col: 0,
            seen_code_on_line: false,
            space_run: 0,
            block_cmt_pad: false,
            saw_semi_on_line: false,
            bracket_depth_on_line: 0,
        }
    }

    fn in_test(&self) -> bool {
        self.file_is_test_only || self.cfg_test.is_in_scope()
    }

    fn cur_indent(&self) -> usize {
        self.lines.get(self.line).map_or(0, |l| l.line_indent)
    }

    fn advance_newline(&mut self, mark_multiline_str: bool, in_test: bool) {
        self.line += 1;
        self.col = 0;
        self.seen_code_on_line = false;
        self.space_run = 0;
        self.block_cmt_pad = false;
        self.saw_semi_on_line = false;
        self.bracket_depth_on_line = 0;
        if let Some(ls) = self.lines.get_mut(self.line) {
            if in_test {
                ls.in_test = true;
            }
            if mark_multiline_str {
                ls.in_multiline_str = true;
                ls.has_code = true;
            }
        }
    }

    fn mark_code_token(&mut self, in_test: bool, allow_leading_pad: bool) {
        if let Some(ls) = self.lines.get_mut(self.line) {
            if in_test {
                ls.in_test = true;
            }
            ls.has_code = true;
            if !in_test {
                let lead_smuggle = allow_leading_pad && !self.seen_code_on_line && self.col >= MAX_COLS;
                let mid_smuggle = self.seen_code_on_line && self.space_run >= 8 && self.col >= MAX_COLS;
                let cmt_smuggle = self.block_cmt_pad && self.col >= MAX_COLS;
                if lead_smuggle || mid_smuggle || cmt_smuggle {
                    ls.has_pad_smuggle = true;
                }
                if self.saw_semi_on_line {
                    ls.mid_line_semi = true;
                }
            }
        }
        self.seen_code_on_line = true;
        self.space_run = 0;
    }

    fn push_tok(&mut self, kind: TokKind, text: String, line: usize, in_test: bool, in_os: bool) {
        let line_indent = self.lines.get(line).map_or(0, |l| l.line_indent);
        self.tokens.push(Token { kind, text, line, line_indent, in_test, in_os_cfg: in_os });
    }

    fn scan_line_comment(&mut self) {
        let start = self.pos;
        while self.pos < self.chars.len() && self.chars[self.pos] != '\n' {
            self.pos += 1;
        }
        let raw: String = self.chars[start..self.pos].iter().filter(|&&c| c != '\r').collect();
        let is_doc = (raw.starts_with("///") && !raw.starts_with("////")) || raw.starts_with("//!");
        let is_safety = raw.to_ascii_lowercase().contains("safety:");
        let in_test = self.in_test();
        if let Some(ls) = self.lines.get_mut(self.line) {
            if in_test {
                ls.in_test = true;
            }
            if is_doc {
                ls.has_doc_comment = true;
            } else if is_safety {
                ls.has_safety_comment = true;
            } else {
                ls.has_nondoc_comment = true;
                let body = raw.trim_start_matches('/').trim();
                if !ls.comment_text.is_empty() {
                    ls.comment_text.push(' ');
                }
                ls.comment_text.push_str(body);
            }
        }
    }

    fn scan_block_comment(&mut self) {
        let start_line = self.line;
        let start_col = self.col;
        let is_doc = !self.is_go
            && matches!(self.chars.get(self.pos + 2), Some('*') | Some('!'))
            && self.chars.get(self.pos + 3) != Some(&'/');
        let mut depth = 1usize;
        self.pos += 2;
        self.col += 2;
        let mut cmt_buf = String::new();

        while self.pos < self.chars.len() && depth > 0 {
            let ch = self.chars[self.pos];
            if ch == '\r' {
                self.pos += 1;
                continue;
            }
            if ch == '\n' {
                self.mark_block_comment_line(is_doc, false);
                let in_test = self.in_test();
                self.advance_newline(false, in_test);
                self.pos += 1;
                continue;
            }
            if !self.is_go && ch == '/' && self.chars.get(self.pos + 1) == Some(&'*') {
                depth += 1;
                self.pos += 2;
                self.col += 2;
            } else if ch == '*' && self.chars.get(self.pos + 1) == Some(&'/') {
                depth -= 1;
                self.pos += 2;
                self.col += 2;
            } else {
                cmt_buf.push(ch);
                self.pos += 1;
                self.col += 1;
            }
        }

        let is_safety = cmt_buf.to_ascii_lowercase().contains("safety:");
        self.mark_block_comment_line(is_doc, is_safety);
        if self.line == start_line && start_col < MAX_COLS && self.col.saturating_sub(start_col) >= 24 {
            self.block_cmt_pad = true;
        }
    }

    fn mark_block_comment_line(&mut self, is_doc: bool, is_safety: bool) {
        let in_test = self.in_test();
        if let Some(ls) = self.lines.get_mut(self.line) {
            if in_test {
                ls.in_test = true;
            }
            if is_doc {
                ls.has_doc_comment = true;
            } else if is_safety {
                ls.has_safety_comment = true;
            } else {
                ls.has_nondoc_comment = true;
            }
        }
    }

    fn peek_rust_attr(&mut self) {
        if self.is_go || self.chars[self.pos] != '#' || self.attr_end.is_some() {
            return;
        }
        if self.chars.get(self.pos + 1) == Some(&'[') {
            let close = find_matching_bracket(&self.chars, self.pos + 1);
            let norm = normalized_attr(&self.chars, self.pos + 1, close);
            if norm == "cfg(test)" {
                self.cfg_test.arm();
                if let Some(ls) = self.lines.get_mut(self.line) {
                    ls.in_test = true;
                }
            } else if norm == "cfg(windows)" || norm == "cfg(unix)" {
                self.os_cfg.arm();
            }
            self.attr_end = Some(close);
        } else if self.chars.get(self.pos + 1) == Some(&'!')
            && self.chars.get(self.pos + 2) == Some(&'[')
        {
            self.attr_end = Some(find_matching_bracket(&self.chars, self.pos + 2));
        }
    }

    fn scan_raw_string(&mut self, q_idx: usize, hashes: usize, in_test: bool, in_os: bool) {
        let tok_line = self.line;
        self.mark_code_token(in_test, false);
        let prefix_len = (q_idx + 1) - self.pos;
        self.pos = q_idx + 1;
        self.col += prefix_len;

        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch == '\r' {
                self.pos += 1;
                continue;
            }
            if ch == '\n' {
                self.advance_newline(true, in_test);
                self.pos += 1;
                continue;
            }
            if ch == '"' && self.raw_hashes_match(hashes) {
                self.pos += 1 + hashes;
                self.col += 1 + hashes;
                break;
            }
            self.pos += 1;
            self.col += 1;
        }
        self.push_tok(TokKind::StrLit, String::new(), tok_line, in_test, in_os);
    }

    fn raw_hashes_match(&self, hashes: usize) -> bool {
        (0..hashes).all(|h| self.chars.get(self.pos + 1 + h) == Some(&'#'))
    }

    fn scan_go_backtick(&mut self, in_test: bool, in_os: bool) {
        let tok_line = self.line;
        if let Some(ls) = self.lines.get_mut(self.line) {
            ls.has_code = true;
        }
        self.seen_code_on_line = true;
        self.space_run = 0;
        self.pos += 1;
        self.col += 1;
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            self.pos += 1;
            if ch == '\n' {
                self.advance_newline(true, in_test);
                continue;
            }
            self.col += 1;
            if ch == '`' {
                break;
            }
        }
        self.push_tok(TokKind::StrLit, String::new(), tok_line, in_test, in_os);
    }

    fn scan_quoted_string(&mut self, is_prefixed: bool, in_test: bool, in_os: bool) {
        let tok_line = self.line;
        self.mark_code_token(in_test, false);
        let step = if is_prefixed { 2 } else { 1 };
        self.pos += step;
        self.col += step;

        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch == '\r' {
                self.pos += 1;
                continue;
            }
            if ch == '\\' {
                if self.chars.get(self.pos + 1) == Some(&'\n') {
                    self.pos += 2;
                    self.advance_newline(true, in_test);
                } else {
                    self.pos = (self.pos + 2).min(self.chars.len());
                    self.col += 2;
                }
                continue;
            }
            if ch == '\n' {
                self.advance_newline(true, in_test);
                self.pos += 1;
                continue;
            }
            self.pos += 1;
            self.col += 1;
            if ch == '"' {
                break;
            }
        }
        self.push_tok(TokKind::StrLit, String::new(), tok_line, in_test, in_os);
    }

    fn try_scan_char_lit(&mut self, in_test: bool, in_os: bool) -> bool {
        let ch = self.chars[self.pos];
        let end = if !self.is_go
            && ch == 'b'
            && (self.pos == 0 || !is_ident_cont(self.chars[self.pos - 1]))
        {
            char_lit_end(&self.chars, self.pos + 1)
        } else if ch == '\'' {
            char_lit_end(&self.chars, self.pos)
        } else {
            None
        };
        let Some(end_idx) = end else { return false };
        if let Some(ls) = self.lines.get_mut(self.line) {
            if in_test {
                ls.in_test = true;
            }
            ls.has_code = true;
            if !in_test && self.saw_semi_on_line {
                ls.mid_line_semi = true;
            }
        }
        self.seen_code_on_line = true;
        self.space_run = 0;
        let text: String = self.chars[self.pos..end_idx].iter().collect();
        let len = end_idx - self.pos;
        self.push_tok(TokKind::NumOrChar, text, self.line, in_test, in_os);
        self.pos = end_idx;
        self.col += len;
        true
    }

    fn scan_code_token(&mut self, in_test: bool, in_os: bool, in_attr: bool) {
        self.mark_code_token(in_test, true);
        let ch = self.chars[self.pos];

        if !self.is_go
            && ch == 'r'
            && self.chars.get(self.pos + 1) == Some(&'#')
            && self.chars.get(self.pos + 2).is_some_and(|&nc| is_ident_start(nc))
        {
            self.scan_raw_ident(in_test, in_os);
            return;
        }
        if is_ident_start(ch) {
            self.scan_ident(in_test, in_os);
            return;
        }
        if ch.is_ascii_digit() {
            self.scan_number(in_test, in_os);
            return;
        }
        self.scan_punct(in_attr);
    }

    fn scan_raw_ident(&mut self, in_test: bool, in_os: bool) {
        let start_col = self.col;
        let mut end = self.pos + 2;
        while end < self.chars.len() && is_ident_cont(self.chars[end]) {
            end += 1;
        }
        let text: String = self.chars[self.pos + 2..end].iter().collect();
        let raw_len = end - self.pos;
        if !in_test && let Some(ls) = self.lines.get_mut(self.line) {
            if start_col + raw_len > MAX_COLS {
                ls.tail_ident_chars += raw_len.saturating_sub(MAX_COLS.saturating_sub(start_col));
            }
            if text.len() >= 3 {
                ls.code_idents.push(text.to_ascii_lowercase());
            }
        }
        self.push_tok(TokKind::Ident, text, self.line, in_test, in_os);
        self.pos = end;
        self.col += raw_len;
    }

    fn scan_ident(&mut self, in_test: bool, in_os: bool) {
        let start_col = self.col;
        let mut end = self.pos + 1;
        while end < self.chars.len() && is_ident_cont(self.chars[end]) {
            end += 1;
        }
        let text: String = self.chars[self.pos..end].iter().collect();
        let len = end - self.pos;
        if !in_test && let Some(ls) = self.lines.get_mut(self.line) {
            if start_col + len > MAX_COLS {
                ls.tail_ident_chars += len.saturating_sub(MAX_COLS.saturating_sub(start_col));
                if is_cf_keyword(&text) {
                    ls.tail_has_cf = true;
                }
            }
            if text.len() >= 3 {
                ls.code_idents.push(text.to_ascii_lowercase());
            }
        }
        self.push_tok(TokKind::Ident, text, self.line, in_test, in_os);
        self.pos = end;
        self.col += len;
    }

    fn scan_number(&mut self, in_test: bool, in_os: bool) {
        let start_col = self.col;
        let mut end = self.pos + 1;
        while end < self.chars.len()
            && (self.chars[end].is_ascii_alphanumeric() || self.chars[end] == '_')
        {
            end += 1;
        }
        let text: String = self.chars[self.pos..end].iter().collect();
        let len = end - self.pos;
        if !in_test && start_col + len > MAX_COLS && let Some(ls) = self.lines.get_mut(self.line) {
            ls.tail_ident_chars += len.saturating_sub(MAX_COLS.saturating_sub(start_col));
        }
        self.push_tok(TokKind::NumOrChar, text, self.line, in_test, in_os);
        self.pos = end;
        self.col += len;
    }

    fn scan_punct(&mut self, in_attr: bool) {
        let start_col = self.col;
        let ch = self.chars[self.pos];
        let next_c = self.chars.get(self.pos + 1).copied();
        let third_c = self.chars.get(self.pos + 2).copied();
        let (kind, text, step) = classify_punct(ch, next_c, third_c);

        if !in_attr && !self.is_go {
            self.cfg_test.step_before(kind);
            self.os_cfg.step_before(kind);
        }
        let tok_in_test = self.in_test();
        let tok_in_os = self.os_cfg.is_in_scope();

        if kind == TokKind::LBracket {
            self.bracket_depth_on_line += 1;
        } else if kind == TokKind::RBracket {
            self.bracket_depth_on_line = self.bracket_depth_on_line.saturating_sub(1);
        } else if kind == TokKind::Semi && self.bracket_depth_on_line == 0 {
            self.saw_semi_on_line = true;
            if !tok_in_test && let Some(ls) = self.lines.get_mut(self.line) {
                ls.total_semis += 1;
            }
        }
        if !tok_in_test && start_col >= MAX_COLS && is_tail_op(kind, &text) {
            if let Some(ls) = self.lines.get_mut(self.line) {
                ls.tail_ops += 1;
            }
        }

        let line_indent = self.cur_indent();
        self.tokens.push(Token {
            kind,
            text,
            line: self.line,
            line_indent,
            in_test: tok_in_test,
            in_os_cfg: tok_in_os,
        });
        if !in_attr && !self.is_go {
            self.cfg_test.step_after(kind);
            self.os_cfg.step_after(kind);
        }
        self.pos += step;
        self.col += step;
    }

    fn step_one(&mut self) {
        let ch = self.chars[self.pos];
        if ch == '\r' {
            self.pos += 1;
            return;
        }
        if ch == '\n' {
            self.pos += 1;
            let in_test = self.in_test();
            self.advance_newline(false, in_test);
            return;
        }
        if ch == ' ' || ch == '\t' {
            let step = if ch == '\t' { 4 - (self.col % 4) } else { 1 };
            self.col += step;
            if self.seen_code_on_line {
                self.space_run += step;
            }
            self.pos += 1;
            return;
        }
        if self.attr_end.is_some_and(|end| self.pos > end) {
            self.attr_end = None;
        }
        if ch == '/' && self.chars.get(self.pos + 1) == Some(&'/') {
            self.scan_line_comment();
            return;
        }
        if ch == '/' && self.chars.get(self.pos + 1) == Some(&'*') {
            self.scan_block_comment();
            return;
        }
        self.step_non_comment(ch);
    }

    fn step_non_comment(&mut self, ch: char) {
        self.peek_rust_attr();
        let in_test = self.in_test();
        let in_os = self.os_cfg.is_in_scope();
        let in_attr = self.attr_end.is_some();

        if !self.is_go && let Some((q_idx, hashes)) = raw_str_start(&self.chars, self.pos) {
            self.scan_raw_string(q_idx, hashes, in_test, in_os);
            return;
        }
        if self.is_go && ch == '`' {
            self.scan_go_backtick(in_test, in_os);
            return;
        }
        let is_prefixed = !self.is_go
            && (ch == 'b' || ch == 'c')
            && self.chars.get(self.pos + 1) == Some(&'"')
            && (self.pos == 0 || !is_ident_cont(self.chars[self.pos - 1]));
        if ch == '"' || is_prefixed {
            self.scan_quoted_string(is_prefixed, in_test, in_os);
            return;
        }
        if self.try_scan_char_lit(in_test, in_os) {
            return;
        }
        self.scan_code_token(in_test, in_os, in_attr);
    }

    fn run(mut self) -> LexOutput {
        while self.pos < self.chars.len() {
            self.step_one();
        }
        LexOutput { tokens: self.tokens, lines: self.lines }
    }
}

fn classify_punct(ch: char, next_c: Option<char>, third_c: Option<char>) -> (TokKind, String, usize) {
    match (ch, next_c) {
        ('=', Some('>')) => (TokKind::FatArrow, "=>".into(), 2),
        ('&', Some('&')) => (TokKind::AndAnd, "&&".into(), 2),
        ('|', Some('|')) => (TokKind::OrOr, "||".into(), 2),
        ('-', Some('>')) => (TokKind::Punct, "->".into(), 2),
        ('=', Some('=')) => (TokKind::Punct, "==".into(), 2),
        ('!', Some('=')) => (TokKind::Punct, "!=".into(), 2),
        ('<', Some('=')) => (TokKind::Punct, "<=".into(), 2),
        ('>', Some('=')) => (TokKind::Punct, ">=".into(), 2),
        ('+', Some('=')) => (TokKind::Punct, "+=".into(), 2),
        ('-', Some('=')) => (TokKind::Punct, "-=".into(), 2),
        ('*', Some('=')) => (TokKind::Punct, "*=".into(), 2),
        ('/', Some('=')) => (TokKind::Punct, "/=".into(), 2),
        (':', Some(':')) => (TokKind::Punct, "::".into(), 2),
        ('.', Some('.')) if third_c == Some('=') => (TokKind::Punct, "..=".into(), 3),
        ('.', Some('.')) => (TokKind::Punct, "..".into(), 2),
        ('{', _) => (TokKind::LBrace, "{".into(), 1),
        ('}', _) => (TokKind::RBrace, "}".into(), 1),
        ('(', _) => (TokKind::LParen, "(".into(), 1),
        (')', _) => (TokKind::RParen, ")".into(), 1),
        ('[', _) => (TokKind::LBracket, "[".into(), 1),
        (']', _) => (TokKind::RBracket, "]".into(), 1),
        (';', _) => (TokKind::Semi, ";".into(), 1),
        (',', _) => (TokKind::Comma, ",".into(), 1),
        (':', _) => (TokKind::Colon, ":".into(), 1),
        ('|', _) => (TokKind::Pipe, "|".into(), 1),
        ('#', _) => (TokKind::Hash, "#".into(), 1),
        ('!', _) => (TokKind::Bang, "!".into(), 1),
        ('.', _) => (TokKind::Dot, ".".into(), 1),
        ('=', _) => (TokKind::Eq, "=".into(), 1),
        ('<', _) => (TokKind::Lt, "<".into(), 1),
        ('>', _) => (TokKind::Gt, ">".into(), 1),
        _ => (TokKind::Punct, ch.to_string(), 1),
    }
}

fn lex_source(src: &str, file_is_test_only: bool, is_go: bool) -> LexOutput {
    Lexer::new(src, file_is_test_only, is_go).run()
}

fn detect_dynamic_indent(lines: &[LineScan]) -> (&'static str, usize) {
    let mut tab_lines = 0usize;
    let mut space_lines = 0usize;
    let mut deltas = [0usize; 9];
    let mut prev_sp = 0usize;

    for ls in lines.iter().skip(1) {
        if ls.in_test || !ls.has_code || ls.in_multiline_str {
            continue;
        }
        if ls.leading_tabs > 0 {
            tab_lines += 1;
        } else {
            let sp = ls.leading_spaces;
            if sp > 0 {
                space_lines += 1;
            }
            let diff = sp.abs_diff(prev_sp);
            if (1..=8).contains(&diff) {
                deltas[diff] += 1;
            }
            prev_sp = sp;
        }
    }
    if tab_lines >= space_lines && tab_lines > 0 {
        return ("tabs", 1);
    }
    let mut best_q = 4usize;
    let mut best_cnt = 0usize;
    for q in [4usize, 2, 8, 3] {
        if deltas[q] > best_cnt {
            best_cnt = deltas[q];
            best_q = q;
        }
    }
    ("spaces", best_q)
}

fn check_bidi_and_invisibles(src: &str, findings: &mut Vec<StrictFinding>) {
    for (idx, line) in src.lines().enumerate() {
        let ln = idx + 1;
        for (ci, ch) in line.chars().enumerate() {
            let desc = match ch {
                '\u{202A}' => Some("U+202A LRE"),
                '\u{202B}' => Some("U+202B RLE"),
                '\u{202C}' => Some("U+202C PDF"),
                '\u{202D}' => Some("U+202D LRO"),
                '\u{202E}' => Some("U+202E RLO (Trojan Source)"),
                '\u{2066}' => Some("U+2066 LRI (Trojan Source)"),
                '\u{2067}' => Some("U+2067 RLI (Trojan Source)"),
                '\u{2068}' => Some("U+2068 FSI (Trojan Source)"),
                '\u{2069}' => Some("U+2069 PDI (Trojan Source)"),
                '\u{200E}' => Some("U+200E LRM"),
                '\u{200F}' => Some("U+200F RLM"),
                '\u{061C}' => Some("U+061C ALM"),
                '\u{200B}' => Some("U+200B ZWSP"),
                '\u{200C}' => Some("U+200C ZWNJ"),
                '\u{200D}' => Some("U+200D ZWJ"),
                '\u{2060}' => Some("U+2060 WJ"),
                '\u{FEFF}' if !(ln == 1 && ci == 0) => Some("U+FEFF mid-file BOM/ZWNBSP"),
                _ => None,
            };
            if let Some(tag) = desc {
                findings.push(StrictFinding {
                    severity: Severity::Veto,
                    rule: "bidi_trojan_source",
                    line: ln,
                    detail: format!("forbidden invisible/BiDi control {tag} (CVE-2021-42574)"),
                });
                break;
            }
        }
    }
}

fn uts39_confusable_desc(ch: char) -> Option<(&'static str, char)> {
    match ch {
        '\u{0410}' => Some(("Cyrillic", 'A')),
        '\u{0412}' => Some(("Cyrillic", 'B')),
        '\u{0421}' => Some(("Cyrillic", 'C')),
        '\u{0415}' => Some(("Cyrillic", 'E')),
        '\u{041D}' => Some(("Cyrillic", 'H')),
        '\u{0406}' => Some(("Cyrillic", 'I')),
        '\u{0408}' => Some(("Cyrillic", 'J')),
        '\u{041A}' => Some(("Cyrillic", 'K')),
        '\u{041C}' => Some(("Cyrillic", 'M')),
        '\u{041E}' => Some(("Cyrillic", 'O')),
        '\u{0420}' => Some(("Cyrillic", 'P')),
        '\u{0405}' => Some(("Cyrillic", 'S')),
        '\u{0422}' => Some(("Cyrillic", 'T')),
        '\u{0425}' => Some(("Cyrillic", 'X')),
        '\u{0423}' => Some(("Cyrillic", 'Y')),
        '\u{0430}' => Some(("Cyrillic", 'a')),
        '\u{0441}' => Some(("Cyrillic", 'c')),
        '\u{0435}' => Some(("Cyrillic", 'e')),
        '\u{0456}' => Some(("Cyrillic", 'i')),
        '\u{0458}' => Some(("Cyrillic", 'j')),
        '\u{043E}' => Some(("Cyrillic", 'o')),
        '\u{0440}' => Some(("Cyrillic", 'p')),
        '\u{0455}' => Some(("Cyrillic", 's')),
        '\u{0445}' => Some(("Cyrillic", 'x')),
        '\u{0443}' => Some(("Cyrillic", 'y')),
        '\u{0391}' => Some(("Greek", 'A')),
        '\u{0392}' => Some(("Greek", 'B')),
        '\u{0395}' => Some(("Greek", 'E')),
        '\u{0396}' => Some(("Greek", 'Z')),
        '\u{0397}' => Some(("Greek", 'H')),
        '\u{0399}' => Some(("Greek", 'I')),
        '\u{039A}' => Some(("Greek", 'K')),
        '\u{039C}' => Some(("Greek", 'M')),
        '\u{039D}' => Some(("Greek", 'N')),
        '\u{039F}' => Some(("Greek", 'O')),
        '\u{03A1}' => Some(("Greek", 'P')),
        '\u{03A4}' => Some(("Greek", 'T')),
        '\u{03A5}' => Some(("Greek", 'Y')),
        '\u{03A7}' => Some(("Greek", 'X')),
        '\u{03B1}' => Some(("Greek", 'a')),
        '\u{03B9}' => Some(("Greek", 'i')),
        '\u{03BA}' => Some(("Greek", 'k')),
        '\u{03BD}' => Some(("Greek", 'v')),
        '\u{03BF}' => Some(("Greek", 'o')),
        '\u{03C1}' => Some(("Greek", 'p')),
        '\u{03C4}' => Some(("Greek", 't')),
        '\u{03C5}' => Some(("Greek", 'u')),
        '\u{03C7}' => Some(("Greek", 'x')),
        '\u{037E}' => Some(("Greek question mark", ';')),
        _ => None,
    }
}

fn is_cyrillic_or_greek(ch: char) -> bool {
    let cp = ch as u32;
    (0x0370..=0x03FF).contains(&cp)
        || (0x1F00..=0x1FFF).contains(&cp)
        || (0x0400..=0x052F).contains(&cp)
}

fn check_uts39_homoglyphs(
    tokens: &[Token],
    lines: &[LineScan],
    findings: &mut Vec<StrictFinding>,
) {
    let mut flagged = BTreeSet::new();
    for tok in tokens {
        if tok.kind == TokKind::StrLit || tok.kind == TokKind::NumOrChar {
            continue;
        }
        if let Some(detail) = find_homoglyph_in_token(&tok.text) && flagged.insert(tok.line) {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "uts39_homoglyph",
                line: tok.line,
                detail,
            });
        }
    }
    for (ln, ls) in lines.iter().enumerate().skip(1) {
        if ls.comment_text.is_empty() || flagged.contains(&ln) {
            continue;
        }
        if let Some(ch) = find_mixed_script_comment_char(&ls.comment_text) {
            flagged.insert(ln);
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "uts39_homoglyph",
                line: ln,
                detail: format!(
                    "mixed Latin + Cyrillic/Greek homoglyph U+{:04X} in comment word",
                    ch as u32
                ),
            });
        }
    }
}

fn find_homoglyph_in_token(text: &str) -> Option<String> {
    for ch in text.chars() {
        if let Some((script, lookalike)) = uts39_confusable_desc(ch) {
            return Some(format!(
                "{script} confusable U+{:04X} (looks like '{lookalike}') in code token",
                ch as u32
            ));
        }
        if is_cyrillic_or_greek(ch) {
            return Some(format!("Cyrillic/Greek script char U+{:04X} in code token", ch as u32));
        }
    }
    None
}

fn find_mixed_script_comment_char(comment: &str) -> Option<char> {
    for word in comment.split_whitespace() {
        if word.chars().any(|c| c.is_ascii_alphabetic())
            && let Some(ch) = word.chars().find(|&c| is_cyrillic_or_greek(c))
        {
            return Some(ch);
        }
    }
    None
}

fn check_polyglot_suppressions(lines: &[LineScan], findings: &mut Vec<StrictFinding>) {
    for (ln, ls) in lines.iter().enumerate().skip(1) {
        if ls.in_test || ls.comment_text.is_empty() {
            continue;
        }
        let lower = ls.comment_text.to_ascii_lowercase();
        let bad = lower.contains("@ts-ignore")
            || lower.contains("@ts-nocheck")
            || lower.contains("eslint-disable")
            || lower.contains("noqa");
        if bad {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "lint_suppression",
                line: ln,
                detail: "forbidden lint/typecheck suppression comment".into(),
            });
        }
    }
}

fn check_attrs_and_macros(tokens: &[Token], findings: &mut Vec<StrictFinding>) {
    let mut idx = 0usize;
    while idx < tokens.len() {
        let tok = &tokens[idx];
        if tok.in_test {
            idx += 1;
            continue;
        }
        if tok.kind == TokKind::Hash && let Some(next_idx) = inspect_attr(tokens, idx, findings) {
            idx = next_idx;
            continue;
        }
        if is_placeholder_macro(tokens, idx) {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "prod_placeholder_macro",
                line: tok.line,
                detail: format!("forbidden production `{}!()` placeholder", tok.text),
            });
        }
        idx += 1;
    }
}

fn inspect_attr(tokens: &[Token], idx: usize, findings: &mut Vec<StrictFinding>) -> Option<usize> {
    let open_idx = if tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::LBracket) {
        idx + 1
    } else if tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::Bang)
        && tokens.get(idx + 2).is_some_and(|t| t.kind == TokKind::LBracket)
    {
        idx + 2
    } else {
        return None;
    };

    let mut depth = 1usize;
    let mut pos = open_idx + 1;
    while pos < tokens.len() && depth > 0 {
        if tokens[pos].kind == TokKind::LBracket {
            depth += 1;
        } else if tokens[pos].kind == TokKind::RBracket {
            depth -= 1;
        }
        pos += 1;
    }
    let inner = &tokens[open_idx + 1..pos.saturating_sub(1)];
    check_attr_inner(inner, tokens[idx].line, findings);
    Some(pos)
}

fn check_attr_inner(inner: &[Token], line: usize, findings: &mut Vec<StrictFinding>) {
    let Some(first) = inner.first() else { return };
    if first.kind != TokKind::Ident {
        return;
    }
    match first.text.as_str() {
        "allow" => findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "lint_suppression",
            line,
            detail: "forbidden allow-attribute lint suppression".into(),
        }),
        "cfg_attr" if inner.iter().any(|t| t.kind == TokKind::Ident && t.text == "allow") => {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "lint_suppression",
                line,
                detail: "forbidden cfg_attr allow lint suppression".into(),
            });
        }
        "expect" => {
            let has_reason = inner.windows(2).any(|w| {
                w[0].kind == TokKind::Ident && w[0].text == "reason" && w[1].kind == TokKind::Eq
            });
            if !has_reason {
                findings.push(StrictFinding {
                    severity: Severity::Veto,
                    rule: "lint_suppression",
                    line,
                    detail: "expect attribute missing explicit `reason = ...`".into(),
                });
            }
        }
        "serde" if inner.iter().any(|t| t.kind == TokKind::Ident && t.text == "flatten") => {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "serde_flatten",
                line,
                detail: "forbidden serde flatten attribute".into(),
            });
        }
        _ => {}
    }
}

fn is_placeholder_macro(tokens: &[Token], idx: usize) -> bool {
    let tok = &tokens[idx];
    tok.kind == TokKind::Ident
        && (tok.text == "todo" || tok.text == "unimplemented")
        && tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::Bang)
        && tokens
            .get(idx + 2)
            .is_some_and(|t| matches!(t.kind, TokKind::LParen | TokKind::LBracket | TokKind::LBrace))
}

fn check_unsafe_and_unwraps(
    tokens: &[Token],
    lines: &[LineScan],
    metrics: &mut StrictMetrics,
    findings: &mut Vec<StrictFinding>,
) {
    for idx in 0..tokens.len() {
        let tok = &tokens[idx];
        if tok.in_test {
            continue;
        }
        if is_dot_unwrap(tokens, idx) {
            metrics.prod_unwraps += 1;
            if let Some(lock_fn) = preceded_by_lock_call(tokens, idx) {
                findings.push(StrictFinding {
                    severity: Severity::Veto,
                    rule: "std_lock_unwrap",
                    line: tok.line,
                    detail: format!("forbidden `.{lock_fn}().unwrap()` lock-poison panic in prod"),
                });
            }
        }
        if tok.kind == TokKind::Ident && tok.text == "unsafe" {
            check_unsafe_site(tokens, idx, lines, metrics, findings);
        }
    }

    if metrics.prod_unwraps >= 10 {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "prod_unwrap_spam",
            line: 1,
            detail: format!("excessive production `.unwrap()` count: {}", metrics.prod_unwraps),
        });
    } else if metrics.prod_unwraps > 0 {
        findings.push(StrictFinding {
            severity: Severity::Warning,
            rule: "prod_unwrap_advisory",
            line: 1,
            detail: format!("{} invariant-guarded prod `.unwrap()` call(s)", metrics.prod_unwraps),
        });
    }
}

fn is_dot_unwrap(tokens: &[Token], idx: usize) -> bool {
    tokens[idx].kind == TokKind::Dot
        && tokens
            .get(idx + 1)
            .is_some_and(|t| t.kind == TokKind::Ident && t.text == "unwrap")
        && tokens.get(idx + 2).is_some_and(|t| t.kind == TokKind::LParen)
}

fn preceded_by_lock_call(tokens: &[Token], idx: usize) -> Option<&str> {
    if idx < 4 {
        return None;
    }
    let ok = tokens[idx - 4].kind == TokKind::Dot
        && tokens[idx - 3].kind == TokKind::Ident
        && matches!(tokens[idx - 3].text.as_str(), "lock" | "read" | "write")
        && tokens[idx - 2].kind == TokKind::LParen
        && tokens[idx - 1].kind == TokKind::RParen;
    ok.then_some(tokens[idx - 3].text.as_str())
}

fn check_unsafe_site(
    tokens: &[Token],
    idx: usize,
    lines: &[LineScan],
    metrics: &mut StrictMetrics,
    findings: &mut Vec<StrictFinding>,
) {
    // Edition 2024 `#[unsafe(export_name = ...)]` attributes and `unsafe extern "C" { ... }`
    // FFI declarations do not perform unchecked operations at the declaration site.
    let is_edition2024_unsafe_attr = idx > 0
        && tokens[idx - 1].kind == TokKind::LBracket
        && tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::LParen);
    let is_edition2024_unsafe_extern = tokens
        .get(idx + 1)
        .is_some_and(|t| t.kind == TokKind::Ident && t.text == "extern");
    if is_edition2024_unsafe_attr || is_edition2024_unsafe_extern {
        return;
    }

    let tok = &tokens[idx];
    metrics.unsafe_blocks += 1;
    let lo = tok.line.saturating_sub(3).max(1);
    let has_safety = (lo..=tok.line)
        .any(|ln| lines.get(ln).is_some_and(|ls| ls.has_safety_comment));
    if has_safety {
        return;
    }

    let has_ub = unsafe_block_contains_ub(tokens, idx);
    if tok.in_os_cfg && !has_ub {
        findings.push(StrictFinding {
            severity: Severity::Warning,
            rule: "unsafe_os_ffi_missing_safety_doc",
            line: tok.line,
            detail: "OS-FFI `unsafe` inside `#[cfg(windows|unix)]` lacks `// SAFETY:` doc".into(),
        });
    } else {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "unsafe_undocumented",
            line: tok.line,
            detail: "undocumented `unsafe` block/fn missing `// SAFETY:` comment".into(),
        });
    }
}

fn unsafe_block_contains_ub(tokens: &[Token], idx: usize) -> bool {
    const UB_INTRINSICS: &[&str] = &[
        "transmute",
        "transmute_copy",
        "get_unchecked",
        "get_unchecked_mut",
        "from_raw_parts",
        "from_raw_parts_mut",
        "set_len",
        "assume_init",
        "uninitialized",
        "unreachable_unchecked",
    ];
    let mut pos = idx + 1;
    while pos < tokens.len()
        && tokens[pos].kind != TokKind::LBrace
        && tokens[pos].kind != TokKind::Semi
    {
        pos += 1;
    }
    if tokens.get(pos).is_none_or(|t| t.kind != TokKind::LBrace) {
        return false;
    }
    let mut depth = 1usize;
    let mut scan = pos + 1;
    while scan < tokens.len() && depth > 0 {
        match tokens[scan].kind {
            TokKind::LBrace => depth += 1,
            TokKind::RBrace => depth -= 1,
            TokKind::Ident => {
                if UB_INTRINSICS.contains(&tokens[scan].text.as_str()) {
                    return true;
                }
                if tokens[scan].text == "static"
                    && tokens
                        .get(scan + 1)
                        .is_some_and(|t| t.kind == TokKind::Ident && t.text == "mut")
                {
                    return true;
                }
            }
            _ => {}
        }
        scan += 1;
    }
    false
}

#[derive(Default)]
struct BlankCommentAcc {
    consec_blank: usize,
    max_consec_blank: usize,
    max_consec_blank_line: usize,
    nonblank_run: usize,
    alt_blank_run: usize,
    max_alt_blank: usize,
    max_alt_blank_line: usize,
    nondoc_pure_comments: usize,
    nonblank_prod_lines: usize,
    echo_pairs: usize,
    first_echo_line: usize,
    commented_code_lines: usize,
    first_cmt_code_line: usize,
}

impl BlankCommentAcc {
    fn reset_test_boundary(&mut self) {
        self.consec_blank = 0;
        self.nonblank_run = 0;
        self.alt_blank_run = 0;
    }

    fn record_blank(&mut self, ln: usize, metrics: &mut StrictMetrics) {
        metrics.blank_prod_lines += 1;
        self.consec_blank += 1;
        if self.consec_blank > self.max_consec_blank {
            self.max_consec_blank = self.consec_blank;
            self.max_consec_blank_line = ln;
        }
        if self.nonblank_run == 1 {
            self.alt_blank_run += 1;
            if self.alt_blank_run > self.max_alt_blank {
                self.max_alt_blank = self.alt_blank_run;
                self.max_alt_blank_line = ln;
            }
        } else if self.nonblank_run >= 2 {
            self.alt_blank_run = 0;
        }
        self.nonblank_run = 0;
    }

    fn record_nonblank(&mut self) {
        self.consec_blank = 0;
        self.nonblank_run += 1;
        if self.nonblank_run >= 2 {
            self.alt_blank_run = 0;
        }
        self.nonblank_prod_lines += 1;
    }

    fn record_pure_nondoc_comment(&mut self, lines: &[LineScan], ln: usize, body: &str) {
        self.nondoc_pure_comments += 1;
        if looks_like_commented_rust_code(body) {
            self.commented_code_lines += 1;
            if self.first_cmt_code_line == 0 {
                self.first_cmt_code_line = ln;
            }
        }
        if is_comment_echo_pair(lines, ln, body) {
            self.echo_pairs += 1;
            if self.first_echo_line == 0 {
                self.first_echo_line = ln;
            }
        }
    }
}

fn check_line_tails_blanks_and_comments(
    lines: &[LineScan],
    metrics: &mut StrictMetrics,
    findings: &mut Vec<StrictFinding>,
) {
    let (indent_mode, indent_quantum) = detect_dynamic_indent(lines);
    metrics.indent_mode = indent_mode;
    metrics.indent_quantum = indent_quantum;
    let mut acc = BlankCommentAcc {
        max_consec_blank_line: 1,
        max_alt_blank_line: 1,
        ..BlankCommentAcc::default()
    };

    for (ln, ls) in lines.iter().enumerate().skip(1) {
        metrics.total_lines += 1;
        if ls.in_test {
            acc.reset_test_boundary();
            continue;
        }
        metrics.prod_lines += 1;
        metrics.max_col = metrics.max_col.max(ls.raw_cols);

        if ls.has_code && !ls.in_multiline_str {
            let denom = if indent_mode == "tabs" { 4 } else { indent_quantum.max(1) };
            let logical_d = ls.leading_tabs + (ls.leading_spaces / denom);
            metrics.max_logical_depth = metrics.max_logical_depth.max(logical_d);
        }

        check_dense_tail_line(ln, ls, metrics, findings);
        let is_blank = !ls.has_code
            && !ls.has_nondoc_comment
            && !ls.has_doc_comment
            && !ls.has_safety_comment
            && !ls.in_multiline_str;
        if is_blank {
            acc.record_blank(ln, metrics);
        } else {
            acc.record_nonblank();
        }
        if ls.has_nondoc_comment && !ls.has_code && !ls.has_doc_comment && !ls.has_safety_comment {
            acc.record_pure_nondoc_comment(lines, ln, ls.comment_text.trim());
        }
    }

    metrics.nondoc_comment_lines = acc.nondoc_pure_comments;
    emit_blank_and_comment_vetoes(metrics, &acc, findings);
}

fn check_dense_tail_line(
    ln: usize,
    ls: &LineScan,
    metrics: &mut StrictMetrics,
    findings: &mut Vec<StrictFinding>,
) {
    if ls.raw_cols <= MAX_COLS || ls.in_multiline_str {
        return;
    }
    let tail_score = ls.tail_ident_chars + ls.tail_ops;
    metrics.max_tail_code_chars = metrics.max_tail_code_chars.max(tail_score);
    let dense_tail = ls.tail_has_cf
        || ls.mid_line_semi
        || ls.total_semis >= 2
        || ls.has_pad_smuggle
        || (ls.tail_ops >= 3 && tail_score >= 16)
        || ls.tail_ident_chars >= 24;
    if dense_tail {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "hidden_dense_tail",
            line: ln,
            detail: format!(
                "line width {} > {MAX_COLS} with dense/smuggled code tail (idents={}, ops={}, cf={}, mid_semi={}, pad={})",
                ls.raw_cols, ls.tail_ident_chars, ls.tail_ops, ls.tail_has_cf, ls.mid_line_semi, ls.has_pad_smuggle
            ),
        });
    }
}

fn is_comment_echo_pair(lines: &[LineScan], ln: usize, body: &str) -> bool {
    let c_words = extract_meaningful_words(body);
    if c_words.len() < 2 {
        return false;
    }
    for next_ln in (ln + 1)..=(ln + 2).min(lines.len().saturating_sub(1)) {
        let next_ls = &lines[next_ln];
        if !next_ls.in_test && next_ls.has_code {
            let code_set: BTreeSet<&str> = next_ls.code_idents.iter().map(String::as_str).collect();
            let shared = c_words.iter().filter(|w| code_set.contains(w.as_str())).count();
            return shared >= 2 && (shared * 100) / c_words.len() >= 50;
        }
    }
    false
}

fn emit_blank_and_comment_vetoes(
    metrics: &StrictMetrics,
    acc: &BlankCommentAcc,
    findings: &mut Vec<StrictFinding>,
) {
    if metrics.prod_lines >= 12 {
        if acc.max_consec_blank >= 3 {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "blank_line_stuffing",
                line: acc.max_consec_blank_line,
                detail: format!("{} consecutive blank lines (>= 3)", acc.max_consec_blank),
            });
        } else if metrics.blank_prod_lines >= 6
            && (metrics.blank_prod_lines as f64 / metrics.prod_lines as f64) > 0.32
        {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "blank_line_stuffing",
                line: 1,
                detail: format!(
                    "excessive blank-line ratio {}/{} ({:.1}% > 32.0%)",
                    metrics.blank_prod_lines,
                    metrics.prod_lines,
                    (metrics.blank_prod_lines as f64 / metrics.prod_lines as f64) * 100.0
                ),
            });
        } else if acc.max_alt_blank >= 5 {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "blank_line_stuffing",
                line: acc.max_alt_blank_line,
                detail: format!("{} consecutive interleaved single-line + blank pairs", acc.max_alt_blank),
            });
        }
    }

    if acc.echo_pairs >= 3 {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "comment_echo_stuffing",
            line: acc.first_echo_line,
            detail: format!("{} non-doc comments echoing adjacent code identifiers", acc.echo_pairs),
        });
    } else if acc.commented_code_lines >= 4 {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "comment_echo_stuffing",
            line: acc.first_cmt_code_line,
            detail: format!("{} lines of commented-out code", acc.commented_code_lines),
        });
    } else if acc.nonblank_prod_lines >= 10
        && acc.nondoc_pure_comments >= 6
        && (acc.nondoc_pure_comments as f64 / acc.nonblank_prod_lines as f64) > 0.35
    {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "comment_echo_stuffing",
            line: 1,
            detail: format!(
                "excessive non-doc comment stuffing {}/{} ({:.1}% > 35.0%)",
                acc.nondoc_pure_comments,
                acc.nonblank_prod_lines,
                (acc.nondoc_pure_comments as f64 / acc.nonblank_prod_lines as f64) * 100.0
            ),
        });
    }
}

fn looks_like_commented_rust_code(body: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "let ", "fn ", "pub fn ", "pub struct ", "if ", "for ", "while ", "match ", "return ",
        "use ", "struct ", "impl ", "const ",
    ];
    PREFIXES.iter().any(|p| body.starts_with(p))
        || (body.ends_with(';') && (body.contains('=') || body.contains('(')))
}

fn extract_meaningful_words(text: &str) -> BTreeSet<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "from", "that", "this", "into", "not", "are", "use", "all",
        "can", "has", "was", "but", "when", "then", "else", "only", "each", "also",
    ];
    let mut out = BTreeSet::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 3 && !STOP.contains(&cur.as_str()) {
                out.insert(cur.clone());
            }
            cur.clear();
        }
    }
    if cur.len() >= 3 && !STOP.contains(&cur.as_str()) {
        out.insert(cur);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BraceKind {
    FnBody,
    ControlFlow,
    Closure,
    Other,
}

struct BraceFrame {
    kind: BraceKind,
    line: usize,
    line_indent: usize,
}

struct PendingCf {
    paren_depth: usize,
    bracket_depth: usize,
    kw_line: usize,
    line_indent: usize,
}

struct PendingFn {
    name: String,
    line: usize,
    paren_depth: usize,
    bracket_depth: usize,
    total_params: usize,
    golf_params: usize,
}

struct FnAcc {
    name: String,
    line: usize,
    max_cf_depth: usize,
    cognitive: usize,
    total_bindings: usize,
    golf_1char: usize,
    indent_compressed: usize,
    packed_cf_line: bool,
    cf_opens_by_line: BTreeMap<usize, usize>,
    last_bool_op: Option<TokKind>,
}

fn is_short_idiom_var(name: &str) -> bool {
    matches!(name, "i" | "j" | "k" | "x" | "y" | "n" | "id" | "ok" | "err")
}

fn is_value_ending_tok(tok: &Token) -> bool {
    match tok.kind {
        TokKind::NumOrChar
        | TokKind::StrLit
        | TokKind::RParen
        | TokKind::RBracket
        | TokKind::RBrace => true,
        TokKind::Ident => !matches!(
            tok.text.as_str(),
            "return"
                | "move"
                | "in"
                | "if"
                | "else"
                | "while"
                | "for"
                | "match"
                | "switch"
                | "let"
                | "break"
                | "continue"
                | "async"
        ),
        _ => false,
    }
}

fn extract_pattern_idents(slice: &[Token], total: &mut usize, golf: &mut usize) {
    for (idx, tok) in slice.iter().enumerate() {
        if tok.kind != TokKind::Ident {
            continue;
        }
        let name = tok.text.as_str();
        let skip_kw = matches!(
            name,
            "mut" | "ref" | "self" | "Self" | "box" | "Some" | "None" | "Ok" | "Err"
        );
        let is_type_ctor = name.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        let is_field_label = slice.get(idx + 1).is_some_and(|n| n.kind == TokKind::Colon);
        let is_path_seg = slice
            .get(idx + 1)
            .is_some_and(|n| n.kind == TokKind::Punct && n.text == "::");
        if !skip_kw && !is_type_ctor && !is_field_label && !is_path_seg && !name.starts_with('_') {
            *total += 1;
            if name.chars().count() == 1 && !is_short_idiom_var(name) {
                *golf += 1;
            }
        }
    }
}

fn skip_balanced(tokens: &[Token], mut pos: usize, open: TokKind, close: TokKind) -> usize {
    let mut depth = 1usize;
    pos += 1;
    while pos < tokens.len() && depth > 0 {
        if tokens[pos].kind == open {
            depth += 1;
        } else if tokens[pos].kind == close {
            depth -= 1;
        }
        pos += 1;
    }
    pos
}

fn find_top_colon(seg: &[Token]) -> Option<usize> {
    let (mut pd, mut bd, mut cd) = (0usize, 0usize, 0usize);
    for (idx, tok) in seg.iter().enumerate() {
        match tok.kind {
            TokKind::LParen => pd += 1,
            TokKind::RParen => pd = pd.saturating_sub(1),
            TokKind::LBracket => bd += 1,
            TokKind::RBracket => bd = bd.saturating_sub(1),
            TokKind::LBrace => cd += 1,
            TokKind::RBrace => cd = cd.saturating_sub(1),
            TokKind::Colon if pd == 0 && bd == 0 && cd == 0 => return Some(idx),
            _ => {}
        }
    }
    None
}

fn count_rust_fn_params(tokens: &[Token], param_start: usize, param_end: usize) -> (usize, usize) {
    let (mut total, mut golf) = (0usize, 0usize);
    let mut seg_start = param_start;
    let (mut sp, mut sb, mut sa, mut sc) = (0usize, 0usize, 0usize, 0usize);

    for idx in param_start..=param_end {
        let is_end = idx == param_end;
        let is_comma = !is_end
            && tokens[idx].kind == TokKind::Comma
            && sp == 0
            && sb == 0
            && sa == 0
            && sc == 0;
        if is_end || is_comma {
            let seg = &tokens[seg_start..idx];
            if let Some(col_pos) = find_top_colon(seg) {
                extract_pattern_idents(&seg[..col_pos], &mut total, &mut golf);
            }
            seg_start = idx + 1;
        } else {
            match tokens[idx].kind {
                TokKind::LParen => sp += 1,
                TokKind::RParen => sp = sp.saturating_sub(1),
                TokKind::LBracket => sb += 1,
                TokKind::RBracket => sb = sb.saturating_sub(1),
                TokKind::Lt => sa += 1,
                TokKind::Gt => sa = sa.saturating_sub(1),
                TokKind::LBrace => sc += 1,
                TokKind::RBrace => sc = sc.saturating_sub(1),
                _ => {}
            }
        }
    }
    (total, golf)
}

fn parse_fn_header(tokens: &[Token], fn_idx: usize, is_go: bool) -> Option<(PendingFn, bool)> {
    let mut name_pos = fn_idx + 1;
    if is_go && tokens.get(name_pos).is_some_and(|t| t.kind == TokKind::LParen) {
        name_pos = skip_balanced(tokens, name_pos, TokKind::LParen, TokKind::RParen);
    }
    let name_tok = tokens.get(name_pos)?;
    if name_tok.kind != TokKind::Ident {
        return None;
    }
    let mut pos = name_pos + 1;
    if !is_go && tokens.get(pos).is_some_and(|t| t.kind == TokKind::Lt) {
        pos = skip_balanced(tokens, pos, TokKind::Lt, TokKind::Gt);
    }
    if !tokens.get(pos).is_some_and(|t| t.kind == TokKind::LParen) {
        return None;
    }
    let param_start = pos + 1;
    pos = skip_balanced(tokens, pos, TokKind::LParen, TokKind::RParen);
    let param_end = pos.saturating_sub(1);
    let (total_params, golf_params) = if is_go {
        (0, 0)
    } else {
        count_rust_fn_params(tokens, param_start, param_end)
    };

    let mut has_body = false;
    let mut scan = pos;
    while scan < tokens.len() {
        match tokens[scan].kind {
            TokKind::LBrace => {
                has_body = true;
                break;
            }
            TokKind::Semi => break,
            _ => scan += 1,
        }
    }
    Some((
        PendingFn {
            name: name_tok.text.clone(),
            line: name_tok.line,
            paren_depth: 0,
            bracket_depth: 0,
            total_params,
            golf_params,
        },
        has_body,
    ))
}

fn is_match_guard_if(tokens: &[Token], if_idx: usize) -> bool {
    let (mut pd, mut bd) = (0usize, 0usize);
    let mut pos = if_idx + 1;
    while pos < tokens.len() {
        match tokens[pos].kind {
            TokKind::LParen => pd += 1,
            TokKind::RParen => pd = pd.saturating_sub(1),
            TokKind::LBracket => bd += 1,
            TokKind::RBracket => bd = bd.saturating_sub(1),
            TokKind::FatArrow if pd == 0 && bd == 0 => return true,
            TokKind::LBrace | TokKind::Semi | TokKind::RBrace if pd == 0 && bd == 0 => return false,
            _ => {}
        }
        pos += 1;
    }
    false
}

fn closure_opens_brace(tokens: &[Token], pipe_idx: usize) -> bool {
    let mut pos = pipe_idx + 1;
    let start_line = tokens[pipe_idx].line;
    while pos < tokens.len() && tokens[pos].line <= start_line + 3 {
        match tokens[pos].kind {
            TokKind::Pipe => {
                let mut after = pos + 1;
                if tokens.get(after).is_some_and(|t| t.kind == TokKind::Punct && t.text == "->") {
                    after += 1;
                    while after < tokens.len()
                        && !matches!(
                            tokens[after].kind,
                            TokKind::LBrace | TokKind::Semi | TokKind::Comma
                        )
                    {
                        after += 1;
                    }
                }
                return tokens.get(after).is_some_and(|t| t.kind == TokKind::LBrace);
            }
            TokKind::Semi | TokKind::LBrace | TokKind::RBrace => return false,
            _ => pos += 1,
        }
    }
    false
}

struct FnAnalyzer {
    brace_stack: Vec<BraceFrame>,
    fn_stack: Vec<FnAcc>,
    pending_fn: Option<PendingFn>,
    pending_cf: Vec<PendingCf>,
    pending_closure: Vec<(usize, usize)>,
    paren_depth: usize,
    bracket_depth: usize,
}

impl FnAnalyzer {
    fn new() -> Self {
        Self {
            brace_stack: Vec::new(),
            fn_stack: Vec::new(),
            pending_fn: None,
            pending_cf: Vec::new(),
            pending_closure: Vec::new(),
            paren_depth: 0,
            bracket_depth: 0,
        }
    }

    fn clear_bool_op(&mut self) {
        if let Some(cur) = self.fn_stack.last_mut() {
            cur.last_bool_op = None;
        }
    }

    fn handle_lbrace(&mut self, tok: &Token, tokens: &[Token]) {
        if let Some(pfn) = self.pending_fn.take() {
            if pfn.paren_depth == self.paren_depth && pfn.bracket_depth == self.bracket_depth {
                self.brace_stack.push(BraceFrame {
                    kind: BraceKind::FnBody,
                    line: tok.line,
                    line_indent: tok.line_indent,
                });
                self.fn_stack.push(FnAcc {
                    name: pfn.name,
                    line: pfn.line,
                    max_cf_depth: 0,
                    cognitive: 0,
                    total_bindings: pfn.total_params,
                    golf_1char: pfn.golf_params,
                    indent_compressed: 0,
                    packed_cf_line: false,
                    cf_opens_by_line: BTreeMap::new(),
                    last_bool_op: None,
                });
                return;
            }
            self.pending_fn = Some(pfn);
        }

        let cf_ready = self
            .pending_cf
            .last()
            .is_some_and(|p| p.paren_depth == self.paren_depth && p.bracket_depth == self.bracket_depth);
        if cf_ready && let Some(pcf) = self.pending_cf.pop() {
            self.push_cf_brace(pcf, tok, tokens);
            return;
        }

        let cls_ready = self
            .pending_closure
            .last()
            .is_some_and(|&(cp, cb)| cp == self.paren_depth && cb == self.bracket_depth);
        if cls_ready {
            self.pending_closure.pop();
            self.brace_stack.push(BraceFrame {
                kind: BraceKind::Closure,
                line: tok.line,
                line_indent: tok.line_indent,
            });
            self.clear_bool_op();
            return;
        }

        self.brace_stack.push(BraceFrame {
            kind: BraceKind::Other,
            line: tok.line,
            line_indent: tok.line_indent,
        });
        self.clear_bool_op();
    }

    fn push_cf_brace(&mut self, pcf: PendingCf, tok: &Token, tokens: &[Token]) {
        let parent_cf = self
            .brace_stack
            .iter()
            .rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .find(|f| f.kind == BraceKind::ControlFlow);
        let compressed = parent_cf
            .is_some_and(|pf| pcf.kw_line > pf.line && pcf.line_indent <= pf.line_indent + 1);
        self.brace_stack.push(BraceFrame {
            kind: BraceKind::ControlFlow,
            line: pcf.kw_line,
            line_indent: pcf.line_indent,
        });
        let depth = self
            .brace_stack
            .iter()
            .rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .filter(|f| f.kind == BraceKind::ControlFlow)
            .count();
        let same_line_cf_depth = self
            .brace_stack
            .iter()
            .rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .filter(|f| f.kind == BraceKind::ControlFlow && f.line == tok.line)
            .count();

        if let Some(cur) = self.fn_stack.last_mut() {
            cur.max_cf_depth = cur.max_cf_depth.max(depth);
            if compressed {
                cur.indent_compressed += 1;
            }
            let cnt = cur.cf_opens_by_line.entry(tok.line).or_insert(0);
            *cnt += 1;
            let semis_on_line = tokens
                .iter()
                .filter(|t| t.line == tok.line && t.kind == TokKind::Semi)
                .count();
            if same_line_cf_depth >= 3 || (*cnt >= 3 && (same_line_cf_depth >= 2 || semis_on_line >= 2)) {
                cur.packed_cf_line = true;
            }
            cur.last_bool_op = None;
        }
    }

    fn handle_fn_body_ident(
        &mut self,
        tokens: &[Token],
        idx: usize,
        prev: Option<&Token>,
        is_go: bool,
    ) {
        let tok = &tokens[idx];
        let cog_nest = self
            .brace_stack
            .iter()
            .rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .filter(|f| matches!(f.kind, BraceKind::ControlFlow | BraceKind::Closure))
            .count();

        match tok.text.as_str() {
            "if" => self.handle_if(tokens, idx, prev, is_go, cog_nest),
            "else" => self.handle_else(tokens, idx, prev, cog_nest),
            "for" => {
                if !tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::Lt) {
                    self.add_cf(tok, 1 + cog_nest);
                }
            }
            "while" | "loop" | "match" | "switch" => self.add_cf(tok, 1 + cog_nest),
            "let" if !is_go => self.handle_let(tokens, idx, prev),
            _ => {}
        }
    }

    fn add_cf(&mut self, tok: &Token, cog_add: usize) {
        if let Some(cur) = self.fn_stack.last_mut() {
            cur.cognitive += cog_add;
            cur.last_bool_op = None;
        }
        self.pending_cf.push(PendingCf {
            paren_depth: self.paren_depth,
            bracket_depth: self.bracket_depth,
            kw_line: tok.line,
            line_indent: tok.line_indent,
        });
    }

    fn handle_if(
        &mut self,
        tokens: &[Token],
        idx: usize,
        prev: Option<&Token>,
        is_go: bool,
        cog_nest: usize,
    ) {
        if !is_go && is_match_guard_if(tokens, idx) {
            if let Some(cur) = self.fn_stack.last_mut() {
                cur.cognitive += 1;
                cur.last_bool_op = None;
            }
            return;
        }
        let is_else_if = prev.is_some_and(|p| p.kind == TokKind::Ident && p.text == "else");
        let add = if is_else_if { 1 } else { 1 + cog_nest };
        self.add_cf(&tokens[idx], add);
    }

    fn handle_else(&mut self, tokens: &[Token], idx: usize, prev: Option<&Token>, cog_nest: usize) {
        let next_is_if = tokens
            .get(idx + 1)
            .is_some_and(|t| t.kind == TokKind::Ident && t.text == "if");
        if next_is_if {
            return;
        }
        let after_rbrace = prev.is_some_and(|p| p.kind == TokKind::RBrace);
        let add = if after_rbrace { 1 } else { 1 + cog_nest };
        self.add_cf(&tokens[idx], add);
    }

    fn handle_let(&mut self, tokens: &[Token], idx: usize, prev: Option<&Token>) {
        let is_cond_let = prev.is_some_and(|p| {
            p.kind == TokKind::AndAnd
                || (p.kind == TokKind::Ident && matches!(p.text.as_str(), "if" | "while"))
        });
        if is_cond_let {
            return;
        }
        let mut pos = idx + 1;
        let (mut lp, mut lb, mut lc) = (0usize, 0usize, 0usize);
        while pos < tokens.len() {
            let tk = tokens[pos].kind;
            if lp == 0 && lb == 0 && lc == 0 && matches!(tk, TokKind::Eq | TokKind::Colon | TokKind::Semi) {
                break;
            }
            match tk {
                TokKind::LParen => lp += 1,
                TokKind::RParen => lp = lp.saturating_sub(1),
                TokKind::LBracket => lb += 1,
                TokKind::RBracket => lb = lb.saturating_sub(1),
                TokKind::LBrace => lc += 1,
                TokKind::RBrace => lc = lc.saturating_sub(1),
                _ => {}
            }
            pos += 1;
        }
        if let Some(cur) = self.fn_stack.last_mut() {
            extract_pattern_idents(&tokens[idx + 1..pos], &mut cur.total_bindings, &mut cur.golf_1char);
        }
    }
}

fn analyze_functions_and_golf(
    tokens: &[Token],
    is_go: bool,
    metrics: &mut StrictMetrics,
    findings: &mut Vec<StrictFinding>,
) {
    let mut state = FnAnalyzer::new();
    for idx in 0..tokens.len() {
        let tok = &tokens[idx];
        if tok.in_test {
            continue;
        }
        let prev = (idx > 0 && !tokens[idx - 1].in_test).then(|| &tokens[idx - 1]);
        check_decl_golf(tokens, idx, is_go, &mut state, findings);

        match tok.kind {
            TokKind::LParen => state.paren_depth += 1,
            TokKind::RParen => state.paren_depth = state.paren_depth.saturating_sub(1),
            TokKind::LBracket => state.bracket_depth += 1,
            TokKind::RBracket => state.bracket_depth = state.bracket_depth.saturating_sub(1),
            TokKind::Semi | TokKind::Comma | TokKind::FatArrow => state.clear_bool_op(),
            TokKind::LBrace => state.handle_lbrace(tok, tokens),
            TokKind::RBrace => {
                if let Some(popped) = state.brace_stack.pop() {
                    if popped.kind == BraceKind::FnBody {
                        if let Some(acc) = state.fn_stack.pop() {
                            finalize_fn(&acc, metrics, findings);
                        }
                    } else {
                        state.clear_bool_op();
                    }
                }
            }
            _ => step_fn_body_token(&mut state, tokens, idx, prev, is_go),
        }
    }

    if !is_go
        && metrics.prod_bindings >= 10
        && (metrics.prod_golf_1char * 100) / metrics.prod_bindings >= 40
    {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "identifier_golf",
            line: 1,
            detail: format!(
                "file-wide 1-char identifier golf {}/{} ({:.1}% >= 40.0%)",
                metrics.prod_golf_1char,
                metrics.prod_bindings,
                (metrics.prod_golf_1char as f64 / metrics.prod_bindings as f64) * 100.0
            ),
        });
    }
}

fn check_decl_golf(
    tokens: &[Token],
    idx: usize,
    is_go: bool,
    state: &mut FnAnalyzer,
    findings: &mut Vec<StrictFinding>,
) {
    let tok = &tokens[idx];
    if !is_go
        && tok.kind == TokKind::Ident
        && matches!(tok.text.as_str(), "struct" | "enum" | "trait")
        && let Some(nt) = tokens.get(idx + 1)
        && nt.kind == TokKind::Ident
        && nt.text.chars().count() == 1
    {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "identifier_golf",
            line: nt.line,
            detail: format!("1-char {} name `{}`", tok.text, nt.text),
        });
    }

    let is_fn_kw = tok.kind == TokKind::Ident
        && ((is_go && tok.text == "func") || (!is_go && tok.text == "fn"));
    if is_fn_kw && let Some((mut pfn, has_body)) = parse_fn_header(tokens, idx, is_go) {
        if !is_go && pfn.name.chars().count() == 1 {
            findings.push(StrictFinding {
                severity: Severity::Veto,
                rule: "identifier_golf",
                line: pfn.line,
                detail: format!("1-char prod function name `{}`", pfn.name),
            });
        }
        if has_body {
            pfn.paren_depth = state.paren_depth;
            pfn.bracket_depth = state.bracket_depth;
            state.pending_fn = Some(pfn);
        }
    }
}

fn step_fn_body_token(
    state: &mut FnAnalyzer,
    tokens: &[Token],
    idx: usize,
    prev: Option<&Token>,
    is_go: bool,
) {
    if state.fn_stack.is_empty() {
        return;
    }
    let tok = &tokens[idx];
    if tok.kind == TokKind::AndAnd {
        if let Some(cur) = state.fn_stack.last_mut() && cur.last_bool_op != Some(TokKind::AndAnd) {
            cur.cognitive += 1;
            cur.last_bool_op = Some(TokKind::AndAnd);
        }
    } else if tok.kind == TokKind::OrOr {
        if prev.is_some_and(is_value_ending_tok) {
            if let Some(cur) = state.fn_stack.last_mut() && cur.last_bool_op != Some(TokKind::OrOr) {
                cur.cognitive += 1;
                cur.last_bool_op = Some(TokKind::OrOr);
            }
        } else if !is_go && tokens.get(idx + 1).is_some_and(|t| t.kind == TokKind::LBrace) {
            state.pending_closure.push((state.paren_depth, state.bracket_depth));
        }
    } else if !is_go && tok.kind == TokKind::Pipe {
        if prev.is_none_or(|p| !is_value_ending_tok(p)) && closure_opens_brace(tokens, idx) {
            state.pending_closure.push((state.paren_depth, state.bracket_depth));
        }
    } else if tok.kind == TokKind::Ident {
        state.handle_fn_body_ident(tokens, idx, prev, is_go);
    }
}

fn finalize_fn(acc: &FnAcc, metrics: &mut StrictMetrics, findings: &mut Vec<StrictFinding>) {
    metrics.max_cf_depth = metrics.max_cf_depth.max(acc.max_cf_depth);
    metrics.max_cognitive = metrics.max_cognitive.max(acc.cognitive);
    metrics.prod_bindings += acc.total_bindings;
    metrics.prod_golf_1char += acc.golf_1char;

    if acc.max_cf_depth >= MAX_CF_DEPTH_VETO
        || acc.cognitive > MAX_COGNITIVE_VETO
        || acc.indent_compressed >= 4
        || acc.packed_cf_line
    {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "deep_nesting_spaghetti",
            line: acc.line,
            detail: format!(
                "fn `{}`: cf_depth={} (limit <{MAX_CF_DEPTH_VETO}), cognitive={} (limit <={MAX_COGNITIVE_VETO}), compressed_blocks={}, packed_cf_line={}",
                acc.name, acc.max_cf_depth, acc.cognitive, acc.indent_compressed, acc.packed_cf_line
            ),
        });
    } else if acc.max_cf_depth >= WARN_CF_DEPTH || acc.cognitive >= WARN_COGNITIVE {
        findings.push(StrictFinding {
            severity: Severity::Warning,
            rule: "complexity_borderline",
            line: acc.line,
            detail: format!(
                "fn `{}` borderline: cf_depth={}, cognitive={}",
                acc.name, acc.max_cf_depth, acc.cognitive
            ),
        });
    }

    if acc.total_bindings >= 5
        && acc.golf_1char >= 4
        && (acc.golf_1char * 100) / acc.total_bindings >= 55
    {
        findings.push(StrictFinding {
            severity: Severity::Veto,
            rule: "identifier_golf",
            line: acc.line,
            detail: format!(
                "fn `{}` 1-char variable golf {}/{} ({:.1}% >= 55.0%)",
                acc.name,
                acc.golf_1char,
                acc.total_bindings,
                (acc.golf_1char as f64 / acc.total_bindings as f64) * 100.0
            ),
        });
    }
}
