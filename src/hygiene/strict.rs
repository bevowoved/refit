use super::ops::{SparseAttnTelemetry, sigmoid_f64};
use super::weights::should_use_5trit_lut;
use crate::lex::{LangKind, LexCursor, LexKind, LexTok};
pub use crate::types::FileEgoContext;
use crate::types::HygieneResult;
use aho_corasick::AhoCorasick;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const MAX_COLS: usize = 128;
const MAX_CF_DEPTH_VETO: usize = 6;
const MAX_COGNITIVE_VETO: usize = 35;
const WARN_CF_DEPTH: usize = 5;
const WARN_COGNITIVE: usize = 25;

/// Sorted ascending by `char as u32` for `O(log N)` binary search (49 UTS #39 confusables).
const UTS39_CONFUSABLES: [(char, &str, char); 49] = [
    ('\u{037E}', "Greek question mark", ';'), ('\u{0391}', "Greek", 'A'), ('\u{0392}', "Greek", 'B'),
    ('\u{0395}', "Greek", 'E'), ('\u{0396}', "Greek", 'Z'), ('\u{0397}', "Greek", 'H'),
    ('\u{0399}', "Greek", 'I'), ('\u{039A}', "Greek", 'K'), ('\u{039C}', "Greek", 'M'),
    ('\u{039D}', "Greek", 'N'), ('\u{039F}', "Greek", 'O'), ('\u{03A1}', "Greek", 'P'),
    ('\u{03A4}', "Greek", 'T'), ('\u{03A5}', "Greek", 'Y'), ('\u{03A7}', "Greek", 'X'),
    ('\u{03B1}', "Greek", 'a'), ('\u{03B9}', "Greek", 'i'), ('\u{03BA}', "Greek", 'k'),
    ('\u{03BD}', "Greek", 'v'), ('\u{03BF}', "Greek", 'o'), ('\u{03C1}', "Greek", 'p'),
    ('\u{03C4}', "Greek", 't'), ('\u{03C5}', "Greek", 'u'), ('\u{03C7}', "Greek", 'x'),
    ('\u{0405}', "Cyrillic", 'S'), ('\u{0406}', "Cyrillic", 'I'), ('\u{0408}', "Cyrillic", 'J'),
    ('\u{0410}', "Cyrillic", 'A'), ('\u{0412}', "Cyrillic", 'B'), ('\u{0415}', "Cyrillic", 'E'),
    ('\u{041A}', "Cyrillic", 'K'), ('\u{041C}', "Cyrillic", 'M'), ('\u{041D}', "Cyrillic", 'H'),
    ('\u{041E}', "Cyrillic", 'O'), ('\u{0420}', "Cyrillic", 'P'), ('\u{0421}', "Cyrillic", 'C'),
    ('\u{0422}', "Cyrillic", 'T'), ('\u{0423}', "Cyrillic", 'Y'), ('\u{0425}', "Cyrillic", 'X'),
    ('\u{0430}', "Cyrillic", 'a'), ('\u{0435}', "Cyrillic", 'e'), ('\u{043E}', "Cyrillic", 'o'),
    ('\u{0440}', "Cyrillic", 'p'), ('\u{0441}', "Cyrillic", 'c'), ('\u{0443}', "Cyrillic", 'y'),
    ('\u{0445}', "Cyrillic", 'x'), ('\u{0455}', "Cyrillic", 's'), ('\u{0456}', "Cyrillic", 'i'),
    ('\u{0458}', "Cyrillic", 'j'),
];

const _: () = {
    let mut i = 1usize;
    while i < UTS39_CONFUSABLES.len() {
        assert!((UTS39_CONFUSABLES[i - 1].0 as u32) < (UTS39_CONFUSABLES[i].0 as u32));
        i += 1;
    }
};

const PROD_IDIOMS: [(&str, f64); 15] = [
    ("todo!(", 0.90), ("unimplemented!(", 0.90), ("transmute", 0.85), (".unwrap()", 0.55),
    (concat!("#!", "[allow("), 0.90), (concat!("#", "[allow("), 0.75), ("serde(flatten)", 0.80),
    ("@ts-nocheck", 0.90), ("@ts-ignore", 0.80), ("eslint-disable", 0.80), ("except: pass", 0.85),
    ("as any", 0.65), ("catch {}", 0.75), ("panic!(", 0.55), ("unsafe {", 0.45),
];

static IDIOM_AC: OnceLock<AhoCorasick> = OnceLock::new();

fn idiom_automaton() -> &'static AhoCorasick {
    IDIOM_AC.get_or_init(|| AhoCorasick::new(PROD_IDIOMS.iter().map(|(p, _)| *p)).expect("static 15-idiom AC"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity { Veto, Warning }

#[derive(Debug, Clone)]
pub struct StrictFinding { pub severity: Severity, pub rule: &'static str, pub line: usize, pub detail: String }

fn push_finding(
    findings: &mut Vec<StrictFinding>,
    severity: Severity,
    rule: &'static str,
    line: usize,
    detail: impl Into<String>,
) {
    findings.push(StrictFinding { severity, rule, line, detail: detail.into() });
}

#[derive(Debug, Clone)]
pub struct StrictMetrics {
    pub total_lines: usize, pub prod_lines: usize, pub blank_prod_lines: usize,
    pub nondoc_comment_lines: usize, pub indent_mode: &'static str, pub indent_quantum: usize,
    pub max_logical_depth: usize, pub max_cf_depth: usize, pub max_cognitive: usize,
    pub max_col: usize, pub max_tail_code_chars: usize, pub prod_bindings: usize,
    pub prod_golf_1char: usize, pub prod_unwraps: usize, pub unsafe_blocks: usize,
}

impl Default for StrictMetrics {
    fn default() -> Self {
        Self {
            total_lines: 0, prod_lines: 0, blank_prod_lines: 0, nondoc_comment_lines: 0,
            indent_mode: "spaces", indent_quantum: 4, max_logical_depth: 0, max_cf_depth: 0,
            max_cognitive: 0, max_col: 0, max_tail_code_chars: 0, prod_bindings: 0,
            prod_golf_1char: 0, prod_unwraps: 0, unsafe_blocks: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockModuleMetrics {
    pub fn_count: usize, pub max_fn_loc: usize, pub max_cognitive: usize, pub max_params: usize,
    pub max_bool_params: usize, pub pub_items: usize, pub impl_loc: usize, pub piir: f64,
    pub field_leaks: usize, pub getter_leaks: usize, pub passthrough_fns: usize,
    pub shallow_module: bool, pub module_cycle: bool, pub tangle: bool,
    pub block_score: f64, pub delta_block: f64, pub structural_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAudit {
    pub proof_tier: String, pub attn_mode: String, pub block_score: f64, pub piir: f64,
    pub pairs_evaluated: usize, pub cobs_blocks: usize, pub engram_far_hits: usize, pub engram_bytes: usize,
    pub mli_reused: bool, pub mli_reuse_hits: usize, pub lsa_slab_reuses: usize, pub idiom_hits: usize,
    pub raw_v07: f64, pub delta_block: f64, pub delta_cross: f64, pub delta_slop: f64,
    pub veto_cap: f64, pub final_score: f64, pub use_5trit_lut: bool, pub structural_rules: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StrictReport {
    pub metrics: StrictMetrics, pub block: BlockModuleMetrics, pub findings: Vec<StrictFinding>,
    pub veto_cap: f64, pub strict_score: f64, pub idiom_hits: usize, pub idiom_risk: f64,
}

impl StrictReport {
    pub fn veto_rules(&self) -> Vec<&'static str> {
        self.findings.iter().filter(|f| f.severity == Severity::Veto).map(|f| f.rule).collect()
    }
    pub fn formatted_vetoes(&self) -> Vec<String> {
        self.findings.iter().filter(|f| f.severity == Severity::Veto)
            .map(|f| format!("{} (L{}): {}", f.rule, f.line, f.detail)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct AntiGamingAudit {
    pub path: String, pub passed: bool, pub veto_cap: f64, pub indent_mode: &'static str,
    pub indent_quantum: usize, pub before_vetoes: Vec<String>,
    pub after_vetoes: Vec<String>, pub introduced_vetoes: Vec<String>,
}

/// Apply strict CH-V4..V7 anti-gaming + live `FileEgoContext` + truthful `CalibrationAudit`.
/// Leaves canonical `base.score` 100% untouched.
pub fn score_strict_calibrated(
    path: &str, content: &str, base: &HygieneResult, ego: Option<&FileEgoContext>,
    telemetry: Option<&SparseAttnTelemetry>, weight_zero_ratio: f64,
) -> HygieneResult {
    let report = audit_source_full(path, content, base.score, is_test_only_path(path), ego);
    let mut out = base.clone();
    let (mut veto_cap, mut triggered) = (report.veto_cap, report.formatted_vetoes());
    let (delta_block, delta_cross) = (report.block.delta_block, compute_delta_cross(ego));
    let delta_slop = compute_delta_slop(base, report.idiom_hits, report.idiom_risk);
    if report.block.module_cycle || report.block.field_leaks > 0 || report.block.shallow_module {
        veto_cap = veto_cap.min(0.48);
        for rule in &report.block.structural_rules {
            if !triggered.iter().any(|v| v == rule) { triggered.push(rule.clone()); }
        }
    }
    let strict_bound = report.strict_score.min(veto_cap);
    let total_delta = delta_block + delta_cross + delta_slop;
    let calibrated = if total_delta == 0.0 && veto_cap >= 1.0 {
        strict_bound
    } else {
        let raw = sigmoid_f64(logit_f64(base.score) - total_delta).min(veto_cap).min(strict_bound);
        (raw * 10_000.0).round() / 10_000.0
    };
    let t = telemetry.cloned().unwrap_or_default();
    let attn_mode = if t.attn_mode.is_empty() { "v07_exact" } else { t.attn_mode };
    let proof_tier = if veto_cap < 1.0 {
        "tier0_vetoed"
    } else if telemetry.is_some() {
        "tier2_cobs2_engram_lsa_mli"
    } else {
        "tier1_exact_v07"
    };
    let audit = CalibrationAudit {
        proof_tier: proof_tier.into(), attn_mode: attn_mode.into(), block_score: report.block.block_score,
        piir: report.block.piir, pairs_evaluated: t.pairs_evaluated, cobs_blocks: t.cobs_blocks,
        engram_far_hits: t.engram_far_hits, engram_bytes: t.engram_bytes, mli_reused: t.mli_reused,
        mli_reuse_hits: t.mli_reuse_hits, lsa_slab_reuses: t.lsa_slab_reuses, idiom_hits: report.idiom_hits,
        raw_v07: base.score, delta_block, delta_cross, delta_slop, veto_cap, final_score: calibrated,
        use_5trit_lut: should_use_5trit_lut(weight_zero_ratio),
        structural_rules: report.block.structural_rules.clone(),
    };
    out.strict_score = Some(if ego.is_some() { calibrated } else { report.strict_score });
    out.multiscale_score = Some(calibrated);
    out.veto_cap = Some(veto_cap);
    out.triggered_vetoes = triggered;
    out.calibration_log = serde_json::to_string(&audit).ok();
    out
}

pub fn audit_anti_gaming(path: &str, before: &str, after: &str) -> AntiGamingAudit {
    let is_test = is_test_only_path(path);
    let rep_before = audit_source_full(path, before, 0.8808, is_test, None);
    let rep_after = audit_source_full(path, after, 0.8808, is_test, None);
    let before_rules: BTreeSet<&str> = rep_before.veto_rules().into_iter().collect();
    let introduced_vetoes = rep_after.findings.iter()
        .filter(|f| f.severity == Severity::Veto && !before_rules.contains(f.rule))
        .map(|f| format!("{} (L{}): {}", f.rule, f.line, f.detail)).collect();
    let after_vetoes = rep_after.formatted_vetoes();
    AntiGamingAudit {
        path: path.to_string(), passed: after_vetoes.is_empty(), veto_cap: rep_after.veto_cap,
        indent_mode: rep_after.metrics.indent_mode, indent_quantum: rep_after.metrics.indent_quantum,
        before_vetoes: rep_before.formatted_vetoes(), after_vetoes, introduced_vetoes,
    }
}

#[cfg(test)]
pub fn audit_source_with_test_flag(path: &str, content: &str, base_score: f64, is_test_only: bool) -> StrictReport {
    audit_source_full(path, content, base_score, is_test_only, None)
}

pub fn is_test_only_path(rel: &str) -> bool {
    let norm = rel.replace('\\', "/");
    let fname = norm.rsplit('/').next().unwrap_or(&norm);
    norm.starts_with("tests/") || norm.contains("/tests/") || fname == "testutil.rs"
        || fname == "test_util.rs" || fname.ends_with("_test.rs")
        || fname.ends_with("_tests.rs") || fname.ends_with("_test.go")
}

#[cfg(test)]
pub fn find_cfg_test_mod_decls(rel_path: &str, src: &str) -> Vec<String> {
    let mut cur = LexCursor::new(LangKind::Rust, src.as_bytes());
    let mut toks = Vec::new();
    while let Some(t) = cur.next_token() {
        if !matches!(t.kind, LexKind::LineComment | LexKind::BlockComment | LexKind::Newline) { toks.push(t); }
    }
    let parent = rel_path.rsplit_once('/').map_or("", |(dir, _)| dir);
    let mut out = Vec::new();
    for idx in 0..toks.len() {
        let tok = toks[idx];
        if tok.flags.in_test() && tok.kind == LexKind::Ident && tok.slice == b"mod"
            && let Some(nt) = toks.get(idx + 1).filter(|n| n.kind == LexKind::Ident)
            && toks.get(idx + 2).is_some_and(|n| n.kind == LexKind::Semi)
        {
            let name = nt.as_str();
            if parent.is_empty() {
                out.push(format!("{name}.rs")); out.push(format!("{name}/mod.rs"));
            } else {
                out.push(format!("{parent}/{name}.rs")); out.push(format!("{parent}/{name}/mod.rs"));
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrictLang { Rust, Go, Script }

fn classify_lang(path: &str) -> (StrictLang, LangKind) {
    let lk = LangKind::from_ext(path.rsplit('.').next().unwrap_or("rs"));
    let sl = match lk {
        LangKind::Go => StrictLang::Go,
        LangKind::Js | LangKind::Jsx | LangKind::Python => StrictLang::Script,
        LangKind::Rust | LangKind::Other => StrictLang::Rust,
    };
    (sl, if sl == StrictLang::Rust { LangKind::Rust } else { lk })
}

#[derive(Default)]
struct LineScan<'a> {
    raw_line: &'a str, raw_cols: usize, leading_spaces: usize, leading_tabs: usize,
    in_test: bool, has_code: bool, has_doc_comment: bool, has_safety_comment: bool,
    has_nondoc_comment: bool, in_multiline_str: bool, tail_ident_chars: usize, tail_ops: usize,
    tail_has_cf: bool, mid_line_semi: bool, total_semis: usize, has_pad_smuggle: bool,
    comment_slices: Vec<&'a str>, code_idents: Vec<&'a str>,
}

struct StrictTok<'a> {
    kind: LexKind, text: &'a str, line: usize, line_indent: usize, in_test: bool, in_os_cfg: bool,
}

fn init_line_scans(src: &str, file_is_test_only: bool) -> Vec<LineScan<'_>> {
    let mut lines = vec![LineScan::default()];
    for raw in src.split('\n') {
        let trimmed_cr = raw.strip_suffix('\r').unwrap_or(raw);
        let (mut col, mut lead_sp, mut lead_tb, mut lead_done) = (0usize, 0usize, 0usize, false);
        for ch in trimmed_cr.chars() {
            match ch {
                '\t' => { col += 4 - (col % 4); if !lead_done { lead_tb += 1; } }
                ' ' => { col += 1; if !lead_done { lead_sp += 1; } }
                _ => { col += 1; lead_done = true; }
            }
        }
        lines.push(LineScan {
            raw_line: trimmed_cr, raw_cols: col, leading_spaces: lead_sp,
            leading_tabs: lead_tb, in_test: file_is_test_only, ..LineScan::default()
        });
    }
    lines
}

fn strip_comment_delims(kind: LexKind, text: &str) -> &str {
    match kind {
        LexKind::LineComment => text.strip_prefix("///")
            .or_else(|| text.strip_prefix("//!"))
            .or_else(|| text.strip_prefix("//"))
            .or_else(|| text.strip_prefix('#'))
            .unwrap_or(text),
        LexKind::BlockComment => text.strip_prefix("/**")
            .or_else(|| text.strip_prefix("/*!"))
            .or_else(|| text.strip_prefix("/*"))
            .unwrap_or(text).strip_suffix("*/").unwrap_or(text),
        _ => text,
    }
}

fn is_tail_op(kind: LexKind, text: &str) -> bool {
    matches!(
        kind,
        LexKind::Eq | LexKind::Lt | LexKind::Gt | LexKind::AndAnd
            | LexKind::OrOr | LexKind::FatArrow | LexKind::Semi
    ) || matches!(
        text,
        "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<=" | ">="
            | "+=" | "-=" | "*=" | "/=" | "?" | "&" | "|" | "^"
    )
}

#[derive(Default)]
struct LineTracker {
    cur_line: usize, seen_code: bool, prev_end_col: u32,
    block_cmt_pad: bool, saw_semi: bool, bracket_depth: usize,
}

impl LineTracker {
    fn sync_line(&mut self, line: usize) {
        if self.cur_line != line {
            *self = Self { cur_line: line, ..Self::default() };
        }
    }

    fn record_comment<'a>(&mut self, tok: LexTok<'a>, in_test: bool, lines: &mut [LineScan<'a>]) {
        let (start_ln, end_ln) = (tok.line as usize, tok.end_line as usize);
        let (is_doc, is_safety) = (tok.flags.is_doc_comment(), tok.flags.is_safety_comment());
        let body = strip_comment_delims(tok.kind, tok.as_str());
        if tok.kind == LexKind::BlockComment
            && start_ln == end_ln && self.seen_code && tok.end_col - tok.col >= 12
        {
            self.block_cmt_pad = true;
        }
        let hi_ln = end_ln.min(lines.len().saturating_sub(1));
        for ls in &mut lines[start_ln..=hi_ln] {
            if in_test { ls.in_test = true; }
            if is_doc {
                ls.has_doc_comment = true;
            } else if is_safety {
                ls.has_safety_comment = true;
            } else {
                ls.has_nondoc_comment = true;
            }
        }
        if let Some(ls) = lines.get_mut(start_ln) && !body.trim().is_empty() {
            ls.comment_slices.push(body);
        }
        self.prev_end_col = tok.end_col;
    }

    fn record_code<'a>(
        &mut self, tok: LexTok<'a>, in_test: bool,
        lines: &mut [LineScan<'a>], tokens: &mut Vec<StrictTok<'a>>,
    ) {
        let (ln, end_ln, text) = (tok.line as usize, tok.end_line as usize, tok.as_str());
        if let Some(ls) = lines.get_mut(ln) {
            if in_test { ls.in_test = true; }
            ls.has_code = true;
            if !in_test {
                let gap = tok.col.saturating_sub(self.prev_end_col);
                let pad_hit = gap >= 12 || (self.block_cmt_pad && tok.col as usize >= MAX_COLS);
                if self.seen_code && !tok.flags.in_attr() && pad_hit {
                    ls.has_pad_smuggle = true;
                }
                if self.saw_semi && self.bracket_depth == 0 { ls.mid_line_semi = true; }
                record_tail_metrics(tok, text, ls);
            }
        }
        if tok.kind == LexKind::StrLit && end_ln > ln {
            let hi_ln = end_ln.min(lines.len().saturating_sub(1));
            for ls in &mut lines[(ln + 1)..=hi_ln] {
                ls.in_multiline_str = true;
                if in_test { ls.in_test = true; }
            }
        }
        match tok.kind {
            LexKind::LBracket => self.bracket_depth += 1,
            LexKind::RBracket => self.bracket_depth = self.bracket_depth.saturating_sub(1),
            LexKind::Semi if self.bracket_depth == 0 => {
                self.saw_semi = true;
                if !in_test && let Some(ls) = lines.get_mut(ln) { ls.total_semis += 1; }
            }
            _ => {}
        }
        self.seen_code = true;
        self.block_cmt_pad = false;
        self.prev_end_col = if end_ln == ln { tok.end_col } else { 0 };
        let line_indent = lines.get(ln).map_or(0, |ls| ls.leading_tabs * 4 + ls.leading_spaces);
        tokens.push(StrictTok {
            kind: tok.kind, text, line: ln, line_indent, in_test, in_os_cfg: tok.flags.in_os_cfg(),
        });
    }
}

fn record_tail_metrics<'a>(tok: LexTok<'a>, text: &'a str, ls: &mut LineScan<'a>) {
    let sc = tok.col as usize;
    let ec = if tok.end_line == tok.line { tok.end_col as usize } else { sc };
    match tok.kind {
        LexKind::Ident => {
            if ec > MAX_COLS {
                ls.tail_ident_chars += ec.saturating_sub(MAX_COLS.max(sc));
                if matches!(text, "if" | "else" | "for" | "while" | "loop" | "match" | "switch" | "return") {
                    ls.tail_has_cf = true;
                }
            }
            if text.len() >= 3 { ls.code_idents.push(text); }
        }
        LexKind::Number if ec > MAX_COLS => ls.tail_ident_chars += ec.saturating_sub(MAX_COLS.max(sc)),
        LexKind::Number | LexKind::StrLit | LexKind::CharLit => {}
        _ if sc >= MAX_COLS && is_tail_op(tok.kind, text) => ls.tail_ops += 1,
        _ => {}
    }
}

fn scan_with_lex_cursor(
    src: &str, lang: LangKind, file_is_test_only: bool,
) -> (Vec<StrictTok<'_>>, Vec<LineScan<'_>>) {
    let mut lines = init_line_scans(src, file_is_test_only);
    let mut tokens = Vec::with_capacity(src.len() / 8);
    let (mut cur, mut trk) = (LexCursor::new(lang, src.as_bytes()), LineTracker::default());
    while let Some(tok) = cur.next_token() {
        let (in_test, ln) = (file_is_test_only || tok.flags.in_test(), tok.line as usize);
        if tok.kind == LexKind::Newline {
            if in_test && let Some(ls) = lines.get_mut(ln) { ls.in_test = true; }
            continue;
        }
        trk.sync_line(ln);
        if matches!(tok.kind, LexKind::LineComment | LexKind::BlockComment) {
            trk.record_comment(tok, in_test, &mut lines);
        } else {
            trk.record_code(tok, in_test, &mut lines, &mut tokens);
        }
    }
    (tokens, lines)
}

fn audit_source_full(
    path: &str, src: &str, base_score: f64, file_is_test_only: bool, ego: Option<&FileEgoContext>,
) -> StrictReport {
    let (slang, cursor_lang) = classify_lang(path);
    let mut metrics = StrictMetrics::default();
    let mut block = BlockModuleMetrics::default();
    let mut findings = Vec::new();
    if !src.is_ascii() { check_bidi_and_invisibles(src, &mut findings); }
    let (tokens, lines) = scan_with_lex_cursor(src, cursor_lang, file_is_test_only);
    if !src.is_ascii() { check_uts39_homoglyphs(&tokens, &lines, &mut findings); }

    let (mut idiom_hits, mut idiom_risk) = (0usize, 0.0f64);
    if !file_is_test_only {
        if slang == StrictLang::Rust {
            check_attrs_and_macros(&tokens, &mut findings);
            check_unsafe_and_unwraps(&tokens, &lines, &mut metrics, &mut findings);
        } else if slang == StrictLang::Script {
            check_polyglot_suppressions(&lines, &mut findings);
        }
        check_line_tails_blanks_and_comments(&lines, &mut metrics, &mut findings);
        let pub_fields = if slang != StrictLang::Script {
            analyze_functions_and_blocks(&tokens, &lines, slang == StrictLang::Go, &mut metrics, &mut block, &mut findings)
        } else { 0 };
        block.impl_loc = metrics.prod_lines;
        finalize_block_metrics(&mut block, pub_fields, ego);
        (idiom_hits, idiom_risk) = scan_prod_idioms(&lines);
    } else {
        metrics.total_lines = lines.len().saturating_sub(1);
    }
    let (veto_cap, strict_score) = compute_calibration(base_score, &findings);
    StrictReport { metrics, block, findings, veto_cap, strict_score, idiom_hits, idiom_risk }
}

fn scan_prod_idioms(lines: &[LineScan<'_>]) -> (usize, f64) {
    let (ac, mut hits, mut risk_sum) = (idiom_automaton(), 0usize, 0.0f64);
    for ls in lines.iter().skip(1).filter(|l| !l.in_test && l.has_code) {
        for mat in ac.find_iter(ls.raw_line) {
            hits += 1;
            risk_sum += PROD_IDIOMS[mat.pattern().as_usize()].1;
        }
    }
    (hits, if hits > 0 { (risk_sum / (hits as f64)).min(1.0) } else { 0.0 })
}

#[inline]
fn logit_f64(prob: f64) -> f64 { let c = prob.clamp(1e-6, 1.0 - 1e-6); (c / (1.0 - c)).ln() }

fn compute_calibration(base_score: f64, findings: &[StrictFinding]) -> (f64, f64) {
    let (mut veto_cap, mut delta_slop, mut has_veto) = (1.0f64, 0.0f64, false);
    for f in findings.iter().filter(|f| f.severity == Severity::Veto) {
        has_veto = true;
        let (cap, pen) = match f.rule {
            "bidi_trojan_source" | "uts39_homoglyph" => (0.15, 3.5),
            "prod_placeholder_macro" => (0.40, 2.0),
            "lint_suppression" | "serde_flatten" | "unsafe_undocumented" | "std_lock_unwrap" => (0.45, 1.8),
            _ => (0.50, 1.6),
        };
        veto_cap = veto_cap.min(cap);
        delta_slop += pen;
    }
    if !has_veto { return (1.0, base_score); }
    (veto_cap, (sigmoid_f64(logit_f64(base_score) - delta_slop).min(veto_cap) * 10_000.0).round() / 10_000.0)
}

fn compute_delta_cross(ego: Option<&FileEgoContext>) -> f64 {
    let Some(ctx) = ego else { return 0.0 };
    let blast = (ctx.fan_in.saturating_mul(ctx.fan_out) as f64).sqrt();
    let churn_mult = 1.0 + ((ctx.churn_commits as f64) / 20.0).min(1.5);
    let mut raw = if blast > 6.0 { 0.08 * (blast - 6.0) * churn_mult } else { 0.0 };
    if ctx.clone_cluster_hits > 0 { raw += 0.20 * (ctx.clone_cluster_hits.min(4) as f64); }
    (raw * 10_000.0).round() / 10_000.0
}

fn compute_delta_slop(result: &HygieneResult, idiom_hits: usize, idiom_risk: f64) -> f64 {
    let mut delta = result.event_load.filter(|&l| l > 1.5).map_or(0.0, |l| 0.15 * (l - 1.5));
    if idiom_hits >= 2 { delta += 0.25 * idiom_risk; }
    (delta * 10_000.0).round() / 10_000.0
}

fn finalize_block_metrics(m: &mut BlockModuleMetrics, pub_fields: usize, ego: Option<&FileEgoContext>) {
    let impl_loc = m.impl_loc.max(1) as f64;
    m.piir = ((((m.pub_items as f64) * 3.0) / impl_loc).clamp(0.0, 1.0) * 10_000.0).round() / 10_000.0;
    let ext_r = ego.map_or(0, |c| c.external_field_readers);
    if m.getter_leaks >= 2 || (pub_fields > 0 && ext_r >= 2) || (pub_fields > 0 && m.getter_leaks >= 1) {
        m.field_leaks = m.getter_leaks.max(pub_fields).max(ext_r);
        m.structural_rules.push("field-leak".into());
    }
    let pr = if m.fn_count > 0 { (m.passthrough_fns as f64) / (m.fn_count as f64) } else { 0.0 };
    let sd = (m.pub_items as f64) / impl_loc;
    let shallow_big = m.pub_items > 7 && (m.piir >= 0.45 || pr >= 0.50);
    let shallow_small = m.pub_items >= 4 && m.fn_count >= 3 && sd > 0.20 && pr >= 0.60;
    if shallow_big || shallow_small {
        m.shallow_module = true;
        m.structural_rules.push("shallow-module".into());
    }
    if let Some(ctx) = ego {
        if ctx.has_module_cycle { m.module_cycle = true; m.structural_rules.push("module-cycle".into()); }
        let deg = ctx.fan_in + ctx.fan_out;
        let tangle = ctx.tangle || (ctx.fan_in >= 6 && ctx.fan_out >= 6 && ctx.neighbor_dirs >= 2 && deg > 0
            && (0.35..=0.65).contains(&((ctx.fan_out as f64) / (deg as f64))));
        if tangle { m.tangle = true; m.structural_rules.push("tangle".into()); }
    }
    let mut delta = 0.0f64;
    if m.max_fn_loc > 60 { delta += 0.015 * ((m.max_fn_loc - 60) as f64); }
    if m.max_cognitive > 15 { delta += 0.10 * ((m.max_cognitive - 15) as f64); }
    if m.max_params > 5 { delta += 0.25 * ((m.max_params - 5) as f64); }
    if m.max_bool_params >= 2 { delta += 0.35 * ((m.max_bool_params - 1) as f64); }
    if m.field_leaks > 0 { delta += 0.75; }
    if m.shallow_module { delta += 0.65 + 0.15 * (m.pub_items.saturating_sub(7) as f64); }
    if m.module_cycle { delta += 1.20; }
    if m.tangle { delta += 0.80; }
    m.delta_block = (delta * 10_000.0).round() / 10_000.0;
    let bscore = (sigmoid_f64(2.0 - delta) / sigmoid_f64(2.0)).clamp(0.0, 1.0);
    m.block_score = (bscore * 10_000.0).round() / 10_000.0;
}

fn check_bidi_and_invisibles(src: &str, findings: &mut Vec<StrictFinding>) {
    const BIDI: &[(char, &str)] = &[
        ('\u{202A}', "U+202A LRE"), ('\u{202B}', "U+202B RLE"), ('\u{202C}', "U+202C PDF"),
        ('\u{202D}', "U+202D LRO"), ('\u{202E}', "U+202E RLO (Trojan Source)"),
        ('\u{2066}', "U+2066 LRI (Trojan Source)"), ('\u{2067}', "U+2067 RLI (Trojan Source)"),
        ('\u{2068}', "U+2068 FSI (Trojan Source)"), ('\u{2069}', "U+2069 PDI (Trojan Source)"),
        ('\u{200E}', "U+200E LRM"), ('\u{200F}', "U+200F RLM"), ('\u{061C}', "U+061C ALM"),
        ('\u{200B}', "U+200B ZWSP"), ('\u{200C}', "U+200C ZWNJ"), ('\u{200D}', "U+200D ZWJ"),
        ('\u{2060}', "U+2060 WJ"),
    ];
    for (idx, line) in src.lines().enumerate() {
        if line.is_ascii() { continue; }
        let ln = idx + 1;
        for (ci, ch) in line.chars().enumerate() {
            let tag = if ch == '\u{FEFF}' && !(ln == 1 && ci == 0) {
                Some("U+FEFF mid-file BOM/ZWNBSP")
            } else {
                BIDI.iter().find(|&&(c, _)| c == ch).map(|&(_, t)| t)
            };
            if let Some(tag) = tag {
                let msg = format!("forbidden invisible/BiDi control {tag} (CVE-2021-42574)");
                push_finding(findings, Severity::Veto, "bidi_trojan_source", ln, msg);
                break;
            }
        }
    }
}

#[inline]
fn is_cyrillic_or_greek(ch: char) -> bool {
    let cp = ch as u32;
    (0x0370..=0x03FF).contains(&cp) || (0x1F00..=0x1FFF).contains(&cp) || (0x0400..=0x052F).contains(&cp)
}

fn check_uts39_homoglyphs(tokens: &[StrictTok<'_>], lines: &[LineScan<'_>], findings: &mut Vec<StrictFinding>) {
    let mut flagged = BTreeSet::new();
    for tok in tokens {
        if matches!(tok.kind, LexKind::StrLit | LexKind::CharLit | LexKind::Number) || tok.text.is_ascii() {
            continue;
        }
        let detail = tok.text.chars().find_map(|ch| {
            if let Ok(i) = UTS39_CONFUSABLES.binary_search_by_key(&ch, |&(c, _, _)| c) {
                let (_, scr, looks) = UTS39_CONFUSABLES[i];
                Some(format!("{scr} confusable U+{:04X} (looks like '{looks}') in code token", ch as u32))
            } else if is_cyrillic_or_greek(ch) {
                Some(format!("Cyrillic/Greek script char U+{:04X} in code token", ch as u32))
            } else { None }
        });
        if let Some(d) = detail && flagged.insert(tok.line) {
            push_finding(findings, Severity::Veto, "uts39_homoglyph", tok.line, d);
        }
    }
    for (ln, ls) in lines.iter().enumerate().skip(1) {
        if ls.comment_slices.is_empty() || flagged.contains(&ln) { continue; }
        for &cmt in &ls.comment_slices {
            if cmt.is_ascii() { continue; }
            let hit = cmt.split_whitespace().find_map(|w| {
                w.bytes().any(|b| b.is_ascii_alphabetic())
                    .then(|| w.chars().find(|&c| is_cyrillic_or_greek(c))).flatten()
            });
            if let Some(ch) = hit {
                flagged.insert(ln);
                let msg = format!("mixed Latin + Cyrillic/Greek homoglyph U+{:04X} in comment word", ch as u32);
                push_finding(findings, Severity::Veto, "uts39_homoglyph", ln, msg);
                break;
            }
        }
    }
}

fn check_polyglot_suppressions(lines: &[LineScan<'_>], findings: &mut Vec<StrictFinding>) {
    const NEEDLES: &[&[u8]] = &[b"@ts-ignore", b"@ts-nocheck", b"eslint-disable", b"noqa"];
    for (ln, ls) in lines.iter().enumerate().skip(1) {
        if ls.in_test || ls.comment_slices.is_empty() { continue; }
        let bad = ls.comment_slices.iter().any(|c| {
            let b = c.as_bytes();
            NEEDLES.iter().any(|n| crate::lex::find_ascii_ci(b, n).is_some())
        });
        if bad {
            push_finding(findings, Severity::Veto, "lint_suppression", ln, "forbidden lint/typecheck suppression comment");
        }
    }
}

fn check_attrs_and_macros(tokens: &[StrictTok<'_>], findings: &mut Vec<StrictFinding>) {
    let mut idx = 0usize;
    while idx < tokens.len() {
        let tok = &tokens[idx];
        if tok.in_test { idx += 1; continue; }
        if tok.kind == LexKind::Hash && let Some(next_idx) = inspect_attr(tokens, idx, findings) {
            idx = next_idx;
            continue;
        }
        let open_after = tokens.get(idx + 2).is_some_and(|t| {
            matches!(t.kind, LexKind::LParen | LexKind::LBracket | LexKind::LBrace)
        });
        if tok.kind == LexKind::Ident && matches!(tok.text, "todo" | "unimplemented")
            && tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Bang) && open_after
        {
            let msg = format!("forbidden production `{}!()` placeholder", tok.text);
            push_finding(findings, Severity::Veto, "prod_placeholder_macro", tok.line, msg);
        }
        idx += 1;
    }
}

fn inspect_attr(tokens: &[StrictTok<'_>], idx: usize, findings: &mut Vec<StrictFinding>) -> Option<usize> {
    let open = if tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::LBracket) {
        idx + 1
    } else if tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Bang)
        && tokens.get(idx + 2).is_some_and(|t| t.kind == LexKind::LBracket)
    {
        idx + 2
    } else {
        return None;
    };
    let pos = skip_balanced(tokens, open, LexKind::LBracket, LexKind::RBracket);
    let inner = &tokens[open + 1..pos.saturating_sub(1)];
    if let Some(first) = inner.first().filter(|t| t.kind == LexKind::Ident) {
        let ln = tokens[idx].line;
        let has_reason = inner.windows(2).any(|w| {
            w[0].kind == LexKind::Ident && w[0].text == "reason" && w[1].kind == LexKind::Eq
        });
        match first.text {
            "allow" => push_finding(
                findings, Severity::Veto, "lint_suppression", ln, "forbidden allow-attribute lint suppression",
            ),
            "cfg_attr" if inner.iter().any(|t| t.kind == LexKind::Ident && t.text == "allow") => {
                push_finding(
                    findings, Severity::Veto, "lint_suppression", ln,
                    "forbidden cfg_attr allow lint suppression",
                );
            }
            "expect" if !has_reason => {
                push_finding(
                    findings, Severity::Veto, "lint_suppression", ln,
                    "expect attribute missing explicit `reason = ...`",
                );
            }
            "serde" if inner.iter().any(|t| t.kind == LexKind::Ident && t.text == "flatten") => {
                push_finding(
                    findings, Severity::Veto, "serde_flatten", ln, "forbidden serde flatten attribute",
                );
            }
            _ => {}
        }
    }
    Some(pos)
}

fn check_unsafe_and_unwraps(
    tokens: &[StrictTok<'_>], lines: &[LineScan<'_>],
    metrics: &mut StrictMetrics, findings: &mut Vec<StrictFinding>,
) {
    for (idx, tok) in tokens.iter().enumerate() {
        if tok.in_test { continue; }
        let is_unwrap = tok.kind == LexKind::Dot
            && tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Ident && t.text == "unwrap")
            && tokens.get(idx + 2).is_some_and(|t| t.kind == LexKind::LParen);
        if is_unwrap {
            metrics.prod_unwraps += 1;
            if idx >= 4 && tokens[idx - 4].kind == LexKind::Dot && tokens[idx - 3].kind == LexKind::Ident
                && matches!(tokens[idx - 3].text, "lock" | "read" | "write")
                && tokens[idx - 2].kind == LexKind::LParen && tokens[idx - 1].kind == LexKind::RParen
            {
                let msg = format!("forbidden `.{}().unwrap()` lock-poison panic in prod", tokens[idx - 3].text);
                push_finding(findings, Severity::Veto, "std_lock_unwrap", tok.line, msg);
            }
        }
        if tok.kind == LexKind::Ident && tok.text == "unsafe" {
            check_unsafe_site(tokens, idx, lines, metrics, findings);
        }
    }
    if metrics.prod_unwraps >= 10 {
        let msg = format!("excessive production `.unwrap()` count: {}", metrics.prod_unwraps);
        push_finding(findings, Severity::Veto, "prod_unwrap_spam", 1, msg);
    } else if metrics.prod_unwraps > 0 {
        let msg = format!("{} invariant-guarded prod `.unwrap()` call(s)", metrics.prod_unwraps);
        push_finding(findings, Severity::Warning, "prod_unwrap_advisory", 1, msg);
    }
}

fn check_unsafe_site(
    tokens: &[StrictTok<'_>], idx: usize, lines: &[LineScan<'_>],
    metrics: &mut StrictMetrics, findings: &mut Vec<StrictFinding>,
) {
    let is_attr = idx > 0 && tokens[idx - 1].kind == LexKind::LBracket
        && tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::LParen);
    let is_extern = tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Ident && t.text == "extern");
    if is_attr || is_extern { return; }
    let tok = &tokens[idx];
    metrics.unsafe_blocks += 1;
    let has_doc = (tok.line.saturating_sub(3).max(1)..=tok.line)
        .any(|ln| lines.get(ln).is_some_and(|ls| ls.has_safety_comment));
    if has_doc { return; }
    if tok.in_os_cfg && !unsafe_block_contains_ub(tokens, idx) {
        push_finding(
            findings, Severity::Warning, "unsafe_os_ffi_missing_safety_doc", tok.line,
            "OS-FFI `unsafe` inside `#[cfg(windows|unix)]` lacks `// SAFETY:` doc",
        );
    } else {
        push_finding(
            findings, Severity::Veto, "unsafe_undocumented", tok.line,
            "undocumented `unsafe` block/fn missing `// SAFETY:` comment",
        );
    }
}

fn unsafe_block_contains_ub(tokens: &[StrictTok<'_>], idx: usize) -> bool {
    const UB: &[&str] = &[
        "transmute", "transmute_copy", "get_unchecked", "get_unchecked_mut",
        "from_raw_parts", "from_raw_parts_mut", "set_len", "assume_init",
        "uninitialized", "unreachable_unchecked",
    ];
    let mut pos = idx + 1;
    while pos < tokens.len() && !matches!(tokens[pos].kind, LexKind::LBrace | LexKind::Semi) { pos += 1; }
    if tokens.get(pos).is_none_or(|t| t.kind != LexKind::LBrace) { return false; }
    let end = skip_balanced(tokens, pos, LexKind::LBrace, LexKind::RBrace);
    (pos + 1..end.saturating_sub(1)).any(|s| {
        let is_static_mut = tokens[s].text == "static"
            && tokens.get(s + 1).is_some_and(|t| t.kind == LexKind::Ident && t.text == "mut");
        tokens[s].kind == LexKind::Ident && (UB.contains(&tokens[s].text) || is_static_mut)
    })
}

fn detect_dynamic_indent(lines: &[LineScan<'_>]) -> (&'static str, usize) {
    let (mut tabs, mut spaces, mut prev_sp, mut deltas) = (0usize, 0usize, 0usize, [0usize; 9]);
    for ls in lines.iter().skip(1).filter(|l| !l.in_test && l.has_code && !l.in_multiline_str) {
        if ls.leading_tabs > 0 { tabs += 1; } else {
            let sp = ls.leading_spaces;
            if sp > 0 { spaces += 1; }
            let diff = sp.abs_diff(prev_sp);
            if (1..=8).contains(&diff) { deltas[diff] += 1; }
            prev_sp = sp;
        }
    }
    if tabs >= spaces && tabs > 0 { return ("tabs", 1); }
    let (mut best_q, mut best_cnt) = (4usize, 0usize);
    for q in [4usize, 2, 8, 3] { if deltas[q] > best_cnt { best_cnt = deltas[q]; best_q = q; } }
    ("spaces", best_q)
}

#[derive(Default)]
struct BlankCommentAcc {
    consec_blank: usize, max_consec_blank: usize, max_consec_blank_line: usize,
    nonblank_run: usize, alt_blank_run: usize, max_alt_blank: usize, max_alt_blank_line: usize,
    nondoc_pure: usize, nonblank_prod: usize, echo_pairs: usize, first_echo: usize,
    cmt_code: usize, first_cmt_code: usize,
}

impl BlankCommentAcc {
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
        } else if self.nonblank_run >= 2 { self.alt_blank_run = 0; }
        self.nonblank_run = 0;
    }

    fn record_pure_nondoc(&mut self, lines: &[LineScan<'_>], ln: usize, body: &str) {
        const PREFIXES: &[&str] = &[
            "let ", "fn ", "pub fn ", "pub struct ", "if ", "for ",
            "while ", "match ", "return ", "use ", "struct ", "impl ", "const ",
        ];
        self.nondoc_pure += 1;
        let looks_code = PREFIXES.iter().any(|p| body.starts_with(p))
            || (body.ends_with(';') && (body.contains('=') || body.contains('(')));
        if looks_code {
            self.cmt_code += 1;
            if self.first_cmt_code == 0 { self.first_cmt_code = ln; }
        }
        if is_comment_echo_pair(lines, ln, body) {
            self.echo_pairs += 1;
            if self.first_echo == 0 { self.first_echo = ln; }
        }
    }
}

fn check_line_tails_blanks_and_comments(
    lines: &[LineScan<'_>], metrics: &mut StrictMetrics, findings: &mut Vec<StrictFinding>,
) {
    let (indent_mode, indent_quantum) = detect_dynamic_indent(lines);
    metrics.indent_mode = indent_mode;
    metrics.indent_quantum = indent_quantum;
    let mut acc = BlankCommentAcc {
        max_consec_blank_line: 1, max_alt_blank_line: 1, ..BlankCommentAcc::default()
    };

    for (ln, ls) in lines.iter().enumerate().skip(1) {
        metrics.total_lines += 1;
        if ls.in_test {
            acc.consec_blank = 0;
            acc.nonblank_run = 0;
            acc.alt_blank_run = 0;
            continue;
        }
        metrics.prod_lines += 1;
        metrics.max_col = metrics.max_col.max(ls.raw_cols);
        if ls.has_code && !ls.in_multiline_str {
            let denom = if indent_mode == "tabs" { 4 } else { indent_quantum.max(1) };
            let depth = ls.leading_tabs + (ls.leading_spaces / denom);
            metrics.max_logical_depth = metrics.max_logical_depth.max(depth);
        }
        if ls.raw_cols > MAX_COLS && !ls.in_multiline_str {
            let ts = ls.tail_ident_chars + ls.tail_ops;
            metrics.max_tail_code_chars = metrics.max_tail_code_chars.max(ts);
            let dense = ls.tail_has_cf || ls.mid_line_semi || ls.total_semis >= 2
                || ls.has_pad_smuggle || (ls.tail_ops >= 3 && ts >= 16)
                || ls.tail_ident_chars >= 24;
            if dense {
                let msg = format!("line width {} > {MAX_COLS} with dense/smuggled code tail", ls.raw_cols);
                push_finding(findings, Severity::Veto, "hidden_dense_tail", ln, msg);
            }
        }
        let is_blank = !ls.has_code && !ls.has_nondoc_comment
            && !ls.has_doc_comment && !ls.has_safety_comment && !ls.in_multiline_str;
        if is_blank {
            acc.record_blank(ln, metrics);
        } else {
            acc.consec_blank = 0;
            acc.nonblank_run += 1;
            if acc.nonblank_run >= 2 { acc.alt_blank_run = 0; }
            acc.nonblank_prod += 1;
        }
        if ls.has_nondoc_comment && !ls.has_code && !ls.has_doc_comment && !ls.has_safety_comment {
            acc.record_pure_nondoc(lines, ln, ls.comment_slices.first().copied().unwrap_or("").trim());
        }
    }
    metrics.nondoc_comment_lines = acc.nondoc_pure;
    emit_blank_and_comment_vetoes(metrics, &acc, findings);
}

fn is_comment_echo_pair(lines: &[LineScan<'_>], ln: usize, body: &str) -> bool {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "from", "that", "this", "into", "not", "are", "use", "all",
        "can", "has", "was", "but", "when", "then", "else", "only", "each", "also",
    ];
    let mut c_words = BTreeSet::new();
    for raw in crate::lex::iter_idents(body.as_bytes()) {
        if raw.len() >= 3 {
            let w = std::str::from_utf8(raw).unwrap_or("").to_ascii_lowercase();
            if !STOP.contains(&w.as_str()) { c_words.insert(w); }
        }
    }
    if c_words.len() < 2 { return false; }
    let end = (ln + 2).min(lines.len().saturating_sub(1));
    for next_ls in &lines[(ln + 1).min(lines.len())..=end] {
        if !next_ls.in_test && next_ls.has_code {
            let shared = c_words.iter()
                .filter(|w| next_ls.code_idents.iter().any(|id| id.eq_ignore_ascii_case(w)))
                .count();
            return shared >= 2 && (shared * 100) / c_words.len() >= 50;
        }
    }
    false
}

fn emit_blank_and_comment_vetoes(
    metrics: &StrictMetrics, acc: &BlankCommentAcc, findings: &mut Vec<StrictFinding>,
) {
    if metrics.prod_lines >= 12 {
        let blank_ratio = metrics.blank_prod_lines as f64 / metrics.prod_lines as f64;
        if acc.max_consec_blank >= 3 {
            let msg = format!("{} consecutive blank lines (>= 3)", acc.max_consec_blank);
            push_finding(findings, Severity::Veto, "blank_line_stuffing", acc.max_consec_blank_line, msg);
        } else if metrics.blank_prod_lines >= 6 && blank_ratio > 0.32 {
            let msg = format!("excessive blank-line ratio {}/{}", metrics.blank_prod_lines, metrics.prod_lines);
            push_finding(findings, Severity::Veto, "blank_line_stuffing", 1, msg);
        } else if acc.max_alt_blank >= 5 {
            let msg = format!("{} interleaved blank pairs", acc.max_alt_blank);
            push_finding(findings, Severity::Veto, "blank_line_stuffing", acc.max_alt_blank_line, msg);
        }
    }
    if acc.echo_pairs >= 3 {
        let msg = format!("{} non-doc comments echoing adjacent code", acc.echo_pairs);
        push_finding(findings, Severity::Veto, "comment_echo_stuffing", acc.first_echo, msg);
    } else if acc.cmt_code >= 4 {
        let msg = format!("{} lines of commented-out code", acc.cmt_code);
        push_finding(findings, Severity::Veto, "comment_echo_stuffing", acc.first_cmt_code, msg);
    } else if acc.nonblank_prod >= 10 && acc.nondoc_pure >= 6
        && (acc.nondoc_pure as f64 / acc.nonblank_prod as f64) > 0.35
    {
        let msg = format!("excessive non-doc comment stuffing {}/{}", acc.nondoc_pure, acc.nonblank_prod);
        push_finding(findings, Severity::Veto, "comment_echo_stuffing", 1, msg);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BraceKind { FnBody, ControlFlow, Closure, Other }
struct BraceFrame { kind: BraceKind, line: usize, line_indent: usize }
struct PendingCf { paren_depth: usize, bracket_depth: usize, kw_line: usize, line_indent: usize }
struct PendingFn<'a> {
    name: &'a str, line: usize, paren_depth: usize, bracket_depth: usize,
    total_params: usize, golf_params: usize, arity: usize, bool_params: usize, is_pub: bool,
}
struct FnAcc<'a> {
    name: &'a str, line: usize, open_line: usize, open_tok_idx: usize, is_pub: bool,
    arity: usize, bool_params: usize, max_cf_depth: usize, cognitive: usize,
    total_bindings: usize, golf_1char: usize, indent_compressed: usize,
    packed_cf_line: bool, cf_opens_by_line: BTreeMap<usize, usize>, last_bool_op: Option<LexKind>,
}

#[derive(Default)]
struct Delims { p: usize, b: usize, c: usize, a: usize }
impl Delims {
    fn step(&mut self, kind: LexKind) {
        match kind {
            LexKind::LParen => self.p += 1, LexKind::RParen => self.p = self.p.saturating_sub(1),
            LexKind::LBracket => self.b += 1, LexKind::RBracket => self.b = self.b.saturating_sub(1),
            LexKind::LBrace => self.c += 1, LexKind::RBrace => self.c = self.c.saturating_sub(1),
            LexKind::Lt => self.a += 1, LexKind::Gt => self.a = self.a.saturating_sub(1),
            _ => {}
        }
    }
    fn top_pbc(&self) -> bool { self.p == 0 && self.b == 0 && self.c == 0 }
}

fn is_value_ending_tok(tok: &StrictTok<'_>) -> bool {
    match tok.kind {
        LexKind::Number | LexKind::StrLit | LexKind::CharLit
        | LexKind::RParen | LexKind::RBracket | LexKind::RBrace => true,
        LexKind::Ident => !matches!(
            tok.text,
            "return" | "move" | "in" | "if" | "else" | "while"
                | "for" | "match" | "switch" | "let" | "break" | "continue" | "async"
        ),
        _ => false,
    }
}

fn skip_balanced(tokens: &[StrictTok<'_>], mut pos: usize, open: LexKind, close: LexKind) -> usize {
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

fn extract_pattern_idents(slice: &[StrictTok<'_>], total: &mut usize, golf: &mut usize) {
    for (idx, tok) in slice.iter().enumerate() {
        if tok.kind != LexKind::Ident { continue; }
        let name = tok.text;
        let skip = matches!(name, "mut" | "ref" | "self" | "Self" | "box" | "Some" | "None" | "Ok" | "Err")
            || name.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
            || slice.get(idx + 1).is_some_and(|n| matches!(n.kind, LexKind::Colon | LexKind::DoubleColon))
            || name.starts_with('_');
        if !skip {
            *total += 1;
            let short_ok = matches!(name, "i" | "j" | "k" | "x" | "y" | "n" | "id" | "ok" | "err");
            if name.chars().count() == 1 && !short_ok { *golf += 1; }
        }
    }
}

fn count_fn_params(tokens: &[StrictTok<'_>], start: usize, end: usize, is_go: bool) -> (usize, usize, usize, usize) {
    let (mut total, mut golf, mut arity, mut bools) = (0, 0, 0, 0);
    let (mut seg_start, mut d) = (start, Delims::default());
    for idx in start..=end {
        if idx == end || (tokens[idx].kind == LexKind::Comma && d.top_pbc() && d.a == 0) {
            let seg = &tokens[seg_start..idx];
            if !seg.is_empty() && !seg.iter().any(|t| t.kind == LexKind::Ident && t.text == "self") {
                arity += 1;
                if seg.iter().any(|t| t.kind == LexKind::Ident && t.text == "bool") { bools += 1; }
            }
            if !is_go {
                let mut cd = Delims::default();
                let cp = seg.iter().position(|t| {
                    let hit = t.kind == LexKind::Colon && cd.top_pbc();
                    cd.step(t.kind);
                    hit
                });
                if let Some(cp) = cp { extract_pattern_idents(&seg[..cp], &mut total, &mut golf); }
            }
            seg_start = idx + 1;
        } else { d.step(tokens[idx].kind); }
    }
    (total, golf, arity, bools)
}

fn is_pub_before(tokens: &[StrictTok<'_>], idx: usize) -> bool {
    tokens[idx.saturating_sub(4)..idx].iter().any(|t| t.kind == LexKind::Ident && t.text == "pub")
}

fn parse_fn_header<'a>(tokens: &[StrictTok<'a>], fn_idx: usize, is_go: bool) -> Option<(PendingFn<'a>, bool)> {
    let mut np = fn_idx + 1;
    if is_go && tokens.get(np).is_some_and(|t| t.kind == LexKind::LParen) {
        np = skip_balanced(tokens, np, LexKind::LParen, LexKind::RParen);
    }
    let name_tok = tokens.get(np).filter(|t| t.kind == LexKind::Ident)?;
    let mut pos = np + 1;
    if !is_go && tokens.get(pos).is_some_and(|t| t.kind == LexKind::Lt) {
        pos = skip_balanced(tokens, pos, LexKind::Lt, LexKind::Gt);
    }
    if !tokens.get(pos).is_some_and(|t| t.kind == LexKind::LParen) { return None; }
    let p_start = pos + 1;
    pos = skip_balanced(tokens, pos, LexKind::LParen, LexKind::RParen);
    let (total_params, golf_params, arity, bool_params) =
        count_fn_params(tokens, p_start, pos.saturating_sub(1), is_go);
    let has_body = tokens[pos..].iter().find_map(|t| match t.kind {
        LexKind::LBrace => Some(true),
        LexKind::Semi => Some(false),
        _ => None,
    }).unwrap_or(false);
    let pfn = PendingFn {
        name: name_tok.text, line: name_tok.line, paren_depth: 0, bracket_depth: 0,
        total_params, golf_params, arity, bool_params,
        is_pub: !is_go && is_pub_before(tokens, fn_idx),
    };
    Some((pfn, has_body))
}

fn is_match_guard_if(tokens: &[StrictTok<'_>], if_idx: usize) -> bool {
    let mut d = Delims::default();
    for tok in &tokens[if_idx + 1..] {
        if d.p == 0 && d.b == 0 {
            if tok.kind == LexKind::FatArrow { return true; }
            if matches!(tok.kind, LexKind::LBrace | LexKind::Semi | LexKind::RBrace) { return false; }
        }
        d.step(tok.kind);
    }
    false
}

fn closure_opens_brace(tokens: &[StrictTok<'_>], pipe_idx: usize) -> bool {
    let (mut pos, start_ln) = (pipe_idx + 1, tokens[pipe_idx].line);
    while pos < tokens.len() && tokens[pos].line <= start_ln + 3 {
        match tokens[pos].kind {
            LexKind::Pipe => {
                let mut after = pos + 1;
                if tokens.get(after).is_some_and(|t| t.kind == LexKind::Arrow) {
                    after += 1;
                    while after < tokens.len()
                        && !matches!(tokens[after].kind, LexKind::LBrace | LexKind::Semi | LexKind::Comma)
                    { after += 1; }
                }
                return tokens.get(after).is_some_and(|t| t.kind == LexKind::LBrace);
            }
            LexKind::Semi | LexKind::LBrace | LexKind::RBrace => return false,
            _ => pos += 1,
        }
    }
    false
}

#[derive(Default)]
struct FnAnalyzer<'a> {
    brace_stack: Vec<BraceFrame>, fn_stack: Vec<FnAcc<'a>>, pending_fn: Option<PendingFn<'a>>,
    pending_cf: Vec<PendingCf>, pending_closure: Vec<(usize, usize)>,
    paren_depth: usize, bracket_depth: usize,
}

impl<'a> FnAnalyzer<'a> {
    fn clear_bool_op(&mut self) { if let Some(cur) = self.fn_stack.last_mut() { cur.last_bool_op = None; } }

    fn handle_lbrace(&mut self, tok_idx: usize, tokens: &[StrictTok<'a>], lines: &[LineScan<'_>]) {
        let tok = &tokens[tok_idx];
        if let Some(pfn) = self.pending_fn.take() {
            if pfn.paren_depth == self.paren_depth && pfn.bracket_depth == self.bracket_depth {
                self.brace_stack.push(BraceFrame {
                    kind: BraceKind::FnBody, line: tok.line, line_indent: tok.line_indent,
                });
                self.fn_stack.push(FnAcc {
                    name: pfn.name, line: pfn.line, open_line: tok.line, open_tok_idx: tok_idx,
                    is_pub: pfn.is_pub, arity: pfn.arity, bool_params: pfn.bool_params,
                    max_cf_depth: 0, cognitive: 0, total_bindings: pfn.total_params,
                    golf_1char: pfn.golf_params, indent_compressed: 0, packed_cf_line: false,
                    cf_opens_by_line: BTreeMap::new(), last_bool_op: None,
                });
                return;
            }
            self.pending_fn = Some(pfn);
        }
        let match_cf = self.pending_cf.last().is_some_and(|p| {
            p.paren_depth == self.paren_depth && p.bracket_depth == self.bracket_depth
        });
        if match_cf && let Some(pcf) = self.pending_cf.pop() {
            self.push_cf_brace(pcf, tok, lines);
            return;
        }
        let match_cl = self.pending_closure.last()
            .is_some_and(|&(cp, cb)| cp == self.paren_depth && cb == self.bracket_depth);
        if match_cl {
            self.pending_closure.pop();
            self.brace_stack.push(BraceFrame {
                kind: BraceKind::Closure, line: tok.line, line_indent: tok.line_indent,
            });
        } else {
            self.brace_stack.push(BraceFrame {
                kind: BraceKind::Other, line: tok.line, line_indent: tok.line_indent,
            });
        }
        self.clear_bool_op();
    }

    fn push_cf_brace(&mut self, pcf: PendingCf, tok: &StrictTok<'a>, lines: &[LineScan<'_>]) {
        let parent_cf = self.brace_stack.iter().rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .find(|f| f.kind == BraceKind::ControlFlow);
        let compressed = parent_cf.is_some_and(|pf| {
            pcf.kw_line > pf.line && pcf.line_indent <= pf.line_indent + 1
        });
        self.brace_stack.push(BraceFrame {
            kind: BraceKind::ControlFlow, line: pcf.kw_line, line_indent: pcf.line_indent,
        });
        let cf_frames = || self.brace_stack.iter().rev()
            .take_while(|f| f.kind != BraceKind::FnBody)
            .filter(|f| f.kind == BraceKind::ControlFlow);
        let depth = cf_frames().count();
        let same_ln = cf_frames().filter(|f| f.line == tok.line).count();
        if let Some(cur) = self.fn_stack.last_mut() {
            cur.max_cf_depth = cur.max_cf_depth.max(depth);
            if compressed { cur.indent_compressed += 1; }
            let cnt = cur.cf_opens_by_line.entry(tok.line).or_insert(0);
            *cnt += 1;
            let semis = lines.get(tok.line).map_or(0, |l| l.total_semis);
            if same_ln >= 3 || (*cnt >= 3 && (same_ln >= 2 || semis >= 2)) { cur.packed_cf_line = true; }
            cur.last_bool_op = None;
        }
    }

    fn add_cf(&mut self, tok: &StrictTok<'a>, cog_add: usize) {
        if let Some(cur) = self.fn_stack.last_mut() {
            cur.cognitive += cog_add;
            cur.last_bool_op = None;
        }
        self.pending_cf.push(PendingCf {
            paren_depth: self.paren_depth, bracket_depth: self.bracket_depth,
            kw_line: tok.line, line_indent: tok.line_indent,
        });
    }

    fn handle_fn_ident(&mut self, tokens: &[StrictTok<'a>], idx: usize, prev: Option<&StrictTok<'a>>, is_go: bool) {
        let tok = &tokens[idx];
        let nest = self.brace_stack.iter().rev().take_while(|f| f.kind != BraceKind::FnBody)
            .filter(|f| matches!(f.kind, BraceKind::ControlFlow | BraceKind::Closure)).count();
        match tok.text {
            "if" => {
                if !is_go && is_match_guard_if(tokens, idx) {
                    if let Some(cur) = self.fn_stack.last_mut() {
                        cur.cognitive += 1;
                        cur.last_bool_op = None;
                    }
                    return;
                }
                let else_if = prev.is_some_and(|p| p.kind == LexKind::Ident && p.text == "else");
                self.add_cf(tok, if else_if { 1 } else { 1 + nest });
            }
            "else" if !tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Ident && t.text == "if") => {
                let after_brace = prev.is_some_and(|p| p.kind == LexKind::RBrace);
                self.add_cf(tok, if after_brace { 1 } else { 1 + nest });
            }
            "for" if !tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::Lt) => {
                self.add_cf(tok, 1 + nest);
            }
            "while" | "loop" | "match" | "switch" => self.add_cf(tok, 1 + nest),
            "let" if !is_go => self.handle_let(tokens, idx, prev),
            _ => {}
        }
    }

    fn handle_let(&mut self, tokens: &[StrictTok<'a>], idx: usize, prev: Option<&StrictTok<'a>>) {
        let is_let_chain = prev.is_some_and(|p| {
            p.kind == LexKind::AndAnd || (p.kind == LexKind::Ident && matches!(p.text, "if" | "while"))
        });
        if is_let_chain { return; }
        let (mut pos, mut d) = (idx + 1, Delims::default());
        while pos < tokens.len() {
            if d.top_pbc() && matches!(tokens[pos].kind, LexKind::Eq | LexKind::Colon | LexKind::Semi) {
                break;
            }
            d.step(tokens[pos].kind);
            pos += 1;
        }
        if let Some(cur) = self.fn_stack.last_mut() {
            extract_pattern_idents(&tokens[idx + 1..pos], &mut cur.total_bindings, &mut cur.golf_1char);
        }
    }
}

fn count_pub_struct_fields(tokens: &[StrictTok<'_>], struct_idx: usize) -> usize {
    let mut pos = struct_idx + 1;
    while pos < tokens.len() && !matches!(tokens[pos].kind, LexKind::LBrace | LexKind::Semi) { pos += 1; }
    if tokens.get(pos).is_none_or(|t| t.kind != LexKind::LBrace) { return 0; }
    let end = skip_balanced(tokens, pos, LexKind::LBrace, LexKind::RBrace);
    (pos + 1..end.saturating_sub(1)).filter(|&i| {
        tokens[i].kind == LexKind::Ident && tokens[i].text == "pub"
            && tokens.get(i + 1).is_some_and(|n| n.kind == LexKind::Ident)
            && tokens.get(i + 2).is_some_and(|n| n.kind == LexKind::Colon)
    }).count()
}

fn analyze_functions_and_blocks(
    tokens: &[StrictTok<'_>], lines: &[LineScan<'_>], is_go: bool, metrics: &mut StrictMetrics,
    block: &mut BlockModuleMetrics, findings: &mut Vec<StrictFinding>,
) -> usize {
    let (mut state, mut pub_fields) = (FnAnalyzer::default(), 0usize);
    for idx in 0..tokens.len() {
        let tok = &tokens[idx];
        if tok.in_test { continue; }
        let prev = (idx > 0 && !tokens[idx - 1].in_test).then(|| &tokens[idx - 1]);
        pub_fields += check_decl_and_surface(tokens, idx, is_go, &mut state, block, findings);
        match tok.kind {
            LexKind::LParen => state.paren_depth += 1,
            LexKind::RParen => state.paren_depth = state.paren_depth.saturating_sub(1),
            LexKind::LBracket => state.bracket_depth += 1,
            LexKind::RBracket => state.bracket_depth = state.bracket_depth.saturating_sub(1),
            LexKind::Semi | LexKind::Comma | LexKind::FatArrow => state.clear_bool_op(),
            LexKind::LBrace => state.handle_lbrace(idx, tokens, lines),
            LexKind::RBrace => {
                if let Some(popped) = state.brace_stack.pop() {
                    if popped.kind == BraceKind::FnBody {
                        if let Some(acc) = state.fn_stack.pop() {
                            let body = &tokens[(acc.open_tok_idx + 1).min(idx)..idx];
                            finalize_fn(&acc, tok.line, body, metrics, block, findings);
                        }
                    } else { state.clear_bool_op(); }
                }
            }
            _ => step_fn_body_token(&mut state, tokens, idx, prev, is_go),
        }
    }
    if !is_go && metrics.prod_bindings >= 10
        && (metrics.prod_golf_1char * 100) / metrics.prod_bindings >= 40
    {
        let msg = format!(
            "file-wide 1-char identifier golf {}/{}",
            metrics.prod_golf_1char, metrics.prod_bindings
        );
        push_finding(findings, Severity::Veto, "identifier_golf", 1, msg);
    }
    pub_fields
}

fn check_decl_and_surface<'a>(
    tokens: &[StrictTok<'a>], idx: usize, is_go: bool, state: &mut FnAnalyzer<'a>,
    block: &mut BlockModuleMetrics, findings: &mut Vec<StrictFinding>,
) -> usize {
    let (tok, mut pub_fields) = (&tokens[idx], 0usize);
    if !is_go && tok.kind == LexKind::Ident {
        if matches!(tok.text, "struct" | "enum" | "trait") {
            if is_pub_before(tokens, idx) { block.pub_items += 1; }
            if tok.text == "struct" { pub_fields = count_pub_struct_fields(tokens, idx); }
            let golf_name = tokens.get(idx + 1)
                .filter(|n| n.kind == LexKind::Ident && n.text.chars().count() == 1);
            if let Some(nt) = golf_name {
                let msg = format!("1-char {} name `{}`", tok.text, nt.text);
                push_finding(findings, Severity::Veto, "identifier_golf", nt.line, msg);
            }
        } else if matches!(tok.text, "const" | "type")
            && state.fn_stack.is_empty() && is_pub_before(tokens, idx)
        {
            block.pub_items += 1;
        }
    }
    if tok.kind == LexKind::Ident && ((is_go && tok.text == "func") || (!is_go && tok.text == "fn"))
        && let Some((mut pfn, has_body)) = parse_fn_header(tokens, idx, is_go)
    {
        if pfn.is_pub { block.pub_items += 1; }
        if !is_go && pfn.name.chars().count() == 1 {
            let msg = format!("1-char prod function name `{}`", pfn.name);
            push_finding(findings, Severity::Veto, "identifier_golf", pfn.line, msg);
        }
        if has_body {
            pfn.paren_depth = state.paren_depth;
            pfn.bracket_depth = state.bracket_depth;
            state.pending_fn = Some(pfn);
        }
    }
    pub_fields
}

fn step_fn_body_token<'a>(
    state: &mut FnAnalyzer<'a>, tokens: &[StrictTok<'a>], idx: usize,
    prev: Option<&StrictTok<'a>>, is_go: bool,
) {
    if state.fn_stack.is_empty() { return; }
    match tokens[idx].kind {
        LexKind::AndAnd => {
            if let Some(cur) = state.fn_stack.last_mut() && cur.last_bool_op != Some(LexKind::AndAnd) {
                cur.cognitive += 1;
                cur.last_bool_op = Some(LexKind::AndAnd);
            }
        }
        LexKind::OrOr => {
            if prev.is_some_and(is_value_ending_tok) {
                if let Some(cur) = state.fn_stack.last_mut()
                    && cur.last_bool_op != Some(LexKind::OrOr)
                {
                    cur.cognitive += 1;
                    cur.last_bool_op = Some(LexKind::OrOr);
                }
            } else if !is_go && tokens.get(idx + 1).is_some_and(|t| t.kind == LexKind::LBrace) {
                state.pending_closure.push((state.paren_depth, state.bracket_depth));
            }
        }
        LexKind::Pipe if !is_go => {
            if prev.is_none_or(|p| !is_value_ending_tok(p)) && closure_opens_brace(tokens, idx) {
                state.pending_closure.push((state.paren_depth, state.bracket_depth));
            }
        }
        LexKind::Ident => state.handle_fn_ident(tokens, idx, prev, is_go),
        _ => {}
    }
}

fn finalize_fn(
    acc: &FnAcc<'_>, close_line: usize, body: &[StrictTok<'_>], metrics: &mut StrictMetrics,
    block: &mut BlockModuleMetrics, findings: &mut Vec<StrictFinding>,
) {
    let fn_loc = close_line.saturating_sub(acc.open_line) + 1;
    block.fn_count += 1;
    block.max_fn_loc = block.max_fn_loc.max(fn_loc);
    block.max_cognitive = block.max_cognitive.max(acc.cognitive);
    block.max_params = block.max_params.max(acc.arity);
    block.max_bool_params = block.max_bool_params.max(acc.bool_params);
    if fn_loc <= 4 && !body.is_empty() && body.len() <= 14
        && !body.iter().any(|t| t.kind == LexKind::Semi)
    {
        let self_field = body.windows(3).any(|w| {
            w[0].kind == LexKind::Ident && w[0].text == "self"
                && w[1].kind == LexKind::Dot && w[2].kind == LexKind::Ident
        });
        let is_getter = self_field && !body.iter().any(|t| t.kind == LexKind::LParen);
        if acc.is_pub && is_getter {
            block.getter_leaks += 1;
        } else if !is_getter && body.iter().any(|t| t.kind == LexKind::LParen) {
            block.passthrough_fns += 1;
        }
    }
    metrics.max_cf_depth = metrics.max_cf_depth.max(acc.max_cf_depth);
    metrics.max_cognitive = metrics.max_cognitive.max(acc.cognitive);
    metrics.prod_bindings += acc.total_bindings;
    metrics.prod_golf_1char += acc.golf_1char;

    let too_deep = acc.max_cf_depth >= MAX_CF_DEPTH_VETO
        || acc.cognitive > MAX_COGNITIVE_VETO
        || acc.indent_compressed >= 4
        || acc.packed_cf_line;
    if too_deep {
        let msg = format!(
            "fn `{}`: cf_depth={} (<{MAX_CF_DEPTH_VETO}), cognitive={} (<={MAX_COGNITIVE_VETO}), compressed={}, packed={}",
            acc.name, acc.max_cf_depth, acc.cognitive, acc.indent_compressed, acc.packed_cf_line
        );
        push_finding(findings, Severity::Veto, "deep_nesting_spaghetti", acc.line, msg);
    } else if acc.max_cf_depth >= WARN_CF_DEPTH || acc.cognitive >= WARN_COGNITIVE {
        let msg = format!(
            "fn `{}` borderline: cf_depth={}, cognitive={}",
            acc.name, acc.max_cf_depth, acc.cognitive
        );
        push_finding(findings, Severity::Warning, "complexity_borderline", acc.line, msg);
    }
    if acc.total_bindings >= 5 && acc.golf_1char >= 4
        && (acc.golf_1char * 100) / acc.total_bindings >= 55
    {
        let msg = format!(
            "fn `{}` 1-char variable golf {}/{}", acc.name, acc.golf_1char, acc.total_bindings
        );
        push_finding(findings, Severity::Veto, "identifier_golf", acc.line, msg);
    }
}
