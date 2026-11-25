use crate::lex::{ident_byte, left_bound};
use crate::platform::{find_secret, walk_tree};
use crate::source::{ext_of, ext_str, parse_deps, rel_norm, supported_ext};
use crate::types::{InfraFinding, UnusedDep};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use fixedbitset::FixedBitSet;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Eco {
    Npm = 0,
    Cargo = 1,
    Python = 2,
    Go = 3,
}

impl Eco {
    const ALL: [Self; 4] = [Self::Npm, Self::Cargo, Self::Python, Self::Go];

    fn from_ext(ext: &str) -> Self {
        if ext.eq_ignore_ascii_case("rs") {
            Self::Cargo
        } else if ext.eq_ignore_ascii_case("py") {
            Self::Python
        } else if ext.eq_ignore_ascii_case("go") {
            Self::Go
        } else {
            Self::Npm
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "npm" => Self::Npm,
            "cargo" => Self::Cargo,
            "python" => Self::Python,
            _ => Self::Go,
        }
    }

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    fn word_bound(self) -> bool {
        self != Self::Go
    }

    #[inline]
    fn case_insensitive(self) -> bool {
        self == Self::Python
    }
}

struct ParsedDep {
    manifest: String,
    ecosystem: &'static str,
    package: String,
    eco: Eco,
    slot: usize,
}

#[derive(Default)]
struct EcoBuilder {
    pkg_slots: HashMap<String, usize>,
    pat_ids: HashMap<String, usize>,
    patterns: Vec<String>,
    pat_slots: Vec<Vec<usize>>,
}

impl EcoBuilder {
    fn add_dep(&mut self, eco: Eco, package: &str) -> usize {
        let key = if eco.case_insensitive() {
            package.to_ascii_lowercase()
        } else {
            package.to_string()
        };
        let next_slot = self.pkg_slots.len();
        let slot = *self.pkg_slots.entry(key.clone()).or_insert(next_slot);
        if key.is_empty() {
            return slot;
        }
        self.add_pattern(&key, slot);
        if matches!(eco, Eco::Cargo | Eco::Python) && key.contains('-') {
            self.add_pattern(&key.replace('-', "_"), slot);
        }
        slot
    }

    fn add_pattern(&mut self, pat: &str, slot: usize) {
        let pat_id = if let Some(&existing) = self.pat_ids.get(pat) {
            existing
        } else {
            let new_id = self.patterns.len();
            self.patterns.push(pat.to_string());
            self.pat_ids.insert(pat.to_string(), new_id);
            self.pat_slots.push(Vec::new());
            new_id
        };
        let slots = &mut self.pat_slots[pat_id];
        if !slots.contains(&slot) {
            slots.push(slot);
        }
    }

    fn build(self, eco: Eco) -> Option<EcoMatcher> {
        let dep_count = self.pkg_slots.len();
        if dep_count == 0 || self.patterns.is_empty() {
            return None;
        }
        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(eco.case_insensitive())
            .build(&self.patterns)
            .ok()?;
        Some(EcoMatcher {
            ac,
            pat_slots: self.pat_slots,
            dep_count,
            word_bound: eco.word_bound(),
            hits: FixedBitSet::with_capacity(dep_count),
        })
    }
}

struct EcoMatcher {
    ac: AhoCorasick,
    pat_slots: Vec<Vec<usize>>,
    dep_count: usize,
    word_bound: bool,
    hits: FixedBitSet,
}

impl EcoMatcher {
    #[inline]
    fn is_complete(&self) -> bool {
        self.hits.count_ones(..) == self.dep_count
    }

    fn scan_bytes(&mut self, bytes: &[u8]) {
        if self.is_complete() || bytes.is_empty() {
            return;
        }
        for mat in self.ac.find_overlapping_iter(bytes) {
            if self.word_bound && !is_word_span(bytes, mat.start(), mat.end()) {
                continue;
            }
            let mut added = false;
            for &slot in &self.pat_slots[mat.pattern().as_usize()] {
                added |= !self.hits.put(slot);
            }
            if added && self.is_complete() {
                return;
            }
        }
    }
}

#[inline]
fn is_word_span(bytes: &[u8], start: usize, end: usize) -> bool {
    left_bound(bytes, start) && (end == bytes.len() || !ident_byte(bytes[end]))
}

#[derive(Clone, Copy)]
enum Kind {
    Tf,
    Docker,
    Gha,
    Sql,
}

const UNTRUSTED_GHA_SOURCES: &[&str] = &[
    "github.event.issue.title",
    "github.event.issue.body",
    "github.event.pull_request.title",
    "github.event.pull_request.body",
    "github.event.pull_request.head.ref",
    "github.event.pull_request.head.label",
    "github.event.comment.body",
    "github.event.review.body",
    "github.event.review_comment.body",
    "github.event.discussion.title",
    "github.event.discussion.body",
    "github.head_ref",
    "github.event.head_commit.message",
    "github.event.head_commit.author",
];

fn push_finding(
    out: &mut Vec<InfraFinding>,
    file: &str,
    line: usize,
    severity: &str,
    rule_id: &str,
    message: &str,
) {
    out.push(InfraFinding {
        file: file.to_string(),
        line,
        severity: severity.into(),
        rule_id: rule_id.into(),
        message: message.into(),
    });
}

fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hbytes = haystack.as_bytes();
    let nbytes = needle.as_bytes();
    if nbytes.is_empty() || nbytes.len() > hbytes.len() {
        return None;
    }
    hbytes
        .windows(nbytes.len())
        .position(|win| win.eq_ignore_ascii_case(nbytes))
}

fn parse_ignore_directive(raw_line: &str) -> Option<Vec<String>> {
    let pos = find_ascii_ci(raw_line, "refit:")?;
    let rest = raw_line[pos + 6..].trim_start();
    if rest.len() < 6 || !rest[..6].eq_ignore_ascii_case("ignore") {
        return None;
    }
    let after = rest[6..].trim_start();
    if let Some(bracketed) = after.strip_prefix('[')
        && let Some((inner, _)) = bracketed.split_once(']')
    {
        let rules: Vec<String> = inner
            .split(',')
            .map(|tok| tok.trim().to_ascii_lowercase())
            .filter(|tok| !tok.is_empty())
            .collect();
        return Some(rules);
    }
    Some(Vec::new())
}

fn is_suppressed(rule_id: &str, inline: Option<&[String]>, prev: Option<&[String]>) -> bool {
    let matches_list = |list: &[String]| {
        list.is_empty() || list.iter().any(|item| item.eq_ignore_ascii_case(rule_id))
    };
    inline.is_some_and(matches_list) || prev.is_some_and(matches_list)
}

/// Scan Terraform, Dockerfiles, GitHub Actions, SQL, and unused manifest dependencies.
pub fn scan_parts(
    repo_root: &Path,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    let mut infra_count = 0usize;
    let mut findings = Vec::new();
    let mut manifests = Vec::new();
    let mut source_paths: Vec<(Eco, PathBuf)> = Vec::new();

    for abs in walk_tree(repo_root, None, true) {
        let Some(rel) = rel_norm(repo_root, &abs) else { continue };
        let fname = abs.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ext = ext_str(&abs);

        if supported_ext(ext) {
            if preloaded.is_none() {
                source_paths.push((Eco::from_ext(ext), abs));
            }
            continue;
        }

        if matches!(fname, "package.json" | "Cargo.toml" | "pyproject.toml" | "go.mod") {
            if let Ok(txt) = fs::read_to_string(&abs) {
                manifests.push((rel, fname.to_string(), txt));
            }
            continue;
        }

        let ext_lo = ext_of(&abs);
        let Some(kind) = infra_kind(&rel, fname, &ext_lo) else { continue };
        infra_count += 1;
        let Ok(txt) = fs::read_to_string(&abs) else { continue };
        scan_file(&rel, kind, &txt, &mut findings);
    }

    let unused = find_unused_deps(&manifests, preloaded, &source_paths);
    (infra_count, unused, findings)
}

fn infra_kind(rel: &str, fname: &str, ext: &str) -> Option<Option<Kind>> {
    if fname.starts_with("Dockerfile") || fname.ends_with(".dockerfile") {
        return Some(Some(Kind::Docker));
    }
    if ext == "tf" || ext == "tfvars" {
        return Some(Some(Kind::Tf));
    }
    if rel.starts_with(".github/workflows/") && matches!(ext, "yml" | "yaml") {
        return Some(Some(Kind::Gha));
    }
    if ext == "sql" {
        return Some(Some(Kind::Sql));
    }
    if fname.starts_with("docker-compose") && matches!(ext, "yml" | "yaml") {
        return Some(None);
    }
    None
}

fn scan_file(rel: &str, kind: Option<Kind>, txt: &str, out: &mut Vec<InfraFinding>) {
    for (idx, line) in txt.lines().enumerate() {
        if let Some(sk) = find_secret(line) {
            push_finding(
                out,
                rel,
                idx + 1,
                "critical",
                &format!("infra_secret_{sk}"),
                "Hardcoded credential/secret committed in infrastructure config",
            );
        }
    }

    match kind {
        Some(Kind::Gha) => scan_gha(rel, txt, out),
        Some(Kind::Docker) => scan_docker(rel, txt, out),
        Some(Kind::Tf) => scan_tf(rel, txt, out),
        Some(Kind::Sql) => scan_sql(rel, txt, out),
        None => {}
    }
}

fn strip_yaml_inline_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    for idx in 0..bytes.len() {
        let cur = bytes[idx];
        if cur == b'\'' && !in_double {
            in_single = !in_single;
        } else if cur == b'"' && !in_single {
            in_double = !in_double;
        } else if cur == b'#'
            && !in_single
            && !in_double
            && (idx == 0 || bytes[idx - 1].is_ascii_whitespace())
        {
            return &line[..idx];
        }
    }
    line
}

fn extract_gha_uses(code: &str) -> Option<&str> {
    let rest = code
        .strip_prefix("uses:")
        .or_else(|| code.strip_prefix("- uses:"))?;
    let val = rest.trim().trim_matches(['"', '\'']);
    (!val.is_empty()).then_some(val)
}

fn is_pinned_gha_action(uses_ref: &str) -> bool {
    if uses_ref.starts_with("./") || uses_ref.starts_with("../") {
        return true;
    }
    if uses_ref.starts_with("docker://") {
        return uses_ref.contains("@sha256:");
    }
    let Some((_, rev)) = uses_ref.rsplit_once('@') else {
        return false;
    };
    rev.len() == 40 && rev.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn extract_gha_sink_rhs(trimmed: &str) -> Option<&str> {
    let body = trimmed.strip_prefix("- ").map(str::trim_start).unwrap_or(trimmed);
    body.strip_prefix("run:")
        .or_else(|| body.strip_prefix("script:"))
        .map(str::trim)
}

fn is_yaml_block_scalar(rhs: &str) -> bool {
    let clean = strip_yaml_inline_comment(rhs).trim();
    matches!(clean, "|" | "|-" | "|+" | ">" | ">-" | ">+")
}

fn is_untrusted_gha_source(expr_lo: &str) -> bool {
    if UNTRUSTED_GHA_SOURCES.iter().any(|src| expr_lo.contains(src)) {
        return true;
    }
    let bad_commits = expr_lo.contains("github.event.commits")
        && (expr_lo.contains(".message") || expr_lo.contains(".author"));
    let bad_pages = expr_lo.contains("github.event.pages") && expr_lo.contains(".page_name");
    bad_commits || bad_pages
}

fn check_gha_sink_line(
    rel: &str,
    ln: usize,
    line: &str,
    inline_ign: Option<&[String]>,
    prev_ign: Option<&[String]>,
    out: &mut Vec<InfraFinding>,
) {
    let mut rest = line;
    let mut flagged_inj = false;
    let mut flagged_sec = false;

    while let Some(open) = rest.find("${{") {
        let after_open = &rest[open + 3..];
        let Some(close) = after_open.find("}}") else { break };
        let expr_lo = after_open[..close].to_ascii_lowercase();
        if !flagged_inj
            && is_untrusted_gha_source(&expr_lo)
            && !is_suppressed("gha_script_injection", inline_ign, prev_ign)
        {
            flagged_inj = true;
            push_finding(
                out,
                rel,
                ln,
                "high",
                "gha_script_injection",
                "Untrusted `${{ github.event.* }}` interpolation in run/script risks injection",
            );
        }
        if !flagged_sec
            && expr_lo.contains("secrets.")
            && !is_suppressed("gha_secrets_outside_env", inline_ign, prev_ign)
        {
            flagged_sec = true;
            push_finding(
                out,
                rel,
                ln,
                "high",
                "gha_secrets_outside_env",
                "Secret `${{ secrets.* }}` interpolated directly in shell sink instead of `env:`",
            );
        }
        rest = &after_open[close + 2..];
    }
}

fn gha_key_column(indent: usize, trimmed: &str) -> usize {
    if let Some(after_dash) = trimmed.strip_prefix('-') {
        let ws = after_dash.bytes().take_while(|&b| b == b' ').count();
        if ws > 0 {
            return indent + 1 + ws;
        }
    }
    indent
}

fn scan_gha(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    let mut in_run_block = false;
    let mut run_indent = 0usize;
    let mut prev_ignore: Option<Vec<String>> = None;

    for (idx, raw) in txt.lines().enumerate() {
        let ln = idx + 1;
        let indent = raw.bytes().take_while(|&byte| byte == b' ').count();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            prev_ignore = None;
            continue;
        }
        if trimmed.starts_with('#') {
            prev_ignore = parse_ignore_directive(trimmed);
            continue;
        }
        if in_run_block && indent <= run_indent {
            in_run_block = false;
        }

        let inline_ign = parse_ignore_directive(raw);
        let code = strip_yaml_inline_comment(trimmed).trim();

        if !in_run_block {
            if code.contains("write-all")
                && !is_suppressed("gha_write_all_perms", inline_ign.as_deref(), prev_ignore.as_deref())
            {
                push_finding(
                    out,
                    rel,
                    ln,
                    "high",
                    "gha_write_all_perms",
                    "Workflow grants overly broad `write-all` permissions",
                );
            }
            if let Some(uses_ref) = extract_gha_uses(code)
                && !is_pinned_gha_action(uses_ref)
                && !is_suppressed("gha_unpinned_action", inline_ign.as_deref(), prev_ignore.as_deref())
            {
                push_finding(
                    out,
                    rel,
                    ln,
                    "medium",
                    "gha_unpinned_action",
                    "Action not pinned to immutable 40-hex commit SHA",
                );
            }
            if let Some(rhs) = extract_gha_sink_rhs(trimmed) {
                if is_yaml_block_scalar(rhs) {
                    in_run_block = true;
                    run_indent = gha_key_column(indent, trimmed);
                } else {
                    check_gha_sink_line(rel, ln, rhs, inline_ign.as_deref(), prev_ignore.as_deref(), out);
                }
            }
        } else {
            check_gha_sink_line(rel, ln, trimmed, inline_ign.as_deref(), prev_ignore.as_deref(), out);
        }
        prev_ignore = None;
    }
}

fn extract_docker_from_image(trimmed: &str) -> &str {
    for tok in trimmed.split_whitespace().skip(1) {
        if tok.starts_with("--") {
            continue;
        }
        if tok.starts_with('#') {
            return "";
        }
        return tok;
    }
    ""
}

fn is_whitelisted_nonroot_image(image: &str) -> bool {
    let lower = image.trim().to_ascii_lowercase();
    let base = lower.split('@').next().unwrap_or(&lower);
    base == "scratch"
        || base.ends_with(":nonroot")
        || base.contains(":nonroot-")
        || (base.contains("distroless") && base.contains("nonroot"))
}

fn scan_docker(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    let mut has_stage = false;
    let mut stage_non_root = false;
    let mut stage_from_line = 1usize;
    let mut stage_root_ignored = false;
    let mut prev_ignore: Option<Vec<String>> = None;

    for (idx, raw) in txt.lines().enumerate() {
        let ln = idx + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            prev_ignore = None;
            continue;
        }
        if trimmed.starts_with('#') {
            prev_ignore = parse_ignore_directive(trimmed);
            continue;
        }

        let inline_ign = parse_ignore_directive(raw);
        let up = trimmed.to_ascii_uppercase();
        if up.starts_with("FROM ") {
            has_stage = true;
            stage_from_line = ln;
            let image = extract_docker_from_image(trimmed);
            stage_non_root = is_whitelisted_nonroot_image(image);
            stage_root_ignored =
                is_suppressed("docker_runs_as_root", inline_ign.as_deref(), prev_ignore.as_deref());
            if image.to_ascii_lowercase().ends_with(":latest")
                && !is_suppressed("docker_latest_tag", inline_ign.as_deref(), prev_ignore.as_deref())
            {
                push_finding(
                    out,
                    rel,
                    ln,
                    "medium",
                    "docker_latest_tag",
                    "Unpinned `:latest` base image causes non-reproducible builds",
                );
            }
        } else if up.starts_with("USER ") {
            let user = trimmed
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .split(':')
                .next()
                .unwrap_or("");
            stage_non_root = !user.is_empty() && !user.eq_ignore_ascii_case("root") && user != "0";
        } else if up.starts_with("ADD ")
            && (trimmed.contains("http://") || trimmed.contains("https://"))
            && !is_suppressed("docker_add_remote_url", inline_ign.as_deref(), prev_ignore.as_deref())
        {
            push_finding(
                out,
                rel,
                ln,
                "high",
                "docker_add_remote_url",
                "Dockerfile `ADD` fetches remote URL without integrity checksum",
            );
        }

        if has_baked(trimmed, &up)
            && !is_suppressed("docker_baked_secret", inline_ign.as_deref(), prev_ignore.as_deref())
        {
            push_finding(
                out,
                rel,
                ln,
                "high",
                "docker_baked_secret",
                "Sensitive ENV/ARG value baked into Docker image layer",
            );
        }
        prev_ignore = None;
    }

    if has_stage && !stage_non_root && !stage_root_ignored {
        push_finding(
            out,
            rel,
            stage_from_line,
            "high",
            "docker_runs_as_root",
            "Final Dockerfile stage missing non-root `USER` instruction",
        );
    }
}

fn has_baked(trimmed: &str, up: &str) -> bool {
    let is_env = up.starts_with("ENV ") || up.starts_with("ARG ");
    let sensitive = up.contains("SECRET") || up.contains("PASSWORD") || up.contains("API_KEY");
    if !(is_env && sensitive) {
        return false;
    }

    let rest = trimmed[4..].trim();
    let val = rest
        .split_once('=')
        .or_else(|| rest.split_once(char::is_whitespace))
        .map(|(_, v)| v.trim().trim_matches(['"', '\'']))
        .unwrap_or("");
    !val.is_empty() && !val.starts_with('$')
}

fn strip_hcl_line(raw: &str, in_block_comment: &mut bool) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut in_quote = false;
    let mut pos = 0usize;

    while pos < bytes.len() {
        let cur = bytes[pos];
        if *in_block_comment {
            if cur == b'*' && bytes.get(pos + 1) == Some(&b'/') {
                *in_block_comment = false;
                pos += 2;
            } else {
                pos += 1;
            }
            continue;
        }
        if in_quote {
            out.push(cur as char);
            if cur == b'\\' && pos + 1 < bytes.len() {
                pos += 1;
                out.push(bytes[pos] as char);
            } else if cur == b'"' {
                in_quote = false;
            }
            pos += 1;
            continue;
        }
        if cur == b'"' {
            in_quote = true;
            out.push('"');
            pos += 1;
        } else if cur == b'#' || (cur == b'/' && bytes.get(pos + 1) == Some(&b'/')) {
            break;
        } else if cur == b'/' && bytes.get(pos + 1) == Some(&b'*') {
            *in_block_comment = true;
            pos += 2;
        } else {
            out.push(cur as char);
            pos += 1;
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TfBlockKind {
    Other,
    Ingress,
    Egress,
    Route,
    SgRule,
}

struct TfBlock {
    kind: TfBlockKind,
    from_port: Option<u16>,
    to_port: Option<u16>,
    rule_is_egress: bool,
    open_cidr_line: Option<usize>,
}

impl TfBlock {
    fn new(kind: TfBlockKind) -> Self {
        Self { kind, from_port: None, to_port: None, rule_is_egress: false, open_cidr_line: None }
    }
}

fn classify_tf_block(header: &str) -> TfBlockKind {
    let lower = header.trim().to_ascii_lowercase();
    if lower == "ingress" || lower.starts_with("ingress ") || lower.starts_with("ingress\"") {
        return TfBlockKind::Ingress;
    }
    if lower == "egress" || lower.starts_with("egress ") || lower.starts_with("egress\"") {
        return TfBlockKind::Egress;
    }
    if lower.contains("\"aws_vpc_security_group_egress_rule\"") {
        return TfBlockKind::Egress;
    }
    if lower.contains("\"aws_vpc_security_group_ingress_rule\"") {
        return TfBlockKind::Ingress;
    }
    if lower.contains("\"aws_security_group_rule\"") {
        return TfBlockKind::SgRule;
    }
    if lower.contains("\"aws_route\"")
        || lower.contains("\"google_compute_route\"")
        || lower.contains("\"azurerm_route\"")
    {
        return TfBlockKind::Route;
    }
    TfBlockKind::Other
}

fn count_hcl_braces(code: &str) -> (usize, usize) {
    let bytes = code.as_bytes();
    let (mut opens, mut closes, mut in_quote) = (0usize, 0usize, false);
    let mut pos = 0usize;
    while pos < bytes.len() {
        let cur = bytes[pos];
        if in_quote {
            if cur == b'\\' && pos + 1 < bytes.len() {
                pos += 2;
                continue;
            }
            if cur == b'"' {
                in_quote = false;
            }
        } else if cur == b'"' {
            in_quote = true;
        } else if cur == b'{' {
            opens += 1;
        } else if cur == b'}' {
            closes += 1;
        }
        pos += 1;
    }
    (opens, closes)
}

fn parse_hcl_u16_attr(lower: &str, attr: &str) -> Option<u16> {
    let rest = lower.strip_prefix(attr)?.trim_start();
    let val = rest.strip_prefix('=')?.trim();
    val.split(|ch: char| !ch.is_ascii_digit())
        .next()
        .filter(|tok| !tok.is_empty())
        .and_then(|tok| tok.parse::<u16>().ok())
}

fn update_tf_block(blk: &mut TfBlock, code: &str, lower: &str, ln: usize, cidr_suppressed: bool) {
    if let Some(port) = parse_hcl_u16_attr(lower, "from_port") {
        blk.from_port = Some(port);
    }
    if let Some(port) = parse_hcl_u16_attr(lower, "to_port") {
        blk.to_port = Some(port);
    }
    if blk.kind == TfBlockKind::SgRule
        && lower.starts_with("type")
        && lower.contains("\"egress\"")
    {
        blk.rule_is_egress = true;
    }
    if !cidr_suppressed && (code.contains("0.0.0.0/0") || code.contains("::/0")) {
        blk.open_cidr_line.get_or_insert(ln);
    }
}

fn flush_tf_block(rel: &str, blk: TfBlock, out: &mut Vec<InfraFinding>) {
    let is_ingress = blk.kind == TfBlockKind::Ingress
        || (blk.kind == TfBlockKind::SgRule && !blk.rule_is_egress);
    let Some(cidr_ln) = blk.open_cidr_line else {
        return;
    };
    if !is_ingress {
        return;
    }
    let web_port_only = matches!(
        (blk.from_port, blk.to_port),
        (Some(80), Some(80)) | (Some(443), Some(443))
    );
    if !web_port_only {
        push_finding(
            out,
            rel,
            cidr_ln,
            "high",
            "tf_open_cidr",
            "Ingress security rule open to 0.0.0.0/0 or ::/0 on non-80/443 port",
        );
    }
}

fn is_tf_wildcard_iam(code: &str, lower: &str) -> bool {
    code.contains("*:*")
        || ((lower.contains("action") || lower.contains("resources"))
            && (code.contains("\"*\"") || code.contains("\\\"*\\\"")))
}

fn scan_tf(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    let mut in_block_comment = false;
    let mut has_req_providers = false;
    let mut prov_block: Option<(usize, bool)> = None;
    let mut unversioned_provs = Vec::new();
    let mut blocks: Vec<TfBlock> = Vec::new();
    let mut prev_ignore: Option<Vec<String>> = None;

    for (idx, raw) in txt.lines().enumerate() {
        let ln = idx + 1;
        let raw_trim = raw.trim();
        if raw_trim.is_empty() {
            prev_ignore = None;
            continue;
        }
        let code_owned = strip_hcl_line(raw, &mut in_block_comment);
        let code = code_owned.trim();
        if code.is_empty() {
            prev_ignore = parse_ignore_directive(raw_trim);
            continue;
        }

        let inline_ign = parse_ignore_directive(raw);
        let lower = code.to_ascii_lowercase();
        if lower.contains("required_providers") {
            has_req_providers = true;
        }

        if is_tf_wildcard_iam(code, &lower)
            && !is_suppressed("tf_wildcard_iam", inline_ign.as_deref(), prev_ignore.as_deref())
        {
            push_finding(
                out,
                rel,
                ln,
                "critical",
                "tf_wildcard_iam",
                "Overly permissive wildcard IAM policy (`*` / `*:*`)",
            );
        }

        let (opens, closes) = count_hcl_braces(code);
        track_tf_provider(code, &lower, ln, opens, closes, &mut prov_block, &mut unversioned_provs);

        if opens > 0 {
            let header = code.split('{').next().unwrap_or("");
            blocks.push(TfBlock::new(classify_tf_block(header)));
            for _ in 1..opens {
                blocks.push(TfBlock::new(TfBlockKind::Other));
            }
        }
        if let Some(top) = blocks.last_mut() {
            let cidr_sup =
                is_suppressed("tf_open_cidr", inline_ign.as_deref(), prev_ignore.as_deref());
            update_tf_block(top, code, &lower, ln, cidr_sup);
        }
        for _ in 0..closes {
            if let Some(popped) = blocks.pop() {
                flush_tf_block(rel, popped, out);
            }
        }
        prev_ignore = None;
    }

    if !has_req_providers {
        for prov_ln in unversioned_provs {
            push_finding(
                out,
                rel,
                prov_ln,
                "medium",
                "tf_unversioned_provider",
                "Terraform provider lacks pinned version constraint",
            );
        }
    }
}

fn track_tf_provider(
    code: &str,
    lower: &str,
    ln: usize,
    opens: usize,
    closes: usize,
    prov_block: &mut Option<(usize, bool)>,
    unversioned: &mut Vec<usize>,
) {
    if lower.starts_with("provider ") && opens > 0 {
        let has_ver = lower.contains("version");
        if closes >= opens {
            if !has_ver {
                unversioned.push(ln);
            }
        } else {
            *prov_block = Some((ln, has_ver));
        }
        return;
    }
    if let Some((prov_ln, has_ver)) = prov_block.as_mut() {
        *has_ver |= lower.contains("version");
        if code.starts_with('}') || closes > opens {
            let missing_ver = !*has_ver;
            let start_ln = *prov_ln;
            *prov_block = None;
            if missing_ver {
                unversioned.push(start_ln);
            }
        }
    }
}

fn strip_sql_comment(raw: &str) -> &str {
    let bytes = raw.as_bytes();
    let mut in_single = false;
    for idx in 0..bytes.len() {
        let cur = bytes[idx];
        if cur == b'\'' {
            in_single = !in_single;
        } else if !in_single && cur == b'-' && bytes.get(idx + 1) == Some(&b'-') {
            return &raw[..idx];
        }
    }
    raw
}

fn normalize_sql_upper(code: &str) -> String {
    code.split_whitespace()
        .map(str::to_ascii_uppercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_down_migration(rel: &str) -> bool {
    let lower = rel.to_ascii_lowercase();
    lower.ends_with(".down.sql") || lower.ends_with("_down.sql") || lower.contains("/down/")
}

fn scan_sql(rel: &str, txt: &str, out: &mut Vec<InfraFinding>) {
    let down_file = is_down_migration(rel);
    let mut prev_ignore: Option<Vec<String>> = None;

    for (idx, raw) in txt.lines().enumerate() {
        let ln = idx + 1;
        let raw_trim = raw.trim();
        if raw_trim.is_empty() {
            prev_ignore = None;
            continue;
        }
        let code = strip_sql_comment(raw).trim();
        if code.is_empty() {
            prev_ignore = parse_ignore_directive(raw_trim);
            continue;
        }

        let inline_ign = parse_ignore_directive(raw);
        let stmt_up = normalize_sql_upper(code);

        let is_temp_drop = stmt_up.starts_with("DROP TEMP TABLE ")
            || stmt_up.starts_with("DROP TEMPORARY TABLE ");
        let is_destructive = !is_temp_drop
            && ((stmt_up.starts_with("DROP TABLE ") && !stmt_up.contains("IF EXISTS"))
                || stmt_up.starts_with("TRUNCATE TABLE ")
                || stmt_up.starts_with("TRUNCATE "));

        if !down_file
            && is_destructive
            && !is_suppressed("sql_destructive_migration", inline_ign.as_deref(), prev_ignore.as_deref())
        {
            push_finding(
                out,
                rel,
                ln,
                "high",
                "sql_destructive_migration",
                "Unguarded destructive DROP/TRUNCATE statement in SQL migration",
            );
        }

        if (stmt_up.contains("IDENTIFIED BY '") || stmt_up.contains("WITH PASSWORD '"))
            && !is_suppressed("sql_hardcoded_password", inline_ign.as_deref(), prev_ignore.as_deref())
        {
            push_finding(
                out,
                rel,
                ln,
                "critical",
                "sql_hardcoded_password",
                "Hardcoded database password literal in SQL script",
            );
        }
        prev_ignore = None;
    }
}

fn collect_manifest_deps(
    manifests: &[(String, String, String)],
) -> (Vec<ParsedDep>, [Option<EcoMatcher>; 4]) {
    let mut builders: [EcoBuilder; 4] = Default::default();
    let mut parsed = Vec::new();
    for (manifest, fname, txt) in manifests {
        for dep in parse_deps(fname, txt) {
            let eco = Eco::from_name(dep.ecosystem);
            let slot = builders[eco.idx()].add_dep(eco, &dep.package);
            parsed.push(ParsedDep {
                manifest: manifest.clone(),
                ecosystem: dep.ecosystem,
                package: dep.package,
                eco,
                slot,
            });
        }
    }
    let [b0, b1, b2, b3] = builders;
    let matchers = [
        b0.build(Eco::ALL[0]),
        b1.build(Eco::ALL[1]),
        b2.build(Eco::ALL[2]),
        b3.build(Eco::ALL[3]),
    ];
    (parsed, matchers)
}

fn all_matchers_complete(matchers: &[Option<EcoMatcher>; 4]) -> bool {
    matchers.iter().all(|m| m.as_ref().is_none_or(EcoMatcher::is_complete))
}

fn scan_sources_for_deps(
    matchers: &mut [Option<EcoMatcher>; 4],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    source_paths: &[(Eco, PathBuf)],
) {
    if let Some(map) = preloaded {
        for (rel, bytes) in map {
            let eco = Eco::from_ext(&ext_of(rel));
            if let Some(m) = matchers[eco.idx()].as_mut() {
                m.scan_bytes(bytes);
            }
            if all_matchers_complete(matchers) {
                return;
            }
        }
        return;
    }

    for (eco, abs) in source_paths {
        let Some(m) = matchers[eco.idx()].as_mut() else { continue };
        if m.is_complete() {
            continue;
        }
        if let Ok(bytes) = fs::read(abs) {
            m.scan_bytes(&bytes);
        }
        if all_matchers_complete(matchers) {
            return;
        }
    }
}

fn find_unused_deps(
    manifests: &[(String, String, String)],
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
    source_paths: &[(Eco, PathBuf)],
) -> Vec<UnusedDep> {
    if manifests.is_empty() {
        return Vec::new();
    }
    let (parsed, mut matchers) = collect_manifest_deps(manifests);
    scan_sources_for_deps(&mut matchers, preloaded, source_paths);

    let mut unused = Vec::new();
    for dep in parsed {
        let used = matchers[dep.eco.idx()]
            .as_ref()
            .is_some_and(|m| m.hits.contains(dep.slot));
        if !used {
            unused.push(UnusedDep {
                manifest: dep.manifest,
                ecosystem: dep.ecosystem.into(),
                package: dep.package,
            });
        }
    }
    unused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn infra_gha_2q_taint_multistage_docker_tf_sql_precision() {
        let gha_safe = concat!(
            "jobs:\n",
            "  safe:\n",
            "    steps:\n",
            "      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v4\n",
            "      - env:\n",
            "          TITLE: ${{ github.event.issue.title }}\n",
            "          TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n",
            "        if: ${{ github.event.pull_request.number > 0 }}\n",
            "        run: echo \"$TITLE\" ${{ github.event.pull_request.number }}\n",
            "      # never do echo ${{ github.event.issue.title }}\n",
            "      - run: |\n",
            "          echo \"$SAFE_TITLE\"\n",
            "        env:\n",
            "          SAFE_TITLE: ${{ github.event.issue.title }}\n",
            "          SAFE_TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n",
            "      - run: echo ${{ github.event.comment.body }} # refit: ignore[gha_script_injection]\n",
        );
        let gha_unsafe = concat!(
            "jobs:\n",
            "  bad:\n",
            "    steps:\n",
            "      - uses: actions/checkout@v4\n",
            "      - run: |\n",
            "          echo \"${{ github.event.issue.title }}\"\n",
            "          curl -H \"Auth: ${{ secrets.API_TOKEN }}\" https://example.com\n",
        );
        let docker_multistage_root = concat!(
            "FROM node:20 AS builder\n",
            "USER appuser\n",
            "RUN npm run build\n",
            "FROM debian:12-slim\n",
            "COPY --from=builder /app /app\n",
        );
        let docker_scratch = "FROM node:20 AS build\nFROM scratch\nCOPY --from=build /bin/app /app\n";
        let docker_nonroot = "FROM gcr.io/distroless/static-debian12:nonroot\nENTRYPOINT [\"/app\"]\n";
        let tf_cfg = concat!(
            "# cidr_blocks = [\"0.0.0.0/0\"]\n",
            "// policy = \"{\\\"Action\\\": \\\"*:*\\\"}\"\n",
            "terraform { required_providers { aws = { version = \"~> 5.0\" } } }\n",
            "resource \"aws_security_group\" \"web\" {\n",
            "  ingress {\n",
            "    from_port = 443\n",
            "    to_port = 443\n",
            "    cidr_blocks = [\"0.0.0.0/0\"]\n",
            "  }\n",
            "  egress {\n",
            "    from_port = 0\n",
            "    to_port = 0\n",
            "    cidr_blocks = [\"0.0.0.0/0\"]\n",
            "  }\n",
            "  ingress {\n",
            "    from_port = 22\n",
            "    to_port = 22\n",
            "    cidr_blocks = [\"0.0.0.0/0\"]\n",
            "  }\n",
            "}\n",
        );
        let sql_up = "DROP TEMPORARY TABLE scratch_buf;\nDROP TABLE legacy_users;\n";
        let sql_down = "DROP TABLE legacy_users;\nTRUNCATE TABLE audit_log;\n";

        let dir = TempRepo::with_files(
            "infra_prec",
            &[
                (".github/workflows/safe.yml", gha_safe),
                (".github/workflows/unsafe.yml", gha_unsafe),
                ("Dockerfile", docker_multistage_root),
                ("scratch.dockerfile", docker_scratch),
                ("nonroot.dockerfile", docker_nonroot),
                ("infra/sg.tf", tf_cfg),
                ("migrations/001_init.up.sql", sql_up),
                ("migrations/001_init.down.sql", sql_down),
            ],
        );

        let (_, _, findings) = scan_parts(dir.path(), None);
        let by_file = |file: &str, rule: &str| {
            findings.iter().filter(|f| f.file == file && f.rule_id == rule).count()
        };

        assert_eq!(by_file(".github/workflows/safe.yml", "gha_script_injection"), 0);
        assert_eq!(by_file(".github/workflows/safe.yml", "gha_secrets_outside_env"), 0);
        assert_eq!(by_file(".github/workflows/safe.yml", "gha_unpinned_action"), 0);

        assert_eq!(by_file(".github/workflows/unsafe.yml", "gha_script_injection"), 1);
        assert_eq!(by_file(".github/workflows/unsafe.yml", "gha_secrets_outside_env"), 1);
        assert_eq!(by_file(".github/workflows/unsafe.yml", "gha_unpinned_action"), 1);

        assert_eq!(by_file("Dockerfile", "docker_runs_as_root"), 1);
        assert_eq!(by_file("scratch.dockerfile", "docker_runs_as_root"), 0);
        assert_eq!(by_file("nonroot.dockerfile", "docker_runs_as_root"), 0);

        assert_eq!(by_file("infra/sg.tf", "tf_wildcard_iam"), 0);
        assert_eq!(by_file("infra/sg.tf", "tf_open_cidr"), 1);

        assert_eq!(by_file("migrations/001_init.up.sql", "sql_destructive_migration"), 1);
        assert_eq!(by_file("migrations/001_init.down.sql", "sql_destructive_migration"), 0);
    }
}
