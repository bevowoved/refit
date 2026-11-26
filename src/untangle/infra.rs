use crate::lex::{ident_byte, left_bound, quoted_lit, scan_code};
use crate::platform::find_secret;
use crate::source::{RepoSnapshot, ext_of, parse_deps, supported_ext};
use crate::types::{InfraFinding, UnusedDep};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use fixedbitset::FixedBitSet;
use regex::Regex;
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::LazyLock;

static IGNORE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i-u:refit):[ \t]*(?i-u:ignore)(?:[ \t]*\[([^\]]*)\])?").expect("ignore re")
});
static GHA_EXPR_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\$\{\{(.*?)\}\}").expect("gha expr re"));
static GHA_UNTRUSTED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"github\.(?:head_ref|event\.(?:",
        r"issue\.(?:title|body)|pull_request\.(?:title|body|head\.(?:ref|label))|",
        r"(?:comment|review|review_comment)\.body|discussion\.(?:title|body)|",
        r"head_commit\.(?:message|author)|commits.*\.(?:message|author)|pages.*\.page_name))",
    )).expect("gha untrusted re")
});
static GHA_USES_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:-[ \t]+)?uses:[ \t]*['\x22]?([^'\x22]+?)['\x22]?$").expect("gha uses re")
});
static GHA_SHA_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@[0-9a-fA-F]{40}$").expect("gha sha re"));
static GHA_SINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:-[ \t]+)?(?:run|script):[ \t]*(.*)$").expect("gha sink re")
});
static DOCKER_BAKED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?i-u:ENV|ARG)[ \t]+[^ \t=]*(?i-u:SECRET|PASSWORD|API_KEY)",
        r"[^ \t=]*(?:[ \t]*=[ \t]*|[ \t]+)['\x22]?([^ \t='\x22$])",
    )).expect("docker baked re")
});
static HCL_PORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(from|to)_port[ \t]*=[ \t]*([0-9]+)").expect("hcl port re")
});
static SQL_DESTR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?i-u:DROP[ \t]+TABLE[ \t]+|TRUNCATE(?:[ \t]+TABLE)?[ \t]+)").expect("sql destr re")
});
static SQL_IF_EXISTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i-u:IF[ \t]+EXISTS)").expect("sql if-exists re")
});
static SQL_PASS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i-u:IDENTIFIED[ \t]+BY|WITH[ \t]+PASSWORD)[ \t]+'").expect("sql pass re")
});

#[derive(Clone, Copy, PartialEq, Eq)]
enum Eco { Npm = 0, Cargo = 1, Python = 2, Go = 3 }

impl Eco {
    const ALL: [Self; 4] = [Self::Npm, Self::Cargo, Self::Python, Self::Go];
    fn from_ext(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Self::Cargo, "py" => Self::Python, "go" => Self::Go, _ => Self::Npm,
        }
    }
    fn from_name(name: &str) -> Self {
        match name { "npm" => Self::Npm, "cargo" => Self::Cargo, "python" => Self::Python, _ => Self::Go }
    }
    #[inline] fn idx(self) -> usize { self as usize }
    #[inline] fn word_bound(self) -> bool { self != Self::Go }
    #[inline] fn case_insensitive(self) -> bool { self == Self::Python }
}

struct ParsedDep { manifest: String, ecosystem: &'static str, package: String, eco: Eco, slot: usize }

#[derive(Default)]
struct EcoBuilder {
    pkg_slots: HashMap<String, usize>, pat_ids: HashMap<String, usize>,
    patterns: Vec<String>, pat_slots: Vec<Vec<usize>>,
}

impl EcoBuilder {
    fn add_dep(&mut self, eco: Eco, package: &str) -> usize {
        let key = if eco.case_insensitive() { package.to_ascii_lowercase() } else { package.to_string() };
        let next_slot = self.pkg_slots.len();
        let slot = *self.pkg_slots.entry(key.clone()).or_insert(next_slot);
        if key.is_empty() { return slot; }
        self.add_pattern(&key, slot);
        if matches!(eco, Eco::Cargo | Eco::Python) && key.contains('-') {
            self.add_pattern(&key.replace('-', "_"), slot);
        }
        slot
    }

    fn add_pattern(&mut self, pat: &str, slot: usize) {
        let pat_id = *self.pat_ids.entry(pat.to_string()).or_insert_with(|| {
            self.patterns.push(pat.to_string());
            self.pat_slots.push(Vec::new());
            self.patterns.len() - 1
        });
        if !self.pat_slots[pat_id].contains(&slot) { self.pat_slots[pat_id].push(slot); }
    }

    fn build(self, eco: Eco) -> Option<EcoMatcher> {
        let dep_count = self.pkg_slots.len();
        if dep_count == 0 || self.patterns.is_empty() { return None; }
        let ac = AhoCorasickBuilder::new().ascii_case_insensitive(eco.case_insensitive())
            .build(&self.patterns).ok()?;
        Some(EcoMatcher {
            ac, pat_slots: self.pat_slots, dep_count,
            word_bound: eco.word_bound(), hits: FixedBitSet::with_capacity(dep_count),
        })
    }
}

struct EcoMatcher {
    ac: AhoCorasick, pat_slots: Vec<Vec<usize>>, dep_count: usize, word_bound: bool, hits: FixedBitSet,
}

impl EcoMatcher {
    #[inline] fn is_complete(&self) -> bool { self.hits.count_ones(..) == self.dep_count }

    fn scan_bytes(&mut self, bytes: &[u8]) {
        if self.is_complete() || bytes.is_empty() { return; }
        for mat in self.ac.find_overlapping_iter(bytes) {
            if self.word_bound && !is_word_span(bytes, mat.start(), mat.end()) { continue; }
            let mut added = false;
            for &slot in &self.pat_slots[mat.pattern().as_usize()] { added |= !self.hits.put(slot); }
            if added && self.is_complete() { return; }
        }
    }
}

#[inline]
fn is_word_span(bytes: &[u8], start: usize, end: usize) -> bool {
    left_bound(bytes, start) && (end == bytes.len() || !ident_byte(bytes[end]))
}

#[derive(Clone, Copy)]
enum Kind { Tf, Docker, Gha, Sql }

struct FindingSink<'a> {
    rel: &'a str, out: &'a mut Vec<InfraFinding>,
    prev_ignore: Option<Vec<String>>, inline_ignore: Option<Vec<String>>,
}

impl<'a> FindingSink<'a> {
    fn new(rel: &'a str, out: &'a mut Vec<InfraFinding>) -> Self {
        Self { rel, out, prev_ignore: None, inline_ignore: None }
    }

    fn skip_blank_or_comment(&mut self, raw_trim: &str, code_empty: bool) -> bool {
        if raw_trim.is_empty() {
            self.prev_ignore = None;
            return true;
        }
        if code_empty {
            self.prev_ignore = parse_ignore_directive(raw_trim);
            return true;
        }
        false
    }

    fn begin_line(&mut self, raw: &str) { self.inline_ignore = parse_ignore_directive(raw); }
    fn end_line(&mut self) {
        self.prev_ignore = None;
        self.inline_ignore = None;
    }

    fn is_suppressed(&self, rule_id: &str) -> bool {
        let matches = |list: &Vec<String>| {
            list.is_empty() || list.iter().any(|item| item.eq_ignore_ascii_case(rule_id))
        };
        self.inline_ignore.as_ref().is_some_and(matches)
            || self.prev_ignore.as_ref().is_some_and(matches)
    }

    fn emit(&mut self, line: usize, severity: &str, rule_id: &str, message: &str) {
        if !self.is_suppressed(rule_id) { self.emit_raw(line, severity, rule_id, message); }
    }

    fn emit_raw(&mut self, line: usize, severity: &str, rule_id: &str, message: &str) {
        self.out.push(InfraFinding {
            file: self.rel.to_string(), line,
            severity: severity.into(), rule_id: rule_id.into(), message: message.into(),
        });
    }
}

fn parse_ignore_directive(raw_line: &str) -> Option<Vec<String>> {
    let cap = IGNORE_RE.captures(raw_line)?;
    let rules = cap.get(1).map_or_else(Vec::new, |mat| {
        mat.as_str()
            .split(',')
            .map(|tok| tok.trim().to_ascii_lowercase())
            .filter(|tok| !tok.is_empty())
            .collect()
    });
    Some(rules)
}

fn scan_unused_deps(
    repo_root: &Path, snap: &RepoSnapshot, manifests: &[(String, String, String)],
) -> Vec<UnusedDep> {
    if manifests.is_empty() { return Vec::new(); }
    let (parsed, mut matchers) = collect_manifest_deps(manifests);
    for rel in &snap.all_rels {
        let ext = ext_of(rel);
        if !supported_ext(&ext) { continue; }
        let Some(matcher) = matchers[Eco::from_ext(&ext).idx()].as_mut() else { continue };
        if matcher.is_complete() { continue; }
        if let Some(bytes) = snap.preloaded.get(rel) {
            matcher.scan_bytes(bytes);
        } else if let Ok(bytes) = fs::read(repo_root.join(rel)) {
            matcher.scan_bytes(&bytes);
        }
        if matchers.iter().all(|e| e.as_ref().is_none_or(EcoMatcher::is_complete)) { break; }
    }
    collect_unused_from_matchers(parsed, &matchers)
}

/// Snapshot-backed infra & unused-dependency scan sharing the single `RepoSnapshot` tree walk.
pub fn scan_parts_snap(
    repo_root: &Path, snap: &RepoSnapshot,
) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    let (mut infra_count, mut findings, mut manifests) = (0usize, Vec::new(), Vec::new());

    for (rel, fname, abs) in &snap.manifest_rels {
        if let Ok(txt) = fs::read_to_string(abs) { manifests.push((rel.clone(), fname.clone(), txt)); }
    }
    for (rel, abs) in &snap.infra_rels {
        let fname = abs.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(kind) = infra_kind(rel, fname, &ext_of(abs)) else { continue };
        infra_count += 1;
        if let Ok(txt) = fs::read_to_string(abs) { scan_file(rel, kind, &txt, &mut findings); }
    }
    let unused = scan_unused_deps(repo_root, snap, &manifests);
    (infra_count, unused, findings)
}

/// Scan Terraform, Dockerfiles, GitHub Actions, SQL, and unused manifest dependencies.
#[cfg(test)]
pub fn scan_parts(
    repo_root: &Path, _preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> (usize, Vec<UnusedDep>, Vec<InfraFinding>) {
    scan_parts_snap(repo_root, &RepoSnapshot::capture(repo_root, true))
}

fn infra_kind(rel: &str, fname: &str, ext: &str) -> Option<Option<Kind>> {
    if fname.starts_with("Dockerfile") || fname.ends_with(".dockerfile") { Some(Some(Kind::Docker)) }
    else if matches!(ext, "tf" | "tfvars") { Some(Some(Kind::Tf)) }
    else if rel.starts_with(".github/workflows/") && matches!(ext, "yml" | "yaml") { Some(Some(Kind::Gha)) }
    else if ext == "sql" { Some(Some(Kind::Sql)) }
    else if fname.starts_with("docker-compose") && matches!(ext, "yml" | "yaml") { Some(None) }
    else { None }
}

fn scan_file(rel: &str, kind: Option<Kind>, txt: &str, out: &mut Vec<InfraFinding>) {
    let mut sink = FindingSink::new(rel, out);
    for (idx, line) in txt.lines().enumerate() {
        if let Some(sk) = find_secret(line) {
            sink.emit_raw(idx + 1, "critical", &format!("infra_secret_{sk}"),
                "Hardcoded credential/secret committed in infrastructure config");
        }
    }
    match kind {
        Some(Kind::Gha) => scan_gha(txt, &mut sink),
        Some(Kind::Docker) => scan_docker(txt, &mut sink),
        Some(Kind::Tf) => scan_tf(txt, &mut sink),
        Some(Kind::Sql) => scan_sql(txt, &mut sink),
        None => {}
    }
}

fn strip_yaml_inline_comment(line: &str) -> &str {
    let (bytes, mut cut) = (line.as_bytes(), line.len());
    scan_code(bytes, b"'\"", |idx, cur| {
        if cur == b'#' && (idx == 0 || bytes[idx - 1].is_ascii_whitespace()) {
            cut = idx;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    &line[..cut]
}

fn is_pinned_gha_action(uses_ref: &str) -> bool {
    uses_ref.starts_with("./") || uses_ref.starts_with("../")
        || (uses_ref.starts_with("docker://") && uses_ref.contains("@sha256:"))
        || GHA_SHA_RE.is_match(uses_ref)
}

fn check_gha_sink_line(ln: usize, line: &str, sink: &mut FindingSink<'_>) {
    let (mut flagged_inj, mut flagged_sec) = (false, false);
    for cap in GHA_EXPR_RE.captures_iter(line) {
        let expr_lo = cap[1].to_ascii_lowercase();
        if !flagged_inj && GHA_UNTRUSTED_RE.is_match(&expr_lo) {
            flagged_inj = true;
            sink.emit(ln, "high", "gha_script_injection",
                "Untrusted `${{ github.event.* }}` interpolation in run/script risks injection");
        }
        if !flagged_sec && expr_lo.contains("secrets.") {
            flagged_sec = true;
            sink.emit(ln, "high", "gha_secrets_outside_env",
                "Secret `${{ secrets.* }}` interpolated directly in shell sink instead of `env:`");
        }
    }
}

fn yaml_block_indent(trimmed: &str, indent: usize) -> usize {
    let Some(after) = trimmed.strip_prefix('-') else { return indent };
    let ws = after.bytes().take_while(|&b| b == b' ').count();
    if ws > 0 { indent + 1 + ws } else { indent }
}

fn scan_gha_line(ln: usize, indent: usize, trimmed: &str, sink: &mut FindingSink<'_>) -> Option<usize> {
    let code = strip_yaml_inline_comment(trimmed).trim();
    if code.contains("write-all") {
        sink.emit(ln, "high", "gha_write_all_perms",
            "Workflow grants overly broad `write-all` permissions");
    }
    if let Some(m) = GHA_USES_RE.captures(code).and_then(|c| c.get(1))
        && !is_pinned_gha_action(m.as_str().trim())
    {
        sink.emit(ln, "medium", "gha_unpinned_action",
            "Action not pinned to immutable 40-hex commit SHA");
    }
    let rhs = GHA_SINK_RE.captures(trimmed).and_then(|c| c.get(1))?.as_str().trim();
    if matches!(strip_yaml_inline_comment(rhs).trim(), "|" | "|-" | "|+" | ">" | ">-" | ">+") {
        return Some(yaml_block_indent(trimmed, indent));
    }
    check_gha_sink_line(ln, rhs, sink);
    None
}

fn scan_gha(txt: &str, sink: &mut FindingSink<'_>) {
    let mut run_indent: Option<usize> = None;
    for (idx, raw) in txt.lines().enumerate() {
        let (ln, indent, trimmed) = (idx + 1, raw.bytes().take_while(|&b| b == b' ').count(), raw.trim());
        if sink.skip_blank_or_comment(trimmed, trimmed.starts_with('#')) { continue; }
        if run_indent.is_some_and(|ri| indent <= ri) { run_indent = None; }

        sink.begin_line(raw);
        if run_indent.is_some() {
            check_gha_sink_line(ln, trimmed, sink);
        } else {
            run_indent = scan_gha_line(ln, indent, trimmed, sink);
        }
        sink.end_line();
    }
}

fn extract_docker_from_image(trimmed: &str) -> &str {
    trimmed.split_whitespace().skip(1)
        .find(|tok| !tok.starts_with("--"))
        .filter(|tok| !tok.starts_with('#'))
        .unwrap_or("")
}

fn is_whitelisted_nonroot_image(image: &str) -> bool {
    let lower = image.trim().to_ascii_lowercase();
    let base = lower.split('@').next().unwrap_or(&lower);
    base == "scratch" || base.ends_with(":nonroot") || base.contains(":nonroot-")
        || (base.contains("distroless") && base.contains("nonroot"))
}

struct DockerStage { has_stage: bool, non_root: bool, from_ln: usize, root_ignored: bool }

fn scan_docker_from(ln: usize, trimmed: &str, sink: &mut FindingSink<'_>, st: &mut DockerStage) {
    (st.has_stage, st.from_ln) = (true, ln);
    let image = extract_docker_from_image(trimmed);
    st.non_root = is_whitelisted_nonroot_image(image);
    st.root_ignored = sink.is_suppressed("docker_runs_as_root");
    if image.to_ascii_lowercase().ends_with(":latest") {
        sink.emit(ln, "medium", "docker_latest_tag",
            "Unpinned `:latest` base image causes non-reproducible builds");
    }
}

fn scan_docker_stmt(ln: usize, trimmed: &str, up: &str, sink: &mut FindingSink<'_>, st: &mut DockerStage) {
    if up.starts_with("FROM ") {
        scan_docker_from(ln, trimmed, sink, st);
    } else if up.starts_with("USER ") {
        let spec = trimmed.split_whitespace().nth(1).unwrap_or("");
        let user = spec.split(':').next().unwrap_or("");
        st.non_root = !user.is_empty() && !user.eq_ignore_ascii_case("root") && user != "0";
    } else if up.starts_with("ADD ") && (trimmed.contains("http://") || trimmed.contains("https://")) {
        sink.emit(ln, "high", "docker_add_remote_url",
            "Dockerfile `ADD` fetches remote URL without integrity checksum");
    }
}

fn scan_docker(txt: &str, sink: &mut FindingSink<'_>) {
    let mut st = DockerStage { has_stage: false, non_root: false, from_ln: 1, root_ignored: false };
    for (idx, raw) in txt.lines().enumerate() {
        let (ln, trimmed) = (idx + 1, raw.trim());
        if sink.skip_blank_or_comment(trimmed, trimmed.starts_with('#')) { continue; }
        sink.begin_line(raw);
        let up = trimmed.to_ascii_uppercase();
        scan_docker_stmt(ln, trimmed, &up, sink, &mut st);
        if DOCKER_BAKED_RE.is_match(trimmed) {
            sink.emit(ln, "high", "docker_baked_secret",
                "Sensitive ENV/ARG value baked into Docker image layer");
        }
        sink.end_line();
    }
    if st.has_stage && !st.non_root && !st.root_ignored {
        sink.emit_raw(st.from_ln, "high", "docker_runs_as_root",
            "Final Dockerfile stage missing non-root `USER` instruction");
    }
}

fn skip_hcl_block_comment(bytes: &[u8], pos: usize, in_block_comment: &mut bool) -> usize {
    if bytes.get(pos..pos + 2) == Some(b"*/") {
        *in_block_comment = false;
        return pos + 2;
    }
    pos + 1
}

fn strip_hcl_line(raw: &str, in_block_comment: &mut bool) -> String {
    let (bytes, mut out, mut pos) = (raw.as_bytes(), String::with_capacity(raw.len()), 0usize);
    while pos < bytes.len() {
        if *in_block_comment {
            pos = skip_hcl_block_comment(bytes, pos, in_block_comment);
            continue;
        }
        pos = match step_hcl_token(raw, bytes, pos, &mut out, in_block_comment) {
            Some(next) => next,
            None => break,
        };
    }
    out
}

fn step_hcl_token(
    raw: &str, bytes: &[u8], pos: usize, out: &mut String, in_block_comment: &mut bool,
) -> Option<usize> {
    let cur = bytes[pos];
    if cur == b'"' {
        let end = quoted_lit(bytes, pos, true, true).0;
        out.push_str(&raw[pos..end]);
        return Some(end);
    }
    if cur == b'#' || bytes.get(pos..pos + 2) == Some(b"//") { return None; }
    if bytes.get(pos..pos + 2) == Some(b"/*") {
        *in_block_comment = true;
        return Some(pos + 2);
    }
    let ch = raw[pos..].chars().next()?;
    out.push(ch);
    Some(pos + ch.len_utf8())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TfBlockKind { Other, Ingress, SgRule }

struct TfBlock {
    kind: TfBlockKind, from_port: Option<u16>, to_port: Option<u16>,
    rule_is_egress: bool, open_cidr_line: Option<usize>,
}

impl TfBlock {
    fn new(kind: TfBlockKind) -> Self {
        Self { kind, from_port: None, to_port: None, rule_is_egress: false, open_cidr_line: None }
    }
}

fn classify_tf_block(header: &str) -> TfBlockKind {
    let lo = header.trim().to_ascii_lowercase();
    if lo == "ingress" || lo.starts_with("ingress ") || lo.starts_with("ingress\"")
        || lo.contains("\"aws_vpc_security_group_ingress_rule\"")
    { TfBlockKind::Ingress }
    else if lo.contains("\"aws_security_group_rule\"") { TfBlockKind::SgRule }
    else { TfBlockKind::Other }
}

fn count_hcl_braces(code: &str) -> (usize, usize) {
    let (mut opens, mut closes) = (0usize, 0usize);
    scan_code(code.as_bytes(), b"\"", |_, cur| {
        if cur == b'{' {
            opens += 1;
        } else if cur == b'}' {
            closes += 1;
        }
        ControlFlow::Continue(())
    });
    (opens, closes)
}

fn update_tf_block(blk: &mut TfBlock, code: &str, lower: &str, ln: usize, cidr_suppressed: bool) {
    if let Some(cap) = HCL_PORT_RE.captures(lower) && let Ok(port) = cap[2].parse::<u16>() {
        if &cap[1] == "from" {
            blk.from_port = Some(port);
        } else {
            blk.to_port = Some(port);
        }
    }
    if blk.kind == TfBlockKind::SgRule && lower.starts_with("type") && lower.contains("\"egress\"") {
        blk.rule_is_egress = true;
    }
    if !cidr_suppressed && (code.contains("0.0.0.0/0") || code.contains("::/0")) {
        blk.open_cidr_line.get_or_insert(ln);
    }
}

fn flush_tf_block(blk: TfBlock, sink: &mut FindingSink<'_>) {
    let is_ingress = blk.kind == TfBlockKind::Ingress
        || (blk.kind == TfBlockKind::SgRule && !blk.rule_is_egress);
    let Some(cidr_ln) = blk.open_cidr_line.filter(|_| is_ingress) else { return };
    if !matches!((blk.from_port, blk.to_port), (Some(80), Some(80)) | (Some(443), Some(443))) {
        sink.emit_raw(cidr_ln, "high", "tf_open_cidr",
            "Ingress security rule open to 0.0.0.0/0 or ::/0 on non-80/443 port");
    }
}

fn is_tf_wildcard_iam(code: &str, lower: &str) -> bool {
    code.contains("*:*")
        || ((lower.contains("action") || lower.contains("resources"))
            && (code.contains("\"*\"") || code.contains("\\\"*\\\"")))
}

fn apply_tf_braces(code: &str, lower: &str, ln: usize, opens: usize, closes: usize, blocks: &mut Vec<TfBlock>, sink: &mut FindingSink<'_>) {
    if opens > 0 {
        blocks.push(TfBlock::new(classify_tf_block(code.split('{').next().unwrap_or(""))));
        for _ in 1..opens { blocks.push(TfBlock::new(TfBlockKind::Other)); }
    }
    if let Some(top) = blocks.last_mut() {
        update_tf_block(top, code, lower, ln, sink.is_suppressed("tf_open_cidr"));
    }
    for _ in 0..closes {
        if let Some(popped) = blocks.pop() { flush_tf_block(popped, sink); }
    }
}

fn scan_tf(txt: &str, sink: &mut FindingSink<'_>) {
    let (mut in_block_comment, mut has_req_providers) = (false, false);
    let (mut prov_block, mut unversioned_provs, mut blocks) =
        (None::<(usize, bool)>, Vec::new(), Vec::<TfBlock>::new());

    for (idx, raw) in txt.lines().enumerate() {
        let (ln, raw_trim) = (idx + 1, raw.trim());
        let code_owned = strip_hcl_line(raw, &mut in_block_comment);
        let code = code_owned.trim();
        if sink.skip_blank_or_comment(raw_trim, code.is_empty()) { continue; }

        sink.begin_line(raw);
        let lower = code.to_ascii_lowercase();
        has_req_providers |= lower.contains("required_providers");
        if is_tf_wildcard_iam(code, &lower) {
            sink.emit(ln, "critical", "tf_wildcard_iam",
                "Overly permissive wildcard IAM policy (`*` / `*:*`)");
        }
        let (opens, closes) = count_hcl_braces(code);
        track_tf_provider(code, &lower, ln, opens, closes, &mut prov_block, &mut unversioned_provs);
        apply_tf_braces(code, &lower, ln, opens, closes, &mut blocks, sink);
        sink.end_line();
    }
    if has_req_providers { return; }
    for prov_ln in unversioned_provs {
        sink.emit_raw(prov_ln, "medium", "tf_unversioned_provider",
            "Terraform provider lacks pinned version constraint");
    }
}

fn track_tf_provider(
    code: &str, lower: &str, ln: usize, opens: usize, closes: usize,
    prov_block: &mut Option<(usize, bool)>, unversioned: &mut Vec<usize>,
) {
    let has_ver = lower.contains("version");
    if lower.starts_with("provider ") && opens > 0 {
        if closes < opens {
            *prov_block = Some((ln, has_ver));
        } else if !has_ver {
            unversioned.push(ln);
        }
        return;
    }
    if let Some((prov_ln, seen_ver)) = prov_block.as_mut() {
        *seen_ver |= has_ver;
        if code.starts_with('}') || closes > opens {
            let unpinned = (!*seen_ver).then_some(*prov_ln);
            *prov_block = None;
            if let Some(start_ln) = unpinned { unversioned.push(start_ln); }
        }
    }
}

fn strip_sql_comment(raw: &str) -> &str {
    let (bytes, mut cut) = (raw.as_bytes(), raw.len());
    scan_code(bytes, b"'", |idx, cur| {
        if cur == b'-' && bytes.get(idx + 1) == Some(&b'-') {
            cut = idx;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    &raw[..cut]
}

fn is_down_sql(rel: &str) -> bool {
    let lo = rel.to_ascii_lowercase();
    lo.ends_with(".down.sql") || lo.ends_with("_down.sql") || lo.contains("/down/")
}

fn scan_sql(txt: &str, sink: &mut FindingSink<'_>) {
    let down_file = is_down_sql(sink.rel);
    for (idx, raw) in txt.lines().enumerate() {
        let (ln, raw_trim, code) = (idx + 1, raw.trim(), strip_sql_comment(raw).trim());
        if sink.skip_blank_or_comment(raw_trim, code.is_empty()) { continue; }
        sink.begin_line(raw);
        if !down_file && SQL_DESTR_RE.is_match(code) && !SQL_IF_EXISTS_RE.is_match(code) {
            sink.emit(ln, "high", "sql_destructive_migration",
                "Unguarded destructive DROP/TRUNCATE statement in SQL migration");
        }
        if SQL_PASS_RE.is_match(code) {
            sink.emit(ln, "critical", "sql_hardcoded_password",
                "Hardcoded database password literal in SQL script");
        }
        sink.end_line();
    }
}

fn push_manifest_deps(
    manifest: &str, fname: &str, txt: &str, builders: &mut [EcoBuilder; 4], parsed: &mut Vec<ParsedDep>,
) {
    for dep in parse_deps(fname, txt) {
        let eco = Eco::from_name(dep.ecosystem);
        let slot = builders[eco.idx()].add_dep(eco, &dep.package);
        parsed.push(ParsedDep {
            manifest: manifest.to_string(), ecosystem: dep.ecosystem, package: dep.package, eco, slot,
        });
    }
}

fn collect_manifest_deps(
    manifests: &[(String, String, String)],
) -> (Vec<ParsedDep>, [Option<EcoMatcher>; 4]) {
    let (mut builders, mut parsed): ([EcoBuilder; 4], Vec<ParsedDep>) = Default::default();
    for (manifest, fname, txt) in manifests {
        push_manifest_deps(manifest, fname, txt, &mut builders, &mut parsed);
    }
    (parsed, Eco::ALL.map(|eco| std::mem::take(&mut builders[eco.idx()]).build(eco)))
}

fn collect_unused_from_matchers(
    parsed: Vec<ParsedDep>, matchers: &[Option<EcoMatcher>; 4],
) -> Vec<UnusedDep> {
    parsed.into_iter()
        .filter(|d| !matchers[d.eco.idx()].as_ref().is_some_and(|m| m.hits.contains(d.slot)))
        .map(|d| UnusedDep { manifest: d.manifest, ecosystem: d.ecosystem.into(), package: d.package })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn infra_gha_docker_tf_sql() {
        let gha_safe = concat!(
            "jobs:\n  safe:\n    steps:\n",
            "      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v4\n",
            "      - env:\n          TITLE: ${{ github.event.issue.title }}\n",
            "          TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n",
            "        if: ${{ github.event.pull_request.number > 0 }}\n",
            "        run: echo \"$TITLE\" ${{ github.event.pull_request.number }}\n",
            "      # never do echo ${{ github.event.issue.title }}\n",
            "      - run: |\n          echo \"$SAFE_TITLE\"\n        env:\n",
            "          SAFE_TITLE: ${{ github.event.issue.title }}\n",
            "          SAFE_TOKEN: ${{ secrets.DEPLOY_TOKEN }}\n",
            "      - run: echo ${{ github.event.comment.body }} # refit: ignore[gha_script_injection]\n",
        );
        let gha_unsafe = concat!(
            "jobs:\n  bad:\n    steps:\n      - uses: actions/checkout@v4\n      - run: |\n",
            "          echo \"${{ github.event.issue.title }}\"\n",
            "          curl -H \"Auth: ${{ secrets.API_TOKEN }}\" https://example.com\n",
        );
        let docker_root = concat!(
            "FROM node:20 AS builder\nUSER appuser\nRUN npm run build\n",
            "FROM debian:12-slim\nCOPY --from=builder /app /app\n",
        );
        let tf_cfg = concat!(
            "# cidr_blocks = [\"0.0.0.0/0\"]\n// policy = \"{\\\"Action\\\": \\\"*:*\\\"}\"\n",
            "terraform { required_providers { aws = { version = \"~> 5.0\" } } }\n",
            "resource \"aws_security_group\" \"web\" {\n",
            "  ingress {\n    from_port = 443\n    to_port = 443\n    cidr_blocks = [\"0.0.0.0/0\"]\n  }\n",
            "  egress {\n    from_port = 0\n    to_port = 0\n    cidr_blocks = [\"0.0.0.0/0\"]\n  }\n",
            "  ingress {\n    from_port = 22\n    to_port = 22\n    cidr_blocks = [\"0.0.0.0/0\"]\n  }\n}\n",
        );
        let dir = TempRepo::with_files("infra_prec", &[
            (".github/workflows/safe.yml", gha_safe),
            (".github/workflows/unsafe.yml", gha_unsafe),
            ("Dockerfile", docker_root),
            ("scratch.dockerfile", "FROM node:20 AS build\nFROM scratch\nCOPY --from=build /bin/app /app\n"),
            ("nonroot.dockerfile", "FROM gcr.io/distroless/static-debian12:nonroot\nENTRYPOINT [\"/app\"]\n"),
            ("infra/sg.tf", tf_cfg),
            ("migrations/001_init.up.sql", "DROP TEMPORARY TABLE scratch_buf;\nDROP TABLE legacy_users;\n"),
            ("migrations/001_init.down.sql", "DROP TABLE legacy_users;\nTRUNCATE TABLE audit_log;\n"),
        ]);

        let (_, _, findings) = scan_parts(dir.path(), None);
        for (file, rule, expected) in [
            (".github/workflows/safe.yml", "gha_script_injection", 0),
            (".github/workflows/safe.yml", "gha_secrets_outside_env", 0),
            (".github/workflows/safe.yml", "gha_unpinned_action", 0),
            (".github/workflows/unsafe.yml", "gha_script_injection", 1),
            (".github/workflows/unsafe.yml", "gha_secrets_outside_env", 1),
            (".github/workflows/unsafe.yml", "gha_unpinned_action", 1),
            ("Dockerfile", "docker_runs_as_root", 1),
            ("scratch.dockerfile", "docker_runs_as_root", 0),
            ("nonroot.dockerfile", "docker_runs_as_root", 0),
            ("infra/sg.tf", "tf_wildcard_iam", 0),
            ("infra/sg.tf", "tf_open_cidr", 1),
            ("migrations/001_init.up.sql", "sql_destructive_migration", 1),
            ("migrations/001_init.down.sql", "sql_destructive_migration", 0),
        ] {
            let count = findings.iter().filter(|f| f.file == file && f.rule_id == rule).count();
            assert_eq!(count, expected, "{file}:{rule}");
        }
    }
}
