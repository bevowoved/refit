use crate::lex::{ident_byte, left_bound, quoted_lit, scan_code};
use crate::platform::{
    SecretTier, ensure_cachedir_tag, find_secret, find_secret_tiered, now_ms, read_git_head_fast,
    read_zstd, secret_rule_meta, walk_files, write_atomic, write_zstd,
};
use crate::source::{RepoSnapshot, ext_of, parse_deps, rel_norm, supported_ext};
use crate::types::{InfraFinding, UnusedDep};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use anyhow::Result;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::time::{Instant, UNIX_EPOCH};

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
        if raw_trim.is_empty() { self.prev_ignore = None; return true; }
        if code_empty { self.prev_ignore = parse_ignore_directive(raw_trim); return true; }
        false
    }

    fn begin_line(&mut self, raw: &str) { self.inline_ignore = parse_ignore_directive(raw); }
    fn end_line(&mut self) { self.prev_ignore = None; self.inline_ignore = None; }

    fn is_suppressed(&self, rule_id: &str) -> bool {
        let hit = |list: &Vec<String>| list.is_empty() || list.iter().any(|s| s.eq_ignore_ascii_case(rule_id));
        self.inline_ignore.as_ref().is_some_and(hit) || self.prev_ignore.as_ref().is_some_and(hit)
    }

    fn emit(&mut self, line: usize, severity: &str, rule_id: &str, message: &str) {
        if !self.is_suppressed(rule_id) { self.emit_raw(line, severity, rule_id, message); }
    }

    fn emit_raw(&mut self, line: usize, severity: &str, rule_id: &str, message: &str) {
        self.out.push(InfraFinding {
            file: self.rel.into(), line, severity: severity.into(), rule_id: rule_id.into(), message: message.into(),
        });
    }
}

fn parse_ignore_directive(raw_line: &str) -> Option<Vec<String>> {
    let cap = IGNORE_RE.captures(raw_line)?;
    Some(cap.get(1).map_or_else(Vec::new, |m| {
        m.as_str().split(',').map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()).collect()
    }))
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
        if let Some(bytes) = snap.preloaded.get(rel) { matcher.scan_bytes(bytes); }
        else if let Ok(bytes) = fs::read(repo_root.join(rel)) { matcher.scan_bytes(&bytes); }
        if matchers.iter().all(|e| e.as_ref().is_none_or(EcoMatcher::is_complete)) { break; }
    }
    collect_unused_from_matchers(parsed, &matchers)
}

fn default_tier_high() -> String { "high".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretHit {
    pub file: String, pub line: usize, pub kind: String,
    #[serde(default = "default_tier_high")]
    pub tier: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecretScanReport {
    pub mode: String, pub rule_count: usize, pub files_checked: usize, pub cache_hits: usize,
    pub duration_ms: u64, pub blocking_count: usize, pub warning_count: usize,
    pub findings: Vec<SecretHit>,
}

type CachedSecret = (u64, u64, Option<(usize, String, String)>);

#[derive(Default, Serialize, Deserialize)]
struct SecretCacheDisk { rules_hash: String, entries: HashMap<String, CachedSecret> }

struct FileSecretScan {
    file: String, mtime: u64, sz: u64, cached: bool, hit: Option<(usize, String, String)>,
}

fn count_tiers(findings: &[SecretHit]) -> (usize, usize) {
    let blocking = findings.iter().filter(|f| f.tier == "high").count();
    (blocking, findings.len() - blocking)
}

fn scan_git_blobs(repo: &Path, specs: &[(String, String)]) -> Vec<SecretHit> {
    if specs.is_empty() { return Vec::new(); }
    let Ok(mut child) = Command::new("git").args(["cat-file", "--batch"]).current_dir(repo)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()
    else { return Vec::new() };
    if let Some(mut stdin) = child.stdin.take() {
        for (_, obj) in specs { let _ = writeln!(stdin, "{obj}"); }
    }
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    let mut blobs = Vec::with_capacity(specs.len());
    for (rel, _) in specs {
        let mut hdr = String::new();
        if reader.read_line(&mut hdr).ok().filter(|&n| n > 0).is_none() { break; }
        let sz = hdr.split_whitespace().nth(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        if sz == 0 || sz > 2 * 1024 * 1024 {
            let (mut sink, mut rem) = (vec![0u8; sz.min(64 * 1024)], sz);
            while rem > 0 {
                let take = rem.min(sink.len());
                if reader.read_exact(&mut sink[..take]).is_err() { break; }
                rem -= take;
            }
            let mut nl = [0u8; 1];
            let _ = reader.read_exact(&mut nl);
            continue;
        }
        let mut buf = vec![0u8; sz];
        if reader.read_exact(&mut buf).is_ok() {
            let mut nl = [0u8; 1]; let _ = reader.read_exact(&mut nl);
            blobs.push((rel.clone(), buf));
        }
    }
    let _ = child.wait();
    blobs.into_par_iter().filter_map(|(file, buf)| {
        let (line, kind, tier) = find_secret_tiered(&buf)?;
        Some(SecretHit { file, line, kind: kind.into(), tier: tier.as_str().into() })
    }).collect()
}

/// Parallel zero-recompute secret scanner for repo, staged index, or git diff ranges.
pub fn scan_repo_secrets(
    repo_root: &Path, staged: bool, diff_base: Option<&str>, no_cache: bool,
) -> Result<SecretScanReport> {
    let (started, (rule_count, rhash)) = (Instant::now(), secret_rule_meta());
    let cache_hash = format!("{rhash}:v2-tiered");
    if staged || diff_base.is_some() {
        let (mode, specs): (String, Vec<(String, String)>) = if staged {
            let out = Command::new("git").args(["diff-index", "--cached", "--full-index", "HEAD"])
                .current_dir(repo_root).output()?;
            let rows = String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| {
                let p: Vec<&str> = l.split_whitespace().collect();
                (p.len() >= 6 && p[4] != "D").then(|| (p[5].into(), p[3].into()))
            }).collect();
            ("staged".into(), rows)
        } else {
            let base = diff_base.unwrap_or("HEAD~1");
            let rng = if base.contains("..") { base.to_string() } else { format!("{base}..HEAD") };
            let out = Command::new("git").args(["diff", "--name-only", "--diff-filter=ACMR", &rng])
                .current_dir(repo_root).output()?;
            let tip = rng.split("..").last().unwrap_or("HEAD").to_string();
            let rows = String::from_utf8_lossy(&out.stdout).lines()
                .filter(|l| !l.trim().is_empty()).map(|f| (f.into(), format!("{tip}:{f}"))).collect();
            (format!("diff:{rng}"), rows)
        };
        let mut findings = scan_git_blobs(repo_root, &specs);
        findings.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
        let (blocking_count, warning_count) = count_tiers(&findings);
        return Ok(SecretScanReport {
            mode, rule_count, files_checked: specs.len(), cache_hits: 0,
            duration_ms: started.elapsed().as_millis() as u64,
            blocking_count, warning_count, findings,
        });
    }
    let cache_path = repo_root.join(".refit/cache/secrets_cache.zst");
    let prev = if no_cache {
        HashMap::new()
    } else {
        read_zstd::<SecretCacheDisk>(&cache_path)
            .filter(|c| c.rules_hash == cache_hash)
            .map_or_else(HashMap::new, |c| c.entries)
    };
    let files = walk_files(repo_root, None);
    let results: Vec<FileSecretScan> = files
        .par_iter()
        .filter_map(|abs| {
            let rel = rel_norm(repo_root, abs)?;
            let meta = fs::metadata(abs).ok()?;
            let sz = meta.len();
            if sz == 0 || sz > 2 * 1024 * 1024 { return None; }
            let mtime = meta.modified().ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos() as u64);
            if let Some((pm, ps, ph)) = prev.get(&rel) && *pm == mtime && *ps == sz {
                return Some(FileSecretScan { file: rel, mtime, sz, cached: true, hit: ph.clone() });
            }
            let bytes = fs::read(abs).ok()?;
            let hit = find_secret_tiered(&bytes)
                .map(|(l, k, t)| (l, k.to_string(), t.as_str().to_string()));
            Some(FileSecretScan { file: rel, mtime, sz, cached: false, hit })
        })
        .collect();
    let (mut next_map, mut findings, mut cache_hits) = (HashMap::with_capacity(results.len()), Vec::new(), 0usize);
    for row in results {
        if row.cached { cache_hits += 1; }
        if let Some((line, ref kind, ref tier)) = row.hit {
            findings.push(SecretHit {
                file: row.file.clone(), line, kind: kind.clone(), tier: tier.clone(),
            });
        }
        next_map.insert(row.file, (row.mtime, row.sz, row.hit));
    }
    findings.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
    let (blocking_count, warning_count) = count_tiers(&findings);
    if !no_cache {
        if let Some(p) = cache_path.parent() { ensure_cachedir_tag(p); }
        let _ = write_zstd(&cache_path, &SecretCacheDisk { rules_hash: cache_hash, entries: next_map });
    }
    Ok(SecretScanReport {
        mode: "repo".into(), rule_count, files_checked: files.len(), cache_hits,
        duration_ms: started.elapsed().as_millis() as u64,
        blocking_count, warning_count, findings,
    })
}

/// Render a `SecretScanReport` as minimal OASIS SARIF 2.1.0 JSON for GitHub Code Scanning.
pub fn render_secrets_sarif(rep: &SecretScanReport) -> Value {
    let rules: Vec<Value> = rep.findings.iter()
        .map(|f| f.kind.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|id| json!({
            "id": id,
            "shortDescription": { "text": format!("refit secret guard rule: {id}") },
        }))
        .collect();
    let results: Vec<Value> = rep.findings.iter().map(|hit| {
        let level = if hit.tier == "high" { "error" } else { "warning" };
        let fp = blake3::hash(format!("{}:{}:{}", hit.file, hit.line, hit.kind).as_bytes())
            .to_hex()
            .to_string();
        json!({
            "ruleId": hit.kind,
            "level": level,
            "message": { "text": format!("[refit:{}] detected ({} tier)", hit.kind, hit.tier) },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": hit.file },
                    "region": { "startLine": hit.line.max(1) }
                }
            }],
            "partialFingerprints": { "primaryLocationLineHash": fp }
        })
    }).collect();
    json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "refit",
                    "informationUri": "https://github.com/bevowoved/refit",
                    "rules": rules
                }
            },
            "results": results
        }]
    })
}

/// Append a single BLAKE3-fingerprinted result (and rule definition if new) to a SARIF 2.1.0 doc.
pub fn push_sarif_result(
    sarif: &mut Value, rule_id: &str, level: &str, file: &str, line: usize, text: &str,
) {
    if let Some(rules) = sarif.pointer_mut("/runs/0/tool/driver/rules").and_then(Value::as_array_mut)
        && !rules.iter().any(|r| r.get("id").and_then(Value::as_str) == Some(rule_id))
    {
        rules.push(json!({ "id": rule_id, "shortDescription": { "text": format!("refit rule: {rule_id}") } }));
    }
    let Some(results) = sarif.pointer_mut("/runs/0/results").and_then(Value::as_array_mut) else { return };
    let ln = line.max(1);
    let fp = blake3::hash(format!("{file}:{ln}:{rule_id}").as_bytes()).to_hex().to_string();
    results.push(json!({
        "ruleId": rule_id, "level": level, "message": { "text": text },
        "locations": [{ "physicalLocation": {
            "artifactLocation": { "uri": file }, "region": { "startLine": ln }
        } }],
        "partialFingerprints": { "primaryLocationLineHash": fp }
    }));
}

pub fn escape_gh_prop(raw: &str) -> String {
    raw.replace('%', "%25").replace('\r', "%0D").replace('\n', "%0A")
        .replace(':', "%3A").replace(',', "%2C")
}

/// Render a `SecretScanReport` as GitHub Actions workflow annotations (`::error` / `::warning`).
pub fn render_secrets_gh(rep: &SecretScanReport) -> String {
    let mut out = String::new();
    for hit in &rep.findings {
        let level = if hit.tier == "high" { "error" } else { "warning" };
        let file = escape_gh_prop(&hit.file);
        let detail = if hit.tier == "high" {
            "High-confidence secret or invisible-Unicode hazard detected"
        } else {
            "Potential high-entropy secret detected (verify or suppress with refit:ignore)"
        };
        out.push_str(&format!(
            "::{level} file={file},line={}::[refit:{}] {detail}\n",
            hit.line.max(1), hit.kind
        ));
    }
    out
}

const REFIT_HOOK_MARKER: &str = "# refit-managed-hook";

fn resolve_git_hooks_dir(repo_root: &Path) -> PathBuf {
    let common = Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .current_dir(repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|raw| {
            let p = PathBuf::from(&raw);
            if p.is_absolute() { p } else { repo_root.join(p) }
        })
        .unwrap_or_else(|| repo_root.join(".git"));
    common.join("hooks")
}

pub(crate) fn planned_secret_hook_paths(repo_root: &Path) -> Vec<String> {
    let hooks = resolve_git_hooks_dir(repo_root);
    ["pre-commit", "pre-push"]
        .into_iter()
        .map(|name| hooks.join(name).to_string_lossy().into_owned())
        .collect()
}

fn preserve_existing_hook(hook_path: &Path, name: &str) -> Result<()> {
    if !hook_path.exists() { return Ok(()); }
    let existing = fs::read_to_string(hook_path).unwrap_or_default();
    if existing.contains(REFIT_HOOK_MARKER) { return Ok(()); }
    let prev = hook_path.with_file_name(format!("{name}.refit-prev"));
    if !prev.exists() { fs::rename(hook_path, &prev)?; }
    Ok(())
}

/// Install worktree-safe, chained, idempotent pre-commit and pre-push Git hooks.
pub fn install_secret_hooks(repo_root: &Path) -> Result<Vec<String>> {
    let hooks = resolve_git_hooks_dir(repo_root);
    fs::create_dir_all(&hooks)?;
    let pc = concat!(
        "#!/usr/bin/env sh\n",
        "# refit-managed-hook\n",
        "PREV=\"$(dirname \"$0\")/pre-commit.refit-prev\"\n",
        "if [ -x \"$PREV\" ]; then\n  \"$PREV\" \"$@\" || exit $?\n",
        "elif [ -f \"$PREV\" ]; then\n  sh \"$PREV\" \"$@\" || exit $?\nfi\n",
        "exec refit secrets . --staged\n",
    );
    let pp = concat!(
        "#!/usr/bin/env sh\n",
        "# refit-managed-hook\n",
        "STDIN_DATA=\"$(cat)\"\n",
        "PREV=\"$(dirname \"$0\")/pre-push.refit-prev\"\n",
        "if [ -x \"$PREV\" ]; then\n  printf '%s\\n' \"$STDIN_DATA\" | \"$PREV\" \"$@\" || exit $?\n",
        "elif [ -f \"$PREV\" ]; then\n  printf '%s\\n' \"$STDIN_DATA\" | sh \"$PREV\" \"$@\" || exit $?\nfi\n",
        "Z=0000000000000000000000000000000000000000\n",
        "EMPTY=\"$(git hash-object -t tree /dev/null 2>/dev/null || echo 4b825dc642cb6eb9a060e54bf8d69288fbee4904)\"\n",
        "printf '%s\\n' \"$STDIN_DATA\" | while read -r _ l _ r; do\n",
        "  [ -z \"$l\" ] || [ \"$l\" = \"$Z\" ] && continue\n",
        "  if [ \"$r\" = \"$Z\" ]; then\n",
        "    FIRST=\"$(git rev-list \"$l\" --not --remotes 2>/dev/null | tail -n 1)\"\n",
        "    [ -z \"$FIRST\" ] && continue\n",
        "    BASE=\"$(git rev-parse --verify \"${FIRST}^\" 2>/dev/null || true)\"\n",
        "    if [ -n \"$BASE\" ]; then refit secrets . --diff \"$BASE..$l\" || exit 1\n",
        "    else refit secrets . --diff \"$EMPTY..$l\" || exit 1; fi\n",
        "  else\n",
        "    refit secrets . --diff \"$r..$l\" || exit 1\n",
        "  fi\n",
        "done\n",
    );
    let mut written = Vec::new();
    for (name, body) in [("pre-commit", pc), ("pre-push", pp)] {
        let p = hooks.join(name);
        preserve_existing_hook(&p, name)?;
        write_atomic(&p, body.as_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o755));
        }
        written.push(p.to_string_lossy().into_owned());
    }
    Ok(written)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HookHost { Claude, Cursor, Gemini, Kiro }

fn detect_hook_host(val: &Value) -> HookHost {
    let ev = val.get("hook_event_name").and_then(Value::as_str).unwrap_or("");
    if matches!(ev, "fileEdited" | "preToolUse" | "agentStop") { return HookHost::Kiro; }
    if matches!(ev, "beforeShellExecution" | "beforeReadFile" | "afterFileEdit")
        || (val.get("tool_name").is_none()
            && (val.get("command").is_some() || val.get("file_path").is_some() || val.get("edits").is_some()))
    {
        return HookHost::Cursor;
    }
    if matches!(ev, "BeforeTool" | "AfterTool") { return HookHost::Gemini; }
    HookHost::Claude
}

fn is_sensitive_read_path(path: &str) -> bool {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path).to_ascii_lowercase();
    if base.is_empty() { return false; }
    if base == ".env" { return true; }
    if base.starts_with(".env.") {
        return !matches!(base.as_str(), ".env.example" | ".env.sample" | ".env.template");
    }
    matches!(base.as_str(), "id_rsa" | "id_ed25519") || base.ends_with(".pem") || base.ends_with(".key")
}

fn split_shell_segments(cmd: &str) -> Vec<String> {
    let (mut segs, mut cur, mut quote, mut esc) = (Vec::new(), String::new(), 0u8, false);
    for ch in cmd.chars() {
        if esc { cur.push(ch); esc = false; continue; }
        if ch == '\\' && quote != b'\'' { cur.push(ch); esc = true; continue; }
        if quote != 0 {
            if ch as u32 == u32::from(quote) { quote = 0; }
            cur.push(ch);
            continue;
        }
        if ch == '\'' || ch == '"' { quote = ch as u8; cur.push(ch); continue; }
        if matches!(ch, ';' | '|' | '&' | '\n') {
            if !cur.trim().is_empty() { segs.push(std::mem::take(&mut cur)); }
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() { segs.push(cur); }
    segs
}

/// Returns `(is_git_push, has_no_verify_bypass)` from a single `shlex` pass.
fn inspect_git_segment(seg: &str) -> (bool, bool) {
    let toks = shlex::split(seg)
        .unwrap_or_else(|| seg.split_whitespace().map(str::to_string).collect());
    let mut i = 0usize;
    while i < toks.len() {
        let bin = toks[i].rsplit(['/', '\\']).next().unwrap_or(&toks[i]).to_ascii_lowercase();
        if bin == "git" || bin == "git.exe" { break; }
        i += 1;
    }
    if i >= toks.len() { return (false, false); }
    i += 1;
    while i < toks.len() {
        let t = toks[i].as_str();
        if matches!(t, "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace") { i += 2; continue; }
        if t.starts_with('-') { i += 1; continue; }
        break;
    }
    let Some(sub) = toks.get(i).map(String::as_str) else { return (false, false) };
    let is_push = sub == "push";
    if !matches!(sub, "push" | "commit") { return (false, false); }
    let mut j = i + 1;
    while j < toks.len() {
        let arg = toks[j].as_str();
        if arg == "--" { break; }
        if arg == "--no-verify" { return (is_push, true); }
        if sub == "commit" {
            if matches!(arg, "-m" | "-F" | "-C" | "-c" | "--message" | "--file" | "--author") {
                j += 2; continue;
            }
            if arg == "-n" || (arg.starts_with('-') && !arg.starts_with("--") && arg[1..].contains('n')) {
                return (false, true);
            }
        }
        j += 1;
    }
    (is_push, false)
}

fn hook_cwd(val: &Value) -> PathBuf {
    val.get("cwd").and_then(Value::as_str).filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn kane_gate_enabled(cwd: &Path) -> bool {
    std::env::var("REFIT_KANE_GATE").ok().is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        || cwd.join(".refit/kane.json").is_file()
}

fn is_kane_watched_path(path: &str) -> bool {
    let low = path.to_ascii_lowercase();
    low.ends_with(".html") || low.ends_with(".tsx") || low.ends_with(".jsx")
        || low.ends_with(".ts") || low.ends_with(".js") || low.ends_with(".vue")
        || low.ends_with(".svelte") || low.ends_with(".css")
        || low.ends_with("readme.md") || low.ends_with("skill.md")
}

fn check_kane_push_gate(cwd: &Path) -> Option<(&'static str, String, &'static str)> {
    if !kane_gate_enabled(cwd) { return None; }
    let last_raw = fs::read_to_string(cwd.join(".refit/kane_last.json")).ok();
    let Some(last) = last_raw.and_then(|s| serde_json::from_str::<Value>(&s).ok()) else {
        return Some((
            "kane_gate_unverified",
            "no Gate 4 Kane verification receipt found in `.refit/kane_last.json`".into(),
            "Run `refit check . --kane <spec>` before pushing",
        ));
    };
    if last.get("ok").and_then(Value::as_bool) != Some(true) {
        let v = last.get("verdict").and_then(Value::as_str).unwrap_or("failed");
        return Some((
            "kane_gate_unverified",
            format!("last Gate 4 Kane receipt failed (`{v}`)"),
            "Fix the failing browser/UI step and re-run `refit check . --kane <spec>`",
        ));
    }
    if let Some(cur_head) = read_git_head_fast(cwd)
        && let Some(rec_head) = last.get("git_head").and_then(Value::as_str)
        && cur_head != rec_head
    {
        return Some((
            "kane_gate_stale_head",
            "HEAD changed since the last green Gate 4 Kane receipt".into(),
            "Re-run `refit check . --kane <spec>` on the current commit before pushing",
        ));
    }
    let rec_ms = last.get("recorded_ms").and_then(Value::as_u64).unwrap_or(0);
    if let Ok(raw) = fs::read_to_string(cwd.join(".refit/kane_state.json"))
        && let Ok(st) = serde_json::from_str::<Value>(&raw)
        && st.get("dirty_ms").and_then(Value::as_u64).unwrap_or(0) > rec_ms
    {
        return Some((
            "kane_gate_dirty_ui",
            "UI/docs files were edited after the last green Gate 4 Kane run".into(),
            "Re-verify with `refit check . --kane <spec>` before pushing",
        ));
    }
    None
}

fn collect_write_payloads<'a>(val: &'a Value, input: &'a Value) -> Vec<&'a str> {
    let mut out = Vec::new();
    for src in [input, val] {
        for key in ["content", "new_string", "new_source", "newText"] {
            if let Some(s) = src.get(key).and_then(Value::as_str) { out.push(s); }
        }
        if let Some(arr) = src.get("edits").and_then(Value::as_array) {
            for item in arr {
                for key in ["new_string", "newText", "content"] {
                    if let Some(s) = item.get(key).and_then(Value::as_str) { out.push(s); }
                }
            }
        }
    }
    out
}

fn inspect_hook_event(val: &Value) -> Option<(&'static str, String, &'static str)> {
    let ev = val.get("hook_event_name").and_then(Value::as_str).unwrap_or("");
    let tool = val.get("tool_name").and_then(Value::as_str).unwrap_or("");
    let input = val.get("tool_input").unwrap_or(val);

    let is_shell = ev == "beforeShellExecution"
        || matches!(tool, "Bash" | "run_shell_command" | "shell" | "Shell");
    if is_shell {
        let cmd = input.get("command").or_else(|| val.get("command")).and_then(Value::as_str)?;
        let (mut has_push, mut has_bypass) = (false, false);
        for seg in split_shell_segments(cmd) {
            let (p, b) = inspect_git_segment(&seg);
            has_push |= p;
            has_bypass |= b;
        }
        if has_bypass {
            return Some((
                "git_hook_bypass",
                "command attempts to bypass Git verification hooks".into(),
                "Remove --no-verify/-n and fix the underlying hook failure",
            ));
        }
        if let Some((_, kind, SecretTier::High)) = find_secret_tiered(cmd.as_bytes()) {
            return Some((
                kind,
                "high-confidence secret embedded in shell command".into(),
                "Pass credentials via environment variables or a secret manager, never inline",
            ));
        }
        if has_push {
            return check_kane_push_gate(&hook_cwd(val));
        }
        return None;
    }

    let is_read = ev == "beforeReadFile"
        || matches!(tool, "Read" | "read_file" | "ReadFile" | "view_file");
    if is_read {
        let path = input.get("file_path").or_else(|| input.get("path"))
            .or_else(|| val.get("file_path")).or_else(|| val.get("path"))
            .and_then(Value::as_str)?;
        if is_sensitive_read_path(path) {
            return Some((
                "sensitive_file_read",
                format!("reading sensitive credential file `{path}`"),
                "Read `.env.example` or ask the user to inspect secrets outside the agent context",
            ));
        }
        return None;
    }

    let is_write = ev == "afterFileEdit"
        || matches!(tool, "Write" | "Edit" | "MultiEdit" | "NotebookEdit" | "write_file" | "replace" | "edit_file" | "WriteFile" | "Replace");
    if is_write {
        for payload in collect_write_payloads(val, input) {
            if let Some((line, kind, SecretTier::High)) = find_secret_tiered(payload.as_bytes()) {
                return Some((
                    kind,
                    format!("high-confidence secret or Unicode hazard on payload line {line}"),
                    "Remove the secret or invisible Unicode characters, or use an env var placeholder",
                ));
            }
        }
    }
    None
}

fn format_hook_deny(host: HookHost, reason: &str) -> String {
    match host {
        HookHost::Cursor => json!({ "permission": "deny", "agentMessage": reason }).to_string(),
        HookHost::Claude | HookHost::Kiro => json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason
            }
        }).to_string(),
        HookHost::Gemini => json!({ "decision": "deny", "reason": reason }).to_string(),
    }
}

/// Native sub-5ms pre/post-tool hook engine for Claude Code, Cursor, Gemini CLI, and Kiro.
pub fn run_agent_hook(stdin_json: &str) -> (i32, String, String) {
    if stdin_json.trim().is_empty() { return (0, String::new(), String::new()); }
    let Ok(val) = serde_json::from_str::<Value>(stdin_json) else {
        return (0, String::new(), String::new());
    };
    let host = detect_hook_host(&val);
    if let Some((kind, detail, remediation)) = inspect_hook_event(&val) {
        let reason = format!("[refit:hook] blocked {kind}: {detail}. Remediation: {remediation}");
        return (2, format_hook_deny(host, &reason), reason);
    }
    let ev = val.get("hook_event_name").and_then(Value::as_str).unwrap_or("");
    if matches!(ev, "PostToolUse" | "AfterTool" | "afterFileEdit" | "fileEdited") {
        let input = val.get("tool_input").unwrap_or(&val);
        let path = input.get("file_path").or_else(|| input.get("path"))
            .or_else(|| val.get("file_path")).or_else(|| val.get("path"))
            .and_then(Value::as_str).unwrap_or("");
        let cwd = hook_cwd(&val);
        if is_kane_watched_path(path) && kane_gate_enabled(&cwd) {
            let body = json!({ "dirty_ms": now_ms(), "file": path }).to_string();
            let _ = write_atomic(&cwd.join(".refit/kane_state.json"), body.as_bytes());
            if matches!(host, HookHost::Claude | HookHost::Gemini) {
                let ctx = format!("[refit:kane] UI/doc edit in `{path}`. Re-verify via `refit check . --kane <spec>` before pushing.");
                return (0, json!({ "hookSpecificOutput": { "hookEventName": ev, "additionalContext": ctx } }).to_string(), String::new());
            }
        }
    }
    let ok_out = if host == HookHost::Cursor {
        r#"{"permission":"allow"}"#.to_string()
    } else {
        "{}".to_string()
    };
    (0, ok_out, String::new())
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

    #[test]
    fn secrets_tiered_sarif_and_gh_formatters() {
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        let ent = ["api_key = \"", "aB9xQ2mK7pZ4vL8nR3tW6yC1eH5j\""].concat();
        let dir = TempRepo::with_files("sec_tier_fmt", &[
            ("src/a.rs", &format!("let k = \"{aws}\";\n")),
            ("src/b.rs", &format!("{ent}\n")),
            ("src/c.rs", "let ok = \"clean\";\n"),
        ]);
        let rep = scan_repo_secrets(dir.path(), false, None, false).unwrap();
        assert_eq!((rep.blocking_count, rep.warning_count, rep.findings.len()), (1, 1, 2));
        let sarif = render_secrets_sarif(&rep);
        assert_eq!(sarif["version"], "2.1.0");
        assert_eq!(sarif["runs"][0]["tool"]["driver"]["name"], "refit");
        let results = sarif["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["level"], "error");
        assert_eq!(results[1]["level"], "warning");
        assert!(results[0]["partialFingerprints"]["primaryLocationLineHash"].as_str().unwrap().len() == 64);

        let gh = render_secrets_gh(&rep);
        assert!(gh.contains("::error file=src/a.rs,line=1::[refit:aws_access_key_id]"));
        assert!(gh.contains("::warning file=src/b.rs,line=1::[refit:high_entropy_secret]"));
    }

    #[test]
    fn install_hooks_chains_idempotent_and_worktree_safe() {
        let repo = TempRepo::git("hooks_wt", "app", &[("src/lib.rs", "pub fn ok() {}\n")]);
        let hooks_dir = repo.join(".git/hooks");
        fs::create_dir_all(&hooks_dir).unwrap();
        let orig = "#!/usr/bin/env sh\necho custom-pre-commit\n";
        fs::write(hooks_dir.join("pre-commit"), orig).unwrap();

        let first = install_secret_hooks(repo.path()).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(fs::read_to_string(hooks_dir.join("pre-commit.refit-prev")).unwrap(), orig);
        let pc_body = fs::read_to_string(hooks_dir.join("pre-commit")).unwrap();
        let pp_body = fs::read_to_string(hooks_dir.join("pre-push")).unwrap();
        assert!(pc_body.contains(REFIT_HOOK_MARKER) && !pc_body.contains('\r'));
        assert!(pp_body.contains("git rev-list") && !pp_body.contains("HEAD~1..$l") && !pp_body.contains('\r'));

        let _second = install_secret_hooks(repo.path()).unwrap();
        assert_eq!(fs::read_to_string(hooks_dir.join("pre-commit.refit-prev")).unwrap(), orig);

        let wt_path = repo.join("linked_wt");
        let st = Command::new("git")
            .args(["worktree", "add", "-b", "wt-branch", wt_path.to_str().unwrap()])
            .current_dir(repo.path())
            .status()
            .unwrap();
        assert!(st.success());
        assert!(wt_path.join(".git").is_file());
        let wt_hooks = install_secret_hooks(&wt_path).unwrap();
        assert_eq!(wt_hooks.len(), 2);
        assert!(!wt_path.join(".git/hooks").exists());
        assert_eq!(fs::read_to_string(hooks_dir.join("pre-commit.refit-prev")).unwrap(), orig);
    }

    #[test]
    fn agent_hook_blocks_and_allows_all_hosts() {
        let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
        assert_eq!(run_agent_hook(""), (0, String::new(), String::new()));
        assert_eq!(run_agent_hook("{bad json"), (0, String::new(), String::new()));

        let claude_write_bad = json!({
            "hook_event_name": "PreToolUse", "tool_name": "Write",
            "tool_input": { "file_path": "src/cfg.rs", "content": format!("const K: &str = \"{aws}\";") }
        }).to_string();
        let (code, out, err) = run_agent_hook(&claude_write_bad);
        assert_eq!(code, 2);
        assert!(err.contains("aws_access_key_id") && out.contains("\"permissionDecision\":\"deny\""));

        let claude_unicode = json!({
            "tool_name": "Edit", "tool_input": { "new_string": "safe \u{E0001} prompt" }
        }).to_string();
        assert_eq!(run_agent_hook(&claude_unicode).0, 2);

        let claude_bash_bypass = json!({
            "tool_name": "Bash", "tool_input": { "command": "git commit -n -m 'skip'" }
        }).to_string();
        assert_eq!(run_agent_hook(&claude_bash_bypass).0, 2);

        let claude_bash_safe = json!({
            "tool_name": "Bash", "tool_input": { "command": "git commit -m 'note -n inside msg'" }
        }).to_string();
        assert_eq!(run_agent_hook(&claude_bash_safe), (0, "{}".into(), String::new()));

        let cursor_push_bypass = json!({
            "hook_event_name": "beforeShellExecution",
            "command": "git push origin main --no-verify"
        }).to_string();
        let (code, out, _) = run_agent_hook(&cursor_push_bypass);
        assert_eq!(code, 2);
        assert!(out.contains("\"permission\":\"deny\""));

        let cursor_read_env = json!({ "hook_event_name": "beforeReadFile", "file_path": "/app/.env.production" }).to_string();
        assert_eq!(run_agent_hook(&cursor_read_env).0, 2);
        let cursor_read_example = json!({ "hook_event_name": "beforeReadFile", "file_path": "/app/.env.example" }).to_string();
        assert_eq!(run_agent_hook(&cursor_read_example), (0, r#"{"permission":"allow"}"#.into(), String::new()));

        let gemini_read_key = json!({
            "hook_event_name": "BeforeTool", "tool_name": "read_file",
            "tool_input": { "file_path": "/home/u/.ssh/id_ed25519" }
        }).to_string();
        let (code, out, _) = run_agent_hook(&gemini_read_key);
        assert_eq!(code, 2);
        assert!(out.contains("\"decision\":\"deny\""));
    }

    #[test]
    fn agent_hook_kane_push_gate_and_dirty_ui_tracking() {
        let repo = TempRepo::git("kane_hook_gate", "app", &[("docs/index.html", "<h1>ok</h1>\n")]);
        fs::create_dir_all(repo.join(".refit")).unwrap();
        fs::write(repo.join(".refit/kane.json"), r#"{"default_spec":"tests/kane/hud_test.md"}"#).unwrap();
        let cwd = repo.path().display().to_string();

        // 1. Push without receipt is denied when .refit/kane.json exists
        let push_ev = json!({
            "hook_event_name": "PreToolUse", "tool_name": "Bash", "cwd": cwd,
            "tool_input": { "command": "git push origin main" }
        }).to_string();
        let (code, out, err) = run_agent_hook(&push_ev);
        assert_eq!(code, 2);
        assert!(err.contains("kane_gate_unverified") && out.contains("\"permissionDecision\":\"deny\""));

        // 2. PostToolUse UI edit marks .refit/kane_state.json dirty and emits additionalContext
        let edit_ev = json!({
            "hook_event_name": "PostToolUse", "tool_name": "Edit", "cwd": cwd,
            "tool_input": { "file_path": "docs/index.html", "new_string": "<h1>v2</h1>" }
        }).to_string();
        let (code, out, _) = run_agent_hook(&edit_ev);
        assert_eq!(code, 0);
        assert!(out.contains("additionalContext") && repo.join(".refit/kane_state.json").is_file());

        // 3. Green receipt newer than dirty_ms matching HEAD unblocks git push
        let head = read_git_head_fast(repo.path()).unwrap();
        let green = json!({ "ok": true, "verdict": "passed", "git_head": head, "recorded_ms": now_ms() + 10_000 }).to_string();
        fs::write(repo.join(".refit/kane_last.json"), green).unwrap();
        let (code, out, _) = run_agent_hook(&push_ev);
        assert_eq!(code, 0, "expected push allowed after green receipt: {out}");
    }
}
