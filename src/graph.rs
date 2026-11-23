use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::host_os_tag;
use crate::stem::discover_repo;
use crate::types::{AsrfHunk, DagPlan, ModuleUnit};
use anyhow::{Context, Result};
use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Computes canonical BLAKE3 digest of public top-level Python def/class signatures
/// (body-stripped) so unauthorized exported interface changes abort in < 0.5 ms.
pub fn compute_python_abi_hash(source: &str) -> String {
    let mut sigs: Vec<String> = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let is_top_level = !line.starts_with(' ') && !line.starts_with('\t');
        let trimmed = line.trim();
        if is_top_level && (trimmed.starts_with("def ") || trimmed.starts_with("async def ") || trimmed.starts_with("class ")) {
            // Skip private helpers starting with _
            let is_private = trimmed
                .strip_prefix("async def ")
                .or_else(|| trimmed.strip_prefix("def "))
                .or_else(|| trimmed.strip_prefix("class "))
                .map(|rest| rest.starts_with('_'))
                .unwrap_or(false);
            if !is_private {
                let mut sig = trimmed.to_string();
                while !sig.contains(':') && i + 1 < lines.len() {
                    i += 1;
                    sig.push(' ');
                    sig.push_str(lines[i].trim());
                }
                if let Some(colon) = sig.rfind(':') {
                    sig.truncate(colon);
                }
                // Strip type annotations from ABI pin so adding PEP 484 types for Gate 3 `ty`
                // is ALLOWED while keeping function name & parameter names strictly pinned!
                let normalized = normalize_signature_names(&sig);
                sigs.push(normalized);
            }
        }
        i += 1;
    }
    sigs.sort();
    MvccStore::hash_bytes(sigs.join("\n").as_bytes())
}

fn normalize_signature_names(sig: &str) -> String {
    // Keep `def name(arg1, arg2)` or `class Name` with parameter names stripped of type annotations and defaults
    if sig.starts_with("class ") {
        let head = sig.split('(').next().unwrap_or(sig).trim();
        return head.to_string();
    }
    let Some(open) = sig.find('(') else {
        return sig.trim().to_string();
    };
    let Some(close) = sig.rfind(')') else {
        return sig.trim().to_string();
    };
    let prefix = sig[..open].trim();
    let params_raw = &sig[open + 1..close];
    let mut names = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for ch in params_raw.chars() {
        match ch {
            '[' | '(' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            ']' | ')' | '}' => {
                depth = depth.saturating_sub(1);
                cur.push(ch);
            }
            ',' if depth == 0 => {
                let name = extract_param_name(&cur);
                if !name.is_empty() {
                    names.push(name);
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    let last = extract_param_name(&cur);
    if !last.is_empty() {
        names.push(last);
    }
    format!("{prefix}({})", names.join(", "))
}

fn extract_param_name(param: &str) -> String {
    let p = param.trim();
    if p.is_empty() {
        return String::new();
    }
    let before_eq = p.split('=').next().unwrap_or(p).trim();
    let before_colon = before_eq.split(':').next().unwrap_or(before_eq).trim();
    before_colon.to_string()
}

pub fn build_dag_plan(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    hygiene: &HygieneScorer,
) -> Result<DagPlan> {
    let discovery = discover_repo(repo_root, stem_bin_hint);

    let mut source_files: Vec<String> = Vec::new();
    let mut test_files: Vec<String> = Vec::new();

    for entry in WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(
                name.as_ref(),
                ".refit" | ".venv" | ".git" | "__pycache__" | ".pytest_cache" | ".ruff_cache" | "target" | "node_modules" | ".tools"
            )
        })
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("py") {
            continue;
        }
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.ends_with("__init__.py") {
            continue;
        }
        if rel_str.starts_with("tests/") || rel_str.contains("/test_") || rel_str.starts_with("test_") {
            test_files.push(rel_str);
        } else {
            source_files.push(rel_str);
        }
    }

    source_files.sort();
    test_files.sort();

    // Map tests to source modules by stem match (`mod_12` -> `tests/test_mod_12.py`)
    let mut mod_to_tests: HashMap<String, Vec<String>> = HashMap::new();
    for src in &source_files {
        let stem = Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let mut matched: Vec<String> = test_files
            .iter()
            .filter(|t| {
                let ts = Path::new(t).file_stem().and_then(|s| s.to_str()).unwrap_or("");
                ts == format!("test_{stem}") || ts.ends_with(&format!("_{stem}"))
            })
            .cloned()
            .collect();
        if matched.is_empty() && !test_files.is_empty() {
            matched.push(test_files[0].clone());
        }
        mod_to_tests.insert(src.clone(), matched);
    }

    // Build file dependency DiGraph: edge A -> B means A depends on B (so B must complete before A)
    let mut g: DiGraph<String, ()> = DiGraph::new();
    let mut file_to_idx: HashMap<String, NodeIndex> = HashMap::new();
    for f in &source_files {
        let idx = g.add_node(f.clone());
        file_to_idx.insert(f.clone(), idx);
    }
    for (src, dst) in &discovery.edges {
        if let (Some(&u), Some(&v)) = (file_to_idx.get(src), file_to_idx.get(dst)) {
            if u != v && !g.contains_edge(u, v) {
                g.add_edge(u, v, ());
            }
        }
    }

    // Collapse cycles into atomic multi-file SCCs via Tarjan SCC
    let sccs = tarjan_scc(&g);
    let mut file_to_scc: HashMap<String, usize> = HashMap::new();
    let mut scc_files_list: Vec<Vec<String>> = Vec::with_capacity(sccs.len());
    for (scc_idx, comp) in sccs.iter().enumerate() {
        let mut files: Vec<String> = comp.iter().map(|&ni| g[ni].clone()).collect();
        files.sort();
        for f in &files {
            file_to_scc.insert(f.clone(), scc_idx);
        }
        scc_files_list.push(files);
    }

    // Build condensed DAG between SCCs
    let n_sccs = scc_files_list.len();
    let mut scc_deps: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_sccs];
    let mut scc_dependents: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_sccs];

    for (src, dst) in &discovery.edges {
        if let (Some(&su), Some(&sv)) = (file_to_scc.get(src), file_to_scc.get(dst)) {
            if su != sv {
                scc_deps[su].insert(sv);
                scc_dependents[sv].insert(su);
            }
        }
    }

    // Compute Kahn wave depth: leaf dependencies = wave 0, dependents = 1 + max(dep.wave)
    let mut wave_of: Vec<usize> = vec![0; n_sccs];
    let mut in_deg: Vec<usize> = scc_deps.iter().map(|d| d.len()).collect();
    let mut queue: std::collections::VecDeque<usize> = in_deg
        .iter()
        .enumerate()
        .filter_map(|(i, &deg)| if deg == 0 { Some(i) } else { None })
        .collect();

    while let Some(u) = queue.pop_front() {
        let uw = wave_of[u];
        for &dep_of_u in &scc_dependents[u] {
            wave_of[dep_of_u] = wave_of[dep_of_u].max(uw + 1);
            in_deg[dep_of_u] = in_deg[dep_of_u].saturating_sub(1);
            if in_deg[dep_of_u] == 0 {
                queue.push_back(dep_of_u);
            }
        }
    }

    // Assign stable readable IDs
    let scc_ids: Vec<String> = scc_files_list
        .iter()
        .map(|files| {
            if files.len() == 1 {
                Path::new(&files[0])
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mod")
                    .to_string()
            } else {
                let joined: Vec<String> = files
                    .iter()
                    .map(|f| {
                        Path::new(f)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("m")
                            .to_string()
                    })
                    .collect();
                format!("scc_{}", joined.join("_"))
            }
        })
        .collect();

    let mut total_loc = 0usize;
    let mut units = Vec::with_capacity(n_sccs);
    let mut max_wave = 0usize;

    for i in 0..n_sccs {
        let files = &scc_files_list[i];
        let mut base_blake3 = BTreeMap::new();
        let mut abi_hash = BTreeMap::new();
        let mut unit_tests = BTreeSet::new();
        let mut unit_loc = 0usize;
        let mut score_sum = 0.0f64;

        for rel in files {
            let abs = repo_root.join(rel);
            let content = fs::read_to_string(&abs)
                .with_context(|| format!("read source file {}", abs.display()))?;
            let loc = content.lines().count();
            unit_loc += loc;
            total_loc += loc;
            base_blake3.insert(rel.clone(), MvccStore::hash_bytes(content.as_bytes()));
            abi_hash.insert(rel.clone(), compute_python_abi_hash(&content));
            let h = hygiene.score_content(rel, &content);
            score_sum += h.score;
            if let Some(ts) = mod_to_tests.get(rel) {
                for t in ts {
                    unit_tests.insert(t.clone());
                }
            }
        }

        let layer = infer_layer(&files[0]);
        let wave = wave_of[i];
        max_wave = max_wave.max(wave);

        let deps = scc_deps[i].iter().map(|&d| scc_ids[d].clone()).collect();
        let dependents = scc_dependents[i]
            .iter()
            .map(|&d| scc_ids[d].clone())
            .collect();

        units.push(ModuleUnit {
            id: scc_ids[i].clone(),
            files: files.clone(),
            layer,
            wave,
            deps,
            dependents,
            test_files: unit_tests.into_iter().collect(),
            loc: unit_loc,
            base_blake3,
            abi_hash,
            baseline_hygiene: score_sum / (files.len().max(1) as f64),
            allow_abi_change: files.len() > 1,
        });
    }

    units.sort_by(|a, b| a.wave.cmp(&b.wave).then(a.layer.cmp(&b.layer)).then(a.id.cmp(&b.id)));

    Ok(DagPlan {
        repo_root: repo_root.display().to_string(),
        host_os: host_os_tag(),
        stem_used: discovery.used_stem,
        stem_index_ms: discovery.index_ms,
        total_files: source_files.len(),
        total_loc,
        scc_count: units.iter().filter(|u| u.files.len() > 1).count(),
        wave_count: max_wave + 1,
        units,
    })
}

fn infer_layer(rel_path: &str) -> usize {
    if rel_path.contains("/core/") || rel_path.starts_with("core/") {
        0
    } else if rel_path.contains("/engine/") || rel_path.starts_with("engine/") {
        1
    } else if rel_path.contains("/services/") || rel_path.starts_with("services/") {
        2
    } else {
        3
    }
}

/// Micro-slices a unit into a compact dynamic prompt packet (<= ~480 tokens for normal files,
/// keeping exact untouched search context for dirty hotspot functions and imported stubs).
pub fn slice_ast_packet(
    repo_root: &Path,
    unit: &ModuleUnit,
    diagnostics: &[String],
    learn_suffix: &str,
    max_chars: usize,
) -> Result<String> {
    let mut out = String::new();
    out.push_str(&format!(
        "Refactor unit `{}` (wave {}, layer {}, baseline hygiene {:.2}).\n",
        unit.id, unit.wave, unit.layer, unit.baseline_hygiene
    ));

    if !diagnostics.is_empty() {
        out.push_str("Current Gate 1/2/3 diagnostics to fix completely:\n");
        for d in diagnostics.iter().take(36) {
            out.push_str("- ");
            out.push_str(d.trim());
            out.push('\n');
        }
    }
    if unit.files.len() > 1 {
        out.push_str("IMPORTANT: This cyclic SCC unit has multiple files. Emit <<<REFIT_PATCH>>> blocks to fix ALL defects in EVERY file below (including every B006 mutable default, every Optional/+ None type error, every semicolon E702, every >88-col E501 line, and every deep nesting/if-elif chain).\n");
    }

    // Include compact test expectations when tests are failing so Gate 1 passes in 1 shot
    let has_test_fail = diagnostics.iter().any(|d| d.to_lowercase().contains("pytest") || d.contains("FAILED") || d.contains("assert"));
    if has_test_fail {
        for tf in unit.test_files.iter().take(1) {
            if let Ok(tc) = fs::read_to_string(repo_root.join(tf)) {
                let snippet: String = tc.chars().take(700).collect();
                out.push_str(&format!("\nRelevant test `{tf}`:\n```python\n{snippet}\n```\n"));
            }
        }
    }

    for rel in &unit.files {
        let content = fs::read_to_string(repo_root.join(rel))
            .with_context(|| format!("read {rel}"))?;
        let sliced = micro_slice_python(&content, max_chars / unit.files.len().max(1));
        out.push_str(&format!("\nFile `{rel}`:\n```python\n{sliced}\n```\n"));
    }

    if !learn_suffix.is_empty() {
        out.push_str(learn_suffix);
    }

    out.push_str(
        "\nEmit ONLY raw <<<REFIT_PATCH path=\"...\">>> blocks directly in your reply. Preserve public function names and parameter names. Zero lint/type warnings, 0 test failures, hygiene >= 0.85.\n",
    );
    Ok(out)
}

/// Extracts imports + dirty/complex top-level functions when file exceeds max_chars,
/// preserving exact contiguous line spans so SEARCH blocks match verbatim.
fn micro_slice_python(content: &str, budget_chars: usize) -> String {
    if content.len() <= budget_chars {
        return content.to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    let mut blocks: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, high_priority)
    // Header imports/constants block
    let mut header_end = 0;
    while header_end < lines.len() {
        let l = lines[header_end];
        if (l.starts_with("def ") || l.starts_with("async def ") || l.starts_with("class ")) && header_end > 0 {
            break;
        }
        header_end += 1;
    }
    blocks.push((0, header_end, true));
    let mut i = header_end;

    while i < lines.len() {
        let start = i;
        i += 1;
        while i < lines.len() {
            let l = lines[i];
            if !l.starts_with(' ') && !l.starts_with('\t') && (l.starts_with("def ") || l.starts_with("async def ") || l.starts_with("class ")) {
                break;
            }
            i += 1;
        }
        let slice = &lines[start..i];
        let is_hot = slice.iter().any(|l| {
            let indent = l.len().saturating_sub(l.trim_start().len());
            indent >= 12 || l.len() > 88 || l.contains("=[]") || l.contains("={}") || l.contains("None") || l.trim().starts_with("elif ")
        });
        blocks.push((start, i, is_hot));
    }

    let mut out = String::new();
    let mut omitted = 0usize;
    for (start, end, hot) in blocks {
        let chunk = lines[start..end].join("\n");
        if hot || out.len() + chunk.len() <= budget_chars {
            if omitted > 0 {
                out.push_str(&format!("\n# ... ({omitted} clean lines omitted) ...\n"));
                omitted = 0;
            }
            out.push_str(&chunk);
            out.push('\n');
        } else {
            // Keep 1-line signature stub of skipped clean helpers
            if let Some(first) = lines.get(start) {
                if first.starts_with("def ") || first.starts_with("class ") {
                    out.push_str(&format!("# stub: {}\n", first.trim()));
                }
            }
            omitted += end.saturating_sub(start);
        }
    }
    out
}

/// Parses raw unescaped ASRF patches from assistant.message.content.
pub fn parse_asrf_hunks(content: &str) -> Vec<AsrfHunk> {
    let mut hunks = Vec::new();
    let norm = content.replace("\r\n", "\n");
    let mut cursor = 0usize;

    while let Some(rel_start) = norm[cursor..].find("<<<REFIT_PATCH") {
        let start = cursor + rel_start;
        let Some(header_end_rel) = norm[start..].find(">>>") else {
            break;
        };
        let header = &norm[start..start + header_end_rel];
        let path = extract_attr_path(header);
        let after_header = start + header_end_rel + 3;

        let Some(search_rel) = norm[after_header..].find("<<<SEARCH>>>") else {
            cursor = after_header;
            continue;
        };
        let search_start = after_header + search_rel + "<<<SEARCH>>>".len();

        let Some(replace_rel) = norm[search_start..].find("<<<REPLACE>>>") else {
            cursor = search_start;
            continue;
        };
        let replace_start = search_start + replace_rel + "<<<REPLACE>>>".len();

        let Some(end_rel) = norm[replace_start..].find("<<<END>>>") else {
            cursor = replace_start;
            continue;
        };
        let end_pos = replace_start + end_rel;

        let search = strip_single_boundary_newline(&norm[search_start..search_start + replace_rel]);
        let replace = strip_single_boundary_newline(&norm[replace_start..end_pos]);

        if !path.is_empty() && !search.is_empty() {
            hunks.push(AsrfHunk {
                path: path.replace('\\', "/"),
                search,
                replace,
            });
        }

        cursor = end_pos + "<<<END>>>".len();
    }

    hunks
}

fn extract_attr_path(header: &str) -> String {
    if let Some(pos) = header.find("path=") {
        let rest = header[pos + 5..].trim();
        let trimmed = rest.trim_matches(|c| c == '"' || c == '\'' || c == ' ');
        return trimmed.split_whitespace().next().unwrap_or("").trim_matches(|c| c == '"' || c == '\'').to_string();
    }
    String::new()
}

fn strip_single_boundary_newline(s: &str) -> String {
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.to_string()
}

pub fn apply_asrf_hunks(
    base_files: &BTreeMap<String, String>,
    hunks: &[AsrfHunk],
) -> Result<BTreeMap<String, String>> {
    if hunks.is_empty() {
        anyhow::bail!("No <<<REFIT_PATCH>>> blocks found in assistant output");
    }
    let mut updated = base_files.clone();
    for (idx, h) in hunks.iter().enumerate() {
        let target_key = if updated.contains_key(&h.path) {
            h.path.clone()
        } else if let Some(k) = updated.keys().find(|k| k.ends_with(&h.path) || h.path.ends_with(k.as_str())) {
            k.clone()
        } else if updated.len() == 1 {
            updated.keys().next().unwrap().clone()
        } else {
            anyhow::bail!("Hunk #{idx} targets unknown path `{}`", h.path);
        };

        let cur = updated.get(&target_key).cloned().unwrap_or_default().replace("\r\n", "\n");
        let search_norm = h.search.replace("\r\n", "\n");
        let replace_norm = h.replace.replace("\r\n", "\n");

        if let Some(pos) = cur.find(&search_norm) {
            let mut next = String::with_capacity(cur.len() + replace_norm.len());
            next.push_str(&cur[..pos]);
            next.push_str(&replace_norm);
            next.push_str(&cur[pos + search_norm.len()..]);
            updated.insert(target_key, next);
        } else if let Some(next) = fallback_line_trimmed_replace(&cur, &search_norm, &replace_norm) {
            updated.insert(target_key, next);
        } else {
            anyhow::bail!(
                "Hunk #{idx} SEARCH block did not match `{target_key}` (first line: `{}`)",
                search_norm.lines().next().unwrap_or("")
            );
        }
    }
    Ok(updated)
}

fn fallback_line_trimmed_replace(doc: &str, search: &str, replace: &str) -> Option<String> {
    let doc_lines: Vec<&str> = doc.lines().collect();
    let s_lines: Vec<&str> = search.lines().collect();
    if s_lines.is_empty() || s_lines.len() > doc_lines.len() {
        return None;
    }
    let mut match_idx: Option<usize> = None;
    for i in 0..=doc_lines.len() - s_lines.len() {
        let ok = s_lines
            .iter()
            .enumerate()
            .all(|(j, sl)| doc_lines[i + j].trim() == sl.trim());
        if ok {
            if match_idx.is_some() {
                return None; // Ambiguous match
            }
            match_idx = Some(i);
        }
    }
    let start = match_idx?;
    let mut out_lines: Vec<String> = Vec::new();
    for l in &doc_lines[..start] {
        out_lines.push((*l).to_string());
    }
    for l in replace.lines() {
        out_lines.push(l.to_string());
    }
    for l in &doc_lines[start + s_lines.len()..] {
        out_lines.push((*l).to_string());
    }
    let mut joined = out_lines.join("\n");
    if doc.ends_with('\n') {
        joined.push('\n');
    }
    Some(joined)
}

/// Checks if two units share any 1-hop dependency or file (ensuring disjoint parallel cohorts).
pub fn units_interfere(a: &ModuleUnit, b: &ModuleUnit) -> bool {
    if a.id == b.id {
        return true;
    }
    let a_files: HashSet<&String> = a.files.iter().collect();
    if b.files.iter().any(|f| a_files.contains(f)) {
        return true;
    }
    a.deps.contains(&b.id) || b.deps.contains(&a.id)
}
