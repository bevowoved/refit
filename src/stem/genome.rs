//! 100% clean-room Rust git co-change NPMI coupling, recency decay, synthetic summaries,
//! and index-wipe safety guard (zero third-party BUSL code).

use crate::platform::char_prefix;
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_GIT_COMMITS: usize = 2500;
pub const BULK_COMMIT_LIMIT: usize = 40;
pub const MIN_COCHANGE_SUPPORT: u32 = 3;
pub const RECENT_WINDOW_SECS: u64 = 48 * 3600;

#[derive(Debug, Clone, PartialEq)]
pub struct CochangePair {
    pub file_a: String,
    pub file_b: String,
    pub npmi: f64,
    pub commits: u32,
}

#[derive(Debug, Clone, Default)]
pub struct GenomeSummary {
    pub pairs: Vec<CochangePair>,
}

/// Returns `true` when `new_count` drops by more than 80% relative to `prev_count`
/// (with `prev_count >= 50`), guarding against accidental index wipe on broken checkouts.
#[inline]
pub fn check_wipe_guard(prev_count: usize, new_count: usize) -> bool {
    prev_count >= 50 && new_count.saturating_mul(5) < prev_count
}

/// Lockfiles and churn-heavy artifacts excluded from NPMI co-change pairs.
pub fn is_ignored_cochange_path(path: &str) -> bool {
    let fname = path.rsplit('/').next().unwrap_or(path);
    matches!(
        fname,
        "Cargo.lock"
            | "pnpm-lock.yaml"
            | "package-lock.json"
            | "yarn.lock"
            | "bun.lock"
            | "bun.lockb"
            | "uv.lock"
            | "poetry.lock"
            | "Pipfile.lock"
            | "go.sum"
            | "flake.lock"
            | "CHANGELOG.md"
    ) || fname.ends_with(".snap")
        || fname.ends_with(".min.js")
        || fname.ends_with(".min.css")
}

/// Return true if file mtime is within the `[NEW]` 48-hour recency window.
pub fn is_recent_48h(repo_root: &Path, rel: &str) -> bool {
    let Ok(meta) = std::fs::metadata(repo_root.join(rel)) else {
        return false;
    };
    let Ok(mtime) = meta.modified() else {
        return false;
    };
    let Ok(age) = SystemTime::now().duration_since(mtime) else {
        return true;
    };
    age.as_secs() <= RECENT_WINDOW_SECS
}

/// Exponential 14-day recency decay weight in `(0.0, 1.0]`.
pub fn recency_decay(age_secs: u64) -> f64 {
    const FOURTEEN_DAYS: f64 = 14.0 * 86_400.0;
    (-(age_secs as f64) / FOURTEEN_DAYS).exp()
}

/// Extract a compact deterministic zero-LLM module summary with `(->N)` fan-in and `[NEW]` badge.
pub fn synthetic_summary(
    rel: &str,
    src: &str,
    exports: &[String],
    fan_in: usize,
    is_new: bool,
    top_twin: Option<(&str, f64)>,
) -> String {
    let doc = first_doc_sentence(src);
    let mut parts = Vec::with_capacity(4);
    if is_new {
        parts.push("[NEW]".to_string());
    }
    if fan_in > 0 {
        parts.push(format!("(->{fan_in})"));
    }
    if !exports.is_empty() {
        let joined = exports.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
        parts.push(format!("exports:[{joined}]"));
    }
    if let Some(d) = doc {
        parts.push(char_prefix(&d, 72).to_string());
    }
    if let Some((twin, npmi)) = top_twin {
        parts.push(format!("twin:{twin}({npmi:.2})"));
    }
    if parts.is_empty() {
        format!("{rel} (->0)")
    } else {
        format!("{rel} {}", parts.join(" | "))
    }
}

fn first_doc_sentence(src: &str) -> Option<String> {
    for line in src.lines().take(20) {
        let t = line.trim();
        let cleaned = t
            .strip_prefix("///")
            .or_else(|| t.strip_prefix("//!"))
            .or_else(|| t.strip_prefix("/**"))
            .or_else(|| t.strip_prefix("* "))
            .or_else(|| t.strip_prefix("\"\"\""))
            .or_else(|| t.strip_prefix("# "));
        if let Some(raw) = cleaned {
            let trimmed = raw.trim().trim_end_matches("*/").trim_end_matches("\"\"\"").trim();
            if trimmed.len() >= 8 {
                let sentence = trimmed.split(". ").next().unwrap_or(trimmed).trim();
                return Some(sentence.to_string());
            }
        }
    }
    None
}

#[derive(Default)]
struct NpmiCounts {
    path_to_id: FxHashMap<String, usize>,
    id_to_path: Vec<String>,
    file_commits: Vec<u32>,
    co_counts: FxHashMap<u64, u32>,
    valid_commits: u32,
}

impl NpmiCounts {
    fn intern_path(&mut self, path: &str) -> usize {
        if let Some(&existing) = self.path_to_id.get(path) {
            return existing;
        }
        let next_id = self.id_to_path.len();
        self.id_to_path.push(path.to_string());
        self.file_commits.push(0);
        self.path_to_id.insert(path.to_string(), next_id);
        next_id
    }

    fn record_commit(
        &mut self,
        raw: &[String],
        live_files: Option<&FxHashSet<&str>>,
        commit_ids: &mut Vec<usize>,
    ) {
        if raw.len() >= BULK_COMMIT_LIMIT {
            return;
        }
        commit_ids.clear();
        for path in raw {
            if is_ignored_cochange_path(path) {
                continue;
            }
            if live_files.is_some_and(|live| !live.contains(path.as_str())) {
                continue;
            }
            let id = self.intern_path(path);
            commit_ids.push(id);
        }
        commit_ids.sort_unstable();
        commit_ids.dedup();
        if commit_ids.len() < 2 {
            return;
        }
        self.valid_commits += 1;
        for &id in commit_ids.iter() {
            self.file_commits[id] += 1;
        }
        for i in 0..commit_ids.len() {
            for j in (i + 1)..commit_ids.len() {
                let key = ((commit_ids[i] as u64) << 32) | (commit_ids[j] as u64);
                *self.co_counts.entry(key).or_insert(0) += 1;
            }
        }
    }
}

fn score_npmi_pair(
    counts: &NpmiCounts,
    key: u64,
    c_xy: u32,
    min_support: u32,
    n_f: f64,
) -> Option<CochangePair> {
    if c_xy < min_support {
        return None;
    }
    let idx_a = (key >> 32) as usize;
    let idx_b = (key & 0xffff_ffff) as usize;
    let p_xy = (c_xy as f64) / n_f;
    let p_x = (counts.file_commits[idx_a] as f64) / n_f;
    let p_y = (counts.file_commits[idx_b] as f64) / n_f;
    if p_x <= 0.0 || p_y <= 0.0 || p_xy <= 0.0 {
        return None;
    }
    let pmi = (p_xy / (p_x * p_y)).ln();
    let denom = (-p_xy.ln()).max(1e-9);
    let npmi = (pmi / denom).clamp(-1.0, 1.0);
    if npmi <= 0.05 {
        return None;
    }
    let path_a = &counts.id_to_path[idx_a];
    let path_b = &counts.id_to_path[idx_b];
    let (file_a, file_b) = if path_a <= path_b {
        (path_a.clone(), path_b.clone())
    } else {
        (path_b.clone(), path_a.clone())
    };
    Some(CochangePair { file_a, file_b, npmi, commits: c_xy })
}

/// Pure-function NPMI engine over canonicalized commit file lists (newest-to-oldest).
/// Filters `>= 40`-file housekeeping commits, lockfiles, and pairs below `min_support`.
pub fn compute_npmi_commits(
    commits: &[Vec<String>],
    live_files: Option<&FxHashSet<&str>>,
    min_support: u32,
) -> GenomeSummary {
    let mut counts = NpmiCounts::default();
    let mut commit_ids: Vec<usize> = Vec::with_capacity(BULK_COMMIT_LIMIT);
    for raw in commits {
        counts.record_commit(raw, live_files, &mut commit_ids);
    }

    let n_f = counts.valid_commits.max(1) as f64;
    let mut pairs: Vec<CochangePair> = counts
        .co_counts
        .iter()
        .filter_map(|(&key, &c_xy)| score_npmi_pair(&counts, key, c_xy, min_support, n_f))
        .collect();

    pairs.sort_by(|lhs, rhs| {
        rhs.npmi
            .partial_cmp(&lhs.npmi)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| rhs.commits.cmp(&lhs.commits))
            .then_with(|| lhs.file_a.cmp(&rhs.file_a))
            .then_with(|| lhs.file_b.cmp(&rhs.file_b))
    });

    GenomeSummary { pairs }
}

fn parse_git_status_line(
    trimmed: &str,
    rename_map: &mut FxHashMap<String, String>,
    current: &mut Vec<String>,
) {
    let mut cols = trimmed.split('\t');
    let status = cols.next().unwrap_or("");
    if status.starts_with('R') {
        if let (Some(old_p), Some(new_p)) = (cols.next(), cols.next()) {
            let old_norm = old_p.replace('\\', "/");
            let new_norm = new_p.replace('\\', "/");
            let canon = rename_map.get(&new_norm).cloned().unwrap_or(new_norm);
            rename_map.insert(old_norm, canon.clone());
            current.push(canon);
        }
    } else if let Some(path) = cols.next() {
        let norm = path.replace('\\', "/");
        let canon = rename_map.get(&norm).cloned().unwrap_or(norm);
        current.push(canon);
    }
}

/// Stream up to 2,500 non-merge commits from `git log --no-merges --name-status -M`
/// using a 64 KiB `BufReader` (zero full-log buffering) and compute clean-room NPMI twins.
pub fn analyze_git_cochange(repo_root: &Path, source_rels: &[String]) -> GenomeSummary {
    if !repo_root.join(".git").exists() {
        return GenomeSummary::default();
    }
    let Ok(mut child) = Command::new("git")
        .args([
            "log",
            "--no-merges",
            "--name-status",
            "-M",
            "-n",
            "2500",
            "--format=COMMIT %ct",
        ])
        .current_dir(repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return GenomeSummary::default();
    };

    let Some(stdout) = child.stdout.take() else {
        let _ = child.wait();
        return GenomeSummary::default();
    };

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(0);
    let mut reader = BufReader::with_capacity(64 * 1024, stdout);
    let mut line = String::new();
    let mut commits: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut rename_map: FxHashMap<String, String> = FxHashMap::default();
    let mut commit_count = 0usize;

    while let Ok(n) = reader.read_line(&mut line) {
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if let Some(ts_str) = trimmed.strip_prefix("COMMIT ") {
            if !current.is_empty() {
                commits.push(std::mem::take(&mut current));
            }
            commit_count += 1;
            if commit_count > MAX_GIT_COMMITS {
                break;
            }
            if let Ok(ts) = ts_str.trim().parse::<u64>() {
                let _decay = recency_decay(now_secs.saturating_sub(ts));
            }
        } else if !trimmed.is_empty() {
            parse_git_status_line(trimmed, &mut rename_map, &mut current);
        }
        line.clear();
    }
    if !current.is_empty() {
        commits.push(current);
    }
    let _ = child.wait();

    let live: FxHashSet<&str> = source_rels.iter().map(String::as_str).collect();
    let live_opt = (!live.is_empty()).then_some(&live);
    compute_npmi_commits(&commits, live_opt, MIN_COCHANGE_SUPPORT)
}

pub fn format_genome_twins(summary: &GenomeSummary, limit: usize) -> Vec<String> {
    summary
        .pairs
        .iter()
        .take(limit)
        .map(|p| {
            format!(
                "{} <-> {} (npmi={:.2}, commits={})",
                p.file_a, p.file_b, p.npmi, p.commits
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_npmi_hub_demotion_lockfile_filter_and_wipe_guard() {
        let mut commits: Vec<Vec<String>> = Vec::new();
        // 1. Bulk housekeeping commit (>= 40 files) must be discarded completely.
        let mut bulk = vec!["src/hub.rs".to_string(), "Cargo.lock".to_string()];
        for i in 0..45 {
            bulk.push(format!("src/bulk_{i}.rs"));
        }
        commits.push(bulk);

        // 2. Global hub touched across 60 scattered commits with rotating random files,
        //    plus Cargo.lock touched frequently.
        for c in 0..60 {
            let mut files = vec![
                "src/hub.rs".to_string(),
                format!("src/noise_{}.rs", c % 15),
                "Cargo.lock".to_string(),
                "pnpm-lock.yaml".to_string(),
            ];
            if c % 5 == 0 {
                files.push("src/route.rs".to_string());
                files.push("src/schema.rs".to_string());
            }
            if c % 6 == 1 {
                files.push("src/auth.rs".to_string());
                files.push("src/session.rs".to_string());
            }
            commits.push(files);
        }

        let summary = compute_npmi_commits(&commits, None, MIN_COCHANGE_SUPPORT);
        // Zero lockfiles in output!
        assert!(
            !summary
                .pairs
                .iter()
                .any(|p| p.file_a.ends_with(".lock") || p.file_b.ends_with(".lock") || p.file_a.contains("pnpm-lock"))
        );
        // True logical twins beat the global hub!
        assert!(summary.pairs.len() >= 2);
        let top2: Vec<(&str, &str)> = summary
            .pairs
            .iter()
            .take(2)
            .map(|p| (p.file_a.as_str(), p.file_b.as_str()))
            .collect();
        assert!(top2.contains(&("src/route.rs", "src/schema.rs")), "top2={top2:?}");
        assert!(top2.contains(&("src/auth.rs", "src/session.rs")), "top2={top2:?}");

        // Wipe guard check (>80% drop on >= 50 files triggers abort).
        assert!(check_wipe_guard(100, 19));
        assert!(!check_wipe_guard(100, 20));
        assert!(!check_wipe_guard(40, 2));

        // Synthetic summary with (->N) fan-in and [NEW] badge.
        let s = synthetic_summary(
            "src/auth.rs",
            "/// Validates JWT bearer tokens.\npub fn verify() {}\n",
            &["verify".into()],
            7,
            true,
            Some(("src/session.rs", 0.86)),
        );
        assert!(s.contains("[NEW]") && s.contains("(->7)") && s.contains("twin:src/session.rs(0.86)"));
        assert!(recency_decay(0) > 0.99 && recency_decay(14 * 86_400) < 0.40);
    }
}
