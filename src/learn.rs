use crate::platform::atomic_replace_file;
use crate::types::LearnRule;
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_RULES: usize = 16;

pub struct LearnStore {
    path: PathBuf,
    rules: Mutex<Vec<LearnRule>>,
}

impl LearnStore {
    pub fn open(repo_root: &Path) -> Result<Self> {
        let refit_dir = repo_root.join(".refit");
        fs::create_dir_all(&refit_dir)?;
        let path = refit_dir.join("learn.jsonl");
        let mut loaded = Vec::new();
        if path.exists() {
            if let Ok(raw) = fs::read_to_string(&path) {
                for line in raw.lines() {
                    if let Ok(rule) = serde_json::from_str::<LearnRule>(line.trim()) {
                        loaded.push(rule);
                    }
                }
            }
        }
        Ok(Self {
            path,
            rules: Mutex::new(loaded),
        })
    }

    pub fn record_failure(&self, gate: &str, dir_prefix: &str, raw_diagnostic: &str) {
        let (key, advice) = summarize_diagnostic(gate, raw_diagnostic);
        let mut guard = self.rules.lock().unwrap();
        if let Some(existing) = guard
            .iter_mut()
            .find(|r| r.key == key && r.dir_prefix == dir_prefix)
        {
            existing.hits += 1;
        } else {
            guard.push(LearnRule {
                key,
                gate: gate.to_string(),
                dir_prefix: dir_prefix.to_string(),
                advice,
                hits: 1,
                helpful: 0,
            });
            if guard.len() > MAX_RULES {
                guard.sort_by(|a, b| b.hits.cmp(&a.hits));
                guard.truncate(MAX_RULES);
            }
        }
        let _ = self.flush_locked(&guard);
    }

    pub fn record_success(&self, dir_prefix: &str) {
        let mut guard = self.rules.lock().unwrap();
        let mut changed = false;
        for r in guard.iter_mut() {
            if r.dir_prefix == dir_prefix || r.dir_prefix == "*" {
                r.helpful += 1;
                changed = true;
            }
        }
        if changed {
            let _ = self.flush_locked(&guard);
        }
    }

    /// Strict dynamic-suffix injection (<= 3 rules, <= 420 chars / <= 120 tokens).
    /// NEVER place inside the static 2,048-token system prefix.
    pub fn prompt_suffix(&self, dir_prefix: &str) -> String {
        let guard = self.rules.lock().unwrap();
        let mut candidates: Vec<&LearnRule> = guard
            .iter()
            .filter(|r| r.dir_prefix == dir_prefix || r.hits >= 2)
            .collect();
        candidates.sort_by(|a, b| b.hits.cmp(&a.hits));

        if candidates.is_empty() {
            return String::new();
        }

        let mut out = String::from("\nLearned repo rules from prior waves (obey strictly):\n");
        for rule in candidates.into_iter().take(3) {
            let line = format!("- [{}] {}\n", rule.gate, rule.advice);
            if out.len() + line.len() > 420 {
                break;
            }
            out.push_str(&line);
        }
        out
    }

    pub fn stats(&self) -> (usize, u32) {
        let guard = self.rules.lock().unwrap();
        let helpful = guard.iter().map(|r| r.helpful).sum();
        (guard.len(), helpful)
    }

    fn flush_locked(&self, rules: &[LearnRule]) -> Result<()> {
        let mut buf = String::new();
        for r in rules {
            buf.push_str(&serde_json::to_string(r)?);
            buf.push('\n');
        }
        atomic_replace_file(&self.path, buf.as_bytes())
    }
}

fn summarize_diagnostic(gate: &str, raw: &str) -> (String, String) {
    let lower = raw.to_lowercase();
    if lower.contains("abi") {
        return (
            "abi_pin".into(),
            "Preserve exact public def/class signatures and argument names.".into(),
        );
    }
    if lower.contains("b006") || lower.contains("mutable") {
        return (
            "ruff_b006".into(),
            "Replace mutable default args ([]/{}) with None and initialize inside body.".into(),
        );
    }
    if lower.contains("f401") || lower.contains("unused import") {
        return (
            "ruff_f401".into(),
            "Remove every unused import and unused local variable.".into(),
        );
    }
    if lower.contains("e501") || lower.contains("line too long") || lower.contains("dense_line") {
        return (
            "line_len".into(),
            "Keep all Python lines <= 88 columns; break chained calls cleanly.".into(),
        );
    }
    if lower.contains("deep_nesting") {
        return (
            "deep_nesting".into(),
            "Flatten nested if/for blocks to <= 2 levels using early-return guard clauses.".into(),
        );
    }
    if lower.contains("duplicate_block") || lower.contains("repetitive_structure") {
        return (
            "dedup".into(),
            "Extract repeated validation blocks into a helper and use dict dispatch over elif ladders.".into(),
        );
    }
    if lower.contains("ty:") || gate == "gate3_ty" {
        return (
            "ty_types".into(),
            "Add explicit PEP 484 parameter and return types; narrow Optional before indexing or arithmetic.".into(),
        );
    }
    let clean: String = raw
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("fix failing check")
        .trim()
        .chars()
        .take(100)
        .collect();
    (format!("{gate}_generic"), clean)
}
