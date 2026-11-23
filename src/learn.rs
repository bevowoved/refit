use crate::platform::write_atomic;
use crate::types::LearnRule;
use anyhow::Result;
use parking_lot::Mutex;
use std::fs;
use std::path::{Path, PathBuf};

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
                        // Drop legacy file-specific *_generic poison entries on load.
                        if !rule.key.ends_with("_generic") {
                            loaded.push(rule);
                        }
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
        // Never persist raw file-specific error strings or *_generic catch-alls.
        let Some((key, advice)) = classify_diag(gate, raw_diagnostic) else {
            return;
        };
        let snapshot = {
            let mut guard = self.rules.lock();
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
                    guard.sort_by(|a, b| {
                        let score_a = a.hits.saturating_add(a.helpful * 2);
                        let score_b = b.hits.saturating_add(b.helpful * 2);
                        score_b.cmp(&score_a).then_with(|| a.key.cmp(&b.key))
                    });
                    guard.truncate(MAX_RULES);
                }
            }
            guard.clone()
        };
        let _ = self.flush_snapshot(&snapshot);
    }

    /// Increment `helpful` only for rules injected into the committed unit.
    pub fn credit_keys(&self, dir_prefix: &str, applied_keys: &[String]) {
        if applied_keys.is_empty() {
            return;
        }
        let snapshot = {
            let mut guard = self.rules.lock();
            let mut changed = false;
            for k in applied_keys {
                if let Some(r) = guard
                    .iter_mut()
                    .find(|r| r.key == *k && r.dir_prefix == dir_prefix)
                {
                    r.helpful += 1;
                    changed = true;
                } else if let Some(r) = guard
                    .iter_mut()
                    .filter(|r| r.key == *k && (r.dir_prefix == "*" || r.hits >= 2))
                    .max_by_key(|r| r.hits)
                {
                    r.helpful += 1;
                    changed = true;
                }
            }
            if changed { Some(guard.clone()) } else { None }
        };
        if let Some(snap) = snapshot {
            let _ = self.flush_snapshot(&snap);
        }
    }

    #[cfg(test)]
    pub fn prompt_suffix(&self, dir_prefix: &str) -> String {
        self.rule_suffix(dir_prefix).0
    }

    /// Returns `(formatted_suffix, applied_rule_keys)` so callers credit only injected rules.
    pub fn rule_suffix(&self, dir_prefix: &str) -> (String, Vec<String>) {
        let guard = self.rules.lock();
        let mut candidates: Vec<&LearnRule> = guard
            .iter()
            .filter(|r| !r.key.ends_with("_generic") && (r.dir_prefix == dir_prefix || r.hits >= 2))
            .collect();
        candidates.sort_by(|a, b| b.hits.cmp(&a.hits).then_with(|| a.key.cmp(&b.key)));

        if candidates.is_empty() {
            return (String::new(), Vec::new());
        }

        let mut out = String::from("\nLearned repo rules from prior waves (obey strictly):\n");
        let mut keys = Vec::new();
        for rule in candidates.into_iter().take(3) {
            let line = format!("- [{}] {}\n", rule.gate, rule.advice);
            if out.len() + line.len() > 420 {
                break;
            }
            out.push_str(&line);
            keys.push(rule.key.clone());
        }
        (out, keys)
    }

    pub fn stats(&self) -> (usize, u32) {
        let guard = self.rules.lock();
        let helpful = guard.iter().map(|r| r.helpful).sum();
        (guard.len(), helpful)
    }

    fn flush_snapshot(&self, rules: &[LearnRule]) -> Result<()> {
        let mut buf = String::new();
        for r in rules {
            buf.push_str(&serde_json::to_string(r)?);
            buf.push('\n');
        }
        write_atomic(&self.path, buf.as_bytes())
    }
}

/// Map a raw diagnostic to a canonical cross-wave rule, or return `None` for one-off noise.
fn classify_diag(gate: &str, raw: &str) -> Option<(String, String)> {
    let lower = raw.to_lowercase();
    if lower.contains("abi") {
        return Some((
            "abi_pin".into(),
            "Preserve exact exported signatures and argument names unless allow_abi_change is enabled.".into(),
        ));
    }
    if lower.contains("search block did not match") || lower.contains("ambiguous search") || gate == "asrf_apply" {
        return Some((
            "asrf_exact_context".into(),
            "Copy <<<SEARCH>>> lines verbatim with 2-3 surrounding context lines so patches match uniquely.".into(),
        ));
    }
    if lower.contains("b006") || lower.contains("mutable default") {
        return Some((
            "ruff_b006".into(),
            "Replace mutable default args ([]/{}) with None and initialize inside body.".into(),
        ));
    }
    if lower.contains("f401") || lower.contains("unused import") {
        return Some((
            "ruff_f401".into(),
            "Remove every unused import cleanly.".into(),
        ));
    }
    if lower.contains("f841") || lower.contains("assigned to but never used") {
        return Some((
            "ruff_f841".into(),
            "Remove unused local variables or bind intentionally with `_`.".into(),
        ));
    }
    if lower.contains("e501") || lower.contains("e702") || lower.contains("line too long") || lower.contains("dense_line") {
        return Some((
            "line_len".into(),
            "Keep lines concise and split packed statements/chained calls cleanly across lines.".into(),
        ));
    }
    if lower.contains("deep_nesting") {
        return Some((
            "deep_nesting".into(),
            "Flatten nested if/for/match pyramids using early-return guard clauses and small helpers.".into(),
        ));
    }
    if lower.contains("duplicate_block") || lower.contains("repetitive_structure") {
        return Some((
            "dedup".into(),
            "Extract repeated blocks into a shared helper and prefer lookup tables over branch ladders.".into(),
        ));
    }
    if lower.contains("cramped_layout") {
        return Some((
            "cramped_layout".into(),
            "Separate logical phases and top-level definitions with clean blank lines.".into(),
        ));
    }
    if lower.contains("ty:") || gate == "gate3_ty" {
        return Some((
            "ty_types".into(),
            "Add explicit type annotations and narrow Optional/union values before indexing or arithmetic.".into(),
        ));
    }
    if lower.contains("builtin_no_secrets") || lower.contains("secret literal") || gate == "gate3_5_invariants" {
        return Some((
            "invariants_security".into(),
            "Never hardcode credentials or violate .refit/invariants.yml architectural boundaries.".into(),
        ));
    }
    if lower.contains("assert") || lower.contains("pytest") || gate == "gate1_tests" {
        return Some((
            "test_boundary".into(),
            "Trace exact boundary conditions (<= vs <, clamping, None guards, rounding) against failing tests.".into(),
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::now_ms;

    #[test]
    fn probe_20_learn_budget() {
        let dir = std::env::temp_dir().join(format!("refit_learn_b_{}_{}", std::process::id(), now_ms()));
        let store = LearnStore::open(&dir).unwrap();
        store.record_failure("gate3_ruff", "pkg/core", "B006 mutable default");
        store.record_failure("gate3_ty", "pkg/core", "ty: invalid optional");
        store.record_failure("gate2_hygiene", "pkg/core", "deep_nesting L14");
        store.record_failure("gate0_abi", "pkg/core", "AbiPinViolation");

        let s1 = store.prompt_suffix("pkg/core");
        let s2 = store.prompt_suffix("pkg/core");
        assert_eq!(s1, s2);
        assert!(s1.len() <= 420);
        assert!(s1.lines().filter(|l| l.starts_with("- [")).count() <= 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_21_learn_prune() {
        let dir = std::env::temp_dir().join(format!("refit_learn_p_{}_{}", std::process::id(), now_ms()));
        let store = LearnStore::open(&dir).unwrap();
        for i in 0..24 {
            store.record_failure("gate3_ruff", &format!("dir_{i}"), "B006 mutable default");
        }
        assert_eq!(store.stats().0, MAX_RULES);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_89_learn_keys() {
        let dir = std::env::temp_dir().join(format!("refit_learn89_{}_{}", std::process::id(), now_ms()));
        let store = LearnStore::open(&dir).unwrap();
        // Raw file-specific ASRF error must generalize without leaking `mod_54.py`.
        store.record_failure(
            "asrf_apply",
            "pkg/api",
            "asrf_apply: Hunk #0 SEARCH block did not match `pkg/api/mod_54.py` (first line: `def evaluate_record_54`)",
        );
        // Unrecognized one-off error must be ignored rather than creating *_generic poison.
        store.record_failure("weird_gate", "pkg/api", "one-off transient socket hiccup on foo.py");
        store.record_failure("gate3_ruff", "pkg/api", "B006 mutable default");

        let (suffix, keys) = store.rule_suffix("pkg/api");
        assert!(!suffix.contains("mod_54.py"), "leaked filename into advice: {suffix}");
        assert!(!suffix.contains("weird_gate"), "persisted unrecognized generic error: {suffix}");
        assert!(keys.contains(&"asrf_exact_context".to_string()));
        assert_eq!(store.stats().0, 2);

        // Credit only `asrf_exact_context`, NOT `ruff_b006`.
        store.credit_keys("pkg/api", &["asrf_exact_context".to_string()]);
        assert_eq!(store.stats().1, 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
