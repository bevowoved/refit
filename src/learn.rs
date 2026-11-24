use crate::platform::write_atomic;
use crate::types::LearnRule;
use anyhow::Result;
use parking_lot::Mutex;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_RULES: usize = 16;

struct DiagRule {
    gate: &'static str,
    needles: &'static [&'static str],
    key: &'static str,
    advice: &'static str,
}

const DIAG_RULES: &[DiagRule] = &[
    DiagRule {
        gate: "",
        needles: &["abi"],
        key: "abi_pin",
        advice: "Preserve exact exported signatures unless allow_abi_change is enabled.",
    },
    DiagRule {
        gate: "asrf_apply",
        needles: &["search block did not match", "ambiguous search"],
        key: "asrf_exact_context",
        advice: "Copy <<<SEARCH>>> lines verbatim with 2-3 surrounding context lines.",
    },
    DiagRule {
        gate: "",
        needles: &["b006", "mutable default"],
        key: "ruff_b006",
        advice: "Replace mutable default args ([]/{}) with None and init inside body.",
    },
    DiagRule {
        gate: "",
        needles: &["f401", "unused import"],
        key: "ruff_f401",
        advice: "Remove every unused import cleanly.",
    },
    DiagRule {
        gate: "",
        needles: &["f841", "assigned to but never used"],
        key: "ruff_f841",
        advice: "Remove unused local variables or bind intentionally with `_`.",
    },
    DiagRule {
        gate: "",
        needles: &["e501", "e702", "line too long", "dense_line"],
        key: "line_len",
        advice: "Keep lines concise and split packed statements across lines.",
    },
    DiagRule {
        gate: "",
        needles: &["deep_nesting"],
        key: "deep_nesting",
        advice: "Flatten nested pyramids using guard clauses and small helpers.",
    },
    DiagRule {
        gate: "",
        needles: &["duplicate_block", "repetitive_structure"],
        key: "dedup",
        advice: "Extract repeated blocks into helpers and prefer lookup tables.",
    },
    DiagRule {
        gate: "",
        needles: &["cramped_layout"],
        key: "cramped_layout",
        advice: "Separate logical phases and top-level definitions with blank lines.",
    },
    DiagRule {
        gate: "gate3_ty",
        needles: &["ty:"],
        key: "ty_types",
        advice: "Add explicit type annotations and narrow Optional values before use.",
    },
    DiagRule {
        gate: "gate3_5_invariants",
        needles: &["builtin_no_secrets", "secret literal"],
        key: "invariants_security",
        advice: "Never hardcode credentials or violate .refit/invariants.yml boundaries.",
    },
    DiagRule {
        gate: "gate1_tests",
        needles: &["assert", "pytest"],
        key: "test_boundary",
        advice: "Trace exact boundary conditions (<= vs <, clamping, None guards) carefully.",
    },
];

pub struct LearnStore {
    path: PathBuf,
    rules: Mutex<Vec<LearnRule>>,
}

impl LearnStore {
    pub fn open(repo_root: &Path) -> Result<Self> {
        let dir = repo_root.join(".refit");
        fs::create_dir_all(&dir)?;
        let path = dir.join("learn.jsonl");

        let loaded = fs::read_to_string(&path)
            .ok()
            .map(|raw| {
                raw.lines()
                    .filter_map(|l| serde_json::from_str::<LearnRule>(l.trim()).ok())
                    .filter(|r| !r.key.ends_with("_generic"))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            path,
            rules: Mutex::new(loaded),
        })
    }

    pub fn record_failure(&self, gate: &str, dir_prefix: &str, raw_diagnostic: &str) {
        let Some((key, advice)) = classify_diag(gate, raw_diagnostic) else {
            return;
        };

        let snapshot = {
            let mut guard = self.rules.lock();
            if let Some(rule) = guard
                .iter_mut()
                .find(|r| r.key == key && r.dir_prefix == dir_prefix)
            {
                rule.hits += 1;
            } else {
                guard.push(LearnRule {
                    key,
                    gate: gate.to_string(),
                    dir_prefix: dir_prefix.to_string(),
                    advice,
                    hits: 1,
                    helpful: 0,
                });
                prune_rules(&mut guard);
            }
            guard.clone()
        };

        let _ = self.flush_snapshot(&snapshot);
    }

    pub fn credit_keys(&self, dir_prefix: &str, applied_keys: &[String]) {
        if applied_keys.is_empty() {
            return;
        }

        let snapshot = {
            let mut guard = self.rules.lock();
            let mut changed = false;
            for key in applied_keys {
                if credit_one(&mut guard, dir_prefix, key) {
                    changed = true;
                }
            }
            changed.then(|| guard.clone())
        };

        if let Some(snap) = snapshot {
            let _ = self.flush_snapshot(&snap);
        }
    }

    pub fn rule_suffix(&self, dir_prefix: &str) -> (String, Vec<String>) {
        let guard = self.rules.lock();
        let mut cands: Vec<&LearnRule> = guard
            .iter()
            .filter(|r| !r.key.ends_with("_generic"))
            .filter(|r| r.dir_prefix == dir_prefix || r.hits >= 2)
            .collect();
        cands.sort_by(|a, b| b.hits.cmp(&a.hits).then_with(|| a.key.cmp(&b.key)));

        if cands.is_empty() {
            return (String::new(), Vec::new());
        }

        let mut out = String::from("\nLearned repo rules from prior waves (obey strictly):\n");
        let mut keys = Vec::new();
        for rule in cands.into_iter().take(3) {
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

fn prune_rules(rules: &mut Vec<LearnRule>) {
    if rules.len() <= MAX_RULES {
        return;
    }

    rules.sort_by(|a, b| {
        let sa = a.hits.saturating_add(a.helpful * 2);
        let sb = b.hits.saturating_add(b.helpful * 2);
        sb.cmp(&sa).then_with(|| a.key.cmp(&b.key))
    });
    rules.truncate(MAX_RULES);
}

fn credit_one(rules: &mut [LearnRule], dir_prefix: &str, key: &str) -> bool {
    if let Some(r) = rules
        .iter_mut()
        .find(|r| r.key == key && r.dir_prefix == dir_prefix)
    {
        r.helpful += 1;
        return true;
    }

    if let Some(r) = rules
        .iter_mut()
        .filter(|r| r.key == key && (r.dir_prefix == "*" || r.hits >= 2))
        .max_by_key(|r| r.hits)
    {
        r.helpful += 1;
        return true;
    }
    false
}

fn classify_diag(gate: &str, raw: &str) -> Option<(String, String)> {
    let lower = raw.to_lowercase();
    DIAG_RULES
        .iter()
        .find(|r| {
            (!r.gate.is_empty() && gate == r.gate)
                || r.needles.iter().any(|n| lower.contains(n))
        })
        .map(|r| (r.key.to_string(), r.advice.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    fn open_store(tag: &str) -> (TempRepo, LearnStore) {
        let dir = TempRepo::new(tag);
        let store = LearnStore::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn test_learn_budget_keys_and_credit() {
        let (_dir, store) = open_store("learn_keys");
        for (gate, msg) in [
            ("gate3_ruff", "B006 mutable default"),
            ("gate3_ty", "ty: invalid optional"),
            ("gate2_hygiene", "deep_nesting L14"),
            ("gate0_abi", "AbiPinViolation"),
        ] {
            store.record_failure(gate, "pkg/core", msg);
        }
        let s1 = store.rule_suffix("pkg/core").0;
        assert_eq!(s1, store.rule_suffix("pkg/core").0);
        assert!(s1.len() <= 420 && s1.lines().filter(|l| l.starts_with("- [")).count() <= 3);

        let (_dir2, s2) = open_store("learn_api");
        let err = "asrf_apply: Hunk #0 SEARCH block did not match `pkg/api/mod_54.py` (first line: `def evaluate_record_54`)";
        for (gate, msg) in [
            ("asrf_apply", err),
            ("weird_gate", "transient socket hiccup on foo.py"),
            ("gate3_ruff", "B006 mutable default"),
        ] {
            s2.record_failure(gate, "pkg/api", msg);
        }
        let (suffix, keys) = s2.rule_suffix("pkg/api");
        assert!(!suffix.contains("mod_54.py") && !suffix.contains("weird_gate"));
        assert!(keys.contains(&"asrf_exact_context".to_string()) && s2.stats().0 == 2);

        s2.credit_keys("pkg/api", &["asrf_exact_context".to_string()]);
        assert_eq!(s2.stats().1, 1);
    }

    #[test]
    fn test_learn_max_rules_prune() {
        let (_dir, store) = open_store("learn_prune");
        for i in 0..24 {
            store.record_failure("gate3_ruff", &format!("dir_{i}"), "B006 mutable default");
        }
        assert_eq!(store.stats().0, MAX_RULES);
    }
}
