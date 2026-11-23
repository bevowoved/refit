use crate::mvcc::MvccStore;
use crate::types::{HygieneResult, Squiggle};
use dashmap::DashMap;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

struct WarmWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Deserialize)]
struct WorkerReply {
    score: f64,
    #[serde(default)]
    squiggles: Vec<Squiggle>,
    #[serde(default)]
    latency_us: u64,
}

pub struct HygieneScorer {
    worker_script: PathBuf,
    worker: Mutex<Option<WarmWorker>>,
    cache: DashMap<String, HygieneResult>,
    req_id: AtomicU64,
}

impl HygieneScorer {
    pub fn new(worker_script: PathBuf) -> Self {
        Self {
            worker_script,
            worker: Mutex::new(None),
            cache: DashMap::new(),
            req_id: AtomicU64::new(1),
        }
    }

    pub fn resolve_default_worker(repo_root: &Path) -> PathBuf {
        let candidates = [
            repo_root.join("vendor/codehygiene/worker.mjs"),
            PathBuf::from("vendor/codehygiene/worker.mjs"),
            PathBuf::from("D:/cerebras/refit/vendor/codehygiene/worker.mjs"),
        ];
        for c in candidates {
            if c.exists() {
                return c;
            }
        }
        PathBuf::from("vendor/codehygiene/worker.mjs")
    }

    fn spawn_worker(&self) -> Option<WarmWorker> {
        if !self.worker_script.exists() {
            return None;
        }
        let weights = self
            .worker_script
            .parent()
            .map(|p| p.join("weights.npz"))
            .unwrap_or_else(|| PathBuf::from("weights.npz"));

        for runtime in ["bun", "node"] {
            let attempt = Command::new(runtime)
                .arg(&self.worker_script)
                .arg(&weights)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn();
            if let Ok(mut child) = attempt {
                if let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) {
                    return Some(WarmWorker {
                        _child: child,
                        stdin,
                        stdout: BufReader::new(stdout),
                    });
                }
            }
        }
        None
    }

    pub fn score_content(&self, _path: &str, content: &str) -> HygieneResult {
        let key = MvccStore::hash_bytes(content.as_bytes());
        if let Some(hit) = self.cache.get(&key) {
            let mut res = hit.clone();
            res.cached = true;
            res.latency_us = 1;
            return res;
        }

        let id = self.req_id.fetch_add(1, Ordering::Relaxed);
        let req_line = serde_json::json!({
            "id": id,
            "content": content,
            "threshold": 0.55
        })
        .to_string();

        let mut res_opt: Option<HygieneResult> = None;
        {
            let mut guard = self.worker.lock().unwrap();
            for _attempt in 0..2 {
                if guard.is_none() {
                    *guard = self.spawn_worker();
                }
                if let Some(w) = guard.as_mut() {
                    let write_ok = w
                        .stdin
                        .write_all(req_line.as_bytes())
                        .and_then(|_| w.stdin.write_all(b"\n"))
                        .and_then(|_| w.stdin.flush())
                        .is_ok();
                    if write_ok {
                        let mut reply_line = String::new();
                        if w.stdout.read_line(&mut reply_line).is_ok() && !reply_line.is_empty() {
                            if let Ok(rep) = serde_json::from_str::<WorkerReply>(&reply_line) {
                                res_opt = Some(HygieneResult {
                                    score: rep.score.clamp(0.0, 1.0),
                                    squiggles: rep.squiggles,
                                    cached: false,
                                    latency_us: rep.latency_us,
                                });
                                break;
                            }
                        }
                    }
                    // Restart crashed worker once
                    *guard = None;
                } else {
                    break;
                }
            }
        }

        let result = res_opt.unwrap_or_else(|| fallback_geometric_score(content));
        self.cache.insert(key, result.clone());
        result
    }

    pub fn score_repo(
        &self,
        repo_root: &Path,
        rel_paths: &[String],
    ) -> (f64, f64, HashMap<String, HygieneResult>) {
        let mut per_file = HashMap::with_capacity(rel_paths.len());
        let mut scores = Vec::with_capacity(rel_paths.len());
        for rel in rel_paths {
            let abs = repo_root.join(rel);
            let content = std::fs::read_to_string(&abs).unwrap_or_default();
            let res = self.score_content(rel, &content);
            scores.push(res.score);
            per_file.insert(rel.clone(), res);
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if scores.is_empty() {
            return (1.0, 1.0, per_file);
        }
        let median = scores[scores.len() / 2];
        let p10_idx = ((scores.len() as f64) * 0.10).floor() as usize;
        let p10 = scores[p10_idx.min(scores.len() - 1)];
        (median, p10, per_file)
    }
}

/// Deterministic fallback if Bun/Node worker is unavailable during unit tests.
fn fallback_geometric_score(content: &str) -> HygieneResult {
    let mut penalty = 0.0f64;
    let mut squiggles = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim_start();
        let indent = line.len().saturating_sub(trimmed.len());
        if indent >= 20 {
            penalty += 0.04;
            squiggles.push(Squiggle {
                line: idx + 1,
                kind: "deep_nesting".into(),
                probability: 0.78,
                message: "Deep nesting pyramid".into(),
            });
        }
        if line.len() > 100 {
            penalty += 0.03;
            squiggles.push(Squiggle {
                line: idx + 1,
                kind: "dense_line".into(),
                probability: 0.72,
                message: "Wide line over 100 cols".into(),
            });
        }
    }
    HygieneResult {
        score: (0.95 - penalty).clamp(0.10, 0.99),
        squiggles,
        cached: false,
        latency_us: 50,
    }
}
