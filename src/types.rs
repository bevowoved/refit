use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Cross-platform OS-measured execution result for a sandboxed child process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    pub exit_code: i32,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
    pub wall_ms: u64,
    pub cpu_ms: u64,
    pub peak_rss_bytes: u64,
}

/// Self-process OS resource sample.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SelfMetrics {
    pub rss_bytes: u64,
    pub cpu_ms: u64,
}

/// Single geometric squiggle returned by codehygiene v0.7.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Squiggle {
    pub line: usize,
    pub kind: String,
    pub probability: f64,
    #[serde(default)]
    pub message: String,
}

/// Maintainability score returned by Gate 2 (brianbell-x/codehygiene v0.7.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HygieneResult {
    pub score: f64,
    #[serde(default)]
    pub squiggles: Vec<Squiggle>,
    #[serde(default)]
    pub cached: bool,
    #[serde(default)]
    pub latency_us: u64,
}

/// Atomic refactoring unit (single file or Tarjan SCC multi-file cycle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleUnit {
    pub id: String,
    pub files: Vec<String>,
    pub layer: usize,
    pub wave: usize,
    pub deps: Vec<String>,
    pub dependents: Vec<String>,
    pub test_files: Vec<String>,
    pub loc: usize,
    pub base_blake3: BTreeMap<String, String>,
    pub abi_hash: BTreeMap<String, String>,
    pub baseline_hygiene: f64,
    pub allow_abi_change: bool,
}

/// Full repository dependency graph plan built from stem + AST analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagPlan {
    pub repo_root: String,
    pub host_os: String,
    pub stem_used: bool,
    pub stem_index_ms: u64,
    pub total_files: usize,
    pub total_loc: usize,
    pub scc_count: usize,
    pub wave_count: usize,
    pub units: Vec<ModuleUnit>,
}

/// Parsed raw unescaped ASRF patch hunk from assistant.message.content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrfHunk {
    pub path: String,
    pub search: String,
    pub replace: String,
}

/// Verdict for all 4 local verification stages (Gates 0, 1, 2, 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateVerdict {
    pub passed: bool,
    pub failed_gate: Option<String>,
    pub gate0_syntax_ok: bool,
    pub gate0_abi_ok: bool,
    pub gate0_ms: u64,
    pub gate1_tests_passed: usize,
    pub gate1_tests_failed: usize,
    pub gate1_ms: u64,
    pub gate1_cpu_ms: u64,
    pub gate1_peak_rss_bytes: u64,
    pub gate2_score_before: f64,
    pub gate2_score_after: f64,
    pub gate2_ms: u64,
    pub gate3_ruff_errors: usize,
    pub gate3_ty_errors: usize,
    pub gate3_ms: u64,
    pub diagnostics: Vec<String>,
    /// Final file bytes (may reflect ruff --fix formatting applied before commit).
    #[serde(skip)]
    pub final_contents: BTreeMap<String, Vec<u8>>,
}

/// File entry recorded inside an ACID WAL transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalFileEntry {
    pub path: String,
    pub old_blake3: String,
    pub new_blake3: String,
}

/// Append-only ACID WAL record in .refit/wal.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WalRecord {
    RunStart {
        run_id: String,
        ts_ms: u64,
        repo_root: String,
    },
    Prepared {
        tx_id: String,
        module_id: String,
        ts_ms: u64,
        files: Vec<WalFileEntry>,
    },
    Committed {
        tx_id: String,
        module_id: String,
        head_rev: u64,
        ts_ms: u64,
        files: Vec<WalFileEntry>,
    },
    Rollback {
        tx_id: String,
        module_id: String,
        gate: String,
        reason: String,
        ts_ms: u64,
    },
}

/// Learned operational rule stored in .refit/learn.jsonl (injected only in dynamic suffix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnRule {
    pub key: String,
    pub gate: String,
    pub dir_prefix: String,
    pub advice: String,
    pub hits: u32,
    pub helpful: u32,
}

/// Live Cerebras / provider response timing and rate-limit snapshot tapped by :18790 proxy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmCallMetrics {
    pub module_id: String,
    pub attempt: u32,
    pub http_status: u16,
    pub wall_ms: u64,
    pub queue_ms: f64,
    pub prompt_ms: f64,
    pub completion_ms: f64,
    pub decode_tps: f64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,
    pub uncached_tokens: u64,
    pub prefix_drift: bool,
    pub limit_tpm_minute: Option<u64>,
    pub remaining_tpm_minute: Option<u64>,
    pub limit_rpm_minute: Option<u64>,
}

/// Live proxy pacer and circuit-breaker telemetry snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProxySnapshot {
    pub circuit_state: String,
    pub uncached_tpm_rolling: u64,
    pub uncached_tpm_cap: u64,
    pub total_tpm_rolling: u64,
    pub total_tpm_cap: u64,
    pub rpm_rolling: u64,
    pub rpm_cap: u64,
    pub http_429_count: u64,
    pub http_5xx_count: u64,
    pub circuit_pauses: u64,
    pub prefix_drift_events: u64,
    pub total_prompt_tokens: u64,
    pub total_cached_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_uncached_tokens: u64,
}

/// 5-run alternating OS CPU/RSS benchmark summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchComparison {
    pub pairs: usize,
    pub before_cpu_ms_median: f64,
    pub after_cpu_ms_median: f64,
    pub cpu_ratio: f64,
    pub before_rss_mb_median: f64,
    pub after_rss_mb_median: f64,
    pub rss_ratio: f64,
    pub passed_anti_regression: bool,
}

/// Final Phase 0 vs Phase 2 run summary embedded in run_end and UI scorecard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub host_os: String,
    pub model: String,
    pub provider_label: String,
    pub total_files: usize,
    pub total_loc: usize,
    pub total_units: usize,
    pub committed_units: usize,
    pub skipped_resumed_units: usize,
    pub total_wall_s: f64,
    pub avg_queue_s: f64,
    pub median_decode_tps: f64,
    pub hygiene_before_median: f64,
    pub hygiene_before_p10: f64,
    pub hygiene_after_median: f64,
    pub hygiene_after_p10: f64,
    pub tests_total: usize,
    pub tests_before_fail: usize,
    pub tests_after_fail: usize,
    pub gate3_before_ruff: usize,
    pub gate3_before_ty: usize,
    pub gate3_after_ruff: usize,
    pub gate3_after_ty: usize,
    pub cas_conflicts: u64,
    pub rollbacks_g0: u64,
    pub rollbacks_g1: u64,
    pub rollbacks_g2: u64,
    pub rollbacks_g3: u64,
    pub retries_total: u64,
    pub http_429_count: u64,
    pub circuit_pauses: u64,
    pub prefix_drift_events: u64,
    pub prefix_cache_hit_pct: f64,
    pub uncached_tpm_cap: u64,
    pub total_uncached_tokens: u64,
    pub total_cached_tokens: u64,
    pub engine_rss_mb: f64,
    pub bench: Option<BenchComparison>,
}

/// Unified lossless JSONL & 30 Hz SSE telemetry event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub seq: u64,
    pub ts_ms: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wave: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncached_tpm: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<RunSummary>,
}
