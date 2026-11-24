use crate::types::LlmCallMetrics;
use axum::http::HeaderValue;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const MAX_KEY_LEN: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Short,
    Long,
    None,
}

impl CacheMode {
    pub fn parse_mode(explicit: Option<&str>) -> Self {
        let raw = explicit
            .map(String::from)
            .or_else(|| std::env::var("REFIT_CACHE_RETENTION").ok())
            .unwrap_or_else(|| "short".into());
        match raw.trim().to_ascii_lowercase().as_str() {
            "long" | "1h" | "24h" => Self::Long,
            "none" | "off" => Self::None,
            _ => Self::Short,
        }
    }

    pub fn ttl_ms(self, upstream_base_url: &str) -> u64 {
        let u = upstream_base_url.to_ascii_lowercase();
        match self {
            Self::None => 0,
            Self::Long if long_ttl(&u) => 3_600_000,
            Self::Long if u.contains("openai.com") => 1_800_000,
            _ => 300_000,
        }
    }
}

fn long_ttl(url: &str) -> bool {
    url.contains("moonshot.ai") || url.contains("kimi.ai") || url.contains("anthropic.com")
}

#[derive(Clone)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_base_url: String,
    pub api_key: String,
    pub model_override: String,
    pub ui_index_path: PathBuf,
    pub static_prefix_override: Option<String>,
    pub cache_retention: CacheMode,
    pub keep_warm: bool,
    pub auth_token: Option<String>,
    pub lang_prefixes: BTreeMap<String, String>,
    pub preserve_tools: bool,
}

impl ProxyConfig {
    pub fn new(listen: String, upstream: String, api_key: String, model: String) -> Self {
        Self {
            listen_addr: listen,
            upstream_base_url: upstream,
            api_key,
            model_override: model,
            ui_index_path: PathBuf::from("ui/index.html"),
            static_prefix_override: None,
            cache_retention: CacheMode::Short,
            keep_warm: false,
            auth_token: None,
            lang_prefixes: BTreeMap::new(),
            preserve_tools: true,
        }
    }
}

pub fn keepwarm_due(
    idle_ms: u64,
    since_last_ping_ms: u64,
    ttl_ms: u64,
    in_flight: u64,
    pings_sent: u32,
    max_pings: u32,
    disabled: bool,
) -> bool {
    if disabled || ttl_ms < 60_000 || in_flight > 0 || pings_sent >= max_pings {
        return false;
    }
    let trigger = (ttl_ms * 80) / 100;
    let expiry = ttl_ms.saturating_sub(15_000);
    idle_ms >= trigger && idle_ms < expiry && since_last_ping_ms >= trigger
}

/// Infer the retry attempt number from `[attempt=N]` tags or turn count.
pub fn infer_attempt(payload: &Value) -> u32 {
    let Some(msgs) = payload.get("messages").and_then(Value::as_array) else { return 1 };

    for msg in msgs.iter().rev() {
        let Some(text) = msg.get("content").and_then(Value::as_str) else { continue };
        if let Some(n) = parse_tag(text) {
            return n.max(1);
        }
    }
    if msgs.len() >= 4 { ((msgs.len() / 2) as u32).max(2) } else { 1 }
}

fn parse_tag(text: &str) -> Option<u32> {
    let pos = text.find("[attempt=")?;
    let tail = &text[pos + 9..];
    let raw = tail[..tail.find(']')?].trim();
    raw.parse::<u32>().ok()
}

pub fn guard_payload(payload: &mut Value, cfg: &ProxyConfig, module_id: &str, attempt: u32) {
    guard_lang(payload, cfg, module_id, attempt, None);
}

pub fn guard_lang(
    payload: &mut Value,
    cfg: &ProxyConfig,
    module_id: &str,
    attempt: u32,
    lang_hint: Option<&str>,
) {
    let Some(obj) = payload.as_object_mut() else { return };
    guard_core(obj, cfg);
    let has_tools = sort_tools(obj, cfg.preserve_tools);
    guard_module(obj, cfg, module_id, lang_hint, has_tools);
    guard_cache(obj, cfg, module_id, attempt);
}

fn guard_core(obj: &mut Map<String, Value>, cfg: &ProxyConfig) {
    if !cfg.model_override.is_empty() {
        obj.insert("model".into(), Value::String(cfg.model_override.clone()));
    }

    let req = obj.get("reasoning_effort").and_then(Value::as_str);
    let effort = match req.unwrap_or("low") {
        "low" | "medium" | "high" => req.unwrap_or("low"),
        _ => "low",
    };

    obj.insert("reasoning_effort".into(), Value::String(effort.into()));
    obj.insert("reasoning_format".into(), Value::String("parsed".into()));
    obj.insert("temperature".into(), Value::from(1.0));
    obj.remove("disable_reasoning");
    obj.remove("top_p");
    obj.remove("clear_thinking");

    if let Some(mt) = obj.remove("max_tokens") {
        obj.entry("max_completion_tokens").or_insert(mt);
    }
    let max_comp = obj
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(3200)
        .clamp(1, 6144);
    obj.insert("max_completion_tokens".into(), Value::from(max_comp));

    if obj.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        obj.insert("stream_options".into(), json!({"include_usage": true}));
    }
}

fn sort_tools(obj: &mut Map<String, Value>, preserve: bool) -> bool {
    let Some(tools) = obj.get_mut("tools").and_then(Value::as_array_mut) else { return false };
    let non_empty = !tools.is_empty();
    if preserve {
        tools.sort_by(|a, b| {
            let na = a["function"]["name"].as_str().unwrap_or("");
            let nb = b["function"]["name"].as_str().unwrap_or("");
            na.cmp(nb)
        });
    }
    non_empty
}

fn guard_module(
    obj: &mut Map<String, Value>,
    cfg: &ProxyConfig,
    module_id: &str,
    lang_hint: Option<&str>,
    has_tools: bool,
) {
    if module_id.is_empty() || module_id == "probe" {
        return;
    }
    if !cfg.preserve_tools {
        obj.remove("tools");
        obj.remove("tool_choice");
    }
    if cfg.preserve_tools && has_tools {
        return;
    }

    let prefix = lang_hint
        .and_then(|l| cfg.lang_prefixes.get(l))
        .or(cfg.static_prefix_override.as_ref());
    let Some(static_sys) = prefix else { return };

    let Some(msgs) = obj.get_mut("messages").and_then(Value::as_array_mut) else { return };
    let Some(first) = msgs.first_mut() else { return };
    if first.get("role").and_then(Value::as_str) == Some("system") {
        first["content"] = Value::String(static_sys.clone());
    }
}

fn guard_cache(obj: &mut Map<String, Value>, cfg: &ProxyConfig, module_id: &str, attempt: u32) {
    if attempt <= 1 || cfg.cache_retention == CacheMode::None {
        obj.remove("prompt_cache_key");
    } else if let Some(k) = obj.get("prompt_cache_key").and_then(Value::as_str) {
        if k.len() > MAX_KEY_LEN {
            let clamped: String = k.chars().take(MAX_KEY_LEN).collect();
            obj.insert("prompt_cache_key".into(), Value::String(clamped));
        }
    } else if !module_id.is_empty() && module_id != "probe" {
        obj.insert("prompt_cache_key".into(), Value::String(format!("refit:auto:m:{module_id}")));
    }

    if cfg.cache_retention != CacheMode::Long {
        return;
    }

    let up = cfg.upstream_base_url.to_ascii_lowercase();
    if up.contains("moonshot.ai") || up.contains("kimi.ai") {
        obj.insert("prompt_cache_options".into(), json!({"mode": "implicit", "ttl": "1h"}));
    } else if up.contains("openai.com") {
        obj.insert("prompt_cache_retention".into(), Value::String("24h".into()));
    }
}

pub(super) fn sys_text(payload: &Value) -> &str {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

pub(super) fn find_module(payload: &Value) -> String {
    let Some(msgs) = payload.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let needle = "Refactor unit `";

    for msg in msgs {
        let Some(text) = msg.get("content").and_then(Value::as_str) else { continue };
        let Some(pos) = text.find(needle) else { continue };
        let rest = &text[pos + needle.len()..];
        if let Some(end) = rest.find('`') {
            return rest[..end].to_string();
        }
    }
    String::new()
}

pub(super) fn localip(origin: &HeaderValue) -> bool {
    let Ok(s) = origin.to_str() else { return false };
    let lower = s.to_ascii_lowercase();
    let Some(rest) = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"))
    else {
        return false;
    };
    if rest.bytes().any(|b| matches!(b, b'/' | b'@' | b'?' | b'#')) {
        return false;
    }

    for host in ["127.0.0.1", "localhost", "[::1]"] {
        let Some(after) = rest.strip_prefix(host) else { continue };
        if after.is_empty() {
            return true;
        }
        if let Some(port) = after.strip_prefix(':') {
            return !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit());
        }
    }
    false
}

pub(super) fn header_u64(headers: &reqwest::header::HeaderMap, name: &str) -> Option<u64> {
    let raw = headers.get(name)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok()
}

pub(super) fn parse_metrics(
    json: &Value,
    module_id: &str,
    attempt: u32,
    status: u16,
    wall_ms: u64,
    prefix_drift: bool,
) -> LlmCallMetrics {
    let (usage, ti) = (&json["usage"], &json["time_info"]);
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    let reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0);
    let uncached_tokens = prompt_tokens.saturating_sub(cached_tokens) + completion_tokens;

    let queue_ms = ti["queue_time"].as_f64().unwrap_or(0.0) * 1000.0;
    let prompt_ms = ti["prompt_time"].as_f64().unwrap_or(0.0) * 1000.0;
    let comp_s = ti["completion_time"].as_f64().unwrap_or(0.0);
    let decode_tps = if comp_s > 0.0005 && completion_tokens > 0 {
        (completion_tokens as f64) / comp_s
    } else {
        0.0
    };

    LlmCallMetrics {
        module_id: module_id.to_string(),
        attempt,
        http_status: status,
        wall_ms,
        queue_ms,
        prompt_ms,
        completion_ms: comp_s * 1000.0,
        decode_tps,
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        reasoning_tokens,
        uncached_tokens,
        prefix_drift,
        ..LlmCallMetrics::default()
    }
}
