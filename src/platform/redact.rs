use crate::lex::left_bound;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use std::sync::LazyLock;

const DB_SCHEMES: &[&str] = &[
    "postgres://", "postgresql://", "mysql://", "mongodb://",
    "mongodb+srv://", "redis://",
];

// (prefix, kind, redact_tag, detect_min, redact_min, upper_only)
const SECRET_SPECS: &[(&str, &str, &str, usize, usize, bool)] = &[
    ("csk-", "cerebras_api_key", "REDACTED_CSK_", 12, 8, false),
    ("keen_", "keenable_api_key", "REDACTED_KEEN_", 12, 8, false),
    ("sk-ant-", "anthropic_api_key", "REDACTED_ANTHROPIC_", 16, 12, false),
    ("sk-proj-", "openai_api_key", "REDACTED_OPENAI_", 16, 12, false),
    ("sk_live_", "stripe_live_key", "REDACTED_STRIPE_", 16, 12, false),
    ("rk_live_", "stripe_restricted_key", "REDACTED_STRIPE_", 16, 12, false),
    ("ghp_", "github_pat", "REDACTED_GITHUB_", 20, 16, false),
    ("gho_", "github_oauth", "REDACTED_GITHUB_", 20, 16, false),
    ("github_pat_", "github_fine_grained_pat", "REDACTED_GITHUB_", 20, 16, false),
    ("xoxb-", "slack_bot_token", "REDACTED_SLACK_", 16, 12, false),
    ("xoxp-", "slack_user_token", "REDACTED_SLACK_", 16, 12, false),
    ("xoxa-", "slack_app_token", "REDACTED_SLACK_", 16, 12, false),
    ("AKIA", "aws_access_key_id", "REDACTED_AWS_", 16, 16, true),
    ("ASIA", "aws_sts_key_id", "REDACTED_AWS_", 16, 16, true),
    ("sk-", "generic_sk_secret", "REDACTED_SK_", 24, 24, false),
];

const PEM_MARKER: &str = "PRIVATE KEY-----";
const DB_BASE: usize = SECRET_SPECS.len();
const PEM_IDX: usize = DB_BASE + DB_SCHEMES.len();

static SECRET_AC: LazyLock<AhoCorasick> = LazyLock::new(|| {
    let mut patterns: Vec<&str> = Vec::with_capacity(PEM_IDX + 1);
    for &(prefix, _, _, _, _, _) in SECRET_SPECS {
        patterns.push(prefix);
    }
    patterns.extend_from_slice(DB_SCHEMES);
    patterns.push("-----BEGIN ");
    AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
        .expect("secret aho-corasick")
});

fn token_len(tail: &str, upper_only: bool) -> usize {
    if upper_only {
        tail.bytes()
            .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            .count()
    } else {
        tail.bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            .count()
    }
}

fn db_creds_span(after: &str) -> Option<(usize, &str, &str)> {
    let end = after
        .find(['/', ' ', '"', '\'', ')', ']', '\n', '\r'])
        .unwrap_or(after.len());
    let (creds, host) = after[..end].split_once('@')?;
    if host.is_empty() || !creds.contains(':') {
        return None;
    }
    Some((end, creds, host))
}

/// Single-pass SIMD secret detection across all 22 prefix/URI/PEM patterns.
pub fn find_secret(line: &str) -> Option<&'static str> {
    let bytes = line.as_bytes();
    for mat in SECRET_AC.find_iter(line) {
        let pid = mat.pattern().as_usize();
        if pid == PEM_IDX {
            if line[mat.end()..].contains(PEM_MARKER) {
                return Some("pem_private_key");
            }
        } else if pid < DB_BASE {
            let (_, kind, _, dmin, _, upper) = SECRET_SPECS[pid];
            if left_bound(bytes, mat.start()) && token_len(&line[mat.end()..], upper) >= dmin {
                return Some(kind);
            }
        } else if let Some((_, creds, _)) = db_creds_span(&line[mat.end()..])
            && !creds.ends_with(':')
            && !creds.contains("${")
            && !creds.contains("REDACTED")
        {
            return Some("database_credential_url");
        }
    }
    None
}

fn consume_pem_block(s: &str, start: usize, end: usize) -> Option<usize> {
    let hdr_rel = s[end..].find(PEM_MARKER)?;
    let tail_start = end + hdr_rel + PEM_MARKER.len();
    let tail = &s[tail_start..];
    let Some(end_rel) = tail.find("-----END ") else {
        return Some(s.len());
    };
    let Some(close_rel) = tail[end_rel..].find(PEM_MARKER) else {
        return Some(s.len());
    };
    let _ = start;
    Some(tail_start + end_rel + close_rel + PEM_MARKER.len())
}

/// Single-pass SIMD secret redaction across all secret prefixes, DB URIs, and PEM blocks.
pub fn redact_secrets(s: String) -> String {
    if SECRET_AC.find(&s).is_none() && !s.contains(PEM_MARKER) {
        return s;
    }
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;

    for mat in SECRET_AC.find_iter(&s) {
        if mat.start() < cursor {
            continue;
        }
        let pid = mat.pattern().as_usize();
        if pid == PEM_IDX {
            if let Some(next_pos) = consume_pem_block(&s, mat.start(), mat.end()) {
                out.push_str(&s[cursor..mat.start()]);
                out.push_str("[REDACTED_PEM_PRIVATE_KEY]");
                cursor = next_pos;
            }
        } else if pid < DB_BASE {
            let (_, _, tag, _, rmin, _) = SECRET_SPECS[pid];
            let skip = token_len(&s[mat.end()..], false);
            if left_bound(bytes, mat.start()) && skip >= rmin {
                out.push_str(&s[cursor..mat.start()]);
                out.push_str(tag);
                cursor = mat.end() + skip;
            }
        } else if let Some((span_end, _, host)) = db_creds_span(&s[mat.end()..]) {
            out.push_str(&s[cursor..mat.end()]);
            out.push_str("REDACTED:REDACTED@");
            out.push_str(host);
            cursor = mat.end() + span_end;
        }
    }

    out.push_str(&s[cursor..]);
    if out.contains(PEM_MARKER) {
        out.replace(PEM_MARKER, "REDACTED_PRIVATE_KEY-----")
    } else {
        out
    }
}
