use crate::lex::left_bound;

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

fn has_prefix(line: &str, prefix: &str, min_len: usize, upper_only: bool) -> bool {
    let mut rest = line;
    while let Some(pos) = rest.find(prefix) {
        let tail = &rest[pos + prefix.len()..];
        if left_bound(rest.as_bytes(), pos) && token_len(tail, upper_only) >= min_len {
            return true;
        }
        rest = tail;
    }
    false
}

fn has_dburi(line: &str, scheme: &str) -> bool {
    let mut rest = line;
    while let Some(pos) = rest.find(scheme) {
        let after = &rest[pos + scheme.len()..];
        let auth = after.split(['/', ' ', '"', '\'', ')', ']']).next().unwrap_or("");
        if let Some((creds, host)) = auth.split_once('@')
            && !host.is_empty()
            && creds.contains(':')
            && !creds.ends_with(':')
            && !creds.contains("${")
            && !creds.contains("REDACTED")
        {
            return true;
        }
        rest = after;
    }
    false
}

pub fn find_secret(line: &str) -> Option<&'static str> {
    if line.contains("-----BEGIN ") && line.contains("PRIVATE KEY-----") {
        return Some("pem_private_key");
    }

    for &(prefix, kind, _, dmin, _, upper) in SECRET_SPECS {
        if line.contains(prefix) && has_prefix(line, prefix, dmin, upper) {
            return Some(kind);
        }
    }

    for &scheme in DB_SCHEMES {
        if line.contains(scheme) && has_dburi(line, scheme) {
            return Some("database_credential_url");
        }
    }
    None
}

pub fn redact_secrets(mut s: String) -> String {
    for &(prefix, _, tag, _, rmin, _) in SECRET_SPECS {
        if s.contains(prefix) {
            s = redact_prefix(&s, prefix, tag, rmin);
        }
    }

    for &scheme in DB_SCHEMES {
        if s.contains(scheme) {
            s = redact_uris(&s, scheme);
        }
    }

    if s.contains("PRIVATE KEY-----") {
        s = redact_pem(&s);
    }
    s
}

fn redact_pem(input: &str) -> String {
    const MARKER: &str = "PRIVATE KEY-----";
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(begin) = rest.find("-----BEGIN ") {
        let Some(hdr_end) = rest[begin..].find(MARKER) else {
            out.push_str(&rest[..begin + 11]);
            rest = &rest[begin + 11..];
            continue;
        };

        out.push_str(&rest[..begin]);
        out.push_str("[REDACTED_PEM_PRIVATE_KEY]");
        let tail = &rest[begin + hdr_end + MARKER.len()..];

        let Some(end_rel) = tail.find("-----END ") else {
            rest = "";
            break;
        };
        let Some(close_rel) = tail[end_rel..].find(MARKER) else {
            rest = "";
            break;
        };
        rest = &tail[end_rel + close_rel + MARKER.len()..];
    }

    out.push_str(rest);
    out.replace(MARKER, "REDACTED_PRIVATE_KEY-----")
}

fn redact_prefix(input: &str, prefix: &str, replacement: &str, min_tail: usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(pos) = rest.find(prefix) {
        let after = &rest[pos + prefix.len()..];
        let skip = token_len(after, false);
        if left_bound(rest.as_bytes(), pos) && skip >= min_tail {
            out.push_str(&rest[..pos]);
            out.push_str(replacement);
            rest = &after[skip..];
        } else {
            out.push_str(&rest[..pos + prefix.len()]);
            rest = after;
        }
    }

    out.push_str(rest);
    out
}

fn redact_uris(input: &str, scheme: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(pos) = rest.find(scheme) {
        out.push_str(&rest[..pos + scheme.len()]);
        let after = &rest[pos + scheme.len()..];
        let end = after
            .find(['/', ' ', '"', '\'', ')', ']', '\n', '\r'])
            .unwrap_or(after.len());

        let Some((creds, host)) = after[..end].split_once('@') else {
            rest = after;
            continue;
        };
        if creds.contains(':') && !host.is_empty() {
            out.push_str("REDACTED:REDACTED@");
            out.push_str(host);
            rest = &after[end..];
        } else {
            rest = after;
        }
    }

    out.push_str(rest);
    out
}
