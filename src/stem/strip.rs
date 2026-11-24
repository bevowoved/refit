use crate::lex::{SentinelLex, blank_noise_sentinel};

/// Lex `content` into a 1:1 byte-aligned `0xFF`-sentinel view that blanks comments and
/// Rust `#[cfg(test)]` items while retaining exact line/byte offsets and test spans.
pub fn lex_sentinel(ext: &str, content: &str) -> SentinelLex {
    blank_noise_sentinel(ext, content)
}

/// Strip comments and Rust `#[cfg(test)]` items in one pass while keeping strings intact.
pub fn strip_noise(ext: &str, content: &str) -> String {
    blank_noise_sentinel(ext, content).to_clean_string(content)
}
