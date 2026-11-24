use crate::lex::check_balanced_syntax;

/// Check balanced braces and strings across Rust, TypeScript, JavaScript, JSX, and Go.
pub fn check_braces(rel_path: &str, source: &str) -> Result<(), String> {
    check_balanced_syntax(rel_path, source)
}
