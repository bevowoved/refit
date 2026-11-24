//! Leaf-module cross-OS path normalization and fail-closed security validation.
//!
//! DAG invariant (P1-1): imports ONLY `std` and `dunce` so `scc_count == 0`.

use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Once;

static WSL_9P_WARN_ONCE: Once = Once::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathReject {
    Empty,
    ControlOrNul,
    UnbalancedQuote,
    ReservedDeviceName,
    TrailingDotOrSpace,
    TraversalEscape,
    ProtectedDir,
    ShellMetachar,
    AlternateDataStream,
}

impl fmt::Display for PathReject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Empty => "empty path",
            Self::ControlOrNul => "control or NUL byte in path",
            Self::UnbalancedQuote => "unbalanced quotes in path",
            Self::ReservedDeviceName => "Windows reserved device name in path",
            Self::TrailingDotOrSpace => "trailing dot or space in path segment",
            Self::TraversalEscape => "path traversal outside root",
            Self::ProtectedDir => "protected repository metadata directory (.git/.refit)",
            Self::ShellMetachar => "disallowed shell metacharacter in path",
            Self::AlternateDataStream => "colon / alternate data stream in path",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for PathReject {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedPath {
    pub normalized: String,
    pub wsl_9p_warning: bool,
    pub nfd_normalized: bool,
}

pub struct CrossOsPathNormalizer;

impl CrossOsPathNormalizer {
    /// Normalize an OS path string (absolute or relative) across Windows verbatim `\\?\`,
    /// UNC, PowerShell 5.1 quotes, PowerShell 7.6 trailing `\\`, MSYS2 `/d/`, WSL2 `/mnt/d/`,
    /// `\\wsl.localhost\`, and macOS APFS NFD combining marks (`CERT/CC VU#604762`).
    pub fn normalize(raw: &str, check_shell_safe: bool) -> Result<NormalizedPath, PathReject> {
        let unquoted = validate_and_unquote_raw(raw, check_shell_safe)?;
        let (nfc, nfd_normalized) = normalize_nfd_to_nfc(unquoted);
        let slashed = nfc.replace('\\', "/");
        let (prefix, tail, wsl_9p_warning) = extract_os_prefix(&slashed);

        // PowerShell 7.6 trailing backslash collapse (`\\server\share\\` -> `//server/share`).
        let joined = normalize_segments(tail.trim_end_matches('/'))?;
        let normalized = if prefix.is_empty() {
            joined
        } else {
            format!("{prefix}{joined}")
        };

        Ok(NormalizedPath { normalized, wsl_9p_warning, nfd_normalized })
    }

    pub fn check_rel(rel: &str) -> bool {
        Self::normalize_rel(rel).is_some()
    }

    pub fn normalize_rel(rel: &str) -> Option<String> {
        if rel.is_empty() || rel.bytes().any(|b| b == 0 || b < 0x20) {
            return None;
        }
        if rel.contains(':') {
            return None;
        }

        let (nfc, _) = normalize_nfd_to_nfc(rel);
        let slashed = nfc.replace('\\', "/");
        if slashed.starts_with('/') {
            return None;
        }

        let mut stack: Vec<&str> = Vec::new();
        for seg in slashed.split('/') {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." {
                return None;
            }
            if seg.ends_with('.') || seg.ends_with(' ') {
                return None;
            }
            if is_windows_reserved(seg) || is_protected_repo_dir(seg) {
                return None;
            }
            stack.push(seg);
        }

        if stack.is_empty() {
            return None;
        }
        Some(stack.join("/"))
    }

    fn verify_canonical_in_root(root: &Path, cur: &Path) -> bool {
        let Ok(canon) = dunce::canonicalize(cur) else {
            return false;
        };
        if !canon.starts_with(root) {
            return false;
        }
        let Ok(canon_rel) = canon.strip_prefix(root) else {
            return false;
        };
        let canon_str = canon_rel.to_string_lossy().replace('\\', "/");
        canon_str.is_empty() || Self::check_rel(&canon_str)
    }

    pub fn check_target_in_repo(repo_root: &Path, rel: &str) -> bool {
        let Some(norm_rel) = Self::normalize_rel(rel) else {
            return false;
        };
        let Ok(root) = dunce::canonicalize(repo_root) else {
            return false;
        };

        let mut cur = root.clone();
        for seg in norm_rel.split('/') {
            cur.push(seg);
            match fs::symlink_metadata(&cur) {
                Ok(meta) if meta.file_type().is_symlink() || cur.exists() => {
                    if !Self::verify_canonical_in_root(&root, &cur) {
                        return false;
                    }
                }
                Ok(_) => {}
                Err(err) if err.kind() == ErrorKind::NotFound => break,
                Err(_) => return false,
            }
        }
        true
    }

    pub fn rel_to_root(repo_root: &Path, path: &Path) -> Option<String> {
        if let (Ok(c_root), Ok(c_path)) = (dunce::canonicalize(repo_root), dunce::canonicalize(path))
            && let Ok(rel) = c_path.strip_prefix(&c_root)
        {
            let s = rel.to_string_lossy().replace('\\', "/");
            if !s.is_empty() {
                return Some(s);
            }
        }

        let root_norm = dunce::canonicalize(repo_root)
            .map(|p| normalize_for_prefix(&p))
            .unwrap_or_else(|_| normalize_for_prefix(repo_root));
        let path_norm = normalize_for_prefix(path);

        if let Some(rest) = strip_root_str(&path_norm, &root_norm) {
            return Some(rest.to_string());
        }
        if !path.is_absolute() {
            return Self::normalize_rel(&path.to_string_lossy());
        }
        None
    }
}

fn validate_and_unquote_raw(raw: &str, check_shell_safe: bool) -> Result<&str, PathReject> {
    if raw.is_empty() {
        return Err(PathReject::Empty);
    }
    if raw.bytes().any(|b| b == 0 || b < 0x20) {
        return Err(PathReject::ControlOrNul);
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(PathReject::Empty);
    }

    let quote_count = trimmed.chars().filter(|&c| c == '"').count();
    let unquoted = if trimmed.starts_with('"') && trimmed.ends_with('"') && quote_count == 2 {
        &trimmed[1..trimmed.len() - 1]
    } else if quote_count > 0 {
        return Err(PathReject::UnbalancedQuote);
    } else {
        trimmed
    };

    if check_shell_safe
        && unquoted
            .chars()
            .any(|c| matches!(c, '&' | '|' | '^' | '%' | ';' | '`' | '$'))
    {
        return Err(PathReject::ShellMetachar);
    }
    Ok(unquoted)
}

fn format_drive_prefix(drive_byte: u8) -> String {
    format!("{}:/", drive_byte.to_ascii_uppercase() as char)
}

fn strip_slash_drive(rest: &str) -> Option<(String, &str)> {
    let bytes = rest.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b'/' {
        Some((format_drive_prefix(bytes[0]), &rest[2..]))
    } else {
        None
    }
}

fn split_slash_prefix(slashed: &str) -> (String, &str, bool) {
    if let Some(rest) = slashed
        .strip_prefix("//?/UNC/")
        .or_else(|| slashed.strip_prefix("//?/unc/"))
    {
        return ("//".to_string(), rest, false);
    }
    if let Some(rest) = slashed.strip_prefix("//?/") {
        return (String::new(), rest, false);
    }
    if let Some(rest) = slashed.strip_prefix("//wsl.localhost/") {
        return ("wsl://".to_string(), rest, false);
    }
    if let Some(rest) = slashed.strip_prefix("//") {
        return strip_slash_drive(rest)
            .map_or_else(|| ("//".to_string(), rest, false), |(pref, tail)| (pref, tail, false));
    }
    if let Some(rest) = slashed.strip_prefix("/mnt/") {
        if let Some((pref, tail)) = strip_slash_drive(rest) {
            return (pref, tail, true);
        }
        let bytes = rest.as_bytes();
        if bytes.len() == 1 && bytes[0].is_ascii_alphabetic() {
            return (format_drive_prefix(bytes[0]), "", true);
        }
        return (String::new(), slashed, false);
    }
    if let Some(rest) = slashed.strip_prefix('/') {
        return strip_slash_drive(rest)
            .map_or_else(|| ("/".to_string(), rest, false), |(pref, tail)| (pref, tail, false));
    }
    (String::new(), slashed, false)
}

fn extract_os_prefix(slashed: &str) -> (String, &str, bool) {
    let (mut prefix, mut rest, wsl_9p_warning) = split_slash_prefix(slashed);
    let bytes = rest.as_bytes();
    if prefix.is_empty() && bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        prefix = format_drive_prefix(bytes[0]);
        rest = rest[2..].trim_start_matches('/');
    }
    (prefix, rest, wsl_9p_warning)
}

fn validate_path_segment(seg: &str) -> Result<(), PathReject> {
    if seg.contains(':') {
        return Err(PathReject::AlternateDataStream);
    }
    if seg.ends_with('.') || seg.ends_with(' ') {
        return Err(PathReject::TrailingDotOrSpace);
    }
    if is_windows_reserved(seg) {
        return Err(PathReject::ReservedDeviceName);
    }
    if is_protected_repo_dir(seg) {
        return Err(PathReject::ProtectedDir);
    }
    Ok(())
}

fn normalize_segments(path_tail: &str) -> Result<String, PathReject> {
    let mut stack: Vec<&str> = Vec::new();
    for seg in path_tail.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            if stack.pop().is_none() {
                return Err(PathReject::TraversalEscape);
            }
            continue;
        }
        validate_path_segment(seg)?;
        stack.push(seg);
    }
    Ok(stack.join("/"))
}

/// Pure-Rust APFS NFD -> NFC normalization guard (`CERT/CC VU#604762`).
pub fn normalize_nfd_to_nfc(input: &str) -> (String, bool) {
    if input.is_ascii() {
        return (input.to_string(), false);
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut had_nfd = false;
    let mut i = 0usize;

    while i < chars.len() {
        if i + 1 < chars.len() && is_combining_mark(chars[i + 1]) {
            had_nfd = true;
            let composed = match (chars[i], chars[i + 1]) {
                ('a', '\u{0300}') => Some('\u{00E0}'),
                ('a', '\u{0301}') => Some('\u{00E1}'),
                ('a', '\u{0302}') => Some('\u{00E2}'),
                ('a', '\u{0303}') => Some('\u{00E3}'),
                ('a', '\u{0308}') => Some('\u{00E4}'),
                ('a', '\u{030A}') => Some('\u{00E5}'),
                ('c', '\u{0327}') => Some('\u{00E7}'),
                ('e', '\u{0300}') => Some('\u{00E8}'),
                ('e', '\u{0301}') => Some('\u{00E9}'),
                ('e', '\u{0302}') => Some('\u{00EA}'),
                ('e', '\u{0308}') => Some('\u{00EB}'),
                ('i', '\u{0300}') => Some('\u{00EC}'),
                ('i', '\u{0301}') => Some('\u{00ED}'),
                ('i', '\u{0302}') => Some('\u{00EE}'),
                ('i', '\u{0307}') => Some('i'),
                ('i', '\u{0308}') => Some('\u{00EF}'),
                ('n', '\u{0303}') => Some('\u{00F1}'),
                ('o', '\u{0300}') => Some('\u{00F2}'),
                ('o', '\u{0301}') => Some('\u{00F3}'),
                ('o', '\u{0302}') => Some('\u{00F4}'),
                ('o', '\u{0303}') => Some('\u{00F5}'),
                ('o', '\u{0308}') => Some('\u{00F6}'),
                ('u', '\u{0300}') => Some('\u{00F9}'),
                ('u', '\u{0301}') => Some('\u{00FA}'),
                ('u', '\u{0302}') => Some('\u{00FB}'),
                ('u', '\u{0308}') => Some('\u{00FC}'),
                ('A', '\u{0308}') => Some('\u{00C4}'),
                ('E', '\u{0301}') => Some('\u{00C9}'),
                ('N', '\u{0303}') => Some('\u{00D1}'),
                ('O', '\u{0308}') => Some('\u{00D6}'),
                ('U', '\u{0308}') => Some('\u{00DC}'),
                _ => None,
            };
            if let Some(c) = composed {
                out.push(c);
                i += 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    (out, had_nfd)
}

#[inline]
fn is_combining_mark(c: char) -> bool {
    ('\u{0300}'..='\u{036F}').contains(&c)
}

/// Checks Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`, `LPT1..9`)
/// with or without extension at any path segment.
pub fn is_windows_reserved(seg: &str) -> bool {
    let stem = seg.split('.').next().unwrap_or(seg).trim_end_matches(' ');
    let upper = stem.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

fn is_protected_repo_dir(seg: &str) -> bool {
    if seg.is_ascii() {
        return seg.eq_ignore_ascii_case(".git") || seg.eq_ignore_ascii_case(".refit");
    }
    let folded: String = seg
        .chars()
        .filter(|&c| !is_combining_mark(c))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    folded == ".git" || folded == ".refit"
}

/// Validate and normalize a repo-relative path strictly across Linux, macOS, and Windows.
///
/// Rejects traversal (`..`), absolute/UNC/verbatim/NT prefixes, NUL/control bytes,
/// `.git`/`.refit` at any depth (case/NFD-insensitive), Windows reserved device names,
/// ADS/drive-relative (`:`), trailing dots/spaces, and empty/`.` paths.
pub fn normalize_rel(rel: &str) -> Option<String> {
    CrossOsPathNormalizer::normalize_rel(rel)
}

pub fn check_rel(rel: &str) -> bool {
    CrossOsPathNormalizer::check_rel(rel)
}

/// Fail-closed repository target check using `dunce::canonicalize`.
///
/// Rejects if `repo_root` cannot be canonicalized, if `rel` fails `normalize_rel`,
/// if any existing or symlink prefix escapes `repo_root`, if any symlink is dangling,
/// or if any resolved symlink target points into `.git`/`.refit`.
pub fn check_target_in_repo(repo_root: &Path, rel: &str) -> bool {
    CrossOsPathNormalizer::check_target_in_repo(repo_root, rel)
}

/// Canonicalize a repository root via `dunce::canonicalize` and warn once on WSL2 `/mnt/*` 9P paths.
pub fn canonical_root(repo_root: &Path) -> PathBuf {
    let raw = repo_root.to_string_lossy();
    match CrossOsPathNormalizer::normalize(&raw, false) {
        Ok(norm) => {
            if norm.wsl_9p_warning {
                warn_wsl_9p_once(&raw);
            }
            if norm.nfd_normalized {
                eprintln!(
                    "refit note: normalized NFD path `{}` -> `{}`",
                    raw, norm.normalized
                );
            }
        }
        Err(err) => {
            let _ = err.to_string();
        }
    }
    dunce::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf())
}

fn warn_wsl_9p_once(raw: &str) {
    WSL_9P_WARN_ONCE.call_once(|| {
        eprintln!(
            "refit warning: path `{raw}` crosses the WSL2 /mnt/* 9P bridge (10x-20x slower than native ext4)"
        );
    });
}

fn normalize_for_prefix(p: &Path) -> String {
    let raw = p.to_string_lossy();
    if let Ok(n) = CrossOsPathNormalizer::normalize(&raw, false) {
        return n.normalized;
    }
    let s = raw.replace('\\', "/");
    let stripped = s
        .strip_prefix("//?/UNC/")
        .or_else(|| s.strip_prefix("//?/unc/"))
        .map(|rest| format!("//{rest}"))
        .or_else(|| s.strip_prefix("//?/").map(str::to_string))
        .unwrap_or(s);
    let b = stripped.as_bytes();
    let cased = if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        format!("{}{}", (b[0].to_ascii_uppercase() as char), &stripped[1..])
    } else {
        stripped
    };
    cased.trim_end_matches('/').to_string()
}

/// Convert an absolute or tool-reported path to a normalized repo-relative forward-slash path.
///
/// Handles Windows `\\?\` verbatim prefixes, drive-letter casing, MSYS2 `/d/`, and WSL2 `/mnt/d/`.
pub fn rel_to_root(repo_root: &Path, path: &Path) -> Option<String> {
    CrossOsPathNormalizer::rel_to_root(repo_root, path)
}

fn strip_root_str<'a>(path_norm: &'a str, root_norm: &str) -> Option<&'a str> {
    if root_norm.is_empty() {
        return None;
    }
    if let Some(rest) = path_norm.strip_prefix(root_norm)
        && let Some(rel) = rest.strip_prefix('/')
        && !rel.is_empty()
    {
        return Some(rel);
    }
    // Windows drive-letter or case-insensitive root fallback.
    let win_drive = root_norm.len() >= 2
        && root_norm.as_bytes()[0].is_ascii_alphabetic()
        && root_norm.as_bytes()[1] == b':';
    if (cfg!(windows) || win_drive) && path_norm.len() > root_norm.len() {
        let (head, tail) = path_norm.split_at(root_norm.len());
        if head.eq_ignore_ascii_case(root_norm)
            && let Some(rel) = tail.strip_prefix('/')
            && !rel.is_empty()
        {
            return Some(rel);
        }
    }
    None
}

/// Strip an optional UTF-8 BOM (`\u{FEFF}`) without allocating.
#[inline]
pub fn strip_utf8_bom(s: &str) -> &str {
    s.strip_prefix('\u{FEFF}').unwrap_or(s)
}

/// Strip trailing `\r` and `\n` bytes from piped CRLF/LF lines without allocating.
#[inline]
pub fn trim_crlf(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

/// PowerShell BOM-less UTF-8 console output preamble for explicit `pwsh` invocations.
#[inline]
pub fn pwsh_utf8_preamble() -> &'static str {
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static PATH_TEST_SEQ: AtomicU64 = AtomicU64::new(1);

    fn make_temp_dir(tag: &str) -> PathBuf {
        let seq = PATH_TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("refit_paths_{tag}_{}_{seq}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn try_symlink_file(target: &Path, link: &Path) -> bool {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(target, link).is_ok()
        }
    }

    fn try_symlink_dir(target: &Path, link: &Path) -> bool {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_dir(target, link).is_ok()
        }
    }

    #[test]
    fn cross_os_paths_50() {
        let base = make_temp_dir("corpus50");
        let repo = base.join("repo");
        let outside = base.join("outside");
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(repo.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(outside.join("secret.rs"), "const SECRET: &str = \"nope\";\n").unwrap();

        let mut passed = 0usize;

        // Category A (12): Traversal & Absolute/UNC Escapes -> ALL REJECT
        let cat_a = [
            "../a",
            "a/../../b",
            r"..\a",
            r"a\..\..\b",
            "/etc/passwd",
            r"\Windows\System32",
            r"C:\a.rs",
            "C:/a.rs",
            r"\\srv\share\a.rs",
            r"\\?\C:\a.rs",
            r"\??\C:\a.rs",
            "src/a\0.rs",
        ];
        assert_eq!(cat_a.len(), 12);
        for case in cat_a {
            assert!(!check_rel(case), "Cat A check_rel must reject `{case}`");
            assert!(!check_target_in_repo(&repo, case), "Cat A check_target must reject `{case}`");
            passed += 1;
        }

        // Category B (8): Protected Repo Metadata -> ALL REJECT
        let cat_b = [
            ".git/config",
            ".git/hooks/pre-commit",
            r".git\HEAD",
            "sub/.git/hooks/x",
            ".refit/wal.jsonl",
            r".refit\cas\abc",
            ".refit/invariants.yml",
            "a/.refit/x",
        ];
        assert_eq!(cat_b.len(), 8);
        for case in cat_b {
            assert!(!check_rel(case), "Cat B check_rel must reject `{case}`");
            assert!(!check_target_in_repo(&repo, case), "Cat B check_target must reject `{case}`");
            passed += 1;
        }

        // Category C (10): Windows Reserved Devices, ADS, Drive-Relative & Trailing Dots/Spaces -> ALL REJECT
        let cat_c = [
            "CON",
            "NUL",
            "PRN.rs",
            "AUX.txt",
            "COM1.rs",
            "LPT1.py",
            "src/CON.rs",
            "src/a.rs:ads",
            "C:rel.rs",
            "src/bad. ",
        ];
        assert_eq!(cat_c.len(), 10);
        for case in cat_c {
            assert!(!check_rel(case), "Cat C check_rel must reject `{case}`");
            assert!(!check_target_in_repo(&repo, case), "Cat C check_target must reject `{case}`");
            passed += 1;
        }
        // Extra hardening checks for trailing dot & NFD `.git`
        assert!(!check_rel("src/bad."));
        assert!(!check_rel(".gi\u{0307}t/config"));
        assert!(!check_rel(".GIT/HEAD"));
        assert!(!check_rel(".REFIT/wal.jsonl"));

        // Category D (8): Symlink Escapes & Mixed Separators
        // D1: Symlink outside repo -> REJECT
        if try_symlink_file(&outside.join("secret.rs"), &repo.join("escape.rs")) {
            assert!(!check_target_in_repo(&repo, "escape.rs"), "D1 symlink outside repo must reject");
        } else {
            // Fallback when unprivileged Windows lacks SeCreateSymbolicLinkPrivilege:
            // verify fail-closed root rejection holds.
            assert!(!check_target_in_repo(&base.join("nonexistent_root"), "src/main.rs"));
        }
        passed += 1;

        // D2: Dangling symlink with escaping parent -> REJECT
        if try_symlink_dir(&outside, &repo.join("escape_dir")) {
            assert!(
                !check_target_in_repo(&repo, "escape_dir/dangling_child.rs"),
                "D2 escaping symlink parent must reject"
            );
        } else {
            assert!(!check_target_in_repo(&base.join("missing_dir"), "sub/new.rs"));
        }
        if try_symlink_file(&outside.join("does_not_exist.rs"), &repo.join("dangling_link.rs")) {
            assert!(
                !check_target_in_repo(&repo, "dangling_link.rs"),
                "D2 dangling symlink must reject fail-closed"
            );
        }
        passed += 1;

        // D3: Symlink to .git/HEAD -> REJECT
        if try_symlink_file(&repo.join(".git/HEAD"), &repo.join("git_head_link")) {
            assert!(
                !check_target_in_repo(&repo, "git_head_link"),
                "D3 symlink to .git/HEAD must reject"
            );
        } else {
            assert!(!check_target_in_repo(&repo, ".git/HEAD"));
        }
        passed += 1;

        // D4-D5: "" and "." -> REJECT
        assert!(!check_rel("") && !check_target_in_repo(&repo, ""));
        passed += 1;
        assert!(!check_rel(".") && !check_target_in_repo(&repo, "."));
        passed += 1;

        // D6-D8: Mixed separators & CurDir -> ACCEPT & normalize
        assert_eq!(normalize_rel(r"src\core/sub\a.rs").as_deref(), Some("src/core/sub/a.rs"));
        assert!(check_target_in_repo(&repo, r"src\core/sub\a.rs"));
        passed += 1;
        assert_eq!(normalize_rel("./src/a.rs").as_deref(), Some("src/a.rs"));
        assert!(check_target_in_repo(&repo, "./src/a.rs"));
        passed += 1;
        assert_eq!(normalize_rel("src/./a.rs").as_deref(), Some("src/a.rs"));
        assert!(check_target_in_repo(&repo, "src/./a.rs"));
        passed += 1;

        // Category E (12): Valid Polyglot Unicode, Brackets & Dotfiles -> ALL ACCEPT
        let cat_e = [
            "src/main.rs",
            "src/gates/audit.rs",
            "pkg/my file.py",
            "src/日本語/café.rs",
            "app/(marketing)/[slug]/page.tsx",
            "app/[...all]/route.ts",
            "packages/@scope/pkg/index.ts",
            ".github/workflows/ci.yml",
            "Dockerfile.prod",
            "src/a+b=c.ts",
            "a/b/c/d/e/f/g/h/deep.go",
            "src/test_util.rs",
        ];
        assert_eq!(cat_e.len(), 12);
        for case in cat_e {
            assert!(check_rel(case), "Cat E check_rel must accept `{case}`");
            assert!(check_target_in_repo(&repo, case), "Cat E check_target must accept `{case}`");
            passed += 1;
        }

        assert_eq!(passed, 50, "must verify all 50 Gate 6.6 cases");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn cross_os_normalizer_s3_suite() {
        let ok_cases = [
            (r"\\?\D:\cerebras\refit\src\main.rs", "D:/cerebras/refit/src/main.rs", false, false),
            (r"\\?\c:\work\app\lib.rs", "C:/work/app/lib.rs", false, false),
            (r"\\?\UNC\filesrv\repos\refit", "//filesrv/repos/refit", false, false),
            (r"\\?\unc\nas01\share\proj\\", "//nas01/share/proj", false, false),
            (r"\\filesrv\share\folder\\\\", "//filesrv/share/folder", false, false),
            (r"D:\cerebras\refit\\\", "D:/cerebras/refit", false, false),
            (r#""D:\cerebras\refit\Cargo.toml""#, "D:/cerebras/refit/Cargo.toml", false, false),
            ("/d/cerebras/refit/src/mcp.rs", "D:/cerebras/refit/src/mcp.rs", false, false),
            ("//d/cerebras/refit/Cargo.toml", "D:/cerebras/refit/Cargo.toml", false, false),
            ("/mnt/d/cerebras/refit", "D:/cerebras/refit", true, false),
            (r"\\wsl.localhost\Ubuntu\home\dev\refit", "wsl://Ubuntu/home/dev/refit", false, false),
            ("src/cafe\u{0301}/index.ts", "src/caf\u{00E9}/index.ts", false, true),
            ("docs/Bo\u{0308}rke/notes.md", "docs/B\u{00F6}rke/notes.md", false, true),
        ];
        for (input, exp_norm, exp_wsl, exp_nfd) in ok_cases {
            let got = CrossOsPathNormalizer::normalize(input, false).unwrap();
            assert_eq!(got.normalized, exp_norm, "input={input}");
            assert_eq!(got.wsl_9p_warning, exp_wsl, "wsl input={input}");
            assert_eq!(got.nfd_normalized, exp_nfd, "nfd input={input}");
        }

        let err_cases = [
            ("", false, PathReject::Empty),
            ("   ", false, PathReject::Empty),
            ("src/main.rs\0.txt", false, PathReject::ControlOrNul),
            ("src/main.rs\r\n", false, PathReject::ControlOrNul),
            (r#"src/"bad"quote.rs"#, false, PathReject::UnbalancedQuote),
            ("src/CON.rs", false, PathReject::ReservedDeviceName),
            ("src/file.rs.", false, PathReject::TrailingDotOrSpace),
            ("../outside.rs", false, PathReject::TraversalEscape),
            (".gi\u{0307}t/config", false, PathReject::ProtectedDir),
            ("src/a.rs:ads", false, PathReject::AlternateDataStream),
            ("cargo & calc.exe", true, PathReject::ShellMetachar),
            ("ruff | sh", true, PathReject::ShellMetachar),
            ("echo %PATH%", true, PathReject::ShellMetachar),
            ("tool^inject", true, PathReject::ShellMetachar),
        ];
        for (input, shell_safe, exp_err) in err_cases {
            let got = CrossOsPathNormalizer::normalize(input, shell_safe).unwrap_err();
            assert_eq!(got, exp_err, "input={input}");
        }

        // rel_to_root handles Windows verbatim & drive-case mismatch
        assert_eq!(
            rel_to_root(Path::new(r"D:\cerebras\refit"), Path::new(r"\\?\d:\cerebras\refit\src\app.py")).as_deref(),
            Some("src/app.py")
        );
        assert_eq!(
            rel_to_root(Path::new("/mnt/d/cerebras/refit"), Path::new(r"D:\cerebras\refit\src\mod.py")).as_deref(),
            Some("src/mod.py")
        );

        assert_eq!(strip_utf8_bom("\u{FEFF}hello"), "hello");
        assert_eq!(trim_crlf("line\r\n"), "line");
        assert!(pwsh_utf8_preamble().contains("UTF8Encoding"));
    }
}
