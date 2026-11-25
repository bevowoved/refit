//! Cross-OS path normalization and fail-closed containment checks.

use smallvec::SmallVec;
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
        f.write_str(match self {
            Self::Empty => "empty path",
            Self::ControlOrNul => "control or NUL byte in path",
            Self::UnbalancedQuote => "unbalanced quotes in path",
            Self::ReservedDeviceName => "Windows reserved device name in path",
            Self::TrailingDotOrSpace => "trailing dot or space in path segment",
            Self::TraversalEscape => "path traversal outside root",
            Self::ProtectedDir => "protected repository metadata directory (.git/.refit)",
            Self::ShellMetachar => "disallowed shell metacharacter in path",
            Self::AlternateDataStream => "colon / alternate data stream in path",
        })
    }
}

impl std::error::Error for PathReject {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedPath {
    pub normalized: String,
    pub wsl_9p_warning: bool,
    pub nfd_normalized: bool,
}

/// Normalize an OS path across verbatim, UNC, MSYS2, WSL2, and NFD forms.
pub fn normalize_os_path(raw: &str, check_shell_safe: bool) -> Result<NormalizedPath, PathReject> {
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
    let qcount = trimmed.chars().filter(|&c| c == '"').count();
    let unquoted = if trimmed.starts_with('"') && trimmed.ends_with('"') && qcount == 2 {
        &trimmed[1..trimmed.len() - 1]
    } else if qcount > 0 {
        return Err(PathReject::UnbalancedQuote);
    } else {
        trimmed
    };
    let has_shell_meta = unquoted
        .chars()
        .any(|c| matches!(c, '&' | '|' | '^' | '%' | ';' | '`' | '$'));
    if check_shell_safe && has_shell_meta {
        return Err(PathReject::ShellMetachar);
    }

    let (nfc, nfd_normalized) = normalize_nfd_to_nfc(unquoted);
    let slashed = nfc.replace('\\', "/");
    let (prefix, tail, wsl_9p_warning) = extract_os_prefix(&slashed);

    let mut stack: SmallVec<[&str; 16]> = SmallVec::new();
    for seg in tail.trim_end_matches('/').split('/') {
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
    let joined = stack.join("/");
    let normalized = if prefix.is_empty() { joined } else { format!("{prefix}{joined}") };
    Ok(NormalizedPath { normalized, wsl_9p_warning, nfd_normalized })
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

fn format_drive_prefix(drive_byte: u8) -> String {
    format!("{}:/", drive_byte.to_ascii_uppercase() as char)
}

fn strip_slash_drive(rest: &str) -> Option<(String, &str)> {
    let b = rest.as_bytes();
    (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b'/')
        .then(|| (format_drive_prefix(b[0]), &rest[2..]))
}

fn extract_os_prefix(slashed: &str) -> (String, &str, bool) {
    let (mut prefix, mut rest, wsl_9p) = if let Some(r) =
        slashed.strip_prefix("//?/UNC/").or_else(|| slashed.strip_prefix("//?/unc/"))
    {
        ("//".into(), r, false)
    } else if let Some(r) = slashed.strip_prefix("//?/") {
        (String::new(), r, false)
    } else if let Some(r) = slashed
        .strip_prefix("//wsl.localhost/")
        .or_else(|| slashed.strip_prefix("wsl://"))
    {
        ("wsl://".into(), r, false)
    } else if let Some(r) = slashed.strip_prefix("//") {
        strip_slash_drive(r).map_or_else(|| ("//".into(), r, false), |(p, t)| (p, t, false))
    } else if let Some(r) = slashed.strip_prefix("/mnt/") {
        if let Some((p, t)) = strip_slash_drive(r) {
            (p, t, true)
        } else if r.len() == 1 && r.as_bytes()[0].is_ascii_alphabetic() {
            (format_drive_prefix(r.as_bytes()[0]), "", true)
        } else {
            (String::new(), slashed, false)
        }
    } else if let Some(r) = slashed.strip_prefix('/') {
        strip_slash_drive(r).map_or_else(|| ("/".into(), r, false), |(p, t)| (p, t, false))
    } else {
        (String::new(), slashed, false)
    };

    let b = rest.as_bytes();
    if prefix.is_empty() && b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        prefix = format_drive_prefix(b[0]);
        rest = rest[2..].trim_start_matches('/');
    }
    (prefix, rest, wsl_9p)
}

/// Fold common APFS NFD combining pairs into NFC.
fn normalize_nfd_to_nfc(input: &str) -> (String, bool) {
    if input.is_ascii() {
        return (input.to_string(), false);
    }
    let chars: Vec<char> = input.chars().collect();
    let (mut out, mut had_nfd, mut i) = (String::with_capacity(input.len()), false, 0usize);
    while i < chars.len() {
        if i + 1 < chars.len() && is_combining_mark(chars[i + 1]) {
            had_nfd = true;
            let composed = match (chars[i], chars[i + 1]) {
                ('a', '\u{0300}') => Some('\u{00E0}'), ('a', '\u{0301}') => Some('\u{00E1}'),
                ('a', '\u{0302}') => Some('\u{00E2}'), ('a', '\u{0303}') => Some('\u{00E3}'),
                ('a', '\u{0308}') => Some('\u{00E4}'), ('a', '\u{030A}') => Some('\u{00E5}'),
                ('c', '\u{0327}') => Some('\u{00E7}'), ('e', '\u{0300}') => Some('\u{00E8}'),
                ('e', '\u{0301}') => Some('\u{00E9}'), ('e', '\u{0302}') => Some('\u{00EA}'),
                ('e', '\u{0308}') => Some('\u{00EB}'), ('i', '\u{0300}') => Some('\u{00EC}'),
                ('i', '\u{0301}') => Some('\u{00ED}'), ('i', '\u{0302}') => Some('\u{00EE}'),
                ('i', '\u{0307}') => Some('i'),        ('i', '\u{0308}') => Some('\u{00EF}'),
                ('n', '\u{0303}') => Some('\u{00F1}'), ('o', '\u{0300}') => Some('\u{00F2}'),
                ('o', '\u{0301}') => Some('\u{00F3}'), ('o', '\u{0302}') => Some('\u{00F4}'),
                ('o', '\u{0303}') => Some('\u{00F5}'), ('o', '\u{0308}') => Some('\u{00F6}'),
                ('u', '\u{0300}') => Some('\u{00F9}'), ('u', '\u{0301}') => Some('\u{00FA}'),
                ('u', '\u{0302}') => Some('\u{00FB}'), ('u', '\u{0308}') => Some('\u{00FC}'),
                ('A', '\u{0308}') => Some('\u{00C4}'), ('E', '\u{0301}') => Some('\u{00C9}'),
                ('N', '\u{0303}') => Some('\u{00D1}'), ('O', '\u{0308}') => Some('\u{00D6}'),
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

/// Check whether a segment matches a Windows reserved device name.
fn is_windows_reserved(seg: &str) -> bool {
    let base = seg.split('.').next().unwrap_or(seg);
    let stem = base.trim_end_matches(' ');
    matches!(
        stem.to_ascii_uppercase().as_str(),
        "CON" | "PRN" | "AUX" | "NUL"
            | "COM1" | "COM2" | "COM3" | "COM4" | "COM5" | "COM6" | "COM7" | "COM8" | "COM9"
            | "LPT1" | "LPT2" | "LPT3" | "LPT4" | "LPT5" | "LPT6" | "LPT7" | "LPT8" | "LPT9"
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

/// Validate and normalize a repo-relative path across OSes.
pub fn normalize_rel(rel: &str) -> Option<String> {
    if rel.is_empty() || rel.contains(':') || rel.bytes().any(|b| b == 0 || b < 0x20) {
        return None;
    }
    let (nfc, _) = normalize_nfd_to_nfc(rel);
    let slashed = nfc.replace('\\', "/");
    if slashed.starts_with('/') {
        return None;
    }
    let mut stack: SmallVec<[&str; 16]> = SmallVec::new();
    for seg in slashed.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." || validate_path_segment(seg).is_err() {
            return None;
        }
        stack.push(seg);
    }
    (!stack.is_empty()).then(|| stack.join("/"))
}

pub fn check_rel(rel: &str) -> bool {
    normalize_rel(rel).is_some()
}

fn verify_canonical_in_root(root: &Path, cur: &Path) -> bool {
    let Ok(canon) = dunce::canonicalize(cur) else { return false };
    if same_file::is_same_file(&canon, root).unwrap_or(false) {
        return true;
    }
    let Ok(canon_rel) = canon.strip_prefix(root) else { return false };
    let canon_str = canon_rel.to_string_lossy().replace('\\', "/");
    canon_str.is_empty() || check_rel(&canon_str)
}

/// Verify that a relative path stays inside `repo_root` without symlink escape.
pub fn check_target_in_repo(repo_root: &Path, rel: &str) -> bool {
    let Some(norm_rel) = normalize_rel(rel) else { return false };
    let Ok(root) = dunce::canonicalize(repo_root) else { return false };
    let mut cur = root.clone();
    for seg in norm_rel.split('/') {
        cur.push(seg);
        match fs::symlink_metadata(&cur) {
            Ok(meta) if meta.file_type().is_symlink() || cur.exists() => {
                if !verify_canonical_in_root(&root, &cur) {
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

/// Canonicalize a repository root and warn once on WSL2 `/mnt/*` 9P paths.
pub fn canonical_root(repo_root: &Path) -> PathBuf {
    let raw = repo_root.to_string_lossy();
    if let Ok(norm) = normalize_os_path(&raw, false) {
        if norm.wsl_9p_warning {
            WSL_9P_WARN_ONCE.call_once(|| {
                eprintln!(
                    "refit warning: path `{raw}` crosses the WSL2 /mnt/* 9P bridge \
                     (10x-20x slower than native ext4)"
                );
            });
        }
        if norm.nfd_normalized {
            eprintln!("refit note: normalized NFD path `{raw}` -> `{}`", norm.normalized);
        }
    }
    dunce::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf())
}

fn normalize_for_prefix(p: &Path) -> String {
    let raw = p.to_string_lossy();
    if let Ok(n) = normalize_os_path(&raw, false) {
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
pub fn rel_to_root(repo_root: &Path, path: &Path) -> Option<String> {
    if same_file::is_same_file(repo_root, path).unwrap_or(false) {
        return None;
    }
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
    (!path.is_absolute()).then(|| normalize_rel(&path.to_string_lossy())).flatten()
}

fn strip_root_str<'a>(path_norm: &'a str, root_norm: &str) -> Option<&'a str> {
    if root_norm.is_empty() {
        return None;
    }
    if let Some(rel) = path_norm.strip_prefix(root_norm).and_then(|r| r.strip_prefix('/'))
        && !rel.is_empty()
    {
        return Some(rel);
    }
    let win_drive = root_norm.len() >= 2
        && root_norm.as_bytes()[0].is_ascii_alphabetic()
        && root_norm.as_bytes()[1] == b':';
    if (cfg!(windows) || win_drive)
        && path_norm.len() > root_norm.len()
        && path_norm.is_char_boundary(root_norm.len())
    {
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

#[inline]
pub fn strip_utf8_bom(s: &str) -> &str {
    s.strip_prefix('\u{FEFF}').unwrap_or(s)
}

#[inline]
pub fn trim_crlf(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

#[inline]
pub fn pwsh_utf8_preamble() -> &'static str {
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TempRepo, TestRng};

    fn try_symlink(target: &Path, link: &Path, is_dir: bool) -> bool {
        #[cfg(unix)]
        { let _ = is_dir; std::os::unix::fs::symlink(target, link).is_ok() }
        #[cfg(windows)]
        if is_dir {
            std::os::windows::fs::symlink_dir(target, link).is_ok()
                || std::process::Command::new("cmd")
                    .args(["/C", "mklink", "/J"])
                    .arg(link)
                    .arg(target)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok_and(|s| s.success())
        } else {
            std::os::windows::fs::symlink_file(target, link).is_ok()
        }
    }

    #[test]
    fn cross_os_paths() {
        let base = TempRepo::new("paths");
        let (repo, outside) = (base.join("repo"), base.join("outside"));
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(repo.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(outside.join("secret.rs"), "const S: u8 = 1;\n").unwrap();
        let canon = canonical_root(&repo);

        for bad in [
            "../a", "a/../../b", r"..\a", "/etc/passwd", r"\Windows\System32", r"C:\a.rs",
            "C:/a.rs", r"\\srv\share\a.rs", r"\\?\C:\a.rs", r"\??\C:\a.rs", "src/a\0.rs",
            ".git/config", r".git\HEAD", "sub/.git/x", ".refit/wal.jsonl", r".refit\cas\a",
            ".gi\u{0307}t/config", ".GIT/HEAD", ".REFIT/wal.jsonl", "CON", "NUL", "PRN.rs",
            "AUX.txt", "COM1.rs", "LPT1.py", "src/CON.rs", "src/a.rs:ads", "C:rel.rs",
            "src/bad. ", "src/bad.", "", ".",
        ] {
            assert!(!check_rel(bad) && !check_target_in_repo(&canon, bad), "reject `{bad}`");
        }

        let links = [
            (outside.join("secret.rs"), "esc.rs", false),
            (outside.clone(), "esc_dir", true),
            (outside.join("gone.rs"), "dangle.rs", false),
            (repo.join(".git/HEAD"), "git_link", false),
        ];
        let mut tested_links = 0usize;
        for (target, name, is_dir) in links {
            let link_path = repo.join(name);
            if try_symlink(&target, &link_path, is_dir) {
                tested_links += 1;
                let probe = if is_dir { format!("{name}/child.rs") } else { name.into() };
                assert!(!check_target_in_repo(&canon, &probe));
                let _ = if is_dir { fs::remove_dir(&link_path) } else { fs::remove_file(&link_path) };
            }
        }
        assert!(tested_links >= 1, "expected at least one symlink or junction escape check");
        assert!(!check_target_in_repo(&base.join("missing_root"), "src/main.rs"));

        for (raw, want) in [(r"src\core/sub\a.rs", "src/core/sub/a.rs"), ("./src/a.rs", "src/a.rs")] {
            assert_eq!(normalize_rel(raw).as_deref(), Some(want));
            assert!(check_target_in_repo(&canon, raw));
        }
        for ok in [
            "src/main.rs", "pkg/my file.py", "src/日本語/café.rs",
            "app/(marketing)/[slug]/page.tsx", "packages/@scope/pkg/index.ts",
            ".github/workflows/ci.yml", "a/b/c/d/e/f/deep.go",
        ] {
            assert!(check_rel(ok) && check_target_in_repo(&canon, ok), "accept `{ok}`");
        }
        assert_eq!(rel_to_root(&canon, &canon), None);
        assert_eq!(rel_to_root(&canon, &canon.join("src/a.rs")).as_deref(), Some("src/a.rs"));
        assert_eq!(rel_to_root(&canon, &outside.join("secret.rs")), None);
    }

    #[test]
    fn cross_os_path_norm() {
        for (input, exp, wsl, nfd) in [
            (r"\\?\D:\refit\src\main.rs", "D:/refit/src/main.rs", false, false),
            (r"\\?\c:\app\lib.rs", "C:/app/lib.rs", false, false),
            (r"\\?\UNC\srv\repos\refit", "//srv/repos/refit", false, false),
            (r"\\?\unc\nas\sh\p\\", "//nas/sh/p", false, false),
            (r"\\srv\sh\dir\\\\", "//srv/sh/dir", false, false),
            (r#""D:\refit\Cargo.toml""#, "D:/refit/Cargo.toml", false, false),
            ("/d/refit/src/mcp.rs", "D:/refit/src/mcp.rs", false, false),
            ("//d/refit/Cargo.toml", "D:/refit/Cargo.toml", false, false),
            ("/mnt/d/refit", "D:/refit", true, false),
            (r"\\wsl.localhost\Ubuntu\home\dev", "wsl://Ubuntu/home/dev", false, false),
            ("src/cafe\u{0301}/index.ts", "src/caf\u{00E9}/index.ts", false, true),
            ("docs/Bo\u{0308}rke/n.md", "docs/B\u{00F6}rke/n.md", false, true),
        ] {
            let g = normalize_os_path(input, false).unwrap();
            assert_eq!((g.normalized.as_str(), g.wsl_9p_warning, g.nfd_normalized), (exp, wsl, nfd));
        }

        for (input, safe, want) in [
            ("", false, PathReject::Empty), ("a\0b", false, PathReject::ControlOrNul),
            ("a\r\nb", false, PathReject::ControlOrNul), (r#"a/"b"c"#, false, PathReject::UnbalancedQuote),
            ("src/CON.rs", false, PathReject::ReservedDeviceName),
            ("src/a.rs.", false, PathReject::TrailingDotOrSpace),
            ("../x.rs", false, PathReject::TraversalEscape),
            (".gi\u{0307}t/x", false, PathReject::ProtectedDir),
            ("src/a.rs:ads", false, PathReject::AlternateDataStream),
            ("a & b", true, PathReject::ShellMetachar), ("a | b", true, PathReject::ShellMetachar),
            ("%PATH%", true, PathReject::ShellMetachar), ("a^b", true, PathReject::ShellMetachar),
        ] {
            assert_eq!(normalize_os_path(input, safe).unwrap_err(), want, "in={input}");
        }

        assert_eq!(rel_to_root(Path::new(r"D:\r"), Path::new(r"\\?\d:\r\src\a.py")).as_deref(), Some("src/a.py"));
        assert_eq!(rel_to_root(Path::new("/mnt/d/r"), Path::new(r"D:\r\src\b.py")).as_deref(), Some("src/b.py"));
        assert_eq!(rel_to_root(Path::new(r"D:\r\ab"), Path::new("D:\\r\\a\u{00E9}\\x.rs")), None);
        assert_eq!((strip_utf8_bom("\u{FEFF}hi"), trim_crlf("x\r\n")), ("hi", "x"));
        assert!(pwsh_utf8_preamble().contains("UTF8Encoding"));

        let repo = TempRepo::new("path_prop");
        let canon = canonical_root(repo.path());
        let pfx_list = ["", r"\\?\c:\", r"D:\", "/d/", "/mnt/d/", r"\\srv\sh\", r"\\wsl.localhost\u\", "/"];
        let segs = ["src", "pkg", ".", "sub", "cafe\u{0301}", "f.rs", "..", "CON", "a:ads", ".git", "x."];
        let mut rng = TestRng::new(0x50A7_C0DE);
        for _ in 0..500 {
            let pfx = pfx_list[rng.next_usize(pfx_list.len())];
            let mut raw = pfx.to_string();
            for i in 0..=rng.next_usize(4) {
                if i > 0 { raw.push(if rng.next_usize(2) == 0 { '/' } else { '\\' }); }
                raw.push_str(segs[rng.next_usize(segs.len())]);
            }
            if let Ok(n1) = normalize_os_path(&raw, false) && !n1.normalized.is_empty() {
                let n2 = normalize_os_path(&n1.normalized, false).unwrap();
                assert_eq!(n2.normalized, n1.normalized);
                assert!(!n2.nfd_normalized && !n1.normalized.contains('\\'));
            }
            if pfx.is_empty() && let Some(r1) = normalize_rel(&raw) {
                assert_eq!(normalize_rel(&r1).as_deref(), Some(r1.as_str()));
                assert!(check_target_in_repo(&canon, &r1));
            }
        }
    }
}
