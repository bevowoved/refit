//! Cross-OS path normalization and fail-closed containment checks.

use smallvec::SmallVec;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Once;

static WSL_9P_WARN_ONCE: Once = Once::new();

const NFD_PAIRS: &[(char, char, char)] = &[
    ('a', '\u{0300}', '\u{00E0}'), ('a', '\u{0301}', '\u{00E1}'), ('a', '\u{0302}', '\u{00E2}'),
    ('a', '\u{0303}', '\u{00E3}'), ('a', '\u{0308}', '\u{00E4}'), ('a', '\u{030A}', '\u{00E5}'),
    ('c', '\u{0327}', '\u{00E7}'), ('e', '\u{0300}', '\u{00E8}'), ('e', '\u{0301}', '\u{00E9}'),
    ('e', '\u{0302}', '\u{00EA}'), ('e', '\u{0308}', '\u{00EB}'), ('i', '\u{0300}', '\u{00EC}'),
    ('i', '\u{0301}', '\u{00ED}'), ('i', '\u{0302}', '\u{00EE}'), ('i', '\u{0307}', 'i'),
    ('i', '\u{0308}', '\u{00EF}'), ('n', '\u{0303}', '\u{00F1}'), ('o', '\u{0300}', '\u{00F2}'),
    ('o', '\u{0301}', '\u{00F3}'), ('o', '\u{0302}', '\u{00F4}'), ('o', '\u{0303}', '\u{00F5}'),
    ('o', '\u{0308}', '\u{00F6}'), ('u', '\u{0300}', '\u{00F9}'), ('u', '\u{0301}', '\u{00FA}'),
    ('u', '\u{0302}', '\u{00FB}'), ('u', '\u{0308}', '\u{00FC}'), ('A', '\u{0308}', '\u{00C4}'),
    ('E', '\u{0301}', '\u{00C9}'), ('N', '\u{0303}', '\u{00D1}'), ('O', '\u{0308}', '\u{00D6}'),
    ('U', '\u{0308}', '\u{00DC}'),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PathReject {
    #[error("empty path")] Empty,
    #[error("control or NUL byte in path")] ControlOrNul,
    #[error("unbalanced quotes in path")] UnbalancedQuote,
    #[error("Windows reserved device name in path")] ReservedDeviceName,
    #[error("trailing dot or space in path segment")] TrailingDotOrSpace,
    #[error("path traversal outside root")] TraversalEscape,
    #[error("protected repository metadata directory (.git/.refit)")] ProtectedDir,
    #[error("disallowed shell metacharacter in path")] ShellMetachar,
    #[error("colon / alternate data stream in path")] AlternateDataStream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedPath {
    pub normalized: String,
    pub wsl_9p_warning: bool,
    pub nfd_normalized: bool,
}

/// Normalize an OS path across verbatim, UNC, MSYS2, WSL2, and NFD forms.
pub fn normalize_os_path(raw: &str, check_shell_safe: bool) -> Result<NormalizedPath, PathReject> {
    if raw.is_empty() { return Err(PathReject::Empty); }
    if raw.bytes().any(|byte| byte < 0x20) { return Err(PathReject::ControlOrNul); }
    let trimmed = raw.trim();
    if trimmed.is_empty() { return Err(PathReject::Empty); }

    let qcount = trimmed.chars().filter(|&ch| ch == '"').count();
    let unquoted = if trimmed.starts_with('"') && trimmed.ends_with('"') && qcount == 2 {
        &trimmed[1..trimmed.len() - 1]
    } else if qcount > 0 {
        return Err(PathReject::UnbalancedQuote);
    } else {
        trimmed
    };
    if check_shell_safe && unquoted.chars().any(|ch| matches!(ch, '&' | '|' | '^' | '%' | ';' | '`' | '$')) {
        return Err(PathReject::ShellMetachar);
    }

    let (nfc, nfd_normalized) = normalize_nfd_to_nfc(unquoted);
    let slashed = nfc.replace('\\', "/");
    let (prefix, tail, wsl_9p_warning) = extract_os_prefix(&slashed);

    let mut stack: SmallVec<[&str; 16]> = SmallVec::new();
    for seg in tail.trim_end_matches('/').split('/') {
        if seg.is_empty() || seg == "." { continue; }
        if seg == ".." {
            if stack.pop().is_none() { return Err(PathReject::TraversalEscape); }
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
    if seg.contains(':') { return Err(PathReject::AlternateDataStream); }
    if seg.ends_with(['.', ' ']) { return Err(PathReject::TrailingDotOrSpace); }
    if is_windows_reserved(seg) { return Err(PathReject::ReservedDeviceName); }
    if is_protected_repo_dir(seg) { return Err(PathReject::ProtectedDir); }
    Ok(())
}

fn format_drive_prefix(drive_byte: u8) -> String {
    format!("{}:/", drive_byte.to_ascii_uppercase() as char)
}

fn is_drive_prefix(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn strip_slash_drive(rest: &str) -> Option<(String, &str)> {
    let bytes = rest.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b'/')
        .then(|| (format_drive_prefix(bytes[0]), &rest[2..]))
}

fn strip_unc_verbatim(slashed: &str) -> Option<&str> {
    slashed.strip_prefix("//?/UNC/").or_else(|| slashed.strip_prefix("//?/unc/"))
}

fn extract_os_prefix(slashed: &str) -> (String, &str, bool) {
    let (mut prefix, mut rest, wsl_9p) = if let Some(tail) = strip_unc_verbatim(slashed) {
        ("//".into(), tail, false)
    } else if let Some(tail) = slashed.strip_prefix("//?/") {
        (String::new(), tail, false)
    } else if let Some(tail) =
        slashed.strip_prefix("//wsl.localhost/").or_else(|| slashed.strip_prefix("wsl://"))
    {
        ("wsl://".into(), tail, false)
    } else if let Some(tail) = slashed.strip_prefix("//") {
        strip_slash_drive(tail).map_or_else(|| ("//".into(), tail, false), |(drv, rem)| (drv, rem, false))
    } else if let Some(tail) = slashed.strip_prefix("/mnt/") {
        if let Some((drv, rem)) = strip_slash_drive(tail) {
            (drv, rem, true)
        } else if tail.len() == 1 && tail.as_bytes()[0].is_ascii_alphabetic() {
            (format_drive_prefix(tail.as_bytes()[0]), "", true)
        } else {
            (String::new(), slashed, false)
        }
    } else if let Some(tail) = slashed.strip_prefix('/') {
        strip_slash_drive(tail).map_or_else(|| ("/".into(), tail, false), |(drv, rem)| (drv, rem, false))
    } else {
        (String::new(), slashed, false)
    };

    if prefix.is_empty() && is_drive_prefix(rest) {
        prefix = format_drive_prefix(rest.as_bytes()[0]);
        rest = rest[2..].trim_start_matches('/');
    }
    (prefix, rest, wsl_9p)
}

/// Fold common APFS NFD combining pairs into NFC.
fn normalize_nfd_to_nfc(input: &str) -> (String, bool) {
    if input.is_ascii() { return (input.to_string(), false); }
    let chars: Vec<char> = input.chars().collect();
    let (mut out, mut had_nfd, mut idx) = (String::with_capacity(input.len()), false, 0usize);
    while idx < chars.len() {
        if idx + 1 < chars.len() && is_combining_mark(chars[idx + 1]) {
            had_nfd = true;
            if let Some(&(_, _, composed)) =
                NFD_PAIRS.iter().find(|&&(base, mark, _)| base == chars[idx] && mark == chars[idx + 1])
            {
                out.push(composed);
                idx += 2;
                continue;
            }
        }
        out.push(chars[idx]);
        idx += 1;
    }
    (out, had_nfd)
}

#[inline]
fn is_combining_mark(ch: char) -> bool { ('\u{0300}'..='\u{036F}').contains(&ch) }

/// Check whether a segment matches a Windows reserved device name.
fn is_windows_reserved(seg: &str) -> bool {
    let head = seg.split('.').next().unwrap_or(seg);
    let stem = head.trim_end_matches(' ');
    let bytes = stem.as_bytes();
    if bytes.len() == 3 {
        return ["CON", "PRN", "AUX", "NUL"].iter().any(|res| stem.eq_ignore_ascii_case(res));
    }
    bytes.len() == 4
        && (stem[..3].eq_ignore_ascii_case("COM") || stem[..3].eq_ignore_ascii_case("LPT"))
        && (b'1'..=b'9').contains(&bytes[3])
}

fn is_protected_repo_dir(seg: &str) -> bool {
    if seg.is_ascii() {
        return seg.eq_ignore_ascii_case(".git") || seg.eq_ignore_ascii_case(".refit");
    }
    let folded: String = seg.chars().filter(|&ch| !is_combining_mark(ch))
        .map(|ch| ch.to_ascii_lowercase()).collect();
    folded == ".git" || folded == ".refit"
}

/// Validate and normalize a repo-relative path across OSes.
pub fn normalize_rel(rel: &str) -> Option<String> {
    if rel.is_empty() || rel.contains(':') || rel.bytes().any(|byte| byte < 0x20) { return None; }
    let slashed = normalize_nfd_to_nfc(rel).0.replace('\\', "/");
    if slashed.starts_with('/') { return None; }
    let mut stack: SmallVec<[&str; 16]> = SmallVec::new();
    for seg in slashed.split('/') {
        if seg.is_empty() || seg == "." { continue; }
        if seg == ".." || validate_path_segment(seg).is_err() { return None; }
        stack.push(seg);
    }
    (!stack.is_empty()).then(|| stack.join("/"))
}

pub fn check_rel(rel: &str) -> bool { normalize_rel(rel).is_some() }

fn verify_canonical_in_root(root: &Path, cur: &Path) -> bool {
    let Ok(canon) = dunce::canonicalize(cur) else { return false };
    if same_file::is_same_file(&canon, root).unwrap_or(false) { return true; }
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

fn normalize_for_prefix(path: &Path) -> String {
    let raw = path.to_string_lossy();
    if let Ok(norm) = normalize_os_path(&raw, false) { return norm.normalized; }
    let slashed = raw.replace('\\', "/");
    let stripped = strip_unc_verbatim(&slashed)
        .map(|rest| format!("//{rest}"))
        .or_else(|| slashed.strip_prefix("//?/").map(str::to_string))
        .unwrap_or(slashed);
    let cased = if is_drive_prefix(&stripped) {
        format!("{}{}", stripped.as_bytes()[0].to_ascii_uppercase() as char, &stripped[1..])
    } else {
        stripped
    };
    cased.trim_end_matches('/').to_string()
}

/// Convert an absolute or tool-reported path to a normalized repo-relative forward-slash path.
pub fn rel_to_root(repo_root: &Path, path: &Path) -> Option<String> {
    if same_file::is_same_file(repo_root, path).unwrap_or(false) { return None; }
    if let (Ok(c_root), Ok(c_path)) = (dunce::canonicalize(repo_root), dunce::canonicalize(path))
        && let Ok(rel) = c_path.strip_prefix(&c_root)
    {
        let text = rel.to_string_lossy().replace('\\', "/");
        if !text.is_empty() { return Some(text); }
    }
    let root_norm = dunce::canonicalize(repo_root)
        .map_or_else(|_| normalize_for_prefix(repo_root), |canon| normalize_for_prefix(&canon));
    let path_norm = normalize_for_prefix(path);
    if let Some(rest) = strip_root_str(&path_norm, &root_norm) { return Some(rest.to_string()); }
    (!path.is_absolute()).then(|| normalize_rel(&path.to_string_lossy())).flatten()
}

fn strip_root_str<'a>(path_norm: &'a str, root_norm: &str) -> Option<&'a str> {
    if root_norm.is_empty() { return None; }
    if let Some(rel) = path_norm.strip_prefix(root_norm).and_then(|rem| rem.strip_prefix('/'))
        && !rel.is_empty()
    {
        return Some(rel);
    }
    if (cfg!(windows) || is_drive_prefix(root_norm))
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

#[inline] pub fn strip_utf8_bom(text: &str) -> &str { text.trim_prefix('\u{FEFF}') }
#[inline] pub fn trim_crlf(line: &str) -> &str { line.trim_end_matches(['\r', '\n']) }
#[inline] pub fn pwsh_utf8_preamble() -> &'static str {
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
}

#[cfg(test)]
use crate::testutil::{TempRepo, TestRng};

#[cfg(all(test, windows))]
fn junction_reparse_buf(target: &Path) -> Vec<u8> {
    let canon = dunce::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    let raw = canon.to_string_lossy();
    let print_s = raw.trim_prefix(r"\\?\");
    let sub_s = format!(r"\??\{print_s}");
    let (sub, prt): (Vec<u16>, Vec<u16>) =
        (sub_s.encode_utf16().collect(), print_s.encode_utf16().collect());
    let (sb, pb) = ((sub.len() * 2) as u16, (prt.len() * 2) as u16);
    let mut buf = Vec::with_capacity(20 + (sb + pb) as usize);
    buf.extend_from_slice(&0xA000_0003u32.to_le_bytes());
    for word in [8 + sb + 2 + pb + 2, 0u16, 0, sb, sb + 2, pb] {
        buf.extend_from_slice(&word.to_le_bytes());
    }
    for slice in [&sub[..], &[0u16][..], &prt[..], &[0u16][..]] {
        for &unit in slice {
            buf.extend_from_slice(&unit.to_le_bytes());
        }
    }
    buf
}

#[cfg(all(test, windows))]
fn create_junction(target: &Path, link: &Path) -> bool {
    use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn DeviceIoControl(
            h: *mut std::ffi::c_void, code: u32, in_buf: *const std::ffi::c_void, in_len: u32,
            out_buf: *mut std::ffi::c_void, out_len: u32, ret: *mut u32, ov: *mut std::ffi::c_void,
        ) -> i32;
    }
    if fs::create_dir(link).is_err() { return false; }
    let buf = junction_reparse_buf(target);
    let mut opts = fs::OpenOptions::new();
    opts.write(true).custom_flags(0x0220_0000);
    let Ok(dir) = opts.open(link) else {
        let _ = fs::remove_dir(link);
        return false;
    };
    let mut ret = 0u32;
    // SAFETY: `dir` is an open directory handle and `buf` is a valid REPARSE_DATA_BUFFER.
    let ok = unsafe {
        DeviceIoControl(
            dir.as_raw_handle(), 0x0009_00A4, buf.as_ptr().cast(), buf.len() as u32,
            std::ptr::null_mut(), 0, &mut ret, std::ptr::null_mut(),
        ) != 0
    };
    drop(dir);
    if !ok { let _ = fs::remove_dir(link); }
    ok
}

#[cfg(test)]
fn try_symlink(target: &Path, link: &Path, is_dir: bool) -> bool {
    cfg_select! {
        unix => { let _ = is_dir; std::os::unix::fs::symlink(target, link).is_ok() }
        windows => {
            if is_dir {
                std::os::windows::fs::symlink_dir(target, link).is_ok() || create_junction(target, link)
            } else {
                std::os::windows::fs::symlink_file(target, link).is_ok()
            }
        }
    }
}

#[cfg(test)]
#[test]
fn cross_os_paths() {
    let base = TempRepo::with_files(
        "paths",
        &[("repo/.git/HEAD", "ref: refs/heads/main\n"), ("outside/secret.rs", "const S: u8 = 1;\n")],
    );
    let (repo, outside) = (base.join("repo"), base.join("outside"));
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

    let mut tested_links = 0usize;
    let links = [
        (outside.join("secret.rs"), "esc.rs", false),
        (outside.clone(), "esc_dir", true),
        (outside.join("gone.rs"), "dangle.rs", false),
        (repo.join(".git/HEAD"), "git_link", false),
    ];
    for (target, name, is_dir) in links {
        let link_path = repo.join(name);
        if try_symlink(&target, &link_path, is_dir) {
            tested_links += 1;
            let probe = if is_dir { format!("{name}/child.rs") } else { name.into() };
            assert!(!check_target_in_repo(&canon, &probe));
            if is_dir {
                let _ = fs::remove_dir(&link_path);
            } else {
                let _ = fs::remove_file(&link_path);
            }
        }
    }
    assert!(tested_links >= 1, "expected at least one symlink or junction escape check");
    assert!(!check_target_in_repo(&base.join("missing_root"), "src/main.rs"));

    let norm_cases = [
        (r"src\core/sub\a.rs", "src/core/sub/a.rs"),
        ("./src/a.rs", "src/a.rs"),
    ];
    for (raw, want) in norm_cases {
        assert_eq!(normalize_rel(raw).as_deref(), Some(want));
        assert!(check_target_in_repo(&canon, raw));
    }

    let valid_paths = [
        "src/main.rs",
        "pkg/my file.py",
        "src/日本語/café.rs",
        "app/(marketing)/[slug]/page.tsx",
        "packages/@scope/pkg/index.ts",
        ".github/workflows/ci.yml",
        "a/b/c/d/e/f/deep.go",
    ];
    for ok in valid_paths {
        assert!(check_rel(ok) && check_target_in_repo(&canon, ok), "accept `{ok}`");
    }
    assert_eq!(rel_to_root(&canon, &canon), None);
    assert_eq!(rel_to_root(&canon, &canon.join("src/a.rs")).as_deref(), Some("src/a.rs"));
    assert_eq!(rel_to_root(&canon, &outside.join("secret.rs")), None);
}

#[cfg(test)]
#[test]
fn cross_os_path_norm() {
    let cases = [
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
    ];
    for (input, exp, wsl, nfd) in cases {
        let got = normalize_os_path(input, false).unwrap();
        assert_eq!(
            (got.normalized.as_str(), got.wsl_9p_warning, got.nfd_normalized),
            (exp, wsl, nfd)
        );
    }

    let rejects = [
        ("", false, PathReject::Empty),
        ("a\0b", false, PathReject::ControlOrNul),
        ("a\r\nb", false, PathReject::ControlOrNul),
        (r#"a/"b"c"#, false, PathReject::UnbalancedQuote),
        ("src/CON.rs", false, PathReject::ReservedDeviceName),
        ("src/a.rs.", false, PathReject::TrailingDotOrSpace),
        ("../x.rs", false, PathReject::TraversalEscape),
        (".gi\u{0307}t/x", false, PathReject::ProtectedDir),
        ("src/a.rs:ads", false, PathReject::AlternateDataStream),
        ("a & b", true, PathReject::ShellMetachar),
        ("a | b", true, PathReject::ShellMetachar),
        ("%PATH%", true, PathReject::ShellMetachar),
        ("a^b", true, PathReject::ShellMetachar),
    ];
    for (input, safe, want) in rejects {
        assert_eq!(normalize_os_path(input, safe).unwrap_err(), want, "in={input}");
    }

    let rel_cases = [
        (r"D:\r", r"\\?\d:\r\src\a.py", Some("src/a.py")),
        ("/mnt/d/r", r"D:\r\src\b.py", Some("src/b.py")),
        (r"D:\r\ab", "D:\\r\\a\u{00E9}\\x.rs", None),
    ];
    for (root, target, want) in rel_cases {
        assert_eq!(rel_to_root(Path::new(root), Path::new(target)).as_deref(), want);
    }
    assert_eq!((strip_utf8_bom("\u{FEFF}hi"), trim_crlf("x\r\n")), ("hi", "x"));
    assert!(pwsh_utf8_preamble().contains("UTF8Encoding"));
}

#[cfg(test)]
#[test]
fn cross_os_path_fuzz() {
    let repo = TempRepo::new("path_prop");
    let canon = canonical_root(repo.path());
    let pfx_list = [
        "", r"\\?\c:\", r"D:\", "/d/", "/mnt/d/", r"\\srv\sh\", r"\\wsl.localhost\u\", "/",
    ];
    let segs = [
        "src", "pkg", ".", "sub", "cafe\u{0301}", "f.rs", "..", "CON", "a:ads", ".git", "x.",
    ];
    let mut rng = TestRng::new(0x50A7_C0DE);

    for _ in 0..500 {
        let pfx = pfx_list[rng.next_usize(pfx_list.len())];
        let mut raw = pfx.to_string();
        for step in 0..=rng.next_usize(4) {
            if step > 0 {
                raw.push(if rng.next_usize(2) == 0 { '/' } else { '\\' });
            }
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
