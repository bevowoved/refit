use crate::platform::{SidecarLock, check_rel, now_ms, paths::canonical_root};
use anyhow::{Context, Result};
use std::mem::DropGuard;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Debug, Clone)]
pub struct Worktree {
    pub core_repo: PathBuf,
    pub worktree_path: PathBuf,
    pub branch: String,
    pub carried_dirty_changes: bool,
}

fn git_run(dir: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))
}

fn git_checked(dir: &Path, args: &[&str], label: &str) -> Result<()> {
    let out = git_run(dir, args)?;
    if !out.status.success() {
        anyhow::bail!("{label}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

fn git_stdout(dir: &Path, args: &[&str]) -> Option<String> {
    let out = git_run(dir, args).ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn capture_dirty(repo_root: &Path) -> Option<String> {
    if git_stdout(repo_root, &["status", "--porcelain"])?.is_empty() { return None; }
    git_stdout(repo_root, &["stash", "create", "refit-worktree-carryover"]).filter(|oid| !oid.is_empty())
}

fn copy_untracked(repo_root: &Path, wt_path: &Path) -> bool {
    let Some(list) = git_stdout(repo_root, &["ls-files", "--others", "--exclude-standard"]) else {
        return false;
    };
    let mut copied = false;
    for line in list.lines() {
        let rel = line.trim().replace('\\', "/");
        if rel.is_empty() || !check_rel(&rel) { continue; }
        let (src, dst) = (repo_root.join(&rel), wt_path.join(&rel));
        if !src.is_file() { continue; }
        if let Some(parent) = dst.parent() { let _ = std::fs::create_dir_all(parent); }
        copied |= std::fs::copy(&src, &dst).is_ok();
    }
    copied
}

fn git_lock_path(repo_root: &Path) -> PathBuf {
    let dir = git_stdout(repo_root, &["rev-parse", "--git-common-dir"]).map_or_else(
        || repo_root.join(".git"),
        |common| {
            let path = PathBuf::from(common);
            if path.is_absolute() { path } else { repo_root.join(path) }
        },
    );
    dir.join("refit-git.lock")
}

pub fn open_worktree(repo_root: &Path, run_id: &str, include_dirty: bool) -> Result<Worktree> {
    if !git_run(repo_root, &["rev-parse", "--is-inside-work-tree"])?.status.success() {
        anyhow::bail!("{} is not a git repository", repo_root.display());
    }
    let safe: String = run_id.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '-' })
        .collect();
    let canonical_repo = canonical_root(repo_root);
    let _git_lock = SidecarLock::acquire(&git_lock_path(&canonical_repo))?;
    let parent = canonical_repo.parent().filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = canonical_repo.file_name().and_then(|seg| seg.to_str()).unwrap_or("repo");

    let (mut branch, mut wt_path) =
        (format!("refit/{safe}"), parent.join(format!(".{name}-worktree-{safe}")));
    if wt_path.exists() {
        let _ = git_run(&canonical_repo, &["worktree", "prune"]);
        let suffix = now_ms();
        branch = format!("refit/{safe}-{suffix}");
        wt_path = parent.join(format!(".{name}-worktree-{safe}-{suffix}"));
    }

    let dirty_oid = include_dirty.then(|| capture_dirty(&canonical_repo)).flatten();
    let wt_str = wt_path.display().to_string();
    git_checked(
        &canonical_repo, &["worktree", "add", "-b", &branch, &wt_str, "HEAD"], "git worktree add failed",
    )?;
    drop(_git_lock);

    let wt = Worktree {
        core_repo: canonical_repo, worktree_path: wt_path, branch, carried_dirty_changes: false,
    };
    let mut guard = DropGuard::new(wt, |item| { let _ = remove_worktree(&item); });

    if let Some(oid) = dirty_oid {
        git_checked(&guard.worktree_path, &["stash", "apply", "--quiet", &oid], "dirty carryover failed")?;
        guard.carried_dirty_changes = true;
    }
    if include_dirty {
        guard.carried_dirty_changes |= copy_untracked(repo_root, &guard.worktree_path);
    }
    Ok(DropGuard::dismiss(guard))
}

pub fn diffstat(wt: &Worktree) -> String {
    git_stdout(&wt.worktree_path, &["diff", "--stat"]).unwrap_or_default()
}

pub fn list_worktrees(repo_root: &Path) -> Vec<String> {
    let Some(text) = git_stdout(repo_root, &["worktree", "list", "--porcelain"]) else { return Vec::new() };
    let (mut out, mut cur_path) = (Vec::new(), None);
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            cur_path = Some(path.trim());
        } else if let Some(branch) = line.strip_prefix("branch refs/heads/")
            && let Some(path) = cur_path.filter(|_| branch.starts_with("refit/"))
        {
            out.push(format!("{} ({path})", branch.trim()));
        }
    }
    out
}

pub fn remove_worktree(wt: &Worktree) -> Result<()> {
    let _git_lock = SidecarLock::acquire(&git_lock_path(&wt.core_repo)).ok();
    let wt_str = wt.worktree_path.display().to_string();
    let _ = git_run(&wt.core_repo, &["worktree", "remove", "--force", &wt_str]);
    if wt.branch.starts_with("refit/") {
        let _ = git_run(&wt.core_repo, &["branch", "-D", &wt.branch]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;
    use std::fs;

    #[test]
    fn worktree_isolation_carryover() {
        let orig = "def greet() -> str:\n    return 'core'\n";
        let repo = TempRepo::git("wt_all", "core_repo", &[("app.py", orig)]);

        let wt = open_worktree(repo.path(), "test-run-42", false).unwrap();
        assert!(wt.worktree_path.exists() && wt.branch == "refit/test-run-42");
        fs::write(wt.worktree_path.join("app.py"), "def greet() -> str:\n    return 'ref'\n").unwrap();
        assert_eq!(repo.read("app.py"), orig);
        assert!(diffstat(&wt).contains("app.py"));

        repo.write("app.py", "X = 99\n");
        let dirty = open_worktree(repo.path(), "dirty-run", true).unwrap();
        assert!(dirty.carried_dirty_changes);
        let text = fs::read_to_string(dirty.worktree_path.join("app.py")).unwrap();
        assert_eq!(text.replace("\r\n", "\n"), "X = 99\n");
        assert!(list_worktrees(repo.path()).iter().any(|entry| entry.contains("refit/dirty-run")));

        for item in [&wt, &dirty] {
            remove_worktree(item).unwrap();
            assert!(!item.worktree_path.exists());
        }

        let dot = canonical_root(Path::new("."));
        assert!(dot.is_absolute() && dot.file_name().is_some());
        assert!(dot.parent().is_some_and(|dir| !dir.as_os_str().is_empty()));
    }
}
