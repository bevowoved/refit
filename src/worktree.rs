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

fn git_stdout(dir: &Path, args: &[&str]) -> Option<String> {
    let out = git_run(dir, args).ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn capture_dirty(repo_root: &Path) -> Option<String> {
    let status = git_stdout(repo_root, &["status", "--porcelain"])?;
    if status.is_empty() {
        return None;
    }

    let oid = git_stdout(repo_root, &["stash", "create", "refit-worktree-carryover"])?;
    (!oid.is_empty()).then_some(oid)
}

fn copy_untracked(repo_root: &Path, wt_path: &Path) -> bool {
    let Some(list) = git_stdout(repo_root, &["ls-files", "--others", "--exclude-standard"])
    else {
        return false;
    };

    let mut copied = false;
    for line in list.lines() {
        let rel = line.trim().replace('\\', "/");
        if rel.is_empty() || !crate::platform::check_rel(&rel) {
            continue;
        }

        let src = repo_root.join(&rel);
        let dst = wt_path.join(&rel);
        if !src.is_file() {
            continue;
        }

        if let Some(p) = dst.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        copied |= std::fs::copy(&src, &dst).is_ok();
    }
    copied
}

pub fn open_worktree(
    repo_root: &Path,
    run_id: &str,
    include_dirty: bool,
) -> Result<Worktree> {
    if !git_run(repo_root, &["rev-parse", "--is-inside-work-tree"])?.status.success() {
        anyhow::bail!("{} is not a git repository", repo_root.display());
    }

    let safe: String = run_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect();
    let parent = repo_root.parent().unwrap_or_else(|| Path::new("."));
    let name = repo_root.file_name().and_then(|s| s.to_str()).unwrap_or("repo");

    let mut branch = format!("refit/{safe}");
    let mut wt_path = parent.join(format!(".{name}-worktree-{safe}"));
    if wt_path.exists() {
        let _ = git_run(repo_root, &["worktree", "prune"]);
        let suffix = crate::platform::now_ms();
        branch = format!("refit/{safe}-{suffix}");
        wt_path = parent.join(format!(".{name}-worktree-{safe}-{suffix}"));
    }

    let dirty_oid = include_dirty.then(|| capture_dirty(repo_root)).flatten();
    let wt_str = wt_path.display().to_string();
    let add = git_run(repo_root, &["worktree", "add", "-b", &branch, &wt_str, "HEAD"])?;
    if !add.status.success() {
        anyhow::bail!("git worktree add failed: {}", String::from_utf8_lossy(&add.stderr).trim());
    }

    let wt = Worktree {
        core_repo: repo_root.to_path_buf(),
        worktree_path: wt_path,
        branch,
        carried_dirty_changes: false,
    };
    let mut guard = DropGuard::new(wt, |w| {
        let _ = remove_worktree(&w);
    });

    if let Some(oid) = dirty_oid {
        let apply = git_run(&guard.worktree_path, &["stash", "apply", "--quiet", &oid])?;
        if !apply.status.success() {
            let err = String::from_utf8_lossy(&apply.stderr);
            anyhow::bail!("dirty carryover failed: {}", err.trim());
        }
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
    let Some(text) = git_stdout(repo_root, &["worktree", "list", "--porcelain"]) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut cur_path: Option<&str> = None;
    for line in text.lines() {
        if let Some(p) = line.strip_prefix("worktree ") {
            cur_path = Some(p.trim());
        } else if let Some(b) = line.strip_prefix("branch refs/heads/")
            && let Some(p) = cur_path.filter(|_| b.starts_with("refit/"))
        {
            out.push(format!("{} ({p})", b.trim()));
        }
    }
    out
}

pub fn remove_worktree(wt: &Worktree) -> Result<()> {
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
    fn test_worktree_isolation_and_dirty_carryover() {
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
        assert!(list_worktrees(repo.path()).iter().any(|s| s.contains("refit/dirty-run")));

        for item in [&wt, &dirty] {
            remove_worktree(item).unwrap();
            assert!(!item.worktree_path.exists());
        }
    }
}
