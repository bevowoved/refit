use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct Worktree {
    pub core_repo: PathBuf,
    pub worktree_path: PathBuf,
    pub branch: String,
    pub carried_dirty_changes: bool,
}

#[cfg(test)]
pub fn make_worktree(repo_root: &Path, run_id: &str) -> Result<Worktree> {
    open_worktree(repo_root, run_id, false)
}

pub fn open_worktree(
    repo_root: &Path,
    run_id: &str,
    include_dirty: bool,
) -> Result<Worktree> {
    let git_check = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(repo_root)
        .output()
        .context("check git worktree support")?;
    if !git_check.status.success() {
        anyhow::bail!(
            "{} is not a git repository; cannot create isolated --worktree",
            repo_root.display()
        );
    }

    let safe_id: String = run_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect();
    let mut branch = format!("refit/{safe_id}");
    let parent = repo_root.parent().unwrap_or_else(|| Path::new("."));
    let repo_name = repo_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo");
    let mut worktree_path = parent.join(format!(".{repo_name}-worktree-{safe_id}"));

    if worktree_path.exists() {
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(repo_root)
            .output();
        let suffix = crate::platform::now_ms();
        branch = format!("refit/{safe_id}-{suffix}");
        worktree_path = parent.join(format!(".{repo_name}-worktree-{safe_id}-{suffix}"));
    }

    // Check for dirty working tree so uncommitted work is handled explicitly.
    let dirty_patch = if include_dirty {
        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(repo_root)
            .output()
            .ok();
        let is_dirty = status
            .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
            .unwrap_or(false);
        if is_dirty {
            // Non-destructive `git stash create` captures tracked edits without modifying core working tree.
            Command::new("git")
                .args(["stash", "create", "refit-worktree-carryover"])
                .current_dir(repo_root)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        } else {
            None
        }
    } else {
        None
    };

    let out = Command::new("git")
        .args([
            "worktree",
            "add",
            "-b",
            &branch,
            &worktree_path.display().to_string(),
            "HEAD",
        ])
        .current_dir(repo_root)
        .output()
        .context("run git worktree add")?;

    if !out.status.success() {
        anyhow::bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }

    let mut carried_dirty_changes = false;
    if let Some(stash_oid) = dirty_patch {
        let apply = Command::new("git")
            .args(["stash", "apply", "--quiet", &stash_oid])
            .current_dir(&worktree_path)
            .output()
            .context("apply dirty worktree carryover stash")?;
        if !apply.status.success() {
            anyhow::bail!(
                "failed to carry over dirty tracked changes to worktree: {}",
                String::from_utf8_lossy(&apply.stderr).trim()
            );
        }
        carried_dirty_changes = true;
    }
    if include_dirty {
        // Also carry over non-ignored untracked files (`git ls-files --others --exclude-standard`).
        if let Ok(untracked) = Command::new("git")
            .args(["ls-files", "--others", "--exclude-standard"])
            .current_dir(repo_root)
            .output()
        {
            if untracked.status.success() {
                for line in String::from_utf8_lossy(&untracked.stdout).lines() {
                    let rel = line.trim().replace('\\', "/");
                    if rel.is_empty() || !crate::platform::check_rel(&rel) {
                        continue;
                    }
                    let src = repo_root.join(&rel);
                    let dst = worktree_path.join(&rel);
                    if src.is_file() {
                        if let Some(p) = dst.parent() {
                            let _ = std::fs::create_dir_all(p);
                        }
                        if std::fs::copy(&src, &dst).is_ok() {
                            carried_dirty_changes = true;
                        }
                    }
                }
            }
        }
    }

    Ok(Worktree {
        core_repo: repo_root.to_path_buf(),
        worktree_path,
        branch,
        carried_dirty_changes,
    })
}

pub fn diffstat(wt: &Worktree) -> String {
    Command::new("git")
        .args(["diff", "--stat"])
        .current_dir(&wt.worktree_path)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

pub fn list_worktrees(repo_root: &Path) -> Vec<String> {
    let Ok(out) = Command::new("git")
        .args(["worktree", "list", "--porcelain"])
        .current_dir(repo_root)
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut results = Vec::new();
    let mut cur_path: Option<String> = None;
    for line in text.lines() {
        if let Some(p) = line.strip_prefix("worktree ") {
            cur_path = Some(p.trim().to_string());
        } else if let Some(b) = line.strip_prefix("branch refs/heads/") {
            if b.starts_with("refit/") {
                if let Some(ref p) = cur_path {
                    results.push(format!("{} ({})", b.trim(), p));
                }
            }
        }
    }
    results
}

#[allow(dead_code)]
pub fn remove_worktree(wt: &Worktree) -> Result<()> {
    let _ = Command::new("git")
        .args([
            "worktree",
            "remove",
            "--force",
            &wt.worktree_path.display().to_string(),
        ])
        .current_dir(&wt.core_repo)
        .output();
    if wt.branch.starts_with("refit/") {
        let _ = Command::new("git")
            .args(["branch", "-D", &wt.branch])
            .current_dir(&wt.core_repo)
            .output();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn probe_44_worktree_iso() {
        let base = std::env::temp_dir().join(format!(
            "refit_wt_test_{}_{}",
            std::process::id(),
            crate::platform::now_ms()
        ));
        let core = base.join("core_repo");
        fs::create_dir_all(&core).unwrap();

        let run_git = |args: &[&str], dir: &Path| {
            let o = Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git installed");
            assert!(
                o.status.success(),
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&o.stderr)
            );
        };

        run_git(&["init", "-b", "main"], &core);
        run_git(&["config", "user.email", "refit@local.test"], &core);
        run_git(&["config", "user.name", "refit-test"], &core);

        let core_file = core.join("app.py");
        fs::write(&core_file, "def greet() -> str:\n    return 'core'\n").unwrap();
        run_git(&["add", "app.py"], &core);
        run_git(&["commit", "-m", "init core"], &core);

        let wt = make_worktree(&core, "test-run-42").unwrap();
        assert!(wt.worktree_path.exists());
        assert_eq!(wt.branch, "refit/test-run-42");

        let wt_file = wt.worktree_path.join("app.py");
        fs::write(&wt_file, "def greet() -> str:\n    return 'refactored'\n").unwrap();

        let core_after = fs::read_to_string(&core_file).unwrap();
        assert_eq!(core_after, "def greet() -> str:\n    return 'core'\n");
        let stat = diffstat(&wt);
        assert!(stat.contains("app.py"));

        remove_worktree(&wt).unwrap();
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn probe_90_worktree_dirty() {
        let base = std::env::temp_dir().join(format!(
            "refit_wt90_{}_{}",
            std::process::id(),
            crate::platform::now_ms()
        ));
        let core = base.join("core_repo");
        fs::create_dir_all(&core).unwrap();
        let run_git = |args: &[&str], dir: &Path| {
            let o = Command::new("git").args(args).current_dir(dir).output().unwrap();
            assert!(o.status.success());
        };
        run_git(&["init", "-b", "main"], &core);
        run_git(&["config", "user.email", "refit@local.test"], &core);
        run_git(&["config", "user.name", "refit-test"], &core);
        fs::write(core.join("app.py"), "X = 1\n").unwrap();
        run_git(&["add", "app.py"], &core);
        run_git(&["commit", "-m", "init"], &core);

        // Dirty uncommitted change in core repo.
        fs::write(core.join("app.py"), "X = 99\n").unwrap();
        let wt = open_worktree(&core, "dirty-run", true).unwrap();
        assert!(wt.carried_dirty_changes);
        assert_eq!(
            fs::read_to_string(wt.worktree_path.join("app.py"))
                .unwrap()
                .replace("\r\n", "\n"),
            "X = 99\n"
        );
        let listed = list_worktrees(&core);
        assert!(listed.iter().any(|s| s.contains("refit/dirty-run")));
        remove_worktree(&wt).unwrap();
        assert!(!wt.worktree_path.exists());
        let _ = fs::remove_dir_all(&base);
    }
}
