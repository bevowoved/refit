use crate::types::{SandboxResult, SelfMetrics};
use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn host_os_tag() -> String {
    let os = match std::env::consts::OS {
        "macos" => "macos",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{os}-{arch}")
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("create private directory {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o700);
        fs::set_permissions(path, perms)
            .with_context(|| format!("chmod 0700 {}", path.display()))?;
    }
    Ok(())
}

pub fn atomic_replace_file(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent dir {}", parent.display()))?;

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_name = format!(
        ".refit.tmp.{pid}.{seq}.{}",
        target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("blob")
    );
    let tmp_path = parent.join(tmp_name);

    {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("create temp file {}", tmp_path.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write temp file {}", tmp_path.display()))?;
        f.sync_data()
            .with_context(|| format!("sync temp file {}", tmp_path.display()))?;
    }

    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{
            MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
        };
        let src: Vec<u16> = tmp_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let dst: Vec<u16> = target
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut last_err = None;
        for attempt in 0..5 {
            let ok = unsafe {
                MoveFileExW(
                    src.as_ptr(),
                    dst.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            };
            if ok != 0 {
                return Ok(());
            }
            last_err = Some(std::io::Error::last_os_error());
            std::thread::sleep(Duration::from_millis(10 * (attempt + 1)));
        }
        let _ = fs::remove_file(&tmp_path);
        anyhow::bail!(
            "MoveFileExW failed for {}: {:?}",
            target.display(),
            last_err
        );
    }

    #[cfg(unix)]
    {
        fs::rename(&tmp_path, target).with_context(|| {
            let _ = fs::remove_file(&tmp_path);
            format!("atomic rename {} -> {}", tmp_path.display(), target.display())
        })?;
        Ok(())
    }
}

#[cfg(windows)]
pub fn self_metrics() -> SelfMetrics {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    unsafe {
        let proc_h = GetCurrentProcess();
        let mut pmc: PROCESS_MEMORY_COUNTERS = zeroed();
        pmc.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let rss_bytes = if GetProcessMemoryInfo(proc_h, &mut pmc, pmc.cb) != 0 {
            pmc.WorkingSetSize as u64
        } else {
            0
        };

        let mut creation: FILETIME = zeroed();
        let mut exit: FILETIME = zeroed();
        let mut kernel: FILETIME = zeroed();
        let mut user: FILETIME = zeroed();
        let cpu_ms = if GetProcessTimes(proc_h, &mut creation, &mut exit, &mut kernel, &mut user) != 0 {
            let k = ((kernel.dwHighDateTime as u64) << 32) | (kernel.dwLowDateTime as u64);
            let u = ((user.dwHighDateTime as u64) << 32) | (user.dwLowDateTime as u64);
            (k + u) / 10_000
        } else {
            0
        };

        SelfMetrics { rss_bytes, cpu_ms }
    }
}

#[cfg(target_os = "linux")]
pub fn self_metrics() -> SelfMetrics {
    let rss_bytes = fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0);
    let cpu_ms = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let ms = (ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000;
            ms.max(0) as u64
        })
        .unwrap_or(0);
    SelfMetrics { rss_bytes, cpu_ms }
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn self_metrics() -> SelfMetrics {
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let ms = (ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000;
            SelfMetrics {
                rss_bytes: u.max_rss().max(0) as u64,
                cpu_ms: ms.max(0) as u64,
            }
        })
        .unwrap_or_default()
}

pub fn run_sandboxed_cmd(
    program: &str,
    args: &[String],
    cwd: &Path,
    env_overrides: &[(String, String)],
    clear_env: bool,
    timeout_ms: u64,
) -> Result<SandboxResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if clear_env {
        cmd.env_clear();
    }
    for (k, v) in env_overrides {
        cmd.env(k, v);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                let _ = nix::unistd::setpgid(
                    nix::unistd::Pid::from_raw(0),
                    nix::unistd::Pid::from_raw(0),
                );
                Ok(())
            });
        }
    }

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn sandboxed command `{program}` in {}", cwd.display()))?;

    #[cfg(windows)]
    let job_handle = attach_windows_job(&child);

    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();

    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = out_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = err_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        String::from_utf8_lossy(&buf).into_owned()
    });

    let deadline = Duration::from_millis(timeout_ms.max(100));
    let mut timed_out = false;
    let exit_code;

    loop {
        if let Some(status) = child.try_wait()? {
            exit_code = status.code().unwrap_or(-1);
            break;
        }
        if started.elapsed() >= deadline {
            timed_out = true;
            #[cfg(windows)]
            {
                if let Some(job) = job_handle {
                    unsafe {
                        windows_sys::Win32::System::JobObjects::TerminateJobObject(job, 1);
                    }
                }
                let _ = child.kill();
            }
            #[cfg(unix)]
            {
                let pgid = nix::unistd::Pid::from_raw(child.id() as i32);
                let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
                std::thread::sleep(Duration::from_millis(100));
                let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
                let _ = child.kill();
            }
            let _ = child.wait();
            exit_code = -1;
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let wall_ms = started.elapsed().as_millis() as u64;
    let (cpu_ms, peak_rss_bytes) = {
        #[cfg(windows)]
        {
            query_and_close_windows_job(job_handle)
        }
        #[cfg(unix)]
        {
            query_unix_children_rusage()
        }
    };

    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();

    Ok(SandboxResult {
        exit_code,
        timed_out,
        stdout,
        stderr,
        wall_ms,
        cpu_ms,
        peak_rss_bytes,
    })
}

#[cfg(windows)]
fn attach_windows_job(child: &std::process::Child) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use std::mem::{size_of, zeroed};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            CloseHandle(job);
            return None;
        }
        let proc_h = child.as_raw_handle();
        if AssignProcessToJobObject(job, proc_h) == 0 {
            CloseHandle(job);
            return None;
        }
        Some(job)
    }
}

#[cfg(windows)]
fn query_and_close_windows_job(
    job_handle: Option<windows_sys::Win32::Foundation::HANDLE>,
) -> (u64, u64) {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };

    let Some(job) = job_handle else {
        return (0, 0);
    };
    unsafe {
        let mut acct: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = zeroed();
        let cpu_ms = if QueryInformationJobObject(
            job,
            JobObjectBasicAccountingInformation,
            &mut acct as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
            std::ptr::null_mut(),
        ) != 0
        {
            let ticks = (acct.TotalUserTime + acct.TotalKernelTime).max(0) as u64;
            ticks / 10_000
        } else {
            0
        };

        let mut ext: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        let peak_rss = if QueryInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &mut ext as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        ) != 0
        {
            ext.PeakJobMemoryUsed as u64
        } else {
            0
        };

        CloseHandle(job);
        (cpu_ms, peak_rss)
    }
}

#[cfg(unix)]
fn query_unix_children_rusage() -> (u64, u64) {
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_CHILDREN)
        .map(|u| {
            let ut = u.user_time();
            let st = u.system_time();
            let cpu_ms = ((ut.tv_sec() + st.tv_sec()) * 1000
                + (ut.tv_usec() + st.tv_usec()) / 1000)
                .max(0) as u64;
            let raw_rss = u.max_rss().max(0) as u64;
            #[cfg(target_os = "linux")]
            let rss_bytes = raw_rss * 1024;
            #[cfg(not(target_os = "linux"))]
            let rss_bytes = raw_rss;
            (cpu_ms, rss_bytes)
        })
        .unwrap_or((0, 0))
}
