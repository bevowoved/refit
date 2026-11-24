use crate::types::{SandboxResult, SelfMetrics};
use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(windows)]
pub fn self_metrics() -> SelfMetrics {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    // SAFETY: `GetCurrentProcess` returns the pseudo-handle for this process; both memory
    // counter structs and `FILETIME` outputs are zeroed POD with exact struct byte sizes.
    unsafe {
        let proc_h = GetCurrentProcess();
        let mut ex2: PROCESS_MEMORY_COUNTERS_EX2 = zeroed();
        ex2.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32;
        let rss_bytes = if GetProcessMemoryInfo(
            proc_h,
            &mut ex2 as *mut PROCESS_MEMORY_COUNTERS_EX2 as *mut PROCESS_MEMORY_COUNTERS,
            ex2.cb,
        ) != 0
            && ex2.PrivateWorkingSetSize > 0
        {
            ex2.PrivateWorkingSetSize as u64
        } else {
            let mut pmc: PROCESS_MEMORY_COUNTERS = zeroed();
            pmc.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if GetProcessMemoryInfo(proc_h, &mut pmc, pmc.cb) != 0 {
                pmc.WorkingSetSize as u64
            } else {
                0
            }
        };

        let (mut cr, mut ex, mut kn, mut us): (FILETIME, FILETIME, FILETIME, FILETIME) =
            (zeroed(), zeroed(), zeroed(), zeroed());
        let cpu_ms = if GetProcessTimes(proc_h, &mut cr, &mut ex, &mut kn, &mut us) != 0 {
            let k = ((kn.dwHighDateTime as u64) << 32) | (kn.dwLowDateTime as u64);
            let u = ((us.dwHighDateTime as u64) << 32) | (us.dwLowDateTime as u64);
            (k + u) / 10_000
        } else {
            0
        };
        SelfMetrics { rss_bytes, cpu_ms }
    }
}

#[cfg(unix)]
pub fn self_metrics() -> SelfMetrics {
    let usage = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF).ok();
    let cpu_ms = usage
        .as_ref()
        .map(|u| {
            let (ut, st) = (u.user_time(), u.system_time());
            ((ut.tv_sec() + st.tv_sec()) * 1000 + (ut.tv_usec() + st.tv_usec()) / 1000).max(0) as u64
        })
        .unwrap_or(0);

    #[cfg(target_os = "linux")]
    let rss_bytes = linux_uss_bytes().unwrap_or_else(|| {
        usage.map(|u| u.max_rss().max(0) as u64 * 1024).unwrap_or(0)
    });

    #[cfg(not(target_os = "linux"))]
    let rss_bytes = usage.map(|u| u.max_rss().max(0) as u64).unwrap_or(0);

    SelfMetrics { rss_bytes, cpu_ms }
}

#[cfg(target_os = "linux")]
fn linux_uss_bytes() -> Option<u64> {
    if let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        let mut dirty_kb = 0u64;
        let mut clean_kb = 0u64;
        let mut lazy_kb = 0u64;
        let mut found = false;
        for line in rollup.lines() {
            if let Some(rest) = line.strip_prefix("Private_Dirty:") {
                dirty_kb = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
                found = true;
            } else if let Some(rest) = line.strip_prefix("Private_Clean:") {
                clean_kb = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
                found = true;
            } else if let Some(rest) = line.strip_prefix("LazyFree:") {
                lazy_kb = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
        }
        if found {
            return Some(dirty_kb.saturating_add(clean_kb).saturating_sub(lazy_kb) * 1024);
        }
    }
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn spawn_pipe<R: Read + Send + 'static>(pipe: Option<R>, cap_bytes: usize) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut p) = pipe {
            let mut chunk = [0u8; 8192];
            loop {
                match p.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if buf.len() < cap_bytes {
                            let keep = (cap_bytes - buf.len()).min(n);
                            buf.extend_from_slice(&chunk[..keep]);
                        }
                    }
                }
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    })
}

pub fn run_cmd(
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
    set_pgid(&mut cmd);
    #[cfg(unix)]
    let (unix_cpu_before, _) = unix_rusage();

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn `{program}` in {}", cwd.display()))?;

    #[cfg(windows)]
    let job_handle = attach_job(&child);

    let cap = super::budget::budget().pipe_cap_bytes;
    let out_thread = spawn_pipe(child.stdout.take(), cap);
    let err_thread = spawn_pipe(child.stderr.take(), cap);

    let deadline = Duration::from_millis(timeout_ms.max(100));
    let mut timed_out = false;
    let exit_code = match wait_timeout::ChildExt::wait_timeout(&mut child, deadline)? {
        Some(status) => status.code().unwrap_or(-1),
        None => {
            timed_out = true;
            #[cfg(windows)]
            kill_child(&mut child, job_handle);
            #[cfg(unix)]
            kill_child(&mut child);
            -1
        }
    };

    let wall_ms = started.elapsed().as_millis() as u64;
    #[cfg(windows)]
    let (cpu_ms, peak_rss_bytes) = close_job(job_handle);
    #[cfg(unix)]
    let (cpu_ms, peak_rss_bytes) = {
        let (after_cpu, rss) = unix_rusage();
        (after_cpu.saturating_sub(unix_cpu_before), rss)
    };

    Ok(SandboxResult {
        exit_code,
        timed_out,
        stdout: out_thread.join().unwrap_or_default(),
        stderr: err_thread.join().unwrap_or_default(),
        wall_ms,
        cpu_ms,
        peak_rss_bytes,
    })
}

#[cfg(windows)]
fn kill_child(child: &mut Child, job: Option<windows_sys::Win32::Foundation::HANDLE>) {
    if let Some(h) = job {
        // SAFETY: `h` is a valid Job Object handle owned by this `run_cmd` invocation.
        unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(h, 1);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn set_pgid(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: `setpgid(0, 0)` is async-signal-safe and runs solely in the forked child.
    unsafe {
        cmd.pre_exec(|| {
            let zero = nix::unistd::Pid::from_raw(0);
            let _ = nix::unistd::setpgid(zero, zero);
            Ok(())
        });
    }
}

#[cfg(unix)]
fn kill_child(child: &mut Child) {
    let pgid = nix::unistd::Pid::from_raw(child.id() as i32);
    let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
    let gone = wait_timeout::ChildExt::wait_timeout(child, Duration::from_millis(100))
        .ok()
        .flatten()
        .is_some();
    if !gone {
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg(windows)]
fn attach_job(child: &Child) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use std::mem::{size_of, zeroed};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    // SAFETY: Win32 Job Object creation/assignment with zeroed POD struct and valid child handle;
    // any failure path closes the handle before returning `None`.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ptr = &info as *const _ as *const core::ffi::c_void;
        let len = size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;

        if SetInformationJobObject(job, JobObjectExtendedLimitInformation, ptr, len) == 0 {
            CloseHandle(job);
            return None;
        }
        if AssignProcessToJobObject(job, child.as_raw_handle()) == 0 {
            CloseHandle(job);
            return None;
        }
        Some(job)
    }
}

#[cfg(windows)]
fn close_job(job_handle: Option<windows_sys::Win32::Foundation::HANDLE>) -> (u64, u64) {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };

    let Some(job) = job_handle else { return (0, 0) };
    // SAFETY: `job` is a valid owned Job Object handle queried with exact POD buffer sizes and
    // unconditionally closed via `CloseHandle` before returning.
    unsafe {
        let mut acct: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = zeroed();
        let a_ptr = &mut acct as *mut _ as *mut core::ffi::c_void;
        let a_len = size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32;
        let a_ok = QueryInformationJobObject(
            job,
            JobObjectBasicAccountingInformation,
            a_ptr,
            a_len,
            std::ptr::null_mut(),
        );
        let cpu_ms = if a_ok != 0 {
            ((acct.TotalUserTime + acct.TotalKernelTime).max(0) as u64) / 10_000
        } else {
            0
        };

        let mut ext: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        let e_ptr = &mut ext as *mut _ as *mut core::ffi::c_void;
        let e_len = size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
        let e_ok = QueryInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            e_ptr,
            e_len,
            std::ptr::null_mut(),
        );
        let peak_rss = if e_ok != 0 { ext.PeakJobMemoryUsed as u64 } else { 0 };

        CloseHandle(job);
        (cpu_ms, peak_rss)
    }
}

#[cfg(unix)]
fn unix_rusage() -> (u64, u64) {
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_CHILDREN)
        .map(|u| {
            let (ut, st) = (u.user_time(), u.system_time());
            let cpu = ((ut.tv_sec() + st.tv_sec()) * 1000 + (ut.tv_usec() + st.tv_usec()) / 1000)
                .max(0) as u64;
            let raw = u.max_rss().max(0) as u64;
            #[cfg(target_os = "linux")]
            let rss = raw * 1024;
            #[cfg(not(target_os = "linux"))]
            let rss = raw;
            (cpu, rss)
        })
        .unwrap_or((0, 0))
}
