//! Process execution, OS resource budgeting, allocator tuning, and memory admission.

use crate::types::{SandboxResult, SelfMetrics};
use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};
use rayon::prelude::*;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;
const CONSTRAINED_MEM_BYTES: u64 = 384 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode { InlineConstrained, BoundedBalanced }

#[derive(Debug, Clone, Copy)]
pub struct ResourceBudget {
    pub effective_vcpus: usize, pub effective_mem_bytes: u64, pub mode: ExecMode,
    pub http_pool_max: usize, pub pipe_cap_bytes: usize, pub sse_line_cap_bytes: usize,
    pub mem_gate_bytes: usize, pub mcp_blocking_permits: usize, pub rayon_threads: usize,
}

impl ResourceBudget {
    #[inline] pub fn is_constrained(&self) -> bool { self.mode == ExecMode::InlineConstrained }

    pub fn from_limits(vcpus: usize, mem_bytes: u64) -> Self {
        let effective_vcpus = vcpus.max(1);
        let constrained = effective_vcpus <= 1 || mem_bytes <= CONSTRAINED_MEM_BYTES;
        let mode = if constrained { ExecMode::InlineConstrained } else { ExecMode::BoundedBalanced };
        let (http_pool_max, pipe_cap_bytes, sse_line_cap_bytes, mem_gate_bytes, permits, rayon_threads) =
            if constrained {
                (2, 4 * MIB as usize, 2 * MIB as usize, 16 * MIB as usize, 1, 0)
            } else {
                let gate_mb = ((mem_bytes / MIB) / 8).clamp(16, 64) as usize;
                (effective_vcpus.clamp(4, 16), 32 * MIB as usize, 16 * MIB as usize,
                    gate_mb * MIB as usize, effective_vcpus.clamp(2, 4), effective_vcpus.min(8))
            };
        Self {
            effective_vcpus, effective_mem_bytes: mem_bytes, mode, http_pool_max, pipe_cap_bytes,
            sse_line_cap_bytes, mem_gate_bytes, mcp_blocking_permits: permits, rayon_threads,
        }
    }
}

static BUDGET: OnceLock<ResourceBudget> = OnceLock::new();
pub fn budget() -> &'static ResourceBudget { BUDGET.get_or_init(detect_budget) }

#[cfg(feature = "alloc-mimalloc")]
mod alloc_ops {
    use super::{MIB, ResourceBudget};

    unsafe extern "C" {
        fn mi_collect(force: bool);
        fn mi_option_set(option: core::ffi::c_int, value: core::ffi::c_long);
    }

    pub(super) fn name() -> &'static str { "mimalloc" }

    pub(super) fn init(bdg: &ResourceBudget) {
        // SAFETY: Runs once in main before spawning workers; indices match mimalloc v3.
        unsafe {
            mi_option_set(5, 1);
            mi_option_set(15, 75);
            if bdg.effective_mem_bytes <= 256 * MIB {
                mi_option_set(4, 0);
                mi_option_set(43, 0);
            }
        }
    }

    pub(super) fn collect() {
        // SAFETY: `mi_collect(true)` is thread-safe and purges abandoned segments.
        unsafe {
            mi_collect(true);
        }
    }
}

#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
mod alloc_ops {
    use super::ResourceBudget;

    unsafe extern "C" {
        fn _rjem_mallctl(
            name: *const core::ffi::c_char,
            oldp: *mut core::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut core::ffi::c_void,
            newlen: usize,
        ) -> core::ffi::c_int;
    }

    pub(super) fn name() -> &'static str { "jemalloc" }
    pub(super) fn init(bdg: &ResourceBudget) { let _ = bdg.effective_mem_bytes; }

    pub(super) fn purge_all() -> core::ffi::c_int {
        // SAFETY: `arena.4096.purge` (MALLCTL_ARENAS_ALL) is a valid void mallctl.
        unsafe {
            _rjem_mallctl(
                c"arena.4096.purge".as_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
            )
        }
    }

    pub(super) fn collect() {
        let _ = tikv_jemalloc_ctl::epoch::advance();
        let _ = purge_all();
    }
}

#[cfg(not(any(feature = "alloc-mimalloc", all(feature = "alloc-jemalloc", not(target_env = "msvc")))))]
mod alloc_ops {
    use super::ResourceBudget;

    pub(super) fn name() -> &'static str { "system" }
    pub(super) fn init(bdg: &ResourceBudget) { let _ = bdg.effective_mem_bytes; }
    pub(super) fn collect() {}
}

pub fn allocator_name() -> &'static str { alloc_ops::name() }
/// Tune the global allocator once before spawning worker threads.
pub fn init_allocator(bdg: &ResourceBudget) { alloc_ops::init(bdg); }
/// Release purged allocator pages back to the OS.
pub fn collect_os_memory() { alloc_ops::collect(); }

/// Returns `true` if Linux PSI `memory.pressure` indicates direct-reclaim stall pressure.
pub fn under_memory_pressure() -> bool {
    cfg_select! {
        target_os = "linux" => std::fs::read_to_string("/proc/pressure/memory")
            .ok().and_then(|raw| parse_psi_some_avg10(&raw)).is_some_and(|val| val > 5.0),
        _ => false,
    }
}

/// Unified Linux `/proc/*` `Key: <n> kB` field reader.
#[cfg(target_os = "linux")]
fn parse_proc_kb_field(raw: &str, prefix: &str) -> Option<u64> {
    raw.lines().find_map(|line| line.strip_prefix(prefix))
        .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kb| kb.saturating_mul(1024))
}

fn detect_budget() -> ResourceBudget {
    let host_cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let (mut vcpus, mut mem_bytes) = (host_cpus, host_memory_bytes().unwrap_or(4 * 1024 * MIB));
    let (os_cpus, os_mem) = cfg_select! {
        target_os = "linux" => detect_linux_cgroup_limits(),
        windows => detect_windows_job_limits(host_cpus),
        _ => (None::<usize>, None::<u64>),
    };
    if let Some(cpus) = os_cpus { vcpus = vcpus.min(cpus.max(1)); }
    if let Some(mem) = os_mem { mem_bytes = mem_bytes.min(mem); }
    if let Some(mb) = env_parse::<u64>("REFIT_BUDGET_MB").filter(|&val| val >= 64) {
        mem_bytes = mem_bytes.min(mb * MIB);
    }
    if let Some(cpus) = env_parse::<usize>("REFIT_MAX_CPUS").filter(|&val| val >= 1) {
        vcpus = vcpus.min(cpus);
    }
    ResourceBudget::from_limits(vcpus, mem_bytes)
}

/// Return trimmed non-empty env var value.
pub fn env_trimmed(name: &str) -> Option<String> {
    let raw = std::env::var(name).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Return `true` when env var `name` is set to a truthy literal (`1`, `true`, `yes`, `on`).
pub fn env_truthy(name: &str) -> bool {
    let Ok(raw) = std::env::var(name) else { return false };
    let val = raw.trim();
    ["1", "true", "yes", "on"].iter().any(|lit| val.eq_ignore_ascii_case(lit))
}

/// Parse trimmed env var `name` into `T`.
pub fn env_parse<T: FromStr>(name: &str) -> Option<T> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<T>().ok()
}

/// Map `items` serially when constrained (`1-vCPU` / low-RAM), else in parallel via Rayon.
pub fn par_map<T: Sync, R: Send>(items: &[T], func: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    if budget().is_constrained() {
        items.iter().map(func).collect()
    } else {
        items.par_iter().map(func).collect()
    }
}

/// Execute two closures serially when constrained, else concurrently on scoped OS threads.
pub fn par_join2<A: Send, B: Send>(fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send) -> (A, B) {
    if budget().is_constrained() { return (fa(), fb()); }
    std::thread::scope(|scope| {
        let hb = scope.spawn(fb);
        (fa(), hb.join().expect("par_join2"))
    })
}

/// Execute three closures serially when constrained, else concurrently on scoped OS threads.
pub fn par_join3<A: Send, B: Send, C: Send>(
    fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send, fc: impl FnOnce() -> C + Send,
) -> (A, B, C) {
    if budget().is_constrained() { return (fa(), fb(), fc()); }
    std::thread::scope(|scope| {
        let (hb, hc) = (scope.spawn(fb), scope.spawn(fc));
        (fa(), hb.join().expect("par_join3 b"), hc.join().expect("par_join3 c"))
    })
}

fn host_memory_bytes() -> Option<u64> {
    cfg_select! {
        windows => {
            use std::mem::{size_of, zeroed};
            use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
            // SAFETY: `MEMORYSTATUSEX` is POD with `dwLength` initialized to its exact byte size.
            unsafe {
                let mut st: MEMORYSTATUSEX = zeroed();
                st.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
                (GlobalMemoryStatusEx(&mut st) != 0 && st.ullTotalPhys > 0).then_some(st.ullTotalPhys)
            }
        }
        target_os = "linux" => {
            parse_proc_kb_field(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemTotal:")
        }
        target_os = "macos" => {
            let mut mib = [libc::CTL_HW, libc::HW_MEMSIZE];
            let (mut mem, mut len) = (0u64, std::mem::size_of::<u64>());
            // SAFETY: `mib` has 2 ints and `mem`/`len` point to valid initialized stack storage.
            let ok = unsafe {
                libc::sysctl(
                    mib.as_mut_ptr(), 2, (&mut mem as *mut u64).cast(), &mut len, std::ptr::null_mut(), 0,
                )
            };
            (ok == 0 && mem > 0).then_some(mem)
        }
        _ => None,
    }
}

#[cfg(windows)]
// SAFETY: Caller guarantees `T` is the exact Win32 POD struct matching `class` for `job`.
unsafe fn query_job_info<T>(
    job: windows_sys::Win32::Foundation::HANDLE,
    class: windows_sys::Win32::System::JobObjects::JOBOBJECTINFOCLASS,
) -> Option<T> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::JobObjects::QueryInformationJobObject;
    // SAFETY: Caller guarantees `T` is POD matching `class`.
    unsafe {
        let mut info: T = zeroed();
        let ok = QueryInformationJobObject(
            job, class, (&mut info as *mut T).cast(), size_of::<T>() as u32, std::ptr::null_mut(),
        );
        (ok != 0).then_some(info)
    }
}

#[cfg(windows)]
// SAFETY: Caller guarantees `T` is the exact Win32 POD struct matching `class` for `job`.
unsafe fn set_job_info<T>(
    job: windows_sys::Win32::Foundation::HANDLE,
    class: windows_sys::Win32::System::JobObjects::JOBOBJECTINFOCLASS,
    info: &T,
) -> bool {
    use windows_sys::Win32::System::JobObjects::SetInformationJobObject;
    // SAFETY: Caller guarantees `T` is POD matching `class`.
    unsafe {
        SetInformationJobObject(
            job, class, std::ptr::from_ref(info).cast(), std::mem::size_of_val(info) as u32,
        ) != 0
    }
}

#[cfg(windows)]
fn detect_windows_job_limits(host_cpus: usize) -> (Option<usize>, Option<u64>) {
    use windows_sys::Win32::System::JobObjects::{
        JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation,
    };

    let (mut cpu_limit, mut mem_limit) = (None, None);
    // SAFETY: Null job handle queries the current process's active job object into matching POD structs.
    unsafe {
        if let Some(ext) = query_job_info::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(
            std::ptr::null_mut(), JobObjectExtendedLimitInformation,
        ) {
            let flags = ext.BasicLimitInformation.LimitFlags;
            if (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0 && ext.ProcessMemoryLimit > 0 {
                mem_limit = Some(ext.ProcessMemoryLimit as u64);
            }
            if (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0 && ext.JobMemoryLimit > 0 {
                let job_mem = ext.JobMemoryLimit as u64;
                mem_limit = Some(mem_limit.map_or(job_mem, |prev| prev.min(job_mem)));
            }
        }
        let mask = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
        if let Some(rate) = query_job_info::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>(
            std::ptr::null_mut(), JobObjectCpuRateControlInformation,
        ) && (rate.ControlFlags & mask) == mask
            && rate.Anonymous.CpuRate > 0
        {
            let cap = (host_cpus as u64 * rate.Anonymous.CpuRate as u64).div_ceil(10_000).max(1);
            cpu_limit = Some(cap as usize);
        }
    }
    (cpu_limit, mem_limit)
}

#[cfg(target_os = "linux")]
fn detect_linux_cgroup_limits() -> (Option<usize>, Option<u64>) {
    let Ok(cg_file) = std::fs::read_to_string("/proc/self/cgroup") else { return (None, None) };
    let Some(rel) = parse_cgroup_v2_rel(&cg_file) else { return (None, None) };
    let root = Path::new("/sys/fs/cgroup");
    let leaf = if rel.is_empty() || rel == "/" {
        root.to_path_buf()
    } else {
        root.join(rel.trim_start_matches('/'))
    };
    let (mut min_cpus, mut min_mem, mut cur) = (None::<usize>, None::<u64>, Some(leaf));
    while let Some(dir) = cur {
        if let Ok(raw) = std::fs::read_to_string(dir.join("cpu.max"))
            && let Some(cpus) = parse_cgroup_cpu_max(&raw)
        {
            min_cpus = Some(min_cpus.map_or(cpus, |prev| prev.min(cpus)));
        }
        for name in ["memory.max", "memory.high"] {
            if let Ok(raw) = std::fs::read_to_string(dir.join(name))
                && let Some(mem) = parse_cgroup_mem_limit(&raw)
            {
                min_mem = Some(min_mem.map_or(mem, |prev| prev.min(mem)));
            }
        }
        if dir == root { break; }
        cur = dir.parent().map(Path::to_path_buf).filter(|parent| parent.starts_with(root));
    }
    (min_cpus, min_mem)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_v2_rel(raw: &str) -> Option<&str> {
    raw.lines().find_map(|line| line.strip_prefix("0::").map(str::trim))
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_cpu_max(raw: &str) -> Option<usize> {
    let mut parts = raw.split_whitespace();
    let quota_str = parts.next()?;
    if quota_str == "max" { return None; }
    let (quota, period) = (quota_str.parse::<u64>().ok()?, parts.next()?.parse::<u64>().ok()?);
    (quota > 0 && period > 0).then(|| quota.div_ceil(period).max(1) as usize)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_mem_limit(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "max" { return None; }
    trimmed.parse::<u64>().ok().filter(|&val| val > 0)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_psi_some_avg10(raw: &str) -> Option<f64> {
    raw.lines().find_map(|line| line.strip_prefix("some "))
        .and_then(|rest| rest.split_whitespace().find_map(|tok| tok.strip_prefix("avg10=")?.parse().ok()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GateErr {
    #[error("memory gate busy: reservation limit reached")] Busy,
    #[error("memory gate request {requested}B exceeds capacity {capacity}B")]
    TooBig { requested: usize, capacity: usize },
}

/// Condvar-backed byte-budget admission controller (zero 1ms spin-polling).
#[derive(Debug)]
pub struct MemGate { capacity: usize, used: Mutex<usize>, released: Condvar }

#[derive(Debug)]
pub struct MemPermit<'a> { gate: &'a MemGate, bytes: usize }

impl Drop for MemPermit<'_> {
    fn drop(&mut self) {
        let mut used = self.gate.used.lock();
        *used = used.saturating_sub(self.bytes);
        drop(used);
        self.gate.released.notify_all();
    }
}

impl MemGate {
    pub const fn new(capacity: usize) -> Self {
        Self { capacity, used: Mutex::new(0), released: Condvar::new() }
    }

    pub fn global() -> &'static Self {
        static GLOBAL_GATE: OnceLock<MemGate> = OnceLock::new();
        GLOBAL_GATE.get_or_init(|| Self::new(budget().mem_gate_bytes))
    }

    #[inline] pub fn capacity(&self) -> usize { self.capacity }
    #[cfg(test)] #[inline] pub fn used(&self) -> usize { *self.used.lock() }

    fn check_cap(&self, bytes: usize) -> Result<(), GateErr> {
        if bytes > self.capacity {
            Err(GateErr::TooBig { requested: bytes, capacity: self.capacity })
        } else { Ok(()) }
    }

    fn try_take(&self, used: &mut usize, bytes: usize) -> Option<MemPermit<'_>> {
        used.checked_add(bytes).filter(|&next| next <= self.capacity).map(|next| {
            *used = next;
            MemPermit { gate: self, bytes }
        })
    }

    pub fn try_admit(&self, bytes: usize) -> Result<MemPermit<'_>, GateErr> {
        self.check_cap(bytes)?;
        if under_memory_pressure() { return Err(GateErr::Busy); }
        self.try_reserve(bytes)
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<MemPermit<'_>, GateErr> {
        self.check_cap(bytes)?;
        self.try_take(&mut self.used.lock(), bytes).ok_or(GateErr::Busy)
    }

    pub fn reserve_or_wait(&self, bytes: usize, max_wait: Duration) -> Result<MemPermit<'_>, GateErr> {
        self.check_cap(bytes)?;
        let (deadline, mut used) = (Instant::now() + max_wait, self.used.lock());
        loop {
            if let Some(permit) = self.try_take(&mut used, bytes) { return Ok(permit); }
            if self.released.wait_until(&mut used, deadline).timed_out() {
                return self.try_take(&mut used, bytes).ok_or(GateErr::Busy);
            }
        }
    }
}

#[cfg(unix)]
fn raw_rusage(who: libc::c_int) -> Option<libc::rusage> {
    // SAFETY: `rusage` is a C POD struct zero-initialized before passing to `getrusage`.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        (libc::getrusage(who, &mut ru) == 0).then_some(ru)
    }
}

#[cfg(unix)]
fn rusage_cpu_ms(ru: &libc::rusage) -> u64 {
    fn to_i64(v: impl Into<i64>) -> i64 { v.into() }
    let sec = to_i64(ru.ru_utime.tv_sec) + to_i64(ru.ru_stime.tv_sec);
    let usec = to_i64(ru.ru_utime.tv_usec) + to_i64(ru.ru_stime.tv_usec);
    (sec * 1000 + usec / 1000).max(0) as u64
}

#[cfg(target_os = "macos")]
fn macos_phys_footprint() -> Option<u64> {
    // SAFETY: `rusage_info_v2` is a C POD struct filled by `proc_pid_rusage` for current PID.
    unsafe {
        let mut ri: libc::rusage_info_v2 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            libc::getpid(), libc::RUSAGE_INFO_V2, (&mut ri as *mut libc::rusage_info_v2).cast(),
        );
        (rc == 0 && ri.ri_phys_footprint > 0).then_some(ri.ri_phys_footprint)
    }
}

#[cfg(target_os = "linux")]
fn linux_uss_bytes() -> Option<u64> {
    if let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        let (dirty, clean) = (
            parse_proc_kb_field(&rollup, "Private_Dirty:"),
            parse_proc_kb_field(&rollup, "Private_Clean:"),
        );
        if dirty.is_some() || clean.is_some() {
            let lazy = parse_proc_kb_field(&rollup, "LazyFree:").unwrap_or(0);
            return Some(dirty.unwrap_or(0).saturating_add(clean.unwrap_or(0)).saturating_sub(lazy));
        }
    }
    parse_proc_kb_field(&std::fs::read_to_string("/proc/self/status").ok()?, "VmRSS:")
}

#[cfg(windows)]
fn win_self_metrics() -> SelfMetrics {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    // SAFETY: `GetCurrentProcess` returns the process pseudo-handle; POD outputs carry exact sizes.
    unsafe {
        let proc_h = GetCurrentProcess();
        let mut ex2: PROCESS_MEMORY_COUNTERS_EX2 = zeroed();
        ex2.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32;
        let ex2_ptr = (&mut ex2 as *mut PROCESS_MEMORY_COUNTERS_EX2).cast();
        let rss_bytes = if GetProcessMemoryInfo(proc_h, ex2_ptr, ex2.cb) != 0
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
            let kern = ((kn.dwHighDateTime as u64) << 32) | (kn.dwLowDateTime as u64);
            let user = ((us.dwHighDateTime as u64) << 32) | (us.dwLowDateTime as u64);
            (kern + user) / 10_000
        } else {
            0
        };
        SelfMetrics { rss_bytes, cpu_ms }
    }
}

#[cfg(unix)]
fn unix_self_metrics() -> SelfMetrics {
    let usage = raw_rusage(libc::RUSAGE_SELF);
    let cpu_ms = usage.as_ref().map_or(0, rusage_cpu_ms);
    let max_rss = usage.map_or(0, |ru| ru.ru_maxrss.max(0) as u64);
    let rss_bytes = cfg_select! {
        target_os = "linux" => linux_uss_bytes().unwrap_or(max_rss * 1024),
        target_os = "macos" => macos_phys_footprint().unwrap_or(max_rss),
        _ => max_rss,
    };
    SelfMetrics { rss_bytes, cpu_ms }
}

pub fn self_metrics() -> SelfMetrics {
    cfg_select! {
        windows => win_self_metrics(),
        unix => unix_self_metrics(),
        _ => SelfMetrics { rss_bytes: 0, cpu_ms: 0 },
    }
}

fn spawn_pipe<R: Read + Send + 'static>(
    pipe: Option<R>, cap_bytes: usize,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut reader) = pipe {
            let mut chunk = [0u8; 8192];
            while let Ok(read_n) = reader.read(&mut chunk) && read_n > 0 {
                if buf.len() < cap_bytes {
                    buf.extend_from_slice(&chunk[..(cap_bytes - buf.len()).min(read_n)]);
                }
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    })
}

pub fn run_cmd(
    program: &str, args: &[String], cwd: &Path, env_overrides: &[(String, String)],
    clear_env: bool, timeout_ms: u64,
) -> Result<SandboxResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if clear_env { cmd.env_clear(); }
    for (key, val) in env_overrides { cmd.env(key, val); }
    #[cfg(unix)] set_pgid(&mut cmd);
    #[cfg(unix)] let (unix_cpu_before, _) = unix_rusage();

    let started = Instant::now();
    let mut child = cmd.spawn().with_context(|| format!("spawn `{program}` in {}", cwd.display()))?;
    #[cfg(windows)] let job_handle = attach_job(&child);

    let cap = budget().pipe_cap_bytes;
    let (out_t, err_t) = (spawn_pipe(child.stdout.take(), cap), spawn_pipe(child.stderr.take(), cap));
    let mut timed_out = false;
    let wait_dur = Duration::from_millis(timeout_ms.max(100));
    let exit_code = match wait_timeout::ChildExt::wait_timeout(&mut child, wait_dur)? {
        Some(status) => status.code().unwrap_or(-1),
        None => {
            timed_out = true;
            cfg_select! {
                windows => kill_child(&mut child, job_handle),
                unix => kill_child(&mut child),
                _ => { let _ = child.kill(); let _ = child.wait(); }
            }
            -1
        }
    };
    let wall_ms = started.elapsed().as_millis() as u64;
    let (cpu_ms, peak_rss_bytes) = cfg_select! {
        windows => close_job(job_handle),
        unix => unix_rusage_since(unix_cpu_before),
        _ => (0, 0),
    };
    Ok(SandboxResult {
        exit_code, timed_out, stdout: out_t.join().unwrap_or_default(),
        stderr: err_t.join().unwrap_or_default(), wall_ms, cpu_ms, peak_rss_bytes,
    })
}

#[cfg(windows)]
fn kill_child(child: &mut Child, job: Option<windows_sys::Win32::Foundation::HANDLE>) {
    if let Some(handle) = job {
        // SAFETY: `handle` is a valid Job Object handle owned by this `run_cmd` invocation.
        unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(handle, 1); }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn set_pgid(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: `setpgid(0, 0)` is async-signal-safe and runs solely in the forked child.
    unsafe { cmd.pre_exec(|| { let _ = rustix::process::setpgid(None, None); Ok(()) }); }
}

#[cfg(unix)]
fn kill_child(child: &mut Child) {
    if let Some(pgid) = rustix::process::Pid::from_raw(child.id() as i32) {
        let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::TERM);
        let exited = wait_timeout::ChildExt::wait_timeout(child, Duration::from_millis(100)).ok().flatten();
        if exited.is_none() {
            let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::KILL);
            let _ = child.kill();
        }
    } else {
        let _ = child.kill();
    }
    let _ = child.wait();
}

#[cfg(windows)]
fn attach_job(child: &Child) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    };
    // SAFETY: Win32 Job Object creation/assignment with zeroed POD struct and valid child handle.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() { return None; }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if set_job_info(job, JobObjectExtendedLimitInformation, &info)
            && AssignProcessToJobObject(job, child.as_raw_handle()) != 0
        {
            Some(job)
        } else {
            CloseHandle(job);
            None
        }
    }
}

#[cfg(windows)]
fn close_job(job_handle: Option<windows_sys::Win32::Foundation::HANDLE>) -> (u64, u64) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
    };
    let Some(job) = job_handle else { return (0, 0) };
    // SAFETY: `job` is an owned Job Object handle queried with matching POD structs and closed.
    unsafe {
        let cpu_ms = query_job_info::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(
            job, JobObjectBasicAccountingInformation,
        ).map_or(0, |acc| ((acc.TotalUserTime + acc.TotalKernelTime).max(0) as u64) / 10_000);
        let peak_rss = query_job_info::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(
            job, JobObjectExtendedLimitInformation,
        ).map_or(0, |ext| ext.PeakJobMemoryUsed as u64);
        CloseHandle(job);
        (cpu_ms, peak_rss)
    }
}

#[cfg(unix)]
fn unix_rusage() -> (u64, u64) {
    raw_rusage(libc::RUSAGE_CHILDREN).map_or((0, 0), |ru| {
        let raw = ru.ru_maxrss.max(0) as u64;
        (rusage_cpu_ms(&ru), if cfg!(target_os = "linux") { raw * 1024 } else { raw })
    })
}

#[cfg(unix)]
fn unix_rusage_since(before_cpu: u64) -> (u64, u64) {
    let (after_cpu, rss) = unix_rusage();
    (after_cpu.saturating_sub(before_cpu), rss)
}

#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
#[test]
fn cgroup_and_budget_parsing() {
    assert_eq!(parse_cgroup_v2_rel("0::/kubepods.slice/pod1\n"), Some("/kubepods.slice/pod1"));
    for (raw, want) in [
        ("100000 100000\n", Some(1)), ("250000 100000\n", Some(3)), ("max 100000\n", None),
    ] {
        assert_eq!(parse_cgroup_cpu_max(raw), want);
    }
    assert_eq!(parse_cgroup_mem_limit("268435456\n"), Some(268_435_456));
    assert_eq!(parse_cgroup_mem_limit("max\n"), None);
    assert_eq!(parse_psi_some_avg10("some avg10=12.50 avg60=3.10\n"), Some(12.50));

    let tight = ResourceBudget::from_limits(1, 256 * MIB);
    assert_eq!(
        (tight.mode, tight.rayon_threads, tight.http_pool_max),
        (ExecMode::InlineConstrained, 0, 2)
    );
    assert_eq!(ResourceBudget::from_limits(8, 256 * MIB).mode, ExecMode::InlineConstrained);
    let bal = ResourceBudget::from_limits(8, 8 * 1024 * MIB);
    assert!(bal.mode == ExecMode::BoundedBalanced && bal.rayon_threads >= 4);
}

#[cfg(test)]
#[test]
fn memgate_condvar_and_too_big() {
    let gate = Arc::new(MemGate::new(1024));
    assert_eq!(gate.try_reserve(2048).unwrap_err(), GateErr::TooBig { requested: 2048, capacity: 1024 });
    let first_permit = gate.try_reserve(600).unwrap();
    assert_eq!(gate.used(), 600);
    assert_eq!(gate.try_reserve(500).unwrap_err(), GateErr::Busy);

    // Verify Condvar wakes waiter promptly upon permit drop.
    let shared = Arc::clone(&gate);
    let waiter = std::thread::spawn(move || {
        let permit = shared.reserve_or_wait(500, Duration::from_secs(2)).unwrap();
        assert_eq!(shared.used(), 500);
        drop(permit);
    });
    std::thread::sleep(Duration::from_millis(20));
    drop(first_permit);
    waiter.join().unwrap();
    assert_eq!(gate.used(), 0);
}

#[cfg(test)]
#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
#[test]
fn jemalloc_purge_ok() { assert_eq!(alloc_ops::purge_all(), 0); }
