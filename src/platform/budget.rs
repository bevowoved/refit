use std::sync::OnceLock;

const MIB: u64 = 1024 * 1024;
const CONSTRAINED_MEM_BYTES: u64 = 384 * MIB;
#[cfg(feature = "alloc-mimalloc")]
const TIGHT_MEM_BYTES: u64 = 256 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    InlineConstrained,
    BoundedBalanced,
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceBudget {
    pub effective_vcpus: usize,
    pub effective_mem_bytes: u64,
    pub mode: ExecMode,
    pub http_pool_max: usize,
    pub pipe_cap_bytes: usize,
    pub sse_line_cap_bytes: usize,
    pub mem_gate_bytes: usize,
    pub mcp_blocking_permits: usize,
    pub rayon_threads: usize,
}

impl ResourceBudget {
    #[inline]
    pub fn is_constrained(&self) -> bool {
        self.mode == ExecMode::InlineConstrained
    }

    pub fn from_limits(vcpus: usize, mem_bytes: u64) -> Self {
        let effective_vcpus = vcpus.max(1);
        let constrained = effective_vcpus <= 1 || mem_bytes <= CONSTRAINED_MEM_BYTES;
        let mode = if constrained { ExecMode::InlineConstrained } else { ExecMode::BoundedBalanced };

        let (http_pool_max, pipe_cap_bytes, sse_line_cap_bytes, mem_gate_bytes, mcp_permits, rayon) =
            if constrained {
                (
                    2,
                    4 * (MIB as usize),
                    2 * (MIB as usize),
                    16 * (MIB as usize),
                    1,
                    0,
                )
            } else {
                let gate_mb = ((mem_bytes / MIB) / 8).clamp(16, 64) as usize;
                (
                    effective_vcpus.clamp(4, 16),
                    32 * (MIB as usize),
                    16 * (MIB as usize),
                    gate_mb * (MIB as usize),
                    effective_vcpus.clamp(2, 4),
                    effective_vcpus.min(8),
                )
            };

        Self {
            effective_vcpus,
            effective_mem_bytes: mem_bytes,
            mode,
            http_pool_max,
            pipe_cap_bytes,
            sse_line_cap_bytes,
            mem_gate_bytes,
            mcp_blocking_permits: mcp_permits,
            rayon_threads: rayon,
        }
    }
}

static BUDGET: OnceLock<ResourceBudget> = OnceLock::new();

pub fn budget() -> &'static ResourceBudget {
    BUDGET.get_or_init(detect_budget)
}

pub fn allocator_name() -> &'static str {
    #[cfg(feature = "alloc-mimalloc")]
    {
        "mimalloc"
    }
    #[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
    {
        "jemalloc"
    }
    #[cfg(not(any(feature = "alloc-mimalloc", all(feature = "alloc-jemalloc", not(target_env = "msvc")))))]
    {
        "system"
    }
}

#[cfg(feature = "alloc-mimalloc")]
unsafe extern "C" {
    fn mi_collect(force: bool);
    fn mi_option_set(option: core::ffi::c_int, value: core::ffi::c_long);
}

#[cfg(feature = "alloc-mimalloc")]
const MI_OPT_ARENA_EAGER_COMMIT: core::ffi::c_int = 4;
#[cfg(feature = "alloc-mimalloc")]
const MI_OPT_PURGE_DECOMMITS: core::ffi::c_int = 5;
#[cfg(feature = "alloc-mimalloc")]
const MI_OPT_PURGE_DELAY: core::ffi::c_int = 15;
#[cfg(feature = "alloc-mimalloc")]
const MI_OPT_ALLOW_THP: core::ffi::c_int = 43;

/// Tune the global allocator once at startup before any worker threads spawn.
pub fn init_allocator(b: &ResourceBudget) {
    #[cfg(feature = "alloc-mimalloc")]
    {
        // SAFETY: Called once from sync `fn main()` before spawning any Tokio or Rayon threads,
        // avoiding any concurrent `mi_option_set` race. Option indices match mimalloc v3
        // (`libmimalloc-sys 0.1.49` `v3/include/mimalloc.h`).
        unsafe {
            mi_option_set(MI_OPT_PURGE_DECOMMITS, 1);
            mi_option_set(MI_OPT_PURGE_DELAY, 75);
            if b.effective_mem_bytes <= TIGHT_MEM_BYTES {
                mi_option_set(MI_OPT_ARENA_EAGER_COMMIT, 0);
                mi_option_set(MI_OPT_ALLOW_THP, 0);
            }
        }
    }
    #[cfg(not(feature = "alloc-mimalloc"))]
    let _ = b.effective_mem_bytes;
}

#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
unsafe extern "C" {
    fn _rjem_mallctl(
        name: *const core::ffi::c_char,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut core::ffi::c_void,
        newlen: usize,
    ) -> core::ffi::c_int;
}

/// Release purged allocator pages back to the OS at coarse phase boundaries.
pub fn collect_os_memory() {
    #[cfg(feature = "alloc-mimalloc")]
    {
        // SAFETY: `mi_collect(true)` is thread-safe and forces mimalloc to purge abandoned
        // segments back to the OS (`MADV_DONTNEED` / `MEM_DECOMMIT`).
        unsafe {
            mi_collect(true);
        }
    }
    #[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
    {
        let _ = tikv_jemalloc_ctl::epoch::advance();
        let _ = purge_jemalloc_all_arenas();
    }
}

#[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
fn purge_jemalloc_all_arenas() -> core::ffi::c_int {
    // SAFETY: `arena.4096.purge\0` (`MALLCTL_ARENAS_ALL = 4096`) is a valid NUL-terminated
    // jemalloc void mallctl requiring NULL `oldp`/`newp` and 0 `newlen`.
    unsafe {
        _rjem_mallctl(
            b"arena.4096.purge\0".as_ptr().cast(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            0,
        )
    }
}

/// Returns `true` if Linux PSI `memory.pressure` indicates direct-reclaim stall pressure.
pub fn under_memory_pressure() -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(raw) = std::fs::read_to_string("/proc/pressure/memory") {
            return parse_psi_some_avg10(&raw).is_some_and(|v| v > 5.0);
        }
    }
    false
}

fn detect_budget() -> ResourceBudget {
    let host_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut vcpus = host_cpus;
    let mut mem_bytes = host_memory_bytes().unwrap_or(4 * 1024 * MIB);

    #[cfg(target_os = "linux")]
    {
        let (cg_cpus, cg_mem) = detect_linux_cgroup_limits();
        if let Some(c) = cg_cpus {
            vcpus = vcpus.min(c.max(1));
        }
        if let Some(m) = cg_mem {
            mem_bytes = mem_bytes.min(m);
        }
    }

    #[cfg(windows)]
    {
        let (job_cpus, job_mem) = detect_windows_job_limits(host_cpus);
        if let Some(c) = job_cpus {
            vcpus = vcpus.min(c.max(1));
        }
        if let Some(m) = job_mem {
            mem_bytes = mem_bytes.min(m);
        }
    }

    if let Ok(raw) = std::env::var("REFIT_BUDGET_MB")
        && let Ok(mb) = raw.trim().parse::<u64>()
        && mb >= 64
    {
        mem_bytes = mem_bytes.min(mb * MIB);
    }
    if let Ok(raw) = std::env::var("REFIT_MAX_CPUS")
        && let Ok(c) = raw.trim().parse::<usize>()
        && c >= 1
    {
        vcpus = vcpus.min(c);
    }

    ResourceBudget::from_limits(vcpus, mem_bytes)
}

fn host_memory_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        use std::mem::{size_of, zeroed};
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: `MEMORYSTATUSEX` is POD with `dwLength` initialized to its exact byte size.
        unsafe {
            let mut st: MEMORYSTATUSEX = zeroed();
            st.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut st) != 0 && st.ullTotalPhys > 0 {
                return Some(st.ullTotalPhys);
            }
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

#[cfg(windows)]
fn detect_windows_job_limits(host_cpus: usize) -> (Option<usize>, Option<u64>) {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::JobObjects::{
        JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };

    let mut mem_limit: Option<u64> = None;
    let mut cpu_limit: Option<usize> = None;

    // SAFETY: Passing null job handle queries the current process's active job object (if any).
    // Both output structs are zeroed POD with exact byte lengths.
    unsafe {
        let mut ext: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        let ok = QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectExtendedLimitInformation,
            &mut ext as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        );
        if ok != 0 {
            let flags = ext.BasicLimitInformation.LimitFlags;
            if (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0 && ext.ProcessMemoryLimit > 0 {
                mem_limit = Some(ext.ProcessMemoryLimit as u64);
            }
            if (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0 && ext.JobMemoryLimit > 0 {
                let j = ext.JobMemoryLimit as u64;
                mem_limit = Some(mem_limit.map_or(j, |p| p.min(j)));
            }
        }

        let mut rate: JOBOBJECT_CPU_RATE_CONTROL_INFORMATION = zeroed();
        let rok = QueryInformationJobObject(
            std::ptr::null_mut(),
            JobObjectCpuRateControlInformation,
            &mut rate as *mut _ as *mut core::ffi::c_void,
            size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32,
            std::ptr::null_mut(),
        );
        let mask = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
        if rok != 0 && (rate.ControlFlags & mask) == mask {
            let cpu_rate = rate.Anonymous.CpuRate as u64;
            if cpu_rate > 0 {
                let eff = ((host_cpus as u64 * cpu_rate).div_ceil(10_000)).max(1) as usize;
                cpu_limit = Some(eff);
            }
        }
    }

    (cpu_limit, mem_limit)
}

#[cfg(target_os = "linux")]
fn detect_linux_cgroup_limits() -> (Option<usize>, Option<u64>) {
    use std::path::{Path, PathBuf};

    let Ok(cg_file) = std::fs::read_to_string("/proc/self/cgroup") else {
        return (None, None);
    };
    let Some(rel) = parse_cgroup_v2_rel(&cg_file) else {
        return (None, None);
    };

    let root = Path::new("/sys/fs/cgroup");
    let leaf = if rel.is_empty() || rel == "/" {
        root.to_path_buf()
    } else {
        root.join(rel.trim_start_matches('/'))
    };

    let mut min_cpus: Option<usize> = None;
    let mut min_mem: Option<u64> = None;
    let mut cur: Option<PathBuf> = Some(leaf);

    while let Some(dir) = cur {
        if let Ok(raw) = std::fs::read_to_string(dir.join("cpu.max"))
            && let Some(c) = parse_cgroup_cpu_max(&raw)
        {
            min_cpus = Some(min_cpus.map_or(c, |p| p.min(c)));
        }
        for name in ["memory.max", "memory.high"] {
            if let Ok(raw) = std::fs::read_to_string(dir.join(name))
                && let Some(m) = parse_cgroup_mem_limit(&raw)
            {
                min_mem = Some(min_mem.map_or(m, |p| p.min(m)));
            }
        }
        if dir == root {
            break;
        }
        cur = dir.parent().map(Path::to_path_buf).filter(|p| p.starts_with(root));
    }

    (min_cpus, min_mem)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_v2_rel(raw: &str) -> Option<&str> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            return Some(rest.trim());
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_cpu_max(raw: &str) -> Option<usize> {
    let mut parts = raw.split_whitespace();
    let quota_str = parts.next()?;
    if quota_str == "max" {
        return None;
    }
    let quota = quota_str.parse::<u64>().ok()?;
    let period = parts.next()?.parse::<u64>().ok()?;
    if quota == 0 || period == 0 {
        return None;
    }
    Some(quota.div_ceil(period).max(1) as usize)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_cgroup_mem_limit(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s == "max" {
        return None;
    }
    s.parse::<u64>().ok().filter(|&v| v > 0)
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_psi_some_avg10(raw: &str) -> Option<f64> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("some ") {
            for tok in rest.split_whitespace() {
                if let Some(val) = tok.strip_prefix("avg10=") {
                    return val.parse::<f64>().ok();
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_and_budget_parsing() {
        assert_eq!(parse_cgroup_v2_rel("0::/kubepods.slice/pod1\n"), Some("/kubepods.slice/pod1"));
        assert_eq!(parse_cgroup_cpu_max("100000 100000\n"), Some(1));
        assert_eq!(parse_cgroup_cpu_max("250000 100000\n"), Some(3));
        assert_eq!(parse_cgroup_cpu_max("max 100000\n"), None);
        assert_eq!(parse_cgroup_mem_limit("268435456\n"), Some(268_435_456));
        assert_eq!(parse_cgroup_mem_limit("max\n"), None);
        assert_eq!(
            parse_psi_some_avg10("some avg10=12.50 avg60=3.10 avg300=1.00 total=99\n"),
            Some(12.50)
        );

        let tight = ResourceBudget::from_limits(1, 256 * MIB);
        assert_eq!(tight.mode, ExecMode::InlineConstrained);
        assert_eq!(tight.rayon_threads, 0);
        assert_eq!(tight.http_pool_max, 2);

        let mem_tight = ResourceBudget::from_limits(8, 256 * MIB);
        assert_eq!(mem_tight.mode, ExecMode::InlineConstrained);

        let bal = ResourceBudget::from_limits(8, 8 * 1024 * MIB);
        assert_eq!(bal.mode, ExecMode::BoundedBalanced);
        assert!(bal.rayon_threads >= 4);
    }

    #[cfg(all(feature = "alloc-jemalloc", not(target_env = "msvc"), not(feature = "alloc-mimalloc")))]
    #[test]
    fn jemalloc_all_arenas_purge_returns_zero() {
        assert_eq!(purge_jemalloc_all_arenas(), 0);
    }
}
