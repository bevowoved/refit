use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering, fence};

const EPOCH_SLOTS: usize = 64;
const PAYLOAD_MASK: u64 = 0xA5A5_5A5A_1234_5678;
const PAYLOAD_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

#[repr(align(64))]
struct EpochSlot {
    /// Bit 0 = active `EDIT_LEASE`, bits 1..64 = monotonic fencing epoch.
    state: AtomicU64,
    payload_a: AtomicU64,
    payload_b: AtomicU64,
    checksum: AtomicU64,
}

impl EpochSlot {
    fn new(init: u64) -> Self {
        Self {
            state: AtomicU64::new(2),
            payload_a: AtomicU64::new(init),
            payload_b: AtomicU64::new(init ^ PAYLOAD_MASK),
            checksum: AtomicU64::new(init.wrapping_mul(PAYLOAD_MULT)),
        }
    }

    #[inline]
    fn try_read(&self) -> Result<(u64, u64), ()> {
        let s1 = self.state.load(Ordering::Acquire);
        if (s1 & 1) != 0 {
            return Err(());
        }
        let a = self.payload_a.load(Ordering::Relaxed);
        let b = self.payload_b.load(Ordering::Relaxed);
        let c = self.checksum.load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        let s2 = self.state.load(Ordering::Acquire);
        if s1 != s2 || b != (a ^ PAYLOAD_MASK) || c != a.wrapping_mul(PAYLOAD_MULT) {
            return Err(());
        }
        Ok((s1 >> 1, a))
    }

    #[inline]
    fn try_acquire(&self) -> Result<u64, ()> {
        let cur = self.state.load(Ordering::Acquire);
        if (cur & 1) != 0 {
            return Err(());
        }
        self.state
            .compare_exchange(cur, cur | 1, Ordering::AcqRel, Ordering::Relaxed)
            .map(|_| cur)
            .map_err(|_| ())
    }

    #[inline]
    fn commit(&self, base_state: u64, new_val: u64) -> u64 {
        fence(Ordering::Release);
        self.payload_a.store(new_val, Ordering::Relaxed);
        self.payload_b.store(new_val ^ PAYLOAD_MASK, Ordering::Relaxed);
        self.checksum.store(new_val.wrapping_mul(PAYLOAD_MULT), Ordering::Relaxed);
        let next = base_state.wrapping_add(2);
        self.state.store(next, Ordering::Release);
        next >> 1
    }

    #[inline]
    fn abort(&self, base_state: u64) {
        self.state.store(base_state, Ordering::Release);
    }
}

/// Lock-free 64-slot IRC-style seqlock `EpochRegistry` for wait-free reads and edit leases.
pub struct EpochRegistry {
    slots: Box<[EpochSlot]>,
    global_epoch: AtomicU64,
}

pub struct EditLease<'a> {
    reg: &'a EpochRegistry,
    slot_idx: usize,
    base_state: u64,
    committed: bool,
}

impl EditLease<'_> {
    pub fn commit(mut self, payload: u64) -> u64 {
        let ep = self.reg.slots[self.slot_idx].commit(self.base_state, payload);
        self.reg.global_epoch.fetch_add(1, Ordering::Release);
        self.committed = true;
        ep
    }
}

impl Drop for EditLease<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.reg.slots[self.slot_idx].abort(self.base_state);
        }
    }
}

impl EpochRegistry {
    pub fn new() -> Self {
        let slots: Vec<EpochSlot> = (0..EPOCH_SLOTS)
            .map(|i| EpochSlot::new(i as u64 + 1))
            .collect();
        Self { slots: slots.into_boxed_slice(), global_epoch: AtomicU64::new(1) }
    }

    pub fn global() -> &'static Self {
        static GLOBAL: OnceLock<EpochRegistry> = OnceLock::new();
        GLOBAL.get_or_init(Self::new)
    }

    #[inline]
    pub fn global_epoch(&self) -> u64 {
        self.global_epoch.load(Ordering::Acquire)
    }

    #[inline]
    pub fn bump_global(&self) -> u64 {
        self.global_epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    #[inline]
    fn slot_for(key: &str) -> usize {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in key.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        (h as usize) & (EPOCH_SLOTS - 1)
    }

    #[inline]
    pub fn try_read(&self, key: &str) -> Option<(u64, u64)> {
        self.slots[Self::slot_for(key)].try_read().ok()
    }

    #[inline]
    pub fn try_lease(&self, key: &str) -> Option<EditLease<'_>> {
        let slot_idx = Self::slot_for(key);
        let base_state = self.slots[slot_idx].try_acquire().ok()?;
        Some(EditLease { reg: self, slot_idx, base_state, committed: false })
    }

    #[inline]
    pub fn publish(&self, key: &str, rev: u64) -> u64 {
        if let Some(lease) = self.try_lease(key) {
            lease.commit(rev)
        } else {
            self.bump_global()
        }
    }
}

impl Default for EpochRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn epoch_registry_seqlock_zero_torn_reads() {
        let reg = Arc::new(EpochRegistry::new());
        let lease = reg.try_lease("src/main.rs").unwrap();
        // Concurrent lease on same key conflicts cleanly while held.
        assert!(reg.try_lease("src/main.rs").is_none());
        assert!(reg.try_read("src/main.rs").is_none());
        let ep = lease.commit(42);
        assert!(ep >= 2);
        assert_eq!(reg.try_read("src/main.rs"), Some((ep, 42)));

        // Dropping uncommitted lease aborts cleanly without bumping epoch.
        {
            let _aborted = reg.try_lease("src/main.rs").unwrap();
        }
        assert_eq!(reg.try_read("src/main.rs"), Some((ep, 42)));

        // Stress 200k concurrent read/edit ops across threads.
        let r_ed = reg.clone();
        let ed = std::thread::spawn(move || {
            for i in 1..=40_000u64 {
                let k = format!("k{}", i & 15);
                if let Some(l) = r_ed.try_lease(&k) {
                    l.commit(i);
                }
            }
        });
        let mut ok_reads = 0u64;
        for i in 0..160_000u64 {
            let k = format!("k{}", i & 15);
            if let Some((e, _)) = reg.try_read(&k) {
                assert!(e >= 1);
                ok_reads += 1;
            }
        }
        ed.join().unwrap();
        assert!(ok_reads > 0);
        assert!(reg.global_epoch() > 1);
    }
}
