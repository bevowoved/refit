use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateErr {
    Busy,
    TooBig { requested: usize, capacity: usize },
}

impl fmt::Display for GateErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Busy => write!(f, "memory gate busy: reservation limit reached"),
            Self::TooBig { requested, capacity } => {
                write!(f, "memory gate request {requested}B exceeds capacity {capacity}B")
            }
        }
    }
}

impl std::error::Error for GateErr {}

/// Lock-free CAS admission-by-reservation memory gate.
#[derive(Debug)]
pub struct MemGate {
    capacity: usize,
    used: AtomicUsize,
}

#[derive(Debug)]
pub struct MemPermit<'a> {
    gate: &'a MemGate,
    bytes: usize,
}

impl Drop for MemPermit<'_> {
    fn drop(&mut self) {
        self.gate.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl MemGate {
    pub const fn new(capacity: usize) -> Self {
        Self { capacity, used: AtomicUsize::new(0) }
    }

    pub fn global() -> &'static Self {
        static GLOBAL_GATE: OnceLock<MemGate> = OnceLock::new();
        GLOBAL_GATE.get_or_init(|| Self::new(super::budget::budget().mem_gate_bytes))
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(test)]
    #[inline]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<MemPermit<'_>, GateErr> {
        if bytes > self.capacity {
            return Err(GateErr::TooBig { requested: bytes, capacity: self.capacity });
        }
        let mut cur = self.used.load(Ordering::Acquire);
        loop {
            let Some(next) = cur.checked_add(bytes) else {
                return Err(GateErr::Busy);
            };
            if next > self.capacity {
                return Err(GateErr::Busy);
            }
            match self.used.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(MemPermit { gate: self, bytes }),
                Err(actual) => cur = actual,
            }
        }
    }

    pub fn reserve_or_wait(&self, bytes: usize, max_wait: Duration) -> Result<MemPermit<'_>, GateErr> {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            match self.try_reserve(bytes) {
                Ok(permit) => return Ok(permit),
                Err(err @ GateErr::TooBig { .. }) => return Err(err),
                Err(GateErr::Busy) => {
                    if start.elapsed() >= max_wait {
                        return Err(GateErr::Busy);
                    }
                    spins = spins.saturating_add(1);
                    if spins < 16 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            }
        }
    }
}

#[repr(align(64))]
struct CachePadded<T>(T);

/// Zero-`unsafe`, `Copy`-slot Firedancer Tango-style credit ring (immune to CWE-415).
///
/// Uses 64-byte cache-padded `fseq` (consumer progress) and `fctl` (producer sequence)
/// with per-slot atomic state for lock-free bounded burst scheduling.
pub struct TangoCreditRing<T: Copy, const N: usize> {
    prod_seq: CachePadded<AtomicUsize>,
    cons_seq: CachePadded<AtomicUsize>,
    deferred: CachePadded<AtomicU64>,
    slots: Box<[Slot<T>]>,
    mask: usize,
}

struct Slot<T: Copy> {
    seq: AtomicUsize,
    val: parking_lot::Mutex<T>,
}

// Keep zero-unsafe guarantee: `Slot` wraps `T: Copy` in `parking_lot::Mutex<T>` (never
// contended once sequence ownership is acquired by CAS on `prod_seq`/`cons_seq`).

impl<T: Copy + Default, const N: usize> TangoCreditRing<T, N> {
    pub fn new() -> Self {
        assert!(N >= 2 && N.is_power_of_two(), "TangoCreditRing capacity must be power of two >= 2");
        let slots: Vec<Slot<T>> = (0..N)
            .map(|i| Slot { seq: AtomicUsize::new(i), val: parking_lot::Mutex::new(T::default()) })
            .collect();
        Self {
            prod_seq: CachePadded(AtomicUsize::new(0)),
            cons_seq: CachePadded(AtomicUsize::new(0)),
            deferred: CachePadded(AtomicU64::new(0)),
            slots: slots.into_boxed_slice(),
            mask: N - 1,
        }
    }

    #[cfg(test)]
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    #[cfg(test)]
    #[inline]
    pub fn deferred_count(&self) -> u64 {
        self.deferred.0.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    #[inline]
    pub fn len(&self) -> usize {
        let p = self.prod_seq.0.load(Ordering::Acquire);
        let c = self.cons_seq.0.load(Ordering::Acquire);
        p.wrapping_sub(c)
    }

    #[cfg(test)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_push(&self, item: T) -> Result<(), T> {
        let mut pos = self.prod_seq.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[pos & self.mask];
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq as isize).wrapping_sub(pos as isize);
            if diff == 0 {
                match self.prod_seq.0.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        *slot.val.lock() = item;
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                self.deferred.0.fetch_add(1, Ordering::Relaxed);
                return Err(item);
            } else {
                pos = self.prod_seq.0.load(Ordering::Relaxed);
            }
        }
    }

    pub fn try_pop(&self) -> Option<T> {
        let mut pos = self.cons_seq.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[pos & self.mask];
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq as isize).wrapping_sub(pos.wrapping_add(1) as isize);
            if diff == 0 {
                match self.cons_seq.0.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let out = *slot.val.lock();
                        slot.seq.store(pos.wrapping_add(N), Ordering::Release);
                        return Some(out);
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                return None;
            } else {
                pos = self.cons_seq.0.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T: Copy + Default, const N: usize> Default for TangoCreditRing<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn memgate_cas_and_too_big() {
        let g = MemGate::new(1024);
        assert_eq!(g.try_reserve(2048).unwrap_err(), GateErr::TooBig { requested: 2048, capacity: 1024 });
        let p1 = g.try_reserve(600).unwrap();
        assert_eq!(g.used(), 600);
        assert_eq!(g.try_reserve(500).unwrap_err(), GateErr::Busy);
        let p2 = g.try_reserve(424).unwrap();
        assert_eq!(g.used(), 1024);
        drop(p1);
        assert_eq!(g.used(), 424);
        drop(p2);
        assert_eq!(g.used(), 0);
    }

    #[test]
    fn tango_credit_ring_burst_and_checksum() {
        let ring: Arc<TangoCreditRing<u64, 64>> = Arc::new(TangoCreditRing::new());
        let prod = ring.clone();
        let total = 20_000u64;
        let expected_sum: u64 = (1..=total).sum();

        let h = std::thread::spawn(move || {
            for v in 1..=total {
                while prod.try_push(v).is_err() {
                    std::thread::yield_now();
                }
            }
        });

        let mut got_sum = 0u64;
        let mut count = 0u64;
        while count < total {
            if let Some(v) = ring.try_pop() {
                got_sum += v;
                count += 1;
            } else {
                std::thread::yield_now();
            }
        }
        h.join().unwrap();
        assert_eq!(got_sum, expected_sum);
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 64);

        let tiny = TangoCreditRing::<u32, 2>::new();
        assert!(tiny.try_push(1).is_ok());
        assert!(tiny.try_push(2).is_ok());
        assert_eq!(tiny.try_push(3), Err(3));
        assert!(tiny.deferred_count() >= 1);
    }
}
