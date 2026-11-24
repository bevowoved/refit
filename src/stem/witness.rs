use rustc_hash::FxHashMap;

pub const NO_FILE: u32 = u32::MAX;

/// Witness bit flags packed into the upper 4 bits of `RefWitness::flags_and_test`.
pub const WITNESS_TEST_COUNT_MASK: u16 = 0x0FFF;
pub const WITNESS_DYNAMIC_REF: u16 = 1 << 12;
pub const WITNESS_GLOB_REEXPORT: u16 = 1 << 13;
pub const WITNESS_MACRO_ATTR: u16 = 1 << 14;
pub const WITNESS_EXTERNAL: u16 = 1 << 15;

/// Per-definition boundary & visibility flags (`u16`).
pub const BOUNDARY_FFI_NO_MANGLE: u16 = 1 << 0;
pub const BOUNDARY_CTOR_INVENTORY: u16 = 1 << 1;
pub const BOUNDARY_MACRO_DERIVE: u16 = 1 << 2;
pub const BOUNDARY_PY_ALL: u16 = 1 << 3;
pub const BOUNDARY_FRAMEWORK_ROUTE: u16 = 1 << 4;
pub const BOUNDARY_ENTRYPOINT_OR_GLOB: u16 = 1 << 5;
pub const BOUNDARY_DYNAMIC_REF: u16 = 1 << 6;
pub const DEF_VIS_INTERNAL: u16 = 1 << 7;

pub const BOUNDARY_T4_MASK: u16 = BOUNDARY_FFI_NO_MANGLE
    | BOUNDARY_CTOR_INVENTORY
    | BOUNDARY_MACRO_DERIVE
    | BOUNDARY_PY_ALL
    | BOUNDARY_FRAMEWORK_ROUTE
    | BOUNDARY_ENTRYPOINT_OR_GLOB;

/// 16-byte `#[repr(C)]` 2-distinct-file witness replacing `HashMap<String, HashSet<String>>`.
/// Storing two distinct production `FileId`s (`prod_file_a`, `prod_file_b`) proves cross-file
/// liveness for ANY defining file in O(1) space and order-independent insertion.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefWitness {
    pub prod_file_a: u32,
    pub prod_file_b: u32,
    pub test_file: u32,
    pub prod_files_sat: u16,
    pub flags_and_test: u16,
}

const _: () = assert!(std::mem::size_of::<RefWitness>() == 16);

impl Default for RefWitness {
    fn default() -> Self {
        Self {
            prod_file_a: NO_FILE,
            prod_file_b: NO_FILE,
            test_file: NO_FILE,
            prod_files_sat: 0,
            flags_and_test: 0,
        }
    }
}

impl RefWitness {
    #[inline]
    pub fn record_prod(&mut self, fid: u32) {
        if self.prod_file_a == NO_FILE {
            self.prod_file_a = fid;
            self.prod_files_sat = 1;
        } else if self.prod_file_a == fid {
            return;
        } else if self.prod_file_b == NO_FILE {
            self.prod_file_b = fid;
            self.prod_files_sat = 2;
        } else if self.prod_file_b == fid {
            return;
        } else {
            self.prod_files_sat = self.prod_files_sat.saturating_add(1);
        }
    }

    #[inline]
    pub fn record_test(&mut self, fid: u32) {
        let cur = self.flags_and_test & WITNESS_TEST_COUNT_MASK;
        if self.test_file == NO_FILE {
            self.test_file = fid;
            self.set_test_count(cur.saturating_add(1));
        } else if self.test_file != fid {
            self.set_test_count(cur.saturating_add(1));
        }
    }

    #[inline]
    pub fn add_flag(&mut self, flag: u16) {
        self.flags_and_test |= flag & !WITNESS_TEST_COUNT_MASK;
    }

    #[inline]
    pub fn test_count(self) -> u16 {
        self.flags_and_test & WITNESS_TEST_COUNT_MASK
    }

    #[inline]
    fn set_test_count(&mut self, count: u16) {
        let clamped = count.min(WITNESS_TEST_COUNT_MASK);
        self.flags_and_test = (self.flags_and_test & !WITNESS_TEST_COUNT_MASK) | clamped;
    }

    #[inline]
    pub fn has_cross_file_prod_caller(self, def_file: u32) -> bool {
        self.prod_files_sat >= 2
            || (self.prod_files_sat == 1 && self.prod_file_a != NO_FILE && self.prod_file_a != def_file)
    }

    #[inline]
    pub fn has_external_flag(self) -> bool {
        (self.flags_and_test & (WITNESS_GLOB_REEXPORT | WITNESS_MACRO_ATTR | WITNESS_EXTERNAL)) != 0
    }

    #[inline]
    pub fn has_dynamic_flag(self) -> bool {
        (self.flags_and_test & WITNESS_DYNAMIC_REF) != 0
    }
}

/// Compact `u32` path interner eliminating repeated `String` clones across symbol/graph tables.
#[derive(Debug, Default)]
pub struct PathArena {
    paths: Vec<String>,
    by_path: FxHashMap<String, u32>,
}

impl PathArena {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            paths: Vec::with_capacity(cap),
            by_path: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    pub fn intern(&mut self, path: &str) -> u32 {
        if let Some(&id) = self.by_path.get(path) {
            return id;
        }
        let id = self.paths.len() as u32;
        let owned = path.to_string();
        self.paths.push(owned.clone());
        self.by_path.insert(owned, id);
        id
    }

    pub fn id_of(&self, path: &str) -> Option<u32> {
        self.by_path.get(path).copied()
    }
}

/// 4-Tier Deletion-Safety Proof Lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofTier {
    /// Closed-world internal symbol with zero prod, intra-file, test, or boundary references.
    T1ProvenDead,
    /// Workspace-visible symbol with zero static prod, intra-file, test, or boundary references.
    T2LikelyDead,
    /// Referenced only intra-file, inside tests (`#[cfg(test)]` / test suites), or via dynamic reflection.
    T3DynamicOrTestOnly,
    /// External/FFI/linker/macro/plugin/route/re-export boundary — never delete.
    T4ExternalOrMacro,
}

impl ProofTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T1ProvenDead => "T1ProvenDead",
            Self::T2LikelyDead => "T2LikelyDead",
            Self::T3DynamicOrTestOnly => "T3DynamicOrTestOnly",
            Self::T4ExternalOrMacro => "T4ExternalOrMacro",
        }
    }

    #[inline]
    pub fn is_dead_candidate(self) -> bool {
        matches!(self, Self::T1ProvenDead | Self::T2LikelyDead)
    }
}

/// Classify a symbol definition against its 16B `RefWitness`, intra-file non-def hit count,
/// and boundary flags. Returns `None` when the symbol has a verified cross-file prod caller.
pub fn classify_symbol(
    def_file: u32,
    witness: RefWitness,
    intra_non_def_hits: u16,
    boundary_flags: u16,
) -> Option<ProofTier> {
    if witness.has_cross_file_prod_caller(def_file) {
        return None;
    }
    if (boundary_flags & BOUNDARY_T4_MASK) != 0 || witness.has_external_flag() {
        return Some(ProofTier::T4ExternalOrMacro);
    }
    if witness.test_count() > 0
        || intra_non_def_hits > 0
        || (boundary_flags & BOUNDARY_DYNAMIC_REF) != 0
        || witness.has_dynamic_flag()
    {
        return Some(ProofTier::T3DynamicOrTestOnly);
    }
    if (boundary_flags & DEF_VIS_INTERNAL) != 0 {
        Some(ProofTier::T1ProvenDead)
    } else {
        Some(ProofTier::T2LikelyDead)
    }
}

/// Portable 128-bit multiply-XOR mixer (`rapidhash`-style) shared across STEM, clones & SimiFlow.
#[inline]
pub fn rapid_mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline]
pub fn rapid_hash64(bytes: &[u8], seed: u64) -> u64 {
    const S0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const S1: u64 = 0x8bb8_4b93_962e_acc9;
    const S2: u64 = 0x4b33_a62e_d433_d4a3;
    let mut h = seed ^ rapid_mum(seed ^ S0, S1) ^ (bytes.len() as u64);
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let chunk = u64::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
            bytes[i + 4],
            bytes[i + 5],
            bytes[i + 6],
            bytes[i + 7],
        ]);
        h = rapid_mum(h ^ chunk ^ S1, S2);
        i += 8;
    }
    if i < bytes.len() {
        let mut tail = 0u64;
        let mut shift = 0u32;
        while i < bytes.len() {
            tail |= (bytes[i] as u64) << shift;
            shift += 8;
            i += 1;
        }
        h = rapid_mum(h ^ tail ^ S0, S2);
    }
    rapid_mum(h ^ (bytes.len() as u64), S1)
}

#[inline]
pub fn rapid_hash_u64(x: u64, seed: u64) -> u64 {
    const S0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const S1: u64 = 0x8bb8_4b93_962e_acc9;
    rapid_mum(x ^ seed ^ S0, x.rotate_left(31) ^ S1)
}

pub const SYM_HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

#[inline]
pub fn hash_symbol(name: &str) -> u64 {
    rapid_hash64(name.as_bytes(), SYM_HASH_SEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_witness_size_and_two_file_invariant() {
        assert_eq!(std::mem::size_of::<RefWitness>(), 16);
        let mut w = RefWitness::default();
        assert!(!w.has_cross_file_prod_caller(0));

        w.record_prod(3);
        assert!(!w.has_cross_file_prod_caller(3));
        assert!(w.has_cross_file_prod_caller(4));

        // Duplicate same file does not falsely bump saturation to 2.
        w.record_prod(3);
        assert_eq!(w.prod_files_sat, 1);
        assert!(!w.has_cross_file_prod_caller(3));

        // Second distinct file guarantees cross-file caller for ALL definers.
        w.record_prod(7);
        assert_eq!(w.prod_files_sat, 2);
        assert!(w.has_cross_file_prod_caller(3));
        assert!(w.has_cross_file_prod_caller(7));
        assert!(w.has_cross_file_prod_caller(99));

        w.record_test(10);
        w.record_test(10);
        assert_eq!(w.test_count(), 1);
        w.record_test(11);
        assert_eq!(w.test_count(), 2);
    }

    #[test]
    fn test_proof_lattice_t1_through_t4() {
        let empty = RefWitness::default();
        assert_eq!(
            classify_symbol(0, empty, 0, DEF_VIS_INTERNAL),
            Some(ProofTier::T1ProvenDead)
        );
        assert_eq!(classify_symbol(0, empty, 0, 0), Some(ProofTier::T2LikelyDead));

        // Intra-file call -> T3
        assert_eq!(
            classify_symbol(0, empty, 1, 0),
            Some(ProofTier::T3DynamicOrTestOnly)
        );
        // Test-only witness -> T3
        let mut test_w = RefWitness::default();
        test_w.record_test(5);
        assert_eq!(
            classify_symbol(0, test_w, 0, DEF_VIS_INTERNAL),
            Some(ProofTier::T3DynamicOrTestOnly)
        );
        // Dynamic reflection -> T3
        assert_eq!(
            classify_symbol(0, empty, 0, BOUNDARY_DYNAMIC_REF),
            Some(ProofTier::T3DynamicOrTestOnly)
        );
        // FFI / ctor / __all__ / route / glob re-export -> T4
        for flag in [
            BOUNDARY_FFI_NO_MANGLE,
            BOUNDARY_CTOR_INVENTORY,
            BOUNDARY_MACRO_DERIVE,
            BOUNDARY_PY_ALL,
            BOUNDARY_FRAMEWORK_ROUTE,
            BOUNDARY_ENTRYPOINT_OR_GLOB,
        ] {
            assert_eq!(
                classify_symbol(0, empty, 0, flag),
                Some(ProofTier::T4ExternalOrMacro)
            );
        }
        // Cross-file prod caller -> None (live)
        let mut live = RefWitness::default();
        live.record_prod(9);
        assert_eq!(classify_symbol(0, live, 0, 0), None);
    }
}
