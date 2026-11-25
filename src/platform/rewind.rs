use std::collections::HashMap;

/// Amortized break-even threshold in microseconds per file for activating `InvertedRewindIndex`.
/// Sits well above the empirical `0.203 us/file` break-even while bypassing trivial `< 1 us/file` scans.
pub const MIN_KERNEL_COST_US: u64 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Durability {
    Low,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RewindTrigger {
    NewDepAdvisory,
    NewSmellPattern,
    NewInvariantRule,
    AbiChanged,
}

impl RewindTrigger {
    #[inline]
    pub fn include_1hop_rev_deps(self) -> bool {
        matches!(self, Self::NewDepAdvisory | Self::AbiChanged)
    }

    #[inline]
    pub fn skip_high_durability(self) -> bool {
        !matches!(self, Self::NewDepAdvisory | Self::NewInvariantRule)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NewDepAdvisory => "new_dep_advisory",
            Self::NewSmellPattern => "new_smell_pattern",
            Self::NewInvariantRule => "new_invariant_rule",
            Self::AbiChanged => "abi_changed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewindEvent {
    pub trigger: RewindTrigger,
    pub token: String,
    pub token_hash: u32,
    pub rule_key: String,
}

impl RewindEvent {
    pub fn new(trigger: RewindTrigger, token: String, rule_key: String) -> Self {
        let token_hash = hash_token_u32(&token);
        Self { trigger, token, token_hash, rule_key }
    }
}

/// Gate activation of `InvertedRewindIndex` strictly when target rule/kernel cost `>= 50 us/file`.
#[inline]
pub fn should_use_index(est_us_per_file: u64) -> bool {
    est_us_per_file >= MIN_KERNEL_COST_US
}

/// Deterministic 32-bit FNV-1a + avalanche hash for lexical rewind tokens.
#[inline]
pub fn hash_token_u32(token: &str) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for &byte in token.as_bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash
}

/// 2-bit 64-bit Bloom signature mask for a 32-bit token hash.
#[inline]
pub fn bloom_mask(tok: u32) -> u64 {
    let shift_a = (tok.wrapping_mul(0x9E37_79B1) >> 26) & 63;
    let shift_b = (tok.wrapping_mul(0x85EB_CA6B) >> 26) & 63;
    (1u64 << shift_a) | (1u64 << shift_b)
}

#[inline]
fn insert_sorted_unique(list: &mut Vec<u32>, value: u32) {
    if list.last().copied().is_some_and(|last| last < value) || list.is_empty() {
        list.push(value);
        return;
    }
    if let Err(pos) = list.binary_search(&value) {
        list.insert(pos, value);
    }
}

fn extract_ident_hashes(bytes: &[u8]) -> Vec<u32> {
    let mut hashes = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let cur = bytes[pos];
        if cur.is_ascii_alphabetic() || cur == b'_' {
            let start = pos;
            pos += 1;
            while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            if pos - start >= 3
                && let Ok(tok) = std::str::from_utf8(&bytes[start..pos])
            {
                hashes.push(hash_token_u32(tok));
            }
        } else {
            pos += 1;
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

/// Compact inverted signature index + 64-bit per-file Bloom filter + reverse-dependency adjacency.
#[derive(Debug, Default)]
pub struct InvertedRewindIndex {
    postings: HashMap<u32, Vec<u32>>,
    bloom: Vec<u64>,
    rev_deps: Vec<Vec<u32>>,
    durability: Vec<Durability>,
    paths: Vec<String>,
    path_to_id: HashMap<String, u32>,
}

impl InvertedRewindIndex {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(num_files: usize) -> Self {
        Self {
            postings: HashMap::with_capacity(num_files.min(4096)),
            bloom: Vec::with_capacity(num_files),
            rev_deps: Vec::with_capacity(num_files),
            durability: Vec::with_capacity(num_files),
            paths: Vec::with_capacity(num_files),
            path_to_id: HashMap::with_capacity(num_files),
        }
    }

    fn ensure_slot(&mut self, file_id: u32) {
        let needed = (file_id as usize) + 1;
        if self.bloom.len() < needed {
            self.bloom.resize(needed, 0);
            self.rev_deps.resize_with(needed, Vec::new);
            self.durability.resize(needed, Durability::Low);
        }
    }

    pub fn register_file(&mut self, path: &str, durability: Durability) -> u32 {
        if let Some(&existing) = self.path_to_id.get(path) {
            self.set_durability(existing, durability);
            return existing;
        }
        let file_id = self.paths.len() as u32;
        self.paths.push(path.to_string());
        self.path_to_id.insert(path.to_string(), file_id);
        self.ensure_slot(file_id);
        self.durability[file_id as usize] = durability;
        file_id
    }

    pub fn index_file_tokens(&mut self, file_id: u32, tokens: &[u32], durability: Durability) {
        self.ensure_slot(file_id);
        self.durability[file_id as usize] = durability;
        let mut mask = self.bloom[file_id as usize];
        for &tok in tokens {
            mask |= bloom_mask(tok);
            let entry = self.postings.entry(tok).or_default();
            insert_sorted_unique(entry, file_id);
        }
        self.bloom[file_id as usize] = mask;
    }

    pub fn index_file_bytes(&mut self, path: &str, bytes: &[u8], durability: Durability) -> u32 {
        let file_id = self.register_file(path, durability);
        let hashes = extract_ident_hashes(bytes);
        self.index_file_tokens(file_id, &hashes, durability);
        file_id
    }

    pub fn add_dep(&mut self, dependent: u32, dependency: u32) {
        self.ensure_slot(dependent.max(dependency));
        if dependent != dependency {
            insert_sorted_unique(&mut self.rev_deps[dependency as usize], dependent);
        }
    }

    pub fn add_dep_by_path(&mut self, dependent: &str, dependency: &str) {
        let Some(&dep_id) = self.path_to_id.get(dependency) else {
            return;
        };
        let Some(&user_id) = self.path_to_id.get(dependent) else {
            return;
        };
        self.add_dep(user_id, dep_id);
    }

    pub fn set_durability(&mut self, file_id: u32, durability: Durability) {
        self.ensure_slot(file_id);
        self.durability[file_id as usize] = durability;
    }

    pub fn file_count(&self) -> usize {
        self.bloom.len()
    }

    pub fn path_of(&self, file_id: u32) -> Option<&str> {
        self.paths.get(file_id as usize).map(String::as_str)
    }

    #[inline]
    pub fn might_contain(&self, file_id: u32, trigger_token: u32) -> bool {
        let Some(&mask) = self.bloom.get(file_id as usize) else {
            return false;
        };
        let req = bloom_mask(trigger_token);
        (mask & req) == req
    }

    fn is_skipped(&self, file_id: u32, skip_high_durability: bool) -> bool {
        skip_high_durability
            && self
                .durability
                .get(file_id as usize)
                .is_some_and(|&dur| dur == Durability::High)
    }

    /// Return sorted, deduplicated affected file IDs for `trigger_token`, optionally expanding
    /// 1-hop reverse dependents and skipping untouched `Durability::High` files.
    pub fn query_affected(
        &self,
        trigger_token: u32,
        include_1hop_rev_deps: bool,
        skip_high_durability: bool,
    ) -> Vec<u32> {
        let Some(cands) = self.postings.get(&trigger_token) else {
            return Vec::new();
        };
        let mut direct = Vec::new();
        let mut affected = Vec::new();

        for &fid in cands {
            if !self.might_contain(fid, trigger_token) || self.is_skipped(fid, skip_high_durability)
            {
                continue;
            }
            direct.push(fid);
            affected.push(fid);
        }

        if include_1hop_rev_deps {
            for &fid in &direct {
                for &rev_fid in &self.rev_deps[fid as usize] {
                    if !self.is_skipped(rev_fid, skip_high_durability) {
                        affected.push(rev_fid);
                    }
                }
            }
        }

        affected.sort_unstable();
        affected.dedup();
        affected
    }

    /// Amortized break-even gated query: returns `None` when kernel cost `< MIN_KERNEL_COST_US`.
    pub fn query_affected_gated(
        &self,
        trigger_token: u32,
        est_us_per_file: u64,
        include_1hop_rev_deps: bool,
        skip_high_durability: bool,
    ) -> Option<Vec<u32>> {
        if !should_use_index(est_us_per_file) {
            return None;
        }
        Some(self.query_affected(trigger_token, include_1hop_rev_deps, skip_high_durability))
    }

    /// Resolve affected repository file paths for a `RewindEvent` gated by estimated kernel cost.
    pub fn query_event_paths(&self, event: &RewindEvent, est_us_per_file: u64) -> Vec<String> {
        let Some(ids) = self.query_affected_gated(
            event.token_hash,
            est_us_per_file,
            event.trigger.include_1hop_rev_deps(),
            event.trigger.skip_high_durability(),
        ) else {
            return Vec::new();
        };
        ids.into_iter()
            .filter_map(|fid| self.path_of(fid).map(str::to_string))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct SynthFile {
        id: u32,
        tokens: Vec<u32>,
        deps: Vec<u32>,
        durability: Durability,
    }

    fn build_synth_corpus(count: usize) -> Vec<SynthFile> {
        let mut files = Vec::with_capacity(count);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for idx in 0..count {
            let id = idx as u32;
            let mut tokens = Vec::with_capacity(28);
            for _ in 0..24 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                tokens.push((rng % 2048) as u32);
            }
            if id % 120 == 11 {
                tokens.push(9001);
            }
            if id % 95 == 19 {
                tokens.push(9002);
            }
            if id % 80 == 7 {
                tokens.push(9003);
            }
            let mut deps = Vec::new();
            if id > 0 {
                deps.push(id - 1);
            }
            if id > 7 {
                deps.push((id.wrapping_mul(31).wrapping_add(7)) % id);
            }
            let durability = if id % 4 == 0 { Durability::High } else { Durability::Low };
            files.push(SynthFile { id, tokens, deps, durability });
        }
        files
    }

    fn naive_query(
        corpus: &[SynthFile],
        trigger_tok: u32,
        include_rev: bool,
        skip_high: bool,
    ) -> Vec<u32> {
        let mut direct = HashSet::new();
        let mut out = Vec::new();
        for file in corpus {
            if skip_high && file.durability == Durability::High {
                continue;
            }
            if file.tokens.contains(&trigger_tok) {
                direct.insert(file.id);
                out.push(file.id);
            }
        }
        if include_rev {
            for file in corpus {
                if skip_high && file.durability == Durability::High {
                    continue;
                }
                if file.deps.iter().any(|dep| direct.contains(dep)) {
                    out.push(file.id);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn rewind_index_exact_parity_bloom_and_breakeven_gate() {
        assert_eq!(MIN_KERNEL_COST_US, 50);
        assert!(!should_use_index(0));
        assert!(!should_use_index(1));
        assert!(!should_use_index(49));
        assert!(should_use_index(50));
        assert!(should_use_index(250));

        let corpus = build_synth_corpus(1_200);
        let mut index = InvertedRewindIndex::with_capacity(corpus.len());
        for file in &corpus {
            index.index_file_tokens(file.id, &file.tokens, file.durability);
            for &dep in &file.deps {
                index.add_dep(file.id, dep);
            }
        }
        assert_eq!(index.file_count(), 1_200);

        for file in &corpus {
            for &tok in &file.tokens {
                assert!(index.might_contain(file.id, tok), "bloom FN fid={} tok={}", file.id, tok);
            }
        }

        let cases = [
            (9001u32, true, false),
            (9001u32, true, true),
            (9002u32, false, false),
            (9002u32, false, true),
            (9003u32, true, true),
            (999_999u32, true, false),
        ];
        for (tok, inc_rev, skip_hi) in cases {
            let expected = naive_query(&corpus, tok, inc_rev, skip_hi);
            let actual = index.query_affected(tok, inc_rev, skip_hi);
            assert_eq!(actual, expected, "parity mismatch tok={tok} rev={inc_rev} hi={skip_hi}");
            assert!(index.query_affected_gated(tok, 49, inc_rev, skip_hi).is_none());
            assert_eq!(index.query_affected_gated(tok, 50, inc_rev, skip_hi), Some(expected));
        }

        let mut path_idx = InvertedRewindIndex::new();
        path_idx.index_file_bytes("src/crypto.rs", b"pub fn legacy_hmac_sign() {}", Durability::Low);
        path_idx.index_file_bytes("src/caller.rs", b"pub fn call_sign() {}", Durability::Low);
        path_idx.index_file_bytes("src/vendor.rs", b"pub fn legacy_hmac_sign() {}", Durability::High);
        path_idx.add_dep_by_path("src/caller.rs", "src/crypto.rs");

        let ev_smell = RewindEvent::new(
            RewindTrigger::NewSmellPattern,
            "legacy_hmac_sign".into(),
            "smell".into(),
        );
        assert_eq!(path_idx.query_event_paths(&ev_smell, 50), vec!["src/crypto.rs"]);
        assert!(path_idx.query_event_paths(&ev_smell, 10).is_empty());

        let ev_adv = RewindEvent::new(
            RewindTrigger::NewDepAdvisory,
            "legacy_hmac_sign".into(),
            "advisory".into(),
        );
        assert_eq!(
            path_idx.query_event_paths(&ev_adv, 75),
            vec!["src/crypto.rs", "src/caller.rs", "src/vendor.rs"]
        );
    }
}
