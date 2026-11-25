use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::Hash;

struct Node<K, V> {
    key: K,
    val: V,
    weight: usize,
    visited: bool,
    prev: Option<usize>,
    next: Option<usize>,
}

struct SieveInner<K, V> {
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    hand: Option<usize>,
    total_weight: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl<K: Eq + Hash + Clone, V> SieveInner<K, V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            head: None,
            tail: None,
            hand: None,
            total_weight: 0,
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.nodes[idx].as_ref().expect("sieve node");
            (n.prev, n.next)
        };
        if let Some(p) = prev {
            if let Some(pn) = self.nodes[p].as_mut() {
                pn.next = next;
            }
        } else {
            self.head = next;
        }
        if let Some(nx) = next {
            if let Some(nn) = self.nodes[nx].as_mut() {
                nn.prev = prev;
            }
        } else {
            self.tail = prev;
        }
        if self.hand == Some(idx) {
            self.hand = prev;
        }
    }

    fn push_head(&mut self, idx: usize) {
        let old_head = self.head;
        if let Some(n) = self.nodes[idx].as_mut() {
            n.prev = None;
            n.next = old_head;
        }
        if let Some(oh) = old_head {
            if let Some(hn) = self.nodes[oh].as_mut() {
                hn.prev = Some(idx);
            }
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn evict_one(&mut self) -> bool {
        let mut cur = self.hand.or(self.tail);
        while let Some(idx) = cur {
            let (visited, prev) = {
                let n = self.nodes[idx].as_ref().expect("sieve node");
                (n.visited, n.prev)
            };
            if visited {
                if let Some(n) = self.nodes[idx].as_mut() {
                    n.visited = false;
                }
                cur = prev.or(self.tail);
                continue;
            }
            self.hand = prev;
            self.remove_slot(idx);
            return true;
        }
        false
    }

    fn remove_slot(&mut self, idx: usize) -> Option<V> {
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.map.remove(&node.key);
        self.total_weight = self.total_weight.saturating_sub(node.weight);
        self.free.push(idx);
        Some(node.val)
    }

    fn alloc_slot(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            idx
        }
    }

    fn insert(&mut self, key: K, val: V, weight: usize) -> Option<V> {
        let w = weight.max(1);
        if let Some(&idx) = self.map.get(&key) {
            let old = self.nodes[idx].as_mut().expect("sieve node");
            self.total_weight = self.total_weight.saturating_sub(old.weight).saturating_add(w);
            old.weight = w;
            old.visited = true;
            let prev_val = std::mem::replace(&mut old.val, val);
            while self.total_weight > self.max_bytes && self.map.len() > 1 {
                if !self.evict_one() {
                    break;
                }
            }
            return Some(prev_val);
        }

        while self.map.len() >= self.max_entries
            || (self.total_weight.saturating_add(w) > self.max_bytes && !self.map.is_empty())
        {
            if !self.evict_one() {
                break;
            }
        }

        let idx = self.alloc_slot(Node {
            key: key.clone(),
            val,
            weight: w,
            visited: false,
            prev: None,
            next: None,
        });
        self.push_head(idx);
        self.map.insert(key, idx);
        self.total_weight = self.total_weight.saturating_add(w);
        None
    }
}

/// Thread-safe byte- and entry-bounded SIEVE cache (NSDI '24).
pub struct SieveCache<K, V> {
    inner: Mutex<SieveInner<K, V>>,
}

impl<K: Eq + Hash + Clone, V: Clone> SieveCache<K, V> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self { inner: Mutex::new(SieveInner::new(max_entries, max_bytes)) }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut g = self.inner.lock();
        let idx = *g.map.get(key)?;
        let n = g.nodes[idx].as_mut()?;
        n.visited = true;
        Some(n.val.clone())
    }

    pub fn insert(&self, key: K, val: V, weight_bytes: usize) -> Option<V> {
        self.inner.lock().insert(key, val, weight_bytes)
    }

    pub fn get_or_insert_with(
        &self,
        key: K,
        weight_bytes: usize,
        make: impl FnOnce() -> V,
    ) -> V {
        let mut g = self.inner.lock();
        if let Some(&idx) = g.map.get(&key)
            && let Some(n) = g.nodes[idx].as_mut()
        {
            n.visited = true;
            return n.val.clone();
        }
        let val = make();
        g.insert(key, val.clone(), weight_bytes);
        val
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let mut g = self.inner.lock();
        let idx = *g.map.get(key)?;
        g.remove_slot(idx)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    pub fn snapshot(&self) -> Vec<(K, V)> {
        let g = self.inner.lock();
        g.nodes
            .iter()
            .filter_map(|opt| opt.as_ref().map(|n| (n.key.clone(), n.val.clone())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sieve_evicts_unvisited_and_bounds_bytes() {
        let cache = SieveCache::<String, u32>::new(3, 1024);
        cache.insert("hot".into(), 1, 100);
        cache.insert("cold1".into(), 2, 100);
        cache.insert("cold2".into(), 3, 100);
        // Mark "hot" visited; inserting 4th item must evict an unvisited item first.
        assert_eq!(cache.get(&"hot".into()), Some(1));
        cache.insert("fresh".into(), 4, 100);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"hot".into()), Some(1));
        assert_eq!(cache.get(&"fresh".into()), Some(4));

        // Byte-weight cap test.
        let tiny = SieveCache::<u32, u32>::new(16, 250);
        tiny.insert(1, 10, 100);
        tiny.insert(2, 20, 100);
        tiny.insert(3, 30, 100);
        assert!(tiny.len() <= 2);
        assert_eq!(tiny.remove(&3), Some(30));
    }
}
