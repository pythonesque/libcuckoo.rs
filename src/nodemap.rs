//use std::collections::hash_state::DefaultState;
//use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::hash::{Hasher/*, Hash*/};

/*pub type FnvHashMap<K, V> = HashMap<K, V, DefaultState<FnvHasher>>;
pub type FnvHashSet<V> = HashSet<V, DefaultState<FnvHasher>>;

pub type NodeMap<T> = FnvHashMap<ast::NodeId, T>;
pub type DefIdMap<T> = FnvHashMap<ast::DefId, T>;

pub type NodeSet = FnvHashSet<ast::NodeId>;
pub type DefIdSet = FnvHashSet<ast::DefId>;

pub fn FnvHashMap<K: Hash + Eq, V>() -> FnvHashMap<K, V> {
    Default::default()
}
pub fn FnvHashSet<V: Hash + Eq>() -> FnvHashSet<V> {
    Default::default()
}*/

/*pub fn NodeMap<T>() -> NodeMap<T> { FnvHashMap() }
pub fn DefIdMap<T>() -> DefIdMap<T> { FnvHashMap() }
pub fn NodeSet() -> NodeSet { FnvHashSet() }
pub fn DefIdSet() -> DefIdSet { FnvHashSet() }*/

/// A speedy hash algorithm for node ids and def ids. The hashmap in
/// libcollections by default uses SipHash which isn't quite as speedy as we
/// want. In the compiler we're not really worried about DOS attempts, so we
/// just default to a non-cryptographic hash.
///
/// This uses FNV hashing, as described here:
/// http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
pub struct FnvHasher(u64);

impl Default for FnvHasher {
    #[inline]
    fn default() -> FnvHasher { FnvHasher(0xcbf29ce484222325) }
}

impl Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;
        for byte in bytes {
            hash = hash ^ (*byte as u64);
            hash = hash.wrapping_mul(0x100000001b3); // 1099511628211
        }
        *self = FnvHasher(hash);
    }
    #[inline]
    fn finish(&self) -> u64 { self.0 }
}

