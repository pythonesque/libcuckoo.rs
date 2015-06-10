use num_cpus;
use self::InsertError::*;
use std::cmp::Ordering::{Less, Greater, Equal};
use std::cell::UnsafeCell;
//use std::collections::hash_map::RandomState;
use std::collections::hash_state::{DefaultState, HashState};
//use std::fmt;
use std::hash::{Hash, Hasher, SipHasher};
use std::i32;
use std::intrinsics;
use std::mem;
use std::ptr;
use std::thread;
use std::sync::atomic::{AtomicPtr, Ordering};
use super::iter::Range;
use super::hazard_pointer::{check_hazard_pointer, delete_unused, HazardPointer};
use super::spinlock::SpinLock;
use super::sys::arch::cpuid;

/// SLOT_PER_BUCKET is the maximum number of keys per bucket
const SLOT_PER_BUCKET: usize = 4;

#[static_assert]
static _SLOT_PER_BUCKET_NONZERO: bool = SLOT_PER_BUCKET != 0;

/// DEFAULT_SIZE is the default number of elements in an empty hash
/// table
const DEFAULT_SIZE: usize = (1 << 16) * SLOT_PER_BUCKET;

/// Constants used internally

/// number of locks in the locks_ array
const K_NUM_LOCKS: usize = 1 << 16;

#[static_assert]
static _K_NUM_LOCKS_NONZERO: bool = K_NUM_LOCKS != 0;

/// true if the key is small and simple, which means using partial keys would
/// probably slow us down
/*static const bool is_simple =
    std::is_pod<key_type>::value && sizeof(key_type) <= 8;

static const bool value_copy_assignable = std::is_copy_assignable<
    mapped_type>::value;*/

/*pub enum ReadError {
    KeyNotFound,
    SpaceNotEnough,
    FunctionNotSupported,
}*/

enum CuckooError {
    TableFull,
    UnderExpansion,
}

//#[derive(Debug)]
pub enum InsertError {
    KeyDuplicated,
    TableFull,
    UnderExpansion,
}

pub type InsertResult<T, K, V> = Result<T, (InsertError, K, V)>;

//#[unsafe_no_drop_flag]
struct BucketInner<K, V> {
    keys: [K ; SLOT_PER_BUCKET],

    vals: [V ; SLOT_PER_BUCKET],
}

/// The Bucket type holds SLOT_PER_BUCKET keys and values, and a occupied
/// bitset, which indicates whether the slot at the given bit index is in
/// the table or not. It uses aligned_storage arrays to store the keys and
/// values to allow constructing and destroying key-value pairs in place.
//#[unsafe_no_drop_flag]
struct Bucket<K, V> {
    kv: Option<BucketInner<K, V>>,

    occupied: [bool; SLOT_PER_BUCKET],
}

impl<K, V> Bucket<K, V> {
    /// Unsafe because it does not perform bounds checking.
    #[inline(always)]
    unsafe fn occupied(&self, ind: usize) -> bool {
        *self.occupied.get_unchecked(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    unsafe fn key(&self, ind: usize) -> &K {
        self.kv.as_ref().unwrap_or_else( || intrinsics::unreachable()).keys.get_unchecked(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    unsafe fn key_mut(&mut self, ind: usize) -> &mut K {
        self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable()).keys.get_unchecked_mut(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    unsafe fn val(&self, ind: usize) -> &V {
        self.kv.as_ref().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    unsafe fn val_mut(&mut self, ind: usize) -> &mut V {
        self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked_mut(ind)
    }

    /// Unsafe because it does not perform bounds checking.
    /// Does not check to make sure location was not already occupied; can leak.
    unsafe fn set_kv(&mut self, ind: usize, k: K, v: V) {
        *self.occupied.get_unchecked_mut(ind) = true;
        let kv = self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable());
        ptr::write(kv.keys.as_mut_ptr().offset(ind as isize), k);
        ptr::write(kv.vals.as_mut_ptr().offset(ind as isize), v);
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    unsafe fn erase_kv(&mut self, ind: usize) -> (K, V) {
        *self.occupied.get_unchecked_mut(ind) = false;
        let kv = self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable());
        (ptr::read(kv.keys.as_mut_ptr().offset(ind as isize)),
         ptr::read(kv.vals.as_mut_ptr().offset(ind as isize)))
    }

    fn new() -> Self {
        unsafe {
            Bucket {
                occupied: [false; SLOT_PER_BUCKET],
                kv: Some(mem::uninitialized()),
            }
        }
    }

    fn clear(&mut self) {
        let kv = match self.kv {
            Some(ref mut kv) => kv,
            None => return
        };
        let mut keys = kv.keys.as_mut_ptr();
        let mut vals = kv.vals.as_mut_ptr();
        // We explicitly free occupied elements, but don't set kv to `None` to prevent double free.
        unsafe {
            for occupied in &mut self.occupied {
                if *occupied {
                    *occupied = false;
                    ptr::read(keys);
                    ptr::read(vals);
                }
                keys = keys.offset(1);
                vals = vals.offset(1);
            }
        }
    }
}

impl<K, V> Drop for Bucket<K, V> {
    fn drop(&mut self) {
        // We explicitly free occupied elements and then set kv to `None` to prevent double free.
        self.clear();
        unsafe {
            ptr::write(&mut self.kv, None);
        }
    }
}

type CacheAlign = super::simd::u64x8;
//struct CacheAlign;

/// cacheint is a cache-aligned atomic integer type
#[repr(C)]
struct CacheInt {
    num: UnsafeCell<usize>,
    // works for both 32 and 64 bit architectures (inasmuch as 64-bit aligned access works at all
    // in Rust)
    padding: [CacheAlign ; 0],
}

impl CacheInt {
    fn new() -> Self {
        CacheInt { num: UnsafeCell::new(0), padding: [] }
    }

    #[inline(always)]
    fn load_unordered(&self) -> usize {
        unsafe { intrinsics::atomic_load_unordered(self.num.get()) }
    }

    #[inline(always)]
    fn store_relaxed(&self, val: usize) {
        unsafe { intrinsics::atomic_store_relaxed(self.num.get(), val); }
    }

    #[inline(always)]
    /// Unsafe because it's not guaranteed this is unaliased.
    unsafe fn store_notatomic(&self, val: usize) {
        *self.num.get() = val;
    }

    #[inline(always)]
    fn fetch_add_relaxed(&self, val: usize) -> usize {
        unsafe { intrinsics::atomic_xadd_relaxed(self.num.get(), val) }
    }
}

/// TableInfo contains the entire state of the hashtable. We allocate one
/// TableInfo pointer per hash table and store all of the table memory in it,
/// so that all the data can be atomically swapped during expansion.
struct TableInfo<K, V> {
    /// 2**hashpower is the number of buckets
    hashpower: usize,

    /// vector of buckets
    buckets: Vec<Bucket<K, V>>,

    /// array of locks
    locks: [SpinLock ; K_NUM_LOCKS],

    /// per-core counters for the number of inserts and deletes
    num_inserts: Vec<CacheInt>,
    num_deletes: Vec<CacheInt>,
}

impl<K, V> TableInfo<K, V> {
    /// The constructor allocates the memory for the table. It allocates one
    /// cacheint for each core in num_inserts and num_deletes.
    fn new(hashpower: usize) -> Box<UnsafeCell<Self>> /*where
        K: fmt::Debug,
        V: fmt::Debug,*/
    {
        fn from_fn<F, T>(n: usize, f: F) -> Vec<T> where F: Fn() -> T {
            if mem::size_of::<T>() != 0 && n.checked_mul(mem::size_of::<T>()).is_none() {
                //println!("capacity error: new TableInfo");
                unsafe { intrinsics::abort(); }
            }
            let mut vec = Vec::with_capacity(n);
            unsafe {
                let mut ptr = vec.as_mut_ptr();
                let end = ptr.offset(n as isize);
                while ptr != end {
                    let t = f();
                    ptr::write(ptr, t);
                    ptr = ptr.offset(1);
                }
                //mem::forget(element);
                vec.set_len(n);
            }
            vec
        }
        let num_cpus = num_cpus::get();

        unsafe {
            // We can't use `UnsafeCell::new()` directly because it can cause a stack overflow if
            // placement new doesn't kick in, and even `const fn`s appear to break RVO on -O0.
            // (Note that this depends on the size of `K_NUM_LOCKS`, `SpinLock`, the stack size,
            // etc.).
            let ti = box UnsafeCell {
                value: TableInfo {
                    hashpower: hashpower,
                    buckets: from_fn(hashsize(hashpower), || Bucket::new()),
                    locks: mem::uninitialized(),
                    num_inserts: from_fn(num_cpus, || CacheInt::new()),
                    num_deletes: from_fn(num_cpus, || CacheInt::new()),
                }
            };
            for lock in &mut (&mut *ti.get()).locks[..] {
                ptr::write(lock, SpinLock::new());
            }
            ti
        }
    }
}

struct Counter(UnsafeCell<Option<usize>>);

unsafe impl Sync for Counter {}

/// counterid stores the per-thread counter index of each thread.
#[thread_local] static COUNTER_ID: Counter = Counter(UnsafeCell { value: None });

/// check_counterid checks if the counterid has already been determined. If
/// not, it assigns a counterid to the current thread by picking a random
/// core. This should be called at the beginning of any function that changes
/// the number of elements in the table.
#[inline(always)]
fn check_counterid() {
    unsafe {
        let counterid = COUNTER_ID.0.get();
        if (*counterid).is_none() {
            *counterid = Some(cpuid() as usize);
        };
    }
}

/// reserve_calc takes in a parameter specifying a certain number of slots
/// for a table and returns the smallest hashpower that will hold n elements.
fn reserve_calc(n: usize) -> usize {
    let nhd: f64 = (n as f64 / SLOT_PER_BUCKET as f64).log2().ceil();
    let new_hashpower = if nhd <= 0.0 { 1.0 } else { nhd } as usize;
    if !(n <= hashsize(new_hashpower).wrapping_mul(SLOT_PER_BUCKET)) {
        //println!("capacity error: reserve_calc()");
        unsafe { intrinsics::abort() }
    }
    new_hashpower
}

pub struct CuckooHashMap<K, V, S = DefaultState<SipHasher>> {
    table_info: AtomicPtr<TableInfo<K, V>>,

    /// old_table_infos holds pointers to old TableInfos that were replaced
    /// during expansion. This keeps the memory alive for any leftover
    /// operations, until they are deleted by the global hazard pointer manager.
    old_table_infos: UnsafeCell<Vec<Box<UnsafeCell<TableInfo<K, V>>>>>,

    // All hashes are keyed on these values, to prevent hash collision attacks.
    hash_state: S,
}

impl<K, V, S> CuckooHashMap<K, V, S>
    where K: Eq + Hash,
          /*K: fmt::Debug,*/
          /*V: fmt::Debug,*/
          S: HashState,
{
    /// cuckoo_init initializes the hashtable, given an initial hashpower as the
    /// argument.
    pub fn with_capacity_and_hash_state(n: usize, hash_state: S) -> Self {
        let hashpower = reserve_calc(n);
        let ti = TableInfo::<K, V>::new(hashpower);
        let table_info = ti.get();
        mem::forget(ti);
        CuckooHashMap {
            table_info: AtomicPtr::new(table_info),
            old_table_infos: UnsafeCell::new(Vec::new()),
            hash_state: hash_state,
        }
    }

    /// clear removes all the elements in the hash table, calling their
    /// destructors.
    pub fn clear(&self) {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_and_lock_all(&hazard_pointer);
            //debug_assert!(ti == self.table_info.load(Ordering::SeqCst));
            let _au = AllUnlocker::new(ti);
            // cuckoo_clear empties the table, calling the destructors of all the
            // elements it removes from the table. It assumes the locks are taken as
            // necessary.
            for bucket in &mut (*ti).buckets {
                bucket.clear();
            }
            let mut insert = (*ti).num_inserts.iter();
            let mut delete = (*ti).num_deletes.iter();
            while let Some(insert) = insert.next() {
                let delete = delete.next().unwrap_or_else(|| intrinsics::unreachable());
                insert.store_notatomic(0);
                delete.store_notatomic(0);
            }
        }
    }

    /// size returns the number of items currently in the hash table. Since it
    /// doesn't lock the table, elements can be inserted during the computation,
    /// so the result may not necessarily be exact (it may even be negative).
    pub fn size(&self) -> isize {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            cuckoo_size(ti)
        }
    }

    /// empty returns true if the table is empty.
    pub fn empty(&self) -> bool {
        self.size() == 0
    }

    /// hashpower returns the hashpower of the table, which is
    /// log_2(the number of buckets).
    pub fn hashpower(&self) -> usize {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            (*ti).hashpower
        }
    }

    /// bucket_count returns the number of buckets in the table.
    pub fn bucket_count(&self) -> usize {
        /*
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            hashsize((&*ti).hashpower)
        }*/
        hashsize(self.hashpower())
    }

    /// load_factor returns the ratio of the number of items in the table to the
    /// total number of available slots in the table.
    /// The result may not necessarily be exact (it may even be negative).
    pub fn load_factor(&self) -> f64 {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            cuckoo_loadfactor(ti)
        }
    }

    /// find searches through the table for `key`, and returns `Some(value)` if
    /// it finds the value, `None` otherwise.
    pub fn find(&self, key: &K) -> Option<V> where
            K: Copy,
            V: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);

            let st = cuckoo_find(key, hv, ti, i1, i2);
            unlock_two(ti, i1, i2);
            st
        }
    }

    /// contains searches through the table for `key`, and returns true if it
    /// finds it in the table, and false otherwise.
    pub fn contains(&self, key: &K) -> bool where K: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);

            let result = cuckoo_contains(key, hv, ti, i1, i2);
            unlock_two(ti, i1, i2);
            result
        }
    }

    /// insert puts the given key-value pair into the table. It first checks
    /// that `key` isn't already in the table, since the table doesn't support
    /// duplicate keys. If the table is out of space, insert will automatically
    /// expand until it can succeed. Note that expansion can throw an exception,
    /// which insert will propagate. If `key` is already in the table, it
    /// returns false, otherwise it returns true.
    pub fn insert(&self, key: K, v: V) -> InsertResult<(), K, V>
    where K: Copy + Eq + Hash + Send + Sync,
          V: Send + Sync,
          S: Default + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        check_counterid();
        let hv = hashed_key(&self.hash_state, &key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);
            self.cuckoo_insert_loop(&hazard_pointer, key, v, hv, ti, i1, i2)
        }
    }

    /// erase removes `key` and it's associated value from the table, calling
    /// their destructors. If `key` is not there, it returns false, otherwise
    /// it returns true.
    pub fn erase(&self, key: &K) -> Option<V> {
        let hazard_pointer = check_hazard_pointer();
        check_counterid();
        let hv = hashed_key(&self.hash_state, key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);

            let result = cuckoo_delete(key, hv, ti, i1, i2);
            unlock_two(ti, i1, i2);
            result
        }
    }

    /// update changes the value associated with `key` to `val`. If `key` is
    /// not there, it returns false, otherwise it returns true.
    pub fn update(&self, key: &K, val: V) -> Result<V, V> where
            V: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);

            let result = cuckoo_update(key, val, hv, ti, i1, i2);
            unlock_two(ti, i1, i2);
            result
        }
    }

    /// update_fn changes the value associated with `key` with the function
    /// `updater`. `updater` will be passed one argument of type `&mut V` and can
    /// modify the argument as desired, returning any extra information if
    /// desired.  If `key` is not there, it returns `None`, otherwise it returns `Some(ret)` where
    /// `ret` is the return value of `updater`.
    pub fn update_fn<F, T>(&self, key: &K, mut updater: F) -> Option<T> where
            F: FnMut(&mut V) -> T,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        unsafe {
            let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);

            let result = cuckoo_update_fn(key, &mut updater, hv, ti, i1, i2);
            unlock_two(ti, i1, i2);
            result
        }
    }

    /// upsert is a combination of update_fn and insert. It first tries updating
    /// the value associated with `key` using `updater`. If `key` is not in the
    /// table, then it runs an insert with `key` and `val`. It will always
    /// succeed, since if the update fails and the insert finds the key already
    /// inserted, it can retry the update.
    pub fn upsert<F, T>(&self, mut key: K, mut updater: F, mut val: V) -> Option<T> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Default + Send + Sync,
            F: FnMut(&mut V) -> T,
    {
        let hazard_pointer = check_hazard_pointer();
        check_counterid();
        let hv = hashed_key(&self.hash_state, &key);
        unsafe {
            loop {
                let (ti, i1, i2) = self.snapshot_and_lock_two(&hazard_pointer, hv);
                match cuckoo_update_fn(&key, &mut updater, hv, ti, i1, i2) {
                    v @ Some(_) => {
                        unlock_two(ti, i1, i2);
                        return v;
                    },
                    // We run an insert, since the update failed
                    None => match self.cuckoo_insert_loop(&hazard_pointer, key, val, hv, ti, i1, i2) {
                        Ok(()) => return None,
                        // The only valid reason for res being false is if insert
                        // encountered a duplicate key after releasing the locks and
                        // performing cuckoo hashing. In this case, we retry the entire
                        // upsert operation.
                        Err((_, k, v)) => {
                            key = k;
                            val = v;
                        }
                    }
                }
            }
        }
    }

    /// rehash will size the table using a hashpower of `n`. Note that the
    /// number of buckets in the table will be 2^`n` after expansion,
    /// so the table will have 2^`n` * `SLOT_PER_BUCKET`
    /// slots to store items in. If `n` is not larger than the current
    /// hashpower, then the function does nothing. It returns true if the table
    /// expansion succeeded, and false otherwise. rehash can throw an exception
    /// if the expansion fails to allocate enough memory for the larger table.
    pub fn rehash(&self, n: usize) -> Result<(), ()>  where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Default + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            if n <= (*ti).hashpower {
                Err(())
            } else {
                self.cuckoo_expand_simple(&hazard_pointer, n)
            }
        }
    }

    /// reserve will size the table to have enough slots for at least `n`
    /// elements. If the table can already hold that many elements, the function
    /// has no effect. Otherwise, the function will expand the table to a
    /// hashpower sufficient to hold `n` elements. It will return true if there
    /// was an expansion, and false otherwise. `reserve` can throw an exception if
    /// the expansion fails to allocate enough memory for the larger table.
    pub fn reserve(&self, n: usize) -> Result<(), ()> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Default + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        unsafe {
            let ti = self.snapshot_table_nolock(&hazard_pointer);
            if n <= hashsize((*ti).hashpower).wrapping_mul(SLOT_PER_BUCKET) {
                Err(())
            } else {
                self.cuckoo_expand_simple(&hazard_pointer, reserve_calc(n))
            }
        }
    }

    /// snapshot_table_nolock loads the table info pointer and sets the hazard
    /// pointer, whithout locking anything. There is a possibility that after
    /// loading a snapshot and setting the hazard pointer, an expansion runs and
    /// create a new version of the table, leaving the old one for deletion. To
    /// deal with that, we check that the table_info we loaded is the same as the
    /// current one, and if it isn't, we try again. Whenever we check if (ti !=
    /// table_info.load()) after setting the hazard pointer, there is an ABA
    /// issue, where the address of the new table_info equals the address of a
    /// previously deleted one, however it doesn't matter, since we would still
    /// be looking at the most recent table_info in that case.
    fn snapshot_table_nolock(&self, hazard_pointer: &HazardPointer)
                                    -> *mut TableInfo<K, V> {
        unsafe {
            loop {
                let ti = self.table_info.load(Ordering::SeqCst);
                /*unsafe */{
                    // Monotonic store should be okay since it's followed by a SeqCst load (which
                    // has acquire semantics) though it might make more sense to put the Acquire
                    // here.
                    // For now we use SeqCst to stay on the safe side.
                    // NOTE: This is definitely a huge bottleneck!  Seriously investigate relaxing
                    // this to release.
                    intrinsics::atomic_store_rel(hazard_pointer.0.get(), ti as usize);
                }
                // If the table info has changed in the time we set the hazard
                // pointer, ti could have been deleted, so try again.
                // Note that this should provide an acquire fence for the previous operation.
                if ti != self.table_info.load(Ordering::SeqCst) {
                    continue;
                }
                return ti;
            }
        }
    }

    /// snapshot_and_lock_two loads the table_info pointer and locks the buckets
    /// associated with the given hash value. It returns the table_info and the
    /// two locked buckets as a tuple. Since the positions of the bucket locks
    /// depends on the number of buckets in the table, the table_info pointer
    /// needs to be grabbed first.
    fn snapshot_and_lock_two(&self, hazard_pointer: &HazardPointer, hv: usize)
                                    -> (*mut TableInfo<K, V>, usize, usize) {
        unsafe {
            loop {
                let ti = self.table_info.load(Ordering::SeqCst);
                /*unsafe */{
                    // Monotonic store should be okay since it's followed by a SeqCst load (which
                    // has acquire semantics) though it might make more sense to put the Acquire
                    // here.
                    // For now we use SeqCst to stay on the safe side.
                    // NOTE: This is definitely a huge bottleneck!  Seriously investigate relaxing
                    // this to release.
                    intrinsics::atomic_store_rel(hazard_pointer.0.get(), ti as usize);
                }
                // If the table info has changed in the time we set the hazard
                // pointer, ti could have been deleted, so try again.
                if ti != self.table_info.load(Ordering::SeqCst) {
                    continue;
                }
                let i1 = index_hash(ti, hv);
                let i2 = alt_index(ti, hv, i1);
                lock_two(ti, i1, i2);
                // Check the table info again
                if ti != self.table_info.load(Ordering::SeqCst) {
                    unlock_two(ti, i1, i2);
                    continue;
                }
                return (ti, i1, i2);
            }
        }
    }

    /// snapshot_and_lock_all is similar to snapshot_and_lock_two, except that it
    /// takes all the locks in the table.
    fn snapshot_and_lock_all(&self, hazard_pointer: &HazardPointer)
                             -> *mut TableInfo<K, V> {
        unsafe {
            loop {
                let ti = self.table_info.load(Ordering::SeqCst);
                /*unsafe */{
                    // Monotonic store should be okay since it's followed by a SeqCst load (which
                    // has acquire semantics) though it might make more sense to put the Acquire
                    // here.
                    // For now we use SeqCst to stay on the safe side.
                    // NOTE: This is definitely a huge bottleneck!  Seriously investigate relaxing
                    // this to release.
                    intrinsics::atomic_store_rel(hazard_pointer.0.get(), ti as usize);
                }
                // If the table info has changed, ti could have been deleted, so try
                // again
                if ti != self.table_info.load(Ordering::SeqCst) {
                    continue;
                }
                /*unsafe */{
                    let locks = &(*ti).locks;
                    for lock in &locks[..] {
                        lock.lock();
                    }
                }
                // If the table info has changed, unlock the locks and try again.
                if ti != self.table_info.load(Ordering::SeqCst) {
                    let _au = AllUnlocker::new(ti);
                    continue;
                }
                return ti;
            }
        }
    }

    /// slot_search searches for a cuckoo path using breadth-first search. It
    /// starts with the i1 and i2 buckets, and, until it finds a bucket with an
    /// empty slot, adds each slot of the bucket in the b_slot. If the queue runs
    /// out of space, it fails.
    ///
    /// Unsafe because it assumes that ti is valid, the hazard pointer is set, and i1 and i2 are in
    /// bounds.
    unsafe fn slot_search(&self, ti: *mut TableInfo<K, V>, i1: usize, i2: usize) -> BSlot {
        let mut q = BQueue::new();
        // The initial pathcode informs cuckoopath_search which bucket the path
        // starts on
        q.enqueue(BSlot::new(i1, 0, 0));
        q.enqueue(BSlot::new(i2, 1, 0));
        while !q.full() && !q.empty() {
            let mut x = q.dequeue();
            // picks a (sort-of) random slot to start from
            let starting_slot = x.pathcode % SLOT_PER_BUCKET;
            for i in Range::new(0, SLOT_PER_BUCKET) {
                if q.full() { break; }
                let slot = (starting_slot.wrapping_add(i)) % SLOT_PER_BUCKET;
                lock(ti, x.bucket);
                let bucket = (*ti).buckets.get_unchecked(x.bucket);
                if !bucket.occupied(slot) {
                    // We can terminate the search here
                    x.pathcode = x.pathcode.wrapping_mul(SLOT_PER_BUCKET).wrapping_add(slot);
                    unlock(ti, x.bucket);
                    return x;
                }

                // If x has less than the maximum number of path components,
                // create a new b_slot item, that represents the bucket we would
                // have come from if we kicked out the item at this slot.
                if x.depth < (MAX_BFS_PATH_LEN - 1) as Depth {
                    let hv = hashed_key(&self.hash_state, bucket.key(slot));
                    unlock(ti, x.bucket);
                    let y = BSlot::new(alt_index(ti, hv, x.bucket),
                                       x.pathcode.wrapping_mul(SLOT_PER_BUCKET).wrapping_add(slot),
                                       x.depth.wrapping_add(1));
                    q.enqueue(y);
                }
            }
        }
        // We didn't find a short-enough cuckoo path, so the queue ran out of
        // space. Return a failure value.
        BSlot::new(0, 0, -1)
    }

    /// cuckoopath_search finds a cuckoo path from one of the starting buckets to
    /// an empty slot in another bucket. It returns the depth of the discovered
    /// cuckoo path on success, and -1 on failure. Since it doesn't take locks on
    /// the buckets it searches, the data can change between this function and
    /// cuckoopath_move. Thus cuckoopath_move checks that the data matches the
    /// cuckoo path before changing it.
    ///
    /// Unsafe because it assumes that ti is valid, the hazard pointer is set, and i1 and i2 are in
    /// bounds.
    unsafe fn cuckoopath_search(&self, ti: *mut TableInfo<K, V>,
                                cuckoo_path: &mut [CuckooRecord<K>; MAX_BFS_PATH_LEN as usize],
                                i1: usize, i2: usize)
                                -> /*([CuckooRecord<K>; MAX_BFS_PATH_LEN], Depth)*/Depth where
            K: Copy,
    {
        let mut x = self.slot_search(ti, i1, i2);
        if x.depth == -1 {
            return -1;
        }
        // Fill in the cuckoo path slots from the end to the beginning
        let start = cuckoo_path.as_mut_ptr();
        //let end = cuckoo_path.as_ptr().offset(x.depth);
        //let mut curr = end.offset(1);
        let end = cuckoo_path.as_mut_ptr().offset(x.depth as isize).offset(1);
        let mut curr = end;
        while curr != start {
            curr = curr.offset(-1);
            (*curr).slot = x.pathcode % SLOT_PER_BUCKET;
            x.pathcode /= SLOT_PER_BUCKET;
        }
        // Fill in the cuckoo_path buckets and keys from the beginning to the
        // end, using the final pathcode to figure out which bucket the path
        // starts on. Since data could have been modified between slot_search
        // and the computation of the cuckoo path, this could be an invalid
        // cuckoo_path.
        let mut i = if x.pathcode == 0 {
            i1
        } else {
            //debug_assert!(x.pathcode == 1);
            i2
        };
        while curr != end {
            (*curr).bucket = i;
            lock(ti, i);
            let bucket = (*ti).buckets.get_unchecked(i);
            if !bucket.occupied((*curr).slot) {
                // We can terminate here
                unlock(ti, i);
                return ((curr as usize).wrapping_sub(start as usize)
                        / mem::size_of::<CuckooRecord<K>>()) as Depth;
            }
            (*curr).key = *bucket.key((*curr).slot);
            unlock(ti, i);
            let hv = hashed_key(&self.hash_state, &(*curr).key);
            /*debug_assert!((*curr).bucket == index_hash(ti, hv) ||
                          (*curr).bucket == alt_index(ti, hv, index_hash(ti, hv)));*/
            // We get the bucket that this slot is on by computing the alternate
            // index of the previous bucket
            i = alt_index(ti, hv, i);
            curr = curr.offset(1);
        }
        /*(*curr).bucket = i;
        lock(ti, i);
        let bucket = (*ti).buckets.get_unchecked(i);
        if !bucket.occupied((*curr).slot) {
            // We can terminate here
            unlock(ti, i);
            return 0;
        }
        (*curr).key = bucket.key((*curr).slot);
        unlock(ti, i);*/
        /*if (x.pathcode == 0) {
            (*curr).bucket = i1;
            lock(ti, i1);
            if !(*ti).buckets.get_unchecked(i1).occupied((*curr).slot) {
                // We can terminate here
                unlock(ti, i1);
                return 0;
            }
            (*curr).key = (*ti).buckets.get_unchecked(i1).key((*curr).slot);
            unlock(ti, i1);
        } else {
            //debug_assert!(x.pathcode == 1);
            (*curr).bucket = i2;
            lock(ti, i2);
            if !(*ti).buckets.get_unchecked(i2).occupied((*curr).slot) {
                // We can terminate here
                unlock(ti, i2);
                return 0;
            }
            (*curr).key = (*ti).buckets.get_unchecked(i2).key((*curr).slot);
            unlock(ti, i2);
        }*/
        /*while curr != end {
            (*curr).bucket = i;
            lock(ti, i);
            let bucket = (*ti).buckets.get_unchecked(i);
            if !bucket.occupied((*curr).slot) {
                // We can terminate here
                unlock(ti, i);
                return (curr as usize).wrapping_sub(start as usize) as Depth;
            }
            (*curr).key = bucket.key((*curr).slot);
            unlock(ti, i);
            let hv = hashed_key((*curr).key);
            /*debug_assert!((*prev).bucket == index_hash(ti, prevhv) ||
                          (*prev).bucket == alt_index(ti, prevhv, index_hash(ti,
                                                                             prevhv)));*/
            // We get the bucket that this slot is on by computing the alternate
            // index of the previous bucket
            i = alt_index(ti, hv, (*curr).bucket);
            curr = curr.offset(1);
            /*let i = alt_index(ti, hv, (*curr).bucket);
            curr = curr.offset(1);
            (*curr).bucket = i;
            lock(ti, i);
            let bucket = (*ti).buckets.get_unchecked(i);
            if !bucket.occupied((*curr).slot) {
                // We can terminate here
                unlock(ti, i);
                return (curr as usize).wrapping_sub(start as usize) as Depth;
            }
            (*curr).key = bucket.key((*curr).slot);
            unlock(ti, i);*/
        }*/
        x.depth
    }

    /// run_cuckoo performs cuckoo hashing on the table in an attempt to free up
    /// a slot on either i1 or i2. On success, the bucket and slot that was freed
    /// up is stored in insert_bucket and insert_slot. In order to perform the
    /// search and the swaps, it has to unlock both i1 and i2, which can lead to
    /// certain concurrency issues, the details of which are explained in the
    /// function. If run_cuckoo returns ok (success), then the slot it freed up
    /// is still locked. Otherwise it is unlocked.
    ///
    /// Unsafe because it assumes that ti is valid, the hazard pointer is set, the locks are taken, and
    /// i1 and i2 are in bounds.
    unsafe fn run_cuckoo(&self,
                         ti: *mut TableInfo<K, V>, i1: usize, i2: usize)
                         -> Result<(usize, usize), CuckooError> where
            K: Copy,
    {
        // We must unlock i1 and i2 here, so that cuckoopath_search and
        // cuckoopath_move can lock buckets as desired without deadlock.
        // cuckoopath_move has to look at either i1 or i2 as its last slot, and
        // it will lock both buckets and leave them locked after finishing. This
        // way, we know that if cuckoopath_move succeeds, then the buckets
        // needed for insertion are still locked. If cuckoopath_move fails, the
        // buckets are unlocked and we try again. This unlocking does present
        // two problems. The first is that another insert on the same key runs
        // and, finding that the key isn't in the table, inserts the key into
        // the table. Then we insert the key into the table, causing a
        // duplication. To check for this, we search i1 and i2 for the key we
        // are trying to insert before doing so (this is done in cuckoo_insert,
        // and requires that both i1 and i2 are locked). Another problem is that
        // an expansion runs and changes table_info, meaning the cuckoopath_move
        // and cuckoo_insert would have operated on an old version of the table,
        // so the insert would be invalid. For this, we check that ti ==
        // table_info.load() after cuckoopath_move, signaling to the outer
        // insert to try again if the comparison fails.
        unlock_two(ti, i1, i2);

        loop {
            // Note that this mem::uninitialized() is perfectly safe, because CuckoRecord is `Copy` and
            // therefore can't have a destructor.
            let mut cuckoo_path: [CuckooRecord<K>; MAX_BFS_PATH_LEN as usize] = mem::uninitialized();
            //let (cuckoo_path, depth) = (mem::uninitialized::<[CuckooRecord<K>; MAX_BFS_PATH_LEN as usize]>(), -1);//cuckoopath_search(ti, /*&mut cuckoo_path,*/ i1, i2);
            let depth = self.cuckoopath_search(ti, &mut cuckoo_path, i1, i2);
            if depth < 0 {
                break;
            }

            if cuckoopath_move(ti, &cuckoo_path, depth as usize, i1, i2) {
                let insert_bucket = cuckoo_path.get_unchecked(0).bucket;
                let insert_slot = cuckoo_path.get_unchecked(0).slot;
                //debug_assert!(insert_bucket == i1 || insert_bucket == i2);
                //debug_assert!(!(*ti).locks.get_unchecked(lock_ind(i1)).try_lock());
                //debug_assert!(!(*ti).locks.get_unchecked(lock_ind(i2)).try_lock());
                //debug_assert!(!(*ti).buckets.get_unchecked(insert_bucket).occupied(insert_slot));
                return if ti == self.table_info.load(Ordering::SeqCst) {
                    Ok((insert_bucket, insert_slot))
                } else {
                    // Unlock i1 and i2 and signal to cuckoo_insert to try again. Since
                    // we set the hazard pointer to be ti, this check isn't susceptible
                    // to an ABA issue, since a new pointer can't have the same address
                    // as ti.
                    unlock_two(ti, i1, i2);
                    Err(CuckooError::UnderExpansion)
                };
            }
        }

        Err(CuckooError::TableFull)
    }

    /// cuckoo_insert tries to insert the given key-value pair into an empty slot
    /// in i1 or i2, performing cuckoo hashing if necessary. It expects the locks
    /// to be taken outside the function, but they are released here, since
    /// different scenarios require different handling of the locks. Before
    /// inserting, it checks that the key isn't already in the table. cuckoo
    /// hashing presents multiple concurrency issues, which are explained in the
    /// function.
    ///
    /// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be
    /// set, and i1 and i2 to be in bounds.
    unsafe fn cuckoo_insert(&self,
                            key: K, val: V,
                            hv: usize, ti: *mut TableInfo<K, V>,
                            i1: usize, i2: usize) -> InsertResult<(), K, V>
        where K: Copy + Eq,
    {
        //const partial_t partial = partial_key(hv);
        let partial = ();
        match try_find_insert_bucket(ti, partial, &key, i1) {
            Err(KeyDuplicated) => {
                unlock_two(ti, i1, i2);
                return Err((KeyDuplicated, key, val));
            },
            res1 => match try_find_insert_bucket(ti, partial, &key, i2) {
                Err(KeyDuplicated) => {
                    unlock_two(ti, i1, i2);
                    return Err((KeyDuplicated, key, val));
                },
                res2 => {
                    if let Ok(res1) = res1 {
                        add_to_bucket(ti, partial, key, val, i1, res1);
                        unlock_two(ti, i1, i2);
                        return Ok(());
                    }
                    if let Ok(res2) = res2 {
                        add_to_bucket(ti, partial, key, val, i2, res2);
                        unlock_two(ti, i1, i2);
                        return Ok(());
                    }
                }
            }
        }
        // we are unlucky, so let's perform cuckoo hashing
        match self.run_cuckoo(ti, i1, i2) {
            Err(CuckooError::UnderExpansion) => {
                // The run_cuckoo operation operated on an old version of the table,
                // so we have to try again. We signal to the calling insert method
                // to try again by returning failure_under_expansion.
                Err((UnderExpansion, key, val))
            },
            Ok((insert_bucket, insert_slot)) => {
                /*debug_assert!(!(*ti).locks.get_unchecked(lock_ind(i1)).try_lock());
                debug_assert!(!(*ti).locks.get_unchecked(lock_ind(i2)).try_lock());
                debug_assert!(!(*ti).buckets.get_unchecked(insert_bucket).occupied(insert_slot));
                debug_assert!(insert_bucket == index_hash(ti, hv) ||
                              insert_bucket == alt_index(ti, hv, index_hash(ti, hv)));*/
                // Since we unlocked the buckets during run_cuckoo, another insert
                // could have inserted the same key into either i1 or i2, so we
                // check for that before doing the insert.
                if cuckoo_contains(&key, hv, ti, i1, i2) {
                    unlock_two(ti, i1, i2);
                    return Err((KeyDuplicated, key, val));
                }
                add_to_bucket(ti, partial, key, val,
                              insert_bucket, insert_slot);
                unlock_two(ti, i1, i2);
                Ok(())
            },
            Err(CuckooError::TableFull) => {
                // assert(st == failure);
                //LIBCUCKOO_DBG("hash table is full (hashpower = %zu, hash_items = %zu,"
                //              "load factor = %.2f), need to increase hashpower\n",
                //              ti->hashpower_, cuckoo_size(ti), cuckoo_loadfactor(ti));
                Err((TableFull, key, val))
            }
        }
    }

    /// We run cuckoo_insert in a loop until it succeeds in insert and upsert, so
    /// we pulled out the loop to avoid duplicating it. This should be called
    /// directly after snapshot_and_lock_two, and by the end of the function, the
    /// hazard pointer will have been unset.
    ///
    /// Unsafe because it expects the locks to be taken, ti to be appropriately set,
    /// and i1 and i2 to be in bounds.
    unsafe fn cuckoo_insert_loop(&self, hazard_pointer: &HazardPointer,
                                 key: K, val: V,
                                 hv: usize, ti: *mut TableInfo<K, V>,
                                 i1: usize, i2: usize) -> InsertResult<(), K, V>
        where K: Copy + Send + Sync,
              V: Send + Sync,
              S: Default + Send + Sync,
    {
        let mut res = self.cuckoo_insert(key, val, hv, ti, i1, i2);
        loop {
            let (key, val) = match res {
                // If the insert failed with failure_key_duplicated, it returns here
                Err((KeyDuplicated, key, val)) => return Err((KeyDuplicated, key, val)),
                // If it failed with failure_under_expansion, the insert operated on
                // an old version of the table, so we just try again.
                Err((UnderExpansion, key, val)) => (key, val),
                // If it's failure_table_full, we have to expand the table before trying
                // again.
                Err((TableFull, key, val)) => {
                    // TODO: Investigate whether adding 1 is alaways safe here.
                    let hashpower = (*ti).hashpower.checked_add(1)
                                                   .unwrap_or_else( || {
                        //println!("capacity error: off by one in cuckoo_insert_loop");
                        intrinsics::abort()
                    });
                    if let Err(()) =
                        self.cuckoo_expand_simple(hazard_pointer, hashpower) {
                        // LIBCUCKOO_DBG("expansion is on-going\n");
                    }
                    (key, val)
                },
                Ok(()) => return Ok(())
            };
            let (ti, i1, i2) = self.snapshot_and_lock_two(hazard_pointer, hv);
            res = self.cuckoo_insert(key, val, hv, ti, i1, i2);
        }
    }

    /// cuckoo_expand_simple is a simpler version of expansion than
    /// cuckoo_expand, which will double the size of the existing hash table. It
    /// needs to take all the bucket locks, since no other operations can change
    /// the table during expansion. If some other thread is holding the expansion
    /// thread at the time, then it will return failure_under_expansion.
    /// n should be a valid size (as per `reserve_calc`).
    /// Unsafe because it expects the hazard pointer to be set.
    ///
    unsafe fn cuckoo_expand_simple(&self, hazard_pointer: &HazardPointer,
                                   n: usize) -> Result<(), ()> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Default + Send + Sync,
    {
        struct TI<K, V>(*mut TableInfo<K, V>);
        unsafe impl<K, V> Send for TI<K, V> where K: Send, V: Send {}
        unsafe impl<K, V> Sync for TI<K, V> where K: Send + Sync, V: Send + Sync {}
        /// insert_into_table is a helper function used by cuckoo_expand_simple to
        /// fill up the new table.
        ///
        /// Unsafe because it expects the old table to be locked, ti to be valid,
        /// the hazard pointer to be set, and i and end to be in bounds.
        unsafe fn insert_into_table<K, V, S>(new_map: &CuckooHashMap<K, V, S>,
                                             TI(old_ti): TI<K, V>,
                                             i: usize,
                                             end: usize) where
                K: Copy + Eq + Hash + Send + Sync,
                V: Send + Sync,
                S: Default + HashState + Send + Sync,
        {
            let mut bucket = (*old_ti).buckets.as_mut_ptr().offset(i as isize);
            //let e = end;
            //println!("Inserting {}..{}", i, e);
            let end = (*old_ti).buckets.as_mut_ptr().offset(end as isize);
            let mut x = 0;
            while bucket != end {
                x = x + 1;
                if x % 100000 == 0 {
                    //println!("x: {} ({}..{})", x + i, i, e);
                }
                let kv = (*bucket).kv.as_mut().unwrap_or_else( || intrinsics::unreachable());
                let mut keys = kv.keys.as_mut_ptr();
                let mut vals = kv.vals.as_mut_ptr();
                for occupied in &mut (*bucket).occupied {
                    if *occupied {
                        *occupied = false;
                        let key = ptr::read(keys);
                        let val = ptr::read(vals);
                        if let Err(_) = new_map.insert(key, val) {
                            //println!("failed inserting into new map somehow");
                            intrinsics::abort();
                        }
                    }
                    keys = keys.offset(1);
                    vals = vals.offset(1);
                }
                bucket = bucket.offset(1);
            }
            //println!("Done inserting {}..{}", i, e);
        }

        //println!("expand");
        let ti = self.snapshot_and_lock_all(hazard_pointer);
        //println!("locked");
        //debug_assert!(ti == self.table_info.load(Ordering::SeqCst));
        let _au = AllUnlocker::new(ti);
        //TODO: Evaluate whether we actually do need this for some reason.  I don't really
        //understand why we would.
        //let _hpu = HazardPointerUnsetter::new(hazard_pointer);
        if n <= (*ti).hashpower {
            // Most likely another expansion ran before this one could grab the
            // locks
            return Err(());
        }

        // Creates a new hash table with hashpower n and adds all the
        // elements from the old buckets

        let ref new_map = Self::with_capacity_and_hash_state(
            hashsize(n).wrapping_mul(SLOT_PER_BUCKET),
            Default::default());
        //let threadnum = num_cpus::get();
        let threadnum = (*ti).num_inserts.len();
        if threadnum == 0 {
            // Pretty sure this should actually be impossible.
            // TODO: refactor this after I make sure.
            //println!("Shouldn't be possible for thread num to equal zero.");
            intrinsics::abort();
        }
        let buckets_per_thread =
            intrinsics::unchecked_udiv(hashsize((*ti).hashpower), threadnum);

        if mem::size_of::<thread::JoinGuard<()>>() != 0 &&
           threadnum.checked_mul(mem::size_of::<thread::JoinGuard<()>>()).is_none() {
            //println!("Overflow calculating join guard vector length");
            intrinsics::abort();
        }
        {
            //println!("computing");
            let hashsize = hashsize((*ti).hashpower);
            let mut vec = Vec::with_capacity(threadnum);
            let mut ptr = vec.as_mut_ptr();
            let mut i = 0;
            while i < threadnum.wrapping_sub(1) {
                let ti = TI(ti);
                let t = thread::scoped( move || {
                    insert_into_table(new_map, ti,
                                      i.wrapping_mul(buckets_per_thread),
                                      i.wrapping_add(1).wrapping_mul(buckets_per_thread));
                });
                ptr::write(ptr, t);
                i += 1;
                vec.set_len(i);
                ptr = ptr.offset(1);
            }
            {
                let ti = TI(ti);
                let t = thread::scoped( move || {
                    insert_into_table(new_map, ti,
                                      i.wrapping_mul(buckets_per_thread),
                                      hashsize);
                });
                ptr::write(ptr, t);
                vec.set_len(threadnum);
            }
        }
        //println!("done computing");
        // Sets this table_info to new_map's. It then sets new_map's
        // table_info to nullptr, so that it doesn't get deleted when
        // new_map goes out of scope
        // NOTE: Relaxed load is likely sufficient here because we are the only thread that
        // could possibly have access to it at the moment (the other threads that knew about it are
        // joined now, and the join formed a synchronization point).  Keep it SeqCst anyway unless
        // it proves to be a bottleneck (note that there is one other possible way for them to be
        // written to--via deleted_unused--but it shouldn't be possible for that to run with a
        // pointer with the same address as the old one at the same time.
        //
        // Not sure about the store... again, keeping it SeqCst for now.
        self.table_info.store(new_map.table_info.load(Ordering::SeqCst), Ordering::SeqCst);
        new_map.table_info.store(ptr::null_mut(), Ordering::SeqCst);

        // Rather than deleting ti now, we store it in old_table_infos. We then
        // run a delete_unused routine to delete all the old table pointers.
        let old_table_infos = &mut *self.old_table_infos.get();
        old_table_infos.push(mem::transmute(ti));
        delete_unused(old_table_infos);
        Ok(())
    }
}

impl<K, V, S> Drop for CuckooHashMap<K, V, S> {
    fn drop(&mut self) {
        let ti = self.table_info.load(Ordering::Relaxed);
        if !ti.is_null() {
            unsafe { mem::transmute::<_, Box<TableInfo<K, V>>>(ti); }
        }
    }
}

unsafe impl<K, V, S> Send for CuckooHashMap<K, V, S> where
    K: Send,
    V: Send,
    S: Send {}

/// Too conservative?  Maybe.
unsafe impl<K, V, S> Sync for CuckooHashMap<K,V, S> where
    K: Send + Sync,
    V: Send + Sync,
    S: Send + Sync {}


impl<K, V, S> Default for CuckooHashMap<K, V, S>
    where K: Eq + Hash,
          /*K: fmt::Debug,*/
          /*V: fmt::Debug,*/
          S: HashState + Default
{
    fn default() -> Self {
        Self::with_capacity_and_hash_state(DEFAULT_SIZE, Default::default())
    }
}

impl<K, V> CuckooHashMap<K, V, DefaultState<SipHasher>>
    where K: Eq + Hash,
          /*K: fmt::Debug,*/
          /*V: fmt::Debug,*/
{
    fn new() -> Self {
        Default::default()
    }
}

/// lock locks the given bucket index.
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, and i is in
/// bounds.
#[inline(always)]
unsafe fn lock<K, V>(ti: *mut TableInfo<K, V>, i: usize) {
    (*ti).locks.get_unchecked(lock_ind(i)).lock();
}

/// unlock unlocks the given bucket index.
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, i is in bounds,
/// and the lock is taken.
#[inline(always)]
unsafe fn unlock<K, V>(ti: *mut TableInfo<K, V>, i: usize) {
    (*ti).locks.get_unchecked(lock_ind(i)).unlock();
}

/// lock_two locks the two bucket indexes, always locking the earlier index
/// first to avoid deadlock. If the two indexes are the same, it just locks
/// one.
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, and i1 and i2
/// are in bounds.
unsafe fn lock_two<K, V>(ti: *mut TableInfo<K, V>, i1: usize, i2: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let locks = &(*ti).locks;
    match i1.cmp(&i2) {
        Less => {
            locks.get_unchecked(i1).lock();
            locks.get_unchecked(i2).lock();
        },
        Greater => {
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i1).lock();
        },
        Equal => {
            locks.get_unchecked(i1).lock()
        },
    }
    /*if (i1 < i2) {
        ti->locks_[i1].lock();
        ti->locks_[i2].lock();
    } else if (i2 < i1) {
        ti->locks_[i2].lock();
        ti->locks_[i1].lock();
    } else {
        ti->locks_[i1].lock();
    }*/
}

/// unlock_two unlocks both of the given bucket indexes, or only one if they
/// are equal. Order doesn't matter here.
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, i1 and i2
/// are in bounds, and the locks are taken.
unsafe fn unlock_two<K, V>(ti: *mut TableInfo<K, V>, i1: usize, i2: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let locks = &(*ti).locks;
    locks.get_unchecked(i1).unlock();
    if i1 != i2 {
        locks.get_unchecked(i2).unlock();
    }
}

/// lock_three locks the three bucket indexes in numerical order.
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, and i1, i2, and
/// i3 are in bounds.
unsafe fn lock_three<K, V>(ti: *mut TableInfo<K, V>, i1: usize, i2: usize, i3: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let i3 = lock_ind(i3);
    /*let locks = &(*ti).locks;
    match (i1.cmp(i2), i2.cmp(i3), i1.cmp(i3)) {
        // If any are the same, we just run lock_two
        (Equal, _, _) | (_, Equal, _) => lock_two(ti, i1, i3),
        (_, _, Equal) => lock_two(ti, i1, i2),
        (Less, Less, /*Less - if Greater something is wrong.*/_) =>  {
            locks.get_unchecked(i1).lock();
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i3).lock();
        },
        (Less, /*Greater*/_, Less) => {
            locks.get_unchecked(i1).lock();
            locks.get_unchecked(i3).lock();
            locks.get_unchecked(i2).lock();
        },
        (Less, /*Greater*/_, /*Greater*/_) => {
            locks.get_unchecked(i3).lock();
            locks.get_unchecked(i1).lock();
            locks.get_unchecked(i2).lock();
        },
        (/*Greater*/_, Less, Less) => {
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i1).lock();
            locks.get_unchecked(i3).lock();
        },
        (/*Greater*/_, Less, /*Greater*/_) => {
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i3).lock();
            locks.get_unchecked(i1).lock();
        },
        _/*(Greater, Greater, Greater)*/ => {
            locks.get_unchecked(i3).lock();
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i1).lock();
        },
    }*/
    // If any are the same, we just run lock_two
    if i1 == i2 {
        lock_two(ti, i1, i3);
    } else if i2 == i3 {
        lock_two(ti, i1, i3);
    } else if i1 == i3 {
        lock_two(ti, i1, i2);
    } else {
        let locks = &(*ti).locks;
        if i1 < i2 {
            if i2 < i3 {
                locks.get_unchecked(i1).lock();
                locks.get_unchecked(i2).lock();
                locks.get_unchecked(i3).lock();
            } else if i1 < i3 {
                locks.get_unchecked(i1).lock();
                locks.get_unchecked(i3).lock();
                locks.get_unchecked(i2).lock();
            } else {
                locks.get_unchecked(i3).lock();
                locks.get_unchecked(i1).lock();
                locks.get_unchecked(i2).lock();
            }
        } else if i2 < i3 {
            if i1 < i3 {
                locks.get_unchecked(i2).lock();
                locks.get_unchecked(i1).lock();
                locks.get_unchecked(i3).lock();
            } else {
                locks.get_unchecked(i2).lock();
                locks.get_unchecked(i3).lock();
                locks.get_unchecked(i1).lock();
            }
        } else {
            locks.get_unchecked(i3).lock();
            locks.get_unchecked(i2).lock();
            locks.get_unchecked(i1).lock();
        }
    }
}

/// unlock_three unlocks the three given buckets
///
/// Unsafe because it assumes that ti is valid, the hazard pointer has been taken, i1, i2, and
/// i3 are in bounds, and all the locks are taken.
unsafe fn unlock_three<K, V>(ti: *mut TableInfo<K, V>, i1: usize, i2: usize, i3: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let i3 = lock_ind(i3);
    let locks = &(*ti).locks;
    locks.get_unchecked(i1).unlock();
    if i2 != i1 {
        locks.get_unchecked(i2).unlock();
    }
    if i3 != i1 && i3 != i2 {
        locks.get_unchecked(i3).unlock();
    }
}

/// AllUnlocker is an object which releases all the locks on the given table
/// info when its destructor is called.
struct AllUnlocker<K, V> {
    ti: *mut TableInfo<K, V>,
}

impl<K, V> AllUnlocker<K, V> {
    /// Unsafe because it assumes ti will be unaliased on drop if not null, and this is not
    /// enforced; also assumes that all of ti's locks are taken.
    unsafe fn new(ti: *mut TableInfo<K, V>) -> Self {
        AllUnlocker { ti: ti }
    }
}

impl<K, V> Drop for AllUnlocker<K, V> {
    fn drop(&mut self) {
        if !self.ti.is_null() {
            unsafe {
                let ti = &*self.ti;
                for lock in &ti.locks[..] {
                    lock.unlock();
                }
            }
        }
    }
}

/// lock_ind converts an index into buckets_ to an index into locks_.
#[inline(always)]
fn lock_ind(bucket_ind: usize) -> usize {
    bucket_ind & (K_NUM_LOCKS - 1)
}

/// hashsize returns the number of buckets corresponding to a given
/// hashpower.
#[inline(always)]
fn hashsize(hashpower: usize) -> usize {
    // TODO: make sure Rust can't UB on too large inputs or we'll make this unsafe.
    1 << hashpower
}

/// hashmask returns the bitmask for the buckets array corresponding to a
/// given hashpower.
#[inline(always)]
fn hashmask(hashpower: usize) -> usize {
    hashsize(hashpower) - 1
}

/// hashed_key hashes the given key.
#[inline(always)]
fn hashed_key<K: ?Sized, S>(hash_state: &S, key: &K) -> usize
    where K: Hash,
          S: HashState,
{
    let mut state = hash_state.hasher();
    key.hash(&mut state);
    state.finish() as usize
}

/// index_hash returns the first possible bucket that the given hashed key
/// could be.
///
/// Unsafe because it assumes that ti is valid and the hazard pointer is set.
#[inline(always)]
unsafe fn index_hash<K, V>(ti: *const TableInfo<K, V>, hv: usize) -> usize {
    hv & hashmask((*ti).hashpower)
}

/// alt_index returns the other possible bucket that the given hashed key
/// could be. It takes the first possible bucket as a parameter. Note that
/// this function will return the first possible bucket if index is the
/// second possible bucket, so alt_index(ti, hv, alt_index(ti, hv,
/// index_hash(ti, hv))) == index_hash(ti, hv).
///
/// Unsafe because it assumes that ti is valid and the hazard pointer is set.
#[inline(always)]
unsafe fn alt_index<K, V>(ti: *const TableInfo<K, V>, hv: usize, index: usize) -> usize {
    // ensure tag is nonzero for the multiply
    // TODO: figure out if this is UB and how to mitigate it if so.
    let tag = (hv >> (*ti).hashpower).wrapping_add(1);
    // 0x5bd1e995 is the hash constant from MurmurHash2
    (index ^ (tag.wrapping_mul(0x5bd1e995))) & hashmask((*ti).hashpower)
}

/*// A constexpr version of pow that we can use for static_asserts
static constexpr size_t const_pow(size_t a, size_t b) {
    return (b == 0) ? 1 : a * const_pow(a, b - 1);
}*/

/// The maximum number of items in a BFS path.
const MAX_BFS_PATH_LEN: u8 = 5;

/// CuckooRecord holds one position in a cuckoo path.
struct CuckooRecord<K> {
    bucket: usize,
    slot: usize,
    key: K,
}

type Depth = i32;

/// b_slot holds the information for a BFS path through the table
#[derive(Clone, Copy)]
#[repr(packed)]
struct BSlot {
    /// The bucket of the last item in the path
    bucket: usize,
    /// a compressed representation of the slots for each of the buckets in
    /// the path. pathcode is sort of like a base-SLOT_PER_BUCKET number, and
    /// we need to hold at most MAX_BFS_PATH_LEN slots. Thus we need the
    /// maximum pathcode to be at least SLOT_PER_BUCKET^(MAX_BFS_PATH_LEN)
    pathcode: usize,
    /// The 0-indexed position in the cuckoo path this slot occupies. It must
    /// be less than MAX_BFS_PATH_LEN, and also able to hold negative values.
    depth: Depth,
}

impl BSlot {
    #[inline(always)]
    fn new(bucket: usize, pathcode: usize, depth: Depth) -> Self {
        //debug_assert!(depth < MAX_BFS_PATH_LEN as Depth);
        BSlot {
            bucket: bucket,
            pathcode: pathcode,
            depth: depth,
        }
    }
}

/*static_assert(const_pow(SLOT_PER_BUCKET, MAX_BFS_PATH_LEN) <
              std::numeric_limits<decltype(pathcode)>::max(),
              "pathcode may not be large enough to encode a cuckoo"
              " path");*/

#[static_assert]
/// The depth type must able to hold a value of
/// MAX_BFS_PATH_LEN - 1");
static MAX_BFS_PATH_FITS_IN_DEPTH: bool = (MAX_BFS_PATH_LEN - 1) as Depth <= i32::MAX;

#[static_assert]
/// The depth type must be able to hold a value of -1
static NEGATIVE_ONE_FITS_IN_DEPTH: bool = -1 as Depth >= i32::MIN;

/// b_queue is the queue used to store b_slots for BFS cuckoo hashing.
#[repr(packed)]
struct BQueue {
    /// A circular array of b_slots
    slots: [BSlot ; MAX_CUCKOO_COUNT],
    /// The index of the head of the queue in the array
    first: usize,
    /// One past the index of the last item of the queue in the array
    last: usize,
}

/// The maximum size of the BFS queue. Unless it's less than
/// SLOT_PER_BUCKET^MAX_BFS_PATH_LEN, it won't really mean anything. If
/// it's a power of 2, then we can quickly wrap around to the beginning
/// of the array, so we do that.
const MAX_CUCKOO_COUNT: usize = 512;

/*static_assert(const_pow(SLOT_PER_BUCKET, MAX_BFS_PATH_LEN) >=
              MAX_CUCKOO_COUNT, "MAX_CUCKOO_COUNT value is too large"
              " to be useful");*/

#[static_assert]
/// MAX_CUCKOO_COUNT should be a power of 2
static MAX_CUCKOO_COUNT_POWER_OF_2: bool = (MAX_CUCKOO_COUNT & (MAX_CUCKOO_COUNT - 1)) == 0;

/// returns the index in the queue after ind, wrapping around if
/// necessary.
fn increment(ind: usize) -> usize {
    (ind.wrapping_add(1)) & (MAX_CUCKOO_COUNT - 1)
}

impl BQueue{
    #[inline(always)]
    fn new() -> Self where BSlot: Copy {
        BQueue {
            // Perfectly safe because `BSlot` is `Copy`.
            slots: unsafe { mem::uninitialized() },
            first: 0,
            last: 0,
        }
    }

    /// unsafe because it assumes that the queue is not full.
    unsafe fn enqueue(&mut self, x: BSlot) {
        // debug_assert!(!self.full());
        *self.slots.get_unchecked_mut(self.last) = x;
        self.last = increment(self.last);
    }

    /// unsafe because it assumes the queue is nonempty.
    unsafe fn dequeue(&mut self, ) -> BSlot {
        // debug_assert!(!self.empty());
        let x = *self.slots.get_unchecked(self.first);
        self.first = increment(self.first);
        x
    }

    fn empty(&self) -> bool {
        self.first == self.last
    }

    fn full(&self) -> bool {
        increment(self.last) == self.first
    }
}

/// cuckoopath_move moves keys along the given cuckoo path in order to make
/// an empty slot in one of the buckets in cuckoo_insert. Before the start of
/// this function, the two insert-locked buckets were unlocked in run_cuckoo.
/// At the end of the function, if the function returns true (success), then
/// the last bucket it looks at (which is either i1 or i2 in run_cuckoo)
/// remains locked. If the function is unsuccessful, then both insert-locked
/// buckets will be unlocked.
///
/// Note that if i1 or i2 is equal to cuckoo_path[0].bucket and depth == 0, this will deadlock,
/// though this is not technically a safety issue.
///
/// Unsafe because it assumes ti is valid, hazard pointer is set, i1 and i2 are in bounds, and
/// depth is in bounds.
unsafe fn cuckoopath_move<K, V>(ti: *mut TableInfo<K, V>,
                                cuckoo_path: &[CuckooRecord<K> ; MAX_BFS_PATH_LEN as usize],
                                mut depth: usize, i1: usize, i2: usize) -> bool where
        K: Copy + Eq,
{
    if depth == 0 {
        // There is a chance that depth == 0, when try_add_to_bucket sees i1
        // and i2 as full and cuckoopath_search finds one empty. In this
        // case, we lock both buckets. If the bucket that cuckoopath_search
        // found empty isn't empty anymore, we unlock them and return false.
        // Otherwise, the bucket is empty and insertable, so we hold the
        // locks and return true.
        let bucket = cuckoo_path.get_unchecked(0).bucket;
        // debug_assert!(bucket == i1 || bucket == i2);
        lock_two(ti, i1, i2);
        return if !(*ti).buckets.get_unchecked(bucket).occupied(cuckoo_path.get_unchecked(0).slot) {
            true
        } else {
            unlock_two(ti, i1, i2);
            false
        }
    }

    while depth > 0 {
        let from = cuckoo_path.as_ptr().offset(depth.wrapping_sub(1) as isize);
        let to = cuckoo_path.as_ptr().offset(depth as isize);
        let fb = (*from).bucket;
        let fs = (*from).slot;
        let tb = (*to).bucket;
        let ts = (*to).slot;

        let mut ob = 0;
        if depth == 1 {
            // Even though we are only swapping out of i1 or i2, we have to
            // lock both of them along with the slot we are swapping to,
            // since at the end of this function, i1 and i2 must be locked.
            ob = if fb == i1 { i2 } else { i1 };
            lock_three(ti, fb, tb, ob);
        } else {
            lock_two(ti, fb, tb);
        }

        // We plan to kick out fs, but let's check if it is still there;
        // there's a small chance we've gotten scooped by a later cuckoo. If
        // that happened, just... try again. Also the slot we are filling in
        // may have already been filled in by another thread, or the slot we
        // are moving from may be empty, both of which invalidate the swap.
        // &mut is safe because we have tkaen the locks.
        let bucket_fb = (*ti).buckets.get_unchecked_mut(fb);
        let bucket_tb = (*ti).buckets.get_unchecked_mut(tb);
        if bucket_fb.key(fs) != &(*from).key ||
           bucket_tb.occupied(ts) ||
           !bucket_fb.occupied(fs) {
            if depth == 1 {
                unlock_three(ti, fb, tb, ob);
            } else {
                unlock_two(ti, fb, tb);
            }
            return false;
        }

        // For now, we know we are "simple" so we skip this part.
        // if !is_simple {
        //     bucket_tb.partial(ts) = bucket_fb.partial(fs);
        // }
        let (key, val) = bucket_fb.erase_kv(fs);
        bucket_tb.set_kv(ts, key, val);
        if depth == 1 {
            // Don't unlock fb or ob, since they are needed in
            // cuckoo_insert. Only unlock tb if it doesn't unlock the same
            // bucket as fb or ob.
            if lock_ind(tb) != lock_ind(fb) &&
               lock_ind(tb) != lock_ind(ob) {
                unlock(ti, tb);
            }
        } else {
            unlock_two(ti, fb, tb);
        }
        depth -= 1;
    }
    true
}

/// try_read_from_bucket will search the bucket for the given key and store
/// the associated value if it finds it.
///
/// Unsafe because it assumes ti is valid, locks are taken, hazard pointer is set, and i is in
/// bounds.
unsafe fn try_read_from_bucket<K, V, P>
                              (ti: *const TableInfo<K, V>, _partial: P,
                               key: &K, i: usize) -> Option<V>
    where K: Copy + Eq,
          V: Copy,
{
    let bucket = (*ti).buckets.get_unchecked(i);
    for j in Range::new(0, SLOT_PER_BUCKET) {
        if !bucket.occupied(j) {
            continue;
        }
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
        //     continue;
        // }
        if key == bucket.key(j) {
            return Some(*bucket.val(j));
        }
    }
    None
}

/// check_in_bucket will search the bucket for the given key and return true
/// if the key is in the bucket, and false if it isn't.
///
/// Unsafe because it assumes ti is valid, locks are taken, hazard pointer is set, and i is in
/// bounds.
unsafe fn check_in_bucket<K, V, P>
                         (ti: *const TableInfo<K, V>, _partial: P,
                          key: &K, i: usize) -> bool
    where K: Copy + Eq,
{
    let bucket = (*ti).buckets.get_unchecked(i);
    for j in Range::new(0, SLOT_PER_BUCKET) {
        if !bucket.occupied(j) {
            continue;
        }
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
        //     continue;
        // }
        if key == bucket.key(j) {
            return true;
        }
    }
    false
}

/// add_to_bucket will insert the given key-value pair into the slot.
///
/// This function is unsafe because it relies on ti being correctly set, the hazard pointer being
/// set, the correct locks having been taken, and the indexes having been chosen correctly (in the
/// original version, this last bit is *sort of* checked, but not really [there's no bounds
/// checking]).
unsafe fn add_to_bucket<K, V, P>(ti: *mut TableInfo<K, V>, _partial: P,
                                 key: K, val: V,
                                 i: usize, j: usize)
    where K: Copy + Eq,
          /*K: fmt::Debug,*/
          /*V: fmt::Debug,*/
{
    let bucket = (*ti).buckets.get_unchecked_mut(i);
    //debug_assert!(!bucket.occupied(j));
    // For now, we know we are "simple" so we skip this part.
    //if (!is_simple) {
    //    ti->buckets_[i].partial(j) = partial;
    //}
    // &mut should be safe; this function is potected by the lock.
    bucket.set_kv(j, key, val);
    let counterid = (*COUNTER_ID.0.get()).unwrap_or_else( || intrinsics::unreachable() );
    let num_inserts = (*ti).num_inserts.get_unchecked(counterid);
    num_inserts.fetch_add_relaxed(1);
}

/// try_find_insert_bucket will search the bucket and store the index of an
/// empty slot if it finds one, or -1 if it doesn't. Regardless, it will
/// search the entire bucket and return false if it finds the key already in
/// the table (duplicate key error) and true otherwise.
///
/// This function is unsafe because it relies on ti being correctly set, the hazard pointer being
/// set, the correct locks having been taken, and the indexes having been chosen correctly.
unsafe fn try_find_insert_bucket<K, V, P>(ti: *mut TableInfo<K, V>, _partial: P,
                                          key: &K,
                                          i: usize)
                                          -> Result<usize, InsertError>
    where K: Copy + Eq,
{
    let mut found_empty = Err(TableFull);
    let bucket = (*ti).buckets.get_unchecked(i);
    for k in Range::new(0, SLOT_PER_BUCKET) {
        if bucket.occupied(k) {
            // For now, we know we are "simple" so we skip this part.
            // if (!is_simple && partial != ti->buckets_[i].partial(k)) {
            //     continue;
            // }
            if key == bucket.key(k) {
                return Err(KeyDuplicated);
            }
        } else {
            if let Err(_) = found_empty {
                found_empty = Ok(k);
            }
        }
    }
    found_empty
}
/// try_del_from_bucket will search the bucket for the given key, and set the
/// slot of the key to empty if it finds it.
///
/// This function is unsafe because it relies on ti being correctly set, the hazard pointer being
/// set, the correct locks having been taken, and the index being in bounds.
unsafe fn try_del_from_bucket<K, V, P>(ti: *mut TableInfo<K, V>, _partial: P,
                                       key: &K,
                                       i: usize)
                                       -> Option<V>
    where K: Eq,
{
    // Safe because we have the lock.
    let bucket = (*ti).buckets.get_unchecked_mut(i);
    for j in Range::new(0, SLOT_PER_BUCKET) {
        if !bucket.occupied(j) {
            continue;
        }
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        if bucket.key(j) == key {
            let (_, v) = bucket.erase_kv(j);
            let counterid = (*COUNTER_ID.0.get()).unwrap_or_else( || intrinsics::unreachable() );
            let num_deletes = (*ti).num_deletes.get_unchecked(counterid);
            num_deletes.fetch_add_relaxed(1);
            return Some(v);
        }
    }
    None
}

/// try_update_bucket will search the bucket for the given key and change its
/// associated value if it finds it.
///
/// This function is unsafe because it relies on ti being correctly set, the hazard pointer being
/// set, the correct locks having been taken, and the index being in bounds.
unsafe fn try_update_bucket<K, V, P>(ti: *mut TableInfo<K, V>, _partial: P,
                                     key: &K, value: V, i: usize)
                                     -> Result<V, V> where
        K: Eq,
        V: Copy,
{
    // Safe because we have the lock.
    let bucket = (*ti).buckets.get_unchecked_mut(i);
    for j in Range::new(0, SLOT_PER_BUCKET) {
        if !bucket.occupied(j) {
            continue;
        }
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        if bucket.key(j) == key {
            return Ok(mem::replace(bucket.val_mut(j), value))
        }
    }
    Err(value)
}

/// try_update_bucket_fn will search the bucket for the given key and change
/// its associated value with the given function if it finds it.
///
/// This function is unsafe because it relies on ti being correctly set, the hazard pointer being
/// set, the correct locks having been taken, and the index being in bounds.
unsafe fn try_update_bucket_fn<K, V, P, F, T>(ti: *mut TableInfo<K, V>, _partial: P,
                                              key: &K, updater: &mut F, i: usize)
                                              -> Option<T> where
        K: Eq,
        F: FnMut(&mut V) -> T,
{
    // Safe because we have the lock.
    let bucket = (*ti).buckets.get_unchecked_mut(i);
    for j in Range::new(0, SLOT_PER_BUCKET) {
        if !bucket.occupied(j) {
            continue;
        }
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        if bucket.key(j) == key {
            let res = updater(bucket.val_mut(j));
            return Some(res);
        }
    }
    None
}

/// cuckoo_find searches the table for the given key and value, storing the
/// value in the val if it finds the key. It expects the locks to be taken
/// and released outside the function.
///
/// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be
/// set, and i1 and i2 to be in bounds.
unsafe fn cuckoo_find<K, V>(key: &K,
                            _hv: usize, ti: *const TableInfo<K, V>,
                            i1: usize, i2: usize) -> Option<V> where
        K: Copy + Eq,
        V: Copy,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    try_read_from_bucket(ti, partial, key, i1)
        .or_else( || try_read_from_bucket(ti, partial, key, i2))
}

/// cuckoo_contains searches the table for the given key, returning true if
/// it's in the table and false otherwise. It expects the locks to be taken
/// and released outside the function.
///
/// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be
/// set, and i1 and i2 to be in bounds.
unsafe fn cuckoo_contains<K, V>(key: &K,
                                _hv: usize, ti: *const TableInfo<K, V>,
                                i1: usize, i2: usize) -> bool
    where K: Copy + Eq,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    check_in_bucket(ti, partial, key, i1)
        || check_in_bucket(ti, partial, key, i2)
}

/// cuckoo_delete searches the table for the given key and sets the slot with
/// that key to empty if it finds it. It expects the locks to be taken and
/// released outside the function.
///
/// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be set,
/// and i1 and i2 to be in bounds.
unsafe fn cuckoo_delete<K, V>(key: &K,
                              _hv: usize, ti: *mut TableInfo<K, V>,
                              i1: usize, i2: usize) -> Option<V>
    where K: Eq,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    match try_del_from_bucket(ti, partial, key, i1) {
        v @ Some(_) => v,
        None => try_del_from_bucket(ti, partial, key, i2)
    }
}

/// cuckoo_update searches the table for the given key and updates its value
/// if it finds it. It expects the locks to be taken and released outside the
/// function.
///
/// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be set,
/// and i1 and i2 to be in bounds.
unsafe fn cuckoo_update<K,V>(key: &K, val: V,
                             _hv: usize, ti: *mut TableInfo<K, V>,
                             i1: usize, i2: usize) -> Result<V, V> where
        K: Eq,
        V: Copy,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    match try_update_bucket(ti, partial, key, val, i1) {
        v @ Ok(_) => v,
        Err(val) => try_update_bucket(ti, partial, key, val, i2)
    }
}

/// cuckoo_update_fn searches the table for the given key and runs the given
/// function on its value if it finds it, assigning the result of the
/// function to the value. It expects the locks to be taken and released
/// outside the function.
///
/// Unsafe because it expects the locks to be taken, ti to be valid, the hazard pointer to be set,
/// and i1 and i2 to be in bounds.
unsafe fn cuckoo_update_fn<K, V, F, T>(key: &K, updater: &mut F,
                                       _hv: usize, ti: *mut TableInfo<K, V>,
                                       i1: usize, i2: usize) -> Option<T> where
        K: Eq,
        F: FnMut(&mut V) -> T,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    match try_update_bucket_fn(ti, partial, key, updater, i1) {
        v @ Some(_) => v,
        None => try_update_bucket_fn(ti, partial, key, updater, i2)
    }
}

/// cuckoo_size returns the number of elements in the given table.
/// Unsafe because it assumes that ti is valid and that the snapshot has been taken.
/// The number of elements is approximate and may be negative.
unsafe fn cuckoo_size<K, V>(ti: *const TableInfo<K, V>) -> isize {
    let mut inserts = 0usize;
    let mut deletes = 0usize;

    let mut insert = (*ti).num_inserts.iter();
    let mut delete = (*ti).num_deletes.iter();
    while let Some(insert) = insert.next() {
        let delete = delete.next().unwrap_or_else(|| intrinsics::unreachable());
        // We use unordered loads here because we don't care about accuracy and grabbing ti should
        // have given us enough of a fence to ensure memory safety.
        inserts = inserts.wrapping_add(insert.load_unordered());
        deletes = deletes.wrapping_add(delete.load_unordered());
    }
    (inserts as isize).wrapping_sub(deletes as isize)
}

/// cuckoo_loadfactor returns the load factor of the given table.
/// Unsafe because it assumes that ti is valid and that the snapshot has been taken.
/// The load factor is approximate and may be negative.
unsafe fn cuckoo_loadfactor<K, V>(ti: *const TableInfo<K, V>) -> f64 {
    cuckoo_size(ti) as f64 / SLOT_PER_BUCKET as f64 / hashsize((*ti).hashpower) as f64
}

#[cfg(test)]
mod tests {
    use super::CuckooHashMap;

    #[test]
    fn make_hashmap() {
        let foo = CuckooHashMap::<u64, u64>::default();
    }
}
