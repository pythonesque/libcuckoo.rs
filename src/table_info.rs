use num_cpus;
use std::cell::UnsafeCell;
use std::cmp::Ordering::{Less, Greater, Equal};
use std::intrinsics;
use std::mem;
use std::ptr;
use super::hazard_pointer::HazardPointerSet;
use super::spinlock::SpinLock;
use super::sys::arch::cpuid;

/// SLOT_PER_BUCKET is the maximum number of keys per bucket
pub const SLOT_PER_BUCKET: usize = 4;

/// number of locks in the locks_ array
const K_NUM_LOCKS: usize = 1 << 16;

//#[unsafe_no_drop_flag]
pub struct BucketInner<K, V> {
    pub keys: [K ; SLOT_PER_BUCKET],

    pub vals: [V ; SLOT_PER_BUCKET],
}

/// The Bucket type holds SLOT_PER_BUCKET keys and values, and a occupied
/// bitset, which indicates whether the slot at the given bit index is in
/// the table or not. It uses aligned_storage arrays to store the keys and
/// values to allow constructing and destroying key-value pairs in place.
//#[unsafe_no_drop_flag]
pub struct Bucket<K, V> {
    pub kv: Option<BucketInner<K, V>>,

    pub occupied: [bool; SLOT_PER_BUCKET],
}

impl<K, V> Bucket<K, V> {
    /// Unsafe because it does not perform bounds checking.
    #[inline(always)]
    pub unsafe fn occupied(&self, ind: usize) -> bool {
        *self.occupied.get_unchecked(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    pub unsafe fn key(&self, ind: usize) -> &K {
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
    pub unsafe fn val(&self, ind: usize) -> &V {
        self.kv.as_ref().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked(ind)
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    #[inline(always)]
    pub unsafe fn val_mut(&mut self, ind: usize) -> &mut V {
        self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked_mut(ind)
    }

    /// Unsafe because it does not perform bounds checking.
    /// Does not check to make sure location was not already occupied; can leak.
    pub unsafe fn set_kv(&mut self, ind: usize, k: K, v: V) {
        *self.occupied.get_unchecked_mut(ind) = true;
        let kv = self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable());
        ptr::write(kv.keys.as_mut_ptr().offset(ind as isize), k);
        ptr::write(kv.vals.as_mut_ptr().offset(ind as isize), v);
    }

    /// Unsafe because it does not perform bounds checking or ensure the location was already
    /// occupied.
    pub unsafe fn erase_kv(&mut self, ind: usize) -> (K, V) {
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

    pub fn clear(&mut self) {
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
pub struct CacheInt {
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
    pub fn load_unordered(&self) -> usize {
        unsafe { intrinsics::atomic_load_unordered(self.num.get()) }
    }

    #[inline(always)]
    fn store_relaxed(&self, val: usize) {
        unsafe { intrinsics::atomic_store_relaxed(self.num.get(), val); }
    }

    #[inline(always)]
    /// Unsafe because it's not guaranteed this is unaliased.
    pub unsafe fn store_notatomic(&self, val: usize) {
        *self.num.get() = val;
    }

    #[inline(always)]
    pub fn fetch_add_relaxed(&self, val: usize) -> usize {
        unsafe { intrinsics::atomic_xadd_relaxed(self.num.get(), val) }
    }
}

pub struct Counter(pub UnsafeCell<Option<usize>>);

unsafe impl Sync for Counter {}

/// counterid stores the per-thread counter index of each thread.
#[cfg(feature = "counter")]
#[thread_local] pub static COUNTER_ID: Counter = Counter(UnsafeCell::new(None));

/// check_counterid checks if the counterid has already been determined. If
/// not, it assigns a counterid to the current thread by picking a random
/// core. This should be called at the beginning of any function that changes
/// the number of elements in the table.
#[inline(always)]
#[cfg(feature = "counter")]
pub fn check_counterid() {
    unsafe {
        let counterid = COUNTER_ID.0.get();
        if (*counterid).is_none() {
            *counterid = Some(cpuid() as usize);
        };
    }
}
#[cfg(not(feature = "counter"))]
pub fn check_counterid() {}


/// TableInfo contains the entire state of the hashtable. We allocate one
/// TableInfo pointer per hash table and store all of the table memory in it,
/// so that all the data can be atomically swapped during expansion.
pub struct TableInfo<K, V> {
    /// 2**hashpower is the number of buckets
    pub hashpower: usize,

    /// vector of buckets
    pub buckets: Vec<UnsafeCell<Bucket<K, V>>>,

    /// array of locks
    pub locks: [SpinLock ; K_NUM_LOCKS],

    /// per-core counters for the number of inserts and deletes
    pub num_inserts: Vec<CacheInt>,
    pub num_deletes: Vec<CacheInt>,
}

// I believe Send is required since after you take a lock you need to be able to extract a bucket,
// but I may be mistaken.
//unsafe impl<'a, K, V> Sync for HazardPointerSet<'a, TableInfo<K, V>> where
//    K: Send + Sync,
//    V: Send + Sync, {}

impl<K, V> TableInfo<K, V> {
    /// The constructor allocates the memory for the table. It allocates one
    /// cacheint for each core in num_inserts and num_deletes.
    pub fn new(hashpower: usize) -> Box<UnsafeCell<Self>> /*where
        K: fmt::Debug,
        V: fmt::Debug,*/
    {
        fn from_fn<F, T>(n: usize, f: F) -> Vec<T> where
                F: Fn() -> T,
        {
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
            let ti = box UnsafeCell {
                value: TableInfo {
                    hashpower: hashpower,
                    buckets: from_fn(hashsize(hashpower), || UnsafeCell::new(Bucket::new())),
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

pub struct LockTwo<'a, K, V> {
    pub ti: HazardPointerSet<'a, TableInfo<K, V>>,
    pub i1: usize,
    pub i2: usize,
}

impl<'a, K, V> LockTwo<'a, K, V> {
    /// Unsafe because it assumes that i1 and i2 are in bounds and locked.
    pub unsafe fn release(self) {
        unlock_two(&self.ti, self.i1, self.i2);
    }

    /// Unsafe because it assumes that i1 is in bounds and locked.
    pub unsafe fn bucket1(&mut self) -> (&mut Bucket<K, V>, &TableInfo<K, V>) {
        (&mut *self.ti.buckets.get_unchecked(self.i1).get(), &self.ti)
    }

    /// Unsafe because it assumes that i2 is in bounds and locked.
    pub unsafe fn bucket2(&mut self) -> (&mut Bucket<K, V>, &TableInfo<K, V>) {
        (&mut *self.ti.buckets.get_unchecked(self.i2).get(), &self.ti)
    }
}

/// lock locks the given bucket index.
///
/// Unsafe because it assumes that i is in bounds.
#[inline(always)]
pub unsafe fn lock<K, V>(ti: &TableInfo<K, V>, i: usize) {
    ti.locks.get_unchecked(lock_ind(i)).lock();
}

/// unlock unlocks the given bucket index.
///
/// Unsafe because it assumes that i is in bounds and locked.
#[inline(always)]
pub unsafe fn unlock<K, V>(ti: &TableInfo<K, V>, i: usize) {
    ti.locks.get_unchecked(lock_ind(i)).unlock();
}

/// lock_two locks the two bucket indexes, always locking the earlier index
/// first to avoid deadlock. If the two indexes are the same, it just locks
/// one.
///
/// Unsafe because it assumes that i1 and i2 are in bounds.
pub unsafe fn lock_two<K, V>(ti: &TableInfo<K, V>, i1: usize, i2: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let locks = &ti.locks;
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
/// Unsafe because it assumes that the locks are taken and i1 and i2 are in bounds.
pub unsafe fn unlock_two<K, V>(ti: &TableInfo<K, V>, i1: usize, i2: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let locks = &ti.locks;
    locks.get_unchecked(i1).unlock();
    if i1 != i2 {
        locks.get_unchecked(i2).unlock();
    }
}

/// lock_three locks the three bucket indexes in numerical order.
///
/// Unsafe because it assumes that i1, i2, and i3 are in bounds.
pub unsafe fn lock_three<K, V>(ti: &TableInfo<K, V>, i1: usize, i2: usize, i3: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let i3 = lock_ind(i3);
    /*let locks = &ti.locks;
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
        let locks = &ti.locks;
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
/// Unsafe because it assumes that i1, i2, and i3 are in bounds, and all the locks are taken.
pub unsafe fn unlock_three<K, V>(ti: &TableInfo<K, V>, i1: usize, i2: usize, i3: usize) {
    let i1 = lock_ind(i1);
    let i2 = lock_ind(i2);
    let i3 = lock_ind(i3);
    let locks = &ti.locks;
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
pub struct AllUnlocker<'a, K, V> where
        K: 'a,
        V: 'a,
{
    ti: &'a TableInfo<K, V>,
}

impl<'a, K, V> AllUnlocker<'a, K, V> {
    /// Unsafe because it assumes that all of ti's locks are taken.
    pub unsafe fn new(ti: &'a TableInfo<K, V>) -> Self {
        AllUnlocker { ti: ti }
    }
}

impl<'a, K, V> Drop for AllUnlocker<'a, K, V> {
    fn drop(&mut self) {
        for lock in &self.ti.locks[..] {
            lock.unlock();
        }
    }
}

/// lock_ind converts an index into buckets_ to an index into locks_.
#[inline(always)]
pub fn lock_ind(bucket_ind: usize) -> usize {
    bucket_ind & (K_NUM_LOCKS - 1)
}

/// hashsize returns the number of buckets corresponding to a given
/// hashpower.
/// Invariant: always returns a nonzero value.
#[inline(always)]
pub fn hashsize(hashpower: usize) -> usize {
    // TODO: make sure Rust can't UB on too large inputs or we'll make this unsafe.
    1 << hashpower
}
