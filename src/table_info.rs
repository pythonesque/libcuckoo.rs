use num_cpus;
use std::cell::UnsafeCell;
#[cfg(feature = "counter")]
use std::cell::Cell;
use std::cmp::Ordering::{Less, Greater, Equal};
use std::intrinsics;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ptr;
use super::hazard_pointer::HazardPointerSet;
use super::spinlock::SpinLock;
#[cfg(feature = "counter")]
use super::sys::arch::cpuid;

/// SLOT_PER_BUCKET is the maximum number of keys per bucket
pub const SLOT_PER_BUCKET: usize = 4;

/// number of locks in the locks_ array
pub const K_NUM_LOCKS: usize = 1 << 16;

//#[unsafe_no_drop_flag]
pub struct BucketInner<K, V> {
    pub keys: [K ; SLOT_PER_BUCKET],

    pub vals: [V ; SLOT_PER_BUCKET],
}

/// Guaranteed to hold an index that is in-bounds for any bucket's occupied vector(i.e. the index
/// is less than SLOT_PER_BUCKET).
#[derive(Clone,Copy)]
pub struct SlotIndex(usize);

impl SlotIndex {
    #[inline]
    pub fn new(slot: usize) -> Self {
        SlotIndex(slot % SLOT_PER_BUCKET)
    }
}

impl Deref for SlotIndex {
    type Target = usize;

    fn deref(&self) -> &usize {
        &self.0
    }
}

pub struct SlotIndexIter {
    i: usize,
}

impl SlotIndexIter {
    #[inline]
    pub fn new() -> Self {
        SlotIndexIter { i: 0 }
    }
}

impl Iterator for SlotIndexIter
{
    type Item = SlotIndex;

    #[inline]
    fn next(&mut self) -> Option<SlotIndex> {
        // FIXME #24660: this may start returning Some after returning
        // None if the + overflows. This is OK per Iterator's
        // definition, but it would be really nice for a core iterator
        // like `x..y` to be as well behaved as
        // possible. Unfortunately, for types like `i32`, LLVM
        // mishandles the version that places the mutation inside the
        // `if`: it seems to optimise the `Option<i32>` in a way that
        // confuses it.
        let mut n = self.i + 1;
        mem::swap(&mut n, &mut self.i);
        if n < SLOT_PER_BUCKET {
            Some(SlotIndex(n))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (SLOT_PER_BUCKET - self.i, Some(SLOT_PER_BUCKET - self.i))
    }
}

pub struct Slot<'a, K: 'a, V: 'a> {
    slot: &'a Bucket<K, V>,
    index: SlotIndex,
}

impl<'a, K, V> Slot<'a, K, V> {
    pub fn key(&self) -> &K {
        unsafe {
            self.slot.key(self.index)
        }
    }

    pub fn val(&self) -> &V {
        unsafe {
            self.slot.val(self.index)
        }
    }
}

pub struct SlotIter<'a, K: 'a, V: 'a> {
    slot: &'a Bucket<K, V>,
    iter: SlotIndexIter,
}

impl<'a, K, V> Iterator for SlotIter<'a, K, V> {
    type Item = Slot<'a, K, V>;

    #[inline]
    fn next(&mut self) -> Option<Slot<'a, K, V>> {
        for i in &mut self.iter {
            if self.slot.occupied(i) {
                return Some(Slot { slot: self.slot, index: i });
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct SlotMut<'a, K: 'a, V: 'a> {
    slot: *mut Bucket<K, V>,
    index: SlotIndex,
    marker: PhantomData<&'a mut Bucket<K, V>>
}

impl<'a, K, V> SlotMut<'a, K, V> {
    pub fn key(&mut self) -> &mut K {
        unsafe {
            (*self.slot).key_mut(self.index)
        }
    }

    pub fn val(&mut self) -> &mut V {
        unsafe {
            (*self.slot).val_mut(self.index)
        }
    }

    pub fn erase(self) -> (K, V) {
        unsafe {
            (*self.slot).erase_kv(self.index)
        }
    }
}

pub struct SlotIterMut<'a, K: 'a, V: 'a> {
    slot: *mut Bucket<K, V>,
    iter: SlotIndexIter,
    marker: PhantomData<&'a mut Bucket<K, V>>
}

impl<'a, K, V> Iterator for SlotIterMut<'a, K, V> {
    type Item = SlotMut<'a, K, V>;

    #[inline]
    fn next(&mut self) -> Option<SlotMut<'a, K, V>> {
        for i in &mut self.iter {
            unsafe {
                if (*self.slot).occupied(i) {
                    return Some(SlotMut { slot: self.slot, index: i, marker: PhantomData });
                }
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
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
    #[inline(always)]
    pub fn occupied(&self, SlotIndex(ind): SlotIndex) -> bool {
        unsafe {
            *self.occupied.get_unchecked(ind)
        }
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    pub unsafe fn key(&self, SlotIndex(ind): SlotIndex) -> &K {
        self.kv.as_ref().unwrap_or_else( || intrinsics::unreachable()).keys.get_unchecked(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    unsafe fn key_mut(&mut self, SlotIndex(ind): SlotIndex) -> &mut K {
        self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable()).keys.get_unchecked_mut(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    pub unsafe fn val(&self, SlotIndex(ind): SlotIndex) -> &V {
        self.kv.as_ref().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    pub unsafe fn val_mut(&mut self, SlotIndex(ind): SlotIndex) -> &mut V {
        self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable()).vals.get_unchecked_mut(ind)
    }

    /// Does not check to make sure location was not already occupied; can leak.
    pub fn set_kv(&mut self, SlotIndex(ind): SlotIndex, k: K, v: V) {
        unsafe {
            *self.occupied.get_unchecked_mut(ind) = true;
            let kv = self.kv.as_mut().unwrap_or_else( || intrinsics::unreachable());
            ptr::write(kv.keys.as_mut_ptr().offset(ind as isize), k);
            ptr::write(kv.vals.as_mut_ptr().offset(ind as isize), v);
        }
    }

    /// Unsafe because it does not ensure the location was already occupied.
    pub unsafe fn erase_kv(&mut self, SlotIndex(ind): SlotIndex) -> (K, V) {
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

    pub fn slots<'a>(&'a self) -> SlotIter<'a, K, V> {
        SlotIter { slot: self, iter: SlotIndexIter::new() }
    }

    pub fn slots_mut<'a>(&'a mut self) -> SlotIterMut<'a, K, V> {
        SlotIterMut { slot: self, iter: SlotIndexIter::new(), marker: PhantomData }
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

#[cfg(feature = "counter")]
struct Counter(Cell<Option<usize>>);

#[cfg(feature = "counter")]
unsafe impl Sync for Counter {}

/// When set, guaranteed to be set and in bounds for the snapshot with the same lifetime.
pub struct CounterIndex<'a> {
    #[cfg(feature = "counter")]
    pub index: usize,
    marker: InvariantLifetime<'a>,
}

/// counterid stores the per-thread counter index of each thread.
#[cfg(feature = "counter")]
#[thread_local] static COUNTER_ID: Counter = Counter(Cell::new(None));

/// check_counterid checks if the counterid has already been determined. If
/// not, it assigns a counterid to the current thread by picking a random
/// core. This should be called at the beginning of any function that changes
/// the number of elements in the table.
#[inline(always)]
#[cfg(feature = "counter")]
pub fn check_counterid<'a, K, V>(ti: &Snapshot<'a, K, V>) -> CounterIndex<'a> {
    let Counter(ref counterid) = COUNTER_ID;
    let counterid = match counterid.get() {
        Some(counterid) => counterid,
        None => {
            let id = cpuid() as usize;
            counterid.set(Some(id));
            id
        }
    };
    // `num_inserts.len()` and `num_deletes.len()` should be identical.
    // Note that the check probably isn't *really* necessary.  Actually, I'm kind of tempted
    // to remove it, since if it goes wrong num_cpus or cpuid is screwy, but considering that
    // this code path is off by default I will leave it in for now.
    CounterIndex {
        index: if counterid > ti.num_inserts.len() { 0 } else { counterid },
        marker: InvariantLifetime::new(),
    }
}
#[cfg(not(feature = "counter"))]
pub fn check_counterid<'a, K, V>(_: &Snapshot<'a, K, V>) -> CounterIndex<'a> {
    CounterIndex { marker: InvariantLifetime::new() }
}

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

        // Pretty sure this should actually be impossible, but check anyway to be extra sure.
        if num_cpus == 0 {
            unsafe { intrinsics::abort(); }
        }

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

#[allow(raw_pointer_derive)]
#[derive(Clone,Copy)]
struct InvariantLifetime<'id>(
    PhantomData<*mut &'id ()>);

impl<'id> InvariantLifetime<'id> {
    #[inline]
    fn new() -> InvariantLifetime<'id> {
        InvariantLifetime(PhantomData)
    }
}

/// Invariant that must be preserved: only one ti per Snapshot<'a, K, V>.
#[derive(Clone,Copy)]
pub struct Snapshot<'a, K: 'a, V: 'a> {
    ti: &'a TableInfo<K, V>,
    marker: InvariantLifetime<'a>
}

impl<'a, K, V> Snapshot<'a, K, V> {
    /// Unsafe because it does not enfoce that there is only one ti per Snapshot<'a, K, V>.
    pub unsafe fn new(ti: &'a HazardPointerSet<'a, TableInfo<K, V>>) -> Self {
        Snapshot { ti: &ti, marker: InvariantLifetime::new() }
    }
}

impl<'a, K, V> Deref for Snapshot<'a, K, V> {
    type Target = TableInfo<K, V>;

    fn deref(&self) -> &TableInfo<K, V> {
        self.ti
    }
}

/// Guaranteed to hold an index that is in-bounds for all buckets in the `TableInfo<K, V>` with
/// `Snapshot<'a, K, V>`.  The guarantee that must be preserved is that there is precisely one
/// `Snapshot<'a, K, V>` for a `'a`, which ensures that we don't actually need to hold on to a
/// reference to the `TableInfo<K, V>` itself.  We uphold this guarantee using the runSt trick
/// from Haskell (see also `BTreeMap`).
#[derive(Clone,Copy)]
pub struct BucketIndex<'a> {
    bucket: usize,
    marker: InvariantLifetime<'a>
}

impl<'a> BucketIndex<'a> {
    #[inline]
    pub fn new<K, V>(ti: &Snapshot<'a, K, V>, bucket: usize) -> Self {
        BucketIndex {bucket: index_hash(ti, bucket), marker: InvariantLifetime::new() }
    }
}

impl<'a> Deref for BucketIndex<'a> {
    type Target = usize;

    fn deref(&self) -> &usize {
        &self.bucket
    }
}

pub struct LockTwo<'a> {
    pub i1: BucketIndex<'a>,
    pub i2: BucketIndex<'a>,
    dummy: (),
}

impl<'a> LockTwo<'a> {
    /// Unsafe because it assumes that i1 and i2 are in bounds and locked.
    #[inline]
    pub unsafe fn new(i1: BucketIndex<'a>, i2: BucketIndex<'a>) -> Self {
        LockTwo { i1: i1, i2: i2, dummy: () }
    }

    pub fn release<K, V>(self, ti: &Snapshot<'a, K, V>) {
        unsafe {
            unlock_two(ti, self.i1, self.i2);
        }
    }

    pub fn bucket1<K, V>(&mut self, ti: &Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i1).get()
        }
    }

    pub fn bucket2<K, V>(&mut self, ti: &Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i2).get()
        }
    }
}

/// lock locks the given bucket index.
#[inline(always)]
pub fn lock<'a, K, V>(ti: &Snapshot<'a, K, V>, i: BucketIndex<'a>) {
    unsafe {
        ti.locks.get_unchecked(lock_ind(*i)).lock();
    }
}

/// unlock unlocks the given bucket index.
///
/// Unsafe because it assumes that i is locked.
#[inline(always)]
pub unsafe fn unlock<'a, K, V>(ti: &Snapshot<'a, K, V>, i: BucketIndex<'a>) {
    ti.locks.get_unchecked(lock_ind(*i)).unlock();
}

/// lock_two locks the two bucket indexes, always locking the earlier index
/// first to avoid deadlock. If the two indexes are the same, it just locks
/// one.
pub fn lock_two<'a, K, V>(ti: &Snapshot<'a, K, V>,
                          i1: BucketIndex<'a>, i2: BucketIndex<'a>) {
    unsafe {
        let i1 = lock_ind(*i1);
        let i2 = lock_ind(*i2);
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
}

/// unlock_two unlocks both of the given bucket indexes, or only one if they
/// are equal. Order doesn't matter here.
///
/// Unsafe because it assumes that the locks are taken.
pub unsafe fn unlock_two<'a, K, V>(ti: &Snapshot<'a, K, V>,
                                   i1: BucketIndex<'a>, i2: BucketIndex<'a>) {
    let i1 = lock_ind(*i1);
    let i2 = lock_ind(*i2);
    let locks = &ti.locks;
    locks.get_unchecked(i1).unlock();
    if i1 != i2 {
        locks.get_unchecked(i2).unlock();
    }
}

/// lock_three locks the three bucket indexes in numerical order.
pub fn lock_three<'a, K, V>(ti: &Snapshot<'a, K, V>,
                            b1: BucketIndex<'a>, b2: BucketIndex<'a>, b3: BucketIndex<'a>) {
    unsafe {
        let i1 = lock_ind(*b1);
        let i2 = lock_ind(*b2);
        let i3 = lock_ind(*b3);
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
            lock_two(ti, b1, b3);
        } else if i2 == i3 {
            lock_two(ti, b1, b3);
        } else if i1 == i3 {
            lock_two(ti, b1, b2);
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
}

/// unlock_three unlocks the three given buckets
///
/// Unsafe because it assumes that all the locks are taken.
pub unsafe fn unlock_three<'a, K, V>(ti: &Snapshot<'a, K, V>,
                                     i1: BucketIndex<'a>, i2: BucketIndex<'a>, i3: BucketIndex<'a>)
{
    let i1 = lock_ind(*i1);
    let i2 = lock_ind(*i2);
    let i3 = lock_ind(*i3);
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

/// hashmask returns the bitmask for the buckets array corresponding to a
/// given hashpower.
#[inline(always)]
fn hashmask(hashpower: usize) -> usize {
    // Subtract is always safe because hashsize() always returns a positive value.
    hashsize(hashpower).wrapping_sub(1)
}


/// hashsize returns the number of buckets corresponding to a given
/// hashpower.
/// Invariant: always returns a nonzero value.
#[inline(always)]
pub fn hashsize(hashpower: usize) -> usize {
    // TODO: make sure Rust can't UB on too large inputs or we'll make this unsafe.
    1 << hashpower
}

/// index_hash returns the first possible bucket that the given hashed key
/// could be.
#[inline(always)]
fn index_hash<'a, K, V>(ti: &TableInfo<K, V>, hv: usize) -> usize {
    hv & hashmask(ti.hashpower)
}
