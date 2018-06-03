use num_cpus;
use std::cell::UnsafeCell;
#[cfg(feature = "counter")]
use std::cell::Cell;
use std::intrinsics;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
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
        let n = self.i;
        self.i = n.wrapping_add(1);
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

    pub fn val(&self) -> &'a V {
        unsafe {
            self.slot.val(self.index)
        }
    }
}

pub struct SlotIter<'a, K: 'a, V: 'a> {
    slot: &'a Bucket<K, V>,
    i: usize,
}

impl<'a, K, V> Iterator for SlotIter<'a, K, V> {
    type Item = Slot<'a, K, V>;

    #[inline]
    fn next(&mut self) -> Option<Slot<'a, K, V>> {
        loop {
            let i = self.i;
            if i == SLOT_PER_BUCKET {
                return None;
            }
            self.i = i.wrapping_add(1);
            if self.slot.occupied(SlotIndex(i)) {
                return Some(Slot { slot: self.slot, index: SlotIndex(i) });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (SLOT_PER_BUCKET - self.i, Some(SLOT_PER_BUCKET - self.i))
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
pub struct Bucket<K, V> {
    pub kv: ManuallyDrop<BucketInner<K, V>>,

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
        self.kv.keys.get_unchecked(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    unsafe fn key_mut(&mut self, SlotIndex(ind): SlotIndex) -> &mut K {
        self.kv.keys.get_unchecked_mut(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    pub unsafe fn val(&self, SlotIndex(ind): SlotIndex) -> &V {
        self.kv.vals.get_unchecked(ind)
    }

    /// Unsafe because it does not ensure the location was already occupied.
    #[inline(always)]
    pub unsafe fn val_mut(&mut self, SlotIndex(ind): SlotIndex) -> &mut V {
        self.kv.vals.get_unchecked_mut(ind)
    }

    /// Does not check to make sure location was not already occupied; can leak.
    pub fn set_kv(&mut self, SlotIndex(ind): SlotIndex, k: K, v: V) {
        unsafe {
            *self.occupied.get_unchecked_mut(ind) = true;
            let kv = &mut self.kv;
            ptr::write(kv.keys.as_mut_ptr().offset(ind as isize), k);
            ptr::write(kv.vals.as_mut_ptr().offset(ind as isize), v);
        }
    }

    /// Unsafe because it does not ensure the location was already occupied.
    pub unsafe fn erase_kv(&mut self, SlotIndex(ind): SlotIndex) -> (K, V) {
        *self.occupied.get_unchecked_mut(ind) = false;
        let kv = &mut self.kv;
        (ptr::read(kv.keys.as_mut_ptr().offset(ind as isize)),
         ptr::read(kv.vals.as_mut_ptr().offset(ind as isize)))
    }

    fn new() -> Self {
        unsafe {
            Bucket {
                occupied: [false; SLOT_PER_BUCKET],
                kv: ManuallyDrop::new(mem::uninitialized()),
            }
        }
    }

    pub fn clear(&mut self) {
        let kv = &mut self.kv;
        let mut keys = kv.keys.as_mut_ptr();
        let mut vals = kv.vals.as_mut_ptr();
        // We explicitly free occupied elements, instead of calling drop on the ManuallyDrop
        // Bucket.
        unsafe {
            for occupied in &mut self.occupied {
                if *occupied {
                    // Set *occupied to false first, so even if the drop panics nothing untoward
                    // happens.
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
        SlotIter { slot: self, i: 0 }
    }

    pub fn find<'a>(&'a self, key: &K) -> Option<&'a V> where
            K: Eq,
    {
        for i in SlotIndexIter::new() {
            if self.occupied(i) {
                // For now, we know we are "simple" so we skip this part.
                // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
                //     continue;
                // }
                unsafe {
                    if key == self.key(i) {
                        return Some(self.val(i))
                    }
                }
            }
        }
        None
    }

    pub fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> where
            K: Eq,
    {
        for i in SlotIndexIter::new() {
            if self.occupied(i) {
                // For now, we know we are "simple" so we skip this part.
                // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
                //     continue;
                // }
                unsafe {
                    if key == self.key(i) {
                        return Some(self.val_mut(i))
                    }
                }
            }
        }
        None
    }

    pub fn slots_mut<'a>(&'a mut self) -> SlotIterMut<'a, K, V> {
        SlotIterMut { slot: self, iter: SlotIndexIter::new(), marker: PhantomData }
    }
}

impl<K, V> Drop for Bucket<K, V> {
    fn drop(&mut self) {
        // We explicitly free occupied elements, and never drop kv explicitly.
        self.clear();
    }
}

type CacheAlign = ::std::simd::u64x8;
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
    index: usize,
    marker: InvariantLifetime<'a>,
}

#[cfg(feature = "counter")]
impl<'a> CounterIndex<'a> {
    pub fn num_inserts<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &CacheInt {
        unsafe {
            ti.num_inserts.get_unchecked(self.index)
        }
    }

    pub fn num_deletes<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &CacheInt {
        unsafe {
            ti.num_deletes.get_unchecked(self.index)
        }
    }
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
            let mut vec = Vec::<T>::with_capacity(n);
            unsafe {
                let mut ptr = vec.as_mut_ptr();
                let end = ptr.offset(n as isize);
                while ptr != end {
                    ptr::write(ptr, f());
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
            let ti = box UnsafeCell::new(
                TableInfo {
                    hashpower: hashpower,
                    buckets: from_fn(hashsize(hashpower), || UnsafeCell::new(Bucket::new())),
                    locks: mem::uninitialized(),
                    num_inserts: from_fn(num_cpus, || CacheInt::new()),
                    num_deletes: from_fn(num_cpus, || CacheInt::new()),
                }
            );
            for lock in &mut (&mut *ti.get()).locks[..] {
                ptr::write(lock, SpinLock::new());
            }
            ti
        }
    }
}

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

pub struct Lock<'a> {
    i: BucketIndex<'a>,
}

impl<'a> Lock<'a> {
    /// locks the given bucket index.
    #[inline(always)]
    pub fn new<K, V>(ti: &Snapshot<'a, K, V>, i: BucketIndex<'a>) -> Self {
        unsafe {
            ti.locks.get_unchecked(lock_ind(*i)).lock();
            Lock { i: i }
        }
    }

    /// unlocks the given bucket index.
    #[inline(always)]
    pub fn release<K, V>(self, ti: &Snapshot<'a, K, V>) {
        unsafe {
            ti.locks.get_unchecked(lock_ind(*self.i)).unlock();
        }
    }

    pub fn bucket<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &Bucket<K, V> {
        unsafe {
            &*ti.buckets.get_unchecked(*self.i).get()
        }
    }

    pub fn bucket_mut<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i).get()
        }
    }
}

pub struct LockTwo<'a> {
    i1: BucketIndex<'a>,
    i2: BucketIndex<'a>,
}

impl<'a> LockTwo<'a> {
    #[inline]
    /// locks the two bucket indexes, always locking the earlier index
    /// first to avoid deadlock. If the two indexes are the same, it just locks
    /// one.
    pub fn new<K, V>(ti: &Snapshot<'a, K, V>, b1: BucketIndex<'a>, b2: BucketIndex<'a>) -> Self {
        unsafe {
            let mut i1 = lock_ind(*b1);
            let mut i2 = lock_ind(*b2);
            let locks = &ti.locks;
            if i1 > i2 {
                mem::swap(&mut i1, &mut i2);
            }
            locks.get_unchecked(i1).lock();
            if i1 != i2 {
                locks.get_unchecked(i2).lock();
            }
            LockTwo { i1: b1, i2: b2, }
        }
    }

    /// unlocks both of the given bucket indexes, or only one if they
    /// are equal. Order doesn't matter here.
    pub fn release<K, V>(self, ti: &Snapshot<'a, K, V>) -> (BucketIndex<'a>, BucketIndex<'a>) {
        unsafe {
            let i1 = lock_ind(*self.i1);
            let i2 = lock_ind(*self.i2);
            let locks = &ti.locks;
            locks.get_unchecked(i1).unlock();
            if i1 != i2 {
                locks.get_unchecked(i2).unlock();
            }
            (self.i1, self.i2)
        }
    }

    pub fn bucket1<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &Bucket<K, V> {
        unsafe {
            &*ti.buckets.get_unchecked(*self.i1).get()
        }
    }

    pub fn bucket2<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &Bucket<K, V> {
        unsafe {
            &*ti.buckets.get_unchecked(*self.i2).get()
        }
    }

    pub fn bucket1_mut<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i1).get()
        }
    }

    pub fn bucket2_mut<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i2).get()
        }
    }
}

pub struct LockThree<'a> {
    i1: BucketIndex<'a>,
    i2: BucketIndex<'a>,
    i3: BucketIndex<'a>,
}

impl<'a> LockThree<'a> {
    #[inline]
    /// locks the three bucket indexes in numerical order.
    pub fn new<K, V>(ti: &Snapshot<'a, K, V>,
                     b1: BucketIndex<'a>, b2: BucketIndex<'a>, b3: BucketIndex<'a>) -> Self {
        unsafe {
            let mut i1 = lock_ind(*b1);
            let mut i2 = lock_ind(*b2);
            let mut i3 = lock_ind(*b3);
            if i1 > i2 {
                mem::swap(&mut i1, &mut i2);
            }
            if i2 > i3 {
                mem::swap(&mut i2, &mut i3);
            }
            if i1 > i2 {
                mem::swap(&mut i1, &mut i2);
            }
            let locks = &ti.locks;
            locks.get_unchecked(i1).lock();
            if i2 != i1 {
                locks.get_unchecked(i2).lock();
            }
            if i3 != i2 {
                locks.get_unchecked(i3).lock();
            }
            LockThree { i1: b1, i2: b2, i3: b3 }
        }
    }

    /// unlocks the three locked buckets
    pub fn release<K, V>(self, ti: &Snapshot<'a, K, V>) {
        unsafe {
            let i1 = lock_ind(*self.i1);
            let i2 = lock_ind(*self.i2);
            let i3 = lock_ind(*self.i3);
            let locks = &ti.locks;
            locks.get_unchecked(i1).unlock();
            if i2 != i1 {
                locks.get_unchecked(i2).unlock();
            }
            if i3 != i1 && i3 != i2 {
                locks.get_unchecked(i3).unlock();
            }
        }
    }

    pub fn release3<K, V>(self, ti: &Snapshot<'a, K, V>) -> LockTwo<'a> {
        unsafe {
            let i1 = lock_ind(*self.i1);
            let i2 = lock_ind(*self.i2);
            let i3 = lock_ind(*self.i3);
            if i3 != i1 && i3 != i2 {
                ti.locks.get_unchecked(i3).unlock();
            }
            LockTwo { i1: self.i1, i2: self.i2 }
        }
    }

    pub fn bucket1<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &Bucket<K, V> {
        unsafe {
            &*ti.buckets.get_unchecked(*self.i1).get()
        }
    }

    pub fn bucket1_mut<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i1).get()
        }
    }

    pub fn bucket3<'b, K, V>(&'b self, ti: &'b Snapshot<'a, K, V>) -> &Bucket<K, V> {
        unsafe {
            &*ti.buckets.get_unchecked(*self.i3).get()
        }
    }

    pub fn bucket3_mut<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> &mut Bucket<K, V> {
        unsafe {
            &mut *ti.buckets.get_unchecked(*self.i3).get()
        }
    }
}

pub struct LockAll<'a> {
    marker: InvariantLifetime<'a>,
}

impl<'a> LockAll<'a> {
    #[inline]
    pub fn new<K, V>(ti: &Snapshot<'a, K, V>) -> Self {
        for lock in &ti.locks[..] {
            lock.lock();
        }
        LockAll { marker: InvariantLifetime::new() }
    }

    pub fn release<K, V>(self, ti: &Snapshot<'a, K, V>) {
        for lock in &ti.locks[..] {
            lock.unlock();
        }
    }

    #[inline]
    pub fn buckets<'b, K, V>(&'b mut self, ti: &'b Snapshot<'a, K, V>) -> BucketIter<'b, K, V> {
        unsafe {
            let buckets = &ti.buckets;
            let p = buckets.as_ptr();
            intrinsics::assume(!p.is_null());
            if mem::size_of::<Bucket<K, V>>() == 0 {
                BucketIter {ptr: p,
                         end: (p as usize + buckets.len()) as *const UnsafeCell<Bucket<K, V>>,
                         _marker: PhantomData}
            } else {
                BucketIter {ptr: p,
                         end: p.offset(buckets.len() as isize),
                         _marker: PhantomData}
            }
        }
    }
}

pub struct BucketIter<'a, K: 'a, V: 'a> {
    ptr: *const UnsafeCell<Bucket<K, V>>,
    end: *const UnsafeCell<Bucket<K, V>>,
    _marker: PhantomData<&'a mut Bucket<K, V>>
}

impl<'a, K, V> Iterator for BucketIter<'a, K, V> {
    type Item = &'a mut Bucket<K, V>;

    #[inline]
    fn next(&mut self) -> Option<&'a mut Bucket<K, V>> {
        // could be implemented with slices, but this avoids bounds checks
        unsafe {
            intrinsics::assume(!self.ptr.is_null());
            intrinsics::assume(!self.end.is_null());
            if self.ptr == self.end {
                None
            } else {
                if mem::size_of::<Bucket<K, V>>() == 0 {
                    // purposefully don't use 'ptr.offset' because for
                    // vectors with 0-size elements this would return the
                    // same pointer.
                    self.ptr = mem::transmute(self.ptr as usize + 1);

                    // Use a non-null pointer value
                    Some(&mut *(1 as *mut _))
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(mem::transmute(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = (self.end as usize) - (self.ptr as usize);
        let size = mem::size_of::<Bucket<K, V>>();
        let exact = diff / (if size == 0 {1} else {size});
        (exact, Some(exact))
    }
}

/// lock_ind converts an index into buckets_ to an index into locks_.
#[inline(always)]
fn lock_ind(bucket_ind: usize) -> usize {
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
