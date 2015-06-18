use self::InsertError::*;
use std::cell::UnsafeCell;
//use std::collections::hash_map::RandomState;
use std::collections::hash_state::{DefaultState, HashState};
//use std::fmt;
use std::hash::{Hash, Hasher, SipHasher};
use std::intrinsics;
use std::mem;
use std::ptr;
#[cfg(not(feature="nothreads"))]
use std::thread;
use std::sync::atomic::{AtomicPtr, Ordering};
use super::iter::Range;
use super::hazard_pointer::{check_hazard_pointer, delete_unused, HazardPointer, HazardPointerSet};
use super::table_info::{self, AllUnlocker, Bucket, BucketIndex, CounterIndex, LockTwo, SlotIndex, SlotIndexIter, SLOT_PER_BUCKET, Snapshot, TableInfo};

/// DEFAULT_SIZE is the default number of elements in an empty hash
/// table
const DEFAULT_SIZE: usize = (1 << 16) * SLOT_PER_BUCKET;

/*
/// true if the key is small and simple, which means using partial keys would
/// probably slow us down
static const bool is_simple =
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

/// reserve_calc takes in a parameter specifying a certain number of slots
/// for a table and returns the smallest hashpower that will hold n elements.
fn reserve_calc(n: usize) -> usize {
    let nhd: f64 = (n as f64 / SLOT_PER_BUCKET as f64).log2().ceil();
    let new_hashpower = if nhd <= 0.0 { 1.0 } else { nhd } as usize;
    if !(n <= table_info::hashsize(new_hashpower).wrapping_mul(SLOT_PER_BUCKET)) {
        //println!("capacity error: reserve_calc()");
        unsafe { intrinsics::abort() }
    }
    new_hashpower
}

/// In my current benchmarks, inserts with (num_cpus) threads all working at once, even when most
/// of them already exist, are about 80% faster than the speed of normal HashMap inserts with
/// FnvHashMap with identical workloads, initial capacities, algorithm implementations, etc.).
pub struct CuckooHashMap<K, V, S = DefaultState<SipHasher>> {
    table_info: AtomicPtr<TableInfo<K, V>>,

    /// old_table_infos holds pointers to old TableInfos that were replaced
    /// during expansion. This keeps the memory alive for any leftover
    /// operations, until they are deleted by the global hazard pointer manager.
    old_table_infos: UnsafeCell<Vec<Box<UnsafeCell<TableInfo<K, V>>>>>,

    // All hashes are keyed on these values, to prevent hash collision attacks.
    hash_state: S,
}

impl<K, V, S> CuckooHashMap<K, V, S> where
        K: Eq + Hash,
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
            let _au = AllUnlocker::new(&ti);
            // cuckoo_clear empties the table, calling the destructors of all the
            // elements it removes from the table. It assumes the locks are taken as
            // necessary.
            for bucket in &mut (*ti.as_raw()).buckets {
                (*bucket.get()).clear();
            }
            if cfg!(feature = "counter") {
                let mut insert = ti.num_inserts.iter();
                let mut delete = ti.num_deletes.iter();
                while let Some(insert) = insert.next() {
                    let delete = delete.next().unwrap_or_else(|| intrinsics::unreachable());
                    insert.store_notatomic(0);
                    delete.store_notatomic(0);
                }
            }
        }
    }

    /// size returns the number of items currently in the hash table. Since it
    /// doesn't lock the table, elements can be inserted during the computation,
    /// so the result may not necessarily be exact (it may even be negative).
    #[cfg(feature = "counter")]
    pub fn size(&self) -> isize {
        let hazard_pointer = check_hazard_pointer();
        let ti = self.snapshot_table_nolock(&hazard_pointer);
        cuckoo_size(&ti)
    }

    /// empty returns true if the table is empty.
    #[cfg(feature = "counter")]
    pub fn empty(&self) -> bool {
        self.size() == 0
    }

    /// hashpower returns the hashpower of the table, which is
    /// log_2(the number of buckets).
    pub fn hashpower(&self) -> usize {
        let hazard_pointer = check_hazard_pointer();
        let ti = self.snapshot_table_nolock(&hazard_pointer);
        ti.hashpower
    }

    /// bucket_count returns the number of buckets in the table.
    pub fn bucket_count(&self) -> usize {
        table_info::hashsize(self.hashpower())
    }

    /// load_factor returns the ratio of the number of items in the table to the
    /// total number of available slots in the table.
    /// The result may not necessarily be exact (it may even be negative).
    #[cfg(feature = "counter")]
    pub fn load_factor(&self) -> f64 {
        let hazard_pointer = check_hazard_pointer();
        let ti = self.snapshot_table_nolock(&hazard_pointer);
        cuckoo_loadfactor(&ti)
    }

    /// find searches through the table for `key`, and returns `Some(value)` if
    /// it finds the value, `None` otherwise.
    pub fn find(&self, key: &K) -> Option<V> where
            K: Copy,
            V: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        self.snapshot_and_lock_two(&hazard_pointer, hv, move |snapshot, mut lock| {
            let st = cuckoo_find(key, hv, &snapshot, &mut lock);
            lock.release(&snapshot);
            st
        })
    }

    /// contains searches through the table for `key`, and returns true if it
    /// finds it in the table, and false otherwise.
    pub fn contains(&self, key: &K) -> bool where
            K: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        self.snapshot_and_lock_two(&hazard_pointer, hv, move |snapshot, mut lock| {
            let result = cuckoo_contains(key, hv, &snapshot, &mut lock);
            lock.release(&snapshot);
            result
        })
    }

    /// insert puts the given key-value pair into the table. It first checks
    /// that `key` isn't already in the table, since the table doesn't support
    /// duplicate keys. If the table is out of space, insert will automatically
    /// expand until it can succeed. Note that expansion can throw an exception,
    /// which insert will propagate. If `key` is already in the table, it
    /// returns false, otherwise it returns true.
    pub fn insert(&self, key: K, v: V) -> InsertResult<(), K, V> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Clone + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, &key);
        self.snapshot_and_lock_two(&hazard_pointer, hv, |snapshot, lock| {
            self.cuckoo_insert_loop(&hazard_pointer, key, v, hv, (snapshot, lock))
        })
    }

    /// erase removes `key` and it's associated value from the table, calling
    /// their destructors. If `key` is not there, it returns false, otherwise
    /// it returns true.
    pub fn erase(&self, key: &K) -> Option<V> {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        self.snapshot_and_lock_two(&hazard_pointer, hv, move |snapshot, mut lock| {
            let result = cuckoo_delete(key, hv, &snapshot, &mut lock);
            lock.release(&snapshot);
            result
        })
    }

    /// update changes the value associated with `key` to `val`. If `key` is
    /// not there, it returns false, otherwise it returns true.
    pub fn update(&self, key: &K, val: V) -> Result<V, V> where
            V: Copy,
    {
        let hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, key);
        self.snapshot_and_lock_two(&hazard_pointer, hv, move |snapshot, mut lock| {
            let result = cuckoo_update(key, val, hv, &snapshot, &mut lock);
            lock.release(&snapshot);
            result
        })
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
        self.snapshot_and_lock_two(&hazard_pointer, hv, move |snapshot, mut lock| {
            let result = cuckoo_update_fn(key, &mut updater, hv, &snapshot, &mut lock);
            lock.release(&snapshot);
            result
        })
    }

    /// upsert is a combination of update_fn and insert. It first tries updating
    /// the value associated with `key` using `updater`. If `key` is not in the
    /// table, then it runs an insert with `key` and `val`. It will always
    /// succeed, since if the update fails and the insert finds the key already
    /// inserted, it can retry the update.
    pub fn upsert<F, T>(&self, mut key: K, mut updater: F, mut val: V) -> Option<T> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Clone + Send + Sync,
            F: FnMut(&mut V) -> T,
    {
        let ref hazard_pointer = check_hazard_pointer();
        let hv = hashed_key(&self.hash_state, &key);
        loop {
            let updater = &mut updater;
            match self.snapshot_and_lock_two(hazard_pointer, hv, move |snapshot, mut lock| {
                match cuckoo_update_fn(&key, updater, hv, &snapshot, &mut lock) {
                    v @ Some(_) => {
                        lock.release(&snapshot);
                        return Ok(v);
                    },
                    // We run an insert, since the update failed
                    None => return match self.cuckoo_insert_loop(hazard_pointer, key, val, hv,
                                                                 (snapshot, lock)) {
                        Ok(()) => Ok(None),
                        // The only valid reason for res being false is if insert
                        // encountered a duplicate key after releasing the locks and
                        // performing cuckoo hashing. In this case, we retry the entire
                        // upsert operation.
                        Err(e) => Err(e),
                    }
                }
            }) {
                Ok(res) => return res,
                Err((_, k, v)) => {
                    key = k;
                    val = v;
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
            S: Clone + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        let ti = self.snapshot_table_nolock(&hazard_pointer);
        if n <= ti.hashpower {
            Err(())
        } else {
            self.cuckoo_expand_simple(&hazard_pointer, n)
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
            S: Clone + Send + Sync,
    {
        let hazard_pointer = check_hazard_pointer();
        let ti = self.snapshot_table_nolock(&hazard_pointer);
        if n <= table_info::hashsize(ti.hashpower).wrapping_mul(SLOT_PER_BUCKET) {
            Err(())
        } else {
            self.cuckoo_expand_simple(&hazard_pointer, reserve_calc(n))
        }
    }

    fn snapshot<'a>(&self, hazard_pointer: &'a HazardPointer)
                    -> HazardPointerSet<'a, TableInfo<K, V>> {
       loop {
            return unsafe {
                let ti = self.table_info.load(Ordering::SeqCst);
                let ti = hazard_pointer.set(ti);
                // If the table info has changed in the time we set the hazard
                // pointer, ti could have been deleted, so try again.
                // Note that this should provide an acquire fence for the previous operation.
                if ti.as_raw() != self.table_info.load(Ordering::SeqCst) {
                    continue;
                }
                ti
            };
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
    fn snapshot_table_nolock<'a>(&self, hazard_pointer: &'a HazardPointer)
                                 -> HazardPointerSet<'a, TableInfo<K, V>> {
        self.snapshot(hazard_pointer)
    }

    /// snapshot_and_lock_two loads the table_info pointer and locks the buckets
    /// associated with the given hash value. It returns the table_info and the
    /// two locked buckets as a tuple. Since the positions of the bucket locks
    /// depends on the number of buckets in the table, the table_info pointer
    /// needs to be grabbed first.
    fn snapshot_and_lock_two<F, T>(&self, hazard_pointer: &HazardPointer, hv: usize, closure: F)
                                   -> T
            where F: for <'a> FnOnce(Snapshot<'a, K, V>, LockTwo<'a>) -> T
    {
        loop {
            unsafe {
                let ti = self.snapshot(hazard_pointer);
                let ti = Snapshot::new(&ti);
                let i1 = BucketIndex::new(&ti, hv);
                let i2 = alt_index(&ti, hv, *i1);
                table_info::lock_two(&ti, i1, i2);

                // Check the table info again
                if &*ti as *const _ != self.table_info.load(Ordering::SeqCst) {
                    table_info::unlock_two(&ti, i1, i2);
                    continue;
                }

                return closure(ti, LockTwo::new(i1, i2));
            }
        }
    }

    /// snapshot_and_lock_all is similar to snapshot_and_lock_two, except that it
    /// takes all the locks in the table.
    fn snapshot_and_lock_all<'a>(&self, hazard_pointer: &'a HazardPointer)
                                 -> HazardPointerSet<'a, TableInfo<K, V>> {
        loop {
            let ti = self.snapshot_table_nolock(hazard_pointer);
            for lock in &ti.locks[..] {
                lock.lock();
            }
            // If the table info has changed, unlock the locks and try again.
            if ti.as_raw() != self.table_info.load(Ordering::SeqCst) {
                unsafe {
                    let _au = AllUnlocker::new(&ti);
                }
                continue;
            }
            return ti;
        }
    }

    /// slot_search searches for a cuckoo path using breadth-first search. It
    /// starts with the i1 and i2 buckets, and, until it finds a bucket with an
    /// empty slot, adds each slot of the bucket in the b_slot. If the queue runs
    /// out of space, it fails.
    fn slot_search<'a>(&self, ti: &Snapshot<'a, K, V>,
                       i1: BucketIndex<'a>, i2: BucketIndex<'a>) -> BSlot<'a> {
        let mut q = BQueue::new();
        // The initial pathcode informs cuckoopath_search which bucket the path
        // starts on
        unsafe {
            q.enqueue(BSlot::new(i1, 0, 0));
            q.enqueue(BSlot::new(i2, 1, 0));
        }
        while !q.full() && !q.empty() {
            let mut x = unsafe {
                q.dequeue()
            };
            // picks a (sort-of) random slot to start from
            let starting_slot = x.pathcode % SLOT_PER_BUCKET;
            for i in Range::new(0, SLOT_PER_BUCKET) {
                if q.full() { break; }
                let slot = SlotIndex::new(starting_slot.wrapping_add(i));
                unsafe {
                    table_info::lock(ti, x.bucket);
                    let bucket = &*ti.buckets.get_unchecked(*x.bucket).get();
                    if !bucket.occupied(slot) {
                        // We can terminate the search here
                        x.pathcode = x.pathcode.wrapping_mul(SLOT_PER_BUCKET).wrapping_add(*slot);
                        table_info::unlock(ti, x.bucket);
                        return x;
                    }

                    // If x has less than the maximum number of path components,
                    // create a new b_slot item, that represents the bucket we would
                    // have come from if we kicked out the item at this slot.
                    if x.depth < (MAX_BFS_PATH_LEN - 1) as Depth {
                        let hv = hashed_key(&self.hash_state, bucket.key(slot));
                        table_info::unlock(ti, x.bucket);
                        let y = BSlot::new(alt_index(ti, hv, *x.bucket),
                                           x.pathcode.wrapping_mul(SLOT_PER_BUCKET).wrapping_add(*slot),
                                           x.depth.wrapping_add(1));
                        q.enqueue(y);
                    }
                }
            }
        }
        // We didn't find a short-enough cuckoo path, so the queue ran out of
        // space. Return a failure value.
        BSlot::new(BucketIndex::new(ti, 0), 0, -1)
    }

    /// cuckoopath_search finds a cuckoo path from one of the starting buckets to
    /// an empty slot in another bucket. It returns the depth of the discovered
    /// cuckoo path on success, and -1 on failure. Since it doesn't take locks on
    /// the buckets it searches, the data can change between this function and
    /// cuckoopath_move. Thus cuckoopath_move checks that the data matches the
    /// cuckoo path before changing it.
    fn cuckoopath_search<'a>(&self, ti: &Snapshot<'a, K, V>,
                             cuckoo_path: &mut [CuckooRecord<'a, K>; MAX_BFS_PATH_LEN as usize],
                             i1: BucketIndex<'a>, i2: BucketIndex<'a>)
                             -> /*([CuckooRecord<K>; MAX_BFS_PATH_LEN], Depth)*/Depth where
            K: Copy,
    {
        unsafe {
            let mut x = self.slot_search(ti, i1, i2);
            if x.depth == -1 {
                return -1;
            }
            // Fill in the cuckoo path slots from the end to the beginning
            let start = cuckoo_path.as_mut_ptr();
            //let end = cuckoo_path.as_ptr().offset(x.depth);
            //let mut curr = end.offset(1);
            let end = start.offset(x.depth as isize).offset(1);
            let mut curr = end;
            while curr != start {
                curr = curr.offset(-1);
                (*curr).slot = SlotIndex::new(x.pathcode);
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
                table_info::lock(ti, i);
                let bucket = &*ti.buckets.get_unchecked(*i).get();
                if !bucket.occupied((*curr).slot) {
                    // We can terminate here
                    table_info::unlock(ti, i);
                    return ((curr as usize).wrapping_sub(start as usize)
                            / mem::size_of::<CuckooRecord<K>>()) as Depth;
                }
                (*curr).key = *bucket.key((*curr).slot);
                table_info::unlock(ti, i);
                let hv = hashed_key(&self.hash_state, &(*curr).key);
                /*debug_assert!((*curr).bucket == index_hash(ti, hv) ||
                              (*curr).bucket == alt_index(ti, hv, index_hash(ti, hv)));*/
                // We get the bucket that this slot is on by computing the alternate
                // index of the previous bucket
                i = alt_index(ti, hv, *i);
                curr = curr.offset(1);
            }
            /*(*curr).bucket = i;
            lock(ti, i);
            let bucket = ti.buckets.get_unchecked(i);
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
                if !ti.buckets.get_unchecked(i1).occupied((*curr).slot) {
                    // We can terminate here
                    unlock(ti, i1);
                    return 0;
                }
                (*curr).key = ti.buckets.get_unchecked(i1).key((*curr).slot);
                unlock(ti, i1);
            } else {
                //debug_assert!(x.pathcode == 1);
                (*curr).bucket = i2;
                lock(ti, i2);
                if !ti.buckets.get_unchecked(i2).occupied((*curr).slot) {
                    // We can terminate here
                    unlock(ti, i2);
                    return 0;
                }
                (*curr).key = ti.buckets.get_unchecked(i2).key((*curr).slot);
                unlock(ti, i2);
            }*/
            /*while curr != end {
                (*curr).bucket = i;
                lock(ti, i);
                let bucket = ti.buckets.get_unchecked(i);
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
                let bucket = ti.buckets.get_unchecked(i);
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
    }

    /// run_cuckoo performs cuckoo hashing on the table in an attempt to free up
    /// a slot on either i1 or i2. On success, the bucket and slot that was freed
    /// up is stored in insert_bucket and insert_slot. In order to perform the
    /// search and the swaps, it has to unlock both i1 and i2, which can lead to
    /// certain concurrency issues, the details of which are explained in the
    /// function. If run_cuckoo returns ok (success), then the slot it freed up
    /// is still locked. Otherwise it is unlocked.
    ///
    /// Unsafe because it assumes that the locks are taken.
    unsafe fn run_cuckoo<'a>(&self,
                             ti: &Snapshot<'a, K, V>,
                             i1: BucketIndex<'a>, i2: BucketIndex<'a>)
                             -> Result<(&mut Bucket<K, V>, SlotIndex), CuckooError> where
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
        table_info::unlock_two(ti, i1, i2);

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
                //debug_assert!(!ti.locks.get_unchecked(lock_ind(i1)).try_lock());
                //debug_assert!(!ti.locks.get_unchecked(lock_ind(i2)).try_lock());
                //debug_assert!(!ti.buckets.get_unchecked(insert_bucket).occupied(insert_slot));
                return if ti as *const _ as *mut _ == self.table_info.load(Ordering::SeqCst) {
                    let bucket = &mut *ti.buckets.get_unchecked(*insert_bucket).get();
                    Ok((bucket, insert_slot))
                } else {
                    // Unlock i1 and i2 and signal to cuckoo_insert to try again. Since
                    // we set the hazard pointer to be ti, this check isn't susceptible
                    // to an ABA issue, since a new pointer can't have the same address
                    // as ti.
                    table_info::unlock_two(ti, i1, i2);
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
    fn cuckoo_insert<'a>(&self,
                         key: K, val: V,
                         hv: usize,
                         snapshot_lock: (Snapshot<'a, K, V>, LockTwo<'a>))
                         -> InsertResult<(), K, V> where
            K: Copy + Eq,
    {
        let snapshot = &snapshot_lock.0;
        let mut lock = snapshot_lock.1;
        //const partial_t partial = partial_key(hv);
        let partial = ();
        let counterid = table_info::check_counterid(&snapshot_lock.0);
        match try_find_insert_bucket(partial, &key, lock.bucket1(snapshot)) {
            Err(KeyDuplicated) => {
                lock.release(snapshot);
                return Err((KeyDuplicated, key, val));
            },
            res1 => match try_find_insert_bucket(partial, &key, lock.bucket2(snapshot)) {
                Err(KeyDuplicated) => {
                    lock.release(snapshot);
                    return Err((KeyDuplicated, key, val));
                },
                res2 => {
                    if let Ok(res1) = res1 {
                        add_to_bucket(snapshot, &counterid, partial, key, val, lock.bucket1(snapshot), res1);
                        lock.release(snapshot);
                        return Ok(());
                    }
                    if let Ok(res2) = res2 {
                        add_to_bucket(snapshot, &counterid, partial, key, val, lock.bucket2(snapshot), res2);
                        lock.release(snapshot);
                        return Ok(());
                    }
                }
            }
        }
        // we are unlucky, so let's perform cuckoo hashing
        unsafe {
            match self.run_cuckoo(snapshot, lock.i1, lock.i2) {
                Err(CuckooError::UnderExpansion) => {
                    // The run_cuckoo operation operated on an old version of the table,
                    // so we have to try again. We signal to the calling insert method
                    // to try again by returning failure_under_expansion.
                    Err((UnderExpansion, key, val))
                },
                Ok((insert_bucket, insert_slot)) => {
                    /*debug_assert!(!ti.locks.get_unchecked(lock_ind(i1)).try_lock());
                    debug_assert!(!ti.locks.get_unchecked(lock_ind(i2)).try_lock());
                    debug_assert!(!ti.buckets.get_unchecked(insert_bucket).occupied(insert_slot));
                    debug_assert!(insert_bucket == index_hash(ti, hv) ||
                                  insert_bucket == alt_index(ti, hv, index_hash(ti, hv)));*/
                    // Since we unlocked the buckets during run_cuckoo, another insert
                    // could have inserted the same key into either i1 or i2, so we
                    // check for that before doing the insert.
                    if cuckoo_contains(&key, hv, snapshot, &mut lock) {
                        lock.release(snapshot);
                        return Err((KeyDuplicated, key, val));
                    }
                    add_to_bucket(snapshot, &counterid, partial, key, val,
                                  insert_bucket, insert_slot);
                    lock.release(snapshot);
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
    }

    /// We run cuckoo_insert in a loop until it succeeds in insert and upsert, so
    /// we pulled out the loop to avoid duplicating it. This should be called
    /// directly after snapshot_and_lock_two.
    fn cuckoo_insert_loop<'a>(&self, hazard_pointer: &HazardPointer,
                              mut key: K, mut val: V,
                              hv: usize,
                              mut snapshot_lock: (Snapshot<'a, K, V>, LockTwo<'a>))
                              -> InsertResult<(), K, V> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Clone + Send + Sync,
    {
        // TODO: investigate this claim:
        //   "by the end of the function, the hazard pointer will have been unset."
        loop {
            let hashpower = snapshot_lock.0.hashpower;
            let res = self.cuckoo_insert(key, val, hv, snapshot_lock);
            let (key_, val_) = match res {
                // If the insert failed with failure_key_duplicated, it returns here
                Err((KeyDuplicated, key, val)) => return Err((KeyDuplicated, key, val)),
                // If it failed with failure_under_expansion, the insert operated on
                // an old version of the table, so we just try again.
                Err((UnderExpansion, key, val)) => (key, val),
                // If it's failure_table_full, we have to expand the table before trying
                // again.
                Err((TableFull, key, val)) => {
                    // TODO: Investigate whether adding 1 is always safe here.
                    let hashpower = hashpower.checked_add(1)
                                             .unwrap_or_else( || {
                        //println!("capacity error: off by one in cuckoo_insert_loop");
                        unsafe { intrinsics::abort() }
                    });
                    if let Err(()) =
                        self.cuckoo_expand_simple(hazard_pointer, hashpower) {
                        // LIBCUCKOO_DBG("expansion is on-going\n");
                    }
                    (key, val)
                },
                Ok(()) => return Ok(())
            };
            key = key_;
            val = val_;
            snapshot_lock = self.snapshot_and_lock_two(hazard_pointer,
                                                       hv, |snapshot_, lock_| unsafe {
                // This is safe because ti was moved into cuckoo_insert_loop, so it can't be relied
                // on outside of this function anyway, and there are no locals within this function
                // that outlast the loop body and depend on the snapshot remaining the same.
                mem::transmute::<_, (Snapshot<'a, K, V>, LockTwo<'a>)>((snapshot_, lock_))
            });
        }
    }

    /// cuckoo_expand_simple is a simpler version of expansion than
    /// cuckoo_expand, which will double the size of the existing hash table. It
    /// needs to take all the bucket locks, since no other operations can change
    /// the table during expansion. If some other thread is holding the expansion
    /// thread at the time, then it will return failure_under_expansion.
    /// n should be a valid size (as per `reserve_calc`).
    fn cuckoo_expand_simple(&self, hazard_pointer: &HazardPointer,
                                   n: usize) -> Result<(), ()> where
            K: Copy + Send + Sync,
            V: Send + Sync,
            S: Clone + Send + Sync,
    {
        struct TI<'a, K, V>(&'a TableInfo<K, V>) where K: 'a, V: 'a;
        unsafe impl<'a, K, V> Send for TI<'a, K, V> where K: Send, V: Send {}
        unsafe impl<'a, K, V> Sync for TI<'a, K, V> where K: Send + Sync, V: Send + Sync {}
        #[cfg(feature="nothreads")]
        type Res<'a> = ();
        #[cfg(not(feature="nothreads"))]
        type Res<'a> = thread::JoinGuard<'a, ()>;
        /// insert_into_table is a helper function used by cuckoo_expand_simple to
        /// fill up the new table.
        ///
        /// Unsafe because it expects the old table to be locked and i and end to be in bounds.
        unsafe fn insert_into_table_<'a, K, V, S>(new_map: &CuckooHashMap<K, V, S>,
                                                  TI(old_ti): TI<K, V>,
                                                  //old_ti: HazardPointerSet<'a, TableInfo<K, V>>,
                                                  i: usize,
                                                  end: usize) where
                K: Copy + Eq + Hash + Send + Sync,
                V: Send + Sync,
                S: Clone + HashState + Send + Sync,
        {
        //let ref insert_into_table = |new_map: &CuckooHashMap<K, V, S>,
        //                             TI(old_ti): TI<K, V>,
        //                             //old_ti: HazardPointerSet<'a, TableInfo<K, V>>,
        //                             i: usize,
        //                             end: usize| unsafe {
            let mut bucket = (*old_ti).buckets.as_ptr().offset(i as isize);
            //let e = end;
            //println!("Inserting {}..{}", i, e);
            let end = (*old_ti).buckets.as_ptr().offset(end as isize);
            while bucket != end {
                let kv = (*(*bucket).get()).kv.as_mut()
                    .unwrap_or_else( || intrinsics::unreachable());
                let mut keys = kv.keys.as_mut_ptr();
                let mut vals = kv.vals.as_mut_ptr();
                for occupied in &mut (*(*bucket).get()).occupied {
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
        };
        /// insert_into_table is a helper function used by cuckoo_expand_simple to
        /// fill up the new table.
        ///
        /// Unsafe because it expects the old table to be locked and i and end to be in bounds.
        #[cfg(feature="nothreads")]
        unsafe fn insert_into_table<'a, K, V, S>(new_map: &CuckooHashMap<K, V, S>,
                                                 ti: TI<K, V>,
                                                 //old_ti: HazardPointerSet<'a, TableInfo<K, V>>,
                                                 i: usize,
                                                 end: usize) -> Res<'a> where
                K: Copy + Eq + Hash + Send + Sync,
                V: Send + Sync,
                S: Clone + HashState + Send + Sync,
        {
            // FIXME: Properly insert an extra hazard pointer to keep the current one from being
            // blown away by the insert.
            insert_into_table_(new_map, ti, i, end);
        }
        #[cfg(not(feature="nothreads"))]
        unsafe fn insert_into_table<'a, K, V, S>(new_map: &'a CuckooHashMap<K, V, S>,
                                                 ti: TI<'a, K, V>,
                                                 //old_ti: HazardPointerSet<'a, TableInfo<K, V>>,
                                                 i: usize,
                                                 end: usize) -> Res<'a> where
                K: Copy + Eq + Hash + Send + Sync,
                V: Send + Sync,
                S: Clone + HashState + Send + Sync,
        {
            thread::scoped(move || insert_into_table_(new_map, ti, i, end) )
        }

        unsafe {
            //println!("expand");
            let ti = self.snapshot_and_lock_all(hazard_pointer);
            //println!("locked");
            //debug_assert!(ti == self.table_info.load(Ordering::SeqCst));
            let _au = AllUnlocker::new(&ti);
            //TODO: Evaluate whether we actually do need this for some reason.  I don't really
            //understand why we would.
            //let _hpu = HazardPointerUnsetter::new(hazard_pointer);
            if n <= ti.hashpower {
                // Most likely another expansion ran before this one could grab the
                // locks
                return Err(());
            }

            // Creates a new hash table with hashpower n and adds all the
            // elements from the old buckets

            let ref new_map = Self::with_capacity_and_hash_state(
                table_info::hashsize(n).wrapping_mul(SLOT_PER_BUCKET),
                self.hash_state.clone());
            //let threadnum = num_cpus::get();
            let threadnum = ti.num_inserts.len();
            if threadnum == 0 {
                // Pretty sure this should actually be impossible.
                // TODO: refactor this after I make sure.
                //println!("Shouldn't be possible for thread num to equal zero.");
                intrinsics::abort();
            }
            let buckets_per_thread =
                intrinsics::unchecked_udiv(table_info::hashsize(ti.hashpower), threadnum);

            if mem::size_of::<Res>() != 0 &&
               threadnum.checked_mul(mem::size_of::<Res>()).is_none() {
                //println!("Overflow calculating join guard vector length");
                intrinsics::abort();
            }
            {
                //println!("computing");
                let hashsize = table_info::hashsize(ti.hashpower);
                let mut vec = Vec::with_capacity(threadnum);
                let mut ptr = vec.as_mut_ptr();
                let mut i = 0;
                while i < threadnum.wrapping_sub(1) {
                    let ti = TI(&ti);
                    let start = i.wrapping_mul(buckets_per_thread);
                    i = i.wrapping_add(1);
                    let end = i.wrapping_mul(buckets_per_thread);
                    let t = insert_into_table(new_map, ti, start, end);
                    ptr::write(ptr, t);
                    vec.set_len(i);
                    ptr = ptr.offset(1);
                }
                {
                    let ti = TI(&ti);
                    let start = i.wrapping_mul(buckets_per_thread);
                    let t = insert_into_table(new_map, ti, start, hashsize);
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
            // pointer with the same address as the old one at the same time).
            //
            // Not sure about the store... again, keeping it SeqCst for now.
            self.table_info.store(new_map.table_info.load(Ordering::SeqCst), Ordering::SeqCst);
            new_map.table_info.store(ptr::null_mut(), Ordering::SeqCst);

            // Rather than deleting ti now, we store it in old_table_infos. We then
            // run a delete_unused routine to delete all the old table pointers.
            let old_table_infos = &mut *self.old_table_infos.get();
            old_table_infos.push(mem::transmute(ti.as_raw()));
            delete_unused(old_table_infos);
            Ok(())
        }
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


impl<K, V, S> Default for CuckooHashMap<K, V, S> where
        K: Eq + Hash,
        /*K: fmt::Debug,*/
        /*V: fmt::Debug,*/
        S: HashState + Default
{
    fn default() -> Self {
        Self::with_capacity_and_hash_state(DEFAULT_SIZE, Default::default())
    }
}

impl<K, V> CuckooHashMap<K, V, DefaultState<SipHasher>> where
        K: Eq + Hash,
        /*K: fmt::Debug,*/
        /*V: fmt::Debug,*/
{
    fn new() -> Self {
        Default::default()
    }
}

/// hashed_key hashes the given key.
#[inline(always)]
fn hashed_key<K: ?Sized, S>(hash_state: &S, key: &K) -> usize where
        K: Hash,
        S: HashState,
{
    let mut state = hash_state.hasher();
    key.hash(&mut state);
    state.finish() as usize
}

/// alt_index returns the other possible bucket that the given hashed key
/// could be. It takes the first possible bucket as a parameter. Note that
/// this function will return the first possible bucket if index is the
/// second possible bucket, so alt_index(ti, hv, alt_index(ti, hv,
/// index_hash(ti, hv))) == index_hash(ti, hv).
#[inline(always)]
fn alt_index<'a, K, V>(ti: &Snapshot<'a, K, V>, hv: usize, index: usize)
                       -> BucketIndex<'a> {
    // ensure tag is nonzero for the multiply
    // TODO: figure out if this is UB and how to mitigate it if so.
    let tag = (hv >> ti.hashpower).wrapping_add(1);
    // 0x5bd1e995 is the hash constant from MurmurHash2
    BucketIndex::new(ti, index ^ (tag.wrapping_mul(0x5bd1e995)))
}

/*// A constexpr version of pow that we can use for static_asserts
static constexpr size_t const_pow(size_t a, size_t b) {
    return (b == 0) ? 1 : a * const_pow(a, b - 1);
}*/

/// The maximum number of items in a BFS path.
const MAX_BFS_PATH_LEN: u8 = 5;

/// CuckooRecord holds one position in a cuckoo path.
struct CuckooRecord<'a, K> {
    bucket: BucketIndex<'a>,
    slot: SlotIndex,
    key: K,
}

type Depth = i32;

/// b_slot holds the information for a BFS path through the table
#[derive(Clone, Copy)]
#[repr(packed)]
struct BSlot<'a> {
    /// The bucket of the last item in the path
    bucket: BucketIndex<'a>,
    /// a compressed representation of the slots for each of the buckets in
    /// the path. pathcode is sort of like a base-SLOT_PER_BUCKET number, and
    /// we need to hold at most MAX_BFS_PATH_LEN slots. Thus we need the
    /// maximum pathcode to be at least SLOT_PER_BUCKET^(MAX_BFS_PATH_LEN)
    pathcode: usize,
    /// The 0-indexed position in the cuckoo path this slot occupies. It must
    /// be less than MAX_BFS_PATH_LEN, and also able to hold negative values.
    depth: Depth,
}

impl<'a> BSlot<'a> {
    #[inline(always)]
    fn new(bucket: BucketIndex<'a>, pathcode: usize, depth: Depth) -> Self {
        //debug_assert!(depth < MAX_BFS_PATH_LEN as Depth);
        BSlot {
            bucket: bucket,
            pathcode: pathcode,
            depth: depth,
        }
    }
}

/// b_queue is the queue used to store b_slots for BFS cuckoo hashing.
#[repr(packed)]
struct BQueue<'a> {
    /// A circular array of b_slots
    slots: [BSlot<'a> ; MAX_CUCKOO_COUNT],
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

/// returns the index in the queue after ind, wrapping around if
/// necessary.
fn increment(ind: usize) -> usize {
    (ind.wrapping_add(1)) & (MAX_CUCKOO_COUNT - 1)
}

impl<'a> BQueue<'a> {
    #[inline(always)]
    fn new() -> Self where
            BSlot<'a>: Copy,
    {
        BQueue {
            // Perfectly safe because `BSlot` is `Copy`.
            slots: unsafe { mem::uninitialized() },
            first: 0,
            last: 0,
        }
    }

    /// Unsafe because it assumes that the queue is not full.
    unsafe fn enqueue(&mut self, x: BSlot<'a>) {
        // debug_assert!(!self.full());
        *self.slots.get_unchecked_mut(self.last) = x;
        self.last = increment(self.last);
    }

    /// Unsafe because it assumes the queue is nonempty.
    unsafe fn dequeue(&mut self, ) -> BSlot<'a> {
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
/// Unsafe because it assumes depth is in bounds.
unsafe fn cuckoopath_move<'a, K, V>(ti: &Snapshot<'a, K, V>,
                                    cuckoo_path: &[CuckooRecord<'a, K> ; MAX_BFS_PATH_LEN as usize],
                                    mut depth: usize, i1: BucketIndex<'a>, i2: BucketIndex<'a>)
                                   -> bool where
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
        table_info::lock_two(ti, i1, i2);
        return if !(*ti.buckets.get_unchecked(*bucket).get()).occupied(cuckoo_path.get_unchecked(0).slot) {
            true
        } else {
            table_info::unlock_two(ti, i1, i2);
            false
        }
    }

    while depth > 0 {
        let from = cuckoo_path.as_ptr().offset(depth.wrapping_sub(1) as isize);
        let to = cuckoo_path.as_ptr().offset(depth as isize);
        let CuckooRecord { bucket: fb, slot: fs, .. } = *from;
        let CuckooRecord { bucket: tb, slot: ts, .. } = *to;

        let mut ob = BucketIndex::new(ti, 0);
        if depth == 1 {
            // Even though we are only swapping out of i1 or i2, we have to
            // lock both of them along with the slot we are swapping to,
            // since at the end of this function, i1 and i2 must be locked.
            ob = if *fb == *i1 { i2 } else { i1 };
            table_info::lock_three(ti, fb, tb, ob);
        } else {
            table_info::lock_two(ti, fb, tb);
        }

        // We plan to kick out fs, but let's check if it is still there;
        // there's a small chance we've gotten scooped by a later cuckoo. If
        // that happened, just... try again. Also the slot we are filling in
        // may have already been filled in by another thread, or the slot we
        // are moving from may be empty, both of which invalidate the swap.
        // &mut is safe because we have tkaen the locks.
        let bucket_fb = &mut *ti.buckets.get_unchecked(*fb).get();
        let bucket_tb = &mut *ti.buckets.get_unchecked(*tb).get();
        if bucket_fb.key(fs) != &(*from).key ||
           bucket_tb.occupied(ts) ||
           !bucket_fb.occupied(fs) {
            if depth == 1 {
                table_info::unlock_three(ti, fb, tb, ob);
            } else {
                table_info::unlock_two(ti, fb, tb);
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
            if table_info::lock_ind(*tb) != table_info::lock_ind(*fb) &&
               table_info::lock_ind(*tb) != table_info::lock_ind(*ob) {
                table_info::unlock(ti, tb);
            }
        } else {
            table_info::unlock_two(ti, fb, tb);
        }
        // Always safe because depth > 0 was a loop precondition and wasn't modified at all in the
        // rest of the loop.
        depth = depth.wrapping_sub(1);
    }
    true
}

/// try_read_from_bucket will search the bucket for the given key and store
/// the associated value if it finds it.
fn try_read_from_bucket<K, V, P>(_partial: P, key: &K, bucket: &Bucket<K, V>) -> Option<V> where
        K: Copy + Eq,
        V: Copy,
{
    for slot in bucket.slots() {
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
        //     continue;
        // }
        if key == slot.key() {
            return Some(*slot.val());
        }
    }
    None
}

/// check_in_bucket will search the bucket for the given key and return true
/// if the key is in the bucket, and false if it isn't.
fn check_in_bucket<K, V, P>(_partial: P, key: &K, bucket: &Bucket<K, V>) -> bool where
        K: Copy + Eq,
{
    for slot in bucket.slots() {
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && partial != ti->buckets_[i].partial(j)) {
        //     continue;
        // }
        if key == slot.key() {
            return true;
        }
    }
    false
}

/// add_to_bucket will insert the given key-value pair into the slot.
fn add_to_bucket<'a, K, V, P>(ti: &Snapshot<'a, K, V>,
                              counterid: &CounterIndex<'a>,
                              _partial: P,
                              key: K, val: V,
                              bucket: &mut Bucket<K, V>, j: SlotIndex) where
        K: Copy + Eq,
        /*K: fmt::Debug,*/
        /*V: fmt::Debug,*/
{
    //debug_assert!(!bucket.occupied(j));
    // For now, we know we are "simple" so we skip this part.
    //if (!is_simple) {
    //    ti->buckets_[i].partial(j) = partial;
    //}
    // &mut should be safe; this function is protected by the lock.
    bucket.set_kv(j, key, val);
    #[inline(always)]
    #[cfg(feature = "counter")]
    fn insert_counter<'a, K, V>(ti: &Snapshot<'a, K, V>, counterid: &CounterIndex<'a>) {
        let num_inserts = unsafe { ti.num_inserts.get_unchecked(counterid.index) };
        num_inserts.fetch_add_relaxed(1);
    }
    #[cfg(not(feature = "counter"))]
    fn insert_counter<'a, K, V>(_: &Snapshot<'a, K, V>, _: &CounterIndex<'a>) {}
    insert_counter(ti, counterid);
}

/// try_find_insert_bucket will search the bucket and store the index of an
/// empty slot if it finds one, or -1 if it doesn't. Regardless, it will
/// search the entire bucket and return false if it finds the key already in
/// the table (duplicate key error) and true otherwise.
fn try_find_insert_bucket<K, V, P>(_partial: P,
                                   key: &K,
                                   bucket: &Bucket<K, V>)
                                   -> Result<SlotIndex, InsertError> where
        K: Copy + Eq,
{
    let mut found_empty = Err(TableFull);
    for k in SlotIndexIter::new() {
        unsafe {
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
    }
    found_empty
}
/// try_del_from_bucket will search the bucket for the given key, and set the
/// slot of the key to empty if it finds it.
fn try_del_from_bucket<'a, K, V, P>(ti: &Snapshot<'a, K, V>,
                                    counterid: &CounterIndex<'a>,
                                    _partial: P,
                                    key: &K,
                                    bucket: &mut Bucket<K, V>)
                                    -> Option<V> where
        K: Eq,
{
    for mut slot in bucket.slots_mut() {
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        #[inline(always)]
        #[cfg(feature = "counter")]
        fn delete_counter<'a, K, V>(ti: &Snapshot<'a, K, V>, counterid: &CounterIndex<'a>) {
            let num_deletes = unsafe { ti.num_deletes.get_unchecked(counterid.index) };
            num_deletes.fetch_add_relaxed(1);
        }
        #[cfg(not(feature = "counter"))]
        fn delete_counter<'a, K, V>(_: &Snapshot<'a, K, V>, _: &CounterIndex<'a>) {}
        if slot.key() == key {
            let (_, v) = slot.erase();
            delete_counter(ti, counterid);
            return Some(v);
        }
    }
    None
}

/// try_update_bucket will search the bucket for the given key and change its
/// associated value if it finds it.
fn try_update_bucket<K, V, P>(_partial: P, key: &K, value: V, bucket: &mut Bucket<K, V>)
                              -> Result<V, V> where
        K: Eq,
        V: Copy,
{
    for mut slot in bucket.slots_mut() {
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        if slot.key() == key {
            return Ok(mem::replace(slot.val(), value))
        }
    }
    Err(value)
}

/// try_update_bucket_fn will search the bucket for the given key and change
/// its associated value with the given function if it finds it.
fn try_update_bucket_fn<K, V, P, F, T>(_partial: P,
                                       key: &K, updater: &mut F, bucket: &mut Bucket<K, V>)
                                       -> Option<T> where
        K: Eq,
        F: FnMut(&mut V) -> T,
{
    for mut slot in bucket.slots_mut() {
        // For now, we know we are "simple" so we skip this part.
        // if (!is_simple && ti->buckets_[i].partial(j) != partial) {
        //     continue;
        // }
        if slot.key() == key {
            let res = updater(slot.val());
            return Some(res);
        }
    }
    None
}

/// cuckoo_find searches the table for the given key and value, storing the
/// value in the val if it finds the key. It expects the locks to be taken
/// and released outside the function.
fn cuckoo_find<'a, K, V>(key: &K, _hv: usize,
                         snapshot: &Snapshot<'a, K, V>,
                         lock: &mut LockTwo<'a>) -> Option<V> where
        K: Copy + Eq,
        V: Copy,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    try_read_from_bucket(partial, key, lock.bucket1(snapshot))
        .or_else( || try_read_from_bucket(partial, key, lock.bucket2(snapshot)))
}

/// cuckoo_contains searches the table for the given key, returning true if
/// it's in the table and false otherwise. It expects the locks to be taken
/// and released outside the function.
fn cuckoo_contains<'a, K, V>(key: &K, _hv: usize,
                             snapshot: &Snapshot<'a, K, V>,
                             lock: &mut LockTwo<'a>) -> bool where
        K: Copy + Eq,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    check_in_bucket(partial, key, lock.bucket1(snapshot))
        || check_in_bucket(partial, key, lock.bucket2(snapshot))
}

/// cuckoo_delete searches the table for the given key and sets the slot with
/// that key to empty if it finds it. It expects the locks to be taken and
/// released outside the function.
fn cuckoo_delete<'a, K, V>(key: &K,
                           _hv: usize,
                           snapshot: &Snapshot<'a, K, V>,
                           lock: &mut LockTwo<'a>) -> Option<V> where
        K: Eq,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    let counterid = table_info::check_counterid(&snapshot);
    let res = try_del_from_bucket(snapshot, &counterid, partial, key, lock.bucket1(snapshot));
    match res {
        v @ Some(_) => v,
        None => try_del_from_bucket(snapshot, &counterid, partial, key, lock.bucket2(snapshot))
    }
}

/// cuckoo_update searches the table for the given key and updates its value
/// if it finds it. It expects the locks to be taken and released outside the
/// function.
fn cuckoo_update<'a, K, V>(key: &K, val: V,
                           _hv: usize,
                           snapshot: &Snapshot<'a, K, V>,
                           lock: &mut LockTwo<'a>) -> Result<V, V> where
        K: Eq,
        V: Copy,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    match try_update_bucket(partial, key, val, lock.bucket1(snapshot)) {
        v @ Ok(_) => v,
        Err(val) => try_update_bucket(partial, key, val, lock.bucket2(snapshot))
    }
}

/// cuckoo_update_fn searches the table for the given key and runs the given
/// function on its value if it finds it, assigning the result of the
/// function to the value. It expects the locks to be taken and released
/// outside the function.
fn cuckoo_update_fn<'a, K, V, F, T>(key: &K, updater: &mut F,
                                    _hv: usize,
                                    snapshot: &Snapshot<'a, K, V>,
                                    lock: &mut LockTwo<'a>) -> Option<T> where
        K: Eq,
        F: FnMut(&mut V) -> T,
{
    //const partial_t partial = partial_key(hv);
    let partial = ();
    match try_update_bucket_fn(partial, key, updater, lock.bucket1(snapshot)) {
        v @ Some(_) => v,
        None => try_update_bucket_fn(partial, key, updater, lock.bucket2(snapshot))
    }
}

/// cuckoo_size returns the number of elements in the given table.
/// The number of elements is approximate and may be negative.
#[cfg(feature = "counter")]
fn cuckoo_size<K, V>(ti: &TableInfo<K, V>) -> isize {
    let mut inserts = 0usize;
    let mut deletes = 0usize;

    let mut insert = ti.num_inserts.iter();
    let mut delete = ti.num_deletes.iter();
    while let Some(insert) = insert.next() {
        let delete = unsafe { delete.next().unwrap_or_else(|| intrinsics::unreachable()) };
        // We use unordered loads here because we don't care about accuracy and grabbing ti should
        // have given us enough of a fence to ensure memory safety.
        inserts = inserts.wrapping_add(insert.load_unordered());
        deletes = deletes.wrapping_add(delete.load_unordered());
    }
    (inserts as isize).wrapping_sub(deletes as isize)
}

/// cuckoo_loadfactor returns the load factor of the given table.
/// The load factor is approximate and may be negative.
#[cfg(feature = "counter")]
fn cuckoo_loadfactor<K, V>(ti: &TableInfo<K, V>) -> f64 {
    unsafe {
        // The safety here relies on table_info::hashsize() always being nonzero.
        intrinsics::unchecked_sdiv(cuckoo_size(ti) as f64 / SLOT_PER_BUCKET as f64, table_info::hashsize(ti.hashpower) as f64)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use num_cpus;
    use std::collections::hash_state::DefaultState;
    use std::i32;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use super::{Depth, MAX_BFS_PATH_LEN, MAX_CUCKOO_COUNT};
    use super::CuckooHashMap;
    use super::super::iter::Range;
    use super::super::nodemap::FnvHasher;
    use super::super::table_info::{K_NUM_LOCKS, SLOT_PER_BUCKET};

    #[test]
    fn slot_per_bucket_nonzero() {
        assert!(SLOT_PER_BUCKET != 0);
    }

    #[test]
    fn k_num_locks_nonzero() {
        assert!(K_NUM_LOCKS != 0);
    }

    /*static_assert(const_pow(SLOT_PER_BUCKET, MAX_BFS_PATH_LEN) <
                  std::numeric_limits<decltype(pathcode)>::max(),
                  "pathcode may not be large enough to encode a cuckoo"
                  " path");*/

    #[test]
    /// The depth type must able to hold a value of
    /// MAX_BFS_PATH_LEN - 1");
    fn max_bfs_path_fits_in_depth() {
        assert!((MAX_BFS_PATH_LEN - 1) as Depth <= i32::MAX);
    }

    /// The depth type must be able to hold a value of -1
    #[test]
    fn negative_one_fits_in_depth() {
        assert!(-1 as Depth >= i32::MIN);
    }

    /*static_assert(const_pow(SLOT_PER_BUCKET, MAX_BFS_PATH_LEN) >=
                  MAX_CUCKOO_COUNT, "MAX_CUCKOO_COUNT value is too large"
                  " to be useful");*/

    /// MAX_CUCKOO_COUNT should be a power of 2
    #[test]
    fn max_cuckoo_count_power_of_two() {
        assert!((MAX_CUCKOO_COUNT & (MAX_CUCKOO_COUNT - 1)) == 0);
    }

    #[test]
    fn make_hashmap() {
        CuckooHashMap::<u64, u64>::default();
    }

    #[bench]
    fn bench_single_threaded_hashmap(b: &mut test::Bencher) {
        const MAX: u32 = 325_000;
        const SIZE: usize = 1 << 21;
        use std::collections::HashMap;
        // 34ns is the time to beat (i.e. the time the cuckoo hash map seems to take in a real
        // benchmark, on average)
        let state = DefaultState::<FnvHasher>::default();
        let mut map = HashMap::with_capacity_and_hash_state(SIZE, state);
        let num_cpus = num_cpus::get() as u32;
        b.iter( || {
            for i in Range::new(0, num_cpus * MAX) {
                map.insert(i, ());
            }
        });
    }

    #[bench]
    #[cfg(not(feature="nothreads"))]
    fn bench_cuckoo_hashmap(b: &mut test::Bencher) {
        const MAX: u32 = 325_000;
        const SIZE: usize = 1 << 21;
        let state = DefaultState::<FnvHasher>::default();
        let ref map = CuckooHashMap::with_capacity_and_hash_state(SIZE, state);
        let num_cpus = num_cpus::get() as u32;
        let ref done = AtomicBool::new(false);
        let ref barrier = Barrier::new(num_cpus as usize + 1);
        let mut threads = Vec::with_capacity(num_cpus as usize);
        for i in Range::new(0, num_cpus) {
            threads.push(thread::scoped( move || {
                loop {
                    barrier.wait();
                    if done.load(Ordering::SeqCst) { break }
                    for i in Range::new(i * MAX, i.checked_add(1).unwrap() * MAX) {
                        let _ = map.upsert(i, |_| (), ());
                    }
                    barrier.wait();
                }
            }));
        }
        b.iter( move || {
            // Test start
            barrier.wait();
            // Test end
            barrier.wait();
        });
        done.store(true, Ordering::SeqCst);
        barrier.wait();
    }
}
