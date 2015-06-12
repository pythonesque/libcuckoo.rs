//#![no_start]
#![feature(asm,associated_consts,box_syntax,lang_items,libc,start,std_misc,thread_local,no_std,core,unsafe_no_drop_flag,/*rustc_private,*/zero_one,step_trait,optin_builtin_traits,scoped,simd)]
#![feature(const_fn)]
#![cfg_attr(test, feature(test))]
#![allow(dead_code)]
//#![no_std]
//#![no_main]

extern crate core;
extern crate libc;
//extern crate rustc;
extern crate num_cpus;
extern crate rand;

/*extern crate core;


use core::prelude::*;
use core::intrinsics;

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { unsafe { intrinsics::abort(); } }

/*#[no_mangle]
extern "C" fn _rust_begin_unwind() -> ! {
    unsafe {
        intrinsics::abort();
    }
}*/

//extern crate libc;

*/

pub use self::map::CuckooHashMap;

mod iter;
mod hazard_pointer;
mod map;
mod mutex;
mod nodemap;
mod simd;
pub mod spinlock;
mod sys;

#[cfg(test)]
mod tests {
    extern crate test;

    #[test]
    fn test_map() {
        /*let map = CuckooHashMap::<(), ()>::default();
        let size = map.size();
        println!("{}", size);*/
    }

    /*use std::sync::{StaticMutex, MUTEX_INIT};

    use super::sys;

    #[test]
    fn test_cpuid() {
        /*println!("CPU ID: {}", sys::cpuid());*/
    }

    #[bench]
    fn bench_mutex(b: &mut Bencher) {
        static MUTEX: StaticMutex = MUTEX_INIT;
        b.iter( || MUTEX.try_lock() );
    }

    #[bench]
    fn bench_cpuid(b: &mut Bencher) {
        extern crate test;
        b.iter( || sys::cpuid() );
        //println!("{}", unsafe { sys::cpuid() });
    }*/
}

/*#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    main();
    0
}*/

#[cfg(not(test))]
fn main() {
    use core::intrinsics;
    use core::mem;
    use core::ops::Add;
    use core::ptr;
    //use rustc::util::nodemap::FnvHasher;
    use std::collections::hash_state::DefaultState;
    use self::nodemap::FnvHasher;
    use self::iter::Range;
    use std::thread;

//fn start(_argc: isize, _argv: *const *const u8) {
//#[no_mangle] // ensure that this symbol is called `main` in the output
//pub extern fn main(argc: i32, argv: *const *const u8) -> i32 {
    //let map = CuckooHashMap::<(), (), DefaultState<FnvHasher>>::default();
    //let ref map = CuckooHashMap::<_, _, DefaultState<FnvHasher>>::default();
    const CAPACITY: usize = 1 << 22;
    type Key = u64;//u32;//();
    type Val = bool;//[[u64; 8]; 16];//();
    const RED: Val = false;//[[0 ; 8]; 16];//();
    const BLACK: Val = true;//[[!0 ; 8]; 16];//();
    //const MAX: Key = 0x20add; // no cuckoo (133853)
    //const MAX: Key = 0x3df36; // no resize (253750) 4 * 4 * 4 * 253750
    const MAX: Key = 650_000;
    //const MAX: Key = 1_000_000;
    //const MAX: Key = 20 * 3000;
    const CORES: Key = 4;
    const SLICE_THREADS: Key = CORES; // 1
    const NUM_THREADS: Key = 4;
    const LOAD_FACTOR: Key = 1;//6;
    const NUM_READS: Key = LOAD_FACTOR * ((CORES + NUM_THREADS - 1) / NUM_THREADS) * ((CORES + NUM_THREADS - 1) / NUM_THREADS)* SLICE_THREADS;// * 4;//60; // 48 * 4 * 4 * 20 * 3000

    const NUM_ENTRIES: Key = NUM_THREADS * MAX;

    #[derive(Clone,Copy)] struct Stats<S> { upsert: S, delete: S, insert: S, update: S, read: S };
    impl<S> Stats<S> where S: Copy {
        fn fold<B, F>(self, init: B, f: F) -> B where F: Fn(B, S) -> B {
            let Stats { upsert, delete, insert, update, read } = self;
            f(f(f(f(f(init, upsert), delete), insert), update), read)
        }
        fn map<B, F>(self, f: F) -> Stats<B> where F: Fn(S) -> B {
            let Stats { upsert, delete, insert, update, read } = self;
            Stats { upsert: f(upsert), delete: f(delete), insert: f(insert), update: f(update), read: f(read) }
        }
        fn dot<B, F, T>(self, s: Stats<T>, f: F) -> Stats<B> where F: Fn(S, T) -> B {
            let Stats { upsert: u1, delete: d1, insert: i1, update: p1, read: r1 } = self;
            let Stats { upsert: u2, delete: d2, insert: i2, update: p2, read: r2 } = s;
            Stats { upsert: f(u1, u2), delete: f(d1, d2), insert: f(i1, i2), update: f(p1, p2), read: f(r1, r2) }
        }
        const fn new(init: S) -> Self {
            Stats { upsert: init, delete: init, insert: init, update: init, read: init }
        }
    }
    impl<S> Stats<Stats<S>> where S: Copy {
        fn transpose(self) -> Stats<Stats<S>> {
            Stats { upsert: self.map( |s| s.upsert), delete: self.map( |s| s.delete),
                    insert: self.map( |s| s.insert), update: self.map( |s| s.update),
                    read: self.map( |s| s.read) }
        }
    }
    const OPS_PER_ENTRY_PER_TASK: Stats<Stats<Key>> = Stats {
        upsert: Stats { upsert: NUM_THREADS, read: NUM_THREADS - 1, .. Stats::new(0) },
        delete: Stats { delete: 1, read: 1, .. Stats::new(0) },
        insert: Stats { insert: 1, .. Stats::new(0) },
        update: Stats { update: 1, .. Stats::new(0) },
        read: Stats { read: NUM_READS / SLICE_THREADS * NUM_THREADS, .. Stats::new(0) },
    };
    {
        let ops_per_entry = OPS_PER_ENTRY_PER_TASK.transpose().map( |s| s.fold(0, Add::add) );
        let Stats { upsert, delete, insert, update, read } = ops_per_entry.map( |s| s * NUM_ENTRIES );
        let total_per_entry = ops_per_entry.fold(0, Add::add);
        println!("threads: {}, capacity: {}, entries: {}, key size: {}, value size: {}\
                 , upserts: {}, deletes: {}, inserts: {}, updates: {}, reads: {}\
                 , total: {}, {}% writes",
                 NUM_THREADS, CAPACITY, NUM_ENTRIES, mem::size_of::<Key>(), mem::size_of::<Val>(),
                 upsert, delete, insert, update, read,
                 total_per_entry * NUM_ENTRIES,
                 ((total_per_entry - ops_per_entry.read) as f64 / (total_per_entry as f64) * 100.0).round());
    }
    let ref map = CuckooHashMap::<_, _, DefaultState<FnvHasher>>::with_capacity_and_hash_state(CAPACITY, Default::default());
    let upsert = |j: Key, stats: &mut Stats<Key>| {
        let range = /*if SLICE_THREADS == CORES {
            Range::new(th, th + 1)
        } else {
            */Range::new(0, NUM_THREADS)/*
        }*/;
        let mut upserts: Key = 0;
        let mut reads: Key = 0;
        /*'here: */for j in range {
            for i in Range::new(j * MAX, (j + 1) * MAX) {
                upserts = upserts.wrapping_add(1);
                if let None = map.upsert(i, |b| { *b = RED; }, BLACK) {
                    continue;
                }
                if map.find(&i) != Some(RED) {
                    //println!("upsert error");
                    unsafe { intrinsics::abort(); }
                    //break 'here;
                }
                reads = reads.wrapping_add(1);
            }
        }
        stats.upsert = stats.upsert.wrapping_add(upserts);
        stats.read = stats.read.wrapping_add(reads);
        println!("({}) upserts", j);//, upserts);
        //Stats { upsert: inserts.wrapping_add(updates), read: updates, .. Stats::new(0) }
        //println!("({}) {} upserts ({} inserts, {} updates, {} finds)", j, inserts.wrapping_add(updates), inserts, updates, updates);
    };
    let delete = |j: Key, stats: &mut Stats<Key>| {
        let mut deletes: Key = 0;
        let mut reads: Key = 0;
        for i in Range::new(j * MAX, (j + 1) * MAX) {
            if let None = map.erase(&i) {
                //println!("delete error");
                unsafe { intrinsics::abort(); }
                //break;
            }
            deletes = deletes.wrapping_add(1);
            if let Some(_) = map.find(&i) {
                //println!("find after delete error");
                unsafe { intrinsics::abort(); }
                //break;
            }
            reads = reads.wrapping_add(1);
        }
        stats.delete = stats.delete.wrapping_add(deletes);
        stats.read = stats.read.wrapping_add(reads);
        println!("({}) deletes", j);//, writes);
    };
    let insert = |j: Key, stats: &mut Stats<Key>| {
        let mut inserts: Key = 0;
        for i in Range::new(j * MAX, (j + 1) * MAX) {
            if let Err(_) = map.insert(i, BLACK) {
                //println!("insert error");
                unsafe { intrinsics::abort(); }
                //break;
            }
            inserts = inserts.wrapping_add(1);
        }
        stats.insert = stats.insert.wrapping_add(inserts);
        println!("({}) writes", j);//, writes);
    };
    let update = |j: Key, stats: &mut Stats<Key>| {
        let mut updates: Key = 0;
        for i in Range::new(j * MAX, (j + 1) * MAX) {
            if map.update(&i, RED) != Ok(BLACK) {
                //println!("update error");
                unsafe { intrinsics::abort(); }
                //break;
            }
            updates = updates.wrapping_add(1);
        }
        stats.update = stats.update.wrapping_add(updates);
        println!("({}) updates", j);//, updates);
    };
    let find = |th: Key, stats: &mut Stats<Key>| {
        let range = if SLICE_THREADS == CORES {
            Range::new(th, th + 1)
        } else {
            Range::new(0, NUM_THREADS)
        };
        let mut reads = 0;
        /*'here: */for j in range {
            for _ in Range::new(0, NUM_READS) {
                for i in Range::new(j * MAX, (j + 1) * MAX) {
                    if map.find(&i) != Some(RED) {
                        //println!("find error");
                        unsafe { intrinsics::abort(); }
                        //break 'here;
                    }
                    reads += 1;
                }
            }
        }
        stats.read = stats.read.wrapping_add(reads);
        println!("({}) reads", th,);// reads);
    };

    unsafe fn make_threads<'a, F, T>(f: &'a F, s: &'a mut [T ; NUM_THREADS as usize])
                                     -> [thread::JoinGuard<'a, ()>; NUM_THREADS as usize] where
            F: Fn(Key, &'a mut T) + Send + Sync + 'a,
            T: Send,
    {
        let mut t: [thread::JoinGuard<()> ; NUM_THREADS as usize] = mem::uninitialized();
        for j in Range::new(0, NUM_THREADS) {
            let stats = &mut *s.as_mut_ptr().offset(j as isize);
            ptr::write(t.as_mut_ptr().offset(j as isize), thread::scoped(move || {
                //let start = sys::precise_time_ns();
                f(j, stats);
                //let end = sys::precise_time_ns();
                //println!("({}) {} ns", j, end - start);
            }));
        }
        t
    }
    let mut stats = [Stats::new(0) ; NUM_THREADS as usize];
    unsafe {
        {
            make_threads(&upsert, &mut stats);
        }
        if SLICE_THREADS == CORES {
            let all = |th, stats: &mut Stats<Key>| {
                delete(th, stats);
                insert(th, stats);
                update(th, stats);
                find(th, stats);
            };
            make_threads(&all, &mut stats);
        } else {
            {
                make_threads(&delete, &mut stats);
            }
            {
                make_threads(&insert, &mut stats);
            }
            {
                make_threads(&update, &mut stats);
            }
            {
                make_threads(&find, &mut stats);
            }
        }
    }
    {
        let Stats { upsert, delete, insert, update, read } =
            stats.iter().fold(Stats::new(0), |res, &s| res.dot(s, |x: Key, y: Key| x.wrapping_add(y) ) );
        println!("upserts: {}, deletes: {}, inserts: {}, updates: {}, reads: {}",
                 upsert, delete, insert, update, read);
    }

    /*
    upsert(0);
    delete(0);
    insert(0);
    update(0);
    find(0);*/
    /*//extern crate libc;
    //extern crate time;
    extern crate clock_ticks;
    //use std::sync::{StaticMutex, MUTEX_INIT};
    use core::atomic::{AtomicBool, ATOMIC_BOOL_INIT, Ordering};

    fn bench<F>(n: u32, name: &str, mut f: F) where F: core::ops::FnMut() {
        let mut i = n;
        //let start = clock_ticks::precise_time_ns();
        //println!("{} ({} iterations)", n);
        while i > 0 {
            f();
            i -= 1;
        }
        //let end = clock_ticks::precise_time_ns();
        //let mut buffer: [u8; MAX_BUF];
        //println!("{} ({} iterations): {}ns", MAX, end - time);
    }

    const MAX: u32 = 10_000_000;
    let mut foo = [0; 4];
    bench(MAX, "cpuid", || { sys::cpuid(); } );
    //println!("{:?}", foo);
    //static MUTEX: StaticMutex = MUTEX_INIT;
    //bench(MAX, "mutex", || { let _ = MUTEX.try_lock(); } );
    //static ATOMIC: AtomicBool = ATOMIC_BOOL_INIT;
    //bench(MAX, "atomic", || { while (ATOMIC.compare_and_swap(false, true, Ordering::Acquire)) { ATOMIC.compare_and_swap(true, false, Ordering::Release);} } );
    //bench(MAX, "atomic", || { while (ATOMIC.compare_and_swap(false, true, Ordering::Acquire)) { ATOMIC.compare_and_swap(true, false, Ordering::Release);} } );*/
    //0
}

/*pub struct CukooHashMap<K, V, S = RandomState> {
}*/
