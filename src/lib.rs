//#![no_start]
#![feature(asm,associated_consts,box_syntax,lang_items,libc,start,std_misc,thread_local,no_std,core,unsafe_no_drop_flag,static_assert,/*rustc_private,*/zero_one,step_trait,optin_builtin_traits,scoped,simd)]
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

    use self::test::Bencher;

    use super::CuckooHashMap;

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
    let ref map = CuckooHashMap::<_, _, DefaultState<FnvHasher>>::default();
    type Key = u64;
    //const MAX: Key = 0x20add; // no cuckoo (133853)
    //const MAX: Key = 0x3df36; // no resize (253750) 4 * 4 * 4 * 253750
    const MAX: Key = 10_000_000; // no resize
    //const MAX: Key = 20 * 3000;
    const CORES: Key = 4;
    const SLICE_THREADS: Key = 1; //CORES;
    const NUM_THREADS: Key = 4;
    const LOAD_FACTOR: Key = 6;
    const NUM_READS: Key = LOAD_FACTOR * ((CORES + NUM_THREADS - 1) / NUM_THREADS) * ((CORES + NUM_THREADS - 1) / NUM_THREADS)* SLICE_THREADS;// * 4;//60; // 48 * 4 * 4 * 20 * 3000
    println!("threads: {}, reads: {}, writes: {}", NUM_THREADS, NUM_READS * NUM_THREADS * NUM_THREADS / SLICE_THREADS * MAX, MAX);

    let insert = |j| {
        //let mut writes = 0;
        for i in Range::new(j * MAX, (j + 1) * MAX) {
            if let Err(_) = map.insert(i, /*[0u8; 1024]*/()) {
                //println!("insert error");
                unsafe { intrinsics::abort(); }
                //break;
            }
            //writes += 1;
        }
        println!("({}) writes", j);//, writes);
    };
    let find = |th| {
        let range = if SLICE_THREADS == CORES {
            Range::new(th, th + 1)
        } else {
            Range::new(0, NUM_THREADS)
        };
        //let mut reads = 0;
        /*'here: */for j in range {
            for _ in Range::new(0, NUM_READS) {
                for i in Range::new(j * MAX, (j + 1) * MAX) {
                    if let None = map.find(&i) {
                        //println!("find error");
                        unsafe { intrinsics::abort(); }
                        //break 'here;
                    };
                    //reads += 1;
                }
            }
        }
        println!("({}) reads", th,);// reads);
    };

    unsafe fn make_threads<'a, F, T>(f: &'a F) -> [thread::JoinGuard<'a, T>; NUM_THREADS as usize] where F: Fn(Key) -> T + Send + Sync + 'a, T: Send {
        let mut t: [thread::JoinGuard<T> ; NUM_THREADS as usize] = mem::uninitialized();
        for j in Range::new(0, NUM_THREADS) {
            ptr::write(t.as_mut_ptr().offset(j as isize), thread::scoped(move || f(j)));
        }
        t
    }
    unsafe {
        make_threads(&insert);
    }
    unsafe {
        make_threads(&find);
    }

    /*insert(0);
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
