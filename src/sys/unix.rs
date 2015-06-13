use core::cell::UnsafeCell;
use core::intrinsics;
use libc::{self, c_int};
use super::Duration;

#[cfg(all(not(target_os = "macos"), not(target_os = "ios")))]
mod imp {
    use libc::{c_int, timespec};

    // Apparently android provides this in some other library?
    #[cfg(all(not(target_os = "android"),
              not(target_os = "nacl")))]
    #[link(name = "rt")]
    extern {}

    extern {
        pub fn clock_gettime(clk_id: c_int, tp: *mut timespec) -> c_int;
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    use core::atomic::{AtomicBool, Ordering};
    use core::intrinsics;
    use libc::{timeval, timezone, c_int, mach_timebase_info};
    use super::super::super::spinlock::SpinLock;

    extern {
        pub fn gettimeofday(tp: *mut timeval, tzp: *mut timezone) -> c_int;
        pub fn mach_absolute_time() -> u64;
        pub fn mach_timebase_info(info: *mut mach_timebase_info) -> c_int;
    }

    /// After this is called, the return mach_timebase_info is guaranteed to have a nonzero
    /// denominator.
    pub fn info() -> &'static mach_timebase_info {
        static mut INFO: mach_timebase_info = mach_timebase_info {
            numer: 0,
            denom: 0,
        };
        static ONCE: AtomicBool = AtomicBool::new(false);

        #[cold]
        /// After this is called, the return mach_timebase_info is guaranteed to have a nonzero
        /// denominator.
        fn init() {
            static LOCK: SpinLock = SpinLock::new();
            unsafe {
                LOCK.lock();
                if !ONCE.load(Ordering::Relaxed) {
                    mach_timebase_info(&mut INFO);
                    if INFO.denom == 0 { intrinsics::abort(); }
                    ONCE.store(true, Ordering::Relaxed);
                }
                LOCK.unlock();
            }
        }
        unsafe {
            if !ONCE.load(Ordering::Relaxed) {
                init();
            }
            &INFO
        }
    }
}

/**
 * Returns the current value of a high-resolution performance counter
 * in nanoseconds since an unspecified epoch.
 */
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn precise_time_ns() -> u64 {
    unsafe {
        let time = imp::mach_absolute_time();
        let info = imp::info();
        intrinsics::unchecked_udiv(time.wrapping_mul(info.numer as u64), info.denom as u64)
    }
}

/*struct Timer {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn new() -> Self {

        if info.denom == 0 { intrinsics::abort(); }
    }
}*/

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn precise_time_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        imp::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    return (ts.tv_sec as u64) * 1000000000 + (ts.tv_nsec as u64)
}

/// Returns the platform-specific value of errno
pub fn errno() -> i32 {
    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd"))]
    unsafe fn errno_location() -> *const c_int {
        extern { fn __error() -> *const c_int; }
        __error()
    }

    #[cfg(target_os = "bitrig")]
    fn errno_location() -> *const c_int {
        extern {
            fn __errno() -> *const c_int;
        }
        unsafe {
            __errno()
        }
    }

    #[cfg(target_os = "dragonfly")]
    unsafe fn errno_location() -> *const c_int {
        extern { fn __dfly_error() -> *const c_int; }
        __dfly_error()
    }

    #[cfg(target_os = "openbsd")]
    unsafe fn errno_location() -> *const c_int {
        extern { fn __errno() -> *const c_int; }
        __errno()
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe fn errno_location() -> *const c_int {
        extern { fn __errno_location() -> *const c_int; }
        __errno_location()
    }

    unsafe {
        (*errno_location()) as i32
    }
}

pub fn sleep(dur: Duration) {
    let mut ts = libc::timespec {
        tv_sec: dur.secs() as libc::time_t,
        tv_nsec: dur.extra_nanos() as libc::c_long,
    };

    // If we're awoken with a signal then the return value will be -1 and
    // nanosleep will fill in `ts` with the remaining time.
    unsafe {
        while libc::nanosleep(&ts, &mut ts) == -1 {
            if errno() == libc::EINTR { intrinsics::abort(); }
        }
    }
}

pub mod ffi {
    use libc;

    pub use self::os::{PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_RECURSIVE, pthread_mutex_t,
                       pthread_mutexattr_t};
    pub use self::os::{PTHREAD_COND_INITIALIZER, pthread_cond_t};
    pub use self::os::{PTHREAD_RWLOCK_INITIALIZER, pthread_rwlock_t};

    extern {
        // mutexes
        pub fn pthread_mutex_init(lock: *mut pthread_mutex_t, attr: *const pthread_mutexattr_t)
                                -> libc::c_int;
        pub fn pthread_mutex_destroy(lock: *mut pthread_mutex_t) -> libc::c_int;
        pub fn pthread_mutex_lock(lock: *mut pthread_mutex_t) -> libc::c_int;
        pub fn pthread_mutex_trylock(lock: *mut pthread_mutex_t) -> libc::c_int;
        pub fn pthread_mutex_unlock(lock: *mut pthread_mutex_t) -> libc::c_int;

        pub fn pthread_mutexattr_init(attr: *mut pthread_mutexattr_t) -> libc::c_int;
        pub fn pthread_mutexattr_destroy(attr: *mut pthread_mutexattr_t) -> libc::c_int;
        pub fn pthread_mutexattr_settype(attr: *mut pthread_mutexattr_t, _type: libc::c_int)
                                        -> libc::c_int;

        // cvars
        pub fn pthread_cond_wait(cond: *mut pthread_cond_t,
                                 lock: *mut pthread_mutex_t) -> libc::c_int;
        pub fn pthread_cond_timedwait(cond: *mut pthread_cond_t,
                                  lock: *mut pthread_mutex_t,
                                  abstime: *const libc::timespec) -> libc::c_int;
        pub fn pthread_cond_signal(cond: *mut pthread_cond_t) -> libc::c_int;
        pub fn pthread_cond_broadcast(cond: *mut pthread_cond_t) -> libc::c_int;
        pub fn pthread_cond_destroy(cond: *mut pthread_cond_t) -> libc::c_int;
        pub fn gettimeofday(tp: *mut libc::timeval,
                            tz: *mut libc::c_void) -> libc::c_int;

        // rwlocks
        pub fn pthread_rwlock_destroy(lock: *mut pthread_rwlock_t) -> libc::c_int;
        pub fn pthread_rwlock_rdlock(lock: *mut pthread_rwlock_t) -> libc::c_int;
        pub fn pthread_rwlock_tryrdlock(lock: *mut pthread_rwlock_t) -> libc::c_int;
        pub fn pthread_rwlock_wrlock(lock: *mut pthread_rwlock_t) -> libc::c_int;
        pub fn pthread_rwlock_trywrlock(lock: *mut pthread_rwlock_t) -> libc::c_int;
        pub fn pthread_rwlock_unlock(lock: *mut pthread_rwlock_t) -> libc::c_int;
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "openbsd"))]
    mod os {
        use libc;

        pub type pthread_mutex_t = *mut libc::c_void;
        pub type pthread_mutexattr_t = *mut libc::c_void;
        pub type pthread_cond_t = *mut libc::c_void;
        pub type pthread_rwlock_t = *mut libc::c_void;

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = 0 as *mut _;
        pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = 0 as *mut _;
        pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = 0 as *mut _;
        pub const PTHREAD_MUTEX_RECURSIVE: libc::c_int = 2;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    mod os {
        use libc;

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "aarch64"))]
        const __PTHREAD_MUTEX_SIZE__: usize = 56;
        #[cfg(any(target_arch = "x86",
                  target_arch = "arm"))]
        const __PTHREAD_MUTEX_SIZE__: usize = 40;

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "aarch64"))]
        const __PTHREAD_COND_SIZE__: usize = 40;
        #[cfg(any(target_arch = "x86",
                  target_arch = "arm"))]
        const __PTHREAD_COND_SIZE__: usize = 24;

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "aarch64"))]
        const __PTHREAD_RWLOCK_SIZE__: usize = 192;
        #[cfg(any(target_arch = "x86",
                  target_arch = "arm"))]
        const __PTHREAD_RWLOCK_SIZE__: usize = 124;

        const _PTHREAD_MUTEX_SIG_INIT: libc::c_long = 0x32AAABA7;
        const _PTHREAD_COND_SIG_INIT: libc::c_long = 0x3CB0B1BB;
        const _PTHREAD_RWLOCK_SIG_INIT: libc::c_long = 0x2DA8B3B4;

        #[repr(C)]
        pub struct pthread_mutex_t {
            __sig: libc::c_long,
            __opaque: [u8; __PTHREAD_MUTEX_SIZE__],
        }
        #[repr(C)]
        pub struct pthread_mutexattr_t {
            __sig: libc::c_long,
            // note, that this is 16 bytes just to be safe, the actual struct might be smaller.
            __opaque: [u8; 16],
        }
        #[repr(C)]
        pub struct pthread_cond_t {
            __sig: libc::c_long,
            __opaque: [u8; __PTHREAD_COND_SIZE__],
        }
        #[repr(C)]
        pub struct pthread_rwlock_t {
            __sig: libc::c_long,
            __opaque: [u8; __PTHREAD_RWLOCK_SIZE__],
        }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __sig: _PTHREAD_MUTEX_SIG_INIT,
            __opaque: [0; __PTHREAD_MUTEX_SIZE__],
        };
        pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            __sig: _PTHREAD_COND_SIG_INIT,
            __opaque: [0; __PTHREAD_COND_SIZE__],
        };
        pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
            __sig: _PTHREAD_RWLOCK_SIG_INIT,
            __opaque: [0; __PTHREAD_RWLOCK_SIZE__],
        };

        pub const PTHREAD_MUTEX_RECURSIVE: libc::c_int = 2;
    }

    #[cfg(target_os = "linux")]
    mod os {
        use core::cell::UnsafeCell;
        use core::mem;
        use libc;

        // minus 8 because we have an 'align' field
        #[cfg(target_arch = "x86_64")]
        const __SIZEOF_PTHREAD_MUTEX_T: usize = 40 - 8;
        #[cfg(any(target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "mips",
                  target_arch = "mipsel",
                  target_arch = "powerpc"))]
        const __SIZEOF_PTHREAD_MUTEX_T: usize = 24 - 8;
        #[cfg(target_arch = "aarch64")]
        const __SIZEOF_PTHREAD_MUTEX_T: usize = 48 - 8;

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "mips",
                  target_arch = "mipsel",
                  target_arch = "powerpc"))]
        const __SIZEOF_PTHREAD_COND_T: usize = 48 - 8;

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "aarch64"))]
        const __SIZEOF_PTHREAD_RWLOCK_T: usize = 56 - 8;

        #[cfg(any(target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "mips",
                  target_arch = "mipsel",
                  target_arch = "powerpc"))]
        const __SIZEOF_PTHREAD_RWLOCK_T: usize = 32 - 8;

        #[repr(C)]
        pub struct pthread_mutex_t {
            __align: libc::c_longlong,
            size: [u8; __SIZEOF_PTHREAD_MUTEX_T],
        }
        #[repr(C)]
        pub struct pthread_mutexattr_t {
            __align: libc::c_longlong,
            // note, that this is 16 bytes just to be safe, the actual struct might be smaller.
            size: [u8; 16],
        }
        #[repr(C)]
        pub struct pthread_cond_t {
            __align: libc::c_longlong,
            size: [u8; __SIZEOF_PTHREAD_COND_T],
        }
        #[repr(C)]
        pub struct pthread_rwlock_t {
            __align: libc::c_longlong,
            size: [u8; __SIZEOF_PTHREAD_RWLOCK_T],
        }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __align: 0,
            size: [0; __SIZEOF_PTHREAD_MUTEX_T],
        };
        pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            __align: 0,
            size: [0; __SIZEOF_PTHREAD_COND_T],
        };
        pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
            __align: 0,
            size: [0; __SIZEOF_PTHREAD_RWLOCK_T],
        };
        pub const PTHREAD_MUTEX_RECURSIVE: libc::c_int = 1;
    }
    #[cfg(target_os = "android")]
    mod os {
        use libc;

        #[repr(C)]
        pub struct pthread_mutex_t { value: libc::c_int }
        pub type pthread_mutexattr_t = libc::c_long;
        #[repr(C)]
        pub struct pthread_cond_t { value: libc::c_int }
        #[repr(C)]
        pub struct pthread_rwlock_t {
            lock: pthread_mutex_t,
            cond: pthread_cond_t,
            numLocks: libc::c_int,
            writerThreadId: libc::c_int,
            pendingReaders: libc::c_int,
            pendingWriters: libc::c_int,
            reserved: [*mut libc::c_void; 4],
        }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            value: 0,
        };
        pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            value: 0,
        };
        pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
            lock: PTHREAD_MUTEX_INITIALIZER,
            cond: PTHREAD_COND_INITIALIZER,
            numLocks: 0,
            writerThreadId: 0,
            pendingReaders: 0,
            pendingWriters: 0,
            reserved: [0 as *mut _; 4],
        };
        pub const PTHREAD_MUTEX_RECURSIVE: libc::c_int = 1;
    }
}

pub struct Mutex { inner: UnsafeCell<ffi::pthread_mutex_t> }

#[inline]
pub unsafe fn raw(m: &Mutex) -> *mut ffi::pthread_mutex_t {
    m.inner.get()
}

pub const MUTEX_INIT: Mutex = Mutex {
    inner: UnsafeCell::new(ffi::PTHREAD_MUTEX_INITIALIZER),
};

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[allow(dead_code)] // sys isn't exported yet
impl Mutex {
    #[inline]
    pub unsafe fn new() -> Mutex {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        MUTEX_INIT
    }
    #[inline]
    pub unsafe fn lock(&self) {
        let _r = ffi::pthread_mutex_lock(self.inner.get());
        //debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        let _r = ffi::pthread_mutex_unlock(self.inner.get());
        //debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::pthread_mutex_trylock(self.inner.get()) == 0
    }
    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let _r = ffi::pthread_mutex_destroy(self.inner.get());
        //debug_assert_eq!(r, 0);
    }
    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        use libc;
        let _r = ffi::pthread_mutex_destroy(self.inner.get());
        // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
        // mutex that was just initialized with ffi::PTHREAD_MUTEX_INITIALIZER.
        // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
        // this behaviour no longer occurs.
        //debug_assert!(r == 0 || r == libc::EINVAL);
    }
}
