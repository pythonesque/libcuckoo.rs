use core::cell::UnsafeCell;
use core::intrinsics;
use libc::{self, c_int};
use super::Duration;

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

    pub use self::os::{PTHREAD_MUTEX_INITIALIZER, pthread_mutex_t};

    extern {
        // mutexes
        pub fn pthread_mutex_lock(lock: *mut pthread_mutex_t) -> libc::c_int;
        pub fn pthread_mutex_unlock(lock: *mut pthread_mutex_t) -> libc::c_int;

    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "openbsd"))]
    mod os {
        use libc;

        pub type pthread_mutex_t = *mut libc::c_void;

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = 0 as *mut _;
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

        const _PTHREAD_MUTEX_SIG_INIT: libc::c_long = 0x32AAABA7;

        #[repr(C)]
        pub struct pthread_mutex_t {
            __sig: libc::c_long,
            __opaque: [u8; __PTHREAD_MUTEX_SIZE__],
        }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __sig: _PTHREAD_MUTEX_SIG_INIT,
            __opaque: [0; __PTHREAD_MUTEX_SIZE__],
        };
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

        #[repr(C)]
        pub struct pthread_mutex_t {
            __align: libc::c_longlong,
            size: [u8; __SIZEOF_PTHREAD_MUTEX_T],
        }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __align: 0,
            size: [0; __SIZEOF_PTHREAD_MUTEX_T],
        };
    }
    #[cfg(target_os = "android")]
    mod os {
        use libc;

        #[repr(C)]
        pub struct pthread_mutex_t { value: libc::c_int }

        pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            value: 0,
        };
    }
}

pub struct Mutex { inner: UnsafeCell<ffi::pthread_mutex_t> }

pub const MUTEX_INIT: Mutex = Mutex {
    inner: UnsafeCell { value: ffi::PTHREAD_MUTEX_INITIALIZER },
};

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
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
}
