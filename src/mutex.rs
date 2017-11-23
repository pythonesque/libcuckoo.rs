// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cell::UnsafeCell;
use core::marker;
use core::ops::{Deref, DerefMut};
use self::poison::{TryLockError, TryLockResult, /*LockResult*/};
use super::sys::os as sys;

pub mod poison {
    /*use core::marker::Reflect;
    use core::cell::UnsafeCell;
    use core::fmt;
    use std::thread;
    use std::error::{Error};

    pub struct Flag { failed: UnsafeCell<bool> }

    // This flag is only ever accessed with a lock previously held. Note that this
    // a totally private structure.
    unsafe impl Send for Flag {}
    unsafe impl Sync for Flag {}

    pub const FLAG_INIT: Flag = Flag { failed: UnsafeCell { value: false } };

    impl Flag {
        #[inline]
        pub fn borrow(&self) -> LockResult<Guard> {
            let ret = Guard { panicking: thread::panicking() };
            if unsafe { *self.failed.get() } {
                Err(PoisonError::new(ret))
            } else {
                Ok(ret)
            }
        }

        #[inline]
        pub fn done(&self, guard: &Guard) {
            if !guard.panicking && thread::panicking() {
                unsafe { *self.failed.get() = true; }
            }
        }

        #[inline]
        pub fn get(&self) -> bool {
            unsafe { *self.failed.get() }
        }
    }

    pub struct Guard {
        panicking: bool,
    }

    /// A type of error which can be returned whenever a lock is acquired.
    ///
    /// Both Mutexes and RwLocks are poisoned whenever a thread fails while the lock
    /// is held. The precise semantics for when a lock is poisoned is documented on
    /// each lock, but once a lock is poisoned then all future acquisitions will
    /// return this error.
    pub struct PoisonError<T> {
        guard: T,
    }*/

    /// An enumeration of possible errors which can occur while calling the
    /// `try_lock` method.
    pub enum TryLockError/*<T>*/ {
        /// The lock could not be acquired because another thread failed while holding
        /// the lock.
        //Poisoned(PoisonError<T>),
        /// The lock could not be acquired at this time because the operation would
        /// otherwise block.
        WouldBlock,
    }

    /*/// A type alias for the result of a lock method which can be poisoned.
    ///
    /// The `Ok` variant of this result indicates that the primitive was not
    /// poisoned, and the `Guard` is contained within. The `Err` variant indicates
    /// that the primitive was poisoned. Note that the `Err` variant *also* carries
    /// the associated guard, and it can be acquired through the `into_inner`
    /// method.
    pub type LockResult<Guard> = Result<Guard, PoisonError<Guard>>;*/

    /// A type alias for the result of a nonblocking locking method.
    ///
    /// For more information, see `LockResult`. A `TryLockResult` doesn't
    /// necessarily hold the associated guard in the `Err` type as the lock may not
    /// have been acquired for other reasons.
    pub type TryLockResult<Guard> = Result<Guard, TryLockError/*<Guard>*/>;

    /*impl<T> fmt::Debug for PoisonError<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            "PoisonError { inner: .. }".fmt(f)
        }
    }

    impl<T> fmt::Display for PoisonError<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            "poisoned lock: another task failed inside".fmt(f)
        }
    }

    impl<T: Send + Reflect> Error for PoisonError<T> {
        fn description(&self) -> &str {
            "poisoned lock: another task failed inside"
        }
    }

    impl<T> PoisonError<T> {
        /// Creates a `PoisonError`.
        pub fn new(guard: T) -> PoisonError<T> {
            PoisonError { guard: guard }
        }

        /// Consumes this error indicating that a lock is poisoned, returning the
        /// underlying guard to allow access regardless.
        pub fn into_inner(self) -> T { self.guard }

        /// Reaches into this error indicating that a lock is poisoned, returning a
        /// reference to the underlying guard to allow access regardless.
        pub fn get_ref(&self) -> &T { &self.guard }

        /// Reaches into this error indicating that a lock is poisoned, returning a
        /// mutable reference to the underlying guard to allow access regardless.
        pub fn get_mut(&mut self) -> &mut T { &mut self.guard }
    }

    impl<T> From<PoisonError<T>> for TryLockError<T> {
        fn from(err: PoisonError<T>) -> TryLockError<T> {
            TryLockError::Poisoned(err)
        }
    }

    impl<T> fmt::Debug for TryLockError<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match *self {
                TryLockError::Poisoned(..) => "Poisoned(..)".fmt(f),
                TryLockError::WouldBlock => "WouldBlock".fmt(f)
            }
        }
    }

    impl<T: Send + Reflect> fmt::Display for TryLockError<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.description().fmt(f)
        }
    }

    impl<T: Send + Reflect> Error for TryLockError<T> {
        fn description(&self) -> &str {
            match *self {
                TryLockError::Poisoned(ref p) => p.description(),
                TryLockError::WouldBlock => "try_lock failed because the operation would block"
            }
        }

        fn cause(&self) -> Option<&Error> {
            match *self {
                TryLockError::Poisoned(ref p) => Some(p),
                _ => None
            }
        }
    }

    pub fn map_result<T, U, F>(result: LockResult<T>, f: F)
                               -> LockResult<U>
                               where F: FnOnce(T) -> U {
        match result {
            Ok(t) => Ok(f(t)),
            Err(PoisonError { guard }) => Err(PoisonError::new(f(guard)))
        }
    }*/
}

/// The static mutex type is provided to allow for static allocation of mutexes.
///
/// Note that this is a separate type because using a Mutex correctly means that
/// it needs to have a destructor run. In Rust, statics are not allowed to have
/// destructors. As a result, a `StaticMutex` has one extra method when compared
/// to a `Mutex`, a `destroy` method. This method is unsafe to call, and
/// documentation can be found directly on the method.
/// ```
pub struct StaticMutex {
    lock: sys::Mutex,
    //poison: poison::Flag,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be access through this guard via its
/// `Deref` and `DerefMut` implementations
#[must_use]
pub struct MutexGuard<'a, T: ?Sized + 'a> {
    // funny underscores due to how Deref/DerefMut currently work (they
    // disregard field privacy).
    __lock: &'a StaticMutex,
    __data: &'a UnsafeCell<T>,
    //__poison: poison::Guard,
}

impl<'a, T: ?Sized> !marker::Send for MutexGuard<'a, T> {}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub const MUTEX_INIT: StaticMutex = StaticMutex {
    lock: sys::MUTEX_INIT,
    //poison: poison::FLAG_INIT,
};

impl StaticMutex {
    /// Acquires this lock, see `Mutex::lock`
    #[inline]
    pub fn lock(&'static self) -> /*LockResult<*/MutexGuard<()>/*>*/ {
        unsafe { self.lock.lock() }
        MutexGuard::new(self, &DUMMY.0)
    }

    /// Attempts to grab this lock, see `Mutex::try_lock`
    #[inline]
    pub fn try_lock(&'static self) -> TryLockResult<MutexGuard<()>> {
        if unsafe { self.lock.try_lock() } {
            Ok(/*try!(*/MutexGuard::new(self, &DUMMY.0)/*)*/)
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Deallocates resources associated with this static mutex.
    ///
    /// This method is unsafe because it provides no guarantees that there are
    /// no active users of this mutex, and safety is not guaranteed if there are
    /// active users of this mutex.
    ///
    /// This method is required to ensure that there are no memory leaks on
    /// *all* platforms. It may be the case that some platforms do not leak
    /// memory if this method is not called, but this is not guaranteed to be
    /// true on all platforms.
    pub unsafe fn destroy(&'static self) {
        self.lock.destroy()
    }
}

pub fn guard_lock<'a, T: ?Sized>(guard: &MutexGuard<'a, T>) -> &'a sys::Mutex {
    &guard.__lock.lock
}

/*pub fn guard_poison<'a, T: ?Sized>(guard: &MutexGuard<'a, T>) -> &'a poison::Flag {
    &guard.__lock.poison
}*/

struct Dummy(UnsafeCell<()>);
unsafe impl Sync for Dummy {}
static DUMMY: Dummy = Dummy(UnsafeCell::new(()));

impl<'mutex, T: ?Sized> MutexGuard<'mutex, T> {

    fn new(lock: &'mutex StaticMutex, data: &'mutex UnsafeCell<T>)
           -> /*LockResult<*/MutexGuard<'mutex, T>/*>*/ {
        //poison::map_result(lock.poison.borrow(), |guard| {
            MutexGuard {
                __lock: lock,
                __data: data,
                //__poison: guard,
            }
        //})
    }
}

impl<'mutex, T: ?Sized> Deref for MutexGuard<'mutex, T> {
    type Target = T;

    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.__data.get() }
    }
}
impl<'mutex, T: ?Sized> DerefMut for MutexGuard<'mutex, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.__data.get() }
    }
}

impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            //self.__lock.poison.done(&self.__poison);
            self.__lock.lock.unlock();
        }
    }
}
