use core::intrinsics;
use rand::{Rand, Rng, SeedableRng};

#[cfg(any(target_arch="x86_64", target_arch="x86"))]
#[path = "x86.rs"] pub mod arch;

#[cfg(unix)]
#[path = "unix.rs"] pub mod os;

/**
 * Returns the current value of a high-resolution performance counter
 * in nanoseconds since an unspecified epoch.
 */
pub fn precise_time_ns() -> u64 {
    return os::precise_time_ns();
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub const NANOS_PER_SEC: u32 = 1_000_000_000;
pub const NANOS_PER_MILLI: u32 = 1_000_000;
pub const MILLIS_PER_SEC: u64 = 1_000;

pub struct Duration {
    pub secs: u64,
    pub nanos: u32, // Always 0 <= nanos < NANOS_PER_SEC
}

impl Duration {
    /// Crates a new `Duration` from the specified number of seconds and
    /// additional nanosecond precision.
    ///
    /// If the nanoseconds is greater than 1 billion (the number of nanoseconds
    /// in a second), then it will carry over into the seconds provided.
    #[inline]
    pub fn new(secs: u64, nanos: u32) -> Duration {
        let secs = secs + (nanos / NANOS_PER_SEC) as u64;
        let nanos = nanos % NANOS_PER_SEC;
        Duration { secs: secs, nanos: nanos }
    }

    /// Creates a new `Duration` from the specified number of seconds.
    pub fn from_secs(secs: u64) -> Duration {
        Duration { secs: secs, nanos: 0 }
    }

    /// Creates a new `Duration` from the specified number of milliseconds.
    #[inline]
    pub fn from_millis(millis: u64) -> Duration {
        let secs = millis / MILLIS_PER_SEC;
        let nanos = ((millis % MILLIS_PER_SEC) as u32) * NANOS_PER_MILLI;
        Duration { secs: secs, nanos: nanos }
    }

    /// Returns the number of whole seconds represented by this duration.
    ///
    /// The extra precision represented by this duration is ignored (e.g. extra
    /// nanoseconds are not represented in the returned value).
    pub fn secs(&self) -> u64 { self.secs }

    /// Returns the nanosecond precision represented by this duration.
    ///
    /// This method does **not** return the length of the duration when
    /// represented by nanoseconds. The returned number always represents a
    /// fractional portion of a second (e.g. it is less than one billion).
    pub fn extra_nanos(&self) -> u32 { self.nanos }
}

/// An Xorshift[1] random number
/// generator.
///
/// The Xorshift algorithm is not suitable for cryptographic purposes
/// but is very fast. If you do not know for sure that it fits your
/// requirements, use a more secure one such as `IsaacRng` or `OsRng`.
///
/// [1]: Marsaglia, George (July 2003). ["Xorshift
/// RNGs"](http://www.jstatsoft.org/v08/i14/paper). *Journal of
/// Statistical Software*. Vol. 8 (Issue 14).
#[derive(Clone)]
pub struct XorShiftRng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl XorShiftRng {
    /// Creates a new XorShiftRng instance which is not seeded.
    ///
    /// The initial values of this RNG are constants, so all generators created
    /// by this function will yield the same stream of random numbers. It is
    /// highly recommended that this is created through `SeedableRng` instead of
    /// this function
    pub fn new_unseeded() -> XorShiftRng {
        XorShiftRng {
            x: 0x193a6754,
            y: 0xa8a7d469,
            z: 0x97830e05,
            w: 0x113ba7bb,
        }
    }
}

impl Rng for XorShiftRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let x = self.x;
        let t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w = self.w;
        self.w = w ^ (w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

impl SeedableRng<[u32; 4]> for XorShiftRng {
    /// Reseed an XorShiftRng. This will panic if `seed` is entirely 0.
    #[inline]
    fn reseed(&mut self, seed: [u32; 4]) {
        if !seed.iter().all(|&x| x == 0) {
            //println!("XorShiftRng.reseed called with an all zero seed.");
            unsafe { intrinsics::abort(); }
        }

        self.x = seed[0];
        self.y = seed[1];
        self.z = seed[2];
        self.w = seed[3];
    }

    /// Create a new XorShiftRng. This will panic if `seed` is entirely 0.
    fn from_seed(seed: [u32; 4]) -> XorShiftRng {
        if !seed.iter().all(|&x| x == 0) {
            //println!("XorShiftRng::from_seed called with an all zero seed.");
            unsafe { intrinsics::abort(); }
        }

        XorShiftRng {
            x: seed[0],
            y: seed[1],
            z: seed[2],
            w: seed[3]
        }
    }
}

impl Rand for XorShiftRng {
    fn rand<R: Rng>(rng: &mut R) -> XorShiftRng {
        let mut tuple: (u32, u32, u32, u32) = rng.gen();
        while tuple == (0, 0, 0, 0) {
            tuple = rng.gen();
        }
        let (x, y, z, w) = tuple;
        XorShiftRng { x: x, y: y, z: z, w: w }
    }
}
