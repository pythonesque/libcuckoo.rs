use rand::{Closed01, Rng};
use std::cmp;
use std::cell::UnsafeCell;
use std::intrinsics;
use super::sys::{Duration, XorShiftRng};
use super::sys::arch::pause;
use super::sys::os::sleep;

const MIN_DELAY_MSEC: u32 = 1;
const MAX_DELAY_MSEC: u32 = 1000;

//type SpinLockAlign = super::simd::u64x8;
struct SpinLockAlign;

#[cfg(test)]
/// Note: we rely on these tests being correct for optimizations that would otherwise be unsafe!
mod tests {
    use super::{MAX_DELAY_MSEC, MIN_DELAY_MSEC};
    use super::super::sys::{MILLIS_PER_SEC, NANOS_PER_MILLI};

    #[test]
    fn min_delay_fine() {
        ((MIN_DELAY_MSEC as u64 % MILLIS_PER_SEC) as u32).checked_mul(NANOS_PER_MILLI).unwrap();
    }

    #[test]
    fn max_delay_fine() {
        ((MAX_DELAY_MSEC as u64 % MILLIS_PER_SEC) as u32).checked_mul(NANOS_PER_MILLI).unwrap();
    }
}

/// A fast, lightweight spinlock
#[repr(C)]
pub struct SpinLock {
    lock: UnsafeCell<u8>,
    // Plus 15 bytes to get to u64
    padding: [SpinLockAlign; 0],
}

const DEFAULT_SPINS_PER_DELAY: u32 = 100;
#[thread_local] static mut SPINS_PER_DELAY: u32 = DEFAULT_SPINS_PER_DELAY;

pub fn set_spins_per_delay(shared_spins_per_delay: u32) {
    unsafe {
        SPINS_PER_DELAY = shared_spins_per_delay;
    }
}

pub fn update_spins_per_delay(shared_spins_per_delay: u32) -> u32 {
    unsafe {
        let spins_per_delay = SPINS_PER_DELAY;
        (shared_spins_per_delay * 15u32.wrapping_add(spins_per_delay)) / 16
    }
}

const U8_TRUE: u8 = !0;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn spin_delay() {
    pause();
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn spin_delay() { }

impl SpinLock {
    #[inline(always)]
    fn tas(&self) -> bool {
        unsafe { intrinsics::atomic_xchg_acq(self.lock.get(), U8_TRUE) != 0 }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn tas_spin(&self) -> bool {
        unsafe { intrinsics::atomic_load_relaxed(self.lock.get()) != 0 || self.tas() }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    fn tas_spin(&self) -> bool {
        self.tas()
    }

    pub fn new() -> Self {
        SpinLock { lock: UnsafeCell::new(0), padding: [] }
    }

    /*fn spin_lock(&self) {
        while self.tas_spin() {
            // CPU-specific delay each time through the loop
            spin_delay();
        }
    }*/

    fn spin_lock(&self) {
        const MIN_SPINS_PER_DELAY: u32 = 10;
        const MAX_SPINS_PER_DELAY: u32 = 1000;
        const NUM_DELAYS: u32 = 1000;
        {
            let mut spins = 0u32;
            let mut delays = 0u32;
            let mut cur_delay = 0;
            let mut rng = None;

            let spins_per_delay = unsafe { SPINS_PER_DELAY };

            while self.tas_spin() {
                // CPU-specific delay each time through the loop
                spin_delay();

                // Block the process every spins_per_delay tries
                spins = spins.wrapping_add(1u32);
                if spins >= spins_per_delay {
                    delays = delays.wrapping_add(1u32);
                    if delays > NUM_DELAYS {
                        /*unsafe */{
                            //println!("abort");
                            // Lock stuck.  Currently we do nothing.
                            //intrinsics::abort();
                            delays = 0;
                        }
                    }

                    if cur_delay == 0 { // first time to delay?
                        cur_delay = MIN_DELAY_MSEC;
                    }

                    /*unsafe */{
                        //intrinsics::assume(MIN_DELAY_MSEC <= cur_delay && cur_delay <= MAX_DELAY_MSEC);
                        let duration = Duration::from_millis(cur_delay as u64);
                        sleep(duration);
                    }
                    // increase delay by a random fraction between 1X and 2X
                    // TODO: Fix this to actually use proper randomness... the default traits in
                    // the rand crate mostly do lots of panicking / unwinding so I'll probably have
                    // to seed them myself.
                    cur_delay = cur_delay.wrapping_add((cur_delay as f64 * match rng {
                        None => {
                            //let mut rng_ = rand::thread_rng();
                            let mut rng_ = XorShiftRng::new_unseeded();
                            let frac = rng_.gen::<Closed01<f64>>().0;
                            rng = Some(rng_);
                            frac
                        },
                        Some(ref mut rng) => rng.gen::<Closed01<f64>>().0,
                    }) as u32);
                    // wrap back to minimum delay when maximum is exceeded
                    if cur_delay > MAX_DELAY_MSEC {
                        cur_delay = MIN_DELAY_MSEC ;
                    }

                    spins = 0;
                }
            }

            if cur_delay == 0 {
                // we never had to delay
                if spins_per_delay < MAX_SPINS_PER_DELAY {
                    unsafe {
                        SPINS_PER_DELAY = cmp::min(spins_per_delay.wrapping_add(100), MAX_SPINS_PER_DELAY);
                    }
                }
            } else {
                if spins_per_delay > MIN_SPINS_PER_DELAY {
                    unsafe {
                        SPINS_PER_DELAY = cmp::max(spins_per_delay.wrapping_sub(1), MIN_SPINS_PER_DELAY);
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn lock(&self) {
        // fast path
        if self.tas() {
            // slow path
            self.spin_lock();
        }
    }

    #[inline(always)]
    pub fn unlock(&self) {
        unsafe { intrinsics::atomic_store_rel(self.lock.get(), 0) }
    }

    #[inline(always)]
    pub fn try_lock(&self) -> bool {
        self.tas()
    }
}
