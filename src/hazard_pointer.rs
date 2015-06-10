use std::cell::{Cell, UnsafeCell};
use std::intrinsics;
use std::mem;
use std::ops::Deref;
use super::mutex::{MUTEX_INIT, StaticMutex};

//#[derive(Clone,Copy)]
pub struct HazardPointer(pub UnsafeCell<usize>);

//pub type Gc(*mut UnsafeCell<>);

struct ThreadLocalHazardPointer(Cell<Option<&'static HazardPointer>>);

unsafe impl Sync for ThreadLocalHazardPointer {}

/// This is a hazard pointer, used to indicate which version of the TableInfo
/// is currently being used in the thread. Since cuckoohash_map operations
/// can run simultaneously in different threads, this variable is thread
/// local. Note that this variable can be safely shared between different
/// cuckoohash_map instances, since multiple operations cannot occur
/// simultaneously in one thread. The hazard pointer variable points to a
/// pointer inside a global list of pointers, that each map checks before
/// deleting any old TableInfo pointers.
#[thread_local] static HAZARD_POINTER: ThreadLocalHazardPointer =
    ThreadLocalHazardPointer(Cell::new(None));

struct GlobalHazardPointerNode {
    next: Option<&'static mut GlobalHazardPointerNode>,
    hp: HazardPointer,
}

/// A GlobalHazardPointerList stores a list of pointers that cannot be
/// deleted by an expansion thread. Each thread gets its own node in the
/// list, whose data pointer it can modify without contention.
struct GlobalHazardPointerList {
    /// hazard pointer list
    list: Option<&'static mut GlobalHazardPointerNode>,
    /// Mutex
    lock: StaticMutex,
}

#[cold]
pub fn new_hazard_pointer() -> &'static HazardPointer {
    unsafe {
        let _guard = GLOBAL_HAZARD_POINTERS.lock.lock();
        let list = &mut GLOBAL_HAZARD_POINTERS;
        let node = box GlobalHazardPointerNode {
            next: list.list.take(),
            hp: HazardPointer(UnsafeCell { value: 0 }),
        };
        list.list = Some(mem::transmute(node));
        let fst = list.list.as_mut().unwrap_or_else( || intrinsics::unreachable() );
        &fst.hp
    }
}

pub fn delete_unused<T>(old_pointers: &mut Vec<Box<UnsafeCell<T>>>) {
    unsafe {
        let _guard = GLOBAL_HAZARD_POINTERS.lock.lock();
        let list = &GLOBAL_HAZARD_POINTERS.list;
        old_pointers.retain( |ptr| {
            let mut list = list;
            while let Some(&mut GlobalHazardPointerNode { ref next, ref hp }) = *list {
                // Relaxed load should be sufficient, I *think*, but let's use SeqCst anyway until
                // I think about it more carefully.
                if intrinsics::atomic_load(hp.0.get()) == ptr.get() as *const T as usize { return true }
                list = next;
            }
            //println!("Deleting: {:p}", ptr.get());
            false
        });
        //println!("Got here");
    }
}

/// As long as the thread_local hazard_pointer is static, which means each
/// template instantiation of a cuckoohash_map class gets its own per-thread
/// hazard pointer, then each template instantiation of a cuckoohash_map
/// class can get its own global_hazard_pointers list, since different
/// template instantiations won't interfere with each other.
static mut GLOBAL_HAZARD_POINTERS: GlobalHazardPointerList =
    GlobalHazardPointerList { list: None, lock: MUTEX_INIT };

/// check_hazard_pointer should be called before any public method that loads
/// a table snapshot. It checks that the thread local hazard pointer pointer
/// is not null, and gets a new pointer if it is null.
#[inline(always)]
pub fn check_hazard_pointer() -> HazardPointerUnsetter {
    unsafe {
        let ThreadLocalHazardPointer(ref hazard_pointer) = HAZARD_POINTER;
        HazardPointerUnsetter(match hazard_pointer.get() {
            Some(hazard_pointer) => {
                // Prevent cycles which could lead to the hazard pointer mistakenly being unset.
                // I believe making this a nonatomic load is safe because nobody can possibly be
                // writing through this pointer at the moment (since it's thread local unique).
                if *hazard_pointer.0.get() != 0 {
                    //println!("hazard pointer RefCell thing");
                    intrinsics::abort();
                } else {
                    hazard_pointer
                }
            },
            None => {
                let hp = new_hazard_pointer();
                hazard_pointer.set(Some(hp));
                hp
            }
        })
    }
}

/// Once a function is finished with a version of the table, it will want to
/// unset the hazard pointer it set so that it can be freed if it needs to.
/// This is an object which, upon destruction, will unset the hazard pointer.
pub struct HazardPointerUnsetter(&'static HazardPointer);

impl Deref for HazardPointerUnsetter {
    type Target = HazardPointer;

    fn deref(&self) -> &HazardPointer {
        &self.0
    }
}

impl Drop for HazardPointerUnsetter {
    fn drop(&mut self) {
        unsafe {
            let HazardPointerUnsetter(hazard_pointer) = *self;
            // Requires release semantics since we rely on the hazard pointer acting as a fence for
            // memory safety.
            // For now, we use SeqCst to be on the safe side.
            // NOTE: This is definitely a huge bottleneck!  Seriously investigate relaxing this to
            // release.
            intrinsics::atomic_store_rel(hazard_pointer.0.get(), 0);
        }
    }
}
