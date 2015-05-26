#![feature(asm,associated_consts,box_syntax,lang_items,libc,start,std_misc,thread_local,no_std,core,unsafe_no_drop_flag,static_assert,/*rustc_private,*/zero_one,step_trait,optin_builtin_traits,scoped,simd)]

extern crate core;
extern crate libc;
extern crate num_cpus;
extern crate rand;

pub use self::map::CuckooHashMap;

mod iter;
mod hazard_pointer;
mod map;
mod mutex;
mod nodemap;
mod simd;
pub mod spinlock;
mod sys;
