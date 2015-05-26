pub type CpuId = u32;

#[inline(always)]
pub fn cpuid() -> CpuId {
    let cpuid;
    unsafe {
        asm!("movl $$0xbh, %eax" : : : "eax");

        asm!("cpuid" : "={edx}"(cpuid)
                     :
                     : "eax", "ebx", "ecx", "edx"
                     : "volatile" );
    }

    cpuid
}

#[inline(always)]
pub fn pause() {
    unsafe {
        // rep; nop
        asm!("pause")
    }
}
