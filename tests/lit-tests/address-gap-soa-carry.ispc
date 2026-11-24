// Indexing a slice pointer into an soa<> array adds the index to the slice
// offset of the pointer. With 64-bit addressing this used to be a 32-bit add
// for a 32-bit index, so an unsigned index of UINT32_MAX plus the offset 1
// of &g[1] carried out of 32 bits: &q[UINT32_MAX] was &g[0] instead of
// 2^32 elements further. The sum is now split as (index >> 2) * 4 +
// ((index & 3) + offset), which can't carry. No memory is accessed: the test
// only takes the difference of two pointers into an array of 2^32 + 4
// elements, whose address space is reserved without access.

// RUN: %{ispc} --pic --target=avx2-i32x8 --addressing=64 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin
// RUN: %{ispc} --pic --target=avx2-i32x8 --addressing=64 -O0 --woff -h %t0.h %s -o %t0.o
// RUN: %{cc} -O2 -x c -c %s -o %t0.c.o --include %t0.h
// RUN: %{cc} %t0.o %t0.c.o -o %t0.bin
// RUN: %t0.bin

// An index that can't reach 2^31 (an int8 extended, plus 1) can't carry:
// it keeps the plain sum, as before.
// RUN: %{ispc} --target=avx2-i32x8 --addressing=64 -O2 --woff --emit-llvm-text --debug-phase=10:10 %s -o - | FileCheck %s
// CHECK-LABEL: define {{.*}}@soa_diff___
// CHECK: slice_carry
// CHECK-LABEL: define {{.*}}@soa_narrow___
// CHECK-NOT: slice_carry
// CHECK: {{^}}define

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST

#ifdef ISPC
struct P {
    int8 x;
};

export uniform int64 soa_diff(void *uniform mem, uniform unsigned int i) {
    soa<4> P *uniform g = (soa<4> P * uniform)mem;
    soa<4> P *uniform q = &g[1];
    varying unsigned int idx = i;
    soa<4> P *varying r = &q[idx];
    soa<4> P *varying b = &g[0];
    return extract(r - b, 0);
}

export uniform int8 soa_narrow(void *uniform mem, uniform int8 i) {
    soa<4> P *uniform g = (soa<4> P * uniform)mem;
    soa<4> P *uniform q = &g[1];
    varying int8 idx = i;
    return extract(q[idx + 1].x, 0);
}
#else
#include <stdio.h>
#include <sys/mman.h>
int main(void) {
    // 2^32 + 4 elements of 1 byte, in slices of 4.
    size_t bytes = ((size_t)1 << 32) + 4;
    void *mem = mmap(NULL, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED) {
        printf("can't reserve %zu bytes of address space\n", bytes);
        return 1;
    }
    long long d = soa_diff(mem, 0xffffffffu);
    munmap(mem, bytes);
    if (d != 4294967296LL) {
        printf("&q[UINT32_MAX] - &g[0] = %lld, expected 4294967296\n", d);
        return 1;
    }
    return 0;
}
#endif
