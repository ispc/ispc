// Known wrong code: with 32-bit addressing on a 64-bit target, the
// difference of two pointers is truncated to 32 bits before it is divided by
// the element size. q = p + 0x20000000u is valid (the 32-bit byte offset
// 0x80000000 of an unsigned index doesn't wrap), and q - p is 2^29 elements,
// which fits; but the byte difference 2^31 doesn't fit in 32 bits signed, so
// the result is -2^29. No memory is accessed: both pointers are in an array
// of 2^29 + 4 ints whose address space is reserved without access.
//
// XFAIL until the difference is divided at 64 bits.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

#ifdef ISPC
export uniform int diff_uniform(uniform int *uniform p) {
    uniform int *uniform q = p + 0x20000000u;
    return q - p;
}
export uniform int diff_varying(uniform int *uniform p) {
    uniform int *varying q = p + (varying unsigned int)0x20000000u;
    uniform int *varying r = p;
    return extract(q - r, 0);
}
#else
#include <stdio.h>
#include <sys/mman.h>
int main(void) {
    size_t bytes = (((size_t)1 << 29) + 4) * sizeof(int);
    void *mem = mmap(NULL, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED) {
        // Not the bug: exit with success, which shows as an unexpected pass
        // of this expected failure.
        printf("can't reserve %zu bytes of address space\n", bytes);
        return 0;
    }
    int u = diff_uniform((int *)mem), v = diff_varying((int *)mem);
    munmap(mem, bytes);
    if (u != 0x20000000 || v != 0x20000000) {
        printf("q - p = %d (uniform), %d (varying), expected %d\n", u, v, 0x20000000);
        return 1;
    }
    return 0;
}
#endif
