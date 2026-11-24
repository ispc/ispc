// Known address gap: with 32-bit addressing, indexing a slice pointer into
// an soa<> array adds the index to the slice offset of the pointer at 32
// bits. For q = &g[1] and a signed index of INT32_MAX the sum wraps, so
// q[INT32_MAX].x is read 2^31 bytes before g instead of 2^31 bytes after it,
// although its displacement from &q->x, INT32_MAX bytes, fits in 32 bits.
// The block, 2^29 blocks of 4 bytes after g, is too far for a 32-bit offset,
// so a fix needs that product at 64 bits. With 64-bit addressing the sum is
// split and is right, see address-gap-soa-carry.ispc.
//
// The driver reserves 4 GB without access and makes only the page at
// g + 2^31 readable, so the wrong address faults.
//
// XFAIL until the block offset is computed at 64 bits when it can carry.

// RUN: %{ispc} --pic --target=sse4-i32x4 --addressing=32 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

#ifdef ISPC
struct P {
    int8 x;
};

export uniform int soa_read(void *uniform mem, uniform int i) {
    soa<4> P *uniform g = (soa<4> P * uniform)mem;
    soa<4> P *uniform q = &g[1];
    varying int idx = i;
    return extract(q[idx].x, 0);
}
#else
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
int main(void) {
    size_t gb2 = (size_t)1 << 31, page = 4096;
    char *base = (char *)mmap(NULL, 2 * gb2 + page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == (char *)MAP_FAILED) {
        printf("can't reserve %zu bytes of address space\n", 2 * gb2 + page);
        return 1;
    }
    // g is at base + 2^31; element 2^31 is block 2^29, 2^31 bytes after g.
    char *g = base + gb2;
    if (mprotect(g + gb2, page, PROT_READ | PROT_WRITE) != 0) {
        printf("mprotect failed\n");
        return 1;
    }
    g[gb2] = 42;
    int v = soa_read(g, INT32_MAX);
    munmap(base, 2 * gb2 + page);
    if (v != 42) {
        printf("q[INT32_MAX].x = %d, expected 42\n", v);
        return 1;
    }
    return 0;
}
#endif
