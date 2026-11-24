// Known wrong code: a struct with __attribute__((aligned(16))) is padded to
// 16 bytes in the generated C header (__ISPC_ALIGNED_STRUCT__(16)), as in
// C, but ISPC lays out its arrays with the stride of the unpadded struct
// (4 bytes for one int). An exported function that indexes an array of the
// application then reads the wrong elements.
//
// XFAIL until the struct layout of ISPC pads a struct to its alignment.

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

#ifdef ISPC
struct S {
    int x;
} __attribute__((aligned(16)));

export void read_x(uniform S a[], uniform int out[]) { out[programIndex] = a[programIndex].x; }
#else
#include <stdio.h>
int main(void) {
    struct S a[8];
    int out[8], ok = 1;
    for (int i = 0; i < 8; i++)
        a[i].x = 100 + i;
    read_x(a, out);
    for (int i = 0; i < 8; i++)
        ok &= out[i] == 100 + i;
    if (!ok)
        printf("sizeof(struct S) = %zu, out[1] = %d, expected 101\n", sizeof(struct S), out[1]);
    return !ok;
}
#endif
