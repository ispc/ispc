// Ordinary nested indexing can sum two signed-i32 byte offsets to +2^31.
// Row has 3 floats: i = 178956970 gives i * 12 = 2147483640, and j = 2 gives
// j * 4 = 8. Both components fit, both indices are valid, and the allocation
// is smaller than 4GB. There is no integer-to-pointer conversion.
// ImproveMemoryOps narrows their i64 sum to a scale-1 dword index, changing
// +2^31 to -2^31. Fix during factoring/width legality, preserving performance:
// a dword index with scale 4 can represent the correct address.
//
// RUN: %{ispc} %s --pic --target=avx2-i32x8 --addressing=32 -O2 --woff -h %t.h -o %t.o
// RUN: %{cc} -O2 -x c -c %s --include %t.h -o %t.c.o
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin
// RUN: %{ispc} %s --pic --target=avx2-i32x8 --addressing=32 --wrap-signed-int -O2 --woff -o %t-wrap.o
// RUN: %{cc} %t-wrap.o %t.c.o -o %t-wrap.bin
// RUN: %t-wrap.bin
// RUN: %{ispc} %s --pic --target=avx2-i32x8 --addressing=32 -O0 --woff -o %t-o0.o
// RUN: %{cc} %t-o0.o %t.c.o -o %t-o0.bin
// RUN: %t-o0.bin
// Each runtime configuration fails independently. Passing configurations
// share this kernel/driver through address-nested-sum.ispc.
// Like 3985-1.ispc, execution assumes an AVX2-capable x86-64 host.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

#ifdef ISPC
struct Row {
    float v[3];
};

export void nested_sum(uniform Row rows[], uniform int idx[], uniform int jdx[], uniform float out[]) {
    out[programIndex] = rows[idx[programIndex]].v[jdx[programIndex]];
}
#else
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

int main(void) {
    const size_t nrows = 178956971;
    const size_t bytes = nrows * sizeof(struct Row);
    // Only one page is touched; reserve the complete object so the indices
    // stay within their source arrays without biased or integer-built pointers.
    struct Row *rows = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (rows == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    int idx[8], jdx[8];
    float out[8];
    for (int i = 0; i < 8; ++i) {
        idx[i] = 178956970;
        jdx[i] = 2;
        out[i] = -1.0f;
    }
    rows[178956970].v[2] = 9.0f;
    nested_sum(rows, idx, jdx, out);
    int failed = 0;
    for (int i = 0; i < 8; ++i) {
        if (out[i] != 9.0f) {
            printf("lane %d: got %g, expected 9\n", i, out[i]);
            failed = 1;
        }
    }
    munmap(rows, bytes);
    return failed;
}
#endif
