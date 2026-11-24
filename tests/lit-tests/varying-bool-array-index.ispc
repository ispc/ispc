// An array of varying bool is stored with one byte per lane. A varying
// index into it used to be scaled by the size of the mask register type (16
// bytes on sse4-i32x4, 32 on avx2-i32x8), so every lane read the wrong
// element; avx2-i8x32, whose mask is one byte per lane, was right. Varying
// indexing, constant element offsets, pointer differences and the size that
// new allocates now use the size in memory.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %s -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %s -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin

// On avx512skx-x16 the mask register is 2 bytes and the size in memory 16;
// the old size made new allocate 4 bytes for two elements of 16.
// RUN: %{ispc} --target=avx512skx-x16 -O2 --woff --emit-llvm-text %s -o - | FileCheck %s --check-prefix=SKX
// SKX-LABEL: define {{.*}}@new_rows
// SKX: @posix_memalign(ptr {{.*}}, i64 64, i64 32)

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST

#ifdef ISPC

export uniform int width() { return programCount; }

// rows[1] is true in every lane, the others false; every lane reads rows[1].
export void read_rows(uniform int idx[], uniform int out[]) {
    varying bool rows[8];
    for (uniform int k = 0; k < 8; k++)
        rows[k] = false;
    rows[1] = true;
    int i = idx[programIndex];
    out[programIndex] = rows[i] ? 1 : 0;
}

// new allocates one byte per lane for each element, the distance between
// two elements: the second element, written in full, is in the allocation.
export uniform int new_rows() {
    varying bool *uniform p = uniform new varying bool[2];
    p[1] = programIndex == 1;
    uniform int d = (uniform int)((uniform int8 * uniform)(p + 1) - (uniform int8 * uniform)p);
    uniform int ok = d == programCount && extract(p[1], 1);
    delete[] p;
    return ok;
}

#else

#include <stdio.h>

int main(void) {
    int idx[64], out[64], w = width(), bad = 0;
    for (int j = 0; j < 64; j++)
        idx[j] = 1;
    read_rows(idx, out);
    for (int j = 0; j < w; j++)
        bad += out[j] != 1;
    printf("%d of %d lanes wrong\n", bad, w);
    if (!new_rows()) {
        printf("new_rows failed\n");
        bad++;
    }
    return bad != 0;
}

#endif // ISPC
