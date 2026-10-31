// Regression test for https://github.com/ispc/ispc/issues/3985
// Unsigned indices whose byte offset is in [2^31, 2^32) must reach the right
// address with the default 32-bit addressing: such an offset must be zero
// extended, not sign extended. Covers targets without a gather instruction
// (sse4) and without a scatter instruction (avx2), p[i - 1], where the 32-bit
// offset wraps and must not be split into a constant and a variable part in 64
// bits, and p[j - 1] for j > 0, which becomes a vector load or store although
// the offset of lane 0, which is off, wraps, and p[j + 1] for j just below
// 2^32 - 1, where the trailing lanes are off and wrap, and p[j - d] with only
// lane 1 on, whose index is 0 while lane 0 wraps. The memory is reserved
// without access, except for small windows at p, p + 2 GB and below p + 4 GB,
// so a wrong address faults.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %s -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin | FileCheck %s

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --addressing=64 -h %t-sse4-a64.h %s -o %t-sse4-a64.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4-a64.c.o --include %t-sse4-a64.h
// RUN: %{cc} %t-sse4-a64.o %t-sse4-a64.c.o -o %t-sse4-a64.bin
// RUN: %t-sse4-a64.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %s -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 -h %t-avx2-a64.h %s -o %t-avx2-a64.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-a64.c.o --include %t-avx2-a64.h
// RUN: %{cc} %t-avx2-a64.o %t-avx2-a64.c.o -o %t-avx2-a64.bin
// RUN: %t-avx2-a64.bin | FileCheck %s

// InstCombine reassociates the index of minus_uniform_load on generic targets.
// RUN: %{ispc} --pic --target=generic-i1x4 -O2 --woff -h %t-generic.h %s -o %t-generic.o
// RUN: %{cc} -O2 -x c -c %s -o %t-generic.c.o --include %t-generic.h
// RUN: %{cc} %t-generic.o %t-generic.c.o -o %t-generic.bin
// RUN: %t-generic.bin | FileCheck %s

// Like 3513.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST

// CHECK: gather ok
// CHECK-NEXT: gather_i8 ok
// CHECK-NEXT: scatter ok
// CHECK-NEXT: load ok
// CHECK-NEXT: store ok
// CHECK-NEXT: store_rows ok
// CHECK-NEXT: gather_minus_1 ok
// CHECK-NEXT: shift_load ok
// CHECK-NEXT: shift_store ok
// CHECK-NEXT: tail_load ok
// CHECK-NEXT: loop_shift_load ok
// CHECK-NEXT: minus_uniform_load ok

#ifdef ISPC
export uniform int width() { return programCount; }

export void gather(uniform float r[], uniform float p[], uniform unsigned int idx[]) {
    foreach (j = 0 ... programCount) {
        unsigned int i = idx[j];
        r[j] = p[i];
    }
}

export void gather_i8(uniform int8 r[], uniform int8 p[], uniform unsigned int idx[]) {
    foreach (j = 0 ... programCount) {
        unsigned int i = idx[j];
        r[j] = p[i];
    }
}

export void scatter(uniform float p[], uniform unsigned int idx[]) {
    foreach (j = 0 ... programCount) {
        unsigned int i = idx[j];
        p[i] = 2.0f;
    }
}

export void load(uniform float r[], uniform float p[], uniform unsigned int base) {
    foreach (k = 0 ... programCount) {
        r[k] = p[base + k];
    }
}

export void store(uniform float p[], uniform unsigned int base) {
    foreach (k = 0 ... programCount) {
        p[base + k] = 3.0f;
    }
}

// The store of examples/cpu/sgemm.
export void store_rows(uniform float p[], uniform unsigned int m, uniform int K, uniform unsigned int k0) {
    for (uniform unsigned int i = 0; i < 2; i++) {
        foreach (k = 0 ... programCount) {
            p[(m + i) * K + k0 + k] = 4.0f;
        }
    }
}

export void gather_minus_1(uniform float r[], uniform float p[], uniform unsigned int idx[]) {
    foreach (j = 0 ... programCount) {
        unsigned int i = idx[j];
        r[j] = p[i - 1];
    }
}

export void shift_load(uniform float r[], uniform float p[], uniform unsigned int base) {
    unsigned int j = base + programIndex;
    float v = -1.0f;
    if (j > 0) {
        v = p[j - 1];
    }
    r[programIndex] = v;
}

export void shift_store(uniform float p[], uniform unsigned int base) {
    unsigned int j = base + programIndex;
    if (j > 0) {
        p[j - 1] = 5.0f;
    }
}

// The lanes from 2 on are off; their offsets wrap to 0. Lane 0 is on, so its
// offset gives the address.
export void tail_load(uniform int8 r[], uniform int8 p[], uniform unsigned int base) {
    unsigned int j = base + programIndex;
    int8 v = -1;
    if (programIndex < 2) {
        v = p[j + 1];
    }
    r[programIndex] = v;
}
// A varying loop index: j - 1 adds the uniform constant 0xFFFFFFFF, so lane 0,
// which is off in the first iteration, wraps.
export void loop_shift_load(uniform int8 r[], uniform int8 p[], uniform unsigned int n) {
    for (unsigned int j = programIndex; j < n; j += programCount) {
        if (j > 0) {
            r[j] = p[j - 1];
        }
    }
}

// InstCombine may turn j - d into splat(b - d) + <0, 1, ...>, whose lane 0,
// which is off, wraps for b = 0, d = 1.
export void minus_uniform_load(uniform int8 r[], uniform int8 p[], uniform unsigned int b, uniform unsigned int d,
                               uniform int first, uniform int end) {
    unsigned int j = b + programIndex;
    int8 v = -1;
    if (programIndex >= first && programIndex < end) {
        v = p[j - d];
    }
    r[programIndex] = v;
}
#else
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

#define GB (1ull << 30)
#define WINDOW (1 << 16)
// float index whose byte offset is 2^31 + 20
#define FAR ((1u << 29) + 5)

static void report(const char *name, int ok) {
    printf("%s %s\n", name, ok ? "ok" : "wrong");
    fflush(stdout);
}

int main() {
    // p - 2 GB and p + 4 GB are reserved, too.
    char *reserved = (char *)mmap(NULL, 8 * GB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reserved == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    char *base = reserved + 2 * GB;
    if (mprotect(base, WINDOW, PROT_READ | PROT_WRITE) ||
        mprotect(base + 2 * GB - WINDOW, 2 * WINDOW, PROT_READ | PROT_WRITE) ||
        mprotect(base + 4 * GB - WINDOW, WINDOW, PROT_READ | PROT_WRITE)) {
        perror("mprotect");
        return 1;
    }
    float *p = (float *)base;
    int8_t *p8 = (int8_t *)base;
    int w = width();
    uint32_t idx[64];
    float r[64];
    int8_t r8[64];
    int ok;

    for (int j = 0; j < w; j++) {
        idx[j] = FAR + 7 * j;
        p[idx[j]] = 10.0f + j;
    }
    gather(r, p, idx);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r[j] == 10.0f + j;
    report("gather", ok);

    for (int j = 0; j < w; j++) {
        idx[j] = (1u << 31) + 3 + 5 * j;
        p8[idx[j]] = (int8_t)(20 + j);
    }
    gather_i8(r8, p8, idx);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r8[j] == 20 + j;
    report("gather_i8", ok);

    for (int j = 0; j < w; j++)
        idx[j] = FAR + 7 * j;
    scatter(p, idx);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= p[idx[j]] == 2.0f;
    report("scatter", ok);

    for (int j = 0; j < w; j++)
        p[FAR + j] = 30.0f + j;
    load(r, p, FAR);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r[j] == 30.0f + j;
    report("load", ok);

    store(p, FAR);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= p[FAR + j] == 3.0f;
    report("store", ok);

    store_rows(p, 0, 1, FAR);
    ok = 1;
    for (int j = 0; j < w + 1; j++)
        ok &= p[FAR + j] == 4.0f;
    report("store_rows", ok);

    for (int j = 0; j < w; j++) {
        idx[j] = 100 + 3 * j;
        p[idx[j] - 1] = 40.0f + j;
    }
    gather_minus_1(r, p, idx);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r[j] == 40.0f + j;
    report("gather_minus_1", ok);

    float *q = p + 64;
    for (int j = -1; j < w; j++)
        q[j] = (float)j;
    shift_load(r, q, 0);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r[j] == (j == 0 ? -1.0f : (float)(j - 1));
    report("shift_load", ok);

    shift_store(q, 0);
    ok = q[-1] == -1.0f;
    for (int j = 0; j < w - 1; j++)
        ok &= q[j] == 5.0f;
    report("shift_store", ok);

    p8[4 * GB - 2] = 50;
    p8[4 * GB - 1] = 51;
    tail_load(r8, p8, 0xFFFFFFFDu);
    ok = r8[0] == 50 && r8[1] == 51;
    for (int j = 2; j < w; j++)
        ok &= r8[j] == -1;
    report("tail_load", ok);

    for (int j = 0; j < 2 * w + 3; j++) {
        p8[j] = (int8_t)(60 + j);
        r8[j] = -1;
    }
    loop_shift_load(r8, p8, 2 * w + 3);
    ok = r8[0] == -1;
    for (int j = 1; j < 2 * w + 3; j++)
        ok &= r8[j] == 60 + j - 1;
    report("loop_shift_load", ok);

    p8[0] = 43;
    minus_uniform_load(r8, p8, 0, 1, 1, 2);
    ok = 1;
    for (int j = 0; j < w; j++)
        ok &= r8[j] == (j == 1 ? 43 : -1);
    report("minus_uniform_load", ok);
    return 0;
}
#endif // ISPC
