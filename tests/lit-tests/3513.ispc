// This test checks whether ISPC produces correct representation of bool type
// on ABI boundary for targets with i8 masks.

// RUN: %{ispc} --pic --target=avx2-i16x16 -h %t-avx2-i16x16.h %s -o %t-avx2-i16x16.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i16x16.c.o --include %t-avx2-i16x16.h
// RUN: %{cc} %t-avx2-i16x16.o %t-avx2-i16x16.c.o -o %t-avx2-i16x16.c.bin
// RUN: %t-avx2-i16x16.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i16x16.cpp.o --include %t-avx2-i16x16.h
// RUN: %{cc} %t-avx2-i16x16.o %t-avx2-i16x16.cpp.o -o %t-avx2-i16x16.cpp.bin
// RUN: %t-avx2-i16x16.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i32x16 -h %t-avx2-i32x16.h %s -o %t-avx2-i32x16.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i32x16.c.o --include %t-avx2-i32x16.h
// RUN: %{cc} %t-avx2-i32x16.o %t-avx2-i32x16.c.o -o %t-avx2-i32x16.c.bin
// RUN: %t-avx2-i32x16.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i32x16.cpp.o --include %t-avx2-i32x16.h
// RUN: %{cc} %t-avx2-i32x16.o %t-avx2-i32x16.cpp.o -o %t-avx2-i32x16.cpp.bin
// RUN: %t-avx2-i32x16.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i32x4 -h %t-avx2-i32x4.h %s -o %t-avx2-i32x4.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i32x4.c.o --include %t-avx2-i32x4.h
// RUN: %{cc} %t-avx2-i32x4.o %t-avx2-i32x4.c.o -o %t-avx2-i32x4.c.bin
// RUN: %t-avx2-i32x4.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i32x4.cpp.o --include %t-avx2-i32x4.h
// RUN: %{cc} %t-avx2-i32x4.o %t-avx2-i32x4.cpp.o -o %t-avx2-i32x4.cpp.bin
// RUN: %t-avx2-i32x4.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i32x8 -h %t-avx2-i32x8.h %s -o %t-avx2-i32x8.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i32x8.c.o --include %t-avx2-i32x8.h
// RUN: %{cc} %t-avx2-i32x8.o %t-avx2-i32x8.c.o -o %t-avx2-i32x8.c.bin
// RUN: %t-avx2-i32x8.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i32x8.cpp.o --include %t-avx2-i32x8.h
// RUN: %{cc} %t-avx2-i32x8.o %t-avx2-i32x8.cpp.o -o %t-avx2-i32x8.cpp.bin
// RUN: %t-avx2-i32x8.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i64x4 -h %t-avx2-i64x4.h %s -o %t-avx2-i64x4.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i64x4.c.o --include %t-avx2-i64x4.h
// RUN: %{cc} %t-avx2-i64x4.o %t-avx2-i64x4.c.o -o %t-avx2-i64x4.c.bin
// RUN: %t-avx2-i64x4.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i64x4.cpp.o --include %t-avx2-i64x4.h
// RUN: %{cc} %t-avx2-i64x4.o %t-avx2-i64x4.cpp.o -o %t-avx2-i64x4.cpp.bin
// RUN: %t-avx2-i64x4.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=avx2-i8x32 -h %t-avx2-i8x32.h %s -o %t-avx2-i8x32.o
// RUN: %{cc} -O2 -x c -c %s -o %t-avx2-i8x32.c.o --include %t-avx2-i8x32.h
// RUN: %{cc} %t-avx2-i8x32.o %t-avx2-i8x32.c.o -o %t-avx2-i8x32.c.bin
// RUN: %t-avx2-i8x32.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-avx2-i8x32.cpp.o --include %t-avx2-i8x32.h
// RUN: %{cc} %t-avx2-i8x32.o %t-avx2-i8x32.cpp.o -o %t-avx2-i8x32.cpp.bin
// RUN: %t-avx2-i8x32.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=sse4.1-i16x8 -h %t-sse4.1-i16x8.h %s -o %t-sse4.1-i16x8.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.1-i16x8.c.o --include %t-sse4.1-i16x8.h
// RUN: %{cc} %t-sse4.1-i16x8.o %t-sse4.1-i16x8.c.o -o %t-sse4.1-i16x8.c.bin
// RUN: %t-sse4.1-i16x8.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-sse4.1-i16x8.cpp.o --include %t-sse4.1-i16x8.h
// RUN: %{cc} %t-sse4.1-i16x8.o %t-sse4.1-i16x8.cpp.o -o %t-sse4.1-i16x8.cpp.bin
// RUN: %t-sse4.1-i16x8.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=sse4.1-i32x4 -h %t-sse4.1-i32x4.h %s -o %t-sse4.1-i32x4.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.1-i32x4.c.o --include %t-sse4.1-i32x4.h
// RUN: %{cc} %t-sse4.1-i32x4.o %t-sse4.1-i32x4.c.o -o %t-sse4.1-i32x4.c.bin
// RUN: %t-sse4.1-i32x4.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-sse4.1-i32x4.cpp.o --include %t-sse4.1-i32x4.h
// RUN: %{cc} %t-sse4.1-i32x4.o %t-sse4.1-i32x4.cpp.o -o %t-sse4.1-i32x4.cpp.bin
// RUN: %t-sse4.1-i32x4.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=sse4.1-i32x8 -h %t-sse4.1-i32x8.h %s -o %t-sse4.1-i32x8.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.1-i32x8.c.o --include %t-sse4.1-i32x8.h
// RUN: %{cc} %t-sse4.1-i32x8.o %t-sse4.1-i32x8.c.o -o %t-sse4.1-i32x8.c.bin
// RUN: %t-sse4.1-i32x8.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-sse4.1-i32x8.cpp.o --include %t-sse4.1-i32x8.h
// RUN: %{cc} %t-sse4.1-i32x8.o %t-sse4.1-i32x8.cpp.o -o %t-sse4.1-i32x8.cpp.bin
// RUN: %t-sse4.1-i32x8.cpp.bin | FileCheck %s

// RUN: %{ispc} --pic --target=sse4.1-i8x16 -h %t-sse4.1-i8x16.h %s -o %t-sse4.1-i8x16.o
// RUN: %{cc} -O2 -x c -c %s -o %t-sse4.1-i8x16.c.o --include %t-sse4.1-i8x16.h
// RUN: %{cc} %t-sse4.1-i8x16.o %t-sse4.1-i8x16.c.o -o %t-sse4.1-i8x16.c.bin
// RUN: %t-sse4.1-i8x16.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-sse4.1-i8x16.cpp.o --include %t-sse4.1-i8x16.h
// RUN: %{cc} %t-sse4.1-i8x16.o %t-sse4.1-i8x16.cpp.o -o %t-sse4.1-i8x16.cpp.bin
// RUN: %t-sse4.1-i8x16.cpp.bin | FileCheck %s

// TODO: ideally we need to introduce AVX2_HOST and similar features to
// reliably test tests like this but for now assume that AVX2 is always
// available practically on any x86_64 hosts

// REQUIRES: !MACOS_HOST && X86_ENABLED && X86_64_HOST

// CHECK: dst = true, bit representation: 0x01

#ifdef ISPC
export void test_bool_bug(uniform bool *uniform dst) {
    foreach (i = 0...1) {
        dst[i] = true;
    }
}
#else
#include <stdio.h>

#if defined(__cplusplus)
using namespace ispc;
#endif

int main() {
    bool dst;

    // Call ISPC function that sets dst[0] = true
    test_bool_bug(&dst);

    // Check the bool representation
    printf("dst = %s, bit representation: 0x%02x\n",
           dst ? "true" : "false", (unsigned char)dst);

    return 0;
}
#endif // ISPC
