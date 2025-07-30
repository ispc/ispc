// This test checks whether ISPC produces correct representation of bool type
// on ABI boundary for targets with i8 masks.

// RUN: %{ispc} --pic --target=sse4.1-i8x16 -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

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
