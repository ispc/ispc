// This test checks that NaN to bool conversion returns true (consistent with C/C++)

// RUN: %{ispc} --pic --target=host -h %t-host.h %s -o %t-host.o
// RUN: %{cc} -O2 -x c -c %s -o %t-host.c.o --include %t-host.h
// RUN: %{cc} %t-host.o %t-host.c.o -o %t-host.c.bin
// RUN: %t-host.c.bin | FileCheck %s
// RUN: %{cc} -O2 -x c++ -c %s -o %t-host.cpp.o --include %t-host.h
// RUN: %{cc} %t-host.o %t-host.cpp.o -o %t-host.cpp.bin
// RUN: %t-host.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: true true false false false true true

#ifdef ISPC
export void test_nan_to_bool() {
    uniform float a = 0.0;
    uniform int* uniform ptr = 0;
    uniform bool b1 = 0.5;              // b1 == 1 (0.5 converted to int would be zero)
    uniform bool b4 = a / 0.0;          // b4 == 1 (NaN does not compare equal to zero)
    uniform bool b5 = ptr;              // b5 == 0 (nullptr is converted to false)
    uniform bool b6 = (uniform int)0.5; // b6 == 0 (see b1)
    uniform bool b7 = 0.0;              // b7 == 0 (see b4)
    uniform bool b8 = -1;               // b8 == 1
    uniform bool b9 = 2;                // b9 == 1

    print("% % % % % % %\n", b1, b4, b5, b6, b7, b8, b9);
}
#else
#include <stdio.h>

#if defined(__cplusplus)
using namespace ispc;
#endif

int main() {
    test_nan_to_bool();
    return 0;
}
#endif // ISPC
