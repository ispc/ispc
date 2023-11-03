// This test checks whether ISPC generates correct C and C++ header regarding reference as output value.
// Check here that this test is compilable with ISPC/C/CPP compilers and runnable producing the sane output.

// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: uint x=99

#ifdef ISPC
uniform uint x = 99;
export uniform uint& def() {
    return x;
}
#else
#include <stdio.h>
#if defined(__cplusplus)
using namespace ispc;
int main(int argc, char **argv) {
    uint32_t x = def();
    printf("uint x=%d\n", x);
    return 0;
}
#else
int main(int argc, char **argv) {
    uint32_t *x = def();
    printf("uint x=%d\n", *x);
    return 0;
}
#endif
#endif
