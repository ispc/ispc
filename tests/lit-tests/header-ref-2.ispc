// This test checks whether ISPC generates correct C and C++ header regarding reference as output value.
// Check here that this test is compilable with ISPC/C/CPP compilers and runnable producing the sane output.

// RUN: %{cc_ispc_run} host test | FileCheck %s
// RUN: %{cxx_ispc_run} host test | FileCheck %s

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
