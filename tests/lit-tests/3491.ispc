// This test checks that ISPC generates the correct code for selects with bool
// condition for targets with i8 masks.

// RUN: %{ispc} --target=sse4.1-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=sse4.2-i8x16 --emit-llvm-text -O2 -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: sext

#define N 3

uniform uint8<N> embedded_select(uniform bool<N> cond, uniform uint8<N> t, uniform uint8<N> f) {
    uniform uint8<N> result;

    foreach (i = 0 ... N) {
        result[i] = select(cond[i], t[i], f[i]);
    }

    return result;
}
