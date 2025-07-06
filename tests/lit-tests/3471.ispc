// This test checks that the compiler does not crash (fatal error) when using
// the `signed` keyword in types (like in casts or in function return types).

// RUN: %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:

export uniform signed int test() {
    return (const uniform signed int)0;
}
