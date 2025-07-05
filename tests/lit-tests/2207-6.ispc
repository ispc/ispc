// This test checks that the compiler does not accept value modification
// suffixes after type modification suffixes.

// RUN: not %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK: Error:
// CHECK-NOT: FATAL ERROR:

uniform int test()
{
    return 1llG;
}

