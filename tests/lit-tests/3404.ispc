// This test checks that the compiler reports an error when calling a function 
// taking a void-typed value in argument (rather than a float-convertible value), 
// instead of a crash (fatal error).

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Unable to find any matching overload for call to function "cube".
// CHECK-NOT: FATAL ERROR:

void cube(float x) {
    return x * x * x;
}

export void power_9_3(uniform float values[]) {
    cube(cube(values[programIndex]));
}
