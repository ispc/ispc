// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't type cast from type "varying int32" to type "uniform int32"

uniform int foo(int x)  {
    return (uniform int) x;
}

