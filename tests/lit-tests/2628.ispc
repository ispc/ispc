// RUN: %{ispc} --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK: extractvalue [2 x <[[#]] x float>]
// CHECK:  fneg <[[#]] x float>
typedef float<2> float2;

float2 TestFunc(float2 x) {
    return -x;
}
