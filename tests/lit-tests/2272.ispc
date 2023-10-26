// RUN: %{ispc} --target=host --nowrap --nostdlib --debug-phase=first:first --emit-llvm-text %s -o - | FileCheck %s

// CHECK: [[TYPE:%.*]] = type { <[[WIDTH:[0-9]+]] x i8> }
// CHECK: [[VAL:%.*]] = insertvalue [[TYPE]] undef, <[[WIDTH]] x i8> [[BUF:%.*]], 0

struct Test { bool b; };
export void main_kernel(uniform const Test *uniform buffer) {
    Test x = buffer[programIndex];
}
