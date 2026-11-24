// A uniform bool<4> is stored as 4 bytes, and pointer arithmetic steps over
// them; the difference of two pointers to it used to be divided by the size
// of its register type, <4 x i1>, 1 byte, so (p + 1) - p was 4.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: define {{.*}}@diff
// CHECK: ret i32 1
uniform int diff(uniform bool<4> *uniform p) { return (uniform int)((p + 1) - p); }
