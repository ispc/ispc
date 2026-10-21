// A local struct with a member of its own type is incomplete; it must be
// reported without a crash, even if it shadows a complete struct with the
// same name.
// See https://github.com/ispc/ispc/issues/3959

// RUN: not %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O2 -o %t.o 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: Error: incomplete type 'S'.
// CHECK: Error: Attempt to allocate memory for incomplete type "uniform struct S".
struct S { int x; };
uniform S outer;

export void f() {
    struct S { S s[]; };
    uniform S s;
}
