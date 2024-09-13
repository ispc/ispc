// RUN: %{ispc} --target=host --emit-asm %s -o - -h %t.h 2>&1 | FileCheck %s
// RUN: cat %t.h | FileCheck %s --check-prefix=CHECK-HEADER

// CHECK-NOT: foo___:
// CHECK-NOT: Ignoring unknown attribute "external_only"
// CHECK: foo:
// CHECK-HEADER: extern void foo();

export __attribute__((external_only)) void foo() { return; }
