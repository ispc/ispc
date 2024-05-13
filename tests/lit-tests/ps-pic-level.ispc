// RUN: %{ispc} --target-os=ps4 --nostdlib --nowrap %s -o - --emit-llvm-text 2>&1 | FileCheck %s --check-prefix=CHECK-IR-PIC
// RUN: %{ispc} --target-os=ps4 --nostdlib --nowrap %s -o - --emit-llvm-text --pic 2>&1 | FileCheck %s --check-prefix=CHECK-IR-PIC
// RUN: %{ispc} --target-os=ps4 --nostdlib --nowrap %s -o - --emit-llvm-text --PIC 2>&1 | FileCheck %s --check-prefix=CHECK-IR-PIC

// REQUIRES: PS_ENABLED

// CHECK-IR-PIC: !"PIC Level", i32 2

uniform int global_var;

uniform int foo() { return global_var; }
