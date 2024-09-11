// RUN: %{ispc} %s --target=host --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s -fno-function-sections --target=host --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s -ffunction-sections --target=host --emit-asm -o - | FileCheck %s --check-prefix=CHECK-SEC

// REQUIRES: WINDOWS_HOST || LINUX_HOST

// Do not run on macOS, because the behavior of -ffunction-sections is
// different. -ffunction-sections does not create separate sections for
// functions on macOS, although unused functions are still removed if
// -dead_strip is used. In macOS case, we are following clang, which does have
// this option on macOS, but it does nothing.

// CHECK-SEC: .section{{.*}}foo___
// CHECK-SEC: .section{{.*}}foo
// CHECK-NOT: .section        .text.
// CHECK-NOT: .section{{.*}}foo___
// CHECK-NOT: .section{{.*}}foo
export void foo() { return; }
