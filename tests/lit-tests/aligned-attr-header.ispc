// This tests checks whether the user-provided alignment for structs are
// propogated to the generated C/C++ header file.

// RUN: %{ispc} --target=host --nostdlib -h %t.h %s
// RUN: FileCheck -input-file=%t.h %s
// RUN: %{cc} -x c %t.h -E -o - | FileCheck %s --check-prefix=CHECK-C
// RUN: %{cc} -x c++ %t.h -E -o - | FileCheck %s --check-prefix=CHECK-CPP

// REQUIRES: !WINDOWS_HOST

// CHECK: __ISPC_ALIGNED_STRUCT__(256) v{{.*}}_varying_S
// CHECK-C: struct __attribute__((aligned(256))) v{{.*}}_varying_S
// CHECK-CPP: struct alignas(256) v{{.*}}_varying_S

struct S {
    int x;
} __attribute__((aligned(256)));

export void foo1(varying struct S *uniform s) {}


// CHECK: __ISPC_ALIGNED_STRUCT__(128) U
// CHECK-C: struct __attribute__((aligned(128))) U
// CHECK-CPP: struct alignas(128) U

struct U {
    int x;
} __attribute__((aligned(128)));

export void foo2(uniform struct U *uniform s) {}
