// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// sizeof of a reference-returning call is the size of the referenced type,
// both when it is evaluated directly and when it is folded into a constant
// initializer of a local or global variable.
// RUN: %{ispc} %s --arch=x86-64 --target=avx2-i32x8 --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED

uniform int &get();
varying double &get_varying();

// CHECK-DAG: @global_size = {{.*}}global i32 4{{$}}
uniform int global_size = sizeof(get());
// CHECK-DAG: @global_varying_size = {{.*}}global i32 64{{$}}
uniform int global_varying_size = sizeof(get_varying());

// CHECK-LABEL: define {{.*}}i32 @direct_size(
// CHECK: ret i32 4{{$}}
export uniform int direct_size() { return sizeof(get()); }

// CHECK-LABEL: define {{.*}}i32 @local_size(
// CHECK: ret i32 4{{$}}
export uniform int local_size() {
    uniform int n = sizeof(get());
    return n;
}

// CHECK-LABEL: define {{.*}}i32 @direct_varying_size(
// CHECK: ret i32 64{{$}}
export uniform int direct_varying_size() { return sizeof(get_varying()); }

// CHECK-LABEL: define {{.*}}i32 @local_varying_size(
// CHECK: ret i32 64{{$}}
export uniform int local_varying_size() {
    uniform int n = sizeof(get_varying());
    return n;
}

struct S {
    int a, b, c;
};
uniform S &get_struct();

// CHECK-LABEL: define {{.*}}i32 @local_struct_size(
// CHECK: ret i32 12{{$}}
export uniform int local_struct_size() {
    uniform int n = sizeof(get_struct());
    return n;
}

// CHECK-LABEL: define {{.*}}i32 @local_param_size(
// CHECK: ret i32 28{{$}}
export uniform int local_param_size(uniform int (&a)[7]) {
    uniform int n = sizeof(a);
    return n;
}

// CHECK-LABEL: define {{.*}}i32 @local_type_size(
// CHECK: ret i32 4{{$}}
export uniform int local_type_size() {
    uniform int n = sizeof(uniform int &);
    return n;
}
