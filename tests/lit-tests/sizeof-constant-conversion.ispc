// A sizeof that initializes a variable of another type is converted like
// any other constant. It used to be built directly in the type of the
// variable, so a size that doesn't fit, such as 4 for a bool or 400 for an
// int8, or any size for a float, crashed the compiler, and a bool global got
// the register type of a bool instead of its storage type.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-DAG: @g_bool4 = {{.*}}global i8 -1
// CHECK-DAG: @g_vbool = {{.*}}global i8 -1
// CHECK-DAG: @g_int = {{.*}}global i8 -1
// CHECK-DAG: @g_int8 = {{.*}}global i8 -112
// CHECK-DAG: @g_float = {{.*}}global float 8.000000e+00
// CHECK-DAG: @g_varying = {{.*}}global <8 x i8> splat (i8 -1)
// A uniform short vector gets the size in every element, and an soa<>
// member gets it in every lane.
// CHECK-DAG: @g_int4 = {{.*}}global <4 x i32> splat (i32 4)
// CHECK-DAG: @g_bool4v = {{.*}}global <4 x i8> splat (i8 -1)
// CHECK-DAG: @g_soa = {{.*}}global %v8_soa8_S { [8 x i32] [i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4] }
uniform bool g_bool4 = sizeof(uniform bool<4>);
uniform bool g_vbool = sizeof(varying bool);
uniform bool g_int = (uniform bool)sizeof(uniform int);
uniform int8 g_int8 = sizeof(uniform int[100]);
uniform float g_float = sizeof(uniform double);
bool g_varying = sizeof(uniform int);
uniform int<4> g_int4 = sizeof(uniform int);
uniform bool<4> g_bool4v = sizeof(uniform int);
enum E { A };
struct S {
    E x;
};
soa<8> S g_soa = {{(uniform E)sizeof(uniform int)}};

// CHECK-LABEL: define {{.*}}@local_bool
// CHECK: ret i32 1
uniform int local_bool() {
    uniform bool b = sizeof(uniform bool<4>);
    return b;
}

// CHECK-LABEL: define {{.*}}@local_int8
// CHECK: ret i8 -112
uniform int8 local_int8() {
    uniform int8 b = sizeof(uniform int[100]);
    return b;
}

// CHECK-LABEL: define {{.*}}@local_float
// CHECK: ret float 8.000000e+00
uniform float local_float() {
    uniform float f = sizeof(uniform double);
    return f;
}
