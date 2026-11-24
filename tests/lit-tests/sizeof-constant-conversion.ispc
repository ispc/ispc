// A sizeof that initializes a variable of another type is converted like
// any other constant, also under casts. It used to be built directly in the
// type of the variable, so a size that doesn't fit, such as 4 for a bool or
// 400 for an int8, or any size for a float, crashed the compiler, and a bool
// global got the register type of a bool instead of its storage type.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s -DA64 --addressing=64 --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap --woff -O2 --emit-llvm-text -o - | FileCheck %s --check-prefix=A64

// REQUIRES: X86_ENABLED

// A stored true is 0xff, but C and C++ expect 1: the checks accept both.
// CHECK-DAG: @g_bool4 = {{.*}}global i8 {{-1|1}}
// CHECK-DAG: @g_vbool = {{.*}}global i8 {{-1|1}}
// CHECK-DAG: @g_int = {{.*}}global i8 {{-1|1}}
// CHECK-DAG: @g_int8 = {{.*}}global i8 -112
// CHECK-DAG: @g_float = {{.*}}global float 8.000000e+00
// CHECK-DAG: @g_varying = {{.*}}global <8 x i8> splat (i8 {{-1|1}})
// A uniform short vector gets the size in every element, and an soa<>
// member gets it in every lane.
// CHECK-DAG: @g_int4 = {{.*}}global <4 x i32> splat (i32 4)
// CHECK-DAG: @g_bool4v = {{.*}}global <4 x i8> splat (i8 {{-1|1}})
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

// A cast of a sizeof converts the size to the type of the cast, and the
// result to the type of the variable.
// CHECK-DAG: @g_cast = {{.*}}global i32 4
// CHECK-DAG: @g_cast_int8 = {{.*}}global i32 -112
// CHECK-DAG: @g_cast_float = {{.*}}global float -1.120000e+02
// CHECK-DAG: @g_cast_bool = {{.*}}global i8 0
uniform int g_cast = {(uniform int8)sizeof(uniform int)};
uniform int g_cast_int8 = (uniform int8)sizeof(uniform int[100]);
uniform float g_cast_float = (uniform int8)sizeof(uniform int[100]);
uniform bool g_cast_bool = (uniform uint8)sizeof(uniform int[64]);

// An integer is rounded to a float once: 2^54 + 2^30 + 1 rounds up to
// 2^54 + 2^31. Through a double it was 2^54 + 2^30, which rounds to 2^54.
// LLVM 22 prints this float in hex, LLVM 23 in decimal.
// CHECK-DAG: @g_round = {{.*}}global float {{1\.801440e\+16|0x4350000020000000}}
uniform float g_round = (uniform uint64)18014399583223809ull;
#ifdef A64
struct Big {
    int8 a[0x40000000];
};
struct H {
    Big b[0x1000000];
    int8 c[0x40000001];
};
// A64: @g_round_sizeof = {{.*}}global float {{1\.801440e\+16|0x4350000020000000}}
uniform float g_round_sizeof = sizeof(uniform H);
#endif

// The elements that a partial initializer list leaves out are 0 in the type
// of the given ones: a bool in memory is a byte, also for true.
// CHECK-DAG: @g_partial_bool = {{.*}}global [3 x i8] c"{{\\FF|\\01}}\00\00"
// CHECK-DAG: @g_partial_bool_sizeof = {{.*}}global [3 x i8] c"{{\\FF|\\01}}\00\00"
uniform bool g_partial_bool[3] = {true};
uniform bool g_partial_bool_sizeof[3] = {(uniform int)sizeof(uniform int8)};

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

// A local array declared without a size gets the size of its initializer
// list when its declaration is emitted.
// CHECK-LABEL: define {{.*}}@local_unsized
// CHECK: ret i32 3
uniform int local_unsized() {
    const uniform float x[] = {1, 2, 3};
    return sizeof(x) / sizeof(x[0]);
}

// CHECK-LABEL: define {{.*}}@local_partial_bool
// CHECK: ret i32 1
uniform int local_partial_bool() {
    uniform bool a[2] = {(uniform int)sizeof(uniform int8)};
    uniform bool<3> b = {true};
    return a[0] && !a[1] && b[0] && !b[1];
}
