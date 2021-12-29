// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16 -h %t.h --emit-llvm -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s -check-prefix=CHECK_TYPES
// RUN: FileCheck --input-file=%t.h %s -check-prefix=CHECK_ALIGN
// REQUIRES: X86_ENABLED

// This needs to be fixed, see #2164 for more details.
// XFAIL: LLVM_14_0+

// CHECK_TYPES: type { <4 x float> }
// CHECK_TYPES-NEXT: type { <8 x i32> }
// CHECK_TYPES-NEXT: type { <4 x i8> }
// CHECK_TYPES-NEXT: type { <16 x i64> }
// CHECK_TYPES-NEXT: type { <4 x i8> }

typedef float<3> float3;
// CHECK_ALIGN: #ifdef _MSC_VER
// CHECK_ALIGN-NEXT: __declspec( align(16) ) struct float3 { float v[3]; };
// CHECK_ALIGN-NEXT: #else
// CHECK_ALIGN-NEXT: struct float3 { float v[3]; } __attribute__ ((aligned(16)));
// CHECK_ALIGN-NEXT: #endif
struct A {
    uniform float3 value;
};

export void check_A(uniform const A * uniform x){
}

typedef int<5> int5;
// CHECK_ALIGN: #ifdef _MSC_VER
// CHECK_ALIGN-NEXT: __declspec( align(32) ) struct int32_t5 { int32_t v[5]; };
// CHECK_ALIGN-NEXT: #else
// CHECK_ALIGN-NEXT: struct int32_t5 { int32_t v[5]; } __attribute__ ((aligned(32)));
// CHECK_ALIGN-NEXT: #endif
struct B {
    uniform int5 value;
};

export void check_B(uniform const B * uniform x){
}
typedef int8<4> int8_4;
// CHECK_ALIGN: #ifdef _MSC_VER
// CHECK_ALIGN-NEXT: __declspec( align(4) ) struct int8_t4 { int8_t v[4]; };
// CHECK_ALIGN-NEXT: #else
// CHECK_ALIGN-NEXT: struct int8_t4 { int8_t v[4]; } __attribute__ ((aligned(4)));
// CHECK_ALIGN-NEXT: #endif
struct C {
    uniform int8_4 value;
};

export void check_C(uniform const C * uniform x){
}

typedef int64<9> int64_9;
// CHECK_ALIGN: #ifdef _MSC_VER
// CHECK_ALIGN-NEXT: __declspec( align(128) ) struct int64_t9 { int64_t v[9]; };
// CHECK_ALIGN-NEXT: #else
// CHECK_ALIGN-NEXT: struct int64_t9 { int64_t v[9]; } __attribute__ ((aligned(128)));
// CHECK_ALIGN-NEXT: #endif
struct D {
    uniform int64_9 value;
};

export void check_D(uniform const D * uniform x){
}

typedef bool<2> bool2;
// CHECK_ALIGN: #ifdef _MSC_VER
// CHECK_ALIGN-NEXT: __declspec( align(4) ) struct bool2 { bool v[2]; };
// CHECK_ALIGN-NEXT: #else
// CHECK_ALIGN-NEXT: struct bool2 { bool v[2]; } __attribute__ ((aligned(4)));
// CHECK_ALIGN-NEXT: #endif
struct E {
    uniform bool2 value;
};

export void check_E(uniform const E * uniform x){
}
