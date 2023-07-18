// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16 -h %t.h --emit-llvm -o %t.bc
// -opaque-pointers option was introduced in LLVM 14.0 and it's became necessary to pass it explicitely to llvm-dis starting LLVM 16.0
// RUN: llvm-dis -opaque-pointers=0 %t.bc -o - | FileCheck %s -check-prefix=CHECK_TYPES
// REQUIRES: X86_ENABLED && LLVM_14_0+

// The test doesn't make sense in opaque mode, all types will be opaque.
// UNSUPPORTED: OPAQUE_PTRS_ENABLED

// CHECK_TYPES: type { <4 x float> }
// CHECK_TYPES-NEXT: type { <8 x i32> }
// CHECK_TYPES-NEXT: type { <4 x i8> }
// CHECK_TYPES-NEXT: type { <16 x i64> }
// CHECK_TYPES-NEXT: type { <4 x i8> }

typedef float<3> float3;
struct A {
    uniform float3 value;
};

export void check_A(uniform const A * uniform x){
}

typedef int<5> int5;
struct B {
    uniform int5 value;
};

export void check_B(uniform const B * uniform x){
}
typedef int8<4> int8_4;
struct C {
    uniform int8_4 value;
};

export void check_C(uniform const C * uniform x){
}

typedef int64<9> int64_9;
struct D {
    uniform int64_9 value;
};

export void check_D(uniform const D * uniform x){
}

typedef bool<2> bool2;
struct E {
    uniform bool2 value;
};

export void check_E(uniform const E * uniform x){
}
