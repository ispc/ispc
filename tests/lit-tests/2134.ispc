// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16 -h %t.h --emit-llvm -o %t.bc
// RUN: FileCheck --input-file=%t.h %s -check-prefix=CHECK_ALIGN

// REQUIRES: X86_ENABLED

// CHECK_ALIGN: __declspec( align(4) ) struct bool1 { bool v[1]; };
// CHECK_ALIGN: __declspec( align(4) ) struct bool2 { bool v[2]; };
// CHECK_ALIGN: __declspec( align(4) ) struct bool3 { bool v[3]; };
// CHECK_ALIGN: __declspec( align(4) ) struct bool4 { bool v[4]; };
// CHECK_ALIGN: __declspec( align(8) ) struct bool5 { bool v[5]; };
typedef bool<1> bool1;
typedef bool<2> bool2;
typedef bool<3> bool3;
typedef bool<4> bool4;
typedef bool<5> bool5;

// CHECK_ALIGN: __ISPC_ALIGN__(4) bool b[4];
// CHECK_ALIGN: __ISPC_ALIGN__(8) bool b[8];
// CHECK_ALIGN: __ISPC_ALIGN__(16) bool b[16];
struct E {
    varying bool b;
};

export void check_E(uniform bool1 * uniform p1,
                    uniform bool2 * uniform p2,
                    uniform bool3 * uniform p3,
                    uniform bool4 * uniform p4,
                    uniform bool5 * uniform p5,
                    varying bool * uniform p6,
                    varying struct E * uniform p7){
}
