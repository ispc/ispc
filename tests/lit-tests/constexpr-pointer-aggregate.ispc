// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: %{ispc} %s --target=generic-i32x4 --nostdlib --emit-llvm-text -o - | FileCheck %s

typedef uniform int *uniform IntPtr;

struct PtrPair {
    IntPtr p;
    uniform int v;
};

constexpr uniform PtrPair P0 = {NULL, 5};
constexpr uniform PtrPair P1 = {(IntPtr)0, 7};
constexpr uniform PtrPair P2 = {NULL};
constexpr uniform IntPtr GP0 = NULL;
constexpr uniform IntPtr GP1 = (IntPtr)0;

uniform int use() { return P0.v + P1.v + P2.v; }

// CHECK-DAG: @P0 = {{.*}} { ptr null, i32 5 }
// CHECK-DAG: @P1 = {{.*}} { ptr null, i32 7 }
// CHECK-DAG: @P2 = {{.*}} zeroinitializer
// CHECK-DAG: @GP0 = {{.*}} ptr null
// CHECK-DAG: @GP1 = {{.*}} ptr null
