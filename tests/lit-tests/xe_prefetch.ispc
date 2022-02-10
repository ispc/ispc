// RUN: %{ispc} %s --target=xehpc-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// RUN: %{ispc} %s --target=xelpg-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// RUN: %{ispc} %s --target=xehpg-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck -check-prefix=CHECK_NO_PREFETCH %s
// RUN: %{ispc} %s --target=xelp-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck -check-prefix=CHECK_NO_PREFETCH %s

// REQUIRES: XE_ENABLED

// CHECK-NOT: llvm.prefetch
// CHECK_NO_PREFETCH-NOT: .prefetch
export void f_uniform(uniform float aFOO[]) {
  // CHECK: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 6, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
    prefetch_l1(aFOO);
    prefetch_l2(aFOO);
    prefetch_l3(aFOO);
    prefetch_nt(aFOO);
}

export void f_uniform_sized(uniform float aFOO[]) {
  // CHECK: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 2, i8 1, i16 1, i32 0, i8 1, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 1, i8 2, i16 1, i32 0, i8 2, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.i1.i64(i1 true, i8 0, i8 6, i8 2, i16 1, i32 0, i8 4, i8 1, i8 1, i8 0, i64 %{{.*}}, i32 0)
    prefetch_l1(aFOO, 1);
    prefetch_l2(aFOO, 2);
    prefetch_l3(aFOO, 4);
    prefetch_nt(aFOO, 8);
}

int64 zero = 0;
export void f_varying(uniform float aFOO[]) {
    uniform int64 a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
      a[i] = aFOO[i];

    int64 *ptr = &(a[programIndex+zero]);
  // CHECK: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i64> %gep_offset, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i64> %gep_offset, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i64> %gep_offset, i32 0)
  // CHECK-NEXT: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i8 0, i8 6, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i64> %gep_offset, i32 0)
    prefetch_l1(ptr);
    prefetch_l2(ptr);
    prefetch_l3(ptr);
    prefetch_nt(ptr);
}
