// Keep the prefetch immediates and zero-input semantics through the generic builtin wrappers.
// RUN: %{ispc} %s --target=generic-i32x4 --nowrap -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=generic-i32x4 --nowrap -O0 --off-phase=all --emit-llvm-text -o - | FileCheck %s

void prefetch_lines(uniform int8 *uniform p) {
    prefetch_l1(p);
    prefetch_l2(p + 64);
    prefetch_l3(p + 128);
    prefetch_nt(p + 192);
    prefetchw_l1(p + 256);
    prefetchw_l2(p + 320);
    prefetchw_l3(p + 384);
}

// CHECK-LABEL: define void @prefetch_lines___
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 2, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 1, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 2, i32 1)
// CHECK-DAG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 1, i32 1)

uniform int32 count_zeros(uniform int32 x) { return count_leading_zeros(x) + count_trailing_zeros(x); }

// CHECK-LABEL: define {{.*}}i32 @count_zeros___
// CHECK-DAG: call {{.*}}i32 @llvm.ctlz.i32(i32 {{.*}}, i1 false)
// CHECK-DAG: call {{.*}}i32 @llvm.cttz.i32(i32 {{.*}}, i1 false)
