// RUN: %{ispc} %s --nowrap --target=avx512skx-x4 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx512skx-x8 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx512skx-x16 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx512skx-x32 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i8x32 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i16x16 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i32x4 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i32x8 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i32x16 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=avx2-i64x4 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=sse4.2-i8x16 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=sse4.2-i16x8 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=neon-i32x4 --arch=aarch64 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// RUN: %{ispc} %s --nowrap --target=neon-i32x8 --arch=aarch64 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && ARM_ENABLED

#define unmangled __attribute__((unmangled))
#define ATTRS unmasked unmangled

// CHECK: define void @mask_type___(<[[WIDTH:.*]] x [[MASK:.*]]> %__mask)
void mask_type() {}

// CHECK-LABEL: @foo
// CHECK-DAG:   %calltmp = tail call half @llvm.pow.f16(half %x, half %y)
ATTRS uniform float16 foo(uniform float16 x, uniform float16 y) { return @llvm.pow(x, y); }

// CHECK-LABEL: @bar
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x float> @llvm.pow.v[[WIDTH]]f32(<[[WIDTH]] x float> %x, <[[WIDTH]] x float> %y)
ATTRS varying float bar(varying float x, varying float y) { return @llvm.pow(x, y); }

// CHECK-LABEL: @cttz_u32
// CHECK-DAG:   %calltmp = tail call {{.*}}i32 @llvm.cttz.i32(i32 %x, i1 false)
ATTRS uniform int32 cttz_u32(uniform int32 x) { return @llvm.cttz.i32(x, false); }

// CHECK-LABEL: @cttz_v32
// CHECK-DAG:  %calltmp = tail call {{.*}}<[[WIDTH]] x i32> @llvm.cttz.v[[WIDTH]]i32(<[[WIDTH]] x i32> %x, i1 false)
ATTRS int32 cttz_v32(int32 x) { return @llvm.cttz.i32(x, false); }

// CHECK-LABEL: @ctlz_u32
// CHECK-DAG:   %calltmp = tail call {{.*}}i32 @llvm.ctlz.i32(i32 %x, i1 false)
ATTRS uniform int32 ctlz_u32(uniform int32 x) { return @llvm.ctlz.i32(x, false); }

// CHECK-LABEL: @ctlz_v32
// CHECK-DAG:  %calltmp = tail call {{.*}}<[[WIDTH]] x i32> @llvm.ctlz.v[[WIDTH]]i32(<[[WIDTH]] x i32> %x, i1 false)
ATTRS int32 ctlz_v32(int32 x) { return @llvm.ctlz.i32(x, false); }

// CHECK-LABEL: @prefetch
// CHECK-DAG:   tail call void @llvm.prefetch.[[P0:(p0|p0i8)]]([[PTR:(ptr|i8\*)]] %ptr, i32 0, i32 3, i32 1)
// 0l forces integral literals to be int32
ATTRS void prefetch(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 0l, 3l, 1l); }

// CHECK-LABEL: @uadd_sat_u64
// CHECK-DAG:   %calltmp = tail call i64 @llvm.uadd.sat.i64(i64 %x, i64 %y)
ATTRS uniform uint64 uadd_sat_u64(uniform uint64 x, uniform uint64 y) { return @llvm.uadd.sat(x, y); }

// CHECK-LABEL: @uadd_sat_v64
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x i64> @llvm.uadd.sat.v[[WIDTH]]i64(<[[WIDTH]] x i64> %x, <[[WIDTH]] x i64> %y)
ATTRS uint64 uadd_sat_v64(uint64 x, uint64 y) { return @llvm.uadd.sat(x, y); }

// CHECK-LABEL: @uadd_sat_u16
// CHECK-DAG:   %calltmp = tail call i16 @llvm.sadd.sat.i16(i16 %x, i16 %y)
ATTRS uniform int16 uadd_sat_u16(uniform int16 x, uniform int16 y) { return @llvm.sadd.sat(x, y); }

// CHECK-LABEL: @uadd_sat_v16
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x i16> @llvm.sadd.sat.v[[WIDTH]]i16(<[[WIDTH]] x i16> %x, <[[WIDTH]] x i16> %y)
ATTRS int16 uadd_sat_v16(int16 x, int16 y) { return @llvm.sadd.sat(x, y); }

// CHECK-LABEL: @usub_sat_u8
// CHECK-DAG:   %calltmp = tail call i8 @llvm.usub.sat.i8(i8 %x, i8 %y)
ATTRS uniform uint8 usub_sat_u8(uniform uint8 x, uniform uint8 y) { return @llvm.usub.sat(x, y); }

// CHECK-LABEL: @usub_sat_v8
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x i8> @llvm.usub.sat.v[[WIDTH]]i8(<[[WIDTH]] x i8> %x, <[[WIDTH]] x i8> %y)
ATTRS uint8 usub_sat_v8(uint8 x, uint8 y) { return @llvm.usub.sat(x, y); }

// CHECK-LABEL: @ssub_sat_u32
// CHECK-DAG:   %calltmp = tail call i32 @llvm.ssub.sat.i32(i32 %x, i32 %y)
ATTRS uniform int32 ssub_sat_u32(uniform int32 x, uniform int32 y) { return @llvm.ssub.sat(x, y); }

// CHECK-LABEL: @ssub_sat_v32
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x i32> @llvm.ssub.sat.v[[WIDTH]]i32(<[[WIDTH]] x i32> %x, <[[WIDTH]] x i32> %y)
ATTRS int32 ssub_sat_v32(int32 x, int32 y) { return @llvm.ssub.sat(x, y); }

// CHECK-LABEL: @memset
// CHECK-DAG:   tail call void @llvm.memset.[[P0]].i32([[PTR]] align 1 %dst, i8 %val, i32 %len, i1 false)
ATTRS void memset(uniform int8 *uniform dst, uniform int8 val, uniform int32 len) {
    @llvm.memset(dst, val, len, false);
}

// CHECK-LABEL: @memcpy
// CHECK-DAG:   tail call void @llvm.memcpy.[[P0]].[[P0]].i64([[PTR]] align 1 %dst, [[PTR]] align 1 %src, i64 %len, i1 false)
ATTRS void memcpy(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int64 len) {
    @llvm.memcpy(dst, src, len, false);
}

// CHECK-LABEL: @memmove
// CHECK-DAG:   tail call void @llvm.memcpy.[[P0]].[[P0]].i32([[PTR]] align 1 %dst, [[PTR]] align 1 %src, i32 %len, i1 false)
// memcpy because all pointers are non-overlapping in ISPC (noalias)
ATTRS void memmove(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int32 len) {
    @llvm.memmove(dst, src, len, false);
}
