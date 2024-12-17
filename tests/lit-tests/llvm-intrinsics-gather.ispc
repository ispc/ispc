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

// CHECK-LABEL: @masked_load_i8
// CHECK-DAG: %calltmp = tail call <[[WIDTH]] x i8> @llvm.masked.load.v[[WIDTH]]i8.[[P0:(p0|p0i8)]]([[PTR:(ptr|i8*)]] %ptr, i32 1, <[[WIDTH]] x i1> {{%.*}}, <[[WIDTH]] x i8> %passthru)
unmangled int8 masked_load_i8(int8 *uniform ptr, int8 passthru) {
    // 1l forces integral literals to be int32
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, __mask, passthru);
}

// CHECK-LABEL: @gather
// CHECK-DAG: [[PTR_VEC:%.*]] = inttoptr <[[WIDTH]] x [[ADDR_SIZE:(i64|i32)]]> {{%.*}} to <[[WIDTH]] x ptr>
// CHECK-DAGL %calltmp = tail call <[[WIDTH]] x i32> @llvm.masked.gather.v[[WIDTH]]i32.v[[WIDTH]][[P0]](<[[WIDTH]] x ptr> [[PTR_VEC]], i32 1, <[[WIDTH]] x i1> {{%.*}}, <[[WIDTH]] x i32> %passthru)
unmangled int32 gather(uint64 addrs, int32 passthru) {
    return @llvm.masked.gather(addrs, 1l, __mask, passthru);
}

// CHECK-LABEL: @masked_store_float
// CHECK-DAG: tail call void @llvm.masked.store.v[[WIDTH]]f32.[[P0]](<[[WIDTH]] x float> %val, [[PTR]] %ptr, i32 1, <[[WIDTH]] x i1> {{%.*}})
unmangled void masked_store_float(varying float *uniform ptr, varying float val) {
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, __mask);
}

// CHECK-LABEL: @scatter
// CHECK-DAG:  [[VPTR:%.*]] = inttoptr <[[WIDTH]] x [[ADDR_SIZE]]> {{%.*}} to <[[WIDTH]] x ptr>
// CHECK-DAG:  tail call void @llvm.masked.scatter.v[[WIDTH]]i8.v[[WIDTH]][[P0]](<[[WIDTH]] x i8> %val, <[[WIDTH]] x ptr> [[VPTR]], i32 1, <[[WIDTH]] x i1> {{%.*}})
unmangled void scatter(varying uint64 addr, varying int8 val) {
    @llvm.masked.scatter(val, addr, 1l, __mask);
}

// CHECK-LABEL: @reduce_add_float
// CHECK-DAG:   %calltmp = tail call float @llvm.vector.reduce.fadd.v[[WIDTH]]f32(float -0.000000e+00, <[[WIDTH]] x float> %x)
ATTRS uniform float reduce_add_float(float x) { return @llvm.vector.reduce.fadd(-0.0f, x); }

// CHECK-LABEL: @masked_compressstore
// CHECK-DAG: tail call void @llvm.masked.compressstore.v[[WIDTH]]i32(<[[WIDTH]] x i32> %val, [[PTR]] %ptr, <[[WIDTH]] x i1> {{%.*}})
unmangled void masked_compressstore(uniform int8 *uniform ptr, int val, int mask) {
    @llvm.masked.compressstore(val, ptr, __mask);
}

// CHECK-LABEL!: @masked_expandload
// CHECK-DAG: %calltmp = tail call <[[WIDTH]] x i64> @llvm.masked.expandload.v[[WIDTH]]i64([[PTR]] %from, <[[WIDTH]] x i1> {{%.*}}, <[[WIDTH]] x i64> %passthru)
unmangled int64 masked_expandload(uniform int8 *uniform from, int64 passthru, UIntMaskType mask) {
    return @llvm.masked.expandload(from, mask, passthru);
}
