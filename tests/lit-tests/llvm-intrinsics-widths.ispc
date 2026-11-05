// Check that bool masks and uniform short vectors are converted to the types
// of LLVM intrinsic parameters on targets with different mask and element
// widths. uniform short vectors are padded based on the element width, e.g.
// uniform double<2> is <4 x double> on avx2-i32x8 and <16 x double> on
// avx2-i8x32 (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i16x16 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i8x32 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512skx-x16 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: define {{.*}} @width___(<[[WIDTH:[0-9]+]] x
void width() {}

// varying bool is converted to <N x i1>.
// CHECK-LABEL: @masked_load_varying___
// CHECK: call <[[WIDTH]] x i8> @llvm.masked.load.v[[WIDTH]]i8.p0(ptr {{.*}}%p, {{.*}}<[[WIDTH]] x i1> {{.*}}, <[[WIDTH]] x i8> %pt)
int8 masked_load_varying(uniform int8 *uniform p, int8 pt, bool m) {
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load(p, m, pt);
#else
    return @llvm.masked.load(p, 1l, m, pt);
#endif
}

// uniform bool is broadcast to <N x i1>.
// CHECK-LABEL: @masked_load_uniform___
// CHECK: call <[[WIDTH]] x i8> @llvm.masked.load.v[[WIDTH]]i8.p0(ptr {{.*}}%p, {{.*}}<[[WIDTH]] x i1> {{.*}}, <[[WIDTH]] x i8> %pt)
int8 masked_load_uniform(uniform int8 *uniform p, int8 pt, uniform bool m) {
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load(p, m, pt);
#else
    return @llvm.masked.load(p, 1l, m, pt);
#endif
}

// uniform short vectors are padded to at least 128 bits based on the element
// width of the target, and are passed with the width of the intrinsic.
// CHECK-LABEL: @max_ps___
// CHECK: call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
uniform float<4> max_ps(uniform float<4> a, uniform float<4> b) { return @llvm.x86.sse.max.ps(a, b); }

// The result is padded back with poison lanes.
// CHECK-LABEL: @max_pd___
// CHECK: [[A:%[a-z_0-9]+]] = shufflevector <{{[0-9]+}} x double> %a, <{{[0-9]+}} x double> poison, <2 x i32> <i32 0, i32 1>
// CHECK: [[R:%[a-z_0-9]+]] = tail call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> [[A]], <2 x double> %{{.*}})
// CHECK: shufflevector <2 x double> [[R]], <2 x double> poison, <{{[0-9]+}} x i32> <i32 0, i32 1, i32 poison
uniform double<2> max_pd(uniform double<2> a, uniform double<2> b) { return @llvm.x86.sse2.max.pd(a, b); }

// The types of overloaded intrinsics are deduced from the elements of the
// vector, so the padding lanes are not part of the operation.
// CHECK-LABEL: @reduce_add___
// CHECK: call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %{{.*}})
uniform int reduce_add(uniform int<2> v) { return @llvm.vector.reduce.add(v); }

// CHECK-LABEL: @reverse___
// CHECK: call <3 x i32> @llvm.vector.reverse.v3i32(<3 x i32> %{{.*}})
uniform int<3> reverse(uniform int<3> v) { return @llvm.vector.reverse(v); }

// The same for ISPC intrinsics: the result has 4 elements.
// CHECK-LABEL: @concat___
// CHECK: shufflevector <2 x float> %{{.*}}, <2 x float> %{{.*}}, <{{[0-9]+}} x i32> <i32 0, i32 1, i32 2, i32 3{{(>|,)}}
uniform float<4> concat(uniform float<2> a, uniform float<2> b) { return @llvm.ispc.concat(a, b); }

// uniform bool<N> short vectors are masks of the ISPC intrinsics.
// CHECK-LABEL: @select_short___
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
uniform float<4> select_short(uniform bool<4> m, uniform float<4> a, uniform float<4> b) {
    return @llvm.ispc.select(m, a, b);
}

// CHECK-LABEL: @blend_store_short___
// CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
// CHECK: store <4 x float> %{{.*}}, ptr %p
void blend_store_short(uniform float<4> v, uniform int8 *uniform p, uniform bool<4> m) {
    @llvm.ispc.blend_store(v, p, m);
}

// CHECK-LABEL: @packmask_short___
// CHECK: zext i4 %{{.*}} to i64
uniform int64 packmask_short(uniform bool<4> m) { return @llvm.ispc.packmask(m); }

// Atomic floating-point operations on short vectors, which are expanded to
// loops of cmpxchg. Integer operations on short vectors need LLVM 23, see
// llvm-intrinsics-atomic-vectors.ispc.
// CHECK-LABEL: @atomic_fadd_short___
// CHECK: fadd <4 x float>
// CHECK: cmpxchg ptr %p, i128 %{{.*}}, i128 %{{.*}} seq_cst seq_cst, align 16
uniform float<4> atomic_fadd_short(uniform int8 *uniform p, uniform float<4> a) {
    return @llvm.ispc.atomicrmw.fadd.seq_cst(p, a);
}

// An atomic operation is expanded once also if it's the operand of a no-op
// bitcast, which is lowered to the operand.
// CHECK-LABEL: @atomic_fadd_bitcast___
// CHECK: cmpxchg ptr %p, i128 %{{.*}}, i128 %{{.*}} seq_cst seq_cst, align 16
// CHECK-NOT: cmpxchg
// CHECK-NOT: = atomicrmw
uniform float<4> atomic_fadd_bitcast(uniform int8 *uniform p, uniform float<4> a) {
    return @llvm.ispc.bitcast(@llvm.ispc.atomicrmw.fadd.seq_cst(p, a), a);
}
