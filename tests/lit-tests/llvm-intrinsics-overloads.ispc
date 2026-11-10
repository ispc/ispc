// Check calls of LLVM intrinsics: the overloaded types are deduced from the
// arguments, the type suffix in the name is ignored, and ImmArg parameters
// accept compile-time constant expressions
// (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s

// Matrix intrinsics are lowered only when the module calls them, so the pass
// pipeline doesn't depend on the feature otherwise.
// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O0 --debug-phase=10:10 %s -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-NO-MATRIX
// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O0 --debug-phase=10:10 -DMATRIX %s -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK-MATRIX

// REQUIRES: X86_ENABLED

// CHECK-NO-MATRIX: LLVM IR after phase : {{.*}}VerifierPass
// CHECK-MATRIX: LLVM IR after phase : {{.*}}LowerMatrixIntrinsicsPass

// The suffix doesn't have to match the deduced declaration.
// CHECK-LABEL: @ctpop_i64___
// CHECK: call {{.*}}i64 @llvm.ctpop.i64(i64 %a)
uniform int64 ctpop_i64(uniform int64 a) { return @llvm.ctpop.i32(a); }

// CHECK-LABEL: @ctpop_varying___
// CHECK: call {{.*}}<8 x i32> @llvm.ctpop.v8i32(<8 x i32> %a)
int ctpop_varying(int a) { return @llvm.ctpop.v4i32(a); }

// CHECK-LABEL: @ctpop_unsuffixed___
// CHECK: call {{.*}}<8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
int16 ctpop_unsuffixed(int16 a) { return @llvm.ctpop(a); }

// The overloaded type is deduced from the argument, the return type from it.
// CHECK-LABEL: @vreduce_add___
// CHECK: call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %a)
uniform int vreduce_add(int a) { return @llvm.vector.reduce.add(a); }

// The overloaded type of the return value is the type of the matching argument.
// CHECK-LABEL: @fshl___
// CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %a, <8 x i32> %b, <8 x i32> %c)
int fshl(int a, int b, int c) { return @llvm.fshl(a, b, c); }

// Compile-time constant arguments of ImmArg parameters.
// CHECK-LABEL: @ctlz_literal___
// CHECK: call {{.*}}i32 @llvm.ctlz.i32(i32 %a, i1 false)
uniform int ctlz_literal(uniform int a) { return @llvm.ctlz(a, false); }

// CHECK-LABEL: @ctlz_expr___
// CHECK: call {{.*}}i32 @llvm.ctlz.i32(i32 %a, i1 true)
uniform int ctlz_expr(uniform int a) { return @llvm.ctlz(a, 1 == 1); }

// CHECK-LABEL: @ctlz_const___
// CHECK: call {{.*}}i32 @llvm.ctlz.i32(i32 %a, i1 true)
static const uniform bool isZeroPoison = true;
uniform int ctlz_const(uniform int a) { return @llvm.ctlz(a, isZeroPoison); }

// The constant is converted to the parameter type.
// CHECK-LABEL: @ctlz_int___
// CHECK: call {{.*}}i32 @llvm.ctlz.i32(i32 %a, i1 false)
uniform int ctlz_int(uniform int a) { return @llvm.ctlz(a, 0); }

// InstCombine replaces the call with a shuffle.
// CHECK-LABEL: @insertps___
// CHECK: shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
uniform float<4> insertps(uniform float<4> a, uniform float<4> b) { return @llvm.x86.sse41.insertps(a, b, 16); }

// varying bool is converted to <N x i1>.
// CHECK-LABEL: @masked_load___
// CHECK: call <8 x i8> @llvm.masked.load.v8i8.p0(ptr {{.*}}%p, {{.*}}<8 x i1> {{.*}}, <8 x i8> %pt)
int8 masked_load(uniform int8 *uniform p, int8 pt, bool m) {
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load(p, m, pt);
#else
    return @llvm.masked.load(p, 1l, m, pt);
#endif
}

// CHECK-LABEL: @ctlz_template___
// CHECK: call {{.*}}i64 @llvm.ctlz.i64(i64 %a, i1 true)
template <typename T, bool B> T tctlz(T x) { return @llvm.ctlz(x, B); }
uniform int64 ctlz_template(uniform int64 a) { return tctlz<uniform int64, true>(a); }

// The orderings of llvm.ispc.cmpxchg are <success ordering>.<failure ordering>.
// CHECK-LABEL: @cmpxchg_orderings___
// CHECK: cmpxchg ptr %p, i32 %c, i32 %v acq_rel acquire
uniform int cmpxchg_orderings(uniform int8 *uniform p, uniform int c, uniform int v) {
    return @llvm.ispc.cmpxchg.acq_rel.acquire(p, c, v);
}

#ifdef MATRIX
float transpose(float m) { return @llvm.matrix.transpose(m, 2, 4); }
#endif
