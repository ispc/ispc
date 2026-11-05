// Check that malformed calls of LLVM and ISPC intrinsics are reported as errors
// at the call position, using the name as written in the source, and that
// they are reported only once (https://github.com/ispc/ispc/issues/4004).

// RUN: not %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -o %t.o %s 2>&1 | FileCheck %s --implicit-check-not=Error --implicit-check-not=FATAL
// RUN: not %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -DIMMARG -o %t.o %s 2>&1 | FileCheck %s --check-prefix=CHECK-IMMARG --implicit-check-not=Error --implicit-check-not=FATAL

// REQUIRES: X86_ENABLED

#ifndef IMMARG
// The type doesn't match the overloaded type of the intrinsic.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:51: Error: Arguments of types (uniform float) are not valid for LLVM intrinsic "llvm.ctpop.i32".
uniform int ctpop_float(uniform float a) { return @llvm.ctpop.i32(a); }

// LLVM can't derive the element type from a scalar.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:53: Error: Type "uniform bool" of argument 1 is not valid for LLVM intrinsic "llvm.vector.reduce.or.v8i1".
uniform bool reduce_scalar(uniform bool a) { return @llvm.vector.reduce.or.v8i1(a); }

// Wrong number of arguments.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:39: Error: LLVM intrinsic "llvm.fabs.f32" expects 1 argument, 0 provided.
uniform float fabs_no_args() { return @llvm.fabs.f32(); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:52: Error: LLVM intrinsic "llvm.ctpop" expects 1 argument, 2 provided.
uniform int ctpop_two_args(uniform int a) { return @llvm.ctpop(a, a); }

// Struct, metadata and token types are not representable in ISPC.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:46: Error: Return type "{ i32, i1 }" of LLVM intrinsic "llvm.sadd.with.overflow.i32" is not representable as an ISPC type.
uniform int sadd_ovf(uniform int a) { return @llvm.sadd.with.overflow.i32(a, a); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:53: Error: Type "metadata" of parameter 3 of LLVM intrinsic "llvm.experimental.constrained.fadd" is not representable as an ISPC type.
uniform float constrained(uniform float a) { return @llvm.experimental.constrained.fadd(a, a, a, a); }

// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:43: Error: Return type "{ <2 x float>, <2 x float> }" of LLVM intrinsic "llvm.vector.deinterleave2" is not representable as an ISPC type.
void deinterleave(uniform float<4> val) { @llvm.vector.deinterleave2(val); }
// The overloaded type is not the type of any argument.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:39: Error: Can't deduce the overloaded types of LLVM intrinsic "llvm.vector.interleave2" from its arguments.
int interleave(int a, int b) { return @llvm.vector.interleave2(a, b); }

// The architecture check reports the name as written in the source.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:26: Error: LLVM intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8" supported only on "aarch64" target architecture.
int sdot(int a) { return @llvm.aarch64.neon.sdot.v4i32.v16i8(a, a, a); }

// ISPC intrinsics.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:51: Error: ISPC intrinsic "llvm.ispc.bitcast" expects 2 arguments, 1 provided.
uniform int bitcast_one(uniform float a) { return @llvm.ispc.bitcast(a); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:54: Error: Arguments of ISPC intrinsic "llvm.ispc.bitcast" must be values of the same size in bits, "uniform int64" and "const uniform float" provided.
uniform float bitcast_size(uniform int64 a) { return @llvm.ispc.bitcast(a, 0.f); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:41: Error: ISPC intrinsic "llvm.ispc.extract" expects 2 or 3 arguments, 1 provided.
uniform int extract_one(int a) { return @llvm.ispc.extract(a); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:52: Error: Argument 1 of ISPC intrinsic "llvm.ispc.extract" must be a varying or short vector value, "uniform int32" provided.
uniform int extract_scalar(uniform int a) { return @llvm.ispc.extract(a, 0); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:40: Error: ISPC intrinsic "llvm.ispc.select" expects 3 arguments, 2 provided.
int select_two(bool c, int a) { return @llvm.ispc.select(c, a); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:20: Error: Invalid ISPC intrinsic name "llvm.ispc.fence.foo", expected "llvm.ispc.fence.<ordering>".
void bad_fence() { @llvm.ispc.fence.foo(); }

// Errors are reported when the template is instantiated.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:46: Error: Arguments of types (uniform float) are not valid for LLVM intrinsic "llvm.ctpop".
template <typename T> T tctpop(T x) { return @llvm.ctpop(x); }
uniform float use_tctpop(uniform float x) { return tctpop<uniform float>(x); }
#else
// Arguments of ImmArg parameters must be compile-time constants.
// CHECK-IMMARG: llvm-intrinsics-errors.ispc:[[@LINE+1]]:80: Error: Argument 2 of LLVM intrinsic "llvm.ctlz.i32" must be a compile-time constant.
uniform int ctlz_var(uniform int a, uniform bool b) { return @llvm.ctlz.i32(a, b); }
#endif
