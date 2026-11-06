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
// The half-width vector of the result can't be derived from a scalar.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:47: Error: Type "uniform float" of argument 1 is not valid for LLVM intrinsic "llvm.vector.deinterleave2".
void deinterleave_scalar(uniform float val) { @llvm.vector.deinterleave2(val); }

// The architecture check reports the name as written in the source.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:26: Error: LLVM intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8" supported only on "aarch64" target architecture.
int sdot(int a) { return @llvm.aarch64.neon.sdot.v4i32.v16i8(a, a, a); }
// Intrinsics of targets that ispc doesn't compile for.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:21: Error: LLVM intrinsic "llvm.nvvm.read.ptx.sreg.tid.x" is not supported on "x86-64" target architecture.
int nvvm() { return @llvm.nvvm.read.ptx.sreg.tid.x(); }

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
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:45: Error: Argument 2 of ISPC intrinsic "llvm.ispc.extract" must be a uniform integer value, "const varying int32" provided.
uniform int extract_varying(int v) { return @llvm.ispc.extract(v, programIndex); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:34: Error: Argument 3 of ISPC intrinsic "llvm.ispc.insert" must have the LLVM type of the elements of argument 1 ("i32"), "const uniform float" provided.
int insert_float(int v) { return @llvm.ispc.insert(v, 0, 1.0f); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:50: Error: Argument 1 of ISPC intrinsic "llvm.ispc.select" must be a bool value or a vector of bool values, "varying float" provided.
int select_float(float c, int a, int b) { return @llvm.ispc.select(c, a, b); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:50: Error: Argument 1 of ISPC intrinsic "llvm.ispc.select" must be a bool value or a vector of bool values, "varying int64" provided.
int select_int64(int64 c, int a, int b) { return @llvm.ispc.select(c, a, b); }
// Integer values are not masks even if they have the LLVM type of the mask of
// the target, as varying int32 has on avx2-i32x8.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:46: Error: Argument 1 of ISPC intrinsic "llvm.ispc.select" must be a bool value or a vector of bool values, "varying int32" provided.
int select_int(int c, int a, int b) { return @llvm.ispc.select(c, a, b); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:44: Error: Argument 1 of ISPC intrinsic "llvm.ispc.packmask" must be a varying bool value or a vector of bool values, "varying int32" provided.
uniform int64 packmask_int(int c) { return @llvm.ispc.packmask(c); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:57: Error: Argument 3 of ISPC intrinsic "llvm.ispc.blend_store" must be a bool value or a vector of bool values, "varying int32" provided.
void blend_int(int v, uniform int8 *uniform p, int m) { @llvm.ispc.blend_store(v, p, m); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:73: Error: Arguments 1 and 2 of ISPC intrinsic "llvm.ispc.select" must have the same number of elements, "varying bool" and "uniform int32" provided.
uniform int select_width(bool c, uniform int a, uniform int b) { return @llvm.ispc.select(c, a, b); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:48: Error: Argument 1 of ISPC intrinsic "llvm.ispc.packmask" must be a varying bool value or a vector of bool values, "varying float" provided.
uniform int64 packmask_float(float v) { return @llvm.ispc.packmask(v); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:61: Error: Argument 3 of ISPC intrinsic "llvm.ispc.blend_store" must be a bool value or a vector of bool values, "varying float" provided.
void blend_float(int v, uniform int8 *uniform p, float m) { @llvm.ispc.blend_store(v, p, m); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:59: Error: Argument 2 of ISPC intrinsic "llvm.ispc.atomicrmw.add.seq_cst" must be a uniform integer value{{.*}}, "const uniform float" provided.
uniform float add_float(uniform int8 *uniform p) { return @llvm.ispc.atomicrmw.add.seq_cst(p, 1.0f); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:56: Error: Argument 2 of ISPC intrinsic "llvm.ispc.atomicrmw.fadd.seq_cst" must be a uniform floating-point value or a uniform short vector of floating-point values, "const uniform int32" provided.
uniform int fadd_int(uniform int8 *uniform p) { return @llvm.ispc.atomicrmw.fadd.seq_cst(p, 1); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:74: Error: Argument 2 of ISPC intrinsic "llvm.ispc.atomicrmw.xchg.seq_cst" must be a uniform integer, floating-point or pointer value, "uniform bool" provided.
uniform bool xchg_bool(uniform int8 *uniform p, uniform bool b) { return @llvm.ispc.atomicrmw.xchg.seq_cst(p, b); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:63: Error: Argument 2 of ISPC intrinsic "llvm.ispc.cmpxchg.seq_cst.seq_cst" must be a uniform integer or pointer value, "const uniform float" provided.
uniform float cmpxchg_float(uniform int8 *uniform p) { return @llvm.ispc.cmpxchg.seq_cst.seq_cst(p, 1.0f, 2.0f); }
// The size of an atomic operand must be a power of two (96 bits here).
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:77: Error: Argument 2 of ISPC intrinsic "llvm.ispc.atomicrmw.add.seq_cst" must be a uniform integer value{{.*}}, "uniform int32<3>" provided.
uniform int<3> add_int3(uniform int8 *uniform p, uniform int<3> a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:80: Error: Argument 2 of ISPC intrinsic "llvm.ispc.atomicrmw.xchg.seq_cst" must be a uniform integer, floating-point or pointer value, "uniform int32<4>" provided.
uniform int<4> xchg_vector(uniform int8 *uniform p, uniform int<4> a) { return @llvm.ispc.atomicrmw.xchg.seq_cst(p, a); }

// The memory orderings must be valid for the instruction.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:20: Error: Invalid ISPC intrinsic name "llvm.ispc.fence.foo", expected "llvm.ispc.fence.<ordering>" where <ordering> is acquire, release, acq_rel or seq_cst.
void bad_fence() { @llvm.ispc.fence.foo(); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:26: Error: Invalid ISPC intrinsic name "llvm.ispc.fence.monotonic", expected
void fence_monotonic() { @llvm.ispc.fence.monotonic(); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:61: Error: Invalid ISPC intrinsic name "llvm.ispc.atomicrmw.add.unordered", expected "llvm.ispc.atomicrmw.<operation>.<ordering>" where <operation> is {{.*}}, <ordering> is monotonic, acquire, release, acq_rel or seq_cst.
uniform int add_unordered(uniform int8 *uniform p) { return @llvm.ispc.atomicrmw.add.unordered(p, 1); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:58: Error: Invalid ISPC intrinsic name "llvm.ispc.atomicrmw.foo.seq_cst", expected "llvm.ispc.atomicrmw.<operation>.<ordering>" where <operation> is xchg, add, sub, and, nand, or, xor, max, min, umax, umin, fadd, fsub, fmax or fmin, <ordering> is monotonic, acquire, release, acq_rel or seq_cst.
uniform int atomic_foo(uniform int8 *uniform p) { return @llvm.ispc.atomicrmw.foo.seq_cst(p, 1); }
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:63: Error: Invalid ISPC intrinsic name "llvm.ispc.cmpxchg.seq_cst.acq_rel", expected "llvm.ispc.cmpxchg.<success ordering>.<failure ordering>" where <success ordering> is monotonic, acquire, release, acq_rel or seq_cst, <failure ordering> is monotonic, acquire or seq_cst.
uniform int cmpxchg_failure(uniform int8 *uniform p) { return @llvm.ispc.cmpxchg.seq_cst.acq_rel(p, 1, 2); }

// Errors are reported when the template is instantiated.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:46: Error: Arguments of types (uniform float) are not valid for LLVM intrinsic "llvm.ctpop".
template <typename T> T tctpop(T x) { return @llvm.ctpop(x); }
uniform float use_tctpop(uniform float x) { return tctpop<uniform float>(x); }

// The error in the returned expression doesn't hide the other errors of the
// return statement, which are reported when the code is generated.
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+2]]:58: Error: Arguments of types (const uniform float) are not valid for LLVM intrinsic "llvm.ctpop".
// CHECK: llvm-intrinsics-errors.ispc:[[@LINE+1]]:51: Error: "return" statement is illegal inside a "foreach" loop.
int return_in_foreach() { foreach (i = 0 ... 8) { return @llvm.ctpop(1.0f); } return 0; }
#else
// Arguments of ImmArg parameters must be compile-time constants.
// CHECK-IMMARG: llvm-intrinsics-errors.ispc:[[@LINE+1]]:80: Error: Argument 2 of LLVM intrinsic "llvm.ctlz.i32" must be a compile-time constant.
uniform int ctlz_var(uniform int a, uniform bool b) { return @llvm.ctlz.i32(a, b); }
// The argument is checked when the template is instantiated.
// CHECK-IMMARG: llvm-intrinsics-errors.ispc:[[@LINE+1]]:91: Error: Argument 2 of LLVM intrinsic "llvm.ctlz" must be a compile-time constant.
template <bool B> uniform int tctlz(uniform int x, uniform bool b) { return @llvm.ctlz(x, B && b); }
uniform int use_tctlz(uniform int x, uniform bool b) { return tctlz<true>(x, b); }
#endif
