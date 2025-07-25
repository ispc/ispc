;;  Copyright (c) 2020-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

stdlib_core()
scans()
halfTypeGenericImplementation()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stub for mask conversion. LLVM's intrinsics want i1 mask, but we use i8

define i32 @__cast_mask_to_i32 (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i32 = bitcast <WIDTH x i1> %mask to i32
  ret i32 %mask_i32
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %res32 = call i32 @__cast_mask_to_i32 (<WIDTH x MASK> %mask)
  %res = zext i32 %res32 to i64
  ret i64 %res
}

define i1 @__any(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i32 @__cast_mask_to_i32 (<WIDTH x MASK> %mask)
  %res = icmp ne i32 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i32 @__cast_mask_to_i32 (<WIDTH x MASK> %mask)
  %res = icmp eq i32 %intmask, -1
  ret i1 %res
}

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i32 @__cast_mask_to_i32 (<WIDTH x MASK> %mask)
  %res = icmp eq i32 %intmask, 0
  ret i1 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shift/shuffle

; Use vpermw for 16-bit shuffles.
declare <32 x i16> @llvm.x86.avx512.mask.permvar.hi.512(<32 x i16>, <32 x i16>, <32 x i16>, i16)
define <32 x i16> @__shuffle_i16(<32 x i16>, <32 x i32>) nounwind readnone alwaysinline {
  %ind = trunc <32 x i32> %1 to <32 x i16>
  %res = call <32 x i16> @llvm.x86.avx512.mask.permvar.hi.512(<32 x i16> %0, <32 x i16> %ind, <32 x i16> zeroinitializer, i16 -1)
  ret <32 x i16> %res
}

define <32 x i8> @__shuffle_i8(<32 x i8>, <32 x i32>) nounwind readnone alwaysinline {
  %vals = zext <32 x i8> %0 to <32 x i16>
  %res = call <32 x i16> @__shuffle_i16(<32 x i16> %vals, <32 x i32> %1)
  %res_i8 = trunc <32 x i16> %res to <32 x i8>
  ret <32 x i8> %res_i8
}

define <32 x half> @__shuffle_half(<32 x half>, <32 x i32>) nounwind readnone alwaysinline {
  %vals = bitcast <32 x half> %0 to <32 x i16>
  %res = call <32 x i16> @__shuffle_i16(<32 x i16> %vals, <32 x i32> %1)
  %res_half = bitcast <32 x i16> %res to <32 x half>
  ret <32 x half> %res_half
}

declare <16 x i32> @llvm.x86.avx512.vpermi2var.d.512(<16 x i32> %a, <16 x i32> %idx, <16 x i32> %b)
define <32 x i32> @__shuffle_i32(<32 x i32> %input, <32 x i32> %perm) {
    ; Split input into two 512-bit halves (16 x i32 each)
    v32tov16(i32, %input, %low, %high)
    v32tov16(i32, %perm, %perm_low, %perm_high)

    ; Two 512-bit VPERMI2D operations
    %result1 = call <16 x i32> @llvm.x86.avx512.vpermi2var.d.512(<16 x i32> %low, <16 x i32> %perm_low, <16 x i32> %high)
    %result2 = call <16 x i32> @llvm.x86.avx512.vpermi2var.d.512(<16 x i32> %low, <16 x i32> %perm_high, <16 x i32> %high)

    ; Concatenate results
    v16tov32(i32, %result1, %result2, %final)
    ret <32 x i32> %final
}

declare <16 x float> @llvm.x86.avx512.vpermi2var.ps.512(<16 x float> %a, <16 x i32> %idx, <16 x float> %b)
define <32 x float> @__shuffle_float(<32 x float> %input, <32 x i32> %perm) {
    ; Split input into two 512-bit halves (16 x float each)
    v32tov16(float, %input, %low, %high)
    v32tov16(i32, %perm, %perm_low, %perm_high)

    ; Two 512-bit VPERMI2PS operations
    %result1 = call <16 x float> @llvm.x86.avx512.vpermi2var.ps.512(<16 x float> %low, <16 x i32> %perm_low, <16 x float> %high)
    %result2 = call <16 x float> @llvm.x86.avx512.vpermi2var.ps.512(<16 x float> %low, <16 x i32> %perm_high, <16 x float> %high)

    ; Concatenate results
    v16tov32(float, %result1, %result2, %final)
    ret <32 x float> %final
}

declare <32 x i64> @llvm.masked.gather.v32i64.v32p0(<32 x i64*>, i32, <32 x i1>, <32 x i64>)
define <32 x i64> @__shuffle_i64(<32 x i64> %input, <32 x i32> %perm) nounwind readnone alwaysinline {
  ; Store input vector to memory so we can gather from it
  %input_alloca = alloca [32 x i64], align 64
  %input_vec_ptr = bitcast [32 x i64]* %input_alloca to <32 x i64>*
  store <32 x i64> %input, <32 x i64>* %input_vec_ptr, align 64

  ; Create base pointer vector for gather operation
  %base_ptr_scalar = bitcast [32 x i64]* %input_alloca to i64*
  %base_ptr_vec = insertelement <32 x i64*> undef, i64* %base_ptr_scalar, i32 0
  %base_ptr_broadcast = shufflevector <32 x i64*> %base_ptr_vec, <32 x i64*> zeroinitializer, <32 x i32> zeroinitializer

  ; Create pointer vector
  %perm_i64 = sext <32 x i32> %perm to <32 x i64>
  %ptrs = getelementptr i64, <32 x i64*> %base_ptr_broadcast, <32 x i64> %perm_i64

  ; Create mask for gather (all true)
  %true_val = insertelement <32 x i1> undef, i1 true, i32 0
  %mask_all = shufflevector <32 x i1> %true_val, <32 x i1> zeroinitializer, <32 x i32> zeroinitializer

  ; Perform the single gather operation
  %result = call <32 x i64> @llvm.masked.gather.v32i64.v32p0(<32 x i64*> %ptrs, i32 8, <32 x i1> %mask_all, <32 x i64> zeroinitializer)
  ret <32 x i64> %result
}

define <32 x double> @__shuffle_double(<32 x double> %input, <32 x i32> %perm) nounwind readnone alwaysinline {
  %input_i64 = bitcast <32 x double> %input to <32 x i64>
  %res_i64 = call <32 x i64> @__shuffle_i64(<32 x i64> %input_i64, <32 x i32> %perm)
  %res_double = bitcast <32 x i64> %res_i64 to <32 x double>
  ret <32 x double> %res_double
}

define_shuffle2_const()

declare <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16>, <32 x i16>, <32 x i16>)
define <32 x i16> @__shuffle2_i16(<32 x i16> %v1, <32 x i16> %v2, <32 x i32> %perm) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<32 x i32> %perm)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <32 x i16> @__shuffle2_const_i16(<32 x i16> %v1, <32 x i16> %v2, <32 x i32> %perm)
  ret <32 x i16> %res_const

not_const:
  %perm16 = trunc <32 x i32> %perm to <32 x i16>
  %result = call <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16> %v1, <32 x i16> %perm16, <32 x i16> %v2)
  ret <32 x i16> %result
}

define <32 x half> @__shuffle2_half(<32 x half> %v1, <32 x half> %v2, <32 x i32> %perm) nounwind readnone alwaysinline {
  %v1_i16 = bitcast <32 x half> %v1 to <32 x i16>
  %v2_i16 = bitcast <32 x half> %v2 to <32 x i16>
  %res_i16 = call <32 x i16> @__shuffle2_i16(<32 x i16> %v1_i16, <32 x i16> %v2_i16, <32 x i32> %perm)
  %res_half = bitcast <32 x i16> %res_i16 to <32 x half>
  ret <32 x half> %res_half
}

define <32 x i8> @__shuffle2_i8(<32 x i8> %v1, <32 x i8> %v2, <32 x i32> %perm) nounwind readnone alwaysinline {
  %v1_i16 = zext <32 x i8> %v1 to <32 x i16>
  %v2_i16 = zext <32 x i8> %v2 to <32 x i16>
  %res_i16 = call <32 x i16> @__shuffle2_i16(<32 x i16> %v1_i16, <32 x i16> %v2_i16, <32 x i32> %perm)
  %res_i8 = trunc <32 x i16> %res_i16 to <32 x i8>
  ret <32 x i8> %res_i8
}

shuffle2(float)
shuffle2(i32)
shuffle2(double)
shuffle2(i64)

define_vector_permutations()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

;; same as avx2 and avx512skx
;; TODO: hoist to some utility file?

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %vv)
  %r = extractelement <8 x float> %rv, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  ; round to nearest even
  %rv = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv, i32 0)
  %r = extractelement <8 x i16> %rv, i32 0
  ret i16 %r
}

declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %source, <16 x float> %write_through, i16 %mask, i32) nounwind readonly
declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %source, i32, <16 x i16> %write_through, i16 %mask) nounwind readonly

define <32 x float> @__half_to_float_varying(<32 x i16> %v) nounwind readnone alwaysinline {
  v32tov16(i16, %v, %v0, %v1)
  %r0 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v0, <16 x float> undef, i16 -1, i32 4)
  %r1 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v1, <16 x float> undef, i16 -1, i32 4)
  v16tov32(float, %r0, %r1, %r)
  ret <32 x float> %r
}

define <32 x i16> @__float_to_half_varying(<32 x float> %v) nounwind readnone alwaysinline {
  v32tov16(float, %v, %v0, %v1)
  %r0 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v0, i32 0, <16 x i16> undef, i16 -1)
  %r1 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v1, i32 0, <16 x i16> undef, i16 -1)
  v16tov32(i16, %r0, %r1, %r)
  ret <32 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode
fastMathFTZDAZ_x86()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; round/floor/ceil varying float/doubles

declare <16 x float> @llvm.roundeven.v16f32(<16 x float> %p)
declare <16 x float> @llvm.floor.v16f32(<16 x float> %p)
declare <16 x float> @llvm.ceil.v16f32(<16 x float> %p)

define <32 x float> @__round_varying_float(<32 x float> %v) nounwind readonly alwaysinline {
  v32tov16(float, %v, %v0, %v1)
  %r0 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v1)
  v16tov32(float, %r0, %r1, %r)
  ret <32 x float> %r
}

define <32 x float> @__floor_varying_float(<32 x float> %v) nounwind readonly alwaysinline {
  v32tov16(float, %v, %v0, %v1)
  %r0 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v1)
  v16tov32(float, %r0, %r1, %r)
  ret <32 x float> %r
}

define <32 x float> @__ceil_varying_float(<32 x float> %v) nounwind readonly alwaysinline {
  v32tov16(float, %v, %v0, %v1)
  %r0 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v1)
  v16tov32(float, %r0, %r1, %r)
  ret <32 x float> %r
}

declare <8 x double> @llvm.roundeven.v8f64(<8 x double> %p)
declare <8 x double> @llvm.floor.v8f64(<8 x double> %p)
declare <8 x double> @llvm.ceil.v8f64(<8 x double> %p)

define <32 x double> @__round_varying_double(<32 x double> %v) nounwind readonly alwaysinline {
  v32tov8(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v3)
  v8tov32(double, %r0, %r1, %r2, %r3, %r)
  ret <32 x double> %r
}

define <32 x double> @__floor_varying_double(<32 x double> %v) nounwind readonly alwaysinline {
  v32tov8(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v3)
  v8tov32(double, %r0, %r1, %r2, %r3, %r)
  ret <32 x double> %r
}

define <32 x double> @__ceil_varying_double(<32 x double> %v) nounwind readonly alwaysinline {
  v32tov8(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v3)
  v8tov32(double, %r0, %r1, %r2, %r3, %r)
  ret <32 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;; min/max
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

declare i64 @__max_uniform_int64(i64, i64) nounwind readonly alwaysinline
declare i64 @__max_uniform_uint64(i64, i64) nounwind readonly alwaysinline
declare i64 @__min_uniform_int64(i64, i64) nounwind readonly alwaysinline
declare i64 @__min_uniform_uint64(i64, i64) nounwind readonly alwaysinline
declare i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline
declare i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline
declare i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline
declare i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline

;; TODO: these are from neon-common, need to make all of them standard utils.
;; TODO: ogt vs ugt?
define float @__max_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__min_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ult float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp olt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp olt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

define <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp ogt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

;; int32/uint32/float versions
define <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %m = fcmp olt <WIDTH x float> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x float> %0, <WIDTH x float> %1
  ret <WIDTH x float> %r
}

define <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %m = fcmp ogt <WIDTH x float> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x float> %0, <WIDTH x float> %1
  ret <WIDTH x float> %r
}

;; sqrt/rsqrt/rcp
;; TODO: need to use intrinsics and N-R approximation.

;; bit ops

popcnt()
ctlztz()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

;; 8 bit
declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<32 x i8>) nounwind readnone alwaysinline {
  %rv = call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %0,
                                                    <32 x i8> zeroinitializer)
  %r0 = extractelement <4 x i64> %rv, i32 0
  %r1 = extractelement <4 x i64> %rv, i32 1
  %r2 = extractelement <4 x i64> %rv, i32 2
  %r3 = extractelement <4 x i64> %rv, i32 3
  %r01 = add i64 %r0, %r1
  %r23 = add i64 %r2, %r3
  %r = add i64 %r01, %r23
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

;; 16 bit
;; TODO: why returning i16?
define internal <32 x i16> @__add_varying_i16(<32 x i16>,
                                              <32 x i16>) nounwind readnone alwaysinline {
  %r = add <32 x i16> %0, %1
  ret <32 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<32 x i16>) nounwind readnone alwaysinline {
  reduce32(i16, @__add_varying_i16, @__add_uniform_i16)
}

;; 32 bit
;; TODO: why returning i32?
define internal <32 x i32> @__add_varying_int32(<32 x i32>,
                                                <32 x i32>) nounwind readnone alwaysinline {
  %s = add <32 x i32> %0, %1
  ret <32 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<32 x i32>) nounwind readnone alwaysinline {
  reduce32(i32, @__add_varying_int32, @__add_uniform_int32)
}

;; float
;; TODO: __reduce_add_float may use hadd
define internal <32 x float> @__add_varying_float(<32 x float>,
                                                  <32 x float>) nounwind readnone alwaysinline {
  %s = fadd <32 x float> %0, %1
  ret <32 x float> %s
}

define internal float @__add_uniform_float(float, float) nounwind readnone alwaysinline {
  %s = fadd float %0, %1
  ret float %s
}

define float @__reduce_add_float(<32 x float>) nounwind readonly alwaysinline {
  reduce32(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<32 x float>) nounwind readnone alwaysinline {
  reduce32(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<32 x float>) nounwind readnone alwaysinline {
  reduce32(float, @__max_varying_float, @__max_uniform_float)
}

;; 32 bit min/max
define i32 @__reduce_min_int32(<32 x i32>) nounwind readnone alwaysinline {
  reduce32(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<32 x i32>) nounwind readnone alwaysinline {
  reduce32(i32, @__max_varying_int32, @__max_uniform_int32)
}


define i32 @__reduce_min_uint32(<32 x i32>) nounwind readnone alwaysinline {
  reduce32(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<32 x i32>) nounwind readnone alwaysinline {
  reduce32(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;; double

define internal <32 x double> @__add_varying_double(<32 x double>,
                                                   <32 x double>) nounwind readnone alwaysinline {
  %s = fadd <32 x double> %0, %1
  ret <32 x double> %s
}

define internal double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %s = fadd double %0, %1
  ret double %s
}

define double @__reduce_add_double(<32 x double>) nounwind readonly alwaysinline {
  reduce32(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<32 x double>) nounwind readnone alwaysinline {
  reduce32(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<32 x double>) nounwind readnone alwaysinline {
  reduce32(double, @__max_varying_double, @__max_uniform_double)
}

;; int64

define internal <32 x i64> @__add_varying_int64(<32 x i64>,
                                                <32 x i64>) nounwind readnone alwaysinline {
  %s = add <32 x i64> %0, %1
  ret <32 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<32 x i64>) nounwind readnone alwaysinline {
  reduce32(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<32 x i64>) nounwind readnone alwaysinline {
  reduce32(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<32 x i64>) nounwind readnone alwaysinline {
  reduce32(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<32 x i64>) nounwind readnone alwaysinline {
  reduce32(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<32 x i64>) nounwind readnone alwaysinline {
  reduce32(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

declare void @__masked_store_blend_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, <WIDTH x i1>) nounwind alwaysinline
declare void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, <WIDTH x i1>) nounwind alwaysinline
declare void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, <WIDTH x i1>) nounwind alwaysinline
declare void @__masked_store_blend_i64(<WIDTH x i64>* nocapture, <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather(i8)
gen_gather(i16)
gen_gather(half)
gen_gather(i32)
gen_gather(float)
gen_gather(i64)
gen_gather(double)

define(`scatterbo32_64', `
define void @__scatter_base_offsets32_$1(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x MASK> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_$1(i8* %ptr, <WIDTH x i32> %offsets,
      i32 %scale, <WIDTH x i32> zeroinitializer, <WIDTH x $1> %vals, <WIDTH x MASK> %mask)
  ret void
}

define void @__scatter_base_offsets64_$1(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x MASK> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_$1(i8* %ptr, <WIDTH x i64> %offsets,
      i32 %scale, <WIDTH x i64> zeroinitializer, <WIDTH x $1> %vals, <WIDTH x MASK> %mask)
  ret void
}
')

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

scatterbo32_64(i8)
scatterbo32_64(i16)
scatterbo32_64(half)
scatterbo32_64(i32)
scatterbo32_64(float)
scatterbo32_64(i64)
scatterbo32_64(double)

;; TODO better intrinsic implementation is available
packed_load_and_store(FALSE)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

;; TODO: need to defined with intrinsics.
define_prefetches()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

saturation_arithmetic_novec()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product 
