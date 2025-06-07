;;  Copyright (c) 2016-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`ISA',`AVX512SKX')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`target-avx512-utils.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shuffles

;; Implementation for i8 and i16 types is different across avx512 targets.
;; There is no @llvm.x86.avx512.mask.permvar.qi.128 and @llvm.x86.avx512.vpermi2var.qi.128
;; before icl. 
;; @llvm.x86.avx512.vpermi2var.hi.256 is not available for KNL.
;; Look for definitions in particular target files.

declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)
define <16 x i8> @__shuffle_i8(<16 x i8> %data, <16 x i32> %shuffle_mask) nounwind readnone alwaysinline {
  %mask = trunc <16 x i32> %shuffle_mask to <16 x i8>
  %result = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %data, <16 x i8> %mask)
  ret <16 x i8> %result
}

declare <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16>, <16 x i16>, <16 x i16>, i16)
define <16 x i16> @__shuffle_i16(<16 x i16>, <16 x i32>) nounwind readnone alwaysinline {
  %ind = trunc <16 x i32> %1 to <16 x i16>
  %res = call <16 x i16> @llvm.x86.avx512.mask.permvar.hi.256(<16 x i16> %0, <16 x i16> %ind, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

define <16 x half> @__shuffle_half(<16 x half> %v, <16 x i32> %perm) nounwind readnone alwaysinline {
  %vals = bitcast <16 x half> %v to <16 x i16>
  %res = call <16 x i16> @__shuffle_i16(<16 x i16> %vals, <16 x i32> %perm)
  %res_half = bitcast <16 x i16> %res to <16 x half>
  ret <16 x half> %res_half
}

declare <WIDTH x i32> @llvm.x86.avx512.permvar.si.512(<WIDTH x i32>, <WIDTH x i32>)
define <WIDTH x i32> @__shuffle_i32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %res = call <WIDTH x i32>@llvm.x86.avx512.permvar.si.512(<WIDTH x i32> %0, <WIDTH x i32> %1)
  ret <WIDTH x i32> %res
}

declare <WIDTH x float> @llvm.x86.avx512.permvar.sf.512(<WIDTH x float>, <WIDTH x i32>)
define <WIDTH x float> @__shuffle_float(<WIDTH x float>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %res = call <WIDTH x float> @llvm.x86.avx512.permvar.sf.512(<WIDTH x float> %0, <WIDTH x i32> %1)
  ret <WIDTH x float> %res
}

declare <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %a, <8 x i64> %idx, <8 x double> %b, i8 %mask)
define <16 x double> @__shuffle_double(<16 x double> %input, <16 x i32> %perm) {
    ; Split input into two 512-bit halves
    v16tov8(double, %input, %low, %high)
    v16tov8(i32, %perm, %perm_low, %perm_high)

    ; Two 512-bit VPERMI2PD operations
    %indices1 = zext <8 x i32> %perm_low to <8 x i64>
    %indices2 = zext <8 x i32> %perm_high to <8 x i64>
    %result1 = call <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %low, <8 x i64> %indices1, <8 x double> %high, i8 255)
    %result2 = call <8 x double> @llvm.x86.avx512.mask.vpermi2var.pd.512(<8 x double> %low, <8 x i64> %indices2, <8 x double> %high, i8 255)

    ; Concatenate results
    v8tov16(double, %result1, %result2, %final)
    ret <16 x double> %final
}

declare <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64> %a, <8 x i64> %idx, <8 x i64> %b, i8 %mask)
define <16 x i64> @__shuffle_i64(<16 x i64> %input, <16 x i32> %perm) {
    ; Split input into two 512-bit halves
    v16tov8(i64, %input, %low, %high)
    v16tov8(i32, %perm, %perm_low, %perm_high)

    ; Two 512-bit VPERMI2Q operations (Q = quadword = 64-bit integers)
    %indices1 = zext <8 x i32> %perm_low to <8 x i64>
    %indices2 = zext <8 x i32> %perm_high to <8 x i64>
    %result1 = call <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64> %low, <8 x i64> %indices1, <8 x i64> %high, i8 255)
    %result2 = call <8 x i64> @llvm.x86.avx512.mask.vpermi2var.q.512(<8 x i64> %low, <8 x i64> %indices2, <8 x i64> %high, i8 255)

    ; Concatenate results
    v8tov16(i64, %result1, %result2, %final)
    ret <16 x i64> %final
}

define_shuffle2_const()

shuffle2(i8)
shuffle2(half)
shuffle2(i64)
shuffle2(double)

declare <WIDTH x i32> @llvm.x86.avx512.vpermi2var.d.512(<WIDTH x i32>, <WIDTH x i32>, <WIDTH x i32>)
define <WIDTH x i32> @__shuffle2_i32(<WIDTH x i32>, <WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x i32> @__shuffle2_const_i32(<WIDTH x i32> %0, <WIDTH x i32> %1, <WIDTH x i32> %2)
  ret <WIDTH x i32> %res_const

not_const:
  %res = call <WIDTH x i32> @llvm.x86.avx512.vpermi2var.d.512(<WIDTH x i32> %0, <WIDTH x i32> %2, <WIDTH x i32> %1)
  ret <WIDTH x i32> %res
}

declare <WIDTH x float> @llvm.x86.avx512.vpermi2var.ps.512(<WIDTH x float>, <WIDTH x i32>, <WIDTH x float>)
define <WIDTH x float> @__shuffle2_float(<WIDTH x float>, <WIDTH x float>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x float> @__shuffle2_const_float(<WIDTH x float> %0, <WIDTH x float> %1, <WIDTH x i32> %2)
  ret <WIDTH x float> %res_const

not_const:
  %res = call <WIDTH x float> @llvm.x86.avx512.vpermi2var.ps.512(<WIDTH x float> %0, <WIDTH x i32> %2, <WIDTH x float> %1)
  ret <WIDTH x float> %res
}


declare <WIDTH x i16> @llvm.x86.avx512.vpermi2var.hi.256(<WIDTH x i16>, <WIDTH x i16>, <WIDTH x i16>)
define <WIDTH x i16> @__shuffle2_i16(<WIDTH x i16>, <WIDTH x i16>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x i16> @__shuffle2_const_i16(<WIDTH x i16> %0, <WIDTH x i16> %1, <WIDTH x i32> %2)
  ret <WIDTH x i16> %res_const

not_const:
  %ind = trunc <WIDTH x i32> %2 to <WIDTH x i16>
  %res = call <WIDTH x i16> @llvm.x86.avx512.vpermi2var.hi.256(<WIDTH x i16> %0, <WIDTH x i16> %ind, <WIDTH x i16> %1)
  ret <WIDTH x i16> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stub for mask conversion. LLVM's intrinsics want i1 mask, but we use i8

define i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i16 = bitcast <WIDTH x i1> %mask to i16
  ret i16 %mask_i16
}

define i8 @__extract_mask_low (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %mask_low = trunc i16 %mask_i16 to i8 
  ret i8 %mask_low
}

define i8 @__extract_mask_hi (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %mask_shifted = lshr i16 %mask_i16, 8
  %mask_hi = trunc i16 %mask_shifted to i8
  ret i8 %mask_hi
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone {
  %r_0 = shufflevector <16 x i16> %v, <16 x i16> undef,
             <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vr_0 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %r_0)
  %r_1 = shufflevector <16 x i16> %v, <16 x i16> undef,
             <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vr_1 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %r_1)
  %r = shufflevector <8 x float> %vr_0, <8 x float> %vr_1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone {
  %r_0 = shufflevector <16 x float> %v, <16 x float> undef,
             <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vr_0 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %r_0, i32 0)
  %r_1 = shufflevector <16 x float> %v, <16 x float> undef,
             <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vr_1 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %r_1, i32 0)
  %r = shufflevector <8 x i16> %vr_0, <8 x i16> %vr_1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %r
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <16 x float> @llvm.roundeven.v16f32(<16 x float> %p)
declare <16 x float> @llvm.floor.v16f32(<16 x float> %p)
declare <16 x float> @llvm.ceil.v16f32(<16 x float> %p)

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.floor.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.ceil.v16f32(<16 x float> %0)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <8 x double> @llvm.roundeven.v8f64(<8 x double> %p)
declare <8 x double> @llvm.floor.v8f64(<8 x double> %p)
declare <8 x double> @llvm.ceil.v8f64(<8 x double> %p)

define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v1)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64/uint64 min/max

declare <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <16 x i64> @__max_varying_int64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__max_varying_uint64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__min_varying_int64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64> @__min_varying_uint64(<16 x i64>, <16 x i64>) nounwind readonly alwaysinline {
  %v0_lo = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0_hi = shufflevector <16 x i64> %0, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1_lo = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1_hi = shufflevector <16 x i64> %1, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %v0_lo, <8 x i64> %v1_lo, <8 x i64>zeroinitializer, i8 -1)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %v0_hi, <8 x i64> %v1_hi, <8 x i64>zeroinitializer, i8 -1)
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)

define <16 x float> @__max_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %0, <16 x float> %1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

define <16 x float> @__min_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %0, <16 x float> %1, <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32> %0, <16 x i32> %1, 
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32> %0, <16 x i32> %1,
                                                           <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  %a_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_a = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a_0, <8 x double> %a_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %b_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_b = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %b_0, <8 x double> %b_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %res_a, <8 x double> %res_b,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res                       
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  %a_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_a = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a_0, <8 x double> %a_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %b_0 = shufflevector <16 x double> %0, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b_1 = shufflevector <16 x double> %1, <16 x double> undef,
                       <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_b = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %b_0, <8 x double> %b_1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %res_a, <8 x double> %res_b,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.mask.sqrt.ps.512(<16 x float> %0, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse2.sqrt.sd, %0)
  ret double %ret
}

declare <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  %v0 = shufflevector <16 x double> %0, <16 x double> undef, 
                      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %0, <16 x double> undef, 
                      <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r0 = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %v0,  <8 x double> zeroinitializer, i8 -1, i32 4)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %v1,  <8 x double> zeroinitializer, i8 -1, i32 4)
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, 
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = zext i16 %intmask to i64
  ret i64 %res
}

define i1 @__any(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp ne i16 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp eq i16 %intmask, 65535
  ret i1 %res
}

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp eq i16 %intmask, 0
  ret i1 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int8/16 ops

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<16 x i8>) nounwind readnone alwaysinline {
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %0,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

define internal <16 x i16> @__add_varying_i16(<16 x i16>,
                                  <16 x i16>) nounwind readnone alwaysinline {
  %r = add <16 x i16> %0, %1
  ret <16 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<16 x i16>) nounwind readnone alwaysinline {
  reduce16(i16, @__add_varying_i16, @__add_uniform_i16)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vb = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %va, <8 x float> %vb)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %v3 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v2, <8 x float> %v2)
  %scalar1 = extractelement <8 x float> %v3, i32 0
  %scalar2 = extractelement <8 x float> %v3, i32 4
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}

define float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define internal <16 x i32> @__add_varying_int32(<16 x i32>,
                                       <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vb = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vc = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %vd = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %vab = fadd <4 x double> %va, %vb
  %vcd = fadd <4 x double> %vc, %vd

  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %vab, <4 x double> %vcd)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %final0 = extractelement <4 x double> %sum1, i32 0
  %final1 = extractelement <4 x double> %sum1, i32 2
  %sum = fadd double %final0, %final1
  ret double %sum
}

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define internal <16 x i64> @__add_varying_int64(<16 x i64>,
                                                <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

masked_load(half, 2)

declare <16 x i8> @llvm.x86.avx512.mask.loadu.b.128(ptr, <16 x i8>, i16)
define <16 x i8> @__masked_load_i8(i8 * %ptr, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = call <16 x i8> @llvm.x86.avx512.mask.loadu.b.128(i8* %ptr, <16 x i8> zeroinitializer, i16 %mask_i16)
  ret <16 x i8> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.loadu.w.256(ptr, <16 x i16>, i16)
define <16 x i16> @__masked_load_i16(i8 * %ptr, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = call <16 x i16> @llvm.x86.avx512.mask.loadu.w.256(i8* %ptr, <16 x i16> zeroinitializer, i16 %mask_i16)
  ret <16 x i16> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.loadu.d.512(i8*, <16 x i32>, i16)
define <16 x i32> @__masked_load_i32(i8 * %ptr, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = call <16 x i32> @llvm.x86.avx512.mask.loadu.d.512(i8* %ptr, <16 x i32> zeroinitializer, i16 %mask_i16)
  ret <16 x i32> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8*, <8 x i64>, i8)
define <16 x i64> @__masked_load_i64(i8 * %ptr, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_lo_i8 = call i8 @__extract_mask_low (<WIDTH x MASK> %mask) 
  %mask_hi_i8 = call i8 @__extract_mask_hi (<WIDTH x MASK> %mask)
  
  %ptr_d = bitcast i8* %ptr to <16 x i64>*
  %ptr_hi = getelementptr PTR_OP_ARGS(`<16 x i64>') %ptr_d, i32 0, i32 8
  %ptr_hi_i8 = bitcast i64* %ptr_hi to i8*

  %r0 = call <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8* %ptr, <8 x i64> zeroinitializer, i8 %mask_lo_i8)
  %r1 = call <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8* %ptr_hi_i8, <8 x i64> zeroinitializer, i8 %mask_hi_i8)
  
  %res = shufflevector <8 x i64> %r0, <8 x i64> %r1,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}


declare <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8*, <16 x float>, i16)
define <16 x float> @__masked_load_float(i8 * %ptr, <WIDTH x MASK> %mask) readonly alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = call <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8* %ptr, <16 x float> zeroinitializer, i16 %mask_i16)
  ret <16 x float> %res
}

declare <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8*, <8 x double>, i8)
define <16 x double> @__masked_load_double(i8 * %ptr, <WIDTH x MASK> %mask) readonly alwaysinline {
  %mask_lo_i8 = call i8 @__extract_mask_low (<WIDTH x MASK> %mask)
  %mask_hi_i8 = call i8 @__extract_mask_hi (<WIDTH x MASK> %mask)

  %ptr_d = bitcast i8* %ptr to <16 x double>*
  %ptr_hi = getelementptr PTR_OP_ARGS(`<16 x double>') %ptr_d, i32 0, i32 8
  %ptr_hi_i8 = bitcast double* %ptr_hi to i8*

  %r0 = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptr, <8 x double> zeroinitializer, i8 %mask_lo_i8)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptr_hi_i8, <8 x double> zeroinitializer, i8 %mask_hi_i8)
  
  %res = shufflevector <8 x double> %r0, <8 x double> %r1, 
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}


gen_masked_store(half)

declare void @llvm.x86.avx512.mask.storeu.b.128(i8*, <16 x i8>, i16)
define void @__masked_store_i8(<16 x i8>* nocapture, <16 x i8> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %ptr_i8 = bitcast <16 x i8>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.b.128(i8* %ptr_i8, <16 x i8> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.w.256(i8*, <16 x i16>, i16)
define void @__masked_store_i16(<16 x i16>* nocapture, <16 x i16> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %ptr_i8 = bitcast <16 x i16>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.w.256(i8* %ptr_i8, <16 x i16> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.d.512(i8*, <16 x i32>, i16)
define void @__masked_store_i32(<16 x i32>* nocapture, <16 x i32> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %ptr_i8 = bitcast <16 x i32>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.d.512(i8* %ptr_i8, <16 x i32> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.q.512(i8*, <8 x i64>, i8)
define void @__masked_store_i64(<16 x i64>* nocapture, <16 x i64> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_lo_i8 = call i8 @__extract_mask_low (<WIDTH x MASK> %mask)
  %mask_hi_i8 = call i8 @__extract_mask_hi (<WIDTH x MASK> %mask)

  %ptr_i8 = bitcast <16 x i64>* %0 to i8*
  %ptr_lo = getelementptr PTR_OP_ARGS(`<16 x i64>') %0, i32 0, i32 8
  %ptr_lo_i8 = bitcast i64* %ptr_lo to i8*

  %v_lo = shufflevector <16 x i64> %v, <16 x i64> undef,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v_hi = shufflevector <16 x i64> %v, <16 x i64> undef,
                        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.mask.storeu.q.512(i8* %ptr_i8, <8 x i64> %v_lo, i8 %mask_lo_i8)
  call void @llvm.x86.avx512.mask.storeu.q.512(i8* %ptr_lo_i8, <8 x i64> %v_hi, i8 %mask_hi_i8)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.512(i8*, <16 x float>, i16 )
define void @__masked_store_float(<16 x float>* nocapture, <16 x float> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i16 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)

  %ptr_i8 = bitcast <16 x float>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.ps.512(i8* %ptr_i8, <16 x float> %v, i16 %mask_i16)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8)
define void @__masked_store_double(<16 x double>* nocapture, <16 x double> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_lo_i8 = call i8 @__extract_mask_low (<WIDTH x MASK> %mask)
  %mask_hi_i8 = call i8 @__extract_mask_hi (<WIDTH x MASK> %mask)

  %ptr_i8 = bitcast <16 x double>* %0 to i8*
  %ptr_lo = getelementptr PTR_OP_ARGS(`<16 x double>') %0, i32 0, i32 8
  %ptr_lo_i8 = bitcast double* %ptr_lo to i8*

  %v_lo = shufflevector <16 x double> %v, <16 x double> undef,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v_hi = shufflevector <16 x double> %v, <16 x double> undef,
                        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr_i8, <8 x double> %v_lo, i8 %mask_lo_i8)
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr_lo_i8, <8 x double> %v_hi, i8 %mask_hi_i8)
  ret void
}

define void @__masked_store_blend_i8(<16 x i8>* nocapture, <16 x i8>, 
                                     <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x i8> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x i8> %1, <16 x i8> %v
  store <16 x i8> %v1, <16 x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<16 x i16>* nocapture, <16 x i16>, 
                                      <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x i16> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x i16> %1, <16 x i16> %v
  store <16 x i16> %v1, <16 x i16> * %0
  ret void
}

define void @__masked_store_blend_half(<16 x half>* nocapture,
                            <16 x half>, <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x half> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x half> %1, <16 x half> %v
  store <16 x half> %v1, <16 x half> * %0
  ret void
}

define void @__masked_store_blend_i32(<16 x i32>* nocapture, <16 x i32>, 
                                      <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x i32> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x i32> %1, <16 x i32> %v
  store <16 x i32> %v1, <16 x i32> * %0
  ret void
}

define void @__masked_store_blend_float(<16 x float>* nocapture, <16 x float>, 
                                        <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x float> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x float> %1, <16 x float> %v
  store <16 x float> %v1, <16 x float> * %0
  ret void
}

define void @__masked_store_blend_i64(<16 x i64>* nocapture,
                            <16 x i64>, <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x i64> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x i64> %1, <16 x i64> %v
  store <16 x i64> %v1, <16 x i64> * %0
  ret void
}

define void @__masked_store_blend_double(<16 x double>* nocapture,
                            <16 x double>, <WIDTH x MASK>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<16 x double> ')  %0
  %v1 = select <WIDTH x i1> %2, <16 x double> %1, <16 x double> %v
  store <16 x double> %v1, <16 x double> * %0
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

;; We need factored generic implementations when --opt=disable-gathers is used.
;; The util functions for gathers already include factored implementations,
;; so use factored ones here explicitely for remaining types only.

;; gather - i8
gen_gather(i8)

;; gather - i16
gen_gather(i16)
gen_gather(half)

gen_gather_factored_generic(i32)
gen_gather_factored_generic(float)
gen_gather_factored_generic(i64)
gen_gather_factored_generic(double)

;; gather - i32
declare <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512(<16 x i32>, i8*, <16 x i32>, <16 x i1>, i32)
define <16 x i32>
@__gather_base_offsets32_i32(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.dpi.512, 16, i32, ptr, offsets, i32, vecmask, <16 x i1>, offset_scale)
  ret <16 x i32> %res
}

declare <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512 (<8 x i32>, i8*, <8 x i64>, <8 x i1>, i32)
define <16 x i32>
@__gather_base_offsets64_i32(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res1, llvm.x86.avx512.mask.gather.qpi.512, 8, i32, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res2, llvm.x86.avx512.mask.gather.qpi.512, 8, i32, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x i32> %res1, <8 x i32> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i32> %res
}

define <16 x i32>
@__gather32_i32(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i32> @__gather_base_offsets32_i32(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x i32> %res
}

define <16 x i32>
@__gather64_i32(<16 x i64> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i32> @__gather_base_offsets64_i32(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x i32> %res
}

;; gather - i64
declare <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512(<8 x i64>, i8*, <8 x i32>, <8 x i1>, i32)
define <16 x i64>
@__gather_base_offsets32_i64(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res1, llvm.x86.avx512.mask.gather.dpq.512, 8, i64, ptr, offsets_lo, i32, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res2, llvm.x86.avx512.mask.gather.dpq.512, 8, i64, ptr, offsets_hi, i32, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x i64> %res1, <8 x i64> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512 (<8 x i64>, i8*, <8 x i64>, <8 x i1>, i32)
define <16 x i64>
@__gather_base_offsets64_i64(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res1, llvm.x86.avx512.mask.gather.qpq.512, 8, i64, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res2, llvm.x86.avx512.mask.gather.qpq.512, 8, i64, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x i64> %res1, <8 x i64> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i64> %res
}

define <16 x i64>
@__gather32_i64(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i64> @__gather_base_offsets32_i64(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x i64> %res
}

define <16 x i64>
@__gather64_i64(<16 x i64> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x i64> @__gather_base_offsets64_i64(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x i64> %res
}

;; gather - float
declare <16 x float> @llvm.x86.avx512.mask.gather.dps.512 (<16 x float>, i8*, <16 x i32>, <16 x i1>, i32)
define <16 x float>
@__gather_base_offsets32_float(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.dps.512, 16,float, ptr, offsets, i32, vecmask, <16 x i1>, offset_scale)
  ret <16 x float> %res
}

declare <8 x float> @llvm.x86.avx512.mask.gather.qps.512 (<8 x float>, i8*, <8 x i64>, <8 x i1>, i32)
define <16 x float>
@__gather_base_offsets64_float(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res_lo, llvm.x86.avx512.mask.gather.qps.512, 8, float, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res_hi, llvm.x86.avx512.mask.gather.qps.512, 8, float, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x float> %res_lo, <8 x float> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x float> %res
}

define <16 x float>
@__gather32_float(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x float> @__gather_base_offsets32_float(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x float> %res
}

define <16 x float>
@__gather64_float(<16 x i64> %ptrs,  <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x float> @__gather_base_offsets64_float(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x float> %res
}

;; gather - double
declare <8 x double> @llvm.x86.avx512.mask.gather.dpd.512(<8 x double>, i8*, <8 x i32>, <8 x i1>, i32)
define <16 x double>
@__gather_base_offsets32_double(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res1, llvm.x86.avx512.mask.gather.dpd.512, 8, double, ptr, offsets_lo, i32, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res2, llvm.x86.avx512.mask.gather.dpd.512, 8, double, ptr, offsets_hi, i32, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x double> %res1, <8 x double> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.gather.qpd.512 (<8 x double>, i8*, <8 x i64>, <8 x i1>, i32)
define <16 x double>
@__gather_base_offsets64_double(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_gather(res1, llvm.x86.avx512.mask.gather.qpd.512, 8, double, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale)
  convert_scale_to_const_gather(res2, llvm.x86.avx512.mask.gather.qpd.512, 8, double, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale)
  %res = shufflevector <8 x double> %res1, <8 x double> %res2 , <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}

define <16 x double>
@__gather32_double(<16 x i32> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x double> @__gather_base_offsets32_double(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i1> %vecmask)
  ret <16 x double> %res
}

define <16 x double>
@__gather64_double(<16 x i64> %ptrs, <16 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x double> @__gather_base_offsets64_double(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i1> %vecmask)
  ret <16 x double> %res
}


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

;; We need factored generic implementations when --opt=disable-scatters is used.
;; The util functions for scatters already include factored implementations,
;; so use factored ones here explicitely for remaining types only.

;; scatter - i8
scatterbo32_64(i8)
gen_scatter(i8)

;; scatter - i16
scatterbo32_64(i16)
gen_scatter(i16)

;; scatter - half
scatterbo32_64(half)
gen_scatter(half)

gen_scatter_factored(i32)
gen_scatter_factored(float)
gen_scatter_factored(i64)
gen_scatter_factored(double)

;; scatter - i32
declare void @llvm.x86.avx512.mask.scatter.dpi.512 (i8*, <16 x i1>, <16 x i32>, <16 x i32>, i32)
define void
@__scatter_base_offsets32_i32(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i32> %vals, <WIDTH x MASK> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpi.512, 16, vals, i32, ptr, offsets, i32, vecmask, <16 x i1>, offset_scale);
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qpi.512 (i8*, <8 x i1>, <8 x i64>, <8 x i32>, i32)
define void
@__scatter_base_offsets64_i32(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i32> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x i32> %vals, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x i32> %vals, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpi.512, 8, res_lo, i32, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpi.512, 8, res_hi, i32, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_i32(<16 x i32> %ptrs, <16 x i32> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_i32(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i32> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_i32(<16 x i64> %ptrs, <16 x i32> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_i32(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i32> %values, <WIDTH x MASK> %vecmask)
  ret void
}

;; scatter - i64
declare void @llvm.x86.avx512.mask.scatter.dpq.512 (i8*, <8 x i1>, <8 x i32>, <8 x i64>, i32)
define void
@__scatter_base_offsets32_i64(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i64> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x i64> %vals, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x i64> %vals, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpq.512, 8, res_lo, i64, ptr, offsets_lo, i32, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpq.512, 8, res_hi, i64, ptr, offsets_hi, i32, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qpq.512 (i8*, <8 x i1>, <8 x i64>, <8 x i64>, i32)
define void
@__scatter_base_offsets64_i64(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x i64> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x i64> %vals, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x i64> %vals, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpq.512, 8, res_lo, i64, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpq.512, 8, res_hi, i64, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_i64(<16 x i32> %ptrs, <16 x i64> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_i64(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x i64> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_i64(<16 x i64> %ptrs, <16 x i64> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_i64(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x i64> %values, <WIDTH x MASK> %vecmask)
  ret void
}

;; scatter - float
declare void @llvm.x86.avx512.mask.scatter.dps.512 (i8*, <16 x i1>, <16 x i32>, <16 x float>, i32)
define void
@__scatter_base_offsets32_float(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x float> %vals, <WIDTH x MASK> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dps.512, 16, vals, float, ptr, offsets, i32, vecmask, <16 x i1>, offset_scale);
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qps.512 (i8*, <8 x i1>, <8 x i64>, <8 x float>, i32)
define void
@__scatter_base_offsets64_float(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x float> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x float> %vals, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x float> %vals, <16 x float> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qps.512, 8, res_lo, float, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qps.512, 8, res_hi, float, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_float(<16 x i32> %ptrs, <16 x float> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_float(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x float> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_float(<16 x i64> %ptrs, <16 x float> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_float(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x float> %values, <WIDTH x MASK> %vecmask)
  ret void
}

;; scatter - double
declare void @llvm.x86.avx512.mask.scatter.dpd.512 (i8*, <8 x i1>, <8 x i32>, <8 x double>, i32)
define void
@__scatter_base_offsets32_double(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x double> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i32> %offsets, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x double> %vals, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x double> %vals, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpd.512, 8, res_lo, double, ptr, offsets_lo, i32, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpd.512, 8, res_hi, double, ptr, offsets_hi, i32, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qpd.512 (i8*, <8 x i1>, <8 x i64>, <8 x double>, i32)
define void
@__scatter_base_offsets64_double(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x double> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %vecmask_lo = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecmask_hi = shufflevector <16 x i1> %vecmask, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %offsets_lo = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %offsets_hi = shufflevector <16 x i64> %offsets, <16 x i64> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = shufflevector <16 x double> %vals, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res_hi = shufflevector <16 x double> %vals, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpd.512, 8, res_lo, double, ptr, offsets_lo, i64, vecmask_lo, <8 x i1>, offset_scale);
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpd.512, 8, res_hi, double, ptr, offsets_hi, i64, vecmask_hi, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_double(<16 x i32> %ptrs, <16 x double> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_double(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x double> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_double(<16 x i64> %ptrs, <16 x double> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_double(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x double> %values, <WIDTH x MASK> %vecmask)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; packed_load/store
packed_load_and_store(TRUE)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

define_prefetches()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;; Transcendentals are not defined because target definitions for all avx512
;; targets has false for m_hasTranscendentals. This means that no real use of
;; these functions happens in stdlib.ispc.

;; Trigonometry

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt

rcp14_uniform()
;; rcp float
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone
define <16 x float> @__rcp_fast_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %ret = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %0, <16 x float> undef, i16 -1)
  ret <16 x float> %ret
}
define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %call = call <16 x float> @__rcp_fast_varying_float(<16 x float> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  float iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <16 x float> %0, %call
  %two_minus = fsub <16 x float> <float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.>, %v_iv
  %iv_mul = fmul <16 x float> %call,  %two_minus
  ret <16 x float> %iv_mul
}

;; rcp double
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>, i8) nounwind readnone
define <16 x double> @__rcp_fast_varying_double(<16 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val_hi = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %val_lo, <8 x double> undef, i8 -1)
  %res_hi = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %val_hi, <8 x double> undef, i8 -1)
  %res = shufflevector <8 x double> %res_lo, <8 x double> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}
define <16 x double> @__rcp_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %call = call <16 x double> @__rcp_fast_varying_double(<16 x double> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  double iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <16 x double> %0, %call
  %two_minus = fsub <16 x double> <double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.>, %v_iv
  %iv_mul = fmul <16 x double> %call,  %two_minus
  ret <16 x double> %iv_mul
}

rsqrt14_uniform()
;; rsqrt float
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>,  <16 x float>,  i16) nounwind readnone
define <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %ret = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %v,  <16 x float> undef,  i16 -1)
  ret <16 x float> %ret
}
define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %is = call <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v)
  ; Newton-Raphson iteration to improve precision
  ;  float is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x float> %v,  %is
  %v_is_is = fmul <16 x float> %v_is,  %is
  %three_sub = fsub <16 x float> <float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <16 x float> %is,  %three_sub
  %half_scale = fmul <16 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <16 x float> %half_scale
}

;; rsqrt double
declare <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double>,  <8 x double>,  i8) nounwind readnone
define <16 x double> @__rsqrt_fast_varying_double(<16 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val_hi = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = call <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double> %val_lo, <8 x double> undef, i8 -1)
  %res_hi = call <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double> %val_hi, <8 x double> undef, i8 -1)
  %res = shufflevector <8 x double> %res_lo, <8 x double> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}
declare <8 x i1> @llvm.x86.avx512.fpclass.pd.512(<8 x double>, i32)
define <16 x double> @__rsqrt_varying_double(<16 x double> %v) nounwind readonly alwaysinline {
  ; detect +/-0 and +inf to deal with them differently.
  %val_lo = shufflevector <16 x double> %v, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val_hi = shufflevector <16 x double> %v, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %corner_cases_lo = call <8 x i1> @llvm.x86.avx512.fpclass.pd.512(<8 x double> %val_lo, i32 14)
  %corner_cases_hi = call <8 x i1> @llvm.x86.avx512.fpclass.pd.512(<8 x double> %val_hi, i32 14)
  %corner_cases = shufflevector <8 x i1> %corner_cases_lo, <8 x i1> %corner_cases_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %is = call <16 x double> @__rsqrt_fast_varying_double(<16 x double> %v)

  ; Precision refinement sequence based on minimax approximation.
  ; This sequence is a little slower than Newton-Raphson, but has much better precision
  ; Relative error is around 3 ULPs.
  ; t1 = 1.0 - (v * is) * is
  ; t2 = 0.37500000407453632 + t1 * 0.31250000550062401
  ; t3 = 0.5 + t1 * t2
  ; t4 = is + (t1*is) * t3
  %v_is = fmul <16 x double> %v,  %is
  %v_is_is = fmul <16 x double> %v_is,  %is
  %t1 = fsub <16 x double> <double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1.>, %v_is_is
  %t1_03125 = fmul <16 x double> <double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401>, %t1
  %t2 = fadd <16 x double> <double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632>, %t1_03125
  %t1_t2 = fmul <16 x double> %t1, %t2
  %t3 = fadd <16 x double> <double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5>, %t1_t2
  %t1_is = fmul <16 x double> %t1, %is
  %t1_is_t3 = fmul <16 x double> %t1_is, %t3
  %t4 = fadd <16 x double> %is, %t1_is_t3
  %ret = select <16 x i1> %corner_cases, <16 x double> %is, <16 x double> %t4
  ret <16 x double> %ret
}

;;saturation_arithmetic_novec()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
