;;  Copyright (c) 2015-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`target-avx512-utils.ll')

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

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)

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


gen_masked_store(i8) ; llvm.x86.sse2.storeu.dq
gen_masked_store(i16)
gen_masked_store(half)

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

;; Transcendentals

;; exponent
define(`F144', `0x3FF7154760000000') ;; F144 = log(2, e)
define(`D144', `0x3FF71547652B82FE') ;; D144 = log(2, e)

declare <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone
declare <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone
declare <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float>, <16 x float>, <16 x float>, i16, i32)
declare <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double>, <8 x double>, <8 x double>, i8, i32)

define float @__exp_uniform_float(float %a) nounwind readnone alwaysinline {
  %res = call float @__stdlib_expf(float %a)
  ret float %res;
}

define double @__exp_uniform_double(double %a) nounwind readnone alwaysinline {
  %res = call double @__stdlib_exp(double %a)
  ret double %res;
}

define <16 x float> @__exp_varying_float(<16 x float> %a) nounwind readnone alwaysinline {
  %a0 = call <16 x float> @llvm.x86.avx512.mask.mul.ps.512(<16 x float> <float F144, float F144, float F144, float F144,
        float F144, float F144, float F144, float F144, float F144, float F144, float F144, float F144,
        float F144, float F144, float F144, float F144>, <16 x float> %a, <16 x float> zeroinitializer, i16 -1, i32 0)
  %res = call <16 x float> @llvm.x86.avx512.exp2.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8)
  ret <16 x float> %res
}

define <16 x double> @__exp_varying_double(<16 x double> %a) nounwind readnone alwaysinline {
  %alo = shufflevector <16 x double> %a, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %ahi = shufflevector <16 x double> %a, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  %alo0 = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> <double D144, double D144, double D144, 
          double D144, double D144, double D144, double D144, double D144>, <8 x double> %alo, <8 x double> zeroinitializer, i8 -1, i32 0)
  %ahi0 = call <8 x double> @llvm.x86.avx512.mask.mul.pd.512(<8 x double> <double D144, double D144, double D144, 
          double D144, double D144, double D144, double D144, double D144>, <8 x double> %ahi, <8 x double> zeroinitializer, i8 -1, i32 0)
  %res_lo = call <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double> %alo0, <8 x double> zeroinitializer, i8 -1, i32 8)
  %res_hi = call <8 x double> @llvm.x86.avx512.exp2.pd(<8 x double> %ahi0, <8 x double> zeroinitializer, i8 -1, i32 8)
  %res = shufflevector <8 x double> %res_lo, <8 x double> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15> 
  ret <16 x double> %res
}

;; power
define float @__pow_uniform_float(float %a, float %b) nounwind readnone alwaysinline {
  %res = call float @__stdlib_powf(float %a, float %b)
  ret float %res;
}

define double @__pow_uniform_double(double %a, double %b) nounwind readnone alwaysinline {
  %res = call double @__stdlib_pow(double %a, double %b)
  ret double %res;
}

declare <16 x float> @__pow_varying_float(<16 x float> %a, <16 x float> %b) nounwind readnone alwaysinline

;;define <16 x float> @__pow_varying_float(<16 x float> %a, <16 x float> %b) nounwind readnone alwaysinline
;;  ret <16 x float> %a
;;}

declare <16 x double> @__pow_varying_double(<16 x double> %a, <16 x double> %b) nounwind readnone alwaysinline


;; log
declare float @__log_uniform_float(float %a) nounwind readnone alwaysinline
declare double @__log_uniform_double(double %a) nounwind readnone alwaysinline
declare <16 x float> @__log_varying_float(<16 x float> %a) nounwind readnone alwaysinline
declare <16 x double> @__log_varying_double(<16 x double> %a) nounwind readnone alwaysinline

;; Trigonometry
trigonometry_decl()
