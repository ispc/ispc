;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`HAVE_GATHER', `1')
define(`ISA',`AVX2')

include(`target-avx-common-8.ll')

rdrand_definition()
saturation_arithmetic()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare <8 x i32> @llvm.x86.avx2.pmins.d(<8 x i32>, <8 x i32>) nounwind readonly
declare <8 x i32> @llvm.x86.avx2.pmaxs.d(<8 x i32>, <8 x i32>) nounwind readonly

define <8 x i32> @__min_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %m = call <8 x i32> @llvm.x86.avx2.pmins.d(<8 x i32> %0, <8 x i32> %1)
  ret <8 x i32> %m
}

define <8 x i32> @__max_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %m = call <8 x i32> @llvm.x86.avx2.pmaxs.d(<8 x i32> %0, <8 x i32> %1)
  ret <8 x i32> %m
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare <8 x i32> @llvm.x86.avx2.pminu.d(<8 x i32>, <8 x i32>) nounwind readonly
declare <8 x i32> @llvm.x86.avx2.pmaxu.d(<8 x i32>, <8 x i32>) nounwind readonly

define <8 x i32> @__min_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %m = call <8 x i32> @llvm.x86.avx2.pminu.d(<8 x i32> %0, <8 x i32> %1)
  ret <8 x i32> %m
}

define <8 x i32> @__max_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %m = call <8 x i32> @llvm.x86.avx2.pmaxu.d(<8 x i32> %0, <8 x i32> %1)
  ret <8 x i32> %m
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
; 0 is round nearest even
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define <8 x float> @__half_to_float_varying(<8 x i16> %v) nounwind readnone {
  %r = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %v)
  ret <8 x float> %r
}

define <8 x i16> @__float_to_half_varying(<8 x float> %v) nounwind readnone {
  %r = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %v, i32 0)
  ret <8 x i16> %r
}

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %vv)
  %r = extractelement <8 x float> %rv, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  ; round to nearest even
  %rv = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv, i32 0)
  %r = extractelement <8 x i16> %rv, i32 0
  ret i16 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

declare void @llvm.trap() noreturn nounwind

define(`extract_4s', `
  %$2_1 = shufflevector <8 x $1> %$2, <8 x $1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$2_2 = shufflevector <8 x $1> %$2, <8 x $1> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
')

;; We need factored generic implementations when --opt=disable-gathers is used.
;; The util functions for gathers already include factored implementations,
;; so use factored ones here explicitely for remaining types only.
gen_gather(i8)
gen_gather(i16)
gen_gather(half)
gen_gather_factored_generic(i32)
gen_gather_factored_generic(float)
gen_gather_factored_generic(i64)
gen_gather_factored_generic(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 gathers

declare <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> %target, i8 * %ptr,
                       <8 x i32> %indices, <8 x i32> %mask, i8 %scale) readonly nounwind
declare <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i32> %mask, i8 %scale) readonly nounwind

define <8 x i32> @__gather_base_offsets32_i32(i8 * %ptr,
                             i32 %scale, <8 x i32> %offsets,
                             <8 x i32> %vecmask) nounwind readonly alwaysinline {

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.d.256, 8, i32, ptr, offsets, i32, vecmask, i32, scale, i8)

  ret <8 x i32> %v
}


define <8 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <8 x i64> %offsets,
                             <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %scale8 = trunc i32 %scale to i8
  extract_4s(i32, vecmask)
  extract_4s(i64, offsets)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_1, i64, vecmask_1, i32, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_2, i64, vecmask_2, i32, scale, i8)

  %v = shufflevector <4 x i32> %v1, <4 x i32> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i32> %v
}


define <8 x i32> @__gather32_i32(<8 x i32> %ptrs, 
                                 <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %v = call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> undef, i8 * null,
                      <8 x i32> %ptrs, <8 x i32> %vecmask, i8 1)
  ret <8 x i32> %v
}


define <8 x i32> @__gather64_i32(<8 x i64> %ptrs, 
                                 <8 x i32> %vecmask) nounwind readonly alwaysinline {
  extract_4s(i64, ptrs)
  extract_4s(i32, vecmask)

  %v1 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x i32> %vecmask_1, i8 1)
  %v2 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x i32> %vecmask_2, i8 1)

  %v = shufflevector <4 x i32> %v1, <4 x i32> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i32> %v
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

declare <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> %target, i8 * %ptr,
                       <8 x i32> %indices, <8 x float> %mask, i8 %scale8) readonly nounwind
declare <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind

define <8 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <8 x i32> %offsets,
                                  <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <8 x i32> %vecmask to <8 x float>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.ps.256, 8, float, ptr, offsets, i32, mask, float, scale, i8)

  ret <8 x float> %v
}


define <8 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <8 x i64> %offsets,
                                   <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <8 x i32> %vecmask to <8 x float>
  extract_4s(i64, offsets)
  extract_4s(float, mask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_1, i64, mask_1, float, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_2, i64, mask_2, float, scale, i8)

  %v = shufflevector <4 x float> %v1, <4 x float> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %v
}


define <8 x float> @__gather32_float(<8 x i32> %ptrs, 
                                     <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <8 x i32> %vecmask to <8 x float>

  %v = call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, i8 * null,
                     <8 x i32> %ptrs, <8 x float> %mask, i8 1)

  ret <8 x float> %v
}


define <8 x float> @__gather64_float(<8 x i64> %ptrs, 
                                     <8 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <8 x i32> %vecmask to <8 x float>
  extract_4s(i64, ptrs)
  extract_4s(float, mask)

  %v1 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x float> %mask_1, i8 1)
  %v2 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x float> %mask_2, i8 1)

  %v = shufflevector <4 x float> %v1, <4 x float> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

declare <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind
declare <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind

define <8 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <8 x i32> %offsets,
                             <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <8 x i32> %mask32 to <8 x i64>
  extract_4s(i32, offsets)
  extract_4s(i64, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_1, i32, vecmask_1, i64, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_2, i32, vecmask_2, i64, scale, i8)

  %v = shufflevector <4 x i64> %v1, <4 x i64> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %v
}


define <8 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <8 x i64> %offsets,
                             <8 x i32> %mask32) nounwind readonly alwaysinline {
  %scale8 = trunc i32 %scale to i8
  %vecmask = sext <8 x i32> %mask32 to <8 x i64>
  extract_4s(i64, offsets)
  extract_4s(i64, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_1, i64, vecmask_1, i64, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_2, i64, vecmask_2, i64, scale, i8)

  %v = shufflevector <4 x i64> %v1, <4 x i64> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %v
}


define <8 x i64> @__gather32_i64(<8 x i32> %ptrs, 
                                 <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <8 x i32> %mask32 to <8 x i64>

  extract_4s(i32, ptrs)
  extract_4s(i64, vecmask)

  %v1 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_1, <4 x i64> %vecmask_1, i8 1)
  %v2 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_2, <4 x i64> %vecmask_2, i8 1)
  %v = shufflevector <4 x i64> %v1, <4 x i64> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %v
}


define <8 x i64> @__gather64_i64(<8 x i64> %ptrs, 
                                 <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <8 x i32> %mask32 to <8 x i64>
  extract_4s(i64, ptrs)
  extract_4s(i64, vecmask)

  %v1 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x i64> %vecmask_1, i8 1)
  %v2 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x i64> %vecmask_2, i8 1)

  %v = shufflevector <4 x i64> %v1, <4 x i64> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x double> %mask, i8 %scale) readonly nounwind
declare <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x double> %mask, i8 %scale) readonly nounwind

define <8 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <8 x i32> %offsets,
                             <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <8 x i32> %mask32 to <8 x i64>
  %vecmask = bitcast <8 x i64> %vecmask64 to <8 x double>
  extract_4s(i32, offsets)
  extract_4s(double, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_1, i32, vecmask_1, double, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_2, i32, vecmask_2, double, scale, i8)

  %v = shufflevector <4 x double> %v1, <4 x double> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %v
}

define <8 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <8 x i64> %offsets,
                             <8 x i32> %mask32) nounwind readonly alwaysinline {
  %scale8 = trunc i32 %scale to i8
  %vecmask64 = sext <8 x i32> %mask32 to <8 x i64>
  %vecmask = bitcast <8 x i64> %vecmask64 to <8 x double>
  extract_4s(i64, offsets)
  extract_4s(double, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_1, i64, vecmask_1, double, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_2, i64, vecmask_2, double, scale, i8)

  %v = shufflevector <4 x double> %v1, <4 x double> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %v
}

define <8 x double> @__gather32_double(<8 x i32> %ptrs, 
                                       <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <8 x i32> %mask32 to <8 x i64>
  %vecmask = bitcast <8 x i64> %vecmask64 to <8 x double>
  extract_4s(i32, ptrs)
  extract_4s(double, vecmask)

  %v1 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_1, <4 x double> %vecmask_1, i8 1)
  %v2 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_2, <4 x double> %vecmask_2, i8 1)

  %v = shufflevector <4 x double> %v1, <4 x double> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %v
}

define <8 x double> @__gather64_double(<8 x i64> %ptrs, 
                                       <8 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <8 x i32> %mask32 to <8 x i64>
  %vecmask = bitcast <8 x i64> %vecmask64 to <8 x double>
  extract_4s(i64, ptrs)
  extract_4s(double, vecmask)

  %v1 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x double> %vecmask_1, i8 1)
  %v2 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x double> %vecmask_2, i8 1)

  %v = shufflevector <4 x double> %v1, <4 x double> %v2,
                     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

  ret <8 x double> %v
}
