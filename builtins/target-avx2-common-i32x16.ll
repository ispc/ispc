;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`HAVE_GATHER', `1')
define(`ISA',`AVX2')

include(`target-avx-common-16.ll')

rdrand_definition()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare <8 x i32> @llvm.x86.avx2.pmins.d(<8 x i32>, <8 x i32>) nounwind readonly
declare <8 x i32> @llvm.x86.avx2.pmaxs.d(<8 x i32>, <8 x i32>) nounwind readonly

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary8to16(m, i32, @llvm.x86.avx2.pmins.d, %0, %1)
  ret <16 x i32> %m
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary8to16(m, i32, @llvm.x86.avx2.pmaxs.d, %0, %1)
  ret <16 x i32> %m
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare <8 x i32> @llvm.x86.avx2.pminu.d(<8 x i32>, <8 x i32>) nounwind readonly
declare <8 x i32> @llvm.x86.avx2.pmaxu.d(<8 x i32>, <8 x i32>) nounwind readonly

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary8to16(m, i32, @llvm.x86.avx2.pminu.d, %0, %1)
  ret <16 x i32> %m
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary8to16(m, i32, @llvm.x86.avx2.pmaxu.d, %0, %1)
  ret <16 x i32> %m
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
; 0 is round nearest even
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

; $1: type
; $2: var base name
define(`extract_4s', `
  %$2_1 = shufflevector <16 x $1> %$2, <16 x $1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$2_2 = shufflevector <16 x $1> %$2, <16 x $1> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %$2_3 = shufflevector <16 x $1> %$2, <16 x $1> undef, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %$2_4 = shufflevector <16 x $1> %$2, <16 x $1> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
')

; $1: type
; $2: var base name
define(`extract_8s', `
  %$2_1 = shufflevector <16 x $1> %$2, <16 x $1> undef,
                    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %$2_2 = shufflevector <16 x $1> %$2, <16 x $1> undef,
                    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
')

; $1: element type
; $2: ret name
; $3: v1
; $4: v2
define(`assemble_8s', `
  %$2 = shufflevector <8 x $1> %$3, <8 x $1> %$4,
                      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                  i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
')

; $1: element type
; $2: ret name
; $3: v1
; $4: v2
; $5: v3
; $6: v4
define(`assemble_4s', `
  %$2_1 = shufflevector <4 x $1> %$3, <4 x $1> %$4,
                    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %$2_2 = shufflevector <4 x $1> %$5, <4 x $1> %$6,
                    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  assemble_8s($1, $2, $2_1, $2_2)
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

define <16 x i32> @__gather_base_offsets32_i32(i8 * %ptr, i32 %scale, <16 x i32> %offsets,
                             <16 x i32> %vecmask) nounwind readonly alwaysinline {
  extract_8s(i32, offsets)
  extract_8s(i32, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.d.256, 8, i32, ptr, offsets_1, i32, vecmask_1, i32, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.d.256, 8, i32, ptr, offsets_2, i32, vecmask_2, i32, scale, i8)

  assemble_8s(i32, v, v1, v2)

  ret <16 x i32> %v
}


define <16 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x i32> %vecmask) nounwind readonly alwaysinline {

  extract_4s(i32, vecmask)
  extract_4s(i64, offsets)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_1, i64, vecmask_1, i32, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_2, i64, vecmask_2, i32, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_3, i64, vecmask_3, i32, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets_4, i64, vecmask_4, i32, scale, i8)

  assemble_4s(i32, v, v1, v2, v3, v4)

  ret <16 x i32> %v
}


define <16 x i32> @__gather32_i32(<16 x i32> %ptrs, 
                                  <16 x i32> %vecmask) nounwind readonly alwaysinline {
  extract_8s(i32, ptrs)
  extract_8s(i32, vecmask)

  %v1 = call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> undef, i8 * null,
                       <8 x i32> %ptrs_1, <8 x i32> %vecmask_1, i8 1)
  %v2 = call <8 x i32> @llvm.x86.avx2.gather.d.d.256(<8 x i32> undef, i8 * null,
                       <8 x i32> %ptrs_2, <8 x i32> %vecmask_2, i8 1)

  assemble_8s(i32, v, v1, v2)

  ret <16 x i32> %v
}


define <16 x i32> @__gather64_i32(<16 x i64> %ptrs, 
                                  <16 x i32> %vecmask) nounwind readonly alwaysinline {
  extract_4s(i64, ptrs)
  extract_4s(i32, vecmask)

  %v1 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x i32> %vecmask_1, i8 1)
  %v2 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x i32> %vecmask_2, i8 1)
  %v3 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_3, <4 x i32> %vecmask_3, i8 1)
  %v4 = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs_4, <4 x i32> %vecmask_4, i8 1)

  assemble_4s(i32, v, v1, v2, v3, v4)

  ret <16 x i32> %v
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

declare <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> %target, i8 * %ptr,
                       <8 x i32> %indices, <8 x float> %mask, i8 %scale8) readonly nounwind
declare <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind

define <16 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <16 x i32> %offsets,
                                  <16 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i32> %vecmask to <16 x float>
  extract_8s(i32, offsets)
  extract_8s(float, mask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.ps.256, 8, float, ptr, offsets_1, i32, mask_1, float, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.ps.256, 8, float, ptr, offsets_2, i32, mask_2, float, scale, i8)

  assemble_8s(float, v, v1, v2)

  ret <16 x float> %v
}


define <16 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <16 x i64> %offsets,
                                   <16 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i32> %vecmask to <16 x float>
  extract_4s(i64, offsets)
  extract_4s(float, mask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_1, i64, mask_1, float, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_2, i64, mask_2, float, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_3, i64, mask_3, float, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets_4, i64, mask_4, float, scale, i8)

  assemble_4s(float, v, v1, v2, v3, v4)

  ret <16 x float> %v
}


define <16 x float> @__gather32_float(<16 x i32> %ptrs, 
                                      <16 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i32> %vecmask to <16 x float>
  extract_8s(float, mask)
  extract_8s(i32, ptrs)

  %v1 = call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, i8 * null,
                     <8 x i32> %ptrs_1, <8 x float> %mask_1, i8 1)
  %v2 = call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, i8 * null,
                     <8 x i32> %ptrs_2, <8 x float> %mask_2, i8 1)

  assemble_8s(float, v, v1, v2)

  ret <16 x float> %v
}


define <16 x float> @__gather64_float(<16 x i64> %ptrs, 
                                      <16 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <16 x i32> %vecmask to <16 x float>
  extract_4s(i64, ptrs)
  extract_4s(float, mask)

  %v1 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x float> %mask_1, i8 1)
  %v2 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x float> %mask_2, i8 1)
  %v3 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_3, <4 x float> %mask_3, i8 1)
  %v4 = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs_4, <4 x float> %mask_4, i8 1)

  assemble_4s(float, v, v1, v2, v3, v4)

  ret <16 x float> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

declare <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind
declare <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind

define <16 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <16 x i32> %mask32 to <16 x i64>
  extract_4s(i32, offsets)
  extract_4s(i64, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_1, i32, vecmask_1, i64, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_2, i32, vecmask_2, i64, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_3, i32, vecmask_3, i64, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets_4, i32, vecmask_4, i64, scale, i8)

  assemble_4s(i64, v, v1, v2, v3, v4)

  ret <16 x i64> %v
}


define <16 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <16 x i32> %mask32 to <16 x i64>
  extract_4s(i64, offsets)
  extract_4s(i64, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_1, i64, vecmask_1, i64, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_2, i64, vecmask_2, i64, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_3, i64, vecmask_3, i64, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets_4, i64, vecmask_4, i64, scale, i8)

  assemble_4s(i64, v, v1, v2, v3, v4)

  ret <16 x i64> %v
}


define <16 x i64> @__gather32_i64(<16 x i32> %ptrs, 
                                  <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <16 x i32> %mask32 to <16 x i64>
  extract_4s(i32, ptrs)
  extract_4s(i64, vecmask)

  %v1 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_1, <4 x i64> %vecmask_1, i8 1)
  %v2 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_2, <4 x i64> %vecmask_2, i8 1)
  %v3 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_3, <4 x i64> %vecmask_3, i8 1)
  %v4 = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs_4, <4 x i64> %vecmask_4, i8 1)

  assemble_4s(i64, v, v1, v2, v3, v4)

  ret <16 x i64> %v
}

define <16 x i64> @__gather64_i64(<16 x i64> %ptrs, 
                                  <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask = sext <16 x i32> %mask32 to <16 x i64>
  extract_4s(i64, ptrs)
  extract_4s(i64, vecmask)

  %v1 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x i64> %vecmask_1, i8 1)
  %v2 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x i64> %vecmask_2, i8 1)
  %v3 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_3, <4 x i64> %vecmask_3, i8 1)
  %v4 = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs_4, <4 x i64> %vecmask_4, i8 1)

  assemble_4s(i64, v, v1, v2, v3, v4)

  ret <16 x i64> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x double> %mask, i8 %scale) readonly nounwind
declare <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x double> %mask, i8 %scale) readonly nounwind

define <16 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x i32> %mask32) nounwind readonly alwaysinline {
  %scale8 = trunc i32 %scale to i8
  %vecmask64 = sext <16 x i32> %mask32 to <16 x i64>
  %vecmask = bitcast <16 x i64> %vecmask64 to <16 x double>
  extract_4s(i32, offsets)
  extract_4s(double, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_1, i32, vecmask_1, double, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_2, i32, vecmask_2, double, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_3, i32, vecmask_3, double, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets_4, i32, vecmask_4, double, scale, i8)

  assemble_4s(double, v, v1, v2, v3, v4)

  ret <16 x double> %v
}


define <16 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <16 x i32> %mask32 to <16 x i64>
  %vecmask = bitcast <16 x i64> %vecmask64 to <16 x double>
  extract_4s(i64, offsets)
  extract_4s(double, vecmask)

  convert_scale_to_const(v1, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_1, i64, vecmask_1, double, scale, i8)
  convert_scale_to_const(v2, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_2, i64, vecmask_2, double, scale, i8)
  convert_scale_to_const(v3, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_3, i64, vecmask_3, double, scale, i8)
  convert_scale_to_const(v4, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets_4, i64, vecmask_4, double, scale, i8)

  assemble_4s(double, v, v1, v2, v3, v4)

  ret <16 x double> %v
}


define <16 x double> @__gather32_double(<16 x i32> %ptrs, 
                                        <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <16 x i32> %mask32 to <16 x i64>
  %vecmask = bitcast <16 x i64> %vecmask64 to <16 x double>
  extract_4s(i32, ptrs)
  extract_4s(double, vecmask)

  %v1 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_1, <4 x double> %vecmask_1, i8 1)
  %v2 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_2, <4 x double> %vecmask_2, i8 1)
  %v3 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_3, <4 x double> %vecmask_3, i8 1)
  %v4 = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs_4, <4 x double> %vecmask_4, i8 1)

  assemble_4s(double, v, v1, v2, v3, v4)

  ret <16 x double> %v
}


define <16 x double> @__gather64_double(<16 x i64> %ptrs, 
                                        <16 x i32> %mask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <16 x i32> %mask32 to <16 x i64>
  %vecmask = bitcast <16 x i64> %vecmask64 to <16 x double>
  extract_4s(i64, ptrs)
  extract_4s(double, vecmask)

  %v1 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_1, <4 x double> %vecmask_1, i8 1)
  %v2 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_2, <4 x double> %vecmask_2, i8 1)
  %v3 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_3, <4 x double> %vecmask_3, i8 1)
  %v4 = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs_4, <4 x double> %vecmask_4, i8 1)

  assemble_4s(double, v, v1, v2, v3, v4)

  ret <16 x double> %v
}
