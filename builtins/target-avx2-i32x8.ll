;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Same target as target-avx2-i32x8 but without native VNNI

define(`HAVE_GATHER', `1')
define(`ISA',`AVX2')
define(`WIDTH',`8')
define(`MASK',`i32')
include(`util.m4')

declare i1 @__is_compile_time_constant_mask(<WIDTH x MASK> %mask)
declare i1 @__is_compile_time_constant_uniform_int32(i32)
declare i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32>)

declare void @__masked_store_blend_i32(<8 x i32>* nocapture, <8 x i32>, 
                                      <8 x i32>) nounwind alwaysinline
declare void @__masked_store_blend_i64(<8 x i64>* nocapture %ptr, <8 x i64> %new, 
                                      <8 x i32> %i32mask) nounwind alwaysinline
declare i64 @__movmsk(<8 x i32>) nounwind readnone alwaysinline

rdrand_definition()
saturation_arithmetic()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; switch macro
;; This is required to ensure that gather intrinsics are used with constant scale value.
;; This particular implementation of the routine is used by non-avx512 targets currently(avx2-i64x4, avx2-i32x8, avx2-i32x16).
;; $1: Return value
;; $2: funcName
;; $3: Width
;; $4: scalar type of array
;; $5: ptr
;; $6: offset
;; $7: scalar type of offset
;; $8: vecMask
;; $9: scalar type of vecMask
;; $10: scale
;; $11: scale type

define(`convert_scale_to_const', `


 switch i32 %argn(`10',$@), label %default_$1 [ i32 1, label %on_one_$1
                                                i32 2, label %on_two_$1
                                                i32 4, label %on_four_$1
                                                i32 8, label %on_eight_$1]

on_one_$1:
  %$1_1 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, <$3 x $9> %$8, argn(`11',$@) 1)
  br label %end_bb_$1

on_two_$1:
  %$1_2 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, <$3 x $9> %$8, argn(`11',$@) 2)
  br label %end_bb_$1

on_four_$1:
  %$1_4 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, <$3 x $9> %$8, argn(`11',$@) 4)
  br label %end_bb_$1

on_eight_$1:
  %$1_8 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, <$3 x $9> %$8, argn(`11',$@) 8)
  br label %end_bb_$1

default_$1:
  unreachable

end_bb_$1:
  %$1 = phi <$3 x $4> [ %$1_1, %on_one_$1 ], [ %$1_2, %on_two_$1 ], [ %$1_4, %on_four_$1 ], [ %$1_8, %on_eight_$1 ]
'
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; optimized shuf version

shuffle1(i8)
shuffle1(i16)
shuffle1(half)
shuffle1(double)
shuffle1(i64)

declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>)
define <8 x i32> @__shuffle_i32(<8 x i32>, <8 x i32>) nounwind readnone alwaysinline {
  %res = call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %0, <8 x i32> %1)
  ret <8 x i32> %res
}

declare <8 x float> @llvm.x86.avx2.permps(<8 x float>, <8 x i32>)
define <8 x float> @__shuffle_float(<8 x float>, <8 x i32>) nounwind readnone alwaysinline {
  %res = call <8 x float> @llvm.x86.avx2.permps(<8 x float> %0, <8 x i32> %1)
  ret <8 x float> %res
}

define_shuffle2_const()

shuffle2(i8)
shuffle2(i16)
shuffle2(half)
shuffle2(double)
shuffle2(i64)

define <WIDTH x i32> @__shuffle2_i32(<WIDTH x i32>, <WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x i32> @__shuffle2_const_i32(<WIDTH x i32> %0, <WIDTH x i32> %1, <WIDTH x i32> %2)
  ret <WIDTH x i32> %res_const

not_const:
  %v1 = call <8 x i32> @__shuffle_i32(<8 x i32> %0, <WIDTH x i32> %2)
  %perm2 = sub <8 x i32> %2, const_vector(i32, 8)
  %v2 = call <8 x i32> @__shuffle_i32(<8 x i32> %1, <8 x i32> %perm2)
  %mask = icmp slt <8 x i32> %2, const_vector(i32, 8)
  %res = select <8 x i1> %mask, <8 x i32> %v1, <8 x i32> %v2
  ret <8 x i32> %res
}

define <WIDTH x float> @__shuffle2_float(<WIDTH x float>, <WIDTH x float>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x float> @__shuffle2_const_float(<WIDTH x float> %0, <WIDTH x float> %1, <WIDTH x i32> %2)
  ret <WIDTH x float> %res_const

not_const:
  %v1 = call <8 x float> @__shuffle_float(<8 x float> %0, <8 x i32> %2)
  %perm2 = sub <8 x i32> %2, const_vector(i32, 8)
  %v2 = call <8 x float> @__shuffle_float(<8 x float> %1, <8 x i32> %perm2)
  %mask = icmp slt <8 x i32> %2, const_vector(i32, 8)
  %res = select <8 x i1> %mask, <8 x float> %v1, <8 x float> %v2
  ret <8 x float> %res
}

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pack / unpack

declare void @llvm.masked.compressstore.v8f16(<8 x half> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
declare void @llvm.masked.compressstore.v8f32(<8 x float> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
declare void @llvm.masked.compressstore.v8f64(<8 x double> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
declare void @llvm.masked.compressstore.v8i8(<8 x i8> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
declare void @llvm.masked.compressstore.v8i16(<8 x i16> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
declare void @llvm.masked.compressstore.v8i32(<8 x i32>, ptr, <8 x i1>)
declare void @llvm.masked.compressstore.v8i64(<8 x i64>, ptr, <8 x i1>)
declare <8 x half> @llvm.masked.expandload.v8f16(ptr %startptr_typed, <8 x i1> %i1mask, <8 x half> %data)
declare <8 x float> @llvm.masked.expandload.v8f32(ptr %startptr_typed, <8 x i1> %i1mask, <8 x float> %data)
declare <8 x double> @llvm.masked.expandload.v8f64(ptr %startptr_typed, <8 x i1> %i1mask, <8 x double> %data)
declare <8 x i8> @llvm.masked.expandload.v8i8(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i8> %data)
declare <8 x i16> @llvm.masked.expandload.v8i16(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i16> %data)
declare <8 x i32> @llvm.masked.expandload.v8i32(ptr, <8 x i1>, <8 x i32>)
declare <8 x i64> @llvm.masked.expandload.v8i64(ptr, <8 x i1>, <8 x i64>)
declare i8 @llvm.ctpop.i8(i8)
declare i32 @llvm.x86.bmi.pdep.32(i32, i32)
declare i32 @llvm.x86.bmi.pext.32(i32, i32)
declare void @llvm.x86.avx2.maskstore.d.256(ptr, <8 x i32>, <8 x i32>)

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activei8(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x i8>, ptr %val_ptr_typed, align 8
  %vec_load = call <8 x i8> @llvm.masked.expandload.v8i8(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i8> %data)
  store <8 x i8> %vec_load, ptr %val_ptr_typed, align 1
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activei8(ptr %startptr, <8 x i8> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8i8(<8 x i8> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2i8(ptr %startptr, <8 x i8> %vals, <8 x i32> %full_mask) #0 {
entry:
  %startptr_typed = bitcast ptr %startptr to ptr
  %mask = call i64 @__movmsk(<8 x i32> %full_mask)
  %mask_known = call i1 @__is_compile_time_constant_mask(<8 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:                                       ; preds = %entry
  %allon = icmp eq i64 %mask, 255
  br i1 %allon, label %all_on, label %unknown_mask

all_on:                                           ; preds = %known_mask
  %vecptr = bitcast ptr %startptr_typed to ptr
  store <8 x i8> %vals, ptr %vecptr, align 4
  ret i32 8

unknown_mask:                                     ; preds = %known_mask, %entry
  br label %loop

loop:                                             ; preds = %loop, %unknown_mask
  %offset = phi i32 [ 0, %unknown_mask ], [ %ch_offset, %loop ]
  %i = phi i32 [ 0, %unknown_mask ], [ %ch_i, %loop ]
  %storeval = extractelement <8 x i8> %vals, i32 %i
  %offset1 = zext i32 %offset to i64
  %storeptr = getelementptr i8, ptr %startptr_typed, i64 %offset1
  store i8 %storeval, ptr %storeptr, align 1
  %mull_mask = extractelement <8 x i32> %full_mask, i32 %i
  %ch_offset = sub i32 %offset, %mull_mask
  %ch_i = add i32 %i, 1
  %test = icmp ne i32 %ch_i, 8
  br i1 %test, label %loop, label %done

done:                                             ; preds = %loop
  ret i32 %ch_offset
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activei16(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x i16>, ptr %val_ptr_typed, align 16
  %vec_load = call <8 x i16> @llvm.masked.expandload.v8i16(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i16> %data)
  store <8 x i16> %vec_load, ptr %val_ptr_typed, align 2
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activei16(ptr %startptr, <8 x i16> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8i16(<8 x i16> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2i16(ptr %startptr, <8 x i16> %vals, <8 x i32> %full_mask) #0 {
entry:
  %startptr_typed = bitcast ptr %startptr to ptr
  %mask = call i64 @__movmsk(<8 x i32> %full_mask)
  %mask_known = call i1 @__is_compile_time_constant_mask(<8 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:                                       ; preds = %entry
  %allon = icmp eq i64 %mask, 255
  br i1 %allon, label %all_on, label %unknown_mask

all_on:                                           ; preds = %known_mask
  %vecptr = bitcast ptr %startptr_typed to ptr
  store <8 x i16> %vals, ptr %vecptr, align 4
  ret i32 8

unknown_mask:                                     ; preds = %known_mask, %entry
  br label %loop

loop:                                             ; preds = %loop, %unknown_mask
  %offset = phi i32 [ 0, %unknown_mask ], [ %ch_offset, %loop ]
  %i = phi i32 [ 0, %unknown_mask ], [ %ch_i, %loop ]
  %storeval = extractelement <8 x i16> %vals, i32 %i
  %offset1 = zext i32 %offset to i64
  %storeptr = getelementptr i16, ptr %startptr_typed, i64 %offset1
  store i16 %storeval, ptr %storeptr, align 2
  %mull_mask = extractelement <8 x i32> %full_mask, i32 %i
  %ch_offset = sub i32 %offset, %mull_mask
  %ch_i = add i32 %i, 1
  %test = icmp ne i32 %ch_i, 8
  br i1 %test, label %loop, label %done

done:                                             ; preds = %loop
  ret i32 %ch_offset
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activei32(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x i32>, ptr %val_ptr_typed, align 32
  %vec_load = call <8 x i32> @llvm.masked.expandload.v8i32(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i32> %data)
  store <8 x i32> %vec_load, ptr %val_ptr_typed, align 4
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activei32(ptr %startptr, <8 x i32> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8i32(<8 x i32> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Based on this stack overflow article: https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
; unsigned int compress256(__m256i src, __m256i full_mask, int* a) {
;   __m256i zero_vector = _mm256_setzero_si256();
;   __m256i cmp_result = _mm256_cmpgt_epi32(full_mask, zero_vector);
;   unsigned int i1mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));
;   uint32_t i32mask = (uint32_t)i1mask;
;   unsigned int mask_cnt = _mm_popcnt_u32(i32mask);
;   uint32_t expanded_mask = _pdep_u32(i32mask, 0x011111111);
;   expanded_mask *= 0xF;
;   unsigned int compressed_mask = (1u << mask_cnt) - 1;
;   uint32_t wip_output_mask = _pdep_u32(compressed_mask, 0x11111111);
;   __m128i temp = _mm_cvtsi32_si128(wip_output_mask);
;   __m128i mask = _mm_set1_epi8(0x0F);  // Mask to isolate nibbles
;   __m128i low_nibbles = _mm_and_si128(temp, mask);
;   __m128i high_nibbles = _mm_and_si128(_mm_srli_epi32(temp, 4), mask);
;   __m128i result = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
;   result *= 0x80;
;   __m256i store_mask = _mm256_cvtepi8_epi32(result);
;   const uint32_t identity_indices = 0x076543210;
;   uint32_t wanted_indices = _pext_u32(identity_indices, expanded_mask);
;   __m128i bytevec  = _mm_cvtsi32_si128(wanted_indices);
;   low_nibbles      = _mm_and_si128(bytevec, mask);
;   high_nibbles     = _mm_and_si128(_mm_srli_epi32(bytevec, 4), mask);
;   bytevec          = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
;   __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
;   src = _mm256_permutevar8x32_epi32(src, shufmask);
;   _mm256_maskstore_epi32(a, store_mask, src);
;   return mask_cnt;
; }
;
; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2i32(ptr %startptr, <8 x i32> %vals, <8 x i32> %full_mask) #0 {
entry:
  %cmp.i = icmp ne <8 x i32> %full_mask, zeroinitializer
  %0 = bitcast <8 x i1> %cmp.i to i8
  %1 = zext i8 %0 to i32
  %2 = call i32 @llvm.ctpop.i32(i32 %1)
  %3 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %1, i32 286331153)
  %mul = mul i32 %3, 15
  %notmask = shl nsw i32 -1, %2
  %sub = xor i32 %notmask, -1
  %4 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %sub, i32 286331153)
  %vecinit3.i = insertelement <4 x i32> <i32 poison, i32 0, i32 poison, i32 poison>, i32 %4, i64 0
  %.scalar = lshr i32 %4, 4
  %5 = insertelement <4 x i32> <i32 poison, i32 0, i32 poison, i32 poison>, i32 %.scalar, i64 0
  %6 = bitcast <4 x i32> %vecinit3.i to <16 x i8>
  %7 = bitcast <4 x i32> %5 to <16 x i8>
  %8 = shufflevector <16 x i8> %6, <16 x i8> %7, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %9 = bitcast <16 x i8> %8 to <2 x i64>
  %10 = shl <2 x i64> %9, <i64 7, i64 7>
  %11 = bitcast <2 x i64> %10 to <16 x i8>
  %12 = and <16 x i8> %11, <i8 -128, i8 -121, i8 -121, i8 -121, i8 -121, i8 -121, i8 -121, i8 -121, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %shuffle.i37 = shufflevector <16 x i8> %12, <16 x i8> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %conv.i = sext <8 x i8> %shuffle.i37 to <8 x i32>
  %13 = tail call i32 @llvm.x86.bmi.pext.32(i32 1985229328, i32 %mul)
  %vecinit3.i36 = insertelement <4 x i32> poison, i32 %13, i64 0
  %.scalar42 = lshr i32 %13, 4
  %14 = insertelement <4 x i32> poison, i32 %.scalar42, i64 0
  %15 = bitcast <4 x i32> %vecinit3.i36 to <16 x i8>
  %16 = and <16 x i8> %15, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %17 = bitcast <4 x i32> %14 to <16 x i8>
  %18 = and <16 x i8> %17, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %shuffle.i40 = shufflevector <16 x i8> %16, <16 x i8> %18, <8 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19>
  %conv.i41 = zext <8 x i8> %shuffle.i40 to <8 x i32>
  %19 = tail call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %vals, <8 x i32> %conv.i41)
  tail call void @llvm.x86.avx2.maskstore.d.256(ptr %startptr, <8 x i32> %conv.i, <8 x i32> %19)
  ret i32 %2
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activef16(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x half>, ptr %val_ptr_typed, align 16
  %vec_load = call <8 x half> @llvm.masked.expandload.v8f16(ptr %startptr_typed, <8 x i1> %i1mask, <8 x half> %data)
  store <8 x half> %vec_load, ptr %val_ptr_typed, align 2
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activef16(ptr %startptr, <8 x half> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8f16(<8 x half> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2f16(ptr %startptr, <8 x half> %vals, <8 x i32> %full_mask) #0 {
entry:
  %startptr_typed = bitcast ptr %startptr to ptr
  %mask = call i64 @__movmsk(<8 x i32> %full_mask)
  %mask_known = call i1 @__is_compile_time_constant_mask(<8 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:                                       ; preds = %entry
  %allon = icmp eq i64 %mask, 255
  br i1 %allon, label %all_on, label %unknown_mask

all_on:                                           ; preds = %known_mask
  %vecptr = bitcast ptr %startptr_typed to ptr
  store <8 x half> %vals, ptr %vecptr, align 4
  ret i32 8

unknown_mask:                                     ; preds = %known_mask, %entry
  br label %loop

loop:                                             ; preds = %loop, %unknown_mask
  %offset = phi i32 [ 0, %unknown_mask ], [ %ch_offset, %loop ]
  %i = phi i32 [ 0, %unknown_mask ], [ %ch_i, %loop ]
  %storeval = extractelement <8 x half> %vals, i32 %i
  %offset1 = zext i32 %offset to i64
  %storeptr = getelementptr half, ptr %startptr_typed, i64 %offset1
  store half %storeval, ptr %storeptr, align 2
  %mull_mask = extractelement <8 x i32> %full_mask, i32 %i
  %ch_offset = sub i32 %offset, %mull_mask
  %ch_i = add i32 %i, 1
  %test = icmp ne i32 %ch_i, 8
  br i1 %test, label %loop, label %done

done:                                             ; preds = %loop
  ret i32 %ch_offset
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activef32(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x float>, ptr %val_ptr_typed, align 32
  %vec_load = call <8 x float> @llvm.masked.expandload.v8f32(ptr %startptr_typed, <8 x i1> %i1mask, <8 x float> %data)
  store <8 x float> %vec_load, ptr %val_ptr_typed, align 4
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activef32(ptr %startptr, <8 x float> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8f32(<8 x float> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2f32(ptr %startptr, <8 x float> %vals, <8 x i32> %full_mask) #0 {
entry:
  %startptr_typed = bitcast ptr %startptr to ptr
  %mask = call i64 @__movmsk(<8 x i32> %full_mask)
  %mask_known = call i1 @__is_compile_time_constant_mask(<8 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:                                       ; preds = %entry
  %allon = icmp eq i64 %mask, 255
  br i1 %allon, label %all_on, label %unknown_mask

all_on:                                           ; preds = %known_mask
  %vecptr = bitcast ptr %startptr_typed to ptr
  store <8 x float> %vals, ptr %vecptr, align 4
  ret i32 8

unknown_mask:                                     ; preds = %known_mask, %entry
  br label %loop

loop:                                             ; preds = %loop, %unknown_mask
  %offset = phi i32 [ 0, %unknown_mask ], [ %ch_offset, %loop ]
  %i = phi i32 [ 0, %unknown_mask ], [ %ch_i, %loop ]
  %storeval = extractelement <8 x float> %vals, i32 %i
  %offset1 = zext i32 %offset to i64
  %storeptr = getelementptr float, ptr %startptr_typed, i64 %offset1
  store float %storeval, ptr %storeptr, align 4
  %mull_mask = extractelement <8 x i32> %full_mask, i32 %i
  %ch_offset = sub i32 %offset, %mull_mask
  %ch_i = add i32 %i, 1
  %test = icmp ne i32 %ch_i, 8
  br i1 %test, label %loop, label %done

done:                                             ; preds = %loop
  ret i32 %ch_offset
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activei64(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x i64>, ptr %val_ptr_typed, align 64
  %vec_load = call <8 x i64> @llvm.masked.expandload.v8i64(ptr %startptr_typed, <8 x i1> %i1mask, <8 x i64> %data)
  store <8 x i64> %vec_load, ptr %val_ptr_typed, align 8
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activei64(ptr %startptr, <8 x i64> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8i64(<8 x i64> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}


; Based on this stack overflow article: https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
; unsigned int compress256x2(__m256i vals_low, __m256i vals_high, __m256i full_mask, int* a){
;   __m256i zero_vector = _mm256_setzero_si256();
;   __m256i cmp_result = _mm256_cmpgt_epi32(full_mask, zero_vector);
;   unsigned int i1mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));
;   uint32_t i32mask = (uint32_t)i1mask;
;   uint32_t expanded_mask = _pdep_u32(i32mask, 0x01010101);
;   unsigned int mask_cnt = _mm_popcnt_u32(expanded_mask);
;   expanded_mask *= 0xFF;
;   unsigned int compressed_mask = (1u << mask_cnt) - 1;
;   uint32_t wip_output_mask = _pdep_u32(compressed_mask, 0x80808080);
;   __m128i temp = _mm_cvtsi32_si128(wip_output_mask);
;   __m128i temp_se = _mm_cvtepi8_epi32(temp);
;   __m256i store_mask = _mm256_cvtepi32_epi64(temp_se);
;   const uint32_t identity_indices = 0x076543210;
;   uint32_t wanted_indices = _pext_u32(identity_indices, expanded_mask);
;   __m128i mask = _mm_set1_epi8(0x0F);
;   __m128i bytevec  = _mm_cvtsi32_si128(wanted_indices);
;   __m128i low_nibbles      = _mm_and_si128(bytevec, mask);
;   __m128i high_nibbles     = _mm_and_si128(_mm_srli_epi32(bytevec, 4), mask);
;   bytevec          = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
;   __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
;   vals_low = _mm256_permutevar8x32_epi32(vals_low, shufmask);
;   _mm256_maskstore_epi32(a, store_mask, vals_low);
;   a += mask_cnt * 2;
;   i32mask >>= 4;
;   expanded_mask = _pdep_u32(i32mask, 0x01010101);
;   unsigned int mask_cnt2 = _mm_popcnt_u64(expanded_mask);
;   expanded_mask *= 0xFF;
;   compressed_mask = (1u << mask_cnt2) - 1;
;   wip_output_mask = _pdep_u32(compressed_mask, 0x80808080);
;   temp = _mm_cvtsi32_si128(wip_output_mask);
;   temp_se = _mm_cvtepi8_epi32(temp);
;   store_mask = _mm256_cvtepi32_epi64(temp_se);
;   wanted_indices = _pext_u32(identity_indices, expanded_mask);
;   bytevec  = _mm_cvtsi32_si128(wanted_indices);
;   low_nibbles      = _mm_and_si128(bytevec, mask);
;   high_nibbles     = _mm_and_si128(_mm_srli_epi32(bytevec, 4), mask);
;   bytevec          = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
;   shufmask = _mm256_cvtepu8_epi32(bytevec);
;   vals_high = _mm256_permutevar8x32_epi32(vals_high, shufmask);
;   _mm256_maskstore_epi32(a, store_mask, vals_high);
;   return mask_cnt + mask_cnt2;
; }
;
; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2i64(ptr %startptr, <8 x i64> %vals, <8 x i32> %full_mask) #0 {
entry:
  %vals_low = shufflevector <8 x i64> %vals, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vals_high = shufflevector <8 x i64> %vals, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %cmp.i = icmp ne <8 x i32> %full_mask, zeroinitializer
  %0 = bitcast <8 x i1> %cmp.i to i8
  %conv = zext i8 %0 to i32
  %1 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %conv, i32 16843009)
  %2 = call i32 @llvm.ctpop.i32(i32 %1)
  %mul = mul i32 %1, 255
  %notmask = shl nsw i32 -1, %2
  %sub = xor i32 %notmask, -1
  %3 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %sub , i32 -2139062144)
  %shuffle.i = bitcast i32 %3 to <4 x i8>
  %conv.i = sext <4 x i8> %shuffle.i to <4 x i64>
  %4 = tail call i32 @llvm.x86.bmi.pext.32(i32 1985229328, i32 %mul)
  %vecinit3.i71 = insertelement <4 x i32> poison, i32 %4, i64 0
  %.scalar = lshr i32 %4, 4
  %5 = insertelement <4 x i32> poison, i32 %.scalar, i64 0
  %6 = bitcast <4 x i32> %vecinit3.i71 to <16 x i8>
  %7 = and <16 x i8> %6, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %8 = bitcast <4 x i32> %5 to <16 x i8>
  %9 = and <16 x i8> %8, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %shuffle.i74 = shufflevector <16 x i8> %7, <16 x i8> %9, <8 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19>
  %conv.i75 = zext <8 x i8> %shuffle.i74 to <8 x i32>
  %10 = bitcast <4 x i64> %vals_low to <8 x i32>
  %11 = tail call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %10, <8 x i32> %conv.i75)
  %12 = bitcast <4 x i64> %conv.i to <8 x i32>
  tail call void @llvm.x86.avx2.maskstore.d.256(ptr %startptr, <8 x i32> %12, <8 x i32> %11)
  %mul15 = shl nuw nsw i32 %2, 1
  %idx.ext = zext i32 %mul15 to i64
  %add.ptr = getelementptr inbounds i32, ptr %startptr, i64 %idx.ext
  %shr = lshr i32 %conv, 4
  %13 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %shr, i32 16843009)
  %14 = tail call i32 @llvm.ctpop.i32(i32 %13)
  %mul20 = mul i32 %13, 255
  %notmask52 = shl nsw i32 -1, %14
  %sub22 = xor i32 %notmask52, -1
  %15 = tail call i32 @llvm.x86.bmi.pdep.32(i32 %sub22, i32 -2139062144)
  %shuffle.i54 = bitcast i32 %15 to <4 x i8>
  %conv.i50 = sext <4 x i8> %shuffle.i54 to <4 x i64>
  %16 = tail call i32 @llvm.x86.bmi.pext.32(i32 1985229328, i32 %mul20)
  %vecinit3.i80 = insertelement <4 x i32> poison, i32 %16, i64 0
  %.scalar86 = lshr i32 %16, 4
  %17 = insertelement <4 x i32> poison, i32 %.scalar86, i64 0
  %18 = bitcast <4 x i32> %vecinit3.i80 to <16 x i8>
  %19 = and <16 x i8> %18, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %20 = bitcast <4 x i32> %17 to <16 x i8>
  %21 = and <16 x i8> %20, <i8 15, i8 15, i8 15, i8 15, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>
  %shuffle.i84 = shufflevector <16 x i8> %19, <16 x i8> %21, <8 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19>
  %conv.i85 = zext <8 x i8> %shuffle.i84 to <8 x i32>
  %22 = bitcast <4 x i64> %vals_high to <8 x i32>
  %23 = tail call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %22, <8 x i32> %conv.i85)
  %24 = bitcast <4 x i64> %conv.i50 to <8 x i32>
  tail call void @llvm.x86.avx2.maskstore.d.256(ptr %add.ptr, <8 x i32> %24, <8 x i32> %23)
  %add = add nuw nsw i32 %14, %2
  ret i32 %add
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_load_activef64(ptr %startptr, ptr %val_ptr, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %val_ptr_typed = bitcast ptr %val_ptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  %data = load <8 x double>, ptr %val_ptr_typed, align 64
  %vec_load = call <8 x double> @llvm.masked.expandload.v8f64(ptr %startptr_typed, <8 x i1> %i1mask, <8 x double> %data)
  store <8 x double> %vec_load, ptr %val_ptr_typed, align 8
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_activef64(ptr %startptr, <8 x double> %vals, <8 x i32> %full_mask) #0 {
  %startptr_typed = bitcast ptr %startptr to ptr
  %i1mask = icmp ne <8 x i32> %full_mask, zeroinitializer
  call void @llvm.masked.compressstore.v8f64(<8 x double> %vals, ptr %startptr_typed, <8 x i1> %i1mask)
  %i8mask = bitcast <8 x i1> %i1mask to i8
  %i32mask = zext i8 %i8mask to i32
  %ret = call i32 @llvm.ctpop.i32(i32 %i32mask)
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define i32 @__packed_store_active2f64(ptr %startptr, <8 x double> %vals, <8 x i32> %full_mask) #0 {
entry:
  %startptr_typed = bitcast ptr %startptr to ptr
  %mask = call i64 @__movmsk(<8 x i32> %full_mask)
  %mask_known = call i1 @__is_compile_time_constant_mask(<8 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:                                       ; preds = %entry
  %allon = icmp eq i64 %mask, 255
  br i1 %allon, label %all_on, label %unknown_mask

all_on:                                           ; preds = %known_mask
  %vecptr = bitcast ptr %startptr_typed to ptr
  store <8 x double> %vals, ptr %vecptr, align 4
  ret i32 8

unknown_mask:                                     ; preds = %known_mask, %entry
  br label %loop

loop:                                             ; preds = %loop, %unknown_mask
  %offset = phi i32 [ 0, %unknown_mask ], [ %ch_offset, %loop ]
  %i = phi i32 [ 0, %unknown_mask ], [ %ch_i, %loop ]
  %storeval = extractelement <8 x double> %vals, i32 %i
  %offset1 = zext i32 %offset to i64
  %storeptr = getelementptr double, ptr %startptr_typed, i64 %offset1
  store double %storeval, ptr %storeptr, align 8
  %mull_mask = extractelement <8 x i32> %full_mask, i32 %i
  %ch_offset = sub i32 %offset, %mull_mask
  %ch_i = add i32 %i, 1
  %test = icmp ne i32 %ch_i, 8
  br i1 %test, label %loop, label %done

done:                                             ; preds = %loop
  ret i32 %ch_offset
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
