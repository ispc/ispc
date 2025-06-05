;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Same target as target-avx2-i32x4 but without native VNNI

define(`HAVE_GATHER', `1')
; Define common 4-wide stuff
define(`WIDTH',`4')
define(`MASK',`i32')
define(`ISA',`AVX2')
include(`util.m4')

declare i1 @__is_compile_time_constant_mask(<WIDTH x MASK> %mask)
declare i1 @__is_compile_time_constant_uniform_int32(i32)
declare i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32>)
declare void @__masked_store_blend_i32(<4 x i32>* nocapture, <4 x i32>, 
                                      <4 x i32> %mask) nounwind alwaysinline
declare void @__masked_store_blend_i64(<4 x i64>* nocapture %ptr, <4 x i64> %new,
                                      <4 x i32> %i32mask) nounwind alwaysinline
declare i64 @__movmsk(<4 x i32>) nounwind readnone alwaysinline

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

declare <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float>, <4 x i32>)
define <4 x i32> @__shuffle_i32(<4 x i32>, <4 x i32>) nounwind readnone alwaysinline {
  %vec = bitcast <4 x i32> %0 to <4 x float>
  %res = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %vec, <4 x i32> %1)
  %res_casted = bitcast <4 x float> %res to <4 x i32>
  ret <4 x i32> %res_casted
}

define <4 x float> @__shuffle_float(<4 x float>, <4 x i32>) nounwind readnone alwaysinline {
  %res = call <4 x float> @llvm.x86.avx.vpermilvar.ps(<4 x float> %0, <4 x i32> %1)
  ret <4 x float> %res
}

define_shuffle2_const()

shuffle2(i8)
shuffle2(i16)
shuffle2(half)
shuffle2(double)
shuffle2(i64)

define <4 x i32> @__shuffle2_i32(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<4 x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <4 x i32> @__shuffle2_const_i32(<4 x i32> %0, <4 x i32> %1, <4 x i32> %2)
  ret <4 x i32> %res_const

not_const:
  %v1 = call <4 x i32> @__shuffle_i32(<4 x i32> %0, <4 x i32> %2)
  %perm2 = sub <4 x i32> %2, const_vector(i32, 4)
  %v2 = call <4 x i32> @__shuffle_i32(<4 x i32> %1, <4 x i32> %perm2)
  %mask = icmp slt <4 x i32> %2, const_vector(i32, 4)
  %res = select <4 x i1> %mask, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %res
}

define <4 x float> @__shuffle2_float(<4 x float>, <4 x float>, <4 x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<4 x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <4 x float> @__shuffle2_const_float(<4 x float> %0, <4 x float> %1, <4 x i32> %2)
  ret <4 x float> %res_const

not_const:
  %v1 = call <4 x float> @__shuffle_float(<4 x float> %0, <4 x i32> %2)
  %perm2 = sub <4 x i32> %2, const_vector(i32, 4)
  %v2 = call <4 x float> @__shuffle_float(<4 x float> %1, <4 x i32> %perm2)
  %mask = icmp slt <4 x i32> %2, const_vector(i32, 4)
  %res = select <4 x i1> %mask, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions



define(`expand_4to8', `
  %$3 = shufflevector <4 x $1> %$2, <4 x $1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
')
define(`extract_4from8', `
  %$3 = shufflevector <8 x $1> %$2, <8 x $1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
')

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
; 0 is round nearest even
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define <4 x float> @__half_to_float_varying(<4 x i16> %v4) nounwind readnone {
  expand_4to8(i16, v4, v) 
  %r = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %v)
  extract_4from8(float, r, ret)
  ret <4 x float> %ret
}

define <4 x i16> @__float_to_half_varying(<4 x float> %v4) nounwind readnone {
  expand_4to8(float, v4, v) 
  %r = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %v, i32 0)
  extract_4from8(i16, r, ret)
  ret <4 x i16> %ret
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
;; gather/scatter

; define these with the macros from stdlib.m4

declare void @llvm.trap() noreturn nounwind

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

declare <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i32> %mask, i8 %scale) readonly nounwind
declare <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i32> %mask, i8 %scale) readonly nounwind

define <4 x i32> @__gather_base_offsets32_i32(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i32> %vecmask) nounwind readonly alwaysinline {

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.d, 4, i32, ptr, offsets, i32, vecmask, i32, scale, i8)
  ret <4 x i32> %v
}


define <4 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i32> %vecmask) nounwind readonly alwaysinline {

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets, i64, vecmask, i32, scale, i8)

  ret <4 x i32> %v
}


define <4 x i32> @__gather32_i32(<4 x i32> %ptrs,
                                 <4 x i32> %vecmask) nounwind readonly alwaysinline {

  %v = call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x i32> %vecmask, i8 1)
  
  ret <4 x i32> %v
}


define <4 x i32> @__gather64_i32(<4 x i64> %ptrs, 
                                 <4 x i32> %vecmask) nounwind readonly alwaysinline {

  %v = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x i32> %vecmask, i8 1)

  ret <4 x i32> %v
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

declare <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind
declare <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind

define <4 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <4 x i32> %offsets,
                                  <4 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.ps, 4, float, ptr, offsets, i32, mask, float, scale, i8)

  ret <4 x float> %v
}


define <4 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <4 x i64> %offsets,
                                   <4 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets, i64, mask, float, scale, i8)

  ret <4 x float> %v
}


define <4 x float> @__gather32_float(<4 x i32> %ptrs, 
                                     <4 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  %v = call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> undef, i8 * null,
                     <4 x i32> %ptrs, <4 x float> %mask, i8 1)

  ret <4 x float> %v
}


define <4 x float> @__gather64_float(<4 x i64> %ptrs, 
                                     <4 x i32> %vecmask) nounwind readonly alwaysinline {
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  %v = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x float> %mask, i8 1)

  ret <4 x float> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

declare <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind
declare <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind

define <4 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i32> %vecmask32) nounwind readonly alwaysinline {

  %vecmask = sext <4 x i32> %vecmask32 to <4 x i64>
  convert_scale_to_const(v, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets, i32, vecmask, i64, scale, i8)

  ret <4 x i64> %v
}


define <4 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i32> %vecmask32) nounwind readonly alwaysinline {

  %vecmask = sext <4 x i32> %vecmask32 to <4 x i64>
  convert_scale_to_const(v, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets, i64, vecmask, i64, scale, i8)

  ret <4 x i64> %v
}


define <4 x i64> @__gather32_i64(<4 x i32> %ptrs, 
                                 <4 x i32> %vecmask32) nounwind readonly alwaysinline {

  %vecmask = sext <4 x i32> %vecmask32 to <4 x i64>
  %v = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x i64> %vecmask, i8 1)
  ret <4 x i64> %v
}


define <4 x i64> @__gather64_i64(<4 x i64> %ptrs, 
                                 <4 x i32> %vecmask32) nounwind readonly alwaysinline {

  %vecmask = sext <4 x i32> %vecmask32 to <4 x i64>
  %v = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x i64> %vecmask, i8 1)
  ret <4 x i64> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x double> %mask, i8 %scale) readonly nounwind
declare <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x double> %mask, i8 %scale) readonly nounwind

define <4 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i32> %vecmask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <4 x i32> %vecmask32 to <4 x i64>
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets, i32, vecmask, double, scale, i8)
  ret <4 x double> %v
}

define <4 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i32> %vecmask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <4 x i32> %vecmask32 to <4 x i64>
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets, i64, vecmask, double, scale, i8)

  ret <4 x double> %v
}

define <4 x double> @__gather32_double(<4 x i32> %ptrs, 
                                       <4 x i32> %vecmask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <4 x i32> %vecmask32 to <4 x i64>
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  %v = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x double> %vecmask, i8 1)

  ret <4 x double> %v
}

define <4 x double> @__gather64_double(<4 x i64> %ptrs, 
                                       <4 x i32> %vecmask32) nounwind readonly alwaysinline {
  %vecmask64 = sext <4 x i32> %vecmask32 to <4 x i64>
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  %v = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x double> %vecmask, i8 1)

  ret <4 x double> %v
}

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
