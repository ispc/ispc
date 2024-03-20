;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause


;; This file defines the target for "double-pumped" SSE4, i.e. running
;; with 8-wide vectors

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standard 8-wide definitions from m4 macros

define(`WIDTH',`8')
define(`MASK',`i32')
define(`ISA',`SSE4')
include(`util.m4')

stdlib_core()
packed_load_and_store(FALSE)
scans()
int64minmax()
saturation_arithmetic()

include(`target-sse4-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

define <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  unary4to8(call, float, @llvm.x86.sse.rcp.ps, %0)
  ; do one N-R iteration
  %v_iv = fmul <8 x float> %0, %call
  %two_minus = fsub <8 x float> <float 2., float 2., float 2., float 2.,
                                 float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <8 x float> %call, %two_minus
  ret <8 x float> %iv_mul
}

define <8 x float> @__rcp_fast_varying_float(<8 x float>) nounwind readonly alwaysinline {
  unary4to8(ret, float, @llvm.x86.sse.rcp.ps, %0)
  ret <8 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary4to8(is, float, @llvm.x86.sse.rsqrt.ps, %v)
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <8 x float> %v, %is
  %v_is_is = fmul <8 x float> %v_is, %is
  %three_sub = fsub <8 x float> <float 3., float 3., float 3., float 3.,
                                 float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <8 x float> %is, %three_sub
  %half_scale = fmul <8 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                  float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <8 x float> %half_scale
}

define <8 x float> @__rsqrt_fast_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  unary4to8(ret, float, @llvm.x86.sse.rsqrt.ps, %v)
  ret <8 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @__sqrt_varying_float(<8 x float>) nounwind readonly alwaysinline {
  unary4to8(call, float, @llvm.x86.sse.sqrt.ps, %0)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @__max_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.max.ps, %0, %1)
  ret <8 x float> %call
}

define <8 x float> @__min_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.min.ps, %0, %1)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

define <8 x i32> @__min_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <8 x i32> %call
}

define <8 x i32> @__max_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <8 x i32> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

define <8 x i32> @__min_varying_uint32(<8 x i32>,
                                       <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <8 x i32> %call
}

define <8 x i32> @__max_varying_uint32(<8 x i32>,
                                       <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <8 x i32> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

define i64 @__movmsk(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %v64 = zext i32 %v to i64
  ret i64 %v64
}

define i1 @__any(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp eq i32 %v, 255
  ret i1 %cmp
}

define i1 @__none(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp eq i32 %v, 0
  ret i1 %cmp
}

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<8 x i8>) nounwind readnone alwaysinline {
  %wide8 = shufflevector <8 x i8> %0, <8 x i8> zeroinitializer,
      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %wide8,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

define internal <8 x i16> @__add_varying_i16(<8 x i16>,
                                  <8 x i16>) nounwind readnone alwaysinline {
  %r = add <8 x i16> %0, %1
  ret <8 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<8 x i16>) nounwind readnone alwaysinline {
  reduce8(i16, @__add_varying_i16, @__add_uniform_i16)
}

define float @__reduce_min_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8by4(float, @llvm.x86.sse.min.ps, @__min_uniform_float)
}

define float @__reduce_max_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8by4(float, @llvm.x86.sse.max.ps, @__max_uniform_float)
}

; helper function for reduce_add_int32
define <4 x i32> @__vec4_add_int32(<4 x i32> %v0,
                                   <4 x i32> %v1) nounwind readnone alwaysinline {
  %v = add <4 x i32> %v0, %v1
  ret <4 x i32> %v
}

; helper function for reduce_add_int32
define i32 @__add_int32(i32, i32) nounwind readnone alwaysinline {
  %v = add i32 %0, %1
  ret i32 %v
}

define i32 @__reduce_add_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @__vec4_add_int32, @__add_int32)
}

define i32 @__reduce_min_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pminsd, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pmaxsd, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pminud, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pmaxud, @__max_uniform_uint32)
}

define <4 x double> @__add_varying_double(<4 x double>,
                                     <4 x double>) nounwind readnone alwaysinline {
  %r = fadd <4 x double> %0, %1
  ret <4 x double> %r
}

define double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %r = fadd double %0, %1
  ret double %r
}

define double @__reduce_add_double(<8 x double>) nounwind readnone {
  reduce8by4(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<8 x double>) nounwind readnone {
  reduce8(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<8 x double>) nounwind readnone {
  reduce8(double, @__max_varying_double, @__max_uniform_double)
}

define <4 x i64> @__add_varying_int64(<4 x i64>,
                                      <4 x i64>) nounwind readnone alwaysinline {
  %r = add <4 x i64> %0, %1
  ret <4 x i64> %r
}

define i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %r = add i64 %0, %1
  ret i64 %r
}

define i64 @__reduce_add_int64(<8 x i64>) nounwind readnone {
  reduce8by4(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts


masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(half)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float rounding

declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone

define <8 x float> @__round_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round4to8(%0, 8)
}

define <8 x float> @__floor_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round4to8(%0, 9)
}

define <8 x float> @__ceil_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round4to8(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone

define <8 x double> @__round_varying_double(<8 x double>) nounwind readonly alwaysinline {
  round2to8double(%0, 8)
}

define <8 x double> @__floor_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round2to8double(%0, 9)
}

define <8 x double> @__ceil_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round2to8double(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define float @__reduce_add_float(<8 x float>) nounwind readonly alwaysinline {
  %a = shufflevector <8 x float> %0, <8 x float> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %b = shufflevector <8 x float> %0, <8 x float> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %ab = fadd <4 x float> %a, %b
  %hab = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %ab, <4 x float> %ab)
  %a_scalar = extractelement <4 x float> %hab, i32 0
  %b_scalar = extractelement <4 x float> %hab, i32 1
  %sum = fadd float %a_scalar, %b_scalar
  ret float %sum
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

masked_store_blend_8_16_by_8()

declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>,
                                             <4 x float>) nounwind readnone

define void @__masked_store_blend_i32(<8 x i32>* nocapture, <8 x i32>, 
                                      <8 x i32> %mask) nounwind alwaysinline {
  ; do two 4-wide blends with blendvps
  %mask_as_float = bitcast <8 x i32> %mask to <8 x float>
  %mask_a = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %mask_b = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %oldValue = load PTR_OP_ARGS(`<8 x i32>')  %0, align 4
  %oldAsFloat = bitcast <8 x i32> %oldValue to <8 x float>
  %newAsFloat = bitcast <8 x i32> %1 to <8 x float>
  %old_a = shufflevector <8 x float> %oldAsFloat, <8 x float> undef,
                <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %old_b = shufflevector <8 x float> %oldAsFloat, <8 x float> undef,
                <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %new_a = shufflevector <8 x float> %newAsFloat, <8 x float> undef,
                <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %new_b = shufflevector <8 x float> %newAsFloat, <8 x float> undef,
                <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %blend_a = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old_a, <4 x float> %new_a,
                                                       <4 x float> %mask_a)
  %blend_b = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old_b, <4 x float> %new_b,
                                                       <4 x float> %mask_b)
  %blend = shufflevector <4 x float> %blend_a, <4 x float> %blend_b,
               <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %blendAsInt = bitcast <8 x float> %blend to <8 x i32>
  store <8 x i32> %blendAsInt, <8 x i32>* %0, align 4
  ret void
}

define void @__masked_store_blend_i64(<8 x i64>* nocapture %ptr, <8 x i64> %new,
                                      <8 x i32> %mask) nounwind alwaysinline {
  ; implement this as 4 blends of <4 x i32>s, which are actually bitcast
  ; <2 x i64>s...

  %mask_as_float = bitcast <8 x i32> %mask to <8 x float>

  %old = load PTR_OP_ARGS(`<8 x i64>')  %ptr, align 8

  ; set up the first two 64-bit values
  %old01 = shufflevector <8 x i64> %old, <8 x i64> undef, <2 x i32> <i32 0, i32 1>
  %old01f = bitcast <2 x i64> %old01 to <4 x float>
  %new01  = shufflevector <8 x i64> %new, <8 x i64> undef, <2 x i32> <i32 0, i32 1>
  %new01f = bitcast <2 x i64> %new01 to <4 x float>
  ; compute mask--note that the values mask0 and mask1 are doubled-up
  %mask01 = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                          <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  ; and blend the two of them values
  %result01f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old01f,
                                                         <4 x float> %new01f,
                                                         <4 x float> %mask01)
  %result01 = bitcast <4 x float> %result01f to <2 x i64>

  ; and again
  %old23 = shufflevector <8 x i64> %old, <8 x i64> undef, <2 x i32> <i32 2, i32 3>
  %old23f = bitcast <2 x i64> %old23 to <4 x float>
  %new23  = shufflevector <8 x i64> %new, <8 x i64> undef, <2 x i32> <i32 2, i32 3>
  %new23f = bitcast <2 x i64> %new23 to <4 x float>
  %mask23 = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                          <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  %result23f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old23f,
                                                         <4 x float> %new23f,
                                                         <4 x float> %mask23)
  %result23 = bitcast <4 x float> %result23f to <2 x i64>

  %old45 = shufflevector <8 x i64> %old, <8 x i64> undef, <2 x i32> <i32 4, i32 5>
  %old45f = bitcast <2 x i64> %old45 to <4 x float>
  %new45  = shufflevector <8 x i64> %new, <8 x i64> undef, <2 x i32> <i32 4, i32 5>
  %new45f = bitcast <2 x i64> %new45 to <4 x float>
  %mask45 = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                          <4 x i32> <i32 4, i32 4, i32 5, i32 5>
  %result45f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old45f,
                                                         <4 x float> %new45f,
                                                         <4 x float> %mask45)
  %result45 = bitcast <4 x float> %result45f to <2 x i64>

  %old67 = shufflevector <8 x i64> %old, <8 x i64> undef, <2 x i32> <i32 6, i32 7>
  %old67f = bitcast <2 x i64> %old67 to <4 x float>
  %new67  = shufflevector <8 x i64> %new, <8 x i64> undef, <2 x i32> <i32 6, i32 7>
  %new67f = bitcast <2 x i64> %new67 to <4 x float>
  %mask67 = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                          <4 x i32> <i32 6, i32 6, i32 7, i32 7>
  %result67f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old67f,
                                                         <4 x float> %new67f,
                                                         <4 x float> %mask67)
  %result67 = bitcast <4 x float> %result67f to <2 x i64>

  %final0123 = shufflevector <2 x i64> %result01, <2 x i64> %result23,
       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %final4567 = shufflevector <2 x i64> %result45, <2 x i64> %result67,
       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %final = shufflevector <4 x i64> %final0123, <4 x i64> %final4567,
       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i64> %final, <8 x i64> * %ptr, align 8
  ret void
}

masked_store_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <8 x double> @__sqrt_varying_double(<8 x double>) nounwind alwaysinline {
  unary2to8(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <8 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision float min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <8 x double> %ret
}

define <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <8 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half
rcph_rsqrth_decl

transcendetals_decl()
trigonometry_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
dot_product_vnni_decl()
