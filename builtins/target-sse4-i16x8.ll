;;  Copyright (c) 2013, 2015, Google, Inc.
;;  Copyright(c) 2019-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Define common 4-wide stuff
define(`WIDTH',`8')
define(`MASK',`i16')
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

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  unary4to8(call, float, @llvm.x86.sse.rcp.ps, %0)
  ; do one N-R iteration to improve precision
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);
  %v_iv = fmul <8 x float> %0, %call
  %two_minus = fsub <8 x float> <float 2., float 2., float 2., float 2.,
                                 float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <8 x float> %call, %two_minus
  ret <8 x float> %iv_mul
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  unary4to8(ret, float, @llvm.x86.sse.rcp.ps, %0)
  ret <8 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary4to8(is, float, @llvm.x86.sse.rsqrt.ps, %v)
   ; Newton-Raphson iteration to improve precision
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

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %v) nounwind readonly alwaysinline {
  unary4to8(ret, float, @llvm.x86.sse.rsqrt.ps, %v)
  ret <8 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @__sqrt_varying_float(<8 x float>) nounwind readonly alwaysinline {
  unary4to8(call, float, @llvm.x86.sse.sqrt.ps, %0)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <8 x double> @__sqrt_varying_double(<8 x double>) nounwind
alwaysinline {
  unary2to8(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <8 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

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
  round2to8double(%0, 9)
}

define <8 x double> @__ceil_varying_double(<8 x double>) nounwind readonly alwaysinline {
  round2to8double(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

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

define <8 x i32> @__min_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <8 x i32> %call
}

define <8 x i32> @__max_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <8 x i32> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone {
  binary2to8(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <8 x double> %ret
}

define <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone {
  binary2to8(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <8 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone

define i64 @__movmsk(<8 x MASK>) nounwind readnone alwaysinline {
  %m8 = trunc <8 x MASK> %0 to <8 x i8>
  %mask8 = shufflevector <8 x i8> %m8, <8 x i8> zeroinitializer,
      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %mask8)
  %m64 = zext i32 %m to i64
  ret i64 %m64
}

define i1 @__any(<8 x MASK>) nounwind readnone alwaysinline {
  %m = call i64 @__movmsk(<8 x MASK> %0)
  %mne = icmp ne i64 %m, 0
  ret i1 %mne
}

define i1 @__all(<8 x MASK>) nounwind readnone alwaysinline {
  %m = call i64 @__movmsk(<8 x MASK> %0)
  %meq = icmp eq i64 %m, ALL_ON_MASK
  ret i1 %meq
}

define i1 @__none(<8 x MASK>) nounwind readnone alwaysinline {
  %m = call i64 @__movmsk(<8 x MASK> %0)
  %meq = icmp eq i64 %m, 0
  ret i1 %meq
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

define internal <8 x float> @__add_varying_float(<8 x float>, <8 x float>) {
  %r = fadd <8 x float> %0, %1
  ret <8 x float> %r
}

define internal float @__add_uniform_float(float, float) {
  %r = fadd float %0, %1
  ret float %r
}

define float @__reduce_add_float(<8 x float>) nounwind readonly alwaysinline {
  reduce8(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<8 x float>) nounwind readnone {
  reduce8(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<8 x float>) nounwind readnone {
  reduce8(float, @__max_varying_float, @__max_uniform_float)
}

define internal <8 x i32> @__add_varying_int32(<8 x i32>, <8 x i32>) {
  %r = add <8 x i32> %0, %1
  ret <8 x i32> %r
}

define internal i32 @__add_uniform_int32(i32, i32) {
  %r = add i32 %0, %1
  ret i32 %r
}

define i32 @__reduce_add_int32(<8 x i32>) nounwind readnone {
  reduce8(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<8 x i32>) nounwind readnone {
  reduce8(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<8 x i32>) nounwind readnone {
  reduce8(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<8 x i32>) nounwind readnone {
  reduce8(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<8 x i32>) nounwind readnone {
  reduce8(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

define internal <8 x double> @__add_varying_double(<8 x double>, <8 x double>) {
  %r = fadd <8 x double> %0, %1
  ret <8 x double> %r
}

define internal double @__add_uniform_double(double, double) {
  %r = fadd double %0, %1
  ret double %r
}

define double @__reduce_add_double(<8 x double>) nounwind readnone {
  reduce8(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<8 x double>) nounwind readnone {
  reduce8(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<8 x double>) nounwind readnone {
  reduce8(double, @__max_varying_double, @__max_uniform_double)
}

define internal <8 x i64> @__add_varying_int64(<8 x i64>, <8 x i64>) {
  %r = add <8 x i64> %0, %1
  ret <8 x i64> %r
}

define internal i64 @__add_uniform_int64(i64, i64) {
  %r = add i64 %0, %1
  ret i64 %r
}

define i64 @__reduce_add_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__add_varying_int64, @__add_uniform_int64)
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
;; masked store

define void @__masked_store_blend_i64(<8 x i64>* nocapture, <8 x i64>,
                                      <8 x MASK> %mask) nounwind
                                      alwaysinline {
  %mask_as_i1 = trunc <8 x MASK> %mask to <8 x i1>
  %old = load PTR_OP_ARGS(`<8 x i64>')  %0, align 4
  %blend = select <8 x i1> %mask_as_i1, <8 x i64> %1, <8 x i64> %old
  store <8 x i64> %blend, <8 x i64>* %0, align 4
  ret void
}

define void @__masked_store_blend_i32(<8 x i32>* nocapture, <8 x i32>, 
                                      <8 x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <8 x MASK> %mask to <8 x i1>
  %old = load PTR_OP_ARGS(`<8 x i32>')  %0, align 4
  %blend = select <8 x i1> %mask_as_i1, <8 x i32> %1, <8 x i32> %old
  store <8 x i32> %blend, <8 x i32>* %0, align 4
  ret void
}

define void @__masked_store_blend_i16(<8 x i16>* nocapture, <8 x i16>,
                                     <8 x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <8 x MASK> %mask to <8 x i1>
  %old = load PTR_OP_ARGS(`<8 x i16>')  %0, align 4
  %blend = select <8 x i1> %mask_as_i1, <8 x i16> %1, <8 x i16> %old
  store <8 x i16> %blend, <8 x i16>* %0, align 4
  ret void
}

define void @__masked_store_blend_i8(<8 x i8>* nocapture, <8 x i8>,
                                     <8 x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <8 x MASK> %mask to <8 x i1>
  %old = load PTR_OP_ARGS(`<8 x i8>')  %0, align 4
  %blend = select <8 x i1> %mask_as_i1, <8 x i8> %1, <8 x i8> %old
  store <8 x i8> %blend, <8 x i8>* %0, align 4
  ret void
}

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

masked_store_float_double()

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

; define these with the macros from stdlib.m4

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
;; int8/int16 builtins

declare <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i8> @__avg_up_uint8(<8 x i8>, <8 x i8>) {
  %v0 = shufflevector <8 x i8> %0, <8 x i8> undef,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  %v1 = shufflevector <8 x i8> %1, <8 x i8> undef,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  %r16 = call <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8> %v0, <16 x i8> %v1)
  %r = shufflevector <16 x i8> %r16, <16 x i8> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %r
}

declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @__avg_up_uint16(<8 x i16>, <8 x i16>) {
  %r = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %0, <8 x i16> %1)
  ret <8 x i16> %r
}

define_avg_up_int8()
define_avg_up_int16()
define_down_avgs()

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
