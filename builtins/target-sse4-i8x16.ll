;;  Copyright (c) 2013, 2015, Google, Inc.
;;  Copyright(c) 2019-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Define common 4-wide stuff
define(`WIDTH',`16')
define(`MASK',`i8')
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
  unary4to16(call, float, @llvm.x86.sse.rcp.ps, %0)
  ; do one N-R iteration to improve precision
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);
  %v_iv = fmul <16 x float> %0, %call
  %two_minus = fsub <16 x float> <float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <16 x float> %call, %two_minus
  ret <16 x float> %iv_mul
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  unary4to16(ret, float, @llvm.x86.sse.rcp.ps, %0)
  ret <16 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary4to16(is, float, @llvm.x86.sse.rsqrt.ps, %v)
   ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x float> %v, %is
  %v_is_is = fmul <16 x float> %v_is, %is
  %three_sub = fsub <16 x float> <float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <16 x float> %is, %three_sub
  %half_scale = fmul <16 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <16 x float> %half_scale
}

define <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  unary4to16(ret, float, @llvm.x86.sse.rsqrt.ps, %v)
  ret <16 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  unary4to16(call, float, @llvm.x86.sse.sqrt.ps, %0)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind
alwaysinline {
  unary2to16(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <16 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round4to16(%0, 8)
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round4to16(%0, 9)
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round4to16(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone

define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
    round2to16double(%0, 8)
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1001 = 9
    round2to16double(%0, 9)
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1010 = 10
    round2to16double(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <16 x float> @__max_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  binary4to16(call, float, @llvm.x86.sse.max.ps, %0, %1)
  ret <16 x float> %call
}

define <16 x float> @__min_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
  binary4to16(call, float, @llvm.x86.sse.min.ps, %0, %1)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(call, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <16 x i32> %call
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(call, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <16 x i32> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(call, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <16 x i32> %call
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(call, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <16 x i32> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone {
  binary2to16(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <16 x double> %ret
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone {
  binary2to16(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <16 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone

define i64 @__movmsk(<16 x i8>) nounwind readnone alwaysinline {
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %0)
  %m64 = zext i32 %m to i64
  ret i64 %m64
}

define i1 @__any(<16 x i8>) nounwind readnone alwaysinline {
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %0)
  %mne = icmp ne i32 %m, 0
  ret i1 %mne
}

define i1 @__all(<16 x i8>) nounwind readnone alwaysinline {
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %0)
  %meq = icmp eq i32 %m, ALL_ON_MASK
  ret i1 %meq
}

define i1 @__none(<16 x i8>) nounwind readnone alwaysinline {
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %0)
  %meq = icmp eq i32 %m, 0
  ret i1 %meq
}

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

define internal <16 x float> @__add_varying_float(<16 x float>, <16 x float>) {
  %r = fadd <16 x float> %0, %1
  ret <16 x float> %r
}

define internal float @__add_uniform_float(float, float) {
  %r = fadd float %0, %1
  ret float %r
}

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  reduce16(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<16 x float>) nounwind readnone {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

define internal <16 x i32> @__add_varying_int32(<16 x i32>, <16 x i32>) {
  %r = add <16 x i32> %0, %1
  ret <16 x i32> %r
}

define internal i32 @__add_uniform_int32(i32, i32) {
  %r = add i32 %0, %1
  ret i32 %r
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

define internal <16 x double> @__add_varying_double(<16 x double>, <16 x double>) {
  %r = fadd <16 x double> %0, %1
  ret <16 x double> %r
}

define internal double @__add_uniform_double(double, double) {
  %r = fadd double %0, %1
  ret double %r
}

define double @__reduce_add_double(<16 x double>) nounwind readnone {
  reduce16(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<16 x double>) nounwind readnone {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

define internal <16 x i64> @__add_varying_int64(<16 x i64>, <16 x i64>) {
  %r = add <16 x i64> %0, %1
  ret <16 x i64> %r
}

define internal i64 @__add_uniform_int64(i64, i64) {
  %r = add i64 %0, %1
  ret i64 %r
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

define void @__masked_store_blend_i64(<16 x i64>* nocapture, <16 x i64>,
                                      <16 x i8> %mask) nounwind
                                      alwaysinline {
  %mask_as_i1 = trunc <16 x MASK> %mask to <16 x i1>
  %old = load PTR_OP_ARGS(`<16 x i64>')  %0, align 4
  %blend = select <16 x i1> %mask_as_i1, <16 x i64> %1, <16 x i64> %old
  store <16 x i64> %blend, <16 x i64>* %0, align 4
  ret void
}

define void @__masked_store_blend_i32(<16 x i32>* nocapture, <16 x i32>, 
                                      <16 x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <16 x MASK> %mask to <16 x i1>
  %old = load PTR_OP_ARGS(`<16 x i32>')  %0, align 4
  %blend = select <16 x i1> %mask_as_i1, <16 x i32> %1, <16 x i32> %old
  store <16 x i32> %blend, <16 x i32>* %0, align 4
  ret void
}

define void @__masked_store_blend_i16(<16 x i16>* nocapture, <16 x i16>,
                                     <16 x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <16 x MASK> %mask to <16 x i1>
  %old = load PTR_OP_ARGS(`<16 x i16>')  %0, align 4
  %blend = select <16 x i1> %mask_as_i1, <16 x i16> %1, <16 x i16> %old
  store <16 x i16> %blend, <16 x i16>* %0, align 4
  ret void
}

declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

define void @__masked_store_blend_i8(<16 x i8>* nocapture, <16 x i8>,
                                     <16 x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<16 x i8>')  %0, align 4
  %blend = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %old, <16 x i8> %1,
                                                   <16 x i8> %mask)
  store <16 x i8> %blend, <16 x i8>* %0, align 4
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

define <16 x i8> @__avg_up_uint8(<16 x i8>, <16 x i8>) nounwind readnone {
  %r = call <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %r
}

declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @__avg_up_uint16(<16 x i16>, <16 x i16>) nounwind readnone {
  v16tov8(i16, %0, %a0, %b0)
  v16tov8(i16, %1, %a1, %b1)
  %r0 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %a0, <8 x i16> %a1)
  %r1 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %b0, <8 x i16> %b1)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
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

