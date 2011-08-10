;;  Copyright (c) 2010-2011, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

;; This file declares implementations of various stdlib builtins that
;; only require SSE version 1 and 2 functionality; this file, in turn
;; is then included by builtins-sse2.ll and builtins-sse4.ll to provide
;; those definitions for them.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

int64minmax(4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

define internal <4 x float> @__rcp_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %0)
  ; do one N-R iteration to improve precision
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);
  %v_iv = fmul <4 x float> %0, %call
  %two_minus = fsub <4 x float> <float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <4 x float> %call, %two_minus
  ret <4 x float> %iv_mul
}

define internal float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  ; do the rcpss call
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %vecval)
  %scall = extractelement <4 x float> %call, i32 0

  ; do one N-R iteration to improve precision, as above
  %v_iv = fmul float %0, %scall
  %two_minus = fsub float 2., %v_iv  
  %iv_mul = fmul float %scall, %two_minus
  ret float %iv_mul
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define internal <4 x float> @__rsqrt_varying_float(<4 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  %is = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %v)
  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <4 x float> %v, %is
  %v_is_is = fmul <4 x float> %v_is, %is
  %three_sub = fsub <4 x float> <float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <4 x float> %is, %three_sub
  %half_scale = fmul <4 x float> <float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <4 x float> %half_scale
}

define internal float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  ;  uniform float is = extract(__rsqrt_u(v), 0);
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %v)
  %is = extractelement <4 x float> %vis, i32 0

  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul float %0, %is
  %v_is_is = fmul float %v_is, %is
  %three_sub = fsub float 3., %v_is_is
  %is_mul = fmul float %is, %three_sub
  %half_scale = fmul float 0.5, %is_mul
  ret float %half_scale
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define internal <4 x float> @__sqrt_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %0)
  ret <4 x float> %call
}

define internal float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode

declare void @llvm.x86.sse.stmxcsr(i8 *) nounwind
declare void @llvm.x86.sse.ldmxcsr(i8 *) nounwind

define internal void @__fastmath() nounwind alwaysinline {
  %ptr = alloca i32
  %ptr8 = bitcast i32 * %ptr to i8 *
  call void @llvm.x86.sse.stmxcsr(i8 * %ptr8)
  %oldval = load i32 *%ptr

  ; turn on DAZ (64)/FTZ (32768) -> 32832
  %update = or i32 %oldval, 32832
  store i32 %update, i32 *%ptr
  call void @llvm.x86.sse.ldmxcsr(i8 * %ptr8)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; svml stuff

declare <4 x float> @__svml_sinf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_cosf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_sincosf4(<4 x float> *, <4 x float>) nounwind readnone
declare <4 x float> @__svml_tanf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_atanf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_atan2f4(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @__svml_expf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_logf4(<4 x float>) nounwind readnone
declare <4 x float> @__svml_powf4(<4 x float>, <4 x float>) nounwind readnone


define internal <4 x float> @__svml_sin(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_sinf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_cos(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_cosf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal void @__svml_sincos(<4 x float>, <4 x float> *, <4 x float> *) nounwind readnone alwaysinline {
  %s = call <4 x float> @__svml_sincosf4(<4 x float> * %2, <4 x float> %0)
  store <4 x float> %s, <4 x float> * %1
  ret void
}

define internal <4 x float> @__svml_tan(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_tanf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_atan(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_atanf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_atan2(<4 x float>, <4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_atan2f4(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_exp(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_expf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_log(<4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_logf4(<4 x float> %0)
  ret <4 x float> %ret
}

define internal <4 x float> @__svml_pow(<4 x float>, <4 x float>) nounwind readnone alwaysinline {
  %ret = call <4 x float> @__svml_powf4(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

define internal <4 x float> @__max_varying_float(<4 x float>, <4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %call
}

define internal float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.max.ss, %0, %1)
  ret float %ret
}

define internal <4 x float> @__min_varying_float(<4 x float>, <4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %call
}

define internal float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.min.ss, %0, %1)
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define internal <4 x double> @__sqrt_varying_double(<4 x double>) nounwind alwaysinline {
  unary2to4(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <4 x double> %ret
}


define internal double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse2.sqrt.sd, %0)
  ret double %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

define internal <4 x double> @__min_varying_double(<4 x double>, <4 x double>) nounwind readnone {
  binary2to4(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <4 x double> %ret
}


define internal double @__min_uniform_double(double, double) nounwind readnone {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.min.sd, %0, %1)
  ret double %ret
}


define internal <4 x double> @__max_varying_double(<4 x double>, <4 x double>) nounwind readnone {
  binary2to4(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <4 x double> %ret
}


define internal double @__max_uniform_double(double, double) nounwind readnone {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.max.sd, %0, %1)
  ret double %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

define internal i32 @__movmsk(<4 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i32> %0 to <4 x float>
  %v = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %floatmask) nounwind readnone
  ret i32 %v
}

define internal float @__reduce_min_float(<4 x float>) nounwind readnone {
  reduce4(float, @__min_varying_float, @__min_uniform_float)
}

define internal float @__reduce_max_float(<4 x float>) nounwind readnone {
  reduce4(float, @__max_varying_float, @__max_uniform_float)
}

define internal i32 @__reduce_add_int32(<4 x i32> %v) nounwind readnone {
  %v1 = shufflevector <4 x i32> %v, <4 x i32> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = add <4 x i32> %v1, %v
  %m1a = extractelement <4 x i32> %m1, i32 0
  %m1b = extractelement <4 x i32> %m1, i32 1
  %sum = add i32 %m1a, %m1b
  ret i32 %sum
}

define internal i32 @__reduce_min_int32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__min_varying_int32, @__min_uniform_int32)
}

define internal i32 @__reduce_max_int32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__max_varying_int32, @__max_uniform_int32)
}

define internal i32 @__reduce_add_uint32(<4 x i32> %v) nounwind readnone {
  %r = call i32 @__reduce_add_int32(<4 x i32> %v)
  ret i32 %r
}

define internal i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define internal i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__max_varying_uint32, @__max_uniform_uint32)
 }


define internal double @__reduce_add_double(<4 x double>) nounwind readnone {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = fadd <2 x double> %v0, %v1
  %e0 = extractelement <2 x double> %sum, i32 0
  %e1 = extractelement <2 x double> %sum, i32 1
  %m = fadd double %e0, %e1
  ret double %m
}

define internal double @__reduce_min_double(<4 x double>) nounwind readnone {
  reduce4(double, @__min_varying_double, @__min_uniform_double)
}

define internal double @__reduce_max_double(<4 x double>) nounwind readnone {
  reduce4(double, @__max_varying_double, @__max_uniform_double)
}

define internal i64 @__reduce_add_int64(<4 x i64>) nounwind readnone {
  %v0 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = add <2 x i64> %v0, %v1
  %e0 = extractelement <2 x i64> %sum, i32 0
  %e1 = extractelement <2 x i64> %sum, i32 1
  %m = add i64 %e0, %e1
  ret i64 %m
}

define internal i64 @__reduce_min_int64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__min_varying_int64, @__min_uniform_int64)
}

define internal i64 @__reduce_max_int64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__max_varying_int64, @__max_uniform_int64)
}

define internal i64 @__reduce_min_uint64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define internal i64 @__reduce_max_uint64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

masked_store_blend_8_16_by_4()

gen_masked_store(4, i8, 8)
gen_masked_store(4, i16, 16)
gen_masked_store(4, i32, 32)
gen_masked_store(4, i64, 64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

load_and_broadcast(4, i8, 8)
load_and_broadcast(4, i16, 16)
load_and_broadcast(4, i32, 32)
load_and_broadcast(4, i64, 64)

load_masked(4, i8,  8,  1)
load_masked(4, i16, 16, 2)
load_masked(4, i32, 32, 4)
load_masked(4, i64, 64, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

; define these with the macros from stdlib.m4

gen_gather(4, i8)
gen_gather(4, i16)
gen_gather(4, i32)
gen_gather(4, i64)

gen_scatter(4, i8)
gen_scatter(4, i16)
gen_scatter(4, i32)
gen_scatter(4, i64)
