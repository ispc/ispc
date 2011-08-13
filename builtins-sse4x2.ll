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


;; This file defines the target for "double-pumped" SSE4, i.e. running
;; with 8-wide vectors

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standard 8-wide definitions from m4 macros

stdlib_core(8)
packed_load_and_store(8)
scans(8)
int64minmax(8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

define internal <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
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

define internal float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
;    uniform float iv = extract(__rcp_u(v), 0);
;    return iv * (2. - v * iv);
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %vecval)
  %scall = extractelement <4 x float> %call, i32 0

  ; do one N-R iteration
  %v_iv = fmul float %0, %scall
  %two_minus = fsub float 2., %v_iv  
  %iv_mul = fmul float %scall, %two_minus
  ret float %iv_mul
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define internal <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
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

define internal float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  ;  uniform float is = extract(__rsqrt_u(v), 0);
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %v)
  %is = extractelement <4 x float> %vis, i32 0

  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul float %0, %is
  %v_is_is = fmul float %v_is, %is
  %three_sub = fsub float 3., %v_is_is
  %is_mul = fmul float %is, %three_sub
  %half_scale = fmul float 0.5, %is_mul
  ret float %half_scale
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define internal <8 x float> @__sqrt_varying_float(<8 x float>) nounwind readonly alwaysinline {
  unary4to8(call, float, @llvm.x86.sse.sqrt.ps, %0)
  ret <8 x float> %call
}

define internal float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math

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


define internal <8 x float> @__svml_sin(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_sinf4, %0)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_cos(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_cosf4, %0)
  ret <8 x float> %ret
}

define internal void @__svml_sincos(<8 x float>, <8 x float> *,
                                    <8 x float> *) nounwind readnone alwaysinline {
  ; call svml_sincosf4 two times with the two 4-wide sub-vectors
  %a = shufflevector <8 x float> %0, <8 x float> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %b = shufflevector <8 x float> %0, <8 x float> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>

  %cospa = alloca <4 x float>
  %sa = call <4 x float> @__svml_sincosf4(<4 x float> * %cospa, <4 x float> %a)

  %cospb = alloca <4 x float>
  %sb = call <4 x float> @__svml_sincosf4(<4 x float> * %cospb, <4 x float> %b)

  %sin = shufflevector <4 x float> %sa, <4 x float> %sb,
         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
                    i32 4, i32 5, i32 6, i32 7>
  store <8 x float> %sin, <8 x float> * %1

  %cosa = load <4 x float> * %cospa
  %cosb = load <4 x float> * %cospb
  %cos = shufflevector <4 x float> %cosa, <4 x float> %cosb,
         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
                    i32 4, i32 5, i32 6, i32 7>
  store <8 x float> %cos, <8 x float> * %2

  ret void
}

define internal <8 x float> @__svml_tan(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_tanf4, %0)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_atan(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_atanf4, %0)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_atan2(<8 x float>,
                                          <8 x float>) nounwind readnone alwaysinline {
  binary4to8(ret, float, @__svml_atan2f4, %0, %1)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_exp(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_expf4, %0)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_log(<8 x float>) nounwind readnone alwaysinline {
  unary4to8(ret, float, @__svml_logf4, %0)
  ret <8 x float> %ret
}

define internal <8 x float> @__svml_pow(<8 x float>,
                                        <8 x float>) nounwind readnone alwaysinline {
  binary4to8(ret, float, @__svml_powf4, %0, %1)
  ret <8 x float> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

define internal <8 x float> @__max_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.max.ps, %0, %1)
  ret <8 x float> %call
}

define internal float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.max.ss, %0, %1)
  ret float %ret
}

define internal <8 x float> @__min_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.min.ps, %0, %1)
  ret <8 x float> %call
}

define internal float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.min.ss, %0, %1)
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define internal <8 x i32> @__min_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <8 x i32> %call
}

define internal i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret i32 %ret
}

define internal <8 x i32> @__max_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <8 x i32> %call
}

define internal i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

define internal <8 x i32> @__min_varying_uint32(<8 x i32>,
                                                <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <8 x i32> %call
}

define internal i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret i32 %ret
}

define internal <8 x i32> @__max_varying_uint32(<8 x i32>,
                                                <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(call, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <8 x i32> %call
}

define internal i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret i32 %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

define internal i32 @__movmsk(<8 x i32>) nounwind readnone alwaysinline {
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
  ret i32 %v
}

define internal float @__reduce_min_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8by4(float, @llvm.x86.sse.min.ps, @__min_uniform_float)
}

define internal float @__reduce_max_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8by4(float, @llvm.x86.sse.max.ps, @__max_uniform_float)
}

; helper function for reduce_add_int32
define internal <4 x i32> @__vec4_add_int32(<4 x i32> %v0,
                                            <4 x i32> %v1) nounwind readnone alwaysinline {
  %v = add <4 x i32> %v0, %v1
  ret <4 x i32> %v
}

; helper function for reduce_add_int32
define internal i32 @__add_int32(i32, i32) nounwind readnone alwaysinline {
  %v = add i32 %0, %1
  ret i32 %v
}

define internal i32 @__reduce_add_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @__vec4_add_int32, @__add_int32)
}

define internal i32 @__reduce_min_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pminsd, @__min_uniform_int32)
}

define internal i32 @__reduce_max_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pmaxsd, @__max_uniform_int32)
}

define internal i32 @__reduce_add_uint32(<8 x i32> %v) nounwind readnone alwaysinline {
  %r = call i32 @__reduce_add_int32(<8 x i32> %v)
  ret i32 %r
}

define internal i32 @__reduce_min_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pminud, @__min_uniform_uint32)
}

define internal i32 @__reduce_max_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @llvm.x86.sse41.pmaxud, @__max_uniform_uint32)
}

define internal <4 x double> @__add_varying_double(<4 x double>,
                                     <4 x double>) nounwind readnone alwaysinline {
  %r = fadd <4 x double> %0, %1
  ret <4 x double> %r
}

define internal double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %r = fadd double %0, %1
  ret double %r
}

define internal double @__reduce_add_double(<8 x double>) nounwind readnone {
  reduce8by4(double, @__add_varying_double, @__add_uniform_double)
}

define internal double @__reduce_min_double(<8 x double>) nounwind readnone {
  reduce8(double, @__min_varying_double, @__min_uniform_double)
}

define internal double @__reduce_max_double(<8 x double>) nounwind readnone {
  reduce8(double, @__max_varying_double, @__max_uniform_double)
}

define internal <4 x i64> @__add_varying_int64(<4 x i64>,
                                               <4 x i64>) nounwind readnone alwaysinline {
  %r = add <4 x i64> %0, %1
  ret <4 x i64> %r
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %r = add i64 %0, %1
  ret i64 %r
}

define internal i64 @__reduce_add_int64(<8 x i64>) nounwind readnone {
  reduce8by4(i64, @__add_varying_int64, @__add_uniform_int64)
}

define internal i64 @__reduce_min_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_int64, @__min_uniform_int64)
}

define internal i64 @__reduce_max_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_int64, @__max_uniform_int64)
}

define internal i64 @__reduce_min_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define internal i64 @__reduce_max_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

load_and_broadcast(8, i8, 8)
load_and_broadcast(8, i16, 16)
load_and_broadcast(8, i32, 32)
load_and_broadcast(8, i64, 64)

load_masked(8, i8,  8,  1)
load_masked(8, i16, 16, 2)
load_masked(8, i32, 32, 4)
load_masked(8, i64, 64, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather(8, i8)
gen_gather(8, i16)
gen_gather(8, i32)
gen_gather(8, i64)

gen_scatter(8, i8)
gen_scatter(8, i16)
gen_scatter(8, i32)
gen_scatter(8, i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float rounding

declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

define internal <8 x float> @__round_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round4to8(%0, 8)
}

define internal float @__round_uniform_float(float) nounwind readonly alwaysinline {
  ; roundss, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  ; the roundss intrinsic is a total mess--docs say:
  ;
  ;  __m128 _mm_round_ss (__m128 a, __m128 b, const int c)
  ;       
  ;  b is a 128-bit parameter. The lowest 32 bits are the result of the rounding function
  ;  on b0. The higher order 96 bits are copied directly from input parameter a. The
  ;  return value is described by the following equations:
  ;
  ;  r0 = RND(b0)
  ;  r1 = a1
  ;  r2 = a2
  ;  r3 = a3
  ;
  ;  It doesn't matter what we pass as a, since we only need the r0 value
  ;  here.  So we pass the same register for both.  
  %xi = insertelement <4 x float> undef, float %0, i32 0
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 8)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal <8 x float> @__floor_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  round4to8(%0, 9)
}

define internal float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal <8 x float> @__ceil_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  round4to8(%0, 10)
}

define internal float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone
declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define internal <8 x double> @__round_varying_double(<8 x double>) nounwind readonly alwaysinline {
  round2to8double(%0, 8)
}

define internal double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define internal <8 x double> @__floor_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  round2to8double(%0, 9)
}

define internal double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define internal <8 x double> @__ceil_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  round2to8double(%0, 10)
}

define internal double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 10)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

define internal i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define internal i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define internal float @__reduce_add_float(<8 x float>) nounwind readonly alwaysinline {
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

gen_masked_store(8, i8, 8)
gen_masked_store(8, i16, 16)
gen_masked_store(8, i32, 32)
gen_masked_store(8, i64, 64)

masked_store_blend_8_16_by_8()

declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>,
                                             <4 x float>) nounwind readnone

define void @__masked_store_blend_32(<8 x i32>* nocapture, <8 x i32>, 
                                     <8 x i32> %mask) nounwind alwaysinline {
  ; do two 4-wide blends with blendvps
  %mask_as_float = bitcast <8 x i32> %mask to <8 x float>
  %mask_a = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %mask_b = shufflevector <8 x float> %mask_as_float, <8 x float> undef,
                <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %oldValue = load <8 x i32>* %0, align 4
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

define void @__masked_store_blend_64(<8 x i64>* nocapture %ptr, <8 x i64> %new,
                                     <8 x i32> %mask) nounwind alwaysinline {
  ; implement this as 4 blends of <4 x i32>s, which are actually bitcast
  ; <2 x i64>s...

  %mask_as_float = bitcast <8 x i32> %mask to <8 x float>

  %old = load <8 x i64>* %ptr, align 8

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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define internal <8 x double> @__sqrt_varying_double(<8 x double>) nounwind alwaysinline {
  unary2to8(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <8 x double> %ret
}


define internal double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret double %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision float min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

define internal <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <8 x double> %ret
}

define internal double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret double %ret
}

define internal <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <8 x double> %ret
}

define internal double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret double %ret

}
