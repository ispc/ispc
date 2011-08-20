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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; *** Untested *** AVX target implementation.
;;
;; The LLVM AVX code generator is incomplete, so the ispc AVX target
;; hasn't yet been tested.  There is therefore a higher-than-normal
;; chance that there are bugs in the code in this file.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

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
;; rounding floats

declare <4 x float> @llvm.x86.sse.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

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
  %xr = call <4 x float> @llvm.x86.sse.round.ss(<4 x float> %xi, <4 x float> %xi, i32 8)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <4 x float> @llvm.x86.sse.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %xr = call <4 x float> @llvm.x86.sse.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define internal double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define internal double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
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
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

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

declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define internal float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fastmath

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
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

define internal float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.max.ss, %0, %1)
  ret float %ret
}

define internal float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, float, @llvm.x86.sse.min.ss, %0, %1)
  ret float %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define internal i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret i32 %ret
}

define internal i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret i32 %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

define internal i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret i32 %ret
}

define internal i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse.sqrt.sd(<2 x double>) nounwind readnone

define internal double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse.sqrt.sd, %0)
  ret double %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

define internal double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.min.sd, %0, %1)
  ret double %ret
}

define internal double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  sse_binary_scalar(ret, 2, double, @llvm.x86.sse2.max.sd, %0, %1)
  ret double %ret
}
