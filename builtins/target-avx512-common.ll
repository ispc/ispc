;;  Copyright (c) 2015, Intel Corporation
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

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_definition()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

define_shuffles()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

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
;; fast math mode

declare void @llvm.x86.sse.stmxcsr(i8 *) nounwind
declare void @llvm.x86.sse.ldmxcsr(i8 *) nounwind

define void @__fastmath() nounwind alwaysinline {
  %ptr = alloca i32
  %ptr8 = bitcast i32 * %ptr to i8 *
  call void @llvm.x86.sse.stmxcsr(i8 * %ptr8)
  %oldval = load PTR_OP_ARGS(`i32 ') %ptr

  ; turn on DAZ (64)/FTZ (32768) -> 32832
  %update = or i32 %oldval, 32832
  store i32 %update, i32 *%ptr
  call void @llvm.x86.sse.ldmxcsr(i8 * %ptr8)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; round/floor/ceil

declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
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
  ;  here.  So we pass the same register for both.  Further, only the 0th
  ;  element of the b parameter matters
  %xi = insertelement <4 x float> undef, float %0, i32 0
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 8)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 10)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone
define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round8to16(%0, 8)
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round8to16(%0, 9)
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round8to16(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone
define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 8)
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 9)
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

int64minmax()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

define float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %1, float %0
  ret float %ret
}

define float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %0, float %1
  ret float %ret
}

declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <16 x float> @__max_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.max.ps.256, %0, %1)
  ret <16 x float> %call
}

define <16 x float> @__min_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.min.ps.256, %0, %1)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

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
;; double precision min/max

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %0, double %1
  ret double %ret
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %1, double %0
  ret double %ret
}

declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.min.pd.256, %0, %1)
  ret <16 x double> %ret
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.max.pd.256, %0, %1)
  ret <16 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
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

declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary8to16(is, float, @llvm.x86.avx.rsqrt.ps.256, %v)
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  ; do the rcpss call
  ;    uniform float iv = extract(__rcp_u(v), 0);
  ;    return iv * (2. - v * iv);
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %vecval)
  %scall = extractelement <4 x float> %call, i32 0

  ; do one N-R iteration to improve precision, as above
  %v_iv = fmul float %0, %scall
  %two_minus = fsub float 2., %v_iv
  %iv_mul = fmul float %scall, %two_minus
  ret float %iv_mul
}

declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  unary8to16(call, float, @llvm.x86.avx.rcp.ps.256, %0)
  ; do one N-R iteration
  %v_iv = fmul <16 x float> %0, %call
  %two_minus = fsub <16 x float> <float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.>, %v_iv
  %iv_mul = fmul <16 x float> %call, %two_minus
  ret <16 x float> %iv_mul
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}

declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  unary8to16(call, float, @llvm.x86.avx.sqrt.ps.256, %0)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  sse_unary_scalar(ret, 2, double, @llvm.x86.sse2.sqrt.sd, %0)
  ret double %ret
}

declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  unary4to16(ret, double, @llvm.x86.avx.sqrt.pd.256, %0)
  ret <16 x double> %ret
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; bit ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

define i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}
ctlztz()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = zext i16 %intmask to i64
  ret i64 %res
}

define i1 @__any(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp ne i16 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp eq i16 %intmask, 65535
  ret i1 %res
}

define i1 @__none(<WIDTH x i1>) nounwind readnone alwaysinline {
  %intmask = bitcast <WIDTH x i1> %0 to i16
  %res = icmp eq i16 %intmask, 0
  ret i1 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int8/16 ops

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vb = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %va, <8 x float> %vb)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %v3 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v2, <8 x float> %v2)
  %scalar1 = extractelement <8 x float> %v3, i32 0
  %scalar2 = extractelement <8 x float> %v3, i32 4
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}

define float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define internal <16 x i32> @__add_varying_int32(<16 x i32>,
                                       <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vb = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vc = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %vd = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %vab = fadd <4 x double> %va, %vb
  %vcd = fadd <4 x double> %vc, %vd

  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %vab, <4 x double> %vcd)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %final0 = extractelement <4 x double> %sum1, i32 0
  %final1 = extractelement <4 x double> %sum1, i32 2
  %sum = fadd double %final0, %final1
  ret double %sum
}

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define internal <16 x i64> @__add_varying_int64(<16 x i64>,
                                                <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(i32, 4)
masked_load(i64, 8)

masked_load_float_double()

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

define void @__masked_store_float(<WIDTH x float> * nocapture, <WIDTH x float>,
                                  <WIDTH x MASK>) nounwind alwaysinline {
  %ptr = bitcast <WIDTH x float> * %0 to <WIDTH x i32> *
  %val = bitcast <WIDTH x float> %1 to <WIDTH x i32>
  call void @__masked_store_i32(<WIDTH x i32> * %ptr, <WIDTH x i32> %val, <WIDTH x MASK> %2)
  ret void
}

define void @__masked_store_double(<WIDTH x double> * nocapture, <WIDTH x double>,
                                   <WIDTH x MASK>) nounwind alwaysinline {
  %ptr = bitcast <WIDTH x double> * %0 to <WIDTH x i64> *
  %val = bitcast <WIDTH x double> %1 to <WIDTH x i64>
  call void @__masked_store_i64(<WIDTH x i64> * %ptr, <WIDTH x i64> %val, <WIDTH x MASK> %2)
  ret void
}

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i8> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i16> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i32> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i32> %1, <WIDTH x i32> %v
  store <WIDTH x i32> %v1, <WIDTH x i32> * %0
  ret void
}

define void @__masked_store_blend_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                        <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x float> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x float> %1, <WIDTH x float> %v
  store <WIDTH x float> %v1, <WIDTH x float> * %0
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i64> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}

define void @__masked_store_blend_double(<WIDTH x double>* nocapture,
                            <WIDTH x double>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x double> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x double> %1, <WIDTH x double> %v
  store <WIDTH x double> %v1, <WIDTH x double> * %0
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

define(`scatterbo32_64', `
define void @__scatter_base_offsets32_$1(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_$1(i8* %ptr, <16 x i32> %offsets,
      i32 %scale, <16 x i32> zeroinitializer, <16 x $1> %vals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_$1(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_$1(i8* %ptr, <16 x i64> %offsets,
      i32 %scale, <16 x i64> zeroinitializer, <16 x $1> %vals, <WIDTH x i1> %mask)
  ret void
} 
')


gen_gather(i8)
gen_gather(i16)
gen_gather(i32)
gen_gather(i64)
gen_gather(float)
gen_gather(double)

scatterbo32_64(i8)
scatterbo32_64(i16)
scatterbo32_64(i32)
scatterbo32_64(i64)
scatterbo32_64(float)
scatterbo32_64(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(i32)
gen_scatter(i64)
gen_scatter(float)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; packed_load/store

declare <16 x i32> @llvm.x86.avx512.mask.expand.load.d.512(i8* %addr, <16 x i32> %data, i16 %mask)

define i32 @__packed_load_active(i32 * %startptr, <16 x i32> * %val_ptr,
                                 <16 x i1> %full_mask) nounwind alwaysinline {
  %addr = bitcast i32* %startptr to i8*
  %data = load PTR_OP_ARGS(`<16 x i32> ') %val_ptr
  %mask = bitcast <16 x i1> %full_mask to i16
  %store_val = call <16 x i32> @llvm.x86.avx512.mask.expand.load.d.512(i8* %addr, <16 x i32> %data, i16 %mask)
  store <16 x i32> %store_val, <16 x i32> * %val_ptr
  %mask_i32 = zext i16 %mask to i32
  %res = call i32 @llvm.ctpop.i32(i32 %mask_i32)
  ret i32 %res
}

declare void @llvm.x86.avx512.mask.compress.store.d.512(i8* %addr, <16 x i32> %data, i16 %mask)

define i32 @__packed_store_active(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask) nounwind alwaysinline {
  %addr = bitcast i32* %startptr to i8*
  %mask = bitcast <16 x i1> %full_mask to i16
  call void @llvm.x86.avx512.mask.compress.store.d.512(i8* %addr, <16 x i32> %vals, i16 %mask)
  %mask_i32 = zext i16 %mask to i32
  %res = call i32 @llvm.ctpop.i32(i32 %mask_i32)
  ret i32 %res
}

define i32 @__packed_store_active2(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask) nounwind alwaysinline {
  %res = call i32 @__packed_store_active(i32 * %startptr, <16 x i32> %vals,
                                   <16 x i1> %full_mask)
  ret i32 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

define_prefetches()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()
declare_nvptx()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()
