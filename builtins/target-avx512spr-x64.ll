;;  Copyright (c) 2020-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`64')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()
halfTypeGenericImplementation()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stub for mask conversion. LLVM's intrinsics want i1 mask, but we use i8

define i64 @__cast_mask_to_i64 (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i64 = bitcast <WIDTH x i1> %mask to i64
  ret i64 %mask_i64
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %res = call i64 @__cast_mask_to_i64 (<WIDTH x MASK> %mask)
  ret i64 %res
}

define i1 @__any(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i64 @__cast_mask_to_i64 (<WIDTH x MASK> %mask)
  %res = icmp ne i64 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i64 @__cast_mask_to_i64 (<WIDTH x MASK> %mask)
  %res = icmp eq i64 %intmask, -1
  ret i1 %res
}

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i64 @__cast_mask_to_i64 (<WIDTH x MASK> %mask)
  %res = icmp eq i64 %intmask, 0
  ret i1 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shift/shuffle

define_shuffles()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

;; same as avx2 and avx512skx
;; TODO: hoist to some utility file?

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %vv)
  %r = extractelement <8 x float> %rv, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  ; round to nearest even
  %rv = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv, i32 0)
  %r = extractelement <8 x i16> %rv, i32 0
  ret i16 %r
}

declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %source, <16 x float> %write_through, i16 %mask, i32) nounwind readonly
declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %source, i32, <16 x i16> %write_through, i16 %mask) nounwind readonly

define <64 x float> @__half_to_float_varying(<64 x i16> %v) nounwind readnone alwaysinline {
  v64tov16(i16, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v0, <16 x float> undef, i16 -1, i32 4)
  %r1 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v1, <16 x float> undef, i16 -1, i32 4)
  %r2 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v2, <16 x float> undef, i16 -1, i32 4)
  %r3 = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v3, <16 x float> undef, i16 -1, i32 4)
  v16tov64(float, %r0, %r1, %r2, %r3, %r)
  ret <64 x float> %r
}

define <64 x i16> @__float_to_half_varying(<64 x float> %v) nounwind readnone alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v0, i32 0, <16 x i16> undef, i16 -1)
  %r1 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v1, i32 0, <16 x i16> undef, i16 -1)
  %r2 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v2, i32 0, <16 x i16> undef, i16 -1)
  %r3 = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v3, i32 0, <16 x i16> undef, i16 -1)
  v16tov64(i16, %r0, %r1, %r2, %r3, %r)
  ret <64 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half precision rcp and rsqrt using native SPR instriuctions
;; they will be invoked only when hardware support is available.

declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rcp_uniform_half(half) nounwind readonly alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rcp = tail call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rcp, i32 0
  ret half %ret
}

declare <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half>, <32 x half>, i32)
define <64 x half> @__rcp_varying_half(<64 x half>) nounwind readonly alwaysinline {
  %vec1 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %vec2 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %rcp1 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %vec1, <32 x half> undef, i32 -1)
  %rcp2 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %vec2, <32 x half> undef, i32 -1)
  %ret = shufflevector <32 x half> %rcp1, <32 x half> %rcp2,
                       <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                    i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  ret <64 x half> %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rsqrt_uniform_half(half) nounwind readnone alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rsqrt, i32 0
  ret half %ret
}

declare <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half>, <32 x half>, i32)
define <64 x half> @__rsqrt_varying_half(<64 x half>) nounwind readnone alwaysinline {
  %vec1 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %vec2 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %rsqrt1 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %vec1, <32 x half> undef, i32 -1)
  %rsqrt2 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %vec2, <32 x half> undef, i32 -1)
  %ret = shufflevector <32 x half> %rsqrt1, <32 x half> %rsqrt2,
                       <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                    i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  ret <64 x half> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode
fastMathFTZDAZ_x86()

;; round/floor/ceil

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; round/floor/ceil uniform float

;; TODO implement through native LLVM intrinsics for round/floor/ceil float/double

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
;; round/floor/ceil uniform doubles

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
;; round/floor/ceil varying float/doubles

declare <16 x float> @llvm.roundeven.v16f32(<16 x float> %p)
declare <16 x float> @llvm.floor.v16f32(<16 x float> %p)
declare <16 x float> @llvm.ceil.v16f32(<16 x float> %p)

define <64 x float> @__round_varying_float(<64 x float> %v) nounwind readonly alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v1)
  %r2 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v2)
  %r3 = call <16 x float> @llvm.roundeven.v16f32(<16 x float> %v3)
  v16tov64(float, %r0, %r1, %r2, %r3, %r)
  ret <64 x float> %r
}

define <64 x float> @__floor_varying_float(<64 x float> %v) nounwind readonly alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v1)
  %r2 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v2)
  %r3 = call <16 x float> @llvm.floor.v16f32(<16 x float> %v3)
  v16tov64(float, %r0, %r1, %r2, %r3, %r)
  ret <64 x float> %r
}

define <64 x float> @__ceil_varying_float(<64 x float> %v) nounwind readonly alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v1)
  %r2 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v2)
  %r3 = call <16 x float> @llvm.ceil.v16f32(<16 x float> %v3)
  v16tov64(float, %r0, %r1, %r2, %r3, %r)
  ret <64 x float> %r
}

declare <8 x double> @llvm.roundeven.v8f64(<8 x double> %p)
declare <8 x double> @llvm.floor.v8f64(<8 x double> %p)
declare <8 x double> @llvm.ceil.v8f64(<8 x double> %p)

define <64 x double> @__round_varying_double(<64 x double> %v) nounwind readonly alwaysinline {
  v64tov8(double, %v, %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7)
  %r0 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v3)
  %r4 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v4)
  %r5 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v5)
  %r6 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v6)
  %r7 = call <8 x double> @llvm.roundeven.v8f64(<8 x double> %v7)
  v8tov64(double, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r)
  ret <64 x double> %r
}

define <64 x double> @__floor_varying_double(<64 x double> %v) nounwind readonly alwaysinline {
  v64tov8(double, %v, %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7)
  %r0 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v3)
  %r4 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v4)
  %r5 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v5)
  %r6 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v6)
  %r7 = call <8 x double> @llvm.floor.v8f64(<8 x double> %v7)
  v8tov64(double, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r)
  ret <64 x double> %r
}

define <64 x double> @__ceil_varying_double(<64 x double> %v) nounwind readonly alwaysinline {
  v64tov8(double, %v, %v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7)
  %r0 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v0)
  %r1 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v1)
  %r2 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v2)
  %r3 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v3)
  %r4 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v4)
  %r5 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v5)
  %r6 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v6)
  %r7 = call <8 x double> @llvm.ceil.v8f64(<8 x double> %v7)
  v8tov64(double, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r)
  ret <64 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;; min/max
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

;; TODO: these are from neon-common, need to make all of them standard utils.
;; TODO: remove int64-minmax from utils
;; TODO: ogt vs ugt?
;; TODO: do uint32/int32 versions through comparison

define float @__max_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__min_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ult float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define i32 @__min_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp slt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp sgt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ult i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ugt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp olt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp olt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

define <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp ogt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

;; int32/uint32/float versions
define <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i32> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i32> %0, <WIDTH x i32> %1
  ret <WIDTH x i32> %r
}

define <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %m = fcmp olt <WIDTH x float> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x float> %0, <WIDTH x float> %1
  ret <WIDTH x float> %r
}

define <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %m = fcmp ogt <WIDTH x float> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x float> %0, <WIDTH x float> %1
  ret <WIDTH x float> %r
}

;; sqrt/rsqrt/rcp

;; implementation note: sqrt uses native LLVM intrinsics
;declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone
declare float @llvm.sqrt.f32(float %Val)
define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @llvm.sqrt.f32(float %0)
  ret float %ret
}

declare <16 x float> @llvm.sqrt.v16f32(<16 x float> %Val)
define <64 x float> @__sqrt_varying_float(<64 x float> %v) nounwind readnone alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v0)
  %r1 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v1)
  %r2 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v2)
  %r3 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v3)
  v16tov64(float, %r0, %r1, %r2, %r3, %r)
  ret <64 x float> %r
}

declare double @llvm.sqrt.f64(double %Val)
define double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @llvm.sqrt.f64(double %0)
  ret double %ret
}

declare <16 x double> @llvm.sqrt.v16f64(<16 x double> %Val)
define <64 x double> @__sqrt_varying_double(<64 x double> %v) nounwind readnone alwaysinline {
  v64tov16(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %v0)
  %r1 = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %v1)
  %r2 = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %v2)
  %r3 = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %v3)
  v16tov64(double, %r0, %r1, %r2, %r3, %r)
  ret <64 x double> %r
}

;; TODO: need to use intrinsics and N-R approximation.
define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %s = call float @llvm.sqrt.f32(float %0)
  %ret = fdiv float 1., %s
  ret float %ret
}

define <64 x float> @__rsqrt_varying_float(<64 x float> %v) nounwind readnone alwaysinline {
  v64tov16(float, %v, %v0, %v1, %v2, %v3)
  %r0 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v0)
  %r0r = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>, %r0
  %r1 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v1)
  %r1r = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>, %r1
  %r2 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v2)
  %r2r = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>, %r2
  %r3 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v3)
  %r3r = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>, %r3
  v16tov64(float, %r0r, %r1r, %r2r, %r3r, %r)
  ret <64 x float> %r
}

;; TODO: need to use intrinsics
define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @__rsqrt_uniform_float(float %0)
  ret float %ret
}

define <64 x float> @__rsqrt_fast_varying_float(<64 x float> %v) nounwind readnone alwaysinline {
  %ret = call <64 x float> @__rsqrt_varying_float(<64 x float> %v)
  ret <64 x float> %ret
}

;; TODO: need to use intrinsics and N-R approximation.
define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %ret = fdiv float 1., %0
  ret float %ret
}

define <64 x float> @__rcp_varying_float(<64 x float> %v) nounwind readnone alwaysinline {
  %ret = fdiv <64 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.,
                            float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.,
                            float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.,
                            float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>,
                            %v
  ret <64 x float> %ret
}

;; TODO: need to use intrinsics
define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @__rcp_uniform_float(float %0)
  ret float %ret
}

define <64 x float> @__rcp_fast_varying_float(<64 x float> %v) nounwind readnone alwaysinline {
  %ret = call <64 x float> @__rcp_varying_float(<64 x float> %v)
  ret <64 x float> %ret
}


;declare float @__rsqrt_uniform_float(float) nounwind readnone
;declare float @__rcp_uniform_float(float) nounwind readnone
;declare float @__rcp_fast_uniform_float(float) nounwind readnone
;declare float @__rsqrt_fast_uniform_float(float) nounwind readnone
;declare <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone
;declare <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone
;declare <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readnone
;declare <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float>) nounwind readnone

;declare float @__sqrt_uniform_float(float) nounwind readnone
;declare <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone
;declare double @__sqrt_uniform_double(double) nounwind readnone
;declare <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone

;; bit ops

popcnt()
ctlztz()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

;declare i64 @__movmsk(<WIDTH x i1>) nounwind readnone
;declare i1 @__any(<WIDTH x i1>) nounwind readnone
;declare i1 @__all(<WIDTH x i1>) nounwind readnone
;declare i1 @__none(<WIDTH x i1>) nounwind readnone

;declare i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone
;declare i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone

;declare float @__reduce_add_float(<WIDTH x float>) nounwind readnone
;declare float @__reduce_min_float(<WIDTH x float>) nounwind readnone
;declare float @__reduce_max_float(<WIDTH x float>) nounwind readnone

;declare i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone alwaysinline
;declare i32 @__reduce_min_int32(<WIDTH x i32>) nounwind readnone
;declare i32 @__reduce_max_int32(<WIDTH x i32>) nounwind readnone
;declare i32 @__reduce_min_uint32(<WIDTH x i32>) nounwind readnone
;declare i32 @__reduce_max_uint32(<WIDTH x i32>) nounwind readnone

;declare double @__reduce_add_double(<WIDTH x double>) nounwind readnone
;declare double @__reduce_min_double(<WIDTH x double>) nounwind readnone
;declare double @__reduce_max_double(<WIDTH x double>) nounwind readnone

;declare i64 @__reduce_add_int64(<WIDTH x i64>) nounwind readnone
;declare i64 @__reduce_min_int64(<WIDTH x i64>) nounwind readnone
;declare i64 @__reduce_max_int64(<WIDTH x i64>) nounwind readnone
;declare i64 @__reduce_min_uint64(<WIDTH x i64>) nounwind readnone
;declare i64 @__reduce_max_uint64(<WIDTH x i64>) nounwind readnone

;; 8 bit
declare <8 x i64> @llvm.x86.avx512.psad.bw.512(<64 x i8>, <64 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<64 x i8>) nounwind readnone alwaysinline {
  %rv = call <8 x i64> @llvm.x86.avx512.psad.bw.512(<64 x i8> %0,
                                                    <64 x i8> zeroinitializer)
  %r0 = extractelement <8 x i64> %rv, i32 0
  %r1 = extractelement <8 x i64> %rv, i32 1
  %r2 = extractelement <8 x i64> %rv, i32 2
  %r3 = extractelement <8 x i64> %rv, i32 3
  %r4 = extractelement <8 x i64> %rv, i32 4
  %r5 = extractelement <8 x i64> %rv, i32 5
  %r6 = extractelement <8 x i64> %rv, i32 6
  %r7 = extractelement <8 x i64> %rv, i32 7
  %r01 = add i64 %r0, %r1
  %r23 = add i64 %r2, %r3
  %r45 = add i64 %r4, %r5
  %r67 = add i64 %r6, %r7
  %r0123 = add i64 %r01, %r23
  %r4567 = add i64 %r45, %r67
  %r = add i64 %r0123, %r4567
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

;; 16 bit
;; TODO: why returning i16?
define internal <64 x i16> @__add_varying_i16(<64 x i16>,
                                              <64 x i16>) nounwind readnone alwaysinline {
  %r = add <64 x i16> %0, %1
  ret <64 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<64 x i16>) nounwind readnone alwaysinline {
  reduce64(i16, @__add_varying_i16, @__add_uniform_i16)
}

;; 32 bit
;; TODO: why returning i32?
define internal <64 x i32> @__add_varying_int32(<64 x i32>,
                                                <64 x i32>) nounwind readnone alwaysinline {
  %s = add <64 x i32> %0, %1
  ret <64 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<64 x i32>) nounwind readnone alwaysinline {
  reduce64(i32, @__add_varying_int32, @__add_uniform_int32)
}

;; float
;; TODO: __reduce_add_float may use hadd
define internal <64 x float> @__add_varying_float(<64 x float>,
                                                  <64 x float>) nounwind readnone alwaysinline {
  %s = fadd <64 x float> %0, %1
  ret <64 x float> %s
}

define internal float @__add_uniform_float(float, float) nounwind readnone alwaysinline {
  %s = fadd float %0, %1
  ret float %s
}

define float @__reduce_add_float(<64 x float>) nounwind readonly alwaysinline {
  reduce64(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<64 x float>) nounwind readnone alwaysinline {
  reduce64(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<64 x float>) nounwind readnone alwaysinline {
  reduce64(float, @__max_varying_float, @__max_uniform_float)
}

;; 32 bit min/max
define i32 @__reduce_min_int32(<64 x i32>) nounwind readnone alwaysinline {
  reduce64(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<64 x i32>) nounwind readnone alwaysinline {
  reduce64(i32, @__max_varying_int32, @__max_uniform_int32)
}


define i32 @__reduce_min_uint32(<64 x i32>) nounwind readnone alwaysinline {
  reduce64(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<64 x i32>) nounwind readnone alwaysinline {
  reduce64(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;; double

define internal <64 x double> @__add_varying_double(<64 x double>,
                                                   <64 x double>) nounwind readnone alwaysinline {
  %s = fadd <64 x double> %0, %1
  ret <64 x double> %s
}

define internal double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %s = fadd double %0, %1
  ret double %s
}

define double @__reduce_add_double(<64 x double>) nounwind readonly alwaysinline {
  reduce64(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<64 x double>) nounwind readnone alwaysinline {
  reduce64(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<64 x double>) nounwind readnone alwaysinline {
  reduce64(double, @__max_varying_double, @__max_uniform_double)
}

;; int64

define internal <64 x i64> @__add_varying_int64(<64 x i64>,
                                                <64 x i64>) nounwind readnone alwaysinline {
  %s = add <64 x i64> %0, %1
  ret <64 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<64 x i64>) nounwind readnone alwaysinline {
  reduce64(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<64 x i64>) nounwind readnone alwaysinline {
  reduce64(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<64 x i64>) nounwind readnone alwaysinline {
  reduce64(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<64 x i64>) nounwind readnone alwaysinline {
  reduce64(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<64 x i64>) nounwind readnone alwaysinline {
  reduce64(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

masked_store_float_double()

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

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i64> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather(i8)
gen_gather(i16)
gen_gather(half)
gen_gather(i32)
gen_gather(float)
gen_gather(i64)
gen_gather(double)

define(`scatterbo32_64', `
define void @__scatter_base_offsets32_$1(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x MASK> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_$1(i8* %ptr, <WIDTH x i32> %offsets,
      i32 %scale, <WIDTH x i32> zeroinitializer, <WIDTH x $1> %vals, <WIDTH x MASK> %mask)
  ret void
}

define void @__scatter_base_offsets64_$1(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x $1> %vals, <WIDTH x MASK> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_$1(i8* %ptr, <WIDTH x i64> %offsets,
      i32 %scale, <WIDTH x i64> zeroinitializer, <WIDTH x $1> %vals, <WIDTH x MASK> %mask)
  ret void
}
')

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

scatterbo32_64(i8)
scatterbo32_64(i16)
scatterbo32_64(half)
scatterbo32_64(i32)
scatterbo32_64(float)
scatterbo32_64(i64)
scatterbo32_64(double)

;; TODO better intrinsic implementation is available
packed_load_and_store(FALSE)
;declare i32 @__packed_load_active(i8 * nocapture, i8 * nocapture,
;                                  <WIDTH x i1>) nounwind
;declare i32 @__packed_store_active(i8 * nocapture, <WIDTH x i32> %vals,
;                                   <WIDTH x i1>) nounwind
;declare i32 @__packed_store_active2(i8 * nocapture, <WIDTH x i32> %vals,
;                                   <WIDTH x i1>) nounwind


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

;; TODO: need to defined with intrinsics.
define_prefetches()

;declare void @__prefetch_read_uniform_1(i8 * nocapture) nounwind
;declare void @__prefetch_read_uniform_2(i8 * nocapture) nounwind
;declare void @__prefetch_read_uniform_3(i8 * nocapture) nounwind
;declare void @__prefetch_read_uniform_nt(i8 * nocapture) nounwind

;declare void @__prefetch_read_varying_1(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_1_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_2(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_2_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_3(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_3_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_nt(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
;declare void @__prefetch_read_varying_nt_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()

saturation_arithmetic_novec()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8i8packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8i8packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_i16packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_i16packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}
