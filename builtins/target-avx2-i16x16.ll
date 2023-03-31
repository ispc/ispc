;;  Copyright (c) 2020-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`MASK',`i16')
define(`HAVE_GATHER',`1')
define(`ISA',`AVX2')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()
halfTypeGenericImplementation()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Stub for mask conversion. LLVM's intrinsics want i1 mask, but we use i8

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone

define i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask) alwaysinline {
  %mask_i8 = trunc <16 x MASK> %mask to <16 x i8>
  %m = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %mask_i8)
  %m16 = trunc i32 %m to i16
  ret i16 %m16
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %res32 = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = zext i16 %res32 to i64
  ret i64 %res
}

define i1 @__any(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp ne i16 %intmask, 0
  ret i1 %res
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp eq i16 %intmask, -1
  ret i1 %res
}

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %intmask = call i16 @__cast_mask_to_i16 (<WIDTH x MASK> %mask)
  %res = icmp eq i16 %intmask, 0
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

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone {
  v16tov8(i16, %v, %v0, %v1)
  %r0 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %v0)
  %r1 = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %v1)
  v8tov16(float, %r0, %r1, %r)
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone {
  v16tov8(float, %v, %v0, %v1)
  %r0 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %v0, i32 0)
  %r1 = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %v1, i32 0)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
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

declare <8 x float> @llvm.nearbyint.v8f32(<8 x float> %p)
declare <8 x float> @llvm.floor.v8f32(<8 x float> %p)
declare <8 x float> @llvm.ceil.v8f32(<8 x float> %p)

define <16 x float> @__round_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  v16tov8(float, %v, %v0, %v1)
  %r0 = call <8 x float> @llvm.nearbyint.v8f32(<8 x float> %v0)
  %r1 = call <8 x float> @llvm.nearbyint.v8f32(<8 x float> %v1)
  v8tov16(float, %r0, %r1, %r)
  ret <16 x float> %r
}

define <16 x float> @__floor_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  v16tov8(float, %v, %v0, %v1)
  %r0 = call <8 x float> @llvm.floor.v8f32(<8 x float> %v0)
  %r1 = call <8 x float> @llvm.floor.v8f32(<8 x float> %v1)
  v8tov16(float, %r0, %r1, %r)
  ret <16 x float> %r
}

define <16 x float> @__ceil_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  v16tov8(float, %v, %v0, %v1)
  %r0 = call <8 x float> @llvm.ceil.v8f32(<8 x float> %v0)
  %r1 = call <8 x float> @llvm.ceil.v8f32(<8 x float> %v1)
  v8tov16(float, %r0, %r1, %r)
  ret <16 x float> %r
}

declare <4 x double> @llvm.nearbyint.v4f64(<4 x double> %p)
declare <4 x double> @llvm.floor.v4f64(<4 x double> %p)
declare <4 x double> @llvm.ceil.v4f64(<4 x double> %p)

define <16 x double> @__round_varying_double(<16 x double> %v) nounwind readonly alwaysinline {
  v16tov4(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %v0)
  %r1 = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %v1)
  %r2 = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %v2)
  %r3 = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %v3)
  v4tov16(double, %r0, %r1, %r2, %r3, %r)
  ret <16 x double> %r
}

define <16 x double> @__floor_varying_double(<16 x double> %v) nounwind readonly alwaysinline {
  v16tov4(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <4 x double> @llvm.floor.v4f64(<4 x double> %v0)
  %r1 = call <4 x double> @llvm.floor.v4f64(<4 x double> %v1)
  %r2 = call <4 x double> @llvm.floor.v4f64(<4 x double> %v2)
  %r3 = call <4 x double> @llvm.floor.v4f64(<4 x double> %v3)
  v4tov16(double, %r0, %r1, %r2, %r3, %r)
  ret <16 x double> %r
}

define <16 x double> @__ceil_varying_double(<16 x double> %v) nounwind readonly alwaysinline {
  v16tov4(double, %v, %v0, %v1, %v2, %v3)
  %r0 = call <4 x double> @llvm.ceil.v4f64(<4 x double> %v0)
  %r1 = call <4 x double> @llvm.ceil.v4f64(<4 x double> %v1)
  %r2 = call <4 x double> @llvm.ceil.v4f64(<4 x double> %v2)
  %r3 = call <4 x double> @llvm.ceil.v4f64(<4 x double> %v3)
  v4tov16(double, %r0, %r1, %r2, %r3, %r)
  ret <16 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float/double

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
define <16 x float> @__sqrt_varying_float(<16 x float> %v) nounwind readnone alwaysinline {
  %r = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v)
  ret <16 x float> %r
}

declare double @llvm.sqrt.f64(double %Val)
define double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @llvm.sqrt.f64(double %0)
  ret double %ret
}

declare <16 x double> @llvm.sqrt.v16f64(<16 x double> %Val)
define <16 x double> @__sqrt_varying_double(<16 x double> %v) nounwind readnone alwaysinline {
  %r = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %v)
  ret <16 x double> %r
}

;; TODO: need to use intrinsics and N-R approximation.
define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %s = call float @llvm.sqrt.f32(float %0)
  %ret = fdiv float 1., %s
  ret float %ret
}

define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readnone alwaysinline {
  %r0 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %v)
  %r = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>, %r0
  ret <16 x float> %r
}

;; TODO: need to use intrinsics
define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @__rsqrt_uniform_float(float %0)
  ret float %ret
}

define <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v) nounwind readnone alwaysinline {
  %ret = call <16 x float> @__rsqrt_varying_float(<16 x float> %v)
  ret <16 x float> %ret
}

;; TODO: need to use intrinsics and N-R approximation.
define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %ret = fdiv float 1., %0
  ret float %ret
}

define <16 x float> @__rcp_varying_float(<16 x float> %v) nounwind readnone alwaysinline {
  %ret = fdiv <16 x float> <float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1., float 1.>,
                            %v
  ret <16 x float> %ret
}

;; TODO: need to use intrinsics
define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @__rcp_uniform_float(float %0)
  ret float %ret
}

define <16 x float> @__rcp_fast_varying_float(<16 x float> %v) nounwind readnone alwaysinline {
  %ret = call <16 x float> @__rcp_varying_float(<16 x float> %v)
  ret <16 x float> %ret
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
declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<16 x i8>) nounwind readnone alwaysinline {
  %ext = shufflevector <16 x i8> %0, <16 x i8> undef,
      <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                 i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %ext,
                                                    <32 x i8> zeroinitializer)
  %r0 = extractelement <4 x i64> %rv, i32 0
  %r1 = extractelement <4 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

;; 16 bit
;; TODO: why returning i16?
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

;; 32 bit
;; TODO: why returning i32?
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

;; float
;; TODO: __reduce_add_float may use hadd
define internal <16 x float> @__add_varying_float(<16 x float>,
                                                  <16 x float>) nounwind readnone alwaysinline {
  %s = fadd <16 x float> %0, %1
  ret <16 x float> %s
}

define internal float @__add_uniform_float(float, float) nounwind readnone alwaysinline {
  %s = fadd float %0, %1
  ret float %s
}

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  reduce16(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

;; 32 bit min/max
define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}


define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;; double

define internal <16 x double> @__add_varying_double(<16 x double>,
                                                   <16 x double>) nounwind readnone alwaysinline {
  %s = fadd <16 x double> %0, %1
  ret <16 x double> %s
}

define internal double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %s = fadd double %0, %1
  ret double %s
}

define double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
  reduce16(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

;; int64

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
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %old = load PTR_OP_ARGS(`<WIDTH x i8>')  %0
  %blend = select <WIDTH x i1> %mask_as_i1, <WIDTH x i8> %1, <WIDTH x i8> %old
  store <WIDTH x i8> %blend, <WIDTH x i8>* %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %old = load PTR_OP_ARGS(`<WIDTH x i16>')  %0
  %blend = select <WIDTH x i1> %mask_as_i1, <WIDTH x i16> %1, <WIDTH x i16> %old
  store <WIDTH x i16> %blend, <WIDTH x i16>* %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_as_i1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %old = load PTR_OP_ARGS(`<WIDTH x i32>')  %0
  %blend = select <WIDTH x i1> %mask_as_i1, <WIDTH x i32> %1, <WIDTH x i32> %old
  store <WIDTH x i32> %blend, <WIDTH x i32>* %0
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture, <WIDTH x i64>,
                                      <WIDTH x MASK> %mask) nounwind
                                      alwaysinline {
  %mask_as_i1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %old = load PTR_OP_ARGS(`<WIDTH x i64>')  %0
  %blend = select <WIDTH x i1> %mask_as_i1, <WIDTH x i64> %1, <WIDTH x i64> %old
  store <WIDTH x i64> %blend, <WIDTH x i64>* %0
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half
rcph_rsqrth_decl

transcendetals_decl()
trigonometry_decl()

saturation_arithmetic_novec()
