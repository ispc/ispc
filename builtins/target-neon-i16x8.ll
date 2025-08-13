;;
;; target-neon-16.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2025 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`MASK',`i16')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mask operations

declare <2 x i32> @NEON_PREFIX_PADDLU.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <4 x i32> @NEON_PREFIX_PADDLU.v4i32.v8i16(<WIDTH x i16>)

define i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %and_mask = and <WIDTH x i16> %0,
    <i16 1, i16 2, i16 4, i16 8, i16 16, i16 32, i16 64, i16 128>
  %v4 = call <4 x i32> @NEON_PREFIX_PADDLU.v4i32.v8i16(<8 x i16> %and_mask)
  %v2 = call <2 x i64> @NEON_PREFIX_PADDLU.v2i64.v4i32(<4 x i32> %v4)
  %va = extractelement <2 x i64> %v2, i32 0
  %vb = extractelement <2 x i64> %v2, i32 1
  %v = or i64 %va, %vb
  ret i64 %v
}

define i1 @__any(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v8tov4(MASK, %0, %v0123, %v4567)
  %vor = or <4 x MASK> %v0123, %v4567
  %v0 = extractelement <4 x MASK> %vor, i32 0
  %v1 = extractelement <4 x MASK> %vor, i32 1
  %v2 = extractelement <4 x MASK> %vor, i32 2
  %v3 = extractelement <4 x MASK> %vor, i32 3
  %v01 = or MASK %v0, %v1
  %v23 = or MASK %v2, %v3
  %v = or MASK %v01, %v23
  %cmp = icmp ne MASK %v, 0
  ret i1 %cmp
}

define i1 @__all(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v8tov4(MASK, %0, %v0123, %v4567)
  %vand = and <4 x MASK> %v0123, %v4567
  %v0 = extractelement <4 x MASK> %vand, i32 0
  %v1 = extractelement <4 x MASK> %vand, i32 1
  %v2 = extractelement <4 x MASK> %vand, i32 2
  %v3 = extractelement <4 x MASK> %vand, i32 3
  %v01 = and MASK %v0, %v1
  %v23 = and MASK %v2, %v3
  %v = and MASK %v01, %v23
  %cmp = icmp ne MASK %v, 0
  ret i1 %cmp
}

define i1 @__none(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<WIDTH x MASK> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

;; rsqrt/rcp

declare <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to8(x0, float, @NEON_PREFIX_RECPEQ.v4f32, %d)
  binary4to8(x0_nr, float, @NEON_PREFIX_RECPSQ.v4f32, %d, %x0)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  binary4to8(x1_nr, float, @NEON_PREFIX_RECPSQ.v4f32, %d, %x1)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to8(x0, float, @NEON_PREFIX_RECPEQ.v4f32, %d)
  ret <WIDTH x float> %x0
}

declare <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to8(x0, float, @NEON_PREFIX_RSQRTEQ.v4f32, %d)
  %x0_2 = fmul <WIDTH x float> %x0, %x0
  binary4to8(x0_nr, float, @NEON_PREFIX_RSQRTSQ.v4f32, %d, %x0_2)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  %x1_2 = fmul <WIDTH x float> %x1, %x1
  binary4to8(x1_nr, float, @NEON_PREFIX_RSQRTSQ.v4f32, %d, %x1_2)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to8(x0, float, @NEON_PREFIX_RSQRTEQ.v4f32, %d)
  ret <WIDTH x float> %x0
}

define float @__rsqrt_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <8 x float> undef, float %0, i32 0
  %vr = call <8 x float> @__rsqrt_varying_float(<8 x float> %vs)
  %r = extractelement <8 x float> %vr, i32 0
  ret float %r
}

define float @__rsqrt_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <8 x float> undef, float %0, i32 0
  %vr = call <8 x float> @__rsqrt_fast_varying_float(<8 x float> %vs)
  %r = extractelement <8 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <8 x float> @__rcp_varying_float(<8 x float> %vs)
  %r = extractelement <8 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_fast_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <8 x float> @__rcp_fast_varying_float(<8 x float> %vs)
  %r = extractelement <8 x float> %vr, i32 0
  ret float %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

half_uniform_conversions()

define <8 x float> @__half_to_float_varying(<8 x i16> %v) nounwind readnone alwaysinline {
  unary4to8conv(r, i16, float, @NEON_PREFIX.vcvthf2fp, %v)
  ret <8 x float> %r
}

define <8 x i16> @__float_to_half_varying(<8 x float> %v) nounwind readnone alwaysinline {
  unary4to8conv(r, float, i16, @NEON_PREFIX.vcvtfp2hf, %v)
  ret <8 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product

declare <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <8 x i32> @__dot4add_u8u8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  v8tov4(i32, %a, %a0, %a1)
  v8tov4(i32, %b, %b0, %b1)
  v8tov4(i32, %acc, %acc0, %acc1)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  v4tov8(i32, %ret0, %ret1, %ret)
  ret <8 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <8 x i32> @__dot4add_i8i8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  v8tov4(i32, %a, %a0, %a1)
  v8tov4(i32, %b, %b0, %b1)
  v8tov4(i32, %acc, %acc0, %acc1)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  v4tov8(i32, %ret0, %ret1, %ret)
  ret <8 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <8 x i32> @__dot4add_u8i8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  v8tov4(i32, %a, %a0, %a1)
  v8tov4(i32, %b, %b0, %b1)
  v8tov4(i32, %acc, %acc0, %acc1)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  v4tov8(i32, %ret0, %ret1, %ret)
  ret <8 x i32> %ret
}
