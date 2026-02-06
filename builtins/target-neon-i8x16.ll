;;
;; target-neon-8.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2026 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`MASK',`i8')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mask operations

declare <8 x i16> @NEON_PREFIX_PADDLU.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @NEON_PREFIX_PADDLU.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @NEON_PREFIX_PADDLU.v2i64.v4i32(<4 x i32>) nounwind readnone

define i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %and_mask = and <WIDTH x i8> %0,
    <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128,
     i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128>
  %v8 = call <8 x i16> @NEON_PREFIX_PADDLU.v8i16.v16i8(<16 x i8> %and_mask)
  %v4 = call <4 x i32> @NEON_PREFIX_PADDLU.v4i32.v8i16(<8 x i16> %v8)
  %v2 = call <2 x i64> @NEON_PREFIX_PADDLU.v2i64.v4i32(<4 x i32> %v4)
  %va = extractelement <2 x i64> %v2, i32 0
  %vb = extractelement <2 x i64> %v2, i32 1
  %vbshift = shl i64 %vb, 8
  %v = or i64 %va, %vbshift
  ret i64 %v
}

define i1 @__any(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v16tov8(MASK, %0, %v8a, %v8b)
  %vor8 = or <8 x MASK> %v8a, %v8b
  %v16 = sext <8 x i8> %vor8 to <8 x i16>
  v8tov4(i16, %v16, %v16a, %v16b)
  %vor16 = or <4 x i16> %v16a, %v16b
  %v32 = sext <4 x i16> %vor16 to <4 x i32>
  v4tov2(i32, %v32, %v32a, %v32b)
  %vor32 = or <2 x i32> %v32a, %v32b
  %v0 = extractelement <2 x i32> %vor32, i32 0
  %v1 = extractelement <2 x i32> %vor32, i32 1
  %v = or i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v16tov8(MASK, %0, %v8a, %v8b)
  %vand8 = and <8 x MASK> %v8a, %v8b
  %v16 = sext <8 x i8> %vand8 to <8 x i16>
  v8tov4(i16, %v16, %v16a, %v16b)
  %vand16 = and <4 x i16> %v16a, %v16b
  %v32 = sext <4 x i16> %vand16 to <4 x i32>
  v4tov2(i32, %v32, %v32a, %v32b)
  %vand32 = and <2 x i32> %v32a, %v32b
  %v0 = extractelement <2 x i32> %vand32, i32 0
  %v1 = extractelement <2 x i32> %vand32, i32 1
  %v = and i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__none(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<WIDTH x MASK> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt/rcp

declare <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(x0, float, @NEON_PREFIX_RECPEQ.v4f32, %d)
  binary4to16(x0_nr, float, @NEON_PREFIX_RECPSQ.v4f32, %d, %x0)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  binary4to16(x1_nr, float, @NEON_PREFIX_RECPSQ.v4f32, %d, %x1)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(ret, float, @NEON_PREFIX_RECPEQ.v4f32, %d)
  ret <WIDTH x float> %ret
}

declare <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(x0, float, @NEON_PREFIX_RSQRTEQ.v4f32, %d)
  %x0_2 = fmul <WIDTH x float> %x0, %x0
  binary4to16(x0_nr, float, @NEON_PREFIX_RSQRTSQ.v4f32, %d, %x0_2)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  %x1_2 = fmul <WIDTH x float> %x1, %x1
  binary4to16(x1_nr, float, @NEON_PREFIX_RSQRTSQ.v4f32, %d, %x1_2)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(ret, float, @NEON_PREFIX_RSQRTEQ.v4f32, %d)
  ret <WIDTH x float> %ret
}

define float @__rsqrt_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <16 x float> @__rsqrt_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rsqrt_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <16 x float> undef, float %0, i32 0
  %vr = call <16 x float> @__rsqrt_fast_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <16 x float> @__rcp_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <16 x float> undef, float %0, i32 0
  %vr = call <16 x float> @__rcp_fast_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

half_uniform_conversions()

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone alwaysinline {
  unary4to16conv(r, i16, float, @NEON_PREFIX.vcvthf2fp, %v)
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone alwaysinline {
  unary4to16conv(r, float, i16, @NEON_PREFIX.vcvtfp2hf, %v)
  ret <16 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product

declare <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <16 x i32> @__dot4add_u8u8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov4(i32, %a, %a0, %a1, %a2, %a3)
  v16tov4(i32, %b, %b0, %b1, %b2, %b3)
  v16tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  v4tov16(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <16 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <16 x i32> @__dot4add_i8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov4(i32, %a, %a0, %a1, %a2, %a3)
  v16tov4(i32, %b, %b0, %b1, %b2, %b3)
  v16tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  v4tov16(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <16 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov4(i32, %a, %a0, %a1, %a2, %a3)
  v16tov4(i32, %b, %b0, %b1, %b2, %b3)
  v16tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  v4tov16(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <16 x i32> %ret
}
