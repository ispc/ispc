;;
;; target-neon-32.ll
;;
;;  Copyright(c) 2012-2013 Matt Pharr
;;  Copyright(c) 2013, 2015 Google, Inc.
;;  Copyright(c) 2019-2025 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`MASK',`i32')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mask operations

ifelse(
RUNTIME, `64', `
declare i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32>)

define i64 @__movmsk(<4 x i32>) nounwind readnone alwaysinline {
  %and_mask = and <4 x i32> %0, <i32 1, i32 2, i32 4, i32 8>
  %v = call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %and_mask)
  %mask64 = zext i32 %v to i64
  ret i64 %mask64
}

define i1 @__any(<4 x i32>) nounwind readnone alwaysinline {
  %v = call i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32> %0)
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<4 x i32>) nounwind readnone alwaysinline {
  %v = call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %0)
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__none(<4 x i32>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<4 x i32> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}
',
RUNTIME, `32',`
declare i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt/rcp

declare <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %x0 = call <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float> %d)
  %x0_nr = call <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float> %d, <4 x float> %x0)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_nr = call <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float> %d, <4 x float> %x1)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %ret = call <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float> %d)
  ret <4 x float> %ret
}

declare <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %x0 = call <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float> %d)
  %x0_2 = fmul <4 x float> %x0, %x0
  %x0_nr = call <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float> %d, <4 x float> %x0_2)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_2 = fmul <4 x float> %x1, %x1
  %x1_nr = call <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float> %d, <4 x float> %x1_2)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

define float @__rsqrt_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rsqrt_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %ret = call <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float> %d)
  ret <4 x float> %ret
}

define float @__rsqrt_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <4 x float> undef, float %0, i32 0
  %vr = call <4 x float> @__rsqrt_fast_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rcp_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <4 x float> undef, float %0, i32 0
  %vr = call <4 x float> @__rcp_fast_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

half_uniform_conversions()

define <4 x float> @__half_to_float_varying(<4 x i16> %v) nounwind readnone alwaysinline {
  %r = call <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16> %v)
  ret <4 x float> %r
}

define <4 x i16> @__float_to_half_varying(<4 x float> %v) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float> %v)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product

declare <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <4 x i32> @__dot4add_u8u8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %a_cast = bitcast <4 x i32> %a to <16 x i8>
  %b_cast = bitcast <4 x i32> %b to <16 x i8>
  %ret = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc, <16 x i8> %a_cast, <16 x i8> %b_cast)
  ret <4 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <4 x i32> @__dot4add_i8i8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %a_cast = bitcast <4 x i32> %a to <16 x i8>
  %b_cast = bitcast <4 x i32> %b to <16 x i8>
  %ret = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc, <16 x i8> %a_cast, <16 x i8> %b_cast)
  ret <4 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <4 x i32> @__dot4add_u8i8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %a_cast = bitcast <4 x i32> %a to <16 x i8>
  %b_cast = bitcast <4 x i32> %b to <16 x i8>
  %ret = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc, <16 x i8> %a_cast, <16 x i8> %b_cast)
  ret <4 x i32> %ret
}
