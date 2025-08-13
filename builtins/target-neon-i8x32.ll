;;  Copyright(c) 2025 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`MASK',`i8')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

declare i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

half_uniform_conversions()

define <32 x float> @__half_to_float_varying(<32 x i16> %v) nounwind readnone alwaysinline {
  unary4to32conv(r, i16, float, @NEON_PREFIX.vcvthf2fp, %v)
  ret <32 x float> %r;
}

define <32 x i16> @__float_to_half_varying(<32 x float> %v) nounwind readnone alwaysinline {
  unary4to32conv(r, float, i16, @NEON_PREFIX.vcvtfp2hf, %v)
  ret <32 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product

declare <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <32 x i32> @__dot4add_u8u8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov4(i32, %a, %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7)
  v32tov4(i32, %b, %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7)
  v32tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3, %acc4, %acc5, %acc6, %acc7)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %a4_cast = bitcast <4 x i32> %a4 to <16 x i8>
  %b4_cast = bitcast <4 x i32> %b4 to <16 x i8>
  %a5_cast = bitcast <4 x i32> %a5 to <16 x i8>
  %b5_cast = bitcast <4 x i32> %b5 to <16 x i8>
  %a6_cast = bitcast <4 x i32> %a6 to <16 x i8>
  %b6_cast = bitcast <4 x i32> %b6 to <16 x i8>
  %a7_cast = bitcast <4 x i32> %a7 to <16 x i8>
  %b7_cast = bitcast <4 x i32> %b7 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  %ret4 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc4, <16 x i8> %a4_cast, <16 x i8> %b4_cast)
  %ret5 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc5, <16 x i8> %a5_cast, <16 x i8> %b5_cast)
  %ret6 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc6, <16 x i8> %a6_cast, <16 x i8> %b6_cast)
  %ret7 = call <4 x i32> @NEON_PREFIX_UDOT.v4i32.v16i8(<4 x i32> %acc7, <16 x i8> %a7_cast, <16 x i8> %b7_cast)
  v4tov32(i32, %ret0, %ret1, %ret2, %ret3, %ret4, %ret5, %ret6, %ret7, %ret)
  ret <32 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <32 x i32> @__dot4add_i8i8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov4(i32, %a, %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7)
  v32tov4(i32, %b, %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7)
  v32tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3, %acc4, %acc5, %acc6, %acc7)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %a4_cast = bitcast <4 x i32> %a4 to <16 x i8>
  %b4_cast = bitcast <4 x i32> %b4 to <16 x i8>
  %a5_cast = bitcast <4 x i32> %a5 to <16 x i8>
  %b5_cast = bitcast <4 x i32> %b5 to <16 x i8>
  %a6_cast = bitcast <4 x i32> %a6 to <16 x i8>
  %b6_cast = bitcast <4 x i32> %b6 to <16 x i8>
  %a7_cast = bitcast <4 x i32> %a7 to <16 x i8>
  %b7_cast = bitcast <4 x i32> %b7 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  %ret4 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc4, <16 x i8> %a4_cast, <16 x i8> %b4_cast)
  %ret5 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc5, <16 x i8> %a5_cast, <16 x i8> %b5_cast)
  %ret6 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc6, <16 x i8> %a6_cast, <16 x i8> %b6_cast)
  %ret7 = call <4 x i32> @NEON_PREFIX_SDOT.v4i32.v16i8(<4 x i32> %acc7, <16 x i8> %a7_cast, <16 x i8> %b7_cast)
  v4tov32(i32, %ret0, %ret1, %ret2, %ret3, %ret4, %ret5, %ret6, %ret7, %ret)
  ret <32 x i32> %ret
}

declare <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>) nounwind readnone
define <32 x i32> @__dot4add_u8i8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov4(i32, %a, %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7)
  v32tov4(i32, %b, %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7)
  v32tov4(i32, %acc, %acc0, %acc1, %acc2, %acc3, %acc4, %acc5, %acc6, %acc7)
  %a0_cast = bitcast <4 x i32> %a0 to <16 x i8>
  %b0_cast = bitcast <4 x i32> %b0 to <16 x i8>
  %a1_cast = bitcast <4 x i32> %a1 to <16 x i8>
  %b1_cast = bitcast <4 x i32> %b1 to <16 x i8>
  %a2_cast = bitcast <4 x i32> %a2 to <16 x i8>
  %b2_cast = bitcast <4 x i32> %b2 to <16 x i8>
  %a3_cast = bitcast <4 x i32> %a3 to <16 x i8>
  %b3_cast = bitcast <4 x i32> %b3 to <16 x i8>
  %a4_cast = bitcast <4 x i32> %a4 to <16 x i8>
  %b4_cast = bitcast <4 x i32> %b4 to <16 x i8>
  %a5_cast = bitcast <4 x i32> %a5 to <16 x i8>
  %b5_cast = bitcast <4 x i32> %b5 to <16 x i8>
  %a6_cast = bitcast <4 x i32> %a6 to <16 x i8>
  %b6_cast = bitcast <4 x i32> %b6 to <16 x i8>
  %a7_cast = bitcast <4 x i32> %a7 to <16 x i8>
  %b7_cast = bitcast <4 x i32> %b7 to <16 x i8>
  %ret0 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc0, <16 x i8> %a0_cast, <16 x i8> %b0_cast)
  %ret1 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc1, <16 x i8> %a1_cast, <16 x i8> %b1_cast)
  %ret2 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc2, <16 x i8> %a2_cast, <16 x i8> %b2_cast)
  %ret3 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc3, <16 x i8> %a3_cast, <16 x i8> %b3_cast)
  %ret4 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc4, <16 x i8> %a4_cast, <16 x i8> %b4_cast)
  %ret5 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc5, <16 x i8> %a5_cast, <16 x i8> %b5_cast)
  %ret6 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc6, <16 x i8> %a6_cast, <16 x i8> %b6_cast)
  %ret7 = call <4 x i32> @NEON_PREFIX_USDOT.v4i32.v16i8(<4 x i32> %acc7, <16 x i8> %a7_cast, <16 x i8> %b7_cast)
  v4tov32(i32, %ret0, %ret1, %ret2, %ret3, %ret4, %ret5, %ret6, %ret7, %ret)
  ret <32 x i32> %ret
}
