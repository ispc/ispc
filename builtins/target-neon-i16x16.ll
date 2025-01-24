;;  Copyright(c) 2025 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`MASK',`i16')
define(`ISA',`NEON')

include(`util.m4')

define(`NEON_PREFIX',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon',
        RUNTIME, `32', `llvm.arm.neon')')

define(`NEON_PREFIX_UDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.udot',
        RUNTIME, `32', `llvm.arm.neon.udot')')

define(`NEON_PREFIX_SDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.sdot',
        RUNTIME, `32', `llvm.arm.neon.sdot')')

define(`NEON_PREFIX_USDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.usdot',
        RUNTIME, `32', `llvm.arm.neon.usdot')')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16>) nounwind readnone

define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline {
  %v1 = bitcast i16 %v to <1 x i16>
  %vec = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16> %vec)
  %r = extractelement <4 x float> %h, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline {
  %v1 = bitcast float %v to <1 x float>
  %vec = shufflevector <1 x float> %v1, <1 x float> undef,
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float> %vec)
  %r = extractelement <4 x i16> %h, i32 0
  ret i16 %r
}

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