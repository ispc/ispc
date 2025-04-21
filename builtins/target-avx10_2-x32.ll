;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`ISA',`AVX10_2')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_i8i8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_i8i8packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_u8u8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_u8u8packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_u16i16packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_u16i16packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_u16u16packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_u16u16packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}