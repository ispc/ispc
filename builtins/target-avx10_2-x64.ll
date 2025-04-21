;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`64')
define(`ISA',`AVX10_2')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_i8i8packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_i8i8packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8u8packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8u8packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_u16i16packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_u16i16packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_u16u16packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_u16u16packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}