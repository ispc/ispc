;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_i8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpbssd.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}
declare <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_i8i8packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpbssds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8u8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpbuud.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}
declare <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8u8packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpbuuds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_u16u16packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpwuud.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_u16u16packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpwuuds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_u16i16packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpwusd.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_u16i16packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx10.vpdpwusds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}