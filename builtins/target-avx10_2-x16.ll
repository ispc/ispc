;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`ISA',`AVX10_2')

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