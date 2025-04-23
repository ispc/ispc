;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`ISA',`AVX10_2')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <8 x i32> @llvm.x86.avx2.vpdpbssd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_i8i8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx2.vpdpbssd.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx2.vpdpbssds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_i8i8packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx2.vpdpbssds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx2.vpdpbuud.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8u8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx2.vpdpbuud.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx2.vpdpbuuds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8u8packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx2.vpdpbuuds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}

