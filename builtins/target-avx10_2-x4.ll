;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`ISA',`AVX10_2')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <4 x i32> @llvm.x86.avx2.vpdpbssd.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_i8i8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx2.vpdpbssd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx2.vpdpbssds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_i8i8packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx2.vpdpbssds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}

declare <4 x i32> @llvm.x86.avx2.vpdpbuud.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8u8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx2.vpdpbuud.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx2.vpdpbuuds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8u8packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx2.vpdpbuuds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}

