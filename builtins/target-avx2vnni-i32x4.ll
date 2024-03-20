;;  Copyright (c) 2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Same target as target-avx2-i32x4 but with native VNNI
include(`target-avx2-common-i32x4.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8i8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8i8packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}

declare <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot2add_i16packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot2add_i16packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}