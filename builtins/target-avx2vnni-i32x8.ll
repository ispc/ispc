;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8i8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8i8packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot2add_i16i16packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot2add_i16i16packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
