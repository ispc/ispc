;;  Copyright (c) 2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Same target as target-avx2-i32x16 but with native VNNI
include(`target-avx2-common-i32x16.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a0, %a1)
  v16tov8(i32, %b, %b0, %b1)
  v16tov8(i32, %acc, %acc0, %acc1)
  %ret0 = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc0, <8 x i32> %a0, <8 x i32> %b0)
  %ret1 = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc1, <8 x i32> %a1, <8 x i32> %b1)
  v8tov16(i32, %ret0, %ret1, %ret)
  ret <16 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a0, %a1)
  v16tov8(i32, %b, %b0, %b1)
  v16tov8(i32, %acc, %acc0, %acc1)
  %ret0 = call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc0, <8 x i32> %a0, <8 x i32> %b0)
  %ret1= call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc1, <8 x i32> %a1, <8 x i32> %b1)
  v8tov16(i32, %ret0, %ret1, %ret)
  ret <16 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a0, %a1)
  v16tov8(i32, %b, %b0, %b1)
  v16tov8(i32, %acc, %acc0, %acc1)
  %ret0 = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc0, <8 x i32> %a0, <8 x i32> %b0)
  %ret1 = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc1, <8 x i32> %a1, <8 x i32> %b1)
  v8tov16(i32, %ret0, %ret1, %ret)
  ret <16 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a0, %a1)
  v16tov8(i32, %b, %b0, %b1)
  v16tov8(i32, %acc, %acc0, %acc1)
  %ret0 = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc0, <8 x i32> %a0, <8 x i32> %b0)
  %ret1 = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc1, <8 x i32> %a1, <8 x i32> %b1)
  v8tov16(i32, %ret0, %ret1, %ret)
  ret <16 x i32> %ret
}