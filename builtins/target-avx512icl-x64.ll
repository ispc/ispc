;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`64')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8i8packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot4add_u8i8packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_i16packed(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <64 x i32> @__dot2add_i16packed_sat(<64 x i32> %a, <64 x i32> %b, <64 x i32> %acc) nounwind readnone alwaysinline {
  v64tov16(i32, %a, %a0, %a1, %a2, %a3)
  v64tov16(i32, %b, %b0, %b1, %b2, %b3)
  v64tov16(i32, %acc, %acc0, %acc1, %acc2, %acc3)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  %ret2 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc2, <16 x i32> %a2, <16 x i32> %b2)
  %ret3 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc3, <16 x i32> %a3, <16 x i32> %b3)
  v16tov64(i32, %ret0, %ret1, %ret2, %ret3, %ret)
  ret <64 x i32> %ret
}
