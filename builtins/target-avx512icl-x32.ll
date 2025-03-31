;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product 
declare <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_u8i8packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot4add_u8i8packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_i16packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_i16packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}
