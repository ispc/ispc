;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`MASK',`i1')
define(`ISA',`AVX512SKX')

include(`util.m4')
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
define <4 x i32> @__dot2add_i16i16packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot2add_i16i16packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}

declare <WIDTH x i32> @llvm.ctpop.TYPE_SUFFIX(i32)(<WIDTH x i32>) nounwind readnone
declare <WIDTH x i64> @llvm.ctpop.TYPE_SUFFIX(i64)(<WIDTH x i64>) nounwind readnone

define <WIDTH x i32> @__popcnt_int32_varying(<WIDTH x i32> %0, <WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %call = call <WIDTH x i32> @llvm.ctpop.TYPE_SUFFIX(i32)(<WIDTH x i32> %0)
  %masked = select <WIDTH x i1> %mask, <WIDTH x i32> %call, <WIDTH x i32> zeroinitializer
  ret <WIDTH x i32> %masked
}

define <WIDTH x i32> @__popcnt_int64_varying(<WIDTH x i64> %0, <WIDTH x MASK> %mask)  nounwind readnone alwaysinline {
  %call = call <WIDTH x i64> @llvm.ctpop.TYPE_SUFFIX(i64)(<WIDTH x i64> %0)
  %trunc = trunc <WIDTH x i64> %call to <WIDTH x i32>
  %masked = select <WIDTH x i1> %mask, <WIDTH x i32> %trunc, <WIDTH x i32> zeroinitializer
  ret <WIDTH x i32> %masked
}