;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`MASK',`i1')
define(`ISA',`AVX512SKX')

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; popcnt
declare <WIDTH x i32> @llvm.ctpop.v`'WIDTH`'i32(<WIDTH x i32>) nounwind readnone
declare <WIDTH x i64> @llvm.ctpop.v`'WIDTH`'i64(<WIDTH x i64>) nounwind readnone

define <WIDTH x i32> @__popcnt_int32_varying(<WIDTH x i32>, <WIDTH x MASK>) nounwind readonly alwaysinline {
  %call = call <WIDTH x i32> @llvm.ctpop.v`'WIDTH`'i32(<WIDTH x i32> %0)
  %result = select <WIDTH x MASK> %1, <WIDTH x i32> %call, <WIDTH x i32> zeroinitializer
  ret <WIDTH x i32> %call
}

define <WIDTH x i32> @__popcnt_int64_varying(<WIDTH x i64>, <WIDTH x MASK>) nounwind readonly alwaysinline {
  %call = call <WIDTH x i64> @llvm.ctpop.v`'WIDTH`'i64(<WIDTH x i64> %0)
  %trunc = trunc <WIDTH x i64> %call to <WIDTH x i32>
  %result = select <WIDTH x MASK> %1, <WIDTH x i32> %trunc, <WIDTH x i32> zeroinitializer
  ret <WIDTH x i32> %trunc
}
