;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shuffles

define_shuffle2_const()

declare i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32>)
declare <WIDTH x i8> @llvm.x86.avx512.mask.permvar.qi.128(<WIDTH x i8>, <WIDTH x i8>, <WIDTH x i8>, i16)
define <WIDTH x i8> @__shuffle_i8(<WIDTH x i8>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %ind = trunc <WIDTH x i32> %1 to <WIDTH x i8>
  %res = call <WIDTH x i8> @llvm.x86.avx512.mask.permvar.qi.128(<WIDTH x i8> %0, <WIDTH x i8> %ind, <WIDTH x i8> zeroinitializer, i16 -1)
  ret <WIDTH x i8> %res
}

declare <WIDTH x i8> @llvm.x86.avx512.vpermi2var.qi.128(<WIDTH x i8>, <WIDTH x i8>, <WIDTH x i8>)
define <WIDTH x i8> @__shuffle2_i8(<WIDTH x i8>, <WIDTH x i8>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <WIDTH x i8> @__shuffle2_const_i8(<WIDTH x i8> %0, <WIDTH x i8> %1, <WIDTH x i32> %2)
  ret <WIDTH x i8> %res_const

not_const:
  %ind = trunc <WIDTH x i32> %2 to <WIDTH x i8>
  %res = call <WIDTH x i8> @llvm.x86.avx512.vpermi2var.qi.128(<WIDTH x i8> %0, <WIDTH x i8> %ind, <WIDTH x i8> %1)
  ret <WIDTH x i8> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}
declare <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.vpdpbusds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}
declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc, <16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %ret
}
