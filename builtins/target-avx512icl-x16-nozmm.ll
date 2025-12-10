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
declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a_lo, %a_hi)
  v16tov8(i32, %b, %b_lo, %b_hi)
  v16tov8(i32, %acc, %acc_lo, %acc_hi)
  %ret_lo = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc_lo, <8 x i32> %a_lo, <8 x i32> %b_lo)
  %ret_hi = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc_hi, <8 x i32> %a_hi, <8 x i32> %b_hi)
  v8tov16(i32, %ret_lo, %ret_hi, %ret)
  ret <16 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot4add_u8i8packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a_lo, %a_hi)
  v16tov8(i32, %b, %b_lo, %b_hi)
  v16tov8(i32, %acc, %acc_lo, %acc_hi)
  %ret_lo = call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc_lo, <8 x i32> %a_lo, <8 x i32> %b_lo)
  %ret_hi = call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc_hi, <8 x i32> %a_hi, <8 x i32> %b_hi)
  v8tov16(i32, %ret_lo, %ret_hi, %ret)
  ret <16 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16i16packed(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a_lo, %a_hi)
  v16tov8(i32, %b, %b_lo, %b_hi)
  v16tov8(i32, %acc, %acc_lo, %acc_hi)
  %ret_lo = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc_lo, <8 x i32> %a_lo, <8 x i32> %b_lo)
  %ret_hi = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc_hi, <8 x i32> %a_hi, <8 x i32> %b_hi)
  v8tov16(i32, %ret_lo, %ret_hi, %ret)
  ret <16 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <16 x i32> @__dot2add_i16i16packed_sat(<16 x i32> %a, <16 x i32> %b, <16 x i32> %acc) nounwind readnone alwaysinline {
  v16tov8(i32, %a, %a_lo, %a_hi)
  v16tov8(i32, %b, %b_lo, %b_hi)
  v16tov8(i32, %acc, %acc_lo, %acc_hi)
  %ret_lo = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc_lo, <8 x i32> %a_lo, <8 x i32> %b_lo)
  %ret_hi = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc_hi, <8 x i32> %a_hi, <8 x i32> %b_hi)
  v8tov16(i32, %ret_lo, %ret_hi, %ret)
  ret <16 x i32> %ret
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