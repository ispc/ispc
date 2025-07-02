;;  Copyright (c) 2024-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
define(`ISA',`AVX512SKX')

include(`util.m4')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shuffle

declare <32 x i8> @llvm.x86.avx512.mask.permvar.qi.256(<32 x i8>, <32 x i8>, <32 x i8>, i32)
define <32 x i8> @__shuffle_i8(<32 x i8> %v, <32 x i32> %perm) nounwind readnone alwaysinline {
  %ind = trunc <32 x i32> %perm to <32 x i8>
  %res = call <32 x i8> @llvm.x86.avx512.mask.permvar.qi.256(<32 x i8> %v, <32 x i8> %ind, <32 x i8> zeroinitializer, i32 -1)
  ret <32 x i8> %res
}

define_shuffle2_const()

declare <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8>, <32 x i8>, <32 x i8>)
declare i1 @__is_compile_time_constant_varying_int32(<32 x i32>)
define <32 x i8> @__shuffle2_i8(<32 x i8> %v1, <32 x i8> %v2, <32 x i32> %perm) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<32 x i32> %perm)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <32 x i8> @__shuffle2_const_i8(<32 x i8> %v1, <32 x i8> %v2, <32 x i32> %perm)
  ret <32 x i8> %res_const

not_const:
  %ind = trunc <32 x i32> %perm to <32 x i8>
  %res = call <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8> %v1, <32 x i8> %ind, <32 x i8> %v2)
  ret <32 x i8> %res
}

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
define <32 x i32> @__dot2add_i16i16packed(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}

declare <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32>, <16 x i32>, <16 x i32>) nounwind readnone
define <32 x i32> @__dot2add_i16i16packed_sat(<32 x i32> %a, <32 x i32> %b, <32 x i32> %acc) nounwind readnone alwaysinline {
  v32tov16(i32, %a, %a0, %a1)
  v32tov16(i32, %b, %b0, %b1)
  v32tov16(i32, %acc, %acc0, %acc1)
  %ret0 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc0, <16 x i32> %a0, <16 x i32> %b0)
  %ret1 = call <16 x i32> @llvm.x86.avx512.vpdpwssds.512(<16 x i32> %acc1, <16 x i32> %a1, <16 x i32> %b1)
  v16tov32(i32, %ret0, %ret1, %ret)
  ret <32 x i32> %ret
}
