;;  Copyright (c) 2020-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`64')
define(`ISA',`AVX512SKX')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half precision rcp and rsqrt using native SPR instriuctions
;; they will be invoked only when hardware support is available.

declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rcp_uniform_half(half) nounwind readonly alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rcp = tail call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rcp, i32 0
  ret half %ret
}

declare <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half>, <32 x half>, i32)
define <64 x half> @__rcp_varying_half(<64 x half>) nounwind readonly alwaysinline {
  %vec1 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %vec2 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %rcp1 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %vec1, <32 x half> undef, i32 -1)
  %rcp2 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %vec2, <32 x half> undef, i32 -1)
  %ret = shufflevector <32 x half> %rcp1, <32 x half> %rcp2,
                       <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                    i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  ret <64 x half> %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rsqrt_uniform_half(half) nounwind readnone alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rsqrt, i32 0
  ret half %ret
}

declare <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half>, <32 x half>, i32)
define <64 x half> @__rsqrt_varying_half(<64 x half>) nounwind readnone alwaysinline {
  %vec1 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %vec2 = shufflevector <64 x half> %0, <64 x half> undef,
                        <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %rsqrt1 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %vec1, <32 x half> undef, i32 -1)
  %rsqrt2 = tail call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %vec2, <32 x half> undef, i32 -1)
  %ret = shufflevector <32 x half> %rsqrt1, <32 x half> %rsqrt2,
                       <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                    i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  ret <64 x half> %ret
}
