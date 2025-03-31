;;  Copyright (c) 2016-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
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

declare <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half>, <16 x half>, i16)
define <16 x half> @__rcp_varying_half(<16 x half>) nounwind readonly alwaysinline {
  %ret = tail call <16 x half> @llvm.x86.avx512fp16.mask.rcp.ph.256(<16 x half> %0, <16 x half> undef, i16 -1)
  ret <16 x half> %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rsqrt_uniform_half(half) nounwind readnone alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rsqrt, i32 0
  ret half %ret
}

declare <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half>, <16 x half>, i16)
define <16 x half> @__rsqrt_varying_half(<16 x half>) nounwind readnone alwaysinline {
  %ret = tail call <16 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.256(<16 x half> %0, <16 x half> undef, i16 -1)
  ret <16 x half> %ret
}
