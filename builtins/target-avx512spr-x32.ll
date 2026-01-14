;;  Copyright (c) 2020-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`ISA',`AVX512SKX')

include(`target-spr-amx-utils.ll')

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
define <32 x half> @__rcp_varying_half(<32 x half>) nounwind readonly alwaysinline {
  %ret = tail call <32 x half> @llvm.x86.avx512fp16.mask.rcp.ph.512(<32 x half> %0, <32 x half> undef, i32 -1)
  ret <32 x half> %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rsqrt_uniform_half(half) nounwind readnone alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rsqrt, i32 0
  ret half %ret
}

declare <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half>, <32 x half>, i32)
define <32 x half> @__rsqrt_varying_half(<32 x half>) nounwind readnone alwaysinline {
  %ret = tail call <32 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.512(<32 x half> %0, <32 x half> undef, i32 -1)
  ret <32 x half> %ret
}
