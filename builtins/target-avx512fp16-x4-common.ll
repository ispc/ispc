;;  Copyright (c) 2016-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half precision rcp and rsqrt using native AVX512FP16 instructions
;; they will be invoked only when hardware support is available.

declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rcp_uniform_half(half) nounwind readonly alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rcp = tail call <8 x half> @llvm.x86.avx512fp16.mask.rcp.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rcp, i32 0
  ret half %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half>, <8 x half>, i8)
define <4 x half> @__rcp_varying_half(<4 x half>) nounwind readonly alwaysinline {
  %vec = shufflevector <4 x half> %0, <4 x half> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %rcp = tail call <8 x half> @llvm.x86.avx512fp16.mask.rcp.ph.128(<8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = shufflevector <8 x half> %rcp, <8 x half> undef,
                       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x half> %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half>, <8 x half>, <8 x half>, i8)
define half @__rsqrt_uniform_half(half) nounwind readnone alwaysinline {
  %vec = insertelement <8 x half> undef, half %0, i32 0
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.sh(<8 x half> %vec, <8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = extractelement <8 x half> %rsqrt, i32 0
  ret half %ret
}

declare <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half>, <8 x half>, i8)
define <4 x half> @__rsqrt_varying_half(<4 x half>) nounwind readnone alwaysinline {
  %vec = shufflevector <4 x half> %0, <4 x half> undef,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %rsqrt = tail call <8 x half> @llvm.x86.avx512fp16.mask.rsqrt.ph.128(<8 x half> %vec, <8 x half> undef, i8 -1)
  %ret = shufflevector <8 x half> %rsqrt, <8 x half> undef,
                       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x half> %ret
}
