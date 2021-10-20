;;  Copyright (c) 2015-2021, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

define(`WIDTH',`16')
define(`ISA',`AVX512KNL')

include(`target-avx512-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt
;; On KNL use rcp14/rsqrt14 for fast versions and rcp28/rsqrt28 for regular versions.
;; And no need for Newton-Raphson iterations.

;; rcp float
declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone
define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %vecval, <4 x float> %vecval, <4 x float> undef, i8 -1)
  %scall = extractelement <4 x float> %call, i32 0
  ret float %scall
}
declare <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone
define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float> %vecval, <4 x float> %vecval, <4 x float> undef, i8 -1, i32 8)
  %scall = extractelement <4 x float> %call, i32 0
  ret float %scall
}
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone
define <16 x float> @__rcp_fast_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %0, <16 x float> undef, i16 -1)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone
define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float> %0, <16 x float> undef, i16 -1, i32 8)
  ret <16 x float> %res
}

;; rsqrt float
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone
define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %v, <4 x float> %v, <4 x float> undef, i8 -1)
  %is = extractelement <4 x float> %vis, i32 0
  ret float %is
}
declare <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone
define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %v, <4 x float> %v, <4 x float> undef, i8 -1, i32 8)
  %is = extractelement <4 x float> %vis, i32 0
  ret float %is
}
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone
define <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %v, <16 x float> undef, i16 -1)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone
define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %v, <16 x float> undef, i16 -1, i32 8)
  ret <16 x float> %res
}

;;saturation_arithmetic_novec()
