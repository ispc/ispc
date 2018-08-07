;;  Copyright (c) 2016-2018, Intel Corporation
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


ifelse(LLVM_VERSION, LLVM_3_8,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_3_9,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_4_0,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_5_0,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_6_0,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_7_0,
    `include(`target-avx512-common.ll')',
         LLVM_VERSION, LLVM_8_0,
    `include(`target-avx512-common.ll')'
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt

define(`rcp_rsqrt_varying_float_skx',`
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone
define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %call = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %0, <16 x float> undef, i16 -1)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  float iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <16 x float> %0`,' %call
  %two_minus = fsub <16 x float> <float 2.`,' float 2.`,' float 2.`,' float 2.`,'
                                  float 2.`,' float 2.`,' float 2.`,' float 2.`,'
                                  float 2.`,' float 2.`,' float 2.`,' float 2.`,'
                                  float 2.`,' float 2.`,' float 2.`,' float 2.>`,' %v_iv
  %iv_mul = fmul <16 x float> %call`,'  %two_minus
  ret <16 x float> %iv_mul
}
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>`,'  <16 x float>`,'  i16) nounwind readnone
define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %is = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %v`,'  <16 x float> undef`,'  i16 -1)
  ; Newton-Raphson iteration to improve precision
  ;  float is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x float> %v`,'  %is
  %v_is_is = fmul <16 x float> %v_is`,'  %is
  %three_sub = fsub <16 x float> <float 3.`,' float 3.`,' float 3.`,' float 3.`,'
                                  float 3.`,' float 3.`,' float 3.`,' float 3.`,'
                                  float 3.`,' float 3.`,' float 3.`,' float 3.`,'
                                  float 3.`,' float 3.`,' float 3.`,' float 3.>`,' %v_is_is
  %is_mul = fmul <16 x float> %is`,'  %three_sub
  %half_scale = fmul <16 x float> <float 0.5`,' float 0.5`,' float 0.5`,' float 0.5`,'
                                   float 0.5`,' float 0.5`,' float 0.5`,' float 0.5`,'
                                   float 0.5`,' float 0.5`,' float 0.5`,' float 0.5`,'
                                   float 0.5`,' float 0.5`,' float 0.5`,' float 0.5>`,' %is_mul
  ret <16 x float> %half_scale
}
')

ifelse(LLVM_VERSION, LLVM_3_8,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_3_9,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_4_0,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_5_0,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_6_0,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_7_0,
    rcp_rsqrt_varying_float_skx(),
         LLVM_VERSION, LLVM_8_0,
    rcp_rsqrt_varying_float_skx()
  )

;;saturation_arithmetic_novec()
