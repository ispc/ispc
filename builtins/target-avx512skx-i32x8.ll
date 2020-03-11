;;  Copyright (c) 2016-2020, Intel Corporation
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

define(`WIDTH',`8')

include(`target-avx512-common-8.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt

define(`rcp_rsqrt_varying_float_skx_8',`
declare <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone
define <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %call = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %0, <8 x float> undef, i8 -1)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  float iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <8 x float> %0`,' %call
  %two_minus = fsub <8 x float> <float 2.`,' float 2.`,' float 2.`,' float 2.`,'
                                  float 2.`,' float 2.`,' float 2.`,' float 2.>`,' %v_iv
  %iv_mul = fmul <8 x float> %call`,'  %two_minus
  ret <8 x float> %iv_mul
}
define <8 x float> @__rcp_fast_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %ret = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %0, <8 x float> undef, i8 -1)
  ret <8 x float> %ret
}

declare <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float>`,'  <8 x float>`,'  i8) nounwind readnone
define <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  %is = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %v`,'  <8 x float> undef`,'  i8 -1)
  ; Newton-Raphson iteration to improve precision
  ;  float is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <8 x float> %v`,'  %is
  %v_is_is = fmul <8 x float> %v_is`,'  %is
  %three_sub = fsub <8 x float> <float 3.`,' float 3.`,' float 3.`,' float 3.`,'
                                  float 3.`,' float 3.`,' float 3.`,' float 3.>`,' %v_is_is
  %is_mul = fmul <8 x float> %is`,'  %three_sub
  %half_scale = fmul <8 x float> <float 0.5`,' float 0.5`,' float 0.5`,' float 0.5`,'
                                   float 0.5`,' float 0.5`,' float 0.5`,' float 0.5>`,' %is_mul
  ret <8 x float> %half_scale
}
define <8 x float> @__rsqrt_fast_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  %ret = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %v`,'  <8 x float> undef`,'  i8 -1)
  ret <8 x float> %ret
}
')

rcp_rsqrt_varying_float_skx_8()

;;saturation_arithmetic_novec()
