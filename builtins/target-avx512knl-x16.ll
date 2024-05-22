;;  Copyright (c) 2015-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`ISA',`AVX512KNL')

include(`target-avx512-common-16.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half
rcph_rsqrth_decl

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

;; rcp double
declare <2 x double> @llvm.x86.avx512.rcp28.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone
define double @__rcp_fast_uniform_double(double) nounwind readonly alwaysinline {
  %vecval = insertelement <2 x double> undef, double %0, i32 0
  %call = call <2 x double> @llvm.x86.avx512.rcp28.sd(<2 x double> %vecval, <2 x double> %vecval, <2 x double> undef, i8 -1, i32 8)
  %scall = extractelement <2 x double> %call, i32 0
  ret double %scall
}
define double @__rcp_uniform_double(double %v) nounwind readonly alwaysinline {
  %iv = call double @__rcp_fast_uniform_double(double %v)

  ; do one N-R iteration to improve precision
  ; iv = rcp(v)
  ; iv * (2. - v * iv)
  %v_iv = fmul double %v, %iv
  %two_minus = fsub double 2., %v_iv
  %iv_mul = fmul double %iv, %two_minus
  ret double %iv_mul
}
declare <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone
define <16 x double> @__rcp_fast_varying_double(<16 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val_hi = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = call <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double> %val_lo, <8 x double> undef, i8 -1, i32 8)
  %res_hi = call <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double> %val_hi, <8 x double> undef, i8 -1, i32 8)
  %res = shufflevector <8 x double> %res_lo, <8 x double> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}
define <16 x double> @__rcp_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %call = call <16 x double> @__rcp_fast_varying_double(<16 x double> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  double iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <16 x double> %0, %call
  %two_minus = fsub <16 x double> <double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.>, %v_iv
  %iv_mul = fmul <16 x double> %call,  %two_minus
  ret <16 x double> %iv_mul
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

;; rsqrt double
declare <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double>, <2 x double>, <2 x double>, i8, i32) nounwind readnone
define double @__rsqrt_fast_uniform_double(double) nounwind readonly alwaysinline {
  %v = insertelement <2 x double> undef, double %0, i32 0
  %vis = call <2 x double> @llvm.x86.avx512.rsqrt28.sd(<2 x double> %v, <2 x double> %v, <2 x double> undef, i8 -1, i32 8)
  %is = extractelement <2 x double> %vis, i32 0
  ret double %is
}
define double @__rsqrt_uniform_double(double) nounwind readonly alwaysinline {
  %is = call double @__rsqrt_fast_uniform_double(double %0)

  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul double %0, %is
  %v_is_is = fmul double %v_is, %is
  %three_sub = fsub double 3., %v_is_is
  %is_mul = fmul double %is, %three_sub
  %half_scale = fmul double 0.5, %is_mul
  ret double %half_scale
}
declare <8 x double> @llvm.x86.avx512.rsqrt28.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone
define <16 x double> @__rsqrt_fast_varying_double(<16 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val_hi = shufflevector <16 x double> %val, <16 x double> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res_lo = call <8 x double> @llvm.x86.avx512.rsqrt28.pd(<8 x double> %val_lo, <8 x double> undef, i8 -1, i32 8)
  %res_hi = call <8 x double> @llvm.x86.avx512.rsqrt28.pd(<8 x double> %val_hi, <8 x double> undef, i8 -1, i32 8)
  %res = shufflevector <8 x double> %res_lo, <8 x double> %res_hi, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %res
}
define <16 x double> @__rsqrt_varying_double(<16 x double> %val) nounwind readonly alwaysinline {
  %is = call <16 x double> @__rsqrt_fast_varying_double(<16 x double> %val)

  ; Newton-Raphson iteration to improve precision
  ;  double is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x double> %val,  %is
  %v_is_is = fmul <16 x double> %v_is,  %is
  %three_sub = fsub <16 x double> <double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.>, %v_is_is
  %is_mul = fmul <16 x double> %is,  %three_sub
  %half_scale = fmul <16 x double> <double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5>, %is_mul
  ret <16 x double> %half_scale
}

;;saturation_arithmetic_novec()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
dot_product_vnni_decl()
