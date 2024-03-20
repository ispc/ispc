;;  Copyright (c) 2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`ISA',`AVX512SKX')

include(`target-avx512-common-8.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half
rcph_rsqrth_decl

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt

rcp14_uniform()
;; rcp float
declare <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone
define <8 x float> @__rcp_fast_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %ret = call <8 x float> @llvm.x86.avx512.rcp14.ps.256(<8 x float> %0, <8 x float> undef, i8 -1)
  ret <8 x float> %ret
}
define <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %call = call <8 x float> @__rcp_fast_varying_float(<8 x float> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  float iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <8 x float> %0, %call
  %two_minus = fsub <8 x float> <float 2., float 2., float 2., float 2.,
                                 float 2., float 2., float 2., float 2.>, %v_iv
  %iv_mul = fmul <8 x float> %call,  %two_minus
  ret <8 x float> %iv_mul
}

;; rcp double
declare <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone
define <8 x double> @__rcp_fast_varying_double(<8 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <8 x double> %val, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val_hi = shufflevector <8 x double> %val, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %res_lo = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %val_lo, <4 x double> undef, i8 -1)
  %res_hi = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %val_hi, <4 x double> undef, i8 -1)
  %res = shufflevector <4 x double> %res_lo, <4 x double> %res_hi, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %res
}
define <8 x double> @__rcp_varying_double(<8 x double>) nounwind readonly alwaysinline {
  %call = call <8 x double> @__rcp_fast_varying_double(<8 x double> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  double iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <8 x double> %0, %call
  %two_minus = fsub <8 x double> <double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.>, %v_iv
  %iv_mul = fmul <8 x double> %call,  %two_minus
  ret <8 x double> %iv_mul
}

rsqrt14_uniform()
;; rsqrt float
declare <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float>,  <8 x float>,  i8) nounwind readnone
define <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  %is = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %v,  <8 x float> undef,  i8 -1)
  ; Newton-Raphson iteration to improve precision
  ;  float is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <8 x float> %v,  %is
  %v_is_is = fmul <8 x float> %v_is,  %is
  %three_sub = fsub <8 x float> <float 3., float 3., float 3., float 3.,
                                 float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <8 x float> %is,  %three_sub
  %half_scale = fmul <8 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                  float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <8 x float> %half_scale
}
define <8 x float> @__rsqrt_fast_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  %ret = call <8 x float> @llvm.x86.avx512.rsqrt14.ps.256(<8 x float> %v,  <8 x float> undef,  i8 -1)
  ret <8 x float> %ret
}

;; rsqrt double
declare <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double>,  <4 x double>,  i8) nounwind readnone
define <8 x double> @__rsqrt_fast_varying_double(<8 x double> %val) nounwind readonly alwaysinline {
  %val_lo = shufflevector <8 x double> %val, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val_hi = shufflevector <8 x double> %val, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %res_lo = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %val_lo, <4 x double> undef, i8 -1)
  %res_hi = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %val_hi, <4 x double> undef, i8 -1)
  %res = shufflevector <4 x double> %res_lo, <4 x double> %res_hi, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %res
}
declare <4 x i1> @llvm.x86.avx512.fpclass.pd.256(<4 x double>, i32)
define <8 x double> @__rsqrt_varying_double(<8 x double> %v) nounwind readonly alwaysinline {
  ; detect +/-0 and +inf to deal with them differently.
  %val_lo = shufflevector <8 x double> %v, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val_hi = shufflevector <8 x double> %v, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %corner_cases_lo = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256(<4 x double> %val_lo, i32 14)
  %corner_cases_hi = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256(<4 x double> %val_hi, i32 14)
  %corner_cases = shufflevector <4 x i1> %corner_cases_lo, <4 x i1> %corner_cases_hi, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %is = call <8 x double> @__rsqrt_fast_varying_double(<8 x double> %v)

  ; Precision refinement sequence based on minimax approximation.
  ; This sequence is a little slower than Newton-Raphson, but has much better precision
  ; Relative error is around 3 ULPs.
  ; t1 = 1.0 - (v * is) * is
  ; t2 = 0.37500000407453632 + t1 * 0.31250000550062401
  ; t3 = 0.5 + t1 * t2
  ; t4 = is + (t1*is) * t3
  %v_is = fmul <8 x double> %v,  %is
  %v_is_is = fmul <8 x double> %v_is,  %is
  %t1 = fsub <8 x double> <double 1., double 1., double 1., double 1., double 1., double 1., double 1., double 1.>, %v_is_is
  %t1_03125 = fmul <8 x double> <double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401>, %t1
  %t2 = fadd <8 x double> <double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632>, %t1_03125
  %t1_t2 = fmul <8 x double> %t1, %t2
  %t3 = fadd <8 x double> <double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5, double 0.5>, %t1_t2
  %t1_is = fmul <8 x double> %t1, %is
  %t1_is_t3 = fmul <8 x double> %t1_is, %t3
  %t4 = fadd <8 x double> %is, %t1_is_t3
  %ret = select <8 x i1> %corner_cases, <8 x double> %is, <8 x double> %t4
  ret <8 x double> %ret
}

;;saturation_arithmetic_novec()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8i8packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot4add_u8i8packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpbusds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}

declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot2add_i16packed(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpwssd.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>) nounwind readnone
define <8 x i32> @__dot2add_i16packed_sat(<8 x i32> %a, <8 x i32> %b, <8 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <8 x i32> @llvm.x86.avx512.vpdpwssds.256(<8 x i32> %acc, <8 x i32> %a, <8 x i32> %b)
  ret <8 x i32> %ret
}
