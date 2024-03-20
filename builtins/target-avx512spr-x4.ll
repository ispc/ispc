;;  Copyright (c) 2016-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`ISA',`AVX512SKX')

include(`target-avx512-common-4.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp, rsqrt

rcp14_uniform()
;; rcp float
declare <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float>, <4 x float>, i8) nounwind readnone
define <4 x float> @__rcp_fast_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %ret = call <4 x float> @llvm.x86.avx512.rcp14.ps.128(<4 x float> %0, <4 x float> undef, i8 -1)
  ret <4 x float> %ret
}
define <4 x float> @__rcp_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @__rcp_fast_varying_float(<4 x float> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  float iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <4 x float> %0, %call
  %two_minus = fsub <4 x float> <float 2., float 2., float 2., float 2.>, %v_iv
  %iv_mul = fmul <4 x float> %call,  %two_minus
  ret <4 x float> %iv_mul
}

;; rcp double
declare <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone
define <4 x double> @__rcp_fast_varying_double(<4 x double> %val) nounwind readonly alwaysinline {
  %res = call <4 x double> @llvm.x86.avx512.rcp14.pd.256(<4 x double> %val, <4 x double> undef, i8 -1)
  ret <4 x double> %res
}
define <4 x double> @__rcp_varying_double(<4 x double>) nounwind readonly alwaysinline {
  %call = call <4 x double> @__rcp_fast_varying_double(<4 x double> %0)
  ;; do one Newton-Raphson iteration to improve precision
  ;;  double iv = __rcp_v(v);
  ;;  return iv * (2. - v * iv);
  %v_iv = fmul <4 x double> %0, %call
  %two_minus = fsub <4 x double> <double 2., double 2., double 2., double 2.>, %v_iv
  %iv_mul = fmul <4 x double> %call,  %two_minus
  ret <4 x double> %iv_mul
}

rsqrt14_uniform()
;; rsqrt float
declare <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float>,  <4 x float>,  i8) nounwind readnone
define <4 x float> @__rsqrt_fast_varying_float(<4 x float> %v) nounwind readonly alwaysinline {
  %ret = call <4 x float> @llvm.x86.avx512.rsqrt14.ps.128(<4 x float> %v,  <4 x float> undef,  i8 -1)
  ret <4 x float> %ret
}
define <4 x float> @__rsqrt_varying_float(<4 x float> %v) nounwind readonly alwaysinline {
  %is = call <4 x float> @__rsqrt_fast_varying_float(<4 x float> %v)
  ; Newton-Raphson iteration to improve precision
  ;  float is = __rsqrt_v(v);
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <4 x float> %v,  %is
  %v_is_is = fmul <4 x float> %v_is,  %is
  %three_sub = fsub <4 x float> <float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <4 x float> %is,  %three_sub
  %half_scale = fmul <4 x float> <float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <4 x float> %half_scale
}

;; rsqrt double
declare <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double>,  <4 x double>,  i8) nounwind readnone
define <4 x double> @__rsqrt_fast_varying_double(<4 x double> %val) nounwind readonly alwaysinline {
  %res = call <4 x double> @llvm.x86.avx512.rsqrt14.pd.256(<4 x double> %val, <4 x double> undef, i8 -1)
  ret <4 x double> %res
}
declare <4 x i1> @llvm.x86.avx512.fpclass.pd.256(<4 x double>, i32)
define <4 x double> @__rsqrt_varying_double(<4 x double> %v) nounwind readonly alwaysinline {
  %corner_cases = call <4 x i1> @llvm.x86.avx512.fpclass.pd.256(<4 x double> %v, i32 14)
  %is = call <4 x double> @__rsqrt_fast_varying_double(<4 x double> %v)

  ; Precision refinement sequence based on minimax approximation.
  ; This sequence is a little slower than Newton-Raphson, but has much better precision
  ; Relative error is around 3 ULPs.
  ; t1 = 1.0 - (v * is) * is
  ; t2 = 0.37500000407453632 + t1 * 0.31250000550062401
  ; t3 = 0.5 + t1 * t2
  ; t4 = is + (t1*is) * t3
  %v_is = fmul <4 x double> %v,  %is
  %v_is_is = fmul <4 x double> %v_is,  %is
  %t1 = fsub <4 x double> <double 1., double 1., double 1., double 1.>, %v_is_is
  %t1_03125 = fmul <4 x double> <double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401, double 0.31250000550062401>, %t1
  %t2 = fadd <4 x double> <double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632, double 0.37500000407453632>, %t1_03125
  %t1_t2 = fmul <4 x double> %t1, %t2
  %t3 = fadd <4 x double> <double 0.5, double 0.5, double 0.5, double 0.5>, %t1_t2
  %t1_is = fmul <4 x double> %t1, %is
  %t1_is_t3 = fmul <4 x double> %t1_is, %t3
  %t4 = fadd <4 x double> %is, %t1_is_t3
  %ret = select <4 x i1> %corner_cases, <4 x double> %is, <4 x double> %t4
  ret <4 x double> %ret
}

;;saturation_arithmetic_novec()

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
declare <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8i8packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpbusd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot4add_u8i8packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpbusds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}

declare <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot2add_i16packed(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssd.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
declare <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
define <4 x i32> @__dot2add_i16packed_sat(<4 x i32> %a, <4 x i32> %b, <4 x i32> %acc) nounwind readnone alwaysinline {
  %ret = call <4 x i32> @llvm.x86.avx512.vpdpwssds.128(<4 x i32> %acc, <4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %ret
}
