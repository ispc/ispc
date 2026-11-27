;;  Copyright (c) 2016-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`ISA',`AVX512SKX')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

;; avx512skx-x8 keeps 8-wide vectors of 64-bit types (double, int64) in zmm registers.
;; This file only overrides the builtins that operate on such vectors; everything else is
;; inherited from avx512skx-x8-nozmm (target-avx512skx-x8-nozmm.ll), which is also used
;; on its own when --opt=disable-zmm is passed.
;; The gather/scatter wrappers and __rcp/__rsqrt_varying_double call overridden builtins, so they
;; are overridden as well: otherwise linking them from the parent would also pull in the parent's
;; split implementations. The parent's __reduce_{min,max}_* still use its 256-bit min/max, which is
;; how a horizontal reduction narrows anyway.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shuffles
declare <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double>, <8 x i64>)
define <8 x double> @__shuffle_double(<8 x double> %input, <8 x i32> %perm) nounwind readnone alwaysinline {
    %indices = zext <8 x i32> %perm to <8 x i64>
    %res = call <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double> %input, <8 x i64> %indices)
    ret <8 x double> %res
}

declare <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64>, <8 x i64>)
define <8 x i64> @__shuffle_i64(<8 x i64> %input, <8 x i32> %perm) nounwind readnone alwaysinline {
    %indices = zext <8 x i32> %perm to <8 x i64>
    %res = call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %input, <8 x i64> %indices)
    ret <8 x i64> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; two-input shuffles
declare i1 @__is_compile_time_constant_varying_int32(<WIDTH x i32>)
shuffle2_const(double)
shuffle2_const(i64)

declare <8 x double> @llvm.x86.avx512.vpermi2var.pd.512(<8 x double>, <8 x i64>, <8 x double>)
define <8 x double> @__shuffle2_double(<8 x double>, <8 x double>, <8 x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<8 x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <8 x double> @__shuffle2_const_double(<8 x double> %0, <8 x double> %1, <8 x i32> %2)
  ret <8 x double> %res_const

not_const:
  %ind = zext <8 x i32> %2 to <8 x i64>
  %res = call <8 x double> @llvm.x86.avx512.vpermi2var.pd.512(<8 x double> %0, <8 x i64> %ind, <8 x double> %1)
  ret <8 x double> %res
}

declare <8 x i64> @llvm.x86.avx512.vpermi2var.q.512(<8 x i64>, <8 x i64>, <8 x i64>)
define <8 x i64> @__shuffle2_i64(<8 x i64>, <8 x i64>, <8 x i32>) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_varying_int32(<8 x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  %res_const = tail call <8 x i64> @__shuffle2_const_i64(<8 x i64> %0, <8 x i64> %1, <8 x i32> %2)
  ret <8 x i64> %res_const

not_const:
  %ind = zext <8 x i32> %2 to <8 x i64>
  %res = call <8 x i64> @llvm.x86.avx512.vpermi2var.q.512(<8 x i64> %0, <8 x i64> %ind, <8 x i64> %1)
  ret <8 x i64> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64/uint64 min/max
declare <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
declare <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)
define <8 x i64> @__max_varying_int64(<8 x i64>, <8 x i64>) nounwind readonly alwaysinline {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %0, <8 x i64> %1, <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @__max_varying_uint64(<8 x i64>, <8 x i64>) nounwind readonly alwaysinline {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxu.q.512(<8 x i64> %0, <8 x i64> %1, <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @__min_varying_int64(<8 x i64>, <8 x i64>) nounwind readonly alwaysinline {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmins.q.512(<8 x i64> %0, <8 x i64> %1, <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @__min_varying_uint64(<8 x i64>, <8 x i64>) nounwind readonly alwaysinline {
  %res = call <8 x i64> @llvm.x86.avx512.mask.pminu.q.512(<8 x i64> %0, <8 x i64> %1, <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max
declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)
define <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  %res = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %0, <8 x double> %1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)
define <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  %res = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %0, <8 x double> %1,
                <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt
declare <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone
define <8 x double> @__sqrt_varying_double(<8 x double>) nounwind alwaysinline {
  %res = call <8 x double> @llvm.x86.avx512.mask.sqrt.pd.512(<8 x double> %0,  <8 x double> zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked load/store
declare <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8*, <8 x i64>, i8)
define <8 x i64> @__masked_load_i64(i8 * %ptr, <WIDTH x MASK> %mask) readonly alwaysinline {
  %mask_i8 = bitcast <WIDTH x MASK> %mask to i8
  %res = call <8 x i64> @llvm.x86.avx512.mask.loadu.q.512(i8* %ptr, <8 x i64> zeroinitializer, i8 %mask_i8)
  ret <8 x i64> %res
}

declare <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8*, <8 x double>, i8)
define <8 x double> @__masked_load_double(i8 * %ptr, <WIDTH x MASK> %mask) readonly alwaysinline {
  %mask_i8 = bitcast <WIDTH x MASK> %mask to i8
  %res = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptr, <8 x double> zeroinitializer, i8 %mask_i8)
  ret <8 x double> %res
}

declare void @llvm.x86.avx512.mask.storeu.q.512(i8*, <8 x i64>, i8)
define void @__masked_store_i64(<8 x i64>* nocapture, <8 x i64> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i8 = bitcast <WIDTH x MASK> %mask to i8
  %ptr_i8 = bitcast <8 x i64>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.q.512(i8* %ptr_i8, <8 x i64> %v, i8 %mask_i8)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8)
define void @__masked_store_double(<8 x double>* nocapture, <8 x double> %v, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %mask_i8 = bitcast <WIDTH x MASK> %mask to i8
  %ptr_i8 = bitcast <8 x double>* %0 to i8*
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr_i8, <8 x double> %v, i8 %mask_i8)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather
declare <8 x i32> @llvm.x86.avx512.mask.gather.qpi.512(<8 x i32>, i8*, <8 x i64>, <8 x i1>, i32)
define <8 x i32>
@__gather_base_offsets64_i32(i8 * %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.qpi.512, 8, i32, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale)
  ret <8 x i32> %res
}

define <8 x i32>
@__gather64_i32(<8 x i64> %ptrs, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x i32> @__gather_base_offsets64_i32(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i1> %vecmask)
  ret <8 x i32> %res
}

declare <8 x float> @llvm.x86.avx512.mask.gather.qps.512(<8 x float>, i8*, <8 x i64>, <8 x i1>, i32)
define <8 x float>
@__gather_base_offsets64_float(i8 * %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.qps.512, 8, float, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale)
  ret <8 x float> %res
}

define <8 x float>
@__gather64_float(<8 x i64> %ptrs,  <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x float> @__gather_base_offsets64_float(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i1> %vecmask)
  ret <8 x float> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.gather.dpq.512(<8 x i64>, i8*, <8 x i32>, <8 x i1>, i32)
define <8 x i64>
@__gather_base_offsets32_i64(i8 * %ptr, i32 %offset_scale, <8 x i32> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.dpq.512, 8, i64, ptr, offsets, i32, vecmask, <8 x i1>, offset_scale)
  ret <8 x i64> %res
}

define <8 x i64>
@__gather32_i64(<8 x i32> %ptrs, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x i64> @__gather_base_offsets32_i64(i8 * zeroinitializer, i32 1, <8 x i32> %ptrs, <8 x i1> %vecmask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512(<8 x i64>, i8*, <8 x i64>, <8 x i1>, i32)
define <8 x i64>
@__gather_base_offsets64_i64(i8 * %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.qpq.512, 8, i64, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale)
  ret <8 x i64> %res
}

define <8 x i64>
@__gather64_i64(<8 x i64> %ptrs, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x i64> @__gather_base_offsets64_i64(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i1> %vecmask)
  ret <8 x i64> %res
}

declare <8 x double> @llvm.x86.avx512.mask.gather.dpd.512(<8 x double>, i8*, <8 x i32>, <8 x i1>, i32)
define <8 x double>
@__gather_base_offsets32_double(i8 * %ptr, i32 %offset_scale, <8 x i32> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.dpd.512, 8, double, ptr, offsets, i32, vecmask, <8 x i1>, offset_scale)
  ret <8 x double> %res
}

define <8 x double>
@__gather32_double(<8 x i32> %ptrs, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x double> @__gather_base_offsets32_double(i8 * zeroinitializer, i32 1, <8 x i32> %ptrs, <8 x i1> %vecmask)
  ret <8 x double> %res
}

declare <8 x double> @llvm.x86.avx512.mask.gather.qpd.512(<8 x double>, i8*, <8 x i64>, <8 x i1>, i32)
define <8 x double>
@__gather_base_offsets64_double(i8 * %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  convert_scale_to_const_gather(res, llvm.x86.avx512.mask.gather.qpd.512, 8, double, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale)
  ret <8 x double> %res
}

define <8 x double>
@__gather64_double(<8 x i64> %ptrs, <8 x i1> %vecmask) nounwind readonly alwaysinline {
  %res = call <8 x double> @__gather_base_offsets64_double(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i1> %vecmask)
  ret <8 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; scatter
declare void @llvm.x86.avx512.mask.scatter.qpi.512(i8*, <8 x i1>, <8 x i64>, <8 x i32>, i32)
define void
@__scatter_base_offsets64_i32(i8* %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i32> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpi.512, 8, vals, i32, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter64_i32(<8 x i64> %ptrs, <8 x i32> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_i32(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i32> %values, <8 x i1> %vecmask)
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qps.512(i8*, <8 x i1>, <8 x i64>, <8 x float>, i32)
define void
@__scatter_base_offsets64_float(i8* %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x float> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qps.512, 8, vals, float, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter64_float(<8 x i64> %ptrs, <8 x float> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_float(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x float> %values, <8 x i1> %vecmask)
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.dpq.512(i8*, <8 x i1>, <8 x i32>, <8 x i64>, i32)
define void
@__scatter_base_offsets32_i64(i8* %ptr, i32 %offset_scale, <8 x i32> %offsets, <8 x i64> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpq.512, 8, vals, i64, ptr, offsets, i32, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_i64(<8 x i32> %ptrs, <8 x i64> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_i64(i8 * zeroinitializer, i32 1, <8 x i32> %ptrs, <8 x i64> %values, <8 x i1> %vecmask)
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qpq.512(i8*, <8 x i1>, <8 x i64>, <8 x i64>, i32)
define void
@__scatter_base_offsets64_i64(i8* %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x i64> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpq.512, 8, vals, i64, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter64_i64(<8 x i64> %ptrs, <8 x i64> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_i64(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x i64> %values, <8 x i1> %vecmask)
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.dpd.512(i8*, <8 x i1>, <8 x i32>, <8 x double>, i32)
define void
@__scatter_base_offsets32_double(i8* %ptr, i32 %offset_scale, <8 x i32> %offsets, <8 x double> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.dpd.512, 8, vals, double, ptr, offsets, i32, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter32_double(<8 x i32> %ptrs, <8 x double> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_double(i8 * zeroinitializer, i32 1, <8 x i32> %ptrs, <8 x double> %values, <8 x i1> %vecmask)
  ret void
}

declare void @llvm.x86.avx512.mask.scatter.qpd.512(i8*, <8 x i1>, <8 x i64>, <8 x double>, i32)
define void
@__scatter_base_offsets64_double(i8* %ptr, i32 %offset_scale, <8 x i64> %offsets, <8 x double> %vals, <8 x i1> %vecmask) nounwind {
  convert_scale_to_const_scatter(llvm.x86.avx512.mask.scatter.qpd.512, 8, vals, double, ptr, offsets, i64, vecmask, <8 x i1>, offset_scale);
  ret void
}

define void
@__scatter64_double(<8 x i64> %ptrs, <8 x double> %values, <8 x i1> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_double(i8 * zeroinitializer, i32 1, <8 x i64> %ptrs, <8 x double> %values, <8 x i1> %vecmask)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision rcp/rsqrt
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>, i8) nounwind readnone
define <8 x double> @__rcp_fast_varying_double(<8 x double> %val) nounwind readonly alwaysinline {
  %res = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %val, <8 x double> poison, i8 -1)
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
  ; clamp the factor from below, as in __rcp_varying_float, so that 0 * Inf does not give NaN
  %keep = fcmp ogt <8 x double> %two_minus, <double 0.5, double 0.5, double 0.5, double 0.5,
                                             double 0.5, double 0.5, double 0.5, double 0.5>
  %factor = select <8 x i1> %keep, <8 x double> %two_minus, <8 x double> <double 0.5, double 0.5, double 0.5, double 0.5,
                                                                          double 0.5, double 0.5, double 0.5, double 0.5>
  %iv_mul = fmul <8 x double> %call,  %factor
  ret <8 x double> %iv_mul
}

declare <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double>,  <8 x double>,  i8) nounwind readnone
define <8 x double> @__rsqrt_fast_varying_double(<8 x double> %val) nounwind readonly alwaysinline {
  %res = call <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double> %val, <8 x double> poison, i8 -1)
  ret <8 x double> %res
}

declare <8 x i1> @llvm.x86.avx512.fpclass.pd.512(<8 x double>, i32)
define <8 x double> @__rsqrt_varying_double(<8 x double> %v) nounwind readonly alwaysinline {
  ; detect +/-0 and +inf to deal with them differently.
  %corner_cases = call <8 x i1> @llvm.x86.avx512.fpclass.pd.512(<8 x double> %v, i32 14)
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
