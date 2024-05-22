;;  Copyright (c) 2019-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

target datalayout = "e-p:32:32-i64:64-n8:16:32";

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
include(`util-xe.m4')

define(`CONCAT',`$1$2')
define(`XE_TYPE',
`ifelse($1, `i1', `i1',
        $1, `i8', `i8',
        $1, `i16', `i16',
        $1, `half', `f16',
        $1, `i32', `i32',
        $1, `float', `f32',
        $1, `double', `f64',
        $1, `i64', `i64')')


define(`XE_SUFFIXN',`CONCAT(`v', CONCAT($2, XE_TYPE($1)))')

define(`SIZEOF',
`ifelse($1, `i1', 1,
        $1, `i8', 1,
        $1, `i16', 2,
        $1, `half', 2,
        $1, `i32', 4,
        $1, `float', 4,
        $1, `double', 8,
        $1, `i64', 8)')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

stdlib_core()
packed_load_and_store()
scans()
ctlztz()
define_prefetches()
define_shuffles()
aossoa()
rdrand_decls()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare float @llvm.genx.rndd.f32(float)
declare float @llvm.genx.rndu.f32(float)
declare float @llvm.genx.rnde.f32(float)
declare <WIDTH x float> @llvm.genx.rndu.XE_SUFFIX(float)(<WIDTH x float>)
declare <WIDTH x float> @llvm.genx.rndd.XE_SUFFIX(float)(<WIDTH x float>)
declare <WIDTH x float> @llvm.genx.rnde.XE_SUFFIX(float)(<WIDTH x float>)


define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.rndd.f32(float %0)
    ret float %res
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.rndu.f32(float %0)
    ret float %res
}

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
  %res = call float @llvm.genx.rnde.f32(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding 16-bit floats

define half @__floor_uniform_half(half) nounwind readonly alwaysinline {
  %conv_fp32 = fpext half %0 to float
  %res = call float @llvm.genx.rndd.f32(float %conv_fp32)
  %conv_hf = fptrunc float %res to half
  ret half %conv_hf
}

define half @__ceil_uniform_half(half) nounwind readonly alwaysinline {
  %conv_fp32 = fpext half %0 to float
  %res = call float @llvm.genx.rndu.f32(float %conv_fp32)
  %conv_hf = fptrunc float %res to half
  ret half %conv_hf
}

define half @__round_uniform_half(half) nounwind readonly alwaysinline {
  %conv_fp32 = fpext half %0 to float
  %res = call float @llvm.genx.rnde.f32(float %conv_fp32)
  %conv_hf = fptrunc float %res to half
  ret half %conv_hf
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast double %0 to i64
  %bitop.i.i = and i64 %float_to_int_bitcast.i.i.i.i, -9223372036854775808
  %bitop.i = xor i64 %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast i64 %bitop.i to double
  %binop.i = fadd double %int_to_float_bitcast.i.i40.i, 4.5036e+15
  %binop21.i = fadd double %binop.i, -4.5036e+15
  %float_to_int_bitcast.i.i.i = bitcast double %binop21.i to i64
  %bitop31.i = xor i64 %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast i64 %bitop31.i to double
  ret double %int_to_float_bitcast.i.i.i
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  %calltmp.i = tail call double @__round_uniform_double(double %0) nounwind
  %bincmp.i = fcmp ogt double %calltmp.i, %0
  %val_to_boolvec32.i = sext i1 %bincmp.i to i64
  %bitop.i = and i64 %val_to_boolvec32.i, -4616189618054758400
  %int_to_float_bitcast.i.i.i = bitcast i64 %bitop.i to double
  %binop.i = fadd double %calltmp.i, %int_to_float_bitcast.i.i.i
  ret double %binop.i
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  %calltmp.i = tail call double @__round_uniform_double(double %0) nounwind
  %bincmp.i = fcmp olt double %calltmp.i, %0
  %val_to_boolvec32.i = sext i1 %bincmp.i to i64
  %bitop.i = and i64 %val_to_boolvec32.i, 4607182418800017408
  %int_to_float_bitcast.i.i.i = bitcast i64 %bitop.i to double
  %binop.i = fadd double %calltmp.i, %int_to_float_bitcast.i.i.i
  ret double %binop.i
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp
declare half @__spirv_ocl_native_recip_DvWIDTH1Dh(half)

define half @__rcp_uniform_half(half) nounwind readonly alwaysinline {
  ;; No need to make NR iteration to improve precision since precision
  ;; on Xe is high already (1UP)
  %res = call half @__rcp_fast_uniform_half(half %0)
  ret half %res
}

define half @__rcp_fast_uniform_half(half) nounwind readonly alwaysinline {
  %res = call half @__spirv_ocl_native_recip_DvWIDTH1Dh(half %0)
  ret half %res
}

declare float @__spirv_ocl_native_recip_DvWIDTH1f(float)

define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  ;; No need to make NR iteration to improve precision since precision
  ;; on Xe is high already (1UP)
  %res = call float @__rcp_fast_uniform_float(float %0)
  ret float %res
}

define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %res = call float @__spirv_ocl_native_recip_DvWIDTH1f(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare float @__spirv_ocl_native_rsqrt_DvWIDTH1f(float)
define float @__rsqrt_uniform_float(float %v) nounwind readonly alwaysinline {
  %r = call float @__spirv_ocl_native_rsqrt_DvWIDTH1f(float %v)
  ;; Newton-Raphson iteration to improve precision
  ;;  return 0.5 * r * (3. - (v * r) * r);
  %mult = fmul float %v, %r
  %mult2 = fmul float %mult, %r
  %three_sub = fsub float 3., %mult2
  %mult3 = fmul float %r, %three_sub
  %res = fmul float 0.5, %mult3
  ret float %res
}

define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %res = call float @__spirv_ocl_native_rsqrt_DvWIDTH1f(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half precision rsqrt

declare half @__spirv_ocl_native_rsqrt_DvWIDTH1Dh(half)
define half @__rsqrt_uniform_half(half %v) nounwind readonly alwaysinline {
  %res = call half @__spirv_ocl_native_rsqrt_DvWIDTH1Dh(half %v)
  ret half %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare float @__spirv_ocl_native_sqrt_DvWIDTH1f(float)
define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %res = call float @__spirv_ocl_native_sqrt_DvWIDTH1f(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half precision sqrt

declare half @__spirv_ocl_native_sqrt_DvWIDTH1Dh(half)
define half @__sqrt_uniform_half(half) nounwind readonly alwaysinline {
  %res = call half @__spirv_ocl_native_sqrt_DvWIDTH1Dh(half %0)
  ret half %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare double @llvm.genx.ieee.sqrt.f64(double)
define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  %res = call double @llvm.genx.ieee.sqrt.f64(double %0)
  ret double %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode

;; In CPU fastmath set FTZ (flush-to-zero) and DAZ (denormals-are-zero)
;; Xe CM have per kernel setting of CM_DENORM_RTZ (Set all denorms to zero) - applied as attribute to kernel function; enabled by default
;; So in Xe fastmath enabled by default
define void @__fastmath() nounwind alwaysinline {
  ret void
}

define i32 @__set_ftz_daz_flags() nounwind alwaysinline {
  ret i32 0
}

define void @__restore_ftz_daz_flags(i32 %oldVal) nounwind alwaysinline {
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %pred = fcmp olt double %0, %1
  %res = select i1 %pred, double %0, double %1
  ret double %res
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %pred = fcmp ogt double %0, %1
  %res = select i1 %pred, double %0, double %1
  ret double %res
}

define <WIDTH x double> @__min_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind readnone {
  %pred = fcmp olt <WIDTH x double> %0, %1
  %res = select <WIDTH x i1> %pred, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %res
}

define <WIDTH x double> @__max_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind readnone {
  %pred = fcmp ogt <WIDTH x double> %0, %1
  %res = select <WIDTH x i1> %pred, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %res
}

;; Generates rdregion intrinsics needed for reductions
;; $1 LLVM IR type
define(`xe_rdregion', `
  declare <16 x $1> @llvm.genx.$2.XE_SUFFIXN($1,16).XE_SUFFIXN($1,32).i16(<32 x $1>, i32, i32, i32, i16, i32)
  declare <8 x $1> @llvm.genx.$2.XE_SUFFIXN($1,8).XE_SUFFIXN($1,16).i16(<16 x $1>, i32, i32, i32, i16, i32)
  declare <4 x $1> @llvm.genx.$2.XE_SUFFIXN($1,4).XE_SUFFIXN($1,8).i16(<8 x $1>, i32, i32, i32, i16, i32)
  declare <2 x $1> @llvm.genx.$2.XE_SUFFIXN($1,2).XE_SUFFIXN($1,4).i16(<4 x $1>, i32, i32, i32, i16, i32)
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Generates max/min builtins for unfiorm and varying
;; $1 LLVM IR type
;; $2 Xe intrinsic min name
;; $3 Xe intrinsic max name
;; $4 type-based builtin suffix
define(`xe_maxmin', `
declare $1 @llvm.genx.$2.XE_TYPE($1).XE_TYPE($1)($1, $1)
declare $1 @llvm.genx.$3.XE_TYPE($1).XE_TYPE($1)($1, $1)
declare <32 x $1> @llvm.genx.$2.XE_SUFFIXN($1, 32).XE_SUFFIXN($1, 32)(<32 x $1>, <32 x $1>)
declare <16 x $1> @llvm.genx.$2.XE_SUFFIXN($1, 16).XE_SUFFIXN($1, 16)(<16 x $1>, <16 x $1>)
declare <8 x $1> @llvm.genx.$2.XE_SUFFIXN($1, 8).XE_SUFFIXN($1, 8)(<8 x $1>, <8 x $1>)
declare <4 x $1> @llvm.genx.$2.XE_SUFFIXN($1, 4).XE_SUFFIXN($1, 4)(<4 x $1>, <4 x $1>)
declare <2 x $1> @llvm.genx.$2.XE_SUFFIXN($1, 2).XE_SUFFIXN($1, 2)(<2 x $1>, <2 x $1>)

declare <32 x $1> @llvm.genx.$3.XE_SUFFIXN($1, 32).XE_SUFFIXN($1, 32)(<32 x $1>, <32 x $1>)
declare <16 x $1> @llvm.genx.$3.XE_SUFFIXN($1, 16).XE_SUFFIXN($1, 16)(<16 x $1>, <16 x $1>)
declare <8 x $1> @llvm.genx.$3.XE_SUFFIXN($1, 8).XE_SUFFIXN($1, 8)(<8 x $1>, <8 x $1>)
declare <4 x $1> @llvm.genx.$3.XE_SUFFIXN($1, 4).XE_SUFFIXN($1, 4)(<4 x $1>, <4 x $1>)
declare <2 x $1> @llvm.genx.$3.XE_SUFFIXN($1, 2).XE_SUFFIXN($1, 2)(<2 x $1>, <2 x $1>)

define $1 @__max_uniform_$4($1, $1) nounwind readonly alwaysinline {
  %res = call $1 @llvm.genx.$3.XE_TYPE($1).XE_TYPE($1)($1 %0, $1 %1)
  ret $1 %res
}

define $1 @__min_uniform_$4($1, $1) nounwind readonly alwaysinline {
  %res = call $1 @llvm.genx.$2.XE_TYPE($1).XE_TYPE($1)($1 %0, $1 %1)
  ret $1 %res
}

define <WIDTH x $1> @__max_varying_$4(<WIDTH x $1>, <WIDTH x $1>) nounwind readonly alwaysinline {
  %res = call <WIDTH x $1> @llvm.genx.$3.XE_SUFFIX($1).XE_SUFFIX($1)(<WIDTH x $1> %0, <WIDTH x $1> %1)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1> @__min_varying_$4(<WIDTH x $1>, <WIDTH x $1>) nounwind readonly alwaysinline {
  %res = call <WIDTH x $1> @llvm.genx.$2.XE_SUFFIX($1).XE_SUFFIX($1)(<WIDTH x $1> %0, <WIDTH x $1> %1)
  ret <WIDTH x $1> %res
}
')

xe_maxmin(half, fmin, fmax, half)
xe_maxmin(float, fmin, fmax, float)
xe_maxmin(i32, smin, smax, int32)
xe_maxmin(i64, smin, smax, int64)
xe_maxmin(i32, umin, umax, uint32)
xe_maxmin(i64, umin, umax, uint64)

xe_rdregion(half, rdregionf)
xe_rdregion(float, rdregionf)
xe_rdregion(i32, rdregioni)
xe_rdregion(i64, rdregioni)

;; int8 and int16 types are processed differently so declare them in advance
declare <32 x i8> @llvm.genx.rdregioni.XE_SUFFIXN(i8,32).XE_SUFFIXN(i8, 128).i16(<128 x i8>, i32, i32, i32, i16, i32)
declare <32 x i16> @llvm.genx.rdregioni.XE_SUFFIX(i16,32).XE_SUFFIXN(i16, 64).i16(<64 x i16>, i32, i32, i32, i16, i32)
declare <16 x i8> @llvm.genx.rdregioni.XE_SUFFIXN(i8,16).XE_SUFFIXN(i8, 64).i16(<64 x i8>, i32, i32, i32, i16, i32)
declare <16 x i16> @llvm.genx.rdregioni.XE_SUFFIXN(i16,16).XE_SUFFIXN(i16,32).i16(<32 x i16>, i32, i32, i32, i16, i32)
declare <8 x i8> @llvm.genx.rdregioni.XE_SUFFIXN(i8,8).XE_SUFFIXN(i8, 32).i16(<32 x i8>, i32, i32, i32, i16, i32)
declare <8 x i16> @llvm.genx.rdregioni.XE_SUFFIX(i16,8).XE_SUFFIXN(i16, 16).i16(<16 x i16>, i32, i32, i32, i16, i32)

declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,32).XE_SUFFIXN(i64,32).XE_SUFFIXN(i16, 64)(<32 x MASK>, i32, <32 x i64>, <64 x i16>)
declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,32).XE_SUFFIXN(i64,32).XE_SUFFIXN(i8, 128)(<32 x MASK>, i32, <32 x i64>, <128 x i8>)
declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN(i16, 32)(<16 x MASK>, i32, <16 x i64>, <32 x i16>)
declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN(i8, 64)(<16 x MASK>, i32, <16 x i64>, <64 x i8>)
declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,8).XE_SUFFIXN(i64,8).XE_SUFFIXN(i16, 16)(<8 x MASK>, i32, <8 x i64>, <16 x i16>)
declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,8).XE_SUFFIXN(i64,8).XE_SUFFIXN(i8, 32)(<8 x MASK>, i32, <8 x i64>, <32 x i8>)

declare <128 x i8> @llvm.genx.svm.gather.XE_SUFFIXN(i8, 128).XE_SUFFIXN(i1,32).XE_SUFFIXN(i64,32)(<32 x MASK>, i32, <32 x i64>, <32 x i8>)
declare <64 x i16> @llvm.genx.svm.gather.XE_SUFFIXN(i16, 64).XE_SUFFIXN(i1,32).XE_SUFFIXN(i64,32)(<32 x MASK>, i32, <32 x i64>, <32 x i16>)
declare <64 x i8> @llvm.genx.svm.gather.XE_SUFFIXN(i8, 64).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK>, i32, <16 x i64>, <16 x i8>)
declare <32 x i16> @llvm.genx.svm.gather.XE_SUFFIXN(i16, 32).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK>, i32, <16 x i64>, <16 x i16>)
declare <32 x i8> @llvm.genx.svm.gather.XE_SUFFIXN(i8, 32).XE_SUFFIXN(i1,8).XE_SUFFIXN(i64,8)(<8 x MASK>, i32, <8 x i64>, <8 x i8>)
declare <16 x i16> @llvm.genx.svm.gather.XE_SUFFIXN(i16, 16).XE_SUFFIXN(i1,8).XE_SUFFIXN(i64,8)(<8 x MASK>, i32, <8 x i64>, <8 x i16>)

declare <128 x i8> @llvm.genx.wrregioni.XE_SUFFIXN(i8, 128).XE_SUFFIXN(i8,32).i16.XE_SUFFIXN(i1,32)(<128 x i8>, <32 x i8>, i32, i32, i32, i16, i32, <32 x MASK>)
declare <64 x i16> @llvm.genx.wrregioni.XE_SUFFIXN(i16, 64).XE_SUFFIXN(i16,32).i16.XE_SUFFIXN(i1,32)(<64 x i16>, <32 x i16>, i32, i32, i32, i16, i32, <32 x MASK>)
declare <64 x i8> @llvm.genx.wrregioni.XE_SUFFIXN(i8, 64).XE_SUFFIXN(i8,16).i16.XE_SUFFIXN(i1,16)(<64 x i8>, <16 x i8>, i32, i32, i32, i16, i32, <16 x MASK>)
declare <32 x i16> @llvm.genx.wrregioni.XE_SUFFIXN(i16, 32).XE_SUFFIXN(i16,16).i16.XE_SUFFIXN(i1,16)(<32 x i16>, <16 x i16>, i32, i32, i32, i16, i32, <16 x MASK>)
declare <32 x i8> @llvm.genx.wrregioni.XE_SUFFIXN(i8, 32).XE_SUFFIXN(i8,8).i16.XE_SUFFIXN(i1,8)(<32 x i8>, <8 x i8>, i32, i32, i32, i16, i32, <8 x MASK>)
declare <16 x i16> @llvm.genx.wrregioni.XE_SUFFIXN(i16, 16).XE_SUFFIXN(i16,8).i16.XE_SUFFIXN(i1,8)(<16 x i16>, <8 x i16>, i32, i32, i32, i16, i32, <8 x MASK>)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal ops / reductions

declare i32 @llvm.genx.cbit.i32 (i32)

define i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %c = call i32 @llvm.genx.cbit.i32 (i32 %0)
  ret i32 %c
}

define i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %lo = trunc i64 %0 to i32
  %hi.init = lshr i64 %0, 32
  %hi = trunc i64 %hi.init to i32
  %lo.cbit = call i32 @llvm.genx.cbit.i32 (i32 %lo)
  %hi.cbit = call i32 @llvm.genx.cbit.i32 (i32 %hi)
  %res.32 = add i32 %lo.cbit, %hi.cbit
  %res = zext i32 %res.32 to i64
  ret i64 %res
}

declare i64 @__spirv_BuiltInWorkgroupId(i32 %dim)
declare i64 @__spirv_BuiltInLocalInvocationId(i32 %dim)
declare i64 @__spirv_BuiltInNumWorkgroups(i32 %dim)
declare i64 @__spirv_BuiltInWorkgroupSize(i32 %dim)

define i32 @__task_index()  nounwind readnone alwaysinline {
;; linear_group_id() * linear_local_size() + linear_local_id();
;; linear_group_id = group_count(0) * (group_count(1) * group_id(2) +
;;                   + group_id(1)) + group_id(0);
;; linear_local_size = local_size(0) * local_size(1) * local_size(2);
;; linear_local_id = local_size(0) * (local_size(1) * local_id(2) +
;;                   + local_id(1)) + local_id(0);
;; linear_group_id
  %gr_id_x_64 = call i64 @__spirv_BuiltInWorkgroupId(i32 0)
  %gr_id_y_64 = call i64 @__spirv_BuiltInWorkgroupId(i32 1)
  %gr_id_z_64 = call i64 @__spirv_BuiltInWorkgroupId(i32 2)
  %gr_id_x = trunc i64 %gr_id_x_64 to i32
  %gr_id_y = trunc i64 %gr_id_y_64 to i32
  %gr_id_z = trunc i64 %gr_id_z_64 to i32
  %gr_count_x_64 = call i64 @__spirv_BuiltInNumWorkgroups(i32 0)
  %gr_count_y_64 = call i64 @__spirv_BuiltInNumWorkgroups(i32 1)
  %gr_count_z_64 = call i64 @__spirv_BuiltInNumWorkgroups(i32 2)
  %gr_count_x = trunc i64 %gr_count_x_64 to i32
  %gr_count_y = trunc i64 %gr_count_y_64 to i32
  %gr_count_z = trunc i64 %gr_count_z_64 to i32
  %gr_count_y_z = mul i32 %gr_count_y, %gr_id_z
  %gr_id_y_z_temp = add i32 %gr_count_y_z, %gr_id_y
  %gr_id_x_y_z_temp = mul i32 %gr_id_y_z_temp, %gr_count_x
  %gr_id = add i32 %gr_id_x_y_z_temp, %gr_id_x

;; linear_local_size
  %l_size_x_64 = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
  %l_size_y_64 = call i64 @__spirv_BuiltInWorkgroupSize(i32 1)
  %l_size_z_64 = call i64 @__spirv_BuiltInWorkgroupSize(i32 2)
  %l_size_x = trunc i64 %l_size_x_64 to i32
  %l_size_y = trunc i64 %l_size_y_64 to i32
  %l_size_z = trunc i64 %l_size_z_64 to i32
  %l_size_xy = mul i32 %l_size_x, %l_size_y
  %l_size_xyz = mul i32 %l_size_xy, %l_size_z

;; linear_local_id
  %l_id_x_64 = call i64 @__spirv_BuiltInLocalInvocationId(i32 0)
  %l_id_y_64 = call i64 @__spirv_BuiltInLocalInvocationId(i32 1)
  %l_id_z_64 = call i64 @__spirv_BuiltInLocalInvocationId(i32 2)
  %l_id_x = trunc i64 %l_id_x_64 to i32
  %l_id_y = trunc i64 %l_id_y_64 to i32
  %l_id_z = trunc i64 %l_id_z_64 to i32
  %l_is_y_z_size = mul i32 %l_size_y, %l_id_z
  %l_is_y_z_size_temp = add i32 %l_is_y_z_size, %l_id_y
  %l_is_x_y_z_size_temp = mul i32 %l_is_y_z_size_temp, %l_size_x
  %l_local_id = add i32 %l_is_x_y_z_size_temp, %l_id_x

  %res_temp = mul i32 %gr_id, %l_size_xyz
  %res = add i32 %res_temp, %l_local_id
  ret i32 %res
}

define i32 @__task_count()  nounwind readnone alwaysinline {
;; linear_group_count * linear_local_size
;; linear_group_count = group_count(0) * group_count(1) * group_count(2);
;; linear_local_size = local_size(0) * local_size(1) * local_size(2);
;; linear_local_size
  %l_size_x_ = call i64 @__spirv_BuiltInWorkgroupSize(i32 0)
  %l_size_y_ = call i64 @__spirv_BuiltInWorkgroupSize(i32 1)
  %l_size_z_ = call i64 @__spirv_BuiltInWorkgroupSize(i32 2)
  %l_size_x = trunc i64 %l_size_x_ to i32
  %l_size_y = trunc i64 %l_size_y_ to i32
  %l_size_z = trunc i64 %l_size_z_ to i32
  %l_size_1 = insertelement <3 x i32> undef, i32 %l_size_x, i32 0
  %l_size_2 = insertelement <3 x i32> %l_size_1, i32 %l_size_y, i32 1
  %l_size_3 = insertelement <3 x i32> %l_size_2, i32 %l_size_z, i32 2
  %gr_count_x_ = call i64 @__spirv_BuiltInNumWorkgroups(i32 0)
  %gr_count_y_ = call i64 @__spirv_BuiltInNumWorkgroups(i32 1)
  %gr_count_z_ = call i64 @__spirv_BuiltInNumWorkgroups(i32 2)
  %gr_count_x = trunc i64 %gr_count_x_ to i32
  %gr_count_y = trunc i64 %gr_count_y_ to i32
  %gr_count_z = trunc i64 %gr_count_z_ to i32
  %gr_count_1 = insertelement <3 x i32> undef, i32 %gr_count_x, i32 0
  %gr_count_2 = insertelement <3 x i32> %gr_count_1, i32 %gr_count_y, i32 1
  %gr_count_3 = insertelement <3 x i32> %gr_count_2, i32 %gr_count_z, i32 2
  %size_gr = mul <3 x i32> %l_size_3, %gr_count_3
  %size_gr_0 = extractelement <3 x i32> %size_gr, i32 0
  %size_gr_1 = extractelement <3 x i32> %size_gr, i32 1
  %size_gr_2 = extractelement <3 x i32> %size_gr, i32 2
  %res_ = mul i32 %size_gr_0, %size_gr_1
  %res = mul i32 %res_, %size_gr_2
  ret i32 %res
}

define(`__xe_task_count', `
  %l_size_64 = call i64 @__spirv_BuiltInWorkgroupSize(i32 $1)
  %l_size = trunc i64 %l_size_64 to i32
  %gr_count_64 = call i64 @__spirv_BuiltInNumWorkgroups(i32 $1)
  %gr_count = trunc i64 %gr_count_64 to i32
  %res = mul i32 %l_size, %gr_count
  ret i32 %res
')

define i32 @__task_count0()  nounwind readnone alwaysinline {
   __xe_task_count(0)
}

define i32 @__task_count1()  nounwind readnone alwaysinline {
  __xe_task_count(1)
}

define i32 @__task_count2()  nounwind readnone alwaysinline {
  __xe_task_count(2)
}

define(`__xe_task_index', `
  %gr_id_64 = call i64 @__spirv_BuiltInWorkgroupId(i32 $1)
  %gr_id = trunc i64 %gr_id_64 to i32
  %l_id_64 = call i64 @__spirv_BuiltInLocalInvocationId(i32 $1)
  %l_id = trunc i64 %l_id_64 to i32
  %l_size_64 = call i64 @__spirv_BuiltInWorkgroupSize(i32 $1)
  %l_size = trunc i64 %l_size_64 to i32
  %res_tmp = mul i32 %gr_id, %l_size
  %res = add i32 %res_tmp, %l_id
  ret i32 %res
')

define i32 @__task_index0()  nounwind readnone alwaysinline {
   __xe_task_index(0)
}

define i32 @__task_index1()  nounwind readnone alwaysinline {
   __xe_task_index(1)
}

define i32 @__task_index2()  nounwind readnone alwaysinline {
   __xe_task_index(2)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %hf = bitcast i16 %v to half
  %ft = fpext half %hf to float
  ret float %ft
}

define <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone {
  %hf = bitcast <WIDTH x i16> %v to <WIDTH x half>
  %ft = fpext <WIDTH x half> %hf to <WIDTH x float>
  ret <WIDTH x float> %ft
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %hf = fptrunc float %v to half
  %hf.bitcast = bitcast half %hf to i16
  ret i16 %hf.bitcast
}

define <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone {
  %hf = fptrunc <WIDTH x float> %v to <WIDTH x half>
  %hf.bitcast = bitcast <WIDTH x half> %hf to <WIDTH x i16>
  ret <WIDTH x i16> %hf.bitcast
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp
declare <WIDTH x float> @__spirv_ocl_native_recip_DvWIDTHf(<WIDTH x float> %0)
define <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  ;; No need to make NR iteration to improve precision since precision
  ;; on Xe is high already (1UP)
  %res = call <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = call <WIDTH x float> @__spirv_ocl_native_recip_DvWIDTHf(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

;; rcp
declare <WIDTH x half> @__spirv_ocl_native_recip_DvWIDTHDh(<WIDTH x half> %0)
define <WIDTH x half> @__rcp_varying_half(<WIDTH x half>) nounwind readonly alwaysinline {
  ;; No need to make NR iteration to improve precision since precision
  ;; on Xe is high already (1UP)
  %res = call <WIDTH x half> @__rcp_fast_varying_half(<WIDTH x half> %0)
  ret <WIDTH x half> %res
}

define <WIDTH x half> @__rcp_fast_varying_half(<WIDTH x half>) nounwind readonly alwaysinline {
  %res = call <WIDTH x half> @__spirv_ocl_native_recip_DvWIDTHDh(<WIDTH x half> %0)
  ret <WIDTH x half> %res
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <WIDTH x float> @__spirv_ocl_native_rsqrt_DvWIDTHf(<WIDTH x float>)
define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %v) nounwind readonly alwaysinline {
  %r = call <WIDTH x float> @__spirv_ocl_native_rsqrt_DvWIDTHf(<WIDTH x float> %v)
  ;; Newton-Raphson iteration to improve precision
  ;;  return 0.5 * r * (3. - (v * r) * r);
  %mult = fmul <WIDTH x float> %v, %r
  %mult2 = fmul <WIDTH x float> %mult, %r
  %three_sub = fsub <WIDTH x float> const_vector(float, 3.), %mult2
  %mult3 = fmul <WIDTH x float> %r, %three_sub
  %res = fmul <WIDTH x float> const_vector(float, 0.5), %mult3
  ret <WIDTH x float> %res
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = call <WIDTH x float> @__spirv_ocl_native_rsqrt_DvWIDTHf(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; half precision rsqrt

declare <WIDTH x half> @__spirv_ocl_native_rsqrt_DvWIDTHDh(<WIDTH x half>)
define <WIDTH x half> @__rsqrt_varying_half(<WIDTH x half> %v) nounwind readonly alwaysinline {
  %res = call <WIDTH x half> @__spirv_ocl_native_rsqrt_DvWIDTHDh(<WIDTH x half> %v)
  ret <WIDTH x half> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <WIDTH x float> @__spirv_ocl_native_sqrt_DvWIDTHf(<WIDTH x float>)
define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = call <WIDTH x float> @__spirv_ocl_native_sqrt_DvWIDTHf(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; half precision sqrt

declare <WIDTH x half> @__spirv_ocl_native_sqrt_DvWIDTHDh(<WIDTH x half>)
define <WIDTH x half> @__sqrt_varying_half(<WIDTH x half>) nounwind readonly alwaysinline {
  %res = call <WIDTH x half> @__spirv_ocl_native_sqrt_DvWIDTHDh(<WIDTH x half> %0)
  ret <WIDTH x half> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <WIDTH x double> @llvm.genx.ieee.sqrt.XE_SUFFIX(double)(<WIDTH x double>)
define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind alwaysinline {
  %res = call <WIDTH x double> @llvm.genx.ieee.sqrt.XE_SUFFIX(double)(<WIDTH x double> %0)
  ret <WIDTH x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding 16-bit floats

define <WIDTH x half> @__round_varying_half(<WIDTH x half>) nounwind readonly alwaysinline {
  %conv_fp32 = fpext <WIDTH x half> %0 to <WIDTH x float>
  %res = call <WIDTH x float> @llvm.genx.rnde.XE_SUFFIX(float)(<WIDTH x float> %conv_fp32)
  %conv_hf = fptrunc <WIDTH x float> %res to <WIDTH x half>
  ret <WIDTH x half> %conv_hf
}

define <WIDTH x half> @__floor_varying_half(<WIDTH x half>) nounwind readonly alwaysinline {
    %conv_fp32 = fpext <WIDTH x half> %0 to <WIDTH x float>
    %res = call <WIDTH x float> @llvm.genx.rndd.XE_SUFFIX(float)(<WIDTH x float> %conv_fp32)
    %conv_hf = fptrunc <WIDTH x float> %res to <WIDTH x half>
    ret <WIDTH x half> %conv_hf
}

define <WIDTH x half> @__ceil_varying_half(<WIDTH x half>) nounwind readonly alwaysinline  {
    %conv_fp32 = fpext <WIDTH x half> %0 to <WIDTH x float>
    %res = call <WIDTH x float> @llvm.genx.rndu.XE_SUFFIX(float)(<WIDTH x float> %conv_fp32)
    %conv_hf = fptrunc <WIDTH x float> %res to <WIDTH x half>
    ret <WIDTH x half> %conv_hf
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

define <WIDTH x float> @__round_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = call <WIDTH x float> @llvm.genx.rnde.XE_SUFFIX(float)(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

define <WIDTH x float> @__floor_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
    %res = call <WIDTH x float> @llvm.genx.rndd.XE_SUFFIX(float)(<WIDTH x float> %0)
    ret <WIDTH x float> %res
}

define <WIDTH x float> @__ceil_varying_float(<WIDTH x float>) nounwind readonly alwaysinline  {
    %res = call <WIDTH x float> @llvm.genx.rndu.XE_SUFFIX(float)(<WIDTH x float> %0)
    ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

define <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <WIDTH x double> %0 to <WIDTH x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 -9223372036854775808, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <WIDTH x i32> zeroinitializer
  %bitop.i.i = and <WIDTH x i64> %float_to_int_bitcast.i.i.i.i, %vec_lit
  %bitop.i = xor <WIDTH x i64> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <WIDTH x i64> %bitop.i to <WIDTH x double>
  ; create vector of float literals
  %vec_lit_pos.i = insertelement <1 x double> undef, double 4.5036e+15, i32 0
  %vec_lit_pos = shufflevector <1 x double> %vec_lit_pos.i, <1 x double> undef, <WIDTH x i32> zeroinitializer
  ; create vector of float literals
  %vec_lit_neg.i = insertelement <1 x double> undef, double -4.5036e+15, i32 0
  %vec_lit_neg = shufflevector <1 x double> %vec_lit_neg.i, <1 x double> undef, <WIDTH x i32> zeroinitializer
  %binop.i = fadd <WIDTH x double> %int_to_float_bitcast.i.i40.i, %vec_lit_pos
  %binop21.i = fadd <WIDTH x double> %binop.i, %vec_lit_neg
  %float_to_int_bitcast.i.i.i = bitcast <WIDTH x double> %binop21.i to <WIDTH x i64>
  %bitop31.i = xor <WIDTH x i64> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <WIDTH x i64> %bitop31.i to <WIDTH x double>
  ret <WIDTH x double> %int_to_float_bitcast.i.i.i
}

define <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <WIDTH x double> @__round_varying_double(<WIDTH x double> %0) nounwind
  %bincmp.i = fcmp ogt <WIDTH x double> %calltmp.i, %0
  %val_to_boolvec32.i = sext <WIDTH x i1> %bincmp.i to <WIDTH x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 -4616189618054758400, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <WIDTH x i32> zeroinitializer
  %bitop.i = and <WIDTH x i64> %val_to_boolvec32.i, %vec_lit
  %int_to_float_bitcast.i.i.i = bitcast <WIDTH x i64> %bitop.i to <WIDTH x double>
  %binop.i = fadd <WIDTH x double> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <WIDTH x double> %binop.i
}

define <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <WIDTH x double> @__round_varying_double(<WIDTH x double> %0) nounwind
  %bincmp.i = fcmp olt <WIDTH x double> %calltmp.i, %0
  %val_to_boolvec32.i = sext <WIDTH x i1> %bincmp.i to <WIDTH x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 4607182418800017408, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <WIDTH x i32> zeroinitializer
  %bitop.i = and <WIDTH x i64> %val_to_boolvec32.i, %vec_lit
  %int_to_float_bitcast.i.i.i = bitcast <WIDTH x i64> %bitop.i to <WIDTH x double>
  %binop.i = fadd <WIDTH x double> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <WIDTH x double> %binop.i
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i1 @llvm.genx.any.XE_SUFFIX(i1)(<WIDTH x MASK>)
declare i1 @llvm.genx.all.XE_SUFFIX(i1)(<WIDTH x MASK>)

define i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %v = bitcast <WIDTH x MASK> %0 to BITCAST_WIDTH
  %zext = zext BITCAST_WIDTH %v to i64
  ret i64 %zext
}

define i1 @__any(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.any.XE_SUFFIX(i1)(<WIDTH x MASK> %0)
  ret i1 %v
}

define i1 @__all(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.all.XE_SUFFIX(i1)(<WIDTH x MASK> %0) nounwind readnone
  ret i1 %v
}

define i1 @__none(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.any.XE_SUFFIX(i1)(<WIDTH x MASK> %0) nounwind readnone
  %v_not = icmp eq i1 %v, 0
  ret i1 %v_not
}

define(`xe_add', `
define internal <WIDTH x $1> @__add_varying_$2(<WIDTH x $1>,
                                  <WIDTH x $1>) nounwind readnone alwaysinline {
  %r = add <WIDTH x $1> %0, %1
  ret <WIDTH x $1> %r
}

define internal $1 @__add_uniform_$2($1, $1) nounwind readnone alwaysinline {
  %r = add $1 %0, %1
  ret $1 %r
}
')

xe_add(i16, i16)
xe_add(i32, int32)
xe_add(i64, int64)

define(`xe_fadd', `
define internal <WIDTH x $1> @__fadd_varying_$1(<WIDTH x $1>,
                                  <WIDTH x $1>) nounwind readnone alwaysinline {
  %r = fadd <WIDTH x $1> %0, %1
  ret <WIDTH x $1> %r
}

define internal $1 @__fadd_uniform_$1($1, $1) nounwind readnone alwaysinline {
  %r = fadd $1 %0, %1
  ret $1 %r
}
')

xe_fadd(half)
xe_fadd(float)
xe_fadd(double)

define(`reduce_func',
`ifelse(WIDTH, `32', `reduce32($1, $2, $3, $4)',
        WIDTH, `16', `reduce16($1, $2, $3, $4)',
                     `reduce8($1, $2, $3, $4)')')

define(`reducexe_func',
`ifelse(WIDTH, `32', `reducexe32($1, $2, $3, $4, $5)',
        WIDTH, `16', `reducexe16($1, $2, $3, $4, $5)',
                     `reducexe8($1, $2, $3, $4, $5)')')

define i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone alwaysinline {
  %ext = zext <WIDTH x i8> %0 to <WIDTH x i16>
  reduce_func(i16, @__add_varying_i16, @__add_uniform_i16, %ext)
}

define i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone alwaysinline {
  %ext = zext <WIDTH x i16> %0 to <WIDTH x i32>
  reduce_func(i32, @__add_varying_int32, @__add_uniform_int32, %ext)
}

define half @__reduce_add_half(<WIDTH x half>) nounwind readonly alwaysinline {
  reduce_func(half, @__fadd_varying_half, @__fadd_uniform_half, %0)
}

define i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone {
  %ext = zext <WIDTH x i32> %0 to <WIDTH x i64>
  reduce_func(i64, @__add_varying_int64, @__add_uniform_int64, %ext)
}

define float @__reduce_add_float(<WIDTH x float>) nounwind readonly alwaysinline {
  reduce_func(float, @__fadd_varying_float, @__fadd_uniform_float, %0)
}

define double @__reduce_add_double(<WIDTH x double>) nounwind readnone {
  reduce_func(double, @__fadd_varying_double, @__fadd_uniform_double, %0)
}

define i64 @__reduce_add_int64(<WIDTH x i64>) nounwind readnone {
  reduce_func(i64, @__add_varying_int64, @__add_uniform_int64, %0)
}

define i32 @__reduce_min_int32(<WIDTH x i32>) nounwind readnone {
  reducexe_func(i32, smin, rdregioni, %0, 4)
}

define i32 @__reduce_max_int32(<WIDTH x i32>) nounwind readnone {
  reducexe_func(i32, smax, rdregioni, %0, 4)
}

define i32 @__reduce_min_uint32(<WIDTH x i32>) nounwind readnone {
  reducexe_func(i32, umin, rdregioni, %0, 4)
}

define i32 @__reduce_max_uint32(<WIDTH x i32>) nounwind readnone {
  reducexe_func(i32, umax, rdregioni, %0, 4)
}

define float @__reduce_min_float(<WIDTH x float>) nounwind readnone {
  reducexe_func(float, fmin, rdregionf, %0, 4)
}

define float @__reduce_max_float(<WIDTH x float>) nounwind readnone {
  reducexe_func(float, fmax, rdregionf, %0, 4)
}

define half @__reduce_min_half(<WIDTH x half>) nounwind readnone {
  reducexe_func(half, fmin, rdregionf, %0, 2)
}

define half @__reduce_max_half(<WIDTH x half>) nounwind readnone {
  reducexe_func(half, fmax, rdregionf, %0, 2)
}

define double @__reduce_min_double(<WIDTH x double>) nounwind readnone {
  reduce_func(double, @__min_varying_double, @__min_uniform_double, %0)
}

define double @__reduce_max_double(<WIDTH x double>) nounwind readnone {
  reduce_func(double, @__max_varying_double, @__max_uniform_double, %0)
}

define i64 @__reduce_min_int64(<WIDTH x i64>) nounwind readnone {
  reducexe_func(i64, smin, rdregioni, %0, 8)
}

define i64 @__reduce_max_int64(<WIDTH x i64>) nounwind readnone {
  reducexe_func(i64, smax, rdregioni, %0, 8)
}

define i64 @__reduce_min_uint64(<WIDTH x i64>) nounwind readnone {
  reducexe_func(i64, umin, rdregioni, %0, 8)
}

define i64 @__reduce_max_uint64(<WIDTH x i64>) nounwind readnone {
  reducexe_func(i64, umax, rdregioni, %0, 8)
}

reduce_equal(WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

define(`xe_masked_store_blend', `
declare void @llvm.genx.vstore.XE_SUFFIX($1)(<WIDTH x $1>, <WIDTH x $1>*)
declare <WIDTH x $1> @llvm.genx.vload.XE_SUFFIX($1)(<WIDTH x $1>*)

define void @__masked_store_blend_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>,
                                      <WIDTH x MASK> %mask) nounwind
                                      alwaysinline {
  %old = load <WIDTH x $1>, <WIDTH x $1>* %0
  %blend = select <WIDTH x MASK> %mask, <WIDTH x $1> %1, <WIDTH x $1> %old
  store <WIDTH x $1> %blend, <WIDTH x $1>* %0
  ret void
}
')

xe_masked_store_blend(i8)
xe_masked_store_blend(i16)
xe_masked_store_blend(half)
xe_masked_store_blend(i32)
xe_masked_store_blend(float)
xe_masked_store_blend(double)
xe_masked_store_blend(i64)

define(`xe_masked_store', `
define void @__masked_store_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %ptr = bitcast <WIDTH x $1>* %0 to i8*
  %broadcast_init = insertelement <WIDTH x i32> undef, i32 SIZEOF($1), i32 0
  %shuffle = shufflevector <WIDTH x i32> %broadcast_init, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %offsets = mul LINEAR_VECTOR(i32), %shuffle
ifelse(RUNTIME, `32',
`
  call void @__scatter_base_offsets32_$1(i8* %ptr, i32 1, <WIDTH x i32> %offsets, <WIDTH x $1> %1, <WIDTH x MASK> %mask)
  ',
  RUNTIME, `64',
`
  %offsets64 = zext <WIDTH x i32> %offsets to <WIDTH x i64>
  call void @__scatter_base_offsets64_$1(i8* %ptr, i32 1, <WIDTH x i64> %offsets64, <WIDTH x $1> %1, <WIDTH x MASK> %mask)
')
  ret void
}

')

xe_masked_store(i8)
xe_masked_store(i16)
xe_masked_store(half)
xe_masked_store(i32)
xe_masked_store(float)
xe_masked_store(double)
xe_masked_store(i64)

define(`xe_masked_load', `
; Blend version is NOT safe w.r.t. crossing page boundaries, even if the mask is off
; for the lanes that cross the page boundaries.
define <WIDTH x $1> @__masked_load_blend_$1(i8 *, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %bitptr = bitcast i8* %0 to <WIDTH x $1>*
  %res = load PTR_OP_ARGS(`<WIDTH x $1> ') %bitptr, align SIZEOF($1)
  %res_masked = select <WIDTH x MASK> %mask, <WIDTH x $1> %res, <WIDTH x $1> undef
  ret <WIDTH x $1> %res_masked
}

; This version is safe w.r.t. crossing page boundaries and it contains the optimization
; that is useful for Gen9 and XeLP, but needs to be revised for later hardware.
; The optimization has runtime check for first and last values of the mask and doing
; either block load (if it is safe) or gather (if it is not safe).
define <WIDTH x $1> @__masked_load_$1(i8 *, <WIDTH x MASK> %mask) nounwind alwaysinline {
entry:
  %retptr = alloca <WIDTH x $1>
  %mm = call i64 @__movmsk(<WIDTH x MASK> %mask)

  ; if the first lane and the last lane are on, then it is safe to do a vector load
  ; of the whole thing--what the lanes in the middle want turns out to not matter...
  %mm_and_low = and i64 %mm, 1
  %mm_and_high = and i64 %mm, MASK_HIGH_BIT_ON
  %mm_and_high_shift = lshr i64 %mm_and_high, eval(WIDTH-1)
  %mm_and_low_i1 = trunc i64 %mm_and_low to i1
  %mm_and_high_shift_i1 = trunc i64 %mm_and_high_shift to i1
  %can_vload = and i1 %mm_and_low_i1, %mm_and_high_shift_i1

  ; if we are not able to do a singe vload, we will accumulate lanes in this memory..
  %retptr32 = bitcast <WIDTH x $1> * %retptr to $1 *
  br i1 %can_vload, label %vload, label %vgather

vload:
  ;; Blend version for v8i8 cannot be used here since it reads past the last lane,
  ;; so our checks are not enough to be sure in safety of such operation.
  ;; TODO: currently blend_i8 is not applied anywhere, maybe we can use it
  ;; in a different way somehow.
  ifelse($1,i8, `
    ifelse(WIDTH,8, `
      br label %vgather
    ',`
    %res = call <WIDTH x $1> @__masked_load_blend_$1(i8* %0, <WIDTH x MASK> %mask)
    ret <WIDTH x $1> %res
    ')
  ', $1,i64, `
    ifelse(WIDTH,32, `
      br label %vgather
    ',`
      %res = call <WIDTH x $1> @__masked_load_blend_$1(i8* %0, <WIDTH x MASK> %mask)
      ret <WIDTH x $1> %res
    ')
  ', $1,double, `
    ifelse(WIDTH,32, `
      br label %vgather
    ',`
      %res = call <WIDTH x $1> @__masked_load_blend_$1(i8* %0, <WIDTH x MASK> %mask)
      ret <WIDTH x $1> %res
    ')
  ',`
    %res = call <WIDTH x $1> @__masked_load_blend_$1(i8* %0, <WIDTH x MASK> %mask)
    ret <WIDTH x $1> %res
  ')


vgather:
  %broadcast_init = insertelement <WIDTH x i32> undef, i32 SIZEOF($1), i32 0
  %shuffle = shufflevector <WIDTH x i32> %broadcast_init, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %offsets = mul LINEAR_VECTOR(i32), %shuffle
  ifelse(RUNTIME, `32',
  `
    %res_gather = call <WIDTH x $1> @__gather_base_offsets32_$1(i8 * %0, i32 1, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask)
  ',
  RUNTIME, `64',
  `
    %offsets64 = zext <WIDTH x i32> %offsets to <WIDTH x i64>
    %res_gather = call <WIDTH x $1> @__gather_base_offsets64_$1(i8 * %0, i32 1, <WIDTH x i64> %offsets64, <WIDTH x MASK> %mask)
  ')

  ret <WIDTH x $1> %res_gather
}

')

xe_masked_load(i8)
xe_masked_load(i16)
xe_masked_load(half)
xe_masked_load(i32)
xe_masked_load(float)
xe_masked_load(double)
xe_masked_load(i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter
;; TODO_GEN: add computation of the block size and the number of blocks for svm gather/scatter.
define(`xe_gather', `
ifelse(WIDTH, 32,`
  declare <16 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1,16).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK>, i32, <16 x i64>, <16 x $1>)
',`
  declare <WIDTH x $1> @llvm.genx.svm.gather.XE_SUFFIX($1).XE_SUFFIX(i1).XE_SUFFIX(i64)(<WIDTH x MASK>, i32, <WIDTH x i64>, <WIDTH x $1>)
')


define <WIDTH x $1>
@__gather_base_offsets32_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %scale = insertelement <WIDTH x i32> undef, i32 %offset_scale, i32 0
  %scale_shuffle = shufflevector <WIDTH x i32> %scale, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets_scaled = mul <WIDTH x i32> %offsets, %scale_shuffle
  %ptr_to_int = ptrtoint i8* %ptr to i32
  %base = insertelement <WIDTH x i32> undef, i32 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i32> %base, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i32> %new_offsets_scaled, %shuffle
  %res = call <WIDTH x $1> @__gather32_$1(<WIDTH x i32> %new_offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather_base_offsets64_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %offset_scale64 = zext i32 %offset_scale to i64
  %scale = insertelement <WIDTH x i64> undef, i64 %offset_scale64, i32 0
  %scale_shuffle = shufflevector <WIDTH x i64> %scale, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets_scaled = mul <WIDTH x i64> %offsets, %scale_shuffle
  %ptr_to_int = ptrtoint i8* %ptr to i64
  %base = insertelement <WIDTH x i64> undef, i64 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i64> %base, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i64> %new_offsets_scaled, %shuffle
  %res = call <WIDTH x $1> @__gather64_$1(<WIDTH x i64> %new_offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather32_$1(<WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %offsets64 = zext <WIDTH x i32> %offsets to <WIDTH x i64>
  %res = call <WIDTH x $1> @__gather64_$1(<WIDTH x i64> %offsets64, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather64_$1(<WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  ifelse(WIDTH,32,`
   %offsets1 = shufflevector <WIDTH x i64> %offsets, <WIDTH x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
   %offsets2 = shufflevector <WIDTH x i64> %offsets, <WIDTH x i64> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
   %vecmask1 = shufflevector <WIDTH x MASK> %vecmask, <WIDTH x MASK> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
   %vecmask2 = shufflevector <WIDTH x MASK> %vecmask, <WIDTH x MASK> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
   ifelse($1, i8,`
      %res64_1 = call <64 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, 64).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask1, i32 0, <16 x i64> %offsets1, <16 x $1> undef)
      %res_1 = call <16 x $1> @llvm.genx.rdregioni.XE_SUFFIXN($1,16).XE_SUFFIXN($1, 64).i16(<64 x $1> %res64_1, i32 0, i32 16, i32 4, i16 0, i32 undef)
      %res64_2 = call <64 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, 64).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask2, i32 0, <16 x i64> %offsets2, <16 x $1> undef)
      %res_2 = call <16 x $1> @llvm.genx.rdregioni.XE_SUFFIXN($1,16).XE_SUFFIXN($1, 64).i16(<64 x $1> %res64_2, i32 0, i32 16, i32 4, i16 0, i32 undef)
      %res = shufflevector <16 x $1> %res_1, <16 x $1> %res_2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    ', $1,i16, `
      %res64_1 = call <32 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, 32).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask1, i32 1, <16 x i64> %offsets1, <16 x $1> undef)
      %res_1 = call <16 x $1> @llvm.genx.rdregioni.XE_SUFFIXN($1,16).XE_SUFFIXN($1, 32).i16(<32 x $1> %res64_1, i32 0, i32 16, i32 2, i16 0, i32 undef)
      %res64_2 = call <32 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, 32).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask2, i32 1, <16 x i64> %offsets2, <16 x $1> undef)
      %res_2 = call <16 x $1> @llvm.genx.rdregioni.XE_SUFFIXN($1,16).XE_SUFFIXN($1, 32).i16(<32 x $1> %res64_2, i32 0, i32 16, i32 2, i16 0, i32 undef)
      %res = shufflevector <16 x $1> %res_1, <16 x $1> %res_2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    ',`
      %res1 = call <16 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1,16).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask1, i32 0, <16 x i64> %offsets1, <16 x $1> undef)
      %res2 = call <16 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1,16).XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16)(<16 x MASK> %vecmask2, i32 0, <16 x i64> %offsets2, <16 x $1> undef)
      %res = shufflevector <16 x $1> %res1, <16 x $1> %res2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    ')
      ret <WIDTH x $1> %res
  ',`
    ifelse($1, i8,`
        %res64 = call <WIDTH_X4 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, WIDTH_X4).XE_SUFFIX(i1).XE_SUFFIX(i64)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %offsets, <WIDTH x $1> undef)
        %res = call <WIDTH x $1> @llvm.genx.rdregioni.XE_SUFFIX($1).XE_SUFFIXN($1, WIDTH_X4).i16(<WIDTH_X4 x $1> %res64, i32 0, i32 WIDTH, i32 4, i16 0, i32 undef)
      ', $1,i16, `
        %res64 = call <WIDTH_X2 x $1> @llvm.genx.svm.gather.XE_SUFFIXN($1, WIDTH_X2).XE_SUFFIX(i1).XE_SUFFIX(i64)(<WIDTH x MASK> %vecmask, i32 1, <WIDTH x i64> %offsets, <WIDTH x $1> undef)
        %res = call <WIDTH x $1> @llvm.genx.rdregioni.XE_SUFFIX($1).XE_SUFFIXN($1, WIDTH_X2).i16(<WIDTH_X2 x $1> %res64, i32 0, i32 WIDTH, i32 2, i16 0, i32 undef)
      ',`
        %res = call <WIDTH x $1> @llvm.genx.svm.gather.XE_SUFFIX($1).XE_SUFFIX(i1).XE_SUFFIX(i64)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %offsets, <WIDTH x $1> undef)
      ')
      ret <WIDTH x $1> %res
  ')
}
')
xe_gather(i8)
xe_gather(i16)
xe_gather(half)
xe_gather(i32)
xe_gather(float)
xe_gather(i64)
xe_gather(double)

; We need factored generic implementations when --opt=disable-gathers is used
gen_gather_factored_generic(i8)
gen_gather_factored_generic(i16)
gen_gather_factored_generic(half)
gen_gather_factored_generic(i32)
gen_gather_factored_generic(float)
gen_gather_factored_generic(i64)
gen_gather_factored_generic(double)

define(`xe_scatter', `
ifelse(WIDTH, 32,`
  declare void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1,16)(<16 x MASK>, i32, <16 x i64>, <16 x $1>)
',`
  declare void @llvm.genx.svm.scatter.XE_SUFFIX(i1).XE_SUFFIX(i64).XE_SUFFIX($1)(<WIDTH x MASK>, i32, <WIDTH x i64>, <WIDTH x $1>)
')
define void
@__scatter_base_offsets32_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %scale = insertelement <WIDTH x i32> undef, i32 %offset_scale, i32 0
  %scale_shuffle = shufflevector <WIDTH x i32> %scale, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets_scaled = mul <WIDTH x i32> %offsets, %scale_shuffle
  %ptr_to_int = ptrtoint i8* %ptr to i32
  %base = insertelement <WIDTH x i32> undef, i32 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i32> %base, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i32> %new_offsets_scaled, %shuffle
  call void @__scatter32_$1(<WIDTH x i32> %new_offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter_base_offsets64_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %offset_scale64 = zext i32 %offset_scale to i64
  %scale = insertelement <WIDTH x i64> undef, i64 %offset_scale64, i32 0
  %scale_shuffle = shufflevector <WIDTH x i64> %scale, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets_scaled = mul <WIDTH x i64> %offsets, %scale_shuffle
  %ptr_to_int = ptrtoint i8* %ptr to i64
  %base = insertelement <WIDTH x i64> undef, i64 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i64> %base, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i64> %new_offsets_scaled, %shuffle
  call void @__scatter64_$1(<WIDTH x i64> %new_offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter32_$1(<WIDTH x i32> %ptrs, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  %offsets64 = zext <WIDTH x i32> %ptrs to <WIDTH x i64>
  call void @__scatter64_$1(<WIDTH x i64> %offsets64, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_$1(<WIDTH x i64> %ptrs, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
   ifelse(WIDTH,32,`
    %ptrs1 = shufflevector <WIDTH x i64> %ptrs, <WIDTH x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %ptrs2 = shufflevector <WIDTH x i64> %ptrs, <WIDTH x i64> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %values1 = shufflevector <WIDTH x $1> %values, <WIDTH x $1> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %values2 = shufflevector <WIDTH x $1> %values, <WIDTH x $1> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %vecmask1 = shufflevector <WIDTH x MASK> %vecmask, <WIDTH x MASK> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %vecmask2 = shufflevector <WIDTH x MASK> %vecmask, <WIDTH x MASK> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    ifelse($1,i8, `
      %res1 = tail call <64 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, 64).XE_SUFFIXN($1,16).i16.XE_SUFFIXN(i1,16)(<64 x $1> undef, <16 x $1> %values1, i32 0, i32 16, i32 4, i16 0, i32 0, <16 x MASK> %vecmask1)
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1, 64)(<16 x MASK> %vecmask1, i32 0, <16 x i64> %ptrs1, <64 x $1> %res1)
      %res2 = tail call <64 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, 64).XE_SUFFIXN($1,16).i16.XE_SUFFIXN(i1,16)(<64 x $1> undef, <16 x $1> %values2, i32 0, i32 16, i32 4, i16 0, i32 0, <16 x MASK> %vecmask2)
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1, 64)(<16 x MASK> %vecmask2, i32 0, <16 x i64> %ptrs2, <64 x $1> %res2)
    ', $1,i16, `
      %res1 = tail call <32 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, 32).XE_SUFFIXN($1,16).i16.XE_SUFFIXN(i1,16)(<32 x $1> undef, <16 x $1> %values1, i32 0, i32 16, i32 2, i16 0, i32 0, <16 x MASK> %vecmask1)
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1, 32)(<16 x MASK> %vecmask1, i32 1, <16 x i64> %ptrs1, <32 x $1> %res1)
      %res2 = tail call <32 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, 32).XE_SUFFIXN($1,16).i16.XE_SUFFIXN(i1,16)(<32 x $1> undef, <16 x $1> %values2, i32 0, i32 16, i32 2, i16 0, i32 0, <16 x MASK> %vecmask2)
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1, 32)(<16 x MASK> %vecmask2, i32 1, <16 x i64> %ptrs2, <32 x $1> %res2)
    ',`
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1,16)(<16 x MASK> %vecmask1, i32 0, <16 x i64> %ptrs1, <16 x $1> %values1)
      call void @llvm.genx.svm.scatter.XE_SUFFIXN(i1,16).XE_SUFFIXN(i64,16).XE_SUFFIXN($1,16)(<16 x MASK> %vecmask2, i32 0, <16 x i64> %ptrs2, <16 x $1> %values2)
    ')
  ',`
    ifelse($1,i8, `
      %res = tail call <WIDTH_X4 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, WIDTH_X4).XE_SUFFIX($1).i16.XE_SUFFIX(i1)(<WIDTH_X4 x $1> undef, <WIDTH x $1> %values, i32 0, i32 WIDTH, i32 4, i16 0, i32 0, <WIDTH x MASK> %vecmask)
      call void @llvm.genx.svm.scatter.XE_SUFFIX(i1).XE_SUFFIX(i64).XE_SUFFIXN(i8, WIDTH_X4)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %ptrs, <WIDTH_X4 x $1> %res)
    ', $1,i16, `
      %res = tail call <WIDTH_X2 x $1> @llvm.genx.wrregioni.XE_SUFFIXN($1, WIDTH_X2).XE_SUFFIX($1).i16.XE_SUFFIX(i1)(<WIDTH_X2 x $1> undef, <WIDTH x $1> %values, i32 0, i32 WIDTH, i32 2, i16 0, i32 0, <WIDTH x MASK> %vecmask)
      call void @llvm.genx.svm.scatter.XE_SUFFIX(i1).XE_SUFFIX(i64).XE_SUFFIXN($1, WIDTH_X2)(<WIDTH x MASK> %vecmask, i32 1, <WIDTH x i64> %ptrs, <WIDTH_X2 x $1> %res)
    ',`
      call void @llvm.genx.svm.scatter.XE_SUFFIX(i1).XE_SUFFIX(i64).XE_SUFFIX($1)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %ptrs, <WIDTH x $1> %values)
    ')
  ')
  ret void
}

')

xe_scatter(i8)
xe_scatter(i16)
xe_scatter(half)
xe_scatter(i32)
xe_scatter(float)
xe_scatter(i64)
xe_scatter(double)

; We need factored generic implementations when --opt=disable-scatters is used
gen_scatter_factored(i8)
gen_scatter_factored(i16)
gen_scatter_factored(half)
gen_scatter_factored(i32)
gen_scatter_factored(float)
gen_scatter_factored(i64)
gen_scatter_factored(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; native transcendetals

define(`EXP', `0x4005BF0A80000000')
define(`LOG2E', `0x3FF7154760000000') ;; LOG2E = log(2, e)

declare float @__spirv_ocl_native_log2_DvWIDTH1f(float) nounwind readnone
define float @__log_uniform_float(float) nounwind readnone {
  %res2base = call float @__spirv_ocl_native_log2_DvWIDTH1f(float %0)
  %res = fdiv float %res2base, LOG2E
  ret float %res
}

declare <WIDTH x float> @__spirv_ocl_native_log2_DvWIDTHf(<WIDTH x float>) nounwind readnone
define <WIDTH x float> @__log_varying_float(<WIDTH x float>) nounwind readnone {
  %res2base = call <WIDTH x float> @__spirv_ocl_native_log2_DvWIDTHf(<WIDTH x float> %0)
  %log2e = insertelement <WIDTH x float> undef, float LOG2E, i32 0
  %log2e_shuffle = shufflevector <WIDTH x float> %log2e, <WIDTH x float> undef, <WIDTH x i32> zeroinitializer
  %res = fdiv <WIDTH x float> %res2base, %log2e_shuffle
  ret <WIDTH x float> %res
}


define(`EXPF16', `0xH4170')
define(`LOG2EF16', `0xH3DC5') ;; LOG2EF16 = log(2, e)
declare half @__spirv_ocl_native_log2_DvWIDTH1Dh(half) nounwind readnone
define half @__log_uniform_half(half) nounwind readnone {
  %res2base = call half @__spirv_ocl_native_log2_DvWIDTH1Dh(half %0)
  %res = fdiv half %res2base, LOG2EF16
  ret half %res
}

declare <WIDTH x half> @__spirv_ocl_native_log2_DvWIDTHDh(<WIDTH x half>) nounwind readnone
define <WIDTH x half> @__log_varying_half(<WIDTH x half>) nounwind readnone {
  %res2base = call <WIDTH x half> @__spirv_ocl_native_log2_DvWIDTHDh(<WIDTH x half> %0)
  %log2e = insertelement <WIDTH x half> undef, half LOG2EF16, i32 0
  %log2e_shuffle = shufflevector <WIDTH x half> %log2e, <WIDTH x half> undef, <WIDTH x i32> zeroinitializer
  %res = fdiv <WIDTH x half> %res2base, %log2e_shuffle
  ret <WIDTH x half> %res
}

declare float @__spirv_ocl_native_powr_DvWIDTH1f(float, float) nounwind readnone
define float @__pow_uniform_float(float, float) nounwind readnone {
  %res = call float @__spirv_ocl_native_powr_DvWIDTH1f(float %0, float %1)
  ret float %res
}

declare <WIDTH x float> @__spirv_ocl_native_powr_DvWIDTHf(<WIDTH x float>, <WIDTH x float>) nounwind readnone
define <WIDTH x float> @__pow_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone {
  %res = call <WIDTH x float> @__spirv_ocl_native_powr_DvWIDTHf(<WIDTH x float> %0, <WIDTH x float> %1)
  ret <WIDTH x float> %res
}

declare half @__spirv_ocl_native_powr_DvWIDTH1Dh(half, half) nounwind readnone
define half @__pow_uniform_half(half, half) nounwind readnone {
  %res = call half @__spirv_ocl_native_powr_DvWIDTH1Dh(half %0, half %1)
  ret half %res
}

declare <WIDTH x half> @__spirv_ocl_native_powr_DvWIDTHDh(<WIDTH x half>, <WIDTH x half>) nounwind readnone
define <WIDTH x half> @__pow_varying_half(<WIDTH x half>, <WIDTH x half>) nounwind readnone {
  %res = call <WIDTH x half> @__spirv_ocl_native_powr_DvWIDTHDh(<WIDTH x half> %0, <WIDTH x half> %1)
  ret <WIDTH x half> %res
}

define float @__exp_uniform_float(float) nounwind readnone {
  %res = call float @__spirv_ocl_native_powr_DvWIDTH1f(float EXP, float %0)
  ret float %res
}

define <WIDTH x float> @__exp_varying_float(<WIDTH x float>) nounwind readnone {
  %exp = insertelement <WIDTH x float> undef, float EXP, i32 0
  %exp_shuffle = shufflevector <WIDTH x float> %exp, <WIDTH x float> undef, <WIDTH x i32> zeroinitializer
  %res = call <WIDTH x float> @__spirv_ocl_native_powr_DvWIDTHf(<WIDTH x float> %exp_shuffle, <WIDTH x float> %0)
  ret <WIDTH x float> %res
}

define half @__exp_uniform_half(half) nounwind readnone {
  %res = call half @__spirv_ocl_native_powr_DvWIDTH1Dh(half EXPF16, half %0)
  ret half %res
}

define <WIDTH x half> @__exp_varying_half(<WIDTH x half>) nounwind readnone {
  %exp = insertelement <WIDTH x half> undef, half EXPF16, i32 0
  %exp_shuffle = shufflevector <WIDTH x half> %exp, <WIDTH x half> undef, <WIDTH x i32> zeroinitializer
  %res = call <WIDTH x half> @__spirv_ocl_native_powr_DvWIDTHDh(<WIDTH x half> %exp_shuffle, <WIDTH x half> %0)
  ret <WIDTH x half> %res
}


;; Generates double math builtins for unfiorm and varying
;; $1 operation (e.g. pow, sin etc)
define(`xe_double_math', `
declare double @__spirv_ocl_$1_DvWIDTH1d(double) nounwind readnone
define double @__$1_uniform_double(double) nounwind readnone {
  %res = call double @__spirv_ocl_$1_DvWIDTH1d(double %0)
  ret double %res
}

ifelse(WIDTH,32,`
  declare <16 x double> @__spirv_ocl_$1_DvWIDTHd(<16 x double>)
',`
  declare <WIDTH x double> @__spirv_ocl_$1_DvWIDTHd(<WIDTH x double>)
')
define <WIDTH x double> @__$1_varying_double(<WIDTH x double>) nounwind readnone {
  ifelse(WIDTH,32,`
    %in1 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in2 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %res1 = call <16 x double> @__spirv_ocl_$1_DvWIDTHd(<16 x double> %in1)
    %res2 = call <16 x double> @__spirv_ocl_$1_DvWIDTHd(<16 x double> %in2)
    %res = shufflevector <16 x double> %res1, <16 x double> %res2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ',`
    %res = call <WIDTH x double> @__spirv_ocl_$1_DvWIDTHd(<WIDTH x double> %0)
  ')
  ret <WIDTH x double> %res
}
')

xe_double_math(exp)
xe_double_math(log)
xe_double_math(sin)
xe_double_math(cos)
xe_double_math(tan)
xe_double_math(asin)
xe_double_math(acos)
xe_double_math(atan)

;; sin is returned value
;; cos is returned through pointer
declare double @__spirv_ocl_sincos_DvWIDTH1d(double, double*) nounwind
define void @__sincos_uniform_double(double, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to double*
  %ptr2 = bitcast i8* %2 to double*
  %sin = call double @__spirv_ocl_sincos_DvWIDTH1d(double %0, double* %ptr2)
  store double %sin, double* %ptr1
  ret void
}

ifelse(WIDTH,32,`
  declare <16 x double> @__spirv_ocl_sincos_DvWIDTHd(<16 x double>, <16 x double>*) nounwind
',`
  declare <WIDTH x double> @__spirv_ocl_sincos_DvWIDTHd(<WIDTH x double>, <WIDTH x double>*) nounwind
')

define void @__sincos_varying_double(<WIDTH x double>, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to <WIDTH x double>*
  %ptr2 = bitcast i8* %2 to <WIDTH x double>*
  ifelse(WIDTH,32,`
    %in0_1 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in0_2 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %ptr2_16_1 = bitcast <32 x double>* %ptr2 to <16 x double>*
    %ptrtoint = ptrtoint <32 x double>* %ptr2 to i64
    %ptrtoint_add = add i64 %ptrtoint, 128
    %ptr2_16_2 = inttoptr i64 %ptrtoint_add to <16 x double>*
    %sin1 = call <16 x double> @__spirv_ocl_sincos_DvWIDTHd(<16 x double> %in0_1, <16 x double>* %ptr2_16_1)
    %sin2 = call <16 x double> @__spirv_ocl_sincos_DvWIDTHd(<16 x double> %in0_2, <16 x double>* %ptr2_16_2)
    %sin = shufflevector <16 x double> %sin1, <16 x double> %sin2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ',`
    %sin = call <WIDTH x double> @__spirv_ocl_sincos_DvWIDTHd(<WIDTH x double> %0, <WIDTH x double>* %ptr2)
  ')
  store <WIDTH x double> %sin, <WIDTH x double>* %ptr1
  ret void
}

declare double @__spirv_ocl_pow_DvWIDTH1d(double, double) nounwind readnone
define double @__pow_uniform_double(double, double) nounwind {
  %res = call double @__spirv_ocl_pow_DvWIDTH1d(double %0, double %1)
  ret double %res
}

ifelse(WIDTH,32,`
  declare <16 x double> @__spirv_ocl_pow_DvWIDTHd(<16 x double>, <16 x double>) nounwind readnone
',`
  declare <WIDTH x double> @__spirv_ocl_pow_DvWIDTHd(<WIDTH x double>, <WIDTH x double>) nounwind readnone
')
define <WIDTH x double> @__pow_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind {
  ifelse(WIDTH,32,`
    %in1_1 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in1_2 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %in2_1 = shufflevector <32 x double> %1, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in2_2 = shufflevector <32 x double> %1, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %res1 = call <16 x double> @__spirv_ocl_pow_DvWIDTHd(<16 x double> %in1_1, <16 x double> %in2_1)
    %res2 = call <16 x double> @__spirv_ocl_pow_DvWIDTHd(<16 x double> %in1_2, <16 x double> %in2_2)
    %res = shufflevector <16 x double> %res1, <16 x double> %res2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ',`
    %res = call <WIDTH x double> @__spirv_ocl_pow_DvWIDTHd(<WIDTH x double> %0, <WIDTH x double> %1)
  ')

  ret <WIDTH x double> %res
}

declare double @__spirv_ocl_atan2_DvWIDTH1d(double, double) nounwind readnone
define double @__atan2_uniform_double(double, double) nounwind {
  %res = call double @__spirv_ocl_atan2_DvWIDTH1d(double %0, double %1)
  ret double %res
}

ifelse(WIDTH,32,`
  declare <16 x double> @__spirv_ocl_atan2_DvWIDTHd(<16 x double>, <16 x double>) nounwind readnone
',`
  declare <WIDTH x double> @__spirv_ocl_atan2_DvWIDTHd(<WIDTH x double>, <WIDTH x double>) nounwind readnone
')

define <WIDTH x double> @__atan2_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind {
  ifelse(WIDTH,32,`
    %in1_1 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in1_2 = shufflevector <32 x double> %0, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %in2_1 = shufflevector <32 x double> %1, <32 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
    %in2_2 = shufflevector <32 x double> %1, <32 x double> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    %res1 = call <16 x double> @__spirv_ocl_atan2_DvWIDTHd(<16 x double> %in1_1, <16 x double> %in2_1)
    %res2 = call <16 x double> @__spirv_ocl_atan2_DvWIDTHd(<16 x double> %in1_2, <16 x double> %in2_2)
    %res = shufflevector <16 x double> %res1, <16 x double> %res2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ',`
    %res = call <WIDTH x double> @__spirv_ocl_atan2_DvWIDTHd(<WIDTH x double> %0, <WIDTH x double> %1)
  ')

  ret <WIDTH x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; native trigonometry

declare float @__spirv_ocl_native_sin_DvWIDTH1f(float) nounwind readnone
define float @__sin_uniform_float(float) nounwind readnone {
  %res = call float @__spirv_ocl_native_sin_DvWIDTH1f(float %0)
  ret float %res
}

declare <WIDTH x float> @__spirv_ocl_native_sin_DvWIDTHf(<WIDTH x float>) nounwind readnone
define <WIDTH x float> @__sin_varying_float(<WIDTH x float>) nounwind readnone {
  %res = call <WIDTH x float> @__spirv_ocl_native_sin_DvWIDTHf(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

declare float @__spirv_ocl_native_cos_DvWIDTH1f(float) nounwind readnone
define float @__cos_uniform_float(float) nounwind readnone {
  %res = call float @__spirv_ocl_native_cos_DvWIDTH1f(float %0)
  ret float %res
}

declare <WIDTH x float> @__spirv_ocl_native_cos_DvWIDTHf(<WIDTH x float>) nounwind readnone
define <WIDTH x float> @__cos_varying_float(<WIDTH x float>) nounwind readnone {
  %res = call <WIDTH x float> @__spirv_ocl_native_cos_DvWIDTHf(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

define float @__tan_uniform_float(float) nounwind readnone {
  %cos = call float @__spirv_ocl_native_cos_DvWIDTH1f(float %0)
  %sin = call float @__spirv_ocl_native_sin_DvWIDTH1f(float %0)
  %res = fdiv float %sin, %cos
  ret float %res
}

define <WIDTH x float> @__tan_varying_float(<WIDTH x float>) nounwind readnone {
  %cos = call <WIDTH x float> @__spirv_ocl_native_cos_DvWIDTHf(<WIDTH x float> %0)
  %sin = call <WIDTH x float> @__spirv_ocl_native_sin_DvWIDTHf(<WIDTH x float> %0)
  %res = fdiv <WIDTH x float> %sin, %cos
  ret <WIDTH x float> %res
}

define void @__sincos_uniform_float(float, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to float*
  %ptr2 = bitcast i8* %2 to float*
  %cos = call float @__spirv_ocl_native_cos_DvWIDTH1f(float %0)
  %sin = call float @__spirv_ocl_native_sin_DvWIDTH1f(float %0)
  store float %sin, float* %ptr1
  store float %cos, float* %ptr2
  ret void
}

define void @__sincos_varying_float(<WIDTH x float>, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to <WIDTH x float>*
  %ptr2 = bitcast i8* %2 to <WIDTH x float>*
  %cos = call <WIDTH x float> @__spirv_ocl_native_cos_DvWIDTHf(<WIDTH x float> %0)
  %sin = call <WIDTH x float> @__spirv_ocl_native_sin_DvWIDTHf(<WIDTH x float> %0)
  store <WIDTH x float> %sin, <WIDTH x float>* %ptr1
  store <WIDTH x float> %cos, <WIDTH x float>* %ptr2
  ret void
}

declare half @__spirv_ocl_native_sin_DvWIDTH1Dh(half) nounwind readnone
define half @__sin_uniform_half(half) nounwind readnone {
  %res = call half @__spirv_ocl_native_sin_DvWIDTH1Dh(half %0)
  ret half %res
}

declare <WIDTH x half> @__spirv_ocl_native_sin_DvWIDTHDh(<WIDTH x half>) nounwind readnone
define <WIDTH x half> @__sin_varying_half(<WIDTH x half>) nounwind readnone {
  %res = call <WIDTH x half> @__spirv_ocl_native_sin_DvWIDTHDh(<WIDTH x half> %0)
  ret <WIDTH x half> %res
}

declare half @__spirv_ocl_native_cos_DvWIDTH1Dh(half) nounwind readnone
define half @__cos_uniform_half(half) nounwind readnone {
  %res = call half @__spirv_ocl_native_cos_DvWIDTH1Dh(half %0)
  ret half %res
}

declare <WIDTH x half> @__spirv_ocl_native_cos_DvWIDTHDh(<WIDTH x half>) nounwind readnone
define <WIDTH x half> @__cos_varying_half(<WIDTH x half>) nounwind readnone {
  %res = call <WIDTH x half> @__spirv_ocl_native_cos_DvWIDTHDh(<WIDTH x half> %0)
  ret <WIDTH x half> %res
}

define half @__tan_uniform_half(half) nounwind readnone {
  %cos = call half @__spirv_ocl_native_cos_DvWIDTH1Dh(half %0)
  %sin = call half @__spirv_ocl_native_sin_DvWIDTH1Dh(half %0)
  %res = fdiv half %sin, %cos
  ret half %res
}

define <WIDTH x half> @__tan_varying_half(<WIDTH x half>) nounwind readnone {
  %cos = call <WIDTH x half> @__spirv_ocl_native_cos_DvWIDTHDh(<WIDTH x half> %0)
  %sin = call <WIDTH x half> @__spirv_ocl_native_sin_DvWIDTHDh(<WIDTH x half> %0)
  %res = fdiv <WIDTH x half> %sin, %cos
  ret <WIDTH x half> %res
}

define void @__sincos_uniform_half(half, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to half*
  %ptr2 = bitcast i8* %2 to half*
  %cos = call half @__spirv_ocl_native_cos_DvWIDTH1Dh(half %0)
  %sin = call half @__spirv_ocl_native_sin_DvWIDTH1Dh(half %0)
  store half %sin, half* %ptr1
  store half %cos, half* %ptr2
  ret void
}

define void @__sincos_varying_half(<WIDTH x half>, i8*, i8*) nounwind {
  %ptr1 = bitcast i8* %1 to <WIDTH x half>*
  %ptr2 = bitcast i8* %2 to <WIDTH x half>*
  %cos = call <WIDTH x half> @__spirv_ocl_native_cos_DvWIDTHDh(<WIDTH x half> %0)
  %sin = call <WIDTH x half> @__spirv_ocl_native_sin_DvWIDTHDh(<WIDTH x half> %0)
  store <WIDTH x half> %sin, <WIDTH x half>* %ptr1
  store <WIDTH x half> %cos, <WIDTH x half>* %ptr2
  ret void
}

trigonometry_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; atomics
;; Generates atomics intrinsics. Xe intrinsics are supported for WIDTH = 1, 2, 4, 8
;; so for WIDTH = 16 or more we will use 8-wide width intrinsics.
;; $1 atomic operation (e.g. max, min...)

define(`xe_atomics_decl', `
  declare <1 x i32> @llvm.genx.svm.atomic.$1.v1i32.v1i1.v1i64(<1 x i1>, <1 x i64>, <1 x i32>, <1 x i32>)
  declare <1 x i64> @llvm.genx.svm.atomic.$1.v1i64.v1i1.v1i64(<1 x i1>, <1 x i64>, <1 x i64>, <1 x i64>)
  declare <8 x i32> @llvm.genx.svm.atomic.$1.v8i32.v8i1.v8i64(<8 x i1>, <8 x i64>, <8 x i32>, <8 x i32>)
  declare <8 x i64> @llvm.genx.svm.atomic.$1.v8i64.v8i1.v8i64(<8 x i1>, <8 x i64>, <8 x i64>, <8 x i64>)
')
;; cmpxchg has another signature, declare them separately
declare <1 x i32> @llvm.genx.svm.atomic.cmpxchg.v1i32.v1i1.v1i64(<1 x i1>, <1 x i64>, <1 x i32>, <1 x i32>, <1 x i32>)
declare <1 x i64> @llvm.genx.svm.atomic.cmpxchg.v1i64.v1i1.v1i64(<1 x i1>, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>)
declare <8 x i32> @llvm.genx.svm.atomic.cmpxchg.v8i32.v8i1.v8i64(<8 x i1>, <8 x i64>, <8 x i32>, <8 x i32>, <8 x i32>)
declare <8 x i64> @llvm.genx.svm.atomic.cmpxchg.v8i64.v8i1.v8i64(<8 x i1>, <8 x i64>, <8 x i64>, <8 x i64>, <8 x i64>)

xe_atomics_decl(add)
xe_atomics_decl(xchg)
xe_atomics_decl(sub)
xe_atomics_decl(and)
xe_atomics_decl(or)
xe_atomics_decl(xor)
xe_atomics_decl(max)
xe_atomics_decl(imax)
xe_atomics_decl(min)
xe_atomics_decl(imin)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; idiv implementation
;; For Xe target the fastest way to make idiv operations is to generate
;; LLVM instructions, VC backend effectevly process it.
;; $1 LLVM type (e.g. i8, i32)
;; $2 ISPC stdlib type (e.g int8, uint32)
;; $3 llvm function

define(`xe_idiv_decl', `
  define <WIDTH x $1> @__idiv_$2(<WIDTH x $1>, <WIDTH x $1>) nounwind readnone alwaysinline{
    %res = $3 <WIDTH x $1> %0, %1
    ret <WIDTH x $1> %res
  }
')

xe_idiv_decl(i8, int8, sdiv)
xe_idiv_decl(i16, int16, sdiv)
xe_idiv_decl(i32, int32, sdiv)
xe_idiv_decl(i8, uint8, udiv)
xe_idiv_decl(i16, uint16, udiv)
xe_idiv_decl(i32, uint32, udiv)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
dot_product_vnni_decl()
