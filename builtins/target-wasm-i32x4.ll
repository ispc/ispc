;;  Copyright (c) 2020-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;; Authors:
;; Anton Schreiner

;; TODO
;; * Implement actual *fast* functions to be faster than default
;; * Analize code to find uses for wasm specific intrinsics(@llvm.wasm.anytrue.v4i32 etc)

define(`WIDTH',`4')
;; FIXME: Workaround for "BUILD_OS should be defined to either UNIX or WINDOWS" error
define(`BUILD_OS',`UNIX')
define(`MASK',`i32')
define(`ISA',`WASM')
;; Wasm has custom clock function
define(`HAS_CUSTOM_CLOCK',`1')

include(`util.m4')

stdlib_core()
scans()
rdrand_decls()
define_shuffles()
aossoa()
ctlztz()
trigonometry_decl()
transcendetals_decl()
include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)
define_avgs()
saturation_arithmetic()
halfTypeGenericImplementation()
dot_product_vnni_decl()

;; rcp/rsqrt for double
rsqrtd_decl()
rcpd_decl()

;; rcp/rsqrt for half
rcph_rsqrth_decl

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone
declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare double    @llvm.sqrt.f64(double %Val)
declare float     @llvm.sin.f32(float %Val)
declare float     @llvm.asin.f32(float %Val)
declare float     @llvm.cos.f32(float %Val)
declare float     @llvm.sqrt.f32(float %Val)
declare float     @llvm.exp.f32(float %Val)
declare float     @llvm.log.f32(float %Val)
declare float     @llvm.pow.f32(float %f, float %e)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)
declare double @round(double)
declare double @floor(double)
declare double @ceil(double)

define i1 @__wasm_cmp_msk_eq(<4 x i32> %v1, <4 x i32> %v2) nounwind readnone alwaysinline {
  %v1_i128 = bitcast <4 x i32> %v1 to i128
  %v2_i128 = bitcast <4 x i32> %v2 to i128
  %ret = icmp eq i128 %v1_i128, %v2_i128
  ret i1 %ret
}

define i64 @__clock() {
entry:
  %call = tail call i32 @clock()
  %conv = sext i32 %call to i64
  ret i64 %conv
}

declare i32 @clock()

define void @__fastmath() {
entry:
  ret void
}

define i32 @__set_ftz_daz_flags() nounwind alwaysinline {
  ret i32 0
}

define void @__restore_ftz_daz_flags(i32 %oldVal) nounwind alwaysinline {
  ret void
}

define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone alwaysinline {
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %0)
  ret <WIDTH x double> %r
}

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture %ptr, <WIDTH x i8> %new,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i8> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i8> %new, <WIDTH x i8> %old
  store <WIDTH x i8> %result, <WIDTH x i8> * %ptr
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture %ptr, <WIDTH x i16> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i16> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i16> %new, <WIDTH x i16> %old
  store <WIDTH x i16> %result, <WIDTH x i16> * %ptr
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture %ptr, <WIDTH x i32> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i32> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i32> %new, <WIDTH x i32> %old
  store <WIDTH x i32> %result, <WIDTH x i32> * %ptr
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture %ptr,
                            <WIDTH x i64> %new, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i64> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i64> %new, <WIDTH x i64> %old
  store <WIDTH x i64> %result, <WIDTH x i64> * %ptr
  ret void
}

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %res = bitcast <WIDTH x i1> %mask1 to i4
  %res_i64 = zext i4 %res to i64
  ret i64 %res_i64
}

define i1 @__any(<4 x MASK> %mask) nounwind readnone alwaysinline {
  entry:
    %any_true = bitcast <WIDTH x MASK> %mask to i128
    %cmp = icmp ne i128 %any_true, 0
    ret i1 %cmp
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  entry:
    %all_true = bitcast <WIDTH x MASK> %mask to i128
    %cmp = icmp eq i128 %all_true, -1
    ret i1 %cmp
}

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %any = call i1 @__any(<WIDTH x MASK> %mask)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

define <4 x float> @__rsqrt_varying_float(<4 x float> %v) nounwind readnone alwaysinline {
entry:
  %0 = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %v)
  %mul.i16 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %0
  ; %mul.i = fmul <4 x float> %0, %v
  ; %mul.i18 = fmul <4 x float> %0, %mul.i
  ; %sub.i = fsub <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, %mul.i18
  ; %mul.i17 = fmul <4 x float> %0, %sub.i
  ; %mul.i16 = fmul <4 x float> %mul.i17, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  ret <4 x float> %mul.i16
}

define <4 x float> @__rsqrt_fast_varying_float(<4 x float> %v) nounwind readnone alwaysinline {
entry:
  %0 = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %v)
  %mul.i16 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %0
  ; %mul.i = fmul <4 x float> %0, %v
  ; %mul.i18 = fmul <4 x float> %0, %mul.i
  ; %sub.i = fsub <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, %mul.i18
  ; %mul.i17 = fmul <4 x float> %0, %sub.i
  ; %mul.i16 = fmul <4 x float> %mul.i17, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  ret <4 x float> %mul.i16
}

define <4 x float> @__sqrt_varying_float(<4 x float> %v) nounwind readnone alwaysinline {
entry:
  %0 = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %v)
  ret <4 x float> %0
}

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast float %0 to i32
  %bitop.i.i = and i32 %float_to_int_bitcast.i.i.i.i, -2147483648
  %bitop.i = xor i32 %bitop.i.i, %float_to_int_bitcast.i.i.i.i
  %int_to_float_bitcast.i.i40.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %int_to_float_bitcast.i.i40.i, 8.388608e+06
  %binop21.i = fadd float %binop.i, -8.388608e+06
  %float_to_int_bitcast.i.i.i = bitcast float %binop21.i to i32
  %bitop31.i = xor i32 %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop31.i to float
  ret float %int_to_float_bitcast.i.i.i
}

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp ogt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, -1082130432
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp olt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, 1065353216
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

define <4 x float> @__round_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <4 x float> %0 to <4 x i32>
  %bitop.i.i = and <4 x i32> %float_to_int_bitcast.i.i.i.i, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %bitop.i = xor <4 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %int_to_float_bitcast.i.i40.i, <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06>
  %binop21.i = fadd <4 x float> %binop.i, <float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06>
  %float_to_int_bitcast.i.i.i = bitcast <4 x float> %binop21.i to <4 x i32>
  %bitop31.i = xor <4 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop31.i to <4 x float>
  ret <4 x float> %int_to_float_bitcast.i.i.i
}

define <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp ogt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

define <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp olt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

define <4 x double> @__round_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @round)
}

define <4 x double> @__floor_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @floor)
}

define <4 x double> @__ceil_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @ceil)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define float @__max_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__min_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ult float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define i32 @__min_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp slt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp sgt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ult i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ugt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp olt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define <4 x i32> @__vselect_i32(<4 x i32>, <4 x i32> ,
                                <4 x i32> %mask) nounwind readnone alwaysinline {
  %notmask = xor <4 x i32> %mask, <i32 -1, i32 -1, i32 -1, i32 -1>
  %cleared_old = and <4 x i32> %0, %notmask
  %masked_new = and <4 x i32> %1, %mask
  %new = or <4 x i32> %cleared_old, %masked_new
  ret <4 x i32> %new
}

define <4 x float> @__vselect_float(<4 x float>, <4 x float>,
                                    <4 x i32> %mask) nounwind readnone alwaysinline {
  %v0 = bitcast <4 x float> %0 to <4 x i32>
  %v1 = bitcast <4 x float> %1 to <4 x i32>
  %r = call <4 x i32> @__vselect_i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %mask)
  %rf = bitcast <4 x i32> %r to <4 x float>
  ret <4 x float> %rf
}

define <4 x i32> @__min_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp slt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define <4 x i32> @__max_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp sgt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define <4 x i32> @__min_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ult <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define <4 x i32> @__max_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ugt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <4 x float> @__min_varying_float(<4 x float> %0, <4 x float> %1) {
  %3 = fcmp olt <4 x float> %0, %1
  %4 = select <4 x i1> %3, <4 x float>  %0, <4 x float>  %1
  ret <4 x float>  %4
}

define <4 x float> @__max_varying_float(<4 x float> %0, <4 x float> %1) unnamed_addr #0 {
  %3 = fcmp ogt <4 x float> %0, %1
  %4 = select <4 x i1> %3, <4 x float>  %0, <4 x float>  %1
  ret <4 x float>  %4
}

define <4 x double> @__max_varying_double(<4 x double> %a, <4 x double> %b) {
entry:
  %vecinit2 = shufflevector <4 x double> %a, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %vecinit7 = shufflevector <4 x double> %b, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %0 = tail call <2 x double> @llvm.maximum.v2f64(<2 x double> %vecinit2, <2 x double> %vecinit7) #5
  %vecinit12 = shufflevector <4 x double> %a, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %vecinit17 = shufflevector <4 x double> %b, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %1 = tail call <2 x double> @llvm.maximum.v2f64(<2 x double> %vecinit12, <2 x double> %vecinit17) #5
  %vecinit6.i = shufflevector <2 x double> %0, <2 x double> %1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %vecinit6.i
}

define <4 x double> @__min_varying_double(<4 x double> %a, <4 x double> %b) {
entry:
  %vecinit2 = shufflevector <4 x double> %a, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %vecinit7 = shufflevector <4 x double> %b, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %0 = tail call <2 x double> @llvm.minimum.v2f64(<2 x double> %vecinit2, <2 x double> %vecinit7) #5
  %vecinit12 = shufflevector <4 x double> %a, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %vecinit17 = shufflevector <4 x double> %b, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %1 = tail call <2 x double> @llvm.minimum.v2f64(<2 x double> %vecinit12, <2 x double> %vecinit17) #5
  %vecinit6.i = shufflevector <2 x double> %0, <2 x double> %1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %vecinit6.i
}

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @round(double %0)
  ret double %r
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @floor(double %0)
  ret double %r
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @ceil(double %0)
  ret double %r
}

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(half)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)
masked_store_float_double()

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

packed_load_and_store(4)
define_prefetches()
popcnt()

define i16 @__reduce_add_int8(<4 x i8> %v) {
entry:
  %vecext = extractelement <4 x i8> %v, i32 0
  %conv = sext i8 %vecext to i16
  %vecext.1 = extractelement <4 x i8> %v, i32 1
  %conv.1 = sext i8 %vecext.1 to i16
  %add.1 = add nsw i16 %conv, %conv.1
  %vecext.2 = extractelement <4 x i8> %v, i32 2
  %conv.2 = sext i8 %vecext.2 to i16
  %add.2 = add nsw i16 %add.1, %conv.2
  %vecext.3 = extractelement <4 x i8> %v, i32 3
  %conv.3 = sext i8 %vecext.3 to i16
  %add.3 = add nsw i16 %add.2, %conv.3
  ret i16 %add.3
}

define internal <4 x i16> @__add_varying_i16(<4 x i16>,
                                  <4 x i16>) nounwind readnone alwaysinline {
  %r = add <4 x i16> %0, %1
  ret <4 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<4 x i16>) nounwind readnone alwaysinline {
  reduce4(i16, @__add_varying_i16, @__add_uniform_i16)
}

define float @__reduce_add_float(<4 x float> %v) nounwind readonly alwaysinline {
  %v1 = shufflevector <4 x float> %v, <4 x float> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = fadd <4 x float> %v1, %v
  %m1a = extractelement <4 x float> %m1, i32 0
  %m1b = extractelement <4 x float> %m1, i32 1
  %sum = fadd float %m1a, %m1b
  ret float %sum
}

define float @__reduce_min_float(<4 x float>) nounwind readnone {
  reduce4(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<4 x float>) nounwind readnone {
  reduce4(float, @__max_varying_float, @__max_uniform_float)
}

define i32 @__reduce_add_int32(<4 x i32> %v) nounwind readnone alwaysinline {
  %v1 = shufflevector <4 x i32> %v, <4 x i32> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = add <4 x i32> %v1, %v
  %m1a = extractelement <4 x i32> %m1, i32 0
  %m1b = extractelement <4 x i32> %m1, i32 1
  %sum = add i32 %m1a, %m1b
  ret i32 %sum
}

define double @__reduce_add_double(<4 x double>) nounwind readnone {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = fadd <2 x double> %v0, %v1
  %e0 = extractelement <2 x double> %sum, i32 0
  %e1 = extractelement <2 x double> %sum, i32 1
  %m = fadd double %e0, %e1
  ret double %m
}

define double @__reduce_min_double(<4 x double>) nounwind readnone {
  reduce4(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<4 x double>) nounwind readnone {
  reduce4(double, @__max_varying_double, @__max_uniform_double)
}

define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

define hidden <4 x float> @__rcp_varying_float(<4 x float> %v) local_unnamed_addr #0 {
entry:
  %div.i = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %v
  ret <4 x float> %div.i
}

define hidden <4 x float> @__rcp_fast_varying_float(<4 x float> %v) local_unnamed_addr #0 {
entry:
  %div.i = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %v
  ret <4 x float> %div.i
}

define  float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @llvm.sqrt.f32(float %0)
  ret float %ret
}

define  double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @llvm.sqrt.f64(double %0)
  ret double %ret
}

define  float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %s = call float @__sqrt_uniform_float(float %0)
  %r = call float @__rcp_uniform_float(float %s)
  ret float %r
}

define  float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %s = call float @__sqrt_uniform_float(float %0)
  %r = call float @__rcp_uniform_float(float %s)
  ret float %r
}

define  float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %r = fdiv float 1.,%0
  ret float %r
}

define  float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %r = fdiv float 1.,%0
  ret float %r
}

define i64 @__reduce_add_int64(<4 x i64>) nounwind readnone alwaysinline {
  %v0 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = add <2 x i64> %v0, %v1
  %e0 = extractelement <2 x i64> %sum, i32 0
  %e1 = extractelement <2 x i64> %sum, i32 1
  %m = add i64 %e0, %e1
  ret i64 %m
}

define i64 @__reduce_min_int64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(4)
