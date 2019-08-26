;;  Copyright (c) 2019, Intel Corporation
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

target datalayout = "e-p:32:32-i64:64-n8:16:32";

define(`WIDTH',`16')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')
include(`util-genx.m4')

define(`GEN_SUFFIX',
`ifelse($1, `i1', `v16i1',
        $1, `i8', `v16i8',
        $1, `i16', `v16i16',
        $1, `i32', `v16i32',
        $1, `float', `v16f32',
        $1, `double', `v16f64',
        $1, `i64', `v16i64')')

define(`SIZEOF',
`ifelse($1, `i1', 1,
        $1, `i8', 1,
        $1, `i16', 2,
        $1, `i32', 4,
        $1, `float', 4,
        $1, `double', 8,
        $1, `i64', 8)')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

stdlib_core()
packed_load_and_store()
scans()
int64minmax()
saturation_arithmetic()
ctlztz()
define_prefetches()
define_shuffles()
aossoa()
rdrand_decls()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare float @llvm.genx.rndd.f32(float)
declare float @llvm.genx.rndu.f32(float)
declare <16 x float> @llvm.genx.rndu.GEN_SUFFIX(float)(<16 x float>)
declare <16 x float> @llvm.genx.rndd.GEN_SUFFIX(float)(<16 x float>)


define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.rndd.f32(float %0)
    ret float %res
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.rndu.f32(float %0)
    ret float %res
}

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast float %0 to i32
  %bitop.i.i = and i32 %float_to_int_bitcast.i.i.i.i, -2147483648
  %bitop.i = xor i32 %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %int_to_float_bitcast.i.i40.i, 8.388608e+06
  %binop21.i = fadd float %binop.i, -8.388608e+06
  %float_to_int_bitcast.i.i.i = bitcast float %binop21.i to i32
  %bitop31.i = xor i32 %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop31.i to float
  ret float %int_to_float_bitcast.i.i.i
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
;; rcp

define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %mid_res = fdiv float 1., %0
  ;; do one N-R iteration to improve precision
  ;; return (2. - v * r) * r;
  %mult = fmul float %0, %mid_res
  %two_minus = fsub float 2., %mult
  %res = fmul float %mid_res, %two_minus
  ret float %res
}

define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %res = fdiv float 1., %0
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare float @llvm.genx.rsqrt.float.f32(float)
define float @__rsqrt_uniform_float(float %v) nounwind readonly alwaysinline {
  %r = call float @llvm.genx.rsqrt.float.f32(float %v)
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
  %res = call float @llvm.genx.rsqrt.float.f32(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare float @llvm.genx.sqrt.f32(float)
define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %res = call float @llvm.genx.sqrt.f32(float %0)
  ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare double @llvm.genx.ieee.sqrt.d64(double)
define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  %res = call double @llvm.genx.ieee.sqrt.d64(double %0)
  ret double %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode

;; In CPU fastmath set FTZ (flush-to-zero) and DAZ (denormals-are-zero)
;; GenX CM have per kernel setting of CM_DENORM_RTZ (Set all denorms to zero) - applied as attribute to kernel function; enabled by default
;; So in GenX fastmath enabled by default
define void @__fastmath() nounwind alwaysinline {
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare i32 @llvm.genx.smin.i32.i32(i32, i32)
declare i32 @llvm.genx.smax.i32.i32(i32, i32)
declare i32 @llvm.genx.umin.i32.i32(i32, i32)
declare i32 @llvm.genx.umax.i32.i32(i32, i32)
declare float @llvm.genx.fmin.f32.f32(float, float)
declare float @llvm.genx.fmax.f32.f32(float, float)
declare <16 x i32> @llvm.genx.smin.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.genx.smax.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.genx.umin.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.genx.umax.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32>, <16 x i32>)
declare <16 x float> @llvm.genx.fmin.GEN_SUFFIX(float).GEN_SUFFIX(float)(<16 x float>, <16 x float>)
declare <16 x float> @llvm.genx.fmax.GEN_SUFFIX(float).GEN_SUFFIX(float)(<16 x float>, <16 x float>)


define float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.fmax.f32.f32(float %0, float %1)
    ret float %res
}

define float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
    %res = call float @llvm.genx.fmin.f32.f32(float %0, float %1)
    ret float %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
    %res = call i32 @llvm.genx.smin.i32.i32(i32 %0, i32 %1)
    ret i32 %res
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
    %res = call i32 @llvm.genx.smax.i32.i32(i32 %0, i32 %1)
    ret i32 %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
    %res = call i32 @llvm.genx.umin.i32.i32(i32 %0, i32 %1)
    ret i32 %res
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
    %res = call i32 @llvm.genx.umax.i32.i32(i32 %0, i32 %1)
    ret i32 %res
}

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

declare_nvptx()

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

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %r = fdiv <WIDTH x float> const_vector(float, 1.), %0
  ;; do one N-R iteration to improve precision
  ;; return (2. - v * r) * r;
  %mult = fmul <WIDTH x float> %0, %r
  %two_minus = fsub <WIDTH x float> const_vector(float, 2.), %mult
  %res = fmul <WIDTH x float> %r, %two_minus
  ret <WIDTH x float> %res
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = fdiv <WIDTH x float> const_vector(float, 1.), %0
  ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float>)
define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %v) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float> %v)
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
  %res = call <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float> %0)
  ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <16 x float> @llvm.genx.sqrt.v16f32(<16 x float>)
define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readonly alwaysinline {
  %res = call <16 x float> @llvm.genx.sqrt.v16f32(<WIDTH x float> %0)
  ret <WIDTH x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <16 x double> @llvm.genx.ieee.sqrt.v16d64(<16 x double>)
define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind alwaysinline {
  %res = call <16 x double> @llvm.genx.ieee.sqrt.v16d64(<WIDTH x double> %0)
  ret <WIDTH x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <16 x float> %0 to <16 x i32>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i32> undef, i32 -2147483648, i32 0
  %vec_lit = shufflevector <1 x i32> %vec_lit.i, <1 x i32> undef, <16 x i32> zeroinitializer
  %bitop.i.i = and <16 x i32> %float_to_int_bitcast.i.i.i.i, %vec_lit
  %bitop.i = xor <16 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <16 x i32> %bitop.i to <16 x float>
  ; create vector of float literals
  %vec_lit_pos.i = insertelement <1 x float> undef, float 8.388608e+06, i32 0
  %vec_lit_pos = shufflevector <1 x float> %vec_lit_pos.i, <1 x float> undef, <16 x i32> zeroinitializer
  ; create vector of float literals
  %vec_lit_neg.i = insertelement <1 x float> undef, float -8.388608e+06, i32 0
  %vec_lit_neg = shufflevector <1 x float> %vec_lit_neg.i, <1 x float> undef, <16 x i32> zeroinitializer
  %binop.i = fadd <16 x float> %int_to_float_bitcast.i.i40.i, %vec_lit_pos
  %binop21.i = fadd <16 x float> %binop.i, %vec_lit_neg
  %float_to_int_bitcast.i.i.i = bitcast <16 x float> %binop21.i to <16 x i32>
  %bitop31.i = xor <16 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <16 x i32> %bitop31.i to <16 x float>
  ret <16 x float> %int_to_float_bitcast.i.i.i
}


define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
    %res = call <16 x float> @llvm.genx.rndd.GEN_SUFFIX(float)(<16 x float> %0)
    ret <16 x float> %res
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline  {
    %res = call <16 x float> @llvm.genx.rndu.GEN_SUFFIX(float)(<16 x float> %0)
    ret <16 x float> %res
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <16 x double> %0 to <16 x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 -9223372036854775808, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <16 x i32> zeroinitializer
  %bitop.i.i = and <16 x i64> %float_to_int_bitcast.i.i.i.i, %vec_lit
  %bitop.i = xor <16 x i64> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <16 x i64> %bitop.i to <16 x double>
  ; create vector of float literals
  %vec_lit_pos.i = insertelement <1 x double> undef, double 4.5036e+15, i32 0
  %vec_lit_pos = shufflevector <1 x double> %vec_lit_pos.i, <1 x double> undef, <16 x i32> zeroinitializer
  ; create vector of float literals
  %vec_lit_neg.i = insertelement <1 x double> undef, double -4.5036e+15, i32 0
  %vec_lit_neg = shufflevector <1 x double> %vec_lit_neg.i, <1 x double> undef, <16 x i32> zeroinitializer
  %binop.i = fadd <16 x double> %int_to_float_bitcast.i.i40.i, %vec_lit_pos
  %binop21.i = fadd <16 x double> %binop.i, %vec_lit_neg
  %float_to_int_bitcast.i.i.i = bitcast <16 x double> %binop21.i to <16 x i64>
  %bitop31.i = xor <16 x i64> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <16 x i64> %bitop31.i to <16 x double>
  ret <16 x double> %int_to_float_bitcast.i.i.i
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <16 x double> @__round_varying_double(<16 x double> %0) nounwind
  %bincmp.i = fcmp ogt <16 x double> %calltmp.i, %0
  %val_to_boolvec32.i = sext <16 x i1> %bincmp.i to <16 x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 -4616189618054758400, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <16 x i32> zeroinitializer
  %bitop.i = and <16 x i64> %val_to_boolvec32.i, %vec_lit
  %int_to_float_bitcast.i.i.i = bitcast <16 x i64> %bitop.i to <16 x double>
  %binop.i = fadd <16 x double> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <16 x double> %binop.i
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <16 x double> @__round_varying_double(<16 x double> %0) nounwind
  %bincmp.i = fcmp olt <16 x double> %calltmp.i, %0
  %val_to_boolvec32.i = sext <16 x i1> %bincmp.i to <16 x i64>
  ; create vector of literals
  %vec_lit.i = insertelement <1 x i64> undef, i64 4607182418800017408, i32 0
  %vec_lit = shufflevector <1 x i64> %vec_lit.i, <1 x i64> undef, <16 x i32> zeroinitializer
  %bitop.i = and <16 x i64> %val_to_boolvec32.i, %vec_lit
  %int_to_float_bitcast.i.i.i = bitcast <16 x i64> %bitop.i to <16 x double>
  %binop.i = fadd <16 x double> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <16 x double> %binop.i
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

define <16 x float> @__max_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
    %res = call <16 x float> @llvm.genx.fmax.GEN_SUFFIX(float).GEN_SUFFIX(float)(<16 x float> %0, <16 x float> %1)
    ret <16 x float> %res
}

define <16 x float> @__min_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline {
    %res = call <16 x float> @llvm.genx.fmin.GEN_SUFFIX(float).GEN_SUFFIX(float)(<16 x float> %0, <16 x float> %1)
    ret <16 x float> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
    %res = call <16 x i32> @llvm.genx.smin.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32> %0, <16 x i32> %1)
    ret <16 x i32> %res
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
    %res = call <16 x i32> @llvm.genx.smax.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32> %0, <16 x i32> %1)
    ret <16 x i32> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
    %res = call <16 x i32> @llvm.genx.umin.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32> %0, <16 x i32> %1)
    ret <16 x i32> %res
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
    %res = call <16 x i32> @llvm.genx.umax.GEN_SUFFIX(i32).GEN_SUFFIX(i32)(<16 x i32> %0, <16 x i32> %1)
    ret <16 x i32> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone {
  %pred = fcmp olt <16 x double> %0, %1
  %res = select <16 x i1> %pred, <16 x double> %0, <16 x double> %1
  ret <16 x double> %res
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone {
  %pred = fcmp ogt <16 x double> %0, %1
  %res = select <16 x i1> %pred, <16 x double> %0, <16 x double> %1
  ret <16 x double> %res
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i1 @llvm.genx.any.v16i1(<16 x MASK>)
declare i1 @llvm.genx.all.v16i1(<16 x MASK>)

define i64 @__movmsk(<16 x MASK>) nounwind readnone alwaysinline {
  %v = bitcast <16 x MASK> %0 to i16
  %zext = zext i16 %v to i64
  ret i64 %zext
}

define i1 @__any(<16 x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.any.v16i1(<16 x MASK> %0)
  ret i1 %v
}

define i1 @__all(<16 x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.all.v16i1(<16 x MASK> %0) nounwind readnone
  ret i1 %v
}

define i1 @__none(<16 x MASK>) nounwind readnone alwaysinline {
  %v = call i1 @llvm.genx.all.v16i1(<16 x MASK> %0) nounwind readnone
  %v_not = icmp eq i1 %v, 0
  ret i1 %v_not
}

declare i16 @__reduce_add_int8(<16 x MASK>) nounwind readnone alwaysinline

define internal <16 x i16> @__add_varying_i16(<16 x i16>,
                                  <16 x i16>) nounwind readnone alwaysinline {
  %r = add <16 x i16> %0, %1
  ret <16 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<16 x i16>) nounwind readnone alwaysinline {
  reduce16(i16, @__add_varying_i16, @__add_uniform_i16)
}

define internal <16 x float> @__add_varying_float(<16 x float>, <16 x float>) {
  %r = fadd <16 x float> %0, %1
  ret <16 x float> %r
}

define internal float @__add_uniform_float(float, float) {
  %r = fadd float %0, %1
  ret float %r
}

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  reduce16(float, @__add_varying_float, @__add_uniform_float)
}

define float @__reduce_min_float(<16 x float>) nounwind readnone {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<16 x float>) nounwind readnone {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

define internal <16 x i32> @__add_varying_int32(<16 x i32>, <16 x i32>) {
  %r = add <16 x i32> %0, %1
  ret <16 x i32> %r
}

define internal i32 @__add_uniform_int32(i32, i32) {
  %r = add i32 %0, %1
  ret i32 %r
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}

define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

define internal <16 x double> @__add_varying_double(<16 x double>, <16 x double>) {
  %r = fadd <16 x double> %0, %1
  ret <16 x double> %r
}

define internal double @__add_uniform_double(double, double) {
  %r = fadd double %0, %1
  ret double %r
}

define double @__reduce_add_double(<16 x double>) nounwind readnone {
  reduce16(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<16 x double>) nounwind readnone {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

define internal <16 x i64> @__add_varying_int64(<16 x i64>, <16 x i64>) {
  %r = add <16 x i64> %0, %1
  ret <16 x i64> %r
}

define internal i64 @__add_uniform_int64(i64, i64) {
  %r = add i64 %0, %1
  ret i64 %r
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

define(`genx_masked_store_blend', `
declare void @llvm.genx.vstore.GEN_SUFFIX($1)(<WIDTH x $1>, <WIDTH x $1>*)
declare <WIDTH x $1> @llvm.genx.vload.GEN_SUFFIX($1)(<WIDTH x $1>*)

define void @__masked_store_blend_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>,
                                      <WIDTH x MASK> %mask) nounwind
                                      alwaysinline {
  %old = load <WIDTH x $1>, <WIDTH x $1>* %0
  %blend = select <WIDTH x MASK> %mask, <16 x $1> %1, <16 x $1> %old
  store <WIDTH x $1> %blend, <WIDTH x $1>* %0
  ret void
}
')

genx_masked_store_blend(i8)
genx_masked_store_blend(i16)
genx_masked_store_blend(i32)
genx_masked_store_blend(float)
genx_masked_store_blend(double)
genx_masked_store_blend(i64)

define(`genx_masked_store', `
declare void @llvm.genx.svm.block.st.GEN_SUFFIX($1)(i64, <WIDTH x $1>)
define void @__masked_store_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %ptr = bitcast <WIDTH x $1>* %0 to i8*
  %broadcast_init = insertelement <WIDTH x i32> undef, i32 SIZEOF($1), i32 0
  %shuffle = shufflevector <WIDTH x i32> %broadcast_init, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %offsets = mul <WIDTH x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %shuffle
  call void @__scatter_base_offsets32_$1(i8* %ptr, i32 1, <WIDTH x i32> %offsets, <WIDTH x $1> %1, <WIDTH x MASK> %mask)
  ret void
}

define void @__masked_store_private_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %ptr = bitcast <WIDTH x $1>* %0 to i8*
  %broadcast_init = insertelement <WIDTH x i32> undef, i32 SIZEOF($1), i32 0
  %shuffle = shufflevector <WIDTH x i32> %broadcast_init, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %offsets = mul <WIDTH x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %shuffle
  call void @__scatter_base_offsets32_private_$1(i8* %ptr, i32 1, <WIDTH x i32> %offsets, <WIDTH x $1> %1, <WIDTH x MASK> %mask)
  ret void
}
')

genx_masked_store(i8)
genx_masked_store(i16)
genx_masked_store(i32)
genx_masked_store(float)
genx_masked_store(double)
genx_masked_store(i64)

define(`genx_masked_load', `
declare <WIDTH x $1> @llvm.genx.svm.block.ld.GEN_SUFFIX($1)(i64)
define <WIDTH x $1> @__masked_load_$1(i8 *, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %broadcast_init = insertelement <WIDTH x i32> undef, i32 SIZEOF($1), i32 0
  %shuffle = shufflevector <WIDTH x i32> %broadcast_init, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %offsets = mul <WIDTH x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %shuffle
  %res = call <WIDTH x $1> @__gather_base_offsets32_$1(i8 * %0, i32 1, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1> @__masked_load_private_$1(i8 *, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %ptr_bitcast = bitcast i8* %0 to <WIDTH x $1>*
  %res = load <WIDTH x $1>, <WIDTH x $1>* %ptr_bitcast
  %masked_res = select <WIDTH x MASK> %mask, <WIDTH x $1> %res, <WIDTH x $1> zeroinitializer
  ret <WIDTH x $1> %masked_res
}
')

genx_masked_load(i8)
genx_masked_load(i16)
genx_masked_load(i32)
genx_masked_load(float)
genx_masked_load(double)
genx_masked_load(i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter
;; TODO_GEN: add computation of the block size and the number of blocks for svm gather/scatter.
define(`genx_gather', `
declare <WIDTH x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<WIDTH x MASK>, i32, <WIDTH x i64>, <WIDTH x $1>)
declare <WIDTH x $1> @llvm.genx.gather.private.GEN_SUFFIX($1).v16i1.pi8.v16i32(<WIDTH x MASK>, i8*, <WIDTH x i32>, <WIDTH x $1>)

define <WIDTH x $1>
@__gather_base_offsets32_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %ptr_to_int = ptrtoint i8* %ptr to i32
  %base = insertelement <WIDTH x i32> undef, i32 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i32> %base, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i32> %offsets, %shuffle
  %res = call <WIDTH x $1> @__gather32_$1(<WIDTH x i32> %new_offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather_base_offsets64_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %ptr_to_int = ptrtoint i8* %ptr to i64
  %base = insertelement <WIDTH x i64> undef, i64 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i64> %base, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i64> %offsets, %shuffle
  %res = call <WIDTH x $1> @__gather64_$1(<WIDTH x i64> %new_offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather_base_offsets32_private_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %res = call <WIDTH x $1> @llvm.genx.gather.private.GEN_SUFFIX($1).v16i1.pi8.v16i32(<WIDTH x MASK> %vecmask, i8* %ptr, <WIDTH x i32> %offsets, <WIDTH x $1> undef)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather_base_offsets64_private_$1(i8 * %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %offsets32 = trunc <WIDTH x i64> %offsets to <WIDTH x i32>
  %res = call <WIDTH x $1> @llvm.genx.gather.private.GEN_SUFFIX($1).v16i1.pi8.v16i32(<WIDTH x MASK> %vecmask, i8* %ptr, <WIDTH x i32> %offsets32, <WIDTH x $1> undef)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather32_$1(<WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %offsets64 = zext <WIDTH x i32> %offsets to <WIDTH x i64>
  %res = call <WIDTH x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<16 x MASK> %vecmask, i32 0, <WIDTH x i64> %offsets64, <WIDTH x $1> undef)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather64_$1(<WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %res = call <WIDTH x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %offsets, <WIDTH x $1> undef)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather32_private_$1(<WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  ;; CM cannot process zeroinitializer as a base so we need to generate here llvm gather
  %res = call <WIDTH x $1> @__gather_base_offsets32_private_$1(i8 * zeroinitializer, i32 0, <WIDTH x i32> %offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}

define <WIDTH x $1>
@__gather64_private_$1(<WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  ;; CM cannot process zeroinitializer as a base so we need to generate here llvm gather
  %res = call <WIDTH x $1> @__gather_base_offsets64_private_$1(i8 * zeroinitializer, i32 0, <WIDTH x i64> %offsets, <WIDTH x MASK> %vecmask)
  ret <WIDTH x $1> %res
}
')
genx_gather(i8)
genx_gather(i16)
genx_gather(i32)
genx_gather(float)
genx_gather(i64)
genx_gather(double)

define(`genx_scatter', `
declare void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<WIDTH x MASK>, i32, <WIDTH x i64>, <WIDTH x $1>)
declare void @llvm.genx.scatter.private.v16i1.pi8.v16i32.GEN_SUFFIX($1)(<WIDTH x MASK>, i8*, <WIDTH x i32>, <WIDTH x $1>)

define void
@__scatter_base_offsets32_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %ptr_to_int = ptrtoint i8* %ptr to i32
  %base = insertelement <WIDTH x i32> undef, i32 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i32> %base, <WIDTH x i32> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i32> %offsets, %shuffle
  call void @__scatter32_$1(<WIDTH x i32> %new_offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter_base_offsets64_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %ptr_to_int = ptrtoint i8* %ptr to i64
  %base = insertelement <WIDTH x i64> undef, i64 %ptr_to_int, i32 0
  %shuffle = shufflevector <WIDTH x i64> %base, <WIDTH x i64> undef, <WIDTH x i32> zeroinitializer
  %new_offsets = add <WIDTH x i64> %offsets, %shuffle
  call void @__scatter64_$1(<WIDTH x i64> %new_offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter_base_offsets32_private_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i32> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  call void @llvm.genx.scatter.private.v16i1.pi8.v16i32.GEN_SUFFIX($1)(<WIDTH x MASK> %vecmask, i8* %ptr, <WIDTH x i32> %offsets, <WIDTH x $1> %vals)
  ret void
}

define void
@__scatter_base_offsets64_private_$1(i8* %ptr, i32 %offset_scale, <WIDTH x i64> %offsets, <WIDTH x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
   %offsets32 = trunc <WIDTH x i64> %offsets to <WIDTH x i32>
   call void @llvm.genx.scatter.private.v16i1.pi8.v16i32.GEN_SUFFIX($1)(<WIDTH x MASK> %vecmask, i8* %ptr, <WIDTH x i32> %offsets32, <WIDTH x $1> %vals)
   ret void
}

define void
@__scatter32_$1(<WIDTH x i32> %ptrs, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  %offsets64 = zext <WIDTH x i32> %ptrs to <WIDTH x i64>
  call void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %offsets64, <WIDTH x $1> %values)
  ret void
}

define void
@__scatter64_$1(<WIDTH x i64> %ptrs, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<WIDTH x MASK> %vecmask, i32 0, <WIDTH x i64> %ptrs, <WIDTH x $1> %values)
  ret void
}

define void
@__scatter32_private_$1(<WIDTH x i32> %offsets, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  ;; CM cannot process zeroinitializer as a base so we need to generate here llvm scatter
  call void @__scatter_base_offsets32_private_$1(i8* zeroinitializer, i32 1, <WIDTH x i32> %offsets, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_private_$1(<WIDTH x i64> %offsets, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  ;; CM cannot process zeroinitializer as a base so we need to generate here llvm scatter
  call void @__scatter_base_offsets64_private_$1(i8* zeroinitializer, i32 1, <WIDTH x i64> %offsets, <WIDTH x $1> %values, <WIDTH x MASK> %vecmask)
  ret void
}
')

genx_scatter(i8)
genx_scatter(i16)
genx_scatter(i32)
genx_scatter(float)
genx_scatter(i64)
genx_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()
