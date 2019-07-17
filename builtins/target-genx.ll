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
`ifelse($1, `i8', `v16i8',
        $1, `i16', `v16i16',
        $1, `i32', `v16i32',
        $1, `float', `v16f32',
        $1, `double', `v16f64',
        $1, `i64', `v16i64')')

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

declare <1 x float> @llvm.genx.rndd(<1 x float>)
declare <1 x float> @llvm.genx.rndu(<1 x float>)
declare <16 x float> @llvm.genx.rndu.GEN_SUFFIX(float)(<16 x float>)
declare <16 x float> @llvm.genx.rndd.GEN_SUFFIX(float)(<16 x float>)


define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
    %res.i.i = insertelement <1 x float> undef, float %0, i32 0
    %res.i = call <1 x float> @llvm.genx.rndd(<1 x float> %res.i.i)
    %res = extractelement <1 x float> %res.i, i32 0
    ret float %res
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
    %res.i.i = insertelement <1 x float> undef, float %0, i32 0
    %res.i = call <1 x float> @llvm.genx.rndu(<1 x float> %res.i.i)
    %res = extractelement <1 x float> %res.i, i32 0
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
  %ret = fdiv float 1., %0
  ret float %ret
}

define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %ret = fdiv float 1., %0
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare float @llvm.genx.rsqrt.float.f32(float)
define float @__rsqrt_uniform_float(float %v) nounwind readonly alwaysinline {
  %r = call float @llvm.genx.rsqrt.float.f32(float %v)
  ;; Newton-Raphson iteration to improve precision
  ;;  return 0.5 * r * (3. - (v * r) * r);
  %v_is = fmul float %v, %r
  %v_is_is = fmul float %v_is, %r
  %three_sub = fsub float 3., %v_is_is
  %is_mul = fmul float %r, %three_sub
  %res = fmul float 0.5, %is_mul
  ret float %res
}

define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %r = call float @llvm.genx.rsqrt.float.f32(float %0)
  ret float %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare float @llvm.genx.sqrt.f32(float)
define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %r = call float @llvm.genx.sqrt.f32(float %0)
  ret float %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare double @llvm.genx.ieee.sqrt.d64(double)
define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  %r = call double @llvm.genx.ieee.sqrt.d64(double %0)
  ret double %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fast math mode

declare void @__fastmath() nounwind alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare float @__max_uniform_float(float, float) nounwind readonly alwaysinline

declare float @__min_uniform_float(float, float) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare double @__min_uniform_double(double, double) nounwind readnone alwaysinline

declare double @__max_uniform_double(double, double) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline

declare i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline

declare i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal ops / reductions

declare i32 @__popcnt_int32(i32) nounwind readonly alwaysinline

declare i64 @__popcnt_int64(i64) nounwind readonly alwaysinline

declare_nvptx()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readonly alwaysinline
declare <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float>)
define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float> %v)
  ;; Newton-Raphson iteration to improve precision
  ;;  return 0.5 * r * (3. - (v * r) * r);
  %v_is = fmul <16 x float> %v, %r
  %v_is_is = fmul <16 x float> %v_is, %r
  %three_sub = fsub <16 x float> <float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <16 x float> %r, %three_sub
  %res = fmul <16 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                            float 0.5, float 0.5, float 0.5, float 0.5,
                            float 0.5, float 0.5, float 0.5, float 0.5,
                            float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <16 x float> %res
}

define <16 x float> @__rsqrt_fast_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.genx.rsqrt.v16f32(<16 x float> %0)
  ret <16 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <16 x float> @llvm.genx.sqrt.v16f32(<16 x float>)
define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.genx.sqrt.v16f32(<16 x float> %0)
  ret <16 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <16 x double> @llvm.genx.ieee.sqrt.v16d64(<16 x double>)
define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  %r = call <16 x double> @llvm.genx.ieee.sqrt.v16d64(<16 x double> %0)
  ret <16 x double> %r
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

declare <16 x float> @__max_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline

declare <16 x float> @__min_varying_float(<16 x float>, <16 x float>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

declare <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

declare <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

declare <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

declare <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone

declare <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone

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
declare void @llvm.genx.vstore.GEN_SUFFIX($1)(<16 x $1>, <16 x $1>*)
declare <16 x $1> @llvm.genx.vload.GEN_SUFFIX($1)(<16 x $1>*)

define void @__masked_store_blend_$1(<16 x $1>* nocapture, <16 x $1>,
                                      <16 x MASK> %mask) nounwind
                                      alwaysinline {
  %old = call <16 x $1> @llvm.genx.vload.GEN_SUFFIX($1)(<16 x $1>* %0)
  %blend = select <16 x i1> %mask, <16 x $1> %1, <16 x $1> %old
  call void @llvm.genx.vstore.GEN_SUFFIX($1)(<16 x $1> %blend, <16 x $1>* %0)
  ret void
}
')

genx_masked_store_blend(i8)
genx_masked_store_blend(i16)
genx_masked_store_blend(i32)
genx_masked_store_blend(float)
genx_masked_store_blend(double)
genx_masked_store_blend(i64)

;; llvm.genx.svm.block.st must be predicated with llvm.genx.simdcf.predicate
;; but since CMSimdCFLowering pass is run before ISPC passes
;; llvm.genx.simdcf.predicate will not be lowered.
;; TODO_GEN: insert predication

define(`genx_masked_store', `
declare void @llvm.genx.svm.block.st.GEN_SUFFIX($1)(i64, <WIDTH x $1>)
define void @__masked_store_$1(<WIDTH x $1>* nocapture, <WIDTH x $1>, <WIDTH x MASK>) nounwind alwaysinline {
  %bitcast = bitcast <WIDTH x $1>* %0 to i32*
  %ptrtoint = ptrtoint i32* %bitcast to i32
  %zext = zext i32 %ptrtoint to i64
  call void @llvm.genx.svm.block.st.GEN_SUFFIX($1)(i64 %zext, <WIDTH x $1> %1)
  ret void
}
')

genx_masked_store(i8)
genx_masked_store(i16)
genx_masked_store(i32)
genx_masked_store(float)
genx_masked_store(double)
genx_masked_store(i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts
;; llvm.genx.svm.block.ld must be predicated with llvm.genx.simdcf.predicate
;; but since CMSimdCFLowering pass is run before ISPC passes
;; llvm.genx.simdcf.predicate will not be lowered.
;; TODO_GEN: insert predication

define(`genx_masked_load', `
declare <WIDTH x $1> @llvm.genx.svm.block.ld.GEN_SUFFIX($1)(i64)
define <WIDTH x $1> @__masked_load_$1(i8 *, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %bitcast = bitcast i8* %0 to i32*
  %ptrtoint = ptrtoint i32* %bitcast to i32
  %zext = zext i32 %ptrtoint to i64
  %res = call <WIDTH x $1> @llvm.genx.svm.block.ld.GEN_SUFFIX($1)(i64 %zext)
  ret <WIDTH x $1> %res
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
declare <16 x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<16 x i1>, i32, <16 x i64>, <16 x $1>)
define <16 x $1>
@__gather_base_offsets32_$1(i8 * %ptr, i32 %offset_scale, <16 x i32> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %offsets64 = zext <16 x i32> %offsets to <16 x i64>
  %res = call <16 x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<16 x i1> %vecmask, i32 0, <16 x i64> %offsets64, <16 x $1> undef)
  ret <16 x $1> %res
}

define <16 x $1>
@__gather_base_offsets64_$1(i8 * %ptr, i32 %offset_scale, <16 x i64> %offsets, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x $1> @llvm.genx.svm.gather.GEN_SUFFIX($1).v16i1.v16i64(<16 x i1> %vecmask, i32 0, <16 x i64> %offsets, <16 x $1> undef)
  ret <16 x $1> %res
}

define <16 x $1>
@__gather32_$1(<16 x i32> %ptrs, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x $1> @__gather_base_offsets32_$1(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <WIDTH x MASK> %vecmask)
  ret <16 x $1> %res
}

define <16 x $1>
@__gather64_$1(<16x i64> %ptrs, <WIDTH x MASK> %vecmask) nounwind readonly alwaysinline {
  %res = call <16 x $1> @__gather_base_offsets64_$1(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <WIDTH x MASK> %vecmask)
  ret <16 x $1> %res
}
')
genx_gather(i8)
genx_gather(i16)
genx_gather(i32)
genx_gather(float)
genx_gather(i64)
genx_gather(double)

define(`genx_scatter', `
declare void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<16 x i1>, i32, <16 x i64>, <16 x $1>)
define void
@__scatter_base_offsets32_$1(i8* %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
  %offsets64 = zext <16 x i32> %offsets to <16 x i64>
  call void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<16 x i1> %vecmask, i32 0, <16 x i64> %offsets64, <16 x $1> %vals)
  ret void
}

define void
@__scatter_base_offsets64_$1(i8* %ptr, i32 %offset_scale, <16 x i64> %offsets, <16 x $1> %vals, <WIDTH x MASK> %vecmask) nounwind {
   call void @llvm.genx.svm.scatter.v16i1.v16i64.GEN_SUFFIX($1)(<16 x i1> %vecmask, i32 0, <16 x i64> %offsets, <16 x $1> %vals)
   ret void
}

define void
@__scatter32_$1(<16 x i32> %ptrs, <16 x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets32_$1(i8 * zeroinitializer, i32 1, <16 x i32> %ptrs, <16 x $1> %values, <WIDTH x MASK> %vecmask)
  ret void
}

define void
@__scatter64_$1(<16 x i64> %ptrs, <16 x $1> %values, <WIDTH x MASK> %vecmask) nounwind alwaysinline {
  call void @__scatter_base_offsets64_$1(i8 * zeroinitializer, i32 1, <16 x i64> %ptrs, <16 x $1> %values, <WIDTH x MASK> %vecmask)
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

declare <16 x i8> @__avg_up_uint8(<16 x i8>, <16 x i8>) nounwind readnone

declare <16 x i16> @__avg_up_uint16(<16 x i16>, <16 x i16>) nounwind readnone

define_avg_up_int8()
define_avg_up_int16()
define_down_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()
