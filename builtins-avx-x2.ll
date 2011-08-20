;;  Copyright (c) 2010-2011, Intel Corporation
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; *** Untested *** AVX target implementation.
;;
;; The LLVM AVX code generator is incomplete, so the ispc AVX target
;; hasn't yet been tested.  There is therefore a higher-than-normal
;; chance that there are bugs in the code in this file.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Basic 16-wide definitions

stdlib_core(16)
packed_load_and_store(16)
scans(16)
int64minmax(16)

include(`builtins-avx-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

define internal <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  unary8to16(call, float, @llvm.x86.avx.rcp.ps.256, %0)
  ; do one N-R iteration
  %v_iv = fmul <16 x float> %0, %call
  %two_minus = fsub <16 x float> <float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.,
                                  float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <16 x float> %call, %two_minus
  ret <16 x float> %iv_mul
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone

define internal <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round8to16(%0, 8)
}

define internal <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round8to16(%0, 9)
}

define internal <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round8to16(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone

define internal <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 8)
}

define internal <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 9)
}

define internal <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 10)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone

define internal <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary8to16(is, float, @llvm.x86.avx.rsqrt.ps.256, %v)
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x float> %v, %is
  %v_is_is = fmul <16 x float> %v_is, %is
  %three_sub = fsub <16 x float> <float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.,
                                  float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <16 x float> %is, %three_sub
  %half_scale = fmul <16 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5,
                                   float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <16 x float> %half_scale
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone

define internal <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  unary8to16(call, float, @llvm.x86.avx.sqrt.ps.256, %0)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones 4x with our 16-wide
; vectors...

declare <16 x float> @__svml_sin(<16 x float>)
declare <16 x float> @__svml_cos(<16 x float>)
declare void @__svml_sincos(<16 x float>, <16 x float> *, <16 x float> *)
declare <16 x float> @__svml_tan(<16 x float>)
declare <16 x float> @__svml_atan(<16 x float>)
declare <16 x float> @__svml_atan2(<16 x float>, <16 x float>)
declare <16 x float> @__svml_exp(<16 x float>)
declare <16 x float> @__svml_log(<16 x float>)
declare <16 x float> @__svml_pow(<16 x float>, <16 x float>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define internal <16 x float> @__max_varying_float(<16 x float>,
                                                  <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.max.ps.256, %0, %1)
  ret <16 x float> %call
}

define internal <16 x float> @__min_varying_float(<16 x float>,
                                                  <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.min.ps.256, %0, %1)
  ret <16 x float> %call
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define internal <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <16 x i32> %ret
}

define internal <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <16 x i32> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define internal <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <16 x i32> %ret
}

define internal <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <16 x i32> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone

define internal i32 @__movmsk(<16 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <16 x i32> %0 to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) nounwind readnone
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask1) nounwind readnone

  %v1shift = shl i32 %v1, 8
  %v = or i32 %v1shift, %v0
  ret i32 %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define internal float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vb = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %va, <8 x float> %vb)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %v3 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v2, <8 x float> %v2)
  %scalar1 = extractelement <8 x float> %v2, i32 0
  %scalar2 = extractelement <8 x float> %v2, i32 4
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}


define internal float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}


define internal float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

reduce_equal(16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define internal <16 x i32> @__add_varying_int32(<16 x i32>,
                                                <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define internal i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}


define internal i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}


define internal i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define internal i32 @__reduce_add_uint32(<16 x i32> %v) nounwind readnone alwaysinline {
  %r = call i32 @__reduce_add_int32(<16 x i32> %v)
  ret i32 %r
}

define internal i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}


define internal i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define internal double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vb = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vc = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %vd = shufflevector <16 x double> %0, <16 x double> undef,
         <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %vab = fadd <4 x double> %va, %vb
  %vcd = fadd <4 x double> %vc, %vd

  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %vab, <4 x double> %vcd)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %scalar1 = extractelement <4 x double> %sum0, i32 0
  %scalar2 = extractelement <4 x double> %sum1, i32 1
  %sum = fadd double %scalar1, %scalar2
  ret double %sum
}

define internal double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}


define internal double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define internal <16 x i64> @__add_varying_int64(<16 x i64>,
                                                <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define internal i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}


define internal i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}


define internal i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define internal i64 @__reduce_add_uint64(<16 x i64> %v) nounwind readnone alwaysinline {
  %r = call i64 @__reduce_add_int64(<16 x i64> %v)
  ret i64 %r
}

define internal i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}


define internal i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

load_and_broadcast(16, i8, 8)
load_and_broadcast(16, i16, 16)
load_and_broadcast(16, i32, 32)
load_and_broadcast(16, i64, 64)

; no masked load instruction for i8 and i16 types??
load_masked(16, i8,  8,  1)
load_masked(16, i16, 16, 2)

declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8 *, <8 x float> %mask)
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8 *, <4 x double> %mask)
 
define <16 x i32> @__load_masked_32(i8 *, <16 x i32> %mask) nounwind alwaysinline {
  %floatmask = bitcast <16 x i32> %mask to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val0 = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8 * %0, <8 x float> %mask0)
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
     <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %ptr1 = getelementptr i8 * %0, i32 32   ;; 8x4 bytes = 32
  %val1 = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8 * %ptr1, <8 x float> %mask1)

  %retval = shufflevector <8 x float> %val0, <8 x float> %val1,
     <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %reti32 = bitcast <16 x float> %retval to <16 x i32>
  ret <16 x i32> %reti32
}


define <16 x i64> @__load_masked_64(i8 *, <16 x i32> %mask) nounwind alwaysinline {
  ; double up masks, bitcast to doubles
  %mask0 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask2 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11>
  %mask3 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  %mask0d = bitcast <8 x i32> %mask0 to <4 x double>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x double>
  %mask2d = bitcast <8 x i32> %mask2 to <4 x double>
  %mask3d = bitcast <8 x i32> %mask3 to <4 x double>

  %val0d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %0, <4 x double> %mask0d)
  %ptr1 = getelementptr i8 * %0, i32 32
  %val1d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr1, <4 x double> %mask1d)
  %ptr2 = getelementptr i8 * %0, i32 64
  %val2d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr2, <4 x double> %mask2d)
  %ptr3 = getelementptr i8 * %0, i32 96
  %val3d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr3, <4 x double> %mask3d)

  %val01 = shufflevector <4 x double> %val0d, <4 x double> %val1d,
      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val23 = shufflevector <4 x double> %val2d, <4 x double> %val3d,
      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val0123 = shufflevector <8 x double> %val01, <8 x double> %val23,
      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %val = bitcast <16 x double> %val0123 to <16 x i64>
  ret <16 x i64> %val
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

; FIXME: there is no AVX instruction for these, but we could be clever
; by packing the bits down and setting the last 3/4 or half, respectively,
; of the mask to zero...  Not sure if this would be a win in the end
gen_masked_store(16, i8, 8)
gen_masked_store(16, i16, 16)

; note that mask is the 2nd parameter, not the 3rd one!!
declare void @llvm.x86.avx.maskstore.ps.256(i8 *, <8 x float>, <8 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8 *, <4 x double>, <4 x double>)

define void @__masked_store_32(<16 x i32>* nocapture, <16 x i32>, 
                               <16 x i32>) nounwind alwaysinline {
  %ptr = bitcast <16 x i32> * %0 to i8 *
  %val = bitcast <16 x i32> %1 to <16 x float>
  %mask = bitcast <16 x i32> %2 to <16 x float>

  %val0 = shufflevector <16 x float> %val, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val1 = shufflevector <16 x float> %val, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %mask0 = shufflevector <16 x float> %mask, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mask1 = shufflevector <16 x float> %mask, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx.maskstore.ps.256(i8 * %ptr, <8 x float> %mask0, <8 x float> %val0)
  %ptr1 = getelementptr i8 * %ptr, i32 32
  call void @llvm.x86.avx.maskstore.ps.256(i8 * %ptr1, <8 x float> %mask1, <8 x float> %val1)

  ret void
}

define void @__masked_store_64(<16 x i64>* nocapture, <16 x i64>,
                               <16 x i32> %mask) nounwind alwaysinline {
  %ptr = bitcast <16 x i64> * %0 to i8 *
  %val = bitcast <16 x i64> %1 to <16 x double>

  ; double up masks, bitcast to doubles
  %mask0 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask2 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11>
  %mask3 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  %mask0d = bitcast <8 x i32> %mask0 to <4 x double>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x double>
  %mask2d = bitcast <8 x i32> %mask2 to <4 x double>
  %mask3d = bitcast <8 x i32> %mask3 to <4 x double>

  %val0 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val1 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %val2 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %val3 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr, <4 x double> %mask0d, <4 x double> %val0)
  %ptr1 = getelementptr i8 * %ptr, i32 32
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr1, <4 x double> %mask1d, <4 x double> %val1)
  %ptr2 = getelementptr i8 * %ptr, i32 64
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr2, <4 x double> %mask2d, <4 x double> %val2)
  %ptr3 = getelementptr i8 * %ptr, i32 96
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr3, <4 x double> %mask3d, <4 x double> %val3)

  ret void
}

masked_store_blend_8_16_by_16()

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>,
                                                <8 x float>) nounwind readnone


define void @__masked_store_blend_32(<16 x i32>* nocapture, <16 x i32>,
                                     <16 x i32>) nounwind alwaysinline {
  %maskAsFloat = bitcast <16 x i32> %2 to <16 x float>
  %oldValue = load <16 x i32>* %0, align 4
  %oldAsFloat = bitcast <16 x i32> %oldValue to <16 x float>
  %newAsFloat = bitcast <16 x i32> %1 to <16 x float>

  %old0 = shufflevector <16 x float> %oldAsFloat, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %old1 = shufflevector <16 x float> %oldAsFloat, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %new0 = shufflevector <16 x float> %newAsFloat, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %new1 = shufflevector <16 x float> %newAsFloat, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %mask0 = shufflevector <16 x float> %maskAsFloat, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mask1 = shufflevector <16 x float> %maskAsFloat, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %blend0 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %old0,
                                                         <8 x float> %new0,
                                                         <8 x float> %mask0)
  %blend1 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %old1,
                                                         <8 x float> %new1,
                                                         <8 x float> %mask1)
  %blend = shufflevector <8 x float> %blend0, <8 x float> %blend1,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %blendAsInt = bitcast <16 x float> %blend to <16 x i32>
  store <16 x i32> %blendAsInt, <16 x i32>* %0, align 4
  ret void
}


declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>,
                                                 <4 x double>) nounwind readnone

define void @__masked_store_blend_64(<16 x i64>* nocapture %ptr, <16 x i64> %newi64,
                                     <16 x i32> %mask) nounwind alwaysinline {
  %oldValue = load <16 x i64>* %ptr, align 8
  %old = bitcast <16 x i64> %oldValue to <16 x double>
  %old0d = shufflevector <16 x double> %old, <16 x double> undef,
     <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %old1d = shufflevector <16 x double> %old, <16 x double> undef,
     <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %old2d = shufflevector <16 x double> %old, <16 x double> undef,
     <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %old3d = shufflevector <16 x double> %old, <16 x double> undef,
     <4 x i32> <i32 12, i32 13, i32 14, i32 15>

  %new = bitcast <16 x i64> %newi64 to <16 x double>
  %new0d = shufflevector <16 x double> %new, <16 x double> undef,
     <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %new1d = shufflevector <16 x double> %new, <16 x double> undef,
     <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %new2d = shufflevector <16 x double> %new, <16 x double> undef,
     <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %new3d = shufflevector <16 x double> %new, <16 x double> undef,
     <4 x i32> <i32 12, i32 13, i32 14, i32 15>

  %mask0 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask2 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11>
  %mask3 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  %mask0d = bitcast <8 x i32> %mask0 to <4 x double>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x double>
  %mask2d = bitcast <8 x i32> %mask2 to <4 x double>
  %mask3d = bitcast <8 x i32> %mask3 to <4 x double>

  %result0d = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %old0d,
                                 <4 x double> %new0d, <4 x double> %mask0d)
  %result1d = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %old1d,
                                 <4 x double> %new1d, <4 x double> %mask1d)
  %result2d = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %old2d,
                                 <4 x double> %new2d, <4 x double> %mask2d)
  %result3d = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %old3d,
                                 <4 x double> %new3d, <4 x double> %mask3d)

  %result01 = shufflevector <4 x double> %result0d, <4 x double> %result1d,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %result23 = shufflevector <4 x double> %result2d, <4 x double> %result3d,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

  %result = shufflevector <8 x double> %result01, <8 x double> %result23,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %result64 = bitcast <16 x double> %result to <16 x i64>
  store <16 x i64> %result64, <16 x i64> * %ptr
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather(16, i8)
gen_gather(16, i16)
gen_gather(16, i32)
gen_gather(16, i64)

gen_scatter(16, i8)
gen_scatter(16, i16)
gen_scatter(16, i32)
gen_scatter(16, i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define internal <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  unary4to16(ret, double, @llvm.x86.avx.sqrt.pd.256, %0)
  ret <16 x double> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define internal <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.min.pd.256, %0, %1)
  ret <16 x double> %ret
}

define internal <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.max.pd.256, %0, %1)
  ret <16 x double> %ret
}
