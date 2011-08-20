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
;; Basic 8-wide definitions

stdlib_core(8)
packed_load_and_store(8)
scans(8)
int64minmax(8)

include(`builtins-avx-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

define internal <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  %call = call <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %0)
  ; do one N-R iteration
  %v_iv = fmul <8 x float> %0, %call
  %two_minus = fsub <8 x float> <float 2., float 2., float 2., float 2.,
                                 float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <8 x float> %call, %two_minus
  ret <8 x float> %iv_mul
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone

define internal <8 x float> @__round_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  %call = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %0, i32 8)
  ret <8 x float> %call
}

define internal <8 x float> @__floor_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %call = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %0, i32 9)
  ret <8 x float> %call
}

define internal <8 x float> @__ceil_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %call = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %0, i32 10)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone

define internal <8 x double> @__round_varying_double(<8 x double>) nounwind readonly alwaysinline {
  round4to8double(%0, 8)
}

define internal <8 x double> @__floor_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  round4to8double(%0, 9)
}


define internal <8 x double> @__ceil_varying_double(<8 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  round4to8double(%0, 10)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone

define internal <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  %is = call <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float> %v)
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <8 x float> %v, %is
  %v_is_is = fmul <8 x float> %v_is, %is
  %three_sub = fsub <8 x float> <float 3., float 3., float 3., float 3., float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <8 x float> %is, %three_sub
  %half_scale = fmul <8 x float> <float 0.5, float 0.5, float 0.5, float 0.5, float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <8 x float> %half_scale
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone

define internal <8 x float> @__sqrt_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %call = call <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float> %0)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

declare <8 x float> @__svml_sin(<8 x float>)
declare <8 x float> @__svml_cos(<8 x float>)
declare void @__svml_sincos(<8 x float>, <8 x float> *, <8 x float> *)
declare <8 x float> @__svml_tan(<8 x float>)
declare <8 x float> @__svml_atan(<8 x float>)
declare <8 x float> @__svml_atan2(<8 x float>, <8 x float>)
declare <8 x float> @__svml_exp(<8 x float>)
declare <8 x float> @__svml_log(<8 x float>)
declare <8 x float> @__svml_pow(<8 x float>, <8 x float>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define internal <8 x float> @__max_varying_float(<8 x float>,
                                                 <8 x float>) nounwind readonly alwaysinline {
  %call = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %0, <8 x float> %1)
  ret <8 x float> %call
}

define internal <8 x float> @__min_varying_float(<8 x float>,
                                                 <8 x float>) nounwind readonly alwaysinline {
  %call = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %0, <8 x float> %1)
  ret <8 x float> %call
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define internal <8 x i32> @__min_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(ret, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <8 x i32> %ret
}

define internal <8 x i32> @__max_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(ret, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <8 x i32> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define internal <8 x i32> @__min_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(ret, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <8 x i32> %ret
}

define internal <8 x i32> @__max_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  binary4to8(ret, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <8 x i32> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone

define internal i32 @__movmsk(<8 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %v = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask) nounwind readnone
  ret i32 %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define internal float @__reduce_add_float(<8 x float>) nounwind readonly alwaysinline {
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %0, <8 x float> %0)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %scalar1 = extractelement <8 x float> %v2, i32 0
  %scalar2 = extractelement <8 x float> %v2, i32 1
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}


define internal float @__reduce_min_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8(float, @__min_varying_float, @__min_uniform_float)
}


define internal float @__reduce_max_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8(float, @__max_varying_float, @__max_uniform_float)
}

reduce_equal(8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define internal <8 x i32> @__add_varying_int32(<8 x i32>,
                                               <8 x i32>) nounwind readnone alwaysinline {
  %s = add <8 x i32> %0, %1
  ret <8 x i32> %s
}

define internal i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define internal i32 @__reduce_add_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__add_varying_int32, @__add_uniform_int32)
}


define internal i32 @__reduce_min_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__min_varying_int32, @__min_uniform_int32)
}


define internal i32 @__reduce_max_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__max_varying_int32, @__max_uniform_int32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define internal i32 @__reduce_add_uint32(<8 x i32> %v) nounwind readnone alwaysinline {
  %r = call i32 @__reduce_add_int32(<8 x i32> %v)
  ret i32 %r
}

define internal i32 @__reduce_min_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__min_varying_uint32, @__min_uniform_uint32)
}


define internal i32 @__reduce_max_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__max_varying_uint32, @__max_uniform_uint32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define internal double @__reduce_add_double(<8 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <8 x double> %0, <8 x double> undef,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v1 = shufflevector <8 x double> %0, <8 x double> undef,
                      <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %v0, <4 x double> %v1)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %scalar1 = extractelement <4 x double> %sum0, i32 0
  %scalar2 = extractelement <4 x double> %sum1, i32 1
  %sum = fadd double %scalar1, %scalar2
  ret double %sum
}

define internal double @__reduce_min_double(<8 x double>) nounwind readnone alwaysinline {
  reduce8(double, @__min_varying_double, @__min_uniform_double)
}


define internal double @__reduce_max_double(<8 x double>) nounwind readnone alwaysinline {
  reduce8(double, @__max_varying_double, @__max_uniform_double)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define internal <8 x i64> @__add_varying_int64(<8 x i64>,
                                               <8 x i64>) nounwind readnone alwaysinline {
  %s = add <8 x i64> %0, %1
  ret <8 x i64> %s
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define internal i64 @__reduce_add_int64(<8 x i64>) nounwind readnone alwaysinline {
  reduce8(i64, @__add_varying_int64, @__add_uniform_int64)
}


define internal i64 @__reduce_min_int64(<8 x i64>) nounwind readnone alwaysinline {
  reduce8(i64, @__min_varying_int64, @__min_uniform_int64)
}


define internal i64 @__reduce_max_int64(<8 x i64>) nounwind readnone alwaysinline {
  reduce8(i64, @__max_varying_int64, @__max_uniform_int64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define internal i64 @__reduce_add_uint64(<8 x i64> %v) nounwind readnone alwaysinline {
  %r = call i64 @__reduce_add_int64(<8 x i64> %v)
  ret i64 %r
}

define internal i64 @__reduce_min_uint64(<8 x i64>) nounwind readnone alwaysinline {
  reduce8(i64, @__min_varying_uint64, @__min_uniform_uint64)
}


define internal i64 @__reduce_max_uint64(<8 x i64>) nounwind readnone alwaysinline {
  reduce8(i64, @__max_varying_uint64, @__max_uniform_uint64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

load_and_broadcast(8, i8, 8)
load_and_broadcast(8, i16, 16)
load_and_broadcast(8, i32, 32)
load_and_broadcast(8, i64, 64)

; no masked load instruction for i8 and i16 types??
load_masked(8, i8,  8,  1)
load_masked(8, i16, 16, 2)

declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8 *, <8 x float> %mask)
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8 *, <4 x double> %mask)
 
define <8 x i32> @__load_masked_32(i8 *, <8 x i32> %mask) nounwind alwaysinline {
  %floatmask = bitcast <8 x i32> %mask to <8 x float>
  %floatval = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8 * %0, <8 x float> %floatmask)
  %retval = bitcast <8 x float> %floatval to <8 x i32>
  ret <8 x i32> %retval
}


define <8 x i64> @__load_masked_64(i8 *, <8 x i32> %mask) nounwind alwaysinline {
  ; double up masks, bitcast to doubles
  %mask0 = shufflevector <8 x i32> %mask, <8 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <8 x i32> %mask, <8 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask0d = bitcast <8 x i32> %mask0 to <4 x double>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x double>

  %val0d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %0, <4 x double> %mask0d)
  %ptr1 = getelementptr i8 * %0, i32 32
  %val1d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr1, <4 x double> %mask1d)

  %vald = shufflevector <4 x double> %val0d, <4 x double> %val1d,
      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val = bitcast <8 x double> %vald to <8 x i64>
  ret <8 x i64> %val
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

; FIXME: there is no AVX instruction for these, but we could be clever
; by packing the bits down and setting the last 3/4 or half, respectively,
; of the mask to zero...  Not sure if this would be a win in the end
gen_masked_store(8, i8, 8)
gen_masked_store(8, i16, 16)

; note that mask is the 2nd parameter, not the 3rd one!!
declare void @llvm.x86.avx.maskstore.ps.256(i8 *, <8 x float>, <8 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8 *, <4 x double>, <4 x double>)

define void @__masked_store_32(<8 x i32>* nocapture, <8 x i32>, 
                               <8 x i32>) nounwind alwaysinline {
  %ptr = bitcast <8 x i32> * %0 to i8 *
  %val = bitcast <8 x i32> %1 to <8 x float>
  %mask = bitcast <8 x i32> %2 to <8 x float>
  call void @llvm.x86.avx.maskstore.ps.256(i8 * %ptr, <8 x float> %mask, <8 x float> %val)
  ret void
}

define void @__masked_store_64(<8 x i64>* nocapture, <8 x i64>,
                               <8 x i32> %mask) nounwind alwaysinline {
  %ptr = bitcast <8 x i64> * %0 to i8 *
  %val = bitcast <8 x i64> %1 to <8 x double>

  %mask0 = shufflevector <8 x i32> %mask, <8 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <8 x i32> %mask, <8 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>

  %mask0d = bitcast <8 x i32> %mask0 to <4 x double>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x double>

  %val0 = shufflevector <8 x double> %val, <8 x double> undef,
     <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val1 = shufflevector <8 x double> %val, <8 x double> undef,
     <4 x i32> <i32 4, i32 5, i32 6, i32 7>

  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr, <4 x double> %mask0d, <4 x double> %val0)
  %ptr1 = getelementptr i8 * %ptr, i32 32
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr1, <4 x double> %mask1d, <4 x double> %val1)
  ret void
}

masked_store_blend_8_16_by_8()

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>,
                                                <8 x float>) nounwind readnone


define void @__masked_store_blend_32(<8 x i32>* nocapture, <8 x i32>,
                                     <8 x i32>) nounwind alwaysinline {
  %mask_as_float = bitcast <8 x i32> %2 to <8 x float>
  %oldValue = load <8 x i32>* %0, align 4
  %oldAsFloat = bitcast <8 x i32> %oldValue to <8 x float>
  %newAsFloat = bitcast <8 x i32> %1 to <8 x float>
  %blend = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %oldAsFloat,
                                                        <8 x float> %newAsFloat,
                                                        <8 x float> %mask_as_float)
  %blendAsInt = bitcast <8 x float> %blend to <8 x i32>
  store <8 x i32> %blendAsInt, <8 x i32>* %0, align 4
  ret void
}


define void @__masked_store_blend_64(<8 x i64>* nocapture %ptr, <8 x i64> %new,
                                     <8 x i32> %i32mask) nounwind alwaysinline {
  %oldValue = load <8 x i64>* %ptr, align 8
  %mask = bitcast <8 x i32> %i32mask to <8 x float>

  ; Do 4x64-bit blends by doing two <8 x i32> blends, where the <8 x i32> values
  ; are actually bitcast <4 x i64> values
  ;
  ; set up the first four 64-bit values
  %old01  = shufflevector <8 x i64> %oldValue, <8 x i64> undef,
                          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %old01f = bitcast <4 x i64> %old01 to <8 x float>
  %new01  = shufflevector <8 x i64> %new, <8 x i64> undef,
                          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %new01f = bitcast <4 x i64> %new01 to <8 x float>
  ; compute mask--note that the indices are all doubled-up
  %mask01 = shufflevector <8 x float> %mask, <8 x float> undef,
                          <8 x i32> <i32 0, i32 0, i32 1, i32 1,
                                     i32 2, i32 2, i32 3, i32 3>
  ; and blend them
  %result01f = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %old01f,
                                                            <8 x float> %new01f,
                                                            <8 x float> %mask01)
  %result01 = bitcast <8 x float> %result01f to <4 x i64>

  ; and again
  %old23  = shufflevector <8 x i64> %oldValue, <8 x i64> undef,
                          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %old23f = bitcast <4 x i64> %old23 to <8 x float>
  %new23  = shufflevector <8 x i64> %new, <8 x i64> undef,
                          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %new23f = bitcast <4 x i64> %new23 to <8 x float>
  ; compute mask--note that the values are doubled-up...
  %mask23 = shufflevector <8 x float> %mask, <8 x float> undef,
                          <8 x i32> <i32 4, i32 4, i32 5, i32 5,
                                     i32 6, i32 6, i32 7, i32 7>
  ; and blend them
  %result23f = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %old23f,
                                                            <8 x float> %new23f,
                                                            <8 x float> %mask23)
  %result23 = bitcast <8 x float> %result23f to <4 x i64>

  ; reconstruct the final <8 x i64> vector
  %final = shufflevector <4 x i64> %result01, <4 x i64> %result23,
                         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
                                    i32 4, i32 5, i32 6, i32 7>
  store <8 x i64> %final, <8 x i64> * %ptr, align 8
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

gen_gather(8, i8)
gen_gather(8, i16)
gen_gather(8, i32)
gen_gather(8, i64)

gen_scatter(8, i8)
gen_scatter(8, i16)
gen_scatter(8, i32)
gen_scatter(8, i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define internal <8 x double> @__sqrt_varying_double(<8 x double>) nounwind alwaysinline {
  unary4to8(ret, double, @llvm.x86.avx.sqrt.pd.256, %0)
  ret <8 x double> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define internal <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary4to8(ret, double, @llvm.x86.avx.min.pd.256, %0, %1)
  ret <8 x double> %ret
}

define internal <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary4to8(ret, double, @llvm.x86.avx.max.pd.256, %0, %1)
  ret <8 x double> %ret
}

