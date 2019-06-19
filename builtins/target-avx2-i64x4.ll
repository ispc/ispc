;;  Copyright (c) 2013-2019, Intel Corporation
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

define(`HAVE_GATHER', `1')

include(`target-avx1-i64x4base.ll')

rdrand_definition()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

;; declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone
;; declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readonly

define <4 x i32> @__min_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %m = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %m
}

define <4 x i32> @__max_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %m = call <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %m
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

;; declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readonly
;; declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readonly

define <4 x i32> @__min_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %m = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %m
}

define <4 x i32> @__max_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %m = call <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %m
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions



define(`expand_4to8', `
  %$3 = shufflevector <4 x $1> %$2, <4 x $1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
')
define(`extract_4from8', `
  %$3 = shufflevector <8 x $1> %$2, <8 x $1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
')

declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readnone
; 0 is round nearest even
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readnone

define <4 x float> @__half_to_float_varying(<4 x i16> %v4) nounwind readnone {
  expand_4to8(i16, v4, v) 
  %r = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %v)
  extract_4from8(float, r, ret)
  ret <4 x float> %ret
}

define <4 x i16> @__float_to_half_varying(<4 x float> %v4) nounwind readnone {
  expand_4to8(float, v4, v) 
  %r = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %v, i32 0)
  extract_4from8(i16, r, ret)
  ret <4 x i16> %ret
}

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %rv = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %vv)
  %r = extractelement <8 x float> %rv, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
           <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  ; round to nearest even
  %rv = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %vv, i32 0)
  %r = extractelement <8 x i16> %rv, i32 0
  ret i16 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

declare void @llvm.trap() noreturn nounwind

gen_gather(i8)
gen_gather(i16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 gathers

declare <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i32> %mask, i8 %scale) readonly nounwind
declare <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i32> %mask, i8 %scale) readonly nounwind

define <4 x i32> @__gather_base_offsets32_i32(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.d, 4, i32, ptr, offsets, i32, vecmask, i32, scale, i8)
  ret <4 x i32> %v
}


define <4 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.d.256, 4, i32, ptr, offsets, i64, vecmask, i32, scale, i8)

  ret <4 x i32> %v
}


define <4 x i32> @__gather32_i32(<4 x i32> %ptrs,
                                 <4 x i64> %vecmask64) nounwind readonly alwaysinline {

  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>

  %v = call <4 x i32> @llvm.x86.avx2.gather.d.d(<4 x i32> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x i32> %vecmask, i8 1)
  
  ret <4 x i32> %v
}


define <4 x i32> @__gather64_i32(<4 x i64> %ptrs, 
                                 <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>

  %v = call <4 x i32> @llvm.x86.avx2.gather.q.d.256(<4 x i32> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x i32> %vecmask, i8 1)

  ret <4 x i32> %v
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

declare <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind
declare <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x float> %mask, i8 %scale8) readonly nounwind

define <4 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <4 x i32> %offsets,
                                  <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.ps, 4, float, ptr, offsets, i32, mask, float, scale, i8)

  ret <4 x float> %v
}


define <4 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <4 x i64> %offsets,
                                   <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.ps.256, 4, float, ptr, offsets, i64, mask, float, scale, i8)

  ret <4 x float> %v
}


define <4 x float> @__gather32_float(<4 x i32> %ptrs, 
                                     <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  %v = call <4 x float> @llvm.x86.avx2.gather.d.ps(<4 x float> undef, i8 * null,
                     <4 x i32> %ptrs, <4 x float> %mask, i8 1)

  ret <4 x float> %v
}


define <4 x float> @__gather64_float(<4 x i64> %ptrs, 
                                     <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = trunc <4 x i64> %vecmask64 to <4 x i32>
  %mask = bitcast <4 x i32> %vecmask to <4 x float>

  %v = call <4 x float> @llvm.x86.avx2.gather.q.ps.256(<4 x float> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x float> %mask, i8 1)

  ret <4 x float> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

declare <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind
declare <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x i64> %mask, i8 %scale) readonly nounwind

define <4 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i64> %vecmask) nounwind readonly alwaysinline {

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.q.256, 4, i64, ptr, offsets, i32, vecmask, i64, scale, i8)

  ret <4 x i64> %v
}


define <4 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i64> %vecmask) nounwind readonly alwaysinline {

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.q.256, 4, i64, ptr, offsets, i64, vecmask, i64, scale, i8)

  ret <4 x i64> %v
}


define <4 x i64> @__gather32_i64(<4 x i32> %ptrs, 
                                 <4 x i64> %vecmask) nounwind readonly alwaysinline {

  %v = call <4 x i64> @llvm.x86.avx2.gather.d.q.256(<4 x i64> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x i64> %vecmask, i8 1)
  ret <4 x i64> %v
}


define <4 x i64> @__gather64_i64(<4 x i64> %ptrs, 
                                 <4 x i64> %vecmask) nounwind readonly alwaysinline {
  %v = call <4 x i64> @llvm.x86.avx2.gather.q.q.256(<4 x i64> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x i64> %vecmask, i8 1)
  ret <4 x i64> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i64> %indices, <4 x double> %mask, i8 %scale) readonly nounwind
declare <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> %target, i8 * %ptr,
                       <4 x i32> %indices, <4 x double> %mask, i8 %scale) readonly nounwind

define <4 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <4 x i32> %offsets,
                             <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  convert_scale_to_const(v, llvm.x86.avx2.gather.d.pd.256, 4, double, ptr, offsets, i32, vecmask, double, scale, i8)
  ret <4 x double> %v
}

define <4 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <4 x i64> %offsets,
                             <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  convert_scale_to_const(v, llvm.x86.avx2.gather.q.pd.256, 4, double, ptr, offsets, i64, vecmask, double, scale, i8)

  ret <4 x double> %v
}

define <4 x double> @__gather32_double(<4 x i32> %ptrs, 
                                       <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  %v = call <4 x double> @llvm.x86.avx2.gather.d.pd.256(<4 x double> undef, i8 * null,
                      <4 x i32> %ptrs, <4 x double> %vecmask, i8 1)

  ret <4 x double> %v
}

define <4 x double> @__gather64_double(<4 x i64> %ptrs, 
                                       <4 x i64> %vecmask64) nounwind readonly alwaysinline {
  %vecmask = bitcast <4 x i64> %vecmask64 to <4 x double>

  %v = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> undef, i8 * null,
                      <4 x i64> %ptrs, <4 x double> %vecmask, i8 1)

  ret <4 x double> %v
}
