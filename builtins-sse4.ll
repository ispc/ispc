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

; Define common 4-wide stuff
stdlib_core(4)
packed_load_and_store(4)
scans(4)

; Define the stuff that can be done with base SSE1/SSE2 instructions
include(`builtins-sse.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone
declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

define internal <4 x float> @__round_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 8)
  ret <4 x float> %call
}

define internal float @__round_uniform_float(float) nounwind readonly alwaysinline {
  ; roundss, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  ; the roundss intrinsic is a total mess--docs say:
  ;
  ;  __m128 _mm_round_ss (__m128 a, __m128 b, const int c)
  ;       
  ;  b is a 128-bit parameter. The lowest 32 bits are the result of the rounding function
  ;  on b0. The higher order 96 bits are copied directly from input parameter a. The
  ;  return value is described by the following equations:
  ;
  ;  r0 = RND(b0)
  ;  r1 = a1
  ;  r2 = a2
  ;  r3 = a3
  ;
  ;  It doesn't matter what we pass as a, since we only need the r0 value
  ;  here.  So we pass the same register for both.  Further, only the 0th
  ;  element of the b parameter matters
  %xi = insertelement <4 x float> undef, float %0, i32 0
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 8)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 9)
  ret <4 x float> %call
}

define internal float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define internal <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 10)
  ret <4 x float> %call
}

define internal float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) nounwind readnone
declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define internal <4 x double> @__round_varying_double(<4 x double>) nounwind readonly alwaysinline {
  round2to4double(%0, 8)
}

define internal double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define internal <4 x double> @__floor_varying_double(<4 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  round2to4double(%0, 9)
}

define internal double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define internal <4 x double> @__ceil_varying_double(<4 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  round2to4double(%0, 10)
}

define internal double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 10)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 min/max

declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define internal <4 x i32> @__min_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %call = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %call
}

define internal i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret i32 %ret
}

define internal <4 x i32> @__max_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %call = call <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %call
}

define internal i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned int min/max

declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

define internal <4 x i32> @__min_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %call = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %call
}

define internal i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret i32 %ret
}

define internal <4 x i32> @__max_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %call = call <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %call
}

define internal i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  sse_binary_scalar(ret, 4, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret i32 %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

define internal i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define internal i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define internal float @__reduce_add_float(<4 x float>) nounwind readonly alwaysinline {
  %v1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %0, <4 x float> %0)
  %v2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %v1, <4 x float> %v1)
  %scalar = extractelement <4 x float> %v2, i32 0
  ret float %scalar
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>,
                                             <4 x float>) nounwind readnone


define void @__masked_store_blend_32(<4 x i32>* nocapture, <4 x i32>, 
                                     <4 x i32> %mask) nounwind alwaysinline {
  %mask_as_float = bitcast <4 x i32> %mask to <4 x float>
  %oldValue = load <4 x i32>* %0, align 4
  %oldAsFloat = bitcast <4 x i32> %oldValue to <4 x float>
  %newAsFloat = bitcast <4 x i32> %1 to <4 x float>
  %blend = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %oldAsFloat,
                                                     <4 x float> %newAsFloat,
                                                     <4 x float> %mask_as_float)
  %blendAsInt = bitcast <4 x float> %blend to <4 x i32>
  store <4 x i32> %blendAsInt, <4 x i32>* %0, align 4
  ret void
}


define void @__masked_store_blend_64(<4 x i64>* nocapture %ptr, <4 x i64> %new,
                                     <4 x i32> %i32mask) nounwind alwaysinline {
  %oldValue = load <4 x i64>* %ptr, align 8
  %mask = bitcast <4 x i32> %i32mask to <4 x float>

  ; Do 4x64-bit blends by doing two <4 x i32> blends, where the <4 x i32> values
  ; are actually bitcast <2 x i64> values
  ;
  ; set up the first two 64-bit values
  %old01  = shufflevector <4 x i64> %oldValue, <4 x i64> undef,
                          <2 x i32> <i32 0, i32 1>
  %old01f = bitcast <2 x i64> %old01 to <4 x float>
  %new01  = shufflevector <4 x i64> %new, <4 x i64> undef,
                          <2 x i32> <i32 0, i32 1>
  %new01f = bitcast <2 x i64> %new01 to <4 x float>
  ; compute mask--note that the indices 0 and 1 are doubled-up
  %mask01 = shufflevector <4 x float> %mask, <4 x float> undef,
                          <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  ; and blend the two of the values
  %result01f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old01f,
                                                         <4 x float> %new01f,
                                                         <4 x float> %mask01)
  %result01 = bitcast <4 x float> %result01f to <2 x i64>

  ; and again
  %old23  = shufflevector <4 x i64> %oldValue, <4 x i64> undef,
                          <2 x i32> <i32 2, i32 3>
  %old23f = bitcast <2 x i64> %old23 to <4 x float>
  %new23  = shufflevector <4 x i64> %new, <4 x i64> undef,
                          <2 x i32> <i32 2, i32 3>
  %new23f = bitcast <2 x i64> %new23 to <4 x float>
  ; compute mask--note that the values 2 and 3 are doubled-up
  %mask23 = shufflevector <4 x float> %mask, <4 x float> undef,
                          <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  ; and blend the two of the values
  %result23f = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %old23f,
                                                         <4 x float> %new23f,
                                                         <4 x float> %mask23)
  %result23 = bitcast <4 x float> %result23f to <2 x i64>

  ; reconstruct the final <4 x i64> vector
  %final = shufflevector <2 x i64> %result01, <2 x i64> %result23,
                         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %final, <4 x i64> * %ptr, align 8
  ret void
}
