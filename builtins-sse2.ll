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
;; Define the standard library builtins for the SSE2 target

; Define some basics for a 4-wide target
stdlib_core(4)
packed_load_and_store(4)
scans(4)

; Include the various definitions of things that only require SSE1 and SSE2
include(`builtins-sse.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding
;;
;; There are not any rounding instructions in SSE2, so we have to emulate
;; the functionality with multiple instructions...

; The code for __round_* is the result of compiling the following source
; code.
;
; export float Round(float x) {
;    unsigned int sign = signbits(x);
;    unsigned int ix = intbits(x);
;    ix ^= sign;
;    x = floatbits(ix);
;    x += 0x1.0p23f;
;    x -= 0x1.0p23f;
;    ix = intbits(x);
;    ix ^= sign;
;    x = floatbits(ix);
;    return x;
;}

define internal <4 x float> @__round_varying_float(<4 x float>) nounwind readonly alwaysinline {
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

define internal float @__round_uniform_float(float) nounwind readonly alwaysinline {
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

;; Similarly, for implementations of the __floor* functions below, we have the
;; bitcode from compiling the following source code...

;export float Floor(float x) {
;    float y = Round(x);
;    unsigned int cmp = y > x ? 0xffffffff : 0;
;    float delta = -1.f;
;    unsigned int idelta = intbits(delta);
;    idelta &= cmp;
;    delta = floatbits(idelta);
;    return y + delta;
;}

define internal <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp ogt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

define internal float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp ogt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, -1082130432
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

;; And here is the code we compiled to get the __ceil* functions below
;
;export uniform float Ceil(uniform float x) {
;    uniform float y = Round(x);
;    uniform int yltx = y < x ? 0xffffffff : 0;
;    uniform float delta = 1.f;
;    uniform int idelta = intbits(delta);
;    idelta &= yltx;
;    delta = floatbits(idelta);
;    return y + delta;
;}

define internal <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp olt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

define internal float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp olt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, 1065353216
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare double @round(double)
declare double @floor(double)
declare double @ceil(double)

define internal <4 x double> @__round_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @round)
}

define internal double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @round(double %0)
  ret double %r
}

define internal <4 x double> @__floor_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @floor)
}

define internal double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @floor(double %0)
  ret double %r
}

define internal <4 x double> @__ceil_varying_double(<4 x double>) nounwind readonly alwaysinline {
  unary1to4(double, @ceil)
}

define internal double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @ceil(double %0)
  ret double %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

; There is no blend instruction with SSE2, so we simulate it with bit
; operations on i32s.  For these two vselect functions, for each
; vector element, if the mask is on, we return the corresponding value
; from %1, and otherwise return the value from %0.

define internal <4 x i32> @__vselect_i32(<4 x i32>, <4 x i32> ,
                                         <4 x i32> %mask) nounwind readnone alwaysinline {
  %notmask = xor <4 x i32> %mask, <i32 -1, i32 -1, i32 -1, i32 -1>
  %cleared_old = and <4 x i32> %0, %notmask
  %masked_new = and <4 x i32> %1, %mask
  %new = or <4 x i32> %cleared_old, %masked_new
  ret <4 x i32> %new
}

define internal <4 x float> @__vselect_float(<4 x float>, <4 x float>,
                                             <4 x i32> %mask) nounwind readnone alwaysinline {
  %v0 = bitcast <4 x float> %0 to <4 x i32>
  %v1 = bitcast <4 x float> %1 to <4 x i32>
  %r = call <4 x i32> @__vselect_i32(<4 x i32> %v0, <4 x i32> %v1, <4 x i32> %mask)
  %rf = bitcast <4 x i32> %r to <4 x float>
  ret <4 x float> %rf
}


; To do vector integer min and max, we do the vector compare and then sign
; extend the i1 vector result to an i32 mask.  The __vselect does the
; rest...

define internal <4 x i32> @__min_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp slt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define internal i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp slt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define internal <4 x i32> @__max_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp sgt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define internal i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp sgt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

; The functions for unsigned ints are similar, just with unsigned
; comparison functions...

define internal <4 x i32> @__min_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ult <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define internal i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ult i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define internal <4 x i32> @__max_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ugt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define internal i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ugt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

; FIXME: this is very inefficient, loops over all 32 bits...

; we could use the LLVM intrinsic declare i32 @llvm.ctpop.i32(i32),
; although that currently ends up generating a POPCNT instruction even
; if we give --target=sse2 on the command line.  We probably need to
; pipe through the 'sse2' request to LLVM via the 'features' string
; at codegen time...  (If e.g. --cpu=penryn is also passed along, then
; it does generate non-POPCNT code and in particular better code than
; the below does.)

define internal i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
entry:
  br label %loop

loop:
  %count = phi i32 [ 0, %entry ], [ %newcount, %loop ]
  %val = phi i32 [ %0, %entry ], [ %newval, %loop ]
  %delta = and i32 %val, 1
  %newcount = add i32 %count, %delta
  %newval = lshr i32 %val, 1
  %done = icmp eq i32 %newval, 0
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %newcount
}

define internal i32 @__popcnt_int64(i64) nounwind readnone alwaysinline {
  %vec = bitcast i64 %0 to <2 x i32>
  %v0 = extractelement <2 x i32> %vec, i32 0
  %v1 = extractelement <2 x i32> %vec, i32 1
  %c0 = call i32 @__popcnt_int32(i32 %v0)
  %c1 = call i32 @__popcnt_int32(i32 %v1)
  %sum = add i32 %c0, %c1
  ret i32 %sum
}


define internal float @__reduce_add_float(<4 x float> %v) nounwind readonly alwaysinline {
  %v1 = shufflevector <4 x float> %v, <4 x float> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = fadd <4 x float> %v1, %v
  %m1a = extractelement <4 x float> %m1, i32 0
  %m1b = extractelement <4 x float> %m1, i32 1
  %sum = fadd float %m1a, %m1b
  ret float %sum
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

define void @__masked_store_blend_32(<4 x i32>* nocapture, <4 x i32>, 
                                     <4 x i32> %mask) nounwind alwaysinline {
  %val = load <4 x i32> * %0, align 4
  %newval = call <4 x i32> @__vselect_i32(<4 x i32> %val, <4 x i32> %1, <4 x i32> %mask) 
  store <4 x i32> %newval, <4 x i32> * %0, align 4
  ret void
}

define void @__masked_store_blend_64(<4 x i64>* nocapture %ptr, <4 x i64> %new,
                                     <4 x i32> %mask) nounwind alwaysinline {
  %oldValue = load <4 x i64>* %ptr, align 8

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
  %mask01 = shufflevector <4 x i32> %mask, <4 x i32> undef,
                          <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  ; and blend the two of the values
  %result01f = call <4 x float> @__vselect_float(<4 x float> %old01f, <4 x float> %new01f, <4 x i32> %mask01)
  %result01 = bitcast <4 x float> %result01f to <2 x i64>

  ; and again
  %old23  = shufflevector <4 x i64> %oldValue, <4 x i64> undef,
                          <2 x i32> <i32 2, i32 3>
  %old23f = bitcast <2 x i64> %old23 to <4 x float>
  %new23  = shufflevector <4 x i64> %new, <4 x i64> undef,
                          <2 x i32> <i32 2, i32 3>
  %new23f = bitcast <2 x i64> %new23 to <4 x float>
  ; compute mask--note that the values 2 and 3 are doubled-up
  %mask23 = shufflevector <4 x i32> %mask, <4 x i32> undef,
                          <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  ; and blend the two of the values
  %result23f = call <4 x float> @__vselect_float(<4 x float> %old23f, <4 x float> %new23f, <4 x i32> %mask23)
  %result23 = bitcast <4 x float> %result23f to <2 x i64>

  ; reconstruct the final <4 x i64> vector
  %final = shufflevector <2 x i64> %result01, <2 x i64> %result23,
                         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %final, <4 x i64> * %ptr, align 8
  ret void
}

