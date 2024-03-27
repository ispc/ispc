;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Define the standard library builtins for the SSE2 target

; Define some basics for a 4-wide target
define(`WIDTH',`4')
define(`MASK',`i32')
define(`ISA',`SSE2')
include(`util.m4')

stdlib_core()
packed_load_and_store(FALSE)
scans()
int64minmax()
saturation_arithmetic()

include(`target-sse2-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone

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

define <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp ogt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
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

define <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp olt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

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
;; min/max

; There is no blend instruction with SSE2, so we simulate it with bit
; operations on i32s.  For these two vselect functions, for each
; vector element, if the mask is on, we return the corresponding value
; from %1, and otherwise return the value from %0.

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


; To do vector integer min and max, we do the vector compare and then sign
; extend the i1 vector result to an i32 mask.  The __vselect does the
; rest...

define <4 x i32> @__min_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp slt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp slt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define <4 x i32> @__max_varying_int32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp sgt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp sgt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

; The functions for unsigned ints are similar, just with unsigned
; comparison functions...

define <4 x i32> @__min_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ult <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ult i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define <4 x i32> @__max_varying_uint32(<4 x i32>, <4 x i32>) nounwind readonly alwaysinline {
  %c = icmp ugt <4 x i32> %0, %1
  %mask = sext <4 x i1> %c to <4 x i32>
  %v = call <4 x i32> @__vselect_i32(<4 x i32> %1, <4 x i32> %0, <4 x i32> %mask)
  ret <4 x i32> %v
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ugt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

define i64 @__movmsk(<4 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i32> %0 to <4 x float>
  %v = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %floatmask) nounwind readnone
  %v64 = zext i32 %v to i64
  ret i64 %v64
}

define i1 @__any(<4 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i32> %0 to <4 x float>
  %v = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %floatmask) nounwind readnone
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<4 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i32> %0 to <4 x float>
  %v = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %floatmask) nounwind readnone
  %cmp = icmp eq i32 %v, 15
  ret i1 %cmp
}

define i1 @__none(<4 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i32> %0 to <4 x float>
  %v = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %floatmask) nounwind readnone
  %cmp = icmp eq i32 %v, 0
  ret i1 %cmp
}

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<4 x i8>) nounwind readnone alwaysinline {
  %wide8 = shufflevector <4 x i8> %0, <4 x i8> zeroinitializer,
      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4,
                  i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %wide8,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
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

define i32 @__reduce_add_int32(<4 x i32> %v) nounwind readnone {
  %v1 = shufflevector <4 x i32> %v, <4 x i32> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = add <4 x i32> %v1, %v
  %m1a = extractelement <4 x i32> %m1, i32 0
  %m1b = extractelement <4 x i32> %m1, i32 1
  %sum = add i32 %m1a, %m1b
  ret i32 %sum
}

define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone {
  reduce4(i32, @__max_varying_uint32, @__max_uniform_uint32)
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

define i64 @__reduce_add_int64(<4 x i64>) nounwind readnone {
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

define i64 @__reduce_min_int64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<4 x i64>) nounwind readnone {
  reduce4(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(4)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

define void @__masked_store_blend_i32(<4 x i32>* nocapture, <4 x i32>, 
                                      <4 x i32> %mask) nounwind alwaysinline {
  %val = load PTR_OP_ARGS(`<4 x i32> ')  %0, align 4
  %newval = call <4 x i32> @__vselect_i32(<4 x i32> %val, <4 x i32> %1, <4 x i32> %mask) 
  store <4 x i32> %newval, <4 x i32> * %0, align 4
  ret void
}

define void @__masked_store_blend_i64(<4 x i64>* nocapture %ptr, <4 x i64> %new,
                                      <4 x i32> %mask) nounwind alwaysinline {
  %oldValue = load PTR_OP_ARGS(`<4 x i64>')  %ptr, align 8

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


masked_store_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

define <4 x float> @__rcp_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %0)
  ; do one N-R iteration to improve precision
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);
  %v_iv = fmul <4 x float> %0, %call
  %two_minus = fsub <4 x float> <float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <4 x float> %call, %two_minus
  ret <4 x float> %iv_mul
}
define <4 x float> @__rcp_fast_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %ret = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %0)
  ret <4 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

define <4 x float> @__rsqrt_varying_float(<4 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  %is = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %v)
  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <4 x float> %v, %is
  %v_is_is = fmul <4 x float> %v_is, %is
  %three_sub = fsub <4 x float> <float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <4 x float> %is, %three_sub
  %half_scale = fmul <4 x float> <float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <4 x float> %half_scale
}
define <4 x float> @__rsqrt_fast_varying_float(<4 x float> %v) nounwind readonly alwaysinline {
  %ret = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %v)
  ret <4 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <4 x float> @__sqrt_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %0)
  ret <4 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; svml

include(`svml.m4')
svml(ISA)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <4 x float> @__max_varying_float(<4 x float>, <4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %call
}

define <4 x float> @__min_varying_float(<4 x float>, <4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %0, <4 x float> %1)
  ret <4 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <4 x double> @__sqrt_varying_double(<4 x double>) nounwind alwaysinline {
  unary2to4(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <4 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @__min_varying_double(<4 x double>, <4 x double>) nounwind readnone {
  binary2to4(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <4 x double> %ret
}

define <4 x double> @__max_varying_double(<4 x double>, <4 x double>) nounwind readnone {
  binary2to4(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <4 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

masked_store_blend_8_16_by_4()

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts


masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

; define these with the macros from stdlib.m4

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(half)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp/rsqrt declarations for half
rcph_rsqrth_decl

transcendetals_decl()
trigonometry_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
dot_product_vnni_decl()
