;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause


;; This file defines the target for "double-pumped" SSE2, i.e. running
;; with 8-wide vectors

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standard 8-wide definitions from m4 macros

define(`WIDTH',`8')
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
;; rcp

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

define <8 x float> @__rcp_varying_float(<8 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  unary4to8(call, float, @llvm.x86.sse.rcp.ps, %0)
  ; do one N-R iteration
  %v_iv = fmul <8 x float> %0, %call
  %two_minus = fsub <8 x float> <float 2., float 2., float 2., float 2.,
                                 float 2., float 2., float 2., float 2.>, %v_iv  
  %iv_mul = fmul <8 x float> %call, %two_minus
  ret <8 x float> %iv_mul
}

define <8 x float> @__rcp_fast_varying_float(<8 x float>) nounwind readonly alwaysinline {
    unary4to8(ret, float, @llvm.x86.sse.rcp.ps, %0)
    ret <8 x float> %ret
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @__rsqrt_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  ;  float is = __rsqrt_v(v);
  unary4to8(is, float, @llvm.x86.sse.rsqrt.ps, %v)
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <8 x float> %v, %is
  %v_is_is = fmul <8 x float> %v_is, %is
  %three_sub = fsub <8 x float> <float 3., float 3., float 3., float 3.,
                                 float 3., float 3., float 3., float 3.>, %v_is_is
  %is_mul = fmul <8 x float> %is, %three_sub
  %half_scale = fmul <8 x float> <float 0.5, float 0.5, float 0.5, float 0.5,
                                  float 0.5, float 0.5, float 0.5, float 0.5>, %is_mul
  ret <8 x float> %half_scale
}


define <8 x float> @__rsqrt_fast_varying_float(<8 x float> %v) nounwind readonly alwaysinline {
  unary4to8(is, float, @llvm.x86.sse.rsqrt.ps, %v)
  ret <8 x float> %is
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @__sqrt_varying_float(<8 x float>) nounwind readonly alwaysinline {
  unary4to8(call, float, @llvm.x86.sse.sqrt.ps, %0)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; svml

include(`svml.m4')
svml(ISA)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @__max_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.max.ps, %0, %1)
  ret <8 x float> %call
}

define <8 x float> @__min_varying_float(<8 x float>, <8 x float>) nounwind readonly alwaysinline {
  binary4to8(call, float, @llvm.x86.sse.min.ps, %0, %1)
  ret <8 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

; There is no blend instruction with SSE2, so we simulate it with bit
; operations on i32s.  For these two vselect functions, for each
; vector element, if the mask is on, we return the corresponding value
; from %1, and otherwise return the value from %0.

define <8 x i32> @__vselect_i32(<8 x i32>, <8 x i32> ,
                                         <8 x i32> %mask) nounwind readnone alwaysinline {
  %notmask = xor <8 x i32> %mask, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %cleared_old = and <8 x i32> %0, %notmask
  %masked_new = and <8 x i32> %1, %mask
  %new = or <8 x i32> %cleared_old, %masked_new
  ret <8 x i32> %new
}

define <8 x float> @__vselect_float(<8 x float>, <8 x float>,
                                             <8 x i32> %mask) nounwind readnone alwaysinline {
  %v0 = bitcast <8 x float> %0 to <8 x i32>
  %v1 = bitcast <8 x float> %1 to <8 x i32>
  %r = call <8 x i32> @__vselect_i32(<8 x i32> %v0, <8 x i32> %v1, <8 x i32> %mask)
  %rf = bitcast <8 x i32> %r to <8 x float>
  ret <8 x float> %rf
}


; To do vector integer min and max, we do the vector compare and then sign
; extend the i1 vector result to an i32 mask.  The __vselect does the
; rest...

define <8 x i32> @__min_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %c = icmp slt <8 x i32> %0, %1
  %mask = sext <8 x i1> %c to <8 x i32>
  %v = call <8 x i32> @__vselect_i32(<8 x i32> %1, <8 x i32> %0, <8 x i32> %mask)
  ret <8 x i32> %v
}

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp slt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define <8 x i32> @__max_varying_int32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %c = icmp sgt <8 x i32> %0, %1
  %mask = sext <8 x i1> %c to <8 x i32>
  %v = call <8 x i32> @__vselect_i32(<8 x i32> %1, <8 x i32> %0, <8 x i32> %mask)
  ret <8 x i32> %v
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp sgt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

; The functions for unsigned ints are similar, just with unsigned
; comparison functions...

define <8 x i32> @__min_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %c = icmp ult <8 x i32> %0, %1
  %mask = sext <8 x i1> %c to <8 x i32>
  %v = call <8 x i32> @__vselect_i32(<8 x i32> %1, <8 x i32> %0, <8 x i32> %mask)
  ret <8 x i32> %v
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ult i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

define <8 x i32> @__max_varying_uint32(<8 x i32>, <8 x i32>) nounwind readonly alwaysinline {
  %c = icmp ugt <8 x i32> %0, %1
  %mask = sext <8 x i1> %c to <8 x i32>
  %v = call <8 x i32> @__vselect_i32(<8 x i32> %1, <8 x i32> %0, <8 x i32> %mask)
  ret <8 x i32> %v
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ugt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops / reductions

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

define i64 @__movmsk(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %v64 = zext i32 %v to i64
  ret i64 %v64
}

define i1 @__any(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp eq i32 %v, 255
  ret i1 %cmp
}

define i1 @__none(<8 x i32>) nounwind readnone alwaysinline {
  ; first do two 4-wide movmsk calls
  %floatmask = bitcast <8 x i32> %0 to <8 x float>
  %m0 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v0 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m0) nounwind readnone
  %m1 = shufflevector <8 x float> %floatmask, <8 x float> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v1 = call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %m1) nounwind readnone

  ; and shift the first one over by 4 before ORing it with the value 
  ; of the second one
  %v1s = shl i32 %v1, 4
  %v = or i32 %v0, %v1s
  %cmp = icmp eq i32 %v, 0
  ret i1 %cmp
}

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<8 x i8>) nounwind readnone alwaysinline {
  %wide8 = shufflevector <8 x i8> %0, <8 x i8> zeroinitializer,
      <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                  i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %wide8,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

define internal <8 x i16> @__add_varying_i16(<8 x i16>,
                                  <8 x i16>) nounwind readnone alwaysinline {
  %r = add <8 x i16> %0, %1
  ret <8 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<8 x i16>) nounwind readnone alwaysinline {
  reduce8(i16, @__add_varying_i16, @__add_uniform_i16)
}

define <4 x float> @__vec4_add_float(<4 x float> %v0,
                                     <4 x float> %v1) nounwind readnone alwaysinline {
  %v = fadd <4 x float> %v0, %v1
  ret <4 x float> %v
}

define float @__add_float(float, float) nounwind readnone alwaysinline {
  %v = fadd float %0, %1
  ret float %v
}

define float @__reduce_add_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8by4(float, @__vec4_add_float, @__add_float)
}

define float @__reduce_min_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<8 x float>) nounwind readnone alwaysinline {
  reduce8(float, @__max_varying_float, @__max_uniform_float)
}

; helper function for reduce_add_int32
define <4 x i32> @__vec4_add_int32(<4 x i32> %v0,
                                   <4 x i32> %v1) nounwind readnone alwaysinline {
  %v = add <4 x i32> %v0, %v1
  ret <4 x i32> %v
}

; helper function for reduce_add_int32
define i32 @__add_int32(i32, i32) nounwind readnone alwaysinline {
  %v = add i32 %0, %1
  ret i32 %v
}

define i32 @__reduce_add_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8by4(i32, @__vec4_add_int32, @__add_int32)
}

define i32 @__reduce_min_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__min_varying_int32, @__min_uniform_int32)
}

define i32 @__reduce_max_int32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<8 x i32>) nounwind readnone alwaysinline {
  reduce8(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

define <4 x double> @__add_varying_double(<4 x double>,
                                     <4 x double>) nounwind readnone alwaysinline {
  %r = fadd <4 x double> %0, %1
  ret <4 x double> %r
}

define double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %r = fadd double %0, %1
  ret double %r
}

define double @__reduce_add_double(<8 x double>) nounwind readnone {
  reduce8by4(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<8 x double>) nounwind readnone {
  reduce8(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<8 x double>) nounwind readnone {
  reduce8(double, @__max_varying_double, @__max_uniform_double)
}

define <4 x i64> @__add_varying_int64(<4 x i64>,
                                      <4 x i64>) nounwind readnone alwaysinline {
  %r = add <4 x i64> %0, %1
  ret <4 x i64> %r
}

define i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %r = add i64 %0, %1
  ret i64 %r
}

define i64 @__reduce_add_int64(<8 x i64>) nounwind readnone {
  reduce8by4(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<8 x i64>) nounwind readnone {
  reduce8(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

reduce_equal(8)

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
;; float rounding

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

define <8 x float> @__round_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <8 x float> %0 to <8 x i32>
  %bitop.i.i = and <8 x i32> %float_to_int_bitcast.i.i.i.i, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %bitop.i = xor <8 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <8 x i32> %bitop.i to <8 x float>
  %binop.i = fadd <8 x float> %int_to_float_bitcast.i.i40.i, <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06>
  %binop21.i = fadd <8 x float> %binop.i, <float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06>
  %float_to_int_bitcast.i.i.i = bitcast <8 x float> %binop21.i to <8 x i32>
  %bitop31.i = xor <8 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <8 x i32> %bitop31.i to <8 x float>
  ret <8 x float> %int_to_float_bitcast.i.i.i
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

define <8 x float> @__floor_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <8 x float> @__round_varying_float(<8 x float> %0) nounwind
  %bincmp.i = fcmp ogt <8 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <8 x i1> %bincmp.i to <8 x i32>
  %bitop.i = and <8 x i32> %val_to_boolvec32.i, <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <8 x i32> %bitop.i to <8 x float>
  %binop.i = fadd <8 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <8 x float> %binop.i
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

define <8 x float> @__ceil_varying_float(<8 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <8 x float> @__round_varying_float(<8 x float> %0) nounwind
  %bincmp.i = fcmp olt <8 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <8 x i1> %bincmp.i to <8 x i32>
  %bitop.i = and <8 x i32> %val_to_boolvec32.i, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <8 x i32> %bitop.i to <8 x float>
  %binop.i = fadd <8 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <8 x float> %binop.i
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

define <8 x double> @__round_varying_double(<8 x double>) nounwind readonly alwaysinline {
  unary1to8(double, @round)
}

define <8 x double> @__floor_varying_double(<8 x double>) nounwind readonly alwaysinline {
  unary1to8(double, @floor)
}

define <8 x double> @__ceil_varying_double(<8 x double>) nounwind readonly alwaysinline {
  unary1to8(double, @ceil)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)

masked_store_blend_8_16_by_8()

define void @__masked_store_blend_i32(<8 x i32>* nocapture, <8 x i32>, 
                                      <8 x i32> %mask) nounwind alwaysinline {
  %val = load PTR_OP_ARGS(`<8 x i32> ')  %0, align 4
  %newval = call <8 x i32> @__vselect_i32(<8 x i32> %val, <8 x i32> %1, <8 x i32> %mask) 
  store <8 x i32> %newval, <8 x i32> * %0, align 4
  ret void
}

define void @__masked_store_blend_i64(<8 x i64>* nocapture %ptr, <8 x i64> %new,
                                      <8 x i32> %mask) nounwind alwaysinline {
  %oldValue = load PTR_OP_ARGS(`<8 x i64>')  %ptr, align 8

  ; Do 8x64-bit blends by doing two <8 x i32> blends, where the <8 x i32> values
  ; are actually bitcast <2 x i64> values
  ;
  ; set up the first two 64-bit values
  %old0123  = shufflevector <8 x i64> %oldValue, <8 x i64> undef,
                            <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %old0123f = bitcast <4 x i64> %old0123 to <8 x float>
  %new0123  = shufflevector <8 x i64> %new, <8 x i64> undef,
                            <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %new0123f = bitcast <4 x i64> %new0123 to <8 x float>
  ; compute mask--note that the indices are doubled-up
  %mask0123 = shufflevector <8 x i32> %mask, <8 x i32> undef,
              <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ; and blend the first 4 values
  %result0123f = call <8 x float> @__vselect_float(<8 x float> %old0123f, <8 x float> %new0123f,
                                                   <8 x i32> %mask0123)
  %result0123 = bitcast <8 x float> %result0123f to <4 x i64>

  ; and again
  %old4567  = shufflevector <8 x i64> %oldValue, <8 x i64> undef,
                            <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %old4567f = bitcast <4 x i64> %old4567 to <8 x float>
  %new4567  = shufflevector <8 x i64> %new, <8 x i64> undef,
                            <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %new4567f = bitcast <4 x i64> %new4567 to <8 x float>
  ; compute mask--note that the values are doubled-up
  %mask4567 = shufflevector <8 x i32> %mask, <8 x i32> undef,
              <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  ; and blend the two of the values
  %result4567f = call <8 x float> @__vselect_float(<8 x float> %old4567f, <8 x float> %new4567f,
                                                   <8 x i32> %mask4567)
  %result4567 = bitcast <8 x float> %result4567f to <4 x i64>

  ; reconstruct the final <8 x i64> vector
  %final = shufflevector <4 x i64> %result0123, <4 x i64> %result4567,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i64> %final, <8 x i64> * %ptr, align 8
  ret void
}

masked_store_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <8 x double> @__sqrt_varying_double(<8 x double>) nounwind alwaysinline {
  unary2to8(ret, double, @llvm.x86.sse2.sqrt.pd, %0)
  ret <8 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision float min/max

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <8 x double> @__min_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.min.pd, %0, %1)
  ret <8 x double> %ret
}

define <8 x double> @__max_varying_double(<8 x double>, <8 x double>) nounwind readnone alwaysinline {
  binary2to8(ret, double, @llvm.x86.sse2.max.pd, %0, %1)
  ret <8 x double> %ret
}

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
