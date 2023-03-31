;;  Copyright (c) 2010-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Basic 16-wide definitions

define(`WIDTH',`16')
define(`MASK',`i32')
include(`util.m4')

stdlib_core()
packed_load_and_store(FALSE)
scans()
int64minmax()
saturation_arithmetic()

include(`target-avx-utils.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
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

define <16 x float> @__rcp_fast_varying_float(<16 x float>) nounwind readonly alwaysinline {
  unary8to16(ret, float, @llvm.x86.avx.rcp.ps.256, %0)
  ret <16 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) nounwind readnone

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  round8to16(%0, 8)
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  round8to16(%0, 9)
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  round8to16(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone

define <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 8)
}

define <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 9)
}

define <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline {
  round4to16double(%0, 10)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float/double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <8 x float> @llvm.x86.avx.rsqrt.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
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

define <16 x float> @__rsqrt_fast_varying_float(<16 x float> %v) nounwind readonly alwaysinline {
  unary8to16(ret, float, @llvm.x86.avx.rsqrt.ps.256, %v)
  ret <16 x float> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  unary8to16(call, float, @llvm.x86.avx.sqrt.ps.256, %0)
  ret <16 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <16 x float> @__max_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.max.ps.256, %0, %1)
  ret <16 x float> %call
}

define <16 x float> @__min_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  binary8to16(call, float, @llvm.x86.avx.min.ps.256, %0, %1)
  ret <16 x float> %call
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone

define i64 @__movmsk(<16 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <16 x i32> %0 to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) nounwind readnone
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask1) nounwind readnone

  %v1shift = shl i32 %v1, 8
  %v = or i32 %v1shift, %v0
  %v64 = zext i32 %v to i64
  ret i64 %v64
}

define i1 @__any(<16 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <16 x i32> %0 to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) nounwind readnone
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask1) nounwind readnone

  %v1shift = shl i32 %v1, 8
  %v = or i32 %v1shift, %v0
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<16 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <16 x i32> %0 to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) nounwind readnone
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask1) nounwind readnone

  %v1shift = shl i32 %v1, 8
  %v = or i32 %v1shift, %v0
  %cmp = icmp eq i32 %v, 65535
  ret i1 %cmp
}

define i1 @__none(<16 x i32>) nounwind readnone alwaysinline {
  %floatmask = bitcast <16 x i32> %0 to <16 x float>
  %mask0 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v0 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask0) nounwind readnone
  %mask1 = shufflevector <16 x float> %floatmask, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %mask1) nounwind readnone

  %v1shift = shl i32 %v1, 8
  %v = or i32 %v1shift, %v0
  %cmp = icmp eq i32 %v, 0
  ret i1 %cmp
}



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline {
  %va = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vb = shufflevector <16 x float> %0, <16 x float> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v1 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %va, <8 x float> %vb)
  %v2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v1, <8 x float> %v1)
  %v3 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %v2, <8 x float> %v2)
  %scalar1 = extractelement <8 x float> %v3, i32 0
  %scalar2 = extractelement <8 x float> %v3, i32 4
  %sum = fadd float %scalar1, %scalar2
  ret float %sum
}


define float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__min_varying_float, @__min_uniform_float)
}


define float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline {
  reduce16(float, @__max_varying_float, @__max_uniform_float)
}

reduce_equal(16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<16 x i8>) nounwind readnone alwaysinline {
  %rv = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %0,
                                              <16 x i8> zeroinitializer)
  %r0 = extractelement <2 x i64> %rv, i32 0
  %r1 = extractelement <2 x i64> %rv, i32 1
  %r = add i64 %r0, %r1
  %r16 = trunc i64 %r to i16
  ret i16 %r16
}

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

define <16 x i32> @__add_varying_int32(<16 x i32>,
                                       <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}


define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}


define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint32 ops

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}


define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline {
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
  %final0 = extractelement <4 x double> %sum1, i32 0
  %final1 = extractelement <4 x double> %sum1, i32 2
  %sum = fadd double %final0, %final1
  ret double %sum
}

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}


define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define <16 x i64> @__add_varying_int64(<16 x i64>,
                                                <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}


define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}


define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; horizontal uint64 ops

define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}


define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

; no masked load instruction for i8 and i16 types??
masked_load(i8,  1)
masked_load(i16, 2)

declare <8 x float> @llvm.x86.avx.maskload.ps.256(i8 *, <8 x MfORi32> %mask)
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8 *, <4 x MdORi64> %mask)
 
define <16 x i32> @__masked_load_i32(i8 *, <16 x i32> %mask) nounwind alwaysinline {
  %floatmask = bitcast <16 x i32> %mask to <16 x MfORi32>
  %mask0 = shufflevector <16 x MfORi32> %floatmask, <16 x MfORi32> undef,
     <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val0 = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8 * %0, <8 x MfORi32> %mask0)
  %mask1 = shufflevector <16 x MfORi32> %floatmask, <16 x MfORi32> undef,
     <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %ptr1 = getelementptr PTR_OP_ARGS(`i8') %0, i32 32    ;; 8x4 bytes = 32
  %val1 = call <8 x float> @llvm.x86.avx.maskload.ps.256(i8 * %ptr1, <8 x MfORi32> %mask1)

  %retval = shufflevector <8 x float> %val0, <8 x float> %val1,
     <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %reti32 = bitcast <16 x float> %retval to <16 x i32>
  ret <16 x i32> %reti32
}


define <16 x i64> @__masked_load_i64(i8 *, <16 x i32> %mask) nounwind alwaysinline {
  ; double up masks, bitcast to doubles
  %mask0 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask2 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11>
  %mask3 = shufflevector <16 x i32> %mask, <16 x i32> undef,
     <8 x i32> <i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
  %mask0d = bitcast <8 x i32> %mask0 to <4 x MdORi64>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x MdORi64>
  %mask2d = bitcast <8 x i32> %mask2 to <4 x MdORi64>
  %mask3d = bitcast <8 x i32> %mask3 to <4 x MdORi64>

  %val0d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %0, <4 x MdORi64> %mask0d)
  %ptr1 = getelementptr PTR_OP_ARGS(`i8') %0, i32 32
  %val1d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr1, <4 x MdORi64> %mask1d)
  %ptr2 = getelementptr PTR_OP_ARGS(`i8') %0, i32 64
  %val2d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr2, <4 x MdORi64> %mask2d)
  %ptr3 = getelementptr PTR_OP_ARGS(`i8') %0, i32 96
  %val3d = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %ptr3, <4 x MdORi64> %mask3d)

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

masked_load_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

; FIXME: there is no AVX instruction for these, but we could be clever
; by packing the bits down and setting the last 3/4 or half, respectively,
; of the mask to zero...  Not sure if this would be a win in the end
gen_masked_store(i8)
gen_masked_store(i16)

; note that mask is the 2nd parameter, not the 3rd one!!
declare void @llvm.x86.avx.maskstore.ps.256(i8 *, <8 x MfORi32>, <8 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8 *, <4 x MdORi64>, <4 x double>)

define void @__masked_store_i32(<16 x i32>* nocapture, <16 x i32>, 
                                <16 x i32>) nounwind alwaysinline {
  %ptr = bitcast <16 x i32> * %0 to i8 *
  %val = bitcast <16 x i32> %1 to <16 x float>
  %mask = bitcast <16 x i32> %2 to <16 x MfORi32>

  %val0 = shufflevector <16 x float> %val, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %val1 = shufflevector <16 x float> %val, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %mask0 = shufflevector <16 x MfORi32> %mask, <16 x MfORi32> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mask1 = shufflevector <16 x MfORi32> %mask, <16 x MfORi32> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx.maskstore.ps.256(i8 * %ptr, <8 x MfORi32> %mask0, <8 x float> %val0)
  %ptr1 = getelementptr PTR_OP_ARGS(`i8') %ptr, i32 32
  call void @llvm.x86.avx.maskstore.ps.256(i8 * %ptr1, <8 x MfORi32> %mask1, <8 x float> %val1)

  ret void
}

define void @__masked_store_i64(<16 x i64>* nocapture, <16 x i64>,
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
  %mask0d = bitcast <8 x i32> %mask0 to <4 x MdORi64>
  %mask1d = bitcast <8 x i32> %mask1 to <4 x MdORi64>
  %mask2d = bitcast <8 x i32> %mask2 to <4 x MdORi64>
  %mask3d = bitcast <8 x i32> %mask3 to <4 x MdORi64>

  %val0 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val1 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %val2 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %val3 = shufflevector <16 x double> %val, <16 x double> undef,
     <4 x i32> <i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr, <4 x MdORi64> %mask0d, <4 x double> %val0)
  %ptr1 = getelementptr PTR_OP_ARGS(`i8') %ptr, i32 32
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr1, <4 x MdORi64> %mask1d, <4 x double> %val1)
  %ptr2 = getelementptr PTR_OP_ARGS(`i8') %ptr, i32 64
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr2, <4 x MdORi64> %mask2d, <4 x double> %val2)
  %ptr3 = getelementptr PTR_OP_ARGS(`i8') %ptr, i32 96
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr3, <4 x MdORi64> %mask3d, <4 x double> %val3)

  ret void
}

masked_store_float_double()

masked_store_blend_8_16_by_16()

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>,
                                                <8 x float>) nounwind readnone

define void @__masked_store_blend_i32(<16 x i32>* nocapture, <16 x i32>, 
                                      <16 x i32>) nounwind alwaysinline {
  %maskAsFloat = bitcast <16 x i32> %2 to <16 x float>
  %oldValue = load PTR_OP_ARGS(`<16 x i32>')  %0, align 4
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

define void @__masked_store_blend_i64(<16 x i64>* nocapture %ptr, <16 x i64> %newi64, 
                                      <16 x i32> %mask) nounwind alwaysinline {
  %oldValue = load PTR_OP_ARGS(`<16 x i64>')  %ptr, align 8
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
;; scatter

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  unary4to16(ret, double, @llvm.x86.avx.sqrt.pd.256, %0)
  ret <16 x double> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.min.pd.256, %0, %1)
  ret <16 x double> %ret
}

define <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline {
  binary4to16(ret, double, @llvm.x86.avx.max.pd.256, %0, %1)
  ret <16 x double> %ret
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
