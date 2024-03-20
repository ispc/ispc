;;  Copyright (c) 2013-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Basic 4-wide definitions

define(`WIDTH',`4')
define(`MASK',`i64')
include(`util.m4')

stdlib_core()
packed_load_and_store(FALSE)
scans()
int64minmax()
saturation_arithmetic()

include(`target-avx-utils.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

;; sse intrinsic
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

define <4 x float> @__rcp_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ;  float iv = __rcp_v(v);
  ;  return iv * (2. - v * iv);

  %call = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %0)
  ; do one N-R iteration
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
;; rounding floats

;; sse intrinsic
declare <4 x float> @llvm.x86.sse41.round.ps(<4 x float>, i32) nounwind readnone

define <4 x float> @__round_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round mode nearest 0b00 | don't signal precision exceptions 0b1000 = 8
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 8)
  ret <4 x float> %call
}

define <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 9)
  ret <4 x float> %call
}

define <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %call = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %0, i32 10)
  ret <4 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

;; avx intrinsic
declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) nounwind readnone

define <4 x double> @__round_varying_double(<4 x double>) nounwind readonly alwaysinline {
  %call = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %0, i32 8)
  ret <4 x double> %call
}

define <4 x double> @__floor_varying_double(<4 x double>) nounwind readonly alwaysinline {
  ; roundpd, round down 0b01 | don't signal precision exceptions 0b1000 = 9
  %call = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %0, i32 9)
  ret <4 x double> %call
}


define <4 x double> @__ceil_varying_double(<4 x double>) nounwind readonly alwaysinline {
  ; roundpd, round up 0b10 | don't signal precision exceptions 0b1000 = 10
  %call = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %0, i32 10)
  ret <4 x double> %call
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

;; sse intrinsic
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
;; trunc() float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

;; sse intrinsic
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <4 x float> @__sqrt_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %call = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %0)
  ret <4 x float> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

;; avx intrinsic
declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define <4 x double> @__sqrt_varying_double(<4 x double>) nounwind alwaysinline {
  %call = call <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double> %0)
  ret <4 x double> %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

;; sse intrinsics
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
; horizontal ops

;; sse intrinsic 
declare i32 @llvm.x86.avx.movmsk.pd.256(<4 x double>) nounwind readnone

define i64 @__movmsk(<4 x i64>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i64> %0 to <4 x double>
  %v = call i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %floatmask) nounwind readnone
  %v64 = zext i32 %v to i64
  ret i64 %v64
}

define i1 @__any(<4 x i64>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i64> %0 to <4 x double>
  %v = call i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %floatmask) nounwind readnone
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<4 x i64>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i64> %0 to <4 x double>
  %v = call i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %floatmask) nounwind readnone
  %cmp = icmp eq i32 %v, 15
  ret i1 %cmp
}

define i1 @__none(<4 x i64>) nounwind readnone alwaysinline {
  %floatmask = bitcast <4 x i64> %0 to <4 x double>
  %v = call i32 @llvm.x86.avx.movmsk.pd.256(<4 x double> %floatmask) nounwind readnone
  %cmp = icmp eq i32 %v, 0
  ret i1 %cmp
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

;; sse intrinsic
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define float @__reduce_add_float(<4 x float>) nounwind readonly alwaysinline {
  %v1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %0, <4 x float> %0)
  %v2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %v1, <4 x float> %v1)
  %scalar = extractelement <4 x float> %v2, i32 0
  ret float %scalar
}

define float @__reduce_min_float(<4 x float>) nounwind readnone {
  reduce4(float, @__min_varying_float, @__min_uniform_float)
}

define float @__reduce_max_float(<4 x float>) nounwind readnone {
  reduce4(float, @__max_varying_float, @__max_uniform_float)
}

reduce_equal(4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int8 ops

declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<4 x i8>) nounwind readnone alwaysinline 
{
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int16 ops

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define <4 x i32> @__add_varying_int32(<4 x i32>,
                                      <4 x i32>) nounwind readnone alwaysinline {
  %s = add <4 x i32> %0, %1
  ret <4 x i32> %s
}

define i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__add_varying_int32, @__add_uniform_int32)
}


define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__min_varying_int32, @__min_uniform_int32)
}


define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone alwaysinline {
  reduce4(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define double @__reduce_add_double(<4 x double>) nounwind readonly alwaysinline {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v1 = shufflevector <4 x double> <double 0.,double 0.,double 0.,double 0.>, <4 x double> undef,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
;;  %v1 = <4 x double> <double 0., double 0., double 0., double 0.>
  %sum0 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %v0,   <4 x double> %v1)
  %sum1 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %sum0, <4 x double> %sum0)
  %final0 = extractelement <4 x double> %sum1, i32 0
  %final1 = extractelement <4 x double> %sum1, i32 2
  %sum = fadd double %final0, %final1

  ret double %sum
}

define double @__reduce_min_double(<4 x double>) nounwind readnone alwaysinline {
  reduce4(double, @__min_varying_double, @__min_uniform_double)
}


define double @__reduce_max_double(<4 x double>) nounwind readnone alwaysinline {
  reduce4(double, @__max_varying_double, @__max_uniform_double)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define <4 x i64> @__add_varying_int64(<4 x i64>,
                                      <4 x i64>) nounwind readnone alwaysinline {
  %s = add <4 x i64> %0, %1
  ret <4 x i64> %s
}

define i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__add_varying_int64, @__add_uniform_int64)
}


define i64 @__reduce_min_int64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__min_varying_int64, @__min_uniform_int64)
}


define i64 @__reduce_max_int64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__max_varying_int64, @__max_uniform_int64)
}


define i64 @__reduce_min_uint64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__min_varying_uint64, @__min_uniform_uint64)
}


define i64 @__reduce_max_uint64(<4 x i64>) nounwind readnone alwaysinline {
  reduce4(i64, @__max_varying_uint64, @__max_uniform_uint64)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts


; no masked load instruction for i8 and i16 types??
masked_load(i8,  1)
masked_load(i16, 2)

;; avx intrinsics
declare <4 x float> @llvm.x86.avx.maskload.ps(i8 *, <4 x MfORi32> %mask)
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8 *, <4 x MdORi64> %mask)
 
define <4 x i32> @__masked_load_i32(i8 *, <4 x i64> %mask64) nounwind alwaysinline {
  %mask      = trunc <4 x i64> %mask64 to <4 x i32>
  %floatmask = bitcast <4 x i32> %mask to <4 x MfORi32>
  %floatval = call <4 x float> @llvm.x86.avx.maskload.ps(i8 * %0, <4 x MfORi32> %floatmask)
  %retval = bitcast <4 x float> %floatval to <4 x i32>
  ret <4 x i32> %retval
}


define <4 x i64> @__masked_load_i64(i8 *, <4 x i64> %mask) nounwind alwaysinline {
  %doublemask = bitcast <4 x i64> %mask to <4 x MdORi64>
  %doubleval  = call <4 x double> @llvm.x86.avx.maskload.pd.256(i8 * %0, <4 x MdORi64> %doublemask)
  %retval = bitcast <4 x double> %doubleval to <4 x i64>
  ret <4 x i64> %retval
}

masked_load_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

gen_masked_store(i8)
gen_masked_store(i16)

; note that mask is the 2nd parameter, not the 3rd one!!
;; avx intrinsics
declare void @llvm.x86.avx.maskstore.ps    (i8 *, <4 x MfORi32>,  <4 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8 *, <4 x MdORi64>, <4 x double>)

define void @__masked_store_i32(<4 x i32>* nocapture, <4 x i32>, 
                                <4 x i64>) nounwind alwaysinline {
  %mask32 = trunc <4 x i64> %2 to <4 x i32>

  %ptr    = bitcast <4 x i32> * %0 to i8 *
  %val    = bitcast <4 x i32> %1 to <4 x float>
  %mask   = bitcast <4 x i32> %mask32 to <4 x MfORi32>
  call void @llvm.x86.avx.maskstore.ps(i8 * %ptr, <4 x MfORi32> %mask, <4 x float> %val)
  ret void
}

define void @__masked_store_i64(<4 x i64>* nocapture, <4 x i64>,
                                <4 x i64>) nounwind alwaysinline {
  %ptr  = bitcast <4 x i64> * %0 to i8 *
  %val  = bitcast <4 x i64> %1 to <4 x double>
  %mask = bitcast <4 x i64> %2 to <4 x MdORi64>
  call void @llvm.x86.avx.maskstore.pd.256(i8 * %ptr, <4 x MdORi64> %mask, <4 x double> %val)
  ret void
}


masked_store_blend_8_16_by_4_mask64()

;; sse intrinsic
declare <4 x float>  @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>,
                                             <4 x float>) nounwind readnone

define void @__masked_store_blend_i32(<4 x i32>* nocapture, <4 x i32>, 
                                      <4 x i64>) nounwind alwaysinline {
  %mask          = trunc   <4 x i64> %2 to <4 x i32>
  %mask_as_float = bitcast <4 x i32> %mask to <4 x float>
  %oldValue      = load PTR_OP_ARGS(`   <4 x i32>')  %0, align 4
  %oldAsFloat    = bitcast <4 x i32> %oldValue to <4 x float>
  %newAsFloat    = bitcast <4 x i32> %1 to <4 x float>
  %blend         = call    <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %oldAsFloat,
                                                             <4 x float> %newAsFloat,
                                                             <4 x float> %mask_as_float)
  %blendAsInt = bitcast <4 x float> %blend to <4 x i32>
  store <4 x i32> %blendAsInt, <4 x i32>* %0, align 4
  ret void
}

;; avx intrinsic
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>,
                                                <4 x double>) nounwind readnone

define void @__masked_store_blend_i64(<4 x i64>* nocapture , <4 x i64>,
                                      <4 x i64>) nounwind alwaysinline {
  %mask_as_double = bitcast <4 x i64>  %2 to <4 x double>
  %oldValue       = load PTR_OP_ARGS(`   <4 x i64>')  %0, align 4
  %oldAsDouble    = bitcast <4 x i64>  %oldValue to <4 x double>
  %newAsDouble    = bitcast <4 x i64>  %1 to <4 x double>
  %blend          = call    <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %oldAsDouble,
                                                                        <4 x double> %newAsDouble,
                                                                        <4 x double> %mask_as_double)
  %blendAsInt = bitcast <4 x double> %blend to <4 x i64>
  store <4 x i64> %blendAsInt, <4 x i64>* %0, align 4
  ret void
}

masked_store_float_double()

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
;; double precision min/max

declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x double> @__min_varying_double(<4 x double>, <4 x double>) nounwind readnone alwaysinline {
  %call = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %0, <4 x double> %1)
  ret <4 x double> %call
}

define <4 x double> @__max_varying_double(<4 x double>, <4 x double>) nounwind readnone alwaysinline {
  %call = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %0, <4 x double> %1)
  ret <4 x double> %call
}

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
