;;  Copyright (c) 2015-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_definition()
popcnt()
ctlztz()
halfTypeGenericImplementation()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

define_shuffles()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

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
;; fast math mode
fastMathFTZDAZ_x86()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; round/floor/ceil

declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32) nounwind readnone

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
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

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 9)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <4 x float> undef, float %0, i32 0
  ; roundps, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %xi, <4 x float> %xi, i32 10)
  %rs = extractelement <4 x float> %xr, i32 0
  ret float %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <2 x double> @llvm.x86.sse41.round.sd(<2 x double>, <2 x double>, i32) nounwind readnone

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %xi = insertelement <2 x double> undef, double %0, i32 0
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 8)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round down 0b01 | don't signal precision exceptions 0b1001 = 9
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 9)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  ; see above for round_ss instrinsic discussion...
  %xi = insertelement <2 x double> undef, double %0, i32 0
  ; roundsd, round up 0b10 | don't signal precision exceptions 0b1010 = 10
  %xr = call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %xi, <2 x double> %xi, i32 10)
  %rs = extractelement <2 x double> %xr, i32 0
  ret double %rs
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64/uint64 min/max
define i64 @__max_uniform_int64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp sgt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ugt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp slt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ult i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

define float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %1, float %0
  ret float %ret
}

define float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %0, float %1
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %0, double %1
  ret double %ret
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %1, double %0
  ret double %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

define(`rsqrt14_uniform', `
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone
define float @__rsqrt_fast_uniform_float(float) nounwind readonly alwaysinline {
  %v = insertelement <4 x float> undef, float %0, i32 0
  %vis = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %v, <4 x float> %v, <4 x float> undef, i8 -1)
  %is = extractelement <4 x float> %vis, i32 0
  ret float %is
}

define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %is = call float @__rsqrt_fast_uniform_float(float %0)

  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul float %0, %is
  %v_is_is = fmul float %v_is, %is
  %three_sub = fsub float 3., %v_is_is
  %is_mul = fmul float %is, %three_sub
  %half_scale = fmul float 0.5, %is_mul
  ret float %half_scale
}

declare <2 x double> @llvm.x86.avx512.rsqrt14.sd(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone
define double @__rsqrt_fast_uniform_double(double) nounwind readonly alwaysinline {
  %v = insertelement <2 x double> undef, double %0, i32 0
  %vis = call <2 x double> @llvm.x86.avx512.rsqrt14.sd(<2 x double> %v, <2 x double> %v, <2 x double> undef, i8 -1)
  %is = extractelement <2 x double> %vis, i32 0
  ret double %is
}

declare i8 @llvm.x86.avx512.mask.fpclass.sd(<2 x double>, i32, i8)
define double @__rsqrt_uniform_double(double %v) nounwind readonly alwaysinline {
  ; detect +/-0 and +inf to deal with them differently.
  %vec = insertelement <2 x double> undef, double %v, i32 0
  %corner_cases_i8 = call i8 @llvm.x86.avx512.mask.fpclass.sd(<2 x double> %vec, i32 14, i8 -1)
  %corner_cases = icmp ne i8 %corner_cases_i8, 0
  %is = call double @__rsqrt_fast_uniform_double(double %v)

  ; Precision refinement sequence based on minimax approximation.
  ; This sequence is a little slower than Newton-Raphson, but has much better precision
  ; Relative error is around 3 ULPs.
  ; t1 = 1.0 - (v * is) * is
  ; t2 = 0.37500000407453632 + t1 * 0.31250000550062401
  ; t3 = 0.5 + t1 * t2
  ; t4 = is + (t1*is) * t3
  %v_is = fmul double %v,  %is
  %v_is_is = fmul double %v_is,  %is
  %t1 = fsub double 1., %v_is_is
  %t1_03125 = fmul double 0.31250000550062401, %t1
  %t2 = fadd double 0.37500000407453632, %t1_03125
  %t1_t2 = fmul double %t1, %t2
  %t3 = fadd double 0.5, %t1_t2
  %t1_is = fmul double %t1, %is
  %t1_is_t3 = fmul double %t1_is, %t3
  %t4 = fadd double %is, %t1_is_t3
  %ret = select i1 %corner_cases, double %is, double %t4
  ret double %ret
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

define(`rcp14_uniform', `
declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone
define float @__rcp_fast_uniform_float(float) nounwind readonly alwaysinline {
  %vecval = insertelement <4 x float> undef, float %0, i32 0
  %call = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %vecval, <4 x float> %vecval, <4 x float> undef, i8 -1)
  %scall = extractelement <4 x float> %call, i32 0
  ret float %scall
}

define float @__rcp_uniform_float(float %v) nounwind readonly alwaysinline {
  %iv = call float @__rcp_fast_uniform_float(float %v)

  ; do one N-R iteration to improve precision
  ; iv = rcp(v)
  ; iv * (2. - v * iv)
  %v_iv = fmul float %v, %iv
  %two_minus = fsub float 2., %v_iv
  %iv_mul = fmul float %iv, %two_minus
  ret float %iv_mul
}

declare <2 x double> @llvm.x86.avx512.rcp14.sd(<2 x double>, <2 x double>, <2 x double>, i8) nounwind readnone
define double @__rcp_fast_uniform_double(double) nounwind readonly alwaysinline {
  %vecval = insertelement <2 x double> undef, double %0, i32 0
  %call = call <2 x double> @llvm.x86.avx512.rcp14.sd(<2 x double> %vecval, <2 x double> %vecval, <2 x double> undef, i8 -1)
  %scall = extractelement <2 x double> %call, i32 0
  ret double %scall
}

define double @__rcp_uniform_double(double %v) nounwind readonly alwaysinline {
  %iv = call double @__rcp_fast_uniform_double(double %v)

  ; do one N-R iteration to improve precision
  ; iv = rcp(v)
  ; iv * (2. - v * iv)
  %v_iv = fmul double %v, %iv
  %two_minus = fsub double 2., %v_iv
  %iv_mul = fmul double %iv, %two_minus
  ret double %iv_mul
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; switch macro
;; This is required to ensure that gather intrinsics are used with constant scale value.
;; This particular implementation of the routine is used by avx512 targets only.
;; $1: Return value
;; $2: funcName
;; $3: Width
;; $4: scalar type of array
;; $5: ptr
;; $6: offset
;; $7: scalar type of offset
;; $8: vecMask
;; $9: scalar type of vecMask
;; $10: scale
define(`convert_scale_to_const_gather', `


 switch i32 %argn(`10',$@), label %default_$1 [ i32 1, label %on_one_$1
                                                i32 2, label %on_two_$1
                                                i32 4, label %on_four_$1
                                                i32 8, label %on_eight_$1]

on_one_$1:
  %$1_1 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, $9 %$8, i32 1)
  br label %end_bb_$1

on_two_$1:
  %$1_2 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, $9 %$8, i32 2)
  br label %end_bb_$1

on_four_$1:
  %$1_4 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, $9 %$8, i32 4)
  br label %end_bb_$1

on_eight_$1:
  %$1_8 = call <$3 x $4> @$2(<$3 x $4> undef, i8 * %$5, <$3 x $7> %$6, $9 %$8, i32 8)
  br label %end_bb_$1

default_$1:
  unreachable

end_bb_$1:
  %$1 = phi <$3 x $4> [ %$1_1, %on_one_$1 ], [ %$1_2, %on_two_$1 ], [ %$1_4, %on_four_$1 ], [ %$1_8, %on_eight_$1 ]
'
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; switch macro
;; This is required to ensure that scatter intrinsics are used with constant scale value.
;; This is used by avx512 targets only.
;; $1: funcName
;; $2: Width
;; $3: Array
;; $4: scalar type of array
;; $5: ptr
;; $6: offset
;; $7: scalar type of offset
;; $8: vecMask
;; $9: scalar type of vecMask
;; $10: scale
define(`convert_scale_to_const_scatter', `


 switch i32 %argn(`10',$@), label %default_$3 [ i32 1, label %on_one_$3
                                                i32 2, label %on_two_$3
                                                i32 4, label %on_four_$3
                                                i32 8, label %on_eight_$3]

on_one_$3:
  call void @$1(i8* %$5, $9 %$8, <$2 x $7> %$6, <$2 x $4> %$3, i32 1)
  br label %end_bb_$3

on_two_$3:
  call void @$1(i8* %$5, $9 %$8, <$2 x $7> %$6, <$2 x $4> %$3, i32 2)
  br label %end_bb_$3

on_four_$3:
  call void @$1(i8* %$5, $9 %$8, <$2 x $7> %$6, <$2 x $4> %$3, i32 4)
  br label %end_bb_$3

on_eight_$3:
  call void @$1(i8* %$5, $9 %$8, <$2 x $7> %$6, <$2 x $4> %$3, i32 8)
  br label %end_bb_$3

default_$3:
  unreachable

end_bb_$3:
'
)

