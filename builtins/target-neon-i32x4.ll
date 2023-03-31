;;
;; target-neon-32.ll
;;
;;  Copyright(c) 2012-2013 Matt Pharr
;;  Copyright(c) 2013, 2015 Google, Inc.
;;  Copyright(c) 2019-2023 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`MASK',`i32')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

define <4 x float> @__half_to_float_varying(<4 x i16> %v) nounwind readnone alwaysinline {
  %r = call <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16> %v)
  ret <4 x float> %r
}

define <4 x i16> @__float_to_half_varying(<4 x float> %v) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float> %v)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

;; round/floor/ceil

;; FIXME: Modify for 32 bit arm.
;; instructions for these.  Is there a better approach for NEON?

declare <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %a) nounwind readnone
define <4 x float> @__round_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %ret = call <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %0)
  ret <4 x float> %ret
}

declare <4 x float> @llvm.floor.v4f32(<4 x float> %a) nounwind readnone
define <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
    %ret = call <4 x float> @llvm.floor.v4f32(<4 x float> %0)
    ret <4 x float> %ret
}

declare <4 x float> @llvm.ceil.v4f32(<4 x float> %a) nounwind readnone
define <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
    %ret = call <4 x float> @llvm.ceil.v4f32(<4 x float> %0)
    ret <4 x float> %ret
}

;; FIXME: Modify for 32 bit arm.
declare <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %a) nounwind readnone
define <4 x double> @__round_varying_double(<4 x double>) nounwind readonly alwaysinline {
    %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
    %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
    %v0_round = call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %v0)
    %v1_round = call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %v1)
    %ret = shufflevector <2 x double> %v0_round, <2 x double> %v1_round,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    ret <4 x double> %ret
}

declare <2 x double> @llvm.floor.v2f64(<2 x double> %a) nounwind readnone
define <4 x double> @__floor_varying_double(<4 x double>) nounwind readonly alwaysinline {
    %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
    %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
    %v0_round = call <2 x double> @llvm.floor.v2f64(<2 x double> %v0)
    %v1_round = call <2 x double> @llvm.floor.v2f64(<2 x double> %v1)
    %ret = shufflevector <2 x double> %v0_round, <2 x double> %v1_round,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    ret <4 x double> %ret
}

declare  <2 x double> @llvm.ceil.v2f64(<2 x double> %a) nounwind readnone
define <4 x double> @__ceil_varying_double(<4 x double>) nounwind readonly alwaysinline {
    %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
    %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
    %v0_round = call <2 x double> @llvm.ceil.v2f64(<2 x double> %v0)
    %v1_round = call <2 x double> @llvm.ceil.v2f64(<2 x double> %v1)
    %ret = shufflevector <2 x double> %v0_round, <2 x double> %v1_round,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    ret <4 x double> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float and double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

declare <4 x float> @NEON_PREFIX_FMIN.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_FMAX.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %r = call <4 x float> @NEON_PREFIX_FMAX.v4f32(<4 x float> %0, <4 x float> %1)
  ret <WIDTH x float> %r
}

define <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  %r = call <4 x float> @NEON_PREFIX_FMIN.v4f32(<4 x float> %0, <4 x float> %1)
  ret <WIDTH x float> %r
}

declare <4 x i32> @NEON_PREFIX_IMINS.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @NEON_PREFIX_IMINU.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @NEON_PREFIX_IMAXS.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @NEON_PREFIX_IMAXU.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %r = call <4 x i32> @NEON_PREFIX_IMINS.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %r = call <4 x i32> @NEON_PREFIX_IMAXS.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %r = call <4 x i32> @NEON_PREFIX_IMINU.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  %r = call <4 x i32> @NEON_PREFIX_IMAXU.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

;; sqrt/rsqrt/rcp

declare <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %x0 = call <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float> %d)
  %x0_nr = call <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float> %d, <4 x float> %x0)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_nr = call <4 x float> @NEON_PREFIX_RECPSQ.v4f32(<4 x float> %d, <4 x float> %x1)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %ret = call <4 x float> @NEON_PREFIX_RECPEQ.v4f32(<4 x float> %d)
  ret <4 x float> %ret
}

declare <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %x0 = call <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float> %d)
  %x0_2 = fmul <4 x float> %x0, %x0
  %x0_nr = call <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float> %d, <4 x float> %x0_2)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_2 = fmul <4 x float> %x1, %x1
  %x1_nr = call <4 x float> @NEON_PREFIX_RSQRTSQ.v4f32(<4 x float> %d, <4 x float> %x1_2)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

define float @__rsqrt_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rsqrt_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  %ret = call <4 x float> @NEON_PREFIX_RSQRTEQ.v4f32(<4 x float> %d)
  ret <4 x float> %ret
}

define float @__rsqrt_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <4 x float> undef, float %0, i32 0
  %vr = call <4 x float> @__rsqrt_fast_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rcp_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <4 x float> undef, float %0, i32 0
  %vr = call <4 x float> @__rcp_fast_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline {
  %result = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)
;; this returns nan for v=0, which is undesirable..
;;  %rsqrt = call <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %0)
;;  %result = fmul <4 x float> %rsqrt, %0
  ret <4 x float> %result
}

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)

define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone alwaysinline {
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %0)
  ret <4 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

ifelse(
RUNTIME, `64', `
dnl Reductions across lanes are only available in ARM64.
declare i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32>)
declare float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float>)
declare float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float>)
declare float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float>)
declare i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8>)
declare i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16>)
declare i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32>)
declare double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double>)
declare <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double>, <2 x double>)
declare double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double>)
declare <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double>, <2 x double>)
declare double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double>)
declare i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64>)

define i64 @__movmsk(<4 x i32>) nounwind readnone alwaysinline {
  %and_mask = and <4 x i32> %0, <i32 1, i32 2, i32 4, i32 8>
  %v = call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %and_mask)
  %mask64 = zext i32 %v to i64
  ret i64 %mask64
}

define i1 @__any(<4 x i32>) nounwind readnone alwaysinline {
  %v = call i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32> %0)
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<4 x i32>) nounwind readnone alwaysinline {
  %v = call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %0)
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__none(<4 x i32>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<4 x i32> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

define float @__reduce_add_float(<4 x float>) nounwind readnone alwaysinline {
  %r = call float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float> %0)
  ret float %r
}

define float @__reduce_min_float(<4 x float>) nounwind readnone alwaysinline {
  %r = call float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float> %0)
  ret float %r
}

define float @__reduce_max_float(<4 x float>) nounwind readnone alwaysinline {
  %r = call float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float> %0)
  ret float %r
}

define i16 @__reduce_add_int8(<4 x i8>) nounwind readnone alwaysinline {
  %v8 = shufflevector <4 x i8> %0, <4 x i8> zeroinitializer,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %r = call i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8> %v8)
  %r16 = trunc i32 %r to i16
  ret i16 %r16
}

define i32 @__reduce_add_int16(<4 x i16>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16> %0)
  ret i32 %r
}

define i64 @__reduce_add_int32(<4 x i32>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %0)
  %r64 = sext i32 %r to i64
  ret i64 %r64
}

define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %0)
  ret i32 %r
}

define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32> %0)
  ret i32 %r
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %0)
  ret i32 %r
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone alwaysinline {
  %r = call i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32> %0)
  ret i32 %r
}

define double @__reduce_add_double(<4 x double>) nounwind readnone alwaysinline {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = fadd <2 x double> %v0, %v1
  %m = call double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double> %sum)
  ret double %m
}

define double @__reduce_min_double(<4 x double>) nounwind readnone alwaysinline {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
  %x = call <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double> %v0, <2 x double> %v1)
  %m = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> %x)
  ret double %m
}

define double @__reduce_max_double(<4 x double>) nounwind readnone alwaysinline {
  %v0 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x double> %0, <4 x double> undef,
                      <2 x i32> <i32 2, i32 3>
  %x = call <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double> %v0, <2 x double> %v1)
  %m = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> %x)
  ret double %m
}

define i64 @__reduce_add_int64(<4 x i64>) nounwind readnone alwaysinline {
  %v0 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x i64> %0, <4 x i64> undef,
                      <2 x i32> <i32 2, i32 3>
  %sum = add <2 x i64> %v0, %v1
  %m = call i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64> %sum)
  ret i64 %m
}
',
RUNTIME, `32',
`
define i64 @__movmsk(<4 x MASK>) nounwind readnone alwaysinline {
  %and_mask = and <4 x MASK> %0, <MASK 1, MASK 2, MASK 4, MASK 8>
  %v01 = shufflevector <4 x i32> %and_mask, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %v23 = shufflevector <4 x i32> %and_mask, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vor = or <2 x i32> %v01, %v23
  %v0 = extractelement <2 x i32> %vor, i32 0
  %v1 = extractelement <2 x i32> %vor, i32 1
  %v = or i32 %v0, %v1
  %mask64 = zext i32 %v to i64
  ret i64 %mask64
}

define i1 @__any(<4 x i32>) nounwind readnone alwaysinline {
  %v01 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %v23 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vor = or <2 x i32> %v01, %v23
  %v0 = extractelement <2 x i32> %vor, i32 0
  %v1 = extractelement <2 x i32> %vor, i32 1
  %v = or i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<4 x i32>) nounwind readnone alwaysinline {
  %v01 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %v23 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vor = and <2 x i32> %v01, %v23
  %v0 = extractelement <2 x i32> %vor, i32 0
  %v1 = extractelement <2 x i32> %vor, i32 1
  %v = and i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__none(<4 x i32>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<4 x i32> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}
'
dnl $1: scalar type
dnl $2: vector reduce function (2 x <2 x vec> -> <2 x vec>)
dnl $3 scalar reduce function

define(`neon_reduce', `
  %v0 = shufflevector <4 x $1> %0, <4 x $1> undef, <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x $1> %0, <4 x $1> undef, <2 x i32> <i32 2, i32 3>
  %vh = call <2 x $1> $2(<2 x $1> %v0, <2 x $1> %v1)
  %vh0 = extractelement <2 x $1> %vh, i32 0
  %vh1 = extractelement <2 x $1> %vh, i32 1
  %r = call $1$3 ($1 %vh0, $1 %vh1)
  ret $1 %r
')
`
declare <2 x float> @NEON_PREFIX_FPADD.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @add_f32(float, float) nounwind readnone alwaysinline {
  %r = fadd float %0, %1
  ret float %r
}

define float @__reduce_add_float(<4 x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @NEON_PREFIX_FPADD.v2f32, @add_f32)
}

declare <2 x float> @NEON_PREFIX_PMINF.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @min_f32(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp olt float %0, %1

  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_min_float(<4 x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @NEON_PREFIX_PMINF.v2f32, @min_f32)
}

declare <2 x float> @NEON_PREFIX_PMAXF.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @max_f32(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_max_float(<4 x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @NEON_PREFIX_PMAXF.v2f32, @max_f32)
}

declare <4 x i16> @NEON_PREFIX_PADDLS.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @NEON_PREFIX_PADDLU.v2i32.v4i16(<4 x i16>) nounwind readnone

define i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone alwaysinline {
  %v8 = shufflevector <4 x i8> %0, <4 x i8> zeroinitializer,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %a16 = call <4 x i16> @NEON_PREFIX_PADDLS.v4i16.v8i8(<8 x i8> %v8)
  %a32 = call <2 x i32> @NEON_PREFIX_PADDLU.v2i32.v4i16(<4 x i16> %a16)
  %a0 = extractelement <2 x i32> %a32, i32 0
  %a1 = extractelement <2 x i32> %a32, i32 1
  %r = add i32 %a0, %a1
  %r16 = trunc i32 %r to i16
  ret i16 %r16
}

define i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone alwaysinline {
  %a32 = call <2 x i32> @NEON_PREFIX_PADDLU.v2i32.v4i16(<4 x i16> %0)
  %a0 = extractelement <2 x i32> %a32, i32 0
  %a1 = extractelement <2 x i32> %a32, i32 1
  %r = add i32 %a0, %a1
  ret i32 %r
}

declare <2 x i64> @NEON_PREFIX_PADDLU.v2i64.v4i32(<4 x i32>) nounwind readnone

define i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone alwaysinline {
  %a64 = call <2 x i64> @NEON_PREFIX_PADDLU.v2i64.v4i32(<4 x i32> %0)
  %a0 = extractelement <2 x i64> %a64, i32 0
  %a1 = extractelement <2 x i64> %a64, i32 1
  %r = add i64 %a0, %a1
  ret i64 %r
}

declare <2 x i32> @NEON_PREFIX_PMINS.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_si32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @NEON_PREFIX_PMINS.v2i32, @min_si32)
}

declare <2 x i32> @NEON_PREFIX_PMAXS.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_si32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @NEON_PREFIX_PMAXS.v2i32, @max_si32)
}

declare <2 x i32> @NEON_PREFIX_PMINU.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_ui32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @NEON_PREFIX_PMINU.v2i32, @min_ui32)
}

declare <2 x i32> @NEON_PREFIX_PMAXU.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_ui32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @NEON_PREFIX_PMAXU.v2i32, @max_ui32)
}

define double @__reduce_add_double(<4 x double>) nounwind readnone alwaysinline {
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

define double @__reduce_min_double(<4 x double>) nounwind readnone alwaysinline {
  reduce4(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<4 x double>) nounwind readnone alwaysinline {
  reduce4(double, @__max_varying_double, @__max_uniform_double)
}

define i64 @__reduce_add_int64(<4 x i64>) nounwind readnone alwaysinline {
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
')

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
;; int8/int16

declare <8 x i8> @NEON_PREFIX_RHADDLU.v8i8(<8 x i8>, <8 x i8>) nounwind readnone

define <4 x i8> @__avg_up_uint8(<4 x i8>, <4 x i8>) nounwind readnone alwaysinline {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_RHADDLU.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <8 x i8> @NEON_PREFIX_RHADDLS.v8i8(<8 x i8>, <8 x i8>) nounwind readnone

define <4 x i8> @__avg_up_int8(<4 x i8>, <4 x i8>) nounwind readnone alwaysinline {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_RHADDLS.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <8 x i8> @NEON_PREFIX_HADDLU.v8i8(<8 x i8>, <8 x i8>) nounwind readnone

define <4 x i8> @__avg_down_uint8(<4 x i8>, <4 x i8>) nounwind readnone alwaysinline {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_HADDLU.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <8 x i8> @NEON_PREFIX_HADDLS.v8i8(<8 x i8>, <8 x i8>) nounwind readnone

define <4 x i8> @__avg_down_int8(<4 x i8>, <4 x i8>) nounwind readnone alwaysinline {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_HADDLS.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <4 x i16> @NEON_PREFIX_RHADDLU.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_up_uint16(<4 x i16>, <4 x i16>) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX_RHADDLU.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @NEON_PREFIX_RHADDLS.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_up_int16(<4 x i16>, <4 x i16>) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX_RHADDLS.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @NEON_PREFIX_HADDLU.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_down_uint16(<4 x i16>, <4 x i16>) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX_HADDLU.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @NEON_PREFIX_HADDLS.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_down_int16(<4 x i16>, <4 x i16>) nounwind readnone alwaysinline {
  %r = call <4 x i16> @NEON_PREFIX_HADDLS.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; saturation arithmetic

declare <8 x i8> @NEON_PREFIX_QADDS.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
define <4 x i8> @__padds_vi8(<4 x i8>, <4 x i8>) {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_QADDS.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <4 x i16> @NEON_PREFIX_QADDS.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
define <4 x i16> @__padds_vi16(<4 x i16>, <4 x i16>) {
  %r = call <4 x i16> @NEON_PREFIX_QADDS.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <8 x i8> @NEON_PREFIX_QADDU.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
define <4 x i8> @__paddus_vi8(<4 x i8>, <4 x i8>) {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_QADDU.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <4 x i16> @NEON_PREFIX_QADDU.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
define <4 x i16> @__paddus_vi16(<4 x i16>, <4 x i16>) {
  %r = call <4 x i16> @NEON_PREFIX_QADDU.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <8 x i8> @NEON_PREFIX_QSUBS.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
define <4 x i8> @__psubs_vi8(<4 x i8>, <4 x i8>) {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_QSUBS.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <4 x i16> @NEON_PREFIX_QSUBS.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
define <4 x i16> @__psubs_vi16(<4 x i16>, <4 x i16>) {
  %r = call <4 x i16> @NEON_PREFIX_QSUBS.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <8 x i8> @NEON_PREFIX_QSUBU.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
define <4 x i8> @__psubus_vi8(<4 x i8>, <4 x i8>) {
  convert4to8(i8, %0, %v0)
  convert4to8(i8, %1, %v1)
  %r8 = call <8 x i8> @NEON_PREFIX_QSUBU.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  convert8to4(i8, %r8, %r)
  ret <4 x i8> %r
}

declare <4 x i16> @NEON_PREFIX_QSUBU.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
define <4 x i16> @__psubus_vi16(<4 x i16>, <4 x i16>) {
  %r = call <4 x i16> @NEON_PREFIX_QSUBU.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
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
