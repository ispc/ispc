;;
;; target-neon-8.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2023 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`MASK',`i8')
define(`ISA',`NEON')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone alwaysinline {
  unary4to16conv(r, i16, float, @NEON_PREFIX.vcvthf2fp, %v)
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone alwaysinline {
  unary4to16conv(r, float, i16, @NEON_PREFIX.vcvtfp2hf, %v)
  ret <16 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

;; round/floor/ceil

;; FIXME: grabbed these from the sse2 target, which does not have native
;; instructions for these.  Is there a better approach for NEON?

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <16 x float> %0 to <16 x i32>
  %bitop.i.i = and <16 x i32> %float_to_int_bitcast.i.i.i.i,
    <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648,
     i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648,
     i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648,
     i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %bitop.i = xor <16 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <16 x i32> %bitop.i to <16 x float>
  %binop.i = fadd <16 x float> %int_to_float_bitcast.i.i40.i,
    <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06,
     float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06,
     float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06,
     float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06>
  %binop21.i = fadd <16 x float> %binop.i,
    <float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06,
     float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06,
     float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06,
     float -8.388608e+06, float -8.388608e+06, float -8.388608e+06, float -8.388608e+06>
  %float_to_int_bitcast.i.i.i = bitcast <16 x float> %binop21.i to <16 x i32>
  %bitop31.i = xor <16 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <16 x i32> %bitop31.i to <16 x float>
  ret <16 x float> %int_to_float_bitcast.i.i.i
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <16 x float> @__round_varying_float(<16 x float> %0) nounwind
  %bincmp.i = fcmp ogt <16 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <16 x i1> %bincmp.i to <16 x i32>
  %bitop.i = and <16 x i32> %val_to_boolvec32.i,
    <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432,
     i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432,
     i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432,
     i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <16 x i32> %bitop.i to <16 x float>
  %binop.i = fadd <16 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <16 x float> %binop.i
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <16 x float> @__round_varying_float(<16 x float> %0) nounwind
  %bincmp.i = fcmp olt <16 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <16 x i1> %bincmp.i to <16 x i32>
  %bitop.i = and <16 x i32> %val_to_boolvec32.i,
    <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216,
     i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216,
     i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216,
     i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <16 x i32> %bitop.i to <16 x float>
  %binop.i = fadd <16 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <16 x float> %binop.i
}

;; FIXME: rounding doubles and double vectors needs to be implemented
declare <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readnone 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; trunc float/double

truncate()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  binary4to16(r, float, @llvm.arm.neon.vmaxs.v4f32, %0, %1)
  ret <WIDTH x float> %r
}

define <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone alwaysinline {
  binary4to16(r, float, @llvm.arm.neon.vmins.v4f32, %0, %1)
  ret <WIDTH x float> %r
}

declare <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  binary4to16(r, i32, @llvm.arm.neon.vmins.v4i32, %0, %1)
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  binary4to16(r, i32, @llvm.arm.neon.vmaxs.v4i32, %0, %1)
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  binary4to16(r, i32, @llvm.arm.neon.vminu.v4i32, %0, %1)
  ret <WIDTH x i32> %r
}

define <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone alwaysinline {
  binary4to16(r, i32, @llvm.arm.neon.vmaxu.v4i32, %0, %1)
  ret <WIDTH x i32> %r
}

;; sqrt/rsqrt/rcp

declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(x0, float, @llvm.arm.neon.vrecpe.v4f32, %d)
  binary4to16(x0_nr, float, @llvm.arm.neon.vrecps.v4f32, %d, %x0)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  binary4to16(x1_nr, float, @llvm.arm.neon.vrecps.v4f32, %d, %x1)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(ret, float, @llvm.arm.neon.vrecpe.v4f32, %d)
  ret <WIDTH x float> %ret
}

declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(x0, float, @llvm.arm.neon.vrsqrte.v4f32, %d)
  %x0_2 = fmul <WIDTH x float> %x0, %x0
  binary4to16(x0_nr, float, @llvm.arm.neon.vrsqrts.v4f32, %d, %x0_2)
  %x1 = fmul <WIDTH x float> %x0, %x0_nr
  %x1_2 = fmul <WIDTH x float> %x1, %x1
  binary4to16(x1_nr, float, @llvm.arm.neon.vrsqrts.v4f32, %d, %x1_2)
  %x2 = fmul <WIDTH x float> %x1, %x1_nr
  ret <WIDTH x float> %x2
}

define <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float> %d) nounwind readnone alwaysinline {
  unary4to16(ret, float, @llvm.arm.neon.vrsqrte.v4f32, %d)
  ret <WIDTH x float> %ret
}

define float @__rsqrt_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <16 x float> @__rsqrt_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rsqrt_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <16 x float> undef, float %0, i32 0
  %vr = call <16 x float> @__rsqrt_fast_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone alwaysinline {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef,
                      i32 undef, i32 undef, i32 undef, i32 undef>
  %vr = call <16 x float> @__rcp_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_fast_uniform_float(float) nounwind readnone alwaysinline {
  %vs = insertelement <16 x float> undef, float %0, i32 0
  %vr = call <16 x float> @__rcp_fast_varying_float(<16 x float> %vs)
  %r = extractelement <16 x float> %vr, i32 0
  ret float %r
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline {
  unary4to16(result, float, @llvm.sqrt.v4f32, %0)
;; this returns nan for v=0, which is undesirable..
;;  %rsqrt = call <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %0)
;;  %result = fmul <4 x float> %rsqrt, %0
  ret <16 x float> %result
}

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)

define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone alwaysinline {
  unary4to16(r, double, @llvm.sqrt.v4f64, %0)
  ret <WIDTH x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %and_mask = and <WIDTH x i8> %0,
    <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128,
     i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128>
  %v8 = call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %and_mask)
  %v4 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %v8)
  %v2 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %v4)
  %va = extractelement <2 x i64> %v2, i32 0
  %vb = extractelement <2 x i64> %v2, i32 1
  %vbshift = shl i64 %vb, 8
  %v = or i64 %va, %vbshift
  ret i64 %v
}

define i1 @__any(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v16tov8(MASK, %0, %v8a, %v8b)
  %vor8 = or <8 x MASK> %v8a, %v8b
  %v16 = sext <8 x i8> %vor8 to <8 x i16>
  v8tov4(i16, %v16, %v16a, %v16b)
  %vor16 = or <4 x i16> %v16a, %v16b
  %v32 = sext <4 x i16> %vor16 to <4 x i32>
  v4tov2(i32, %v32, %v32a, %v32b)
  %vor32 = or <2 x i32> %v32a, %v32b
  %v0 = extractelement <2 x i32> %vor32, i32 0
  %v1 = extractelement <2 x i32> %vor32, i32 1
  %v = or i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__all(<WIDTH x MASK>) nounwind readnone alwaysinline {
  v16tov8(MASK, %0, %v8a, %v8b)
  %vand8 = and <8 x MASK> %v8a, %v8b
  %v16 = sext <8 x i8> %vand8 to <8 x i16>
  v8tov4(i16, %v16, %v16a, %v16b)
  %vand16 = and <4 x i16> %v16a, %v16b
  %v32 = sext <4 x i16> %vand16 to <4 x i32>
  v4tov2(i32, %v32, %v32a, %v32b)
  %vand32 = and <2 x i32> %v32a, %v32b
  %v0 = extractelement <2 x i32> %vand32, i32 0
  %v1 = extractelement <2 x i32> %vand32, i32 1
  %v = and i32 %v0, %v1
  %cmp = icmp ne i32 %v, 0
  ret i1 %cmp
}

define i1 @__none(<WIDTH x MASK>) nounwind readnone alwaysinline {
  %any = call i1 @__any(<WIDTH x MASK> %0)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

;; $1: scalar type
;; $2: vector/vector reduce function (2 x <WIDTH x vec> -> <WIDTH x vec>)
;; $3: pairwise vector reduce function (2 x <2 x vec> -> <2 x vec>)
;; $4: scalar reduce function

define(`neon_reduce', `
  v16tov8($1, %0, %va, %vb)
  %va_16 = shufflevector <8 x $1> %va, <8 x $1> undef,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  %vb_16 = shufflevector <8 x $1> %vb, <8 x $1> undef,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  %v8 = call <16 x $1> $2(<16 x $1> %va_16, <16 x $1> %vb_16)

  %v8a = shufflevector <16 x $1> %v8, <16 x $1> undef,
    <16 x i32> <i32 0, i32 1, i32 2, i32 3, 
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>
  %v8b = shufflevector <16 x $1> %v8, <16 x $1> undef,
    <16 x i32> <i32 4, i32 5, i32 6, i32 7, 
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef,
                i32 undef, i32 undef, i32 undef, i32 undef>

  %v4 = call <16 x $1> $2(<16 x $1> %v8a, <16 x $1> %v8b)

  %vfirst_4 = shufflevector <16 x $1> %v4, <16 x $1> undef,
    <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  v4tov2($1, %vfirst_4, %v0, %v1)
  %vh = call <2 x $1> $3(<2 x $1> %v0, <2 x $1> %v1)
  %vh0 = extractelement <2 x $1> %vh, i32 0
  %vh1 = extractelement <2 x $1> %vh, i32 1
  %r = call $1 $4($1 %vh0, $1 %vh1)
  ret $1 %r
')

declare <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @add_f32(float, float) nounwind readnone alwaysinline {
  %r = fadd float %0, %1
  ret float %r
}

define internal <WIDTH x float> @__add_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone alwaysinline {
  %r = fadd <WIDTH x float> %0, %1
  ret <WIDTH x float> %r
}

define float @__reduce_add_float(<WIDTH x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @__add_varying_float, @llvm.arm.neon.vpadd.v2f32, @add_f32)
}

declare <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @min_f32(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp olt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_min_float(<WIDTH x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @__min_varying_float, @llvm.arm.neon.vpmins.v2f32, @min_f32)
}

declare <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @max_f32(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_max_float(<WIDTH x float>) nounwind readnone alwaysinline {
  neon_reduce(float, @__max_varying_float, @llvm.arm.neon.vpmaxs.v2f32, @max_f32)
}

declare <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) nounwind readnone

define i64 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone alwaysinline {
  %a16 = call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %0)
  %a32 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %a16)
  %a64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %a32)
  %a0 = extractelement <2 x i64> %a64, i32 0
  %a1 = extractelement <2 x i64> %a64, i32 1
  %r = add i64 %a0, %a1
  ret i64 %r
}

define i64 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone alwaysinline {
  v16tov8(i16, %0, %va, %vb)
  %a32 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %va)
  %b32 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %vb)
  %a64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %a32)
  %b64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %b32)
  %sum = add <2 x i64> %a64, %b64
  %a0 = extractelement <2 x i64> %sum, i32 0
  %a1 = extractelement <2 x i64> %sum, i32 1
  %r = add i64 %a0, %a1
  ret i64 %r
}

define i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone alwaysinline {
  v16tov4(i32, %0, %va, %vb, %vc, %vd)
  %a64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %va)
  %b64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %vb)
  %c64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %vc)
  %d64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %vd)
  %ab = add <2 x i64> %a64, %b64
  %cd = add <2 x i64> %c64, %d64
  %sum = add <2 x i64> %ab, %cd
  %a0 = extractelement <2 x i64> %sum, i32 0
  %a1 = extractelement <2 x i64> %sum, i32 1
  %r = add i64 %a0, %a1
  ret i64 %r
}

declare <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_si32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_int32(<WIDTH x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @__min_varying_int32, @llvm.arm.neon.vpmins.v2i32, @min_si32)
}

declare <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_si32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_int32(<WIDTH x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @__max_varying_int32, @llvm.arm.neon.vpmaxs.v2i32, @max_si32)
}

declare <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_ui32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_uint32(<WIDTH x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @__min_varying_uint32, @llvm.arm.neon.vpmins.v2i32, @min_ui32)
}

declare <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_ui32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_uint32(<WIDTH x i32>) nounwind readnone alwaysinline {
  neon_reduce(i32, @__max_varying_uint32, @llvm.arm.neon.vpmaxs.v2i32, @max_ui32)
}

define internal double @__add_uniform_double(double, double) nounwind readnone alwaysinline {
  %r = fadd double %0, %1
  ret double %r
}

define internal <WIDTH x double> @__add_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind readnone alwaysinline {
  %r = fadd <WIDTH x double> %0, %1
  ret <WIDTH x double> %r
}

define double @__reduce_add_double(<WIDTH x double>) nounwind readnone alwaysinline {
  reduce16(double, @__add_varying_double, @__add_uniform_double)
}

define double @__reduce_min_double(<WIDTH x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<WIDTH x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}

define internal i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %r = add i64 %0, %1
  ret i64 %r
}

define internal <WIDTH x i64> @__add_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %r = add <WIDTH x i64> %0, %1
  ret <WIDTH x i64> %r
}

define i64 @__reduce_add_int64(<WIDTH x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}

define i64 @__reduce_min_int64(<WIDTH x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}

define i64 @__reduce_max_int64(<WIDTH x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}

define i64 @__reduce_min_uint64(<WIDTH x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}

define i64 @__reduce_max_uint64(<WIDTH x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

declare <16 x i8> @llvm.arm.neon.vrhaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @__avg_up_uint8(<16 x i8>, <16 x i8>) nounwind readnone alwaysinline {
  %r = call <16 x i8> @llvm.arm.neon.vrhaddu.v16i8(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %r
}

declare <16 x i8> @llvm.arm.neon.vrhadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @__avg_up_int8(<16 x i8>, <16 x i8>) nounwind readnone alwaysinline {
  %r = call <16 x i8> @llvm.arm.neon.vrhadds.v16i8(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %r
}

declare <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @__avg_down_uint8(<16 x i8>, <16 x i8>) nounwind readnone alwaysinline {
  %r = call <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %r
}

declare <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @__avg_down_int8(<16 x i8>, <16 x i8>) nounwind readnone alwaysinline {
  %r = call <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %r
}

declare <8 x i16> @llvm.arm.neon.vrhaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @__avg_up_uint16(<16 x i16>, <16 x i16>) nounwind readnone alwaysinline {
  v16tov8(i16, %0, %a0, %b0)
  v16tov8(i16, %1, %a1, %b1)
  %r0 = call <8 x i16> @llvm.arm.neon.vrhaddu.v8i16(<8 x i16> %a0, <8 x i16> %a1)
  %r1 = call <8 x i16> @llvm.arm.neon.vrhaddu.v8i16(<8 x i16> %b0, <8 x i16> %b1)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
}

declare <8 x i16> @llvm.arm.neon.vrhadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @__avg_up_int16(<16 x i16>, <16 x i16>) nounwind readnone alwaysinline {
  v16tov8(i16, %0, %a0, %b0)
  v16tov8(i16, %1, %a1, %b1)
  %r0 = call <8 x i16> @llvm.arm.neon.vrhadds.v8i16(<8 x i16> %a0, <8 x i16> %a1)
  %r1 = call <8 x i16> @llvm.arm.neon.vrhadds.v8i16(<8 x i16> %b0, <8 x i16> %b1)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
}

declare <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @__avg_down_uint16(<16 x i16>, <16 x i16>) nounwind readnone alwaysinline {
  v16tov8(i16, %0, %a0, %b0)
  v16tov8(i16, %1, %a1, %b1)
  %r0 = call <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16> %a0, <8 x i16> %a1)
  %r1 = call <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16> %b0, <8 x i16> %b1)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
}

declare <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i16> @__avg_down_int16(<16 x i16>, <16 x i16>) nounwind readnone alwaysinline {
  v16tov8(i16, %0, %a0, %b0)
  v16tov8(i16, %1, %a1, %b1)
  %r0 = call <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16> %a0, <8 x i16> %a1)
  %r1 = call <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16> %b0, <8 x i16> %b1)
  v8tov16(i16, %r0, %r1, %r)
  ret <16 x i16> %r
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
saturation_arithmetic()
