;;
;; target-neon-32.ll
;;
;;  Copyright(c) 2012-2013 Matt Pharr
;;  Copyright(c) 2013, 2015 Google, Inc.
;;
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
;;    * Neither the name of Matt Pharr nor the names of its
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

define(`WIDTH',`4')
define(`MASK',`i32')

include(`util.m4')
include(`target-neon-common.ll')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

define <4 x float> @__half_to_float_varying(<4 x i16> %v) nounwind readnone {
  %r = call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %v)
  ret <4 x float> %r
}

define <4 x i16> @__float_to_half_varying(<4 x float> %v) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %v)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

;; round/floor/ceil

;; FIXME: grabbed these from the sse2 target, which does not have native
;; instructions for these.  Is there a better approach for NEON?

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

define <4 x float> @__floor_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp ogt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 -1082130432, i32 -1082130432, i32 -1082130432, i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

define <4 x float> @__ceil_varying_float(<4 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <4 x float> @__round_varying_float(<4 x float> %0) nounwind
  %bincmp.i = fcmp olt <4 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <4 x i1> %bincmp.i to <4 x i32>
  %bitop.i = and <4 x i32> %val_to_boolvec32.i, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %int_to_float_bitcast.i.i.i = bitcast <4 x i32> %bitop.i to <4 x float>
  %binop.i = fadd <4 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <4 x float> %binop.i
}

;; FIXME: rounding doubles and double vectors needs to be implemented
declare <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readnone 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone {
  %r = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> %0, <4 x float> %1)
  ret <WIDTH x float> %r
}

define <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                            <WIDTH x float>) nounwind readnone {
  %r = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %0, <4 x float> %1)
  ret <WIDTH x float> %r
}

declare <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone {
  %r = call <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone {
  %r = call <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone {
  %r = call <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

define <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone {
  %r = call <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %r
}

;; sqrt/rsqrt/rcp

declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float> %d) nounwind readnone {
  %x0 = call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %d)
  %x0_nr = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %d, <4 x float> %x0)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_nr = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %d, <4 x float> %x1)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float>, <4 x float>) nounwind readnone

define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %d) nounwind readnone {
  %x0 = call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %d)
  %x0_2 = fmul <4 x float> %x0, %x0
  %x0_nr = call <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float> %d, <4 x float> %x0_2)
  %x1 = fmul <4 x float> %x0, %x0_nr
  %x1_2 = fmul <4 x float> %x1, %x1
  %x1_nr = call <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float> %d, <4 x float> %x1_2)
  %x2 = fmul <4 x float> %x1, %x1_nr
  ret <4 x float> %x2
}

define float @__rsqrt_uniform_float(float) nounwind readnone {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rsqrt_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

define float @__rcp_uniform_float(float) nounwind readnone {
  %v1 = bitcast float %0 to <1 x float>
  %vs = shufflevector <1 x float> %v1, <1 x float> undef,
          <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %vr = call <4 x float> @__rcp_varying_float(<4 x float> %vs)
  %r = extractelement <4 x float> %vr, i32 0
  ret float %r
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone {
  %result = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)
;; this returns nan for v=0, which is undesirable..
;;  %rsqrt = call <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float> %0)
;;  %result = fmul <4 x float> %rsqrt, %0
  ret <4 x float> %result
}

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)

define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone {
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %0)
  ret <4 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define i64 @__movmsk(<4 x MASK>) nounwind readnone {
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

;; $1: scalar type
;; $2: vector reduce function (2 x <2 x vec> -> <2 x vec>)
;; $3 scalar reduce function

define(`neon_reduce', `
  %v0 = shufflevector <4 x $1> %0, <4 x $1> undef, <2 x i32> <i32 0, i32 1>
  %v1 = shufflevector <4 x $1> %0, <4 x $1> undef, <2 x i32> <i32 2, i32 3>
  %vh = call <2 x $1> $2(<2 x $1> %v0, <2 x $1> %v1)
  %vh0 = extractelement <2 x $1> %vh, i32 0
  %vh1 = extractelement <2 x $1> %vh, i32 1
  %r = call $1$3 ($1 %vh0, $1 %vh1)
  ret $1 %r
')

declare <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @add_f32(float, float) {
  %r = fadd float %0, %1
  ret float %r
}

define float @__reduce_add_float(<4 x float>) nounwind readnone {
  neon_reduce(float, @llvm.arm.neon.vpadd.v2f32, @add_f32)
}

declare <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @min_f32(float, float) {
  %cmp = fcmp olt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_min_float(<4 x float>) nounwind readnone {
  neon_reduce(float, @llvm.arm.neon.vpmins.v2f32, @min_f32)
}

declare <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone

define internal float @max_f32(float, float) {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__reduce_max_float(<4 x float>) nounwind readnone {
  neon_reduce(float, @llvm.arm.neon.vpmaxs.v2f32, @max_f32)
}

declare <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8>) nounwind readnone

define i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone {
  %v8 = shufflevector <4 x i8> %0, <4 x i8> zeroinitializer,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %a16 = call <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8> %v8)
  %a32 = call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %a16)
  %a0 = extractelement <2 x i32> %a32, i32 0
  %a1 = extractelement <2 x i32> %a32, i32 1
  %r = add i32 %a0, %a1
  %r16 = trunc i32 %r to i16
  ret i16 %r16
}

declare <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16>) nounwind readnone

define i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone {
  %a32 = call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %0)
  %a0 = extractelement <2 x i32> %a32, i32 0
  %a1 = extractelement <2 x i32> %a32, i32 1
  %r = add i32 %a0, %a1
  ret i32 %r
}

declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) nounwind readnone

define i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone {
  %a64 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %0)
  %a0 = extractelement <2 x i64> %a64, i32 0
  %a1 = extractelement <2 x i64> %a64, i32 1
  %r = add i64 %a0, %a1
  ret i64 %r
}

declare <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_si32(i32, i32) {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_int32(<4 x i32>) nounwind readnone {
  neon_reduce(i32, @llvm.arm.neon.vpmins.v2i32, @min_si32)
}

declare <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_si32(i32, i32) {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_int32(<4 x i32>) nounwind readnone {
  neon_reduce(i32, @llvm.arm.neon.vpmaxs.v2i32, @max_si32)
}

declare <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @min_ui32(i32, i32) {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_min_uint32(<4 x i32>) nounwind readnone {
  neon_reduce(i32, @llvm.arm.neon.vpmins.v2i32, @min_ui32)
}

declare <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define internal i32 @max_ui32(i32, i32) {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__reduce_max_uint32(<4 x i32>) nounwind readnone {
  neon_reduce(i32, @llvm.arm.neon.vpmaxs.v2i32, @max_ui32)
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16

declare <4 x i8> @llvm.arm.neon.vrhaddu.v4i8(<4 x i8>, <4 x i8>) nounwind readnone

define <4 x i8> @__avg_up_uint8(<4 x i8>, <4 x i8>) nounwind readnone {
  %r = call <4 x i8> @llvm.arm.neon.vrhaddu.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %r
}

declare <4 x i8> @llvm.arm.neon.vrhadds.v4i8(<4 x i8>, <4 x i8>) nounwind readnone

define <4 x i8> @__avg_up_int8(<4 x i8>, <4 x i8>) nounwind readnone {
  %r = call <4 x i8> @llvm.arm.neon.vrhadds.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %r
}

declare <4 x i8> @llvm.arm.neon.vhaddu.v4i8(<4 x i8>, <4 x i8>) nounwind readnone

define <4 x i8> @__avg_down_uint8(<4 x i8>, <4 x i8>) nounwind readnone {
  %r = call <4 x i8> @llvm.arm.neon.vhaddu.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %r
}

declare <4 x i8> @llvm.arm.neon.vhadds.v4i8(<4 x i8>, <4 x i8>) nounwind readnone

define <4 x i8> @__avg_down_int8(<4 x i8>, <4 x i8>) nounwind readnone {
  %r = call <4 x i8> @llvm.arm.neon.vhadds.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %r
}

declare <4 x i16> @llvm.arm.neon.vrhaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_up_uint16(<4 x i16>, <4 x i16>) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vrhaddu.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @llvm.arm.neon.vrhadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_up_int16(<4 x i16>, <4 x i16>) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vrhadds.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_down_uint16(<4 x i16>, <4 x i16>) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

declare <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define <4 x i16> @__avg_down_int16(<4 x i16>, <4 x i16>) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()
saturation_arithmetic()
