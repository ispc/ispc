;;
;; target-neon.ll
;;
;;  Copyright(c) 2012-2013 Matt Pharr
;;  Copyright(c) 2013 Google, Inc.
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

target datalayout = "e-p:32:32:32-S32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f16:16:16-f32:32:32-f64:32:64-f128:128:128-v64:32:64-v128:32:128-a0:0:64-n32"

define(`WIDTH',`4')

define(`MASK',`i32')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()
define_shuffles()
aossoa()
ctlztz()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16>) nounwind readnone

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %v1 = bitcast i16 %v to <1 x i16>
  %vec = shufflevector <1 x i16> %v1, <1 x i16> undef, 
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %vec)
  %r = extractelement <4 x float> %h, i32 0
  ret float %r
}

define <4 x float> @__half_to_float_varying(<4 x i16> %v) nounwind readnone {
  %r = call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %v)
  ret <4 x float> %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %v1 = bitcast float %v to <1 x float>
  %vec = shufflevector <1 x float> %v1, <1 x float> undef, 
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %vec)
  %r = extractelement <4 x i16> %h, i32 0
  ret i16 %r
}


define <4 x i16> @__float_to_half_varying(<4 x float> %v) nounwind readnone {
  %r = call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %v)
  ret <4 x i16> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

define void @__fastmath() nounwind {
  ret void
}

;; round/floor/ceil

;; FIXME: grabbed these from the sse2 target, which does not have native
;; instructions for these.  Is there a better approach for NEON?

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
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

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp ogt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, -1082130432
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp olt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, 1065353216
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

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
declare double @__round_uniform_double(double) nounwind readnone 
declare double @__floor_uniform_double(double) nounwind readnone 
declare double @__ceil_uniform_double(double) nounwind readnone 

declare <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readnone 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

define float @__max_uniform_float(float, float) nounwind readnone {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__min_uniform_float(float, float) nounwind readnone {
  %cmp = fcmp ult float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define i32 @__min_uniform_int32(i32, i32) nounwind readnone {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_int32(i32, i32) nounwind readnone {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readnone {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readnone {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readnone {
  %cmp = icmp slt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_int64(i64, i64) nounwind readnone {
  %cmp = icmp sgt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readnone {
  %cmp = icmp ult i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readnone {
  %cmp = icmp ugt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define double @__min_uniform_double(double, double) nounwind readnone {
  %cmp = fcmp olt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define double @__max_uniform_double(double, double) nounwind readnone {
  %cmp = fcmp ogt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

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

define <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone {
  %m = icmp slt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone {
  %m = icmp sgt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone {
  %m = icmp ult <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone {
  %m = icmp ugt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone {
  %m = fcmp olt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

define <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone {
  %m = fcmp ogt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
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

declare float @llvm.sqrt.f32(float)

define float @__sqrt_uniform_float(float) nounwind readnone {
  %r = call float @llvm.sqrt.f32(float %0)
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

declare double @llvm.sqrt.f64(double)

define double @__sqrt_uniform_double(double) nounwind readnone {
  %r = call double @llvm.sqrt.f64(double %0)
  ret double %r
}

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)

define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone {
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %0)
  ret <4 x double> %r
}

;; bit ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define i32 @__popcnt_int32(i32) nounwind readnone {
  %v = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %v
}

define i64 @__popcnt_int64(i64) nounwind readnone {
  %v = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %v
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

define internal i32 @add_i32(i32, i32) {
  %r = add i32 %0, %1
  ret i32 %r
}

declare <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define i32 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone {
  neon_reduce(i32, @llvm.arm.neon.vpadd.v2i32, @add_i32)
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
;; unaligned loads/loads+broadcasts

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)
masked_store_float_double()

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture %ptr, <WIDTH x i8> %new,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load <WIDTH x i8> * %ptr
  %mask1 = trunc <4 x MASK> %mask to <4 x i1>
  %result = select <4 x i1> %mask1, <4 x i8> %new, <4 x i8> %old
  store <WIDTH x i8> %result, <WIDTH x i8> * %ptr
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture %ptr, <WIDTH x i16> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load <WIDTH x i16> * %ptr
  %mask1 = trunc <4 x MASK> %mask to <4 x i1>
  %result = select <4 x i1> %mask1, <4 x i16> %new, <4 x i16> %old
  store <WIDTH x i16> %result, <WIDTH x i16> * %ptr
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture %ptr, <WIDTH x i32> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load <WIDTH x i32> * %ptr
  %mask1 = trunc <4 x MASK> %mask to <4 x i1>
  %result = select <4 x i1> %mask1, <4 x i32> %new, <4 x i32> %old
  store <WIDTH x i32> %result, <WIDTH x i32> * %ptr
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture %ptr,
                            <WIDTH x i64> %new, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load <WIDTH x i64> * %ptr
  %mask1 = trunc <4 x MASK> %mask to <4 x i1>
  %result = select <4 x i1> %mask1, <4 x i64> %new, <4 x i64> %old
  store <WIDTH x i64> %result, <WIDTH x i64> * %ptr
  ret void
}

;; yuck.  We need declarations of these, even though we shouldnt ever
;; actually generate calls to them for the NEON target...

declare <WIDTH x float> @__svml_sin(<WIDTH x float>)
declare <WIDTH x float> @__svml_cos(<WIDTH x float>)
declare void @__svml_sincos(<WIDTH x float>, <WIDTH x float> *, <WIDTH x float> *)
declare <WIDTH x float> @__svml_tan(<WIDTH x float>)
declare <WIDTH x float> @__svml_atan(<WIDTH x float>)
declare <WIDTH x float> @__svml_atan2(<WIDTH x float>, <WIDTH x float>)
declare <WIDTH x float> @__svml_exp(<WIDTH x float>)
declare <WIDTH x float> @__svml_log(<WIDTH x float>)
declare <WIDTH x float> @__svml_pow(<WIDTH x float>, <WIDTH x float>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

packed_load_and_store(4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

define_prefetches()
