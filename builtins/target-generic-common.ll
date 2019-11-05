;;  Copyright (c) 2010-2019, Intel Corporation
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

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128-v16:16:16-v32:32:32-v4:128:128";

define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

declare <WIDTH x float> @__smear_float(float) nounwind readnone
declare <WIDTH x double> @__smear_double(double) nounwind readnone
declare <WIDTH x i8> @__smear_i8(i8) nounwind readnone
declare <WIDTH x i16> @__smear_i16(i16) nounwind readnone
declare <WIDTH x i32> @__smear_i32(i32) nounwind readnone
declare <WIDTH x i64> @__smear_i64(i64) nounwind readnone

declare <WIDTH x float> @__setzero_float() nounwind readnone
declare <WIDTH x double> @__setzero_double() nounwind readnone
declare <WIDTH x i8> @__setzero_i8() nounwind readnone
declare <WIDTH x i16> @__setzero_i16() nounwind readnone
declare <WIDTH x i32> @__setzero_i32() nounwind readnone
declare <WIDTH x i64> @__setzero_i64() nounwind readnone

declare <WIDTH x float> @__undef_float() nounwind readnone
declare <WIDTH x double> @__undef_double() nounwind readnone
declare <WIDTH x i8> @__undef_i8() nounwind readnone
declare <WIDTH x i16> @__undef_i16() nounwind readnone
declare <WIDTH x i32> @__undef_i32() nounwind readnone
declare <WIDTH x i64> @__undef_i64() nounwind readnone

declare <WIDTH x float> @__broadcast_float(<WIDTH x float>, i32) nounwind readnone
declare <WIDTH x double> @__broadcast_double(<WIDTH x double>, i32) nounwind readnone
declare <WIDTH x i8> @__broadcast_i8(<WIDTH x i8>, i32) nounwind readnone
declare <WIDTH x i16> @__broadcast_i16(<WIDTH x i16>, i32) nounwind readnone
declare <WIDTH x i32> @__broadcast_i32(<WIDTH x i32>, i32) nounwind readnone
declare <WIDTH x i64> @__broadcast_i64(<WIDTH x i64>, i32) nounwind readnone

declare <WIDTH x i8> @__rotate_i8(<WIDTH x i8>, i32) nounwind readnone
declare <WIDTH x i16> @__rotate_i16(<WIDTH x i16>, i32) nounwind readnone
declare <WIDTH x float> @__rotate_float(<WIDTH x float>, i32) nounwind readnone
declare <WIDTH x i32> @__rotate_i32(<WIDTH x i32>, i32) nounwind readnone
declare <WIDTH x double> @__rotate_double(<WIDTH x double>, i32) nounwind readnone
declare <WIDTH x i64> @__rotate_i64(<WIDTH x i64>, i32) nounwind readnone

declare <WIDTH x i8> @__shift_i8(<WIDTH x i8>, i32) nounwind readnone
declare <WIDTH x i16> @__shift_i16(<WIDTH x i16>, i32) nounwind readnone
declare <WIDTH x float> @__shift_float(<WIDTH x float>, i32) nounwind readnone
declare <WIDTH x i32> @__shift_i32(<WIDTH x i32>, i32) nounwind readnone
declare <WIDTH x double> @__shift_double(<WIDTH x double>, i32) nounwind readnone
declare <WIDTH x i64> @__shift_i64(<WIDTH x i64>, i32) nounwind readnone

declare <WIDTH x i8> @__shuffle_i8(<WIDTH x i8>, <WIDTH x i32>) nounwind readnone
declare <WIDTH x i8> @__shuffle2_i8(<WIDTH x i8>, <WIDTH x i8>,
                                    <WIDTH x i32>) nounwind readnone
declare <WIDTH x i16> @__shuffle_i16(<WIDTH x i16>, <WIDTH x i32>) nounwind readnone
declare <WIDTH x i16> @__shuffle2_i16(<WIDTH x i16>, <WIDTH x i16>,
                                      <WIDTH x i32>) nounwind readnone
declare <WIDTH x float> @__shuffle_float(<WIDTH x float>,
                                         <WIDTH x i32>) nounwind readnone
declare <WIDTH x float> @__shuffle2_float(<WIDTH x float>, <WIDTH x float>,
                                          <WIDTH x i32>) nounwind readnone
declare <WIDTH x i32> @__shuffle_i32(<WIDTH x i32>,
                                     <WIDTH x i32>) nounwind readnone
declare <WIDTH x i32> @__shuffle2_i32(<WIDTH x i32>, <WIDTH x i32>,
                                      <WIDTH x i32>) nounwind readnone
declare <WIDTH x double> @__shuffle_double(<WIDTH x double>,
                                           <WIDTH x i32>) nounwind readnone
declare <WIDTH x double> @__shuffle2_double(<WIDTH x double>,
                                            <WIDTH x double>, <WIDTH x i32>) nounwind readnone
declare <WIDTH x i64> @__shuffle_i64(<WIDTH x i64>,
                                     <WIDTH x i32>) nounwind readnone
declare <WIDTH x i64> @__shuffle2_i64(<WIDTH x i64>, <WIDTH x i64>,
                                      <WIDTH x i32>) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

declare void @__soa_to_aos3_float(<WIDTH x float> %v0, <WIDTH x float> %v1,
                                  <WIDTH x float> %v2, float * noalias %p) nounwind
declare void @__aos_to_soa3_float(float * noalias %p, <WIDTH x float> * %out0,
                                  <WIDTH x float> * %out1, <WIDTH x float> * %out2) nounwind
declare void @__soa_to_aos4_float(<WIDTH x float> %v0, <WIDTH x float> %v1,
                                  <WIDTH x float> %v2, <WIDTH x float> %v3,
                                  float * noalias %p) nounwind
declare void @__aos_to_soa4_float(float * noalias %p, <WIDTH x float> * noalias %out0,
                                  <WIDTH x float> * noalias %out1,
                                  <WIDTH x float> * noalias %out2,
                                  <WIDTH x float> * noalias %out3) nounwind
declare void @__soa_to_aos3_double(<WIDTH x double> %v0, <WIDTH x double> %v1,
                                  <WIDTH x double> %v2, double * noalias %p) nounwind
declare void @__aos_to_soa3_double(double * noalias %p, <WIDTH x double> * noalias %out0,
                                  <WIDTH x double> * noalias %out1,
                                  <WIDTH x double> * noalias %out2) nounwind
declare void @__soa_to_aos4_double(<WIDTH x double> %v0, <WIDTH x double> %v1,
                                  <WIDTH x double> %v2, <WIDTH x double> %v3,
                                  double * noalias %p) nounwind
declare void @__aos_to_soa4_double(double * noalias %p, <WIDTH x double> * noalias %out0,
                                  <WIDTH x double> * noalias %out1,
                                  <WIDTH x double> * noalias %out2,
                                  <WIDTH x double> * noalias %out3) nounwind

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

declare void @__fastmath() nounwind 

;; round/floor/ceil

declare float @__round_uniform_float(float) nounwind readnone 
declare float @__floor_uniform_float(float) nounwind readnone 
declare float @__ceil_uniform_float(float) nounwind readnone 

declare double @__round_uniform_double(double) nounwind readnone 
declare double @__floor_uniform_double(double) nounwind readnone 
declare double @__ceil_uniform_double(double) nounwind readnone 

declare <WIDTH x float> @__round_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__floor_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__ceil_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readnone 

;; min/max

declare float @__max_uniform_float(float, float) nounwind readnone 
declare float @__min_uniform_float(float, float) nounwind readnone 
declare i32 @__min_uniform_int32(i32, i32) nounwind readnone 
declare i32 @__max_uniform_int32(i32, i32) nounwind readnone 
declare i32 @__min_uniform_uint32(i32, i32) nounwind readnone 
declare i32 @__max_uniform_uint32(i32, i32) nounwind readnone 
declare i64 @__min_uniform_int64(i64, i64) nounwind readnone 
declare i64 @__max_uniform_int64(i64, i64) nounwind readnone 
declare i64 @__min_uniform_uint64(i64, i64) nounwind readnone 
declare i64 @__max_uniform_uint64(i64, i64) nounwind readnone 
declare double @__min_uniform_double(double, double) nounwind readnone 
declare double @__max_uniform_double(double, double) nounwind readnone 

declare <WIDTH x float> @__max_varying_float(<WIDTH x float>,
                                             <WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__min_varying_float(<WIDTH x float>,
                                             <WIDTH x float>) nounwind readnone 
declare <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
declare <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
declare <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
declare <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
declare <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                               <WIDTH x double>) nounwind readnone
declare <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                               <WIDTH x double>) nounwind readnone 

;; sqrt/rsqrt/rcp

declare float @__rsqrt_uniform_float(float) nounwind readnone 
declare float @__rcp_uniform_float(float) nounwind readnone 
declare float @__sqrt_uniform_float(float) nounwind readnone 
declare float @__rcp_fast_uniform_float(float) nounwind readnone 
declare float @__rsqrt_fast_uniform_float(float) nounwind readnone 
declare <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__rcp_fast_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__rsqrt_fast_varying_float(<WIDTH x float>) nounwind readnone

declare <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone 

declare double @__sqrt_uniform_double(double) nounwind readnone
declare <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone

;; bit ops

declare i32 @__popcnt_int32(i32) nounwind readnone
declare i64 @__popcnt_int64(i64) nounwind readnone 

declare i32 @__count_trailing_zeros_i32(i32) nounwind readnone
declare i64 @__count_trailing_zeros_i64(i64) nounwind readnone
declare i32 @__count_leading_zeros_i32(i32) nounwind readnone
declare i64 @__count_leading_zeros_i64(i64) nounwind readnone

; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

declare i64 @__movmsk(<WIDTH x i1>) nounwind readnone 
declare i1 @__any(<WIDTH x i1>) nounwind readnone 
declare i1 @__all(<WIDTH x i1>) nounwind readnone 
declare i1 @__none(<WIDTH x i1>) nounwind readnone 

declare i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone
declare i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone

declare float @__reduce_add_float(<WIDTH x float>) nounwind readnone
declare float @__reduce_min_float(<WIDTH x float>) nounwind readnone 
declare float @__reduce_max_float(<WIDTH x float>) nounwind readnone 

declare i64 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone
declare i32 @__reduce_min_int32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_max_int32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_min_uint32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_max_uint32(<WIDTH x i32>) nounwind readnone 

declare double @__reduce_add_double(<WIDTH x double>) nounwind readnone 
declare double @__reduce_min_double(<WIDTH x double>) nounwind readnone 
declare double @__reduce_max_double(<WIDTH x double>) nounwind readnone 

declare i64 @__reduce_add_int64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_min_int64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_max_int64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_min_uint64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_max_uint64(<WIDTH x i64>) nounwind readnone 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts


declare <WIDTH x i8> @__masked_load_i8(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i16> @__masked_load_i16(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i32> @__masked_load_i32(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x float> @__masked_load_float(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i64> @__masked_load_i64(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x double> @__masked_load_double(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly

declare void @__masked_store_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                <WIDTH x i1>) nounwind 
declare void @__masked_store_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                 <WIDTH x i1>) nounwind 
declare void @__masked_store_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                 <WIDTH x i1>) nounwind 
declare void @__masked_store_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                   <WIDTH x i1>) nounwind 
declare void @__masked_store_i64(<WIDTH x i64>* nocapture, <WIDTH x i64>,
                                 <WIDTH x i1> %mask) nounwind 
declare void @__masked_store_double(<WIDTH x double>* nocapture, <WIDTH x double>,
                                    <WIDTH x i1> %mask) nounwind 


define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i8> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i16> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i32> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i32> %1, <WIDTH x i32> %v
  store <WIDTH x i32> %v1, <WIDTH x i32> * %0
  ret void
}

define void @__masked_store_blend_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                        <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x float> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x float> %1, <WIDTH x float> %v
  store <WIDTH x float> %v1, <WIDTH x float> * %0
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i64> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}

define void @__masked_store_blend_double(<WIDTH x double>* nocapture,
                            <WIDTH x double>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x double> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x double> %1, <WIDTH x double> %v
  store <WIDTH x double> %v1, <WIDTH x double> * %0
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

define(`gather_scatter', `
declare <WIDTH x $1> @__gather_base_offsets32_$1(i8 * nocapture, i32, <WIDTH x i32>,
                                                 <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather_base_offsets64_$1(i8 * nocapture, i32, <WIDTH x i64>,
                                                  <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather32_$1(<WIDTH x i32>, 
                                    <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather64_$1(<WIDTH x i64>, 
                                    <WIDTH x i1>) nounwind readonly 

declare void @__scatter_base_offsets32_$1(i8* nocapture, i32, <WIDTH x i32>,
                                          <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter_base_offsets64_$1(i8* nocapture, i32, <WIDTH x i64>,
                                          <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter32_$1(<WIDTH x i32>, <WIDTH x $1>,
                             <WIDTH x i1>) nounwind 
declare void @__scatter64_$1(<WIDTH x i64>, <WIDTH x $1>,
                              <WIDTH x i1>) nounwind 
')

gather_scatter(i8)
gather_scatter(i16)
gather_scatter(i32)
gather_scatter(float)
gather_scatter(i64)
gather_scatter(double)

declare i32 @__packed_load_active(i32 * nocapture, <WIDTH x i32> * nocapture,
                                  <WIDTH x i1>) nounwind
declare i32 @__packed_store_active(i32 * nocapture, <WIDTH x i32> %vals,
                                   <WIDTH x i1>) nounwind
declare i32 @__packed_store_active2(i32 * nocapture, <WIDTH x i32> %vals,
                                   <WIDTH x i1>) nounwind


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

declare void @__prefetch_read_uniform_1(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_2(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_3(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_nt(i8 * nocapture) nounwind 

declare void @__prefetch_read_varying_1(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_1_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_2(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_2_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_3(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_3_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_nt(<WIDTH x i64> %addr, <WIDTH x MASK> %mask) nounwind
declare void @__prefetch_read_varying_nt_native(i8 * %base, i32 %scale, <WIDTH x i32> %offsets, <WIDTH x MASK> %mask) nounwind
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reciprocals in double precision, if supported

rsqrtd_decl()
rcpd_decl()

transcendetals_decl()
trigonometry_decl()
