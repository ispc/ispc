;;  Copyright (c) 2010-2011, Intel Corporation
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

define(`MASK',`i1')
include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

declare <WIDTH x float> @__smear_float(float) nounwind readnone
declare <WIDTH x double> @__smear_double(double) nounwind readnone
declare <WIDTH x i8> @__smear_i8(i8) nounwind readnone
declare <WIDTH x i16> @__smear_i16(i16) nounwind readnone
declare <WIDTH x i32> @__smear_i32(i32) nounwind readnone
declare <WIDTH x i64> @__smear_i64(i64) nounwind readnone

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
declare <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone 
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

;; svml

; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

declare <WIDTH x float> @__svml_sin(<WIDTH x float>)
declare <WIDTH x float> @__svml_cos(<WIDTH x float>)
declare void @__svml_sincos(<WIDTH x float>, <WIDTH x float> *, <WIDTH x float> *)
declare <WIDTH x float> @__svml_tan(<WIDTH x float>)
declare <WIDTH x float> @__svml_atan(<WIDTH x float>)
declare <WIDTH x float> @__svml_atan2(<WIDTH x float>, <WIDTH x float>)
declare <WIDTH x float> @__svml_exp(<WIDTH x float>)
declare <WIDTH x float> @__svml_log(<WIDTH x float>)
declare <WIDTH x float> @__svml_pow(<WIDTH x float>, <WIDTH x float>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

declare i32 @__movmsk(<WIDTH x i1>) nounwind readnone 

declare float @__reduce_add_float(<WIDTH x float>) nounwind readnone
declare float @__reduce_min_float(<WIDTH x float>) nounwind readnone 
declare float @__reduce_max_float(<WIDTH x float>) nounwind readnone 

declare i32 @__reduce_add_int32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_min_int32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_max_int32(<WIDTH x i32>) nounwind readnone 

declare i32 @__reduce_add_uint32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_min_uint32(<WIDTH x i32>) nounwind readnone 
declare i32 @__reduce_max_uint32(<WIDTH x i32>) nounwind readnone 

declare double @__reduce_add_double(<WIDTH x double>) nounwind readnone 
declare double @__reduce_min_double(<WIDTH x double>) nounwind readnone 
declare double @__reduce_max_double(<WIDTH x double>) nounwind readnone 

declare i64 @__reduce_add_int64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_min_int64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_max_int64(<WIDTH x i64>) nounwind readnone 

declare i64 @__reduce_add_uint64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_min_uint64(<WIDTH x i64>) nounwind readnone 
declare i64 @__reduce_max_uint64(<WIDTH x i64>) nounwind readnone 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

load_and_broadcast(WIDTH, i8, 8)
load_and_broadcast(WIDTH, i16, 16)
load_and_broadcast(WIDTH, i32, 32)
load_and_broadcast(WIDTH, i64, 64)

declare <WIDTH x i8> @__masked_load_8(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i16> @__masked_load_16(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i32> @__masked_load_32(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly
declare <WIDTH x i64> @__masked_load_64(i8 * nocapture, <WIDTH x i1> %mask) nounwind readonly

declare void @__masked_store_8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                               <WIDTH x i1>) nounwind 
declare void @__masked_store_16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                <WIDTH x i1>) nounwind 
declare void @__masked_store_32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                <WIDTH x i1>) nounwind 
declare void @__masked_store_64(<WIDTH x i64>* nocapture, <WIDTH x i64>,
                                <WIDTH x i1> %mask) nounwind 

ifelse(LLVM_VERSION, `LLVM_3_1svn',`
define void @__masked_store_blend_8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i8> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i16> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i32> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i32> %1, <WIDTH x i32> %v
  store <WIDTH x i32> %v1, <WIDTH x i32> * %0
  ret void
}

define void @__masked_store_blend_64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i64> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}
',`
declare void @__masked_store_blend_8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind 
declare void @__masked_store_blend_16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind 
declare void @__masked_store_blend_32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind 
declare void @__masked_store_blend_64(<WIDTH x i64>* nocapture, <WIDTH x i64>,
                                      <WIDTH x i1> %mask) nounwind 
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

define(`gather_scatter', `
declare <WIDTH x $1> @__gather_base_offsets32_$1(i8 * nocapture, <WIDTH x i32>,
                        i32, <WIDTH x i32>, <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather_base_offsets64_$1(i8 * nocapture, <WIDTH x i64>,
                        i32, <WIDTH x i64>, <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather32_$1(<WIDTH x i32>, 
                                    <WIDTH x i1>) nounwind readonly 
declare <WIDTH x $1> @__gather64_$1(<WIDTH x i64>, 
                                    <WIDTH x i1>) nounwind readonly 

declare void @__scatter_base_offsets32_$1(i8* nocapture, <WIDTH x i32>,
                  i32, <WIDTH x i32>, <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter_base_offsets64_$1(i8* nocapture, <WIDTH x i64>,
                  i32, <WIDTH x i64>, <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter32_$1(<WIDTH x i32>, <WIDTH x $1>,
                             <WIDTH x i1>) nounwind 
declare void @__scatter64_$1(<WIDTH x i64>, <WIDTH x $1>,
                              <WIDTH x i1>) nounwind 
')

gather_scatter(i8)
gather_scatter(i16)
gather_scatter(i32)
gather_scatter(i64)

declare i32 @__packed_load_active(i32 * nocapture, <WIDTH x i32> * nocapture,
                                  <WIDTH x i1>) nounwind
declare i32 @__packed_store_active(i32 * nocapture, <WIDTH x i32> %vals,
                                   <WIDTH x i1>) nounwind


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

declare void @__prefetch_read_uniform_1(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_2(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_3(i8 * nocapture) nounwind 
declare void @__prefetch_read_uniform_nt(i8 * nocapture) nounwind 

