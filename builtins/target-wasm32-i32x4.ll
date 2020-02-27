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

;; Define i1x4 mask

define(`WIDTH',`4')
;; FIXME: Workaround for "BUILD_OS should be defined to either UNIX or WINDOWS" error
define(`BUILD_OS',`UNIX')
define(`RUNTIME',`32')
define(`MASK',`i1')

include(`util.m4')

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()
define_shuffles()
aossoa()
ctlztz()

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture %ptr, <WIDTH x i8> %new,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i8> ')  %ptr
  %result = select <WIDTH x i1> %mask, <WIDTH x i8> %new, <WIDTH x i8> %old
  store <WIDTH x i8> %result, <WIDTH x i8> * %ptr
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture %ptr, <WIDTH x i16> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i16> ')  %ptr
  %result = select <WIDTH x i1> %mask, <WIDTH x i16> %new, <WIDTH x i16> %old
  store <WIDTH x i16> %result, <WIDTH x i16> * %ptr
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture %ptr, <WIDTH x i32> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i32> ')  %ptr
  %result = select <WIDTH x i1> %mask, <WIDTH x i32> %new, <WIDTH x i32> %old
  store <WIDTH x i32> %result, <WIDTH x i32> * %ptr
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture %ptr,
                            <WIDTH x i64> %new, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i64> ')  %ptr
  %result = select <WIDTH x i1> %mask, <WIDTH x i64> %new, <WIDTH x i64> %old
  store <WIDTH x i64> %result, <WIDTH x i64> * %ptr
  ret void
}

define i64 @__movmsk(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %mask_i4 = bitcast <WIDTH x MASK> %mask to i4
  %res = zext i4 %mask_i4 to i64
  ret i64 %res
}

define i1 @__any(<4 x MASK> %mask) nounwind readnone alwaysinline {
  entry:
    %mask_i4 = bitcast <WIDTH x MASK> %mask to i4
    %cmp = icmp ne i4 %mask_i4, 0
    ret i1 %cmp
}

define i1 @__all(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  entry:
    %mask_i4 = bitcast <WIDTH x MASK> %mask to i4
    %cmp = icmp eq i4 %mask_i4, 15
    ret i1 %cmp
}

declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)

define i1 @__none(<WIDTH x MASK> %mask) nounwind readnone alwaysinline {
  %any = call i1 @__any(<WIDTH x MASK> %mask)
  %none = icmp eq i1 %any, 0
  ret i1 %none
}

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

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

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

packed_load_and_store(4)
define_prefetches()