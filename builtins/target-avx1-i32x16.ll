;;  Copyright (c) 2010-2024, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`ISA',`AVX1')
include(`target-avx-common-16.ll')


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; svml

include(`svml.m4')
svml(ISA)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pminsd, %0, %1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pmaxsd, %0, %1)
  ret <16 x i32> %ret
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pminud, %0, %1)
  ret <16 x i32> %ret
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  binary4to16(ret, i32, @llvm.x86.sse41.pmaxud, %0, %1)
  ret <16 x i32> %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(half)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dot product
