; Copyright (c) 2026, Intel Corporation
; SPDX-License-Identifier: BSD-3-Clause
;
; RUN: %{ispc-opt} --target=avx512icl-x64 --passes=peephole %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

; Every upper lane is poison in at least one operand. Lane 1 is also poison
; and must still be compared as such. Freeze keeps the entire result live.
; CHECK-LABEL: @mixed_poison_tail(
; CHECK: [[LEFT:%[^ ]+]] = shufflevector <16 x i32> %left, <16 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: [[RIGHT:%[^ ]+]] = shufflevector <16 x i32> %right, <16 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: [[CMP:%[^ ]+]] = icmp sgt <4 x i32> [[LEFT]], [[RIGHT]]
; CHECK-NEXT: [[MASK:%[^ ]+]] = sext <4 x i1> [[CMP]] to <4 x i32>
; CHECK-NEXT: [[PAD:%[^ ]+]] = shufflevector <4 x i32> [[MASK]], <4 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3{{(, i32 poison){12}>}}
; CHECK-NEXT: freeze <16 x i32> [[PAD]]
define <16 x i32> @mixed_poison_tail(<4 x i32> %a, <4 x i32> %b) {
  %left = shufflevector <4 x i32> %a, <4 x i32> poison, <16 x i32> <i32 0, i32 poison, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %right = shufflevector <4 x i32> %b, <4 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %cmp = icmp sgt <16 x i32> %left, %right
  %wide = sext <16 x i1> %cmp to <16 x i32>
  %frozen = freeze <16 x i32> %wide
  ret <16 x i32> %frozen
}

; Lane 10 might yield a defined result. Keep the full-width comparison.
; CHECK-LABEL: @defined_tail_lane(
; CHECK: icmp sgt <16 x i32>
; CHECK-NEXT: sext <16 x i1>
; CHECK-NEXT: ret <16 x i32> %wide
define <16 x i32> @defined_tail_lane(<4 x i32> %a, <16 x i32> %b) {
  %left = shufflevector <4 x i32> %a, <4 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %cmp = icmp sgt <16 x i32> %left, %b
  %wide = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %wide
}

; Undef operands do not guarantee a poison result.
; CHECK-LABEL: @undef_tail(
; CHECK: icmp sgt <16 x i32>
; CHECK-NEXT: sext <16 x i1>
; CHECK-NEXT: ret <16 x i32> %wide
define <16 x i32> @undef_tail(i32 %a, <16 x i32> %b) {
  %left = insertelement <16 x i32> undef, i32 %a, i32 0
  %cmp = icmp sgt <16 x i32> %left, %b
  %wide = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %wide
}

; Narrowing would leave the full-width comparison live for the other use.
; CHECK-LABEL: @shared_compare(
; CHECK: icmp sgt <16 x i32>
; CHECK-NEXT: sext <16 x i1>
; CHECK-NEXT: store <16 x i1> %cmp
define <16 x i32> @shared_compare(<4 x i32> %a, <16 x i32> %b, ptr %out) {
  %left = shufflevector <4 x i32> %a, <4 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %cmp = icmp sgt <16 x i32> %left, %b
  %wide = sext <16 x i1> %cmp to <16 x i32>
  store <16 x i1> %cmp, ptr %out
  ret <16 x i32> %wide
}
