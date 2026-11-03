; Step 1 of #3990: unsuccessful base/offset matching must leave IR unchanged.
; SSE4 has no vector prefetch, so this operation cannot be converted. The
; current matcher constructs an unused offset before rejecting it.
; Unlike codegen pins, this checks before dead-code cleanup can hide mutation.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED
; XFAIL: *

declare void @__pseudo_prefetch_read_varying_1(<4 x i64>, <4 x i32>)

; CHECK-LABEL: define void @prefetch_without_native_support(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: call void @__pseudo_prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> %mask)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @prefetch_without_native_support(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  call void @__pseudo_prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> %mask)
  ret void
}
