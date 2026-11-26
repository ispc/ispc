; The scalar load of a gather whose lanes all read one address is made only if
; a lane is on, unless the mask is known to be all on: it is a masked load of
; one element whose mask is any lane on.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

declare <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)

; CHECK-LABEL: define <4 x i32> @mixed_mask(
; CHECK: %[[ON:.*]] = icmp slt <4 x i32> %mask, zeroinitializer
; CHECK: %[[ANY:.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %[[ON]])
; CHECK: %[[M:.*]] = bitcast i1 %[[ANY]] to <1 x i1>
; CHECK: call <1 x i32> @llvm.masked.load.v1i32.p0(ptr {{.*}}%ptr, {{(i32 4, )?}}<1 x i1> %[[M]], <1 x i32> poison)
; CHECK-NOT: @__pseudo_gather
; CHECK: ret
define <4 x i32> @mixed_mask(ptr %base, i64 %u, <4 x i32> %mask) {
  %s = insertelement <4 x i64> poison, i64 %u, i32 0
  %offsets = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %offsets, i32 4, <4 x i64> zeroinitializer, <4 x i32> %mask)
  ret <4 x i32> %v
}

; CHECK-LABEL: define <4 x i32> @all_on(
; CHECK-NOT: @llvm.masked.load
; CHECK: load i32, ptr
; CHECK-NOT: @__pseudo_gather
; CHECK: ret
define <4 x i32> @all_on(ptr %base, i64 %u) {
  %s = insertelement <4 x i64> poison, i64 %u, i32 0
  %offsets = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %offsets, i32 4, <4 x i64> zeroinitializer, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
  ret <4 x i32> %v
}
