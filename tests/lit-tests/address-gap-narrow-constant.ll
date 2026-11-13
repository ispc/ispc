; An unfolded constant displacement of +2^31 is not safe to truncate to i32.
; The factored narrowing path folds it and keeps the 64-bit form. 3985-2.ispc already
; covers a folded constant; gather-index-width.ispc's big_const is truncated
; by the frontend before ImproveMemoryOps, so it does not exercise this case.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__masked_load_float(ptr, <4 x i32>)

; CHECK-LABEL: define <4 x float> @unfolded_displacement(
; CHECK-NOT: call {{.*}}@__pseudo_gather_factored_base_offsets32_float
; CHECK: call <4 x float> @__pseudo_gather_factored_base_offsets64_float(
; CHECK-NOT: call {{.*}}@__pseudo_gather_factored_base_offsets32_float
; CHECK: ret
define <4 x float> @unfolded_displacement(ptr %base, <4 x i32> %x, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %sx = sext <4 x i32> %x to <4 x i64>
  %c = add <4 x i64> <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>, <i64 1, i64 1, i64 1, i64 1>
  %off = add <4 x i64> %sx, %c
  %addresses = add <4 x i64> %bases, %off
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}
