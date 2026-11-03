; nneg constrains the result of the add, not its operands. For x = INT32_MIN,
; add i32 x, INT32_MIN wraps to zero, so the original offset is defined and
; zero. Splitting it into sext(x) + sext(INT32_MIN) gives -2^32.
; Unlike address-gap-minus-const.ispc, this exercises zext nneg, not sext.
; Accept either the original 64-bit %off or the 32-bit %sum itself: nneg
; bounds the sum below 2^31. Tie the call operand to those values so dropping
; the constant and passing %x cannot satisfy the check.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED
; XFAIL: *

declare <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x i8> @__pseudo_gather_factored_base_offsets32_i8(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x i8> @__masked_load_i8(ptr, <4 x i32>)

; CHECK-LABEL: define <4 x i8> @nneg_sum(
; CHECK: %sum = add <4 x i32> %x,
; CHECK: %off = zext nneg <4 x i32> %sum to <4 x i64>
; CHECK-NOT: sext <4 x i32> %x
; CHECK-NOT: sext <4 x i32> {{splat \(i32 -2147483648\)|<i32 -2147483648}}
; CHECK: call <4 x i8> @__pseudo_gather_factored_base_offsets{{32_i8\(ptr %base, <4 x i32> %sum, i32 1, <4 x i32>|64_i8\(ptr %base, <4 x i64> %off, i32 1, <4 x i64>}} zeroinitializer, <4 x i32> %mask)
; CHECK-NOT: sext <4 x i32> %x
; CHECK: ret
define <4 x i8> @nneg_sum(ptr %base, <4 x i32> %x, <4 x i32> %mask) {
  %sum = add <4 x i32> %x, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %off = zext nneg <4 x i32> %sum to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  ret <4 x i8> %v
}
