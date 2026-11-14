; zext nneg(mul32(x, 4)) does not permit moving the scale outside the cast.
; For x = -1073741823, the unflagged multiply wraps to 4: its result is
; nonnegative, but x is not. The old scale extraction copies nneg to the
; extension of x, introducing poison for a defined active-lane address.
; This is independent of constant splitting (address-gap-nneg-split.ll).
;
; RUN: %{ispc-opt} --target=avx2-i32x8 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED
declare <8 x float> @__pseudo_gather64_float(<8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <8 x i32>, <8 x i32>)
declare <8 x float> @__masked_load_float(ptr, <8 x i32>)

; The index may be narrowed after the multiply because its result is nneg;
; it must still include the wrapping multiply, with hardware scale 1.
; CHECK-LABEL: define <8 x float> @nneg_product(
; CHECK-NOT: zext nneg <8 x i32> %x to <8 x i64>
; CHECK: call <8 x float> @__pseudo_gather_base_offsets{{32|64}}_float(ptr {{[^,]+}}, i32 1,
; CHECK-NOT: zext nneg <8 x i32> %x to <8 x i64>
; CHECK: ret
define <8 x float> @nneg_product(ptr %base, <8 x i32> %x, <8 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <8 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
  %scaled = mul <8 x i32> %x, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %off = zext nneg <8 x i32> %scaled to <8 x i64>
  %addresses = add <8 x i64> %bases, %off
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %v
}
