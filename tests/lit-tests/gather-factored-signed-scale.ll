; The scale of a factored gather is signed, as in the lowering
; (__gather_elt64_* in builtins/util.m4). With 64-bit offsets, zero extending
; a scale of -4 gives 2^32 - 4: a gather that becomes a load, a constant moved
; from the offsets into the delta and the common base of coalesced gathers
; must use the sign extended scale.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=improve-memory-ops %s -o - | FileCheck %s --check-prefix=LOAD
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=gather-coalesce %s -o - | FileCheck %s --check-prefix=COAL
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)

; LOAD-LABEL: define <4 x float> @load_negative_scale(
; LOAD-NOT: 4294967295
; LOAD: mul i64 -1, %x
; LOAD: load float
; LOAD-NOT: call
; LOAD: ret
define <4 x float> @load_negative_scale(ptr %base, i64 %x) {
  %xi = insertelement <4 x i64> poison, i64 %x, i32 0
  %xs = shufflevector <4 x i64> %xi, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> %xs, i32 -1, <4 x i64> splat (i64 8), <4 x i32> splat (i32 -1))
  ret <4 x float> %v
}

; LOAD-LABEL: define <4 x float> @split_negative_scale(
; LOAD-NOT: 4294967292
; LOAD: [[C:%.*]] = mul <4 x i64> splat (i64 2), splat (i64 -4)
; LOAD: [[D:%.*]] = add <4 x i64> [[C]], zeroinitializer
; LOAD: call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> %v, i32 -4, <4 x i64> [[D]],
; LOAD: ret
define <4 x float> @split_negative_scale(ptr %base, <4 x i64> %v, <4 x i32> %mask) {
  %o = add <4 x i64> %v, splat (i64 2)
  %r = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> %o, i32 -4, <4 x i64> zeroinitializer, <4 x i32> %mask)
  ret <4 x float> %r
}

; COAL-LABEL: define <4 x i32> @coalesce_negative_scale(
; COAL-NOT: zext i32 -4
; COAL: [[S:%.*]] = sext i32 -4 to i64
; COAL: mul i64 %x, [[S]]
; COAL-NOT: call
; COAL: ret
define <4 x i32> @coalesce_negative_scale(ptr %base, i64 %x) {
  %xi = insertelement <4 x i64> poison, i64 %x, i32 0
  %xs = shufflevector <4 x i64> %xi, <4 x i64> poison, <4 x i32> zeroinitializer
  %a = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %xs, i32 -4, <4 x i64> <i64 0, i64 8, i64 16, i64 24>, <4 x i32> splat (i32 -1))
  %b = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %xs, i32 -4, <4 x i64> <i64 4, i64 12, i64 20, i64 28>, <4 x i32> splat (i32 -1))
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}
