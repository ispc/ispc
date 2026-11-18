; The Wrapped 32-bit forms, whose offset is the i32 sum offsets * scale +
; delta, sign extended once. A gather that becomes a load computes that sum at
; 32 bits. Wrapped and Extended gathers are coalesced together, and a Wrapped
; form is lowered to the builtin of the Extended form.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=improve-memory-ops %s -o - | FileCheck %s --check-prefix=LOAD
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=gather-coalesce %s -o - | FileCheck %s --check-prefix=COAL
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=replace-pseudo-memory-ops %s -o - | FileCheck %s --check-prefix=LOWER
; REQUIRES: X86_ENABLED

declare <4 x i8> @__pseudo_gather_wrapped_factored_base_offsets32_i8(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_wrapped_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_wrapped_factored_base_offsets32_i32(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x i8> @__gather_factored_base_offsets32_i8(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__gather_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x i32> @__gather_factored_base_offsets32_i32(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)

; LOAD-LABEL: define <4 x i8> @load_wrapped(
; LOAD-NOT: sext
; LOAD: [[S:%.*]] = mul i32 1, %x
; LOAD: [[O:%.*]] = add i32 [[S]], -2147483648
; LOAD: getelementptr i8, ptr %base, i32 [[O]]
; LOAD: ret
define <4 x i8> @load_wrapped(ptr %base, i32 %x) {
  %xi = insertelement <4 x i32> poison, i32 %x, i32 0
  %xs = shufflevector <4 x i32> %xi, <4 x i32> poison, <4 x i32> zeroinitializer
  %v = call <4 x i8> @__pseudo_gather_wrapped_factored_base_offsets32_i8(ptr %base, <4 x i32> %xs, i32 1, <4 x i32> splat (i32 -2147483648), <4 x i32> splat (i32 -1))
  ret <4 x i8> %v
}

; LOAD-LABEL: define <4 x float> @load_wrapped_base_offsets(
; LOAD-NOT: sext
; LOAD: [[O:%.*]] = mul i32 4, %x
; LOAD: getelementptr i8, ptr %base, i32 [[O]]
; LOAD: ret
define <4 x float> @load_wrapped_base_offsets(ptr %base, i32 %x) {
  %xi = insertelement <4 x i32> poison, i32 %x, i32 0
  %xs = shufflevector <4 x i32> %xi, <4 x i32> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather_wrapped_base_offsets32_float(ptr %base, i32 4, <4 x i32> %xs, <4 x i32> splat (i32 -1))
  ret <4 x float> %v
}

; COAL-LABEL: define <4 x i32> @coalesce_mixed(
; COAL-NOT: call
; COAL: load i32
; COAL-NOT: call
; COAL: ret
define <4 x i32> @coalesce_mixed(ptr %base, i32 %x) {
  %xi = insertelement <4 x i32> poison, i32 %x, i32 0
  %xs = shufflevector <4 x i32> %xi, <4 x i32> poison, <4 x i32> zeroinitializer
  %a = call <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr %base, <4 x i32> %xs, i32 4, <4 x i32> <i32 0, i32 8, i32 16, i32 24>, <4 x i32> splat (i32 -1))
  %b = call <4 x i32> @__pseudo_gather_wrapped_factored_base_offsets32_i32(ptr %base, <4 x i32> %xs, i32 4, <4 x i32> <i32 4, i32 12, i32 20, i32 28>, <4 x i32> splat (i32 -1))
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}

; LOWER-LABEL: define void @lower(
; LOWER: call <4 x i8> @__gather_factored_base_offsets32_i8(ptr %p, <4 x i32> %o, i32 1, <4 x i32> %o, <4 x i32> %m)
; LOWER: call <4 x float> @__gather_base_offsets32_float(ptr %p, i32 4, <4 x i32> %o, <4 x i32> %m)
; LOWER: ret
define void @lower(ptr %p, <4 x i32> %o, <4 x i32> %m, ptr %out) {
  %a = call <4 x i8> @__pseudo_gather_wrapped_factored_base_offsets32_i8(ptr %p, <4 x i32> %o, i32 1, <4 x i32> %o, <4 x i32> %m)
  %b = call <4 x float> @__pseudo_gather_wrapped_base_offsets32_float(ptr %p, i32 4, <4 x i32> %o, <4 x i32> %m)
  store <4 x i8> %a, ptr %out
  store <4 x float> %b, ptr %out
  ret void
}
