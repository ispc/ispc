; Read-only base recovery must retain the general varying-address fallback
; when no common base is established. These rejection cases previously
; depended on the local external corpus. No target builtin definitions are
; needed: the calls must remain unchanged.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare void @__pseudo_scatter64_float(<4 x i64>, <4 x float>, <4 x i32>)
declare <2 x i32> @__pseudo_gather64_i32(<2 x i64>, <2 x i32>)
declare <2 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <2 x i64>, i32, <2 x i64>, <2 x i32>)
declare <2 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr, <2 x i32>, i32, <2 x i32>, <2 x i32>)
declare void @__pseudo_scatter64_i32(<2 x i64>, <2 x i32>, <2 x i32>)
declare void @__pseudo_scatter_factored_base_offsets64_i32(ptr, <2 x i64>, i32, <2 x i64>, <2 x i32>, <2 x i32>)
declare void @__pseudo_scatter_factored_base_offsets32_i32(ptr, <2 x i32>, i32, <2 x i32>, <2 x i32>, <2 x i32>)

; CHECK-LABEL: define <4 x float> @unknown_addresses(
; CHECK-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v
define <4 x float> @unknown_addresses(<4 x i64> %addresses, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; The same integer-vector type can contain pointers from different objects.
; Merely finding ptrtoint in an insert chain does not establish a common base.
; CHECK-LABEL: define void @different_bases(
; CHECK-NEXT: %a = ptrtoint ptr %p to i64
; CHECK-NEXT: %b = ptrtoint ptr %q to i64
; CHECK-NEXT: %v0 = insertelement <4 x i64> poison, i64 %a, i32 0
; CHECK-NEXT: %v1 = insertelement <4 x i64> %v0, i64 %b, i32 1
; CHECK-NEXT: %v2 = insertelement <4 x i64> %v1, i64 %a, i32 2
; CHECK-NEXT: %v3 = insertelement <4 x i64> %v2, i64 %b, i32 3
; CHECK-NEXT: call void @__pseudo_scatter64_float(<4 x i64> %v3, <4 x float> %values, <4 x i32> %mask)
; CHECK-NEXT: ret void
define void @different_bases(ptr %p, ptr %q, <4 x float> %values, <4 x i32> %mask) {
  %a = ptrtoint ptr %p to i64
  %b = ptrtoint ptr %q to i64
  %v0 = insertelement <4 x i64> poison, i64 %a, i32 0
  %v1 = insertelement <4 x i64> %v0, i64 %b, i32 1
  %v2 = insertelement <4 x i64> %v1, i64 %a, i32 2
  %v3 = insertelement <4 x i64> %v2, i64 %b, i32 3
  call void @__pseudo_scatter64_float(<4 x i64> %v3, <4 x float> %values, <4 x i32> %mask)
  ret void
}

; The conversion emits vectors of the target width, so a call with another
; lane count keeps its general form even when its addresses have a common
; base and the replacement functions are declared: an insert chain and a
; broadcast, for a gather and a scatter.
; CHECK-LABEL: define <2 x i32> @other_width_insert_chain(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <2 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = insertelement <2 x i64> %s, i64 %b, i32 1
; CHECK-NEXT: %addresses = add <2 x i64> %bases, %offsets
; CHECK-NEXT: %v = call <2 x i32> @__pseudo_gather64_i32(<2 x i64> %addresses, <2 x i32> %mask)
; CHECK-NEXT: ret <2 x i32> %v
define <2 x i32> @other_width_insert_chain(ptr %base, <2 x i64> %offsets, <2 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <2 x i64> poison, i64 %b, i32 0
  %bases = insertelement <2 x i64> %s, i64 %b, i32 1
  %addresses = add <2 x i64> %bases, %offsets
  %v = call <2 x i32> @__pseudo_gather64_i32(<2 x i64> %addresses, <2 x i32> %mask)
  ret <2 x i32> %v
}

; CHECK-LABEL: define void @other_width_broadcast(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <2 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <2 x i64> %s, <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <2 x i64> %bases, %offsets
; CHECK-NEXT: call void @__pseudo_scatter64_i32(<2 x i64> %addresses, <2 x i32> %values, <2 x i32> %mask)
; CHECK-NEXT: ret void
define void @other_width_broadcast(ptr %base, <2 x i64> %offsets, <2 x i32> %values, <2 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <2 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <2 x i64> %s, <2 x i64> poison, <2 x i32> zeroinitializer
  %addresses = add <2 x i64> %bases, %offsets
  call void @__pseudo_scatter64_i32(<2 x i64> %addresses, <2 x i32> %values, <2 x i32> %mask)
  ret void
}
