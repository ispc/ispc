; Read-only base recovery must retain the general varying-address fallback
; when no common base is established. These rejection cases previously
; depended on the local external corpus. No target builtin definitions are
; needed: the calls must remain unchanged.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=improve-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare void @__pseudo_scatter64_float(<4 x i64>, <4 x float>, <4 x i32>)

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
