; The conversion needs its replacement functions. The
; frontend always declares them; hand-written IR may not. A module without
; the function the conversion may select keeps the general call, without
; inserting anything, instead of crashing. Under --addressing=32 either width
; may be selected, so both must be declared; here only the 64-bit factored
; form is, which suffices under --addressing=64.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A32 %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A64 %s
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x double> @__pseudo_gather64_double(<4 x i64>, <4 x i32>)

; CHECK-LABEL: define <4 x float> @only_64bit_form_declared(
; CHECK:      %addresses = add <4 x i64> %bases, %offsets
; A32-NEXT:   %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; A32-NEXT:   ret <4 x float> %v
; A64-NEXT:   {{%.*}} = add <4 x i64> zeroinitializer, %offsets
; A64-NEXT:   [[BASE:%.*]] = inttoptr i64 %b to ptr
; A64:        [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; A64-NEXT:   ret <4 x float> [[V]]
define <4 x float> @only_64bit_form_declared(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; CHECK-LABEL: define <4 x double> @nothing_declared(
; CHECK:      %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: %v = call <4 x double> @__pseudo_gather64_double(<4 x i64> %addresses, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x double> %v
define <4 x double> @nothing_declared(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x double> @__pseudo_gather64_double(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x double> %v
}
