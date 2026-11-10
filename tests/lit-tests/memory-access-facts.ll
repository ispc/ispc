; The address facts a consumer can ask for, implemented only
; where a constant or a splat proves them directly: the base/index/scale/
; displacement candidate a base-offset or factored call already carries and
; the arithmetic contract of its form, a narrower index whose sign extension
; is the index, and the lane pattern of the address operand. Everything else
; is unknown. Printing rewrites nothing.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o /dev/null 2>&1 | FileCheck %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o - 2>/dev/null | FileCheck --check-prefix=IR %s
; REQUIRES: X86_ENABLED

; Address space 1 has 32-bit pointers here.
target datalayout = "e-p:64:64-p1:32:32"

declare <4 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x half> @__pseudo_gather_base_offsets32_half(ptr addrspace(1), i32, <4 x i32>, <4 x i32>)
declare <4 x i64> @__pseudo_gather_base_offsets64_i64(ptr addrspace(1), i32, <4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_base_offsets32_i32(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare void @__pseudo_scatter_base_offsets64_i8(ptr, i32, <4 x i64>, <4 x i8>, <4 x i32>)

; 32-bit offsets on a 64-bit target: the lowering extends them, the index
; is already narrow.
; CHECK-LABEL: memory accesses of @index32:
; CHECK: candidate: base ptr %base + index <4 x i32> %offsets * 4 (lowering extends i32)
; CHECK-NEXT: narrow index: <4 x i32> %offsets
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 not known defined
define <4 x float> @index32(ptr %base, <4 x i32> %offsets, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_base_offsets32_float(ptr %base, i32 4, <4 x i32> %offsets, <4 x i32> %mask)
  ret <4 x float> %v
}

; A sign-extended i32 index narrows to its operand.
; CHECK-LABEL: memory accesses of @sext_index:
; CHECK: candidate: base ptr %base + index <4 x i64> %wide * 1 (address width)
; CHECK-NEXT: narrow index: <4 x i32> %index
define <4 x float> @sext_index(ptr %base, <4 x i32> %index, <4 x i32> %mask) {
  %wide = sext <4 x i32> %index to <4 x i64>
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %wide, <4 x i32> %mask)
  ret <4 x float> %v
}

; So does a zext nneg: it equals the sext.
; CHECK-LABEL: memory accesses of @zext_nneg_index:
; CHECK: candidate: base ptr %base + index <4 x i64> %wide * 2 (address width)
; CHECK-NEXT: narrow index: <4 x i32> %index
define <4 x float> @zext_nneg_index(ptr %base, <4 x i32> %index, <4 x i32> %mask) {
  %wide = zext nneg <4 x i32> %index to <4 x i64>
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 2, <4 x i64> %wide, <4 x i32> %mask)
  ret <4 x float> %v
}

; A plain zext may carry a set sign bit: unknown.
; CHECK-LABEL: memory accesses of @zext_index:
; CHECK: narrow index: unknown
define <4 x float> @zext_index(ptr %base, <4 x i32> %index, <4 x i32> %mask) {
  %wide = zext <4 x i32> %index to <4 x i64>
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %wide, <4 x i32> %mask)
  ret <4 x float> %v
}

; A sum needs a range proof of the complete expression: unknown here, even
; though each addend narrows. (The legacy narrowing accepts this sum; see
; address-gap-nested-sum.ispc for what that costs.)
; CHECK-LABEL: memory accesses of @sum_index:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 not known defined
define <4 x float> @sum_index(ptr %base, <4 x i32> %index, <4 x i32> %mask) {
  %wide = sext <4 x i32> %index to <4 x i64>
  %sum = add <4 x i64> %wide, <i64 0, i64 4, i64 8, i64 12>
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %sum, <4 x i32> %mask)
  ret <4 x float> %v
}

; i32 constants keep their signed value, -4 ...
; CHECK-LABEL: memory accesses of @negative_i32_lanes:
; CHECK: candidate: base ptr %base + index <4 x i32> splat (i32 -4) * 1 + displacement <4 x i32> <i32 0, i32 4, i32 8, i32 12> (lowering extends i32)
; CHECK-NEXT: narrow index: <4 x i32> splat (i32 -4), displacement lanes 0 4 8 12
; CHECK-NEXT: lane pattern of offsets: all equal, all lanes; lane 0 defined
define <4 x float> @negative_i32_lanes(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_factored_base_offsets32_float(ptr %base, <4 x i32> <i32 -4, i32 -4, i32 -4, i32 -4>, i32 1, <4 x i32> <i32 0, i32 4, i32 8, i32 12>, <4 x i32> %mask)
  ret <4 x float> %v
}

; ... while the same bit pattern zero extended to i64 is 4294967292, which
; does not fit a sign-extended i32 (#3984).
; CHECK-LABEL: memory accesses of @same_bits_zero_extended:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: all equal, all lanes; lane 0 defined
define <4 x float> @same_bits_zero_extended(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> <i64 4294967292, i64 4294967292, i64 4294967292, i64 4294967292>, i32 1, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i32> %mask)
  ret <4 x float> %v
}

; Negative i64 lanes within the signed i32 range narrow, including
; INT32_MIN.
; CHECK-LABEL: memory accesses of @negative_i64_lanes:
; CHECK: narrow index: lanes -4 0 4 8, displacement lanes -2147483648 -2147483648 -2147483648 -2147483648
; CHECK-NEXT: lane pattern of offsets: linear, stride 4, all lanes; lane 0 defined
define <4 x float> @negative_i64_lanes(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> <i64 -4, i64 0, i64 4, i64 8>, i32 4, <4 x i64> <i64 -2147483648, i64 -2147483648, i64 -2147483648, i64 -2147483648>, <4 x i32> %mask)
  ret <4 x float> %v
}

; A displacement of 2^31 does not fit, so the pair is unknown.
; CHECK-LABEL: memory accesses of @displacement_too_wide:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: linear, stride 4, all lanes; lane 0 defined
define <4 x float> @displacement_too_wide(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, i32 1, <4 x i64> <i64 2147483648, i64 2147483648, i64 2147483648, i64 2147483648>, <4 x i32> %mask)
  ret <4 x float> %v
}

; A splat of an SSA value that is provably not undef: all lanes equal, but
; lane 0 may be poison (here if the add overflows).
; CHECK-LABEL: memory accesses of @splat_offsets:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: all equal, all lanes; lane 0 not known defined
define void @splat_offsets(ptr %base, i64 noundef %x, <4 x i8> %bytes, <4 x i32> %mask) {
  %o = add nsw i64 %x, 1
  %s = insertelement <4 x i64> poison, i64 %o, i32 0
  %offsets = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  call void @__pseudo_scatter_base_offsets64_i8(ptr %base, i32 1, <4 x i64> %offsets, <4 x i8> %bytes, <4 x i32> %mask)
  ret void
}

; A splat of a value that may be undef proves nothing: an argument without
; noundef, or a value computed from undef. LLVM folds a broadcast of such a
; value to an undef vector, whose lanes need not agree.
; CHECK-LABEL: memory accesses of @splat_maybe_undef:
; CHECK: lane pattern of offsets: unknown; lane 0 not known defined
; CHECK: lane pattern of offsets: unknown; lane 0 not known defined
define <4 x float> @splat_maybe_undef(ptr %base, i64 %arg, <4 x i32> %mask) {
  %s = insertelement <4 x i64> poison, i64 %arg, i32 0
  %args = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %args, <4 x i32> %mask)
  %u = add i64 undef, 0
  %t = insertelement <4 x i64> poison, i64 %u, i32 0
  %undefs = shufflevector <4 x i64> %t, <4 x i64> poison, <4 x i32> zeroinitializer
  %w = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %undefs, <4 x i32> %mask)
  ret <4 x float> %v
}

; A broadcast whose mask has a poison lane leaves that lane poison: no
; pattern covers all lanes.
; CHECK-LABEL: memory accesses of @poison_mask_lane:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 not known defined
define <4 x float> @poison_mask_lane(ptr %base, i64 %o, <4 x i32> %mask) {
  %s = insertelement <4 x i64> poison, i64 %o, i32 0
  %offsets = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> <i32 0, i32 poison, i32 0, i32 0>
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %offsets, <4 x i32> %mask)
  ret <4 x float> %v
}

; A broadcast of an undef source lane proves no pattern either: LLVM folds
; it to an undef vector, whose lanes need not agree. A broadcast of a defined
; constant lane does.
; CHECK-LABEL: memory accesses of @undef_source_lane:
; CHECK: lane pattern of offsets: unknown; lane 0 not known defined
; CHECK: lane pattern of offsets: unknown; lane 0 not known defined
; CHECK: lane pattern of offsets: unknown; lane 0 not known defined
; CHECK: lane pattern of offsets: all equal, all lanes; lane 0 not known defined
define <4 x float> @undef_source_lane(ptr %base, i64 %o, <4 x i32> %mask) {
  %undefs = shufflevector <4 x i64> undef, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %undefs, <4 x i32> %mask)
  %other_lane = insertelement <4 x i64> undef, i64 %o, i32 1
  %undef_lane0 = shufflevector <4 x i64> %other_lane, <4 x i64> poison, <4 x i32> zeroinitializer
  %u = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %undef_lane0, <4 x i32> %mask)
  %poisons = shufflevector <4 x i64> <i64 poison, i64 4, i64 8, i64 12>, <4 x i64> poison, <4 x i32> zeroinitializer
  %w = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %poisons, <4 x i32> %mask)
  %defined = shufflevector <4 x i64> <i64 poison, i64 4, i64 8, i64 12>, <4 x i64> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %x = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %defined, <4 x i32> %mask)
  ret <4 x float> %v
}

; A constant with a poison lane proves no pattern.
; CHECK-LABEL: memory accesses of @poison_lane:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 not known defined
define <4 x float> @poison_lane(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> <i64 0, i64 poison, i64 8, i64 12>, <4 x i32> %mask)
  ret <4 x float> %v
}

; One lane out of the signed i32 range is enough to reject the narrowing,
; wherever it is; INT32_MAX itself fits.
; CHECK-LABEL: memory accesses of @second_lane_too_wide:
; CHECK: narrow index: unknown
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 defined
define <4 x float> @second_lane_too_wide(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> <i64 0, i64 2147483648, i64 8, i64 2147483647>, <4 x i32> %mask)
  ret <4 x float> %v
}

; CHECK-LABEL: memory accesses of @boundary_lanes:
; CHECK: narrow index: lanes 2147483647 -2147483648 0 -1
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 defined
define <4 x float> @boundary_lanes(ptr %base, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> <i64 2147483647, i64 -2147483648, i64 0, i64 -1>, <4 x i32> %mask)
  ret <4 x float> %v
}

; The stride is taken modulo the lane width: this i32 sequence wraps between
; lanes 1 and 2 and is still "linear" here, which is exactly why a lane
; pattern of a narrow operand is not a pattern of effective addresses.
; CHECK-LABEL: memory accesses of @wrapping_i32:
; CHECK: narrow index: <4 x i32> <i32 2147483646, i32 2147483647, i32 -2147483648, i32 -2147483647>
; CHECK-NEXT: lane pattern of offsets: linear, stride 1, all lanes; lane 0 defined
define <4 x i32> @wrapping_i32(ptr %base, <4 x i32> %mask) {
  %v = call <4 x i32> @__pseudo_gather_base_offsets32_i32(ptr %base, i32 4, <4 x i32> <i32 2147483646, i32 2147483647, i32 -2147483648, i32 -2147483647>, <4 x i32> %mask)
  ret <4 x i32> %v
}

; The arithmetic contract follows the base's address space: 32-bit offsets
; with a 32-bit pointer are address width; 64-bit offsets with a 32-bit
; pointer are a relationship no lowering is written for.
; CHECK-LABEL: memory accesses of @addrspace_32bit_pointer:
; CHECK: candidate: base ptr addrspace(1) %base + index <4 x i32> %offsets * 2 (address width)
; CHECK-NEXT: narrow index: <4 x i32> %offsets
; CHECK-NEXT: lane pattern of offsets: unknown; lane 0 not known defined
; CHECK-NOT: candidate
; CHECK: lane pattern of offsets: linear, stride 8, all lanes; lane 0 defined
define <4 x half> @addrspace_32bit_pointer(ptr addrspace(1) %base, <4 x i32> %offsets, <4 x i32> %mask) {
  %v = call <4 x half> @__pseudo_gather_base_offsets32_half(ptr addrspace(1) %base, i32 2, <4 x i32> %offsets, <4 x i32> %mask)
  %w = call <4 x i64> @__pseudo_gather_base_offsets64_i64(ptr addrspace(1) %base, i32 1, <4 x i64> <i64 0, i64 8, i64 16, i64 24>, <4 x i32> %mask)
  ret <4 x half> %v
}

; Without a constant scale there is no candidate; the lane pattern of the
; offsets is still known.
; CHECK-LABEL: memory accesses of @variable_scale:
; CHECK: mask: <4 x i32> %mask (unknown)
; CHECK-NEXT: source: unknown
; CHECK-NEXT: lane pattern of offsets: linear, stride 4, all lanes; lane 0 defined
define <4 x float> @variable_scale(ptr %base, i32 %scale, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 %scale, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i32> %mask)
  ret <4 x float> %v
}

; Constant absolute addresses: linear, every lane; no common base.
; CHECK-LABEL: memory accesses of @constant_addresses:
; CHECK: mask: <4 x i32> <i32 -1, i32 -1, i32 0, i32 0> (mixed)
; CHECK-NEXT: source: unknown
; CHECK-NEXT: common base: unknown
; CHECK-NEXT: lane pattern of addresses: linear, stride 4, all lanes; lane 0 defined
define <4 x float> @constant_addresses(<4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 4096, i64 4100, i64 4104, i64 4108>, <4 x i32> <i32 -1, i32 -1, i32 0, i32 0>)
  ret <4 x float> %v
}

; base + <0, 4, 8, 12> is linear for LLVMVectorIsLinear(), which looks
; through the arithmetic; the lane pattern query proves nothing about it.
; CHECK-LABEL: memory accesses of @linear_but_not_proven:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT: provenance: unknown
; CHECK-NEXT: lane pattern of addresses: unknown; lane 0 not known defined
define <4 x float> @linear_but_not_proven(ptr %base, <4 x i64> %u, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; IR-LABEL: define <4 x float> @sum_index(
; IR-NEXT: %wide = sext <4 x i32> %index to <4 x i64>
; IR-NEXT: %sum = add <4 x i64> %wide, <i64 0, i64 4, i64 8, i64 12>
; IR-NEXT: %v = call <4 x float> @__pseudo_gather_base_offsets64_float(ptr %base, i32 1, <4 x i64> %sum, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
; IR-LABEL: define <4 x float> @negative_i64_lanes(
; IR-NEXT: %v = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %base, <4 x i64> <i64 -4, i64 0, i64 4, i64 8>, i32 4, <4 x i64> splat (i64 -2147483648), <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
; IR-LABEL: define <4 x float> @linear_but_not_proven(
; IR-NEXT: %b = ptrtoint ptr %base to i64
; IR-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; IR-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; IR-NEXT: %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
; IR-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
