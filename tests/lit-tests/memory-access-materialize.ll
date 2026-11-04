; The accepted base/offset recipe is materialized into the
; same instructions, names and insertion points the mutating matcher
; produced, and feeds the existing factoring, narrowing and emission. The
; expected output below was recorded from the pass before the change; the
; dead instructions of the old matcher's intermediate results are part of
; it. CHECK lines hold for both addressing modes, A32/A64 lines for one.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A32 %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A64 %s
; REQUIRES: X86_ENABLED

@table = global [16 x float] zeroinitializer

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare void @__pseudo_scatter64_float(<4 x i64>, <4 x float>, <4 x i32>)
declare <4 x i32> @__pseudo_gather64_i32(<4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather32_i32(<4 x i32>, <4 x i32>)
declare <4 x float> @__masked_load_float(ptr, <4 x i32>)
declare <4 x i32> @__masked_load_i32(ptr, <4 x i32>)
declare void @__pseudo_masked_store_float(ptr, <4 x float>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare void @__pseudo_scatter_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x float>, <4 x i32>)
declare void @__pseudo_scatter_factored_base_offsets32_float(ptr, <4 x i32>, i32, <4 x i32>, <4 x float>, <4 x i32>)
declare void @__pseudo_scatter_base_offsets64_float(ptr, i32, <4 x i64>, <4 x float>, <4 x i32>)
declare void @__pseudo_scatter_base_offsets32_float(ptr, i32, <4 x i32>, <4 x float>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_base_offsets64_i32(ptr, i32, <4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_base_offsets32_i32(ptr, i32, <4 x i32>, <4 x i32>)

; A pointer broadcast through an insert chain plus offsets. The matcher's
; offset sum starts at zero; the factoring below strips it again and leaves
; the dead add behind, as before.
; CHECK-LABEL: define <4 x float> @insert_chain(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %v0 = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %v1 = insertelement <4 x i64> %v0, i64 %b, i32 1
; CHECK-NEXT: %v2 = insertelement <4 x i64> %v1, i64 %b, i32 2
; CHECK-NEXT: %v3 = insertelement <4 x i64> %v2, i64 %b, i32 3
; CHECK-NEXT: %addresses = add <4 x i64> %v3, %offsets
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v4 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %b_2void, <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v4
; CHECK-NEXT: }
define <4 x float> @insert_chain(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %v0 = insertelement <4 x i64> poison, i64 %b, i32 0
  %v1 = insertelement <4 x i64> %v0, i64 %b, i32 1
  %v2 = insertelement <4 x i64> %v1, i64 %b, i32 2
  %v3 = insertelement <4 x i64> %v2, i64 %b, i32 3
  %addresses = add <4 x i64> %v3, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Nested adds: the terms are added innermost first, the base may be either
; operand.
; CHECK-LABEL: define void @nested_adds(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %inner = add <4 x i64> %a, %bases
; CHECK-NEXT: %addresses = add <4 x i64> %c, %inner
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %a
; CHECK-NEXT: %new_offsets1 = add <4 x i64> %new_offsets, %c
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %new_add = add <4 x i64> %a, %c
; CHECK-NEXT: %add_a_c = add <4 x i64> %a, %c
; CHECK-NEXT: %add_a_c2 = add <4 x i64> %a, %c
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %new_add
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: call void @__pseudo_scatter_factored_base_offsets64_float(ptr %b_2void, <4 x i64> %new_add, i32 1, <4 x i64> zeroinitializer, <4 x float> %values, <4 x i32> %mask)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @nested_adds(ptr %base, <4 x i64> %a, <4 x i64> %c, <4 x float> %values, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %inner = add <4 x i64> %a, %bases
  %addresses = add <4 x i64> %c, %inner
  call void @__pseudo_scatter64_float(<4 x i64> %addresses, <4 x float> %values, <4 x i32> %mask)
  ret void
}

; The lane-0 constant of a broadcast is splat by a shuffle inserted before
; the add it came from.
; CHECK-LABEL: define <4 x float> @broadcast_constant(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %shuffle = shufflevector <4 x i64> <i64 8, i64 poison, i64 poison, i64 poison>, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <4 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison>
; CHECK-NEXT: %bases = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %shuffle
; CHECK-NEXT: %new_offsets1 = add <4 x i64> %new_offsets, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %new_add = add <4 x i64> %shuffle, %offsets
; CHECK-NEXT: %add_shuffle_offsets = add <4 x i64> %shuffle, %offsets
; CHECK-NEXT: %add_shuffle_offsets3 = add <4 x i64> %shuffle, %offsets
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %new_add
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v2 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %b_2void, <4 x i64> %new_add, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v2
; CHECK-NEXT: }
define <4 x float> @broadcast_constant(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %plus = add <4 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison>
  %bases = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Lane 0 of another operation is splat by a shuffle inserted after it; all
; lanes then share one address and the gather becomes a scalar load.
; CHECK-LABEL: define <4 x float> @broadcast_part(
; CHECK-NEXT: %first_elt = extractelement <4 x i64> %vec, i32 0
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %part.elt0 = shl i64 %first_elt, 2
; CHECK-NEXT: %part = shl <4 x i64> %vec, <i64 2, i64 poison, i64 poison, i64 poison>
; CHECK-NEXT: %shuffle = shufflevector <4 x i64> %part, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <4 x i64> %part, %s
; CHECK-NEXT: %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %shuffle
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %scaled_varying.elt0 = mul i64 1, %part.elt0
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %shuffle
; CHECK-NEXT: %"varying+const_offsets.elt0" = add i64 %scaled_varying.elt0, 0
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %ptr = getelementptr i8, ptr %b_2void, i64 %"varying+const_offsets.elt0"
; CHECK-NEXT: %v12 = load float, ptr %ptr, align 4
; CHECK-NEXT: %v13 = insertelement <4 x float> poison, float %v12, i32 0
; CHECK-NEXT: %v14 = shufflevector <4 x float> %v13, <4 x float> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: ret <4 x float> %v14
; CHECK-NEXT: }
define <4 x float> @broadcast_part(ptr %base, <4 x i64> %vec, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %part = shl <4 x i64> %vec, <i64 2, i64 poison, i64 poison, i64 poison>
  %plus = add <4 x i64> %part, %s
  %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A vector zext of the bases is replayed as a scalar zext of the base.
; CHECK-LABEL: define <4 x float> @cast_chain(
; CHECK-NEXT: %b = ptrtoint ptr %base to i32
; CHECK-NEXT: %s = insertelement <4 x i32> poison, i32 %b, i32 0
; CHECK-NEXT: %bases32 = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %bases = zext <4 x i32> %bases32 to <4 x i64>
; CHECK-NEXT: %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: %b_cast = zext i32 %b to i64
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %b_cast_2void = inttoptr i64 %b_cast to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v1 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %b_cast_2void, <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v1
; CHECK-NEXT: }
define <4 x float> @cast_chain(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i32
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %bases32 = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %bases = zext <4 x i32> %bases32 to <4 x i64>
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A constant splat of a global's address.
; CHECK-LABEL: define <4 x float> @constant_splat(
; CHECK-NEXT: %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 ptrtoint (ptr @table to i64), i64 ptrtoint (ptr @table to i64), i64 ptrtoint (ptr @table to i64)>, %offsets
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v1 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %_2void, <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v1
; CHECK-NEXT: }
define <4 x float> @constant_splat(<4 x i64> %offsets, <4 x i32> %mask) {
  %addresses = add <4 x i64> splat (i64 ptrtoint (ptr @table to i64)), %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; One global plus per-lane displacements: the displacements start the
; offset sum.
; CHECK-LABEL: define <4 x float> @constant_lanes_plus(
; CHECK-NEXT: %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, %extra
; CHECK-NEXT: %new_offsets = add <4 x i64> <i64 0, i64 4, i64 8, i64 12>, %extra
; CHECK-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %extra
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT: %v1 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %_2void, <4 x i64> %extra, i32 1, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v1
; CHECK-NEXT: }
define <4 x float> @constant_lanes_plus(<4 x i64> %extra, <4 x i32> %mask) {
  %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, %extra
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Linear displacements alone become a masked load.
; A32-LABEL: define <4 x float> @constant_lanes_only(
; A32-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32-NEXT: %_trunc = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_trunc1.elt0 = trunc i64 0 to i32
; A32-NEXT: %_trunc1 = trunc <4 x i64> <i64 0, i64 4, i64 8, i64 12> to <4 x i32>
; A32-NEXT: %_cast.elt0 = trunc i64 0 to i32
; A32-NEXT: %_cast = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_cast3.elt0 = mul i32 %_cast.elt0, 1
; A32-NEXT: %_cast3 = mul <4 x i32> %_cast, splat (i32 1)
; A32-NEXT: %_trunc14.elt0 = add i32 %_cast3.elt0, %_trunc1.elt0
; A32-NEXT: %_trunc14 = add <4 x i32> %_cast3, %_trunc1
; A32-NEXT: %scaled_varying.elt0 = mul i32 1, 0
; A32-NEXT: %scaled_varying = mul <4 x i32> splat (i32 1), zeroinitializer
; A32-NEXT: %"varying+const_offsets.elt0" = add i32 %scaled_varying.elt0, %_trunc14.elt0
; A32-NEXT: %"varying+const_offsets" = add <4 x i32> %scaled_varying, %_trunc14
; A32-NEXT: %ptr = getelementptr i8, ptr %_2void, i32 %"varying+const_offsets.elt0"
; A32-NEXT: %ptr_masked_load = call <4 x float> @__masked_load_float(ptr %ptr, <4 x i32> %mask)
; A32-NEXT: ret <4 x float> %ptr_masked_load
; A32-NEXT: }
; A64-LABEL: define <4 x float> @constant_lanes_only(
; A64-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A64-NEXT: %scaled_varying.elt0 = mul i64 1, 0
; A64-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), zeroinitializer
; A64-NEXT: %"varying+const_offsets.elt0" = add i64 %scaled_varying.elt0, 0
; A64-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, <i64 0, i64 4, i64 8, i64 12>
; A64-NEXT: %ptr = getelementptr i8, ptr %_2void, i64 %"varying+const_offsets.elt0"
; A64-NEXT: %ptr_masked_load = call <4 x float> @__masked_load_float(ptr %ptr, <4 x i32> %mask)
; A64-NEXT: ret <4 x float> %ptr_masked_load
; A64-NEXT: }
define <4 x float> @constant_lanes_only(<4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, <4 x i32> %mask)
  ret <4 x float> %v
}

; Addresses extracted from an aggregate built with insertvalue.
; CHECK-LABEL: define <4 x float> @aggregate(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: %agg0 = insertvalue { <4 x i64>, <4 x i64> } poison, <4 x i64> %unrelated, 0
; CHECK-NEXT: %agg1 = insertvalue { <4 x i64>, <4 x i64> } %agg0, <4 x i64> %addresses, 1
; CHECK-NEXT: %extracted = extractvalue { <4 x i64>, <4 x i64> } %agg1, 1
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v1 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %b_2void, <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> %v1
; CHECK-NEXT: }
define <4 x float> @aggregate(ptr %base, <4 x i64> %offsets, <4 x i64> %unrelated, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %agg0 = insertvalue { <4 x i64>, <4 x i64> } poison, <4 x i64> %unrelated, 0
  %agg1 = insertvalue { <4 x i64>, <4 x i64> } %agg0, <4 x i64> %addresses, 1
  %extracted = extractvalue { <4 x i64>, <4 x i64> } %agg1, 1
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %extracted, <4 x i32> %mask)
  ret <4 x float> %v
}

; A sign-extended 32-bit index selects the 32-bit factored form under
; --addressing=32 and keeps the 64-bit form under --addressing=64.
; A32-LABEL: define <4 x i32> @sext_index(
; A32-NEXT: %b = ptrtoint ptr %base to i64
; A32-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; A32-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; A32-NEXT: %scaled = shl <4 x i32> %index, splat (i32 2)
; A32-NEXT: %wide = sext <4 x i32> %scaled to <4 x i64>
; A32-NEXT: %addresses = add <4 x i64> %bases, %wide
; A32-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %wide
; A32-NEXT: %b_2void = inttoptr i64 %b to ptr
; A32-NEXT: %offset_cast = sext <4 x i32> %scaled to <4 x i64>
; A32-NEXT: %scaled_cast = sext <4 x i32> %scaled to <4 x i64>
; A32-NEXT: %_trunc = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %scaled_varying = mul <4 x i32> splat (i32 1), %scaled
; A32-NEXT: %"varying+const_offsets" = add <4 x i32> %scaled_varying, %_trunc
; A32-NEXT: %v1 = call <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr %b_2void, <4 x i32> %scaled, i32 1, <4 x i32> %_trunc, <4 x i32> %mask)
; A32-NEXT: ret <4 x i32> %v1
; A32-NEXT: }
; A64-LABEL: define <4 x i32> @sext_index(
; A64-NEXT: %b = ptrtoint ptr %base to i64
; A64-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; A64-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; A64-NEXT: %scaled = shl <4 x i32> %index, splat (i32 2)
; A64-NEXT: %wide = sext <4 x i32> %scaled to <4 x i64>
; A64-NEXT: %addresses = add <4 x i64> %bases, %wide
; A64-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %wide
; A64-NEXT: %b_2void = inttoptr i64 %b to ptr
; A64-NEXT: %offset_cast = sext <4 x i32> %scaled to <4 x i64>
; A64-NEXT: %scaled_cast = sext <4 x i32> %scaled to <4 x i64>
; A64-NEXT: %scaled_cast2 = sext <4 x i32> %scaled to <4 x i64>
; A64-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offset_cast
; A64-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; A64-NEXT: %v1 = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %b_2void, <4 x i64> %offset_cast, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; A64-NEXT: ret <4 x i32> %v1
; A64-NEXT: }
define <4 x i32> @sext_index(ptr %base, <4 x i32> %index, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %scaled = shl <4 x i32> %index, splat (i32 2)
  %wide = sext <4 x i32> %scaled to <4 x i64>
  %addresses = add <4 x i64> %bases, %wide
  %v = call <4 x i32> @__pseudo_gather64_i32(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x i32> %v
}

; The revisit chain within one pass invocation: general gather, factored
; gather, masked load, then an ordinary load for the all-on mask.
; A32-LABEL: define <4 x float> @chain_to_load(
; A32-NEXT: %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, zeroinitializer
; A32-NEXT: %new_offsets = add <4 x i64> <i64 0, i64 4, i64 8, i64 12>, zeroinitializer
; A32-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32-NEXT: %_trunc = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_trunc1.elt0 = trunc i64 0 to i32
; A32-NEXT: %_trunc1 = trunc <4 x i64> <i64 0, i64 4, i64 8, i64 12> to <4 x i32>
; A32-NEXT: %_cast.elt0 = trunc i64 0 to i32
; A32-NEXT: %_cast = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_cast3.elt0 = mul i32 %_cast.elt0, 1
; A32-NEXT: %_cast3 = mul <4 x i32> %_cast, splat (i32 1)
; A32-NEXT: %_trunc14.elt0 = add i32 %_cast3.elt0, %_trunc1.elt0
; A32-NEXT: %_trunc14 = add <4 x i32> %_cast3, %_trunc1
; A32-NEXT: %scaled_varying.elt0 = mul i32 1, 0
; A32-NEXT: %scaled_varying = mul <4 x i32> splat (i32 1), zeroinitializer
; A32-NEXT: %"varying+const_offsets.elt0" = add i32 %scaled_varying.elt0, %_trunc14.elt0
; A32-NEXT: %"varying+const_offsets" = add <4 x i32> %scaled_varying, %_trunc14
; A32-NEXT: %ptr = getelementptr i8, ptr %_2void, i32 %"varying+const_offsets.elt0"
; A32-NEXT: %ptr_masked_load5 = load <4 x float>, ptr %ptr, align 4
; A32-NEXT: ret <4 x float> %ptr_masked_load5
; A32-NEXT: }
; A64-LABEL: define <4 x float> @chain_to_load(
; A64-NEXT: %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, zeroinitializer
; A64-NEXT: %new_offsets = add <4 x i64> <i64 0, i64 4, i64 8, i64 12>, zeroinitializer
; A64-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A64-NEXT: %scaled_varying.elt0 = mul i64 1, 0
; A64-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), zeroinitializer
; A64-NEXT: %"varying+const_offsets.elt0" = add i64 %scaled_varying.elt0, 0
; A64-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, <i64 0, i64 4, i64 8, i64 12>
; A64-NEXT: %ptr = getelementptr i8, ptr %_2void, i64 %"varying+const_offsets.elt0"
; A64-NEXT: %ptr_masked_load2 = load <4 x float>, ptr %ptr, align 4
; A64-NEXT: ret <4 x float> %ptr_masked_load2
; A64-NEXT: }
define <4 x float> @chain_to_load(<4 x i64> %extra) {
  %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, zeroinitializer
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> splat (i32 -1))
  ret <4 x float> %v
}

; A scatter under an all-off mask becomes a masked store that is removed.
; A32-LABEL: define void @store_removed(
; A32-NEXT: %b = ptrtoint ptr %base to i64
; A32-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; A32-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; A32-NEXT: %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
; A32-NEXT: %new_offsets = add <4 x i64> zeroinitializer, <i64 0, i64 4, i64 8, i64 12>
; A32-NEXT: %b_2void = inttoptr i64 %b to ptr
; A32-NEXT: %_trunc = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_trunc1.elt0 = trunc i64 0 to i32
; A32-NEXT: %_trunc1 = trunc <4 x i64> <i64 0, i64 4, i64 8, i64 12> to <4 x i32>
; A32-NEXT: %_cast.elt0 = trunc i64 0 to i32
; A32-NEXT: %_cast = trunc <4 x i64> zeroinitializer to <4 x i32>
; A32-NEXT: %_cast2.elt0 = mul i32 %_cast.elt0, 1
; A32-NEXT: %_cast2 = mul <4 x i32> %_cast, splat (i32 1)
; A32-NEXT: %_trunc13.elt0 = add i32 %_cast2.elt0, %_trunc1.elt0
; A32-NEXT: %_trunc13 = add <4 x i32> %_cast2, %_trunc1
; A32-NEXT: %scaled_varying.elt0 = mul i32 1, 0
; A32-NEXT: %scaled_varying = mul <4 x i32> splat (i32 1), zeroinitializer
; A32-NEXT: %"varying+const_offsets.elt0" = add i32 %scaled_varying.elt0, %_trunc13.elt0
; A32-NEXT: %"varying+const_offsets" = add <4 x i32> %scaled_varying, %_trunc13
; A32-NEXT: %ptr = getelementptr i8, ptr %b_2void, i32 %"varying+const_offsets.elt0"
; A32-NEXT: ret void
; A32-NEXT: }
; A64-LABEL: define void @store_removed(
; A64-NEXT: %b = ptrtoint ptr %base to i64
; A64-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; A64-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; A64-NEXT: %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
; A64-NEXT: %new_offsets = add <4 x i64> zeroinitializer, <i64 0, i64 4, i64 8, i64 12>
; A64-NEXT: %b_2void = inttoptr i64 %b to ptr
; A64-NEXT: %scaled_varying.elt0 = mul i64 1, 0
; A64-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), zeroinitializer
; A64-NEXT: %"varying+const_offsets.elt0" = add i64 %scaled_varying.elt0, 0
; A64-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, <i64 0, i64 4, i64 8, i64 12>
; A64-NEXT: %ptr = getelementptr i8, ptr %b_2void, i64 %"varying+const_offsets.elt0"
; A64-NEXT: ret void
; A64-NEXT: }
define void @store_removed(ptr %base, <4 x i64> %offsets, <4 x float> %values) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
  call void @__pseudo_scatter64_float(<4 x i64> %addresses, <4 x float> %values, <4 x i32> zeroinitializer)
  ret void
}

; The ISPC position metadata is copied to the new base pointer and call.
; CHECK-LABEL: define <4 x float> @keeps_metadata(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: %new_offsets = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr, !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
; CHECK-NEXT: %scaled_varying = mul <4 x i64> splat (i64 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v1 = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr %b_2void, <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask), !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
; CHECK-NEXT: ret <4 x float> %v1
; CHECK-NEXT: }
define <4 x float> @keeps_metadata(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask), !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
  ret <4 x float> %v
}

; With 32-bit addresses the offsets have the addresses' type; the old
; matcher used the target's pointer width for its zero vector, which did
; not match in this case.
; CHECK-LABEL: define <4 x i32> @addresses32(
; CHECK-NEXT: %b = ptrtoint ptr %base to i32
; CHECK-NEXT: %s = insertelement <4 x i32> poison, i32 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <4 x i32> %bases, %offsets
; CHECK-NEXT: %new_offsets = add <4 x i32> zeroinitializer, %offsets
; CHECK-NEXT: %b_2void = inttoptr i32 %b to ptr
; CHECK-NEXT: %scaled_varying = mul <4 x i32> splat (i32 1), %offsets
; CHECK-NEXT: %"varying+const_offsets" = add <4 x i32> %scaled_varying, zeroinitializer
; CHECK-NEXT: %v1 = call <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr %b_2void, <4 x i32> %offsets, i32 1, <4 x i32> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x i32> %v1
define <4 x i32> @addresses32(ptr %base, <4 x i32> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i32
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %bases = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i32> %bases, %offsets
  %v = call <4 x i32> @__pseudo_gather32_i32(<4 x i32> %addresses, <4 x i32> %mask)
  ret <4 x i32> %v
}

!0 = !{!"kernel.ispc"}
!1 = !{i32 12}
!2 = !{i32 7}
!3 = !{i32 20}
