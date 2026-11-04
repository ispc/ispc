; Native gather path: on a target with a gather
; instruction the recovered base and offsets feed the non-factored
; base-offset form through the existing scale extraction and narrowing.
; The expected output was recorded from the pass before the change. CHECK
; lines hold for both addressing modes, A32/A64 lines for one.
;
; RUN: %{ispc-opt} --target=avx2-i32x8 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A32 %s
; RUN: %{ispc-opt} --target=avx2-i32x8 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck --check-prefixes=CHECK,A64 %s
; REQUIRES: X86_ENABLED

@table = global [16 x float] zeroinitializer

declare <8 x float> @__pseudo_gather64_float(<8 x i64>, <8 x i32>)
declare void @__pseudo_scatter64_float(<8 x i64>, <8 x float>, <8 x i32>)
declare <8 x i32> @__pseudo_gather64_i32(<8 x i64>, <8 x i32>)
declare <8 x float> @__masked_load_float(ptr, <8 x i32>)
declare <8 x i32> @__masked_load_i32(ptr, <8 x i32>)
declare void @__pseudo_masked_store_float(ptr, <8 x float>, <8 x i32>)
declare <8 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <8 x i64>, i32, <8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <8 x i32>, i32, <8 x i32>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <8 x i32>, <8 x i32>)
declare void @__pseudo_scatter_factored_base_offsets64_float(ptr, <8 x i64>, i32, <8 x i64>, <8 x float>, <8 x i32>)
declare void @__pseudo_scatter_factored_base_offsets32_float(ptr, <8 x i32>, i32, <8 x i32>, <8 x float>, <8 x i32>)
declare void @__pseudo_scatter_base_offsets64_float(ptr, i32, <8 x i64>, <8 x float>, <8 x i32>)
declare void @__pseudo_scatter_base_offsets32_float(ptr, i32, <8 x i32>, <8 x float>, <8 x i32>)
declare <8 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <8 x i64>, i32, <8 x i64>, <8 x i32>)
declare <8 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr, <8 x i32>, i32, <8 x i32>, <8 x i32>)
declare <8 x i32> @__pseudo_gather_base_offsets64_i32(ptr, i32, <8 x i64>, <8 x i32>)
declare <8 x i32> @__pseudo_gather_base_offsets32_i32(ptr, i32, <8 x i32>, <8 x i32>)

; CHECK-LABEL: define <8 x float> @insert_chain(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %v0 = insertelement <8 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %v1 = insertelement <8 x i64> %v0, i64 %b, i32 1
; CHECK-NEXT: %v2 = insertelement <8 x i64> %v1, i64 %b, i32 2
; CHECK-NEXT: %v3 = insertelement <8 x i64> %v2, i64 %b, i32 3
; CHECK-NEXT: %v4 = insertelement <8 x i64> %v3, i64 %b, i32 4
; CHECK-NEXT: %v5 = insertelement <8 x i64> %v4, i64 %b, i32 5
; CHECK-NEXT: %v6 = insertelement <8 x i64> %v5, i64 %b, i32 6
; CHECK-NEXT: %v7 = insertelement <8 x i64> %v6, i64 %b, i32 7
; CHECK-NEXT: %addresses = add <8 x i64> %v7, %offsets
; CHECK-NEXT: %new_offsets = add <8 x i64> zeroinitializer, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %v8 = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr %b_2void, i32 1, <8 x i64> %offsets, <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> %v8
; CHECK-NEXT: }
define <8 x float> @insert_chain(ptr %base, <8 x i64> %offsets, <8 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %v0 = insertelement <8 x i64> poison, i64 %b, i32 0
  %v1 = insertelement <8 x i64> %v0, i64 %b, i32 1
  %v2 = insertelement <8 x i64> %v1, i64 %b, i32 2
  %v3 = insertelement <8 x i64> %v2, i64 %b, i32 3
  %v4 = insertelement <8 x i64> %v3, i64 %b, i32 4
  %v5 = insertelement <8 x i64> %v4, i64 %b, i32 5
  %v6 = insertelement <8 x i64> %v5, i64 %b, i32 6
  %v7 = insertelement <8 x i64> %v6, i64 %b, i32 7
  %addresses = add <8 x i64> %v7, %offsets
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %v
}

; CHECK-LABEL: define void @nested_adds(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <8 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
; CHECK-NEXT: %inner = add <8 x i64> %a, %bases
; CHECK-NEXT: %addresses = add <8 x i64> %c, %inner
; CHECK-NEXT: %new_offsets = add <8 x i64> zeroinitializer, %a
; CHECK-NEXT: %new_offsets1 = add <8 x i64> %new_offsets, %c
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %new_add = add <8 x i64> %a, %c
; CHECK-NEXT: %add_a_c = add <8 x i64> %a, %c
; CHECK-NEXT: %add_a_c2 = add <8 x i64> %a, %c
; CHECK-NEXT: %scaled_varying = mul <8 x i64> splat (i64 1), %new_add
; CHECK-NEXT: %"varying+const_offsets" = add <8 x i64> %scaled_varying, zeroinitializer
; CHECK-NEXT: call void @__pseudo_scatter_factored_base_offsets64_float(ptr %b_2void, <8 x i64> %new_add, i32 1, <8 x i64> zeroinitializer, <8 x float> %values, <8 x i32> %mask)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @nested_adds(ptr %base, <8 x i64> %a, <8 x i64> %c, <8 x float> %values, <8 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <8 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
  %inner = add <8 x i64> %a, %bases
  %addresses = add <8 x i64> %c, %inner
  call void @__pseudo_scatter64_float(<8 x i64> %addresses, <8 x float> %values, <8 x i32> %mask)
  ret void
}

; CHECK-LABEL: define <8 x float> @broadcast_constant(
; CHECK-NEXT: %b = ptrtoint ptr %base to i64
; CHECK-NEXT: %s = insertelement <8 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: %shuffle = shufflevector <8 x i64> <i64 8, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>, <8 x i64> poison, <8 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <8 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>
; CHECK-NEXT: %bases = shufflevector <8 x i64> %plus, <8 x i64> poison, <8 x i32> zeroinitializer
; CHECK-NEXT: %addresses = add <8 x i64> %bases, %offsets
; CHECK-NEXT: %new_offsets = add <8 x i64> zeroinitializer, %shuffle
; CHECK-NEXT: %new_add = add <8 x i64> %shuffle, %offsets
; CHECK-NEXT: %new_offsets1 = add <8 x i64> %new_offsets, %offsets
; CHECK-NEXT: %b_2void = inttoptr i64 %b to ptr
; CHECK-NEXT: %v2 = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr %b_2void, i32 1, <8 x i64> %new_add, <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> %v2
; CHECK-NEXT: }
define <8 x float> @broadcast_constant(ptr %base, <8 x i64> %offsets, <8 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <8 x i64> poison, i64 %b, i32 0
  %plus = add <8 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>
  %bases = shufflevector <8 x i64> %plus, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %offsets
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %v
}

; CHECK-LABEL: define <8 x float> @constant_lanes_plus(
; CHECK-NEXT: %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, %extra
; CHECK-NEXT: %new_add = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, %extra
; CHECK-NEXT: %new_offsets = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, %extra
; CHECK-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK-NEXT: %v1 = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr %_2void, i32 1, <8 x i64> %new_add, <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> %v1
; CHECK-NEXT: }
define <8 x float> @constant_lanes_plus(<8 x i64> %extra, <8 x i32> %mask) {
  %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, %extra
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %v
}

; A32-LABEL: define <8 x i32> @sext_index(
; A32-NEXT: %b = ptrtoint ptr %base to i64
; A32-NEXT: %s = insertelement <8 x i64> poison, i64 %b, i32 0
; A32-NEXT: %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
; A32-NEXT: %scaled = shl <8 x i32> %index, splat (i32 2)
; A32-NEXT: %offset_cast = sext <8 x i32> %scaled to <8 x i64>
; A32-NEXT: %wide = sext <8 x i32> %scaled to <8 x i64>
; A32-NEXT: %addresses = add <8 x i64> %bases, %wide
; A32-NEXT: %new_offsets = add <8 x i64> zeroinitializer, %wide
; A32-NEXT: %b_2void = inttoptr i64 %b to ptr
; A32-NEXT: %v1 = call <8 x i32> @__pseudo_gather_base_offsets32_i32(ptr %b_2void, i32 1, <8 x i32> %scaled, <8 x i32> %mask)
; A32-NEXT: ret <8 x i32> %v1
; A32-NEXT: }
; A64-LABEL: define <8 x i32> @sext_index(
; A64-NEXT: %b = ptrtoint ptr %base to i64
; A64-NEXT: %s = insertelement <8 x i64> poison, i64 %b, i32 0
; A64-NEXT: %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
; A64-NEXT: %scaled = shl <8 x i32> %index, splat (i32 2)
; A64-NEXT: %offset_cast = sext <8 x i32> %scaled to <8 x i64>
; A64-NEXT: %wide = sext <8 x i32> %scaled to <8 x i64>
; A64-NEXT: %addresses = add <8 x i64> %bases, %wide
; A64-NEXT: %new_offsets = add <8 x i64> zeroinitializer, %wide
; A64-NEXT: %b_2void = inttoptr i64 %b to ptr
; A64-NEXT: %v1 = call <8 x i32> @__pseudo_gather_base_offsets64_i32(ptr %b_2void, i32 1, <8 x i64> %offset_cast, <8 x i32> %mask)
; A64-NEXT: ret <8 x i32> %v1
; A64-NEXT: }
define <8 x i32> @sext_index(ptr %base, <8 x i32> %index, <8 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <8 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <8 x i64> %s, <8 x i64> poison, <8 x i32> zeroinitializer
  %scaled = shl <8 x i32> %index, splat (i32 2)
  %wide = sext <8 x i32> %scaled to <8 x i64>
  %addresses = add <8 x i64> %bases, %wide
  %v = call <8 x i32> @__pseudo_gather64_i32(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x i32> %v
}

; A32-LABEL: define <8 x float> @chain_to_load(
; A32-NEXT: %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, zeroinitializer
; A32-NEXT: %new_offsets = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, zeroinitializer
; A32-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32-NEXT: %_trunc.elt0 = trunc i64 0 to i32
; A32-NEXT: %_trunc = trunc <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28> to <8 x i32>
; A32-NEXT: %ptr = getelementptr i8, ptr %_2void, i32 %_trunc.elt0
; A32-NEXT: %ptr_masked_load2 = load <8 x float>, ptr %ptr, align 4
; A32-NEXT: ret <8 x float> %ptr_masked_load2
; A32-NEXT: }
; A64-LABEL: define <8 x float> @chain_to_load(
; A64-NEXT: %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, zeroinitializer
; A64-NEXT: %new_offsets = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, zeroinitializer
; A64-NEXT: %_2void = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A64-NEXT: %ptr = getelementptr i8, ptr %_2void, i64 0
; A64-NEXT: %ptr_masked_load2 = load <8 x float>, ptr %ptr, align 4
; A64-NEXT: ret <8 x float> %ptr_masked_load2
; A64-NEXT: }
define <8 x float> @chain_to_load(<8 x i64> %extra) {
  %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, zeroinitializer
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> splat (i32 -1))
  ret <8 x float> %v
}
