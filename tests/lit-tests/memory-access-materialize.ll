; The accepted base/offset recipe is materialized as the sum of its terms,
; innermost first, followed by the base cast, and feeds the existing
; factoring, narrowing and emission. The checks cover the materialized
; offsets, the shuffles of broadcast lanes next to the instruction they
; come from, the base and the replacement call, load or store; the
; instructions of the existing emission in between are not pinned. CHECK
; lines hold for both addressing modes, A32/A64 lines for one.
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

; A pointer broadcast through an insert chain plus offsets. The offset sum
; starts at zero; the factoring strips it again, so the call takes the
; offsets directly.
; CHECK-LABEL: define <4 x float> @insert_chain(
; CHECK:      %addresses = add <4 x i64> %v3, %offsets
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK:      %addresses = add <4 x i64> %c, %inner
; CHECK-NEXT: [[SUM:%.*]] = add <4 x i64> zeroinitializer, %a
; CHECK-NEXT: {{%.*}} = add <4 x i64> [[SUM]], %c
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK-NEXT: [[VAR:%.*]] = add <4 x i64> %a, %c
; CHECK-NOT:  @__pseudo_scatter64_float
; CHECK:      call void @__pseudo_scatter_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> [[VAR]], i32 1, <4 x i64> zeroinitializer, <4 x float> %values, <4 x i32> %mask)
; CHECK-NEXT: ret void
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
; CHECK:      %s = insertelement <4 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x i64> <i64 8, i64 poison, i64 poison, i64 poison>, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <4 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison>
; CHECK:      %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: [[SUM:%.*]] = add <4 x i64> zeroinitializer, [[SPLAT]]
; CHECK-NEXT: {{%.*}} = add <4 x i64> [[SUM]], %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK-NEXT: [[VAR:%.*]] = add <4 x i64> [[SPLAT]], %offsets
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> [[VAR]], i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK:      %part = shl <4 x i64> %vec, <i64 2, i64 poison, i64 poison, i64 poison>
; CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <4 x i64> %part, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <4 x i64> %part, %s
; CHECK-NEXT: %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, [[SPLAT]]
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK:      [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i64 {{%.*}}
; CHECK-NEXT: [[ELT:%.*]] = load float, ptr [[PTR]], align 4
; CHECK-NEXT: [[INS:%.*]] = insertelement <4 x float> poison, float [[ELT]], i32 0
; CHECK-NEXT: [[V:%.*]] = shufflevector <4 x float> [[INS]], <4 x float> poison, <4 x i32> zeroinitializer
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK:      %addresses = add <4 x i64> %bases, %offsets
; CHECK-NEXT: [[CAST:%.*]] = zext i32 %b to i64
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 [[CAST]] to ptr
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK:      %addresses = add <4 x i64>
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
define <4 x float> @constant_splat(<4 x i64> %offsets, <4 x i32> %mask) {
  %addresses = add <4 x i64> splat (i64 ptrtoint (ptr @table to i64)), %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; One global plus per-lane displacements: the displacements start the
; offset sum.
; CHECK-LABEL: define <4 x float> @constant_lanes_plus(
; CHECK:      %addresses = add <4 x i64>
; CHECK-NEXT: {{%.*}} = add <4 x i64> <i64 0, i64 4, i64 8, i64 12>, %extra
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %extra, i32 1, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
define <4 x float> @constant_lanes_plus(<4 x i64> %extra, <4 x i32> %mask) {
  %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, %extra
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Linear displacements alone become a masked load.
; CHECK-LABEL: define <4 x float> @constant_lanes_only(
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i32 {{%.*}}
; A64:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i64 {{%.*}}
; CHECK-NEXT: [[V:%.*]] = call <4 x float> @__masked_load_float(ptr [[PTR]], <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
define <4 x float> @constant_lanes_only(<4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, <4 x i32> %mask)
  ret <4 x float> %v
}

; Addresses extracted from an aggregate built with insertvalue.
; CHECK-LABEL: define <4 x float> @aggregate(
; CHECK:      %extracted = extractvalue { <4 x i64>, <4 x i64> } %agg1, 1
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK-LABEL: define <4 x i32> @sext_index(
; CHECK:      %addresses = add <4 x i64> %bases, %wide
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, %wide
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; A32:        [[V:%.*]] = call <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr [[BASE]], <4 x i32> %scaled, i32 1, <4 x i32> {{%.*}}, <4 x i32> %mask)
; A64:        [[V:%.*]] = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr [[BASE]], <4 x i64> {{%.*}}, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x i32> [[V]]
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
; CHECK-LABEL: define <4 x float> @chain_to_load(
; CHECK:      %addresses = add <4 x i64>
; CHECK-NEXT: {{%.*}} = add <4 x i64> <i64 0, i64 4, i64 8, i64 12>, zeroinitializer
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i32 {{%.*}}
; A64:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i64 {{%.*}}
; CHECK-NEXT: [[V:%.*]] = load <4 x float>, ptr [[PTR]], align 4
; CHECK-NEXT: ret <4 x float> [[V]]
define <4 x float> @chain_to_load(<4 x i64> %extra) {
  %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, zeroinitializer
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> splat (i32 -1))
  ret <4 x float> %v
}

; A scatter under an all-off mask becomes a masked store that is removed.
; CHECK-LABEL: define void @store_removed(
; CHECK:      %addresses = add <4 x i64> %bases, <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT: {{%.*}} = add <4 x i64> zeroinitializer, <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT: inttoptr i64 %b to ptr
; CHECK-NOT:  call
; CHECK:      ret void
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
; CHECK:      [[BASE:%.*]] = inttoptr i64 %b to ptr, !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
; CHECK:      [[V:%.*]] = call <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr [[BASE]], <4 x i64> %offsets, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask), !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
; CHECK-NEXT: ret <4 x float> [[V]]
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
; CHECK:      %addresses = add <4 x i32> %bases, %offsets
; CHECK-NEXT: {{%.*}} = add <4 x i32> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i32 %b to ptr
; CHECK:      [[V:%.*]] = call <4 x i32> @__pseudo_gather_factored_base_offsets32_i32(ptr [[BASE]], <4 x i32> %offsets, i32 1, <4 x i32> zeroinitializer, <4 x i32> %mask)
; CHECK-NEXT: ret <4 x i32> [[V]]
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
