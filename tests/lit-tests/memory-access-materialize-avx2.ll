; Native gather path: on a target with a gather instruction the recovered
; base and offsets feed the non-factored base-offset form through the
; existing scale extraction and narrowing. The checks cover the
; materialized offsets, the base and the replacement call or load. CHECK
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
; CHECK:      %addresses = add <8 x i64> %v7, %offsets
; CHECK-NEXT: {{%.*}} = add <8 x i64> zeroinitializer, %offsets
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK:      [[V:%.*]] = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr [[BASE]], i32 1, <8 x i64> %offsets, <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> [[V]]
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
; CHECK:      %addresses = add <8 x i64> %c, %inner
; CHECK-NEXT: [[SUM:%.*]] = add <8 x i64> zeroinitializer, %a
; CHECK-NEXT: {{%.*}} = add <8 x i64> [[SUM]], %c
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK-NEXT: [[VAR:%.*]] = add <8 x i64> %a, %c
; CHECK-NOT:  @__pseudo_scatter64_float
; CHECK:      call void @__pseudo_scatter_factored_base_offsets64_float(ptr [[BASE]], <8 x i64> [[VAR]], i32 1, <8 x i64> zeroinitializer, <8 x float> %values, <8 x i32> %mask)
; CHECK-NEXT: ret void
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
; CHECK:      %s = insertelement <8 x i64> poison, i64 %b, i32 0
; CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <8 x i64> <i64 8, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>, <8 x i64> poison, <8 x i32> zeroinitializer
; CHECK-NEXT: %plus = add <8 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison, i64 poison>
; CHECK:      %addresses = add <8 x i64> %bases, %offsets
; CHECK-NEXT: [[SUM:%.*]] = add <8 x i64> zeroinitializer, [[SPLAT]]
; CHECK-DAG:  [[VAR:%.*]] = add <8 x i64> [[SPLAT]], %offsets
; CHECK-DAG:  {{%.*}} = add <8 x i64> [[SUM]], %offsets
; CHECK-DAG:  [[BASE:%.*]] = inttoptr i64 %b to ptr
; CHECK:      [[V:%.*]] = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr [[BASE]], i32 1, <8 x i64> [[VAR]], <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> [[V]]
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
; CHECK:      %addresses = add <8 x i64>
; CHECK-DAG:  [[VAR:%.*]] = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, %extra
; CHECK-DAG:  [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; CHECK:      [[V:%.*]] = call <8 x float> @__pseudo_gather_base_offsets64_float(ptr [[BASE]], i32 1, <8 x i64> [[VAR]], <8 x i32> %mask)
; CHECK-NEXT: ret <8 x float> [[V]]
define <8 x float> @constant_lanes_plus(<8 x i64> %extra, <8 x i32> %mask) {
  %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, %extra
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %v
}

; CHECK-LABEL: define <8 x i32> @sext_index(
; CHECK:      %addresses = add <8 x i64> %bases, %wide
; CHECK-NEXT: {{%.*}} = add <8 x i64> zeroinitializer, %wide
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 %b to ptr
; A32:        [[V:%.*]] = call <8 x i32> @__pseudo_gather_base_offsets32_i32(ptr [[BASE]], i32 1, <8 x i32> %scaled, <8 x i32> %mask)
; A64:        [[V:%.*]] = call <8 x i32> @__pseudo_gather_base_offsets64_i32(ptr [[BASE]], i32 1, <8 x i64> {{%.*}}, <8 x i32> %mask)
; CHECK-NEXT: ret <8 x i32> [[V]]
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

; CHECK-LABEL: define <8 x float> @chain_to_load(
; CHECK:      %addresses = add <8 x i64>
; CHECK-NEXT: {{%.*}} = add <8 x i64> <i64 0, i64 4, i64 8, i64 12, i64 16, i64 20, i64 24, i64 28>, zeroinitializer
; CHECK-NEXT: [[BASE:%.*]] = inttoptr i64 ptrtoint (ptr @table to i64) to ptr
; A32:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i64 {{%.*}}
; A64:        [[PTR:%.*]] = getelementptr i8, ptr [[BASE]], i64 0
; CHECK-NEXT: [[V:%.*]] = load <8 x float>, ptr [[PTR]], align 4
; CHECK-NEXT: ret <8 x float> [[V]]
define <8 x float> @chain_to_load(<8 x i64> %extra) {
  %addresses = add <8 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12), i64 add (i64 ptrtoint (ptr @table to i64), i64 16), i64 add (i64 ptrtoint (ptr @table to i64), i64 20), i64 add (i64 ptrtoint (ptr @table to i64), i64 24), i64 add (i64 ptrtoint (ptr @table to i64), i64 28)>, zeroinitializer
  %v = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> splat (i32 -1))
  ret <8 x float> %v
}
