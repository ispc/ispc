; Read-only recovery of a common base and per-lane offsets
; from the varying addresses of a general gather, scatter or prefetch. The
; query describes existing values; the second RUN line checks that printing
; every recovery leaves the module unchanged: no new instructions, operands,
; or declarations. A recovered numeric base is not pointer provenance.
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o /dev/null 2>&1 | FileCheck %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o - 2>/dev/null | FileCheck --check-prefix=IR %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o - 2>/dev/null | grep -c "^declare" | FileCheck --check-prefix=DECLS %s
; REQUIRES: X86_ENABLED

; The module keeps exactly its own declarations.
; DECLS: 5

@table = global [16 x float] zeroinitializer
@other = global [16 x float] zeroinitializer

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare void @__pseudo_scatter64_float(<4 x i64>, <4 x float>, <4 x i32>)
declare void @__pseudo_prefetch_read_varying_1(<4 x i64>, <4 x i32>)
declare <4 x i32> @__pseudo_gather32_i32(<4 x i32>, <4 x i32>)
declare <4 x i8> @__pseudo_gather64_i8(<4 x i64>, <4 x i32>)

; A uniform pointer broadcast through an insert chain, plus offsets.
; CHECK-LABEL: memory accesses of @insert_chain:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
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

; Nested adds give the terms innermost first; the base may be either operand.
; CHECK-LABEL: memory accesses of @nested_adds:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %a + <4 x i64> %c
; CHECK-NEXT: provenance: unknown
define void @nested_adds(ptr %base, <4 x i64> %a, <4 x i64> %c, <4 x float> %values, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %inner = add <4 x i64> %a, %bases
  %addresses = add <4 x i64> %c, %inner
  call void @__pseudo_scatter64_float(<4 x i64> %addresses, <4 x float> %values, <4 x i32> %mask)
  ret void
}

; An or that is an add (disjoint bits) is followed like an add.
; CHECK-LABEL: memory accesses of @disjoint_or:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %low
; CHECK-NEXT: provenance: unknown
define <4 x float> @disjoint_or(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %low = and <4 x i64> %offsets, splat (i64 3)
  %addresses = or disjoint <4 x i64> %bases, %low
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A lane-0 broadcast of base + <c, poison, ...> carries c as an offset.
; CHECK-LABEL: memory accesses of @broadcast_constant:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + lane 0 of <4 x i64> <i64 8, i64 poison, i64 poison, i64 poison> + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
define <4 x float> @broadcast_constant(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %plus = add <4 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison>
  %bases = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Or lane 0 of another operation that computes part of the offset.
; CHECK-LABEL: memory accesses of @broadcast_part:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + lane 0 of <4 x i64> %part
; CHECK-NEXT: provenance: unknown
define <4 x float> @broadcast_part(ptr %base, <4 x i64> %vec, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %part = shl <4 x i64> %vec, <i64 2, i64 poison, i64 poison, i64 poison>
  %plus = add <4 x i64> %part, %s
  %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A broadcast whose add has no recognized offset part is not matched.
; CHECK-LABEL: memory accesses of @broadcast_unrecognized:
; CHECK: common base: unknown
define <4 x float> @broadcast_unrecognized(ptr %base, i64 %x, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %t = insertelement <4 x i64> poison, i64 %x, i32 0
  %plus = add <4 x i64> %t, %s
  %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; Vector casts of the addresses are replayed on the scalar base.
; CHECK-LABEL: memory accesses of @cast_chain:
; CHECK: common base: i32 %b (ptrtoint), zext to i64
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
define <4 x float> @cast_chain(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i32
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %bases32 = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %bases = zext <4 x i32> %bases32 to <4 x i64>
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A cast that changes the lane count has no scalar counterpart: two equal
; i128 lanes do not make four equal i64 lanes.
; CHECK-LABEL: memory accesses of @lane_changing_cast:
; CHECK: common base: unknown
define <4 x i8> @lane_changing_cast(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i128
  %s0 = insertelement <2 x i128> poison, i128 %b, i32 0
  %s1 = insertelement <2 x i128> %s0, i128 %b, i32 1
  %bases = bitcast <2 x i128> %s1 to <4 x i64>
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x i8> @__pseudo_gather64_i8(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x i8> %v
}

; The cast of a splat that is not a pointer in the end is not a base.
; CHECK-LABEL: memory accesses of @cast_of_integer:
; CHECK: common base: unknown
define <4 x float> @cast_of_integer(i32 %n, <4 x i64> %offsets, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %n, i32 0
  %ns = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %wide = zext <4 x i32> %ns to <4 x i64>
  %addresses = add <4 x i64> %wide, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; An integer loaded from memory is accepted as a pointer: the legacy
; heuristic for a pointer that LLVM loaded as i64. Its origin is recorded.
; CHECK-LABEL: memory accesses of @loaded_integer:
; CHECK: common base: i64 %p (loaded integer)
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
define <4 x float> @loaded_integer(ptr %slot, <4 x i64> %offsets, <4 x i32> %mask) {
  %p = load i64, ptr %slot
  %s = insertelement <4 x i64> poison, i64 %p, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; A constant splat of a global's address.
; CHECK-LABEL: memory accesses of @constant_splat:
; CHECK: common base: i64 ptrtoint (ptr @table to i64) (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
define <4 x float> @constant_splat(<4 x i64> %offsets, <4 x i32> %mask) {
  %addresses = add <4 x i64> splat (i64 ptrtoint (ptr @table to i64)), %offsets
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; One global plus per-lane constant displacements: the lanes are kept as
; constants at their width, no LLVM constant is created by the query.
; CHECK-LABEL: memory accesses of @constant_lanes:
; CHECK: common base: i64 ptrtoint (ptr @table to i64) (ptrtoint)
; CHECK-NEXT: offsets: lanes 0 4 8 -4 (i64)
; CHECK-NEXT: provenance: unknown
define <4 x float> @constant_lanes(<4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 -4)>, <4 x i32> %mask)
  ret <4 x float> %v
}

; Constant lanes plus further terms: the lanes start the sum.
; CHECK-LABEL: memory accesses of @constant_lanes_plus:
; CHECK: common base: i64 ptrtoint (ptr @table to i64) (ptrtoint)
; CHECK-NEXT: offsets: lanes 0 4 8 12 (i64) + <4 x i64> %extra
; CHECK-NEXT: provenance: unknown
define <4 x float> @constant_lanes_plus(<4 x i64> %extra, <4 x i32> %mask) {
  %addresses = add <4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, %extra
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; 32-bit lane constants keep their width.
; CHECK-LABEL: memory accesses of @constant_lanes32:
; CHECK: common base: i32 ptrtoint (ptr @table to i32) (ptrtoint)
; CHECK-NEXT: offsets: lanes 0 -8 0 -8 (i32) + <4 x i32> %extra
define <4 x i32> @constant_lanes32(<4 x i32> %extra, <4 x i32> %mask) {
  %addresses = add <4 x i32> <i32 ptrtoint (ptr @table to i32), i32 add (i32 ptrtoint (ptr @table to i32), i32 -8), i32 ptrtoint (ptr @table to i32), i32 add (i32 ptrtoint (ptr @table to i32), i32 -8)>, %extra
  %v = call <4 x i32> @__pseudo_gather32_i32(<4 x i32> %addresses, <4 x i32> %mask)
  ret <4 x i32> %v
}

; Lanes from two globals have no common base.
; CHECK-LABEL: memory accesses of @constant_lanes_two_globals:
; CHECK: common base: unknown
define <4 x float> @constant_lanes_two_globals(<4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 ptrtoint (ptr @table to i64), i64 ptrtoint (ptr @other to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 12)>, <4 x i32> %mask)
  ret <4 x float> %v
}

; The addresses extracted from an aggregate built with insertvalue.
; CHECK-LABEL: memory accesses of @aggregate:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
; CHECK-NEXT: provenance: unknown
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

; An aggregate without the inserted element is unknown.
; CHECK-LABEL: memory accesses of @aggregate_missing:
; CHECK: common base: unknown
define <4 x float> @aggregate_missing({ <4 x i64>, <4 x i64> } %agg, <4 x i32> %mask) {
  %extracted = extractvalue { <4 x i64>, <4 x i64> } %agg, 1
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %extracted, <4 x i32> %mask)
  ret <4 x float> %v
}

; A prefetch is decoded and its addresses are described like a gather's, but
; nothing converts it: there is no vector prefetch instruction.
; CHECK-LABEL: memory accesses of @prefetch:
; CHECK: prefetch __pseudo_prefetch_read_varying_1:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %offsets
define void @prefetch(ptr %base, <4 x i64> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  call void @__pseudo_prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> %mask)
  ret void
}

; 32-bit addresses on a 32-bit form.
; CHECK-LABEL: memory accesses of @addresses32:
; CHECK: common base: i32 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i32> %offsets
define <4 x i32> @addresses32(ptr %base, <4 x i32> %offsets, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i32
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %bases = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i32> %bases, %offsets
  %v = call <4 x i32> @__pseudo_gather32_i32(<4 x i32> %addresses, <4 x i32> %mask)
  ret <4 x i32> %v
}

; Unknown addresses and a mul are not followed.
; CHECK-LABEL: memory accesses of @unknown:
; CHECK: common base: unknown
; CHECK: common base: unknown
define <4 x float> @unknown(ptr %base, <4 x i64> %addresses, <4 x i64> %factors, <4 x i32> %mask) {
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %scaled = mul <4 x i64> %bases, %factors
  %w = call <4 x float> @__pseudo_gather64_float(<4 x i64> %scaled, <4 x i32> %mask)
  ret <4 x float> %v
}

; The query visits a bounded number of values. Following this tree of shared
; adds, which has no base, visits more than 500 values before the base in the
; second operand is reached; the legacy matcher followed them all, the query
; reports unknown.
; CHECK-LABEL: memory accesses of @exhausted_budget:
; CHECK: common base: unknown
define <4 x float> @exhausted_budget(ptr %base, <4 x i64> %o, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %f1 = add <4 x i64> %o, %o
  %f2 = add <4 x i64> %f1, %f1
  %f3 = add <4 x i64> %f2, %f2
  %f4 = add <4 x i64> %f3, %f3
  %f5 = add <4 x i64> %f4, %f4
  %f6 = add <4 x i64> %f5, %f5
  %f7 = add <4 x i64> %f6, %f6
  %addresses = add <4 x i64> %f7, %bases
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; An insert chain longer than the budget is unknown, too.
; CHECK-LABEL: memory accesses of @long_insert_chain:
; CHECK: common base: unknown
define <4 x float> @long_insert_chain(ptr %base, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %i0 = insertelement <4 x i64> poison, i64 %b, i32 0
  %i1 = insertelement <4 x i64> %i0, i64 %b, i32 1
  %i2 = insertelement <4 x i64> %i1, i64 %b, i32 2
  %i3 = insertelement <4 x i64> %i2, i64 %b, i32 3
  %i4 = insertelement <4 x i64> %i3, i64 %b, i32 0
  %i5 = insertelement <4 x i64> %i4, i64 %b, i32 1
  %i6 = insertelement <4 x i64> %i5, i64 %b, i32 2
  %i7 = insertelement <4 x i64> %i6, i64 %b, i32 3
  %i8 = insertelement <4 x i64> %i7, i64 %b, i32 0
  %i9 = insertelement <4 x i64> %i8, i64 %b, i32 1
  %i10 = insertelement <4 x i64> %i9, i64 %b, i32 2
  %i11 = insertelement <4 x i64> %i10, i64 %b, i32 3
  %i12 = insertelement <4 x i64> %i11, i64 %b, i32 0
  %i13 = insertelement <4 x i64> %i12, i64 %b, i32 1
  %i14 = insertelement <4 x i64> %i13, i64 %b, i32 2
  %i15 = insertelement <4 x i64> %i14, i64 %b, i32 3
  %i16 = insertelement <4 x i64> %i15, i64 %b, i32 0
  %i17 = insertelement <4 x i64> %i16, i64 %b, i32 1
  %i18 = insertelement <4 x i64> %i17, i64 %b, i32 2
  %i19 = insertelement <4 x i64> %i18, i64 %b, i32 3
  %i20 = insertelement <4 x i64> %i19, i64 %b, i32 0
  %i21 = insertelement <4 x i64> %i20, i64 %b, i32 1
  %i22 = insertelement <4 x i64> %i21, i64 %b, i32 2
  %i23 = insertelement <4 x i64> %i22, i64 %b, i32 3
  %i24 = insertelement <4 x i64> %i23, i64 %b, i32 0
  %i25 = insertelement <4 x i64> %i24, i64 %b, i32 1
  %i26 = insertelement <4 x i64> %i25, i64 %b, i32 2
  %i27 = insertelement <4 x i64> %i26, i64 %b, i32 3
  %i28 = insertelement <4 x i64> %i27, i64 %b, i32 0
  %i29 = insertelement <4 x i64> %i28, i64 %b, i32 1
  %i30 = insertelement <4 x i64> %i29, i64 %b, i32 2
  %i31 = insertelement <4 x i64> %i30, i64 %b, i32 3
  %i32 = insertelement <4 x i64> %i31, i64 %b, i32 0
  %i33 = insertelement <4 x i64> %i32, i64 %b, i32 1
  %i34 = insertelement <4 x i64> %i33, i64 %b, i32 2
  %i35 = insertelement <4 x i64> %i34, i64 %b, i32 3
  %i36 = insertelement <4 x i64> %i35, i64 %b, i32 0
  %i37 = insertelement <4 x i64> %i36, i64 %b, i32 1
  %i38 = insertelement <4 x i64> %i37, i64 %b, i32 2
  %i39 = insertelement <4 x i64> %i38, i64 %b, i32 3
  %i40 = insertelement <4 x i64> %i39, i64 %b, i32 0
  %i41 = insertelement <4 x i64> %i40, i64 %b, i32 1
  %i42 = insertelement <4 x i64> %i41, i64 %b, i32 2
  %i43 = insertelement <4 x i64> %i42, i64 %b, i32 3
  %i44 = insertelement <4 x i64> %i43, i64 %b, i32 0
  %i45 = insertelement <4 x i64> %i44, i64 %b, i32 1
  %i46 = insertelement <4 x i64> %i45, i64 %b, i32 2
  %i47 = insertelement <4 x i64> %i46, i64 %b, i32 3
  %i48 = insertelement <4 x i64> %i47, i64 %b, i32 0
  %i49 = insertelement <4 x i64> %i48, i64 %b, i32 1
  %i50 = insertelement <4 x i64> %i49, i64 %b, i32 2
  %i51 = insertelement <4 x i64> %i50, i64 %b, i32 3
  %i52 = insertelement <4 x i64> %i51, i64 %b, i32 0
  %i53 = insertelement <4 x i64> %i52, i64 %b, i32 1
  %i54 = insertelement <4 x i64> %i53, i64 %b, i32 2
  %i55 = insertelement <4 x i64> %i54, i64 %b, i32 3
  %i56 = insertelement <4 x i64> %i55, i64 %b, i32 0
  %i57 = insertelement <4 x i64> %i56, i64 %b, i32 1
  %i58 = insertelement <4 x i64> %i57, i64 %b, i32 2
  %i59 = insertelement <4 x i64> %i58, i64 %b, i32 3
  %i60 = insertelement <4 x i64> %i59, i64 %b, i32 0
  %i61 = insertelement <4 x i64> %i60, i64 %b, i32 1
  %i62 = insertelement <4 x i64> %i61, i64 %b, i32 2
  %i63 = insertelement <4 x i64> %i62, i64 %b, i32 3
  %i64 = insertelement <4 x i64> %i63, i64 %b, i32 0
  %i65 = insertelement <4 x i64> %i64, i64 %b, i32 1
  %i66 = insertelement <4 x i64> %i65, i64 %b, i32 2
  %i67 = insertelement <4 x i64> %i66, i64 %b, i32 3
  %i68 = insertelement <4 x i64> %i67, i64 %b, i32 0
  %i69 = insertelement <4 x i64> %i68, i64 %b, i32 1
  %i70 = insertelement <4 x i64> %i69, i64 %b, i32 2
  %i71 = insertelement <4 x i64> %i70, i64 %b, i32 3
  %i72 = insertelement <4 x i64> %i71, i64 %b, i32 0
  %i73 = insertelement <4 x i64> %i72, i64 %b, i32 1
  %i74 = insertelement <4 x i64> %i73, i64 %b, i32 2
  %i75 = insertelement <4 x i64> %i74, i64 %b, i32 3
  %i76 = insertelement <4 x i64> %i75, i64 %b, i32 0
  %i77 = insertelement <4 x i64> %i76, i64 %b, i32 1
  %i78 = insertelement <4 x i64> %i77, i64 %b, i32 2
  %i79 = insertelement <4 x i64> %i78, i64 %b, i32 3
  %i80 = insertelement <4 x i64> %i79, i64 %b, i32 0
  %i81 = insertelement <4 x i64> %i80, i64 %b, i32 1
  %i82 = insertelement <4 x i64> %i81, i64 %b, i32 2
  %i83 = insertelement <4 x i64> %i82, i64 %b, i32 3
  %i84 = insertelement <4 x i64> %i83, i64 %b, i32 0
  %i85 = insertelement <4 x i64> %i84, i64 %b, i32 1
  %i86 = insertelement <4 x i64> %i85, i64 %b, i32 2
  %i87 = insertelement <4 x i64> %i86, i64 %b, i32 3
  %i88 = insertelement <4 x i64> %i87, i64 %b, i32 0
  %i89 = insertelement <4 x i64> %i88, i64 %b, i32 1
  %i90 = insertelement <4 x i64> %i89, i64 %b, i32 2
  %i91 = insertelement <4 x i64> %i90, i64 %b, i32 3
  %i92 = insertelement <4 x i64> %i91, i64 %b, i32 0
  %i93 = insertelement <4 x i64> %i92, i64 %b, i32 1
  %i94 = insertelement <4 x i64> %i93, i64 %b, i32 2
  %i95 = insertelement <4 x i64> %i94, i64 %b, i32 3
  %i96 = insertelement <4 x i64> %i95, i64 %b, i32 0
  %i97 = insertelement <4 x i64> %i96, i64 %b, i32 1
  %i98 = insertelement <4 x i64> %i97, i64 %b, i32 2
  %i99 = insertelement <4 x i64> %i98, i64 %b, i32 3
  %i100 = insertelement <4 x i64> %i99, i64 %b, i32 0
  %i101 = insertelement <4 x i64> %i100, i64 %b, i32 1
  %i102 = insertelement <4 x i64> %i101, i64 %b, i32 2
  %i103 = insertelement <4 x i64> %i102, i64 %b, i32 3
  %i104 = insertelement <4 x i64> %i103, i64 %b, i32 0
  %i105 = insertelement <4 x i64> %i104, i64 %b, i32 1
  %i106 = insertelement <4 x i64> %i105, i64 %b, i32 2
  %i107 = insertelement <4 x i64> %i106, i64 %b, i32 3
  %i108 = insertelement <4 x i64> %i107, i64 %b, i32 0
  %i109 = insertelement <4 x i64> %i108, i64 %b, i32 1
  %i110 = insertelement <4 x i64> %i109, i64 %b, i32 2
  %i111 = insertelement <4 x i64> %i110, i64 %b, i32 3
  %i112 = insertelement <4 x i64> %i111, i64 %b, i32 0
  %i113 = insertelement <4 x i64> %i112, i64 %b, i32 1
  %i114 = insertelement <4 x i64> %i113, i64 %b, i32 2
  %i115 = insertelement <4 x i64> %i114, i64 %b, i32 3
  %i116 = insertelement <4 x i64> %i115, i64 %b, i32 0
  %i117 = insertelement <4 x i64> %i116, i64 %b, i32 1
  %i118 = insertelement <4 x i64> %i117, i64 %b, i32 2
  %i119 = insertelement <4 x i64> %i118, i64 %b, i32 3
  %i120 = insertelement <4 x i64> %i119, i64 %b, i32 0
  %i121 = insertelement <4 x i64> %i120, i64 %b, i32 1
  %i122 = insertelement <4 x i64> %i121, i64 %b, i32 2
  %i123 = insertelement <4 x i64> %i122, i64 %b, i32 3
  %i124 = insertelement <4 x i64> %i123, i64 %b, i32 0
  %i125 = insertelement <4 x i64> %i124, i64 %b, i32 1
  %i126 = insertelement <4 x i64> %i125, i64 %b, i32 2
  %i127 = insertelement <4 x i64> %i126, i64 %b, i32 3
  %i128 = insertelement <4 x i64> %i127, i64 %b, i32 0
  %i129 = insertelement <4 x i64> %i128, i64 %b, i32 1
  %i130 = insertelement <4 x i64> %i129, i64 %b, i32 2
  %i131 = insertelement <4 x i64> %i130, i64 %b, i32 3
  %i132 = insertelement <4 x i64> %i131, i64 %b, i32 0
  %i133 = insertelement <4 x i64> %i132, i64 %b, i32 1
  %i134 = insertelement <4 x i64> %i133, i64 %b, i32 2
  %i135 = insertelement <4 x i64> %i134, i64 %b, i32 3
  %i136 = insertelement <4 x i64> %i135, i64 %b, i32 0
  %i137 = insertelement <4 x i64> %i136, i64 %b, i32 1
  %i138 = insertelement <4 x i64> %i137, i64 %b, i32 2
  %i139 = insertelement <4 x i64> %i138, i64 %b, i32 3
  %i140 = insertelement <4 x i64> %i139, i64 %b, i32 0
  %i141 = insertelement <4 x i64> %i140, i64 %b, i32 1
  %i142 = insertelement <4 x i64> %i141, i64 %b, i32 2
  %i143 = insertelement <4 x i64> %i142, i64 %b, i32 3
  %i144 = insertelement <4 x i64> %i143, i64 %b, i32 0
  %i145 = insertelement <4 x i64> %i144, i64 %b, i32 1
  %i146 = insertelement <4 x i64> %i145, i64 %b, i32 2
  %i147 = insertelement <4 x i64> %i146, i64 %b, i32 3
  %i148 = insertelement <4 x i64> %i147, i64 %b, i32 0
  %i149 = insertelement <4 x i64> %i148, i64 %b, i32 1
  %i150 = insertelement <4 x i64> %i149, i64 %b, i32 2
  %i151 = insertelement <4 x i64> %i150, i64 %b, i32 3
  %i152 = insertelement <4 x i64> %i151, i64 %b, i32 0
  %i153 = insertelement <4 x i64> %i152, i64 %b, i32 1
  %i154 = insertelement <4 x i64> %i153, i64 %b, i32 2
  %i155 = insertelement <4 x i64> %i154, i64 %b, i32 3
  %i156 = insertelement <4 x i64> %i155, i64 %b, i32 0
  %i157 = insertelement <4 x i64> %i156, i64 %b, i32 1
  %i158 = insertelement <4 x i64> %i157, i64 %b, i32 2
  %i159 = insertelement <4 x i64> %i158, i64 %b, i32 3
  %i160 = insertelement <4 x i64> %i159, i64 %b, i32 0
  %i161 = insertelement <4 x i64> %i160, i64 %b, i32 1
  %i162 = insertelement <4 x i64> %i161, i64 %b, i32 2
  %i163 = insertelement <4 x i64> %i162, i64 %b, i32 3
  %i164 = insertelement <4 x i64> %i163, i64 %b, i32 0
  %i165 = insertelement <4 x i64> %i164, i64 %b, i32 1
  %i166 = insertelement <4 x i64> %i165, i64 %b, i32 2
  %i167 = insertelement <4 x i64> %i166, i64 %b, i32 3
  %i168 = insertelement <4 x i64> %i167, i64 %b, i32 0
  %i169 = insertelement <4 x i64> %i168, i64 %b, i32 1
  %i170 = insertelement <4 x i64> %i169, i64 %b, i32 2
  %i171 = insertelement <4 x i64> %i170, i64 %b, i32 3
  %i172 = insertelement <4 x i64> %i171, i64 %b, i32 0
  %i173 = insertelement <4 x i64> %i172, i64 %b, i32 1
  %i174 = insertelement <4 x i64> %i173, i64 %b, i32 2
  %i175 = insertelement <4 x i64> %i174, i64 %b, i32 3
  %i176 = insertelement <4 x i64> %i175, i64 %b, i32 0
  %i177 = insertelement <4 x i64> %i176, i64 %b, i32 1
  %i178 = insertelement <4 x i64> %i177, i64 %b, i32 2
  %i179 = insertelement <4 x i64> %i178, i64 %b, i32 3
  %i180 = insertelement <4 x i64> %i179, i64 %b, i32 0
  %i181 = insertelement <4 x i64> %i180, i64 %b, i32 1
  %i182 = insertelement <4 x i64> %i181, i64 %b, i32 2
  %i183 = insertelement <4 x i64> %i182, i64 %b, i32 3
  %i184 = insertelement <4 x i64> %i183, i64 %b, i32 0
  %i185 = insertelement <4 x i64> %i184, i64 %b, i32 1
  %i186 = insertelement <4 x i64> %i185, i64 %b, i32 2
  %i187 = insertelement <4 x i64> %i186, i64 %b, i32 3
  %i188 = insertelement <4 x i64> %i187, i64 %b, i32 0
  %i189 = insertelement <4 x i64> %i188, i64 %b, i32 1
  %i190 = insertelement <4 x i64> %i189, i64 %b, i32 2
  %i191 = insertelement <4 x i64> %i190, i64 %b, i32 3
  %i192 = insertelement <4 x i64> %i191, i64 %b, i32 0
  %i193 = insertelement <4 x i64> %i192, i64 %b, i32 1
  %i194 = insertelement <4 x i64> %i193, i64 %b, i32 2
  %i195 = insertelement <4 x i64> %i194, i64 %b, i32 3
  %i196 = insertelement <4 x i64> %i195, i64 %b, i32 0
  %i197 = insertelement <4 x i64> %i196, i64 %b, i32 1
  %i198 = insertelement <4 x i64> %i197, i64 %b, i32 2
  %i199 = insertelement <4 x i64> %i198, i64 %b, i32 3
  %i200 = insertelement <4 x i64> %i199, i64 %b, i32 0
  %i201 = insertelement <4 x i64> %i200, i64 %b, i32 1
  %i202 = insertelement <4 x i64> %i201, i64 %b, i32 2
  %i203 = insertelement <4 x i64> %i202, i64 %b, i32 3
  %i204 = insertelement <4 x i64> %i203, i64 %b, i32 0
  %i205 = insertelement <4 x i64> %i204, i64 %b, i32 1
  %i206 = insertelement <4 x i64> %i205, i64 %b, i32 2
  %i207 = insertelement <4 x i64> %i206, i64 %b, i32 3
  %i208 = insertelement <4 x i64> %i207, i64 %b, i32 0
  %i209 = insertelement <4 x i64> %i208, i64 %b, i32 1
  %i210 = insertelement <4 x i64> %i209, i64 %b, i32 2
  %i211 = insertelement <4 x i64> %i210, i64 %b, i32 3
  %i212 = insertelement <4 x i64> %i211, i64 %b, i32 0
  %i213 = insertelement <4 x i64> %i212, i64 %b, i32 1
  %i214 = insertelement <4 x i64> %i213, i64 %b, i32 2
  %i215 = insertelement <4 x i64> %i214, i64 %b, i32 3
  %i216 = insertelement <4 x i64> %i215, i64 %b, i32 0
  %i217 = insertelement <4 x i64> %i216, i64 %b, i32 1
  %i218 = insertelement <4 x i64> %i217, i64 %b, i32 2
  %i219 = insertelement <4 x i64> %i218, i64 %b, i32 3
  %i220 = insertelement <4 x i64> %i219, i64 %b, i32 0
  %i221 = insertelement <4 x i64> %i220, i64 %b, i32 1
  %i222 = insertelement <4 x i64> %i221, i64 %b, i32 2
  %i223 = insertelement <4 x i64> %i222, i64 %b, i32 3
  %i224 = insertelement <4 x i64> %i223, i64 %b, i32 0
  %i225 = insertelement <4 x i64> %i224, i64 %b, i32 1
  %i226 = insertelement <4 x i64> %i225, i64 %b, i32 2
  %i227 = insertelement <4 x i64> %i226, i64 %b, i32 3
  %i228 = insertelement <4 x i64> %i227, i64 %b, i32 0
  %i229 = insertelement <4 x i64> %i228, i64 %b, i32 1
  %i230 = insertelement <4 x i64> %i229, i64 %b, i32 2
  %i231 = insertelement <4 x i64> %i230, i64 %b, i32 3
  %i232 = insertelement <4 x i64> %i231, i64 %b, i32 0
  %i233 = insertelement <4 x i64> %i232, i64 %b, i32 1
  %i234 = insertelement <4 x i64> %i233, i64 %b, i32 2
  %i235 = insertelement <4 x i64> %i234, i64 %b, i32 3
  %i236 = insertelement <4 x i64> %i235, i64 %b, i32 0
  %i237 = insertelement <4 x i64> %i236, i64 %b, i32 1
  %i238 = insertelement <4 x i64> %i237, i64 %b, i32 2
  %i239 = insertelement <4 x i64> %i238, i64 %b, i32 3
  %i240 = insertelement <4 x i64> %i239, i64 %b, i32 0
  %i241 = insertelement <4 x i64> %i240, i64 %b, i32 1
  %i242 = insertelement <4 x i64> %i241, i64 %b, i32 2
  %i243 = insertelement <4 x i64> %i242, i64 %b, i32 3
  %i244 = insertelement <4 x i64> %i243, i64 %b, i32 0
  %i245 = insertelement <4 x i64> %i244, i64 %b, i32 1
  %i246 = insertelement <4 x i64> %i245, i64 %b, i32 2
  %i247 = insertelement <4 x i64> %i246, i64 %b, i32 3
  %i248 = insertelement <4 x i64> %i247, i64 %b, i32 0
  %i249 = insertelement <4 x i64> %i248, i64 %b, i32 1
  %i250 = insertelement <4 x i64> %i249, i64 %b, i32 2
  %i251 = insertelement <4 x i64> %i250, i64 %b, i32 3
  %i252 = insertelement <4 x i64> %i251, i64 %b, i32 0
  %i253 = insertelement <4 x i64> %i252, i64 %b, i32 1
  %i254 = insertelement <4 x i64> %i253, i64 %b, i32 2
  %i255 = insertelement <4 x i64> %i254, i64 %b, i32 3
  %i256 = insertelement <4 x i64> %i255, i64 %b, i32 0
  %i257 = insertelement <4 x i64> %i256, i64 %b, i32 1
  %i258 = insertelement <4 x i64> %i257, i64 %b, i32 2
  %i259 = insertelement <4 x i64> %i258, i64 %b, i32 3
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %i259, <4 x i32> %mask)
  ret <4 x float> %v
}

; The same tree two levels smaller fits the budget.
; CHECK-LABEL: memory accesses of @within_budget:
; CHECK: common base: i64 %b (ptrtoint)
; CHECK-NEXT: offsets: 0 + <4 x i64> %f5
define <4 x float> @within_budget(ptr %base, <4 x i64> %o, <4 x i32> %mask) {
  %b = ptrtoint ptr %base to i64
  %s = insertelement <4 x i64> poison, i64 %b, i32 0
  %bases = shufflevector <4 x i64> %s, <4 x i64> poison, <4 x i32> zeroinitializer
  %f1 = add <4 x i64> %o, %o
  %f2 = add <4 x i64> %f1, %f1
  %f3 = add <4 x i64> %f2, %f2
  %f4 = add <4 x i64> %f3, %f3
  %f5 = add <4 x i64> %f4, %f4
  %addresses = add <4 x i64> %f5, %bases
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %v
}

; The module after printing: identical to the input.
; IR-LABEL: define <4 x float> @broadcast_constant(
; IR-NEXT: %b = ptrtoint ptr %base to i64
; IR-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; IR-NEXT: %plus = add <4 x i64> %s, <i64 8, i64 poison, i64 poison, i64 poison>
; IR-NEXT: %bases = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
; IR-NEXT: %addresses = add <4 x i64> %bases, %offsets
; IR-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
; IR-LABEL: define <4 x float> @broadcast_part(
; IR-NEXT: %b = ptrtoint ptr %base to i64
; IR-NEXT: %s = insertelement <4 x i64> poison, i64 %b, i32 0
; IR-NEXT: %part = shl <4 x i64> %vec, <i64 2, i64 poison, i64 poison, i64 poison>
; IR-NEXT: %plus = add <4 x i64> %part, %s
; IR-NEXT: %addresses = shufflevector <4 x i64> %plus, <4 x i64> poison, <4 x i32> zeroinitializer
; IR-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
; IR-LABEL: define <4 x float> @cast_chain(
; IR-NEXT: %b = ptrtoint ptr %base to i32
; IR-NEXT: %s = insertelement <4 x i32> poison, i32 %b, i32 0
; IR-NEXT: %bases32 = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
; IR-NEXT: %bases = zext <4 x i32> %bases32 to <4 x i64>
; IR-NEXT: %addresses = add <4 x i64> %bases, %offsets
; IR-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
; IR-LABEL: define <4 x float> @constant_lanes(
; IR-NEXT: %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> <i64 ptrtoint (ptr @table to i64), i64 add (i64 ptrtoint (ptr @table to i64), i64 4), i64 add (i64 ptrtoint (ptr @table to i64), i64 8), i64 add (i64 ptrtoint (ptr @table to i64), i64 -4)>, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %v
