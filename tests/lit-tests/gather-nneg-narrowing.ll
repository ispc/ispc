; A 32-bit native gather sign extends its index before scaling. On defined
; lanes, zext nneg has the same value as sext; an ordinary zext does not.
;
; RUN: %{ispc-opt} --target=avx2-i32x8 --addressing=32 --passes=improve-memory-ops %s -o - | FileCheck %s --check-prefix=A32
; RUN: %{ispc-opt} --target=avx2-i32x8 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck %s --check-prefix=A64
; REQUIRES: X86_ENABLED, LLVM_20_0+

declare <8 x float> @__pseudo_gather64_float(<8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <8 x i64>, <8 x i32>)
declare <8 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <8 x i32>, <8 x i32>)
declare <8 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <8 x i32>, i32, <8 x i32>, <8 x i32>)
declare <8 x float> @__masked_load_float(ptr, <8 x i32>)

; A32-LABEL: define <8 x float> @signed_index(
; A32: call <8 x float> @__pseudo_gather_base_offsets32_float(ptr {{[^,]+}}, i32 1, <8 x i32> %index,
; A64-LABEL: define <8 x float> @signed_index(
; A64: call <8 x float> @__pseudo_gather_base_offsets64_float(
define <8 x float> @signed_index(ptr %base, <8 x i32> %index, <8 x i32> %mask) {
  %wide = sext <8 x i32> %index to <8 x i64>
  %baseint = ptrtoint ptr %base to i64
  %insert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %insert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %wide
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %value
}

; A32-LABEL: define <8 x float> @nonnegative_index(
; A32: call <8 x float> @__pseudo_gather_base_offsets32_float(ptr {{[^,]+}}, i32 1, <8 x i32> %index,
; A64-LABEL: define <8 x float> @nonnegative_index(
; A64: call <8 x float> @__pseudo_gather_base_offsets64_float(
define <8 x float> @nonnegative_index(ptr %base, <8 x i32> %index, <8 x i32> %mask) {
  %wide = zext nneg <8 x i32> %index to <8 x i64>
  %baseint = ptrtoint ptr %base to i64
  %insert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %insert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %wide
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %value
}

; A32-LABEL: define <8 x float> @unsigned_index(
; A32: call <8 x float> @__pseudo_gather_base_offsets64_float(
; A64-LABEL: define <8 x float> @unsigned_index(
; A64: call <8 x float> @__pseudo_gather_base_offsets64_float(
define <8 x float> @unsigned_index(ptr %base, <8 x i32> %index, <8 x i32> %mask) {
  %wide = zext <8 x i32> %index to <8 x i64>
  %baseint = ptrtoint ptr %base to i64
  %insert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %insert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %wide
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %value
}

; The full offset of a load/store rewrite is built as the native gather
; computes it: index sign extended to i64, then scaled. A bounded index and an
; unbounded one both become a load, without an i32 multiply.
; A32-LABEL: define <8 x float> @bounded_scaled_broadcast(
; A32-NOT: call <8 x float> @__pseudo_gather
; A32: %[[B:.*]] = sext i32 %bounded to i64
; A32: %[[S:.*]] = mul nsw i64 4, %[[B]]
; A32: getelementptr i8, ptr {{[^,]+}}, i64 %[[S]]
; A32: load float, ptr
; A32-NOT: call <8 x float> @__pseudo_gather
; A32: ret <8 x float>
define <8 x float> @bounded_scaled_broadcast(ptr %base, i32 %index) {
  %bounded = and i32 %index, 536870911
  %insert = insertelement <8 x i32> poison, i32 %bounded, i32 0
  %indices = shufflevector <8 x i32> %insert, <8 x i32> poison, <8 x i32> zeroinitializer
  %wide = zext nneg <8 x i32> %indices to <8 x i64>
  %offsets = mul nsw <8 x i64> %wide, splat (i64 4)
  %baseint = ptrtoint ptr %base to i64
  %baseinsert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %baseinsert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %offsets
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> splat (i32 -1))
  ret <8 x float> %value
}

; A32-LABEL: define <8 x float> @scaled_linear(
; A32-NOT: mul i32
; A32: %[[I:.*]] = sext i32 %{{.*}} to i64
; A32: %[[S:.*]] = mul nsw i64 4, %[[I]]
; A32: %[[P:.*]] = getelementptr i8, ptr %base, i64 %[[S]]
; A32: call <8 x float> @__masked_load_float(ptr %[[P]], <8 x i32> %mask)
; A32: ret <8 x float>
; The add is nsw: lanes that wrap from INT32_MAX to INT32_MIN are not linear
; after sign extension (address-gap-wrap.ispc).
define <8 x float> @scaled_linear(ptr %base, i32 %index, <8 x i32> %mask) {
  %insert = insertelement <8 x i32> poison, i32 %index, i32 0
  %broadcast = shufflevector <8 x i32> %insert, <8 x i32> poison, <8 x i32> zeroinitializer
  %indices = add nsw <8 x i32> %broadcast, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %value = call <8 x float> @__pseudo_gather_base_offsets32_float(ptr %base, i32 4, <8 x i32> %indices, <8 x i32> %mask)
  ret <8 x float> %value
}

; An unscaled i32 index already fits; this ordinary broadcast still loads.
; A32-LABEL: define <8 x float> @unscaled_broadcast(
; A32: load float, ptr
; A32: ret <8 x float>
define <8 x float> @unscaled_broadcast(ptr %base, i32 %index) {
  %insert = insertelement <8 x i32> poison, i32 %index, i32 0
  %indices = shufflevector <8 x i32> %insert, <8 x i32> poison, <8 x i32> zeroinitializer
  %value = call <8 x float> @__pseudo_gather_base_offsets32_float(ptr %base, i32 1, <8 x i32> %indices,
                                                                   <8 x i32> splat (i32 -1))
  ret <8 x float> %value
}

; A zext nneg of a flagged shift or multiply keeps the 32-bit product with
; scale 1 and narrows it; its scale is not moved outside the cast (see
; lExtractNUWScale()). address-gap-nneg-scale.ll has the unflagged multiply.
; A32-LABEL: define <8 x float> @nsw_shl_product(
; A32: call <8 x float> @__pseudo_gather_base_offsets32_float(ptr {{[^,]+}}, i32 1, <8 x i32> %scaled,
; A64-LABEL: define <8 x float> @nsw_shl_product(
; A64: call <8 x float> @__pseudo_gather_base_offsets64_float(ptr {{[^,]+}}, i32 1, <8 x i64> %wide,
define <8 x float> @nsw_shl_product(ptr %base, <8 x i32> %x, <8 x i32> %mask) {
  %scaled = shl nsw <8 x i32> %x, splat (i32 2)
  %wide = zext nneg <8 x i32> %scaled to <8 x i64>
  %baseint = ptrtoint ptr %base to i64
  %insert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %insert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %wide
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %value
}

; A32-LABEL: define <8 x float> @nuw_mul_product(
; A32: call <8 x float> @__pseudo_gather_base_offsets32_float(ptr {{[^,]+}}, i32 1, <8 x i32> %scaled,
; A64-LABEL: define <8 x float> @nuw_mul_product(
; A64: call <8 x float> @__pseudo_gather_base_offsets64_float(ptr {{[^,]+}}, i32 1, <8 x i64> %wide,
define <8 x float> @nuw_mul_product(ptr %base, <8 x i32> %x, <8 x i32> %mask) {
  %scaled = mul nuw <8 x i32> %x, splat (i32 8)
  %wide = zext nneg <8 x i32> %scaled to <8 x i64>
  %baseint = ptrtoint ptr %base to i64
  %insert = insertelement <8 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <8 x i64> %insert, <8 x i64> poison, <8 x i32> zeroinitializer
  %addresses = add <8 x i64> %bases, %wide
  %value = call <8 x float> @__pseudo_gather64_float(<8 x i64> %addresses, <8 x i32> %mask)
  ret <8 x float> %value
}

; The lowering sign extends a negative factored scale: index 1 with scale -1
; and delta 8 addresses base + 7, not base + 2^32 + 7.
; A32-LABEL: define <8 x float> @negative_factored_scale(
; A32: %[[I:.*]] = sext i32 %index to i64
; A32: %[[S:.*]] = mul nsw i64 -1, %[[I]]
; A32: %[[O:.*]] = add nsw i64 %[[S]], 8
; A32: getelementptr i8, ptr %base, i64 %[[O]]
; A32: load float, ptr
; A32: ret <8 x float>
define <8 x float> @negative_factored_scale(ptr %base, i32 %index) {
  %insert = insertelement <8 x i32> poison, i32 %index, i32 0
  %indices = shufflevector <8 x i32> %insert, <8 x i32> poison, <8 x i32> zeroinitializer
  %value = call <8 x float> @__pseudo_gather_factored_base_offsets32_float(ptr %base, <8 x i32> %indices, i32 -1,
                                                                            <8 x i32> splat (i32 8), <8 x i32> splat (i32 -1))
  ret <8 x float> %value
}
