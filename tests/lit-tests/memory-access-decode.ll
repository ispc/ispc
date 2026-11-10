; The memory access decoder identifies every call form by the callee's
; name and signature, not by the target's gather/scatter
; preference. SSE4 has neither a gather nor a scatter instruction, yet the
; non-factored base-offset forms decode with their own operand roles. Masks
; of any integer width are accepted. Printing the accesses rewrites nothing.
; No general form here has a recoverable common base (see
; memory-access-recovery.ll for those).
;
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o /dev/null 2>&1 | FileCheck %s
; RUN: %{ispc-opt} --target=sse4-i32x4 --passes=print-memory-access %s -o - 2>/dev/null | FileCheck --check-prefix=IR %s
; REQUIRES: X86_ENABLED

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare <4 x i16> @__pseudo_gather32_i16(<4 x i32>, <4 x i32>)
declare void @__pseudo_scatter64_double(<4 x i64>, <4 x double>, <4 x i32>)
declare <4 x half> @__pseudo_gather_base_offsets32_half(ptr, i32, <4 x i32>, <4 x i32>)
declare void @__pseudo_scatter_base_offsets64_i8(ptr, i32, <4 x i64>, <4 x i8>, <4 x i32>)
declare <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr, <4 x i64>, i32, <4 x i64>, <4 x i1>)
declare void @__pseudo_scatter_factored_base_offsets32_i64(ptr, <4 x i32>, i32, <4 x i32>, <4 x i64>, <4 x i32>)
declare void @__pseudo_prefetch_read_varying_1(<4 x i64>, <4 x i32>)
declare void @__prefetch_write_varying_3(<4 x i64>, <4 x i32>)
declare <4 x i16> @__masked_load_i16(ptr, <4 x i1>)
declare <4 x double> @__masked_load_blend_double(ptr, <4 x i32>)
declare void @__pseudo_masked_store_float(ptr, <4 x float>, <4 x i32>)
declare void @__masked_store_blend_i8(ptr, <4 x i8>, <4 x i32>)
declare void @__masked_store_i64(ptr, <4 x i64>, <4 x i32>)
declare <4 x i32> @__gather32_generic_i32(<4 x i32>, <4 x i32>)
declare void @__scatter_base_offsets64_float(ptr, i32, <4 x i64>, <4 x float>, <4 x i32>)
; Declarations with a known name and another signature, for @malformed.
declare <4 x i8> @__pseudo_gather64_i8(<4 x i64>, <4 x i32>, <4 x i32>)
declare <4 x i64> @__pseudo_gather_base_offsets64_i64(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x double> @__pseudo_gather_factored_base_offsets64_double(ptr, <4 x i64>, <4 x i32>, <4 x i64>, <4 x i32>)
declare void @__pseudo_scatter64_float(<4 x i64>, <4 x i32>, <4 x i32>)
declare <4 x i32> @__masked_load_i32(i64, <4 x i1>)
declare void @__pseudo_prefetch_read_varying_2(<4 x i64>, <4 x float>)
declare void @__masked_store_i32(ptr, <8 x i32>, <4 x i32>)
declare <4 x i32> @__pseudo_gather32_i32(...)
declare <128 x i8> @__pseudo_gather32_i8(<128 x i32>, <128 x i32>)
declare <4 x i32> @__pseudo_gather32_float(<4 x i32>, <4 x i32>)
declare <8 x i16> @__pseudo_gather64_i16(<4 x i64>, <4 x i32>)
declare i32 @__pseudo_scatter32_i8(<4 x i32>, <4 x i8>, <4 x i32>)
declare i32 @__prefetch_read_varying_1(<4 x i64>, <4 x i32>)
declare <4 x float> @__masked_load_float(ptr, <4 x i32>)

; CHECK-LABEL: memory accesses of @general:
; CHECK-NEXT: gather __pseudo_gather64_float: general, pseudo, float (4 bytes), 4 lanes, 64-bit addresses
; CHECK-NEXT:   result: <4 x float> %g
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: gather __pseudo_gather32_i16: general, pseudo, i16 (2 bytes), 4 lanes, 32-bit addresses
; CHECK-NEXT:   result: <4 x i16> %g32
; CHECK-NEXT:   addresses: <4 x i32> %addresses32
; CHECK-NEXT:   mask: <4 x i32> splat (i32 -1) (all on)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: scatter __pseudo_scatter64_double: general, pseudo, double (8 bytes), 4 lanes, 64-bit addresses
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   value: <4 x double> %values
; CHECK-NEXT:   mask: <4 x i32> <i32 -1, i32 0, i32 -1, i32 0> (mixed)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: prefetch __pseudo_prefetch_read_varying_1: general, pseudo, 4 lanes, 64-bit addresses
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> zeroinitializer (all off)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: prefetch __prefetch_write_varying_3: general, lowered, 4 lanes, 64-bit addresses
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
define <4 x float> @general(<4 x i64> %addresses, <4 x i32> %addresses32, <4 x double> %values, <4 x i32> %mask) {
  %g = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
  %g32 = call <4 x i16> @__pseudo_gather32_i16(<4 x i32> %addresses32, <4 x i32> splat (i32 -1))
  call void @__pseudo_scatter64_double(<4 x i64> %addresses, <4 x double> %values, <4 x i32> <i32 -1, i32 0, i32 -1, i32 0>)
  call void @__pseudo_prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> zeroinitializer)
  call void @__prefetch_write_varying_3(<4 x i64> %addresses, <4 x i32> %mask)
  ret <4 x float> %g
}

; The scale is the second operand of a base-offset call and the third of a
; factored call; the roles come from the name, not from the target.
; CHECK-LABEL: memory accesses of @base_offsets:
; CHECK-NEXT: gather __pseudo_gather_base_offsets32_half: base-offsets, pseudo, half (2 bytes), 4 lanes, 32-bit offsets
; CHECK-NEXT:   result: <4 x half> %h
; CHECK-NEXT:   base: ptr %base
; CHECK-NEXT:   scale: i32 2
; CHECK-NEXT:   offsets: <4 x i32> %offsets32
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   candidate: base ptr %base + index <4 x i32> %offsets32 * 2 (lowering extends i32)
; CHECK-NEXT:   narrow index: <4 x i32> %offsets32
; CHECK-NEXT:   lane pattern of offsets: unknown; lane 0 not known defined
; CHECK-NEXT: scatter __pseudo_scatter_base_offsets64_i8: base-offsets, pseudo, i8 (1 bytes), 4 lanes, 64-bit offsets
; CHECK-NEXT:   base: ptr %base
; CHECK-NEXT:   scale: i32 1
; CHECK-NEXT:   offsets: <4 x i64> %offsets64
; CHECK-NEXT:   value: <4 x i8> %bytes
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   candidate: base ptr %base + index <4 x i64> %offsets64 * 1 (address width)
; CHECK-NEXT:   narrow index: unknown
; CHECK-NEXT:   lane pattern of offsets: unknown; lane 0 not known defined
; CHECK-NEXT: gather __pseudo_gather_factored_base_offsets64_i32: factored, pseudo, i32 (4 bytes), 4 lanes, 64-bit offsets
; CHECK-NEXT:   result: <4 x i32> %f
; CHECK-NEXT:   base: ptr %base
; CHECK-NEXT:   scale: i32 4
; CHECK-NEXT:   offsets: <4 x i64> %offsets64
; CHECK-NEXT:   delta: <4 x i64> <i64 0, i64 4, i64 8, i64 12>
; CHECK-NEXT:   mask: <4 x i1> %mask1 (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   candidate: base ptr %base + index <4 x i64> %offsets64 * 4 + displacement <4 x i64> <i64 0, i64 4, i64 8, i64 12> (address width)
; CHECK-NEXT:   narrow index: unknown
; CHECK-NEXT:   lane pattern of offsets: unknown; lane 0 not known defined
; CHECK-NEXT: scatter __pseudo_scatter_factored_base_offsets32_i64: factored, pseudo, i64 (8 bytes), 4 lanes, 32-bit offsets
; CHECK-NEXT:   base: ptr %base
; CHECK-NEXT:   scale: i32 8
; CHECK-NEXT:   offsets: <4 x i32> %offsets32
; CHECK-NEXT:   delta: <4 x i32> splat (i32 -4)
; CHECK-NEXT:   value: <4 x i64> %longs
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   candidate: base ptr %base + index <4 x i32> %offsets32 * 8 + displacement <4 x i32> splat (i32 -4) (lowering extends i32)
; CHECK-NEXT:   narrow index: <4 x i32> %offsets32, displacement lanes -4 -4 -4 -4
; CHECK-NEXT:   lane pattern of offsets: unknown; lane 0 not known defined
define <4 x half> @base_offsets(ptr %base, <4 x i32> %offsets32, <4 x i64> %offsets64, <4 x i8> %bytes, <4 x i64> %longs, <4 x i32> %mask, <4 x i1> %mask1) {
  %h = call <4 x half> @__pseudo_gather_base_offsets32_half(ptr %base, i32 2, <4 x i32> %offsets32, <4 x i32> %mask)
  call void @__pseudo_scatter_base_offsets64_i8(ptr %base, i32 1, <4 x i64> %offsets64, <4 x i8> %bytes, <4 x i32> %mask)
  %f = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %offsets64, i32 4, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i1> %mask1)
  call void @__pseudo_scatter_factored_base_offsets32_i64(ptr %base, <4 x i32> %offsets32, i32 8, <4 x i32> splat (i32 -4), <4 x i64> %longs, <4 x i32> %mask)
  ret <4 x half> %h
}

; CHECK-LABEL: memory accesses of @pointer_forms:
; CHECK-NEXT: masked load __masked_load_i16: pointer, lowered, i16 (2 bytes), 4 lanes
; CHECK-NEXT:   result: <4 x i16> %l
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   mask: <4 x i1> <i1 true, i1 true, i1 false, i1 false> (mixed)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT: masked load __masked_load_blend_double: pointer, blend, double (8 bytes), 4 lanes
; CHECK-NEXT:   result: <4 x double> %lb
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT: masked store __pseudo_masked_store_float: pointer, pseudo, float (4 bytes), 4 lanes
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   value: <4 x float> %floats
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT: masked store __masked_store_blend_i8: pointer, blend, i8 (1 bytes), 4 lanes
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   value: <4 x i8> %bytes
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT: masked store __masked_store_i64: pointer, lowered, i64 (8 bytes), 4 lanes
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   value: <4 x i64> %longs
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
define <4 x i16> @pointer_forms(ptr %p, <4 x float> %floats, <4 x i8> %bytes, <4 x i64> %longs, <4 x i32> %mask) {
  %l = call <4 x i16> @__masked_load_i16(ptr %p, <4 x i1> <i1 true, i1 true, i1 false, i1 false>)
  %lb = call <4 x double> @__masked_load_blend_double(ptr %p, <4 x i32> %mask)
  call void @__pseudo_masked_store_float(ptr %p, <4 x float> %floats, <4 x i32> %mask)
  call void @__masked_store_blend_i8(ptr %p, <4 x i8> %bytes, <4 x i32> %mask)
  call void @__masked_store_i64(ptr %p, <4 x i64> %longs, <4 x i32> %mask)
  ret <4 x i16> %l
}

; The lowered gather and scatter builtins decode like their pseudo forms.
; CHECK-LABEL: memory accesses of @lowered_forms:
; CHECK-NEXT: gather __gather32_generic_i32: general, generic, i32 (4 bytes), 4 lanes, 32-bit addresses
; CHECK-NEXT:   result: <4 x i32> %g
; CHECK-NEXT:   addresses: <4 x i32> %addresses
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: scatter __scatter_base_offsets64_float: base-offsets, lowered, float (4 bytes), 4 lanes, 64-bit offsets
; CHECK-NEXT:   base: ptr %p
; CHECK-NEXT:   scale: i32 4
; CHECK-NEXT:   offsets: <4 x i64> %offsets
; CHECK-NEXT:   value: <4 x float> %floats
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   candidate: base ptr %p + index <4 x i64> %offsets * 4 (address width)
; CHECK-NEXT:   narrow index: unknown
; CHECK-NEXT:   lane pattern of offsets: unknown; lane 0 not known defined
define <4 x i32> @lowered_forms(ptr %p, <4 x i32> %addresses, <4 x i64> %offsets, <4 x float> %floats, <4 x i32> %mask) {
  %g = call <4 x i32> @__gather32_generic_i32(<4 x i32> %addresses, <4 x i32> %mask)
  call void @__scatter_base_offsets64_float(ptr %p, i32 4, <4 x i64> %offsets, <4 x float> %floats, <4 x i32> %mask)
  ret <4 x i32> %g
}

; The ISPC source position comes from the call's metadata when present.
; CHECK-LABEL: memory accesses of @with_position:
; CHECK-NEXT: gather __pseudo_gather64_float: general, pseudo, float (4 bytes), 4 lanes, 64-bit addresses
; CHECK-NEXT:   result: <4 x float> %g
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: kernel.ispc:12:7
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
define <4 x float> @with_position(<4 x i64> %addresses, <4 x i32> %mask) {
  %g = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask), !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
  ret <4 x float> %g
}

; Malformed position metadata, as hand-written IR may carry, reports no
; position instead of being trusted.
; CHECK-LABEL: memory accesses of @with_bad_position:
; CHECK-NEXT: gather __pseudo_gather64_float:
; CHECK-NEXT:   result: <4 x float> %g
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> %mask (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
define <4 x float> @with_bad_position(<4 x i64> %addresses, <4 x i32> %mask) {
  %g = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask), !filename !4, !first_line !0, !first_column !2, !last_line !1, !last_column !3
  ret <4 x float> %g
}

!0 = !{!"kernel.ispc"}
!1 = !{i32 12}
!2 = !{i32 7}
!3 = !{i32 20}
!4 = !{}

; Calls of a known name whose signature differs from the form's layout are
; reported, not decoded under the assumed layout: an extra operand, 32-bit
; offsets in a 64-bit form, a vector scale, a value whose element type
; differs from the function's, a non-pointer base, a non-integer mask, a
; value with another lane count than the mask, a variadic declaration, more
; lanes than any target has, a result of another element type or lane count,
; a scatter or prefetch that returns a value, and a call whose function type
; differs from the callee's.
; CHECK-LABEL: memory accesses of @malformed:
; CHECK-NEXT: malformed __pseudo_gather64_i8: not decoded
; CHECK-NEXT: malformed __pseudo_gather_base_offsets64_i64: not decoded
; CHECK-NEXT: malformed __pseudo_gather_factored_base_offsets64_double: not decoded
; CHECK-NEXT: malformed __pseudo_scatter64_float: not decoded
; CHECK-NEXT: malformed __masked_load_i32: not decoded
; CHECK-NEXT: malformed __pseudo_prefetch_read_varying_2: not decoded
; CHECK-NEXT: malformed __masked_store_i32: not decoded
; CHECK-NEXT: malformed __pseudo_gather32_i32: not decoded
; CHECK-NEXT: malformed __pseudo_gather32_i8: not decoded
; CHECK-NEXT: malformed __pseudo_gather32_float: not decoded
; CHECK-NEXT: malformed __pseudo_gather64_i16: not decoded
; CHECK-NEXT: malformed __pseudo_scatter32_i8: not decoded
; CHECK-NEXT: malformed __prefetch_read_varying_1: not decoded
; CHECK-NEXT: malformed __masked_load_float: not decoded
; CHECK-NOT: gather
; CHECK-NOT: scatter
define void @malformed(<4 x i64> %addresses, <4 x i32> %offsets32, <4 x i64> %offsets64, <4 x i32> %ints, <8 x i32> %wide, <4 x i32> %mask, <4 x float> %floats, i64 %int, <128 x i32> %huge) {
  %a = call <4 x i8> @__pseudo_gather64_i8(<4 x i64> %addresses, <4 x i32> %mask, <4 x i32> %mask)
  %b = call <4 x i64> @__pseudo_gather_base_offsets64_i64(ptr null, i32 1, <4 x i32> %offsets32, <4 x i32> %mask)
  %c = call <4 x double> @__pseudo_gather_factored_base_offsets64_double(ptr null, <4 x i64> %offsets64, <4 x i32> %mask, <4 x i64> %offsets64, <4 x i32> %mask)
  call void @__pseudo_scatter64_float(<4 x i64> %addresses, <4 x i32> %ints, <4 x i32> %mask)
  %d = call <4 x i32> @__masked_load_i32(i64 %int, <4 x i1> zeroinitializer)
  call void @__pseudo_prefetch_read_varying_2(<4 x i64> %addresses, <4 x float> %floats)
  call void @__masked_store_i32(ptr null, <8 x i32> %wide, <4 x i32> %mask)
  %e = call <4 x i32> (...) @__pseudo_gather32_i32(<4 x i32> %offsets32, <4 x i32> %mask)
  %f = call <128 x i8> @__pseudo_gather32_i8(<128 x i32> %huge, <128 x i32> %huge)
  %g = call <4 x i32> @__pseudo_gather32_float(<4 x i32> %offsets32, <4 x i32> %mask)
  %h = call <8 x i16> @__pseudo_gather64_i16(<4 x i64> %addresses, <4 x i32> %mask)
  %i = call i32 @__pseudo_scatter32_i8(<4 x i32> %offsets32, <4 x i8> zeroinitializer, <4 x i32> %mask)
  %j = call i32 @__prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> %mask)
  %k = call <4 x float> @__masked_load_float(ptr null, <4 x i64> %addresses)
  ret void
}

; A constant mask with undef or poison lanes, or one that is undef or poison
; as a whole, proves no status: choosing those lanes is a refinement, not a
; fact about the mask.
; CHECK-LABEL: memory accesses of @undef_mask_lanes:
; CHECK-NEXT: gather __pseudo_gather64_float:
; CHECK-NEXT:   result: <4 x float> %on
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> <i32 -1, i32 undef, i32 undef, i32 undef> (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: gather __pseudo_gather64_float:
; CHECK-NEXT:   result: <4 x float> %off
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison> (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: gather __pseudo_gather64_float:
; CHECK-NEXT:   result: <4 x float> %undef
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> undef (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
; CHECK-NEXT: gather __pseudo_gather64_float:
; CHECK-NEXT:   result: <4 x float> %poison
; CHECK-NEXT:   addresses: <4 x i64> %addresses
; CHECK-NEXT:   mask: <4 x i32> poison (unknown)
; CHECK-NEXT:   source: unknown
; CHECK-NEXT:   common base: unknown
; CHECK-NEXT:   lane pattern of addresses: unknown; lane 0 not known defined
define <4 x float> @undef_mask_lanes(<4 x i64> %addresses) {
  %on = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> <i32 -1, i32 undef, i32 undef, i32 undef>)
  %off = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>)
  %undef = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> undef)
  %poison = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> poison)
  ret <4 x float> %on
}

; IR-LABEL: define <4 x float> @general(
; IR-NEXT: %g = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: %g32 = call <4 x i16> @__pseudo_gather32_i16(<4 x i32> %addresses32, <4 x i32> splat (i32 -1))
; IR-NEXT: call void @__pseudo_scatter64_double(<4 x i64> %addresses, <4 x double> %values, <4 x i32> <i32 -1, i32 0, i32 -1, i32 0>)
; IR-NEXT: call void @__pseudo_prefetch_read_varying_1(<4 x i64> %addresses, <4 x i32> zeroinitializer)
; IR-NEXT: call void @__prefetch_write_varying_3(<4 x i64> %addresses, <4 x i32> %mask)
; IR-NEXT: ret <4 x float> %g
; IR-LABEL: define <4 x half> @base_offsets(
; IR-NEXT: %h = call <4 x half> @__pseudo_gather_base_offsets32_half(ptr %base, i32 2, <4 x i32> %offsets32, <4 x i32> %mask)
; IR-NEXT: call void @__pseudo_scatter_base_offsets64_i8(ptr %base, i32 1, <4 x i64> %offsets64, <4 x i8> %bytes, <4 x i32> %mask)
; IR-NEXT: %f = call <4 x i32> @__pseudo_gather_factored_base_offsets64_i32(ptr %base, <4 x i64> %offsets64, i32 4, <4 x i64> <i64 0, i64 4, i64 8, i64 12>, <4 x i1> %mask1)
; IR-NEXT: call void @__pseudo_scatter_factored_base_offsets32_i64(ptr %base, <4 x i32> %offsets32, i32 8, <4 x i32> splat (i32 -4), <4 x i64> %longs, <4 x i32> %mask)
; IR-NEXT: ret <4 x half> %h
