; Each pseudo gather, scatter, prefetch and masked store is lowered to the
; target builtin of the same kind, form, element and offset width; a masked
; store to a vector-typed alloca uses the blend variant.
;
; RUN: %{ispc-opt} --target=avx2-i32x8 --passes=replace-pseudo-memory-ops %s -o - | FileCheck %s
; REQUIRES: X86_ENABLED

declare <8 x i32> @__pseudo_gather32_i32(<8 x i32>, <8 x i32>)
declare <8 x float> @__pseudo_gather64_float(<8 x i64>, <8 x i32>)
declare <8 x i16> @__pseudo_gather_base_offsets32_i16(ptr, i32, <8 x i32>, <8 x i32>)
declare <8 x double> @__pseudo_gather_factored_base_offsets64_double(ptr, <8 x i64>, i32, <8 x i64>, <8 x i32>)
declare void @__pseudo_scatter64_i8(<8 x i64>, <8 x i8>, <8 x i32>)
declare void @__pseudo_scatter_base_offsets64_half(ptr, i32, <8 x i64>, <8 x half>, <8 x i32>)
declare void @__pseudo_scatter_factored_base_offsets32_i64(ptr, <8 x i32>, i32, <8 x i32>, <8 x i64>, <8 x i32>)
declare void @__pseudo_prefetch_read_varying_nt(<8 x i64>, <8 x i32>)
declare void @__pseudo_prefetch_write_varying_3(<8 x i64>, <8 x i32>)
declare void @__pseudo_masked_store_float(ptr, <8 x float>, <8 x i32>)

declare <8 x i32> @__gather32_i32(<8 x i32>, <8 x i32>)
declare <8 x float> @__gather64_float(<8 x i64>, <8 x i32>)
declare <8 x i32> @__gather32_generic_i32(<8 x i32>, <8 x i32>)
declare <8 x float> @__gather64_generic_float(<8 x i64>, <8 x i32>)
declare <8 x i16> @__gather_base_offsets32_i16(ptr, i32, <8 x i32>, <8 x i32>)
declare <8 x double> @__gather_factored_base_offsets64_double(ptr, <8 x i64>, i32, <8 x i64>, <8 x i32>)
declare void @__scatter64_i8(<8 x i64>, <8 x i8>, <8 x i32>)
declare void @__scatter64_generic_i8(<8 x i64>, <8 x i8>, <8 x i32>)
declare void @__scatter_base_offsets64_half(ptr, i32, <8 x i64>, <8 x half>, <8 x i32>)
declare void @__scatter_factored_base_offsets32_i64(ptr, <8 x i32>, i32, <8 x i32>, <8 x i64>, <8 x i32>)
declare void @__prefetch_read_varying_nt(<8 x i64>, <8 x i32>)
declare void @__prefetch_write_varying_3(<8 x i64>, <8 x i32>)
declare void @__masked_store_float(ptr, <8 x float>, <8 x i32>)
declare void @__masked_store_blend_float(ptr, <8 x float>, <8 x i32>)

; CHECK-LABEL: define void @lower_all(
; CHECK: call <8 x i32> @__gather32_i32(<8 x i32> %a32, <8 x i32> %m)
; CHECK: call <8 x float> @__gather64_float(<8 x i64> %a64, <8 x i32> %m)
; CHECK: call <8 x i16> @__gather_base_offsets32_i16(ptr %p, i32 2, <8 x i32> %a32, <8 x i32> %m)
; CHECK: call <8 x double> @__gather_factored_base_offsets64_double(ptr %p, <8 x i64> %a64, i32 8, <8 x i64> zeroinitializer, <8 x i32> %m)
; CHECK: call void @__scatter64_i8(<8 x i64> %a64, <8 x i8> %v8, <8 x i32> %m)
; CHECK: call void @__scatter_base_offsets64_half(ptr %p, i32 2, <8 x i64> %a64, <8 x half> %vh, <8 x i32> %m)
; CHECK: call void @__scatter_factored_base_offsets32_i64(ptr %p, <8 x i32> %a32, i32 8, <8 x i32> zeroinitializer, <8 x i64> %v64, <8 x i32> %m)
; CHECK: call void @__prefetch_read_varying_nt(<8 x i64> %a64, <8 x i32> %m)
; CHECK: call void @__prefetch_write_varying_3(<8 x i64> %a64, <8 x i32> %m)
; CHECK: call void @__masked_store_float(ptr %p, <8 x float> %vf, <8 x i32> %m)
; CHECK: call void @__masked_store_blend_float(ptr %local, <8 x float> %vf, <8 x i32> %m)
; CHECK-NOT: @__pseudo_
; CHECK: ret void
define void @lower_all(ptr %p, <8 x i32> %a32, <8 x i64> %a64, <8 x i8> %v8, <8 x half> %vh, <8 x i64> %v64,
                       <8 x float> %vf, <8 x i32> %m) {
  %local = alloca <8 x float>
  %g1 = call <8 x i32> @__pseudo_gather32_i32(<8 x i32> %a32, <8 x i32> %m)
  %g2 = call <8 x float> @__pseudo_gather64_float(<8 x i64> %a64, <8 x i32> %m)
  %g3 = call <8 x i16> @__pseudo_gather_base_offsets32_i16(ptr %p, i32 2, <8 x i32> %a32, <8 x i32> %m)
  %g4 = call <8 x double> @__pseudo_gather_factored_base_offsets64_double(ptr %p, <8 x i64> %a64, i32 8, <8 x i64> zeroinitializer, <8 x i32> %m)
  call void @__pseudo_scatter64_i8(<8 x i64> %a64, <8 x i8> %v8, <8 x i32> %m)
  call void @__pseudo_scatter_base_offsets64_half(ptr %p, i32 2, <8 x i64> %a64, <8 x half> %vh, <8 x i32> %m)
  call void @__pseudo_scatter_factored_base_offsets32_i64(ptr %p, <8 x i32> %a32, i32 8, <8 x i32> zeroinitializer, <8 x i64> %v64, <8 x i32> %m)
  call void @__pseudo_prefetch_read_varying_nt(<8 x i64> %a64, <8 x i32> %m)
  call void @__pseudo_prefetch_write_varying_3(<8 x i64> %a64, <8 x i32> %m)
  call void @__pseudo_masked_store_float(ptr %p, <8 x float> %vf, <8 x i32> %m)
  call void @__pseudo_masked_store_float(ptr %local, <8 x float> %vf, <8 x i32> %m)
  ret void
}
