; A zext nneg of an i32 index can use a 32-bit-index gather, whose lowering
; sign extends the index before scaling it. When such a gather becomes a
; scalar load, the load must not multiply the index at i32 width. At index
; 0x20000000, scale 4 addresses base + 2 GiB; the wrapped i32 product
; addresses base - 2 GiB. AVX2 reaches the load through the native
; base-offset form, SSE4 and AVX1 through the factored form.
;
; RUN: %{ispc-opt} --target=avx2-i32x4 --addressing=32 --passes=improve-memory-ops %s -o %t-avx2.ll
; RUN: FileCheck %s --input-file=%t-avx2.ll
; RUN: %{cc} -O2 -x ir %t-avx2.ll -x c %S/gather-nneg-scaled-broadcast.inc -o %t-avx2.bin
; RUN: %t-avx2.bin
; RUN: %{ispc-opt} --target=avx1-i32x4 --addressing=32 --passes=improve-memory-ops %s -o %t-avx1.ll
; RUN: FileCheck %s --input-file=%t-avx1.ll
; RUN: %{cc} -O2 -x ir %t-avx1.ll -x c %S/gather-nneg-scaled-broadcast.inc -o %t-avx1.bin
; RUN: %t-avx1.bin
; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=32 --passes=improve-memory-ops %s -o %t-sse4.ll
; RUN: FileCheck %s --input-file=%t-sse4.ll
; RUN: %{cc} -O2 -x ir %t-sse4.ll -x c %S/gather-nneg-scaled-broadcast.inc -o %t-sse4.bin
; RUN: %t-sse4.bin
; REQUIRES: X86_ENABLED, X86_64_HOST, LINUX_ENABLED, LLVM_20_0+

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets32_float(ptr, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_base_offsets64_float(ptr, i32, <4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets32_float(ptr, <4 x i32>, i32, <4 x i32>, <4 x i32>)
declare <4 x float> @__pseudo_gather_factored_base_offsets64_float(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)

; CHECK-LABEL: define float @read_scaled(
; CHECK-NOT: mul i32
; CHECK: sext i32 %index to i64
; CHECK-NOT: call <4 x float> @__pseudo_gather
; CHECK: load float
; CHECK-NOT: call <4 x float> @__pseudo_gather
; CHECK: ret float
define float @read_scaled(ptr %base, i32 noundef %index) {
  %insert = insertelement <4 x i32> poison, i32 %index, i32 0
  %indices = shufflevector <4 x i32> %insert, <4 x i32> poison, <4 x i32> zeroinitializer
  %wide = zext nneg <4 x i32> %indices to <4 x i64>
  %offsets = mul nsw <4 x i64> %wide, splat (i64 4)
  %baseint = ptrtoint ptr %base to i64
  %baseinsert = insertelement <4 x i64> poison, i64 %baseint, i32 0
  %bases = shufflevector <4 x i64> %baseinsert, <4 x i64> poison, <4 x i32> zeroinitializer
  %addresses = add <4 x i64> %bases, %offsets
  %value = call <4 x float> @__pseudo_gather64_float(<4 x i64> %addresses, <4 x i32> splat (i32 -1))
  %scalar = extractelement <4 x float> %value, i32 0
  ret float %scalar
}
