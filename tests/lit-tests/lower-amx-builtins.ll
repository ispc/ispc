; RUN: %{ispc-opt} --target=host --passes=lower-amx-builtins %s -o - | FileCheck %s

; Test AMX tile zero
; CHECK-LABEL: @test_tile_zero
; CHECK: call void @llvm.x86.tilezero(i8 0)
; CHECK: ret void
declare void @__ispc_amx_tile_zero(i8)
define void @test_tile_zero() {
  call void @__ispc_amx_tile_zero(i8 0)
  ret void
}

; Test AMX tile load
; CHECK-LABEL: @test_tile_load
; CHECK: call void @llvm.x86.tileloadd64(i8 1, ptr %data, i64 %stride)
; CHECK: ret void
declare void @__ispc_amx_tile_load(i8, ptr, i64)
define void @test_tile_load(ptr %data, i64 %stride) {
  call void @__ispc_amx_tile_load(i8 1, ptr %data, i64 %stride)
  ret void
}

; Test AMX tile load with T1 hint
; CHECK-LABEL: @test_tile_load_t1
; CHECK: call void @llvm.x86.tileloaddt164(i8 2, ptr %data, i64 %stride)
; CHECK: ret void
declare void @__ispc_amx_tile_load_t1(i8, ptr, i64)
define void @test_tile_load_t1(ptr %data, i64 %stride) {
  call void @__ispc_amx_tile_load_t1(i8 2, ptr %data, i64 %stride)
  ret void
}

; Test AMX tile store
; CHECK-LABEL: @test_tile_store
; CHECK: call void @llvm.x86.tilestored64(i8 3, ptr %data, i64 %stride)
; CHECK: ret void
declare void @__ispc_amx_tile_store(i8, ptr, i64)
define void @test_tile_store(ptr %data, i64 %stride) {
  call void @__ispc_amx_tile_store(i8 3, ptr %data, i64 %stride)
  ret void
}

; Test AMX INT8 dot products (signed/signed)
; CHECK-LABEL: @test_dpbssd
; CHECK: call void @llvm.x86.tdpbssd(i8 0, i8 1, i8 2)
; CHECK: ret void
declare void @__ispc_amx_dpbssd(i8, i8, i8)
define void @test_dpbssd() {
  call void @__ispc_amx_dpbssd(i8 0, i8 1, i8 2)
  ret void
}

; Test AMX INT8 dot products (signed/unsigned)
; CHECK-LABEL: @test_dpbsud
; CHECK: call void @llvm.x86.tdpbsud(i8 3, i8 4, i8 5)
; CHECK: ret void
declare void @__ispc_amx_dpbsud(i8, i8, i8)
define void @test_dpbsud() {
  call void @__ispc_amx_dpbsud(i8 3, i8 4, i8 5)
  ret void
}

; Test AMX INT8 dot products (unsigned/signed)
; CHECK-LABEL: @test_dpbusd
; CHECK: call void @llvm.x86.tdpbusd(i8 6, i8 7, i8 0)
; CHECK: ret void
declare void @__ispc_amx_dpbusd(i8, i8, i8)
define void @test_dpbusd() {
  call void @__ispc_amx_dpbusd(i8 6, i8 7, i8 0)
  ret void
}

; Test AMX INT8 dot products (unsigned/unsigned)
; CHECK-LABEL: @test_dpbuud
; CHECK: call void @llvm.x86.tdpbuud(i8 1, i8 2, i8 3)
; CHECK: ret void
declare void @__ispc_amx_dpbuud(i8, i8, i8)
define void @test_dpbuud() {
  call void @__ispc_amx_dpbuud(i8 1, i8 2, i8 3)
  ret void
}

; Test AMX BF16 dot product
; CHECK-LABEL: @test_dpbf16ps
; CHECK: call void @llvm.x86.tdpbf16ps(i8 4, i8 5, i8 6)
; CHECK: ret void
declare void @__ispc_amx_dpbf16ps(i8, i8, i8)
define void @test_dpbf16ps() {
  call void @__ispc_amx_dpbf16ps(i8 4, i8 5, i8 6)
  ret void
}

; Test AMX FP16 dot product
; CHECK-LABEL: @test_dpfp16ps
; CHECK: call void @llvm.x86.tdpfp16ps(i8 7, i8 0, i8 1)
; CHECK: ret void
declare void @__ispc_amx_dpfp16ps(i8, i8, i8)
define void @test_dpfp16ps() {
  call void @__ispc_amx_dpfp16ps(i8 7, i8 0, i8 1)
  ret void
}

; Test that constants through alloca/load (O0 pattern) are handled correctly
; CHECK-LABEL: @test_tile_zero_through_alloca
; CHECK: call void @llvm.x86.tilezero(i8 5)
; CHECK: ret void
define void @test_tile_zero_through_alloca() {
entry:
  %tile = alloca i8, align 1
  store i8 5, ptr %tile, align 1
  %loaded = load i8, ptr %tile, align 1
  call void @__ispc_amx_tile_zero(i8 %loaded)
  ret void
}
