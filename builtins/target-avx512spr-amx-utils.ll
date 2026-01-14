;;  Copyright (c) 2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AMX-TILE intrinsics

declare void @llvm.x86.ldtilecfg(i8*)
define void @__amx_tile_loadconfig(i8* nocapture readonly %config) nounwind alwaysinline {
  call void @llvm.x86.ldtilecfg(i8* %config)
  ret void
}

declare void @llvm.x86.sttilecfg(i8*)
define void @__amx_tile_storeconfig(i8* nocapture writeonly %config) nounwind alwaysinline {
  call void @llvm.x86.sttilecfg(i8* %config)
  ret void
}

declare void @llvm.x86.tilerelease()
define void @__amx_tile_release() nounwind alwaysinline {
  call void @llvm.x86.tilerelease()
  ret void
}

declare void @__ispc_amx_tile_zero(i8)
define void @__amx_tile_zero(i8 %tile) nounwind alwaysinline {
  call void @__ispc_amx_tile_zero(i8 %tile)
  ret void
}

declare void @__ispc_amx_tile_load(i8, i8*, i64)
define void @__amx_tile_load(i8 %tile, i8* nocapture readonly %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_tile_load(i8 %tile, i8* %data, i64 %stride)
  ret void
}

declare void @__ispc_amx_tile_load_t1(i8, i8*, i64)
define void @__amx_tile_load_t1(i8 %tile, i8* nocapture readonly %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_tile_load_t1(i8 %tile, i8* %data, i64 %stride)
  ret void
}

declare void @__ispc_amx_tile_store(i8, i8*, i64)
define void @__amx_tile_store(i8 %tile, i8* nocapture writeonly %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_tile_store(i8 %tile, i8* %data, i64 %stride)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AMX-INT8 intrinsics

declare void @__ispc_amx_dpbssd(i8, i8, i8)
define void @__amx_dpbssd(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpbssd(i8 %dst, i8 %src1, i8 %src2)
  ret void
}

declare void @__ispc_amx_dpbsud(i8, i8, i8)
define void @__amx_dpbsud(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpbsud(i8 %dst, i8 %src1, i8 %src2)
  ret void
}

declare void @__ispc_amx_dpbusd(i8, i8, i8)
define void @__amx_dpbusd(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpbusd(i8 %dst, i8 %src1, i8 %src2)
  ret void
}

declare void @__ispc_amx_dpbuud(i8, i8, i8)
define void @__amx_dpbuud(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpbuud(i8 %dst, i8 %src1, i8 %src2)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AMX-BF16 intrinsic

declare void @__ispc_amx_dpbf16ps(i8, i8, i8)
define void @__amx_dpbf16ps(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpbf16ps(i8 %dst, i8 %src1, i8 %src2)
  ret void
}
