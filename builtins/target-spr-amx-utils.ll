;;  Copyright (c) 2015-2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AMX-TILE intrinsics

declare void @llvm.x86.ldtilecfg(i8*)
define void @__amx_loadconfig(i8* %config) nounwind alwaysinline {
  call void @llvm.x86.ldtilecfg(i8* %config)
  ret void
}

declare void @llvm.x86.sttilecfg(i8*)
define void @__amx_storeconfig(i8* %config) nounwind alwaysinline {
  call void @llvm.x86.sttilecfg(i8* %config)
  ret void
}

declare void @llvm.x86.tilerelease()
define void @__amx_release() nounwind alwaysinline {
  call void @llvm.x86.tilerelease()
  ret void
}

declare void @__ispc_amx_zero(i8)
define void @__amx_zero(i8 %tile) nounwind alwaysinline {
  call void @__ispc_amx_zero(i8 %tile)
  ret void
}

declare void @__ispc_amx_load(i8, i8*, i64)
define void @__amx_load(i8 %tile, i8* %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_load(i8 %tile, i8* %data, i64 %stride)
  ret void
}

declare void @__ispc_amx_load_t1(i8, i8*, i64)
define void @__amx_load_t1(i8 %tile, i8* %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_load_t1(i8 %tile, i8* %data, i64 %stride)
  ret void
}

declare void @__ispc_amx_store(i8, i8*, i64)
define void @__amx_store(i8 %tile, i8* %data, i64 %stride) nounwind alwaysinline {
  call void @__ispc_amx_store(i8 %tile, i8* %data, i64 %stride)
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
