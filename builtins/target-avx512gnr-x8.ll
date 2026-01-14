;;  Copyright (c) 2025-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`ISA',`AVX512SKX')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AMX-FP16 intrinsic

declare void @__ispc_amx_dpfp16ps(i8, i8, i8)
define void @__amx_dpfp16ps(i8 %dst, i8 %src1, i8 %src2) nounwind alwaysinline {
  call void @__ispc_amx_dpfp16ps(i8 %dst, i8 %src1, i8 %src2)
  ret void
}
