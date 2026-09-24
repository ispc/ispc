;;  Copyright (c) 2025-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

;; Functions shared by every rvv-<VLEN>b target. The vector register width
;; does not matter for anything defined here.

define i64 @__clock() nounwind {
  %r = call i64 asm sideeffect "rdtime $0", "=r"() nounwind
  ret i64 %r
}
