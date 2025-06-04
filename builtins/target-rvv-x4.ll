;;  Copyright (c) 2025, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define i64 @__clock() nounwind {
  %r = call i64 asm sideeffect "rdtime $0", "=r"() nounwind
  ret i64 %r
}
