;;  Copyright (c) 2022-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`32')
define(`WIDTH_X2',`64')
define(`WIDTH_X4',`128')
define(`XE_SUFFIX',`CONCAT(`v32', XE_TYPE($1))')
define(`BITCAST_WIDTH',`i32')

include(`target-xe.ll')
