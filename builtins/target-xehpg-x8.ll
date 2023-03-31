;;  Copyright (c) 2019-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`WIDTH_X2',`16')
define(`WIDTH_X4',`32')
define(`XE_SUFFIX',`CONCAT(`v8', XE_TYPE($1))')
define(`BITCAST_WIDTH',`i8')

include(`target-xe.ll')
