;;  Copyright (c) 2020-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`WIDTH_X2',`32')
define(`WIDTH_X4',`64')
define(`XE_SUFFIX',`CONCAT(`v16', XE_TYPE($1))')
define(`BITCAST_WIDTH',`i16')

include(`target-xe.ll')
