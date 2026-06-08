;;  Copyright (c) 2016-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`4')
define(`ISA',`AVX512SKX')

include(`target-avx512spr-amx-utils.ll')
include(`target-avx512fp16-x4-common.ll')
