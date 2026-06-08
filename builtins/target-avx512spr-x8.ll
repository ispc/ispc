;;  Copyright (c) 2016-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`8')
define(`ISA',`AVX512SKX')

include(`target-avx512spr-amx-utils.ll')
include(`target-avx512fp16-x8-common.ll')
