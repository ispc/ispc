;;  Copyright (c) 2020-2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`64')
define(`ISA',`AVX512SKX')

include(`target-avx512spr-amx-utils.ll')
include(`target-avx512fp16-x64-common.ll')
