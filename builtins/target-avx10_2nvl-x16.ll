;;  Copyright (c) 2026, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause

define(`WIDTH',`16')
define(`ISA',`NVL_AVX10_2')
include(`target-avx10_2-x16-common.ll')
include(`target-avx512fp16-x16-common.ll')
