#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# List the benchmarks

compile_benchmark_test(01_sse_intrinsics)
target_compile_options(01_sse_intrinsics PRIVATE -ffast-math)
compile_benchmark_test(02_sse2_intrinsics)
target_compile_options(02_sse2_intrinsics PRIVATE -ffast-math)
