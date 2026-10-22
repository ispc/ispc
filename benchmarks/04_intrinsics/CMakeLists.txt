#
#  Copyright (c) 2025-2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# List the benchmarks

compile_benchmark_test(01_sse_intrinsics CXX_FLAGS -ffast-math)
compile_benchmark_test(02_sse2_intrinsics CXX_FLAGS -ffast-math)
