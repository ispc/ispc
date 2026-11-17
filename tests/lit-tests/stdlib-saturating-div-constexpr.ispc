// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Ordinary signed saturating_div calls accept both uniform and varying constexpr inputs.
// RUN: %{ispc} %s %{generic_i32x4} --werror --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: GENERIC_I32X4

static_assert(saturating_div((uniform int8)INT8_MIN, (uniform int8)-1) == INT8_MAX);
static_assert(saturating_div((uniform int16)INT16_MIN, (uniform int16)-1) == INT16_MAX);
static_assert(saturating_div((uniform int32)INT32_MIN, (uniform int32)-1) == INT32_MAX);
static_assert(saturating_div((uniform int64)INT64_MIN, (uniform int64)-1) == INT64_MAX);
static_assert(saturating_div((uniform int32)-41, (uniform int32)4) == -10);
static_assert(saturating_div((uniform int32)INT32_MIN, (uniform int32)2) == INT32_MIN / 2);

constexpr varying int32 varying_result = saturating_div(INT32_MIN + programIndex, (varying int32)-1);

// CHECK-DAG: @varying_result = {{.*}} constant <4 x i32> <i32 2147483647, i32 2147483647, i32 2147483646, i32 2147483645>
