// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Check integer library calls before LLVM optimization, including varying values.
// RUN: %{ispc} %s %{generic_i32x4} --werror --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// REQUIRES: GENERIC_I32X4

template <typename T> constexpr uniform bool signed_operations(uniform T low, uniform T high) {
    uniform T one = 1;
    uniform T zero = 0;
    uniform T minus_one = -1;
    return min(low, high) == low && max(low, high) == high && clamp(zero, one, high) == one &&
           clamp(zero, high, low) == low && saturating_add(high, one) == high &&
           saturating_add(low, minus_one) == low && saturating_sub(low, one) == low &&
           saturating_sub(high, minus_one) == high && saturating_sub(minus_one, low) == high;
}

template <typename T> constexpr uniform bool unsigned_operations(uniform T high) {
    uniform T zero = 0;
    uniform T one = 1;
    uniform T two = 2;
    return min(zero, high) == zero && max(zero, high) == high && clamp(high, zero, one) == one &&
           saturating_add(high, one) == high && saturating_sub(zero, one) == zero &&
           saturating_div(high, two) == (high >> 1);
}

static_assert(signed_operations<int8>(INT8_MIN, INT8_MAX));
static_assert(signed_operations<int16>(INT16_MIN, INT16_MAX));
static_assert(signed_operations<int32>(INT32_MIN, INT32_MAX));
static_assert(signed_operations<int64>(INT64_MIN, INT64_MAX));
static_assert(unsigned_operations<uint8>(UINT8_MAX));
static_assert(unsigned_operations<uint16>(UINT16_MAX));
static_assert(unsigned_operations<uint32>(UINT32_MAX));
static_assert(unsigned_operations<uint64>(UINT64_MAX));

constexpr varying int32 clamped = clamp(programIndex, (varying int32)1, (varying int32)2);
constexpr varying int8 sat_add = saturating_add((varying int8)126, (varying int8)programIndex);
constexpr varying uint16 sat_sub = saturating_sub((varying uint16)programIndex, (varying uint16)2);
constexpr varying uint64 quotient = saturating_div((varying uint64)programIndex, (varying uint64)2);
constexpr varying int8 average_up = avg_up(-(varying int8)programIndex, (varying int8)-1);
constexpr varying int16 average_down = avg_down(-(varying int16)programIndex, (varying int16)-1);
constexpr varying uint8 unsigned_average_up = avg_up((varying uint8)programIndex, (varying uint8)255);
constexpr varying uint16 unsigned_average_down = avg_down((varying uint16)programIndex, (varying uint16)65535);

// CHECK-DAG: @clamped = {{.*}}constant <4 x i32> <i32 1, i32 1, i32 2, i32 2>
// CHECK-DAG: @sat_add = {{.*}}constant <4 x i8> <i8 126, i8 127, i8 127, i8 127>
// CHECK-DAG: @sat_sub = {{.*}}constant <4 x i16> <i16 0, i16 0, i16 0, i16 1>
// CHECK-DAG: @quotient = {{.*}}constant <4 x i64> <i64 0, i64 0, i64 1, i64 1>
// CHECK-DAG: @average_up = {{.*}}constant <4 x i8> <i8 0, i8 -1, i8 -1, i8 -2>
// CHECK-DAG: @average_down = {{.*}}constant <4 x i16> <i16 -1, i16 -1, i16 -2, i16 -2>
// CHECK-DAG: @unsigned_average_up = {{.*}}constant <4 x i8> <i8 -128, i8 -128, i8 -127, i8 -127>
// CHECK-DAG: @unsigned_average_down = {{.*}}constant <4 x i16> <i16 32767, i16 -32768, i16 -32768, i16 -32767>
