// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// RUN: %{ispc} %s --target=host --nowrap --werror --emit-llvm-text -o - | FileCheck %s

// Use integral assertions to distinguish both kinds of NaN and their payloads.
constexpr uniform float16 half_snan = float16bits((uniform uint16)0xfc35);
constexpr uniform float float_snan = floatbits((uniform int32)0xff800135);
constexpr uniform double double_snan = doublebits(0xfff0000000000135ull);
static_assert(intbits(half_snan) == 0xfc35);
static_assert(intbits(float_snan) == 0xff800135u);
static_assert(intbits(double_snan) == 0xfff0000000000135ull);
static_assert(intbits(float16bits((uniform int16)0x7e35)) == 0x7e35);
static_assert(intbits(floatbits(0x7fc00135u)) == 0x7fc00135u);
static_assert(intbits(doublebits(0x7ff8000000000135ull)) == 0x7ff8000000000135ull);
static_assert(intbits(abs(half_snan)) == 0x7c35);
static_assert(intbits(abs(float_snan)) == 0x7f800135u);
static_assert(intbits(abs(double_snan)) == 0x7ff0000000000135ull);
static_assert(intbits(abs(float16bits((uniform uint16)0x8000))) == 0);
static_assert(signbits(floatbits(0x80000000u)) == 0x80000000u);
static_assert(signbits(doublebits(0x8000000000000000ull)) == 0x8000000000000000ull);
static_assert(isinf(float16bits((uniform uint16)0xfc00)) && !isinf(half_snan));
static_assert(isinf(floatbits(0xff800000u)) && !isinf(float_snan));
static_assert(isinf(doublebits(0xfff0000000000000ull)) && !isinf(double_snan));
static_assert(isfinite(float16bits((uniform uint16)0x0001)) && !isfinite(half_snan));
static_assert(isfinite(floatbits(0x00000001u)) && !isfinite(float_snan));
static_assert(isfinite(doublebits(0x0000000000000001ull)) && !isfinite(double_snan));

// Both operands of the intrinsic are evaluated, including the type operand.
constexpr uniform bool type_operand_side_effect() {
    uniform float marker = 0;
    uniform float value = @llvm.ispc.bitcast(0x80000000u, ++marker);
    return marker == 1 && intbits(value) == 0x80000000u;
}
static_assert(type_operand_side_effect());

// Confirm that emitting a constexpr value also preserves its payload.
varying uint32 compiled_lanes() {
    constexpr varying float value = floatbits((varying uint32)(programIndex & 1 ? 0x7f800135u : 0x80000000u));
    return intbits(value);
}
// CHECK-LABEL: @compiled_lanes
// CHECK: ret <{{[0-9]+}} x i32> <i32 -2147483648, i32 2139095349{{.*}}>

// The same public conversion still accepts runtime operands.
varying uint32 runtime_sign(varying uint32 bits) { return signbits(floatbits(bits)); }
// CHECK-LABEL: @runtime_sign
// CHECK: and <{{[0-9]+}} x i32> {{.*}}, {{.*}}-2147483648
// CHECK: ret <{{[0-9]+}} x i32>
