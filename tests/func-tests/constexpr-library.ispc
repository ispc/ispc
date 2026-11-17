// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"

// Compare constant evaluation with runtime evaluation using the same helpers.
constexpr uniform int logical_and(uniform bool b) {
    uniform int n = 0;
    uniform bool value = b && (++n > 0);
    return n;
}

constexpr uniform int logical_or(uniform bool b) {
    uniform int n = 0;
    uniform bool value = b || (++n > 0);
    return n;
}

constexpr uniform int conditional(uniform bool b) {
    uniform int n = 0;
    uniform int value = b ? ++n : ++n;
    return n;
}

constexpr uniform int and_true = logical_and(true);
constexpr uniform int and_false = logical_and(false);
constexpr uniform int or_true = logical_or(true);
constexpr uniform int or_false = logical_or(false);
constexpr uniform int select_true = conditional(true);
constexpr uniform int select_false = conditional(false);
constexpr uniform float newton_positive = _cbrt_newton_iter(2.f, 20.f);
constexpr uniform float newton_other = _cbrt_newton_iter(1.f, 1.f);
constexpr uniform double halley_positive = _cbrt_halley_iter(1.d, 4.d);
constexpr uniform double halley_other = _cbrt_halley_iter(2.d, 8.d);
constexpr uniform int8 magnitude_small = abs((uniform int8)-3);
constexpr uniform int16 magnitude_medium = abs((uniform int16)-3);
constexpr uniform int32 magnitude_int = abs((uniform int32)-3);
constexpr uniform int64 magnitude_long = abs((uniform int64)-3);
constexpr uniform bool nan_float = isnan(0.0f / 0.0f);
constexpr uniform bool nan_double = isnan(0.0d / 0.0d);

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform bool b = aFOO[0] > 0;
    uniform bool logic_ok = logical_and(b) == (b ? and_true : and_false) &&
                            logical_or(b) == (b ? or_true : or_false) &&
                            conditional(b) == (b ? select_true : select_false);
    bool lane_positive = aFOO[programIndex] > 0;
    float newton = _cbrt_newton_iter(lane_positive ? 2.f : 1.f, lane_positive ? 20.f : 1.f);
    double halley = _cbrt_halley_iter(lane_positive ? 1.d : 2.d, lane_positive ? 4.d : 8.d);
    bool math_ok = newton == (lane_positive ? newton_positive : newton_other) &&
                   halley == (lane_positive ? halley_positive : halley_other);
    uniform int uniform_value = b ? -3 : 3;
    int lane_value = lane_positive ? -3 : 3;
    uniform bool uniform_abs_ok = abs((uniform int8)uniform_value) == magnitude_small &&
                                  abs((uniform int16)uniform_value) == magnitude_medium &&
                                  abs((uniform int32)uniform_value) == magnitude_int &&
                                  abs((uniform int64)uniform_value) == magnitude_long;
    bool lane_abs_ok = abs((int8)lane_value) == magnitude_small && abs((int16)lane_value) == magnitude_medium &&
                       abs((int32)lane_value) == magnitude_int && abs((int64)lane_value) == magnitude_long;
    uniform uint32 uniform_bits = b ? 0x7fc00000u : 0x80000000u;
    uint32 lane_bits = lane_positive ? 0x7fc00000u : 0x80000000u;
    uniform float uniform_fp = floatbits(uniform_bits);
    float lane_fp = floatbits(lane_bits);
    uniform bool uniform_nan_ok = isnan(uniform_fp) == (b && nan_float) &&
                                  isnan((uniform double)uniform_fp) == (b && nan_double);
    bool lane_nan_ok = isnan(lane_fp) == (lane_positive && nan_float) &&
                       isnan((double)lane_fp) == (lane_positive && nan_double);
    RET[programIndex] = logic_ok && math_ok && uniform_abs_ok && lane_abs_ok && uniform_nan_ok && lane_nan_ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
