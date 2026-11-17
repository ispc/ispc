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
    RET[programIndex] = logic_ok && math_ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
