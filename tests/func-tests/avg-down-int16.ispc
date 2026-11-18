// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float bf) {
    int16 a = aFOO[programIndex];
    int16 b = bf;
    bool negative = (programIndex & 2) != 0;
    RET[programIndex] = avg_down(negative ? -a : a, negative ? -b : b);
}

task void result(uniform float RET[]) {
    int sum = (int)programIndex + 1 + 5;
    RET[programIndex] = (programIndex & 2) != 0 ? -((sum + 1) / 2) : sum / 2;
}
