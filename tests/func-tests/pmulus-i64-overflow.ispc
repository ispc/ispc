// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"
task void f_v(uniform float RET[]) {
    uniform uint64 a[] = {0x8000000000000000, 0x8000000000000001, 0xffffffff00000000, 0xffffffff00000000,
                          0x100000001,        0x100000002,        0xffffffff,         UINT64_MAX,
                          UINT64_MAX,         0x100000000,        0x100000000,        0x4000000000000000};
    uniform uint64 b[] = {2, 2, 1, 2, 0xffffffff, 0xffffffff, 0xffffffff, 0, 1, 0x100000000, 0xffffffff, 2};
    uniform uint64 expected[] = {UINT64_MAX, UINT64_MAX, 0xffffffff00000000, UINT64_MAX,
                                 UINT64_MAX, UINT64_MAX, 0xfffffffe00000001, 0,
                                 UINT64_MAX, UINT64_MAX, 0xffffffff00000000, 0x8000000000000000};

    RET[programIndex] = 0;
    for (uniform int i = 0; i < 12; ++i) {
        // Compare integers before converting to the test harness's float result.
        if (saturating_mul(a[i], b[i]) != expected[i] || saturating_mul(b[i], a[i]) != expected[i])
            RET[programIndex] = 1;

        int j = (i + programIndex) % 12;
        uint64 va = a[j], vb = b[j];
        if (saturating_mul(va, vb) != expected[j] || saturating_mul(vb, va) != expected[j])
            RET[programIndex] = 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
