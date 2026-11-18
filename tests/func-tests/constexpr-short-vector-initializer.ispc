// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"

static noinline constexpr uniform int32<5> partial_same(uniform int16 first, uniform int16 second) {
    uniform int32<5> result = {first, second};
    return result;
}

static noinline constexpr uniform int64<5> partial_mixed(uniform int32 first, uniform int16 second, uniform int32 third) {
    uniform int64<5> result = {first, second, third};
    return result;
}

constexpr uniform int32<5> same = partial_same(71, -92);
constexpr uniform int64<5> mixed = partial_mixed(71, -92, 123456);
static_assert(same[0] == 71 && same[1] == -92 && same[2] == 0 && same[3] == 0 && same[4] == 0);
static_assert(mixed[0] == 71 && mixed[1] == -92 && mixed[2] == 123456 && mixed[3] == 0 && mixed[4] == 0);

task void f_f(uniform float RET[], uniform float input[]) {
    // The non-inlined callees initialize their vectors from runtime arguments.
    uniform int offset = (uniform int)input[0] & 7;
    uniform int32<5> runtime_same = partial_same(71 + offset, -92 - offset);
    uniform int64<5> runtime_mixed = partial_mixed(71 + offset, -92 - offset, 123456 + offset);
    uniform bool ok = runtime_same[0] - offset == same[0] && runtime_same[1] + offset == same[1] &&
                      runtime_mixed[0] - offset == mixed[0] && runtime_mixed[1] + offset == mixed[1] &&
                      runtime_mixed[2] - offset == mixed[2] && runtime_same[2] == same[2];
    for (uniform int i = 3; i < 5; ++i) {
        ok = ok && runtime_same[i] == same[i] && runtime_mixed[i] == mixed[i];
    }
    RET[programIndex] = ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
