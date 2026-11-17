#include "test_static.isph"
// rule: skip on arch=ppc64le

// Keep half runtime parity separate: PPC lacks the required conversion routines.
constexpr uniform bool expected_nan = isnan(0.0f16 / 0.0f16);
constexpr uniform bool expected_non_nan = isnan(-0.0f16);
task void f_v(uniform float RET[]) {
    float errors = 0;
    for (uniform int i = -2; i <= 2; ++i) {
        float16 f = log((float16)i);

        if ((i < 0) && (isnan(f) != expected_nan))
            errors++;
        if ((i >= 0) && (isnan(f) != expected_non_nan))
            errors++;
    }

    for (uniform int i = -2; i <= 2; ++i) {
        uniform float16 f = log((uniform float16)i);

        if ((i < 0) && (isnan(f) != expected_nan))
            errors++;
        if ((i >= 0) && (isnan(f) != expected_non_nan))
            errors++;
    }
    RET[programIndex] = errors;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
