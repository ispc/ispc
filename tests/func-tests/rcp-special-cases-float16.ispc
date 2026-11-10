#include "test_static.isph"
// rule: skip on arch=ppc64le

static inline bool check_half_rcp(float x, float y) {
    unsigned int input_bits = intbits(x);
    unsigned int abs_bits = input_bits & 0x7fffffffu;
    unsigned int sign = input_bits & 0x80000000u;
    unsigned int result_bits = intbits(y);
    bool zero_ok = (abs_bits != 0) || (result_bits == (sign | 0x7f800000u));
    bool infinity_ok = (abs_bits != 0x7f800000u) || (result_bits == sign);
    bool ordinary_ok = (x != 4.0f) || (y > 0.249f && y < 0.251f);
    return zero_ok && infinity_ok && ordinary_ok;
}

task void f_v(uniform float RET[]) {
    bool ok = true;
    // Rotate the cases so every lane checks 4.0f even on targets with fewer than five lanes.
    for (uniform int k = 0; k < 5; k++) {
        varying int idx = (programIndex + k) % 5;
        varying float x = 4.0f;
        if (idx == 0) {
            x = 0.0f;
        } else if (idx == 1) {
            x = floatbits(0x80000000u);
        } else if (idx == 2) {
            x = floatbits(0x7f800000u);
        } else if (idx == 3) {
            x = floatbits(0xff800000u);
        }

        varying float16 xh = (float16)x;
        ok = ok && check_half_rcp(x, (float)rcp(xh));
    }

    uniform float inputs[5] = {0.0f, 0.0f, 0.0f, 0.0f, 4.0f};
    inputs[1] = floatbits(0x80000000u);
    inputs[2] = floatbits(0x7f800000u);
    inputs[3] = floatbits(0xff800000u);
    for (uniform int i = 0; i < 5; i++) {
        ok = ok && check_half_rcp(inputs[i], (uniform float)rcp((uniform float16)inputs[i]));
    }

    RET[programIndex] = ok ? 0.0f : 1.0f;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0f;
}
