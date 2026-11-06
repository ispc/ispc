#include "test_static.isph"
// rule: skip on arch=ppc64le

static inline bool check_half_rsqrt(float x, float y) {
    unsigned int result_bits = intbits(y);
    unsigned int zero_result_bits = (intbits(x) & 0x80000000u) | 0x7f800000u;
    bool zero_ok = (x != 0.0f) || (result_bits == zero_result_bits);
    bool infinity_ok = (x != floatbits(0x7f800000u)) || (result_bits == 0);
    bool ordinary_ok = (x != 4.0f) || (y > 0.49f && y < 0.51f);
    return zero_ok && infinity_ok && ordinary_ok;
}

task void f_v(uniform float RET[]) {
    varying float x = 4.0f;
    if ((programIndex & 3) == 0) {
        x = 0.0f;
    } else if ((programIndex & 3) == 1) {
        x = floatbits(0x7f800000u);
    } else if ((programIndex & 3) == 2) {
        x = floatbits(0x80000000u);
    }

    varying float16 xh = (float16)x;
    bool ok = check_half_rsqrt(x, (float)rsqrt(xh));

    uniform float positive_zero = 0.0f;
    uniform float positive_inf = floatbits(0x7f800000u);
    uniform float negative_zero = floatbits(0x80000000u);
    uniform float ordinary = 4.0f;
    ok = ok && check_half_rsqrt(positive_zero, (uniform float)rsqrt((uniform float16)positive_zero));
    ok = ok && check_half_rsqrt(positive_inf, (uniform float)rsqrt((uniform float16)positive_inf));
    ok = ok && check_half_rsqrt(negative_zero, (uniform float)rsqrt((uniform float16)negative_zero));
    ok = ok && check_half_rsqrt(ordinary, (uniform float)rsqrt((uniform float16)ordinary));

    RET[programIndex] = ok ? 0.0f : 1.0f;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0f;
}
