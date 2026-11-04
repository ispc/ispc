#include "test_static.isph"
// Float denormals. The x86 estimate instructions treat a denormal input like zero, which used to make the
// Newton-Raphson factor -Inf and flip the sign of rcp(). Other architectures refine differently and are not
// covered by this test.
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64

static inline uniform bool rcp_keeps_sign_uniform(uniform float x) {
    uniform float y = rcp(x);
    return (y == y) && ((intbits(y) & 0x80000000u) == (intbits(x) & 0x80000000u));
}

static inline bool rcp_keeps_sign_varying(float x) {
    float y = rcp(x);
    return (y == y) && ((intbits(y) & 0x80000000u) == (intbits(x) & 0x80000000u));
}

// The exact value differs between targets (+Inf on SSE and AVX2, a finite value on AVX-512), so only the
// sign is checked. NaN fails the comparison.
static inline uniform bool rsqrt_is_positive_uniform(uniform float x) { return rsqrt(x) > 0.0f; }

static inline bool rsqrt_is_positive_varying(float x) { return rsqrt(x) > 0.0f; }

task void f_v(uniform float RET[]) {
    // The smallest denormal, 1e-40f and the largest denormal.
    uniform unsigned int denormals[3] = {0x00000001u, 0x000116c2u, 0x007fffffu};
    bool ok = true;
    for (uniform int i = 0; i < 3; i++) {
        uniform float p = floatbits(denormals[i]);
        uniform float n = floatbits(denormals[i] | 0x80000000u);
        ok = ok && rcp_keeps_sign_uniform(p) && rcp_keeps_sign_uniform(n) && rsqrt_is_positive_uniform(p);

        varying float alternating = (programIndex & 1) ? n : p;
        varying float positive = p;
        ok = ok && rcp_keeps_sign_varying(alternating) && rsqrt_is_positive_varying(positive);
    }

    RET[programIndex] = ok ? 0.0f : 1.0f;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0f;
}
