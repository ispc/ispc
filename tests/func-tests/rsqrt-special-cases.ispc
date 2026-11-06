#include "test_static.isph"

// Both signs of zero, Inf and quiet/signaling NaN, and ordinary inputs. Every lane runs every case.
// Keep the double ordinary inputs at 4/-4: native double _fast estimates have a different precision contract.
static const uniform int float_case_count = 18;
static const uniform int double_case_count = 10;
static const uniform unsigned int float_cases[float_case_count] = {
    0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
    0x7fa00000u, 0xffa00000u, 0x3e800000u, 0xbe800000u, 0x3f800000u, 0xbf800000u,
    0x40000000u, 0xc0000000u, 0x40800000u, 0xc0800000u, 0x41200000u, 0xc1200000u};
static const uniform unsigned int64 double_cases[double_case_count] = {
    0x0000000000000000, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000,
    0xfff8000000000000, 0x7ff4000000000000, 0xfff4000000000000, 0x4010000000000000, 0xc010000000000000};

static const uniform int range_case_count = 8;
static const uniform unsigned int64 double_range_cases[range_case_count] = {
    0x7fdfffffffffffff, 0xffdfffffffffffff, 0x7fe0000000000000, 0xffe0000000000000,
    0x7fe0000000000001, 0xffe0000000000001, 0x7fefffffffffffff, 0xffefffffffffffff};

// Expected: +-0 gives +-Inf, +Inf gives 0, NaN and every negative nonzero input give NaN, and an ordinary
// positive input gives a positive 1/sqrt(x) within the tolerance. Squaring alone cannot check the sign.
// Multiply x*y first to avoid the range problems of y*y for extreme inputs.
static inline bool check_float_rsqrt(float x, float y, uniform float max_error = 0.001f) {
    unsigned int input_bits = intbits(x);
    unsigned int abs_bits = input_bits & 0x7fffffffu;
    unsigned int sign = input_bits & 0x80000000u;
    unsigned int result_bits = intbits(y);
    return abs_bits == 0             ? (result_bits == (sign | 0x7f800000u))
           : abs_bits > 0x7f800000u  ? (y != y)
           : sign != 0               ? (y != y)
           : abs_bits == 0x7f800000u ? (result_bits == 0)
                                     : (y > 0.0f && abs((x * y) * y - 1.0f) < max_error);
}

static inline uniform bool check_float_rsqrt(uniform float x, uniform float y, uniform float max_error = 0.001f) {
    uniform unsigned int input_bits = intbits(x);
    uniform unsigned int abs_bits = input_bits & 0x7fffffffu;
    uniform unsigned int sign = input_bits & 0x80000000u;
    uniform unsigned int result_bits = intbits(y);
    return abs_bits == 0             ? (result_bits == (sign | 0x7f800000u))
           : abs_bits > 0x7f800000u  ? (y != y)
           : sign != 0               ? (y != y)
           : abs_bits == 0x7f800000u ? (result_bits == 0)
                                     : (y > 0.0f && abs((x * y) * y - 1.0f) < max_error);
}

static inline bool check_double_rsqrt(double x, double y) {
    unsigned int64 input_bits = intbits(x);
    unsigned int64 abs_bits = input_bits & 0x7fffffffffffffff;
    unsigned int64 sign = input_bits & 0x8000000000000000;
    unsigned int64 result_bits = intbits(y);
    return abs_bits == 0                    ? (result_bits == (sign | 0x7ff0000000000000))
           : abs_bits > 0x7ff0000000000000  ? (y != y)
           : sign != 0                      ? (y != y)
           : abs_bits == 0x7ff0000000000000 ? (result_bits == 0)
                                            : (y > 0.0d && abs((x * y) * y - 1.0d) < 1.0e-9d);
}

static inline uniform bool check_double_rsqrt(uniform double x, uniform double y) {
    uniform unsigned int64 input_bits = intbits(x);
    uniform unsigned int64 abs_bits = input_bits & 0x7fffffffffffffff;
    uniform unsigned int64 sign = input_bits & 0x8000000000000000;
    uniform unsigned int64 result_bits = intbits(y);
    return abs_bits == 0                    ? (result_bits == (sign | 0x7ff0000000000000))
           : abs_bits > 0x7ff0000000000000  ? (y != y)
           : sign != 0                      ? (y != y)
           : abs_bits == 0x7ff0000000000000 ? (result_bits == 0)
                                            : (y > 0.0d && abs((x * y) * y - 1.0d) < 1.0e-9d);
}

// Check class/sign, not a shared precision limit: native double _fast estimates are only 14-bit.
static inline bool check_double_rsqrt_range(double x, double y) {
    return (intbits(x) & 0x8000000000000000) != 0
               ? (y != y)
               : (y > 0.0d && (intbits(y) & 0x7fffffffffffffff) < 0x7ff0000000000000);
}

static inline uniform bool check_double_rsqrt_range(uniform double x, uniform double y) {
    return (intbits(x) & 0x8000000000000000) != 0
               ? (y != y)
               : (y > 0.0d && (intbits(y) & 0x7fffffffffffffff) < 0x7ff0000000000000);
}

// aFOO[0] is 1, so "zero" is 0 at run time and the compiler cannot fold the uniform inputs below.
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform unsigned int zero = (uniform unsigned int)(aFOO[0] - 1.0f);
    bool ok = true;

    for (uniform int k = 0; k < max(float_case_count, double_case_count); k++) {
        varying int idx = (programIndex + k) % float_case_count;
        varying int double_idx = (programIndex + k) % double_case_count;
        varying float x = floatbits(float_cases[idx]);
        varying double xd = doublebits(double_cases[double_idx]);

        ok = ok && check_float_rsqrt(x, rsqrt(x));
        ok = ok && check_float_rsqrt(x, rsqrt_fast(x), 0.01f);
        ok = ok && check_double_rsqrt(xd, rsqrt(xd));
        ok = ok && check_double_rsqrt(xd, rsqrt_fast(xd));

        // Partially active lanes: only the odd lanes make the calls.
        if ((programIndex & 1) == 1) {
            ok = ok && check_float_rsqrt(x, rsqrt(x));
            ok = ok && check_float_rsqrt(x, rsqrt_fast(x), 0.01f);
            ok = ok && check_double_rsqrt(xd, rsqrt(xd));
            ok = ok && check_double_rsqrt(xd, rsqrt_fast(xd));
        }

        uniform float ux = floatbits(float_cases[k % float_case_count] | zero);
        uniform double uxd = doublebits(double_cases[k % double_case_count] | zero);
        ok = ok && check_float_rsqrt(ux, rsqrt(ux));
        ok = ok && check_float_rsqrt(ux, rsqrt_fast(ux), 0.01f);
        ok = ok && check_double_rsqrt(uxd, rsqrt(uxd));
        ok = ok && check_double_rsqrt(uxd, rsqrt_fast(uxd));
    }

    for (uniform int k = 0; k < range_case_count; k++) {
        varying int idx = (programIndex + k) % range_case_count;
        varying double x = doublebits(double_range_cases[idx]);
        ok = ok && check_double_rsqrt_range(x, rsqrt(x));
        ok = ok && check_double_rsqrt_range(x, rsqrt_fast(x));
        if ((programIndex & 1) == 1) {
            ok = ok && check_double_rsqrt_range(x, rsqrt(x));
            ok = ok && check_double_rsqrt_range(x, rsqrt_fast(x));
        }
        uniform double ux = doublebits(double_range_cases[k] | zero);
        ok = ok && check_double_rsqrt_range(ux, rsqrt(ux));
        ok = ok && check_double_rsqrt_range(ux, rsqrt_fast(ux));
    }

    RET[programIndex] = ok ? 0.0f : 1.0f;
}

task void result(uniform float RET[]) { RET[programIndex] = 0.0f; }
