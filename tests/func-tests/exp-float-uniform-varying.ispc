#include "test_static.isph"

// Check that uniform and varying float exp() agree on every active lane.
// Results must have identical bits; any two NaNs are treated as agreeing.
// This checks consistency only, not accuracy against a reference.

static const uniform unsigned int32 input_bits[] = {
    // +-0, +-1, +-Inf, +-NaN, +-min/max subnormal, +-min normal
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
    0x7fc00000, 0xffc00000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
    0x00800000, 0x80800000,
    // Around the implementation's overflow cutoff (first Inf at 0x42b17218)
    0x42b17216, 0x42b17217, 0x42b17218, 0x42b17219,
    // Around the implementation's zero-result cutoff (first zero at 0xc2aeac51)
    0xc2aeac4f, 0xc2aeac50, 0xc2aeac51, 0xc2aeac52,
    // Deep underflow and large magnitudes: ~-103.97, -90, -100, -150, 150
    0xc2cff1b4, 0xc2b40000, 0xc2c80000, 0xc3160000, 0x43160000,
    // Reproducible pseudo-random values in [-100, 100]
    0x4039bd7e, 0xc268ecf4, 0x429b7229, 0x41010c4d, 0xc29c43aa, 0xc29a8aff,
    0x42c6dffc, 0xc1a3d6d9, 0x418efa72, 0xc2a3464f, 0xc2c3ea4a, 0x41d426fc,
    0x42b4f31b, 0xc0d3d9b9, 0xc2c2394f, 0xc2c13dfb, 0xc2a63f78, 0xc2a6689c,
    0xc2c3bda3, 0xc278c922, 0xc1fff23b, 0x429c3dee, 0xc221c474, 0xc2001fee,
    0x427d2777, 0xc2a27046, 0x41ffd42d, 0xc1a643a7, 0x42a5a225, 0x41b429e3,
    0x421f89cf, 0xc283c76e, 0xc231247c,
};

static uniform bool same_result(uniform float a, uniform float b) {
    if (isnan(a) || isnan(b))
        return isnan(a) && isnan(b);
    return intbits(a) == intbits(b);
}

// mask_case: 0 = all lanes, 1 = bounded (partial last block), 2 = odd lanes,
// 3 = even lanes.
static uniform int count_mismatches(float x, uniform int mask_case) {
    const uniform unsigned int64 mask = lanemask();
    uniform int mismatches = 0;
    float vres = exp(x);
    foreach_active (lane) {
        uniform float xin = extract(x, lane);
        uniform float ures = exp(xin);
        uniform float vout = extract(vres, lane);
        if (!same_result(vout, ures)) {
            print("exp mismatch: mask case %, mask %, lane %, input bits %, "
                  "uniform bits %, varying bits %\n",
                  mask_case, mask, lane, intbits(xin), intbits(ures), intbits(vout));
            ++mismatches;
        }
    }
    return mismatches;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
#if ISPC_MATH_LIB_VAL == ISPC_MATH_LIB_SVML_VAL
    // With --math-lib=svml, varying exp() uses SVML while uniform exp() uses
    // the C library, so their results are not expected to match.
    RET[programIndex] = 0;
#else
    const uniform int count = sizeof(input_bits) / sizeof(input_bits[0]);
    // aFOO[i] == i + 1; deriving the lane from it keeps inputs runtime values.
    const int lane = (int)aFOO[programIndex] - 1;
    uniform int mismatches = 0;

    for (uniform int base = 0; base < count; base += programCount) {
        const int i = base + lane;
        // All lanes active; indices past the end wrap around to the start.
        mismatches += count_mismatches(floatbits(input_bits[i % count]), 0);
        // The last block has a partial mask when count % programCount != 0.
        if (i < count) {
            mismatches += count_mismatches(floatbits(input_bits[i]), 1);
            // Repeat with only odd lanes, then only even lanes, active.
            if ((lane & 1) == 1)
                mismatches += count_mismatches(floatbits(input_bits[i]), 2);
            else
                mismatches += count_mismatches(floatbits(input_bits[i]), 3);
        }
    }

    RET[programIndex] = mismatches;
#endif
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
