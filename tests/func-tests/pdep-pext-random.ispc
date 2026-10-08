#include "test_static.isph"
// Compare pdep/pext against references that scan all bit positions, using
// deterministic pseudo-random values and masks of varying density.

#define CHECK(n, cond)                                                                                                 \
    if (err == 0 && !(cond))                                                                                           \
        err = n;

static uint64 ref_pdep64(uint64 v, uint64 m) {
    uint64 r = 0;
    int kv = 0;
    for (uniform int i = 0; i < 64; i++) {
        if (((m >> (uint64)i) & 1ull) != 0ull) {
            if (((v >> (uint64)kv) & 1ull) != 0ull)
                r |= 1ull << (uint64)i;
            kv++;
        }
    }
    return r;
}

static uint64 ref_pext64(uint64 v, uint64 m) {
    uint64 r = 0;
    int kv = 0;
    for (uniform int i = 0; i < 64; i++) {
        if (((m >> (uint64)i) & 1ull) != 0ull) {
            if (((v >> (uint64)i) & 1ull) != 0ull)
                r |= 1ull << (uint64)kv;
            kv++;
        }
    }
    return r;
}

// Low n bits set, valid for n == 64.
static uint64 low_bits64(int n) { return n == 64 ? 0xFFFFFFFFFFFFFFFFull : (1ull << (uint64)n) - 1ull; }

static uint64 next(uint64 &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

task void f_v(uniform float RET[]) {
    int err = 0;
    uint64 s = 0x9E3779B97F4A7C15ull * (uint64)(programIndex + 1);

    for (uniform int iter = 0; iter < 256; iter++) {
        uint64 v = next(s);
        uint64 r1 = next(s);
        uint64 r2 = next(s);
        uint64 m;
        switch (iter % 4) {
        case 0:
            m = r1 & r2 & next(s); // sparse
            break;
        case 1:
            m = r1 | r2; // dense
            break;
        case 2:
            m = r1;
            break;
        default:
            m = r1 >> (uint64)(iter % 64); // only low bits
            break;
        }
        uint32 v32 = (uint32)v;
        uint32 m32 = (uint32)(m >> 16);

        // Varying operations.
        uint64 d64 = pdep64(v, m);
        uint64 e64 = pext64(v, m);
        uint32 d32 = pdep32(v32, m32);
        uint32 e32 = pext32(v32, m32);
        CHECK(1, d64 == ref_pdep64(v, m));
        CHECK(2, e64 == ref_pext64(v, m));
        CHECK(3, d32 == (uint32)ref_pdep64((uint64)v32, (uint64)m32));
        CHECK(4, e32 == (uint32)ref_pext64((uint64)v32, (uint64)m32));

        // Round trips.
        CHECK(5, pdep64(e64, m) == (v & m));
        CHECK(6, pdep32(e32, m32) == (v32 & m32));
        CHECK(7, pext64(d64, m) == (v & low_bits64(popcnt((int64)m))));
        CHECK(8, pext32(d32, m32) == (v32 & (uint32)low_bits64(popcnt((int32)m32))));

        // Uniform and mixed operations, one lane at a time.
        for (uniform int lane = 0; lane < programCount; lane++) {
            uniform uint64 uv = extract(v, lane);
            uniform uint64 um = extract(m, lane);
            uniform uint32 uv32 = extract(v32, lane);
            uniform uint32 um32 = extract(m32, lane);
            if (programIndex == lane) {
                CHECK(11, pdep64(uv, um) == d64);
                CHECK(12, pext64(uv, um) == e64);
                CHECK(13, pdep32(uv32, um32) == d32);
                CHECK(14, pext32(uv32, um32) == e32);
            }
            // Uniform value with varying mask, and varying value with uniform mask.
            CHECK(15, pdep64(uv, m) == ref_pdep64(uv, m));
            CHECK(16, pext64(v, um) == ref_pext64(v, um));
            CHECK(17, pdep32(v32, um32) == (uint32)ref_pdep64((uint64)v32, (uint64)um32));
            CHECK(18, pext32(uv32, m32) == (uint32)ref_pext64((uint64)uv32, (uint64)m32));
        }
    }

    // Full-width masks in the round trips.
    uint64 v = next(s);
    CHECK(21, pext64(pdep64(v, 0xFFFFFFFFFFFFFFFFull), 0xFFFFFFFFFFFFFFFFull) == v);
    CHECK(22, pext32(pdep32((uint32)v, 0xFFFFFFFFu), 0xFFFFFFFFu) == (uint32)v);

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
