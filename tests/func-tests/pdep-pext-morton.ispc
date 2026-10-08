#include "test_static.isph"
// Morton encode/decode with pdep/pext, checked against independent bit
// interleaving.

#define CHECK(n, cond)                                                                                                 \
    if (err == 0 && !(cond))                                                                                           \
        err = n;

// Interleave the low 16 (2D, 32-bit code) or 32 (2D, 64-bit code) bits of x
// and y: x goes to the even bits and y to the odd bits.
static uint64 interleave2(uint64 x, uint64 y, uniform int bits) {
    uint64 r = 0;
    for (uniform int i = 0; i < bits; i++) {
        r |= ((x >> (uint64)i) & 1ull) << (uint64)(2 * i);
        r |= ((y >> (uint64)i) & 1ull) << (uint64)(2 * i + 1);
    }
    return r;
}

// Interleave the low 21 bits of x, y, and z into a 63-bit code.
static uint64 interleave3(uint64 x, uint64 y, uint64 z) {
    uint64 r = 0;
    for (uniform int i = 0; i < 21; i++) {
        r |= ((x >> (uint64)i) & 1ull) << (uint64)(3 * i);
        r |= ((y >> (uint64)i) & 1ull) << (uint64)(3 * i + 1);
        r |= ((z >> (uint64)i) & 1ull) << (uint64)(3 * i + 2);
    }
    return r;
}

task void f_v(uniform float RET[]) {
    int err = 0;

    for (uniform int iter = 0; iter < 64; iter++) {
        uint32 x = (uint32)(programIndex * 2654435761u + iter * 40503u);
        uint32 y = (uint32)(programIndex * 2246822519u ^ iter * 3266489917u);
        uint32 x16 = x & 0xFFFFu;
        uint32 y16 = y & 0xFFFFu;

        // 2D, 32-bit code.
        uint32 code32 = pdep32(x16, 0x55555555u) | pdep32(y16, 0xAAAAAAAAu);
        CHECK(1, code32 == (uint32)interleave2(x16, y16, 16));
        CHECK(2, pext32(code32, 0x55555555u) == x16);
        CHECK(3, pext32(code32, 0xAAAAAAAAu) == y16);

        // 2D, 64-bit code.
        uint64 code64 = pdep64(x, 0x5555555555555555ull) | pdep64(y, 0xAAAAAAAAAAAAAAAAull);
        CHECK(4, code64 == interleave2(x, y, 32));
        CHECK(5, pext64(code64, 0x5555555555555555ull) == (uint64)x);
        CHECK(6, pext64(code64, 0xAAAAAAAAAAAAAAAAull) == (uint64)y);

        // 3D, 63-bit code.
        uint64 x21 = x & 0x1FFFFFu;
        uint64 y21 = y & 0x1FFFFFu;
        uint64 z21 = (x ^ y) & 0x1FFFFFu;
        uint64 code3 = pdep64(x21, 0x1249249249249249ull) | pdep64(y21, 0x2492492492492492ull) |
                       pdep64(z21, 0x4924924924924924ull);
        CHECK(7, code3 == interleave3(x21, y21, z21));
        CHECK(8, pext64(code3, 0x1249249249249249ull) == x21);
        CHECK(9, pext64(code3, 0x2492492492492492ull) == y21);
        CHECK(10, pext64(code3, 0x4924924924924924ull) == z21);

        // Uniform coordinates.
        uniform uint32 ux = extract(x16, 0);
        uniform uint32 uy = extract(y16, 0);
        uniform uint32 ucode = pdep32(ux, 0x55555555u) | pdep32(uy, 0xAAAAAAAAu);
        CHECK(11, ucode == extract(code32, 0));
        CHECK(12, pext32(ucode, 0xAAAAAAAAu) == uy);
    }

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
