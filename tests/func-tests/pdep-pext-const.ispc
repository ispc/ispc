#include "test_static.isph"
// pdep/pext with compile-time constant masks use a separate implementation.
// Compare it with runtime masks (made opaque with a runtime zero) and with a
// reference that scans all bit positions.

#define CHECK(n, cond)                                                                                                 \
    if (err == 0 && !(cond))                                                                                           \
        err = n;

static uint64 ref_pdep64(uint64 v, uint64 m) {
    uint64 r = 0;
    int k = 0;
    for (uniform int i = 0; i < 64; i++) {
        if (((m >> (uint64)i) & 1ull) != 0ull) {
            if (((v >> (uint64)k) & 1ull) != 0ull)
                r |= 1ull << (uint64)i;
            k++;
        }
    }
    return r;
}

static uint64 ref_pext64(uint64 v, uint64 m) {
    uint64 r = 0;
    int k = 0;
    for (uniform int i = 0; i < 64; i++) {
        if (((m >> (uint64)i) & 1ull) != 0ull) {
            if (((v >> (uint64)i) & 1ull) != 0ull)
                r |= 1ull << (uint64)k;
            k++;
        }
    }
    return r;
}

static uint64 next(uint64 &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Each case is in its own noinline function to keep the -O0 stack frames small
// (the wasm32 stack is 64 KB). M is a constant; M ^ z is the same mask, unknown
// at compile time.

#define TEST32(n, M)                                                                                                   \
    static noinline int test32_##n(uint32 v32, uint32 z32, uniform uint32 u32, uniform uint32 uz32) {                  \
        int err = 0;                                                                                                   \
        CHECK(n, pdep32(v32, M) == pdep32(v32, M ^ z32));                                                              \
        CHECK(n, pext32(v32, M) == pext32(v32, M ^ z32));                                                              \
        CHECK(n, pdep32(v32, M) == (uint32)ref_pdep64(v32, M));                                                        \
        CHECK(n, pext32(v32, M) == (uint32)ref_pext64(v32, M));                                                        \
        CHECK(n, pdep32(u32, M) == pdep32(u32, M ^ uz32));                                                             \
        CHECK(n, pext32(u32, M) == pext32(u32, M ^ uz32));                                                             \
        return err;                                                                                                    \
    }

#define TEST64(n, M)                                                                                                   \
    static noinline int test64_##n(uint64 v64, uint64 z64, uniform uint64 u64, uniform uint64 uz64) {                  \
        int err = 0;                                                                                                   \
        CHECK(n, pdep64(v64, M) == pdep64(v64, M ^ z64));                                                              \
        CHECK(n, pext64(v64, M) == pext64(v64, M ^ z64));                                                              \
        CHECK(n, pdep64(v64, M) == ref_pdep64(v64, M));                                                                \
        CHECK(n, pext64(v64, M) == ref_pext64(v64, M));                                                                \
        CHECK(n, pdep64(u64, M) == pdep64(u64, M ^ uz64));                                                             \
        CHECK(n, pext64(u64, M) == pext64(u64, M ^ uz64));                                                             \
        return err;                                                                                                    \
    }

TEST32(1, 0u)
TEST32(2, 0xFFFFFFFFu)
TEST32(3, 1u)
TEST32(4, 0x80000000u)
TEST32(5, 0x55555555u)
TEST32(6, 0xAAAAAAAAu)
TEST32(7, 0x49249249u)
TEST32(8, 0x92492492u)
TEST32(9, 0x0000FF00u)
TEST32(10, 0xFF00FF00u)
TEST32(11, 0x0F0F0F0Fu)
TEST32(12, 0x00F00001u)
TEST32(13, 0x80000001u)
TEST32(14, 0x7FFFFFFEu)
TEST32(15, 0x12345678u)
TEST32(16, 0xDEADBEEFu)
TEST32(17, 0x00010000u)

TEST64(21, 0ull)
TEST64(22, 0xFFFFFFFFFFFFFFFFull)
TEST64(23, 1ull)
TEST64(24, 0x8000000000000000ull)
TEST64(25, 0x5555555555555555ull)
TEST64(26, 0xAAAAAAAAAAAAAAAAull)
TEST64(27, 0x1249249249249249ull)
TEST64(28, 0x2492492492492492ull)
TEST64(29, 0x4924924924924924ull)
TEST64(30, 0x00000000FFFFFFFFull)
TEST64(31, 0xFFFFFFFF00000000ull)
TEST64(32, 0x0000000180000000ull)
TEST64(33, 0x8000000000000001ull)
TEST64(34, 0x123456789ABCDEF0ull)
TEST64(35, 0xF0F0F0F0F0F0F0F0ull)
TEST64(36, 0xFEDCBA9876543210ull)

static noinline int test_lane_masks(uint32 v32, uint32 z32, uint64 v64, uint64 z64) {
    int err = 0;
    // Constant masks that differ between lanes.
    uint32 lm32 = 0x55555555u << (uint32)(programIndex % 32);
    uint64 lm64 = (uint64)(programIndex + 1) * 0x0101010101010101ull;
    CHECK(41, pdep32(v32, lm32) == pdep32(v32, lm32 ^ z32));
    CHECK(42, pext32(v32, lm32) == pext32(v32, lm32 ^ z32));
    CHECK(43, pdep64(v64, lm64) == pdep64(v64, lm64 ^ z64));
    CHECK(44, pext64(v64, lm64) == pext64(v64, lm64 ^ z64));
    return err;
}

static noinline int test_mixed_masks(uint32 v32, uint32 z32, uint64 v64, uint64 z64) {
    int err = 0;
    // Lane-varying constant masks mixing zero, all-ones, and sparse masks.
    uint32 zm32 = (programIndex % 3 == 0) ? 0u : ((programIndex % 3 == 1) ? 0xFFFFFFFFu : 0x92492492u);
    uint64 zm64 =
        (programIndex % 3 == 0) ? 0ull : ((programIndex % 3 == 1) ? 0xFFFFFFFFFFFFFFFFull : 0x8102040810204080ull);
    CHECK(45, pdep32(v32, zm32) == pdep32(v32, zm32 ^ z32));
    CHECK(46, pext32(v32, zm32) == pext32(v32, zm32 ^ z32));
    CHECK(47, pdep64(v64, zm64) == pdep64(v64, zm64 ^ z64));
    CHECK(48, pext64(v64, zm64) == pext64(v64, zm64 ^ z64));
    return err;
}

static noinline int test_partial_mask(uint32 v32, uint64 v64) {
    int err = 0;
    // Partial execution mask; inactive lanes keep their values.
    uint32 r32 = 7;
    uint64 r64 = 7;
    if ((programIndex & 1) != 0) {
        r32 = pdep32(v32, 0x0F0F0F0Fu);
        r64 = pext64(v64, 0x5555555555555555ull);
    }
    if ((programIndex & 1) != 0) {
        CHECK(51, r32 == (uint32)ref_pdep64(v32, 0x0F0F0F0Full));
        CHECK(52, r64 == ref_pext64(v64, 0x5555555555555555ull));
    } else {
        CHECK(53, r32 == 7u);
        CHECK(54, r64 == 7ull);
    }
    return err;
}

#define RUN(call)                                                                                                      \
    if (err == 0)                                                                                                      \
        err = call;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int err = 0;
    // Runtime zeros: aFOO[i] == i + 1.
    uniform uint32 uz32 = (uniform uint32)(aFOO[0] - 1);
    uniform uint64 uz64 = uz32;
    uint32 z32 = (uint32)(aFOO[programIndex] - 1 - programIndex);
    uint64 z64 = z32;
    uint64 s = 0x9E3779B97F4A7C15ull * (uint64)aFOO[programIndex];

    for (uniform int iter = 0; iter < 32; iter++) {
        uint64 v64 = next(s);
        uint32 v32 = (uint32)(v64 >> 13);
        uniform uint64 u64 = extract(v64, 0);
        uniform uint32 u32 = extract(v32, 0);

        RUN(test32_1(v32, z32, u32, uz32));
        RUN(test32_2(v32, z32, u32, uz32));
        RUN(test32_3(v32, z32, u32, uz32));
        RUN(test32_4(v32, z32, u32, uz32));
        RUN(test32_5(v32, z32, u32, uz32));
        RUN(test32_6(v32, z32, u32, uz32));
        RUN(test32_7(v32, z32, u32, uz32));
        RUN(test32_8(v32, z32, u32, uz32));
        RUN(test32_9(v32, z32, u32, uz32));
        RUN(test32_10(v32, z32, u32, uz32));
        RUN(test32_11(v32, z32, u32, uz32));
        RUN(test32_12(v32, z32, u32, uz32));
        RUN(test32_13(v32, z32, u32, uz32));
        RUN(test32_14(v32, z32, u32, uz32));
        RUN(test32_15(v32, z32, u32, uz32));
        RUN(test32_16(v32, z32, u32, uz32));
        RUN(test32_17(v32, z32, u32, uz32));

        RUN(test64_21(v64, z64, u64, uz64));
        RUN(test64_22(v64, z64, u64, uz64));
        RUN(test64_23(v64, z64, u64, uz64));
        RUN(test64_24(v64, z64, u64, uz64));
        RUN(test64_25(v64, z64, u64, uz64));
        RUN(test64_26(v64, z64, u64, uz64));
        RUN(test64_27(v64, z64, u64, uz64));
        RUN(test64_28(v64, z64, u64, uz64));
        RUN(test64_29(v64, z64, u64, uz64));
        RUN(test64_30(v64, z64, u64, uz64));
        RUN(test64_31(v64, z64, u64, uz64));
        RUN(test64_32(v64, z64, u64, uz64));
        RUN(test64_33(v64, z64, u64, uz64));
        RUN(test64_34(v64, z64, u64, uz64));
        RUN(test64_35(v64, z64, u64, uz64));
        RUN(test64_36(v64, z64, u64, uz64));
        RUN(test_lane_masks(v32, z32, v64, z64));
        RUN(test_mixed_masks(v32, z32, v64, z64));
        RUN(test_partial_mask(v32, v64));
    }

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
