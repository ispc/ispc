#include "test_static.isph"
// pdep/pext under divergent control flow, nested masks, and a partial tail
// gang. Inactive lanes must not affect active lanes or their own values.

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

task void f_v(uniform float RET[]) {
    int err = 0;
    uint64 v = 0xF0E1D2C3B4A59687ull * (uint64)(programIndex + 3);
    uint64 m = 0x0123456789ABCDEFull ^ ((uint64)programIndex * 0x1111111111111111ull);
    // Lanes with different mask populations finish at different times.
    m = m >> (uint64)(programIndex % 8);
    uint32 v32 = (uint32)(v >> 7);
    uint32 m32 = (uint32)(m >> 3);

    // Divergent branches.
    uint64 r64 = 7;
    uint32 r32 = 7;
    if ((programIndex & 1) != 0) {
        r64 = pdep64(v, m);
        r32 = pext32(v32, m32);
    } else {
        r64 = pext64(v, m);
        r32 = pdep32(v32, m32);
    }
    if ((programIndex & 1) != 0) {
        CHECK(1, r64 == ref_pdep64(v, m));
        CHECK(2, r32 == (uint32)ref_pext64((uint64)v32, (uint64)m32));
    } else {
        CHECK(3, r64 == ref_pext64(v, m));
        CHECK(4, r32 == (uint32)ref_pdep64((uint64)v32, (uint64)m32));
    }

    // Nested masks; inactive lanes keep their previous values.
    r64 = 7;
    r32 = 7;
    if (programIndex % 3 != 0) {
        if (programIndex > 1) {
            r64 = pext64(v, m);
            r32 = pdep32(v32, m32);
        }
    }
    if (programIndex % 3 != 0 && programIndex > 1) {
        CHECK(11, r64 == ref_pext64(v, m));
        CHECK(12, r32 == (uint32)ref_pdep64((uint64)v32, (uint64)m32));
    } else {
        CHECK(13, r64 == 7ull);
        CHECK(14, r32 == 7u);
    }

    // Operands only defined in the active lanes.
    uint64 partial_v, partial_m;
    if (programIndex < 2) {
        partial_v = v;
        partial_m = m;
    }
    if (programIndex < 2) {
        CHECK(21, pdep64(partial_v, partial_m) == ref_pdep64(v, m));
        CHECK(22, pext64(partial_v, partial_m) == ref_pext64(v, m));
        CHECK(23,
              pdep32((uint32)partial_v, (uint32)partial_m) == (uint32)ref_pdep64(v & 0xFFFFFFFFull, m & 0xFFFFFFFFull));
        CHECK(24,
              pext32((uint32)partial_v, (uint32)partial_m) == (uint32)ref_pext64(v & 0xFFFFFFFFull, m & 0xFFFFFFFFull));
    }

    // Tail gang with fewer active lanes.
    uniform uint64 out64[2 * programCount];
    uniform uint32 out32[2 * programCount];
    uniform int n = programCount + programCount / 2 + 1;
    for (uniform int i = 0; i < 2 * programCount; i++) {
        out64[i] = 7;
        out32[i] = 7;
    }
    foreach (i = 0 ... n) {
        uint64 tv = v + (uint64)i;
        out64[i] = pdep64(tv, m);
        out32[i] = pext32((uint32)tv, m32);
    }
    for (uniform int base = 0; base < 2 * programCount; base += programCount) {
        int i = base + programIndex;
        uint64 tv = v + (uint64)i;
        if (i < n) {
            CHECK(31, out64[i] == ref_pdep64(tv, m));
            CHECK(32, out32[i] == (uint32)ref_pext64(tv & 0xFFFFFFFFull, (uint64)m32));
        } else {
            CHECK(33, out64[i] == 7ull);
            CHECK(34, out32[i] == 7u);
        }
    }

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
