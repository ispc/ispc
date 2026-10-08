#include "test_static.isph"
// Signed, narrower, and wider operands convert to the unsigned type of the
// width-suffixed function: sign-extension for pdep64/pext64, truncation for
// pdep32/pext32.

#define CHECK(n, cond)                                                                                                 \
    if (err == 0 && !(cond))                                                                                           \
        err = n;

task void f_v(uniform float RET[]) {
    int err = 0;

    // Negative int32 is sign-extended for the 64-bit functions.
    uniform int32 un = -2;
    int32 vn = -2 - programIndex;
    CHECK(1, pdep64(un, 0xFFFFFFFF00000000ull) == 0xFFFFFFFE00000000ull);
    CHECK(2, pext64(un, 0xFFFFFFFF00000000ull) == 0xFFFFFFFFull);
    CHECK(3, pdep32(un, 0xFFFF0000u) == 0xFFFE0000u);
    CHECK(4, pext32(un, 0xFFFF0000u) == 0xFFFFu);
    CHECK(5, pext64(vn, 0xFFFFFFFF00000000ull) == 0xFFFFFFFFull);
    CHECK(6, pext32(vn, 0xFFFFFFFFu) == (uint32)vn);
    CHECK(7, pext64(vn, 0xFFFFFFFFFFFFFFFFull) == (uint64)(int64)vn);

    // uint64 operands are truncated by the 32-bit functions.
    uniform uint64 uw = 0x0000000100000003ull;
    uint64 vw = 0x0000000F00000003ull + ((uint64)programIndex << 32);
    CHECK(11, pdep32(uw, 0xFu) == 3u);
    CHECK(12, pext32(uw, 0xFFFFFFFFu) == 3u);
    CHECK(13, pdep32(vw, 0xFFFFFFFFu) == 3u);
    CHECK(14, pext32(vw, vw) == 3u);
    CHECK(15, pdep64(vw, 0xFFFFFFFFFFFFFFFFull) == vw);

    // int8 and int16 operands.
    uniform int8 u8 = -1;
    int8 v8 = (int8)(-128 + programIndex);
    uniform int16 u16 = 0x00F0;
    CHECK(21, pdep32(u8, u8) == 0xFFFFFFFFu);
    CHECK(22, pdep64(u8, u8) == 0xFFFFFFFFFFFFFFFFull);
    CHECK(23, pext32(v8, 0x80000000u) == 1u);
    CHECK(24, pext32(v8, 0xFFu) == (uint32)(uint8)v8);
    CHECK(25, pext64(u16, u16) == 0xFull);
    CHECK(26, pdep32(u16, (uint8)0xF0) == 0u);

    // int64 operands, signed and unsigned literals.
    uniform int64 u64n = -1;
    int64 v64 = (int64)programIndex - 1;
    CHECK(31, pdep64(u64n, 0x5555555555555555ull) == 0x5555555555555555ull);
    CHECK(32, pext64(v64, 0xFFFFFFFFFFFFFFFFull) == (uint64)v64);
    CHECK(33, pdep32(5, 0x52) == 0x42u);
    CHECK(34, pext32(0x42, 0x52) == 5u);
    CHECK(35, pdep64(1, 0x8000000000000000) == 0x8000000000000000ull);
    CHECK(36, pext64(0x8000000000000000, 0x8000000000000000) == 1ull);
    CHECK(37, pdep32(programIndex, 0xFFFFFFFF) == (uint32)programIndex);

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
