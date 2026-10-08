#include "test_static.isph"
// Fixed-value checks for pdep32/pext32/pdep64/pext64 with uniform, varying,
// and mixed operands.

#define CHECK(n, cond)                                                                                                 \
    if (err == 0 && !(cond))                                                                                           \
        err = n;

task void f_v(uniform float RET[]) {
    int err = 0;

    // Uniform operands.
    uniform uint32 ux = 0x12345678u;
    uniform uint64 ux64 = 0x123456789ABCDEF0ull;
    CHECK(1, pdep32(5u, 0x52u) == 0x42u);
    CHECK(2, pext32(0x42u, 0x52u) == 5u);
    CHECK(3, pdep32(ux, 0u) == 0u);
    CHECK(4, pext32(ux, 0u) == 0u);
    CHECK(5, pdep32(ux, 0xFFFFFFFFu) == ux);
    CHECK(6, pext32(ux, 0xFFFFFFFFu) == ux);
    CHECK(7, pdep32(1u, 0x80000000u) == 0x80000000u);
    CHECK(8, pext32(0x80000000u, 0x80000000u) == 1u);
    CHECK(9, pdep32(ux, 0xFF00FF00u) == 0x56007800u);
    CHECK(10, pext32(ux, 0xFF00FF00u) == 0x1256u);
    CHECK(11, pdep32(0xFFFFu, 0x55555555u) == 0x55555555u);
    CHECK(12, pdep32(0xFFFFu, 0xAAAAAAAAu) == 0xAAAAAAAAu);
    CHECK(13, pext32(0xFFFFFFFFu, 0xAAAAAAAAu) == 0xFFFFu);
    CHECK(14, pext32(0x55555555u, 0xAAAAAAAAu) == 0u);
    CHECK(15, pdep32(0u, 0xFFFFFFFFu) == 0u);
    CHECK(16, pdep32(0xFFFFFFFFu, 0x00F00001u) == 0x00F00001u);
    CHECK(17, pext32(0x00A00001u, 0x00F00001u) == 0x15u);

    CHECK(21, pdep64(ux64, 0ull) == 0ull);
    CHECK(22, pext64(ux64, 0ull) == 0ull);
    CHECK(23, pdep64(ux64, 0xFFFFFFFFFFFFFFFFull) == ux64);
    CHECK(24, pext64(ux64, 0xFFFFFFFFFFFFFFFFull) == ux64);
    CHECK(25, pdep64(1ull, 0x8000000000000000ull) == 0x8000000000000000ull);
    CHECK(26, pext64(0x8000000000000000ull, 0x8000000000000000ull) == 1ull);
    CHECK(27, pdep64(0xFFFFFFFFull, 0x5555555555555555ull) == 0x5555555555555555ull);
    CHECK(28, pext64(0xFFFFFFFFFFFFFFFFull, 0xAAAAAAAAAAAAAAAAull) == 0xFFFFFFFFull);
    // Masks crossing bit 31/32.
    CHECK(29, pdep64(3ull, 0x0000000180000000ull) == 0x0000000180000000ull);
    CHECK(30, pext64(0x0000000180000000ull, 0x0000000180000000ull) == 3ull);
    CHECK(31, pdep64(0x5ull, 0x00000003C0000000ull) == 0x0000000140000000ull);
    CHECK(32, pext64(0x0000000140000000ull, 0x00000003C0000000ull) == 0x5ull);
    CHECK(33, pdep64(ux64, 0xFF000000000000FFull) == 0xDE000000000000F0ull);
    CHECK(34, pext64(ux64, 0xFF000000000000FFull) == 0x12F0ull);

    // Varying operands with lane-dependent masks: deposit or extract the
    // single bit at position pos (and pos + 32).
    int pos = programIndex % 32;
    uint32 bit32 = 1u << (uint32)pos;
    uint64 bit64 = (1ull << (uint64)pos) | (1ull << (uint64)(pos + 32));
    CHECK(41, pdep32((uint32)1, bit32) == bit32);
    CHECK(42, pext32(bit32, bit32) == 1u);
    CHECK(43, pext32(~bit32, bit32) == 0u);
    CHECK(44, pdep64((uint64)3, bit64) == bit64);
    CHECK(45, pdep64((uint64)2, bit64) == (1ull << (uint64)(pos + 32)));
    CHECK(46, pext64(bit64, bit64) == 3ull);
    CHECK(47, pext64(1ull << (uint64)(pos + 32), bit64) == 2ull);

    // Mixed uniform/varying operands.
    uint32 v32 = (uint32)programIndex;
    uint64 v64 = (uint64)programIndex << 30;
    CHECK(51, pdep32(v32, 0xFFFFFFFFu) == v32);
    CHECK(52, pext32(v32, 0xFFFFFFFFu) == v32);
    CHECK(53, pdep32(0xFFFFFFFFu, bit32) == bit32);
    CHECK(54, pext32(0xFFFFFFFFu, bit32) == 1u);
    CHECK(55, pdep64(v64, 0xFFFFFFFFFFFFFFFFull) == v64);
    CHECK(56, pext64(v64, 0xFFFFFFFFFFFFFFFFull) == v64);
    CHECK(57, pdep64(0xFFFFFFFFFFFFFFFFull, bit64) == bit64);
    CHECK(58, pext64(0xFFFFFFFFFFFFFFFFull, bit64) == 3ull);
    CHECK(59, pdep32(v32, 0x0000FF00u) == (v32 << 8));
    CHECK(60, pext64(v64, 0xFFFFFFFFC0000000ull) == (uint64)programIndex);

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
