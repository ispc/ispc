#include "test_static.isph"

// Edge cases for the packed dot products, including the native forms and the
// emulated fallbacks. Expected values follow the VNNI pseudocode: exact
// products, wrapping accumulation for the plain forms and one saturation of
// the exact sum for the _sat forms. Each lane reports a bitmask of the
// functions that returned a wrong value for one of its cases (bit n for the
// n-th function checked in f_v).

#define NCASES 10

// Non-static globals so the calls are not constant-folded.
uniform uint32 A[NCASES] = {0xFFFFFFFF, 0x80808080, 0x7F7F7F7F, 0x80008000, 0x80008000,
                            0xFFFFFFFF, 0xFFFF0001, 0x0000007F, 0x12345678, 0xFF00FF00};
uniform uint32 B[NCASES] = {0xFFFFFFFF, 0x80808080, 0x80808080, 0x80008000, 0x80008000,
                            0x80000010, 0x7FFF8000, 0x0000FFFF, 0x9ABCDEF0, 0x00FF00FF};
uniform uint32 ACC[NCASES] = {0x00000000, 0x00000000, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF,
                              0x80008000, 0x7FFFFFF0, 0xF362035C, 0x7FFFFFFF, 0xFFFFFFFF};

uniform uint32 EXP_U8I8[NCASES] = {0xFFFFFC04, 0xFFFF0000, 0x7FFF0200, 0xFFFF7FFF, 0x7FFF7FFF,
                                   0x80001070, 0x80007D72, 0xF36202DD, 0x7FFFD817, 0xFFFFFFFF};
uniform uint32 EXP_U8I8_SAT[NCASES] = {0xFFFFFC04, 0xFFFF0000, 0x80000000, 0xFFFF7FFF, 0x7FFF7FFF,
                                       0x80001070, 0x7FFFFFFF, 0xF36202DD, 0x7FFFD817, 0xFFFFFFFF};
uniform uint32 EXP_U8U8[NCASES] = {0x0003F804, 0x00010000, 0x8000FE00, 0x00007FFF, 0x80007FFF,
                                   0x80010F70, 0x80017C72, 0xF36281DD, 0x8000EC17, 0xFFFFFFFF};
uniform uint32 EXP_U8U8_SAT[NCASES] = {0x0003F804, 0x00010000, 0x8000FE00, 0xFFFFFFFF, 0x80007FFF,
                                       0x80010F70, 0x80017C72, 0xF36281DD, 0x8000EC17, 0xFFFFFFFF};
uniform uint32 EXP_I8I8[NCASES] = {0x00000004, 0x00010000, 0x7FFF0200, 0x00007FFF, 0x80007FFF,
                                   0x80008070, 0x7FFFFF72, 0xF36202DD, 0x7FFFD817, 0xFFFFFFFF};
uniform uint32 EXP_I8I8_SAT[NCASES] = {0x00000004, 0x00010000, 0x80000000, 0x00007FFF, 0x7FFFFFFF,
                                       0x80008070, 0x7FFFFF72, 0xF36202DD, 0x7FFFD817, 0xFFFFFFFF};

task void f_v(uniform float dst[]) {
    int errors = 0;
    foreach (i = 0 ... NCASES) {
        uint32 a = A[i];
        uint32 b = B[i];
        uint32 acc = ACC[i];
        errors |= ((uint32)dot4add_u8i8packed(a, b, (int32)acc) != EXP_U8I8[i]) ? (1 << 0) : 0;
        errors |= ((uint32)dot4add_u8i8packed_sat(a, b, (int32)acc) != EXP_U8I8_SAT[i]) ? (1 << 1) : 0;
        errors |= (dot4add_u8u8packed(a, b, acc) != EXP_U8U8[i]) ? (1 << 2) : 0;
        errors |= (dot4add_u8u8packed_sat(a, b, acc) != EXP_U8U8_SAT[i]) ? (1 << 3) : 0;
        errors |= ((uint32)dot4add_i8i8packed(a, b, (int32)acc) != EXP_I8I8[i]) ? (1 << 4) : 0;
        errors |= ((uint32)dot4add_i8i8packed_sat(a, b, (int32)acc) != EXP_I8I8_SAT[i]) ? (1 << 5) : 0;
    }
    dst[programIndex] = errors;
}

task void result(uniform float dst[]) { dst[programIndex] = 0; }
