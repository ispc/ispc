#include "test_static.isph"

// Edge cases for the packed dot products, including the native forms and the
// emulated fallbacks. Expected values follow the VNNI pseudocode: exact
// products, wrapping accumulation for the plain forms and one saturation of
// the exact sum for the _sat forms. Each lane reports a bitmask of the
// functions that returned a wrong value for one of its cases (bit n for the
// n-th function checked in f_v).

#define NCASES 13

// Non-static globals so the calls are not constant-folded. The last three cases overflow
// the u16i16 product sum in both directions.
uniform uint32 A[NCASES] = {0xFFFFFFFF, 0x80808080, 0x7F7F7F7F, 0x80008000, 0x80008000,
                            0xFFFFFFFF, 0xFFFF0001, 0x0000007F, 0x12345678, 0xFF00FF00,
                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
uniform uint32 B[NCASES] = {0xFFFFFFFF, 0x80808080, 0x80808080, 0x80008000, 0x80008000,
                            0x80000010, 0x7FFF8000, 0x0000FFFF, 0x9ABCDEF0, 0x00FF00FF,
                            0x80008000, 0x7FFF7FFF, 0x80008000};
uniform uint32 ACC[NCASES] = {0x00000000, 0x00000000, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF,
                              0x80008000, 0x7FFFFFF0, 0xF362035C, 0x7FFFFFFF, 0xFFFFFFFF,
                              0x7FFFFFFF, 0x80000000, 0x00000000};

uniform uint32 EXP_I16I16[NCASES] = {0x00000002, 0x7F008000, 0x01007F00, 0x7FFFFFFF, 0xFFFFFFFF,
                                     0x8000FFF0, 0x7FFEFFF1, 0xF36202DD, 0x6DA1C6AF, 0xFFFE01FF,
                                     0x8000FFFF, 0x7FFF0002, 0x00010000};
uniform uint32 EXP_I16I16_SAT[NCASES] = {0x00000002, 0x7F008000, 0x80000000, 0x7FFFFFFF, 0x7FFFFFFF,
                                         0x8000FFF0, 0x7FFEFFF1, 0xF36202DD, 0x6DA1C6AF, 0xFFFE01FF,
                                         0x7FFFFFFF, 0x80000000, 0x00010000};
uniform uint32 EXP_U16U16[NCASES] = {0xFFFC0002, 0x81008000, 0xFFFE7F00, 0x7FFFFFFF, 0xFFFFFFFF,
                                     0x000FFFF0, 0xFFFEFFF1, 0xF3E102DD, 0xD64DC6AF, 0x01FC01FF,
                                     0x7FFEFFFF, 0x7FFD0002, 0xFFFF0000};
uniform uint32 EXP_U16U16_SAT[NCASES] = {0xFFFFFFFF, 0x81008000, 0xFFFE7F00, 0xFFFFFFFF, 0xFFFFFFFF,
                                         0xFFFFFFFF, 0xFFFEFFF1, 0xF3E102DD, 0xD64DC6AF, 0xFFFFFFFF,
                                         0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF0000};
uniform uint32 EXP_U16I16[NCASES] = {0xFFFE0002, 0x80008000, 0x01007F00, 0x7FFFFFFF, 0xFFFFFFFF,
                                     0x0010FFF0, 0xFFFDFFF1, 0xF36202DD, 0x6DA1C6AF, 0x01FC01FF,
                                     0x8000FFFF, 0x7FFD0002, 0x00010000};
uniform uint32 EXP_U16I16_SAT[NCASES] = {0xFFFE0002, 0x80008000, 0x80000000, 0x80000000, 0xFFFFFFFF,
                                         0x80000000, 0x7FFFFFFF, 0xF36202DD, 0x6DA1C6AF, 0x01FC01FF,
                                         0x8000FFFF, 0x7FFD0002, 0x80000000};

task void f_v(uniform float dst[]) {
    int errors = 0;
    foreach (i = 0 ... NCASES) {
        uint32 a = A[i];
        uint32 b = B[i];
        uint32 acc = ACC[i];
        errors |= ((uint32)dot2add_i16i16packed(a, b, (int32)acc) != EXP_I16I16[i]) ? (1 << 0) : 0;
        errors |= ((uint32)dot2add_i16i16packed_sat(a, b, (int32)acc) != EXP_I16I16_SAT[i]) ? (1 << 1) : 0;
        errors |= (dot2add_u16u16packed(a, b, acc) != EXP_U16U16[i]) ? (1 << 2) : 0;
        errors |= (dot2add_u16u16packed_sat(a, b, acc) != EXP_U16U16_SAT[i]) ? (1 << 3) : 0;
        errors |= ((uint32)dot2add_u16i16packed(a, b, (int32)acc) != EXP_U16I16[i]) ? (1 << 4) : 0;
        errors |= ((uint32)dot2add_u16i16packed_sat(a, b, (int32)acc) != EXP_U16I16_SAT[i]) ? (1 << 5) : 0;
    }
    dst[programIndex] = errors;
}

task void result(uniform float dst[]) { dst[programIndex] = 0; }
