// Copyright (c) 2014-2015, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Intel Corporation nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ==========================================================
// Author: Vsevolod Livinskiy
// ==========================================================

#include "knc_test_driver_core.h"

/////////////////////////////////////////////////////////////////////////////////////////////

#define BINARY_OP_TEST(TYPE, VEC_TYPE, OP, FUNC_NAME, TYPE_MOD)                             \
void  FUNC_NAME##_##TYPE_MOD(TYPE *a, TYPE *b) {                                            \
    printf ("%-40s", #FUNC_NAME "_" #TYPE_MOD ":");                                         \
                                                                                            \
    TYPE copy_a[16];                                                                        \
    TYPE copy_b[16];                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        copy_a[i] = a[i];                                                                   \
        copy_b[i] = b[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    VEC_TYPE input_b;                                                                       \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        __insert_element(&input_a, i, (TYPE)copy_a[i]);                                     \
        __insert_element(&input_b, i, (TYPE)copy_b[i]);                                     \
    }                                                                                       \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(input_a, input_b);                                               \
                                                                                            \
    int err_counter = 0;                                                                    \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        result = (TYPE) (a[i]) OP (TYPE) (b[i]);                                            \
        if (check_and_print((TYPE)__extract_element(output, i), (TYPE)result, err_counter)) \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

BINARY_OP_TEST(double , __vec16_d  , +, add, double)
BINARY_OP_TEST(float  , __vec16_f  , +, add, float)
BINARY_OP_TEST(int8_t , __vec16_i8 , +, add, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, +, add, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, +, add, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, +, add, i64)


BINARY_OP_TEST(double , __vec16_d  , -, sub, double)
BINARY_OP_TEST(float  , __vec16_f  , -, sub, float)
BINARY_OP_TEST(int8_t , __vec16_i8 , -, sub, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, -, sub, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, -, sub, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, -, sub, i64)

BINARY_OP_TEST(double , __vec16_d  , *, mul, double)
BINARY_OP_TEST(float  , __vec16_f  , *, mul, float)
BINARY_OP_TEST(int8_t , __vec16_i8 , *, mul, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, *, mul, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, *, mul, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, *, mul, i64)

BINARY_OP_TEST(int8_t , __vec16_i8 , |, or, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, |, or, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, |, or, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, |, or, i64)

BINARY_OP_TEST(int8_t , __vec16_i8 , &, and, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, &, and, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, &, and, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, &, and, i64)

BINARY_OP_TEST(int8_t , __vec16_i8 , ^, xor, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, ^, xor, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, ^, xor, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, ^, xor, i64)

BINARY_OP_TEST(int8_t , __vec16_i8 , <<, shl, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, <<, shl, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, <<, shl, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, <<, shl, i64)

BINARY_OP_TEST(uint8_t , __vec16_i8 , /, udiv, ui8)
BINARY_OP_TEST(uint16_t, __vec16_i16, /, udiv, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, /, udiv, ui32)
BINARY_OP_TEST(uint64_t, __vec16_i64, /, udiv, ui64)

BINARY_OP_TEST(int8_t , __vec16_i8 , /, sdiv, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, /, sdiv, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, /, sdiv, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, /, sdiv, i64)

BINARY_OP_TEST(double, __vec16_d, /, div, double)
BINARY_OP_TEST(float , __vec16_f, /, div, float)

BINARY_OP_TEST(uint8_t , __vec16_i8 , %, urem, ui8)
BINARY_OP_TEST(uint16_t, __vec16_i16, %, urem, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, %, urem, ui32)
//BINARY_OP_TEST(uint64_t, __vec16_i64, %, urem, ui64)

BINARY_OP_TEST(int8_t , __vec16_i8 , %, srem, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, %, srem, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, %, srem, i32)
//BINARY_OP_TEST(int64_t, __vec16_i64, %, srem, i64)

BINARY_OP_TEST(uint8_t , __vec16_i8 , >>, lshr, ui8)
BINARY_OP_TEST(uint16_t, __vec16_i16, >>, lshr, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, >>, lshr, ui32)
BINARY_OP_TEST(uint64_t, __vec16_i64, >>, lshr, ui64)

BINARY_OP_TEST(int8_t , __vec16_i8 , >>, ashr, i8)
BINARY_OP_TEST(int16_t, __vec16_i16, >>, ashr, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, >>, ashr, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, >>, ashr, i64)

#define SHIFT_UNIFORM_TEST(TYPE, VEC_TYPE, OP, FUNC_NAME, TYPE_MOD)                         \
void  FUNC_NAME##_##TYPE_MOD##_uniform(TYPE *a, int32_t *b) {                               \
    printf ("%-40s", #FUNC_NAME "_" #TYPE_MOD ":");                                         \
                                                                                            \
    TYPE copy_a[16];                                                                        \
    int32_t copy_b[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        copy_a[i] = a[i];                                                                   \
        copy_b[i] = b[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input_a, i, (TYPE)copy_a[i]);                                     \
                                                                                            \
    VEC_TYPE output;                                                                        \
    int err_counter = 0;                                                                    \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __##FUNC_NAME(input_a, copy_b[i]);                                         \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            result = (TYPE) (a[j]) OP (b[i]);                                               \
            if (check_and_print((TYPE)__extract_element(output, j), (TYPE)result,           \
                                                                     err_counter))          \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

SHIFT_UNIFORM_TEST(uint8_t , __vec16_i8 , >>, lshr, ui8)
SHIFT_UNIFORM_TEST(uint16_t, __vec16_i16, >>, lshr, ui16)
SHIFT_UNIFORM_TEST(uint32_t, __vec16_i32, >>, lshr, ui32)
SHIFT_UNIFORM_TEST(uint64_t, __vec16_i64, >>, lshr, ui64)

SHIFT_UNIFORM_TEST(int8_t , __vec16_i8 , >>, ashr, i8)
SHIFT_UNIFORM_TEST(int16_t, __vec16_i16, >>, ashr, i16)
SHIFT_UNIFORM_TEST(int32_t, __vec16_i32, >>, ashr, i32)
SHIFT_UNIFORM_TEST(int64_t, __vec16_i64, >>, ashr, i64)

/////////////////////////////////////////////////////////////////////////////////////////////

void test_binary_op() {
    InputData inpData;

    add_double(inpData.no_of_d_32, inpData.no_of_d_32);
    add_float (inpData.no_of_f_32, inpData.no_of_f_32);
    add_i8    (inpData.no_of_i8  , inpData.no_of_i8);
    add_i16   (inpData.no_of_i16 , inpData.no_of_i16);
    add_i32   (inpData.no_of_i32 , inpData.no_of_i32);
    add_i64   (inpData.no_of_i64 , inpData.no_of_i64);

    sub_double(inpData.no_of_d_32, inpData.no_of_d_32);
    sub_float (inpData.no_of_f_32, inpData.no_of_f_32);
    sub_i8    (inpData.no_of_i8  , inpData.no_of_i8);
    sub_i16   (inpData.no_of_i16 , inpData.no_of_i16);
    sub_i32   (inpData.no_of_i32 , inpData.no_of_i32);
    sub_i64   (inpData.no_of_i64 , inpData.no_of_i64);

    mul_double(inpData.no_of_d_32, inpData.no_of_d_32);
    mul_float (inpData.no_of_f_32, inpData.no_of_f_32);
    mul_i8    (inpData.no_of_i8  , inpData.no_of_i8);
    mul_i16   (inpData.no_of_i16 , inpData.no_of_i16);
    mul_i32   (inpData.no_of_i32 , inpData.no_of_i32);
    mul_i64   (inpData.no_of_i64 , inpData.no_of_i64);

    or_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    or_i16(inpData.no_of_i16 , inpData.no_of_i16);
    or_i32(inpData.no_of_i32 , inpData.no_of_i32);
    or_i64(inpData.no_of_i64 , inpData.no_of_i64);

    and_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    and_i16(inpData.no_of_i16 , inpData.no_of_i16);
    and_i32(inpData.no_of_i32 , inpData.no_of_i32);
    and_i64(inpData.no_of_i64 , inpData.no_of_i64);

    xor_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    xor_i16(inpData.no_of_i16 , inpData.no_of_i16);
    xor_i32(inpData.no_of_i32 , inpData.no_of_i32);
    xor_i64(inpData.no_of_i64 , inpData.no_of_i64);

    shl_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    shl_i16(inpData.no_of_i16 , inpData.no_of_i16);
    shl_i32(inpData.no_of_i32 , inpData.no_of_i32);
    shl_i64(inpData.no_of_i64 , inpData.no_of_i64);

    udiv_ui8 (inpData.no_of_ui8  , inpData.no_of_ui8);
    udiv_ui16(inpData.no_of_ui16 , inpData.no_of_ui16);
    udiv_ui32(inpData.no_of_ui32 , inpData.no_of_ui32);
    udiv_ui64(inpData.no_of_ui64 , inpData.no_of_ui64);

    sdiv_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    sdiv_i16(inpData.no_of_i16 , inpData.no_of_i16);
    sdiv_i32(inpData.no_of_i32 , inpData.no_of_i32);
    sdiv_i64(inpData.no_of_i64 , inpData.no_of_i64);

    div_double(inpData.no_of_d_32 , inpData.no_of_d_32);
    div_float (inpData.no_of_f_32 , inpData.no_of_f_32);

    urem_ui8 (inpData.no_of_ui8  , inpData.no_of_ui8);
    urem_ui16(inpData.no_of_ui16 , inpData.no_of_ui16);
    urem_ui32(inpData.no_of_ui32 , inpData.no_of_ui32);
    //urem_ui64(inpData.no_of_ui64 , inpData.no_of_ui64);

    srem_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    srem_i16(inpData.no_of_i16 , inpData.no_of_i16);
    srem_i32(inpData.no_of_i32 , inpData.no_of_i32);
    //srem_i64(inpData.no_of_i64 , inpData.no_of_i64);

    lshr_ui8 (inpData.no_of_ui8  , inpData.no_of_ui8);
    lshr_ui16(inpData.no_of_ui16 , inpData.no_of_ui16);
    lshr_ui32(inpData.no_of_ui32 , inpData.no_of_ui32);
    lshr_ui64(inpData.no_of_ui64 , inpData.no_of_ui64);

    ashr_i8 (inpData.no_of_i8  , inpData.no_of_i8);
    ashr_i16(inpData.no_of_i16 , inpData.no_of_i16);
    ashr_i32(inpData.no_of_i32 , inpData.no_of_i32);
    ashr_i64(inpData.no_of_i64 , inpData.no_of_i64);

    lshr_ui8_uniform (inpData.no_of_ui8 , inpData.no_of_i32);
    lshr_ui16_uniform(inpData.no_of_ui16, inpData.no_of_i32);
    lshr_ui32_uniform(inpData.no_of_ui32, inpData.no_of_i32);
    lshr_ui64_uniform(inpData.no_of_ui64, inpData.no_of_i32);

    ashr_i8_uniform (inpData.no_of_i8 , inpData.no_of_i32);
    ashr_i16_uniform(inpData.no_of_i16, inpData.no_of_i32);
    ashr_i32_uniform(inpData.no_of_i32, inpData.no_of_i32);
    ashr_i64_uniform(inpData.no_of_i64, inpData.no_of_i32);
}
