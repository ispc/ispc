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

#define CAST_BITS_SCALAR_TEST(TO, FROM, FUNC_NAME)                                          \
void FUNC_NAME(FROM *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    union {                                                                                 \
        TO to;                                                                              \
        FROM from;                                                                          \
    } u;                                                                                    \
                                                                                            \
    TO output;                                                                              \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        output = __cast_bits(output, copy_data[i]);                                         \
        u.from = data[i];                                                                   \
        if ((isnan(output) || isnan(u.to)) && (!isnan(output) != !isnan(u.to)))             \
            err_counter++;                                                                  \
        if (!isnan(output) && !isnan(u.to) &&                                               \
            check_and_print(output, (TO) u.to, err_counter))                                \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//CAST_BITS_SCALAR_TEST(uint32_t, float   , cast_bits_scalar_ui32_f)
//CAST_BITS_SCALAR_TEST(int32_t , float   , cast_bits_scalar_i32_f)
//CAST_BITS_SCALAR_TEST(float   , uint32_t, cast_bits_scalar_f_ui32)
//CAST_BITS_SCALAR_TEST(float   , int32_t , cast_bits_scalar_f_i32)
//CAST_BITS_SCALAR_TEST(uint64_t, double  , cast_bits_scalar_ui64_d)
//CAST_BITS_SCALAR_TEST(int64_t , double  , cast_bits_scalar_i64_d)
//CAST_BITS_SCALAR_TEST(double  , uint64_t, cast_bits_scalar_d_ui64)
//CAST_BITS_SCALAR_TEST(double  , int64_t , cast_bits_scalar_d_i64)

#define CAST_BITS_TEST(TO, TO_VEC, FROM, FROM_VEC, FUNC_NAME)                               \
void FUNC_NAME(FROM *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    union {                                                                                 \
        TO to;                                                                              \
        FROM from;                                                                          \
    }u;                                                                                     \
                                                                                            \
    FROM_VEC input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (FROM) copy_data[i]);                                   \
                                                                                            \
    TO_VEC result;                                                                          \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        u.from = data[i];                                                                   \
        __insert_element(&result, i, (TO) u.to);                                            \
    }                                                                                       \
                                                                                            \
    TO_VEC output;                                                                          \
    output = __cast_bits(output, input);                                                    \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if((isnan(__extract_element(output, i)) || isnan(__extract_element(result, i))) &&  \
          (!isnan(__extract_element(output, i)) != !isnan(__extract_element(result, i))))   \
             err_counter++;                                                                 \
        if(!isnan(__extract_element(output, i)) && !isnan(__extract_element(result, i)) &&  \
          (check_and_print(__extract_element(output, i), __extract_element(result, i),      \
                                                         err_counter)))                     \
             err_counter++;                                                                 \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CAST_BITS_TEST(float  , __vec16_f  , int32_t, __vec16_i32, cast_bits_f_i32)
CAST_BITS_TEST(int32_t, __vec16_i32, float  , __vec16_f  , cast_bits_i32_f)
CAST_BITS_TEST(double , __vec16_d  , int64_t, __vec16_i64, cast_bits_d_i64)
CAST_BITS_TEST(int64_t, __vec16_i64, double , __vec16_d  , cast_bits_i64_d)

/////////////////////////////////////////////////////////////////////////////////////////////

void test_cast_bits() {
    InputData inpData;

    //cast_bits_scalar_ui32_f(inpData.f_u32);
    //cast_bits_scalar_i32_f (inpData.f_32);
    //cast_bits_scalar_f_ui32(inpData.ui32);
    //cast_bits_scalar_f_i32 (inpData.i32);
    //cast_bits_scalar_ui64_d(inpData.d_u64);
    //cast_bits_scalar_i64_d (inpData.d_64);
    //cast_bits_scalar_d_ui64(inpData.ui64);
    //cast_bits_scalar_d_i64 (inpData.i64);


    cast_bits_f_i32(inpData.i32);
    cast_bits_i32_f(inpData.f_32);
    cast_bits_d_i64(inpData.i64);
    cast_bits_i64_d(inpData.d_64);

}


