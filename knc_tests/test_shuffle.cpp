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

#define SHUFFLE_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                             \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i ,(int32_t) copy_data[i]);                                \
                                                                                            \
    __vec16_i32 index;                                                                      \
    /* 8|12|9|13|10|14|11|15|0|1|2|3|7|6|5|4 */                                             \
    for (uint32_t i = 0; i < 8; i++)                                                        \
        if (i % 2 == 0)                                                                     \
            __insert_element(&index, i , 8 + i / 2);                                        \
        else                                                                                \
            __insert_element(&index, i , 12 + (i - 1) / 2);                                 \
                                                                                            \
    for (uint32_t i = 8; i < 12; i++)                                                       \
        __insert_element(&index, i , i - 8);                                                \
                                                                                            \
    for (uint32_t i = 12; i < 16; i++)                                                      \
        __insert_element(&index, i , -i + 3);                                               \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(input, index);                                                   \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 8; i++){                                                       \
        if (i % 2 == 0 && check_and_print(__extract_element(output, i),                     \
                                          (TYPE)data[8 + i / 2], err_counter))              \
            err_counter++;                                                                  \
        if(i % 2 != 0 && check_and_print(__extract_element(output, i),                      \
                                        (TYPE)data[12 + (i - 1) / 2], err_counter))         \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    for (uint32_t i = 8; i < 12; i++)                                                       \
        if (check_and_print(__extract_element(output, i), (TYPE) data[i - 8],               \
                                                           err_counter))                    \
            err_counter++;                                                                  \
                                                                                            \
    for (uint32_t i = 12; i < 16; i++)                                                      \
        if (check_and_print(__extract_element(output, i), (TYPE) data[19 - i],              \
                                                          err_counter))                     \
            err_counter++;                                                                  \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//SHUFFLE_TEST(double , __vec16_d  , shuffle_double)
SHUFFLE_TEST(float  , __vec16_f  , shuffle_float )
SHUFFLE_TEST(int8_t , __vec16_i8 , shuffle_i8    )
SHUFFLE_TEST(int16_t, __vec16_i16, shuffle_i16   )
SHUFFLE_TEST(int32_t, __vec16_i32, shuffle_i32   )
//SHUFFLE_TEST(int64_t, __vec16_i64, shuffle_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

void test_shuffle() {
    InputData inpData;

    //shuffle_double(inpData.d_32);
    shuffle_float(inpData.f_32);
    shuffle_i8(inpData.i8);
    shuffle_i16(inpData.i16);
    shuffle_i32(inpData.i32);
    //shuffle_i64(inpData.i64);    

}

