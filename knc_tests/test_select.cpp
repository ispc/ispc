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

#define SELECT_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                              \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input1;                                                                        \
    VEC_TYPE input2;                                                                        \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&input1, i, (TYPE) copy_data[i]);                                  \
        __insert_element(&input2, i, (TYPE) copy_data[i] / 2);                              \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __select(mask, input1, input2);                                                \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 && check_and_print(__extract_element(output, i), (TYPE) (data[i]),    \
                                         err_counter))                                      \
            err_counter++;                                                                  \
        if (m[i] == 0 && check_and_print(__extract_element(output, i), (TYPE)(data[i] / 2), \
                                        err_counter))                                       \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}                                                                                           \
                                                                                            \
void FUNC_NAME##_cond(TYPE *data, int *m) {                                                 \
    printf ("%-40s", #FUNC_NAME "_cond" ":");                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input1;                                                                        \
    VEC_TYPE input2;                                                                        \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&input1, i, (TYPE) copy_data[i]);                                  \
        __insert_element(&input2, i, (TYPE) copy_data[i] * -1);                             \
    }                                                                                       \
                                                                                            \
    VEC_TYPE output;                                                                        \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __select(copy_m[i], input1, input2);                                       \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            if (m[i] != 0 && check_and_print(__extract_element(output, j),                  \
                                             (TYPE)(data[j]), err_counter))                 \
                err_counter++;                                                              \
            if (m[i] == 0 && check_and_print(__extract_element(output, j),                  \
                                            (TYPE)(data[j] * -1), err_counter))             \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SELECT_TEST(double , __vec16_d  , select_double)
SELECT_TEST(float  , __vec16_f  , select_float )
SELECT_TEST(int8_t , __vec16_i8 , select_i8    )
SELECT_TEST(int16_t, __vec16_i16, select_i16   )
SELECT_TEST(int32_t, __vec16_i32, select_i32   )
SELECT_TEST(int64_t, __vec16_i64, select_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

void test_select() {
    InputData inpData;

    select_double(inpData.d_32, inpData.mask);
    select_float(inpData.f_32, inpData.mask);
    select_i8(inpData.i8, inpData.mask);
    select_i16(inpData.i16, inpData.mask);
    select_i32(inpData.i32, inpData.mask);
    select_i64(inpData.i64, inpData.mask);

    select_double_cond(inpData.d_32, inpData.mask);
    select_float_cond(inpData.f_32, inpData.mask);
    select_i8_cond(inpData.i8, inpData.mask);
    select_i16_cond(inpData.i16, inpData.mask);
    select_i32_cond(inpData.i32, inpData.mask);
    select_i64_cond(inpData.i64, inpData.mask);

}


