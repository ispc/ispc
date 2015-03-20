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

#define LOAD(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                          \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE ptrs;                                                                          \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&ptrs, i, (TYPE)copy_data[i]);                                     \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __load<ALIGN_NUM>(&ptrs);                                                      \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(__extract_element(output, i), data[i], err_counter))            \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

LOAD(double , __vec16_d  , load_double, 128)
LOAD(float  , __vec16_f  , load_float , 64)
LOAD(int8_t , __vec16_i8 , load_i8    , 16)
LOAD(int16_t, __vec16_i16, load_i16   , 32)
LOAD(int32_t, __vec16_i32, load_i32   , 64)

void load_i64(int64_t *data) {
    printf ("%-40s", "load_i64" ":");
    int64_t copy_data[16];
    for (uint32_t i = 0; i < 16; i++)
        copy_data[i] = data[i];

    __vec16_i64 ptrs;
    for (uint32_t i = 0; i < 8; i++){
        ((int64_t *)&ptrs.v_hi)[i] = (int64_t)copy_data[i];
        ((int64_t *)&ptrs.v_lo)[i] = (int64_t)copy_data[i + 8];
    }
    __vec16_i64 output;
    output = __load<128>(&ptrs);

    int err_counter = 0;
    for (uint32_t i = 0; i < 16; i++){
        if (__extract_element(output, i) != data[i])
            err_counter++;
    }
    if (err_counter != 0)
        printf(" errors %d\n", err_counter);
    else
        printf(" no fails\n");
}

#define STORE(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                         \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (TYPE)data[i]);                                         \
                                                                                            \
    VEC_TYPE output;                                                                        \
    __store<ALIGN_NUM>(&output, input);                                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(__extract_element(output, i), data[i], err_counter))            \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

STORE(double , __vec16_d  , store_double, 128)
STORE(float  , __vec16_f  , store_float , 64)
STORE(int8_t , __vec16_i8 , store_i8    , 16)
STORE(int16_t, __vec16_i16, store_i16   , 32)
STORE(int32_t, __vec16_i32, store_i32   , 64)

void store_i64(int64_t *data) {
    printf ("%-40s", "store_i64" ":");

    int64_t copy_data[16];
    for (uint32_t i = 0; i < 16; i++)
        copy_data[i] = data[i];

    __vec16_i64 input;
    for (uint32_t i = 0; i < 16; i++){
        if(i % 2 == 0) {
            ((int32_t *)&input.v_lo)[i / 2] = (int64_t)copy_data[i] >> 32;
            ((int32_t *)&input.v_lo)[i / 2 + 8] = (int64_t)copy_data[i];
        }
        else {
            ((int32_t *)&input.v_hi)[i / 2] = (int64_t)copy_data[i] >> 32;
            ((int32_t *)&input.v_hi)[i / 2 + 8] = (int64_t)copy_data[i];
        }
    }

    __vec16_i64 output;
    __store<128>(&output, input);

    int err_counter = 0;
    for (uint32_t i = 0; i < 16; i++){
        if (check_and_print(__extract_element(output, i), data[i], err_counter))
            err_counter++;
    }
    if (err_counter != 0)
        printf(" errors %d\n", err_counter);
    else
        printf(" no fails\n");
}

/////////////////////////////////////////////////////////////////////////////////////////////

void test_load_store () {
    InputData inpData;

    load_double(inpData.d_32);
    load_float(inpData.f_32);
    load_i8(inpData.i8);
    load_i16(inpData.i16);
    load_i32(inpData.i32);
    load_i64(inpData.i64);


    store_double(inpData.d_32);
    store_float(inpData.f_32);
    store_i8(inpData.i8);
    store_i16(inpData.i16);
    store_i32(inpData.i32);
    store_i64(inpData.i64);

}

