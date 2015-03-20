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

#include "knc_test_driver_core.h"

/////////////////////////////////////////////////////////////////////////////////////////////

#define SETZERO_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                             \
void FUNC_NAME() {                                                                          \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME<VEC_TYPE>();                                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        if (check_and_print(__extract_element(output, i), (TYPE) 0, err_counter))           \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SETZERO_TEST(double , __vec16_d  , setzero_double)
SETZERO_TEST(float  , __vec16_f  , setzero_float )
SETZERO_TEST(int8_t , __vec16_i8 , setzero_i8    )
SETZERO_TEST(int16_t, __vec16_i16, setzero_i16   )
SETZERO_TEST(int32_t, __vec16_i32, setzero_i32   )
SETZERO_TEST(int64_t, __vec16_i64, setzero_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

void test_setzero() {
    InputData inpData;

    setzero_double();
    setzero_float();
    setzero_i8();
    setzero_i16();
    setzero_i32();
    setzero_i64();

}

