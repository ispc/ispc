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

#define CAST_TEST(TO, TO_VEC, FROM, FROM_VEC, FUNC_NAME, FUNC_CALL)                         \
void FUNC_NAME(FROM *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    FROM_VEC input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (FROM) copy_data[i]);                                   \
                                                                                            \
    TO_VEC output;                                                                          \
    output = FUNC_CALL(output, input);                                                      \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(!(TO)__extract_element(output, i), !(TO)data[i], err_counter))  \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CAST_TEST(int64_t, __vec16_i64, int32_t, __vec16_i32, cast_i64_i32, __cast_sext)
//CAST_TEST(int64_t, __vec16_i64, int16_t, __vec16_i16, cast_i64_i16, __cast_sext)
CAST_TEST(int64_t, __vec16_i64, int8_t , __vec16_i8 , cast_i64_i8 , __cast_sext)
//CAST_TEST(int64_t, __vec16_i64, bool   , __vec16_i1 , cast_i64_i1 , __cast_sext)
CAST_TEST(int32_t, __vec16_i32, int16_t, __vec16_i16, cast_i32_i16, __cast_sext)
CAST_TEST(int32_t, __vec16_i32, int8_t , __vec16_i8 , cast_i32_i8 , __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, bool   , __vec16_i1 , cast_i32_i1 , __cast_sext)
CAST_TEST(int16_t, __vec16_i16, int8_t , __vec16_i8 , cast_i16_i8 , __cast_sext)
//CAST_TEST(int16_t, __vec16_i16, bool   , __vec16_i1 , cast_i16_i1 , __cast_sext)
//CAST_TEST(int8_t , __vec16_i8 , bool   , __vec16_i1 , cast_i8_i1  , __cast_sext)

CAST_TEST(uint64_t, __vec16_i64, uint32_t, __vec16_i32, cast_ui64_ui32, __cast_zext)
CAST_TEST(uint64_t, __vec16_i64, uint16_t, __vec16_i16, cast_ui64_ui16, __cast_zext)
CAST_TEST(uint64_t, __vec16_i64, uint8_t , __vec16_i8 , cast_ui64_ui8 , __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, bool    , __vec16_i1 , cast_ui64_ui1 , __cast_zext)
CAST_TEST(uint32_t, __vec16_i32, uint16_t, __vec16_i16, cast_ui32_ui16, __cast_zext)
CAST_TEST(uint32_t, __vec16_i32, uint8_t , __vec16_i8 , cast_ui32_ui8 , __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, bool    , __vec16_i1 , cast_ui32_ui1 , __cast_zext)
CAST_TEST(uint16_t, __vec16_i16, uint8_t , __vec16_i8 , cast_ui16_ui8 , __cast_zext)
//CAST_TEST(uint16_t, __vec16_i16, bool    , __vec16_i1 , cast_ui16_ui1 , __cast_zext)
//CAST_TEST(uint8_t , __vec16_i8 , bool    , __vec16_i1 , cast_ui8_ui1  , __cast_zext)

CAST_TEST(int32_t, __vec16_i32, int64_t, __vec16_i64, trunk_i32_i64, __cast_trunc)
CAST_TEST(int16_t, __vec16_i16, int64_t, __vec16_i64, trunk_i16_i64, __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int64_t, __vec16_i64, trunk_i8_i64 , __cast_trunc)
CAST_TEST(int16_t, __vec16_i16, int32_t, __vec16_i32, trunk_i16_i32, __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int32_t, __vec16_i32, trunk_i8_i32 , __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int16_t, __vec16_i16, trunk_i8_i16 , __cast_trunc)

CAST_TEST(float , __vec16_f, int8_t , __vec16_i8,  cast_f_i8,  __cast_sitofp)
CAST_TEST(float , __vec16_f, int16_t, __vec16_i16, cast_f_i16, __cast_sitofp)
CAST_TEST(float , __vec16_f, int32_t, __vec16_i32, cast_f_i32, __cast_sitofp)
CAST_TEST(float , __vec16_f, int64_t, __vec16_i64, cast_f_i64, __cast_sitofp)
CAST_TEST(double, __vec16_d, int8_t , __vec16_i8 , cast_d_i8,  __cast_sitofp)
CAST_TEST(double, __vec16_d, int16_t, __vec16_i16, cast_d_i16, __cast_sitofp)
CAST_TEST(double, __vec16_d, int32_t, __vec16_i32, cast_d_i32, __cast_sitofp)
CAST_TEST(double, __vec16_d, int64_t, __vec16_i64, cast_d_i64, __cast_sitofp)

CAST_TEST(float , __vec16_f, uint8_t , __vec16_i8,  cast_f_ui8,  __cast_uitofp)
CAST_TEST(float , __vec16_f, uint16_t, __vec16_i16, cast_f_ui16, __cast_uitofp)
CAST_TEST(float , __vec16_f, uint32_t, __vec16_i32, cast_f_ui32, __cast_uitofp)
CAST_TEST(float , __vec16_f, uint64_t, __vec16_i64, cast_f_ui64, __cast_uitofp)
CAST_TEST(double, __vec16_d, uint8_t , __vec16_i8 , cast_d_ui8,  __cast_uitofp)
CAST_TEST(double, __vec16_d, uint16_t, __vec16_i16, cast_d_ui16, __cast_uitofp)
CAST_TEST(double, __vec16_d, uint32_t, __vec16_i32, cast_d_ui32, __cast_uitofp)
CAST_TEST(double, __vec16_d, uint64_t, __vec16_i64, cast_d_ui64, __cast_uitofp)

CAST_TEST(int8_t , __vec16_i8 , float , __vec16_f, cast_i8_f , __cast_fptosi)
CAST_TEST(int16_t, __vec16_i16, float , __vec16_f, cast_i16_f, __cast_fptosi)
CAST_TEST(int32_t, __vec16_i32, float , __vec16_f, cast_i32_f, __cast_fptosi)
CAST_TEST(int64_t, __vec16_i64, float , __vec16_f, cast_i64_f, __cast_fptosi)
CAST_TEST(int8_t , __vec16_i8 , double, __vec16_d, cast_i8_d , __cast_fptosi)
CAST_TEST(int16_t, __vec16_i16, double, __vec16_d, cast_i16_d, __cast_fptosi)
CAST_TEST(int32_t, __vec16_i32, double, __vec16_d, cast_i32_d, __cast_fptosi)
CAST_TEST(int64_t, __vec16_i64, double, __vec16_d, cast_i64_d, __cast_fptosi)

CAST_TEST(uint8_t , __vec16_i8 , float , __vec16_f, cast_ui8_f , __cast_fptoui)
CAST_TEST(uint16_t, __vec16_i16, float , __vec16_f, cast_ui16_f, __cast_fptoui)
CAST_TEST(uint32_t, __vec16_i32, float , __vec16_f, cast_ui32_f, __cast_fptoui)
CAST_TEST(uint64_t, __vec16_i64, float , __vec16_f, cast_ui64_f, __cast_fptoui)
CAST_TEST(uint8_t , __vec16_i8 , double, __vec16_d, cast_ui8_d , __cast_fptoui)
CAST_TEST(uint16_t, __vec16_i16, double, __vec16_d, cast_ui16_d, __cast_fptoui)
CAST_TEST(uint32_t, __vec16_i32, double, __vec16_d, cast_ui32_d, __cast_fptoui)
CAST_TEST(uint64_t, __vec16_i64, double, __vec16_d, cast_ui64_d, __cast_fptoui)

CAST_TEST(float , __vec16_f, double, __vec16_d, cast_f_d, __cast_fptrunc)
CAST_TEST(double, __vec16_d, float , __vec16_f, cast_d_f, __cast_fpext)

/////////////////////////////////////////////////////////////////////////////////////////////

void test_cast() {
    InputData inpData;

    cast_i64_i32(inpData.i32);
    //cast_i64_i16(inpData.i16);
    cast_i64_i8 (inpData.i8);
    //cast_i64_i1 (inpData.i1);
    cast_i32_i16(inpData.i16);
    cast_i32_i8 (inpData.i8);
    //cast_i32_i1 (inpData.i1);
    cast_i16_i8 (inpData.i8);
    //cast_i16_i1 (inpData.i1);
    //cast_i8_i1  (inpData.i1);

    cast_ui64_ui32(inpData.ui32);
    cast_ui64_ui16(inpData.ui16);
    cast_ui64_ui8 (inpData.ui8);
    //cast_ui64_ui1 (inpData.i1);
    cast_ui32_ui16(inpData.ui16);
    cast_ui32_ui8 (inpData.ui8);
    //cast_ui32_ui1 (inpData.i1);
    cast_ui16_ui8 (inpData.ui8);
    //cast_ui16_ui1 (inpData.i1);
    //cast_ui8_ui1  (inpData.i1);

    trunk_i32_i64(inpData.i64);
    trunk_i16_i64(inpData.i64);
    trunk_i8_i64 (inpData.i64);
    trunk_i16_i32(inpData.i32);
    trunk_i8_i32 (inpData.i32);
    trunk_i8_i16 (inpData.i16);

    cast_f_i8 (inpData.i8);
    cast_f_i16(inpData.i16);
    cast_f_i32(inpData.i32);
    cast_f_i64(inpData.i64);
    cast_d_i8 (inpData.i8);
    cast_d_i16(inpData.i16);
    cast_d_i32(inpData.i32);
    cast_d_i64(inpData.i64);

    cast_f_ui8 (inpData.ui8);
    cast_f_ui16(inpData.ui16);
    cast_f_ui32(inpData.ui32);
    cast_f_ui64(inpData.ui64);
    cast_d_ui8 (inpData.ui8);
    cast_d_ui16(inpData.ui16);
    cast_d_ui32(inpData.ui32);
    cast_d_ui64(inpData.ui64);

    cast_i8_f (inpData.f_8);
    cast_i16_f(inpData.f_16);
    cast_i32_f(inpData.f_32);
    cast_i64_f(inpData.f_64);
    cast_i8_d (inpData.d_8);
    cast_i16_d(inpData.d_16);
    cast_i32_d(inpData.d_32);
    cast_i64_d(inpData.d_64);

    cast_ui8_f (inpData.f_u8);
    cast_ui16_f(inpData.f_u16);
    cast_ui32_f(inpData.f_u32);
    cast_ui64_f(inpData.f_u64);
    cast_ui8_d (inpData.d_u8);
    cast_ui16_d(inpData.d_u16);
    cast_ui32_d(inpData.d_u32);
    cast_ui64_d(inpData.d_u64);


    cast_f_d(inpData.d_8);
    cast_f_d(inpData.d_16);
    cast_f_d(inpData.d_32);
    cast_f_d(inpData.d_64);
    cast_d_f(inpData.f_8);
    cast_d_f(inpData.f_16);
    cast_d_f(inpData.f_32);
    cast_d_f(inpData.f_64);

}


