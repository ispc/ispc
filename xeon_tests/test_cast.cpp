#include "knc_test_driver_core.h"

void cast_i64_i32(int32_t *i32);
void cast_i64_i16(int16_t *i16);
void cast_i64_i8 (int8_t  *i8);
void cast_i64_i1 (bool    *i1);
void cast_i32_i16(int16_t *i16);
void cast_i32_i8 (int8_t  *i8);
void cast_i32_i1 (bool    *i1);
void cast_i16_i8 (int8_t  *i8);
void cast_i16_i1 (bool    *i1);
void cast_i8_i1 (bool    *i1);

void cast_ui64_ui32(uint32_t *ui32);
void cast_ui64_ui16(uint16_t *ui16);
void cast_ui64_ui8 (uint8_t  *ui8);
void cast_ui64_ui1 (bool      *ui1);
void cast_ui32_ui16(uint16_t *ui16);
void cast_ui32_ui8 (uint8_t  *ui8);
void cast_ui32_ui1 (bool      *ui1);
void cast_ui16_ui8 (uint8_t  *ui8);
void cast_ui16_ui1 (bool      *ui1);
void cast_ui8_ui1 (bool      *ui1);

void trunk_i32_i64(int64_t *i64);
void trunk_i16_i64(int64_t *i64);
void trunk_i8_i64 (int64_t *i64);
void trunk_i16_i32(int32_t *i32);
void trunk_i8_i32 (int32_t *i32);
void trunk_i8_i16 (int16_t *i16);

void cast_f_i8 (int8_t *i8);
void cast_f_i16(int16_t *i16);
void cast_f_i32(int32_t *i32);
void cast_f_i64(int64_t *i64);
void cast_d_i8 (int8_t *i8);
void cast_d_i16(int16_t *i16);
void cast_d_i32(int32_t *i32);
void cast_d_i64(int64_t *i64);

void cast_f_ui8 (uint8_t *ui8);
void cast_f_ui16(uint16_t *ui16);
void cast_f_ui32(uint32_t *ui32);
void cast_f_ui64(uint64_t *ui64);
void cast_d_ui8 (uint8_t *ui8);
void cast_d_ui16(uint16_t *ui16);
void cast_d_ui32(uint32_t *ui32);
void cast_d_ui64(uint64_t *ui64);

void cast_i8_f (float  *f_8);
void cast_i16_f(float  *f_16);
void cast_i32_f(float  *f_32);
void cast_i64_f(float  *f_64);
void cast_i8_d (double *d_8);
void cast_i16_d(double *d_16);
void cast_i32_d(double *d_32);
void cast_i64_d(double *d_64);

void cast_ui8_f (float  *f_u8);
void cast_ui16_f(float  *f_u16);
void cast_ui32_f(float  *f_u32);
void cast_ui64_f(float  *f_u64);
void cast_ui8_d (double *d_u8);
void cast_ui16_d(double *d_u16);
void cast_ui32_d(double *d_u32);
void cast_ui64_d(double *d_u64);

void cast_f_d(double *d);
void cast_d_f(float  *f);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_cast() {
    InputData inpData;

    cast_i64_i32(inpData.i32);
    //cast_i64_i16(inpData.i16);
    //cast_i64_i8 (inpData.i8);
    //cast_i64_i1 (inpData.i1);
    //cast_i32_i16(inpData.i16);
    //cast_i32_i8 (inpData.i8);
    //cast_i32_i1 (inpData.i1);
    //cast_i16_i8 (inpData.i8);
    //cast_i16_i1 (inpData.i1);
    //cast_i8_i1  (inpData.i1);

    cast_ui64_ui32(inpData.ui32);
    //cast_ui64_ui16(inpData.ui16);
    //cast_ui64_ui8 (inpData.ui8);
    //cast_ui64_ui1 (inpData.i1);
    //cast_ui32_ui16(inpData.ui16);
    //cast_ui32_ui8 (inpData.ui8);
    //cast_ui32_ui1 (inpData.i1);
    //cast_ui16_ui8 (inpData.ui8);
    //cast_ui16_ui1 (inpData.i1);
    //cast_ui8_ui1  (inpData.i1);

    trunk_i32_i64(inpData.i64);
    //trunk_i16_i64(inpData.i64);
    //trunk_i8_i64 (inpData.i64);
    //trunk_i16_i32(inpData.i32);
    //trunk_i8_i32 (inpData.i32);
    //trunk_i8_i16 (inpData.i16);

    //cast_f_i8 (inpData.i8);
    //cast_f_i16(inpData.i16);
    cast_f_i32(inpData.i32);
    //cast_f_i64(inpData.i64);
    //cast_d_i8 (inpData.i8);
    //cast_d_i16(inpData.i16);
    cast_d_i32(inpData.i32);
    //cast_d_i64(inpData.i64);

    //cast_f_ui8 (inpData.ui8);
    //cast_f_ui16(inpData.ui16);
    cast_f_ui32(inpData.ui32);
    //cast_f_ui64(inpData.ui64);
    //cast_d_ui8 (inpData.ui8);
    //cast_d_ui16(inpData.ui16);
    //cast_d_ui32(inpData.ui32);
    //cast_d_ui64(inpData.ui64);

    //cast_i8_f (inpData.f_8);
    //cast_i16_f(inpData.f_16);
    cast_i32_f(inpData.f_32);
    //cast_i64_f(inpData.f_64);
    //cast_i8_d (inpData.d_8);
    //cast_i16_d(inpData.d_16);
    //cast_i32_d(inpData.d_32);
    //cast_i64_d(inpData.d_64);

    //cast_ui8_f (inpData.f_u8);
    //cast_ui16_f(inpData.f_u16);
    cast_ui32_f(inpData.f_u32);
    //cast_ui64_f(inpData.f_u64);
    //cast_ui8_d (inpData.d_u8);
    //cast_ui16_d(inpData.d_u16);
    //cast_ui32_d(inpData.d_u32);
    //cast_ui64_d(inpData.d_u64);


    cast_f_d(inpData.d_8);
    cast_f_d(inpData.d_16);
    cast_f_d(inpData.d_32);
    cast_f_d(inpData.d_64);
    cast_d_f(inpData.f_8);
    cast_d_f(inpData.f_16);
    cast_d_f(inpData.f_32);
    cast_d_f(inpData.f_64);

}

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
//CAST_TEST(int64_t, __vec16_i64, int8_t , __vec16_i8 , cast_i64_i8 , __cast_sext)
//CAST_TEST(int64_t, __vec16_i64, bool   , __vec16_i1 , cast_i64_i1 , __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, int16_t, __vec16_i16, cast_i32_i16, __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, int8_t , __vec16_i8 , cast_i32_i8 , __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, bool   , __vec16_i1 , cast_i32_i1 , __cast_sext)
//CAST_TEST(int16_t, __vec16_i16, int8_t , __vec16_i8 , cast_i16_i8 , __cast_sext)
//CAST_TEST(int16_t, __vec16_i16, bool   , __vec16_i1 , cast_i16_i1 , __cast_sext)
//CAST_TEST(int8_t , __vec16_i8 , bool   , __vec16_i1 , cast_i8_i1  , __cast_sext)

CAST_TEST(uint64_t, __vec16_i64, uint32_t, __vec16_i32, cast_ui64_ui32, __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, uint16_t, __vec16_i16, cast_ui64_ui16, __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, uint8_t , __vec16_i8 , cast_ui64_ui8 , __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, bool    , __vec16_i1 , cast_ui64_ui1 , __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, uint16_t, __vec16_i16, cast_ui32_ui16, __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, uint8_t , __vec16_i8 , cast_ui32_ui8 , __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, bool    , __vec16_i1 , cast_ui32_ui1 , __cast_zext)
//CAST_TEST(uint16_t, __vec16_i16, uint8_t , __vec16_i8 , cast_ui16_ui8 , __cast_zext)
//CAST_TEST(uint16_t, __vec16_i16, bool    , __vec16_i1 , cast_ui16_ui1 , __cast_zext)
//CAST_TEST(uint8_t , __vec16_i8 , bool    , __vec16_i1 , cast_ui8_ui1  , __cast_zext)

CAST_TEST(int32_t, __vec16_i32, int64_t, __vec16_i64, trunk_i32_i64, __cast_trunc)
//CAST_TEST(int16_t, __vec16_i16, int64_t, __vec16_i64, trunk_i16_i64, __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int64_t, __vec16_i64, trunk_i8_i64 , __cast_trunc)
//CAST_TEST(int16_t, __vec16_i16, int32_t, __vec16_i32, trunk_i16_i32, __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int32_t, __vec16_i32, trunk_i8_i32 , __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int16_t, __vec16_i16, trunk_i8_i16 , __cast_trunc)

//CAST_TEST(float , __vec16_f, int8_t , __vec16_i8,  cast_f_i8,  __cast_sitofp)
//CAST_TEST(float , __vec16_f, int16_t, __vec16_i16, cast_f_i16, __cast_sitofp)
CAST_TEST(float , __vec16_f, int32_t, __vec16_i32, cast_f_i32, __cast_sitofp)
//CAST_TEST(float , __vec16_f, int64_t, __vec16_i64, cast_f_i64, __cast_sitofp)
//CAST_TEST(double, __vec16_d, int8_t , __vec16_i8 , cast_d_i8,  __cast_sitofp)
//CAST_TEST(double, __vec16_d, int16_t, __vec16_i16, cast_d_i16, __cast_sitofp)
CAST_TEST(double, __vec16_d, int32_t, __vec16_i32, cast_d_i32, __cast_sitofp)
//CAST_TEST(double, __vec16_d, int64_t, __vec16_i64, cast_d_i64, __cast_sitofp)

//CAST_TEST(float , __vec16_f, uint8_t , __vec16_i8,  cast_f_ui8,  __cast_uitofp)
//CAST_TEST(float , __vec16_f, uint16_t, __vec16_i16, cast_f_ui16, __cast_uitofp)
CAST_TEST(float , __vec16_f, uint32_t, __vec16_i32, cast_f_ui32, __cast_uitofp)
//CAST_TEST(float , __vec16_f, uint64_t, __vec16_i64, cast_f_ui64, __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint8_t , __vec16_i8 , cast_d_ui8,  __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint16_t, __vec16_i16, cast_d_ui16, __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint32_t, __vec16_i32, cast_d_ui32, __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint64_t, __vec16_i64, cast_d_ui64, __cast_uitofp)

//CAST_TEST(int8_t , __vec16_i8 , float , __vec16_f, cast_i8_f , __cast_fptosi)
//CAST_TEST(int16_t, __vec16_i16, float , __vec16_f, cast_i16_f, __cast_fptosi)
CAST_TEST(int32_t, __vec16_i32, float , __vec16_f, cast_i32_f, __cast_fptosi)
//CAST_TEST(int64_t, __vec16_i64, float , __vec16_f, cast_i64_f, __cast_fptosi)
//CAST_TEST(int8_t , __vec16_i8 , double, __vec16_d, cast_i8_d , __cast_fptosi)
//CAST_TEST(int16_t, __vec16_i16, double, __vec16_d, cast_i16_d, __cast_fptosi)
//CAST_TEST(int32_t, __vec16_i32, double, __vec16_d, cast_i32_d, __cast_fptosi)
//CAST_TEST(int64_t, __vec16_i64, double, __vec16_d, cast_i64_d, __cast_fptosi)

//CAST_TEST(uint8_t , __vec16_i8 , float , __vec16_f, cast_ui8_f , __cast_fptoui)
//CAST_TEST(uint16_t, __vec16_i16, float , __vec16_f, cast_ui16_f, __cast_fptoui)
CAST_TEST(uint32_t, __vec16_i32, float , __vec16_f, cast_ui32_f, __cast_fptoui)
//CAST_TEST(uint64_t, __vec16_i64, float , __vec16_f, cast_ui64_f, __cast_fptoui)
//CAST_TEST(uint8_t , __vec16_i8 , double, __vec16_d, cast_ui8_d , __cast_fptoui)
//CAST_TEST(uint16_t, __vec16_i16, double, __vec16_d, cast_ui16_d, __cast_fptoui)
//CAST_TEST(uint32_t, __vec16_i32, double, __vec16_d, cast_ui32_d, __cast_fptoui)
//CAST_TEST(uint64_t, __vec16_i64, double, __vec16_d, cast_ui64_d, __cast_fptoui)

CAST_TEST(float , __vec16_f, double, __vec16_d, cast_f_d, __cast_fptrunc)
CAST_TEST(double, __vec16_d, float , __vec16_f, cast_d_f, __cast_fpext)

