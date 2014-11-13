#include "knc_test_driver_core.h"

void smear_double(double  *d  );
void smear_float (float   *f  );
void smear_i8    (int8_t  *i8 );
void smear_i16   (int16_t *i16);
void smear_i32   (int32_t *i32);
void smear_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_smear () {
    InputData inpData;

    smear_double(inpData.d_32);
    smear_float(inpData.f_32);
    //smear_i8(inpData.i8);
    //smear_i16(inpData.i16);
    smear_i32(inpData.i32);
    smear_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define SMEAR_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                               \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE output;                                                                        \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        output = __##FUNC_NAME<VEC_TYPE>(copy_data[i]);                                     \
        for (uint32_t j = 0; j < 16; j++)                                                   \
            if (check_and_print(__extract_element(output, j), data[i], err_counter))        \
                err_counter++;                                                              \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SMEAR_TEST(double , __vec16_d  , smear_double)
SMEAR_TEST(float  , __vec16_f  , smear_float )
//SMEAR_TEST(int8_t , __vec16_i8 , smear_i8    )
//SMEAR_TEST(int16_t, __vec16_i16, smear_i16   )
SMEAR_TEST(int32_t, __vec16_i32, smear_i32   )
SMEAR_TEST(int64_t, __vec16_i64, smear_i64   )

