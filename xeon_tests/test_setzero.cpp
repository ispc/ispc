#include "knc_test_driver_core.h"

void setzero_double();
void setzero_float ();
void setzero_i8    ();
void setzero_i16   ();
void setzero_i32   ();
void setzero_i64   ();

/////////////////////////////////////////////////////////////////////////////////////////////

void test_setzero() {
    InputData inpData;

    setzero_double();
    setzero_float();
    //setzero_i8();
    //setzero_i16();
    setzero_i32();
    setzero_i64();

}

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
//SETZERO_TEST(int8_t , __vec16_i8 , setzero_i8    )
//SETZERO_TEST(int16_t, __vec16_i16, setzero_i16   )
SETZERO_TEST(int32_t, __vec16_i32, setzero_i32   )
SETZERO_TEST(int64_t, __vec16_i64, setzero_i64   )
