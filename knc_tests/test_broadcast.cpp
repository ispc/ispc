#include "knc_test_driver_core.h"

void broadcast_double(double  *d  );
void broadcast_float (float   *f  );
void broadcast_i8    (int8_t  *i8 );
void broadcast_i16   (int16_t *i16);
void broadcast_i32   (int32_t *i32);
void broadcast_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_broadcast() {
    InputData inpData;

    broadcast_double(inpData.d_32);
    broadcast_float(inpData.f_32);
    //broadcast_i8(inpData.i8);
    //broadcast_i16(inpData.i16);
    broadcast_i32(inpData.i32);
    //broadcast_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define BROADCAST_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                           \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (TYPE) copy_data[i]);                                   \
                                                                                            \
    VEC_TYPE output;                                                                        \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        output = __##FUNC_NAME(input, i);                                                   \
        for (int32_t j = 0; j < 16; j++){                                                   \
        if (__extract_element(output, j) != data[i])                                        \
            err_counter++;                                                                  \
        }                                                                                   \
        output = __##FUNC_NAME(input, -i);                                                  \
        for (int32_t j = 0; j < 16; j++){                                                   \
        if (check_and_print(__extract_element(output, j), data[(16 - i) % 16], err_counter))\
            err_counter++;                                                                  \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

BROADCAST_TEST(double , __vec16_d  , broadcast_double)
BROADCAST_TEST(float  , __vec16_f  , broadcast_float )
//BROADCAST_TEST(int8_t , __vec16_i8 , broadcast_i8    )
//BROADCAST_TEST(int16_t, __vec16_i16, broadcast_i16   )
BROADCAST_TEST(int32_t, __vec16_i32, broadcast_i32   )
//BROADCAST_TEST(int64_t, __vec16_i64, broadcast_i64   )

