#include "knc_test_driver_core.h"

void rotate_double(double  *d  );
void rotate_float (float   *f  );
void rotate_i8    (int8_t  *i8 );
void rotate_i16   (int16_t *i16);
void rotate_i32   (int32_t *i32);
void rotate_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_rotate() {
    InputData inpData;

    //rotate_double(inpData.d_32);
    //rotate_float(inpData.f_32);
    //rotate_i8(inpData.i8);
    //rotate_i16(inpData.i16);
    rotate_i32(inpData.i32);
    //rotate_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define ROTATE_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                              \
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
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __##FUNC_NAME(input, i);                                                   \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            if (check_and_print(__extract_element(output, j), data[(j + i) % 16],           \
                                                              err_counter))                 \
                err_counter++;                                                              \
        }                                                                                   \
        output = __##FUNC_NAME(input, -i);                                                  \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            if (check_and_print(__extract_element(output, j), data[(j - i) % 16],           \
                                                              err_counter))                 \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//ROTATE_TEST(double , __vec16_d  , rotate_double)
//ROTATE_TEST(float  , __vec16_f  , rotate_float )
//ROTATE_TEST(int8_t , __vec16_i8 , rotate_i8    )
//ROTATE_TEST(int16_t, __vec16_i16, rotate_i16   )
ROTATE_TEST(int32_t, __vec16_i32, rotate_i32   )
//ROTATE_TEST(int64_t, __vec16_i64, rotate_i64   )

