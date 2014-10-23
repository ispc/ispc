#include "knc_test_driver_core.h"

void shift_double(double  *d  );
void shift_float (float   *f  );
void shift_i8    (int8_t  *i8 );
void shift_i16   (int16_t *i16);
void shift_i32   (int32_t *i32);
void shift_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_shift() {
    InputData inpData;

    //shift_double(inpData.d_32);
    //shift_float(inpData.f_32);
    //shift_i8(inpData.i8);
    //shift_i16(inpData.i16);
    //shift_i32(inpData.i32);
    //shift_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define SHIFT_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                               \
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
        for (uint32_t j = 0; j < 16; j++) {                                                 \
            if ((j + i >=16 || j + i < 0) && check_and_print(__extract_element(output, j),  \
                                                            (TYPE) 0, err_counter))         \
                err_counter++;                                                              \
            if (j + i < 16 && check_and_print(__extract_element(output, j),                 \
                                             (TYPE) data[i + j], err_counter))              \
                err_counter++;                                                              \
        }                                                                                   \
                                                                                            \
        output = __##FUNC_NAME(input, -i);                                                  \
        for (uint32_t j = 0; j < 16; j++) {                                                 \
            if ((j - i >=16 || j - i < 0) && check_and_print(__extract_element(output, j),  \
                                                             (TYPE) 0, err_counter))        \
                err_counter++;                                                              \
            if (j - i < 16 && check_and_print(__extract_element(output, j),                 \
                                             (TYPE) data[j - i], err_counter))              \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//SHIFT_TEST(double , __vec16_d  , shift_double)
//SHIFT_TEST(float  , __vec16_f  , shift_float )
//SHIFT_TEST(int8_t , __vec16_i8 , shift_i8    )
//SHIFT_TEST(int16_t, __vec16_i16, shift_i16   )
//SHIFT_TEST(int32_t, __vec16_i32, shift_i32   )
//SHIFT_TEST(int64_t, __vec16_i64, shift_i64   )

