#include "knc_test_driver_core.h"

void insert_extract_element_double(double  *d  );
void insert_extract_element_float (float   *f  );
void insert_extract_element_i8    (int8_t  *i8 );
void insert_extract_element_i16   (int16_t *i16);
void insert_extract_element_i32   (int32_t *i32);
void insert_extract_element_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_insert_extract() {
    InputData inpData;

    insert_extract_element_double(inpData.d_32);
    insert_extract_element_float(inpData.f_32);
    insert_extract_element_i8(inpData.i8);
    insert_extract_element_i16(inpData.i16);
    insert_extract_element_i32(inpData.i32);
    insert_extract_element_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define INSERT_EXTRACT_ELEMENT(TYPE, VEC_TYPE, FUNC_NAME)                                   \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (TYPE)copy_data[i]);                                    \
                                                                                            \
    TYPE output[16];                                                                        \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        output[i] = __extract_element(input, i);                                            \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(output[i], data[i], err_counter))                               \
            err_counter++;                                                                  \
        if (check_and_print(copy_data[i], data[i], err_counter))                            \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

INSERT_EXTRACT_ELEMENT(double , __vec16_d  , insert_extract_element_double)
INSERT_EXTRACT_ELEMENT(float  , __vec16_f  , insert_extract_element_float )
INSERT_EXTRACT_ELEMENT(int8_t , __vec16_i8 , insert_extract_element_i8    )
INSERT_EXTRACT_ELEMENT(int16_t, __vec16_i16, insert_extract_element_i16   )
INSERT_EXTRACT_ELEMENT(int32_t, __vec16_i32, insert_extract_element_i32   )
INSERT_EXTRACT_ELEMENT(int64_t, __vec16_i64, insert_extract_element_i64   )
