#include "knc_test_driver_core.h"

void select_double(double  *d  , int *mask);
void select_float (float   *f  , int *mask);
void select_i8    (int8_t  *i8 , int *mask);
void select_i16   (int16_t *i16, int *mask);
void select_i32   (int32_t *i32, int *mask);
void select_i64   (int64_t *i64, int *mask);

void select_double_cond(double  *d  , int *mask);
void select_float_cond (float   *f  , int *mask);
void select_i8_cond    (int8_t  *i8 , int *mask);
void select_i16_cond   (int16_t *i16, int *mask);
void select_i32_cond   (int32_t *i32, int *mask);
void select_i64_cond   (int64_t *i64, int *mask);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_select() {
    InputData inpData;

    select_double(inpData.d_32, inpData.mask);
    select_float(inpData.f_32, inpData.mask);
    select_i8(inpData.i8, inpData.mask);
    select_i16(inpData.i16, inpData.mask);
    select_i32(inpData.i32, inpData.mask);
    select_i64(inpData.i64, inpData.mask);

    select_double_cond(inpData.d_32, inpData.mask);
    select_float_cond(inpData.f_32, inpData.mask);
    select_i8_cond(inpData.i8, inpData.mask);
    select_i16_cond(inpData.i16, inpData.mask);
    select_i32_cond(inpData.i32, inpData.mask);
    select_i64_cond(inpData.i64, inpData.mask);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define SELECT_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                              \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input1;                                                                        \
    VEC_TYPE input2;                                                                        \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&input1, i, (TYPE) copy_data[i]);                                  \
        __insert_element(&input2, i, (TYPE) copy_data[i] / 2);                              \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __select(mask, input1, input2);                                                \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 && check_and_print(__extract_element(output, i), (TYPE) (data[i]),    \
                                         err_counter))                                      \
            err_counter++;                                                                  \
        if (m[i] == 0 && check_and_print(__extract_element(output, i), (TYPE)(data[i] / 2), \
                                        err_counter))                                       \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}                                                                                           \
                                                                                            \
void FUNC_NAME##_cond(TYPE *data, int *m) {                                                 \
    printf ("%-40s", #FUNC_NAME "_cond" ":");                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input1;                                                                        \
    VEC_TYPE input2;                                                                        \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&input1, i, (TYPE) copy_data[i]);                                  \
        __insert_element(&input2, i, (TYPE) copy_data[i] * -1);                             \
    }                                                                                       \
                                                                                            \
    VEC_TYPE output;                                                                        \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __select(copy_m[i], input1, input2);                                       \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            if (m[i] != 0 && check_and_print(__extract_element(output, j),                  \
                                             (TYPE)(data[j]), err_counter))                 \
                err_counter++;                                                              \
            if (m[i] == 0 && check_and_print(__extract_element(output, j),                  \
                                            (TYPE)(data[j] * -1), err_counter))             \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SELECT_TEST(double , __vec16_d  , select_double)
SELECT_TEST(float  , __vec16_f  , select_float )
SELECT_TEST(int8_t , __vec16_i8 , select_i8    )
SELECT_TEST(int16_t, __vec16_i16, select_i16   )
SELECT_TEST(int32_t, __vec16_i32, select_i32   )
SELECT_TEST(int64_t, __vec16_i64, select_i64   )

