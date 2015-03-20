#include "knc_test_driver_core.h"

void reduce_add_double(double  *d  );
void reduce_add_float (float   *f  );
void reduce_add_int8  (int8_t  *i8 );
void reduce_add_int16 (int16_t *i16);
void reduce_add_int32 (int32_t *i32);
void reduce_add_int64 (int64_t *i64);


void reduce_min_double (double  *d  );
void reduce_min_float  (float   *f  );
void reduce_min_int32  (int32_t *i32);
void reduce_min_uint32 (uint32_t *ui32);
void reduce_min_int64  (int64_t *i64);
void reduce_min_uint64 (uint64_t *ui64);

void reduce_max_double (double  *d  );
void reduce_max_float  (float   *f  );
void reduce_max_int32  (int32_t *i32);
void reduce_max_uint32 (uint32_t *ui32);
void reduce_max_int64  (int64_t *i64);
void reduce_max_uint64 (uint64_t *ui64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_reduce() {
    InputData inpData;

    reduce_add_double(inpData.no_of_d_32);
    reduce_add_float (inpData.no_of_f_32);
    reduce_add_int8  (inpData.no_of_i8);
    reduce_add_int16 (inpData.no_of_i16);
    reduce_add_int32 (inpData.no_of_i32);
    reduce_add_int64 (inpData.no_of_i64);


    reduce_min_double (inpData.d_32);
    reduce_min_float  (inpData.f_32);
    reduce_min_int32  (inpData.i32);
    reduce_min_uint32 (inpData.ui32);
    reduce_min_int64  (inpData.i64);
    reduce_min_uint64 (inpData.ui64);
    reduce_max_double (inpData.d_32);
    reduce_max_float  (inpData.f_32);
    reduce_max_int32  (inpData.i32);
    reduce_max_uint32 (inpData.ui32);
    reduce_max_int64  (inpData.i64);
    reduce_max_uint64 (inpData.ui64);

}

/////////////////////////////////////////////////////////////////////////////////////////////


#define REDUCE_ADD_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                          \
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
    TYPE output;                                                                            \
    output = __##FUNC_NAME(input);                                                          \
                                                                                            \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        result += (TYPE) data[i];                                                           \
                                                                                            \
    if (check_and_print((TYPE) output, (TYPE) result, 0))                                   \
        printf(" errors 1\n");                                                              \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

REDUCE_ADD_TEST(double , __vec16_d  , reduce_add_double)
REDUCE_ADD_TEST(float  , __vec16_f  , reduce_add_float )
REDUCE_ADD_TEST(int8_t , __vec16_i8 , reduce_add_int8  )
REDUCE_ADD_TEST(int16_t, __vec16_i16, reduce_add_int16 )
REDUCE_ADD_TEST(int32_t, __vec16_i32, reduce_add_int32 )
REDUCE_ADD_TEST(int64_t, __vec16_i64, reduce_add_int64 )

#define REDUCE_MINMAX_TEST(TYPE, VEC_TYPE, RES_NUM, FUNC_NAME)                              \
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
    TYPE output;                                                                            \
    output = __##FUNC_NAME(input);                                                          \
    if (check_and_print((TYPE) output, (TYPE) data[RES_NUM], 0))                            \
        printf(" errors 1\n");                                                              \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

REDUCE_MINMAX_TEST(double  , __vec16_d  , 1, reduce_min_double)
REDUCE_MINMAX_TEST(float   , __vec16_f  , 1, reduce_min_float)
REDUCE_MINMAX_TEST(int32_t , __vec16_i32, 1, reduce_min_int32)
REDUCE_MINMAX_TEST(uint32_t, __vec16_i32, 1, reduce_min_uint32)
REDUCE_MINMAX_TEST(int64_t , __vec16_i64, 1, reduce_min_int64)
REDUCE_MINMAX_TEST(uint64_t, __vec16_i64, 1, reduce_min_uint64)
REDUCE_MINMAX_TEST(double  , __vec16_d  , 0, reduce_max_double)
REDUCE_MINMAX_TEST(float   , __vec16_f  , 0, reduce_max_float)
REDUCE_MINMAX_TEST(int32_t , __vec16_i32, 0, reduce_max_int32)
REDUCE_MINMAX_TEST(uint32_t, __vec16_i32, 0, reduce_max_uint32)
REDUCE_MINMAX_TEST(int64_t , __vec16_i64, 0, reduce_max_int64)
REDUCE_MINMAX_TEST(uint64_t, __vec16_i64, 0, reduce_max_uint64)
