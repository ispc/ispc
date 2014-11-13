#include "knc_test_driver_core.h"

void cast_bits_scalar_ui32_f(float    *f_u32);
void cast_bits_scalar_i32_f (float    *f_32);
void cast_bits_scalar_f_ui32(uint32_t *ui32);
void cast_bits_scalar_f_i32 (int32_t  *i32);
void cast_bits_scalar_ui64_d(double   *d_u64);
void cast_bits_scalar_i64_d (double   *d_64);
void cast_bits_scalar_d_ui64(uint64_t *ui64);
void cast_bits_scalar_d_i64 (int64_t  *i64);

void cast_bits_f_i32(int32_t *i32);
void cast_bits_i32_f(float   *f_32);
void cast_bits_d_i64(int64_t *i64);
void cast_bits_i64_d(double  *d_64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_cast_bits() {
    InputData inpData;

    //cast_bits_scalar_ui32_f(inpData.f_u32);
    //cast_bits_scalar_i32_f (inpData.f_32);
    //cast_bits_scalar_f_ui32(inpData.ui32);
    //cast_bits_scalar_f_i32 (inpData.i32);
    //cast_bits_scalar_ui64_d(inpData.d_u64);
    //cast_bits_scalar_i64_d (inpData.d_64);
    //cast_bits_scalar_d_ui64(inpData.ui64);
    //cast_bits_scalar_d_i64 (inpData.i64);


    cast_bits_f_i32(inpData.i32);
    cast_bits_i32_f(inpData.f_32);
    cast_bits_d_i64(inpData.i64);
    cast_bits_i64_d(inpData.d_64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define CAST_BITS_SCALAR_TEST(TO, FROM, FUNC_NAME)                                          \
void FUNC_NAME(FROM *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    union {                                                                                 \
        TO to;                                                                              \
        FROM from;                                                                          \
    } u;                                                                                    \
                                                                                            \
    TO output;                                                                              \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        output = __cast_bits(output, copy_data[i]);                                         \
        u.from = data[i];                                                                   \
        if ((isnan(output) || isnan(u.to)) && (!isnan(output) != !isnan(u.to)))             \
            err_counter++;                                                                  \
        if (!isnan(output) && !isnan(u.to) &&                                               \
            check_and_print(output, (TO) u.to, err_counter))                                \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//CAST_BITS_SCALAR_TEST(uint32_t, float   , cast_bits_scalar_ui32_f)
//CAST_BITS_SCALAR_TEST(int32_t , float   , cast_bits_scalar_i32_f)
//CAST_BITS_SCALAR_TEST(float   , uint32_t, cast_bits_scalar_f_ui32)
//CAST_BITS_SCALAR_TEST(float   , int32_t , cast_bits_scalar_f_i32)
//CAST_BITS_SCALAR_TEST(uint64_t, double  , cast_bits_scalar_ui64_d)
//CAST_BITS_SCALAR_TEST(int64_t , double  , cast_bits_scalar_i64_d)
//CAST_BITS_SCALAR_TEST(double  , uint64_t, cast_bits_scalar_d_ui64)
//CAST_BITS_SCALAR_TEST(double  , int64_t , cast_bits_scalar_d_i64)

#define CAST_BITS_TEST(TO, TO_VEC, FROM, FROM_VEC, FUNC_NAME)                               \
void FUNC_NAME(FROM *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    union {                                                                                 \
        TO to;                                                                              \
        FROM from;                                                                          \
    }u;                                                                                     \
                                                                                            \
    FROM_VEC input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (FROM) copy_data[i]);                                   \
                                                                                            \
    TO_VEC result;                                                                          \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        u.from = data[i];                                                                   \
        __insert_element(&result, i, (TO) u.to);                                            \
    }                                                                                       \
                                                                                            \
    TO_VEC output;                                                                          \
    output = __cast_bits(output, input);                                                    \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if((isnan(__extract_element(output, i)) || isnan(__extract_element(result, i))) &&  \
          (!isnan(__extract_element(output, i)) != !isnan(__extract_element(result, i))))   \
             err_counter++;                                                                 \
        if(!isnan(__extract_element(output, i)) && !isnan(__extract_element(result, i)) &&  \
          (check_and_print(__extract_element(output, i), __extract_element(result, i),      \
                                                         err_counter)))                     \
             err_counter++;                                                                 \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CAST_BITS_TEST(float  , __vec16_f  , int32_t, __vec16_i32, cast_bits_f_i32)
CAST_BITS_TEST(int32_t, __vec16_i32, float  , __vec16_f  , cast_bits_i32_f)
CAST_BITS_TEST(double , __vec16_d  , int64_t, __vec16_i64, cast_bits_d_i64)
CAST_BITS_TEST(int64_t, __vec16_i64, double , __vec16_d  , cast_bits_i64_d)

