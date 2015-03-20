#include "knc_test_driver_core.h"

void masked_load_double(double  *d  , int *mask);
void masked_load_float (float   *f  , int *mask);
void masked_load_i8    (int8_t  *i8 , int *mask);
void masked_load_i16   (int16_t *i16, int *mask);
void masked_load_i32   (int32_t *i32, int *mask);
void masked_load_i64   (int64_t *i64, int *mask);


void masked_store_double(double  *d  , int *mask);
void masked_store_float (float   *f  , int *mask);
void masked_store_i8    (int8_t  *i8 , int *mask);
void masked_store_i16   (int16_t *i16, int *mask);
void masked_store_i32   (int32_t *i32, int *mask);
void masked_store_i64   (int64_t *i64, int *mask);

void masked_store_blend_double(double  *d  , int *mask);
void masked_store_blend_float (float   *f  , int *mask);
void masked_store_blend_i8    (int8_t  *i8 , int *mask);
void masked_store_blend_i16   (int16_t *i16, int *mask);
void masked_store_blend_i32   (int32_t *i32, int *mask);
void masked_store_blend_i64   (int64_t *i64, int *mask);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_masked_load_store() {
    InputData inpData;

    masked_load_double(inpData.d_32, inpData.mask);
    masked_load_float(inpData.f_32, inpData.mask);
    masked_load_i8(inpData.i8, inpData.mask);
    masked_load_i16(inpData.i16, inpData.mask);
    masked_load_i32(inpData.i32, inpData.mask);
    masked_load_i64(inpData.i64, inpData.mask);


    masked_store_double(inpData.d_32, inpData.mask);
    masked_store_float(inpData.f_32, inpData.mask);
    masked_store_i8(inpData.i8, inpData.mask);
    masked_store_i16(inpData.i16, inpData.mask);
    masked_store_i32(inpData.i32, inpData.mask);
    masked_store_i64(inpData.i64, inpData.mask);

    //masked_store_blend_double(inpData.d_32, inpData.mask);
    masked_store_blend_float(inpData.f_32, inpData.mask);
    //masked_store_blend_i8(inpData.i8, inpData.mask);
    //masked_store_blend_i16(inpData.i16, inpData.mask);
    masked_store_blend_i32(inpData.i32, inpData.mask);
    //masked_store_blend_i64(inpData.i64, inpData.mask);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define MASKED_LOAD(TYPE, VEC_TYPE, FUNC_NAME)                                              \
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
    TYPE ptrs[16];                                                                          \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        ptrs[i] = (TYPE) copy_data[i];                                                      \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(ptrs, mask);                                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 &&                                                                    \
            check_and_print(__extract_element(output, i), data[i], err_counter))            \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

MASKED_LOAD(double , __vec16_d  , masked_load_double)
MASKED_LOAD(float  , __vec16_f  , masked_load_float )
MASKED_LOAD(int8_t , __vec16_i8 , masked_load_i8    )
MASKED_LOAD(int16_t, __vec16_i16, masked_load_i16   )
MASKED_LOAD(int32_t, __vec16_i32, masked_load_i32   )
MASKED_LOAD(int64_t, __vec16_i64, masked_load_i64   )

#define MASKED_STORE(TYPE, VEC_TYPE, FUNC_NAME)                                             \
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
    TYPE ptrs[16];                                                                          \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (TYPE) copy_data[i]);                                   \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    __##FUNC_NAME(ptrs, input, mask);                                                       \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
                         /* Cause segfault on icpc -O2 and icpc -O3 */                      \
        if (m[i] != 0 && check_and_print(ptrs[i], data[i], err_counter))                    \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

MASKED_STORE(double , __vec16_d  , masked_store_double)
MASKED_STORE(float  , __vec16_f  , masked_store_float )
MASKED_STORE(int8_t , __vec16_i8 , masked_store_i8    )
MASKED_STORE(int16_t, __vec16_i16, masked_store_i16   )
MASKED_STORE(int32_t, __vec16_i32, masked_store_i32   )
MASKED_STORE(int64_t, __vec16_i64, masked_store_i64   )

//MASKED_STORE(double , __vec16_d  , masked_store_blend_double)
MASKED_STORE(float  , __vec16_f  , masked_store_blend_float )
//MASKED_STORE(int8_t , __vec16_i8 , masked_store_blend_i8    )
//MASKED_STORE(int16_t, __vec16_i16, masked_store_blend_i16   )
MASKED_STORE(int32_t, __vec16_i32, masked_store_blend_i32   )
//MASKED_STORE(int64_t, __vec16_i64, masked_store_blend_i64   )
