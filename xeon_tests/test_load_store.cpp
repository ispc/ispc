#include "knc_test_driver_core.h"

void load_double(double  *d  );
void load_float (float   *f  );
void load_i8    (int8_t  *i8 );
void load_i16   (int16_t *i16);
void load_i32   (int32_t *i32);
void load_i64   (int64_t *i64);


void store_double(double  *d  );
void store_float (float   *f  );
void store_i8    (int8_t  *i8 );
void store_i16   (int16_t *i16);
void store_i32   (int32_t *i32);
void store_i64   (int64_t *i64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_load_store () {
    InputData inpData;

    load_double(inpData.d_32);
    load_float(inpData.f_32);
    //load_i8(inpData.i8);
    //load_i16(inpData.i16);
    load_i32(inpData.i32);
    load_i64(inpData.i64);


    store_double(inpData.d_32);
    store_float(inpData.f_32);
    //store_i8(inpData.i8);
    //store_i16(inpData.i16);
    store_i32(inpData.i32);
    store_i64(inpData.i64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                          \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE ptrs;                                                                          \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&ptrs, i, (TYPE)copy_data[i]);                                     \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __load<ALIGN_NUM>(&ptrs);                                                      \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(__extract_element(output, i), data[i], err_counter))            \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

LOAD(double , __vec16_d  , load_double, 128)
LOAD(float  , __vec16_f  , load_float , 64)
//LOAD(int8_t , __vec16_i8 , load_i8    , 16)
//LOAD(int16_t, __vec16_i16, load_i16   , 32)
LOAD(int32_t, __vec16_i32, load_i32   , 64)

void load_i64(int64_t *data) {
    printf ("%-40s", "load_i64" ":");
    int64_t copy_data[16];
    for (uint32_t i = 0; i < 16; i++)
        copy_data[i] = data[i];

    __vec16_i64 ptrs;
    for (uint32_t i = 0; i < 8; i++){
        ((int64_t *)&ptrs.v_hi)[i] = (int64_t)copy_data[i];
        ((int64_t *)&ptrs.v_lo)[i] = (int64_t)copy_data[i + 8];
    }
    __vec16_i64 output;
    output = __load<128>(&ptrs);

    int err_counter = 0;
    for (uint32_t i = 0; i < 16; i++){
        if (__extract_element(output, i) != data[i])
            err_counter++;
    }
    if (err_counter != 0)
        printf(" errors %d\n", err_counter);
    else
        printf(" no fails\n");
}

#define STORE(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                         \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (TYPE)data[i]);                                         \
                                                                                            \
    VEC_TYPE output;                                                                        \
    __store<ALIGN_NUM>(&output, input);                                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (check_and_print(__extract_element(output, i), data[i], err_counter))            \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

STORE(double , __vec16_d  , store_double, 128)
STORE(float  , __vec16_f  , store_float , 64)
//STORE(int8_t , __vec16_i8 , store_i8    , 16)
//STORE(int16_t, __vec16_i16, store_i16   , 32)
STORE(int32_t, __vec16_i32, store_i32   , 64)

void store_i64(int64_t *data) {
    printf ("%-40s", "store_i64" ":");

    int64_t copy_data[16];
    for (uint32_t i = 0; i < 16; i++)
        copy_data[i] = data[i];

    __vec16_i64 input;
    for (uint32_t i = 0; i < 16; i++){
        if(i % 2 == 0) {
            ((int32_t *)&input.v_lo)[i / 2] = (int64_t)copy_data[i] >> 32;
            ((int32_t *)&input.v_lo)[i / 2 + 8] = (int64_t)copy_data[i];
        }
        else {
            ((int32_t *)&input.v_hi)[i / 2] = (int64_t)copy_data[i] >> 32;
            ((int32_t *)&input.v_hi)[i / 2 + 8] = (int64_t)copy_data[i];
        }
    }

    __vec16_i64 output;
    __store<128>(&output, input);

    int err_counter = 0;
    for (uint32_t i = 0; i < 16; i++){
        if (check_and_print(__extract_element(output, i), data[i], err_counter))
            err_counter++;
    }
    if (err_counter != 0)
        printf(" errors %d\n", err_counter);
    else
        printf(" no fails\n");
}
