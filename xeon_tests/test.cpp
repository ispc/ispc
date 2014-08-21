// TODO: change input index in insert/extract
#define __STDC_LIMIT_MACROS // enable intN_t limits from stdint.h
#include <stdint.h>

#ifdef KNC_H
    #include "knc.h"
#else
    #include "knc-i1x16.h"
    #include <stdio.h>
#endif

template <typename T>
void allocator(T **array) {
    uint64_t seed = 123456789;
    int m = 100;
    int a = 1103515245;
    int c = 12345;
    T* tmp[4];
    seed = (a * seed + c) % m;
    void* tmp1 = (void*) malloc(seed);

    for (int j = 0; j < 4; j++) {
        for (int i = 4 * j; i < 4 * (j + 1); i++)
            array[i] = (T*) malloc(sizeof(T));
        seed = (a * seed + c) % m;
        tmp[j] = (T*) malloc(seed * sizeof(T));
    }

    for (int j = 0; j < 4; j++)
        free(tmp[j]);
    
    free(tmp1);

}

/////////////////////////////////////////////////////////////////////////////////////////////
/*
    int8_t copy_data[16];
    int copy_m[16];
    for (uint32_t i = 0; i < 16; i++) {
        copy_data[i] = data[i];
        copy_m[i] = m[i];
    }

    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);

    int err_counter = 0;
    for (uint32_t i = 0; i < 16; i++){
        if (m[i] != 0 && __extract_element(output, i) != data[i])
            err_counter++;
    }
    if (err_counter != 0)
        printf(" errors %d\n", err_counter);
    else
         printf(" no fails\n");
*/
/////////////////////////////////////////////////////////////////////////////////////////////
#define INSERT_EXTRACT_ELEMENT(TYPE, VEC_TYPE, FUNC_NAME)                                   \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (output[i] != data[i])                                                           \
            err_counter++;                                                                  \
        if (copy_data[i] != data[i])                                                        \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

INSERT_EXTRACT_ELEMENT(double , __vec16_d  , insert_extract_element_double)
INSERT_EXTRACT_ELEMENT(float  , __vec16_f  , insert_extract_element_float )
INSERT_EXTRACT_ELEMENT(int8_t , __vec16_i8 , insert_extract_element_i8    )
INSERT_EXTRACT_ELEMENT(int16_t, __vec16_i16, insert_extract_element_i16   )
INSERT_EXTRACT_ELEMENT(int32_t, __vec16_i32, insert_extract_element_i32   )
INSERT_EXTRACT_ELEMENT(int64_t, __vec16_i64, insert_extract_element_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define LOAD(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                          \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (__extract_element(output, i) != data[i])                                        \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

LOAD(double , __vec16_d  , load_double, 128)
LOAD(float  , __vec16_f  , load_float , 64)
LOAD(int8_t , __vec16_i8 , load_i8    , 16)
LOAD(int16_t, __vec16_i16, load_i16   , 32)
LOAD(int32_t, __vec16_i32, load_i32   , 64)
LOAD(int64_t, __vec16_i64, load_i64   , 128)

/////////////////////////////////////////////////////////////////////////////////////////////
#define STORE(TYPE, VEC_TYPE, FUNC_NAME, ALIGN_NUM)                                         \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (__extract_element(output, i) != data[i])                                        \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

STORE(double , __vec16_d  , store_double, 128)
STORE(float  , __vec16_f  , store_float , 64)
STORE(int8_t , __vec16_i8 , store_i8    , 16)
STORE(int16_t, __vec16_i16, store_i16   , 32)
STORE(int32_t, __vec16_i32, store_i32   , 64)
STORE(int64_t, __vec16_i64, store_i64   , 128)

/////////////////////////////////////////////////////////////////////////////////////////////
#define SMEAR_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                               \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE output;                                                                        \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        output = __##FUNC_NAME<VEC_TYPE>(copy_data[i]);                                     \
        for (uint32_t j = 0; j < 16; j++)                                                   \
            if (__extract_element(output, j) != data[i])                                    \
                err_counter++;                                                              \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SMEAR_TEST(double , __vec16_d  , smear_double)
SMEAR_TEST(float  , __vec16_f  , smear_float )
SMEAR_TEST(int8_t , __vec16_i8 , smear_i8    )
SMEAR_TEST(int16_t, __vec16_i16, smear_i16   )
SMEAR_TEST(int32_t, __vec16_i32, smear_i32   )
SMEAR_TEST(int64_t, __vec16_i64, smear_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define SETZERO_TEST(VEC_TYPE, FUNC_NAME)                                                   \
void FUNC_NAME() {                                                                          \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME<VEC_TYPE>();                                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        if (__extract_element(output, i) != 0)                                              \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

SETZERO_TEST(__vec16_d  , setzero_double)
SETZERO_TEST(__vec16_f  , setzero_float )
SETZERO_TEST(__vec16_i8 , setzero_i8    )
SETZERO_TEST(__vec16_i16, setzero_i16   )
SETZERO_TEST(__vec16_i32, setzero_i32   )
SETZERO_TEST(__vec16_i64, setzero_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define SELECT_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                              \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (m[i] != 0 && __extract_element(output, i) != data[i])                           \
            err_counter++;                                                                  \
        if (m[i] == 0 && __extract_element(output, i) != data[i] / 2)                       \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

SELECT_TEST(double , __vec16_d  , select_double)
SELECT_TEST(float  , __vec16_f  , select_float )
SELECT_TEST(int8_t , __vec16_i8 , select_i8    )
SELECT_TEST(int16_t, __vec16_i16, select_i16   )
SELECT_TEST(int32_t, __vec16_i32, select_i32   )
SELECT_TEST(int64_t, __vec16_i64, select_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define BROADCAST_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                           \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (__extract_element(output, j) != data[(16 - i) % 16])                            \
            err_counter++;                                                                  \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

BROADCAST_TEST(double , __vec16_d  , broadcast_double)
BROADCAST_TEST(float  , __vec16_f  , broadcast_float )
BROADCAST_TEST(int8_t , __vec16_i8 , broadcast_i8    )
BROADCAST_TEST(int16_t, __vec16_i16, broadcast_i16   )
BROADCAST_TEST(int32_t, __vec16_i32, broadcast_i32   )
BROADCAST_TEST(int64_t, __vec16_i64, broadcast_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define ROTATE_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                              \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
            if (__extract_element(output, j) != data[(j + i) % 16])                         \
                err_counter++;                                                              \
        }                                                                                   \
        output = __##FUNC_NAME(input, -i);                                                  \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            if (__extract_element(output, j) != data[(j - i) % 16])                         \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

ROTATE_TEST(double , __vec16_d  , rotate_double)
ROTATE_TEST(float  , __vec16_f  , rotate_float )
ROTATE_TEST(int8_t , __vec16_i8 , rotate_i8    )
ROTATE_TEST(int16_t, __vec16_i16, rotate_i16   )
ROTATE_TEST(int32_t, __vec16_i32, rotate_i32   )
ROTATE_TEST(int64_t, __vec16_i64, rotate_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define SHIFT_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                               \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
            if ((j + i >=16 || j + i < 0) && __extract_element(output, j) != 0)             \
                err_counter++;                                                              \
            if (j + i < 16 && __extract_element(output, j) != data[i + j])                  \
                err_counter++;                                                              \
        }                                                                                   \
                                                                                            \
        output = __##FUNC_NAME(input, -i);                                                  \
        for (uint32_t j = 0; j < 16; j++) {                                                 \
            if ((j - i >=16 || j - i < 0) && __extract_element(output, j) != 0)             \
                err_counter++;                                                              \
            if (j - i < 16 && __extract_element(output, j) != data[j - i])                  \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

SHIFT_TEST(double , __vec16_d  , shift_double)
SHIFT_TEST(float  , __vec16_f  , shift_float )
SHIFT_TEST(int8_t , __vec16_i8 , shift_i8    )
SHIFT_TEST(int16_t, __vec16_i16, shift_i16   )
SHIFT_TEST(int32_t, __vec16_i32, shift_i32   )
SHIFT_TEST(int64_t, __vec16_i64, shift_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define SHUFFLE_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                             \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i ,(int32_t) copy_data[i]);                                \
                                                                                            \
    __vec16_i32 index;                                                                      \
    /* 8|12|9|13|10|14|11|15|0|1|2|3|7|6|5|4  */                                            \
    for (uint32_t i = 0; i < 8; i++)                                                        \
        if (i % 2 == 0)                                                                     \
            __insert_element(&index, i , 8 + i / 2);                                        \
        else                                                                                \
            __insert_element(&index, i , 12 + (i - 1) / 2);                                 \
                                                                                            \
    for (uint32_t i = 8; i < 12; i++)                                                       \
        __insert_element(&index, i , i - 8);                                                \
                                                                                            \
    for (uint32_t i = 12; i < 16; i++)                                                      \
        __insert_element(&index, i , -i + 3);                                               \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(input, index);                                                   \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 8; i++){                                                       \
        if (i % 2 == 0 && __extract_element(output, i) != data[8 + i / 2])                  \
            err_counter++;                                                                  \
        if(i % 2 != 0 && __extract_element(output, i) != data[12 + (i - 1) / 2])            \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    for (uint32_t i = 8; i < 12; i++)                                                       \
        if (__extract_element(output, i) != data[i - 8])                                    \
            err_counter++;                                                                  \
                                                                                            \
    for (uint32_t i = 12; i < 16; i++)                                                      \
        if (__extract_element(output, i) != data[19 - i])                                   \
            err_counter++;                                                                  \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

SHUFFLE_TEST(double , __vec16_d  , shuffle_double)
SHUFFLE_TEST(float  , __vec16_f  , shuffle_float )
SHUFFLE_TEST(int8_t , __vec16_i8 , shuffle_i8    )
SHUFFLE_TEST(int16_t, __vec16_i16, shuffle_i16   )
SHUFFLE_TEST(int32_t, __vec16_i32, shuffle_i32   )
//SHUFFLE_TEST(int64_t, __vec16_i64, shuffle_i64   ) //undefined

/////////////////////////////////////////////////////////////////////////////////////////////
#define CAST_TEST(TO, TO_VEC, FROM, FROM_VEC, FUNC_NAME, FUNC_CALL)                         \
void FUNC_NAME(FROM *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    FROM copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    FROM_VEC input;                                                                         \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input, i, (FROM) copy_data[i]);                                   \
                                                                                            \
    TO_VEC output;                                                                          \
    output = FUNC_CALL(output, input);                                                      \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (__extract_element(output, i) != (TO)data[i])                                    \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

CAST_TEST(int64_t, __vec16_i64, int32_t, __vec16_i32, cast_i64_i32, __cast_sext)
CAST_TEST(int64_t, __vec16_i64, int16_t, __vec16_i16, cast_i64_i16, __cast_sext)
CAST_TEST(int64_t, __vec16_i64, int8_t , __vec16_i8 , cast_i64_i8 , __cast_sext)
CAST_TEST(int32_t, __vec16_i32, int16_t, __vec16_i16, cast_i32_i16, __cast_sext)
CAST_TEST(int32_t, __vec16_i32, int8_t , __vec16_i8 , cast_i32_i8 , __cast_sext)
CAST_TEST(int16_t, __vec16_i16, int8_t , __vec16_i8 , cast_i16_i8 , __cast_sext)
// TODO: __vec16_i1 cast

CAST_TEST(uint64_t, __vec16_i64, uint32_t, __vec16_i32, cast_ui64_ui32, __cast_zext)
CAST_TEST(uint64_t, __vec16_i64, uint16_t, __vec16_i16, cast_ui64_ui16, __cast_zext)
CAST_TEST(uint64_t, __vec16_i64, uint8_t , __vec16_i8 , cast_ui64_ui8 , __cast_zext)
CAST_TEST(uint32_t, __vec16_i32, uint16_t, __vec16_i16, cast_ui32_ui16, __cast_zext)
CAST_TEST(uint32_t, __vec16_i32, uint8_t , __vec16_i8 , cast_ui32_ui8 , __cast_zext)
CAST_TEST(uint16_t, __vec16_i16, uint8_t , __vec16_i8 , cast_ui16_ui8 , __cast_zext)

CAST_TEST(int32_t, __vec16_i32, int64_t, __vec16_i64, trunk_i32_i64, __cast_trunc)
CAST_TEST(int16_t, __vec16_i16, int64_t, __vec16_i64, trunk_i16_i64, __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int64_t, __vec16_i64, trunk_i8_i64 , __cast_trunc)
CAST_TEST(int16_t, __vec16_i16, int32_t, __vec16_i32, trunk_i16_i32, __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int32_t, __vec16_i32, trunk_i8_i32 , __cast_trunc)
CAST_TEST(int8_t , __vec16_i8 , int16_t, __vec16_i16, trunk_i8_i16 , __cast_trunc)

/////////////////////////////////////////////////////////////////////////////////////////////

#define GATHER(GATHER_SCALAR_TYPE, GATHER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)              \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    GATHER_VEC_TYPE ptrs;                                                                   \
    TYPE *b[16];                                                                            \
    allocator<TYPE>(b);                                                                     \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        *b[i] = (TYPE) copy_data[i];                                                        \
        __insert_element(&ptrs, i, (GATHER_SCALAR_TYPE)b[i]);                               \
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
        if (m[i] != 0 && __extract_element(output, i) != data[i])                           \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
                                                                                            \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        free(b[i]);                                                                         \
}

GATHER(int32_t, __vec16_i32, double , __vec16_d  , gather32_double)
GATHER(int32_t, __vec16_i32, float  , __vec16_f  , gather32_float )
GATHER(int32_t, __vec16_i32, int8_t , __vec16_i8 , gather32_i8    )
GATHER(int32_t, __vec16_i32, int16_t, __vec16_i16, gather32_i16   )
GATHER(int32_t, __vec16_i32, int32_t, __vec16_i32, gather32_i32   )
GATHER(int32_t, __vec16_i32, int64_t, __vec16_i64, gather32_i64   )

GATHER(int64_t, __vec16_i64, double , __vec16_d  , gather64_double)
GATHER(int64_t, __vec16_i64, float  , __vec16_f  , gather64_float )
GATHER(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather64_i8    )
GATHER(int64_t, __vec16_i64, int16_t, __vec16_i16, gather64_i16   )
GATHER(int64_t, __vec16_i64, int32_t, __vec16_i32, gather64_i32   )
GATHER(int64_t, __vec16_i64, int64_t, __vec16_i64, gather64_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define GATHER_OFFSETS(GATHER_SCALAR_TYPE, GATHER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)      \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    TYPE *b[16];                                                                            \
    allocator<TYPE>(b);                                                                     \
    uint8_t *base = (uint8_t*) b[0];                                                        \
    uint32_t scale = sizeof(TYPE);                                                          \
    GATHER_VEC_TYPE offsets;                                                                \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        *b[i] = (TYPE) copy_data[i];                                                        \
        __insert_element(&offsets, i, (GATHER_SCALAR_TYPE)(b[i] - b[0]));                   \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(base, scale, offsets, mask);                                     \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 && __extract_element(output, i) != data[i])                           \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
                                                                                            \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        free(b[i]);                                                                         \
                                                                                            \
}

GATHER_OFFSETS(int32_t, __vec16_i32, double , __vec16_d  , gather_base_offsets32_double)
GATHER_OFFSETS(int32_t, __vec16_i32, float  , __vec16_f  , gather_base_offsets32_float )
GATHER_OFFSETS(int32_t, __vec16_i32, int8_t , __vec16_i8 , gather_base_offsets32_i8    )
GATHER_OFFSETS(int32_t, __vec16_i32, int16_t, __vec16_i16, gather_base_offsets32_i16   )
GATHER_OFFSETS(int32_t, __vec16_i32, int32_t, __vec16_i32, gather_base_offsets32_i32   )
GATHER_OFFSETS(int32_t, __vec16_i32, int64_t, __vec16_i64, gather_base_offsets32_i64   )

GATHER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , gather_base_offsets64_double)
GATHER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , gather_base_offsets64_float )
GATHER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather_base_offsets64_i8    )
GATHER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, gather_base_offsets64_i16   )
GATHER_OFFSETS(int64_t, __vec16_i64, int32_t, __vec16_i32, gather_base_offsets64_i32   )
GATHER_OFFSETS(int64_t, __vec16_i64, int64_t, __vec16_i64, gather_base_offsets64_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define SCATTER(SCATTER_SCALAR_TYPE, SCATTER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)           \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_m[16];                                                                        \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    SCATTER_VEC_TYPE ptrs;                                                                  \
    VEC_TYPE input;                                                                         \
    TYPE *b[16];                                                                            \
    allocator(b);                                                                           \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&ptrs,  i, (SCATTER_SCALAR_TYPE) b[i]);                            \
        __insert_element(&input, i, (TYPE) data[i]);                                        \
    }                                                                                       \
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
        TYPE *p =  (TYPE*) __extract_element(ptrs, i);                                      \
        if (m[i] != 0 && *p  != data[i])                                                    \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
                                                                                            \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        free(b[i]);                                                                         \
                                                                                            \
}

SCATTER(int32_t, __vec16_i32, double , __vec16_d  , scatter32_double)
SCATTER(int32_t, __vec16_i32, float  , __vec16_f  , scatter32_float )
SCATTER(int32_t, __vec16_i32, int8_t , __vec16_i8 , scatter32_i8    )
SCATTER(int32_t, __vec16_i32, int16_t, __vec16_i16, scatter32_i16   )
SCATTER(int32_t, __vec16_i32, int32_t, __vec16_i32, scatter32_i32   )
SCATTER(int32_t, __vec16_i32, int64_t, __vec16_i64, scatter32_i64   )

SCATTER(int64_t, __vec16_i64, double , __vec16_d  , scatter64_double)
SCATTER(int64_t, __vec16_i64, float  , __vec16_f  , scatter64_float )
SCATTER(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter64_i8    )
SCATTER(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter64_i16   )
SCATTER(int64_t, __vec16_i64, int32_t, __vec16_i32, scatter64_i32   )
SCATTER(int64_t, __vec16_i64, int64_t, __vec16_i64, scatter64_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define SCATTER_OFFSETS(SCATTER_SCALAR_TYPE, SCATTER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)   \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_m[16];                                                                        \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_m[i] = m[i];                                                                   \
                                                                                            \
    TYPE *b[16];                                                                            \
    allocator<TYPE>(b);                                                                     \
    uint8_t *base = (uint8_t*) b[0];                                                        \
    uint32_t scale = sizeof(TYPE);                                                          \
    SCATTER_VEC_TYPE offsets;                                                               \
    VEC_TYPE input;                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        __insert_element(&offsets, i, (SCATTER_SCALAR_TYPE) (b[i] - b[0]));                 \
        __insert_element(&input,   i, (TYPE) data[i]);                                      \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    __##FUNC_NAME(base, scale, offsets, input, mask);                                       \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 && *b[i] != data[i])                                                  \
            err_counter++;                                                                  \
    }                                                                                       \
                                                                                            \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
                                                                                            \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        free(b[i]);                                                                         \
                                                                                            \
}

SCATTER_OFFSETS(int32_t, __vec16_i32, double , __vec16_d  , scatter_base_offsets32_double)
SCATTER_OFFSETS(int32_t, __vec16_i32, float  , __vec16_f  , scatter_base_offsets32_float )
SCATTER_OFFSETS(int32_t, __vec16_i32, int8_t , __vec16_i8 , scatter_base_offsets32_i8    )
SCATTER_OFFSETS(int32_t, __vec16_i32, int16_t, __vec16_i16, scatter_base_offsets32_i16   )
SCATTER_OFFSETS(int32_t, __vec16_i32, int32_t, __vec16_i32, scatter_base_offsets32_i32   )
SCATTER_OFFSETS(int32_t, __vec16_i32, int64_t, __vec16_i64, scatter_base_offsets32_i64   )

SCATTER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , scatter_base_offsets64_double)
SCATTER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , scatter_base_offsets64_float )
SCATTER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter_base_offsets64_i8    )
SCATTER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter_base_offsets64_i16   )
SCATTER_OFFSETS(int64_t, __vec16_i64, int32_t, __vec16_i32, scatter_base_offsets64_i32   )
SCATTER_OFFSETS(int64_t, __vec16_i64, int64_t, __vec16_i64, scatter_base_offsets64_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

#define MASKED_LOAD(TYPE, VEC_TYPE, FUNC_NAME)                                              \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (m[i] != 0 && __extract_element(output, i) != data[i])                           \
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

/////////////////////////////////////////////////////////////////////////////////////////////

#define MASKED_STORE(TYPE, VEC_TYPE, FUNC_NAME)                                             \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (m[i] != 0 && ptrs[i] != data[i])                                                \
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

MASKED_STORE(double , __vec16_d  , masked_store_blend_double)
MASKED_STORE(float  , __vec16_f  , masked_store_blend_float )
MASKED_STORE(int8_t , __vec16_i8 , masked_store_blend_i8    )
MASKED_STORE(int16_t, __vec16_i16, masked_store_blend_i16   )
MASKED_STORE(int32_t, __vec16_i32, masked_store_blend_i32   )
MASKED_STORE(int64_t, __vec16_i64, masked_store_blend_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////

void movmsk(int *m) {
    printf ("movmsk: ");
    
    int copy_m[16];
    for (uint32_t i = 0; i < 16; i++) 
        copy_m[i] = m[i];

    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);

    __vec16_i1 copy_mask = mask;

    __vec16_i1 output;
    output = __movmsk(copy_mask);

    if (output != mask)
        printf(" error 1\n");
    else
        printf(" no fails\n");
}
