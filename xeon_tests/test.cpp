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
#define BINARY_OP_TEST(TYPE, VEC_TYPE, OP, FUNC_NAME, TYPE_MOD)                             \
void  FUNC_NAME##_##TYPE_MOD(TYPE *a, TYPE *b) {                                            \
    printf (#FUNC_NAME "_" #TYPE_MOD ":");                                                  \
                                                                                            \
    TYPE copy_a[16];                                                                        \
    TYPE copy_b[16];                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        copy_a[i] = a[i];                                                                   \
        copy_b[i] = b[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    VEC_TYPE input_b;                                                                       \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        __insert_element(&input_a, i, (TYPE)copy_a[i]);                                     \
        __insert_element(&input_b, i, (TYPE)copy_b[i]);                                     \
    }                                                                                       \
                                                                                            \
    VEC_TYPE output;                                                                        \
    output = __##FUNC_NAME(input_a, input_b);                                               \
                                                                                            \
    int err_counter = 0;                                                                    \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        result = (TYPE) (a[i]) OP (TYPE) (b[i]);                                            \
        if (__extract_element(output, i) != result)                                         \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

BINARY_OP_TEST(double , __vec16_d  , +, add, double)
BINARY_OP_TEST(float  , __vec16_f  , +, add, float)
//BINARY_OP_TEST(int8_t , __vec16_i8 , +, add, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, +, add, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, +, add, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, +, add, i64)


BINARY_OP_TEST(double , __vec16_d  , -, sub, double)
BINARY_OP_TEST(float  , __vec16_f  , -, sub, float)
//BINARY_OP_TEST(int8_t , __vec16_i8 , -, sub, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, -, sub, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, -, sub, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, -, sub, i64)

BINARY_OP_TEST(double , __vec16_d  , *, mul, double)
BINARY_OP_TEST(float  , __vec16_f  , *, mul, float)
//BINARY_OP_TEST(int8_t , __vec16_i8 , *, mul, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, *, mul, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, *, mul, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, *, mul, i64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , |, or, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, |, or, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, |, or, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, |, or, i64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , &, and, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, &, and, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, &, and, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, &, and, i64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , ^, xor, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, ^, xor, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, ^, xor, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, ^, xor, i64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , <<, shl, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, <<, shl, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, <<, shl, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, <<, shl, i64)

//BINARY_OP_TEST(uint8_t , __vec16_i8 , /, udiv, ui8)
//BINARY_OP_TEST(uint16_t, __vec16_i16, /, udiv, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, /, udiv, ui32)
BINARY_OP_TEST(uint64_t, __vec16_i64, /, udiv, ui64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , /, sdiv, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, /, sdiv, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, /, sdiv, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, /, sdiv, i64)

BINARY_OP_TEST(double, __vec16_d, /, div, double)
BINARY_OP_TEST(float , __vec16_f, /, div, float)

//BINARY_OP_TEST(uint8_t , __vec16_i8 , %, urem, ui8)
//BINARY_OP_TEST(uint16_t, __vec16_i16, %, urem, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, %, urem, ui32)
//BINARY_OP_TEST(uint64_t, __vec16_i64, %, urem, ui64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , %, srem, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, %, srem, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, %, srem, i32)
//BINARY_OP_TEST(int64_t, __vec16_i64, %, srem, i64)

//BINARY_OP_TEST(uint8_t , __vec16_i8 , >>, lshr, ui8)
//BINARY_OP_TEST(uint16_t, __vec16_i16, >>, lshr, ui16)
BINARY_OP_TEST(uint32_t, __vec16_i32, >>, lshr, ui32)
BINARY_OP_TEST(uint64_t, __vec16_i64, >>, lshr, ui64)

//BINARY_OP_TEST(int8_t , __vec16_i8 , >>, ashr, i8)
//BINARY_OP_TEST(int16_t, __vec16_i16, >>, ashr, i16)
BINARY_OP_TEST(int32_t, __vec16_i32, >>, ashr, i32)
BINARY_OP_TEST(int64_t, __vec16_i64, >>, ashr, i64)

/////////////////////////////////////////////////////////////////////////////////////////////

#define SHIFT_UNIFORM_TEST(TYPE, VEC_TYPE, OP, FUNC_NAME, TYPE_MOD)                         \
void  FUNC_NAME##_##TYPE_MOD##_uniform(TYPE *a, int32_t *b) {                               \
    printf (#FUNC_NAME "_" #TYPE_MOD ":");                                                  \
                                                                                            \
    TYPE copy_a[16];                                                                        \
    int32_t copy_b[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        copy_a[i] = a[i];                                                                   \
        copy_b[i] = b[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        __insert_element(&input_a, i, (TYPE)copy_a[i]);                                     \
                                                                                            \
    VEC_TYPE output;                                                                        \
    int err_counter = 0;                                                                    \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __##FUNC_NAME(input_a, copy_b[i]);                                         \
        for (uint32_t j = 0; j < 16; j++){                                                  \
            result = (TYPE) (a[j]) OP (b[i]);                                               \
            if (__extract_element(output, j) != result)                                     \
                err_counter++;                                                              \
        }                                                                                   \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}

//SHIFT_UNIFORM_TEST(uint8_t , __vec16_i8 , >>, lshr, ui8)
//SHIFT_UNIFORM_TEST(uint16_t, __vec16_i16, >>, lshr, ui16)
SHIFT_UNIFORM_TEST(uint32_t, __vec16_i32, >>, lshr, ui32)
//SHIFT_UNIFORM_TEST(uint64_t, __vec16_i64, >>, lshr, ui64)

//SHIFT_UNIFORM_TEST(int8_t , __vec16_i8 , >>, ashr, i8)
//SHIFT_UNIFORM_TEST(int16_t, __vec16_i16, >>, ashr, i16)
SHIFT_UNIFORM_TEST(int32_t, __vec16_i32, >>, ashr, i32)
//SHIFT_UNIFORM_TEST(int64_t, __vec16_i64, >>, ashr, i64)

/////////////////////////////////////////////////////////////////////////////////////////////

#define CMP(TYPE, VEC_TYPE, OP, FUNC_NAME)                                                  \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    VEC_TYPE input_b;                                                                       \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (i % 2 == 0)                                                                     \
            __insert_element(&input_a, i, 10);                                              \
        else                                                                                \
            __insert_element(&input_a, i, (TYPE)copy_data[i]);                              \
        __insert_element(&input_b, i, (TYPE)copy_data[i]);                                  \
    }                                                                                       \
                                                                                            \
    __vec16_i1 output;                                                                      \
    output = __##FUNC_NAME(input_a, input_b);                                               \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (i % 2 == 0 && !__extract_element(output, i) != !(10 OP data[i]))                \
            err_counter++;                                                                  \
        if (i % 2 != 0 &&  !__extract_element(output, i) != !(data[i] OP data[i]))          \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}                                                                                           \
                                                                                            \
void FUNC_NAME##_and_mask(TYPE *data, int *m) {                                             \
    printf (#FUNC_NAME "_and_mask" ":");                                                    \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    int copy_m[16];                                                                         \
    for (uint32_t i = 0; i < 16; i++) {                                                     \
        copy_data[i] = data[i];                                                             \
        copy_m[i] = m[i];                                                                   \
    }                                                                                       \
                                                                                            \
    VEC_TYPE input_a;                                                                       \
    VEC_TYPE input_b;                                                                       \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (i % 2 == 0)                                                                     \
            __insert_element(&input_a, i, 10);                                              \
        else                                                                                \
            __insert_element(&input_a, i, (TYPE)copy_data[i]);                              \
        __insert_element(&input_b, i, (TYPE)copy_data[i]);                                  \
    }                                                                                       \
                                                                                            \
    __vec16_i1 mask = __vec16_i1(copy_m[0],  copy_m[1],  copy_m[2],  copy_m[3],             \
                                 copy_m[4],  copy_m[5],  copy_m[6],  copy_m[7],             \
                                 copy_m[8],  copy_m[9],  copy_m[10], copy_m[11],            \
                                 copy_m[12], copy_m[13], copy_m[14], copy_m[15]);           \
                                                                                            \
    __vec16_i1 output;                                                                      \
    output = __##FUNC_NAME##_and_mask(input_a, input_b, mask);                              \
                                                                                            \
    int err_counter = 0;                                                                    \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        if (m[i] != 0 && i % 2 == 0 && !__extract_element(output, i) != !(10 OP data[i]))   \
            err_counter++;                                                                  \
        if (m[i]!=0 && i % 2 != 0 && !__extract_element(output,i) != !(data[i] OP data[i])) \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CMP(double , __vec16_d  , ==, equal_double)
CMP(float  , __vec16_f  , ==, equal_float )
//CMP(int8_t , __vec16_i8 , ==, equal_i8    )
//CMP(int16_t, __vec16_i16, ==, equal_i16   )
CMP(int32_t, __vec16_i32, ==, equal_i32   )
CMP(int64_t, __vec16_i64, ==, equal_i64   )

CMP(double , __vec16_d  ,  !=, not_equal_double)
CMP(float  , __vec16_f  ,  !=, not_equal_float )
//CMP(int8_t , __vec16_i8 ,  !=, not_equal_i8    )
//CMP(int16_t, __vec16_i16,  !=, not_equal_i16   )
CMP(int32_t, __vec16_i32,  !=, not_equal_i32   )
CMP(int64_t, __vec16_i64,  !=, not_equal_i64   )

//CMP(uint8_t , __vec16_i8 ,  <=, unsigned_less_equal_i8 )
//CMP(uint16_t, __vec16_i16,  <=, unsigned_less_equal_i16)
CMP(uint32_t, __vec16_i32,  <=, unsigned_less_equal_i32)
//CMP(uint64_t, __vec16_i64,  <=, unsigned_less_equal_i64)

//CMP(int8_t , __vec16_i8 ,  <=, signed_less_equal_i8 )
//CMP(int16_t, __vec16_i16,  <=, signed_less_equal_i16)
CMP(int32_t, __vec16_i32,  <=, signed_less_equal_i32)
//CMP(int64_t, __vec16_i64,  <=, signed_less_equal_i64)

CMP(double , __vec16_d  ,  <=, less_equal_double)
CMP(float  , __vec16_f  ,  <=, less_equal_float )

//CMP(uint8_t , __vec16_i8 ,  >=, unsigned_greater_equal_i8 )
//CMP(uint16_t, __vec16_i16,  >=, unsigned_greater_equal_i16)
CMP(uint32_t, __vec16_i32,  >=, unsigned_greater_equal_i32)
//CMP(uint64_t, __vec16_i64,  >=, unsigned_greater_equal_i64)

//CMP(int8_t , __vec16_i8 ,  >=, signed_greater_equal_i8 )
//CMP(int16_t, __vec16_i16,  >=, signed_greater_equal_i16)
CMP(int32_t, __vec16_i32,  >=, signed_greater_equal_i32)
//CMP(int64_t, __vec16_i64,  >=, signed_greater_equal_i64)

CMP(double , __vec16_d  ,  >=, greater_equal_double)
CMP(float  , __vec16_f  ,  >=, greater_equal_float )

//CMP(uint8_t , __vec16_i8 ,  <, unsigned_less_than_i8 )
//CMP(uint16_t, __vec16_i16,  <, unsigned_less_than_i16)
CMP(uint32_t, __vec16_i32,  <, unsigned_less_than_i32)
//CMP(uint64_t, __vec16_i64,  <, unsigned_less_than_i64)

//CMP(int8_t , __vec16_i8 ,  <, signed_less_than_i8 )
//CMP(int16_t, __vec16_i16,  <, signed_less_than_i16)
CMP(int32_t, __vec16_i32,  <, signed_less_than_i32)
//CMP(int64_t, __vec16_i64,  <, signed_less_than_i64)

CMP(double , __vec16_d  ,  <, less_than_double)
CMP(float  , __vec16_f  ,  <, less_than_float )

//CMP(uint8_t , __vec16_i8 ,  >, unsigned_greater_than_i8 )
//CMP(uint16_t, __vec16_i16,  >, unsigned_greater_than_i16)
CMP(uint32_t, __vec16_i32,  >, unsigned_greater_than_i32)
//CMP(uint64_t, __vec16_i64,  >, unsigned_greater_than_i64)

//CMP(int8_t , __vec16_i8 ,  >, signed_greater_than_i8 )
//CMP(int16_t, __vec16_i16,  >, signed_greater_than_i16)
CMP(int32_t, __vec16_i32,  >, signed_greater_than_i32)
//CMP(int64_t, __vec16_i64,  >, signed_greater_than_i64)

CMP(double , __vec16_d  ,  >, greater_than_double)
CMP(float  , __vec16_f  ,  >, greater_than_float )

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
        printf(" no fails\n");                                                              \
}

INSERT_EXTRACT_ELEMENT(double , __vec16_d  , insert_extract_element_double)
INSERT_EXTRACT_ELEMENT(float  , __vec16_f  , insert_extract_element_float )
//INSERT_EXTRACT_ELEMENT(int8_t , __vec16_i8 , insert_extract_element_i8    )
//INSERT_EXTRACT_ELEMENT(int16_t, __vec16_i16, insert_extract_element_i16   )
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
        printf("%d | %d | %d\n", i, __extract_element(output, i), data[i]);                   \
        if (__extract_element(output, i) != data[i])                                        \
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
LOAD(int64_t, __vec16_i64, load_i64   , 128)
/*
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
//STORE(int8_t , __vec16_i8 , store_i8    , 16)
//STORE(int16_t, __vec16_i16, store_i16   , 32)
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
//SMEAR_TEST(int8_t , __vec16_i8 , smear_i8    )
//SMEAR_TEST(int16_t, __vec16_i16, smear_i16   )
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
//SETZERO_TEST(__vec16_i8 , setzero_i8    )
//SETZERO_TEST(__vec16_i16, setzero_i16   )
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
}                                                                                           \
                                                                                            \
void FUNC_NAME##_cond(TYPE *data, int *m) {                                                 \
    printf (#FUNC_NAME "_cond" ":");                                                        \
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
            if (m[i] != 0 && __extract_element(output, j) != (TYPE)data[j])                 \
                err_counter++;                                                              \
            if (m[i] == 0 && __extract_element(output, j) != (TYPE)(data[j] * -1))          \
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
//SELECT_TEST(int8_t , __vec16_i8 , select_i8    )
//SELECT_TEST(int16_t, __vec16_i16, select_i16   )
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
        printf(" no fails\n");                                                              \
}

BROADCAST_TEST(double , __vec16_d  , broadcast_double)
BROADCAST_TEST(float  , __vec16_f  , broadcast_float )
//BROADCAST_TEST(int8_t , __vec16_i8 , broadcast_i8    )
//BROADCAST_TEST(int16_t, __vec16_i16, broadcast_i16   )
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
        printf(" no fails\n");                                                              \
}

ROTATE_TEST(double , __vec16_d  , rotate_double)
ROTATE_TEST(float  , __vec16_f  , rotate_float )
//ROTATE_TEST(int8_t , __vec16_i8 , rotate_i8    )
//ROTATE_TEST(int16_t, __vec16_i16, rotate_i16   )
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
        printf(" no fails\n");                                                              \
}

SHIFT_TEST(double , __vec16_d  , shift_double)
SHIFT_TEST(float  , __vec16_f  , shift_float )
//SHIFT_TEST(int8_t , __vec16_i8 , shift_i8    )
//SHIFT_TEST(int16_t, __vec16_i16, shift_i16   )
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
    /* 8|12|9|13|10|14|11|15|0|1|2|3|7|6|5|4                                              \
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
        printf(" no fails\n");                                                              \
}

SHUFFLE_TEST(double , __vec16_d  , shuffle_double)
SHUFFLE_TEST(float  , __vec16_f  , shuffle_float )
//SHUFFLE_TEST(int8_t , __vec16_i8 , shuffle_i8    )
//SHUFFLE_TEST(int16_t, __vec16_i16, shuffle_i16   )
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
        if (!(TO)__extract_element(output, i) != !(TO)data[i])                              \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CAST_TEST(int64_t, __vec16_i64, int32_t, __vec16_i32, cast_i64_i32, __cast_sext)
//CAST_TEST(int64_t, __vec16_i64, int16_t, __vec16_i16, cast_i64_i16, __cast_sext)
//CAST_TEST(int64_t, __vec16_i64, int8_t , __vec16_i8 , cast_i64_i8 , __cast_sext)
CAST_TEST(int64_t, __vec16_i64, bool   , __vec16_i1 , cast_i64_i1 , __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, int16_t, __vec16_i16, cast_i32_i16, __cast_sext)
//CAST_TEST(int32_t, __vec16_i32, int8_t , __vec16_i8 , cast_i32_i8 , __cast_sext)
CAST_TEST(int32_t, __vec16_i32, bool   , __vec16_i1 , cast_i32_i1 , __cast_sext)
//CAST_TEST(int16_t, __vec16_i16, int8_t , __vec16_i8 , cast_i16_i8 , __cast_sext)
//CAST_TEST(int16_t, __vec16_i16, bool   , __vec16_i1 , cast_i16_i1 , __cast_sext)
//CAST_TEST(int8_t , __vec16_i8 , bool   , __vec16_i1 , cast_i8_i1  , __cast_sext)

CAST_TEST(uint64_t, __vec16_i64, uint32_t, __vec16_i32, cast_ui64_ui32, __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, uint16_t, __vec16_i16, cast_ui64_ui16, __cast_zext)
//CAST_TEST(uint64_t, __vec16_i64, uint8_t , __vec16_i8 , cast_ui64_ui8 , __cast_zext)
CAST_TEST(uint64_t, __vec16_i64, bool    , __vec16_i1 , cast_ui64_ui1 , __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, uint16_t, __vec16_i16, cast_ui32_ui16, __cast_zext)
//CAST_TEST(uint32_t, __vec16_i32, uint8_t , __vec16_i8 , cast_ui32_ui8 , __cast_zext)
CAST_TEST(uint32_t, __vec16_i32, bool    , __vec16_i1 , cast_ui32_ui1 , __cast_zext)
//CAST_TEST(uint16_t, __vec16_i16, uint8_t , __vec16_i8 , cast_ui16_ui8 , __cast_zext)
//CAST_TEST(uint16_t, __vec16_i16, bool    , __vec16_i1 , cast_ui16_ui1 , __cast_zext)
//CAST_TEST(uint8_t , __vec16_i8 , bool    , __vec16_i1 , cast_ui8_ui1  , __cast_zext)

CAST_TEST(int32_t, __vec16_i32, int64_t, __vec16_i64, trunk_i32_i64, __cast_trunc)
//CAST_TEST(int16_t, __vec16_i16, int64_t, __vec16_i64, trunk_i16_i64, __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int64_t, __vec16_i64, trunk_i8_i64 , __cast_trunc)
//CAST_TEST(int16_t, __vec16_i16, int32_t, __vec16_i32, trunk_i16_i32, __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int32_t, __vec16_i32, trunk_i8_i32 , __cast_trunc)
//CAST_TEST(int8_t , __vec16_i8 , int16_t, __vec16_i16, trunk_i8_i16 , __cast_trunc)

//CAST_TEST(float , __vec16_f, int8_t , __vec16_i8,  cast_f_i8,  __cast_sitofp)
//CAST_TEST(float , __vec16_f, int16_t, __vec16_i16, cast_f_i16, __cast_sitofp)
CAST_TEST(float , __vec16_f, int32_t, __vec16_i32, cast_f_i32, __cast_sitofp)
CAST_TEST(float , __vec16_f, int64_t, __vec16_i64, cast_f_i64, __cast_sitofp)
//CAST_TEST(double, __vec16_d, int8_t , __vec16_i8 , cast_d_i8,  __cast_sitofp)
//CAST_TEST(double, __vec16_d, int16_t, __vec16_i16, cast_d_i16, __cast_sitofp)
CAST_TEST(double, __vec16_d, int32_t, __vec16_i32, cast_d_i32, __cast_sitofp)
CAST_TEST(double, __vec16_d, int64_t, __vec16_i64, cast_d_i64, __cast_sitofp)

//CAST_TEST(float , __vec16_f, uint8_t , __vec16_i8,  cast_f_ui8,  __cast_uitofp)
//CAST_TEST(float , __vec16_f, uint16_t, __vec16_i16, cast_f_ui16, __cast_uitofp)
CAST_TEST(float , __vec16_f, uint32_t, __vec16_i32, cast_f_ui32, __cast_uitofp)
CAST_TEST(float , __vec16_f, uint64_t, __vec16_i64, cast_f_ui64, __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint8_t , __vec16_i8 , cast_d_ui8,  __cast_uitofp)
//CAST_TEST(double, __vec16_d, uint16_t, __vec16_i16, cast_d_ui16, __cast_uitofp)
CAST_TEST(double, __vec16_d, uint32_t, __vec16_i32, cast_d_ui32, __cast_uitofp)
CAST_TEST(double, __vec16_d, uint64_t, __vec16_i64, cast_d_ui64, __cast_uitofp)

//CAST_TEST(int8_t , __vec16_i8 , float , __vec16_f, cast_i8_f , __cast_fptosi)
//CAST_TEST(int16_t, __vec16_i16, float , __vec16_f, cast_i16_f, __cast_fptosi)
CAST_TEST(int32_t, __vec16_i32, float , __vec16_f, cast_i32_f, __cast_fptosi)
CAST_TEST(int64_t, __vec16_i64, float , __vec16_f, cast_i64_f, __cast_fptosi)
//CAST_TEST(int8_t , __vec16_i8 , double, __vec16_d, cast_i8_d , __cast_fptosi)
//CAST_TEST(int16_t, __vec16_i16, double, __vec16_d, cast_i16_d, __cast_fptosi)
CAST_TEST(int32_t, __vec16_i32, double, __vec16_d, cast_i32_d, __cast_fptosi)
CAST_TEST(int64_t, __vec16_i64, double, __vec16_d, cast_i64_d, __cast_fptosi)

//CAST_TEST(uint8_t , __vec16_i8 , float , __vec16_f, cast_ui8_f , __cast_fptoui)
//CAST_TEST(uint16_t, __vec16_i16, float , __vec16_f, cast_ui16_f, __cast_fptoui)
CAST_TEST(uint32_t, __vec16_i32, float , __vec16_f, cast_ui32_f, __cast_fptoui)
CAST_TEST(uint64_t, __vec16_i64, float , __vec16_f, cast_ui64_f, __cast_fptoui)
//CAST_TEST(uint8_t , __vec16_i8 , double, __vec16_d, cast_ui8_d , __cast_fptoui)
//CAST_TEST(uint16_t, __vec16_i16, double, __vec16_d, cast_ui16_d, __cast_fptoui)
CAST_TEST(uint32_t, __vec16_i32, double, __vec16_d, cast_ui32_d, __cast_fptoui)
CAST_TEST(uint64_t, __vec16_i64, double, __vec16_d, cast_ui64_d, __cast_fptoui)

CAST_TEST(float , __vec16_f, double, __vec16_d, cast_f_d, __cast_fptrunc)
CAST_TEST(double, __vec16_d, float , __vec16_f, cast_d_f, __cast_fpext)

/////////////////////////////////////////////////////////////////////////////////////////////
#define CAST_BITS_TEST(TO, TO_VEC, FROM, FROM_VEC, FUNC_NAME)                               \
void FUNC_NAME(FROM *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
          (__extract_element(output, i) != __extract_element(result, i)))                   \
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

/////////////////////////////////////////////////////////////////////////////////////////////

#define CAST_BITS_SCALAR_TEST(TO, FROM, FUNC_NAME)                                          \
void FUNC_NAME(FROM *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
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
        if (!isnan(output) && !isnan(u.to) && output != (TO) u.to)                          \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CAST_BITS_SCALAR_TEST(uint32_t, float   , cast_bits_scalar_ui32_f)
CAST_BITS_SCALAR_TEST(int32_t , float   , cast_bits_scalar_i32_f)
CAST_BITS_SCALAR_TEST(float   , uint32_t, cast_bits_scalar_f_ui32)
CAST_BITS_SCALAR_TEST(float   , int32_t , cast_bits_scalar_f_i32)
CAST_BITS_SCALAR_TEST(uint64_t, double  , cast_bits_scalar_ui64_d)
CAST_BITS_SCALAR_TEST(int64_t , double  , cast_bits_scalar_i64_d)
CAST_BITS_SCALAR_TEST(double  , uint64_t, cast_bits_scalar_d_ui64)
CAST_BITS_SCALAR_TEST(double  , int64_t , cast_bits_scalar_d_i64)

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
        printf(" no fails\n");                                                              \
                                                                                            \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        free(b[i]);                                                                         \
}

GATHER(int32_t, __vec16_i32, double , __vec16_d  , gather32_double)
GATHER(int32_t, __vec16_i32, float  , __vec16_f  , gather32_float )
//GATHER(int32_t, __vec16_i32, int8_t , __vec16_i8 , gather32_i8    )
//GATHER(int32_t, __vec16_i32, int16_t, __vec16_i16, gather32_i16   )
GATHER(int32_t, __vec16_i32, int32_t, __vec16_i32, gather32_i32   )
GATHER(int32_t, __vec16_i32, int64_t, __vec16_i64, gather32_i64   )

GATHER(int64_t, __vec16_i64, double , __vec16_d  , gather64_double)
GATHER(int64_t, __vec16_i64, float  , __vec16_f  , gather64_float )
GATHER(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather64_i8    )
//GATHER(int64_t, __vec16_i64, int16_t, __vec16_i16, gather64_i16   )
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
//GATHER_OFFSETS(int32_t, __vec16_i32, int8_t , __vec16_i8 , gather_base_offsets32_i8    )
//GATHER_OFFSETS(int32_t, __vec16_i32, int16_t, __vec16_i16, gather_base_offsets32_i16   )
GATHER_OFFSETS(int32_t, __vec16_i32, int32_t, __vec16_i32, gather_base_offsets32_i32   )
GATHER_OFFSETS(int32_t, __vec16_i32, int64_t, __vec16_i64, gather_base_offsets32_i64   )

GATHER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , gather_base_offsets64_double)
GATHER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , gather_base_offsets64_float )
//GATHER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather_base_offsets64_i8    )
//GATHER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, gather_base_offsets64_i16   )
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
//SCATTER(int32_t, __vec16_i32, int8_t , __vec16_i8 , scatter32_i8    )
//SCATTER(int32_t, __vec16_i32, int16_t, __vec16_i16, scatter32_i16   )
SCATTER(int32_t, __vec16_i32, int32_t, __vec16_i32, scatter32_i32   )
SCATTER(int32_t, __vec16_i32, int64_t, __vec16_i64, scatter32_i64   )

SCATTER(int64_t, __vec16_i64, double , __vec16_d  , scatter64_double)
SCATTER(int64_t, __vec16_i64, float  , __vec16_f  , scatter64_float )
//SCATTER(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter64_i8    )
//SCATTER(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter64_i16   )
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
//SCATTER_OFFSETS(int32_t, __vec16_i32, int8_t , __vec16_i8 , scatter_base_offsets32_i8    )
//SCATTER_OFFSETS(int32_t, __vec16_i32, int16_t, __vec16_i16, scatter_base_offsets32_i16   )
SCATTER_OFFSETS(int32_t, __vec16_i32, int32_t, __vec16_i32, scatter_base_offsets32_i32   )
SCATTER_OFFSETS(int32_t, __vec16_i32, int64_t, __vec16_i64, scatter_base_offsets32_i64   )

SCATTER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , scatter_base_offsets64_double)
SCATTER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , scatter_base_offsets64_float )
//SCATTER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter_base_offsets64_i8    )
//SCATTER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter_base_offsets64_i16   )
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
//MASKED_LOAD(int8_t , __vec16_i8 , masked_load_i8    )
//MASKED_LOAD(int16_t, __vec16_i16, masked_load_i16   )
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
//MASKED_STORE(int8_t , __vec16_i8 , masked_store_i8    )
//MASKED_STORE(int16_t, __vec16_i16, masked_store_i16   )
MASKED_STORE(int32_t, __vec16_i32, masked_store_i32   )
MASKED_STORE(int64_t, __vec16_i64, masked_store_i64   )

MASKED_STORE(double , __vec16_d  , masked_store_blend_double)
MASKED_STORE(float  , __vec16_f  , masked_store_blend_float )
//MASKED_STORE(int8_t , __vec16_i8 , masked_store_blend_i8    )
//MASKED_STORE(int16_t, __vec16_i16, masked_store_blend_i16   )
MASKED_STORE(int32_t, __vec16_i32, masked_store_blend_i32   )
MASKED_STORE(int64_t, __vec16_i64, masked_store_blend_i64   )

/////////////////////////////////////////////////////////////////////////////////////////////
#define REDUCE_ADD_TEST(TYPE, VEC_TYPE, FUNC_NAME)                                          \
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
    TYPE output;                                                                            \
    output = __##FUNC_NAME(input);                                                          \
                                                                                            \
    TYPE result = 0;                                                                        \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        result += (TYPE) data[i];                                                           \
                                                                                            \
    if ((TYPE) output != (TYPE) result)                                                     \
        printf(" errors 1\n");                                                              \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

REDUCE_ADD_TEST(double , __vec16_d  , reduce_add_double)
REDUCE_ADD_TEST(float  , __vec16_f  , reduce_add_float )
//REDUCE_ADD_TEST(int8_t , __vec16_i8 , reduce_add_int8  )
//REDUCE_ADD_TEST(int16_t, __vec16_i16, reduce_add_int16 )
REDUCE_ADD_TEST(int32_t, __vec16_i32, reduce_add_int32 )
REDUCE_ADD_TEST(int64_t, __vec16_i64, reduce_add_int64 )

/////////////////////////////////////////////////////////////////////////////////////////////
#define REDUCE_MINMAX_TEST(TYPE, VEC_TYPE, RES_NUM, FUNC_NAME)                              \
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
    TYPE output;                                                                            \
    output = __##FUNC_NAME(input);                                                          \
    if ((TYPE) output != (TYPE) data[RES_NUM])                                              \
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

/////////////////////////////////////////////////////////////////////////////////////////////
#define POPCNT_TEST(TYPE, FUNC_NAME)                                                        \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    int err_counter = 0;                                                                    \
    int32_t output = 0;                                                                     \
    int32_t result = 0;                                                                     \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __##FUNC_NAME(copy_data[i]);                                               \
        result = 0;                                                                         \
        for (result = 0; copy_data[i] != 0; result++)                                       \
             copy_data[i] &= copy_data[i] - 1;                                              \
        if (output != result)                                                               \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

POPCNT_TEST(uint32_t, popcnt_int32)
POPCNT_TEST(uint64_t, popcnt_int64)

/////////////////////////////////////////////////////////////////////////////////////////////
#define COUNT_TRAILING_ZEROS(TYPE, BIT_NUM, FUNC_NAME)                                      \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    int err_counter = 0;                                                                    \
    int32_t output = 0;                                                                     \
    int32_t result = 0;                                                                     \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __##FUNC_NAME(copy_data[i]);                                               \
        TYPE mask = 1;                                                                      \
        for (TYPE j = 0; j < BIT_NUM; j++, mask <<= 1)                                      \
            if ((data[i] & mask) != 0){                                                     \
                result = j;                                                                 \
                break;                                                                      \
            }                                                                               \
        if (data[i] == 0)                                                                   \
            result = BIT_NUM;                                                               \
        if (output != result)                                                               \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

COUNT_TRAILING_ZEROS(uint32_t, 32, count_trailing_zeros_i32)
COUNT_TRAILING_ZEROS(uint64_t, 64, count_trailing_zeros_i64)

/////////////////////////////////////////////////////////////////////////////////////////////
#define COUNT_LEADING_ZEROS(TYPE, BIT_NUM, FUNC_NAME)                                       \
void FUNC_NAME(TYPE *data) {                                                                \
    printf (#FUNC_NAME, ":");                                                               \
                                                                                            \
    TYPE copy_data[16];                                                                     \
    for (uint32_t i = 0; i < 16; i++)                                                       \
        copy_data[i] = data[i];                                                             \
                                                                                            \
    int err_counter = 0;                                                                    \
    int32_t output = 0;                                                                     \
    int32_t result = 0;                                                                     \
    for (uint32_t i = 0; i < 16; i++){                                                      \
        output = __count_leading_zeros_i32(copy_data[i]);                                   \
        TYPE mask = 1 << 31;                                                                \
        for (uint32_t j = 0; j < 32; j++, mask >>= 1)                                       \
            if ((data[i] & mask) != 0){                                                     \
                result = j;                                                                 \
                break;                                                                      \
            }                                                                               \
        if (data[i] == 0)                                                                   \
            result = 32;                                                                    \
        if (output != result)                                                               \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

COUNT_LEADING_ZEROS(uint32_t, 32, count_leading_zeros_i32)
COUNT_LEADING_ZEROS(uint64_t, 64, count_leading_zeros_i64)

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
*/
