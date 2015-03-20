#include "knc_test_driver_core.h"

void equal_double(double  *d  );
void equal_float (float   *f  );
void equal_i8    (int8_t  *i8 );
void equal_i16   (int16_t *i16);
void equal_i32   (int32_t *i32);
void equal_i64   (int64_t *i64);

void not_equal_double(double  *d  );
void not_equal_float (float   *f  );
void not_equal_i8    (int8_t  *i8 );
void not_equal_i16   (int16_t *i16);
void not_equal_i32   (int32_t *i32);
void not_equal_i64   (int64_t *i64);

void unsigned_less_equal_i8  (uint8_t  *ui8 );
void unsigned_less_equal_i16 (uint16_t *ui16);
void unsigned_less_equal_i32 (uint32_t *ui32);
void unsigned_less_equal_i64 (uint64_t *ui64);

void signed_less_equal_i8  (int8_t  *i8 );
void signed_less_equal_i16 (int16_t *i16);
void signed_less_equal_i32 (int32_t *i32);
void signed_less_equal_i64 (int64_t *i64);

void less_equal_double(double *d);
void less_equal_float (float  *f);

void unsigned_greater_equal_i8  (uint8_t  *ui8 );
void unsigned_greater_equal_i16 (uint16_t *ui16);
void unsigned_greater_equal_i32 (uint32_t *ui32);
void unsigned_greater_equal_i64 (uint64_t *ui64);

void signed_greater_equal_i8  (int8_t  *i8 );
void signed_greater_equal_i16 (int16_t *i16);
void signed_greater_equal_i32 (int32_t *i32);
void signed_greater_equal_i64 (int64_t *i64);

void greater_equal_double(double *d);
void greater_equal_float (float  *f);

void unsigned_less_than_i8  (uint8_t  *ui8 );
void unsigned_less_than_i16 (uint16_t *ui16);
void unsigned_less_than_i32 (uint32_t *ui32);
void unsigned_less_than_i64 (uint64_t *ui64);

void signed_less_than_i8  (int8_t  *i8 );
void signed_less_than_i16 (int16_t *i16);
void signed_less_than_i32 (int32_t *i32);
void signed_less_than_i64 (int64_t *i64);

void less_than_double(double *d);
void less_than_float (float  *f);

void unsigned_greater_than_i8  (uint8_t  *ui8 );
void unsigned_greater_than_i16 (uint16_t *ui16);
void unsigned_greater_than_i32 (uint32_t *ui32);
void unsigned_greater_than_i64 (uint64_t *ui64);

void signed_greater_than_i8  (int8_t  *i8 );
void signed_greater_than_i16 (int16_t *i16);
void signed_greater_than_i32 (int32_t *i32);
void signed_greater_than_i64 (int64_t *i64);

void greater_than_double(double *d);
void greater_than_float (float  *f);

void equal_double_and_mask(double  *d  , int *mask);
void equal_float_and_mask (float   *f  , int *mask);
void equal_i8_and_mask    (int8_t  *i8 , int *mask);
void equal_i16_and_mask   (int16_t *i16, int *mask);
void equal_i32_and_mask   (int32_t *i32, int *mask);
void equal_i64_and_mask   (int64_t *i64, int *mask);

void not_equal_double_and_mask(double  *d  , int *mask);
void not_equal_float_and_mask (float   *f  , int *mask);
void not_equal_i8_and_mask    (int8_t  *i8 , int *mask);
void not_equal_i16_and_mask   (int16_t *i16, int *mask);
void not_equal_i32_and_mask   (int32_t *i32, int *mask);
void not_equal_i64_and_mask   (int64_t *i64, int *mask);

void unsigned_less_equal_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_less_equal_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_less_equal_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_less_equal_i64_and_mask (uint64_t *ui64, int *mask);

void signed_less_equal_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_less_equal_i16_and_mask (int16_t *i16, int *mask);
void signed_less_equal_i32_and_mask (int32_t *i32, int *mask);
void signed_less_equal_i64_and_mask (int64_t *i64, int *mask);

void less_equal_double_and_mask(double *d, int *mask);
void less_equal_float_and_mask (float  *f, int *mask);

void unsigned_greater_equal_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_greater_equal_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_greater_equal_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_greater_equal_i64_and_mask (uint64_t *ui64, int *mask);

void signed_greater_equal_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_greater_equal_i16_and_mask (int16_t *i16, int *mask);
void signed_greater_equal_i32_and_mask (int32_t *i32, int *mask);
void signed_greater_equal_i64_and_mask (int64_t *i64, int *mask);

void greater_equal_double_and_mask(double *d, int *mask);
void greater_equal_float_and_mask (float  *f, int *mask);

void unsigned_less_than_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_less_than_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_less_than_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_less_than_i64_and_mask (uint64_t *ui64, int *mask);

void signed_less_than_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_less_than_i16_and_mask (int16_t *i16, int *mask);
void signed_less_than_i32_and_mask (int32_t *i32, int *mask);
void signed_less_than_i64_and_mask (int64_t *i64, int *mask);

void less_than_double_and_mask(double *d, int *mask);
void less_than_float_and_mask (float  *f, int *mask);

void unsigned_greater_than_i8_and_mask  (uint8_t  *ui8 , int *mask);
void unsigned_greater_than_i16_and_mask (uint16_t *ui16, int *mask);
void unsigned_greater_than_i32_and_mask (uint32_t *ui32, int *mask);
void unsigned_greater_than_i64_and_mask (uint64_t *ui64, int *mask);

void signed_greater_than_i8_and_mask  (int8_t  *i8 , int *mask);
void signed_greater_than_i16_and_mask (int16_t *i16, int *mask);
void signed_greater_than_i32_and_mask (int32_t *i32, int *mask);
void signed_greater_than_i64_and_mask (int64_t *i64, int *mask);

void greater_than_double_and_mask(double *d, int *mask);
void greater_than_float_and_mask (float  *f, int *mask);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_cmp() {
    InputData inpData;

    equal_double(inpData.d_32);
    equal_float (inpData.f_32);
    equal_i8    (inpData.i8);
    equal_i16   (inpData.i16);
    equal_i32   (inpData.i32);
    equal_i64   (inpData.i64);

    not_equal_double(inpData.d_32);
    not_equal_float (inpData.f_32);
    not_equal_i8    (inpData.i8);
    not_equal_i16   (inpData.i16);
    not_equal_i32   (inpData.i32);
    not_equal_i64   (inpData.i64);

    //unsigned_less_equal_i8  (inpData.ui8);
    //unsigned_less_equal_i16 (inpData.ui16);
    unsigned_less_equal_i32 (inpData.ui32);
    unsigned_less_equal_i64 (inpData.ui64);

    //signed_less_equal_i8  (inpData.i8);
    //signed_less_equal_i16 (inpData.i16);
    signed_less_equal_i32 (inpData.i32);
    signed_less_equal_i64 (inpData.i64);

    less_equal_double(inpData.d_32);
    less_equal_float (inpData.f_32);

    //unsigned_greater_equal_i8  (inpData.ui8);
    //unsigned_greater_equal_i16 (inpData.ui16);
    unsigned_greater_equal_i32 (inpData.ui32);
    unsigned_greater_equal_i64 (inpData.ui64);

    //signed_greater_equal_i8  (inpData.i8);
    //signed_greater_equal_i16 (inpData.i16);
    signed_greater_equal_i32 (inpData.i32);
    signed_greater_equal_i64 (inpData.i64);

    greater_equal_double(inpData.d_32);
    greater_equal_float (inpData.f_32);

    //unsigned_less_than_i8  (inpData.ui8);
    //unsigned_less_than_i16 (inpData.ui16);
    unsigned_less_than_i32 (inpData.ui32);
    unsigned_less_than_i64 (inpData.ui64);

    //signed_less_than_i8  (inpData.i8);
    //signed_less_than_i16 (inpData.i16);
    signed_less_than_i32 (inpData.i32);
    signed_less_than_i64 (inpData.i64);

    less_than_double(inpData.d_32);
    less_than_float (inpData.f_32);

    //unsigned_greater_than_i8  (inpData.ui8);
    //unsigned_greater_than_i16 (inpData.ui16);
    unsigned_greater_than_i32 (inpData.ui32);
    unsigned_greater_than_i64 (inpData.ui64);

    //signed_greater_than_i8  (inpData.i8);
    //signed_greater_than_i16 (inpData.i16);
    signed_greater_than_i32 (inpData.i32);
    signed_greater_than_i64 (inpData.i64);

    greater_than_double(inpData.d_32);
    greater_than_float (inpData.f_32);

    equal_double_and_mask(inpData.d_32, inpData.mask);
    equal_float_and_mask (inpData.f_32, inpData.mask);
    equal_i8_and_mask    (inpData.i8, inpData.mask);
    equal_i16_and_mask   (inpData.i16, inpData.mask);
    equal_i32_and_mask   (inpData.i32, inpData.mask);
    equal_i64_and_mask   (inpData.i64, inpData.mask);

    not_equal_double_and_mask(inpData.d_32, inpData.mask);
    not_equal_float_and_mask (inpData.f_32, inpData.mask);
    not_equal_i8_and_mask    (inpData.i8, inpData.mask);
    not_equal_i16_and_mask   (inpData.i16, inpData.mask);
    not_equal_i32_and_mask   (inpData.i32, inpData.mask);
    not_equal_i64_and_mask   (inpData.i64, inpData.mask);

    //unsigned_less_equal_i8_and_mask  (inpData.ui8, inpData.mask);
    //unsigned_less_equal_i16_and_mask (inpData.ui16, inpData.mask);
    unsigned_less_equal_i32_and_mask (inpData.ui32, inpData.mask);
    unsigned_less_equal_i64_and_mask (inpData.ui64, inpData.mask);

    //signed_less_equal_i8_and_mask  (inpData.i8, inpData.mask);
    //signed_less_equal_i16_and_mask (inpData.i16, inpData.mask);
    signed_less_equal_i32_and_mask (inpData.i32, inpData.mask);
    signed_less_equal_i64_and_mask (inpData.i64, inpData.mask);

    less_equal_double_and_mask(inpData.d_32, inpData.mask);
    less_equal_float_and_mask (inpData.f_32, inpData.mask);

    //unsigned_greater_equal_i8_and_mask  (inpData.ui8, inpData.mask);
    //unsigned_greater_equal_i16_and_mask (inpData.ui16, inpData.mask);
    unsigned_greater_equal_i32_and_mask (inpData.ui32, inpData.mask);
    unsigned_greater_equal_i64_and_mask (inpData.ui64, inpData.mask);

    //signed_greater_equal_i8_and_mask  (inpData.i8, inpData.mask);
    //signed_greater_equal_i16_and_mask (inpData.i16, inpData.mask);
    signed_greater_equal_i32_and_mask (inpData.i32, inpData.mask);
    signed_greater_equal_i64_and_mask (inpData.i64, inpData.mask);

    greater_equal_double_and_mask(inpData.d_32, inpData.mask);
    greater_equal_float_and_mask (inpData.f_32, inpData.mask);

    //unsigned_less_than_i8_and_mask  (inpData.ui8, inpData.mask);
    //unsigned_less_than_i16_and_mask (inpData.ui16, inpData.mask);
    unsigned_less_than_i32_and_mask (inpData.ui32, inpData.mask);
    unsigned_less_than_i64_and_mask (inpData.ui64, inpData.mask);

    //signed_less_than_i8_and_mask  (inpData.i8, inpData.mask);
    //signed_less_than_i16_and_mask (inpData.i16, inpData.mask);
    signed_less_than_i32_and_mask (inpData.i32, inpData.mask);
    signed_less_than_i64_and_mask (inpData.i64, inpData.mask);

    less_than_double_and_mask(inpData.d_32, inpData.mask);
    less_than_float_and_mask (inpData.f_32, inpData.mask);

    //unsigned_greater_than_i8_and_mask  (inpData.ui8, inpData.mask);
    //unsigned_greater_than_i16_and_mask (inpData.ui16, inpData.mask);
    unsigned_greater_than_i32_and_mask (inpData.ui32, inpData.mask);
    unsigned_greater_than_i64_and_mask (inpData.ui64, inpData.mask);

    //signed_greater_than_i8_and_mask  (inpData.i8, inpData.mask);
    //signed_greater_than_i16_and_mask (inpData.i16, inpData.mask);
    signed_greater_than_i32_and_mask (inpData.i32, inpData.mask);
    signed_greater_than_i64_and_mask (inpData.i64, inpData.mask);

    greater_than_double_and_mask(inpData.d_32, inpData.mask);
    greater_than_float_and_mask (inpData.f_32, inpData.mask);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define CMP(TYPE, VEC_TYPE, OP, FUNC_NAME)                                                  \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
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
        if (i % 2 == 0 && check_and_print(!__extract_element(output, i), !(10 OP data[i]),  \
                                                                         err_counter))      \
            err_counter++;                                                                  \
        if (i % 2 != 0 && check_and_print(!__extract_element(output, i),                    \
                                          !(data[i] OP data[i]), err_counter))              \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
         printf(" no fails\n");                                                             \
}                                                                                           \
                                                                                            \
void FUNC_NAME##_and_mask(TYPE *data, int *m) {                                             \
    printf ("%-40s", #FUNC_NAME "_and_mask" ":");                                           \
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
        if (m[i] != 0 && i % 2 == 0 && check_and_print(!__extract_element(output, i),       \
                                                       !(10 OP data[i]), err_counter))      \
            err_counter++;                                                                  \
        if (m[i]!=0 && i % 2 != 0 && check_and_print(!__extract_element(output, i),         \
                                                     !(data[i] OP data[i]), err_counter))   \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

CMP(double , __vec16_d  , ==, equal_double)
CMP(float  , __vec16_f  , ==, equal_float )
CMP(int8_t , __vec16_i8 , ==, equal_i8    )
CMP(int16_t, __vec16_i16, ==, equal_i16   )
CMP(int32_t, __vec16_i32, ==, equal_i32   )
CMP(int64_t, __vec16_i64, ==, equal_i64   )

CMP(double , __vec16_d  ,  !=, not_equal_double)
CMP(float  , __vec16_f  ,  !=, not_equal_float )
CMP(int8_t , __vec16_i8 ,  !=, not_equal_i8    )
CMP(int16_t, __vec16_i16,  !=, not_equal_i16   )
CMP(int32_t, __vec16_i32,  !=, not_equal_i32   )
CMP(int64_t, __vec16_i64,  !=, not_equal_i64   )

//CMP(uint8_t , __vec16_i8 ,  <=, unsigned_less_equal_i8 )
//CMP(uint16_t, __vec16_i16,  <=, unsigned_less_equal_i16)
CMP(uint32_t, __vec16_i32,  <=, unsigned_less_equal_i32)
CMP(uint64_t, __vec16_i64,  <=, unsigned_less_equal_i64)

//CMP(int8_t , __vec16_i8 ,  <=, signed_less_equal_i8 )
//CMP(int16_t, __vec16_i16,  <=, signed_less_equal_i16)
CMP(int32_t, __vec16_i32,  <=, signed_less_equal_i32)
CMP(int64_t, __vec16_i64,  <=, signed_less_equal_i64)

CMP(double , __vec16_d  ,  <=, less_equal_double)
CMP(float  , __vec16_f  ,  <=, less_equal_float )

//CMP(uint8_t , __vec16_i8 ,  >=, unsigned_greater_equal_i8 )
//CMP(uint16_t, __vec16_i16,  >=, unsigned_greater_equal_i16)
CMP(uint32_t, __vec16_i32,  >=, unsigned_greater_equal_i32)
CMP(uint64_t, __vec16_i64,  >=, unsigned_greater_equal_i64)

//CMP(int8_t , __vec16_i8 ,  >=, signed_greater_equal_i8 )
//CMP(int16_t, __vec16_i16,  >=, signed_greater_equal_i16)
CMP(int32_t, __vec16_i32,  >=, signed_greater_equal_i32)
CMP(int64_t, __vec16_i64,  >=, signed_greater_equal_i64)

CMP(double , __vec16_d  ,  >=, greater_equal_double)
CMP(float  , __vec16_f  ,  >=, greater_equal_float )

//CMP(uint8_t , __vec16_i8 ,  <, unsigned_less_than_i8 )
//CMP(uint16_t, __vec16_i16,  <, unsigned_less_than_i16)
CMP(uint32_t, __vec16_i32,  <, unsigned_less_than_i32)
CMP(uint64_t, __vec16_i64,  <, unsigned_less_than_i64)

//CMP(int8_t , __vec16_i8 ,  <, signed_less_than_i8 )
//CMP(int16_t, __vec16_i16,  <, signed_less_than_i16)
CMP(int32_t, __vec16_i32,  <, signed_less_than_i32)
CMP(int64_t, __vec16_i64,  <, signed_less_than_i64)

CMP(double , __vec16_d  ,  <, less_than_double)
CMP(float  , __vec16_f  ,  <, less_than_float )

//CMP(uint8_t , __vec16_i8 ,  >, unsigned_greater_than_i8 )
//CMP(uint16_t, __vec16_i16,  >, unsigned_greater_than_i16)
CMP(uint32_t, __vec16_i32,  >, unsigned_greater_than_i32)
CMP(uint64_t, __vec16_i64,  >, unsigned_greater_than_i64)

//CMP(int8_t , __vec16_i8 ,  >, signed_greater_than_i8 )
//CMP(int16_t, __vec16_i16,  >, signed_greater_than_i16)
CMP(int32_t, __vec16_i32,  >, signed_greater_than_i32)
CMP(int64_t, __vec16_i64,  >, signed_greater_than_i64)

CMP(double , __vec16_d  ,  >, greater_than_double)
CMP(float  , __vec16_f  ,  >, greater_than_float )
