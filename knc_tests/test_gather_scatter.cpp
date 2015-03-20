#include "knc_test_driver_core.h"

void gather32_double(double  *d  , int *mask);
void gather32_float (float   *f  , int *mask);
void gather32_i8    (int8_t  *i8 , int *mask);
void gather32_i16   (int16_t *i16, int *mask);
void gather32_i32   (int32_t *i32, int *mask);
void gather32_i64   (int64_t *i64, int *mask);

void gather64_double(double  *d  , int *mask);
void gather64_float (float   *f  , int *mask);
void gather64_i8    (int8_t  *i8 , int *mask);
void gather64_i16   (int16_t *i16, int *mask);
void gather64_i32   (int32_t *i32, int *mask);
void gather64_i64   (int64_t *i64, int *mask);


void gather_base_offsets32_double(double  *d  , int *mask);
void gather_base_offsets32_float (float   *f  , int *mask);
void gather_base_offsets32_i8    (int8_t  *i8 , int *mask);
void gather_base_offsets32_i16   (int16_t *i16, int *mask);
void gather_base_offsets32_i32   (int32_t *i32, int *mask);
void gather_base_offsets32_i64   (int64_t *i64, int *mask);

void gather_base_offsets64_double(double  *d  , int *mask);
void gather_base_offsets64_float (float   *f  , int *mask);
void gather_base_offsets64_i8    (int8_t  *i8 , int *mask);
void gather_base_offsets64_i16   (int16_t *i16, int *mask);
void gather_base_offsets64_i32   (int32_t *i32, int *mask);
void gather_base_offsets64_i64   (int64_t *i64, int *mask);


void scatter32_double(double  *d  , int *mask);
void scatter32_float (float   *f  , int *mask);
void scatter32_i8    (int8_t  *i8 , int *mask);
void scatter32_i16   (int16_t *i16, int *mask);
void scatter32_i32   (int32_t *i32, int *mask);
void scatter32_i64   (int64_t *i64, int *mask);

void scatter64_double(double  *d  , int *mask);
void scatter64_float (float   *f  , int *mask);
void scatter64_i8    (int8_t  *i8 , int *mask);
void scatter64_i16   (int16_t *i16, int *mask);
void scatter64_i32   (int32_t *i32, int *mask);
void scatter64_i64   (int64_t *i64, int *mask);


void scatter_base_offsets32_double(double  *d  , int *mask);
void scatter_base_offsets32_float (float   *f  , int *mask);
void scatter_base_offsets32_i8    (int8_t  *i8 , int *mask);
void scatter_base_offsets32_i16   (int16_t *i16, int *mask);
void scatter_base_offsets32_i32   (int32_t *i32, int *mask);
void scatter_base_offsets32_i64   (int64_t *i64, int *mask);

void scatter_base_offsets64_double(double  *d  , int *mask);
void scatter_base_offsets64_float (float   *f  , int *mask);
void scatter_base_offsets64_i8    (int8_t  *i8 , int *mask);
void scatter_base_offsets64_i16   (int16_t *i16, int *mask);
void scatter_base_offsets64_i32   (int32_t *i32, int *mask);
void scatter_base_offsets64_i64   (int64_t *i64, int *mask);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_gather_scatter () {
    InputData inpData;

    //gather32_double(inpData.d_32, inpData.mask);
    //gather32_float(inpData.f_32, inpData.mask);
    //gather32_i8(inpData.i8, inpData.mask);
    //gather32_i16(inpData.i16, inpData.mask);
    //gather32_i32(inpData.i32, inpData.mask);
    //gather32_i64(inpData.i64, inpData.mask);

    gather64_double(inpData.d_64, inpData.mask);
    gather64_float(inpData.f_64, inpData.mask);
    gather64_i8(inpData.i8, inpData.mask);
    gather64_i16(inpData.i16, inpData.mask);
    gather64_i32(inpData.i32, inpData.mask);
    gather64_i64(inpData.i64, inpData.mask);


    gather_base_offsets32_double(inpData.d_32, inpData.mask);
    gather_base_offsets32_float(inpData.f_32, inpData.mask);
    gather_base_offsets32_i8(inpData.i8, inpData.mask);
    gather_base_offsets32_i16(inpData.i16, inpData.mask);
    gather_base_offsets32_i32(inpData.i32, inpData.mask);
    gather_base_offsets32_i64(inpData.i64, inpData.mask);

    //gather_base_offsets64_double(inpData.d_64, inpData.mask);
    gather_base_offsets64_float(inpData.f_64, inpData.mask);
    gather_base_offsets64_i8(inpData.i8, inpData.mask);
    gather_base_offsets64_i16(inpData.i16, inpData.mask);
    gather_base_offsets64_i32(inpData.i32, inpData.mask);
    //gather_base_offsets64_i64(inpData.i64, inpData.mask);


    //scatter32_double(inpData.d_32, inpData.mask);
    //scatter32_float(inpData.f_32, inpData.mask);
    //scatter32_i8(inpData.i8, inpData.mask);
    //scatter32_i16(inpData.i16, inpData.mask);
    //scatter32_i32(inpData.i32, inpData.mask);
    //scatter32_i64(inpData.i64, inpData.mask);

    //scatter64_double(inpData.d_64, inpData.mask);
    scatter64_float(inpData.f_64, inpData.mask);
    //scatter64_i8(inpData.i8, inpData.mask);
    //scatter64_i16(inpData.i16, inpData.mask);
    scatter64_i32(inpData.i32, inpData.mask);
    scatter64_i64(inpData.i64, inpData.mask);


    scatter_base_offsets32_double(inpData.d_32, inpData.mask);
    scatter_base_offsets32_float(inpData.f_32, inpData.mask);
    scatter_base_offsets32_i8(inpData.i8, inpData.mask);
    scatter_base_offsets32_i16(inpData.i16, inpData.mask);
    scatter_base_offsets32_i32(inpData.i32, inpData.mask);
    scatter_base_offsets32_i64(inpData.i64, inpData.mask);

    //scatter_base_offsets64_double(inpData.d_64, inpData.mask);
    scatter_base_offsets64_float(inpData.f_64, inpData.mask);
    scatter_base_offsets64_i8(inpData.i8, inpData.mask);
    //scatter_base_offsets64_i16(inpData.i16, inpData.mask);
    scatter_base_offsets64_i32(inpData.i32, inpData.mask);
    scatter_base_offsets64_i64(inpData.i64, inpData.mask);
}

/////////////////////////////////////////////////////////////////////////////////////////////

#define GATHER(GATHER_SCALAR_TYPE, GATHER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)              \
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
        if (m[i] != 0 &&                                                                    \
            check_and_print(__extract_element(output, i), data[i], err_counter))            \
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

//GATHER(int32_t, __vec16_i32, double , __vec16_d  , gather32_double)
//GATHER(int32_t, __vec16_i32, float  , __vec16_f  , gather32_float )
//GATHER(int32_t, __vec16_i32, int8_t , __vec16_i8 , gather32_i8    )
//GATHER(int32_t, __vec16_i32, int16_t, __vec16_i16, gather32_i16   )
//GATHER(int32_t, __vec16_i32, int32_t, __vec16_i32, gather32_i32   )
//GATHER(int32_t, __vec16_i32, int64_t, __vec16_i64, gather32_i64   )

GATHER(int64_t, __vec16_i64, double , __vec16_d  , gather64_double)
GATHER(int64_t, __vec16_i64, float  , __vec16_f  , gather64_float )
GATHER(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather64_i8    )
GATHER(int64_t, __vec16_i64, int16_t, __vec16_i16, gather64_i16   )
GATHER(int64_t, __vec16_i64, int32_t, __vec16_i32, gather64_i32   )
GATHER(int64_t, __vec16_i64, int64_t, __vec16_i64, gather64_i64   )

#define GATHER_OFFSETS(GATHER_SCALAR_TYPE, GATHER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)      \
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
        if (m[i] != 0 &&                                                                    \
            check_and_print(__extract_element(output, i), data[i], err_counter))            \
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

//GATHER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , gather_base_offsets64_double)
GATHER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , gather_base_offsets64_float )
GATHER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , gather_base_offsets64_i8    )
GATHER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, gather_base_offsets64_i16   )
GATHER_OFFSETS(int64_t, __vec16_i64, int32_t, __vec16_i32, gather_base_offsets64_i32   )
//GATHER_OFFSETS(int64_t, __vec16_i64, int64_t, __vec16_i64, gather_base_offsets64_i64   )

#define SCATTER(SCATTER_SCALAR_TYPE, SCATTER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)           \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
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
        if (m[i] != 0 &&                                                                    \
            check_and_print(*p, data[i], err_counter))                                      \
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

//SCATTER(int32_t, __vec16_i32, double , __vec16_d  , scatter32_double)
//SCATTER(int32_t, __vec16_i32, float  , __vec16_f  , scatter32_float )
//SCATTER(int32_t, __vec16_i32, int8_t , __vec16_i8 , scatter32_i8    )
//SCATTER(int32_t, __vec16_i32, int16_t, __vec16_i16, scatter32_i16   )
//SCATTER(int32_t, __vec16_i32, int32_t, __vec16_i32, scatter32_i32   )
//SCATTER(int32_t, __vec16_i32, int64_t, __vec16_i64, scatter32_i64   )

//SCATTER(int64_t, __vec16_i64, double , __vec16_d  , scatter64_double)
SCATTER(int64_t, __vec16_i64, float  , __vec16_f  , scatter64_float )
//SCATTER(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter64_i8    )
//SCATTER(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter64_i16   )
SCATTER(int64_t, __vec16_i64, int32_t, __vec16_i32, scatter64_i32   )
SCATTER(int64_t, __vec16_i64, int64_t, __vec16_i64, scatter64_i64   )

#define SCATTER_OFFSETS(SCATTER_SCALAR_TYPE, SCATTER_VEC_TYPE, TYPE, VEC_TYPE, FUNC_NAME)   \
void FUNC_NAME(TYPE *data, int *m) {                                                        \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
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
        if (m[i] != 0 &&                                                                    \
            check_and_print(*b[i], data[i], err_counter))                                   \
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

//SCATTER_OFFSETS(int64_t, __vec16_i64, double , __vec16_d  , scatter_base_offsets64_double)
SCATTER_OFFSETS(int64_t, __vec16_i64, float  , __vec16_f  , scatter_base_offsets64_float )
SCATTER_OFFSETS(int64_t, __vec16_i64, int8_t , __vec16_i8 , scatter_base_offsets64_i8    )
//SCATTER_OFFSETS(int64_t, __vec16_i64, int16_t, __vec16_i16, scatter_base_offsets64_i16   )
SCATTER_OFFSETS(int64_t, __vec16_i64, int32_t, __vec16_i32, scatter_base_offsets64_i32   )
SCATTER_OFFSETS(int64_t, __vec16_i64, int64_t, __vec16_i64, scatter_base_offsets64_i64   )
