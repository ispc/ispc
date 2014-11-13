#ifndef __DATA_CNTRL_H__
#define __DATA_CNTRL_H__

#define __STDC_LIMIT_MACROS // enable intN_t limits from stdint.h
#include <stdint.h>
#include <limits>

#include "knc.h"

void test_gather_scatter();
void test_masked_load_store();
void test_insert_extract();
void test_load_store();
void test_smear();
void test_setzero();
void test_select();
void test_broadcast();
void test_rotate();
void test_shift();
void test_shuffle();
void test_cast();
void test_binary_op();
void test_cmp();
void test_cast_bits();
void test_reduce();
void test_popcnt();
void test_count_zeros();
void test_other();

struct InputData {
    int      mask [16];
    double   d_8  [16];
    double   d_16 [16];
    double   d_32 [16];
    double   d_64 [16];
    float    f_8  [16];
    float    f_16 [16];
    float    f_32 [16];
    float    f_64 [16];
    double   d_u8 [16];
    double   d_u16[16];
    double   d_u32[16];
    double   d_u64[16];
    float    f_u8 [16];
    float    f_u16[16];
    float    f_u32[16];
    float    f_u64[16];
    bool     i1   [16];
    int8_t   i8   [16];
    int16_t  i16  [16];
    int32_t  i32  [16];
    int64_t  i64  [16];
    uint8_t  ui8  [16];
    uint16_t ui16 [16];
    uint32_t ui32 [16];
    uint64_t ui64 [16];

// variables, which can't cause overflow
    double   no_of_d_8  [16];
    double   no_of_d_16 [16];
    double   no_of_d_32 [16];
    double   no_of_d_64 [16];
    float    no_of_f_8  [16];
    float    no_of_f_16 [16];
    float    no_of_f_32 [16];
    float    no_of_f_64 [16];
    double   no_of_d_u8 [16];
    double   no_of_d_u16[16];
    double   no_of_d_u32[16];
    double   no_of_d_u64[16];
    float    no_of_f_u8 [16];
    float    no_of_f_u16[16];
    float    no_of_f_u32[16];
    float    no_of_f_u64[16];
    bool     no_of_i1   [16];
    int8_t   no_of_i8   [16];
    int16_t  no_of_i16  [16];
    int32_t  no_of_i32  [16];
    int64_t  no_of_i64  [16];
    uint8_t  no_of_ui8  [16];
    uint16_t no_of_ui16 [16];
    uint32_t no_of_ui32 [16];
    uint64_t no_of_ui64 [16];

///////////////////////////////////////////////////////////////////////
   
    InputData(); 
};

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


template <typename T>
bool check_and_print (T a, T b, int err_counter) {
    bool ret = (a != b);
    if (ret && err_counter < 10)
        std::cout << "result: " << a << " expected: " << b << std::endl;
    return ret;
}

template <>
bool check_and_print <double>(double a, double b, int err_counter) {
    bool ret = fabs(a - b) > std::numeric_limits<double>::epsilon() * fabs(a);
    if (ret && err_counter < 10)
        std::cout << "result: " << a << " expected: " << b << std::endl;
    return ret;
}

template <>
bool check_and_print <float>(float a, float b, int err_counter) {
    bool ret = fabs(a - b) > std::numeric_limits<float>::epsilon() * fabs(a);
    if (ret && err_counter < 10)
        std::cout << "result: " << a << " expected: " << b << std::endl;
    return ret;
}

#endif
