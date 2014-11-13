#include "knc_test_driver_core.h"

void count_trailing_zeros_i32(uint32_t *ui32);
void count_trailing_zeros_i64(uint64_t *ui64);

void count_leading_zeros_i32(uint32_t *ui32);
void count_leading_zeros_i64(uint64_t *ui64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_count_zeros() {
    InputData inpData;

    count_trailing_zeros_i32(inpData.ui32);
    count_trailing_zeros_i64(inpData.ui64);

    //count_leading_zeros_i32(inpData.ui32);
    //count_leading_zeros_i64(inpData.ui64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define COUNT_TRAILING_ZEROS(TYPE, BIT_NUM, FUNC_NAME)                                      \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
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
        if (check_and_print(output, result, err_counter))                                   \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

COUNT_TRAILING_ZEROS(uint32_t, 32, count_trailing_zeros_i32)
COUNT_TRAILING_ZEROS(uint64_t, 64, count_trailing_zeros_i64)

#define COUNT_LEADING_ZEROS(TYPE, BIT_NUM, FUNC_NAME)                                       \
void FUNC_NAME(TYPE *data) {                                                                \
    printf ("%-40s", #FUNC_NAME ":");                                                       \
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
        if (check_and_print(output, result, err_counter))                                   \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

//COUNT_LEADING_ZEROS(uint32_t, 32, count_leading_zeros_i32)
//COUNT_LEADING_ZEROS(uint64_t, 64, count_leading_zeros_i64)

