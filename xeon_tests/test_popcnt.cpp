#include "knc_test_driver_core.h"

void popcnt_int32(uint32_t *ui32);
void popcnt_int64(uint64_t *ui64);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_popcnt() {
    InputData inpData;

    popcnt_int32(inpData.ui32);
    popcnt_int64(inpData.ui64);

}

/////////////////////////////////////////////////////////////////////////////////////////////

#define POPCNT_TEST(TYPE, FUNC_NAME)                                                        \
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
        result = 0;                                                                         \
        for (result = 0; copy_data[i] != 0; result++)                                       \
             copy_data[i] &= copy_data[i] - 1;                                              \
        if (check_and_print(output, result, err_counter))                                   \
            err_counter++;                                                                  \
    }                                                                                       \
    if (err_counter != 0)                                                                   \
        printf(" errors %d\n", err_counter);                                                \
    else                                                                                    \
        printf(" no fails\n");                                                              \
}

POPCNT_TEST(uint32_t, popcnt_int32)
POPCNT_TEST(uint64_t, popcnt_int64)

