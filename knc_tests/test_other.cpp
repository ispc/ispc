#include "knc_test_driver_core.h"

void movmsk(int *mask);

/////////////////////////////////////////////////////////////////////////////////////////////

void test_other() {
    InputData inpData;

    movmsk(inpData.mask);

}

/////////////////////////////////////////////////////////////////////////////////////////////

void movmsk(int *m) {
    printf ("%-40s", "movmsk: ");

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

    if (check_and_print(output, mask, 0))
        printf(" error 1\n");
    else
        printf(" no fails\n");
}

