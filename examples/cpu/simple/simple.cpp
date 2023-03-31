/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdio.h>
#include <stdlib.h>

// Include the header file that the ispc compiler generates
#include "simple_ispc.h"
using namespace ispc;

int main() {
    float vin[16], vout[16];

    // Initialize input buffer
    for (int i = 0; i < 16; ++i)
        vin[i] = (float)i;

    // Call simple() function from simple.ispc file
    simple(vin, vout, 16);

    // Print results
    for (int i = 0; i < 16; ++i)
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);

    return 0;
}
