/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef __UTIL_H__
#define __UTIL_H__

#include "matrix.h"
#include <stdio.h>

inline void printMatrix(DenseMatrix &M, const char *name) {
    printf("Matrix %s:\n", name);
    for (size_t row = 0; row < M.rows(); row++) {
        printf("row %2d: ", (int)row + 1);
        for (size_t col = 0; col < M.cols(); col++)
            printf("%6f ", M(row, col));
        printf("\n");
    }
    printf("\n");
}

#endif
