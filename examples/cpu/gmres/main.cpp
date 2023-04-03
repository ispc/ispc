/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "../../common/timing.h"
#include "algorithm.h"
#include "matrix.h"
#include "util.h"
#include <cmath>

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s <input-matrix> <input-rhs> <output-file>\n", argv[0]);
        return -1;
    }

    double gmres_cycles;

    DEBUG_PRINT("Loading A...\n");
    Matrix *A = CRSMatrix::matrix_from_mtf(argv[1]);
    if (A == nullptr)
        return -1;
    DEBUG_PRINT("... size: %lu\n", A->cols());

    DEBUG_PRINT("Loading b...\n");
    Vector *b = Vector::vector_from_mtf(argv[2]);
    if (b == nullptr)
        return -1;

    Vector x(A->cols());
    DEBUG_PRINT("Beginning gmres...\n");
    reset_and_start_timer();
    gmres(*A, *b, x, A->cols() / 2, .01);
    gmres_cycles = get_elapsed_mcycles();
    // Write result out to file
    x.to_mtf(argv[argc - 1]);

    // Compute residual (double-check)
#ifdef DEBUG
    Vector bprime(b->size());
    A->multiply(x, bprime);
    Vector resid(bprime.size(), &(bprime[0]));
    resid.subtract(*b);
    DEBUG_PRINT("residual error check: %lg\n", resid.norm() / b->norm());
#endif
    // Print profiling results
    DEBUG_PRINT("-- Total mcycles to solve : %.03f --\n", gmres_cycles);
}
