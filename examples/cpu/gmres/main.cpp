/*
  Copyright (c) 2012, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    if (A == NULL)
        return -1;
    DEBUG_PRINT("... size: %lu\n", A->cols());

    DEBUG_PRINT("Loading b...\n");
    Vector *b = Vector::vector_from_mtf(argv[2]);
    if (b == NULL)
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
