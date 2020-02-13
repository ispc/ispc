/*
  Copyright (c) 2019, Intel Corporation
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

#include "Matrix.h"
#include <iostream>

/*#ifdef _WIN32
#include <tchar.h>
#else
#include <unistd.h>
#endif*/
#ifdef LINUX
#define FALSE 0
#define TRUE 1
#endif

#include "bitmap_helpers.h"
#include "cm_rt_helpers.h"
#include "host_helpers.h"
#include "isa_helpers.h"

#include <chrono>
#include <string>

using namespace hostutil;

int run(int m, int niter, int gx, int gy) {
    storage_type_t st = RowMajor;
    float alpha = +1.0, beta = +0.0;

    int n = m, k = m;

    // Initialization
    m = (m / TILE_m) * TILE_m;
    n = k = m;

    int lda = ((k + 15) & ~15);
    int ldb = ((n + 15) & ~15);
    int ldc = ldb;
    printf("SGEMM: C(%d, %d) = %.2f * C(%d, %d) + %.2f A(%d, %d) * B(%d, %d)\n", m, n, beta, m, n, alpha, m, k, k, n);
    printf("Thread-group setting: %d x %d \n", gx, gy);

    CmDevice *device = nullptr;
    CmProgram *program = nullptr;
    CmKernel *kernel = nullptr;

#ifndef CMHOST
    CMInitContext(device, kernel, program, "naive_sgemm_mt.isa", "SGEMM_naive_task");
#else
    CMInitContext(device, kernel, program, "naive_sgemm_mt_cm.isa", "sgemm_kernel");
#endif

    // Allocate matrices
    Matrix A(m, k, lda, NULL, true, "A", st);
    Matrix B(k, n, ldb, NULL, true, "B", st);
    Matrix C_gold(m, n, ldc, NULL, false, "C_gold", st);
    //Matrix C(C_gold, "C");
    Matrix C(m, n, ldc, NULL, false, "C", st);
    Matrix zero(C_gold, "C");

    if (niter == 1) {
        printf("** validation run, only one iteration **\n");
        printf("** For performance run, add cmd-args: Sgemm 2048 1000 ** \n");
        // Compute gold result
        printf("Compute gold result\n");
        sgemmNxN(m, n, k, alpha, &A(0, 0), A.l_dim(), &B(0, 0), B.l_dim(), beta, &C_gold(0, 0), C_gold.l_dim(), st);
    } else {
        printf("CPU result not computed: Make #iterations=1 to compute CPU result\n");
    }

    kernel->SetKernelArg(3, sizeof(int), &m);
    kernel->SetKernelArg(4, sizeof(int), &n);
    kernel->SetKernelArg(5, sizeof(int), &k);

    CmBufferSVM *pASurf = nullptr;
    CmBufferSVM *pBSurf = nullptr;
    CmBufferSVM *pCSurf = nullptr;
    void *a_ref = &A(0, 0);
    void *b_ref = &B(0, 0);
    void *c_ref = &C(0, 0);
    cm_result_check(device->CreateBufferSVM(A.l_dim() * m * sizeof(float), a_ref, CM_SVM_ACCESS_FLAG_DEFAULT, pASurf));
    cm_result_check(
        device->CreateBufferSVM(B.l_dim() * B.n_row() * sizeof(float), b_ref, CM_SVM_ACCESS_FLAG_DEFAULT, pBSurf));
    cm_result_check(device->CreateBufferSVM(C.l_dim() * m * sizeof(float), c_ref, CM_SVM_ACCESS_FLAG_DEFAULT, pCSurf));

    kernel->SetKernelArg(0, sizeof(void *), &a_ref);
    kernel->SetKernelArg(1, sizeof(void *), &b_ref);
    kernel->SetKernelArg(2, sizeof(void *), &c_ref);

    auto timings = execute(device, kernel, gx, gy, niter, false, 2000);
    timings.print(niter);

    void *r = &C(0, 0);
    cm_result_check(pCSurf->GetAddress(r));

    bool pass = false;
    if (niter == 1) {
        if (C == C_gold) {
            printf("PASSED\n");
            pass = true;
        } else
            printf("FAILED\n");
    } else
        printf("Result not checked - make #iterations=1 to check result!\n");

    printf("----------------------------\n");

    cm_result_check(device->DestroyBufferSVM(pASurf));
    cm_result_check(device->DestroyBufferSVM(pBSurf));
    cm_result_check(device->DestroyBufferSVM(pCSurf));
    cm_result_check(device->DestroyKernel(kernel));

    cm_result_check(::DestroyCmDevice(device));

    if (pass)
        return 0;
    else
        return 1;
}

int main(int argc, char *argv[]) {
    std::cout << "HELLO!" << std::endl;
    int m = GEMM_BLOCK;
    int niterations = 1;
    int gx = 2, gy = 1;
    if (argc >= 3) {
        m = atoi(argv[1]);
        niterations = atoi(argv[2]);
        if (argc == 5) {
            gx = atoi(argv[3]);
            gy = atoi(argv[4]);
        }
    }

    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << m << " matrix size." << std::endl;
    run(m, niterations, gx, gy);

    return success;
}
