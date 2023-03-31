/*
  Copyright (c) 2019, Intel Corporation
  All rights reserved.

  SPDX-License-Identifier: BSD-3-Clause
*/

#define GEN_KERNEL
#undef CM_DEBUG

#include "cm/cm.h"

// alpha = 1; beta = 0;
const short init_0_15[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#ifdef LINUX
extern "C" _GENX_MAIN_ void sgemm_kernel(svmptr_t indxA [[type("svmptr_t")]], svmptr_t indxB [[type("svmptr_t")]],
                                         svmptr_t indxC [[type("svmptr_t")]], int M, int N, int K) {
#else
extern "C" _GENX_MAIN_ void sgemm_kernel(svmptr_t indxA, svmptr_t indxB, svmptr_t indxC, int M, int N, int K) {
#endif
    unsigned int x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    unsigned int y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);

    unsigned int threadNum = cm_linear_global_size();
    unsigned int threadid = cm_linear_global_id();

    // doesn't work for threadNum > K
    unsigned int rowPerTask = M / cm_linear_global_size();

    unsigned int start = threadid * rowPerTask;
    unsigned int end = start + rowPerTask;

    /*printf("%u Thread says rowPerTask is %u: start=%u;end=%u\n", threadid, rowPerTask, start, end);*/

    const int simd_size = 16;
    vector<float, simd_size> a, c; // assuming this is 2 consequent registers
    vector<float, 1> b;

    vector<uint, simd_size> off(init_0_15);
    vector<svmptr_t, simd_size> offseta;
    vector<svmptr_t, 1> offsetb;
    vector<svmptr_t, simd_size> offsetc;

    unsigned int m = start;
    for (; m < end; m += simd_size) {
        // don't let simd overlap
        if (m + simd_size > end) {
            break;
        }
        for (int k = 0; k < K; k++) {
            c = 0;
            for (int n = 0; n < N; n++) {
                offseta = indxA + (m * N + n) * sizeof(float) + off * N * sizeof(float);
                offsetb = indxB + (n * K + k) * sizeof(float);
                cm_svm_scatter_read(offseta, a);
                cm_svm_scatter_read(offsetb, b);

                c += a * b[0];
            }
            offsetc = indxC + (m * K + k) * sizeof(float) + off * K * (sizeof(float));
            // printf("%u Thread says OFFSET[%u]\n", threadid, indxC + (k*M + m) * sizeof(float));
            cm_svm_scatter_write(offsetc, c);
        }
    }
    // printf("%u thread says: m=%u end=%u\n", threadid, m, end);

    if (m != end) {
        // remainder loop
        off += m;
        SIMD_IF_BEGIN(off >= end) { off = end - 1; }
        SIMD_IF_END;

        vector<float, simd_size> a_rem, c_rem;
        vector<svmptr_t, simd_size> a_rem_offset, c_rem_offset;
        vector<svmptr_t, 1> b_rem_offset;
        for (int k = 0; k < K; k++) {
            c_rem = 0;
            for (int n = 0; n < N; n++) {
                a_rem_offset = indxA + (off * N + n) * sizeof(float);
                b_rem_offset = indxB + (n * K + k) * sizeof(float);
                cm_svm_scatter_read(a_rem_offset, a_rem);
                cm_svm_scatter_read(b_rem_offset, b);
                c_rem += a_rem * b[0];
            }
            c_rem_offset = indxC + (off * K + k) * sizeof(float);
            cm_svm_scatter_write(c_rem_offset, c_rem);
        }
    }
}
