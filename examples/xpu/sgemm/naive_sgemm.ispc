/*
  Copyright (c) 2019-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

task void SGEMM_naive_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[], uniform int M,
                             uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    foreach (m = uRowStart... uRowEnd) {
        uniform unsigned int n, k;
        varying float sum;

        for (k = 0; k < K; k++) {
            sum = 0.0f;
            for (n = 0; n < N; n++) {
#pragma ignore warning(perf)
                sum += matrixA[m * N + n] * matrixB[n * K + k];
            }
#pragma ignore warning(perf)
            matrixC[m * K + k] = sum;
        }
    }
}
