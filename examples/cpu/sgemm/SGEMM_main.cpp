/*
  Copyright (c) 2018-2020, Intel Corporation
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

// Simple harness for testing SGEMM implementations in ISPC
// Junkins, September 2018

// clang-format off
/**
Matrix layout, rows x col, row major storage:

        N                     K                     K
   ---     ---           ---     ---           ---     ---
   |         |           |         |           |         |
M  |         |   X     N |         |   =     M |         |
   |         |           |         |           |         |
   ---     ---           ---     ---           ---     ---

        A        X            B        =            C
**/
// clang-format on

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../../common/timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OS Independent millisec wall timers
#ifdef _WIN32
#define WINDOWS
#endif
#ifdef _WIN64
#define WINDOWS
#endif
#ifdef __linux__
#define LINUX
#endif

#ifdef WINDOWS
#define TIMER_DECLARE_AND_INIT()                                                                                       \
    LARGE_INTEGER beginClock, endClock, cpuClockFreq;                                                                  \
    QueryPerformanceFrequency(&cpuClockFreq);

#define TIMER_RESET_AND_START() QueryPerformanceCounter(&beginClock);

#define TIMER_GET_ELAPSED_MSEC()                                                                                       \
    (QueryPerformanceCounter(&endClock) & 0) + (QueryPerformanceFrequency(&cpuClockFreq) & 0) +                        \
        (double(endClock.QuadPart - beginClock.QuadPart) * 1000.0f / cpuClockFreq.QuadPart)
#else
#define TIMER_DECLARE_AND_INIT()
#define TIMER_RESET_AND_START() reset_and_start_timer();
#define TIMER_GET_ELAPSED_MSEC() get_elapsed_msec();
#endif

// Include the header file that the ispc compiler generates
#include "SGEMM_kernels_ispc.h"
using namespace ispc;

void init_matrix(float M[], unsigned int rows, unsigned int cols, float value) {
    for (unsigned int r = 0; r < rows; r++)
        for (unsigned int c = 0; c < cols; c++)
            M[r * cols + c] = value;
}

void init_matrix_rand(float M[], unsigned int rows, unsigned int cols, float rangeValue) {
    for (unsigned int r = 0; r < rows; r++)
        for (unsigned int c = 0; c < cols; c++) {
            float rnd = ((float)rand() / (float)(RAND_MAX)) * rangeValue;
            M[r * cols + c] = rnd;
        }
}

void print_matrix(char *pcName, float M[], unsigned int rows, unsigned int cols) {
    printf("%s:\n", pcName);
    for (unsigned int r = 0; r < rows; r++) {
        for (unsigned int c = 0; c < cols; c++) {
            printf("%8.2f ", M[r * cols + c]);
        }
        printf("\n");
    }
    printf("\n");
}

void SGEMM_CPU_validation(float matrixA[], float matrixB[], float matrixC[], unsigned int M, unsigned int N,
                          unsigned int K) {
    for (unsigned int m = 0; m < M; m++) {
        unsigned int n, k;
        float sum;
        for (k = 0; k < K; k++) {
            sum = 0.0f;
            for (n = 0; n < N; n++) {
                sum += matrixA[m * N + n] * matrixB[n * K + k];
            }

            matrixC[m * K + k] = sum;
        }
    }
}

#define EPSILON 0.01f
bool Validate_result(float matrixC[], float matrixValid[], unsigned int M, unsigned int K) {
    for (unsigned int m = 0; m < M; m++) {
        for (unsigned int k = 0; k < K; k++) {
            float delta = (float)fabs(matrixC[m * K + k] - matrixValid[m * K + k]);
            if (delta > EPSILON)
                return false;
        }
    }
    return true;
}

typedef void (*SGEMMFuncPtr)(void);
typedef void (*SGEMMFuncPtr_SingleThreaded)(float matrixA[], float matrixB[], float matrixC[], unsigned int M,
                                            unsigned int N, unsigned int K);
typedef void (*SGEMMFuncPtr_MultiThreaded)(float matrixA[], float matrixB[], float matrixC[], unsigned int M,
                                           unsigned int N, unsigned int K);

void Test_SGEMM(SGEMMFuncPtr SGEMMFunc, char *pcFuncName, float matrixA[], float matrixB[], float matrixC[],
                unsigned int M, unsigned int N, unsigned int K, bool tasks, unsigned int numIterations,
                float matrixValid[]) {
    double totalWallTime;
    float avgTime;
    unsigned int i;
    bool bValid;
    char psValid[256];
    TIMER_DECLARE_AND_INIT();

    // Total = MNK mults + MN(K-1) adds.
    float fFlopsPerGEMM = (float)(M * N * K) + (M * N * (K - 1));

    if (tasks == false) {
        // type cast
        SGEMMFuncPtr_SingleThreaded SGEMMFunc_ST = (SGEMMFuncPtr_SingleThreaded)SGEMMFunc;

        TIMER_RESET_AND_START();
        for (i = 0; i < numIterations; i++)
            SGEMMFunc_ST(matrixA, matrixB, matrixC, M, N, K);
        totalWallTime = TIMER_GET_ELAPSED_MSEC();
    } else {
        // type cast
        SGEMMFuncPtr_MultiThreaded SGEMMFunc_MT = (SGEMMFuncPtr_MultiThreaded)SGEMMFunc;

        TIMER_RESET_AND_START();
        for (i = 0; i < numIterations; i++)
            SGEMMFunc_MT(matrixA, matrixB, matrixC, M, N, K);
        totalWallTime = TIMER_GET_ELAPSED_MSEC();
    }

    avgTime = (float)totalWallTime / (float)numIterations;
    bValid = Validate_result(matrixC, matrixValid, M, K);
    if (bValid)
        sprintf(psValid, "valid");
    else
        sprintf(psValid, "ERROR");
    printf("%40s %10.4f millisecs %10.4f GFLOPs Validation: %s.\n", pcFuncName, avgTime,
           (fFlopsPerGEMM / (avgTime / 1000.0f)) / 1000000000.0f, psValid);
    init_matrix(matrixC, M, K, 0.0f);
}

int main(int argc, char **argv) {
    // Random number filled matrix test case:

    // Default values for input parameters
    int ITERATIONS = 500;
    int M = 256;
    int N = 64;
    int K = 512;

    printf("\nUsage: SGEMM (optional)[ispc iterations] (optional)[[Matrix A Rows] [Matrix A Columns/ matrix B Rows] "
           "[Matrix B Columns]]\n");
    if (argc < 2) {
        printf("ispc iterations = %d[default],\t Matrix A Rows = %d[default],\t Matrix A Columns/ matrix B Rows = "
               "%d[default], Matrix B Columns = %d[default]\n",
               ITERATIONS, M, N, K);
    } else if (argc == 2) {
        ITERATIONS = atoi(argv[1]);
        printf("%s\n", argv[0]);
        printf("ispc iterations = %d,\t Matrix A Rows = %d[default],\t Matrix A Columns/ matrix B Rows = %d[default], "
               "Matrix B Columns = %d[default]\n",
               ITERATIONS, M, N, K);
    } else if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        printf("%s\n", argv[0]);
        printf("ispc iterations = %d[default],\t Matrix A Rows = %d,\t Matrix A Columns/ matrix B Rows = %d, Matrix B "
               "Columns = %d\n",
               ITERATIONS, M, N, K);
    } else if (argc == 5) {
        ITERATIONS = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        K = atoi(argv[4]);
        printf("%s\n", argv[0]);
        printf("ispc iterations = %d,\t Matrix A Rows = %d,\t Matrix A Columns/ matrix B Rows = %d, Matrix B Columns = "
               "%d\n",
               ITERATIONS, M, N, K);
    } else {
        printf("%s\n", argv[0]);
        printf("\nInvalid number of inputs\n");
        exit(-1);
    }

    int programCount = SGEMM_get_program_count();
    int tileSize = SGEMM_get_tile_size();
    if (K % programCount != 0 || K % tileSize != 0) {
        printf("\nNumber of columns in Matrix B (K) must be a multiple of %d (target width) and %d (tile size)!\n",
               programCount, tileSize);
        exit(-1);
    }

    if (M % programCount != 0) {
        printf("\nNumber of rows in Matrix A (M) must be a multiple of %d (target width)!\n", programCount);
        exit(-1);
    }

    if (N % programCount != 0) {
        printf("\nNumber of columns in Matrix A (N), which is also number of rows in Matrix B, "
               "must be a multiple of %d (target width)!\n",
               programCount);
        exit(-1);
    }

    float *matrixA;
    matrixA = (float *)malloc(M * N * sizeof(float));
    init_matrix_rand(matrixA, M, N, 10.0f);
    float *matrixB;
    matrixB = (float *)malloc(N * K * sizeof(float));
    init_matrix_rand(matrixB, N, K, 10.0f);
    float *matrixC;
    matrixC = (float *)malloc(M * K * sizeof(float));
    init_matrix(matrixC, M, K, 0.0f);
    float *matrixValid;
    matrixValid = (float *)malloc(M * K * sizeof(float));
    bool tasks = false;

    // Generate a validation matrix using CPU code:
    SGEMM_CPU_validation(matrixA, matrixB, matrixValid, M, N, K);

    // Single threaded test cases:
    Test_SGEMM((SGEMMFuncPtr)SGEMM_naive, (char *)"SGEMM_naive", matrixA, matrixB, matrixC, M, N, K, tasks, ITERATIONS,
               matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileShuffle, (char *)"SGEMM_tileShuffle", matrixA, matrixB, matrixC, M, N, K, tasks,
               ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileReduceAdd, (char *)"SGEMM_tileReduceAdd", matrixA, matrixB, matrixC, M, N, K,
               tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileAtomicAdd, (char *)"SGEMM_tileAtomicAdd", matrixA, matrixB, matrixC, M, N, K,
               tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileNoSIMDIntrin, (char *)"SGEMM_tileNoSIMDIntrin", matrixA, matrixB, matrixC, M, N,
               K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin, (char *)"SGEMM_tileBlockNoSIMDIntrin", matrixA, matrixB,
               matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2, (char *)"SGEMM_tileBlockNoSIMDIntrin_2", matrixA, matrixB,
               matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    printf("\n");

    // Multi-threaded test cases:
    tasks = true;
    Test_SGEMM((SGEMMFuncPtr)SGEMM_naive_withTasks, (char *)"SGEMM_naive_withTasks", matrixA, matrixB, matrixC, M, N, K,
               tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileShuffle_withTasks, (char *)"SGEMM_tileShuffle_withTasks", matrixA, matrixB,
               matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileReduceAdd_withTasks, (char *)"SGEMM_tileReduceAdd_withTasks", matrixA, matrixB,
               matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileAtomicAdd_withTasks, (char *)"SGEMM_tileAtomicAdd_withTasks", matrixA, matrixB,
               matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileNoSIMDIntrin_withTasks, (char *)"SGEMM_tileNoSIMDIntrin_withTasks", matrixA,
               matrixB, matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_withTasks, (char *)"SGEMM_tileBlockNoSIMDIntrin_withTasks",
               matrixA, matrixB, matrixC, M, N, K, tasks, ITERATIONS, matrixValid);
    Test_SGEMM((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_withTasks, (char *)"SGEMM_tileBlockNoSIMDIntrin_2_withTasks",
               matrixA, matrixB, matrixC, M, N, K, tasks, ITERATIONS, matrixValid);

    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixValid);
    return 0;
}
