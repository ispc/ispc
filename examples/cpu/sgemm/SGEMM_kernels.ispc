/*
  Copyright (c) 2018-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Various ISPC SGEMM kernel and task/kernel implementations
// Junkins, September 2018

#define TILE_SIZE 32

export uniform int SGEMM_get_program_count() {
    return programCount;
}

export uniform int SGEMM_get_tile_size() {
    return TILE_SIZE;
}

// Naive implementation. The outer foreach achieves some parallelism.
export void SGEMM_naive(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
                        uniform int M, uniform int N, uniform int K) {
    uniform unsigned int n, k;
    varying float aValue, bValue, sum;

    foreach (m = 0 ... M) {
        for (k = 0; k < K; k++) {
            sum = 0.0f;
            for (n = 0; n < N; n++) {
                #pragma ignore warning(perf)
                aValue = matrixA[m*N + n];
                bValue = matrixB[n*K + k];
                sum +=  aValue * bValue;
            }

            #pragma ignore warning(perf)
            matrixC[m*K + k] = sum;
        }
    }
}


// Naive implementation with task parallelism added. The matrix rows are divided into chunks
// which are calculated by separate tasks.
static task void SGEMM_naive_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
                                    uniform int M, uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart       = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd         = uRowStart + uNumRowsPerTask;

    foreach (m = uRowStart ... uRowEnd) {
        uniform unsigned int n, k;
        varying float sum;

        for (k = 0; k < K; k++) {
            sum = 0.0f;
            for (n = 0; n < N; n++) {
                #pragma ignore warning(perf)
                sum += matrixA[m*N + n] * matrixB[n*K + k];
            }
            #pragma ignore warning(perf)
            matrixC[m*K + k] = sum;
        }
    }
}

export void SGEMM_naive_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
                                    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_naive_task(matA, matB, matC, M, N, K);
}

// Single task version:
// The optimization added here is use of a tile to reuse the loaded matrix value.
// A SIMD width of tile values are stored and re-used using the shuffle intrinsic.
export void SGEMM_tileShuffle(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
                                            uniform int M, uniform int N, uniform int K) {
    uniform int m, n, nn;
    varying float tileOfA, aValue, sum;

    // Launch SPMD control flow.
    foreach (k = 0 ... K) {
        for (m = 0; m < M; m++) {
            sum = 0.0f;
            for (n = 0; n < N; n+=programCount) {
                // A SIMD width's tile of A values...to be re-used.
                tileOfA = matrixA[m*N + n + programIndex];

                // Each program instance will tile multiply, rather naive scalar multiply
                for (nn = 0; nn < programCount; nn++) {
                    aValue = shuffle(tileOfA, nn);
                    // Re-use tile values, by shuffling them.
                    sum +=  aValue * matrixB[(n+nn)*K + k];
                }
            }
            matrixC[m*K + k] = sum;
        }
    }
}

// Multiple task version of the above:
static task void SGEMM_tileShuffle_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
                                            uniform int M, uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumColsPerTask    = K / taskCount;
    uniform unsigned int uColStart          = uNumColsPerTask * taskIndex;
    uniform unsigned int uColEnd            = uColStart + uNumColsPerTask;

    uniform int m, n, nn;
    varying float tileOfA, aValue, sum;

    // Now launch SPMD control flow over columns.
    foreach (k = uColStart ... uColEnd) {
        for (m = 0; m < M; m++) {
            sum = 0.0f;
            for (n = 0; n < N; n+=programCount) {
                // A SIMD width's tile of A values...to be re-used.
                tileOfA = matrixA[m*N + n + programIndex];
                // Each program instance will tile multiply.
                for (nn = 0; nn < programCount; nn++) {
                    // Re-use tile values, by shuffling them.
                    aValue = shuffle(tileOfA, nn);
                    sum +=  aValue * matrixB[(n+nn)*K + k];
                }
            }
            matrixC[m*K + k] = sum;
        }
    }
}

export void SGEMM_tileShuffle_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides columns in matrix C (K size) between tasks.
    // We want each task to process programCount colums in C matrix to maximize SIMD usage.
    uniform int numTasks = K / programCount;
    launch[numTasks] SGEMM_tileShuffle_task(matA, matB, matC, M, N, K);
 }

// Single task version:
// This version uses reduce_add(). This allows reduction across program instances.
// This allows re-use of stored tile without shuffle.
export void SGEMM_tileReduceAdd(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform int m, k, k0;
    uniform float sumTile[TILE_SIZE];
    varying float tileOfA;

    for (m = 0; m < M; m++) {
        for (k0 = 0; k0 < K; k0+=TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                sumTile[ki] = 0.0f;
            }

            // SPMD "vertically" over the matrix N dimension:
            foreach (n = 0 ... N) {
                tileOfA = matrixA[m*N + n];
                for (uniform int kt = 0; kt < TILE_SIZE; kt++) {
                    #pragma ignore warning(perf)
                    sumTile[kt] += reduce_add(tileOfA * matrixB[n*K + k0 + kt]);
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach(ki = 0 ... TILE_SIZE) {
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

// Multiple task version of the above:
static task void SGEMM_tileReduceAdd_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    uniform int m, k, k0;
    uniform float sumTile[TILE_SIZE];
    varying float tileOfA;

    for(m = uRowStart; m < uRowEnd; m++) {
        for (k0 = 0; k0 < K; k0 += TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                sumTile[ki] = 0.0f;
            }

            // SPMD "vertically" over the matrix N dimension:
            foreach (n = 0 ... N) {
                tileOfA = matrixA[m*N + n];
                for (uniform int kt = 0; kt < TILE_SIZE; kt++) {
                    #pragma ignore warning(perf)
                    sumTile[kt] += reduce_add(tileOfA * matrixB[n*K + k0 + kt]);
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

export void SGEMM_tileReduceAdd_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileReduceAdd_task(matA, matB, matC, M, N, K);
}

// This version uses atomic_add_local() to perform addition.
// This causes the version to be significantly slower due to the bottleneck here.
export void SGEMM_tileAtomicAdd(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform int m, k, k0;
    uniform float sumTile[TILE_SIZE];
    varying float tileOfA;

    for (m = 0; m < M; m++) {
        for (k0 = 0; k0 < K; k0 += TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                sumTile[ki] = 0.0f;
            }

            // SPMD "vertically" over the matrix N dimension:
            foreach (n = 0 ... N) {
                tileOfA = matrixA[m*N + n];

                for (uniform int kt = 0; kt < TILE_SIZE; kt++) {

                    #pragma ignore warning(perf)
                    varying float tileProd = tileOfA * matrixB[n*K + k0 + kt];
                    atomic_add_local(&sumTile[kt], tileProd);
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileAtomicAdd_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    uniform int m, k, k0;
    uniform float sumTile[TILE_SIZE];
    varying float tileOfA;

    for (m = uRowStart; m < uRowEnd; m++) {
        for (k0 = 0; k0 < K; k0 += TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                sumTile[ki] = 0.0f;
            }

            // SPMD "vertically" over the matrix N dimension:
            foreach (n = 0 ... N) {
                tileOfA = matrixA[m*N + n];

                for (uniform int kt = 0; kt < TILE_SIZE; kt++) {
                    #pragma ignore warning(perf)
                    varying float tileProd = tileOfA * matrixB[n*K + k0 + kt];
                    atomic_add_local(&sumTile[kt], tileProd);
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

export void SGEMM_tileAtomicAdd_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileAtomicAdd_task(matA, matB, matC, M, N, K);
}

// This SGEMM_tileNoSIMDIntrin() code is (much!) faster than above implementations for serveral reasons:
// 1) Every memory fetch and store to MatrixA, MatrixB, and MatrixC is achieved w/o general gather or scatter.
//    This is because ISPC allows repeatedly changing the "axis of SPMD parallelism" (e.g. multiple foreach()
//    loops, or even nesting a foreach() loop within a conventional for() loop.  Thus the compiler easily knows
//    what loop iteration variable is used to index memory, and since adjacent it can easily block load adjacent
//    elements in memory.
// 2) No atomics or horizonal intrinsics were used in this operation.
// 3) The code caches a wide sumTile[] of read/write values for re-use,
//    presumably "locally" in multiple SIMD registers.  Importantly, b.c the compiler knows the single threaded
//    foundation of the ISPC SPMD programmming model, it is easy for it to enable register allocation of such
//    tiled data.
//
// But most importantly, this code is simple to undestand, debug, and easy to maintain.
export void SGEMM_tileNoSIMDIntrin(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_SIZE], oneAVal;

    for (uniform unsigned int m = 0; m < M; m++) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                // No scatter required.
                sumTile[ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                oneAVal = matrixA[m*N + n];
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_SIZE) {
                    // Note, no gather required.
                    varying float matB  = matrixB[n*K + k0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[kt] += oneAVal * matB;
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                // Note, no scatter required.
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

// Multiple task version of the above:
static task void SGEMM_tileNoSIMDIntrin_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    uniform float sumTile[TILE_SIZE], oneAVal;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m++) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_SIZE) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                // No scatter required.
                sumTile[ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                oneAVal = matrixA[m*N + n];
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_SIZE) {
                    // Note, no gather required.
                    varying float matB = matrixB[n*K + k0 + kt];
                    // Pure SIMD FMAC;
                    sumTile[kt] += oneAVal * matB;
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_SIZE) {
                // Note, no scatter required.
                matrixC[m*K + k0 + ki] = sumTile[ki];
            }
        }
    }
}

export void SGEMM_tileNoSIMDIntrin_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileNoSIMDIntrin_task(matA, matB, matC, M, N, K);
}


// This version is modified version of 'SGEMM_tileNoSIMDIntrin'.
// The tile used to read/write values for re-use is a 2D block of height 2 instead of a n array of same width.
#define TILE_HEIGHT 2
#define TILE_WIDTH 32
export void SGEMM_tileBlockNoSIMDIntrin(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    // No scatter required.
                    sumTile[i][ki] = 0.0f;
                }
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++)
                    oneAVal[i] = matrixA[(m + i)*N + n];

                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[n*K + k0 + kt];
                    for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                        // Pure SIMD FMAC:
                        sumTile[i][kt] += oneAVal[i] * matB1;
                    }
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < 2; i++) {
                    matrixC[(m + i)*K + k0 + ki] = sumTile[i][ki];
                }

            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileBlockNoSIMDIntrin_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    // No scatter required.
                    sumTile[i][ki] = 0.0f;
                }
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++)
                    oneAVal[i] = matrixA[(m + i)*N + n];

                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[n*K + k0 + kt];
                    for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                        // Pure SIMD FMAC:
                        sumTile[i][kt] += oneAVal[i] * matB1;
                    }
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < 2; i++) {
                    matrixC[(m + i)*K + k0 + ki] = sumTile[i][ki];
                }

            }
        }
    }
}

export void SGEMM_tileBlockNoSIMDIntrin_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileBlockNoSIMDIntrin_task(matA, matB, matC, M, N, K);
}

// This version is a further modified version of 'SGEMM_tileBlockNoSIMDIntrin'.
// Since we already know the tile height, the loop used to access the tile vertically is replaced.
export void SGEMM_tileBlockNoSIMDIntrin_2(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0f;
                sumTile[1][ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileBlockNoSIMDIntrin_2_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0f;
                sumTile[1][ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

export void SGEMM_tileBlockNoSIMDIntrin_2_withTasks(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileBlockNoSIMDIntrin_2_task(matA, matB, matC, M, N, K);
}
