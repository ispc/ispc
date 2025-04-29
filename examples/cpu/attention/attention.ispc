//  Copyright (c) 2025, Intel Corporation
//  SPDX-License-Identifier: BSD-3-Clause

#if TARGET_WIDTH == 8
#define K_C 512
#define N_C 512
#define M_C 32
#define N_R 8
#define M_R 12
#elif TARGET_WIDTH == 16
#define K_C 256
#define N_C 1024
#define M_C 256
#define N_R 32
#define M_R 8
#endif

// assume K % K_C == 0
// assume N % N_C == 0
// assume N_C % N_R == 0

// defaults if not defined previously
#ifndef K_C
#define K_C 512
#endif
#ifndef N_C
#define N_C 512
#endif
#ifndef M_C
#define M_C 64
#endif
#ifndef N_R
#define N_R 16
#endif
#ifndef M_R
#define M_R 8
#endif

// A matrix is stored in column-major order
void loop_0(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {

  // assert(M == M_R);
  // assert(N == N_R);
  // assert(K == K_C);
  // assert(lda == M_R);
  // assert(ldb == N_R);

  uniform float sumTile[M_R][N_R] = { { 0 } };
  uniform float oneAVal[M_R] = { 0 };

  for (uniform int k = 0; k < K; k++) {
    for (uniform int m = 0; m < M_R; m++) {
      oneAVal[m] = A[m + k * lda];
    }

    foreach (vn = 0 ... N_R) {
      varying float matB1 = B[k * ldb + vn];
      for (uniform int m = 0; m < M_R; m++) {
        sumTile[m][vn] += oneAVal[m] * matB1;
      }
    }
  }

  foreach (vn = 0 ... N_R) {
    for (uniform int m = 0; m < M_R; m++) {
      C[m * ldc + vn] += sumTile[m][vn];
    }
  }

}

// A matrix is stored in column-major order
// It is different from loop_0 as tile is M x N_R
void loop_0_r(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {

  uniform float sumTile[M_R][N_R] = { { 0 } };
  uniform float oneAVal[M_R] = { 0 };

  for (uniform int k = 0; k < K; k++) {
    for (uniform int m = 0; m < M; m++) {
      oneAVal[m] = A[m + k * lda];
    }

    foreach (vn = 0 ... N_R) {
      varying float matB1 = B[k * ldb + vn];
      for (uniform int m = 0; m < M; m++) {
        sumTile[m][vn] += oneAVal[m] * matB1;
      }
    }
  }

  foreach (vn = 0 ... N_R) {
    for (uniform int m = 0; m < M; m++) {
      C[m * ldc + vn] += sumTile[m][vn];
    }
  }

}

// A matrix is stored in column-major order
void loop_0_naive(uniform int M, uniform int N, uniform int K,
                  uniform float A[], uniform int lda, uniform float B[],
                  uniform int ldb, uniform float C[], uniform int ldc) {
  for (uniform int m = 0; m < M; m++) {
    for (uniform int n = 0; n < N; n++) {
      for (uniform int k = 0; k < K; k++) {
        C[m * ldc + n] += A[m + k * lda] * B[k * ldb + n];
      }
    }
  }
}

task void loop_1_task(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  uniform int R = M % M_R;
  if (R > 0) {
    loop_0_r(R, N, K, &A[(M - R) * lda], R, B, ldb, &C[(M - R) * ldc], ldc);
  } else {
    uniform int ir = taskIndex * M_R;
    loop_0(M_R, N, K, &A[ir * lda], M_R, B, ldb, &C[ir * ldc], ldc);
  }
}

void loop_1(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
#ifndef ISPC_TASKED_LOOP1
  for (uniform int ir = 0; (ir < M) && !(ir + M_R > M); ir += M_R) {
    loop_0(M_R, N, K, &A[ir * lda], M_R, B, ldb, &C[ir * ldc], ldc);
  }

  uniform int R = M % M_R;
  if (R > 0) {
    loop_0_r(R, N, K, &A[(M - R) * lda], R, B, ldb, &C[(M - R) * ldc], ldc);
  }
#else
  uniform int numTasks = M / M_R;
  launch[numTasks] loop_1_task(M, N, K, A, lda, B, ldb, C, ldc);
  sync;
#endif // ISPC_TASKED_LOOP1
}

task void loop_2_task(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  uniform int jr = taskIndex * N_R;
  loop_1(M, N_R, K, A, lda, &B[jr * K], N_R, &C[jr], ldc);
}

void loop_2(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  // assert(N % N_R == 0);
#ifndef ISPC_TASKED_LOOP2
  for (uniform int jr = 0; jr < N; jr += N_R) {
    loop_1(M, N_R, K, A, lda, &B[jr * K], N_R, &C[jr], ldc);
  }
#else
  uniform int numTasks = N / N_R;
  launch[numTasks] loop_2_task(M, N, K, A, lda, B, ldb, C, ldc);
  sync;
#endif // ISPC_TASKED_LOOP2
}

void pack_A(uniform int M, uniform int K, uniform float A[],
                          uniform int lda, uniform float A_packed[]) {
  // assert(M % M_R == 0);
  for (uniform int r = 0; (r < M) && !(r + M_R > M); r += M_R) {
    for (uniform int k = 0; k < K; k++) {
      for (uniform int mr = 0; mr < M_R; mr++) {
        A_packed[r * K + mr + k * M_R] = A[(r + mr) * lda + k];
      }
    }
  }
  uniform int R = M % M_R;
  if (R > 0) {
    for (uniform int k = 0; k < K; k++) {
      for (uniform int mr = 0; mr < R; mr++) {
        A_packed[(M - R) * K + mr + k * R] = A[(M - R + mr) * lda + k];
      }
    }
  }
}

void loop_3(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  // assert(M % M_C == 0);
  uniform float A_data[M_C][K_C] = { { 0 } };
  uniform float *uniform A_packed = (uniform float *uniform) &A_data[0][0];
  for (uniform int i = 0; i < M; i += M_C) {
    pack_A(M_C, K, &A[i * lda], lda, A_packed);
    // leading dimension of A_packed is somewhat confusing here
    loop_2(M_C, N, K, A_packed, K, B, ldb, &C[i * ldc], ldc);
  }
}

void pack_B(uniform int K, uniform int N, uniform float B[],
                          uniform int ldb, uniform float B_packed[]) {
  // assert(N % N_R == 0);
  for (uniform int r = 0; r < N; r += N_R) {
    for (uniform int k = 0; k < K; k++) {
      foreach (nr = 0 ... N_R) {
        B_packed[r * K + k * N_R + nr] = B[k * ldb + (r + nr)];
      }
    }
  }
}

void loop_4(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  // assert(K % K_C == 0);
  uniform float B_data[K_C][N_C] = { { 0 } };
  uniform float *uniform B_packed = (uniform float *uniform) &B_data[0][0];
  for (uniform int p = 0; p < K; p += K_C) {
    pack_B(K_C, N, &B[p * ldb], ldb, B_packed);
    // leading dimension of B_packed is somewhat confusing here
    loop_3(M, N, K_C, &A[p], lda, B_packed, N, C, ldc);
  }
}

void loop_5(uniform int M, uniform int N, uniform int K,
                          uniform float A[], uniform int lda, uniform float B[],
                          uniform int ldb, uniform float C[], uniform int ldc) {
  // assert(N % N_C == 0);
  for (uniform int j = 0; j < N; j += N_C) {
    loop_4(M, N_C, K, A, lda, &B[j], ldb, &C[j], ldc);
  }
}

void gemm(uniform float A[], uniform float B[], uniform float C[],
          uniform int M_0, uniform int N_0, uniform int K_0) {
  uniform int lda = K_0;
  uniform int ldb = N_0;
  uniform int ldc = N_0;

  loop_5(M_0, N_0, K_0, A, lda, B, ldb, C, ldc);
}

#if TARGET_WIDTH == 8
#define TILE_WIDTH 16
#elif TARGET_WIDTH == 16
#define TILE_WIDTH 32
#endif
#define TILE_HEIGHT 8
#define NEG_INF -1e30f

void matmul(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[], uniform int M, uniform int K,
            uniform int N) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int n0 = 0; n0 < N; n0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ni = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    // No scatter required.
                    sumTile[i][ni] = 0.0f;
                }
            }

            // Loop over the matrix K dimension:
            for (uniform unsigned int k = 0; k < K; k++) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++)
                    oneAVal[i] = matrixA[(m + i) * K + k];

                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (nt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[k * N + n0 + nt];
                    for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                        // Pure SIMD FMAC:
                        sumTile[i][nt] += oneAVal[i] * matB1;
                    }
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ni = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    matrixC[(m + i) * N + n0 + ni] = sumTile[i][ni];
                }
            }
        }
    }
}

// Multiple task version of the above:
task void matmul_task(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[], uniform int M,
                      uniform int N, uniform int K) {
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
                    oneAVal[i] = matrixA[(m + i) * N + n];

                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[n * K + k0 + kt];
                    for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                        // Pure SIMD FMAC:
                        sumTile[i][kt] += oneAVal[i] * matB1;
                    }
                }
            }

            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    matrixC[(m + i) * K + k0 + ki] = sumTile[i][ki];
                }
            }
        }
    }
}

void transpose_matrix(uniform float input[], uniform float output[], uniform int width, uniform int height) {
    // Process the matrix in tiles
    for (uniform int tile_y = 0; tile_y < height; tile_y += TILE_HEIGHT) {
        for (uniform int tile_x = 0; tile_x < width; tile_x += TILE_WIDTH) {

            // Calculate the end of the current tile (clamped to matrix dimensions)
            uniform int end_y = min(tile_y + TILE_HEIGHT, height);
            uniform int end_x = min(tile_x + TILE_WIDTH, width);

            // Process all elements within the current tile in parallel
            foreach (y = tile_y... end_y, x = tile_x... end_x) {
                output[x * height + y] = input[y * width + x];
            }
        }
    }
}

task void transpose_matrix_task(uniform float input[], uniform float output[], uniform int width, uniform int height) {
    // Determine workset for this task instance
    uniform int rows_per_task = height / taskCount;
    uniform int row_start = rows_per_task * taskIndex;
    uniform int row_end = row_start + rows_per_task;

    // Process the assigned rows in tiles
    for (uniform int tile_y = row_start; tile_y < row_end; tile_y += TILE_HEIGHT) {
        for (uniform int tile_x = 0; tile_x < width; tile_x += TILE_WIDTH) {
            // Calculate the end of the current tile (clamped to matrix dimensions)
            uniform int end_y = min(tile_y + TILE_HEIGHT, row_end);
            uniform int end_x = min(tile_x + TILE_WIDTH, width);

            // Process all elements within the current tile in parallel
            foreach (y = tile_y... end_y, x = tile_x... end_x) {
                output[x * height + y] = input[y * width + x];
            }
        }
    }
}

void softmax(uniform float matrix[], uniform float scale, uniform int seq_len) {
    // Process each row assigned to this task
    for (uniform int row = 0; row < seq_len; row++) {
        // Find maximum value in the row (applying scaling inside this step)
        uniform float row_max = NEG_INF;
        foreach (j = 0 ... seq_len) {
            // Scale during max calculation
            varying float scaled_val = matrix[row * seq_len + j] * scale;
            matrix[row * seq_len + j] = scaled_val; // Store scaled value for later use
            row_max = max((uniform float)reduce_max(scaled_val), row_max);
        }

        // Compute exp(x - max) and sum
        uniform float row_sum = 0.0f;
        foreach (j = 0 ... seq_len) {
            // No need to scale again, already done in previous step
            varying float exp_val = exp(matrix[row * seq_len + j] - row_max);
            matrix[row * seq_len + j] = exp_val; // Store exp value
            row_sum += reduce_add(exp_val);
        }

        // Normalize
        foreach (j = 0 ... seq_len) {
            matrix[row * seq_len + j] /= row_sum;
        }
    }
}

task void softmax_task(uniform float matrix[], uniform float scale, uniform int seq_len) {
    // Determine workset for this task instance
    uniform int rows_per_task = seq_len / taskCount;
    uniform int row_start = rows_per_task * taskIndex;
    uniform int row_end = row_start + rows_per_task;

    // Process each row assigned to this task
    for (uniform int row = row_start; row < row_end; row++) {
        // Find maximum value in the row (applying scaling inside this step)
        uniform float row_max = NEG_INF;
        foreach (j = 0 ... seq_len) {
            // Scale during max calculation
            varying float scaled_val = matrix[row * seq_len + j] * scale;
            matrix[row * seq_len + j] = scaled_val; // Store scaled value for later use
            row_max = max((uniform float)reduce_max(scaled_val), row_max);
        }

        // Compute exp(x - max) and sum
        uniform float row_sum = 0.0f;
        foreach (j = 0 ... seq_len) {
            // No need to scale again, already done in previous step
            varying float exp_val = exp(matrix[row * seq_len + j] - row_max);
            matrix[row * seq_len + j] = exp_val; // Store exp value
            row_sum += reduce_add(exp_val);
        }

        // Normalize
        foreach (j = 0 ... seq_len) {
            matrix[row * seq_len + j] /= row_sum;
        }
    }
}

export void single_head_attention(uniform float Q[], uniform float K[], uniform float V[], uniform float output[],
                                  uniform int seq_len, uniform int d_model) {
    uniform float scale = rsqrt((uniform float)d_model);

    // Calculate total memory needed, use a single memory pool
    uniform int total_memory_size = seq_len * seq_len * 2 + d_model * seq_len;
    uniform float *uniform memory_pool = uniform new uniform float[total_memory_size];
    foreach (i = 0 ... total_memory_size) {
        memory_pool[i] = 0.0f;
    }
    foreach (i = 0 ... seq_len * d_model) {
        output[i] = 0.0f;
    }

    // Partition memory pool for different uses
    uniform float *uniform attention_scores = memory_pool;
    uniform float *uniform K_transposed = memory_pool + seq_len * seq_len;

    uniform int numTasks = seq_len / programCount;
    // Transpose K for QK^T calculation
#ifndef ONE_THREAD
    launch[numTasks] transpose_matrix_task(K, K_transposed, d_model, seq_len);
    sync;
#else
    transpose_matrix(K, K_transposed, d_model, seq_len);
#endif // ONE_THREAD

    // Compute QK^T using matmul: (seq_len x d_model) * (d_model x seq_len) = (seq_len x seq_len)
#ifdef GOTO_IMPL
    gemm(Q, K_transposed, attention_scores, seq_len, seq_len, d_model);
#else
#ifndef ONE_THREAD
    launch[numTasks] matmul_task(Q, K_transposed, attention_scores, seq_len, d_model, seq_len);
    sync;
#else
    matmul(Q, K_transposed, attention_scores, seq_len, d_model, seq_len);
#endif // ONE_THREAD
#endif // GOTO_IMPL

    // Apply softmax to each row of attention_scores
#ifndef ONE_THREAD
    launch[numTasks] softmax_task(attention_scores, scale, seq_len);
    sync;
#else
    softmax(attention_scores, scale, seq_len);
#endif // ONE_THREAD

    // Compute attention_scores * V using matmul: (seq_len x seq_len) * (seq_len x d_model) = (seq_len x d_model)
#ifdef GOTO_IMPL
    gemm(attention_scores, V, output, seq_len, d_model, seq_len);
#else
#ifndef ONE_THREAD
    launch[numTasks] matmul_task(attention_scores, V, output, seq_len, seq_len, d_model);
    sync;
#else
    matmul(attention_scores, V, output, seq_len, seq_len, d_model);
#endif // ONE_THREAD
#endif // GOTO_IMPL

    // Free allocated memory
    delete[] memory_pool;
}
