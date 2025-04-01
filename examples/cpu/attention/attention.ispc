//  Copyright (c) 2025, Intel Corporation
//  SPDX-License-Identifier: BSD-3-Clause

#if TARGET_WIDTH == 8
#define TILE_WIDTH 16
#elif TARGET_WIDTH == 16
#define TILE_WIDTH 32
#endif
#define TILE_HEIGHT 8
#define NEG_INF -1e30f

void matmul(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[], uniform int M, uniform int N,
            uniform int K) {
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

export void single_head_attention(uniform float Q[], uniform float K[], uniform float V[], uniform float output[],
                                  uniform int seq_len, uniform int d_model) {
    uniform float scale = 1.0f / sqrt((uniform float)d_model);

    //  Use a single memory pool instead of three separate allocations
    // Calculate total memory needed
    uniform int total_memory_size = seq_len * seq_len * 2 + d_model * seq_len;
    uniform float *uniform memory_pool = uniform new uniform float[total_memory_size];

    // Partition memory pool for different uses
    uniform float *uniform attention_scores = memory_pool;
    uniform float *uniform K_transposed = memory_pool + seq_len * seq_len;

    // Transpose K for QK^T calculation
    foreach (i = 0 ... seq_len) {
        for (uniform int j = 0; j < d_model; j++) {
            K_transposed[j * seq_len + i] = K[i * d_model + j];
        }
    }

    // Compute QK^T using matmul: (seq_len x d_model) * (d_model x seq_len) = (seq_len x seq_len)
    matmul(Q, K_transposed, attention_scores, seq_len, d_model, seq_len);

    // Scale the dot products in-place
    for (int row = 0; row < seq_len; row++) {
        // Find maximum value in the row (applying scaling inside this step)
        uniform float row_max = NEG_INF;
        foreach (j = 0 ... seq_len) {
            // Scale during max calculation
            varying float scaled_val = attention_scores[row * seq_len + j] * scale;
            attention_scores[row * seq_len + j] = scaled_val; // Store scaled value for later use
            row_max = max((uniform float)reduce_max(scaled_val), row_max);
        }

        // Compute exp(x - max) and sum
        uniform float row_sum = 0.0f;
        foreach (j = 0 ... seq_len) {
            // No need to scale again, already done in previous step
            varying float exp_val = exp(attention_scores[row * seq_len + j] - row_max);
            attention_scores[row * seq_len + j] = exp_val; // Store exp value
            row_sum += reduce_add(exp_val);
        }

        // Normalize
        foreach (j = 0 ... seq_len) {
            attention_scores[row * seq_len + j] /= row_sum;
        }
    }

    // Compute attention_scores * V using matmul: (seq_len x seq_len) * (seq_len x d_model) = (seq_len x d_model)
    matmul(attention_scores, V, output, seq_len, seq_len, d_model);

    // Free allocated memory
    delete[] memory_pool;
}