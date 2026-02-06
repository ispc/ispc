/*
  Copyright (c) 2025-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// ISPC AMX-FP16 Matrix Multiplication using ISPC stdlib
// Demonstrates Intel AMX-FP16 usage through ISPC with tiling for 128x128
// matrices

#include <amx.isph>

// AMX tile configuration structure (64 bytes total)
struct TileConfig {
  int8 palette_id;   // 0: palette_id
  int8 start_row;    // 1: startRow
  int8 reserved[14]; // 2-15: reserved (must be zero)
  int16 colsb[16];   // 16-47: tile0.colsb through tile15.colsb
  int8 rows[16];     // 48-63: tile0.rows through tile15.rows
};

// Constants for AMX-FP16 tile sizes (hardware limits)
static const uniform int TILE_M = 16; // AMX tile rows
static const uniform int TILE_N = 16; // AMX tile cols
static const uniform int TILE_K = 32; // AMX tile inner dimension

// AMX Tile Register Assignments (TMM0-TMM7) with proper cast
#define TMM_C ((uniform int8)0) // Result tile C (16x16 FP32)
#define TMM_A ((uniform int8)1) // Matrix A tile (16x32 FP16)
#define TMM_B ((uniform int8)2) // Matrix B tile (32x16 FP16)

// Matrix dimensions
#ifndef MAT_SIZE_M
#define MAT_SIZE_M 128 // A rows, C rows
#endif
#ifndef MAT_SIZE_N
#define MAT_SIZE_N 128 // B cols, C cols
#endif
#ifndef MAT_SIZE_K
#define MAT_SIZE_K 128 // A cols, B rows
#endif

// Calculate number of tiles needed
static const uniform int TILES_M = (MAT_SIZE_M + TILE_M - 1) / TILE_M;
static const uniform int TILES_N = (MAT_SIZE_N + TILE_N - 1) / TILE_N;
static const uniform int TILES_K = (MAT_SIZE_K + TILE_K - 1) / TILE_K;

// Initialize tile configuration for AMX-FP16
export void init_amx_config(uniform TileConfig config[]) {
  // Clear the entire structure
  for (uniform int i = 0; i < sizeof(uniform TileConfig); ++i) {
    ((uniform int8 *)config)[i] = 0;
  }

  config->palette_id = 1;
  config->start_row = 0;

  // Configure tile 0 (result tile C): TILE_M×TILE_N FP32 values
  config->colsb[0] = TILE_N * 4; // 4 bytes per FP32
  config->rows[0] = TILE_M;

  // Configure tile 1 (matrix A): TILE_M×TILE_K FP16 values
  config->colsb[1] = TILE_K * 2; // 2 bytes per FP16
  config->rows[1] = TILE_M;

  // Configure tile 2 (matrix B): Special AMX layout for TILE_K×TILE_N
  config->colsb[2] = TILE_N * 4; // TILE_N * 4 bytes per row
  config->rows[2] = TILE_K / 2;  // TILE_K/2 rows

  // Load the tile configuration
  amx_tile_loadconfig((uniform int8 * uniform) config);
}

// Release AMX tiles
export void release_amx() { amx_tile_release(); }

// Simple B tile conversion using direct interleaving
void convert_b_tile(uniform float16 b_src[], uniform float16 b_amx[],
                    uniform int b_offset, uniform int stride_b) {
  // Process pairs of rows with simple interleaving
  for (uniform int tile_row = 0; tile_row < TILE_K / 2; ++tile_row) {
    uniform int row0_offset = b_offset + (tile_row * 2) * stride_b;
    uniform int row1_offset = b_offset + (tile_row * 2 + 1) * stride_b;
    uniform int amx_row_offset = tile_row * TILE_N * 2;

    // Simple interleaving: row0[0], row1[0], row0[1], row1[1], ...
    for (uniform int i = 0; i < TILE_N; ++i) {
      b_amx[amx_row_offset + i * 2] = b_src[row0_offset + i];
      b_amx[amx_row_offset + i * 2 + 1] = b_src[row1_offset + i];
    }
  }
}

// AMX-FP16 matrix multiplication with tiling
export void amx_matmul_fp16_ispc(uniform float16 a[], uniform float16 b[],
                                 uniform float c[], uniform int m,
                                 uniform int n, uniform int k,
                                 uniform int stride_a, uniform int stride_b,
                                 uniform int stride_c) {

  // Allocate temporary buffer for B tile
  uniform float16 __attribute__((aligned(64))) b_amx[TILE_K * TILE_N];

  // Process matrix in tiles - single tile at a time
  for (uniform int tile_i = 0; tile_i < TILES_M; ++tile_i) {
    for (uniform int tile_j = 0; tile_j < TILES_N; ++tile_j) {

      // Zero C tile directly in AMX register
      amx_tile_zero(TMM_C);

      // Accumulate across K dimension
      for (uniform int tile_k = 0; tile_k < TILES_K; ++tile_k) {

        // Calculate positions
        uniform int a_start_row = tile_i * TILE_M;
        uniform int a_start_col = tile_k * TILE_K;
        uniform int a_offset = a_start_row * stride_a + a_start_col;
        uniform int64 a_byte_stride = stride_a * 2;

        uniform int b_start_row = tile_k * TILE_K;
        uniform int b_start_col = tile_j * TILE_N;
        uniform int b_offset = b_start_row * stride_b + b_start_col;

        // Convert B tile from source matrix to AMX format
        convert_b_tile(b, b_amx, b_offset, stride_b);

        // Load A and B tiles, then perform multiply-accumulate
        amx_tile_load(TMM_A, (uniform int8 * uniform) & a[a_offset],
                 a_byte_stride);
        amx_tile_load(TMM_B, (uniform int8 * uniform) & b_amx[0],
                 (uniform int64)(TILE_N * 4));
        amx_dpfp16ps(TMM_C, TMM_A, TMM_B);
      }

      // Store final C tile result to destination matrix
      uniform int c_offset = (tile_i * TILE_M) * stride_c + (tile_j * TILE_N);
      uniform int64 c_byte_stride = stride_c * 4;
      amx_tile_store(TMM_C, (uniform int8 * uniform) & c[c_offset],
                c_byte_stride);
    }
  }
}

// Naive matrix multiplication for comparison
export void naive_matmul_fp16_ispc(uniform float16 a[], uniform float16 b[],
                                   uniform float c[], uniform int m,
                                   uniform int n, uniform int k,
                                   uniform int stride_a, uniform int stride_b,
                                   uniform int stride_c) {
  for (uniform int i = 0; i < m; ++i) {
    for (uniform int j = 0; j < n; ++j) {
      uniform float sum = 0.0f;
      for (uniform int kk = 0; kk < k; ++kk) {
        uniform float a_val = (float)a[i * stride_a + kk];
        uniform float b_val = (float)b[kk * stride_b + j];
        sum += a_val * b_val;
      }
      c[i * stride_c + j] = sum;
    }
  }
}

// Verify results match between AMX and naive implementations (with tolerance)
export uniform bool verify_results_fp16(uniform float result_amx[],
                                        uniform float result_naive[],
                                        uniform int size,
                                        uniform float tolerance) {
  uniform int mismatches = 0;
  for (uniform int i = 0; i < size; ++i) {
    uniform float diff = result_amx[i] - result_naive[i];
    if (diff < 0)
      diff = -diff; // abs
    if (diff > tolerance) {
      if (mismatches < 10) { // Limit output
        print("Mismatch at position %: AMX=%, Naive=%, diff=%\n", i,
              result_amx[i], result_naive[i], diff);
      }
      mismatches++;
    }
  }
  if (mismatches > 10) {
    print("... and % more mismatches\n", mismatches - 10);
  }
  return mismatches == 0;
}

// Initialize input matrices A and B with FP16-friendly random values
export void init_input_matrices(uniform float16 a[], uniform float16 b[],
                                uniform int m, uniform int n, uniform int k) {
  // Use simple LCG for deterministic pseudo-random values
  uniform uint32 seed = 12345;

  // Initialize A with small random values in range [-0.5, 0.5]
  for (uniform int i = 0; i < m; ++i) {
    for (uniform int j = 0; j < k; ++j) {
      seed = seed * 1664525 + 1013904223; // LCG parameters
      uniform float val =
          (float)(seed & 0xFFFF) / 65536.0f - 0.5f; // [-0.5, 0.5]
      a[i * k + j] = (float16)val;
    }
  }

  // Initialize B with small random values in range [-0.5, 0.5]
  for (uniform int i = 0; i < k; ++i) {
    for (uniform int j = 0; j < n; ++j) {
      seed = seed * 1664525 + 1013904223; // LCG parameters
      uniform float val =
          (float)(seed & 0xFFFF) / 65536.0f - 0.5f; // [-0.5, 0.5]
      b[i * n + j] = (float16)val;
    }
  }
}

// Zero out matrix C
export void zero_matrix(uniform float c[], uniform int m, uniform int n) {
  foreach (i = 0 ... m * n) {
    c[i] = 0.0f;
  }
}

// Print FP16 matrix
export void print_matrix_fp16(uniform float16 a[], uniform int m, uniform int n,
                              uniform int stride, uniform int max_print_rows,
                              uniform int max_print_cols) {
  print("Matrix (%x%):\n", m, n);
  for (uniform int i = 0; i < m && i < max_print_rows; ++i) {
    for (uniform int j = 0; j < n && j < max_print_cols; ++j) {
      print("% ", (float)a[i * stride + j]);
    }
    if (n > max_print_cols)
      print(" ...");
    print("\n");
  }
  if (m > max_print_rows)
    print("...\n");
  print("\n");
}

// Print FP32 matrix
export void print_matrix_fp32(uniform float c[], uniform int m, uniform int n,
                              uniform int stride, uniform int max_print_rows,
                              uniform int max_print_cols) {
  print("Matrix (%x%):\n", m, n);
  for (uniform int i = 0; i < m && i < max_print_rows; ++i) {
    for (uniform int j = 0; j < n && j < max_print_cols; ++j) {
      print("% ", c[i * stride + j]);
    }
    if (n > max_print_cols)
      print(" ...");
    print("\n");
  }
  if (m > max_print_rows)
    print("...\n");
  print("\n");
}
