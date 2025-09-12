/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

#include "amx_matmul_ispc.h"
using namespace ispc;

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

const int A_ELEMENTS = MAT_SIZE_M * MAT_SIZE_K;
const int B_ELEMENTS = MAT_SIZE_K * MAT_SIZE_N;
const int C_ELEMENTS = MAT_SIZE_M * MAT_SIZE_N;

// AMX system call constants
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// Request AMX permissions from kernel
bool request_amx_permission() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    std::cout << "Failed to enable XFEATURE_XTILEDATA\n";
    std::cout << "Note: AMX not available on this system. Use Intel SDE to "
                 "test AMX functionality.\n";
    std::cout << "Command: sde -dmr -- ./amx_fp16_test\n";
    std::cout << "You can use SDE_PATH to point to your SDE installation.\n";
    return false;
  } else {
    std::cout << "TILE DATA USE SET - OK\n\n";
    return true;
  }
}

// High-resolution timer
double
get_time_diff(const std::chrono::high_resolution_clock::time_point &start,
              const std::chrono::high_resolution_clock::time_point &end) {
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  return duration.count() / 1e9;
}

int main() {
  std::cout << "Intel AMX-FP16 Matrix Multiplication Demo (ISPC)\n";
  std::cout << "A(" << MAT_SIZE_M << "x" << MAT_SIZE_K << ") * B(" << MAT_SIZE_K
            << "x" << MAT_SIZE_N << ") = C(" << MAT_SIZE_M << "x" << MAT_SIZE_N
            << ")\n";
  std::cout
      << "===============================================================\n\n";

  // Request AMX permissions
  if (!request_amx_permission()) {
    return -1;
  }

  // Allocate aligned memory for matrices
  // Matrix A: FP16 elements with padding
  uint16_t *a = (uint16_t *)aligned_alloc(64, A_ELEMENTS * sizeof(uint16_t));
  // Matrix B: FP16 elements with padding
  uint16_t *b = (uint16_t *)aligned_alloc(64, B_ELEMENTS * sizeof(uint16_t));
  // Matrix C: FP32 elements with padding
  float *c_amx = (float *)aligned_alloc(64, C_ELEMENTS * sizeof(float));
  float *c_naive = (float *)aligned_alloc(64, C_ELEMENTS * sizeof(float));

  if (!a || !b || !c_amx || !c_naive) {
    std::cerr << "Failed to allocate memory\n";
    return -1;
  }

  // Initialize matrices using ISPC functions
  std::cout << "Initializing matrices...\n";
  init_input_matrices((__fp16 *)a, (__fp16 *)b, MAT_SIZE_M, MAT_SIZE_N,
                      MAT_SIZE_K);
  zero_matrix(c_amx, MAT_SIZE_M, MAT_SIZE_N);
  zero_matrix(c_naive, MAT_SIZE_M, MAT_SIZE_N);
  std::cout << "Matrix initialization complete.\n\n";

  // Set up AMX tile configuration
  TileConfig config;
  init_amx_config(&config);

  // Warm-up run
  std::cout << "Performing warm-up...\n";
  amx_matmul_fp16_ispc((__fp16 *)a, (__fp16 *)b, c_amx, MAT_SIZE_M, MAT_SIZE_N,
                       MAT_SIZE_K, MAT_SIZE_K, MAT_SIZE_N, MAT_SIZE_N);
  // Reset C matrix for timing
  zero_matrix(c_amx, MAT_SIZE_M, MAT_SIZE_N);

  // Time both implementations
  std::cout << "Timing AMX implementation...\n";
  auto start = std::chrono::high_resolution_clock::now();
  amx_matmul_fp16_ispc((__fp16 *)a, (__fp16 *)b, c_amx, MAT_SIZE_M, MAT_SIZE_N,
                       MAT_SIZE_K, MAT_SIZE_K, MAT_SIZE_N, MAT_SIZE_N);
  auto end = std::chrono::high_resolution_clock::now();
  double amx_time = get_time_diff(start, end);

  std::cout << "Timing naive implementation...\n";
  start = std::chrono::high_resolution_clock::now();
  naive_matmul_fp16_ispc((__fp16 *)a, (__fp16 *)b, c_naive, MAT_SIZE_M,
                         MAT_SIZE_N, MAT_SIZE_K, MAT_SIZE_K, MAT_SIZE_N,
                         MAT_SIZE_N);
  end = std::chrono::high_resolution_clock::now();
  double naive_time = get_time_diff(start, end);

  // Verify results
  std::cout << "Verifying...\n";
  bool results_match = verify_results_fp16(c_amx, c_naive, C_ELEMENTS, 1e-2f);

  // Display matrices only if verification fails
  if (!results_match) {
    std::cout << "Verification FAILED. Showing matrices for debugging:\n\n";
    std::cout << "Input matrices (first 8x8 view):\n";
    print_matrix_fp16((__fp16 *)a, MAT_SIZE_M, MAT_SIZE_K, MAT_SIZE_K, 8, 8);
    print_matrix_fp16((__fp16 *)b, MAT_SIZE_K, MAT_SIZE_N, MAT_SIZE_N, 8, 8);
    std::cout << "Results (first 8x8 view):\n";
    print_matrix_fp32(c_amx, MAT_SIZE_M, MAT_SIZE_N, MAT_SIZE_N, 8, 8);
    print_matrix_fp32(c_naive, MAT_SIZE_M, MAT_SIZE_N, MAT_SIZE_N, 8, 8);
  } else {
    std::cout << "OK\n";
  }

  // Results
  std::cout << "\n====== RESULTS ======\n";
  std::cout << "Matrix: " << MAT_SIZE_M << "x" << MAT_SIZE_K << " * "
            << MAT_SIZE_K << "x" << MAT_SIZE_N << "\n";
  std::cout << "Correctness: " << (results_match ? "PASS" : "FAIL") << "\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "AMX time: " << amx_time << "s, Naive time: " << naive_time
            << "s\n";

  if (amx_time > 0) {
    double speedup = naive_time / amx_time;
    double total_ops = 2.0 * MAT_SIZE_M * MAT_SIZE_N * MAT_SIZE_K;
    std::cout << std::setprecision(2);
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Performance: " << (total_ops / amx_time) / 1e9
              << " GFLOPS (AMX)\n";
  }
  std::cout << "======================\n";

  // Release AMX resources
  release_amx();

  // Cleanup
  free(a);
  free(b);
  free(c_amx);
  free(c_naive);

  return results_match ? 0 : 1;
}
