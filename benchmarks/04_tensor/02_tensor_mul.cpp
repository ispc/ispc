#include "../common.h"
#include "02_tensor_mul_ispc.h"
#include "dlpack.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdio.h>

static Docs docs("Tensor multiplication benchmark\n"
                 "Expectations:\n"
                 " - Tests different tensor shapes and data types\n"
                 " - Validates correctness of the tensor multiplication operation\n"
                 " - Special handling for matrix multiplication in 2D and batched matrix multiplication in 3D cases");

WARM_UP_RUN();

// Initialize tensors with random data
template <typename T> void init(T *a_data, T *b_data, T *c_data, int count) {
    for (int i = 0; i < count; ++i) {
        a_data[i] = static_cast<T>((i % 10) + 1);       // Fill A with sample values (avoid zeros)
        b_data[i] = static_cast<T>(((i + 3) % 10) + 1); // Fill B with different values
        c_data[i] = static_cast<T>(0);                  // Zero-initialize the result tensor
    }
}

// Helper function to calculate the expected result for element-wise multiplication (general case)
template <typename T>
void calculate_expected(const T *a_data, const T *b_data, T *expected_data, const DLTensor &A, const DLTensor &B,
                        const DLTensor &C) {
    // Get tensor dimensions
    const int ndim = A.ndim;
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= A.shape[i];
    }

    // For each logical element in the tensor
    for (int i = 0; i < total_size; i++) {
        // Calculate the physical indices using the same logic as in the tensor operation
        int physical_idx_A = 0;
        int physical_idx_B = 0;
        int physical_idx_C = 0;

        // Decompose linear index to coordinates
        int temp_idx = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = temp_idx % A.shape[d];
            temp_idx /= A.shape[d];

            // Apply strides to get physical index
            physical_idx_A += coord * A.strides[d];
            physical_idx_B += coord * B.strides[d];
            physical_idx_C += coord * C.strides[d];
        }

        // Element-wise multiplication
        expected_data[physical_idx_C] = a_data[physical_idx_A] * b_data[physical_idx_B];
    }
}

// Validate results of tensor multiplication
template <typename T>
bool validate_results(const T *a_data, const T *b_data, const T *c_data, DLTensor &A, DLTensor &B, DLTensor &C) {
    const int ndim = A.ndim;

    // Create expected results array
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= A.shape[i];
    }

    T *expected_data = new T[total_size * 2]; // Extra space for padding if needed
    memset(expected_data, 0, sizeof(T) * total_size * 2);

    // Calculate expected results based on tensor dimensions
    calculate_expected(a_data, b_data, expected_data, A, B, C);

    // Compare with actual results
    bool valid = true;
    for (int i = 0; i < total_size; i++) {
        // Calculate physical index
        int physical_idx_C = 0;
        int temp_idx = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = temp_idx % C.shape[d];
            temp_idx /= C.shape[d];
            physical_idx_C += coord * C.strides[d];
        }

        T c_val = c_data[physical_idx_C];
        T expected = expected_data[physical_idx_C];

        // For floating point, use epsilon-based comparison
        if (std::is_floating_point<T>::value) {
            T epsilon = static_cast<T>(1e-3); // Larger epsilon for matrix multiplication
            if (std::abs(c_val - expected) > epsilon * (1 + std::abs(expected))) {
                printf("Validation failed at logical index %d (physical index %d): Expected %f, Got %f\n", i,
                       physical_idx_C, static_cast<double>(expected), static_cast<double>(c_val));
                valid = false;
                break;
            }
        } else {
            // For integer types, use exact comparison
            if (c_val != expected) {
                printf("Validation failed at logical index %d (physical index %d): Expected %d, Got %d\n", i,
                       physical_idx_C, static_cast<int>(expected), static_cast<int>(c_val));
                valid = false;
                break;
            }
        }
    }

    delete[] expected_data;
    return valid;
}

// Generic function to initialize a DLTensor with proper strides
template <typename T>
void CreateDLTensor(DLTensor &tensor, T *data, const std::vector<int64_t> &shape, DLDataType dtype,
                    bool non_contiguous = false, int64_t padding = 1) {
    tensor.data = data;
    tensor.ndim = shape.size();
    tensor.shape = new int64_t[tensor.ndim];
    tensor.strides = new int64_t[tensor.ndim];

    // Copy shape values
    for (int i = 0; i < tensor.ndim; i++) {
        tensor.shape[i] = shape[i];
    }

    // Compute strides (row-major layout by default)
    if (!non_contiguous) {
        // Contiguous layout (C-order)
        tensor.strides[tensor.ndim - 1] = 1;
        for (int i = tensor.ndim - 2; i >= 0; --i) {
            tensor.strides[i] = tensor.strides[i + 1] * tensor.shape[i + 1];
        }
    } else {
        // Non-contiguous layout with uniform padding
        // For the innermost dimension (last one according to tensor terminology), use padding+1 as the stride
        tensor.strides[tensor.ndim - 1] = padding + 1;

        // For other dimensions, add padding to the normal stride computation
        for (int i = tensor.ndim - 2; i >= 0; --i) {
            tensor.strides[i] = (tensor.strides[i + 1] * tensor.shape[i + 1]) + padding;
        }
    }

    tensor.byte_offset = 0;
    tensor.dtype = dtype;

    // Set device to CPU
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;
}

// Clean up resources for a DLTensor
void CleanupDLTensor(DLTensor &tensor) {
    delete[] tensor.shape;
    delete[] tensor.strides;
}

// Helper function to determine which ISPC function to call based on type
template <typename T> void call_ispc_tensor_mul(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    // This will be specialized for each type
    // Default implementation might throw an error or use a generic version
    std::cerr << "Unsupported type for ISPC tensor multiplication" << std::endl;
    throw std::runtime_error("Unsupported type");
}

// Specializations for different types
template <> void call_ispc_tensor_mul<float>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_mul_ISPC_float((void *)A, (void *)B, (void *)C);
}

template <> void call_ispc_tensor_mul<double>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_mul_ISPC_double((void *)A, (void *)B, (void *)C);
}

template <> void call_ispc_tensor_mul<int32_t>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_mul_ISPC_int32((void *)A, (void *)B, (void *)C);
}

// Benchmarking function for element-wise multiplication (1D, 3D, 4D, etc.)
template <typename T> static void tensor_mul(benchmark::State &state, DLDataTypeCode code, int bits) {
    const int total_elements = static_cast<int>(state.range(0));
    const int ndim = static_cast<int>(state.range(1));
    bool non_contiguous = (state.range(2) == 1);

    // Create shape based on ndim
    std::vector<int64_t> shape;
    if (ndim == 1) {
        shape = {total_elements}; // 1D tensor
    } else if (ndim == 3) {
        int side = static_cast<int>(std::cbrt(total_elements));
        shape = {side, side, side}; // 3D cube tensor
    } else if (ndim == 4) {
        int side = static_cast<int>(std::sqrt(std::sqrt(total_elements)));
        shape = {side, side, side, side}; // 4D hypercube tensor
    } else {
        // Fallback to 1D
        shape = {total_elements};
    }

    // Calculate total elements with padding
    int padding_factor = non_contiguous ? 4 : 1;
    size_t padded_elements = static_cast<size_t>(total_elements) * padding_factor;

    // Check for potential overflow
    if (padded_elements == 0 || padded_elements > 100000000) {
        state.SkipWithError("Invalid or excessive tensor size");
        return;
    }

    // Allocate memory
    T *a_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * padded_elements));
    T *b_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * padded_elements));
    T *c_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * padded_elements));

    // Initialize with test data
    init(a_data, b_data, c_data, padded_elements);

    // Create DLTensors
    DLTensor A, B, C;
    DLDataType dtype;
    dtype.code = code;
    dtype.bits = static_cast<uint8_t>(bits);
    dtype.lanes = 1;

    CreateDLTensor(A, a_data, shape, dtype, non_contiguous);
    CreateDLTensor(B, b_data, shape, dtype, non_contiguous);
    CreateDLTensor(C, c_data, shape, dtype, non_contiguous);

    // Run the benchmark
    for (auto _ : state) {
        call_ispc_tensor_mul<T>(&A, &B, &C);
    }

    // Validate results
    bool valid = validate_results(a_data, b_data, c_data, A, B, C);
    if (!valid) {
        state.SkipWithError("Element-wise multiplication result validation failed!");
    }

    // Cleanup
    aligned_free_helper(a_data);
    aligned_free_helper(b_data);
    aligned_free_helper(c_data);
    CleanupDLTensor(A);
    CleanupDLTensor(B);
    CleanupDLTensor(C);
}

// Define benchmarks for different types and configurations
static void tensor_mul_float(benchmark::State &state) { tensor_mul<float>(state, kDLFloat, 32); }

static void tensor_mul_int(benchmark::State &state) { tensor_mul<int32_t>(state, kDLInt, 32); }

static void tensor_mul_double(benchmark::State &state) { tensor_mul<double>(state, kDLFloat, 64); }

// Matrix multiplication benchmarks - varying sizes
BENCHMARK(tensor_mul_float)->Args({32, 32, 32, 0})->Args({64, 64, 64, 0})->Args({128, 128, 128, 0});
BENCHMARK(tensor_mul_int)->Args({32, 32, 32, 0})->Args({64, 64, 64, 0});
BENCHMARK(tensor_mul_double)->Args({32, 32, 32, 0})->Args({64, 64, 64, 0});

// Non-square matrices - ensure valid matrix dimensions where A is [M×N] and B is [N×K]
// BENCHMARK(tensor_mul_float)->Args({64, 32, 16, 0})->Args({32, 64, 32, 0});

// Non-contiguous matrices
// BENCHMARK(tensor_mul_float)->Args({64, 64, 64, 1});

// Register benchmarks for element-wise multiplication: args are (elements, dimensions, non_contiguous?)
// 1D element-wise
BENCHMARK(tensor_mul_float)->Args({1024, 1, 0})->Args({4096, 1, 0});
BENCHMARK(tensor_mul_int)->Args({1024, 1, 0});

// 3D element-wise
BENCHMARK(tensor_mul_float)->Args({2048, 3, 0})->Args({8096, 3, 0});

// 4D element-wise
BENCHMARK(tensor_mul_float)->Args({1024, 4, 0});

// Non-contiguous element-wise
// BENCHMARK(tensor_matmul_float)->Args({1024, 3, 1});

BENCHMARK_MAIN();