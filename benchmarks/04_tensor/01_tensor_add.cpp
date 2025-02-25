#include "../common.h"
#include "01_tensor_add_ispc.h"
#include "dlpack.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

static Docs docs("Tensor addition benchmark\n"
                 "Expectations:\n"
                 " - Tests different tensor shapes and data types\n"
                 " - Validates correctness of the tensor addition operation");

WARM_UP_RUN();

// Initialize tensors with random data
template <typename T> void init(T *a_data, T *b_data, T *c_data, int count) {
    for (int i = 0; i < count; ++i) {
        a_data[i] = static_cast<T>(i % 100);        // Fill A with sample values
        b_data[i] = static_cast<T>((i + 50) % 100); // Fill B with different values
        c_data[i] = static_cast<T>(0);              // Zero-initialize the result tensor
    }
}

// Validate results of tensor addition
template <typename T>
bool validate_results(const T *a_data, const T *b_data, const T *c_data, DLTensor &A, DLTensor &B, DLTensor &C) {
    // Get tensor dimensions and properties
    const int ndim = A.ndim;
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= A.shape[i];
    }

    // For each logical element in the tensor
    for (int i = 0; i < total_size; i++) {
        // Calculate the physical indices using the same logic as the tensor operation
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

        // Get values at the correct physical locations
        T a_val = a_data[physical_idx_A];
        T b_val = b_data[physical_idx_B];
        T c_val = c_data[physical_idx_C];
        T expected = a_val + b_val;

        // For floating point, use epsilon-based comparison
        if (std::is_floating_point<T>::value) {
            T epsilon = static_cast<T>(1e-5);
            if (std::abs(c_val - expected) > epsilon) {
                printf("Validation failed at logical index %d (physical index %d): Expected %f, Got %f\n", i,
                       physical_idx_C, static_cast<double>(expected), static_cast<double>(c_val));
                printf("A[%d]=%f, B[%d]=%f\n", physical_idx_A, static_cast<double>(a_val), physical_idx_B,
                       static_cast<double>(b_val));
                return false;
            }
        } else {
            // For integer types, use exact comparison
            if (c_val != expected) {
                printf("Validation failed at logical index %d (physical index %d): Expected %d, Got %d\n", i,
                       physical_idx_C, static_cast<int>(expected), static_cast<int>(c_val));
                printf("A[%d]=%d, B[%d]=%d\n", physical_idx_A, static_cast<int>(a_val), physical_idx_B,
                       static_cast<int>(b_val));
                return false;
            }
        }
    }
    return true;
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
template <typename T> void call_ispc_tensor_add(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    // This will be specialized for each type
    // Default implementation might throw an error or use a generic version
    std::cerr << "Unsupported type for ISPC tensor addition" << std::endl;
    throw std::runtime_error("Unsupported type");
}

// Specializations for different types
template <> void call_ispc_tensor_add<float>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_add_ISPC_float((void *)A, (void *)B, (void *)C);
}

template <> void call_ispc_tensor_add<double>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_add_ISPC_double((void *)A, (void *)B, (void *)C);
}

template <> void call_ispc_tensor_add<int32_t>(const DLTensor *A, const DLTensor *B, DLTensor *C) {
    ispc::tensor_add_ISPC_int32((void *)A, (void *)B, (void *)C);
}

// Benchmarking function for ISPC tensor addition with validation
template <typename T> static void tensor_add(benchmark::State &state, DLDataTypeCode code, int bits) {
    const int count = static_cast<int>(state.range(0));
    const int ndim = static_cast<int>(state.range(1));

    // Create tensor shapes based on ndim
    std::vector<int64_t> shape;
    if (ndim == 1) {
        shape = {count}; // 1D tensor
    } else if (ndim == 2) {
        int side = static_cast<int>(std::sqrt(count));
        shape = {side, side}; // 2D square tensor
    } else if (ndim == 3) {
        int side = static_cast<int>(std::cbrt(count));
        shape = {side, side, side}; // 3D cube tensor
    } else {
        // Fallback to 1D
        shape = {count};
    }

    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    bool non_contiguous = (state.range(2) == 1);
    // Determine padding factor based on dimensions for non-contiguous case
    int padding_factor = 2; // Default for 1D
    if (non_contiguous) {
        // For higher dimensions, we need more memory
        if (ndim == 2)
            padding_factor = 3; // For 2D
        else if (ndim >= 3)
            padding_factor = 4; // For 3D and higher
    }

    // Allocate memory with padding for non-contiguous test
    T *a_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * total_elements * padding_factor));
    T *b_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * total_elements * padding_factor));
    T *c_data = static_cast<T *>(aligned_alloc_helper(sizeof(T) * total_elements * padding_factor));

    init(a_data, b_data, c_data, total_elements * padding_factor);

    // Create DLTensors with proper data type
    DLTensor A, B, C;
    // We will not use DLDataType but keep it for now for reference
    DLDataType dtype;
    dtype.code = code;
    dtype.bits = static_cast<uint8_t>(bits);
    dtype.lanes = 1;

    CreateDLTensor(A, a_data, shape, dtype, non_contiguous);
    CreateDLTensor(B, b_data, shape, dtype, non_contiguous);
    CreateDLTensor(C, c_data, shape, dtype, non_contiguous);

    // Run the benchmark
    for (auto _ : state) {
        call_ispc_tensor_add<T>(&A, &B, &C);
    }

    // Validate results
    bool valid = validate_results(a_data, b_data, c_data, A, B, C);
    if (!valid) {
        state.SkipWithError("Result validation failed!");
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
static void tensor_add_float(benchmark::State &state) { tensor_add<float>(state, kDLFloat, 32); }

static void tensor_add_int(benchmark::State &state) { tensor_add<int32_t>(state, kDLInt, 32); }

static void tensor_add_double(benchmark::State &state) { tensor_add<double>(state, kDLFloat, 64); }

// Register benchmarks: args are (elements, dimensions, non_contiguous?)
// 1D contiguous
BENCHMARK(tensor_add_float)->Args({1024, 1, 0})->Args({4096, 1, 0});
BENCHMARK(tensor_add_int)->Args({1024, 1, 0})->Args({4096, 1, 0});

// 2D contiguous
BENCHMARK(tensor_add_float)->Args({1024, 2, 0})->Args({4096, 2, 0});

// 3D contiguous
BENCHMARK(tensor_add_float)->Args({1000, 3, 0})->Args({8000, 3, 0});

// Non-contiguous tests
BENCHMARK(tensor_add_float)->Args({1024, 1, 1})->Args({4096, 2, 1});
BENCHMARK(tensor_add_double)->Args({1024, 2, 0})->Args({1024, 2, 1});

BENCHMARK_MAIN();