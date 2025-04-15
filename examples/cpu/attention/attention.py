#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

# Run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset --cpu-list 0 python attention.py to check performance on 1 thread
import numpy as np
import torch
import time
import math
import ispc_attention  # Import the module name we defined in our CMakeLists.txt

# Number of iterations for reliable performance measurement
NUM_ITERATIONS = 4
# Warm-up iterations before timing
WARM_UP_ITERATIONS = 3

@torch.compile(backend="inductor", mode="reduce-overhead")
def single_head_attention_pytorch(Q, K, V):
    """PyTorch implementation of single-head attention (CPU)"""
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q).to("cpu")
        K = torch.from_numpy(K).to("cpu")
        V = torch.from_numpy(V).to("cpu")
    else:
        # Ensure tensors are on CPU even if they're already PyTorch tensors
        Q = Q.to("cpu")
        K = K.to("cpu")
        V = V.to("cpu")

    # Get dimensions
    seq_len, d_model = Q.shape

    # Compute scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d_model)
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    # Ensure the output is detached and on CPU before converting to numpy
    return output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output

def calculate_attention_flops(seq_len, d_model):
    """Calculate the number of floating-point operations for attention mechanism"""
    # Q*K^T operation: seq_len * seq_len * d_model multiplications and additions
    qk_flops = 2 * seq_len * seq_len * d_model

    # Softmax: neglecting the division by sqrt(d_model) and exp operations as they're relatively small
    # compared to matrix multiplications

    # attention_weights * V: seq_len * seq_len
    av_flops = 3 * seq_len * seq_len

    # Total FLOPs
    total_flops = qk_flops + av_flops

    return total_flops

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Define dimensions
    d_model = 512

    # Test different sequence lengths
    print("\nPerformance and validation for different sequence lengths:")
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Print header for results table
    print("\n{:<8} {:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Seq Len", "ISPC Time (ms)", "PyTorch Time (ms)", "Speedup", "GFLOPS ISPC", "GFLOPS PyTorch"))
    print("-" * 75)

    for sl in seq_lengths:
        # Generate random input matrices
        Q_test = np.random.rand(sl, d_model).astype(np.float32)
        K_test = np.random.rand(sl, d_model).astype(np.float32)
        V_test = np.random.rand(sl, d_model).astype(np.float32)

        # Prepare output array
        output_ispc_test = np.zeros((sl, d_model), dtype=np.float32)

        # Calculate theoretical FLOPs
        flops = calculate_attention_flops(sl, d_model)

        # Warm up ISPC implementation
        for _ in range(WARM_UP_ITERATIONS):
            ispc_attention.single_head_attention(
                Q_test, K_test, V_test, output_ispc_test, sl, d_model
            )

        # Run ISPC implementation with multiple iterations
        ispc_times = []
        for i in range(NUM_ITERATIONS):
            start = time.perf_counter()
            ispc_attention.single_head_attention(
                Q_test, K_test, V_test, output_ispc_test, sl, d_model
            )
            ispc_times.append(time.perf_counter() - start)

        ispc_time_min = min(ispc_times)
        ispc_gflops = (flops / 1e9) / ispc_time_min  # Convert to GFLOPS

        # Warm up PyTorch implementation
        for _ in range(WARM_UP_ITERATIONS):
            _ = single_head_attention_pytorch(Q_test, K_test, V_test)

        # Run PyTorch implementation with multiple iterations
        pytorch_times = []
        for i in range(NUM_ITERATIONS):
            start = time.perf_counter()
            output_pytorch_test = single_head_attention_pytorch(Q_test, K_test, V_test)
            pytorch_times.append(time.perf_counter() - start)

        pytorch_time_min = min(pytorch_times)
        pytorch_gflops = (flops / 1e9) / pytorch_time_min  # Convert to GFLOPS

        # Verify results
        pytorch_match = np.allclose(output_ispc_test, output_pytorch_test, rtol=1e-5, atol=1e-5)

        # Print performance results in table format
        print("{:<8} {:<15.3f} {:<15.3f} {:<12.2f} {:<12.2f} {:<12.2f}".format(
            sl,
            ispc_time_min * 1000,  # Convert to ms
            pytorch_time_min * 1000,  # Convert to ms
            pytorch_time_min / ispc_time_min,
            ispc_gflops,
            pytorch_gflops
        ))

        # Print validation results separately for clarity
        if not pytorch_match:
            print(f"\nValidation issues for sequence length {sl}:")
            max_diff = np.max(np.abs(output_ispc_test - output_pytorch_test))
            print(f"  - Maximum difference: {max_diff}")

            # Calculate relative error for more insight
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_error = np.abs((output_ispc_test - output_pytorch_test) / output_pytorch_test)
                mean_rel_error = np.nanmean(rel_error) * 100  # Convert to percentage
                print(f"  - Mean relative error: {mean_rel_error:.6f}%")

            # Check where the largest differences are
            max_diff_idx = np.unravel_index(np.argmax(np.abs(output_ispc_test - output_pytorch_test)), output_ispc_test.shape)
            print(f"  - Largest difference at index: {max_diff_idx}")
            print(f"  - ISPC value: {output_ispc_test[max_diff_idx]}, PyTorch value: {output_pytorch_test[max_diff_idx]}")

            # Analyze if differences are systematic or random
            diff_matrix = np.abs(output_ispc_test - output_pytorch_test)
            print(f"  - Number of elements with diff > 1e-4: {np.sum(diff_matrix > 1e-4)} out of {diff_matrix.size}")
            print(f"  - Number of elements with diff > 1e-3: {np.sum(diff_matrix > 1e-3)} out of {diff_matrix.size}")

    # Print detailed statistics for the largest size
    largest_sl = seq_lengths[-1]
    print(f"\nDetailed statistics for sequence length {largest_sl}:")
    print(f"  - ISPC implementation:")
    print(f"    * Min time: {min(ispc_times) * 1000:.3f} ms")
    print(f"    * Max time: {max(ispc_times) * 1000:.3f} ms")
    print(f"    * Std dev: {np.std(ispc_times) * 1000:.3f} ms")
    print(f"  - PyTorch implementation:")
    print(f"    * Min time: {min(pytorch_times) * 1000:.3f} ms")
    print(f"    * Max time: {max(pytorch_times) * 1000:.3f} ms")
    print(f"    * Std dev: {np.std(pytorch_times) * 1000:.3f} ms")

    print("\nSummary: Testing completed for all sequence lengths.")