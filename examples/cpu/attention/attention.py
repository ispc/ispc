#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import time
import math
import ispc_attention  # Import the module name we defined in our CMakeLists.txt

def single_head_attention_pytorch(Q, K, V):
    """PyTorch implementation of single-head attention"""
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q)
        K = torch.from_numpy(K)
        V = torch.from_numpy(V)
    
    # Get dimensions
    seq_len, d_model = Q.shape
    
    # Compute scaled dot-product attention
    # In new version of Pytorch there is a predefined function for scaled dot product attention scaled_dot_product_attention
    scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d_model)
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)    
    return output.numpy() if isinstance(output, torch.Tensor) else output

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
    
    for sl in seq_lengths:
        print(f"\nTesting sequence length {sl}:")
        # Generate random input matrices
        Q_test = np.random.rand(sl, d_model).astype(np.float32)
        K_test = np.random.rand(sl, d_model).astype(np.float32)
        V_test = np.random.rand(sl, d_model).astype(np.float32)
        
        # Prepare output array
        output_ispc_test = np.zeros((sl, d_model), dtype=np.float32)
        
        # Run ISPC implementation
        start = time.perf_counter()
        ispc_attention.single_head_attention(
            Q_test, K_test, V_test, output_ispc_test, sl, d_model
        )
        ispc_time = time.perf_counter() - start
        
        # Run PyTorch implementation
        start = time.perf_counter()
        output_pytorch_test = single_head_attention_pytorch(Q_test, K_test, V_test)
        pytorch_time = time.perf_counter() - start
        
        # Verify results
        pytorch_match = np.allclose(output_ispc_test, output_pytorch_test, rtol=1e-5, atol=1e-5)
        print(f"  - Results match: {pytorch_match}")
        
        if not pytorch_match:
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
        
        print(f"  - ISPC time: {ispc_time:.6f}s, PyTorch time: {pytorch_time:.6f}s, Speedup: {pytorch_time / ispc_time:.2f}x")

    print("\nSummary: Testing completed for all sequence lengths.")