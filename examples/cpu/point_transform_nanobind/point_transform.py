#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import time
import math
import ispc_transform  # Import the module name we defined in our CMakeLists.txt

def transform_points_numpy(points_x, points_y, transform_params, strength):
    """NumPy implementation for point transformation"""
    cos_theta = math.cos(transform_params.rotation)
    sin_theta = math.sin(transform_params.rotation)

    translate_x = transform_params.translate_x * strength
    translate_y = transform_params.translate_y * strength

    # Apply scaling (reuse arrays for better memory usage)
    x = points_x * transform_params.scale_x
    y = points_y * transform_params.scale_y

    # Apply rotation
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta

    # Apply translation
    result_x = x_rot + translate_x
    result_y = y_rot + translate_y

    return result_x, result_y

# Example usage
if __name__ == "__main__":
    # Generate random points
    num_points = 1000000
    points_x = np.random.uniform(-10, 10, num_points).astype(np.float32)
    points_y = np.random.uniform(-10, 10, num_points).astype(np.float32)

    # Create a transform object
    transform = ispc_transform.Transform()
    transform.scale_x = 1.5
    transform.scale_y = 0.8
    transform.translate_x = 2.0
    transform.translate_y = -1.0
    transform.rotation = 0.3  # radians (about 17 degrees)

    # Set transformation strength
    strength = 0.75

    # Create result arrays for the ISPC implementation
    ispc_result_x = np.zeros_like(points_x)
    ispc_result_y = np.zeros_like(points_y)

    start = time.perf_counter()
    ispc_transform.transform_points(
        points_x, points_y,
        ispc_result_x, ispc_result_y,
        transform, strength
    )
    ispc_time = time.perf_counter() - start
    print(f"ISPC/nanobind time: {ispc_time:.9f} seconds")

    # Measure NumPy performance
    start = time.perf_counter()
    numpy_result_x, numpy_result_y = transform_points_numpy(
        points_x, points_y,
        transform, strength
    )
    numpy_time = time.perf_counter() - start
    print(f"NumPy time: {numpy_time:.9f} seconds")

    # Verify results
    if np.allclose(ispc_result_x, numpy_result_x, rtol=1e-5, atol=1e-5) and \
       np.allclose(ispc_result_y, numpy_result_y, rtol=1e-5, atol=1e-5):
        print("Results match within tolerance!")
    else:
        print("Warning: Results differ more than the tolerance!")

    if numpy_time > ispc_time:
        print(f"ISPC/nanobind speedup vs NumPy: {numpy_time / ispc_time:.2f}x")
    else:
        print(f"NumPy was faster than ISPC/nanobind by: {ispc_time / numpy_time:.2f}x")
