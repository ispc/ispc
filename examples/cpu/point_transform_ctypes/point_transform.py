#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

import ctypes
import os
import platform
import numpy as np
import time
import math

# Define the Transform structure in Python to match the ISPC struct
class Transform(ctypes.Structure):
    _fields_ = [
        ("scale_x", ctypes.c_float),
        ("scale_y", ctypes.c_float),
        ("translate_x", ctypes.c_float),
        ("translate_y", ctypes.c_float),
        ("rotation", ctypes.c_float)
    ]

# Determine the file extension based on the platform
if platform.system() == 'Windows':
    lib_extension = '.dll'
    lib_filename = 'point_transform_ctypes' + lib_extension
elif platform.system() == 'Darwin':  # macOS
    lib_extension = '.dylib'
    lib_filename = 'libpoint_transform_ctypes' + lib_extension
else:  # Linux and others
    lib_extension = '.so'
    lib_filename = 'libpoint_transform_ctypes' + lib_extension

# Path to the compiled library
lib_path = os.path.join(os.getcwd(), lib_filename)

# Load the library
try:
    ispc_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading library: {e}")
    print("Make sure you've compiled the ISPC file first!")
    exit(1)

def transform_points_ispc(points_x, points_y, transform_params, strength):
    """
    Apply transformation to points using ISPC.
    
    Parameters:
    - points_x, points_y: NumPy arrays of point coordinates
    - transform_params: dict with scale_x, scale_y, translate_x, translate_y, rotation
    - strength: float controlling the intensity of the transformation
    
    Returns:
    - result_x, result_y: transformed point coordinates
    """
    points_x = np.asarray(points_x, dtype=np.float32)
    points_y = np.asarray(points_y, dtype=np.float32)
    
    # Create output arrays
    result_x = np.zeros_like(points_x)
    result_y = np.zeros_like(points_y)
    
    # Get pointers to the data
    points_x_ptr = points_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    points_y_ptr = points_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_x_ptr = result_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_y_ptr = result_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Create the transform structure
    transform = Transform(
        scale_x=transform_params['scale_x'],
        scale_y=transform_params['scale_y'],
        translate_x=transform_params['translate_x'],
        translate_y=transform_params['translate_y'],
        rotation=transform_params['rotation']
    )
    
    # Define function signature
    ispc_lib.transform_points.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(Transform),
        ctypes.c_float,
        ctypes.c_int
    ]
    ispc_lib.transform_points.restype = None
    
    # Call the ISPC function
    ispc_lib.transform_points(
        points_x_ptr,
        points_y_ptr,
        result_x_ptr,
        result_y_ptr,
        ctypes.byref(transform),
        ctypes.c_float(strength),
        ctypes.c_int(len(points_x))
    )
    
    return result_x, result_y



def transform_points_numpy(points_x, points_y, transform_params, strength):
    """NumPy implementation for point transformation"""
    cos_theta = math.cos(transform_params['rotation'])
    sin_theta = math.sin(transform_params['rotation'])
    
    translate_x = transform_params['translate_x'] * strength
    translate_y = transform_params['translate_y'] * strength
    
    # Apply scaling (reuse arrays for better memory usage)
    x = points_x * transform_params['scale_x']
    y = points_y * transform_params['scale_y']
    
    # Apply rotation
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    
    # Apply translation
    result_x = x_rot + translate_x
    result_y = y_rot + translate_y
    
    return result_x, result_y

if __name__ == "__main__":
    # Generate random points
    num_points = 1000000
    points_x = np.random.uniform(-10, 10, num_points).astype(np.float32)
    points_y = np.random.uniform(-10, 10, num_points).astype(np.float32)
    
    # Define transformation parameters
    transform_params = {
        'scale_x': 1.5,
        'scale_y': 0.8,
        'translate_x': 2.0,
        'translate_y': -1.0,
        'rotation': 0.3  # radians (about 17 degrees)
    }
    
    # Set transformation strength
    strength = 0.75
    

    start = time.perf_counter()
    ispc_result_x, ispc_result_y = transform_points_ispc(
        points_x, points_y, transform_params, strength
    )
    ispc_time = time.perf_counter() - start
    print(f"ISPC/ctypes time: {ispc_time:.9f} seconds")
    
    # Measure NumPy performance
    start = time.perf_counter()
    numpy_result_x, numpy_result_y = transform_points_numpy(
        points_x, points_y, transform_params, strength
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
        print(f"ISPC/ctypes speedup: {numpy_time / ispc_time:.2f}x")
    else:
        print(f"NumPy was faster by: {ispc_time / numpy_time:.2f}x")