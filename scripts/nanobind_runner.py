#!/usr/bin/env python3
"""
Isolated test runner for ISPC nanobind tests.
This module runs test functions in a separate subprocess to provide isolation
and avoid conflicts between tests.
"""

import sys
import os
import numpy
import importlib.util
import importlib.machinery

def call_test_function(module_name, test_sig, func_sig, width, verbose=False):
    """
    Call the test function from the compiled ISPC module.

    Args:
        module_name: Name of the compiled module
        test_sig: Test signature identifier (0-34)
        func_sig: Function signature string (e.g., 'f_v(')
        width: Vector width for the test
        verbose: Enable verbose output for debugging

    Returns:
        Status enum value indicating test result
    """
    try:
        # func_sig is 'f_v(', so substitute ( to _cpu_entry_point
        # because f_v_cpu_entry_point is the entry point
        function = func_sig.replace('(', '_cpu_entry_point')

        # Create a FileFinder for the current directory as we suppose that the
        # script is run from the root project directory.
        finder = importlib.machinery.FileFinder(
            './',
            (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
        )

        spec = finder.find_spec(module_name)
        if spec is None:
            raise ImportError(f"Module '{module_name}' not found")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        entry_func = getattr(module, function, None)

        # Check if the functions are found in the loaded module
        if test_sig != 7:
            if entry_func is None:
                raise ImportError(f"Function '{function}' not found in module '{module_name}'")

        # Prepare input data
        ARRAY_SIZE = 256
        res = numpy.zeros(ARRAY_SIZE, dtype=numpy.float32)

        dst = numpy.zeros(ARRAY_SIZE, dtype=numpy.float32)
        vfloat = numpy.arange(1, ARRAY_SIZE + 1, dtype=numpy.float32)
        vdouble = numpy.arange(1, ARRAY_SIZE + 1, dtype=numpy.float64)
        vint = numpy.array([2 * (i + 1) for i in range(ARRAY_SIZE)], dtype=numpy.int32)
        vint2 = numpy.array([i + 5 for i in range(ARRAY_SIZE)], dtype=numpy.int32)
        b = 5.0

        # Call corresponding TEST_SIG functions, it should correspond to test_static.cpp
        if test_sig == 0:
            entry_func(dst)
        elif test_sig == 1:
            entry_func(dst, vfloat)
        elif test_sig == 2:
            entry_func(dst, vfloat, b)
        elif test_sig == 3:
            entry_func(dst, vfloat, vint)
        elif test_sig == 4:
            entry_func(dst, vdouble, b)
        elif test_sig == 5:
            entry_func(dst, vdouble, b)
        elif test_sig == 6:
            entry_func(dst, vdouble, vint2)
        elif test_sig == 7:
            struct = getattr(module, f"v{width}_varying_f_sz")
            # TODO: python object has different size than ISPC struct, so just
            # check that we have expected class
            instance = struct()
            return True
        elif test_sig == 32:
            entry_func(b)
        elif test_sig == 33:
            entry_func(vfloat)
        elif test_sig == 34:
            entry_func(vfloat, b)

        if test_sig < 32:
            result_func = getattr(module, 'result_cpu_entry_point', None)
            if result_func is None:
                raise ImportError(f"Function 'result_cpu_entry_point' not found in module '{module_name}'")

            result_func(res)
            if numpy.array_equal(dst[:width], res[:width]):
                return True
            else:
                if verbose:
                    print(f"Test {module_name} failed: expected {res[:width]}, got {dst[:width]}")
                return False
        else:
            print_func = getattr(module, 'print_result_cpu_entry_point', None)

            if print_func is None:
                raise ImportError(f"Function 'print_result_cpu_entry_point' not found in module '{module_name}'")
            print_func()
            return True

    except Exception as e:
        if verbose:
            print(f"Error in test function: {e}")
        return False

def main():
    """
    Main entry point for subprocess execution.
    Expected command line arguments:
    python nanobind_runner.py <module_name> <test_sig> <func_sig> <width> [verbose]
    """
    if len(sys.argv) < 5:
        print("Usage: nanobind_runner.py <module_name> <test_sig> <func_sig> <width> [verbose]")
        sys.exit(1)

    module_name = sys.argv[1]
    test_sig = int(sys.argv[2])
    func_sig = sys.argv[3]
    width = int(sys.argv[4])
    verbose = len(sys.argv) > 5 and sys.argv[5].lower() == 'true'

    try:
        result = call_test_function(module_name, test_sig, func_sig, width, verbose)
        # Return the status value as exit code
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        if verbose:
            print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
