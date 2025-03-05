#!/usr/bin/env python3
#
#  Copyright (c) 2013-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
from ctypes import c_float, c_double, c_int, POINTER, CDLL
import tomllib
import os
import pathlib
import subprocess


def main():
    args = get_args()

    cfg_fullpath = pathlib.Path(args.cfg).resolve()
    print(f"Loading config file '{cfg_fullpath}'")
    with open(cfg_fullpath, "rb") as stream:
        config = tomllib.load(stream)

    lib_fullpath = pathlib.Path(args.library).resolve()
    if not args.skip_lib_build:
        compile_library(args.ispc_bin, config["tests"], lib_fullpath)

    print(f"Loading library '{lib_fullpath}'")
    lib = CDLL(lib_fullpath)

    global_vars = globals()
    test_failed = False
    for test in config["tests"]:
        if not test["enabled"]:
            continue

        testname = test["name"]
        func_name = f"{testname}_test"
        print(f"Running test '{testname}'")

        num_errors = global_vars[func_name](lib, test["width"])
        if num_errors > 0:
            test_failed = True
            print(f"ERROR: Test '{testname}' failed with {num_errors} errors")
        else:
            print(f"Test '{testname}' passed.")

    if test_failed:
        exit(1)


def get_args():
    parser = ArgumentParser("Runs ctypes bound ispc tests", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--library", type=str, default="./py_bound_test.so", help="Path to ispc library to test.")
    parser.add_argument("--skip_lib_build", action="store_true", default=False, help="Do not build ispc library.")
    parser.add_argument("--cfg", type=str, default="./config.toml", help="Path to test config file.")
    parser.add_argument("--ispc_bin", type=str, required=True, help="Path to ispc binary.")

    args = parser.parse_args()
    args.ispc_bin = pathlib.Path(args.ispc_bin).resolve()
    return args


def compile_library(ispc_bin, tests, library):
    build_path = pathlib.Path("./build").resolve()
    os.makedirs("./build", exist_ok=True)
    src_path = pathlib.Path().resolve()

    # Run ispc
    o_files = []
    for test in tests:
        if not test["enabled"]:
            continue

        test_file = test["test_file"]
        src_file = os.path.join(src_path, test_file)
        o_file = os.path.join(build_path, f"{test_file}.o")
        h_file = os.path.join(build_path, f"{test_file}.h")
        o_files.append(o_file)
        # TODO: Make work for other architectures and targets
        cmd = f"{ispc_bin} --pic --woff {src_file} -o {o_file} --arch=x86-64 --target=sse4-i32x4 -O2 -h {h_file}"
        print(f"Running cmd '{cmd}'")
        subprocess.run(cmd.split(), check=True)

    # Create shared object
    o_files_str = " ".join(o_files)
    cmd = f"clang++ -O2 -m64 {src_path}/top_bind.cpp {o_files_str} -o ./py_bound_test.so -fPIC -shared"
    print(f"Running cmd '{cmd}'")
    subprocess.run(cmd.split(), check=True)


TestArrays = namedtuple(
    "TestArrays", ["array_size", "returned_result", "expected_result", "vfloat", "vdouble", "vint", "vint2"]
)


# Mirrored from test_static.cpp
def gen_TestArrays():
    array_size = 256

    returned_result = (c_float * array_size)()
    expected_result = (c_float * array_size)()
    vfloat = (c_float * array_size)()
    vdouble = (c_double * array_size)()
    vint = (c_int * array_size)()
    vint2 = (c_int * array_size)()

    for i in range(array_size):
        returned_result[i] = -1e20
        expected_result[i] = 0
        vfloat[i] = i + 1
        vdouble[i] = i + 1
        vint[i] = 2 * (i + 1)
        vint2[i] = i + 5

    return TestArrays(array_size, returned_result, expected_result, vfloat, vdouble, vint, vint2)


def check_arrays(result, expected, width):
    num_errors = 0
    for i in range(width):
        if result[i] != expected[i]:
            num_errors += 1
            print(f"value {i} disagrees: returned {float(result[i])}, expected {float(expected[i])}")
    return num_errors


def swizzle3_test(lib, width):
    lib.swizzle3_entry_point.argtypes = [POINTER(c_float)]
    lib.swizzle3_entry_point.restype = None

    lib.swizzle3_result_entry_point.argtypes = [POINTER(c_float)]
    lib.swizzle3_result_entry_point.restype = None

    arrays = gen_TestArrays()
    lib.swizzle3_entry_point(arrays.returned_result)
    lib.swizzle3_result_entry_point(arrays.expected_result)

    return check_arrays(arrays.returned_result, arrays.expected_result, width)


def gb_double_improve_progindex_test(lib, width):
    lib.gb_double_improve_progindex_entry_point.argtypes = [POINTER(c_float), POINTER(c_float), c_float]
    lib.gb_double_improve_progindex_entry_point.restype = None

    lib.gb_double_improve_progindex_result_entry_point.argtypes = [POINTER(c_float)]
    lib.gb_double_improve_progindex_result_entry_point.restype = None

    arrays = gen_TestArrays()
    b = c_float(5)

    lib.gb_double_improve_progindex_entry_point(arrays.returned_result, arrays.vfloat, b)
    lib.gb_double_improve_progindex_result_entry_point(arrays.expected_result)

    return check_arrays(arrays.returned_result, arrays.expected_result, width)


if __name__ == "__main__":
    main()
