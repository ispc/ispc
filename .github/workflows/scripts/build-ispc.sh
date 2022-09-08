#!/bin/bash -e
echo PATH=$PATH
# FIXME: -Wno-deprecated-declarations is here only until support for opaque pointers is not implemented in ISPC
cmake -B build -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-Werror -Wno-deprecated-declarations" -DISPC_PACKAGE_NAME=ispc-trunk-linux $@
cmake --build build --target package -j4
