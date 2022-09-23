#!/bin/bash -e
echo PATH=$PATH

cmake -B build -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-Werror" -DISPC_PACKAGE_NAME=ispc-trunk-linux -DISPC_OPAQUE_PTR_MODE=${ISPC_OPAQUE_PTR_MODE} $@
cmake --build build --target package -j4
