#!/bin/bash -e
echo PATH=$PATH

if [[ $OSTYPE == 'darwin'* ]]; then
  # macOS is treated differently because we don't build benchmarks and package name is different.
  # Benchmarks are not built, because Github Action macOS shared runners run on too old hardware -
  # IvyBridge, i.e. AVX1, while our benchmark setup assumes at least AVX2.
  cmake -B build -DISPC_PREPARE_PACKAGE=ON -DCMAKE_CXX_FLAGS="-Werror" -DISPC_PACKAGE_NAME=ispc-trunk-macos -DISPC_OPAQUE_PTR_MODE=${ISPC_OPAQUE_PTR_MODE} $@
else
  cmake -B build -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_BENCHMARKS=ON -DCMAKE_CXX_FLAGS="-Werror" -DISPC_PACKAGE_NAME=ispc-trunk-linux -DISPC_OPAQUE_PTR_MODE=${ISPC_OPAQUE_PTR_MODE} $@
fi
cmake --build build --target package -j4
