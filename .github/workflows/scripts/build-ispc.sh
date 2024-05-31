#!/bin/bash -e
#
# Copyright 2024, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

echo PATH=$PATH

if [[ $OSTYPE == 'darwin'* ]]; then
  # macOS is treated differently because we don't build benchmarks and package name is different.
  # Benchmarks are not built, because Github Action macOS shared runners run on too old hardware -
  # IvyBridge, i.e. AVX1, while our benchmark setup assumes at least AVX2.
  cmake -B build \
    -DISPC_MACOS_UNIVERSAL_BINARIES=ON \
    -DISPC_PREPARE_PACKAGE=ON \
    -DCMAKE_CXX_FLAGS="-Werror" \
    -DISPC_PACKAGE_NAME=ispc-trunk-macos \
    -DISPC_OPAQUE_PTR_MODE=${ISPC_OPAQUE_PTR_MODE} \
    $@
else
  ARCH=$(uname -m)
  BENCHMARKS=ON
  if [ "$ARCH" == "aarch64" ]; then
    BENCHMARKS=OFF
  fi
  cmake -B build \
    -DISPC_INCLUDE_BENCHMARKS=${BENCHMARKS} \
    -DISPC_PREPARE_PACKAGE=ON \
    -DCMAKE_CXX_FLAGS="-Werror" \
    -DISPC_PACKAGE_NAME=ispc-trunk-linux \
    -DISPC_OPAQUE_PTR_MODE=${ISPC_OPAQUE_PTR_MODE} \
    $@
fi
cmake --build build --target package -j4
