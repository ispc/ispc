#!/bin/bash -e

echo PATH=$PATH

if [[ $OSTYPE == 'darwin'* ]]; then
  # macOS is treated differently because we don't build benchmarks and package name is different.
  # Benchmarks are not built, because Github Action macOS shared runners run on too old hardware -
  # IvyBridge, i.e. AVX1, while our benchmark setup assumes at least AVX2.
  cmake -B build \
    -DISPC_MACOS_UNIVERSAL_BINARIES=ON \
    -DISPC_PREPARE_PACKAGE=ON \
    -DCMAKE_CXX_FLAGS="-Werror" \
    -DISPC_PACKAGE_NAME=ispc-trunk-macos-wincross \
    -DISPC_CROSS=ON \
    -DISPC_WINDOWS_SDK_PATH=winsdk/sdk \
    -DISPC_WINDOWS_VCTOOLS_PATH=winsdk/crt \
    -DISPC_ANDROID_TARGET=OFF \
    -DISPC_IOS_TARGET=OFF \
    -DISPC_LINUX_TARGET=OFF \
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
    -DISPC_PACKAGE_NAME=ispc-trunk-linux-wincross \
    -DISPC_CROSS=ON \
    -DISPC_WINDOWS_SDK_PATH=winsdk/sdk \
    -DISPC_WINDOWS_VCTOOLS_PATH=winsdk/crt \
    -DISPC_ANDROID_TARGET=OFF \
    -DISPC_FREEBSD_TARGET=OFF \
    $@
fi
cmake --build build --target package -j4
