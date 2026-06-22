#!/bin/bash -e

# Define some widely used environment variables in one place

SDE_REPO=${SDE_REPO:-"https://github.com/ispc/ispc.dependencies"}
# Default to the newer SDE, which supports avx10.2 targets (e.g. the -nvl knob)
# needed by the build job's lit tests. The basic/avx512 test matrix jobs override
# SDE_TAR_NAME to the older 9.58.0 SDE (the newer 10.8 SDE hangs on those runs).
SDE_TAR_NAME=${SDE_TAR_NAME:-"sde-external-10.8.0-2026-03-15"}
LLVM_REPO=${LLVM_REPO:-"https://github.com/ispc/ispc.dependencies"}
LLVM_VERSION=${LLVM_VERSION:-"23.1"}
echo "LLVM_VERSION=${LLVM_VERSION}" >> "${GITHUB_ENV}"

OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
    Linux*)
      if [ "$ARCH" == "aarch64" ]; then
        LLVM_TAR=${LLVM_TAR:-"llvm-23.1.0-ubuntu22.04aarch64-Release+Asserts-x86.arm.wasm.tar.xz"}
      else
        LLVM_TAR=${LLVM_TAR:-"llvm-23.1.0-ubuntu22.04-Release+Asserts-x86.arm.wasm.tar.xz"}
      fi
      ;;
    Darwin*)
      LLVM_TAR=${LLVM_TAR:-"llvm-23.1.0-macos-Release+Asserts-universal-x86.arm.wasm.tar.xz"}
      ;;
    *)
      echo "Unsupported OS: $OS"
      exit 1
      ;;
esac

echo "Installing build dependencies for ISPC on $OS"
echo "LLVM version: $LLVM_VERSION"
echo "LLVM tarball: $LLVM_TAR"
