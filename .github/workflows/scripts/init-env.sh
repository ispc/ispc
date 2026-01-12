#!/bin/bash -e

# Define some widely used environment variables in one place

SDE_MIRROR_ID=${SDE_MIRROR_ID:-"859732"}
SDE_TAR_NAME=${SDE_TAR_NAME:-"sde-external-9.58.0-2025-06-16"}
USER_AGENT=${USER_AGENT:-"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"}
LLVM_REPO=${LLVM_REPO:-"https://github.com/ispc/ispc.dependencies"}
LLVM_VERSION=${LLVM_VERSION:-"21.1"}
echo "LLVM_VERSION=${LLVM_VERSION}" >> "${GITHUB_ENV}"

OS=$(uname -s)
case "$OS" in
    Linux*)
      LLVM_TAR=${LLVM_TAR:-"llvm-21.1.8-ubuntu22.04-Release+Asserts-x86.arm.wasm.tar.xz"}
      ;;
    Darwin*)
      LLVM_TAR=${LLVM_TAR:-"llvm-21.1.8-macos-Release+Asserts-universal-x86.arm.wasm.tar.xz"}
      ;;
    *)
      echo "Unsupported OS: $OS"
      exit 1
      ;;
esac

echo "Installing build dependencies for ISPC on $OS"
echo "LLVM version: $LLVM_VERSION"
echo "LLVM tarball: $LLVM_TAR"
