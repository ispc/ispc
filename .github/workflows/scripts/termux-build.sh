#!/bin/bash
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Build ISPC natively in Termux (Android bionic, AArch64) with 32-bit ARM
# libraries disabled, as in the environment reported in #3887.
# Runs inside the termux-docker container.
# Usage: termux-build.sh <ispc source directory>

set -euxo pipefail

SOURCE_DIR="$1"
WORK_DIR="$HOME/ispc"
BUILD_DIR="$HOME/build"

# Termux does not support partial upgrades, so upgrade before installing.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y -o Dpkg::Options::=--force-confnew upgrade
apt-get -y install bison clang cmake flex git llvm-tools m4 ninja python

uname -a
clang --version
llvm-config --version
cmake --version

# The checkout is mounted read-only; build from a private copy.
cp -R "$SOURCE_DIR" "$WORK_DIR"

cmake -S "$WORK_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_32BIT_ARM=OFF \
    -DISPCRT_BUILD_TASK_MODEL=Threads
cmake --build "$BUILD_DIR" -j "$(nproc)"

# The ARM32 C builtin from #3887 must not be generated; AArch64 must be.
test ! -e "$BUILD_DIR/share/ispc/builtins_cpp_32_linux_armv8a.bc"
test -e "$BUILD_DIR/share/ispc/builtins_cpp_64_linux_aarch64.bc"

"$BUILD_DIR/bin/ispc" --version
# With bionic headers resolved, the C builtins are complete and the whole
# lit suite applies, including the #3887 builtin-definition checks.
cmake --build "$BUILD_DIR" --target check-all
