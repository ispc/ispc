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
apt-get -y install bison clang cmake flex llvm-tools m4 ninja python

uname -a
clang --version
llvm-config --version
cmake --version

# The checkout is mounted read-only; build from a private copy.
cp -R "$SOURCE_DIR" "$WORK_DIR"

cmake -S "$WORK_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DARM32_ENABLED=OFF \
    -DISPCRT_BUILD_TASK_MODEL=Threads
cmake --build "$BUILD_DIR" -j "$(nproc)"

"$BUILD_DIR/bin/ispc" --version
cmake --build "$BUILD_DIR" --target check-all -j "$(nproc)"
