#!/bin/bash -e
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Build ISPC natively in Termux (Android, bionic libc) and run the sanity checks.
# It runs inside the termux-docker container, see termux.yml.
# Usage: bash -e build-ispc-termux.sh <ispc source directory>

# Termux doesn't support partial upgrades, so upgrade the installed packages first.
apt-get update
apt-get -y -o Dpkg::Options::=--force-confnew upgrade
apt-get -y install bison clang cmake flex git libtbb llvm-tools m4 ninja python

cmake --version
clang --version
llvm-config --version

# The source directory is mounted read-only, so build in a copy of it.
cp -R "$1" "$HOME/ispc"
cd "$HOME/ispc"

# Cover both ARM32 and AArch64 builtins without target C library headers.
cmake -B build -G Ninja -DARM32_ENABLED=ON
cmake --build build -j"$(nproc)"

bash -e .github/workflows/scripts/check-ispc.sh
