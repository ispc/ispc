#!/bin/bash -e
#
# Copyright 2024-2025, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && sudo apt-get -y update

sudo apt-get --no-install-recommends install -y wget build-essential gcc g++ git python3-dev ca-certificates libtbb-dev ninja-build

# ISPC and LLVM starting 16.0 build in C++17 mode and will fail without modern libstdc++
sudo apt-get --no-install-recommends install -y software-properties-common libstdc++-9-dev

# Install multilib libc needed to build clang_rt
sudo apt-get --no-install-recommends install -y libc6-dev-armhf-cross
sudo apt-get --no-install-recommends install -y m4 bison flex

# This is actually a clang dependency.
sudo apt-get --no-install-recommends install -y libtinfo5
[ -n "$LLVM_REPO" ] && wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $LLVM_REPO/releases/download/llvm-$LLVM_VERSION-ispc-dev/$LLVM_TAR
tar xf $LLVM_TAR
echo "${GITHUB_WORKSPACE}/bin-$LLVM_VERSION/bin" >> $GITHUB_PATH
