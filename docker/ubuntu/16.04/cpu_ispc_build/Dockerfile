#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

ARG LLVM_VERSION=17.0

FROM ubuntu:16.04 AS llvm_build
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION
ARG EXTRA_BUILD_ARG

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

RUN uname -a

# Packages required to build ISPC and Clang.
RUN apt-get -y update && apt-get --no-install-recommends install -y software-properties-common && \
    apt-get --no-install-recommends install -y wget build-essential git ncurses-dev libtinfo-dev libtbb-dev && \
    rm -rf /var/lib/apt/lists/*

# ISPC and LLVM starting 16.0 build in C++17 mode and will fail without modern compiler (gcc>=7).
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get -y update && \
    apt-get --no-install-recommends install -y gcc-7 g++-7 && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 && \
    update-alternatives --set cc /usr/bin/gcc && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 && \
    update-alternatives --set c++ /usr/bin/g++

# Download and install required version of cmake (3.20) for ISPC build
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.sh && \
    sh cmake-3.20.5-linux-x86_64.sh --prefix=/usr/local --skip-license && rm cmake-3.20.5-linux-x86_64.sh

# If you are behind a proxy, you need to configure git.
RUN if [ -v http_proxy ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /usr/local/src

# Zlib is required to build Python3.6.
RUN apt-get -y update && \
    apt-get -y --no-install-recommends install zlib1g-dev zlib1g && \
    rm -rf /var/lib/apt/lists/*
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && \
    tar xf Python-3.6.15.tgz && pushd Python-3.6.15 && \
    ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC && make -j"$(nproc)" && make install && \
    popd && rm -rf /usr/local/src/*

# This is home for Clang builds
ENV LLVM_HOME=/usr/local/src/llvm
WORKDIR /usr/local/src/ispc

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
RUN git clone https://github.com/$REPO.git ispc && \
    git -C ispc checkout $SHA && \
    cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DLLVM_VERSION="$LLVM_VERSION" \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DLLVM_DISABLE_ASSERTIONS="$LLVM_DISABLE_ASSERTIONS" && \
    cmake --build build && \
    rm -rf build

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

FROM llvm_build AS ispc_build
SHELL ["/bin/bash", "-c"]

ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION

# Install regular ISPC dependencies
RUN apt-get -y update && apt-get --no-install-recommends install -y m4 bison flex libc6-dev-i386 && \
    rm -rf /var/lib/apt/lists/*

# Install regular ISPC dependencies
RUN apt-get -y update && apt-get --no-install-recommends install -y m4 bison flex && \
    rm -rf /var/lib/apt/lists/*

# Configure and build ISPC
WORKDIR /usr/local/src
RUN cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DCMAKE_CXX_FLAGS=-Werror \
        -DPREBUILT_STAGE2_PATH="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-"$LLVM_VERSION" && \
    cmake --build build && \
    cmake --build build --target ispc-stage2-check-all && \
    mv build/build-ispc-stage2/src/ispc-stage2-build/*.tar.gz ./ && \
    rm -rf build

#export path for ispc
ENV PATH=/usr/local/src/ispc/bin-$LLVM_VERSION/bin:$PATH
