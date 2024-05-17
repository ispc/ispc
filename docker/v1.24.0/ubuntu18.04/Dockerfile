#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

ARG LLVM_VERSION=17.0

FROM ubuntu:18.04 AS llvm_build_step1
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=v1.24.0
ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION
ARG LLVM_DISABLE_ASSERTIONS

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

RUN uname -a

# Packages required to build ISPC and Clang.
RUN apt-get -y update && apt-get --no-install-recommends install -y wget build-essential gcc g++ git python3-dev ncurses-dev libtinfo-dev ca-certificates libtbb-dev ninja-build && \
    rm -rf /var/lib/apt/lists/*

# ISPC and LLVM starting 16.0 build in C++17 mode and will fail without modern libstdc++
RUN apt-get -y update && apt-get --no-install-recommends install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get -y update && \
    apt-get --no-install-recommends install -y libstdc++-9-dev && \
    rm -rf /var/lib/apt/lists/*

# Install multilib libc needed to build clang_rt
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CROSS_LIBS="libc6-dev-i386=2.27-3ubuntu*"; \
    else \
        export CROSS_LIBS="libc6-dev-armhf-cross=2.27-3ubuntu*"; \
    fi && \
    apt-get -y update && apt-get --no-install-recommends install -y "$CROSS_LIBS" && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.23.5 for both x86 and aarch64) as required for superbuild preset jsons.
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CMAKE_URL="https://cmake.org/files/v3.23/cmake-3.23.5-linux-x86_64.sh"; \
    else \
        export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.23.5/cmake-3.23.5-linux-aarch64.sh"; \
    fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && \
    sh cmake-*.sh --prefix=/usr/local --skip-license && rm -rf cmake-*.sh

# If you are behind a proxy, you need to configure git.
RUN if [ -v http_proxy ]; then git config --global --add http.proxy "$http_proxy"; fi

ENV LLVM_HOME=/usr/local/src/llvm
WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
RUN git clone https://github.com/$REPO.git ispc && \
    git -C ispc checkout $SHA && \
    cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DLLVM_VERSION="$LLVM_VERSION" \
        -DBUILD_STAGE1_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DLLVM_DISABLE_ASSERTIONS="$LLVM_DISABLE_ASSERTIONS" && \
    cmake --build build && \
    rm -rf build

FROM llvm_build_step1 AS llvm_build_step2
SHELL ["/bin/bash", "-c"]

ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION
ARG LLVM_DISABLE_ASSERTIONS

WORKDIR /usr/local/src
RUN cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DLLVM_VERSION="$LLVM_VERSION" \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DPREBUILT_STAGE1_PATH="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DLLVM_DISABLE_ASSERTIONS="$LLVM_DISABLE_ASSERTIONS" && \
    cmake --build build && \
    rm -rf build

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

FROM llvm_build_step2 AS ispc_build
SHELL ["/bin/bash", "-c"]

ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION

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
