#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

ARG LLVM_VERSION=17.0

FROM ubuntu:22.04 AS llvm_build
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION
ARG LLVM_DISABLE_ASSERTIONS

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

RUN uname -a

# Packages
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y \
        wget ninja-build build-essential gcc g++ git python3-dev \
        ca-certificates libtbb2-dev && \
    rm -rf /var/lib/apt/lists/*

# Install multilib libc needed to build clang_rt
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CROSS_LIBS="libc6-dev-i386"; \
    else \
        export CROSS_LIBS="libc6-dev-armhf-cross"; \
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
RUN git clone https://github.com/$REPO.git ispc && \
    git -C ispc checkout $SHA && \
    cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DLLVM_VERSION="$LLVM_VERSION" \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DLLVM_DISABLE_ASSERTIONS="$LLVM_DISABLE_ASSERTIONS" && \
    cmake --build build && \
    rm -rf build

FROM llvm_build AS ispc_build
SHELL ["/bin/bash", "-c"]

ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_VERSION

RUN apt-get -y update && \
    apt-get --no-install-recommends install -y \
        m4 bison flex libc6-dev-i386-cross libc6-dev-arm64-cross libc6-dev-armhf-cross && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/src

# Create new non-root user and switch to it
RUN useradd -m -d /home/ispc_user -s /bin/bash -U ispc_user && \
chown -R ispc_user:ispc_user /usr

USER ispc_user

# Build ISPC
RUN cmake ispc/superbuild \
        -B build \
        --preset os \
        -DX86_ENABLED=ON \
        -DARM_ENABLED=ON \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DCMAKE_CXX_FLAGS=-Werror \
        -DPREBUILT_STAGE2_PATH="$LLVM_HOME"/bin-"$LLVM_VERSION" \
        -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-"$LLVM_VERSION" && \
    cmake --build build -j"$(nproc)" && \
    cmake --build build --target check-all && \
    cmake --install build && \
    rm -rf build