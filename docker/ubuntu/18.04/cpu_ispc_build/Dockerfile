#
#  Copyright (c) 2017-2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:18.04 AS llvm_build
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LTO=OFF
ARG PGO=OFF
ARG LLVM_DISABLE_ASSERTIONS

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

RUN uname -a

# Packages required to build ISPC and Clang.
RUN apt-get -y update && apt-get --no-install-recommends install -y wget build-essential gcc g++ git ca-certificates ninja-build gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install TBB from Intel oneAPI.
# Use oneAPI TBB instead of system TBB to avoid enum-constexpr-conversion errors with LLVM 21+
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get -y update && apt-get --no-install-recommends install -y intel-oneapi-tbb-devel && \
    rm -rf /var/lib/apt/lists/*

# Set TBB_ROOT to point to oneAPI TBB installation
ENV TBB_ROOT=/opt/intel/oneapi/tbb/latest

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

# Zlib is required to build Python3
RUN apt-get -y update && \
    apt-get -y --no-install-recommends install zlib1g-dev zlib1g && \
    rm -rf /var/lib/apt/lists/*
ENV PYTHON_VERSION=3.8.19
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xf Python-${PYTHON_VERSION}.tgz && pushd Python-${PYTHON_VERSION} && \
    ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC && make -j"$(nproc)" && make install && \
    popd && rm -rf /usr/local/src/*

# If you are behind a proxy, you need to configure git.
RUN if [ -v http_proxy ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

# Build Clang with all required patches with version specific in superbuild
# CMAKE preset file under superbuid directory.
RUN git clone https://github.com/$REPO.git ispc && \
    git -C ispc checkout $SHA && \
    cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ \
        -DLLVM_DISABLE_ASSERTIONS="$LLVM_DISABLE_ASSERTIONS" && \
    cmake --build build && \
    rm -rf build

FROM llvm_build AS ispc_build
SHELL ["/bin/bash", "-c"]

ARG LTO=OFF
ARG PGO=OFF

# Install regular ISPC dependencies
RUN apt-get -y update && apt-get --no-install-recommends install -y m4 bison flex && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/src

# Create new non-root user and switch to it
RUN useradd -m -d /home/ispc_user -s /bin/bash -U ispc_user && \
    chown -R ispc_user:ispc_user /usr

USER ispc_user

# Configure and build ISPC
RUN cmake ispc/superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DXE_DEPS=OFF \
        -DCMAKE_CXX_FLAGS=-Werror \
        -DPREBUILT_STAGE2_PATH=/usr/local/ \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ && \
    cmake --build build && \
    cmake --build build --target ispc-stage2-check-all && \
    mv build/build-ispc-stage2/src/ispc-stage2-build/*.tar.gz ./ && \
    rm -rf build
