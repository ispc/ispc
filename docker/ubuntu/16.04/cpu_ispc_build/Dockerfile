#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

ARG LLVM_VERSION=17.0

FROM ubuntu:16.04 AS llvm_build_step1
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

# ISPC and LLVM starting 16.0 build in C++17 mode and will fail without modern libstdc++ and modern compiler (gcc>=7).
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get -y update && \
    apt-get --no-install-recommends install -y libstdc++-9-dev gcc-7 g++-7 && \
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

# Fork ispc on github and clone *your* fork.
RUN mkdir -p /usr/local/src/llvm && \
    git clone https://github.com/$REPO.git ispc

# This is home for Clang builds
ENV LLVM_HOME=/usr/local/src/llvm
ENV ISPC_HOME=/usr/local/src/ispc

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

WORKDIR /usr/local/src/ispc

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
RUN git checkout $SHA && \
    python3.6 ./alloy.py -b --full-checkout --version="$LLVM_VERSION" --selfbuild-phase1 --verbose "$EXTRA_BUILD_ARG" && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION"_temp

FROM llvm_build_step1 AS llvm_build_step2
ARG LLVM_VERSION
ARG EXTRA_BUILD_ARG

RUN python3.6 ./alloy.py -b --version="$LLVM_VERSION" --selfbuild-phase2 --verbose "$EXTRA_BUILD_ARG" && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

FROM llvm_build_step2 AS ispc_build
ARG LLVM_VERSION

# Install regular ISPC dependencies
RUN apt-get -y update && apt-get --no-install-recommends install -y m4 bison flex libc6-dev-i386 && \
    rm -rf /var/lib/apt/lists/*

# Configure ISPC build
RUN cmake . \
        -B build \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_FLAGS=-Werror \
        -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-$LLVM_VERSION

# Build ISPC
RUN cmake --build build -j "$(nproc)" && \
    cmake --build build --target check-all && \
    cmake --install build && \
    rm -rf build

# Export path for ispc
ENV PATH=/usr/local/src/ispc/bin-$LLVM_VERSION/bin:$PATH
