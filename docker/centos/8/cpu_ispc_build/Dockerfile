#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM rockylinux:8 AS llvm_only
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
RUN dnf -y update && dnf install -y wget yum-utils gcc gcc-c++ git make xz tbb-devel python38 python38-devel && \
    dnf clean -y all

# These packages are required if you need to link ISPC with -static.
# Ninja is required for superbuild
RUN dnf -y --enablerepo=powertools install ninja-build libstdc++-static && \
    dnf clean -y all

RUN if [[ $(uname -m) =~ "x86" ]]; then dnf -y update && dnf install -y glibc-devel.i686 && dnf clean -y all; fi

WORKDIR /usr/local/src

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

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/$REPO.git ispc

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
# Add --llvm-disable-assertions for building "release" version.
RUN git checkout $SHA && \
    cmake superbuild \
        -B build \
        --preset os \
        -DXE_DEPS=OFF \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build && \
    rm -rf build

FROM llvm_only AS ispc_build
SHELL ["/bin/bash", "-c"]

RUN dnf -y update && dnf install -y m4 bison flex && dnf clean -y all

# Create new non-root user and switch to it
RUN useradd -m -d /home/ispc_user -s /bin/bash -U ispc_user && \
    chown -R ispc_user:ispc_user /usr

USER ispc_user

# Build ISPC
WORKDIR /usr/local/src/ispc
RUN cmake . \
        -B build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DISPC_PREPARE_PACKAGE=ON \
        -DISPC_CROSS=ON && \
    cmake --build build --target package -j"$(nproc)" && \
    cmake --build build --target check-all && \
    cmake --install build && \
    rm -rf build
