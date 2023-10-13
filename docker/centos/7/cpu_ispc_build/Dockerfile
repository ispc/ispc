#
#  Copyright (c) 2017-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM centos:7 AS llvm_only
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
# libstdc++-static is required if you need to link ISPC with -static.
RUN yum -y update && \
    yum install -y centos-release-scl epel-release && \
    yum install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison && \
    yum install -y flex patch make glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-8 && \
    yum install -y libtool autopoint gettext-devel texinfo tbb-devel help2man && \
    yum install -y ninja-build && \
    yum install -y libstdc++-static && \
    yum clean -y all

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
# Set PATH and LD_LIBRARY_PATH instead of `source /opt/rh/devtoolset-8/enable` due to hadolint warning.
ENV PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-8/root/usr/lib64:/opt/rh/devtoolset-8/root/usr/lib:/opt/rh/devtoolset-8/root/usr/lib64/dyninst:/opt/rh/devtoolset-8/root/usr/lib/dyninst:/opt/rh/devtoolset-8/root/usr/lib64:/opt/rh/devtoolset-8/root/usr/lib:$LD_LIBRARY_PATH

# Build Clang with all required patches.
# Superbuild fetches LLVM_VERSION from preset json file.
# "rm" are just to keep docker image small.
RUN git checkout $SHA && \
    cmake superbuild \
        -B build \
        --preset os \
        -DXE_DEPS=OFF \
        -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DLLVM_DISABLE_ASSERTIONS=ON && \
    cmake --build build && \
    rm -rf build

FROM llvm_only AS ispc_build

# Build ncurses with -fPIC: https://github.com/ispc/ispc/pull/2502#issuecomment-1526931698
WORKDIR /usr/local/src
RUN git clone https://github.com/mirror/ncurses.git
WORKDIR /usr/local/src/ncurses
# Checkout version with fix for CVE-2023-29491
RUN git checkout 87c2c84 && CFLAGS="-fPIC" CXXFLAGS="-fPIC" ./configure --with-termlib && make -j"$(nproc)" && make install -j"$(nproc)"

# Install new flex (2.6.4)
WORKDIR /usr/local/src
RUN git clone https://github.com/westes/flex.git
WORKDIR /usr/local/src/flex
RUN git checkout v2.6.4 && ./autogen.sh && ./configure && make -j"$(nproc)" && make install

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
