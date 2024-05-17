#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM centos:7
LABEL maintainer="Aleksei Nurmukhametov <aleksei.nurmukhametov@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=v1.23.0
ARG TBB=default
ARG LTO=OFF
ARG PGO=OFF

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
# libstdc++-static is required if you need to link ISPC with -static.
RUN yum -y update && \
    yum install -y centos-release-scl epel-release && \
    yum install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison flex patch make && \
    yum install -y glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-8 && \
    yum install -y libtool autopoint gettext-devel texinfo help2man && \
    yum install -y ninja-build && \
    yum install -y libstdc++-static && \
    yum clean -y all

# TBB is parameter to build both packge usual and oneapi-tbb.
RUN if [ "$TBB" == "default" ]; then  yum -y install tbb tbb-devel && yum clean -y all; fi
COPY oneAPI.repo /etc/yum.repos.d/oneAPI.repo
RUN if [ "$TBB" == "oneapi" ]; then yum -y install intel-oneapi-tbb-devel intel-oneapi-tbb && yum clean -y all; fi

# Download and install required version of cmake (3.23.5 for both x86 and aarch64) as required for superbuild preset jsons.
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CMAKE_URL="https://cmake.org/files/v3.23/cmake-3.23.5-linux-x86_64.sh"; \
    else \
        export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.23.5/cmake-3.23.5-linux-aarch64.sh"; \
    fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && \
    sh cmake-*.sh --prefix=/usr/local --skip-license && rm -rf cmake-*.sh

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
# Same with CMAKE_PREFIX_PATH and LD_LIBRARY_PATH for `source /opt/intel/oneapi/setvars.sh`.
ENV PATH=/opt/rh/devtoolset-8/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.11/lib/intel64/gcc4.8:/opt/rh/devtoolset-8/root/usr/lib64:/opt/rh/devtoolset-8/root/usr/lib:/opt/rh/devtoolset-8/root/usr/lib64/dyninst:/opt/rh/devtoolset-8/root/usr/lib/dyninst:/opt/rh/devtoolset-8/root/usr/lib64:/opt/rh/devtoolset-8/root/usr/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/2021.11

# Build Clang/LLVM, XE dependencies and then ISPC.
RUN git checkout $SHA && \
    cmake superbuild \
        -B build \
        --preset os \
        -DLTO=$LTO \
        -DINSTALL_ISPC=ON \
        -DINSTALL_TOOLCHAIN=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build && \
    (cmake --build build --target ispc-stage2-check-all || true) && \
    mv build/build-ispc-stage2/src/ispc-stage2-build/*.tar.gz ./ && \
    rm -rf build
