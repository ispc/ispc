#
#  Copyright (c) 2017-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

ARG LLVM_VERSION=15.0

FROM centos:7 AS llvm_only
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
RUN yum -y update; yum -y install centos-release-scl epel-release; yum clean all
RUN yum install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison && \
    yum install -y flex patch make glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-8 && \
    yum install -y libtool autopoint gettext-devel texinfo tbb-devel help2man && \
    yum clean -y all

# These packages are required if you need to link ISPC with -static.
RUN yum install -y libstdc++-static && \
    yum clean -y all

# Download and install required version of CMake. CMake 3.20 is required starting from LLVM 16.0.
RUN if [[ $(uname -m) =~ "x86" ]]; then export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.sh"; else export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-aarch64.sh"; fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && mkdir /opt/cmake && sh cmake-*.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm -rf cmake-*.sh

# If you are behind a proxy, you need to configure git.
RUN if [ -v "$http_proxy" ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/$REPO.git ispc

# This is home for Clang builds
RUN mkdir /usr/local/src/llvm

ENV ISPC_HOME=/usr/local/src/ispc
ENV LLVM_HOME=/usr/local/src/llvm

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

WORKDIR /usr/local/src/ispc
RUN git checkout $SHA

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
RUN source /opt/rh/devtoolset-8/enable && \
    ./alloy.py -b --version="$LLVM_VERSION" --selfbuild --llvm-disable-assertions --full-checkout && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp "$LLVM_HOME"/build-"$LLVM_VERSION"_temp

FROM llvm_only AS ispc_build

ARG LLVM_VERSION
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Build ncurses with -fPIC: https://github.com/ispc/ispc/pull/2502#issuecomment-1526931698
WORKDIR /usr/local/src
RUN git clone https://github.com/mirror/ncurses.git
WORKDIR /usr/local/src/ncurses
# Checkout version with fix for CVE-2023-29491
RUN git checkout 87c2c84 && CFLAGS="-fPIC" CXXFLAGS="-fPIC" ./configure --with-termlib && make -j"$(nproc)" && make install -j"$(nproc)"

# Install news flex (2.6.4)
WORKDIR /usr/local/src
RUN git clone https://github.com/westes/flex.git
WORKDIR /usr/local/src/flex
RUN git checkout v2.6.4 && ./autogen.sh && ./configure && make -j"$(nproc)" && make install

# Build ISPC
RUN mkdir build
WORKDIR /usr/local/src/ispc/build
RUN cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_CROSS=ON && make -j"$(nproc)" package && make check-all
