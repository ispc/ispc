#
#  Copyright (c) 2017-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM centos:7
LABEL maintainer="Arina Neshlyaeva <arina.neshlyaeva@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=v1.21.0
ARG LLVM_VERSION=15.0
ARG L0L_VER=1.13.5
ARG VC_INTRINSICS_COMMIT_SHA="910db4801d4a029834606e3e42a8d60358e74fdf"
ARG SPIRV_TRANSLATOR_COMMIT_SHA="e82ecc2bd7295604fcf1824e47c95fa6a09c6e63"

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
RUN yum -y update; yum -y install centos-release-scl epel-release; yum clean all
RUN yum install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison flex patch make && \
    yum install -y glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-8 && \
    yum install -y libtool autopoint gettext-devel texinfo help2man && \
    yum clean -y all

# These packages are required if you need to link ISPC with -static.
RUN yum install -y libstdc++-static && \
    yum clean -y all

# Install TBB
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://registrationcenter-download.intel.com/akdlm/IRC_NAS/c95cd995-586b-4688-b7e8-2d4485a1b5bf/l_tbb_oneapi_p_2021.10.0.49543_offline.sh && \
    chmod u+x ./l_tbb_oneapi_p_2021.10.0.49543_offline.sh && \
    ./l_tbb_oneapi_p_2021.10.0.49543_offline.sh -s -a --action install --eula accept --intel-sw-improvement-program-consent decline --silent && \
    source /opt/intel/oneapi/setvars.sh && \
    rm ./l_tbb_oneapi_p_2021.10.0.49543_offline.sh

# Download and install required version of CMake. CMake 3.20 is required starting from LLVM 16.0.
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.20.5-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm cmake-3.20.5-linux-x86_64.sh

# If you are behind a proxy, you need to configure git.
RUN if [ -v "$http_proxy" ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/$REPO.git ispc

# This is home for Clang builds
RUN mkdir /usr/local/src/llvm

ENV ISPC_HOME=/usr/local/src/ispc
ENV LLVM_HOME=/usr/local/src/llvm
ENV XE_DEPS=/usr/local/deps

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

# vc-intrinsics
WORKDIR /usr/local/src
RUN git clone https://github.com/intel/vc-intrinsics.git
WORKDIR /usr/local/src/vc-intrinsics
RUN git checkout $VC_INTRINSICS_COMMIT_SHA && mkdir -p build
WORKDIR /usr/local/src/vc-intrinsics/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm -DCMAKE_INSTALL_PREFIX=$XE_DEPS ../ && make install -j"$(nproc)"

# SPIRV Translator
WORKDIR /usr/local/src
RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
WORKDIR /usr/local/src/SPIRV-LLVM-Translator
RUN git checkout $SPIRV_TRANSLATOR_COMMIT_SHA && mkdir -p build
WORKDIR /usr/local/src/SPIRV-LLVM-Translator/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm/ -DCMAKE_INSTALL_PREFIX=$XE_DEPS ../ && make install -j"$(nproc)"

# L0
WORKDIR /usr/local/src
RUN git clone https://github.com/oneapi-src/level-zero.git
WORKDIR /usr/local/src/level-zero
RUN git checkout v$L0L_VER && mkdir -p build
WORKDIR /usr/local/src/level-zero/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_INSTALL_PREFIX=/usr/local ../ && make install -j"$(nproc)"

# Build ISPC
ENV LD_LIBRARY_PATH=$LLVM_HOME/bin-$LLVM_VERSION/lib:$LD_LIBRARY_PATH
RUN mkdir build
WORKDIR /usr/local/src/ispc/build
RUN cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_CROSS=ON -DXE_ENABLED=ON -DXE_DEPS_DIR=$XE_DEPS && make -j"$(nproc)" package
