#
#  Copyright (c) 2017-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:22.04
LABEL maintainer="Arina Neshlyaeva <arina.neshlyaeva@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG IGC_VER=15985.7
ARG CR_VER=24.05.28454.6
ARG CR_VER_SUFFIX=28454.6
ARG LTO

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y clang-12 build-essential libnuma1 opencl-headers ocl-icd-libopencl1 clinfo vim gcc g++ git python3-dev imagemagick \
    m4 bison flex libncurses-dev libtinfo-dev libc6-dev-i386 cpio lsb-core wget netcat-openbsd libtbb2-dev libglfw3-dev pkgconf gdb gcc-multilib g++-multilib curl ca-certificates ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.23.5 for both x86 and aarch64) as required for superbuild preset jsons.
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CMAKE_URL="https://cmake.org/files/v3.23/cmake-3.23.5-linux-x86_64.sh"; \
    else \
        export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.23.5/cmake-3.23.5-linux-aarch64.sh"; \
    fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && \
    sh cmake-*.sh --prefix=/usr/local --skip-license && rm -rf cmake-*.sh

# Compute runtime
# Install packages and clean up
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-core_1.0.${IGC_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-opencl_1.0.${IGC_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-level-zero-gpu-dbgsym_1.3.${CR_VER_SUFFIX}_amd64.ddeb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-level-zero-gpu_1.3.${CR_VER_SUFFIX}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-opencl-icd-dbgsym_${CR_VER}_amd64.ddeb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-opencl-icd_${CR_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/libigdgmm12_22.3.11_amd64.deb && \
    dpkg -i ./*.deb && \
    rm ./*.deb

WORKDIR /home/src
RUN git clone https://github.com/$REPO.git ispc

WORKDIR /home/src/ispc
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
