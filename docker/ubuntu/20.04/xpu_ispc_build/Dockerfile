#
#  Copyright (c) 2017-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:20.04
LABEL maintainer="Arina Neshlyaeva <arina.neshlyaeva@intel.com>"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LTO=OFF

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y clang-8 build-essential \
    libnuma1 opencl-headers ocl-icd-libopencl1 clinfo vim gcc g++ git python3-dev imagemagick m4 bison flex libncurses-dev \
    libtinfo-dev libc6-dev-i386 cpio lsb-core wget gnupg netcat-openbsd libtbb-dev libglfw3-dev pkgconf gdb gcc-multilib g++-multilib curl ca-certificates && \
    apt-get --no-install-recommends install -y ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.23.5 for both x86 and aarch64) as required for superbuild preset jsons.
RUN if [[ $(uname -m) =~ "x86" ]]; then \
        export CMAKE_URL="https://cmake.org/files/v3.23/cmake-3.23.5-linux-x86_64.sh"; \
    else \
        export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.23.5/cmake-3.23.5-linux-aarch64.sh"; \
    fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && \
    sh cmake-*.sh --prefix=/usr/local --skip-license && rm -rf cmake-*.sh

# Add package repository and install graphics packages
RUN set -eux && wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-devel main' | \
    tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
    apt-get -y update && apt-get --no-install-recommends install -y intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm11 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all && \
    rm -rf /var/lib/apt/lists/*

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
