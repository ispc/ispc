FROM ubuntu:20.04
LABEL maintainer="Arina Neshlyaeva <arina.neshlyaeva@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION=13.0
ARG IGC_VER=11061
ARG L0L_VER=1.7.15
ARG CR_VER=22.17.23034
ARG CR_VER_SUFFIX=23034
ARG VC_INTRINSICS_COMMIT_SHA="561f4ff575a198b36a72fcb790e1997d7d6d6c91"
ARG SPIRV_TRANSLATOR_COMMIT_SHA="d7a030447802718de76355c248b6bb292669683b"

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y clang-8=1:8.0.1-9 build-essential=12.8ubuntu1.1 \
    libnuma1=2.0.12-1 opencl-headers=2.2~2019.08.06-g0d5f18c-1 ocl-icd-libopencl1=2.2.11-1ubuntu1 clinfo=2.2.18.04.06-1 vim=2:8.1.2269-1ubuntu5.11 \
    gcc=4:9.3.0-1ubuntu2 g++=4:9.3.0-1ubuntu2 git=1:2.25.1-1ubuntu3.8 python3-dev=3.8.2-0ubuntu2 imagemagick=8:6.9.10.23+dfsg-2.1ubuntu11.4 \
    m4=1.4.18-4 bison=2:3.5.1+dfsg-1 flex=2.6.4-6.2 libncurses-dev=6.2-0ubuntu2 libtinfo-dev=6.2-0ubuntu2 libc6-dev-i386=2.31-0ubuntu9.9 cpio=2.13+dfsg-2ubuntu0.3 \
    lsb-core=11.1.0ubuntu2 wget=1.20.3-1ubuntu2 netcat-openbsd=1.206-1ubuntu1 libtbb-dev=2020.1-2 libglfw3-dev=3.3.2-1 pkgconf=1.6.3-5 gdb=9.2-0ubuntu1~20.04.1 \
    gcc-multilib=4:9.3.0-1ubuntu2 g++-multilib=4:9.3.0-1ubuntu2 curl=7.68.0-1ubuntu2.15 libomp-dev=1:10.0-50~exp1 ca-certificates=20211016ubuntu0.20.04.1 && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.14) for ISPC build
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && ln -s /opt/cmake/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest && ln -s /opt/cmake/bin/ctest /usr/bin/ctest && rm cmake-3.14.0-Linux-x86_64.sh

WORKDIR /home/src

RUN git clone https://github.com/$REPO.git ispc

WORKDIR /home/src/ispc
RUN git checkout $SHA
ENV ISPC_HOME=/home/src/ispc

# LLVM
ENV LLVM_HOME=/home/tools/llvm
RUN python3 ./alloy.py -b --version="$LLVM_VERSION" --selfbuild && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp "$LLVM_HOME"/build-"$LLVM_VERSION"_temp

WORKDIR /home/packages

# Compute runtime
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-core_1.0.${IGC_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-opencl_1.0.${IGC_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-level-zero-gpu-dbgsym_1.3.${CR_VER_SUFFIX}_amd64.ddeb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-level-zero-gpu_1.3.${CR_VER_SUFFIX}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-opencl-icd-dbgsym_${CR_VER}_amd64.ddeb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-opencl-icd_${CR_VER}_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5  https://github.com/intel/compute-runtime/releases/download/${CR_VER}/libigdgmm12_22.1.2_amd64.deb

# L0 loaader
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v${L0L_VER}/level-zero-devel_${L0L_VER}+u18.04_amd64.deb && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v${L0L_VER}/level-zero_${L0L_VER}+u18.04_amd64.deb

# Install packages and clean up
RUN dpkg -i ./*.deb
WORKDIR /home
RUN rm -rf packages

# vc-intrinsics
WORKDIR /home/src
RUN git clone https://github.com/intel/vc-intrinsics.git
WORKDIR /home/src/vc-intrinsics
RUN git checkout $VC_INTRINSICS_COMMIT_SHA && mkdir -p build
WORKDIR /home/src/vc-intrinsics/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j"$(nproc)"

# SPIRV Translator
WORKDIR /home/src
RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
WORKDIR /home/src/SPIRV-LLVM-Translator
RUN git checkout $SPIRV_TRANSLATOR_COMMIT_SHA && mkdir -p build
WORKDIR /home/src/SPIRV-LLVM-Translator/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm/ -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j"$(nproc)"

# ISPC build
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH
# A note about libomp discoverability (required for libispcrt.so).
# In Ubuntu 18.04 and earlier installing libomp-dev is what you need.
# Starting Ubuntu 20.04 installing libomp-dev is not installing OpenMP libraries and
# they need to be installed with your compiler. We build them as part of LLVM, so
# adding LLVM lib path to LD_LIBRARY_PATH for libomp discoverability.
ENV LD_LIBRARY_PATH=$LLVM_HOME/bin-$LLVM_VERSION/lib:$LD_LIBRARY_PATH
WORKDIR /home/src/ispc
RUN mkdir -p build
WORKDIR /home/src/ispc/build
RUN cmake -DXE_ENABLED=ON -DCMAKE_INSTALL_PREFIX=/home/ispc/ -DXE_DEPS_DIR=/home/deps ../ && make install -j"$(nproc)" && make check-all -j"$(nproc)"
# Add ISPC to PATH
ENV PATH=/home/ispc/bin:$PATH
