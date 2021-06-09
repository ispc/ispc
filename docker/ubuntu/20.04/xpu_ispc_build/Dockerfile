FROM ubuntu:20.04
MAINTAINER Arina Neshlyaeva <arina.neshlyaeva@intel.com>

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION=12.0
ARG IGC_VER=7423
ARG L0L_VER=1.2.3
ARG CR_VER=21.21.19914
ARG CR_VER_SUFFIX=19914
ARG VC_INTRINSICS_COMMIT_SHA="2d0795c6799d262a464e5e8b8f74b09dd3ec1054"
ARG SPIRV_TRANSLATOR_COMMIT_SHA="0592c4f6fc42ec947f462e81628280e960a12042"

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y clang-8 build-essential libnuma1 opencl-headers ocl-icd-libopencl1 clinfo vim gcc g++ git python3 imagemagick \
    m4 bison flex zlib1g-dev libncurses-dev libtinfo-dev libc6-dev-i386 cpio lsb-core wget netcat-openbsd libtbb-dev libglfw3-dev pkgconf gdb gcc-multilib g++-multilib curl libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.14) for ISPC build
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && ln -s /opt/cmake/bin/cmake /usr/bin/cmake && \
    ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest && ln -s /opt/cmake/bin/ctest /usr/bin/ctest && rm cmake-3.14.0-Linux-x86_64.sh

WORKDIR /home/src

RUN git clone https://github.com/$REPO.git ispc
RUN cd ispc && git checkout $SHA && cd ..

WORKDIR /home/src/ispc
ENV ISPC_HOME=/home/src/ispc

# LLVM
ENV LLVM_HOME=/home/tools/llvm
RUN python3 ./alloy.py -b --version=$LLVM_VERSION --selfbuild && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-"$LLVM_VERSION"_temp $LLVM_HOME/build-"$LLVM_VERSION"_temp

WORKDIR /home/packages

# Compute runtime
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-gmmlib_21.1.3_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-core_1.0.${IGC_VER}_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-opencl_1.0.${IGC_VER}_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-opencl_${CR_VER}_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-ocloc_${CR_VER}_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/${CR_VER}/intel-level-zero-gpu_1.1.${CR_VER_SUFFIX}_amd64.deb

RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.${IGC_VER}/intel-igc-opencl-devel_1.0.${IGC_VER}_amd64.deb

# L0 loaader
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v${L0L_VER}/level-zero-devel_${L0L_VER}+u18.04_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v${L0L_VER}/level-zero_${L0L_VER}+u18.04_amd64.deb

# Install packages and clean up
RUN dpkg -i *.deb
WORKDIR /home
RUN rm -rf packages

# vc-intrinsics
WORKDIR /home/src
RUN git clone https://github.com/intel/vc-intrinsics.git
WORKDIR /home/src/vc-intrinsics
RUN git checkout $VC_INTRINSICS_COMMIT_SHA
RUN mkdir -p build
WORKDIR /home/src/vc-intrinsics/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j`nproc`

# SPIRV Translator
WORKDIR /home/src
RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
WORKDIR /home/src/SPIRV-LLVM-Translator
RUN git checkout $SPIRV_TRANSLATOR_COMMIT_SHA
RUN mkdir -p build
WORKDIR /home/src/SPIRV-LLVM-Translator/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm/ -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j`nproc`

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
RUN cmake -DGENX_ENABLED=ON -DCMAKE_INSTALL_PREFIX=/home/ispc/ -DGENX_DEPS_DIR=/home/deps ../ && make install -j`nproc`
RUN make check-all -j`nproc`
# Add ISPC to PATH
ENV PATH=/home/ispc/bin:$PATH
