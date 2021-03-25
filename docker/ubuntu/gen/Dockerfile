FROM ubuntu:18.04
MAINTAINER Arina Neshlyaeva <arina.neshlyaeva@intel.com>

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION=10.0
ARG IGC_VER=5819
ARG L0L_VER=1.0.22
# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages
RUN apt-get -y update && apt-get install -y repo clang-8 build-essential libnuma1 opencl-headers ocl-icd-libopencl1 clinfo vim gcc g++ git python3 imagemagick \
    m4 bison flex zlib1g-dev ncurses-dev libtinfo-dev libc6-dev-i386 cpio lsb-core wget netcat-openbsd libtbb-dev libglfw3-dev pkgconf gdb gcc-multilib g++-multilib curl libomp-dev && \
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
ENV LLVM_HOME=/home/tools/llvm/$LLVM_VERSION
RUN ./alloy.py -b --version=$LLVM_VERSION --selfbuild && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-"$LLVM_VERSION"_temp $LLVM_HOME/build-"$LLVM_VERSION"_temp

WORKDIR /home/packages

# Compute runtime
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/20.50.18716/intel-gmmlib_20.3.2_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.5819/intel-igc-core_1.0.5819_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.5819/intel-igc-opencl_1.0.5819_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/20.50.18716/intel-opencl_20.50.18716_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/20.50.18716/intel-ocloc_20.50.18716_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/compute-runtime/releases/download/20.50.18716/intel-level-zero-gpu_1.0.18716_amd64.deb

RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.$IGC_VER/intel-igc-opencl-devel_1.0.${IGC_VER}_amd64.deb

# L0 loaader
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v$L0L_VER/level-zero-devel_$L0L_VER+u18.04_amd64.deb
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/oneapi-src/level-zero/releases/download/v$L0L_VER/level-zero_$L0L_VER+u18.04_amd64.deb

# Install packages and clean up
RUN dpkg -i *.deb
WORKDIR /home
RUN rm -rf packages

# vc-intrinsics
WORKDIR /home/src
RUN git clone https://github.com/intel/vc-intrinsics.git
WORKDIR /home/src/vc-intrinsics
RUN git checkout 2de2dd45caae5a69361b77412003ade9b5e198b7
RUN mkdir -p build
WORKDIR /home/src/vc-intrinsics/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j`nproc`

# SPIRV Translator
WORKDIR /home/src
RUN git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
WORKDIR /home/src/SPIRV-LLVM-Translator
RUN git checkout ab5e12aa84fd95d7b0cd7b7b3f27e6ea5eaba05e
RUN mkdir -p build
WORKDIR /home/src/SPIRV-LLVM-Translator/build
RUN cmake -DLLVM_DIR=$LLVM_HOME/bin-$LLVM_VERSION/lib/cmake/llvm/ -DCMAKE_INSTALL_PREFIX=/home/deps ../ && make install -j`nproc`

# ISPC build
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH
WORKDIR /home/src/ispc
RUN mkdir -p build
WORKDIR /home/src/ispc/build
RUN cmake -DGENX_ENABLED=ON -DCMAKE_INSTALL_PREFIX=/home/ispc/ -DGENX_DEPS_DIR=/home/deps ../ && make install -j`nproc`
RUN make check-all -j`nproc`
# Add ISPC to PATH
ENV PATH=/home/ispc/bin:$PATH
