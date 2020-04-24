FROM centos:7
MAINTAINER Dmitry Babokin <dmitry.y.babokin@intel.com>

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
RUN yum -y update; yum -y install centos-release-scl epel-release; yum clean all
RUN yum install -y vim wget yum-utils gcc gcc-c++ git subversion python3 m4 bison flex patch make ncurses-devel glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-7 && \
    yum clean -y all

# These packages are required if you need to link IPSC with -static.
RUN yum install -y ncurses-static libstdc++-static && \
    yum clean -y all

# Download and install required version of cmake (3.14) for ISPC build
RUN wget https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm cmake-3.14.0-Linux-x86_64.sh

# If you are behind a proxy, you need to configure git.
#RUN git config --global --add http.proxy http://proxy.yourcompany.com:888

WORKDIR /usr/local/src

# checkout v1.13
RUN git clone https://github.com/ispc/ispc.git && cd ispc && git checkout v1.13.0

# This is home for Clang builds
RUN mkdir /usr/local/src/llvm

ENV ISPC_HOME=/usr/local/src/ispc
ENV LLVM_HOME=/usr/local/src/llvm

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

WORKDIR /usr/local/src/ispc

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
# By default 10.0 is used.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
ARG LLVM_VERSION=10.0
RUN source /opt/rh/devtoolset-7/enable && \
    ./alloy.py -b --version=$LLVM_VERSION --selfbuild && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-$LLVM_VERSION_temp $LLVM_HOME/build-$LLVM_VERSION_temp

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Install newer zlib
WORKDIR /usr/local/src
RUN git clone https://github.com/madler/zlib.git && cd zlib && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && make install

# MacOSX10.14.sdk to enable cross compilation to macOS.
COPY MacOSX10.14.sdk /usr/local/

# Build ISPC
RUN mkdir build
WORKDIR /usr/local/src/ispc/build
RUN cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_CROSS=ON -DISPC_MACOS_SDK_PATH=/usr/local/MacOSX10.14.sdk && make -j8 package
