FROM centos:7 AS llvm_only
MAINTAINER Dmitry Babokin <dmitry.y.babokin@intel.com>

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION=12.0

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
RUN yum -y update; yum -y install centos-release-scl epel-release; yum clean all
RUN yum install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison flex patch make ncurses-devel glibc-devel.x86_64 glibc-devel.i686 xz devtoolset-7 && \
    yum install -y libtool autopoint gettext-devel texinfo help2man && \
    yum clean -y all

# These packages are required if you need to link IPSC with -static.
RUN yum install -y ncurses-static libstdc++-static && \
    yum clean -y all

# Download and install required version of cmake (3.14 for x86, 3.20 for aarch64, as earlier versions are not available as binary distribution) for ISPC build
RUN if [[ `uname -m` =~ "x86" ]]; then export CMAKE_URL="https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh"; else export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3-linux-aarch64.sh"; fi && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $CMAKE_URL && mkdir /opt/cmake && sh cmake-*.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm -rf cmake-*.sh

# If you are behind a proxy, you need to configure git.
RUN if [ -v "$http_proxy" ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/$REPO.git ispc
RUN cd ispc && git checkout $SHA && cd ..

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
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
RUN source /opt/rh/devtoolset-7/enable && \
    ./alloy.py -b --version=$LLVM_VERSION --selfbuild --llvm-disable-assertions && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-"$LLVM_VERSION"_temp $LLVM_HOME/build-"$LLVM_VERSION"_temp

FROM llvm_only AS ispc_build
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Install newer zlib
WORKDIR /usr/local/src
RUN git clone https://github.com/madler/zlib.git && cd zlib && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && make install

# Install news flex (2.6.4)
WORKDIR /usr/local/src
RUN git clone https://github.com/westes/flex.git && cd flex && git checkout v2.6.4 && ./autogen.sh && ./configure && make -j8 && make install

# Build ISPC
RUN mkdir build
WORKDIR /usr/local/src/ispc/build
RUN cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_CROSS=ON && make -j8 package && make check-all
