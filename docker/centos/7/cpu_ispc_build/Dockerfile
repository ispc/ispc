ARG LLVM_VERSION=13.0

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
RUN yum install -y vim-enhanced-7.4.629 wget-1.14 yum-utils-1.1.31 gcc-4.8.5 gcc-c++-4.8.5 git-1.8.3.1 python3-3.6.8 m4-1.4.16 bison-3.0.4 && \
    yum install -y flex-2.5.37 patch-2.7.1 make-3.82 ncurses-devel-5.9 glibc-devel-2.17-326.el7_9.x86_64 glibc-devel-2.17-326.el7_9.i686 xz-5.2.2 devtoolset-7 && \
    yum install -y libtool-2.4.2 gettext-devel-0.19.8.1 texinfo-5.1 help2man-1.41.1 && \
    yum clean -y all

# These packages are required if you need to link IPSC with -static.
RUN yum install -y ncurses-static-5.9 libstdc++-static-4.8.5 && \
    yum clean -y all

# Download and install required version of cmake (3.14 for x86, 3.20 for aarch64, as earlier versions are not available as binary distribution) for ISPC build
RUN if [[ $(uname -m) =~ "x86" ]]; then export CMAKE_URL="https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh"; else export CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3-linux-aarch64.sh"; fi && \
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
RUN source /opt/rh/devtoolset-7/enable && \
    ./alloy.py -b --version="$LLVM_VERSION" --selfbuild --llvm-disable-assertions && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp "$LLVM_HOME"/build-"$LLVM_VERSION"_temp

FROM llvm_only AS ispc_build

ARG LLVM_VERSION
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Install news flex (2.6.4)
WORKDIR /usr/local/src
RUN git clone https://github.com/westes/flex.git
WORKDIR /usr/local/src/flex
RUN git checkout v2.6.4 && ./autogen.sh && ./configure && make -j"$(nproc)" && make install

# Build ISPC
RUN mkdir build
WORKDIR /usr/local/src/ispc/build
RUN cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_CROSS=ON && make -j"$(nproc)" package && make check-all
