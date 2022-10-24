FROM fedora:26
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
# RUN echo "proxy=http://proxy.yourcompany.com:888" >> /etc/dnf/dnf.conf
RUN dnf install -y vim-enhanced-2:8.0.1806 wget-1.19.5 yum-utils-1.1.31 gcc-7.3.1 gcc-c++-7.3.1 git-2.13.6 python3-3.6.4 && \
    dnf install -y m4-1.4.18 bison-3.0.4 flex-2.6.1 patch-2.7.6 make-4.2.1 ncurses-devel-6.0 glibc-devel-2.25-13.fc26.x86_64 glibc-devel-2.25-13.fc26.i686 && \
    dnf clean -y all

# These packages are required if you need to link IPSC with -static.
RUN dnf install -y ncurses-static-6.0 libstdc++-static-7.3.1 glibc-static-2.25 && \
    dnf clean -y all

# Download and install required version of cmake (3.13) for ISPC build
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.13.5-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm cmake-3.13.5-Linux-x86_64.sh

# If you are behind a proxy, you need to configure git and svn.
#RUN git config --global --add http.proxy http://proxy.yourcompany.com:888

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
# By default 8.0 is used.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
ARG LLVM_VERSION=8.0
RUN ./alloy.py -b --version="$LLVM_VERSION" --selfbuild && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp "$LLVM_HOME"/build-"$LLVM_VERSION"_temp

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Configure ISPC build
RUN mkdir build_$LLVM_VERSION
WORKDIR /usr/local/src/ispc/build_$LLVM_VERSION
RUN cmake ../ -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-$LLVM_VERSION

# Build ISPC
RUN make ispc -j"$(nproc)" && make install
WORKDIR /usr/local/src/ispc
RUN rm -rf build_$LLVM_VERSION
