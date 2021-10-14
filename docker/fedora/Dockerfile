FROM fedora:26
MAINTAINER Dmitry Babokin <dmitry.y.babokin@intel.com>
SHELL ["/bin/bash", "-c"]

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# Packages required to build ISPC and Clang.
# RUN echo "proxy=http://proxy.yourcompany.com:888" >> /etc/dnf/dnf.conf
RUN dnf install -y vim wget yum-utils gcc gcc-c++ git python3 m4 bison flex patch make ncurses-devel zlib-devel glibc-devel.x86_64 glibc-devel.i686 && \
    dnf clean -y all

# These packages are required if you need to link IPSC with -static.
RUN dnf install -y ncurses-static libstdc++-static zlib-static glibc-static && \
    dnf clean -y all

# Download and install required version of cmake (3.13) for ISPC build
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.13.5-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm cmake-3.13.5-Linux-x86_64.sh

# If you are behind a proxy, you need to configure git and svn.
#RUN git config --global --add http.proxy http://proxy.yourcompany.com:888

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/ispc/ispc.git

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
# By default 8.0 is used.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
ARG LLVM_VERSION=8.0
RUN ./alloy.py -b --version=$LLVM_VERSION --selfbuild && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-$LLVM_VERSION_temp $LLVM_HOME/build-$LLVM_VERSION_temp

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Configure ISPC build
RUN mkdir build_$LLVM_VERSION
WORKDIR build_$LLVM_VERSION
RUN cmake ../ -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-$LLVM_VERSION

# Build ISPC
RUN make ispc -j8 && make install
WORKDIR ../
RUN rm -rf build_$LLVM_VERSION
