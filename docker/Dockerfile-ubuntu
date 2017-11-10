FROM ubuntu:16.04
MAINTAINER Dmitry Babokin <dmitry.y.babokin@intel.com>

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

# If you are behind a proxy, let apt-get know about it
#ENV http_proxy=http://proxy.yourcompany.com:888

# Packages required to build ISPC and Clang.
RUN apt-get -y update && apt-get install -y vim cmake gcc g++ git subversion python m4 bison flex zlib1g-dev ncurses-dev libtinfo-dev libc6-dev-i386 && \
    rm -rf /var/lib/apt/lists/*

# If you are behind a proxy, you need to configure git and svn.
#RUN git config --global --add http.proxy http://proxy.yourcompany.com:888
# Initialize svn configs
#RUN svn --version --quiet
#RUN echo "http-proxy-host=proxy.yourcompany.com" >> ~/.subversion/servers
#RUN echo "http-proxy-port=888" >> ~/.subversion/servers

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

# Build Clang 5.0 with all required patches.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
RUN ./alloy.py -b --version=5.0 --selfbuild && \
    rm -rf $LLVM_HOME/build-5.0 $LLVM_HOME/llvm-5.0 $LLVM_HOME/bin-5.0_temp $LLVM_HOME/build-5.0_temp

ENV PATH=$LLVM_HOME/bin-5.0/bin:$PATH

RUN make -j8
