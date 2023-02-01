ARG LLVM_VERSION=13.0

FROM ubuntu:20.04 AS llvm_build_only
LABEL maintainer="Dmitry Babokin <dmitry.y.babokin@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main
ARG LLVM_VERSION
ARG EXTRA_BUILD_ARG

# !!! Make sure that your docker config provides enough memory to the container,
# otherwise LLVM build may fail, as it will use all the cores available to container.

RUN uname -a

# Packages
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y wget=1.20.3-1ubuntu2 cmake=3.16.3-1ubuntu1 build-essential=12.8ubuntu1.1 \
    gcc=4:9.3.0-1ubuntu2 g++=4:9.3.0-1ubuntu2 git=1:2.25.1-1ubuntu3.8 python3-dev=3.8.2-0ubuntu2 libncurses-dev=6.2-0ubuntu2 libtinfo-dev=6.2-0ubuntu2 ca-certificates=20211016ubuntu0.20.04.1 && \
    rm -rf /var/lib/apt/lists/*

# If you are behind a proxy, you need to configure git.
RUN if [ -v "$http_proxy" ]; then git config --global --add http.proxy "$http_proxy"; fi

WORKDIR /home/src

RUN git clone https://github.com/$REPO.git ispc

WORKDIR /home/src/ispc
RUN git checkout $SHA
ENV ISPC_HOME=/home/src/ispc

# LLVM
ENV LLVM_HOME=/home/tools/llvm
RUN python3 ./alloy.py -b --version="$LLVM_VERSION" --selfbuild --verbose "$EXTRA_BUILD_ARG" && \
    rm -rf "$LLVM_HOME"/build-"$LLVM_VERSION" "$LLVM_HOME"/llvm-"$LLVM_VERSION" "$LLVM_HOME"/bin-"$LLVM_VERSION"_temp "$LLVM_HOME"/build-"$LLVM_VERSION"_temp
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

FROM llvm_build_only AS ispc_build

ARG LLVM_VERSION

RUN apt-get -y update && apt-get --no-install-recommends install -y m4=1.4.18-4 bison=2:3.5.1+dfsg-1 flex=2.6.4-6.2 \
    libc6-dev-i386-cross=2.31-0ubuntu9.9cross1 libc6-dev-arm64-cross=2.31-0ubuntu9.9cross1 libc6-dev-armhf-cross=2.31-0ubuntu9.9cross1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/src/ispc
RUN mkdir -p build
WORKDIR /home/src/ispc/build
RUN cmake .. -DX86_ENABLED=ON -DARM_ENABLED=ON -DCMAKE_CXX_FLAGS=-Werror && make -j"$(nproc)" && make check-all
# Add ISPC to PATH
ENV PATH=/home/ispc/bin:$PATH
