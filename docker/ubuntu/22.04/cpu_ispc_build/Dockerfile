ARG LLVM_VERSION=14.0

FROM ubuntu:22.04 AS llvm_build_only
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
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y wget cmake build-essential gcc g++ git python3-dev ncurses-dev libtinfo-dev ca-certificates && \
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

RUN apt-get -y update && apt-get --no-install-recommends install -y m4 bison flex \
    libc6-dev-i386-cross libc6-dev-arm64-cross libc6-dev-armhf-cross && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/src/ispc
RUN mkdir -p build
WORKDIR /home/src/ispc/build
RUN cmake .. -DX86_ENABLED=ON -DARM_ENABLED=ON -DCMAKE_CXX_FLAGS=-Werror && make -j"$(nproc)" && make check-all
# Add ISPC to PATH
ENV PATH=/home/ispc/bin:$PATH
