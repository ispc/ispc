#
#  Copyright (c) 2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# Environment to build and run the benchmarks for the performance dashboard (benchmarks-weekly.yml).
# Same packages as ../cpu_ispc_build, but LLVM is the prebuilt package from ispc.dependencies that
# CI uses (see .github/workflows/scripts/init-env.sh). Sources are not part of the image: they are
# mounted by benchmarks-weekly.yml and built by .github/workflows/scripts/run-benchmarks.sh.

FROM ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956
SHELL ["/bin/bash", "-c"]

ARG LLVM_REPO=https://github.com/ispc/ispc.dependencies
ARG LLVM_VERSION
ARG LLVM_TAR

# Packages
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y \
        wget ninja-build build-essential gcc g++ git python3-dev \
        ca-certificates libtbb-dev libc6-dev-i386 xz-utils \
        m4 bison flex libc6-dev-i386-cross libc6-dev-arm64-cross libc6-dev-armhf-cross && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.23.5).
RUN wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 \
        https://cmake.org/files/v3.23/cmake-3.23.5-linux-x86_64.sh && \
    sh cmake-*.sh --prefix=/usr/local --skip-license && rm -rf cmake-*.sh

# Prebuilt LLVM
ENV LLVM_HOME=/usr/local/src/llvm
RUN mkdir -p "$LLVM_HOME" && \
    wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 \
        "$LLVM_REPO/releases/download/llvm-$LLVM_VERSION-ispc-dev/$LLVM_TAR" && \
    tar xf "$LLVM_TAR" -C "$LLVM_HOME" && \
    rm "$LLVM_TAR"
ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Non-root user. benchmarks-weekly.yml runs the container as the runner user anyway.
RUN useradd -m -s /bin/bash -U ispc_user
USER ispc_user
