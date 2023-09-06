#
#  Copyright (c) 2017-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

FROM fedora:39
LABEL maintainer="Nurmukhametov, Aleksei <aleksei.nurmukhametov@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main

# If you are behind a proxy, you need to configure dnf to use it.
RUN if [ -v http_proxy ]; then echo "proxy=$http_proxy" >> /etc/dnf/dnf.conf; fi

# Packages required to build ISPC.
RUN dnf install -y git cmake python3-setuptools && \
    dnf install -y clang-devel llvm-devel glibc-devel.i686 && \
    dnf install -y flex bison tbb-devel && \
    dnf clean -y all

# If you are behind a proxy, you need to configure git to use it.
RUN if [ -v http_proxy ]; then git config --global --add http.proxy "$http_proxy"; fi

# LLVM
# We don't build llvm here because we use system shared libraries.

# Clone, configure and build ISPC
WORKDIR /usr/local/src
RUN git clone --depth=1 https://github.com/$REPO.git ispc && \
    git -C ispc checkout $SHA && \
    cmake ispc -B build-ispc -DCMAKE_INSTALL_PREFIX=/usr && \
    cmake --build build-ispc -v -j "$(nproc)" && \
    cmake --build build-ispc --target check-all && \
    (echo "shared libs deps of ispc" && ldd ./build-ispc/bin/ispc) && \
    cmake --install build-ispc && \
    rm -rf build ispc
