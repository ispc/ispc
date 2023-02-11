FROM fedora:36
LABEL maintainer="Nurmukhametov, Aleksei <aleksei.nurmukhametov@intel.com>"
SHELL ["/bin/bash", "-c"]

ARG REPO=ispc/ispc
ARG SHA=main

# Packages required to build ISPC.
# RUN echo "proxy=http://proxy.yourcompany.com:888" >> /etc/dnf/dnf.conf
RUN dnf install -y git cmake && \
    dnf install -y clang-devel-14.0.5 llvm-devel-14.0.5 glibc-devel.i686 && \
    dnf install -y flex bison && \
    dnf clean -y all

# If you are behind a proxy, you need to configure git and svn.
# RUN git config --global --add http.proxy http://proxy.yourcompany.com:888

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone --depth=1 -b $SHA https://github.com/$REPO.git ispc

ENV ISPC_HOME=/usr/local/src/ispc

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

# Configure ISPC build
RUN mkdir /usr/local/src/ispc/build
WORKDIR /usr/local/src/ispc/build
RUN cmake ../ -DCMAKE_INSTALL_PREFIX=/usr

# LLVM
# We don't build llvm here because we use system shared libraries.

# Build ISPC
RUN make VERBOSE=1 ispc -j"$(nproc)" && make install
RUN make check-all && (echo "shared libs deps of ispc" && ldd ./bin/ispc)
WORKDIR /usr/local/src/ispc
RUN rm -rf build
