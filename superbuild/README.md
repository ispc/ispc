# Superbuild

It may be used to build ISPC with dependencies (LLVM, L0, vc-intrinsics,
SPIRV-Translator), as well as it can produce an archive with dependencies or
consume a pre-built archive to build ISPC only. It also allows to generate LTO
or LTO+PGO enabled build of LLVM and ISPC.

# Use Cases

Typical use cases:

```bash
# Basic configuration.
$ cmake ${ISPC_HOME}/superbuild --preset os
$ cmake --build .
# Produce an archive with dependencies.
$ cmake --build . --target package-stage2
# On Windows, one may want to use nmake generators
$ cmake ${ISPC_HOME}/superbuild --preset os -G "NMake Makefiles"

# Enable LTO.
$ cmake ${ISPC_HOME}/superbuild --preset os -DLTO=ON
# Enable LTO and PGO.
$ cmake ${ISPC_HOME}/superbuild --preset os -DPGO=ON
# Provide LLVM version and source url or directory (same for other dependencies).
$ cmake ${ISPC_HOME}/superbuild --preset os -DLLVM_VERSION=15.0 -DLLVM_URL=/home/llvm-project
$ cmake ${ISPC_HOME}/superbuild --preset os -DLLVM_VERSION=13_0
# Enable cache using.
$ cmake ${ISPC_HOME}/superbuild --preset os -DCCACHE=ON
# Enable cache using where cache located in not default directory.
$ cmake ${ISPC_HOME}/superbuild --preset os -DCCACHE=ON -DCCACHE_DIR=/tmp/shared-ispc-ccache
# Consume pre-built dependencies as archive.
$ cmake ${ISPC_HOME}/superbuild --preset os -DPREBUILT_STAGE2=/path/to/stage2-archive.tgz

# Build and install only stage2 toolchain
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/tmp/stage2-path -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON
$ cmake --build .
# Consume pre-build dependencies as path with installed stage2 toolchain and libs.
$ cmake ${ISPC_HOME}/superbuild --preset os -DPREBUILT_STAGE2_PATH=/tmp/stage2-path

# Build and install stage2 toolchain with XE deps
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/tmp/stage2-path -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON -DINSTALL_WITH_XE_DEPS=ON
$ cmake --build .

# Build and install only stage2 ISPC
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/tmp/ispc -DPREBUILT_STAGE2=/path/to/stage2-archive.tar.gz
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/tmp/ispc -DLTO=ON -DPREBUILT_STAGE2=/path/to/stage2-lto-archive.tar.gz
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/tmp/ispc -DPGO=ON -DPREBUILT_STAGE2=/path/to/stage2-pgo-archive.tar.gz

# Build and install only dependencies
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/opt/spirv-translator -DBUILD_SPIRV_TRANSLATOR_ONLY=ON
$ cmake --build .
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/opt/vc-intrinsics -DBUILD_VC_INTRINSICS_ONLY=ON
$ cmake --build .
$ cmake ${ISPC_HOME}/superbuild --preset os -DCMAKE_INSTALL_PREFIX=/opt/l0-loader -DBUILD_L0_LOADER_ONLY=ON
$ cmake --build .

# Build and install ISPC with XE dependencies
$ cmake ${ISPC_HOME}/superbuild --preset os -DPREBUILT_STAGE2_PATH=/tmp/stage2-path -DCMAKE_INSTALL_PREFIX=/opt/ispc-with-xe/ -DINSTALL_WITH_XE_DEPS=ON
$ cmake --build .

# Run check-all target of ISPC stage2
$ cmake --build . --target ispc-stage2-check-all

# Run check-all, test and ispc_benchmarks targers of ISPC stage2
$ cmake --build . --target ispc-stage2-check

# Package llvm stage2 toolchain that can be used as -DPREBUILT_STAGE2 argument
$ cmake --build . --target package-stage2

# Disable LLVM assertions
$ cmake ${ISPC_HOME}/superbuild --preset os -DLLVM_DISABLE_ASSERTIONS=OFF

# macOS universal builds
$ cmake ${ISPC_HOME}/superbuild --preset os DMACOS_UNIVERSAL_BIN=ON -DISPC_ANDROID_NDK_PATH=<ndk-path>

# macOS x86_64 and arm64 only builds
$ cmake ${ISPC_HOME}/superbuild --preset os DMACOS_UNIVERSAL_BIN=OFF -DCMAKE_OSX_ARCHITECTURES=arm64 -DISPC_ANDROID_NDK_PATH=<ndk-path>
$ cmake ${ISPC_HOME}/superbuild --preset os DMACOS_UNIVERSAL_BIN=OFF -DCMAKE_OSX_ARCHITECTURES=x86_64 -DISPC_ANDROID_NDK_PATH=<ndk-path>

```

# Build Process

The building process consists of several stages.
Base and LTO build consists of two stages. The only difference between them
is that we pass -flto flags to compiler and linker on the second stage. The
first stage creates a minimal compiler toolchain (clang, lld, llvm-dis,
llvm-as) to build LLVM/ISPC.  On the second stage, stage1 compiler builds clang
and LLVM libraries needed for building ISPC. They are combined together with
stage1 clang to become stage2 toolchain, i.e., stage1 and stage2 clangs are
literally same. We may build the stage2 clang by the stage1 clang, but it takes
time especially with LTO/PGO enabled, whereas time profit from better
performant compiler for building ISPC/stage3 is not comparable with LTO/PGO
overhead in the case when system compiler is not ancient. Stage2 compiler is
used to build ISPC.

PGO build has one more stage (3 stages overall). On the first stage, we
additionally build a compiler runtime library needed for profiling and
llvm-profdata tool utilized to merging profiles. During stage2, we generate
instrumented stage2 LLVM libraries and ISPC. That ISPC binary generates
profile files upon every execution. Profile collection happens on source
files from `ispc-corpus`. `ispc-corpus` is the collection of ISPC preprocessed
real world source codes complemented with scripts to build them. It allows us
to collect profile meaningful from the customer point of view. On the last
third stage, stage1 compiler uses that profile to optimize LLVM and ISPC code
with PGO.
