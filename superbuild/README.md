# Superbuild

It may be used to build ISPC with dependencies (LLVM, L0, vc-intrinsics,
SPIRV-Translator), as well as it can produce an archive with dependencies or
consume a pre-built archive to build ISPC only. It also allows to generate LTO
or LTO+PGO enabled build of LLVM and ISPC.

# Use Cases

Typical use cases:

```bash
# Basic configuration.
$ cmake ${ISPC}/superbuild --preset os
$ cmake --build .
# Produce an archive with dependencies.
$ cmake --build . --target package-stage2
# On Windows, one may want to use nmake generators
$ cmake ${ISPC}/superbuild --preset os -G "NMake Makefiles"

# Enable LTO.
$ cmake ${ISPC}/superbuild --preset os -DLTO=ON
# Enable LTO and PGO.
$ cmake ${ISPC}/superbuild --preset os -DPGO=ON
# Provide LLVM version and source url or directory (same for other dependencies).
$ cmake ${ISPC}/superbuild --preset os -DLLVM_VERSION=15.0 -DLLVM_URL=/home/llvm-project
$ cmake ${ISPC}/superbuild --preset os -DLLVM_VERSION=13_0
# Enable cache using.
$ cmake ${ISPC}/superbuild --preset os -DCCACHE=ON
# Enable cache using where cache located in not default directory.
$ cmake ${ISPC}/superbuild --preset os -DCCACHE=ON -DCCACHE_DIR=/tmp/shared-ispc-ccache
# Consume pre-built dependencies as archive.
$ cmake ${ISPC}/superbuild --preset os -DPREBUILT_STAGE2=/path/to/stage2-archive.tgz

# Build and install only stage2 toolchain
$ cmake ${ISPC}/superbuild --preset os -DSTAGE2_TOOLCHAIN_INSTALL_PREFIX=/tmp/stage2-path -DBUILD_STAGE2_TOOLCHAIN_ONLY=ON
$ cmake --build .
$ cmake --buidl . --target install-stage2-toolchain
# Consume pre-build dependencies as path with installed stage2 toolchain and libs.
$ cmake ${ISPC}/superbuild --preset os -DPREBUILT_STAGE2_PATH=/tmp/stage2-path
```

# Building Process.

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
