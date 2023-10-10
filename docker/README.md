Dockerfiles for reproducible builds
===================================

This folder contains a set of `Dockerfile`s for Intel® ISPC builds, which serve as:

 1. Examples for setting up development environment in different OSes.
 2. Actual `Dockerfile`s used in Github Actions CI.
 3. Archive of `Dockerfile`s used for building specific `ispc` versions.

Development builds
------------------

`CentOS`, `Fedora` and `Ubuntu` `Dockerfiles`s serve dual purpose - as examples for setting up an environment for building `ispc` and as integral parts of CI. This means that while we strive for maximum coverage for different OSes, some of the files might be outdated.

Note that adding all of these `Dockerfile`s to regular CI runs is currently problematic for a couple of reasons:

 * We currently use Github Actions shared runners, which have 6 hours job limit. This is approximately the time required to do LLVM self-build, which we do as part of `ispc` build. This means it's hard to get this kind of jobs reliably passing using shared runners.
 * `Dockerfile`s tend to break from time to time due to changes in the base images and package managers updates. And we don't always have time to maintains it. Contributions are welcome - feel free to submit a PR with the distro you care about or fix an existing `Dockerfile`.

By default `Dockerfile`s are assumed to be built as `x86` images, but some can be built as `aarch64` images (note that `ispc` is a cross-compiler, so regardless the host arch, it can target any supported CPU architecuture, if it's enabled in `ispc` build).

 * [ubuntu/16.04/cpu\_ispc\_build/Dockerfile](ubuntu/16.04/cpu_ispc_build/Dockerfile) `Ubuntu 16.04` image. CPU only.
 * [ubuntu/18.04/cpu\_ispc\_build/Dockerfile](ubuntu/18.04/cpu_ispc_build/Dockerfile) `Ubuntu 18.04` image. CPU only. Ubuntu 18.04 is used to enabled maximum compatibility. This Dockerfile does LLVM selfbuild in two stages, which enables splitting it to two separate CI jobs. The image is used in CI for nightly LLVM builds. Works on both `x86` and `aarch64`.
 * [ubuntu/20.04/cpu\_ispc\_build/Dockerfile](ubuntu/20.04/cpu_ispc_build/Dockerfile) `Ubuntu 20.04` image. CPU only. Works on both `x86` and `aarch64`.
 * [ubuntu/20.04/xpu\_ispc\_build/Dockerfile](ubuntu/20.04/xpu_ispc_build/Dockerfile) `Ubuntu 20.04` image. XPU (CPU+GPU). This is the recommended environment for XPU experiments.
 * [ubuntu/22.04/cpu\_ispc\_build/Dockerfile](ubuntu/22.04/cpu_ispc_build/Dockerfile) `Ubuntu 22.04` image. CPU only. Works on both `x86` and `aarch64`.
 * [ubuntu/22.04/xpu\_ispc\_build/Dockerfile](ubuntu/22.04/xpu_ispc_build/Dockerfile) `Ubuntu 22.04` image. XPU (CPU+GPU).
 * [centos/7/cpu\_ispc\_build/Dockerfile](centos/7/cpu_ispc_build/Dockerfile) `CentOS 7` image. CPU only. Works on both `x86` and `aarch64`.
 * [centos/7/xpu\_ispc\_build/Dockerfile](centos/7/xpu_ispc_build/Dockerfile) `CentOS 7` image. XPU (CPU+GPU). `x86` image only. This image is used for building `ispc` package for future binary releases.
 * [centos/8/cpu\_ispc\_build/Dockerfile](centos/8/cpu_ispc_build/Dockerfile) `CentOS 8` image. CPU only. Works on both `x86` and `aarch64`.
 * [fedora/Dockerfile](fedora/Dockerfile) Fedora 39 with CPU only build linked with system LLVM and Clang shared libraries.

XPU-enabled builds
------------------

The term XPU means going beyond CPU (so it is really xPU, where "x" refers to "anything"). With respect to current state of `ispc`, XPU means Intel® GPU support in addition to CPU targets.

When working on XPU-enabled `ispc` builds, it's highly encouraged to do the development using `Dockerfile`s as it requires having multiple parts of GPU software stack with the right versions. Failing to get the right versions of all components is the easiest way to get the broken build. The recommended OS to work on XPU-enabled build is `Ubuntu 20.04` and later. The `Dockerfile`, which has fully functional GPU environment is here: [ubuntu/xpu\_ispc\_build/Dockerfile](ubuntu/xpu_ispc_build/Dockerfile). It's regularly updated with the latest recommended component versions.

Here's how to build and run XPU-enabled docker image:
```bash
cd docker/ubuntu/xpu_ispc_build
docker build -t ispc_xpu_env:latest .
docker run -it --device=/dev/dri:/dev/dri --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME:$HOME ispc_xpu_env:latest /bin/bash
```

XPU docker `CentOS 7` has build argument `TBB`. It has two possible values
`default` and `oneapi`. Under `default`, tbb dependencies is installed from
system repository. When `oneapi` provided, intel-oneapi-tbb is installed
instead of.

Docker switches used in the command line above are:
 * `--device=/dev/dri:/dev/dri` is required to share GPU device between the host and the container.
 * `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` allows using `gdb` inside the container, so it's not needed if you are not going to debug in Docker.
 * `-v $HOME:$HOME` shares your home directory between the host and the container, so it's handy for development inside the container.

Release builds
--------------

The folders corresponding to Intel® ISPC versions, contain `Dockerfile`s that were used for building Linux binary artifacts available on [Github Releases](https://github.com/ispc/ispc/releases).

Note, that we stick to building `ispc` binary in the environment with the oldest available `glibc`, so the resulting binary is functional on all actual Linux distributions. So, most of released binaries are built using `CentOS 7`.
