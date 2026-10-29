#!/bin/bash -e
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Runs the C builtins tests (tests/c-builtins) for Android (#3887): links them with
# the Android NDK and runs them with qemu-user on the Android system libraries
# (bionic) of the termux-docker images, so no Android device or emulator is needed.
# Usage: run-c-builtins-android.sh <directory with android-<arch>.o> <arch>...
# ANDROID_NDK_HOME must be set to the Android NDK.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/../../../tests/c-builtins/run.sh"
OBJ_DIR=$1
shift
CLANG="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
# API level 23 is the first one with the stdout symbol, which the C builtins use.
API=23

for arch in "$@"; do
  case "$arch" in
    x86-64)
      triple=x86_64-linux-android; qemu=qemu-x86_64; platform=linux/amd64
      interpreter=system/bin/linker64; libc=system/lib64/libc.so
      image=termux/termux-docker:x86_64@sha256:cc9e95e81e351daed171781caaf77017678b65a0cb8eadfbe187d84e8bf33809 ;;
    arm)
      triple=armv7a-linux-androideabi; qemu=qemu-arm; platform=linux/arm/v7
      interpreter=system/bin/linker; libc=system/lib/libc.so
      image=termux/termux-docker:arm@sha256:3f5fae666a6bff546e291537f669bfe5b055f5f8e794321254575ef1924070df ;;
    *) echo "Unknown architecture: $arch"; exit 1 ;;
  esac

  # Unpack /system of the image: the Android dynamic linker and bionic. It links
  # to /data/data/com.termux/files/usr/opt/aosp.
  root=$(realpath -m "$PWD/android-root-$arch")
  mkdir -p "$root"
  container=$(docker create --platform "$platform" "$image")
  docker export "$container" | tar -x -C "$root" system data/data/com.termux/files/usr/opt/aosp
  docker rm "$container"

  # Keep /system relative so it resolves inside the extracted filesystem.
  ln -sfn data/data/com.termux/files/usr/opt/aosp "$root/system"

  for library in "$interpreter" "$libc"; do
    resolved=$(realpath -e "$root/$library")
    if [[ "$resolved" != "$root/"* || ! -f "$resolved" ]]; then
      echo "FAIL: /$library does not resolve inside $root: $resolved"
      exit 1
    fi
    echo "/$library -> $resolved"
    "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" -h "$resolved"
  done

  RUN_WRAPPER="$qemu -L $root" sh "$RUN" "$OBJ_DIR/android-$arch.o" "$CLANG" "--target=$triple$API"
done
