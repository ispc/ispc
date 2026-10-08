#!/usr/bin/env python3
#
# Copyright (c) 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Install target C headers used by the current Linux build images."""

import argparse
import hashlib
from pathlib import Path
import shutil
# Commands use fixed argument lists without a shell.
import subprocess  # nosec B404
import tempfile
import zipfile


def download(url, destination, sha256):
    # Ubuntu 18.04 images build Python without the optional SSL module.
    subprocess.run(
        ["wget", "--retry-connrefused", "--waitretry=5", "--timeout=30",
         "--tries=5", "--no-verbose", "-O", str(destination), url],
        check=True,
    )
    digest = hashlib.sha256()
    with destination.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError("Checksum mismatch for " + url)


def install_ndk(directory, temporary):
    archive = temporary / "android-ndk-r27d-linux.zip"
    download(
        "https://dl.google.com/android/repository/android-ndk-r27d-linux.zip",
        archive,
        "601246087a682d1944e1e16dd85bc6e49560fe8b6d61255be2829178c8ed15d9",
    )
    prefix = "android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/"
    # Only C headers are needed. This does not execute NDK host binaries and
    # also works when building the image on an AArch64 host.
    with zipfile.ZipFile(archive) as ndk:
        for entry in ndk.infolist():
            if not entry.filename.startswith(prefix + "usr/include/"):
                continue
            relative = Path(entry.filename[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Invalid path in NDK archive")
            destination = directory / "sysroot" / relative
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with ndk.open(entry) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)


def install_linux_arm_headers(temporary):
    # Supply both ARM cross-header layouts used by ISPC on Fedora and Rocky.
    # Use immutable Debian packages, retaining only their headers.
    packages = (
        ("libc6-dev-arm64-cross_2.41-11cross1_all.deb",
         "61e0b87dec4929abc9fb345ef324139498bf23ddbf0b10ca3a743b59c8ddf8bb"),
        ("linux-libc-dev-arm64-cross_6.12.38-1cross1_all.deb",
         "fd77c13fedf057038732ef1c4fc06edd753d08092fd5f92c4629551f1f69870f"),
        ("libc6-dev-armhf-cross_2.41-11cross1_all.deb",
         "9f0c3d473edb51883949fb4fd61929110aa3e5d6f4a88782aaa95c9ae65031e0"),
        ("linux-libc-dev-armhf-cross_6.12.38-1cross1_all.deb",
         "bf6fa2d938e493a84c34086219b3e3e391aba885d4fb349022eda9fdafa36272"),
    )
    root = temporary / "linux-headers"
    root.mkdir()
    for name, sha256 in packages:
        archive = temporary / name
        download(
            "https://snapshot.debian.org/archive/debian/20250729T142617Z/"
            "pool/main/c/cross-toolchain-base/" + name,
            archive,
            sha256,
        )
        contents = subprocess.run(
            ["ar", "p", str(archive), "data.tar.xz"],
            check=True, stdout=subprocess.PIPE,
        ).stdout
        subprocess.run(["tar", "xJf", "-", "-C", str(root)], input=contents, check=True)
    for triple in ("aarch64-linux-gnu", "arm-linux-gnueabihf"):
        shutil.copytree(
            root / "usr" / triple / "include",
            Path("/usr") / triple / "include",
            dirs_exist_ok=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk-dir", type=Path, default=Path("/usr/local/share/android-ndk"))
    parser.add_argument("--skip-ndk", action="store_true")
    parser.add_argument("--linux-arm-headers", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if not args.skip_ndk:
            install_ndk(args.ndk_dir, root)
        if args.linux_arm_headers:
            install_linux_arm_headers(root)


if __name__ == "__main__":
    main()
