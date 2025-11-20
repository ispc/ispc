#!/usr/bin/env python3
#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

import sys
import re
import argparse
from os.path import basename, dirname
from os import replace
from tempfile import NamedTemporaryFile


# Read input data and put it in the form of byte array in the source file.
def write_data(src, outfile):
    width = 16
    with open(src, 'rb') as file:
        data = file.read()
        for i in range(0, len(data), 1):
            outfile.write("0x%0.2X," % ord(data[i:i+1]))
            if i%width == (width-1):
                outfile.write("\n")
            else:
                outfile.write(" ")
        return len(data)
    return 0


def bundle_header(src, output):
    name = basename(output).replace(".", "_")
    with open(src, 'r') as infile:
        with open(output, 'w') as outfile:
            outfile.write("#include \"bitcode_lib.h\"\n\n")
            outfile.write("using namespace ispc;\n\n")
            outfile.write(f"extern const char {name}_header[] = ")
            outfile.write("{\n")
            # Read input data and put it in the form of byte array in the source file.
            length = write_data(src, outfile)
            outfile.write("0x00 };\n\n")
            outfile.write(f"int {name}_length = {length};\n")

ARCH_CHOICES = ['i686', 'x86', 'x86_64', 'arm', 'armv8a', 'arm64', 'riscv64', 'aarch64', 'wasm32', 'wasm64', 'xe64']

parser = argparse.ArgumentParser()
parser.add_argument("src", help="Source file to process")
parser.add_argument("output", help="Output file")
parser.add_argument("--type", help="Type of processed file", choices=['dispatch', 'builtins-c', 'ispc-target', 'stdlib', 'header'], required=True)
parser.add_argument("--runtime", help="Runtime", choices=['32', '64'], nargs='?', default='')
parser.add_argument("--os", help="Target OS", choices=['windows', 'linux', 'macos', 'freebsd', 'android', 'ios', 'ps4', 'web', 'WINDOWS', 'UNIX', 'WEB'], default='')
parser.add_argument("--arch", help="Target architecture", choices=ARCH_CHOICES, default='')

args = parser.parse_known_args()
src = args[0].src
output = args[0].output
length=0

if args[0].type == 'header':
    bundle_header(src, output)
    sys.exit(0)

target = basename(src)
target = re.sub(r"^builtins_", "", target)
target = re.sub(r"^target_", "", target)
target = re.sub(r"^stdlib_", "", target)
target = re.sub(r"\.bc$", "", target)
target = re.sub(r"\.ll$", "", target)
target = re.sub(r"\.c$", "", target)
target = re.sub(r"_32bit.*$$", "", target)
target = re.sub(r"_64bit.*$$", "", target)

name = target
if args[0].runtime != '':
    name += "_" + args[0].runtime + "bit"

# Macro style arguments "UNIX", "WINDOWS", and "WEB" for .ll to .cpp (dispatch and targets)
if args[0].os == "UNIX":
    target_os_old = "unix"
    target_os = "linux"
elif args[0].os == "WINDOWS":
    target_os_old = "win"
    target_os = "windows"
elif args[0].os == "WEB":
    target_os_old = "web"
    target_os = "web"
# Exact OS names for builtins.c
elif args[0].os in ["windows", "linux", "macos", "freebsd", "android", "ios", "ps4", "web"]:
    target_os_old = args[0].os
    target_os = args[0].os
else:
    sys.stderr.write("Unknown argument for --os: " + args[0].os + "\n")
    sys.exit(1)

target_arch = ""
ispc_arch = ""
if args[0].arch in ARCH_CHOICES:
    target_arch = args[0].arch + "_"
    # Canoncalization of arch value for Arch enum in ISPC.
    if args[0].arch == "i686" or args[0].arch == "x86":
        ispc_arch = "x86"
    elif args[0].arch == "x86_64" or args[0].arch == "amd64":
        ispc_arch = "x86_64"
    elif args[0].arch == "armv8a" or args[0].arch == "arm":
        ispc_arch = "arm"
    elif args[0].arch == "arm64" or args[0].arch == "aarch64":
        ispc_arch = "aarch64"
    elif args[0].arch == "riscv64":
        ispc_arch = "riscv64"
    elif args[0].arch == "wasm32":
        ispc_arch = "wasm32"
    elif args[0].arch == "wasm64":
        ispc_arch = "wasm64"
    elif args[0].arch == "xe64":
        ispc_arch = args[0].arch

name_prefix = "stdlib_bitcode_" if args[0].type == 'stdlib' else "builtins_bitcode_"
name = name_prefix + target_os_old + "_" + target_arch + name;

# with open(output, 'w') as outfile:
with NamedTemporaryFile(mode='w', dir=dirname(output), delete=False) as outfile:
    temp_file_name = outfile.name

    outfile.write("#include \"bitcode_lib.h\"\n\n")

    outfile.write("using namespace ispc;\n\n")

    outfile.write("extern const unsigned char " + name + "[] = {\n")

    length = write_data(src, outfile)

    outfile.write("0x00 };\n\n")
    outfile.write(f"int {name}_length = {length};\n")

    # There are 3 types of bitcodes to handle (dispatch module, builtins-c, and target),
    # each needs to be registered differently.
    if args[0].type == "dispatch":
        # For dispatch the only parameter is TargetOS.
        outfile.write(f"static BitcodeLib {name}_lib({name}, {name}_length, TargetOS::{target_os});\n")
    elif args[0].type == "builtins-c":
        # For builtin-c we care about TargetOS and Arch.
        outfile.write((f"static BitcodeLib {name}_lib({name}, {name}_length, TargetOS::{target_os}, "
                       f"Arch::{ispc_arch});\n"))
    elif args[0].type == 'ispc-target' or args[0].type == 'stdlib':
        # For ISPC target files we care about ISPCTarget id, TargetOS type (Windows/Unix), and runtime type (32/64).
        arch = "error"
        if ("sse" in target) or ("avx" in target):
            arch = "x86" if args[0].runtime == "32" else "x86_64" if args[0].runtime == "64" else "error"
        elif "neon" in target:
            arch = "arm" if args[0].runtime == "32" else "aarch64" if args[0].runtime == "64" else "error"
        elif "rvv" in target:
            arch = "riscv64"
        elif "wasm" in target and not "generic" in target:
            arch = "wasm32" if args[0].runtime == "32" else "wasm64" if args[0].runtime == "64" else "error"
        elif "xe" in target:
            arch = "xe64"
        elif ("generic" in target):
            arch = args[0].arch
            target = "_".join(target.split("_")[0:2])
        else:
            sys.stderr.write("Unknown target detected: " + target + "\n")
            sys.exit(1)
        if args[0].type == 'stdlib':
            outfile.write((f"static BitcodeLib {name}_lib(BitcodeLib::BitcodeLibType::Stdlib, {name}, {name}_length, "
                           f"ISPCTarget::{target}, TargetOS::{target_os}, Arch::{arch});\n"))
        else:
            outfile.write((f"static BitcodeLib {name}_lib({name}, {name}_length, ISPCTarget::{target}, "
                           f"TargetOS::{target_os}, Arch::{arch});\n"))
    else:
        sys.stderr.write("Unknown argument for --type: " + args[0].type + "\n")
        sys.exit(1)

replace(temp_file_name, output)
