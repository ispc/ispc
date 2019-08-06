#!/usr/bin/env python3

import sys
import re
import subprocess
import platform
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src", help="Source file to process")
parser.add_argument("--runtime", help="Runtime", nargs='?', default='')
parser.add_argument("--os", help="Target OS", default='')
parser.add_argument("--arch", help="Target architecture", default='')
parser.add_argument("--fake", help="Produce empty array", dest='fake', action='store_true', default=False)
parser.add_argument("--llvm_as", help="Path to LLVM assembler executable", dest="path_to_llvm_as")
args = parser.parse_known_args()
src = args[0].src
length=0

target = re.sub(".*builtins/target-", "", src)
target = re.sub(r".*builtins\\target-", "", target)
target = re.sub(".*builtins/", "", target)
target = re.sub(r".*builtins\\", "", target)
target = re.sub("\.ll$", "", target)
target = re.sub("\.c$", "", target)
target = re.sub("-", "_", target)

llvm_as="llvm-as"
if args[0].path_to_llvm_as:
    llvm_as = args[0].path_to_llvm_as
else:
    if platform.system() == 'Windows' or platform.system().find("CYGWIN_NT") != -1:
        llvm_as = os.getenv("LLVM_INSTALL_DIR").replace("\\", "/") + "/bin/" + llvm_as

try:
    if not args[0].fake:
        as_out=subprocess.Popen([llvm_as, "-", "-o", "-"], stdout=subprocess.PIPE)
except IOError:
    sys.stderr.write("Couldn't open " + src)
    sys.exit(1)

name = target
if args[0].runtime != '':
    name += "_" + args[0].runtime + "bit"

# Macro style arguments "UNIX" and "WINDOWS" for .ll to .cpp
if args[0].os == "UNIX":
    target_os = "unix"
elif args[0].os == "WINDOWS":
    target_os = "win"
# Exact OS names for builtins.c
elif args[0].os in ["windows", "linux", "macos", "android", "ios", "ps4"]:
    target_os = args[0].os
else:
    sys.stderr.write("Unknown argument for --os: " + args[0].os)
    sys.exit(1)

target_arch = ""
if args[0].arch in ["i386", "x86_64", "armv7", "arm64", "aarch64"]:
    target_arch = args[0].arch + "_"
    

width = 16

sys.stdout.write("extern const unsigned char builtins_bitcode_" + target_os + "_" + target_arch + name + "[] = {\n")

if args[0].fake:
    data = []
else:
    data = as_out.stdout.read()
    for i in range(0, len(data), 1):
        sys.stdout.write("0x%0.2X," % ord(data[i:i+1]))

        if i%width == (width-1):
            sys.stdout.write("\n")
        else:
            sys.stdout.write(" ")

sys.stdout.write("0x00 };\n\n")
sys.stdout.write("int builtins_bitcode_" + target_os + "_" + target_arch + name + "_length = " + str(len(data)) + ";\n")

if not args[0].fake:
    as_out.wait()
    sys.exit(as_out.returncode)
