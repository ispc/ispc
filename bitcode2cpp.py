#!/usr/bin/env python3

import sys
import re
import subprocess
import platform
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src", help="Source file to process")
parser.add_argument("--type", help="Type of processed file", choices=['dispatch', 'builtins-c', 'ispc-target'], required=True)
parser.add_argument("--runtime", help="Runtime", choices=['32', '64'], nargs='?', default='')
parser.add_argument("--os", help="Target OS", choices=['windows', 'linux', 'macos', 'freebsd', 'android', 'ios', 'ps4', 'web', 'WINDOWS', 'UNIX', 'WEB'], default='')
parser.add_argument("--arch", help="Target architecture", choices=['i686', 'x86_64', 'armv7', 'arm64', 'aarch64', 'wasm32', 'xe32', 'xe64'], default='')
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
    as_out=subprocess.Popen([llvm_as, "-", "-o", "-"], stdout=subprocess.PIPE)
except IOError:
    sys.stderr.write("Couldn't open " + src + "\n")
    sys.exit(1)

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
if args[0].arch in ["i686", "x86_64", "amd64", "armv7", "arm64", "aarch64", "wasm32", "xe32", "xe64"]:
    target_arch = args[0].arch + "_"
    # Canoncalization of arch value for Arch enum in ISPC.
    if args[0].arch == "i686":
        ispc_arch = "x86"
    elif args[0].arch == "x86_64" or args[0].arch == "amd64":
        ispc_arch = "x86_64"
    elif args[0].arch == "armv7":
        ispc_arch = "arm"
    elif args[0].arch == "arm64" or args[0].arch == "aarch64":
        ispc_arch = "aarch64"
    elif args[0].arch == "wasm32":
        ispc_arch = "wasm32"
    elif args[0].arch == "xe32" or args[0].arch == "xe64":
        ispc_arch = args[0].arch

width = 16

name = "builtins_bitcode_" + target_os_old + "_" + target_arch + name;

sys.stdout.write("#include \"bitcode_lib.h\"\n\n")

sys.stdout.write("using namespace ispc;\n\n")

sys.stdout.write("extern const unsigned char " + name + "[] = {\n")

# Read input data and put it in the form of byte array in the source file.
data = as_out.stdout.read()
for i in range(0, len(data), 1):
    sys.stdout.write("0x%0.2X," % ord(data[i:i+1]))
    if i%width == (width-1):
        sys.stdout.write("\n")
    else:
        sys.stdout.write(" ")

sys.stdout.write("0x00 };\n\n")
sys.stdout.write("int " + name + "_length = " + str(len(data)) + ";\n")

# There are 3 types of bitcodes to handle (dispatch module, builtins-c, and target),
# each needs to be registered differently.
if args[0].type == "dispatch":
    # For dispatch the only parameter is TargetOS.
    sys.stdout.write("static BitcodeLib " + name + "_lib(" +
        name + ", " +
        name + "_length, " +
        "TargetOS::" + target_os +
        ");\n")
elif args[0].type == "builtins-c":
    # For builtin-c we care about TargetOS and Arch.
    sys.stdout.write("static BitcodeLib " + name + "_lib(" +
        name + ", " +
        name + "_length, " +
        "TargetOS::" + target_os + ", " +
        "Arch::" + ispc_arch +
        ");\n")
elif args[0].type == 'ispc-target':
    # For ISPC target files we care about ISPCTarget id, TargetOS type (Windows/Unix), and runtime type (32/64).
    arch = "error"
    if ("sse" in target) or ("avx" in target):
        arch = "x86" if args[0].runtime == "32" else "x86_64" if args[0].runtime == "64" else "error"
    elif "neon" in target:
        arch = "arm" if args[0].runtime == "32" else "aarch64" if args[0].runtime == "64" else "error"
    elif "wasm" in target:
        arch = "wasm32"
    elif ("gen9" in target) or ("xe" in target):
        arch = "xe32" if args[0].runtime == "32" else "xe64" if args[0].runtime == "64" else "error"
    else:
        sys.stderr.write("Unknown target detected: " + target + "\n")
        sys.exit(1)
    sys.stdout.write("static BitcodeLib " + name + "_lib(" +
        name + ", " +
        name + "_length, " +
        "ISPCTarget::" + target + ", " +
        "TargetOS::" + target_os + ", " +
        "Arch::" + arch +
        ");\n")
else:
    sys.stderr.write("Unknown argument for --type: " + args[0].type + "\n")
    sys.exit(1)

as_out.wait()
sys.exit(as_out.returncode)
