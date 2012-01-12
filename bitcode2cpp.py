#!/usr/bin/python

import sys
import string
import re
import subprocess
import platform
import os

length=0

src=str(sys.argv[1])

target = re.sub("builtins/target-", "", src)
target = re.sub(r"builtins\\target-", "", target)
target = re.sub("builtins/", "", target)
target = re.sub(r"builtins\\", "", target)
target = re.sub("\.ll$", "", target)
target = re.sub("\.c$", "", target)
target = re.sub("-", "_", target)

llvm_as="llvm-as"
if platform.system() == 'Windows' or string.find(platform.system(), "CYGWIN_NT") != -1:
    llvm_as = os.getenv("LLVM_INSTALL_DIR").replace("\\", "/") + "/bin/" + llvm_as

try:
    as_out=subprocess.Popen([llvm_as, "-", "-o", "-"], stdout=subprocess.PIPE)
except IOError:
    sys.stderr.write("Couldn't open " + src)
    sys.exit(1)

width = 16;
sys.stdout.write("unsigned char builtins_bitcode_" + target + "[] = {\n")

data = as_out.stdout.read()
for i in range(0, len(data), 1):
        sys.stdout.write("0x%0.2X, " % ord(data[i:i+1]))

        if i%width == (width-1):
            sys.stdout.write("\n")

sys.stdout.write("0x00 };\n\n")
sys.stdout.write("int builtins_bitcode_" + target + "_length = " + str(i+1) + ";\n")

as_out.wait()

sys.exit(as_out.returncode)
