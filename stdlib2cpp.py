#!/usr/bin/env python3

import sys

t=str(sys.argv[1])

sys.stdout.write("namespace ispc {\n")
sys.stdout.write("extern const char stdlib_" + t + "_code[] = {\n")

width = 16
data = sys.stdin.read()
for i in range(0, len(data), 1):
    sys.stdout.write("0x%0.2X, " % ord(data[i:i+1]))

    if i%width == (width-1):
        sys.stdout.write("\n")

sys.stdout.write("0x00 };\n")
sys.stdout.write("} // namespace ispc\n")
                                    
