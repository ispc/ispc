#!/usr/bin/python

import sys

t=str(sys.argv[1])

sys.stdout.write("char stdlib_" + t + "_code[] = {\n")

num = 0
for line in sys.stdin:
    for c in line:
        num+=1
        sys.stdout.write("0x%0.2X, " % ord(c))
        if num%16 == 0:
            sys.stdout.write("\n")
sys.stdout.write("0 };\n")
