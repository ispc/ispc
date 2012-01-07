#!/usr/bin/python

import sys

t=str(sys.argv[1])

print("char stdlib_" + t + "_code[] = { ")

num = 0
for line in sys.stdin:
    for c in line:
        num+=1
        print("0x%0.2X, " % ord(c), end="")
        if num%16 == 0:
            print()

print("0 };")
