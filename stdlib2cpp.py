#!/usr/bin/python

import sys

t=str(sys.argv[1])

print "char stdlib_" + t + "_code[] = { "

for line in sys.stdin:
    for c in line:
        print ord(c)
        print ", "

print "0 };"
