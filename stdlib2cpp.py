#!/usr/bin/python

import sys

print "char stdlib_code[] = { "

for line in sys.stdin:
    for c in line:
        print ord(c)
        print ", "

print "0 };"
