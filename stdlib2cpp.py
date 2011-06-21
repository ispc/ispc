#!/usr/bin/python

import sys

print "const char *stdlib_code = "
for line in sys.stdin:
    l=line.rstrip()
    l=l.replace('"', '\\"')
    print "\"" + l + "\\n\""

print ";"
