#!/bin/sh
LLC=$HOME/usr/local/llvm/bin-trunk/bin/llc
ISPC=ispc
PTXCC=ptxcc
$(cat $1 |grep -v 'width'|$ISPC --target=nvptx --emit-llvm -o -|$LLC -march=nvptx64 -mcpu=sm_35 -o $1.ptx)
$($PTXCC $1.ptx  -Xptxas=-v -o $1.ptx.o)
