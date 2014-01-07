#!/bin/sh
LLC=$HOME/usr/local/llvm/bin-trunk/bin/llc
DIS=$HOME/usr/local/llvm/bin-3.2/bin/llvm-dis

ISPC=ispc
PTXCC=ptxcc
PTXGEN=~/ptxgen
$(cat $1 |grep -v 'width'|$ISPC --target=nvptx --emit-llvm -o -|$LLC -march=nvptx64 -mcpu=sm_35 -o $1.ptx) && \
#$(cat $1 |grep -v 'width'|$ISPC --target=nvptx --emit-llvm -o -|$DIS -o $1_32_ptx.ll && $PTXGEN $1_32_ptx.ll > $1.ptx) && \
$($PTXCC $1.ptx  -Xptxas=-v -o $1.ptx.o) && \
nvcc -o test_nvptx test_static_nvptx.cpp examples_ptx/nvcc_helpers.cu examples_ptx/ispc_malloc.cpp $1.ptx.o -arch=sm_35 -Iexamples_ptx/ -D_CUDA_ -lcudadevrt -DTEST_SIG=$2



