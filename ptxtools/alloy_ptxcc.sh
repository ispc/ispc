#!/bin/sh

PTXCC=$ISPC_HOME/ptxtools/ptxcc
PTXGEN=$ISPC_HOME/ptxtools/ptxgen
ARGS=${@:2}
if [ "$NVVM" == "1" ];
then
  LLVM32=$HOME/usr/local/llvm/bin-3.2
  LLVMDIS=$LLVM32/bin/llvm-dis
  $($LLVMDIS $1 -o $1.ll) && $($PTXGEN $1.ll -o $1.ptx) && \
  $($PTXCC $1.ptx -o $1.o -Xnvcc="-G") && \
  $(nvcc test_static_nvptx.cpp examples/nvcc_helpers.cu examples/ispc_malloc.cpp $1.o -arch=sm_35 -Iexamples/ -D_CUDA_ -lcudadevrt $ARGS);
else
  $($PTXCC $1 -o $1.o -Xnvcc="-G") && \
  $(nvcc test_static_nvptx.cpp examples/nvcc_helpers.cu examples/ispc_malloc.cpp $1.o -arch=sm_35 -Iexamples/ -D_CUDA_ -lcudadevrt $ARGS);
fi



