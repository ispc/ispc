#!/bin/sh

PTXCC=$ISPC_HOME/ptxtools/ptxcc
PTXGEN=$ISPC_HOME/ptxtools/ptxgen
ARGS=${@:2}
TMPDIR=/tmp
fbname=`basename $1`
if [ "$NVVM" == "1" ];
then
#  LLVM32=$HOME/usr/local/llvm/bin-3.2
#  LLVM34=$HOME/usr/local/llvm/bin-3.4
#  LLVMAS=$LLVM34/bin/llvm-as
#  LLVMDIS=$LLVM32/bin/llvm-dis
#  $($LLVMAS $1 -o $TMPDIR/$fbname.bc) && $($LLVMDIS $TMPDIR/$fbname.bc -o $TMPDIR/$fbname.ll) && $($PTXGEN $TMPDIR/$fbname.ll -o $TMPDIR/$fbname.ptx) && \
  $($PTXGEN $1 -o $TMPDIR/$fbname.ptx) && \
  $($PTXCC $TMPDIR/$fbname.ptx -o $TMPDIR/$fbname.o -Xnvcc="-G") && \
  $(nvcc test_static_nvptx.cpp examples/util/nvcc_helpers.cu examples/util/ispc_malloc.cpp $TMPDIR/$fbname.o -arch=sm_35 -Iexamples/util/ -D_CUDA_ -lcudadevrt $ARGS);
else
  $(sed 's/\.b0/\.b32/g' $1 > $TMPDIR/$fbname) && \
  $($PTXCC $TMPDIR/$fbname -o $TMPDIR/$fbname.o -Xnvcc="-G") && \
  $(nvcc test_static_nvptx.cpp examples/util/nvcc_helpers.cu examples/util/ispc_malloc.cpp $TMPDIR/$fbname.o -arch=sm_35 -Iexamples/util/ -D_CUDA_ -lcudadevrt $ARGS);
fi



