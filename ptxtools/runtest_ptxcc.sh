#!/bin/sh
#
#  Copyright (c) 2014, Evghenii Gaburov
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
# 
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# 
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
# 
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
# 
# 
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
  $(nvcc test_static_nvptx.cpp examples/util/nvcc_helpers.cu examples/util/ispc_malloc.cpp $TMPDIR/$fbname.o -arch=sm_35 -Iexamples/util/ -D_CUDA_ -lcudadevrt $ARGS) && \
  $(/bin/rm -rf $TMPDIR/*$fbname*);
else
  $(sed 's/\.b0/\.b32/g' $1 > $TMPDIR/$fbname) && \
  $($PTXCC $TMPDIR/$fbname -o $TMPDIR/$fbname.o -Xnvcc="-G") && \
  $(nvcc test_static_nvptx.cpp examples/util/nvcc_helpers.cu examples/util/ispc_malloc.cpp $TMPDIR/$fbname.o -arch=sm_35 -Iexamples/util/ -D_CUDA_ -lcudadevrt $ARGS) && \
  $(/bin/rm -rf $TMPDIR/*$fbname*);
fi



