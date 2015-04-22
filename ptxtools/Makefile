#
#  Copyright (c) 2014-2015, Evghenii Gaburov
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

all: ptxcc ptxgen

CXX=clang++
CXXFLAGS += -O3 --std=c++11 -Wno-deprecated-register
CXXFLAGS += -I/opt/local/include

LD=clang++
LDFLAGS += -L/opt/local/lib

FLEX=flex
BISON=bison

CUDATK=/usr/local/cuda
LIBDEVICE_MAJOR=1
LIBDEVICE_MINOR=0

ptxgrammar.cc : ptxgrammar.yy
	$(BISON) -d -v -t  ptxgrammar.yy -o ptxgrammar.cc

ptx.cc: ptx.ll ptxgrammar.cc
	$(FLEX) -t ptx.ll > ptx.cc

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

OBJ= ptxcc.o \
     ptx.o \
		 ptxgrammar.o

ptxcc: $(OBJ)
	$(LD) $(LDFLAGS) $^ -o $@ 

ptxgen: ptxgen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<  \
	  -L$(CUDATK)/nvvm/lib64 -lnvvm \
		-I$(CUDATK)/nvvm/include \
		-I$(CUDATK)/include   \
		-DLIBDEVICE_MAJOR_VERSION=$(LIBDEVICE_MAJOR) \
		-DLIBDEVICE_MINOR_VERSION=$(LIBDEVICE_MINOR) \
		-DLIBNVVM_HOME=$(CUDATK)/nvvm -Wl,-rpath,$(CUDATK)/nvvm/lib64

clean: 
	/bin/rm -f ptxgen ptxcc $(OBJ) ptxgrammar.hh ptxgrammar.cc ptx.cc ptxgrammar.output

$(OBJ):  ptxgrammar.cc ptx.cc PTXParser.h PTXLexer.h GPUTargets.h

