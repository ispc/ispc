all: ptxcc ptxgen

CXX=clang++
CXXFLAGS += -O3
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

$(OBJ):  ptxgrammar.cc ptx.cc PTXParser.h PTXLexer.h

