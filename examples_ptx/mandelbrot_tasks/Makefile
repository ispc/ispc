PROG=mandelbrot_tasks
ISPC_SRC=mandelbrot_tasks.ispc
CXX_SRC=mandelbrot_tasks.cpp  mandelbrot_tasks_serial.cpp
NVCC_SRC=../nvcc_helpers.cu
NVCC_OBJS=objs/nvcc_helpers_nvcc.o

CXX=g++
CXXFLAGS=-O3 -I$(CUDATK)/include -Iobjs/ -D_CUDA_

NVCC=nvcc
NVCC_FLAGS=-O3 -arch=sm_35 -D_CUDA_

LD=nvcc
LDFLAGS=-lcudart -lcudadevrt -arch=sm_35

PTXCC=ptxcc
PTXCC_FLAGS = -maxrregcount=32 -Xptxas=-v

ISPC=ispc
ISPC_FLAGS=-O3 --math-lib=default --target=nvptx64 --opt=fast-math

# PTXGEN = $(HOME)/ptxgen
# PTXGEN += -opt=3
# PTXGEN += -ftz=1 -prec-div=0 -prec-sqrt=0 -fma=1

 .SUFFIXES: .bc .o .cu 


ISPC_OBJS=$(ISPC_SRC:%.ispc=objs/%_ispc.o)
ISPC_BCS=$(ISPC_SRC:%.ispc=objs/%_ispc.bc)
ISPC_HEADERS=$(ISPC_SRC:%.ispc=objs/%_ispc.h)
CXX_OBJS=$(CXX_SRC:%.cpp=objs/%_gcc.o)
#NVCC_OBJS=$(NVCC_SRC:%.cu=objs/%_nvcc.o)

OBJS=$(ISPC_OBJS) $(CXX_OBJS) $(NVCC_OBJS)

all: dirs $(PROG) $(ISPC_BCS) 

dirs:
	/bin/mkdir -p objs/

objs/%.cpp objs/%.o objs/%.h: dirs

clean: 
	echo $(CXX_OBJS)
	/bin/rm -rf $(PROG) objs

$(PROG): $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS)

objs/%_gcc.o: %.cpp $(ISPC_HEADERS)
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs/%_gcc.o: ../%.cpp 
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

objs/%_nvcc.o: ../%.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<
objs/%_nvcc.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

objs/%_ispc.h objs/%_ispc.bc: %.ispc 
	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs/$*_ispc.h -o objs/$*_ispc.bc $<

objs/%_ispc.o: objs/%_ispc.bc 
	$(PTXCC) $< $(PTXCC_FLAGS) -o $@


	 

