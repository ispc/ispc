PROG=mandelbrot_tasks
ISPC_SRC=mandelbrot_tasks.ispc
CXX_SRC=mandelbrot_tasks.cpp  mandelbrot_tasks_serial.cpp
NVCC_SRC=../nvcc_helpers.cu

CXX=g++
CXXFLAGS=-O3 -I$(CUDATK)/include

NVCC=nvcc
NVCC_FLAGS=-O3 -arch=sm_35

LD=nvcc
LDFLAGS=-lcudart -lcudadevrt

PTXCC=ptxcc
PTXCC_FLAGS = -maxrregcount=32 -Xptxas=-v

ISPC=ispc
ISPC_FLAGS=-O3 --math-lib=default --target=nvptx64 --opt=fast-math

# PTXGEN = $(HOME)/ptxgen
# PTXGEN += -opt=3
# PTXGEN += -ftz=1 -prec-div=0 -prec-sqrt=0 -fma=1

###.SUFFIXES: .bc .o .cu 


ISPC_OBJS=objs/$(ISPC_SRC:.ispc=_ispc.o)
ISPC_BCS=objs/$(ISPC_SRC:.ispc=_ispc.bc)
ISPC_HEADERS=objs/$(ISPC_SRC:.ispc=_ispc.h)
CXX_OBJS=objs/$(CXX_SRC:.cpp=_gcc.o)
NVCC_OBJS=objs/$(NVCC_SRC:.cu=_nvcc.o)

OBJS=$(ISPC_OBJS) $(CXX_OBJS) $(NVCC_OBJS)

all: $(PROG)

dirs:
	/bin/mkdir -p objs/

objs/%.cpp objs/%.o objs/%.h: dirs

clean: 
	/bin/rm -rf $(PROG) objs

$(PROG): $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS)

objs/%_gcc.o: %.cpp dirs $(ISPC_HEADERS)
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

objs/%_cu.o: %.cu dirs
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

objs/%_ispc.h objs/%_ispc.bc: %.ispc dirs
	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs/$*_ispc.h -o objs/$*_ispc.bc $<

objs/%_ispc.o: objs/%_ispc.bc 
	$(PTXCC) $(PTXCC_FLAGS) $< -o $@


	 

