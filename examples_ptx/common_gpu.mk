NVCC_SRC=../nvcc_helpers.cu
NVCC_OBJS=objs_gpu/nvcc_helpers_nvcc.o
#
CXX=g++ -ffast-math
CXXFLAGS=-O3 -I$(CUDATK)/include -Iobjs_gpu/ -D_CUDA_
#
NVCC=nvcc
NVCC_FLAGS+=-O3 -arch=sm_35 -D_CUDA_ -I../ -Xptxas=-v -Iobjs_gpu/
ifdef PTXCC_REGMAX
  NVCC_FLAGS += --maxrregcount=$(PTXCC_REGMAX)
endif
NVCC_FLAGS+=--use_fast_math
#
LD=nvcc
LDFLAGS=-lcudart -lcudadevrt -arch=sm_35
#
PTXCC=ptxcc
PTXCC_FLAGS = -Xptxas=-v
ifdef PTXCC_REGMAX
  PTXCC_FLAGS += -maxrregcount=$(PTXCC_REGMAX)
endif

#
ISPC=ispc
ISPC_FLAGS=-O3 --math-lib=default --target=nvptx --opt=fast-math
#
#
#
ISPC_LLVM_OBJS=$(ISPC_SRC:%.ispc=objs_gpu/%_llvm_ispc.o)
ISPC_NVVM_OBJS=$(ISPC_SRC:%.ispc=objs_gpu/%_nvvm_ispc.o)
ISPC_BCS=$(ISPC_SRC:%.ispc=objs_gpu/%_ispc.bc)
ISPC_LLVM_PTX=$(ISPC_SRC:%.ispc=objs_gpu/%_llvm_ispc.ptx)
ISPC_NVVM_PTX=$(ISPC_SRC:%.ispc=objs_gpu/%_nvvm_ispc.ptx)
ISPC_HEADERS=$(ISPC_SRC:%.ispc=objs_gpu/%_ispc.h)
CXX_OBJS=$(CXX_SRC:%.cpp=objs_gpu/%_gcc.o)
CU_OBJS=$(CU_SRC:%.cu=objs_gpu/%_cu.o)
#NVCC_OBJS=$(NVCC_SRC:%.cu=objs_gpu/%_nvcc.o)

CXX_SRC+=ispc_malloc.cpp
CXX_OBJS+=objs_gpu/ispc_malloc_gcc.o

PTXGEN = $(HOME)/ptxgen
PTXGEN += -opt=3
PTXGEN += -ftz=1 -prec-div=0 -prec-sqrt=0 -fma=1

LLVM32=$(HOME)/usr/local/llvm/bin-3.2
LLVM32DIS=$(LLVM32)/bin/llvm-dis

LLC=$(HOME)/usr/local/llvm/bin-trunk/bin/llc
LLC_FLAGS=-march=nvptx64 -mcpu=sm_35

# .SUFFIXES: .bc .o .cu 

ifdef LLVM_GPU
  OBJSgpu_llvm=$(ISPC_LLVM_OBJS) $(CXX_OBJS) $(NVCC_OBJS) 
  PROGgpu_llvm=$(PROG)_llvm_gpu
else
  ISPC_LLVM_PTX=
endif


ifdef NVVM_GPU
  OBJSgpu_nvvm=$(ISPC_NVVM_OBJS) $(CXX_OBJS) $(NVCC_OBJS) $(ISPC_LVVM_PTX)
  PROGgpu_nvvm=$(PROG)_nvvm_gpu
else
  ISPC_NVVM_PTX=
endif

ifdef CU_SRC
  OBJScu=$(CU_OBJS) $(CXX_OBJS) $(NVCC_OBJS)
  PROGcu=$(PROG)_cu
endif


all: dirs  \
	$(PROGgpu_nvvm)  \
	$(PROGgpu_llvm)  \
	$(PROGcu) $(ISPC_BC)  $(ISPC_HEADERS) $(ISPC_NVVM_PTX) $(ISPC_LLVM_PTX)

dirs:
	/bin/mkdir -p objs_gpu/

objs_gpu/%.cpp objs_gpu/%.o objs_gpu/%.h: dirs

clean: 
	/bin/rm -rf $(PROGgpu_nvvm) $(PROGgpu_llvm) $(PROGcu) objs_gpu

# generate binaries
$(PROGgpu_llvm): $(OBJSgpu_llvm)
	$(LD) -o $@ $^ $(LDFLAGS)
$(PROGgpu_nvvm): $(OBJSgpu_nvvm)
	$(LD) -o $@ $^ $(LDFLAGS)
$(PROGcu): $(OBJScu)
	$(LD) -o $@ $^ $(LDFLAGS)

# compile C++ code
objs_gpu/%_gcc.o: %.cpp $(ISPC_HEADERS)
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs_gpu/%_gcc.o: ../%.cpp 
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

# CUDA helpers
objs_gpu/%_cu.o: %.cu $(ISPC_HEADERS)
	$(NVCC) $(NVCC_FLAGS)  -o $@ -dc $<

# compile CUDA code 
objs_gpu/%_nvcc.o: ../%.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<
objs_gpu/%_nvcc.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

# compile ISPC to LLVM BC
objs_gpu/%_ispc.h objs_gpu/%_ispc.bc: %.ispc 
	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs_gpu/$*_ispc.h -o objs_gpu/$*_ispc.bc $<

# generate PTX from LLVM BC
objs_gpu/%_llvm_ispc.ptx: objs_gpu/%_ispc.bc
	$(LLC) $(LLC_FLAGS) -o $@ $<
objs_gpu/%_nvvm_ispc.ptx: objs_gpu/%_ispc.bc
	$(LLVM32DIS) $< -o objs_gpu/$*_ispc-ll32.ll
	$(PTXGEN) objs_gpu/$*_ispc-ll32.ll > $@

# generate an object file from PTX
objs_gpu/%_ispc.o: objs_gpu/%_ispc.ptx
	$(PTXCC) $< $(PTXCC_FLAGS) -o $@


	 


