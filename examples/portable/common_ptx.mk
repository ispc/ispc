NVCC_SRC=../../util/nvcc_helpers.cu
NVCC_OBJS=objs_ptx/nvcc_helpers_nvcc.o
NVARCH ?= sm_35
#
CXX=g++ -ffast-math
CXXFLAGS=-O3 -I$(CUDATK)/include -Iobjs_ptx/ -D_CUDA_ -I../../util -I../../
#
NVCC=nvcc
NVCC_FLAGS+=-O3 -arch=$(NVARCH) -D_CUDA_ -I../../util -Xptxas=-v -Iobjs_ptx/
ifdef PTXCC_REGMAX
  NVCC_FLAGS += --maxrregcount=$(PTXCC_REGMAX)
endif
NVCC_FLAGS+=--use_fast_math
#
LD=nvcc
LDFLAGS=-lcudart -lcudadevrt -arch=$(NVARCH)
#
PTXCC=$(ISPC_HOME)/ptxtools/ptxcc --arch=$(NVARCH)
PTXCC_FLAGS+= -Xptxas=-v 
ifdef PTXCC_REGMAX
  PTXCC_FLAGS += -maxrregcount=$(PTXCC_REGMAX)
endif

#
ISPC=$(ISPC_HOME)/ispc
ISPC_FLAGS+=-O3 --math-lib=fast --target=nvptx --opt=fast-math
#
#
#
ISPC_LLVM_OBJS=$(ISPC_SRC:%.ispc=objs_ptx/%_llvm_ispc.o)
ISPC_NVVM_OBJS=$(ISPC_SRC:%.ispc=objs_ptx/%_nvvm_ispc.o)
#ISPC_BCS=$(ISPC_SRC:%.ispc=objs_ptx/%_ispc.bc)
ISPC_LLS=$(ISPC_SRC:%.ispc=objs_ptx/%_ispc.ll)
ISPC_LLVM_PTX=$(ISPC_SRC:%.ispc=objs_ptx/%_llvm_ispc.ptx)
ISPC_NVVM_PTX=$(ISPC_SRC:%.ispc=objs_ptx/%_nvvm_ispc.ptx)
ISPC_HEADERS=$(ISPC_SRC:%.ispc=objs_ptx/%_ispc.h)
CXX_OBJS=$(CXX_SRC:%.cpp=objs_ptx/%_gcc.o)
CU_OBJS=$(CU_SRC:%.cu=objs_ptx/%_cu.o)
#NVCC_OBJS=$(NVCC_SRC:%.cu=objs_ptx/%_nvcc.o)

CXX_SRC+=ispc_malloc.cpp
CXX_OBJS+=objs_ptx/ispc_malloc_gcc.o

PTXGEN = $(ISPC_HOME)/ptxtools/ptxgen
PTXGEN += --use_fast_math --arch=$(NVARCH)

#LLVM32=$(HOME)/usr/local/llvm/bin-3.2
#LLVM32DIS=$(LLVM32)/bin/llvm-dis

LLC=$(LLVM_ROOT)/bin/llc
LLC_FLAGS=-march=nvptx64 -mcpu=$(NVARCH)

# .SUFFIXES: .bc .o .cu  .ll

ifdef LLVM_GPU
  OBJSptx_llvm=$(ISPC_LLVM_OBJS) $(CXX_OBJS) $(NVCC_OBJS) 
  PROGptx_llvm=$(PROG)_llvm_ptx
else
  ISPC_LLVM_PTX=
endif


ifdef NVVM_GPU
  OBJSptx_nvvm=$(ISPC_NVVM_OBJS) $(CXX_OBJS) $(NVCC_OBJS) $(ISPC_LVVM_PTX)
  PROGptx_nvvm=$(PROG)_nvvm_ptx
else
  ISPC_NVVM_PTX=
endif

ifdef CU_SRC
  OBJScu=$(CU_OBJS) $(CXX_OBJS) $(NVCC_OBJS)
  PROGcu=$(PROG)_cu
endif


all: dirs  \
	$(PROGptx_nvvm)  \
	$(PROGptx_llvm)  \
	$(PROGcu) $(ISPC_BCS) $(ISPC_LLS)  $(ISPC_HEADERS) $(ISPC_NVVM_PTX) $(ISPC_LLVM_PTX)

dirs:
	/bin/mkdir -p objs_ptx/

objs_ptx/%.cpp objs_ptx/%.o objs_ptx/%.h: dirs

clean: 
	/bin/rm -rf $(PROGptx_nvvm) $(PROGptx_llvm) $(PROGcu) objs_ptx

# generate binaries
$(PROGptx_llvm): $(OBJSptx_llvm)
	$(LD) -o $@ $^ $(LDFLAGS)
$(PROGptx_nvvm): $(OBJSptx_nvvm)
	$(LD) -o $@ $^ $(LDFLAGS)
$(PROGcu): $(OBJScu)
	$(LD) -o $@ $^ $(LDFLAGS)

# compile C++ code
objs_ptx/%_gcc.o: %.cpp $(ISPC_HEADERS)
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs_ptx/%_gcc.o: ../../util/%.cpp 
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

# CUDA helpers
objs_ptx/%_cu.o: %.cu $(ISPC_HEADERS)
	$(NVCC) $(NVCC_FLAGS)  -o $@ -dc $<

# compile CUDA code 
objs_ptx/%_nvcc.o: ../../util/%.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<
objs_ptx/%_nvcc.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

# compile ISPC to LLVM BC
#objs_ptx/%_ispc.h objs_ptx/%_ispc.bc: %.ispc 
#	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs_ptx/$*_ispc.h -o objs_ptx/$*_ispc.bc $<
objs_ptx/%_ispc.h objs_ptx/%_ispc.ll: %.ispc 
	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs_ptx/$*_ispc.h -o objs_ptx/$*_ispc.ll $<

# generate PTX from LLVM BC
#objs_ptx/%_llvm_ispc.ptx: objs_ptx/%_ispc.bc
#	$(LLC) $(LLC_FLAGS) -o $@ $<
objs_ptx/%_llvm_ispc.ptx: objs_ptx/%_ispc.ll
	$(LLC) $(LLC_FLAGS) -o $@ $<
#objs_ptx/%_nvvm_ispc.ptx: objs_ptx/%_ispc.bc
#	$(LLVM32DIS) $< -o objs_ptx/$*_ispc-ll32.ll
#	$(PTXGEN) objs_ptx/$*_ispc-ll32.ll -o $@
objs_ptx/%_nvvm_ispc.ptx: objs_ptx/%_ispc.ll
	$(PTXGEN) $< -o $@

# generate an object file from PTX
objs_ptx/%_ispc.o: objs_ptx/%_ispc.ptx
	$(PTXCC) $< -Xnvcc="$(PTXCC_FLAGS)" -o $@


	 


