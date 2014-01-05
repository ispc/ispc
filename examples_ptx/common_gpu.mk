NVCC_SRC=../nvcc_helpers.cu
NVCC_OBJS=objs_gpu/nvcc_helpers_nvcc.o
#
CXX=g++ -ffast-math
CXXFLAGS=-O3 -I$(CUDATK)/include -Iobjs_gpu/ -D_CUDA_
#
NVCC=nvcc
NVCC_FLAGS=-O3 -arch=sm_35 -D_CUDA_ -I../ -Xptxas=-v
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
ISPC_FLAGS=-O3 --math-lib=default --target=nvptx64 --opt=fast-math
#
#
#
ISPC_OBJS=$(ISPC_SRC:%.ispc=objs_gpu/%_ispc.o)
ISPC_BCS=$(ISPC_SRC:%.ispc=objs_gpu/%_ispc.bc)
ISPC_HEADERS=$(ISPC_SRC:%.ispc=objs_gpu/%_ispc.h)
CXX_OBJS=$(CXX_SRC:%.cpp=objs_gpu/%_gcc.o)
CU_OBJS=$(CU_SRC:%.cu=objs_gpu/%_cu.o)
#NVCC_OBJS=$(NVCC_SRC:%.cu=objs_gpu/%_nvcc.o)

CXX_SRC+=ispc_malloc.cpp
CXX_OBJS+=objs_gpu/ispc_malloc_gcc.o

# PTXGEN = $(HOME)/ptxgen
# PTXGEN += -opt=3
# PTXGEN += -ftz=1 -prec-div=0 -prec-sqrt=0 -fma=1

# .SUFFIXES: .bc .o .cu 

OBJSgpu=$(ISPC_OBJS) $(CXX_OBJS) $(NVCC_OBJS)
PROGgpu = $(PROG)_gpu

ifdef CU_SRC
  OBJScu=$(CU_OBJS) $(CXX_OBJS) $(NVCC_OBJS)
  PROGcu=$(PROG)_cu
endif


all: dirs $(PROGgpu) $(PROGcu) $(ISPC_BCS) 

dirs:
	/bin/mkdir -p objs_gpu/

objs_gpu/%.cpp objs_gpu/%.o objs_gpu/%.h: dirs

clean: 
	/bin/rm -rf $(PROGcu) $(PROGgpu) objs_gpu

$(PROGgpu): $(OBJSgpu)
	$(LD) -o $@ $^ $(LDFLAGS)
$(PROGcu): $(OBJScu)
	$(LD) -o $@ $^ $(LDFLAGS)

objs_gpu/%_gcc.o: %.cpp $(ISPC_HEADERS)
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs_gpu/%_gcc.o: ../%.cpp 
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

objs_gpu/%_cu.o: %.cu $(ISPC_HEADERS)
	$(NVCC) $(NVCC_FLAGS)  -o $@ -dc $<

objs_gpu/%_nvcc.o: ../%.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<
objs_gpu/%_nvcc.o: %.cu 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

objs_gpu/%_ispc.h objs_gpu/%_ispc.bc: %.ispc 
	$(ISPC) $(ISPC_FLAGS) --emit-llvm -h objs_gpu/$*_ispc.h -o objs_gpu/$*_ispc.bc $<

objs_gpu/%_ispc.o: objs_gpu/%_ispc.bc 
	$(PTXCC) $< $(PTXCC_FLAGS) -o $@


	 


