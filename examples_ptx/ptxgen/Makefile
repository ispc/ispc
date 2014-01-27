CUDA55 = /usr/local/cuda-5.5
CUDA60 = /usr/local/cuda-6.0
CUDA55 = /opt/cuda/5.5
CUDA60 = /opt/cuda/5.5
all: ptxgen55 ptxgen60

ptxgen60: ptxgen.cpp
	g++ -O3 -o ptxgen60 ptxgen.cpp  -L$(CUDA60)/nvvm/lib64 -lnvvm -I$(CUDA60)/nvvm/include -I$(CUDA60)/include   -DLIBDEVICE_MAJOR_VERSION=1 -DLIBDEVICE_MINOR_VERSION=0 -DLIBNVVM_HOME=$(CUDA60)/nvvm -Wl,-rpath,$(CUDA60)/nvvm/lib64

 
ptxgen55: ptxgen.cpp
	g++ -O3 -o ptxgen55 ptxgen.cpp  -L$(CUDA55)/nvvm/lib64 -lnvvm -I$(CUDA55)/nvvm/include -I$(CUDA55)/include   -DLIBDEVICE_MAJOR_VERSION=1 -DLIBDEVICE_MINOR_VERSION=0 -DLIBNVVM_HOME=$(CUDA55)/nvvm -Wl,-rpath,$(CUDA55)/nvvm/lib64


clean:
	/bin/rm -f ptxgen55 ptxgen60
