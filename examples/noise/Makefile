
CXX=g++ -m64
CXXFLAGS=-Iobjs/ -O3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --target=sse2,sse4,avx-x2 --arch=x86-64

OBJS=objs/noise.o objs/noise_serial.o objs/noise_ispc.o objs/noise_ispc_sse2.o \
	objs/noise_ispc_sse4.o objs/noise_ispc_avx.o 

default: noise

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ noise

noise: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/noise.o: objs/noise_ispc.h 

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
