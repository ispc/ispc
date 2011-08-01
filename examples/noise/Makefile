
CXX=g++ -m64
CXXFLAGS=-Iobjs/ -O3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --target=sse4 --arch=x86-64

default: noise

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ noise

noise: dirs objs/noise.o objs/noise_serial.o objs/noise_ispc.o
	$(CXX) $(CXXFLAGS) -o $@ objs/noise.o objs/noise_ispc.o objs/noise_serial.o -lm

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/noise.o: objs/noise_ispc.h 

objs/%_ispc.h objs/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
