
CXX=g++ -m64
CXXFLAGS=-Iobjs/ -g -Wall
ISPC=ispc
ISPCFLAGS=-O2 --target=sse2,sse4-x2,avx-x2 --arch=x86-64

OBJS=objs/options.o objs/options_serial.o objs/options_ispc.o \
	objs/options_ispc_sse2.o objs/options_ispc_sse4.o \
	objs/options_ispc_avx.o

default: options

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ options

options: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/options.o: objs/options_ispc.h options_defs.h

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
