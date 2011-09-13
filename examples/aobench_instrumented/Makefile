
CXX=g++ -m64
CXXFLAGS=-Iobjs/ -g3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --instrument --arch=x86-64

default: ao

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ ao

ao: dirs objs/ao.o objs/instrument.o objs/ao_ispc.o
	$(CXX) $(CXXFLAGS) -o $@ objs/ao.o objs/ao_ispc.o objs/instrument.o -lm -lpthread

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/ao.o: objs/ao_ispc.h 

objs/%_ispc.h objs/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
