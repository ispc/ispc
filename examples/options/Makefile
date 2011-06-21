
CXX=g++
CXXFLAGS=-Iobjs/ -g -Wall
ISPC=ispc
ISPCFLAGS=-O2 --target=sse4x2

default: options

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ options

options: dirs objs/options.o objs/options_serial.o objs/options_ispc.o
	$(CXX) $(CXXFLAGS) -o $@ objs/options.o objs/options_ispc.o objs/options_serial.o -lm

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/options.o: objs/options_ispc.h options_defs.h

objs/%_ispc.h objs/%_ispc.o: %.ispc options_defs.h
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
