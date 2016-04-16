CXX=g++
CXXFLAGS=`llvm-config --cxxflags` -I$(ISPC_HOME) -DISPC_LIBISPC_ENABLED
CLANG_LIBS=-lclangFrontend -lclangDriver \
           -lclangSerialization -lclangParse -lclangSema \
           -lclangAnalysis -lclangAST -lclangBasic \
           -lclangEdit -lclangLex
LDFLAGS=-L$(ISPC_HOME) -lispc `llvm-config --ldflags` $(CLANG_LIBS) `llvm-config --libs --system-libs`

.PHONY: clean

default: simple_jit

simple_jit: simple_jit.cpp
	g++ -o$@ $(CXXFLAGS) $< $(LDFLAGS)

clean:
	rm -f simple_jit
