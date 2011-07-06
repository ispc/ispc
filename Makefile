#
# ispc Makefile
#

ARCH_OS = $(shell uname)
ARCH_TYPE = $(shell arch)

CLANG=clang
CLANG_LIBS = -lclangFrontend -lclangDriver \
             -lclangSerialization -lclangParse -lclangSema \
             -lclangAnalysis -lclangAST -lclangLex -lclangBasic

LLVM_LIBS=$(shell llvm-config --ldflags --libs) -lpthread -ldl
LLVM_CXXFLAGS=$(shell llvm-config --cppflags)
LLVM_VERSION_DEF=-DLLVM_$(shell llvm-config --version | sed s/\\./_/)

BUILD_DATE=$(shell date +%Y%m%d)
BUILD_VERSION=$(shell git log | head -1)

CXX=g++
CPP=cpp
CXXFLAGS=-g3 $(LLVM_CXXFLAGS) -I. -Iobjs/ -Wall $(LLVM_VERSION_DEF) \
	-DBUILD_DATE="\"$(BUILD_DATE)\"" -DBUILD_VERSION="\"$(BUILD_VERSION)\""

LDFLAGS=
ifeq ($(ARCH_OS),Linux)
  # try to link everything statically under Linux (including libstdc++) so
  # that the binaries we generate will be portable across distributions...
  ifeq ($(ARCH_TYPE),x86_64)
    LDFLAGS=-static -L/usr/lib/gcc/x86_64-linux-gnu/4.4
  else
    LDFLAGS=-L/usr/lib/gcc/i686-redhat-linux/4.6.0
  endif
endif

LEX=flex
YACC=bison -d -v -t

###########################################################################

CXX_SRC=builtins.cpp ctx.cpp decl.cpp expr.cpp ispc.cpp \
	llvmutil.cpp main.cpp module.cpp opt.cpp stmt.cpp sym.cpp type.cpp \
	util.cpp
HEADERS=builtins.h ctx.h decl.h expr.h ispc.h llvmutil.h module.h \
	opt.h stmt.h sym.h type.h util.h
STDLIB_SRC=stdlib-avx.ll stdlib-sse2.ll stdlib-sse4.ll stdlib-sse4x2.ll
BISON_SRC=parse.yy
FLEX_SRC=lex.ll

OBJS=$(addprefix objs/, $(CXX_SRC:.cpp=.o) $(STDLIB_SRC:.ll=.o) stdlib-c.o stdlib_ispc.o \
	$(BISON_SRC:.yy=.o) $(FLEX_SRC:.ll=.o))

default: ispc ispc_test

.PHONY: dirs clean depend doxygen print_llvm_src
.PRECIOUS: objs/stdlib-%.cpp

depend: $(CXX_SRC) $(HEADERS)
	@echo Updating dependencies
	@gcc -MM $(CXXFLAGS) $(CXX_SRC) | sed 's_^\([a-z]\)_objs/\1_g' > depend

-include depend

dirs:
	@echo Creating objs/ directory
	@/bin/mkdir -p objs

print_llvm_src:
	@echo Using LLVM `llvm-config --version` from `llvm-config --libdir`

clean:
	/bin/rm -rf objs ispc ispc_test

doxygen:
	/bin/rm -rf docs/doxygen
	doxygen doxygen.cfg

ispc: print_llvm_src dirs $(OBJS)
	@echo Creating ispc executable
	@$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(CLANG_LIBS) $(LLVM_LIBS)

ispc_test: dirs ispc_test.cpp
	@echo Creating ispc_test executable
	@$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ ispc_test.cpp $(LLVM_LIBS)

objs/%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/parse.cc: parse.yy
	@echo Running bison on $<
	@$(YACC) -o $@ $<

objs/parse.o: objs/parse.cc $(HEADERS)
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/lex.cpp: lex.ll 
	@echo Running flex on $<
	@$(LEX) -o $@ $<

objs/lex.o: objs/lex.cpp $(HEADERS) objs/parse.cc
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/stdlib-%.cpp: stdlib-%.ll stdlib.m4 stdlib-sse.ll
	@echo Creating C++ source from stdlib file $<
	@m4 stdlib.m4 $< | ./bitcode2cpp.py $< > $@

objs/stdlib-%.o: objs/stdlib-%.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/stdlib-c.cpp: stdlib-c.c
	@echo Creating C++ source from stdlib file $<
	@$(CLANG) -I /opt/l1om/usr/include/ -emit-llvm -c $< -o - | llvm-dis - | ./bitcode2cpp.py $< > $@

objs/stdlib-c.o: objs/stdlib-c.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/stdlib_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $<
	@$(CLANG) -E -x c -DISPC=1 -DPI=3.1415926536 $< -o - | ./stdlib2cpp.py > $@

objs/stdlib_ispc.o: objs/stdlib_ispc.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<
