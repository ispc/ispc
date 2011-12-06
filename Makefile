#
# ispc Makefile
#

ARCH_OS = $(shell uname)
ARCH_TYPE = $(shell arch)

ifeq ($(shell llvm-config --version), 3.1svn)
  LLVM_LIBS=-lLLVMAsmParser -lLLVMInstrumentation -lLLVMLinker			\
	-lLLVMArchive -lLLVMBitReader -lLLVMDebugInfo -lLLVMJIT -lLLVMipo	\
	-lLLVMBitWriter -lLLVMTableGen -lLLVMCBackendInfo			\
	-lLLVMX86Disassembler -lLLVMX86CodeGen -lLLVMSelectionDAG		\
	-lLLVMAsmPrinter -lLLVMX86AsmParser -lLLVMX86Desc -lLLVMX86Info		\
	-lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMMCDisassembler	-lLLVMMCParser	\
	-lLLVMCodeGen -lLLVMScalarOpts	-lLLVMInstCombine -lLLVMTransformUtils	\
	-lLLVMipa -lLLVMAnalysis -lLLVMMCJIT -lLLVMRuntimeDyld			\
	-lLLVMExecutionEngine -lLLVMTarget -lLLVMMC -lLLVMObject -lLLVMCore 	\
	-lLLVMSupport
else
  LLVM_LIBS=$(shell llvm-config --libs)
endif

CLANG=clang
CLANG_LIBS = -lclangFrontend -lclangDriver \
             -lclangSerialization -lclangParse -lclangSema \
             -lclangAnalysis -lclangAST -lclangLex -lclangBasic

ISPC_LIBS=$(shell llvm-config --ldflags) $(CLANG_LIBS) $(LLVM_LIBS) \
	-lpthread -ldl

LLVM_CXXFLAGS=$(shell llvm-config --cppflags)
LLVM_VERSION=LLVM_$(shell llvm-config --version | sed s/\\./_/)
LLVM_VERSION_DEF=-D$(LLVM_VERSION)

BUILD_DATE=$(shell date +%Y%m%d)
BUILD_VERSION=$(shell git log --abbrev-commit --abbrev=16 | head -1)

CXX=g++
CPP=cpp
OPT=-g3
CXXFLAGS=$(OPT) $(LLVM_CXXFLAGS) -I. -Iobjs/ -Wall $(LLVM_VERSION_DEF) \
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

CXX_SRC=ast.cpp builtins.cpp ctx.cpp decl.cpp expr.cpp func.cpp ispc.cpp \
	llvmutil.cpp main.cpp module.cpp opt.cpp stmt.cpp sym.cpp type.cpp \
	util.cpp
HEADERS=ast.h builtins.h ctx.h decl.h expr.h func.h ispc.h llvmutil.h module.h \
	opt.h stmt.h sym.h type.h util.h
BUILTINS_SRC=builtins-avx.ll builtins-avx-x2.ll builtins-sse2.ll builtins-sse2-x2.ll \
	builtins-sse4.ll builtins-sse4-x2.ll builtins-dispatch.ll
BISON_SRC=parse.yy
FLEX_SRC=lex.ll

OBJS=$(addprefix objs/, $(CXX_SRC:.cpp=.o) $(BUILTINS_SRC:.ll=.o) \
	builtins-c-32.o builtins-c-64.o stdlib_ispc.o $(BISON_SRC:.yy=.o) \
	$(FLEX_SRC:.ll=.o))

default: ispc

.PHONY: dirs clean depend doxygen print_llvm_src
.PRECIOUS: objs/builtins-%.cpp

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
	/bin/rm -rf objs ispc

doxygen:
	/bin/rm -rf docs/doxygen
	doxygen doxygen.cfg

ispc: print_llvm_src dirs $(OBJS)
	@echo Creating ispc executable
	@$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(ISPC_LIBS)

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

objs/builtins-%.cpp: builtins-%.ll
	@echo Creating C++ source from builtin definitions file $<
	@m4 -DLLVM_VERSION=$(LLVM_VERSION) builtins.m4 $< | ./bitcode2cpp.py $< > $@

objs/builtins-%.o: objs/builtins-%.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/builtins-c-32.cpp: builtins-c.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m32 -emit-llvm -c $< -o - | llvm-dis - | ./bitcode2cpp.py builtins-c-32.c > $@

objs/builtins-c-32.o: objs/builtins-c-32.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/builtins-c-64.cpp: builtins-c.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m64 -emit-llvm -c $< -o - | llvm-dis - | ./bitcode2cpp.py builtins-c-64.c > $@

objs/builtins-c-64.o: objs/builtins-c-64.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/stdlib_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $<
	@$(CLANG) -E -x c -DISPC=1 -DPI=3.1415926536 $< -o - | ./stdlib2cpp.py > $@

objs/stdlib_ispc.o: objs/stdlib_ispc.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/builtins-sse2.cpp: builtins.m4 builtins-sse2-common.ll builtins-sse2.ll
objs/builtins-sse2-x2.cpp: builtins.m4 builtins-sse2-common.ll builtins-sse2-x2.ll
objs/builtins-sse4.cpp: builtins.m4 builtins-sse4-common.ll builtins-sse4.ll
objs/builtins-sse4-x2.cpp: builtins.m4 builtins-sse4-common.ll builtins-sse4-x2.ll
objs/builtins-avx.cpp: builtins.m4 builtins-avx-common.ll builtins-avx.ll
objs/builtins-avx-x2.cpp: builtins.m4 builtins-avx-common.ll builtins-avx-x2.ll
