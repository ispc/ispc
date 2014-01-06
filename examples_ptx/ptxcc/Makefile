all: dePTX

ptxgrammar.cc : ptxgrammar.yy
	bison -d -v -t  ptxgrammar.yy -o ptxgrammar.cc

ptx.cc: ptx.ll ptxgrammar.cc
	flex -t ptx.ll > ptx.cc

%.o: %.cc
	clang++ -O3 -c $< -o $@ -I/opt/local/include

%.o: %.cpp
	clang++ -O3 -c $< -o $@ -I/opt/local/include

OBJ= dePTX.o \
		 ptx.o \
		 ptxgrammar.o

dePTX: $(OBJ)
	clang++ $^ -o $@ -L/opt/local/lib 

clean: 
	/bin/rm -f dePTX $(OBJ) ptxgrammar.hh ptxgrammar.cc ptx.cc ptxgrammar.output

$(OBJ):  ptxgrammar.cc ptx.cc PTXParser.h PTXLexer.h

