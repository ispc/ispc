#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include "PTXParser.h"

int main(int argc, char * argv[])
{
	// open a file handle to a particular file:
  std::istream & input = std::cin;
  std::ostream & error = std::cerr;
  std::ostream & output = std::cout;
  parser::PTXLexer lexer(&input, &error);
  parser::PTXParser state(output);

	// parse through the input until there is no more:
  //

  do {
    ptx::yyparse(lexer, state);
  }
  while (!input.eof());
	
}
