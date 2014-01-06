#pragma once

#include <cstring>
#include <cassert>

namespace parser
{
  class PTXLexer;
  class PTXParser;
}

#include "ptxgrammar.hh"

namespace parser
{
	/*!	\brief A wrapper around yyFlexLexer to allow for a local variable */
	class PTXLexer : public ptxFlexLexer
	{
		public:
			YYSTYPE*     yylval;
			int          column;
			int          nextColumn;

		public:
			PTXLexer( std::istream* arg_yyin, 
				std::ostream* arg_yyout ) :
        yyFlexLexer( arg_yyin, arg_yyout ), yylval( 0 ), column( 0 ), 
        nextColumn( 0 ) { }
	
			int yylex();
      int yylexPosition()
      {
        int token = yylex();
        column = nextColumn;
        nextColumn = column + strlen( YYText() );
        return token;
      }

  };
}
