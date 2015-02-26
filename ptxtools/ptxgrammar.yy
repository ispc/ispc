/*
   Copyright (c) 2014-2015, Evghenii Gaburov
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 * Neither the name of Intel Corporation nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.


 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
   Based on GPU Ocelot PTX parser : https://code.google.com/p/gpuocelot/ 
   */
%locations

%{
	#include <iostream>
	#include "PTXParser.h"
	#include "PTXLexer.h"
	#include <cassert>
	#include <cstring>
  #include <sstream>
  #include <cstdio>
  #include <algorithm>
  #include "GPUTargets.h"

	#define YYERROR_VERBOSE 1

	#ifdef REPORT_BASE
	#undef REPORT_BASE
	#endif

	#define REPORT_BASE 0

	namespace ptx
	{
	
    int yylex( YYSTYPE* token, YYLTYPE* location, parser::PTXLexer& lexer, 
      parser::PTXParser& state );
    void yyerror( YYLTYPE* location, parser::PTXLexer& lexer, 
      parser::PTXParser& state, char const* message );
    
    std::string yyTypeToString( int );
	
%}

%union
{
	char         svalue[1024];
	double       fvalue;
	int          ivalue;
	unsigned int uvalue;
}

%parse-param {parser::PTXLexer& lexer}
%parse-param {parser::PTXParser& state}
%lex-param   {parser::PTXLexer& lexer}
%lex-param   {parser::PTXParser& state}
%pure-parser

// define the constant-string tokens:
%token TOKEN_VERSION TOKEN_TARGET TOKEN_ADDRESS_SIZE
%token TOKEN_VISIBLE TOKEN_FUNC TOKEN_ENTRY
%token TOKEN_PARAM TOKEN_ALIGN 
%token TOKEN_GLOBAL
%token<ivalue> TOKEN_B8 TOKEN_B16 TOKEN_B32 TOKEN_B64
%token<ivalue> TOKEN_U8 TOKEN_U16 TOKEN_U32 TOKEN_U64
%token<ivalue> TOKEN_S8 TOKEN_S16 TOKEN_S32 TOKEN_S64
%token<ivalue> TOKEN_F32 TOKEN_F64

// define the "terminal symbol" token types I'm going to use (in CAPS
// by convention), and associate each with a field of the union:
%token <ivalue> TOKEN_INT
%token <fvalue> TOKEN_FLOAT
%token <svalue> TOKEN_STRING

%type<svalue> identifier
%type<ivalue> arrayDimensionSet
%type<ivalue> alignment

%start ptxsource

%%
// the first rule defined is the highest-level rule, which in our
// case is just the concept of a whole "snazzle file":
ptxsource:
  header ptxbody;

header:
  version target  address_size 
{ 
//  std::cerr << "Done reading PTX \n" << std::endl; 
   state.printHeader(); 
};

version:
  TOKEN_VERSION TOKEN_FLOAT  { assert($2 >= 3.0); } ;//std::cerr << "Reading PTX version " << $2  << std::endl; };
target:
  TOKEN_TARGET TOKEN_STRING  { 
    if (std::find(GPUTargets::computeMode.begin(), GPUTargets::computeMode.end(), std::string($2)) == GPUTargets::computeMode.end())
    {
      fprintf(stderr, "ptxcc fatal : Found wrong Target=\"%s\" in ptx file\n", $2);
      exit(-1);
    }
  }
address_size:
  TOKEN_ADDRESS_SIZE TOKEN_INT  { assert($2 == 64); } //std::cerr << "Address_Size " << $2  << std::endl; };


dataTypeId : 
    TOKEN_U8 | TOKEN_U16 | TOKEN_U32 | TOKEN_U64 
  | TOKEN_S8 | TOKEN_S16 | TOKEN_S32 | TOKEN_S64 
  | TOKEN_B8 | TOKEN_B16 | TOKEN_B32 | TOKEN_B64 
  | TOKEN_F32 | TOKEN_F64;

dataType: dataTypeId { state.dataTypeId($<ivalue>1); }

anytoken: 
  TOKEN_ALIGN 
| TOKEN_PARAM 
| dataTypeId
| TOKEN_STRING | TOKEN_FLOAT | TOKEN_INT
| TOKEN_FUNC | TOKEN_ENTRY
| TOKEN_GLOBAL
| '['
| ']'
| '('
| ')'
| ','
| ';'
| '='
;

ptxbody: 
    ptxbody visibleFunctionDeclaration | visibleFunctionDeclaration
  | ptxbody visibleEntryDeclaration| visibleEntryDeclaration
  | ptxbody visibleInitializableDeclaration| visibleInitializableDeclaration
  | ptxbody anytoken | anytoken;



arrayDimensionSet : '[' TOKEN_INT ']' { $$ = $2; state.arrayDimensions($<ivalue>2); }
// arrayDimensionSet : arrayDimensionSet '[' TOKEN_INT ']' { $$ = $2; }
// arrayDimensionSet : '[' ']' { $$ = 0; }
arrayDimensions : /* empty string */;
arrayDimensions : arrayDimensionSet;

identifier: TOKEN_STRING { strcpy($$, $1); state.identifier($1); }
parameter : TOKEN_PARAM;

alignment : TOKEN_ALIGN TOKEN_INT {$$ = $2; state.alignment($<ivalue>2);}
addressableVariablePrefix : dataType { state.alignment(0); }
addressableVariablePrefix : alignment dataType;

argumentDeclaration : parameter addressableVariablePrefix identifier arrayDimensions
{
  state.argumentDeclaration(@1);
}


argumentListBegin : '(' { state.argumentListBegin(@1); };
argumentListEnd : ')' {state.argumentListEnd(@1); };
argumentListBody : argumentDeclaration;
argumentListBody : /* empty string */;
argumentListBody : argumentListBody ',' argumentDeclaration;
argumentList: argumentListBegin argumentListBody argumentListEnd;

visibleEntryDeclaration: TOKEN_VISIBLE TOKEN_ENTRY identifier argumentList
{
   state.visibleEntryDeclaration($<svalue>3, @1);
};

returnArgumentListBegin : '(' { state.returnArgumentListBegin(@1); }
returnArgumentListEnd : ')' {state.returnArgumentListEnd(@1); }
returnArgumentList : returnArgumentListBegin argumentListBody returnArgumentListEnd;
optionalReturnArgumentList : returnArgumentList | /* empty string */;
visibleFunctionDeclaration: TOKEN_VISIBLE TOKEN_FUNC optionalReturnArgumentList identifier argumentList
{
   state.visibleFunctionDeclaration($<svalue>4, @1);
};

visibleInitializableDeclaration :
  TOKEN_VISIBLE TOKEN_GLOBAL addressableVariablePrefix identifier arrayDimensionSet
  { state.visibleInitializableDeclaration($<svalue>4,@1); }
| TOKEN_VISIBLE TOKEN_GLOBAL addressableVariablePrefix identifier ';'
  {state.arrayDimensions(0); state.visibleInitializableDeclaration($<svalue>4,@1); }
| TOKEN_VISIBLE TOKEN_GLOBAL addressableVariablePrefix identifier '='
  {state.arrayDimensions(0); state.visibleInitializableDeclaration($<svalue>4,@1); }


%%

int yylex( YYSTYPE* token, YYLTYPE* location, parser::PTXLexer& lexer, 
	parser::PTXParser& state )
{
	lexer.yylval = token;
	
	int tokenValue         = lexer.yylexPosition();
	location->first_line   = lexer.lineno();
	location->first_column = lexer.column;

#if 0
	report( " Lexer (" << location->first_line << ","
		<< location->first_column 
		<< "): " << parser::PTXLexer::toString( tokenValue ) << " \"" 
		<< lexer.YYText() << "\"");
#endif
	
	return tokenValue;
}
	
static std::string toString( YYLTYPE& location, parser::PTXParser& state )
{
  std::stringstream stream;
  stream 
#if 0
  << state.fileName 
#else
  << "ptx "
#endif
  << " (" << location.first_line << ", " 
    << location.first_column << "): ";
  return stream.str();
}

void yyerror( YYLTYPE* location, parser::PTXLexer& lexer, 
	parser::PTXParser& state, char const* message )
{
	std::stringstream stream;
	stream << toString( *location, state ) 
		<< " " << message;
  fprintf(stderr, "--Parser ERROR-- %s %s \n", toString(*location, state).c_str(), message);
  exit(-1);
}

}
