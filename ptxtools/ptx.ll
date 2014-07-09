/*
   Copyright (c) 2014, Evghenii Gaburov
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

%option yylineno
%option noyywrap
%option yyclass="parser::PTXLexer"
%option prefix="ptx"
%option c++

%{
#include "PTXLexer.h"
#include <cassert>
#include <sstream>
#include <cstring>
#ifdef LLSETTOKEN
#error "TOKEN is defined"
#endif
#define LLSETTOKEN(tok) yylval->ivalue = tok; return tok;
%}

COMMENT ("//"[^\n]*)
TAB [\t]*

%%
{COMMENT}       {nextColumn += strlen(yytext); /* lCppComment(&yylloc); */ }
".version"      { return TOKEN_VERSION; }
".target"       { return TOKEN_TARGET; }
".address_size" { return TOKEN_ADDRESS_SIZE; }
".func"         { return TOKEN_FUNC; }
".entry"        { return TOKEN_ENTRY; }
".align"        { return TOKEN_ALIGN; }
".visible"      { return TOKEN_VISIBLE; }
".global"       { return TOKEN_GLOBAL; }
".param"        { return TOKEN_PARAM; }
".b0"           { LLSETTOKEN( TOKEN_B32);}   /* fix for buggy llvm-ptx generator */
".b8"           { LLSETTOKEN( TOKEN_B8);}
".b16"          { LLSETTOKEN( TOKEN_B16);}
".b32"          { LLSETTOKEN( TOKEN_B32);}
".b64"          { LLSETTOKEN( TOKEN_B64);}
".u8"           { LLSETTOKEN( TOKEN_U8);}
".u16"          { LLSETTOKEN( TOKEN_U16);}
".u32"          { LLSETTOKEN( TOKEN_U32);}
".u64"          { LLSETTOKEN( TOKEN_U64);}
".s8"           { LLSETTOKEN( TOKEN_S8);}
".s16"          { LLSETTOKEN( TOKEN_S16);}
".s32"          { LLSETTOKEN( TOKEN_S32);}
".s64"          { LLSETTOKEN( TOKEN_S64);}
".f32"          { LLSETTOKEN( TOKEN_F32);}
".f64"          { LLSETTOKEN( TOKEN_F64);}
"["             { return '[';}
"]"             { return ']';}
"("             { return '(';}
")"             { return ')';}
","             { return ',';}
";"             { return ';';}
"="             { return '=';}
[0-9]+\.[0-9]+ { yylval->fvalue = atof(yytext); return TOKEN_FLOAT; }
[0-9]+   { yylval->ivalue = atoi(yytext); return TOKEN_INT; }
[a-zA-Z0-9_]+   { strcpy(yylval->svalue, yytext); return TOKEN_STRING;}
\n {
 //   yylloc.last_line++;
//    yylloc.last_column = 1;
    nextColumn = 1;
}
.              ;
%%

/** Handle a C++-style comment--eat everything up until the end of the line.
 */
#if 0
static void
lCppComment(SourcePos *pos) {
    char c;
    do {
        c = yyinput();
    } while (c != 0 && c != '\n');
    if (c == '\n') {
        pos->last_line++;
        pos->last_column = 1;
    }
}
#endif
