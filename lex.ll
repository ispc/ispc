/*
  Copyright (c) 2010-2011, Intel Corporation
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

%{

#include "ispc.h"
#include "decl.h"
#include "sym.h"
#include "util.h"
#include "module.h"
#include "type.h"
#include "parse.hh"
#include <stdlib.h>

static uint64_t lParseBinary(const char *ptr, SourcePos pos);
static void lCComment(SourcePos *);
static void lCppComment(SourcePos *);
static void lHandleCppHash(SourcePos *);
static void lStringConst(YYSTYPE *, SourcePos *);
static double lParseHexFloat(const char *ptr);

#define YY_USER_ACTION \
    yylloc->first_line = yylloc->last_line; \
    yylloc->first_column = yylloc->last_column; \
    yylloc->last_column += yyleng;

#ifdef ISPC_IS_WINDOWS
inline int isatty(int) { return 0; }
#endif // ISPC_IS_WINDOWS

%}

%option nounput
%option noyywrap
%option bison-bridge
%option bison-locations
%option nounistd

WHITESPACE [ \t\r]+
INT_NUMBER (([0-9]+)|(0x[0-9a-fA-F]+)|(0b[01]+))
FLOAT_NUMBER (([0-9]+|(([0-9]+\.[0-9]*[fF]?)|(\.[0-9]+)))([eE][-+]?[0-9]+)?[fF]?)
HEX_FLOAT_NUMBER (0x[01](\.[0-9a-fA-F]*)?p[-+]?[0-9]+[fF]?)

IDENT [a-zA-Z_][a-zA-Z_0-9]*

%%
"/*"            { lCComment(yylloc); }
"//"            { lCppComment(yylloc); }

bool { return TOKEN_BOOL; }
break { return TOKEN_BREAK; }
case { return TOKEN_CASE; }
cbreak { return TOKEN_CBREAK; }
ccontinue { return TOKEN_CCONTINUE; }
cdo { return TOKEN_CDO; }
cfor { return TOKEN_CFOR; }
char { return TOKEN_CHAR; }
cif { return TOKEN_CIF; }
cwhile { return TOKEN_CWHILE; }
const { return TOKEN_CONST; }
continue { return TOKEN_CONTINUE; }
creturn { return TOKEN_CRETURN; }
default { return TOKEN_DEFAULT; }
do { return TOKEN_DO; }
double { return TOKEN_DOUBLE; }
else { return TOKEN_ELSE; }
enum { return TOKEN_ENUM; }
export { return TOKEN_EXPORT; }
extern { return TOKEN_EXTERN; }
false { return TOKEN_FALSE; }
float { return TOKEN_FLOAT; }
for { return TOKEN_FOR; }
goto { return TOKEN_GOTO; }
if { return TOKEN_IF; }
inline { return TOKEN_INLINE; }
int { return TOKEN_INT; }
int32 { return TOKEN_INT; }
int64 { return TOKEN_INT64; }
launch { return TOKEN_LAUNCH; }
print { return TOKEN_PRINT; }
reference { return TOKEN_REFERENCE; }
return { return TOKEN_RETURN; }
soa { return TOKEN_SOA; }
static { return TOKEN_STATIC; }
struct { return TOKEN_STRUCT; }
switch { return TOKEN_SWITCH; }
sync { return TOKEN_SYNC; }
task { return TOKEN_TASK; }
true { return TOKEN_TRUE; }
typedef { return TOKEN_TYPEDEF; }
uniform { return TOKEN_UNIFORM; }
unsigned { return TOKEN_UNSIGNED; }
varying { return TOKEN_VARYING; }
void { return TOKEN_VOID; }
while { return TOKEN_WHILE; }

L?\"(\\.|[^\\"])*\" { lStringConst(yylval, yylloc); return TOKEN_STRING_LITERAL; }

{IDENT} { 
    /* We have an identifier--is it a type name or an identifier?
       The symbol table will straighten us out... */
    yylval->stringVal = new std::string(yytext);
    if (m->symbolTable->LookupType(yytext) != NULL)
        return TOKEN_TYPE_NAME;
    else
        return TOKEN_IDENTIFIER; 
}

{INT_NUMBER} { 
    char *endPtr = NULL;
    int64_t val;

    if (yytext[0] == '0' && yytext[1] == 'b')
        val = lParseBinary(yytext+2, *yylloc);
    else {
#ifdef ISPC_IS_WINDOWS
        val = _strtoi64(yytext, &endPtr, 0);
#else
        // FIXME: should use strtouq and then issue an error if we can't
        // fit into 64 bits...
        val = strtoull(yytext, &endPtr, 0);
#endif
    }

    // See if we can fit this into a 32-bit integer...
    if ((val & 0xffffffff) == val) {
        yylval->int32Val = (int32_t)val;
        return TOKEN_INT32_CONSTANT; 
    }
    else {
        yylval->int64Val = val;
        return TOKEN_INT64_CONSTANT; 
    }
}

{INT_NUMBER}[uU] {
    char *endPtr = NULL;
    uint64_t val;

    if (yytext[0] == '0' && yytext[1] == 'b')
        val = lParseBinary(yytext+2, *yylloc);
    else {
#ifdef ISPC_IS_WINDOWS
        val = _strtoui64(yytext, &endPtr, 0);
#else
        val = strtoull(yytext, &endPtr, 0);
#endif
    }

    if ((val & 0xffffffff) == val) {
        // we can represent it in a 32-bit value
        yylval->int32Val = (int32_t)val;
        return TOKEN_UINT32_CONSTANT; 
    }
    else {
        yylval->int64Val = val;
        return TOKEN_UINT64_CONSTANT; 
    }
}

{FLOAT_NUMBER} { 
    yylval->floatVal = atof(yytext); 
    return TOKEN_FLOAT_CONSTANT; 
}

{HEX_FLOAT_NUMBER} {
    yylval->floatVal = lParseHexFloat(yytext); 
    return TOKEN_FLOAT_CONSTANT; 
}

"++" { return TOKEN_INC_OP; }
"--" { return TOKEN_DEC_OP; }
"<<" { return TOKEN_LEFT_OP; }
">>" { return TOKEN_RIGHT_OP; }
"<=" { return TOKEN_LE_OP; }
">=" { return TOKEN_GE_OP; }
"==" { return TOKEN_EQ_OP; }
"!=" { return TOKEN_NE_OP; }
"&&" { return TOKEN_AND_OP; }
"||" { return TOKEN_OR_OP; }
"*=" { return TOKEN_MUL_ASSIGN; }
"/=" { return TOKEN_DIV_ASSIGN; }
"%=" { return TOKEN_MOD_ASSIGN; }
"+=" { return TOKEN_ADD_ASSIGN; }
"-=" { return TOKEN_SUB_ASSIGN; }
"<<=" { return TOKEN_LEFT_ASSIGN; }
">>=" { return TOKEN_RIGHT_ASSIGN; }
"&=" { return TOKEN_AND_ASSIGN; }
"^=" { return TOKEN_XOR_ASSIGN; }
"|=" { return TOKEN_OR_ASSIGN; }
";"             { return ';'; }
("{"|"<%")      { return '{'; }
("}"|"%>")      { return '}'; }
","             { return ','; }
":"             { return ':'; }
"="             { return '='; }
"("             { return '('; }
")"             { return ')'; }
("["|"<:")      { return '['; }
("]"|":>")      { return ']'; }
"."             { return '.'; }
"&"             { return '&'; }
"!"             { return '!'; }
"~"             { return '~'; }
"-"             { return '-'; }
"+"             { return '+'; }
"*"             { return '*'; }
"/"             { return '/'; }
"%"             { return '%'; }
"<"             { return '<'; }
">"             { return '>'; }
"^"             { return '^'; }
"|"             { return '|'; }
"?"             { return '?'; }

{WHITESPACE} { }

\n {
    yylloc->last_line++; 
    yylloc->last_column = 1; 
}

#(line)?[ ][0-9]+[ ]\"(\\.|[^\\"])*\"[^\n]* { 
    lHandleCppHash(yylloc); 
}

. {
    Error(*yylloc, "Illegal character: %c (0x%x)", yytext[0], int(yytext[0]));
    YY_USER_ACTION 
}

%%

/*sizeof { return TOKEN_SIZEOF; }*/
/*"->" { return TOKEN_PTR_OP; }*/
/*short { return TOKEN_SHORT; }*/
/*long { return TOKEN_LONG; }*/
/*signed { return TOKEN_SIGNED; }*/
/*volatile { return TOKEN_VOLATILE; }*/
/*"long"[ \t\v\f\n]+"long" { return TOKEN_LONGLONG; }*/
/*union { return TOKEN_UNION; }*/
/*"..." { return TOKEN_ELLIPSIS; }*/

/** Return the integer version of a binary constant from a string.
 */
static uint64_t
lParseBinary(const char *ptr, SourcePos pos) {
    uint64_t val = 0;
    bool warned = false;

    while (*ptr != '\0') {
        /* if this hits, the regexp for 0b... constants is broken */
        assert(*ptr == '0' || *ptr == '1');

        if ((val & (((int64_t)1)<<63)) && warned == false) {
            // We're about to shift out a set bit
            Warning(pos, "Can't represent binary constant with a 64-bit integer type");
            warned = true;
        }

        val = (val << 1) | (*ptr == '0' ? 0 : 1);
        ++ptr;
    }
    return val;
}


/** Handle a C-style comment in the source. 
 */
static void
lCComment(SourcePos *pos) {
    char c, prev = 0;
  
    while ((c = yyinput()) != 0) {
        if (c == '\n') {
            pos->last_line++;
            pos->last_column = 1;
        }
        if (c == '/' && prev == '*')
            return;
        prev = c;
    }
    Error(*pos, "unterminated comment");
}

/** Handle a C++-style comment--eat everything up until the end of the line.
 */
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

/** Handle a line that starts with a # character; this should be something
    left behind by the preprocessor indicating the source file/line
    that our current position corresponds to.
 */
static void lHandleCppHash(SourcePos *pos) {
    char *ptr, *src;

    // Advance past the opening stuff on the line.
    assert(yytext[0] == '#');
    if (yytext[1] == ' ')
        // On Linux/OSX, the preprocessor gives us lines like
        // # 1234 "foo.c"
        ptr = yytext + 2;
    else {
        // On windows, cl.exe's preprocessor gives us lines of the form:
        // #line 1234 "foo.c"
        assert(!strncmp(yytext+1, "line ", 5));
        ptr = yytext + 6;
    }

    // Now we can set the line number based on the integer in the string
    // that ptr is pointing at.
    pos->last_line = strtol(ptr, &src, 10) - 1;
    pos->last_column = 1;
    // Make sure that the character after the integer is a space and that
    // then we have open quotes
    assert(src != ptr && src[0] == ' ' && src[1] == '"');
    src += 2;

    // And the filename is everything up until the closing quotes
    std::string filename;
    while (*src != '"') {
        assert(*src && *src != '\n');
        filename.push_back(*src);
        ++src;
    }
    pos->name = strdup(filename.c_str());
}


/** Given a pointer to a position in a string, return the character that it
    represents, accounting for the escape characters supported in string
    constants.  (i.e. given the literal string "\\", return the character
    '/').  The return value is the new position in the string and the
    decoded character is returned in *pChar.
*/
static char *
lEscapeChar(char *str, char *pChar, SourcePos *pos)
{
    if (*str != '\\') {
        *pChar = *str;
    }
    else {
        char *tail;
        ++str;
        switch (*str) {
        case '\'': *pChar = '\''; break;
        case '\"': *pChar = '\"'; break;
        case '?':  *pChar = '\?'; break;
        case '\\': *pChar = '\\'; break;
        case 'a':  *pChar = '\a'; break;
        case 'b':  *pChar = '\b'; break;
        case 'f':  *pChar = '\f'; break;
        case 'n':  *pChar = '\n'; break;
        case 'r':  *pChar = '\r'; break;
        case 't':  *pChar = '\t'; break;
        case 'v':  *pChar = '\v'; break;
        // octal constants \012
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7':
            *pChar = (char)strtol(str, &tail, 8);
            str = tail - 1;
            break;
        // hexidecimal constant \xff
        case 'x':
            *pChar = (char)strtol(str, &tail, 16);
            str = tail - 1;
            break;
        default:
            Error(*pos, "Bad character escape sequence: '%s'\n.", str);
            break;
        }
    }
    ++str;
    return str;
}


/** Parse a string constant in the source file.  For each character in the
    string, handle any escaped characters with lEscapeChar() and keep eating
    characters until we come to the closing quote.
*/
static void
lStringConst(YYSTYPE *yylval, SourcePos *pos)
{
    char *p;
    std::string str;
    p = strchr(yytext, '"') + 1;
    while (*p != '\"') {
       char cval;
       p = lEscapeChar(p, &cval, pos);
       str.push_back(cval);
    } 
    yylval->stringVal = new std::string(str);
}


/** Compute the value 2^n, where the exponent is given as an integer.
    There are more efficient ways to do this, for example by just slamming
    the bits into the appropriate bits of the double, but let's just do the
    obvious thing. 
*/
static double
ipow2(int exponent) {
    if (exponent < 0)
        return 1. / ipow2(-exponent);

    double ret = 1.;
    while (exponent > 16) {
        ret *= 65536.;
        exponent -= 16;
    }
    while (exponent-- > 0)
        ret *= 2.;
    return ret;
}


/** Parse a hexadecimal-formatted floating-point number (C99 hex float
    constant-style). 
*/
static double
lParseHexFloat(const char *ptr) {
    assert(ptr != NULL);

    assert(ptr[0] == '0' && ptr[1] == 'x');
    ptr += 2;

    // Start initializing the mantissa
    assert(*ptr == '0' || *ptr == '1');
    double mantissa = (*ptr == '1') ? 1. : 0.;
    ++ptr;

    if (*ptr == '.') {
        // Is there a fraction part?  If so, the i'th digit we encounter
        // gives the 1/(16^i) component of the mantissa.
        ++ptr;

        double scale = 1. / 16.;
        // Keep going until we come to the 'p', which indicates that we've
        // come to the exponent
        while (*ptr != 'p') {
            // Figure out the raw value from 0-15
            int digit;
            if (*ptr >= '0' && *ptr <= '9')
                digit = *ptr - '0';
            else if (*ptr >= 'a' && *ptr <= 'f')
                digit = 10 + *ptr - 'a';
            else {
                assert(*ptr >= 'A' && *ptr <= 'F');
                digit = 10 + *ptr - 'A';
            }

            // And add its contribution to the mantissa
            mantissa += scale * digit;
            scale /= 16.;
            ++ptr;
        }
    }
    else
        // If there's not a '.', then we better be going straight to the
        // exponent
        assert(*ptr == 'p');

    ++ptr; // skip the 'p'

    // interestingly enough, the exponent is provided base 10..
    int exponent = (int)strtol(ptr, (char **)NULL, 10);

    // Does stdlib exp2() guarantee exact results for integer n where can
    // be represented exactly as doubles?  I would hope so but am not sure,
    // so let's be sure.
    return mantissa * ipow2(exponent);
}
