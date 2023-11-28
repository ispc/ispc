/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

%{

#include "ispc.h"
#include "sym.h"
#include "util.h"
#include "module.h"
#include "type.h"
#include <stdlib.h>
#include <stdint.h>

using namespace ispc;
#include "parse.hh"

static uint64_t lParseBinary(const char *ptr, SourcePos pos, char **endPtr);
static int lParseInteger(bool dotdotdot);
static int lParseFP();
static int lParseOperator(const char *ptr);
static void lCComment(SourcePos *);
static void lCppComment(SourcePos *);
static void lNextValidChar(SourcePos *, char const*&);
static void lPragmaIgnoreWarning(SourcePos *, std::string);
static void lPragmaUnroll(YYSTYPE *, SourcePos *, std::string, bool);
static bool lConsumePragma(YYSTYPE *, SourcePos *);
static void lHandleCppHash(SourcePos *);
static void lStringConst(YYSTYPE *, SourcePos *);
static double lParseHexFloat(const char *ptr);
extern const char *RegisterDependency(const std::string &fileName);

#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line; \
    yylloc.first_column = yylloc.last_column; \
    yylloc.last_column += yyleng;

#ifdef ISPC_HOST_IS_WINDOWS
inline int isatty(int) { return 0; }
#else
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS

static int allTokens[] = {
  TOKEN_ASSERT, TOKEN_BOOL, TOKEN_BREAK, TOKEN_CASE,
  TOKEN_CDO, TOKEN_CFOR, TOKEN_CIF, TOKEN_CWHILE,
  TOKEN_CONST, TOKEN_CONTINUE, TOKEN_DEFAULT, TOKEN_DO,
  TOKEN_DELETE, TOKEN_DOUBLE, TOKEN_ELSE, TOKEN_ENUM,
  TOKEN_EXPORT, TOKEN_EXTERN, TOKEN_FALSE, TOKEN_FLOAT, TOKEN_FLOAT16, TOKEN_FOR,
  TOKEN_FOREACH, TOKEN_FOREACH_ACTIVE, TOKEN_FOREACH_TILED,
  TOKEN_FOREACH_UNIQUE, TOKEN_GOTO, TOKEN_IF, TOKEN_IN, TOKEN_INLINE,
  TOKEN_INT, TOKEN_INT8, TOKEN_INT16, TOKEN_INT, TOKEN_INT64, TOKEN_LAUNCH,
  TOKEN_UINT, TOKEN_UINT8, TOKEN_UINT16, TOKEN_UINT64,
  TOKEN_NEW, TOKEN_NULL, TOKEN_PRINT, TOKEN_RETURN, TOKEN_SOA, TOKEN_SIGNED,
  TOKEN_SIZEOF, TOKEN_ALLOCA, TOKEN_STATIC, TOKEN_STRUCT, TOKEN_SWITCH, TOKEN_SYNC,
  TOKEN_TASK, TOKEN_TEMPLATE, TOKEN_TRUE, TOKEN_TYPEDEF, TOKEN_TYPENAME,
  TOKEN_UNIFORM, TOKEN_UNMASKED, TOKEN_UNSIGNED, TOKEN_VARYING, TOKEN_VOID, TOKEN_WHILE,
  TOKEN_STRING_C_LITERAL, TOKEN_STRING_SYCL_LITERAL, TOKEN_DOTDOTDOT,
  TOKEN_FLOAT_CONSTANT, TOKEN_FLOAT16_CONSTANT, TOKEN_DOUBLE_CONSTANT,
  TOKEN_INT8_CONSTANT, TOKEN_UINT8_CONSTANT,
  TOKEN_INT16_CONSTANT, TOKEN_UINT16_CONSTANT,
  TOKEN_INT32_CONSTANT, TOKEN_UINT32_CONSTANT,
  TOKEN_INT64_CONSTANT, TOKEN_UINT64_CONSTANT,
  TOKEN_INC_OP, TOKEN_DEC_OP, TOKEN_LEFT_OP, TOKEN_RIGHT_OP, TOKEN_LE_OP,
  TOKEN_GE_OP, TOKEN_EQ_OP, TOKEN_NE_OP, TOKEN_AND_OP, TOKEN_OR_OP,
  TOKEN_MUL_ASSIGN, TOKEN_DIV_ASSIGN, TOKEN_MOD_ASSIGN, TOKEN_ADD_ASSIGN,
  TOKEN_SUB_ASSIGN, TOKEN_LEFT_ASSIGN, TOKEN_RIGHT_ASSIGN, TOKEN_AND_ASSIGN,
  TOKEN_XOR_ASSIGN, TOKEN_OR_ASSIGN, TOKEN_PTR_OP, TOKEN_NOINLINE, TOKEN_VECTORCALL,
  TOKEN_REGCALL, TOKEN_INVOKE_SYCL,
  ';', '{', '}', ',', ':', '=', '(', ')', '[', ']', '.', '&', '!', '~', '-',
  '+', '*', '/', '%', '<', '>', '^', '|', '?',
};

std::map<int, std::string> tokenToName;
std::map<std::string, std::string> tokenNameRemap;

void ParserInit() {
    tokenToName[TOKEN_ASSERT] = "assert";
    tokenToName[TOKEN_BOOL] = "bool";
    tokenToName[TOKEN_BREAK] = "break";
    tokenToName[TOKEN_CASE] = "case";
    tokenToName[TOKEN_CDO] = "cdo";
    tokenToName[TOKEN_CFOR] = "cfor";
    tokenToName[TOKEN_CIF] = "cif";
    tokenToName[TOKEN_CWHILE] = "cwhile";
    tokenToName[TOKEN_CONST] = "const";
    tokenToName[TOKEN_CONTINUE] = "continue";
    tokenToName[TOKEN_DEFAULT] = "default";
    tokenToName[TOKEN_DO] = "do";
    tokenToName[TOKEN_DELETE] = "delete";
    tokenToName[TOKEN_DOUBLE] = "double";
    tokenToName[TOKEN_ELSE] = "else";
    tokenToName[TOKEN_ENUM] = "enum";
    tokenToName[TOKEN_EXPORT] = "export";
    tokenToName[TOKEN_EXTERN] = "extern";
    tokenToName[TOKEN_FALSE] = "false";
    tokenToName[TOKEN_FLOAT] = "float";
    tokenToName[TOKEN_FLOAT16] = "float16";
    tokenToName[TOKEN_FOR] = "for";
    tokenToName[TOKEN_FOREACH] = "foreach";
    tokenToName[TOKEN_FOREACH_ACTIVE] = "foreach_active";
    tokenToName[TOKEN_FOREACH_TILED] = "foreach_tiled";
    tokenToName[TOKEN_FOREACH_UNIQUE] = "foreach_unique";
    tokenToName[TOKEN_GOTO] = "goto";
    tokenToName[TOKEN_IF] = "if";
    tokenToName[TOKEN_IN] = "in";
    tokenToName[TOKEN_INLINE] = "inline";
    tokenToName[TOKEN_NOINLINE] = "noinline";
    tokenToName[TOKEN_VECTORCALL] = "__vectorcall";
    tokenToName[TOKEN_REGCALL] = "__regcall";
    tokenToName[TOKEN_INT] = "int";
    tokenToName[TOKEN_UINT] = "uint";
    tokenToName[TOKEN_INT8] = "int8";
    tokenToName[TOKEN_UINT8] = "uint8";
    tokenToName[TOKEN_INT16] = "int16";
    tokenToName[TOKEN_UINT16] = "uint16";
    tokenToName[TOKEN_INT] = "int";
    tokenToName[TOKEN_INT64] = "int64";
    tokenToName[TOKEN_UINT64] = "uint64";
    tokenToName[TOKEN_LAUNCH] = "launch";
    tokenToName[TOKEN_INVOKE_SYCL] = "invoke_sycl";
    tokenToName[TOKEN_NEW] = "new";
    tokenToName[TOKEN_NULL] = "NULL";
    tokenToName[TOKEN_PRINT] = "print";
    tokenToName[TOKEN_RETURN] = "return";
    tokenToName[TOKEN_SOA] = "soa";
    tokenToName[TOKEN_SIGNED] = "signed";
    tokenToName[TOKEN_SIZEOF] = "sizeof";
    tokenToName[TOKEN_ALLOCA] = "alloca";
    tokenToName[TOKEN_STATIC] = "static";
    tokenToName[TOKEN_STRUCT] = "struct";
    tokenToName[TOKEN_SWITCH] = "switch";
    tokenToName[TOKEN_SYNC] = "sync";
    tokenToName[TOKEN_TASK] = "task";
    tokenToName[TOKEN_TEMPLATE] = "template";
    tokenToName[TOKEN_TRUE] = "true";
    tokenToName[TOKEN_TYPEDEF] = "typedef";
    tokenToName[TOKEN_TYPENAME] = "typename";
    tokenToName[TOKEN_UNIFORM] = "uniform";
    tokenToName[TOKEN_UNMASKED] = "unmasked";
    tokenToName[TOKEN_UNSIGNED] = "unsigned";
    tokenToName[TOKEN_VARYING] = "varying";
    tokenToName[TOKEN_VOID] = "void";
    tokenToName[TOKEN_WHILE] = "while";
    tokenToName[TOKEN_STRING_C_LITERAL] = "\"C\"";
    tokenToName[TOKEN_STRING_SYCL_LITERAL] = "\"SYCL\"";
    tokenToName[TOKEN_DOTDOTDOT] = "...";
    tokenToName[TOKEN_FLOAT_CONSTANT] = "TOKEN_FLOAT_CONSTANT";
    tokenToName[TOKEN_FLOAT16_CONSTANT] = "TOKEN_FLOAT16_CONSTANT";
    tokenToName[TOKEN_DOUBLE_CONSTANT] = "TOKEN_DOUBLE_CONSTANT";
    tokenToName[TOKEN_INT8_CONSTANT] = "TOKEN_INT8_CONSTANT";
    tokenToName[TOKEN_UINT8_CONSTANT] = "TOKEN_UINT8_CONSTANT";
    tokenToName[TOKEN_INT16_CONSTANT] = "TOKEN_INT16_CONSTANT";
    tokenToName[TOKEN_UINT16_CONSTANT] = "TOKEN_UINT16_CONSTANT";
    tokenToName[TOKEN_INT32_CONSTANT] = "TOKEN_INT32_CONSTANT";
    tokenToName[TOKEN_UINT32_CONSTANT] = "TOKEN_UINT32_CONSTANT";
    tokenToName[TOKEN_INT64_CONSTANT] = "TOKEN_INT64_CONSTANT";
    tokenToName[TOKEN_UINT64_CONSTANT] = "TOKEN_UINT64_CONSTANT";
    tokenToName[TOKEN_INC_OP] = "++";
    tokenToName[TOKEN_DEC_OP] = "--";
    tokenToName[TOKEN_LEFT_OP] = "<<";
    tokenToName[TOKEN_RIGHT_OP] = ">>";
    tokenToName[TOKEN_LE_OP] = "<=";
    tokenToName[TOKEN_GE_OP] = ">=";
    tokenToName[TOKEN_EQ_OP] = "==";
    tokenToName[TOKEN_NE_OP] = "!=";
    tokenToName[TOKEN_AND_OP] = "&&";
    tokenToName[TOKEN_OR_OP] = "||";
    tokenToName[TOKEN_MUL_ASSIGN] = "*=";
    tokenToName[TOKEN_DIV_ASSIGN] = "/=";
    tokenToName[TOKEN_MOD_ASSIGN] = "%=";
    tokenToName[TOKEN_ADD_ASSIGN] = "+=";
    tokenToName[TOKEN_SUB_ASSIGN] = "-=";
    tokenToName[TOKEN_LEFT_ASSIGN] = "<<=";
    tokenToName[TOKEN_RIGHT_ASSIGN] = ">>=";
    tokenToName[TOKEN_AND_ASSIGN] = "&=";
    tokenToName[TOKEN_XOR_ASSIGN] = "^=";
    tokenToName[TOKEN_OR_ASSIGN] = "|=";
    tokenToName[TOKEN_PTR_OP] = "->";
    tokenToName[';'] = ";";
    tokenToName['{'] = "{";
    tokenToName['}'] = "}";
    tokenToName[','] = ",";
    tokenToName[':'] = ":";
    tokenToName['='] = "=";
    tokenToName['('] = "(";
    tokenToName[')'] = ")";
    tokenToName['['] = "[";
    tokenToName[']'] = "]";
    tokenToName['.'] = ".";
    tokenToName['&'] = "&";
    tokenToName['!'] = "!";
    tokenToName['~'] = "~";
    tokenToName['-'] = "-";
    tokenToName['+'] = "+";
    tokenToName['*'] = "*";
    tokenToName['/'] = "/";
    tokenToName['%'] = "%";
    tokenToName['<'] = "<";
    tokenToName['>'] = ">";
    tokenToName['^'] = "^";
    tokenToName['|'] = "|";
    tokenToName['?'] = "?";
    tokenToName[';'] = ";";

    tokenNameRemap["TOKEN_ASSERT"] = "\'assert\'";
    tokenNameRemap["TOKEN_BOOL"] = "\'bool\'";
    tokenNameRemap["TOKEN_BREAK"] = "\'break\'";
    tokenNameRemap["TOKEN_CASE"] = "\'case\'";
    tokenNameRemap["TOKEN_CDO"] = "\'cdo\'";
    tokenNameRemap["TOKEN_CFOR"] = "\'cfor\'";
    tokenNameRemap["TOKEN_CIF"] = "\'cif\'";
    tokenNameRemap["TOKEN_CWHILE"] = "\'cwhile\'";
    tokenNameRemap["TOKEN_CONST"] = "\'const\'";
    tokenNameRemap["TOKEN_CONTINUE"] = "\'continue\'";
    tokenNameRemap["TOKEN_DEFAULT"] = "\'default\'";
    tokenNameRemap["TOKEN_DO"] = "\'do\'";
    tokenNameRemap["TOKEN_DELETE"] = "\'delete\'";
    tokenNameRemap["TOKEN_DOUBLE"] = "\'double\'";
    tokenNameRemap["TOKEN_ELSE"] = "\'else\'";
    tokenNameRemap["TOKEN_ENUM"] = "\'enum\'";
    tokenNameRemap["TOKEN_EXPORT"] = "\'export\'";
    tokenNameRemap["TOKEN_EXTERN"] = "\'extern\'";
    tokenNameRemap["TOKEN_FALSE"] = "\'false\'";
    tokenNameRemap["TOKEN_FLOAT"] = "\'float\'";
    tokenNameRemap["TOKEN_FLOAT16"] = "\'float16\'";
    tokenNameRemap["TOKEN_FOR"] = "\'for\'";
    tokenNameRemap["TOKEN_FOREACH"] = "\'foreach\'";
    tokenNameRemap["TOKEN_FOREACH_ACTIVE"] = "\'foreach_active\'";
    tokenNameRemap["TOKEN_FOREACH_TILED"] = "\'foreach_tiled\'";
    tokenNameRemap["TOKEN_FOREACH_UNIQUE"] = "\'foreach_unique\'";
    tokenNameRemap["TOKEN_GOTO"] = "\'goto\'";
    tokenNameRemap["TOKEN_IDENTIFIER"] = "identifier";
    tokenNameRemap["TOKEN_IF"] = "\'if\'";
    tokenNameRemap["TOKEN_IN"] = "\'in\'";
    tokenNameRemap["TOKEN_INLINE"] = "\'inline\'";
    tokenNameRemap["TOKEN_NOINLINE"] = "\'noinline\'";
    tokenNameRemap["TOKEN_VECTORCALL"] = "\'__vectorcall\'";
    tokenNameRemap["TOKEN_REGCALL"] = "\'__regcall\'";
    tokenNameRemap["TOKEN_INT"] = "\'int\'";
    tokenNameRemap["TOKEN_UINT"] = "\'uint\'";
    tokenNameRemap["TOKEN_INT8"] = "\'int8\'";
    tokenNameRemap["TOKEN_UINT8"] = "\'uint8\'";
    tokenNameRemap["TOKEN_INT16"] = "\'int16\'";
    tokenNameRemap["TOKEN_UINT16"] = "\'uint16\'";
    tokenNameRemap["TOKEN_INT"] = "\'int\'";
    tokenNameRemap["TOKEN_INT64"] = "\'int64\'";
    tokenNameRemap["TOKEN_UINT64"] = "\'uint64\'";
    tokenNameRemap["TOKEN_LAUNCH"] = "\'launch\'";
    tokenNameRemap["TOKEN_INVOKE_SYCL"] = "\'invoke_sycl\'";
    tokenNameRemap["TOKEN_NEW"] = "\'new\'";
    tokenNameRemap["TOKEN_NULL"] = "\'NULL\'";
    tokenNameRemap["TOKEN_PRINT"] = "\'print\'";
    tokenNameRemap["TOKEN_RETURN"] = "\'return\'";
    tokenNameRemap["TOKEN_SOA"] = "\'soa\'";
    tokenNameRemap["TOKEN_SIGNED"] = "\'signed\'";
    tokenNameRemap["TOKEN_SIZEOF"] = "\'sizeof\'";
    tokenNameRemap["TOKEN_ALLOCA"] = "\'TOKEN_ALLOCA\'";
    tokenNameRemap["TOKEN_STATIC"] = "\'static\'";
    tokenNameRemap["TOKEN_STRUCT"] = "\'struct\'";
    tokenNameRemap["TOKEN_SWITCH"] = "\'switch\'";
    tokenNameRemap["TOKEN_SYNC"] = "\'sync\'";
    tokenNameRemap["TOKEN_TASK"] = "\'task\'";
    tokenNameRemap["TOKEN_TEMPLATE"] = "\'template\'";
    tokenNameRemap["TOKEN_TRUE"] = "\'true\'";
    tokenNameRemap["TOKEN_TYPEDEF"] = "\'typedef\'";
    tokenNameRemap["TOKEN_TYPENAME"] = "\'typename\'";
    tokenNameRemap["TOKEN_UNIFORM"] = "\'uniform\'";
    tokenNameRemap["TOKEN_UNMASKED"] = "\'unmasked\'";
    tokenNameRemap["TOKEN_UNSIGNED"] = "\'unsigned\'";
    tokenNameRemap["TOKEN_VARYING"] = "\'varying\'";
    tokenNameRemap["TOKEN_VOID"] = "\'void\'";
    tokenNameRemap["TOKEN_WHILE"] = "\'while\'";
    tokenNameRemap["TOKEN_STRING_C_LITERAL"] = "\"C\"";
    tokenNameRemap["TOKEN_STRING_SYCL_LITERAL"] = "\"SYCL\"";
    tokenNameRemap["TOKEN_DOTDOTDOT"] = "\'...\'";
    tokenNameRemap["TOKEN_FLOAT_CONSTANT"] = "float constant";
    tokenNameRemap["TOKEN_FLOAT16_CONSTANT"] = "float16 constant";
    tokenNameRemap["TOKEN_DOUBLE_CONSTANT"] = "double constant";
    tokenNameRemap["TOKEN_INT8_CONSTANT"] = "int8 constant";
    tokenNameRemap["TOKEN_UINT8_CONSTANT"] = "unsigned int8 constant";
    tokenNameRemap["TOKEN_INT16_CONSTANT"] = "int16 constant";
    tokenNameRemap["TOKEN_UINT16_CONSTANT"] = "unsigned int16 constant";
    tokenNameRemap["TOKEN_INT32_CONSTANT"] = "int32 constant";
    tokenNameRemap["TOKEN_UINT32_CONSTANT"] = "unsigned int32 constant";
    tokenNameRemap["TOKEN_INT64_CONSTANT"] = "int64 constant";
    tokenNameRemap["TOKEN_UINT64_CONSTANT"] = "unsigned int64 constant";
    tokenNameRemap["TOKEN_INC_OP"] = "\'++\'";
    tokenNameRemap["TOKEN_DEC_OP"] = "\'--\'";
    tokenNameRemap["TOKEN_LEFT_OP"] = "\'<<\'";
    tokenNameRemap["TOKEN_RIGHT_OP"] = "\'>>\'";
    tokenNameRemap["TOKEN_LE_OP"] = "\'<=\'";
    tokenNameRemap["TOKEN_GE_OP"] = "\'>=\'";
    tokenNameRemap["TOKEN_EQ_OP"] = "\'==\'";
    tokenNameRemap["TOKEN_NE_OP"] = "\'!=\'";
    tokenNameRemap["TOKEN_AND_OP"] = "\'&&\'";
    tokenNameRemap["TOKEN_OR_OP"] = "\'||\'";
    tokenNameRemap["TOKEN_MUL_ASSIGN"] = "\'*=\'";
    tokenNameRemap["TOKEN_DIV_ASSIGN"] = "\'/=\'";
    tokenNameRemap["TOKEN_MOD_ASSIGN"] = "\'%=\'";
    tokenNameRemap["TOKEN_ADD_ASSIGN"] = "\'+=\'";
    tokenNameRemap["TOKEN_SUB_ASSIGN"] = "\'-=\'";
    tokenNameRemap["TOKEN_LEFT_ASSIGN"] = "\'<<=\'";
    tokenNameRemap["TOKEN_RIGHT_ASSIGN"] = "\'>>=\'";
    tokenNameRemap["TOKEN_AND_ASSIGN"] = "\'&=\'";
    tokenNameRemap["TOKEN_XOR_ASSIGN"] = "\'^=\'";
    tokenNameRemap["TOKEN_OR_ASSIGN"] = "\'|=\'";
    tokenNameRemap["TOKEN_PTR_OP"] = "\'->\'";
    tokenNameRemap["$end"] = "end of file";
}


inline int ispcRand() {
#ifdef ISPC_HOST_IS_WINDOWS
    return rand();
#else
    return lrand48();
#endif
}

#define RT \
    do { \
    if (g->enableFuzzTest) { \
        int r = ispcRand() % 40; \
        if (r == 0) { \
            Warning(yylloc, "Fuzz test dropping token"); \
        } \
        else if (r == 1) { \
            Assert (tokenToName.size() > 0); \
            int nt = sizeof(allTokens) / sizeof(allTokens[0]); \
            int tn = ispcRand() % nt; \
            yylval.stringVal = new std::string(yytext); /* just in case */\
            Warning(yylloc, "Fuzz test replaced token with \"%s\"", tokenToName[allTokens[tn]].c_str()); \
            return allTokens[tn]; \
        } \
        else if (r == 2) { \
            Symbol *sym = m->symbolTable->RandomSymbol(); \
            if (sym != nullptr) { \
                yylval.stringVal = new std::string(sym->name); \
                Warning(yylloc, "Fuzz test replaced with identifier \"%s\".", sym->name.c_str()); \
                return TOKEN_IDENTIFIER; \
            } \
        } \
        /*  TOKEN_TYPE_NAME */ \
     } } while(0)

%}

%option nounput
%option noyywrap
%option nounistd

WHITESPACE [ \t\r]+
INT_NUMBER (([0-9]+)|(0[xX][0-9a-fA-F]+)|(0b[01]+))[uUlL]*[kMG]?[uUlL]*
INT_NUMBER_DOTDOTDOT (([0-9]+)|(0[xX][0-9a-fA-F]+)|(0b[01]+))[uUlL]*[kMG]?[uUlL]*\.\.\.
FLOAT_NUMBER_DECIMAL ((([0-9]+\.[0-9]*)|(\.[0-9]+))([dD]|[fF]|[fF]16)?)
FLOAT_NUMBER_DECIMAL_DEPRECATED ([0-9]+[fF])
FLOAT_NUMBER_DECIMAL_ILLEGAL ([0-9]+([dD]|[fF]16))
FLOAT_NUMBER_SCIENTIFIC (([0-9]+|(([0-9]+\.[0-9]*)|(\.[0-9]+)))([eE][-+]?[0-9]+)([dD]|[fF]|[fF]16)?)
FLOAT_NUMBER_HEXADECIMAL (0[xX][01](\.[0-9a-fA-F]*)?[pP][-+]?[0-9]+([dD]|[fF]|[fF]16)?)
FORTRAN_DOUBLE_NUMBER (([0-9]+|(([0-9]+\.[0-9]*)|(\.[0-9]+)))([dD][-+]?[0-9]+))



IDENT [a-zA-Z_][a-zA-Z_0-9]*
INTRINSIC_CALL [@][l][l][v][m][.][.a-zA-Z_0-9]*
ZO_SWIZZLE ([01]+[w-z]+)+|([01]+[rgba]+)+|([01]+[uv]+)+

%%
"/*"            { lCComment(&yylloc); }
"//"            { lCppComment(&yylloc); }
"#pragma" {
    if (lConsumePragma(&yylval, &yylloc)) {
        return TOKEN_PRAGMA;
    }
}


__assert { RT; return TOKEN_ASSERT; }
bool { RT; return TOKEN_BOOL; }
break { RT; return TOKEN_BREAK; }
case { RT; return TOKEN_CASE; }
cbreak { RT; Warning(yylloc, "\"cbreak\" is deprecated. Use \"break\"."); return TOKEN_BREAK; }
ccontinue { RT; Warning(yylloc, "\"ccontinue\" is deprecated. Use \"continue\"."); return TOKEN_CONTINUE; }
cdo { RT; return TOKEN_CDO; }
cfor { RT; return TOKEN_CFOR; }
cif { RT; return TOKEN_CIF; }
cwhile { RT; return TOKEN_CWHILE; }
const { RT; return TOKEN_CONST; }
continue { RT; return TOKEN_CONTINUE; }
creturn { RT; Warning(yylloc, "\"creturn\" is deprecated. Use \"return\"."); return TOKEN_RETURN; }
__declspec { RT; return TOKEN_DECLSPEC; }
default { RT; return TOKEN_DEFAULT; }
do { RT; return TOKEN_DO; }
delete { RT; return TOKEN_DELETE; }
delete\[\] { RT; return TOKEN_DELETE; }
double { RT; return TOKEN_DOUBLE; }
else { RT; return TOKEN_ELSE; }
enum { RT; return TOKEN_ENUM; }
export { RT; return TOKEN_EXPORT; }
extern { RT; return TOKEN_EXTERN; }
false { RT; return TOKEN_FALSE; }
float { RT; return TOKEN_FLOAT; }
for { RT; return TOKEN_FOR; }
foreach { RT; return TOKEN_FOREACH; }
foreach_active { RT; return TOKEN_FOREACH_ACTIVE; }
foreach_tiled { RT; return TOKEN_FOREACH_TILED; }
foreach_unique { RT; return TOKEN_FOREACH_UNIQUE; }
float16 { RT; return TOKEN_FLOAT16; }
goto { RT; return TOKEN_GOTO; }
if { RT; return TOKEN_IF; }
in { RT; return TOKEN_IN; }
inline { RT; return TOKEN_INLINE; }
noinline { RT; return TOKEN_NOINLINE; }
__vectorcall { RT; return TOKEN_VECTORCALL; }
__regcall { RT; return TOKEN_REGCALL; }
int { RT; return TOKEN_INT; }
uint { RT; return TOKEN_UINT; }
int8 { RT; return TOKEN_INT8; }
uint8 { RT; return TOKEN_UINT8; }
int16 { RT; return TOKEN_INT16; }
uint16 { RT; return TOKEN_UINT16; }
int32 { RT; return TOKEN_INT; }
uint32 { RT; return TOKEN_UINT; }
int64 { RT; return TOKEN_INT64; }
uint64 { RT; return TOKEN_UINT64; }
launch { RT; return TOKEN_LAUNCH; }
invoke_sycl { RT; return TOKEN_INVOKE_SYCL; }
new { RT; return TOKEN_NEW; }
NULL { RT; return TOKEN_NULL; }
print { RT; return TOKEN_PRINT; }
return { RT; return TOKEN_RETURN; }
soa { RT; return TOKEN_SOA; }
signed { RT; return TOKEN_SIGNED; }
sizeof { RT; return TOKEN_SIZEOF; }
alloca { RT; return TOKEN_ALLOCA; }
static { RT; return TOKEN_STATIC; }
struct { RT; return TOKEN_STRUCT; }
switch { RT; return TOKEN_SWITCH; }
sync { RT; return TOKEN_SYNC; }
task { RT; return TOKEN_TASK; }
template { RT; return TOKEN_TEMPLATE; }
true { RT; return TOKEN_TRUE; }
typedef { RT; return TOKEN_TYPEDEF; }
typename { RT; return TOKEN_TYPENAME; }
uniform { RT; return TOKEN_UNIFORM; }
unmasked { RT; return TOKEN_UNMASKED; }
unsigned { RT; return TOKEN_UNSIGNED; }
varying { RT; return TOKEN_VARYING; }
void { RT; return TOKEN_VOID; }
while { RT; return TOKEN_WHILE; }
\"C\" { RT; return TOKEN_STRING_C_LITERAL; }
\"SYCL\" { RT; return TOKEN_STRING_SYCL_LITERAL; }
\.\.\. { RT; return TOKEN_DOTDOTDOT; }

"operator*"  { return lParseOperator(yytext); }
"operator+"  { return lParseOperator(yytext); }
"operator-"  { return lParseOperator(yytext); }
"operator<<" { return lParseOperator(yytext); }
"operator>>" { return lParseOperator(yytext); }
"operator/"  { return lParseOperator(yytext); }
"operator%"  { return lParseOperator(yytext); }

L?\"(\\.|[^\\"])*\" { lStringConst(&yylval, &yylloc); return TOKEN_STRING_LITERAL; }

{IDENT} {
    RT;
    /* We have an identifier--is it a type name or an identifier?
       The symbol table will straighten us out... */
    yylval.stringVal = new std::string(yytext);
    if (m->symbolTable->LookupType(yytext) != nullptr)
        return TOKEN_TYPE_NAME;
    else if (m->symbolTable->LookupFunctionTemplate(yytext))
        return TOKEN_TEMPLATE_NAME;
    else
        return TOKEN_IDENTIFIER;
}

{INTRINSIC_CALL} {
    RT;
    /* We have a potential llvm intrinsic call.*/
    yylval.stringVal = new std::string(yytext);
    return TOKEN_INTRINSIC_CALL;
}

{INT_NUMBER} {
    RT;
    return lParseInteger(false);
}

{INT_NUMBER_DOTDOTDOT} {
    RT;
    return lParseInteger(true);
}

{FORTRAN_DOUBLE_NUMBER} {
    RT;
    {
      int i = 0;
      while (yytext[i] != 'd' && yytext[i] != 'D') i++;
      yytext[i] = 'E';
    }
    yylval.doubleVal = atof(yytext);
    return TOKEN_DOUBLE_CONSTANT;
}

{FLOAT_NUMBER_DECIMAL}|{FLOAT_NUMBER_SCIENTIFIC} {
    RT;
    return lParseFP();
}

{FLOAT_NUMBER_DECIMAL_DEPRECATED} {
    RT;
    Warning(yylloc, "single precision floating point literal should have a radix separator (dot)");
    return lParseFP();
}

{FLOAT_NUMBER_DECIMAL_ILLEGAL} {
    RT;
    Error(yylloc, "floating point literal should have a radix separator (dot)");
    return lParseFP();
}

{FLOAT_NUMBER_HEXADECIMAL} {
    RT;
    std::string val(yytext);
    std::string fp16S("f16");
    std::string fp16C("F16");
    if (val.size() >= fp16S.size() && ((val.compare(val.size() - fp16S.size(), fp16S.size(), fp16S) == 0)
           || (val.compare(val.size() - fp16C.size(), fp16C.size(), fp16C) == 0))) {
        yylval.stringVal = new std::string(val.substr(0, val.length() - 3));
        return TOKEN_FLOAT16_CONSTANT;
    }
    double dval = lParseHexFloat(yytext);
    std::string fp64S("d");
    std::string fp64C("D");
    if (val.size() >= fp64S.size() && ((val.compare(val.size() - fp64S.size(), fp64S.size(), fp64S) == 0)
           || (val.compare(val.size() - fp64C.size(), fp64C.size(), fp64C) == 0))) {
        yylval.doubleVal = dval;
        return TOKEN_DOUBLE_CONSTANT;
    }
    yylval.floatVal = (float)dval;
    return TOKEN_FLOAT_CONSTANT;
}



"++" { RT; return TOKEN_INC_OP; }
"--" { RT; return TOKEN_DEC_OP; }
"<<" { RT; return TOKEN_LEFT_OP; }
">>" { RT; return TOKEN_RIGHT_OP; }
"<=" { RT; return TOKEN_LE_OP; }
">=" { RT; return TOKEN_GE_OP; }
"==" { RT; return TOKEN_EQ_OP; }
"!=" { RT; return TOKEN_NE_OP; }
"&&" { RT; return TOKEN_AND_OP; }
"||" { RT; return TOKEN_OR_OP; }
"*=" { RT; return TOKEN_MUL_ASSIGN; }
"/=" { RT; return TOKEN_DIV_ASSIGN; }
"%=" { RT; return TOKEN_MOD_ASSIGN; }
"+=" { RT; return TOKEN_ADD_ASSIGN; }
"-=" { RT; return TOKEN_SUB_ASSIGN; }
"<<=" { RT; return TOKEN_LEFT_ASSIGN; }
">>=" { RT; return TOKEN_RIGHT_ASSIGN; }
"&=" { RT; return TOKEN_AND_ASSIGN; }
"^=" { RT; return TOKEN_XOR_ASSIGN; }
"|=" { RT; return TOKEN_OR_ASSIGN; }
"->" { RT; return TOKEN_PTR_OP; }
";"             { RT; return ';'; }
("{"|"<%")      { RT; return '{'; }
("}"|"%>")      { RT; return '}'; }
","             { RT; return ','; }
":"             { RT; return ':'; }
"="             { RT; return '='; }
"("             { RT; return '('; }
")"             { RT; return ')'; }
("["|"<:")      { RT; return '['; }
("]"|":>")      { RT; return ']'; }
"."             { RT; return '.'; }
"&"             { RT; return '&'; }
"!"             { RT; return '!'; }
"~"             { RT; return '~'; }
"-"             { RT; return '-'; }
"+"             { RT; return '+'; }
"*"             { RT; return '*'; }
"/"             { RT; return '/'; }
"%"             { RT; return '%'; }
"<"             { RT; return '<'; }
">"             { RT; return '>'; }
"^"             { RT; return '^'; }
"|"             { RT; return '|'; }
"?"             { RT; return '?'; }

{WHITESPACE} { }

\n {
    yylloc.last_line++;
    yylloc.last_column = 1;
}

#(line)?[ ][0-9]+[ ]\"(\\.|[^\\"])*\"[^\n]* {
    lHandleCppHash(&yylloc);
}

. {
    Error(yylloc, "Illegal character: %c (0x%x)", yytext[0], int(yytext[0]));
    YY_USER_ACTION
}

%%

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
lParseBinary(const char *ptr, SourcePos pos, char **endPtr) {
    uint64_t val = 0;
    bool warned = false;

    while (*ptr == '0' || *ptr == '1') {
        if ((val & (((int64_t)1)<<63)) && warned == false) {
            // We're about to shift out a set bit
            Warning(pos, "Can't represent binary constant with a 64-bit integer type");
            warned = true;
        }

        val = (val << 1) | (*ptr == '0' ? 0 : 1);
        ++ptr;
    }
    *endPtr = (char *)ptr;
    return val;
}


static int
lParseInteger(bool dotdotdot) {
    int ls = 0, us = 0;

    char *endPtr = nullptr;
    if (yytext[0] == '0' && yytext[1] == 'b')
        yylval.intVal = lParseBinary(yytext+2, yylloc, &endPtr);
    else {
#if defined(ISPC_HOST_IS_WINDOWS) && !defined(__MINGW32__)
        yylval.intVal = _strtoui64(yytext, &endPtr, 0);
#else
        // FIXME: should use strtouq and then issue an error if we can't
        // fit into 64 bits...
        yylval.intVal = strtoull(yytext, &endPtr, 0);
#endif
    }

    bool kilo = false, mega = false, giga = false;
    for (; *endPtr; endPtr++) {
        if (*endPtr == 'k')
            kilo = true;
        else if (*endPtr == 'M')
            mega = true;
        else if (*endPtr == 'G')
            giga = true;
        else if (*endPtr == 'l' || *endPtr == 'L')
            ls++;
        else if (*endPtr == 'u' || *endPtr == 'U')
            us++;
        else
            Assert(dotdotdot && *endPtr == '.');
    }
    if (kilo)
        yylval.intVal *= 1024;
    if (mega)
        yylval.intVal *= 1024*1024;
    if (giga)
        yylval.intVal *= 1024*1024*1024;

    if (dotdotdot) {
        if (ls >= 2)
            return us ? TOKEN_UINT64DOTDOTDOT_CONSTANT : TOKEN_INT64DOTDOTDOT_CONSTANT;
        else if (ls == 1)
            return us ? TOKEN_UINT32DOTDOTDOT_CONSTANT : TOKEN_INT32DOTDOTDOT_CONSTANT;

        // See if we can fit this into a 32-bit integer...
        if ((yylval.intVal & 0xffffffff) == yylval.intVal)
            return us ? TOKEN_UINT32DOTDOTDOT_CONSTANT : TOKEN_INT32DOTDOTDOT_CONSTANT;
        else
            return us ? TOKEN_UINT64DOTDOTDOT_CONSTANT : TOKEN_INT64DOTDOTDOT_CONSTANT;
    }
    else {
        if (ls >= 2)
            return us ? TOKEN_UINT64_CONSTANT : TOKEN_INT64_CONSTANT;
        else if (ls == 1)
            return us ? TOKEN_UINT32_CONSTANT : TOKEN_INT32_CONSTANT;
        else if (us) {
            // u suffix only
            if (yylval.intVal <= 0xffffffffL)
                return TOKEN_UINT32_CONSTANT;
            else
                return TOKEN_UINT64_CONSTANT;
        }
        else {
            // No u or l suffix
            // If we're compiling to an 8-bit mask target and the constant
            // fits into 8 bits, return an 8-bit int.
            if (g->target->getDataTypeWidth() == 8) {
                if (yylval.intVal <= 0x7fULL)
                    return TOKEN_INT8_CONSTANT;
                else if (yylval.intVal <= 0xffULL)
                    return TOKEN_UINT8_CONSTANT;
            }
            // And similarly for 16-bit masks and constants
            if (g->target->getDataTypeWidth() == 16) {
                if (yylval.intVal <= 0x7fffULL)
                    return TOKEN_INT16_CONSTANT;
                else if (yylval.intVal <= 0xffffULL)
                    return TOKEN_UINT16_CONSTANT;
            }
            // Otherwise, see if we can fit this into a 32-bit integer...
            if (yylval.intVal <= 0x7fffffffULL)
                return TOKEN_INT32_CONSTANT;
            else if (yylval.intVal <= 0xffffffffULL)
                return TOKEN_UINT32_CONSTANT;
            else if (yylval.intVal <= 0x7fffffffffffffffULL)
                return TOKEN_INT64_CONSTANT;
            else
                return TOKEN_UINT64_CONSTANT;
        }
    }
}

static int
lParseFP() {
    std::string val(yytext);
    std::string fp64S("d");
    std::string fp64C("D");
    std::string fp16S("f16");
    std::string fp16C("F16");
    if (val.size() >= fp16S.size() && ((val.compare(val.size() - fp16S.size(), fp16S.size(), fp16S) == 0)
           || (val.compare(val.size() - fp16C.size(), fp16C.size(), fp16C) == 0))) {
        yylval.stringVal = new std::string(val.substr(0, val.length() - 3));
        return TOKEN_FLOAT16_CONSTANT;
    } else if (val.size() >= fp64S.size() && ((val.compare(val.size() - fp64S.size(), fp64S.size(), fp64S) == 0)
           || (val.compare(val.size() - fp64C.size(), fp64C.size(), fp64C) == 0))) {
        yylval.doubleVal = atof(yytext);
        return TOKEN_DOUBLE_CONSTANT;
    }
    yylval.floatVal = (float)atof(yytext);
    return TOKEN_FLOAT_CONSTANT;
}


/** Handle a C-style comment in the source.
 */
static void
lCComment(SourcePos *pos) {
    char c, prev = 0;

    while ((c = yyinput()) != 0) {
        ++pos->last_column;

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

static void lNextValidChar(SourcePos *pos, char const*&currChar) {
    while ((*currChar == ' ') || (*currChar == '\t') || (*currChar == '\r')) {
        ++pos->last_column;
        currChar++;
    }
}

/** Handle pragma directive to unroll loops.
*/
static void lPragmaUnroll(YYSTYPE *yylval, SourcePos *pos, std::string fromUserReq, bool isNounroll) {

    const char *currChar = fromUserReq.data();
    yylval->pragmaAttributes = new PragmaAttributes();
    yylval->pragmaAttributes->aType = PragmaAttributes::AttributeType::pragmaloop;
    int count = -1;

    lNextValidChar(pos, currChar);

    if (isNounroll) {
        if (*currChar == '\n') {
            yylval->pragmaAttributes->unrollType = Globals::pragmaUnrollType::nounroll;
            pos->last_column = 1;
            pos->last_line++;
            return;
        }
        pos->last_column = 1;
        pos->last_line++;
        Warning(*pos, "extra tokens at end of '#pragma nounroll'.");
        return;

    }

    if (*currChar == '\n') {
        yylval->pragmaAttributes->unrollType = Globals::pragmaUnrollType::unroll;
        pos->last_column = 1;
        pos->last_line++;
        return;
    }

    bool popPar = false;
    if (*currChar == '(') {
        popPar = true;
        currChar++;
        ++pos->last_column;
    }

    char *endPtr = nullptr;
#if defined(ISPC_HOST_IS_WINDOWS) && !defined(__MINGW32__)
    count = _strtoui64(currChar, &endPtr, 0);
#else
    // FIXME: should use strtouq and then issue an error if we can't
    // fit into 64 bits...
    count = strtoull(currChar, &endPtr, 0);
#endif

    if((count == 0) && (endPtr != currChar)){
        Error(*pos, "'#pragma unroll()' invalid value '0'; must be positive.");
    }

    lNextValidChar(pos, const_cast<const char*&>(endPtr));

    if (popPar == true) {
        if (*endPtr == ')') {
            ++pos->last_column;
            endPtr++;
            lNextValidChar(pos, const_cast<const char*&>(endPtr));
        }
        else {
            Error(*pos, "Incomplete '#pragma unroll()' : expected ')'.");
        }
    }

    yylval->pragmaAttributes->unrollType = Globals::pragmaUnrollType::count;
    yylval->pragmaAttributes->count = count;
    pos->last_line++;
    pos->last_column = 1;
}

/** Handle pragma directive to ignore warning.
*/
static void
lPragmaIgnoreWarning(SourcePos *pos, std::string fromUserReq) {
    std::string userReq;
    const char *currChar = fromUserReq.data();
    bool perfWarningOnly = false;
    lNextValidChar(pos, currChar);

    if (*currChar == '\n') {
        pos->last_column = 1;
        pos->last_line++;
        std::pair<int, std::string> key = std::pair<int, std::string>(pos->last_line, pos->name);
        g->turnOffWarnings[key] = perfWarningOnly;
        return;
    }
    else if (*currChar == '(') {
        currChar++;
        lNextValidChar(pos, currChar);
        while (*currChar != 0 && *currChar != '\n' && *currChar != ' ' && *currChar != ')') {
            userReq += *currChar;
            currChar++;
            ++pos->last_column;
        }
        if ((*currChar == ' ') || (*currChar == ')')) {
            lNextValidChar(pos, currChar);
            if (*currChar == ')') {
                currChar++;
                ++pos->last_column;
                lNextValidChar(pos, currChar);
                if (*currChar == '\n') {
                    pos->last_column = 1;
                    pos->last_line++;
                    if (userReq.compare("perf") == 0) {
                        perfWarningOnly = true;
                        std::pair<int, std::string> key = std::pair<int, std::string>(pos->last_line, pos->name);
                        g->turnOffWarnings[key] = perfWarningOnly;
                    }
                    else if (userReq.compare("all") == 0) {
                        std::pair<int, std::string> key = std::pair<int, std::string>(pos->last_line, pos->name);
                        g->turnOffWarnings[key] = perfWarningOnly;
                    }
                    else {
                        Error(*pos, "Incorrect argument for '#pragma ignore warning()'.");
                    }
                    return;
                }
            }
        }
        else if (*currChar == '\n') {
            Error(*pos, "Incomplete '#pragma ignore warning()' : expected ')'.");
            pos->last_column = 1;
            pos->last_line++;
            return;
        }
    }
    Error(*pos, "Undefined #pragma.");
}

/** Consume line starting with '#pragma' and decide on next action based on
 * directive.
 */
static bool lConsumePragma(YYSTYPE *yylval, SourcePos *pos) {
    char c;
    std::string userReq;
    do {
        c = yyinput();
        ++pos->last_column;
    } while ((c == ' ') || (c == '\t') || (c == '\r'));
    if (c == '\n') {
        // Ignore pragma since - directive provided.
        pos->last_column = 1;
        pos->last_line++;
        return false;
    }

    while (c != '\n') {
        userReq += c;
        c = yyinput();
    }
    userReq += c;
    std::string loopUnroll("unroll"), loopNounroll("nounroll"), ignoreWarning("ignore warning");
    if (loopUnroll == userReq.substr(0, loopUnroll.size())) {
        pos->last_column += loopUnroll.size();
        lPragmaUnroll(yylval, pos, userReq.erase(0, loopUnroll.size()), false);
        return true;
    }
    else if (loopNounroll == userReq.substr(0, loopNounroll.size())) {
        pos->last_column += loopNounroll.size();
        lPragmaUnroll(yylval, pos, userReq.erase(0, loopNounroll.size()), true);
        return true;
    }
    else if (ignoreWarning == userReq.substr(0, ignoreWarning.size())) {
        pos->last_column += ignoreWarning.size();
        lPragmaIgnoreWarning(pos, userReq.erase(0, ignoreWarning.size()));
        return false;
    }

    // Ignore pragma : invalid directive provided.
    Warning(*pos, "unknown pragma ignored.");
    return false;
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
    Assert(yytext[0] == '#');
    if (yytext[1] == ' ')
        // On Linux/OSX, the preprocessor gives us lines like
        // # 1234 "foo.c"
        ptr = yytext + 2;
    else {
        // On windows, cl.exe's preprocessor gives us lines of the form:
        // #line 1234 "foo.c"
        Assert(!strncmp(yytext+1, "line ", 5));
        ptr = yytext + 6;
    }

    // Now we can set the line number based on the integer in the string
    // that ptr is pointing at.
    pos->last_line = strtol(ptr, &src, 10) - 1;
    pos->last_column = 1;
    // Make sure that the character after the integer is a space and that
    // then we have open quotes
    Assert(src != ptr && src[0] == ' ' && src[1] == '"');
    src += 2;

    // And the filename is everything up until the closing quotes
    std::string filename;
    while (*src != '"') {
        Assert(*src && *src != '\n');
        filename.push_back(*src);
        ++src;
    }
    pos->name = RegisterDependency(filename);
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
            Error(*pos, "Bad character escape sequence: '%s'.", str);
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
    if (p == nullptr)
       return;

    while (*p != '\"') {
       char cval = '\0';
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
    Assert(ptr != nullptr);

    Assert(ptr[0] == '0' && (ptr[1] == 'x' || ptr[1] == 'X'));
    ptr += 2;

    // Start initializing the mantissa
    Assert(*ptr == '0' || *ptr == '1');
    double mantissa = (*ptr == '1') ? 1. : 0.;
    ++ptr;

    if (*ptr == '.') {
        // Is there a fraction part?  If so, the i'th digit we encounter
        // gives the 1/(16^i) component of the mantissa.
        ++ptr;

        double scale = 1. / 16.;
        // Keep going until we come to the 'p'/'P', which indicates that we've
        // come to the exponent
        while (*ptr != 'p' && *ptr != 'P') {
            // Figure out the raw value from 0-15
            int digit;
            if (*ptr >= '0' && *ptr <= '9')
                digit = *ptr - '0';
            else if (*ptr >= 'a' && *ptr <= 'f')
                digit = 10 + *ptr - 'a';
            else {
                Assert(*ptr >= 'A' && *ptr <= 'F');
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
        Assert(*ptr == 'p' || *ptr == 'P');

    ++ptr; // skip the 'p'/'P'

    // interestingly enough, the exponent is provided base 10..
    char* endptr = nullptr;
    int exponent = (int)strtol(ptr, &endptr, 10);
    Assert(ptr != endptr);

    // Does stdlib exp2() guarantee exact results for integer n where can
    // be represented exactly as doubles?  I would hope so but am not sure,
    // so let's be sure.
    return mantissa * ipow2(exponent);
}

/** Parse an operator.
*/
static int
lParseOperator(const char *ptr) {
    yylval.stringVal = new std::string(ptr);
    if (m->symbolTable->LookupFunctionTemplate(yytext))
        return TOKEN_TEMPLATE_NAME;
    else
        return TOKEN_IDENTIFIER;
}
