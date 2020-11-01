%option reentrant noyywrap stack
%option extra-type="ispc::TokenConsumer *"
%option nounistd
%option never-interactive
%option prefix="ispc"

%{
#include <ispc/token.h>
#include <ispc/token_consumer.h>

#include "parser.hh"

namespace {

void produce(ispc::TokenConsumer *consumer,
             ispc::TokenType type,
             const char *text,
             std::size_t length) {

    consumer->Consume(ispc::Token { type, text, length });
}

} // namespace

#define PRODUCE(type) \
    produce(yyget_extra(yyscanner), type, yytext, yyleng);
%}

SPACE [ \t]+
NEWLINE ((\r\n)|\n|\r)

DIGIT [0-9]
NON_DIGIT [a-zA-Z_]
IDENTIFIER {NON_DIGIT}+({DIGIT}|{NON_DIGIT})*

BIN_INT (0b[01]+)[uUlL]*[kMG]?[uUlL]*
BIN_INT_INCOMPLETE 0b
BIN_INT_INVALID 0b({DIGIT}|{NON_DIGIT})+

DEC_INT {DIGIT}+[uUlL]*[kMG]?[uUlL]*
DEC_INT_INVALID [^(0(x|b))]{DIGIT}+({DIGIT}|{NON_DIGIT})*

HEX_INT (0x[0-9a-fA-F]+)[uUlL]*[kMG]?[uUlL]*
HEX_INT_INCOMPLETE 0x
HEX_INT_INVALID 0x({DIGIT}|{NON_DIGIT})+

STRING_LITERAL_INCOMPLETE L?\"(\\.|[^\\\n\r"])*
STRING_LITERAL {STRING_LITERAL_INCOMPLETE}\"

INT_NUMBER_DOTDOTDOT (([0-9]+)|(0x[0-9a-fA-F]+)|(0b[01]+))[uUlL]*[kMG]?[uUlL]*\.\.\.
FLOAT_NUMBER (([0-9]+|(([0-9]+\.[0-9]*[fF]?)|(\.[0-9]+)))([eE][-+]?[0-9]+)?[fF]?)
HEX_FLOAT_NUMBER (0x[01](\.[0-9a-fA-F]*)?p[-+]?[0-9]+[fF]?)
FORTRAN_DOUBLE_NUMBER (([0-9]+\.[0-9]*[dD])|([0-9]+\.[0-9]*[dD][-+]?[0-9]+)|([0-9]+[dD][-+]?[0-9]+)|(\.[0-9]*[dD][-+]?[0-9]+))

ZO_SWIZZLE ([01]+[w-z]+)+|([01]+[rgba]+)+|([01]+[uv]+)+
%%

{SPACE} {
    PRODUCE(ispc::TokenType::Space);
}

{NEWLINE} {
    PRODUCE(ispc::TokenType::Newline);
}

{IDENTIFIER} {
    PRODUCE(ispc::TokenType::Identifier);
}

{BIN_INT} {
    PRODUCE(ispc::TokenType::BinInt);
}

{BIN_INT_INCOMPLETE} { PRODUCE(ispc::TokenType::BinIntIncomplete); }

{BIN_INT_INVALID} { PRODUCE(ispc::TokenType::BinIntInvalid); }

{DEC_INT} {
    PRODUCE(ispc::TokenType::DecInt);
}

{DEC_INT_INVALID} {
    PRODUCE(ispc::TokenType::DecIntInvalid);
}

{HEX_INT} {
    PRODUCE(ispc::TokenType::HexInt);
}

{HEX_INT_INVALID} {
    PRODUCE(ispc::TokenType::HexIntInvalid);
}

{HEX_INT_INCOMPLETE} {
    PRODUCE(ispc::TokenType::HexIntIncomplete);
}

{STRING_LITERAL} {
    PRODUCE(ispc::TokenType::StringLiteral);
}

{STRING_LITERAL_INCOMPLETE} {
    PRODUCE(ispc::TokenType::StringLiteralIncomplete);
}

%%
