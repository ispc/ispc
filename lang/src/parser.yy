%{
#include <ispc/ast_node_consumer.h>
#include <ispc/diagnostic.h>
#include <ispc/diagnostic_consumer.h>
#include <ispc/expr.h>
#include <ispc/source_pos.h>
#include <ispc/token.h>

#include <ostream>
#include <string>

using namespace ispc;

namespace {

class SimpleDiagnostic final : public Diagnostic {
    std::string message;
  public:
    SimpleDiagnostic(const char *msg) : message(msg) {}

    void Print(std::ostream &stream) const override {
        stream << message << std::endl;
    }
};

void yyerror(const void *,
             DiagnosticConsumer &diagnosticConsumer,
             ASTNodeConsumer &,
             const char *tok_text,
             std::size_t tok_length,
             const char *message) {

    printf("ERROR: %s\n", message);

    (void)tok_text;
    (void)tok_length;

    SimpleDiagnostic diagnostic(message);

    diagnosticConsumer.Consume(diagnostic);
}

ASTNode *CreateStringLiteral(const char *text, std::size_t length);

} // namespace

%}

%code requires {

#include <cstddef>

namespace ispc {

class DiagnosticConsumer;
class ASTNode;
class ASTNodeConsumer;

struct SourcePos;

} // namespace ispc

}

%define api.prefix {ispc_}

%define api.pure full
%define api.push-pull push

%locations

%define api.location.type {ispc::SourcePos}
%define api.value.type {ispc::ASTNode *}
%define api.token.prefix {ISPC_TOKEN_}

%define parse.error verbose

%debug

%destructor {
    if ($$)
        nodeConsumer.Consume(*$$);
    delete $$;
} integer_literal string_literal primary_expr ast_node

%parse-param {ispc::DiagnosticConsumer &diagnosticConsumer}
%parse-param {ispc::ASTNodeConsumer &nodeConsumer}
%parse-param {const char *token_text}
%parse-param {std::size_t token_length}

%token IDENTIFIER
%token BIN_INT DEC_INT HEX_INT
%token STRING_LITERAL

%start ast_node

%%

string_literal
    : STRING_LITERAL
    {
        $$ = CreateStringLiteral(token_text, token_length);
    }

integer_literal
    : BIN_INT
    {
        $$ = new IntegerLiteral(BinFormat{}, std::string_view(token_text + 2, token_length - 2));
    }
    | DEC_INT
    {
        $$ = new ispc::IntegerLiteral(0);
    }
    | HEX_INT
    {
        $$ = new ispc::IntegerLiteral(0);
    }
    ;

primary_expr
    : integer_literal { $$ = $1; }
    | string_literal  { $$ = $1; }
    ;

ast_node
    : primary_expr { $$ = $1; }
    ;

%%

namespace {

ispc::ASTNode *CreateStringLiteral(const char *text, std::size_t length) {
    ispc::Token token {
        ispc::TokenType::StringLiteral,
        text,
        length
    };
    return new ispc::StringLiteral(token);
}

} // namespace
