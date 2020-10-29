%{
#include <ispc/ast_node_consumer.h>
#include <ispc/diagnostic.h>
#include <ispc/diagnostic_consumer.h>
#include <ispc/expr.h>
#include <ispc/source_pos.h>

#include <ostream>
#include <string>

class SimpleDiagnostic final : public ispc::Diagnostic {
    std::string message;
  public:
    SimpleDiagnostic(const char *msg) : message(msg) {}

    void Print(std::ostream &stream) const override {
        stream << message;
    }
};

void yyerror(const void *,
             ispc::DiagnosticConsumer &diagnosticConsumer,
             ispc::ASTNodeConsumer &,
             const char *tok_text,
             std::size_t tok_length,
             const char *message) {

    (void)tok_text;
    (void)tok_length;

    SimpleDiagnostic diagnostic(message);

    diagnosticConsumer.Consume(diagnostic);
}

%}

%code requires {

#include <cstddef>

namespace ispc {

class DiagnosticConsumer;
class ASTNode;
class ASTNodeConsumer;
class SourcePos;

} // namespace ispc

}

%define api.prefix {ispc_}

%define api.pure full
%define api.push-pull push

%locations

%define api.location.type {ispc::SourcePos}
%define api.value.type {ispc::ASTNode *}
%define api.token.prefix {ISPC_TOKEN_}

%destructor { nodeConsumer.Consume(*$$); delete $$; } integer_literal

%parse-param {ispc::DiagnosticConsumer &diagnosticConsumer}
%parse-param {ispc::ASTNodeConsumer &nodeConsumer}
%parse-param {const char *token_text}
%parse-param {std::size_t token_length}

%token IDENTIFIER
%token BIN_INT
%token DEC_INT
%token HEX_INT

%%

integer_literal: BIN_INT {
    $$ = new ispc::IntegerLiteral(ispc::BinFormat{}, token_text + 2, token_length - 2);
}

integer_literal: DEC_INT { $$ = new ispc::IntegerLiteral(0); }

integer_literal: HEX_INT { $$ = new ispc::IntegerLiteral(0); }
