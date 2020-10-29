#include <ispc/ast_node.h>
#include <ispc/ast_node_consumer.h>
#include <ispc/ast_node_visitor.h>
#include <ispc/diagnostic.h>
#include <ispc/diagnostic_consumer.h>
#include <ispc/expr.h>
#include <ispc/parser.h>
#include <ispc/token.h>

#include <iostream>

namespace {

class DiagPrinter final : public ispc::DiagnosticConsumer {
  public:
    void Consume(const ispc::Diagnostic &d) override {
        d.Print(std::cerr);
    }
};

class ASTPrinter final : public ispc::ASTNodeVisitor, public ispc::ASTNodeConsumer {
  public:
    void Consume(const ispc::ASTNode &node) override {
        node.Accept(*this);
    }
    void Visit(const ispc::IntegerLiteral &) override {
        printf("here!\n");
    }
    void Visit(const ispc::Identifier &) override {
    }
};

} // namespace

int main() {

    ispc::Parser parser;

    parser.AddDiagnosticConsumer(std::unique_ptr<ispc::DiagnosticConsumer>(new DiagPrinter));

    parser.AddASTNodeConsumer(std::unique_ptr<ispc::ASTNodeConsumer>(new ASTPrinter));

    parser.Consume(ispc::Token { ispc::TokenType::Int });

    parser.Finish();

    return 0;
}
