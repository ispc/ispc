#include <ispc/parser.h>

#include <ispc/ast_node.h>
#include <ispc/ast_node_consumer.h>
#include <ispc/diagnostic_consumer.h>
#include <ispc/source_pos.h>
#include <ispc/token.h>

#include "consumer_ref_wrapper.h"
#include "parser.hh"

#include <optional>
#include <stdexcept>
#include <vector>

#include <cstdio>

namespace ispc {

class CompositeDiagnosticConsumer final : public DiagnosticConsumer {
    std::vector<std::unique_ptr<DiagnosticConsumer>> consumers;

  public:
    void AddDiagnosticConsumer(std::unique_ptr<DiagnosticConsumer> &&consumer) {
        consumers.emplace_back(std::move(consumer));
    }
    void Consume(const Diagnostic &d) override {
        for (auto &consumer : consumers)
            consumer->Consume(d);
    }
};

class CompositeASTNodeConsumer final : public ASTNodeConsumer {
    std::vector<std::unique_ptr<ASTNodeConsumer>> consumers;

  public:
    void AddASTNodeConsumer(std::unique_ptr<ASTNodeConsumer> &&consumer) {
        consumers.emplace_back(std::move(consumer));
    }
    void Consume(const ASTNode &node) override {
        for (auto &consumer : consumers)
            consumer->Consume(node);
    }
};

class ParserImpl final {

    SourcePos position;

    ispc_pstate *state = nullptr;

    bool needsMore = false;

    CompositeDiagnosticConsumer diagnosticConsumer;

    CompositeASTNodeConsumer nodeConsumer;

  public:
    ParserImpl() : state(ispc_pstate_new()) {
        if (!state)
            throw std::bad_alloc();
    }

    ~ParserImpl() { ispc_pstate_delete(state); }

    void AddDiagnosticConsumer(std::unique_ptr<DiagnosticConsumer> &&consumer) {
        diagnosticConsumer.AddDiagnosticConsumer(std::move(consumer));
    }

    void AddASTNodeConsumer(std::unique_ptr<ASTNodeConsumer> &&consumer) {
        nodeConsumer.AddASTNodeConsumer(std::move(consumer));
    }

    void Consume(const Token &token) {

        auto bisonType = ToBisonType(token.type);
        if (!bisonType)
            return;

        auto status = ispc_push_parse(state, *bisonType, nullptr, &position, diagnosticConsumer, nodeConsumer,
                                      token.text, token.length);

        needsMore = (status == YYPUSH_MORE);
    }

    void Finish() { ispc_push_parse(state, 0, nullptr, &position, diagnosticConsumer, nodeConsumer, "", 0); }

    bool NeedsMore() const noexcept { return needsMore; }

    static constexpr std::optional<ispc_tokentype> ToBisonType(TokenType type) noexcept {
        switch (type) {
        case TokenType::Identifier:
            return ISPC_TOKEN_IDENTIFIER;
        case TokenType::BinInt:
            return ISPC_TOKEN_BIN_INT;
        case TokenType::DecInt:
            return ISPC_TOKEN_DEC_INT;
        case TokenType::HexInt:
            return ISPC_TOKEN_HEX_INT;
        case TokenType::StringLiteral:
            return ISPC_TOKEN_STRING_LITERAL;
        case TokenType::BinIntIncomplete:
        case TokenType::BinIntInvalid:
        case TokenType::DecIntInvalid:
        case TokenType::HexIntIncomplete:
        case TokenType::HexIntInvalid:
        case TokenType::StringLiteralIncomplete:
            // TODO
            return std::nullopt;
        case TokenType::Space:
        case TokenType::Newline:
            break;
        }
        return std::nullopt;
    }
};

Parser::~Parser() {
    delete self;
    self = nullptr;
}

void Parser::AddASTNodeConsumer(std::unique_ptr<ASTNodeConsumer> &&consumer) {

    if (!self)
        self = new ParserImpl;

    self->AddASTNodeConsumer(std::move(consumer));
}

void Parser::AddASTNodeConsumer(ASTNodeConsumer &consumer) {
    AddASTNodeConsumer(ConsumerRef<ASTNodeConsumer, ASTNode>::make(consumer));
}

void Parser::AddDiagnosticConsumer(std::unique_ptr<DiagnosticConsumer> &&consumer) {

    if (!self)
        self = new ParserImpl;

    self->AddDiagnosticConsumer(std::move(consumer));
}

void Parser::AddDiagnosticConsumer(DiagnosticConsumer &consumer) {
    AddDiagnosticConsumer(ConsumerRef<DiagnosticConsumer, Diagnostic>::make(consumer));
}

void Parser::Consume(const Token &token) {

    if (!self)
        self = new ParserImpl;

    self->Consume(token);
}

void Parser::Finish() {
    if (self)
        self->Finish();
}

bool Parser::NeedsMore() const noexcept { return self ? self->NeedsMore() : false; }

} // namespace ispc
