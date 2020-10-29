#include <gtest/gtest.h>

#include <ispc/ast_node_consumer.h>
#include <ispc/ast_node_visitor.h>
#include <ispc/expr.h>
#include <ispc/parser.h>
#include <ispc/token.h>

namespace {

using namespace ispc;

template <typename ExpectedType>
class MockASTNodeConsumer : public ASTNodeConsumer, public ASTNodeVisitor {
    bool called = false;
  public:
    virtual ~MockASTNodeConsumer() {
        EXPECT_EQ(called, true);
    }
    void Consume(const ASTNode &node) override {
        called = true;
        node.Accept(*this);
    }
    void Visit(const IntegerLiteral &node) override {
        Check(node);
    }
    void Visit(const Identifier &node) override {
        (void)node;
    }
  protected:
    virtual void Check(const ExpectedType &) = 0;
    void Check(const ASTNode &) {
        FAIL() << "Unexpected AST type";
    }
};

class IntegerLiteralChecker final : public MockASTNodeConsumer<IntegerLiteral> {
    std::uint64_t expectedValue = 0;
  public:
    IntegerLiteralChecker(std::uint64_t expectedValue_) noexcept
        : expectedValue(expectedValue_) {}
  protected:
    void Check(const IntegerLiteral &node) override {
        EXPECT_EQ(expectedValue, node.GetValue());
    }
};

void PutToken(Parser &parser, TokenType type, const std::string_view &text) {

    Token token { type, text.data(), text.size() };

    parser.Consume(token);
}

template <typename Checker, typename... Args>
void PutChecker(Parser &parser, Args... args) {

    auto mockConsumer = std::unique_ptr<ASTNodeConsumer>(new Checker(args...));

    parser.AddASTNodeConsumer(std::move(mockConsumer));
}

} // namespace

TEST(Parser, ParseBinLiteral) {

    Parser parser;

    PutChecker<IntegerLiteralChecker>(parser, 11);

    PutToken(parser, TokenType::BinInt, "0b01011");

    parser.Finish();
}
