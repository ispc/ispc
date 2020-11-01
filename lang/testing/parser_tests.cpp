#include <gtest/gtest.h>

#include <ispc/ast_node_consumer.h>
#include <ispc/ast_node_visitor.h>
#include <ispc/expr.h>
#include <ispc/parser.h>
#include <ispc/token.h>

#include <algorithm>

namespace {

using namespace ispc;

std::ostream &operator << (std::ostream &stream, StringSequenceType type) {
    return stream << ToString(type);
}

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
    void Visit(const StringLiteral &node) override {
        Check(node);
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

class StringLiteralChecker final : public MockASTNodeConsumer<StringLiteral> {

    std::string_view expectedText;

    std::vector<StringSequence> expectedSequences;

  public:

    StringLiteralChecker(const std::string_view &expected_, const std::vector<StringSequence> &seqs_)
        : expectedText(expected_), expectedSequences(seqs_) {}

  protected:

    void Check(const StringLiteral &node) override {

        EXPECT_EQ(expectedText, node.GetText());

        EXPECT_EQ(expectedSequences.size(), node.GetSequenceCount());

        auto minSeqCount = std::min(expectedSequences.size(), node.GetSequenceCount());

        for (std::size_t i = 0; i < node.GetSequenceCount(); i++) {
            const auto &expectedSeq = expectedSequences[i];
            const auto &actualSeq = node.GetSequence(i);
            EXPECT_EQ(expectedSeq.type, actualSeq.type);
            EXPECT_EQ(expectedSeq.text, actualSeq.text);
            EXPECT_EQ(expectedSeq.transformedText,
                      actualSeq.transformedText);
        }
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

TEST(Parser, ParseStringLiteral) {

    Parser parser;

    char expectedData[] {
         'A', 'B', '\\',  '"', '\'', '\a',
        '\b', '\f', '\n', '\r',
        '\t', '\v', '\0', '\75',
        '\x1f', '\377', '\0', '\xff', '\x0'
    };

    std::string_view expected(expectedData, sizeof(expectedData));

    std::vector<StringSequence> sequences;

    auto ExpectSeq = [&sequences](auto type, auto text, auto ttext) {
        sequences.emplace_back(StringSequence {
            type, text, ttext
        });
    };

    ExpectSeq(StringSequenceType::Normal,
              std::string_view("AB"),
              std::string_view("AB"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\\\"),
              std::string_view("\\"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\\""),
              std::string_view("\""));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\'"),
              std::string_view("'"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\a"),
              std::string_view("\a"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\b"),
              std::string_view("\b"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\f"),
              std::string_view("\f"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\n"),
              std::string_view("\n"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\r"),
              std::string_view("\r"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\t"),
              std::string_view("\t"));

    ExpectSeq(StringSequenceType::EscapedChar,
              std::string_view("\\v"),
              std::string_view("\v"));

    ExpectSeq(StringSequenceType::OctalDigits,
              std::string_view("\\0", 2),
              std::string_view("\0", 1));

    ExpectSeq(StringSequenceType::OctalDigits,
              std::string_view("\\75"),
              std::string_view("\75"));

    ExpectSeq(StringSequenceType::HexDigits,
              std::string_view("\\x1f"),
              std::string_view("\x1f"));

    ExpectSeq(StringSequenceType::OctalDigits,
              std::string_view("\\377"),
              std::string_view("\377"));

    ExpectSeq(StringSequenceType::OctalDigitsOverflow,
              std::string_view("\\400"),
              std::string_view("\0", 1));

    ExpectSeq(StringSequenceType::HexDigits,
              std::string_view("\\xff"),
              std::string_view("\xff", 1));

    ExpectSeq(StringSequenceType::HexDigitsOverflow,
              std::string_view("\\x100"),
              std::string_view("\x0", 1));

    PutChecker<StringLiteralChecker>(parser, expected, sequences);

    PutToken(parser, TokenType::StringLiteral, R"("AB\\\"\'\a\b\f\n\r\t\v\0\75\x1f\377\400\xff\x100")");

    parser.Finish();
}
