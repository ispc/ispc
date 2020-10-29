#include <gtest/gtest.h>

#include <ispc/scanner.h>
#include <ispc/token.h>
#include <ispc/token_consumer.h>

#include <iostream>
#include <string_view>

#include <cstdlib>

using namespace ispc;

namespace {

std::ostream &operator << (std::ostream &stream, TokenType type) {
    return stream << ToString(type);
}

class MockTokenConsumer final : public TokenConsumer {
    bool called = false;
    TokenType expectedType;
    std::string_view expectedText;
  public:
    MockTokenConsumer(TokenType expectedType_,
                      const std::string_view &expectedText_)
        : expectedType(expectedType_),
          expectedText(expectedText_) {}

    ~MockTokenConsumer() {
        EXPECT_EQ(called, true);
    }

    void Consume(const Token &token) override {

        if (called)
            return; // Ignore tokens following the first one.

        called = true;
        EXPECT_EQ(expectedType, token.type);
        EXPECT_EQ(expectedText, std::string_view(token.text, token.length));
    }
};

void RunTest(const std::string_view &text, TokenType expectedType,
             const std::string_view &expectedText = "") {

    Scanner scanner;

    auto tokenConsumer = std::make_unique<MockTokenConsumer>(expectedType, expectedText);

    scanner.AddTokenConsumer(std::unique_ptr<TokenConsumer>(tokenConsumer.release()));

    scanner.ScanBuffer(text.data(), text.size());
}

} // namespace

TEST(Scanner, Space) {
    RunTest(" \ta", TokenType::Space, " \t");
}

TEST(Scanner, Space2) {
    RunTest("\t  a", TokenType::Space, "\t  ");
}

TEST(Scanner, Newline) {
    RunTest("\n ", TokenType::Newline, "\n");
}

TEST(Scanner, Newline2) {
    RunTest("\r ", TokenType::Newline, "\r");
}

TEST(Scanner, Newline3) {
    RunTest("\r\n ", TokenType::Newline, "\r\n");
}

TEST(Scanner, Newline4) {
    RunTest("\r\n\n ", TokenType::Newline, "\r\n");
}

TEST(Scanner, Identifier) {
    RunTest("abc123_ ", TokenType::Identifier, "abc123_");
}

TEST(Scanner, Identifier2) {
    RunTest("_abc123 ", TokenType::Identifier, "_abc123");
}

TEST(Scanner, BinInt) {
    RunTest("0b10 ", TokenType::BinInt, "0b10");
}

TEST(Scanner, BinIntIncomplete) {
    RunTest("0b ", TokenType::BinIntIncomplete, "0b");
}

TEST(Scanner, BinIntInvalid) {
    RunTest("0b23 ", TokenType::BinIntInvalid, "0b23");
}

TEST(Scanner, DecInt) {
    RunTest("123 ", TokenType::DecInt, "123");
}

TEST(Scanner, InvalidDecInt) {
    RunTest("123akr ", TokenType::DecIntInvalid, "123akr");
}

TEST(Scanner, HexInt) {
    RunTest("0xca11ab1e ", TokenType::HexInt, "0xca11ab1e");
}

TEST(Scanner, HexIntIncomplete) {
    RunTest("0x ", TokenType::HexIntIncomplete, "0x");
}

TEST(Scanner, HexIntInvalid) {
    RunTest("0x4skdf ", TokenType::HexIntInvalid, "0x4skdf");
}
