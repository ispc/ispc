#include <gtest/gtest.h>

#include <ispc/diagnostic.h>
#include <ispc/diagnostic_consumer.h>
#include <ispc/parser.h>

#include <sstream>
#include <string_view>

namespace {

class MockDiagnosticConsumer final {
    std::string_view expectedMessage;
    bool called = false;
  public:
    MockDiagnosticConsumer(const std::string_view &expectedMessage_)
        : expectedMessage(expectedMessage_) {}

    ~MockDiagnosticConsumer() {
        EXPECT_EQ(called, true);
    }

    void Consume(const Diagnostic &d) override {
        called = true;

        std::ostringstream stream;
        d.Print(stream);
        auto msg = stream.str();

        EXPECT_EQ(msg, expectedMessage);
    }
};

void ExpectDiagnostic(Parser &parser, const std::string_view &msg) {

    std::unique_ptr<DiagnosticConsumer> mockConsumer(new MockDiagnosticConsumer(msg));

    parser.AddDiagnosticConsumer(std::move(mockConsumer));
}

} // namespace

TEST(Diagnostics, UnexpectedIdentifier) {

    Parser parser;

    ExpectDiagnostic(parser, "syntax error, unexpected identifier");

    parser.Consume(Token { TokenType::Identifier, "foo" });

    parser.Finish();
}
