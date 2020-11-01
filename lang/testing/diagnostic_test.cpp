#include "diagnostic_test.h"

#include <gtest/gtest.h>

#include <ispc/diagnostic.h>
#include <ispc/diagnostic_printer.h>
#include <ispc/parser.h>
#include <ispc/scanner.h>

#include <fstream>
#include <vector>

namespace ispc {

namespace {

class DiagnosticTest final : public ::testing::Test {
    const char *sourcePath;
    const char *expectedErrorPath;
  public:
    DiagnosticTest(const char *sourcePath_,
                   const char *expectedErrorPath_) noexcept
        : sourcePath(sourcePath_), expectedErrorPath(expectedErrorPath_) {}

    void TestBody() override {

        auto expectedError = OpenFile(expectedErrorPath);

        ASSERT_EQ(expectedError.has_value(), true);

        auto source = OpenFile(sourcePath);

        ASSERT_EQ(source.has_value(), true);

        Scanner scanner;

        Parser parser;

        scanner.AddTokenConsumer(parser);

        std::ostringstream diagnosticStream;

        DiagnosticPrinter diagnosticPrinter(diagnosticStream);

        parser.AddDiagnosticConsumer(diagnosticPrinter);

        scanner.ScanBuffer(source->data(), source->size());

        parser.Finish();

        EXPECT_EQ(diagnosticStream.str(), *expectedError);
    }

    static std::optional<std::string> OpenFile(const char *path) {

        std::ifstream file(path);
        if (!file.good())
            return std::nullopt;

        std::stringstream contentStream;

        contentStream << file.rdbuf();

        return contentStream.str();
    }
};

} // namespace

::testing::Test *MakeDiagnosticTest(const char *sourcePath,
                                    const char *expectedErrorPath) {

  return new DiagnosticTest(sourcePath, expectedErrorPath);
}

} // namespace ispc
