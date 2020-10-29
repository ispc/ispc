#include "ispc/scanner.h"
#include "ispc/token.h"
#include "ispc/token_consumer.h"

#include <cstdio>

namespace {

class TokenPrinter final : public ispc::TokenConsumer {
public:
    void Consume(const ispc::Token&) override {
        printf("Found token\n");
    }
};

} // namespace

int main(int argc, char **argv) {

    for (int i = 1; i < argc; i++) {

        auto printer = std::unique_ptr<ispc::TokenConsumer>(new TokenPrinter);

        ispc::Scanner scanner;

        scanner.AddTokenConsumer(std::move(printer));

        scanner.ScanFile(argv[i]);
    }

    return 0;
}
