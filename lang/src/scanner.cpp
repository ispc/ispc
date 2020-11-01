#include <ispc/scanner.h>

#include <ispc/token_consumer.h>

#include "consumer_ref_wrapper.h"
#include "scanner.h"

#include <stdexcept>
#include <vector>

#include <climits>
#include <cstdio>

namespace ispc {

class ScannerImpl final : public TokenConsumer {
    yyscan_t flexScanner = nullptr;
    std::vector<std::unique_ptr<TokenConsumer>> consumers;
public:
    ScannerImpl() {
        if (ispclex_init_extra(this, &flexScanner) != 0)
            throw std::bad_alloc();
    }

    ~ScannerImpl() {
        ispclex_destroy(flexScanner);
    }

    void AddTokenConsumer(std::unique_ptr<TokenConsumer> &&consumer) {
        consumers.emplace_back(std::move(consumer));
    }

    void ScanBuffer(const char *text, std::size_t length) {

        length = (length > INT_MAX) ? INT_MAX : length;

        yy_buffer_state *bufferState = ispc_scan_bytes(text, (int) length, flexScanner);

        if (!bufferState)
            return; // TODO : Out of memory diagnostic

        ispc_switch_to_buffer(bufferState, flexScanner);

        ispclex(flexScanner);

        ispc_delete_buffer(bufferState, flexScanner);
    }

    bool ScanFile(const char *path) {

        FILE *file = std::fopen(path, "rb");
        if (!file)
            return false;

        ispcset_in(file, flexScanner);

        ispclex(flexScanner);

        ispcset_in(stdin, flexScanner);

        std::fclose(file);

        return true;
    }

    void Consume(const Token& token) override {
        for (auto &consumer : consumers)
            consumer->Consume(token);
    }
};

Scanner::~Scanner() {
    delete self;
    self = nullptr;
}

void Scanner::AddTokenConsumer(std::unique_ptr<TokenConsumer> &&consumer) {

    if (!self)
        self = new ScannerImpl;

    self->AddTokenConsumer(std::move(consumer));
}

void Scanner::AddTokenConsumer(TokenConsumer &consumer) {
    AddTokenConsumer(ConsumerRef<TokenConsumer, Token>::make(consumer));
}

void Scanner::ScanBuffer(const char *text, std::size_t length) {

    if (!self)
        self = new ScannerImpl();

    self->ScanBuffer(text, length);
}

bool Scanner::ScanFile(const char *path) {

    if (!self)
        self = new ScannerImpl();

    return self->ScanFile(path);
}

} // namespace ispc
