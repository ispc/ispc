#pragma once

namespace ispc {

struct Token;

class TokenConsumer {
public:
    virtual ~TokenConsumer() {}

    virtual void Consume(const Token&) = 0;
};

} // namespace ispc
