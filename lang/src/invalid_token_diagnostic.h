#pragma once

#include <ispc/diagnostic.h>
#include <ispc/token.h>

namespace ispc {

class InvalidTokenDiagnostic final : public Diagnostic {
    Token token;
  public:
    InvalidTokenDiagnostic(const Token &t) noexcept
        : token(t) {}

    void Print(std::ostream &stream) const override;
};

} // namespace ispc
