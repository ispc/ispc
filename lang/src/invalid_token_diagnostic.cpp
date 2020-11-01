#include "invalid_token_diagnostic.h"

#include <ostream>

namespace ispc {

void InvalidTokenDiagnostic::Print(std::ostream &stream) const {
    auto print = [&stream](const char *msg) {
        stream << "syntax error, " << msg << std::endl;
    };
    switch (token.type) {
    case TokenType::Identifier:
        print("unexpected identifier");
        break;
    case TokenType::BinInt:
    case TokenType::DecInt:
    case TokenType::HexInt:
        print("unexpected integer literal");
        break;
    case TokenType::BinIntIncomplete:
    case TokenType::HexIntIncomplete:
        print("incomplete integer literal");
        break;
    case TokenType::BinIntInvalid:
        print("malformed binary integer literal");
        break;
    case TokenType::DecIntInvalid:
        print("malformed integer literal");
        break;
    case TokenType::HexIntInvalid:
        print("malformed hex integer literal");
        break;
    case TokenType::StringLiteral:
        print("unexpected string literal");
        break;
    case TokenType::StringLiteralIncomplete:
        print("missing terminating '\"' character");
        break;
    case TokenType::Space:
    case TokenType::Newline:
        break;
    }
}

} // namespace ispc
