#pragma once

namespace ispc {

enum class TokenType {
    Identifier,
    BinInt,
    /** A "0b" prefix with no following characters. */
    BinIntIncomplete,
    /** A "0b" prefix followed by invalid characters. */
    BinIntInvalid,
    DecInt,
    /** A decimal digit followed by invalid characters. */
    DecIntInvalid,
    HexInt,
    /** A "0x" prefix with no following characters. */
    HexIntIncomplete,
    /** A "0x" prefix with invalid following characters. */
    HexIntInvalid,
    /** Consists of a single newline sequence,
     * which may be a single line feed, a single
     * carriage return, or a single carriage return
     * immediately followed by a line feed. */
    Newline,
    /** Consists of any number of spaces or tabs. */
    Space,
    /** A double quoted string constant. */
    StringLiteral,
    /** A string literal with a missing right quote. */
    StringLiteralIncomplete
};

/** Converts the name of the token into
 * a human-readable string. Useful for debugging.
 *
 * @param type The token type to be converted.
 *
 * @return A human-readable string describing @p type.
 *         This function never returns null pointers.
 * */
constexpr const char *ToString(TokenType type) noexcept;

struct Token final {
    TokenType type;
    /** A null terminated string containing the token
     * text from the source code being parsed. */
    const char *text = "";
    /** The number of bytes in the token text. */
    std::size_t length = 0;
};

constexpr const char *ToString(TokenType type) noexcept {
    switch (type) {
        case TokenType::Identifier:
            return "Identifier";
        case TokenType::BinInt:
            return "Binary Integer";
        case TokenType::BinIntIncomplete:
            return "Binary Integer (Incomplete)";
        case TokenType::BinIntInvalid:
            return "Binary Integer (Invalid)";
        case TokenType::DecInt:
            return "Decimal Integer";
        case TokenType::DecIntInvalid:
            return "Decimal Integer (Invalid)";
        case TokenType::HexInt:
            return "Hexidecimal Integer";
        case TokenType::HexIntIncomplete:
            return "Hexidecimal Integer (Incomplete)";
        case TokenType::HexIntInvalid:
            return "Hexidecimal Integer (Invalid)";
        case TokenType::Newline:
            return "Newline";
        case TokenType::Space:
            return "Space";
        case TokenType::StringLiteral:
            return "String Literal";
        case TokenType::StringLiteralIncomplete:
            return "String Literal (Incomplete)";
    }
    return "";
}

} // namespace ispc
